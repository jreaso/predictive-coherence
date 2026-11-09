"""Mean-field Gaussian variational inference, by Bayes by backprop and by IVON.

Both hold a factorised Gaussian over the parameters: Bayes by backprop (Blundell et al., 2015)
follows reparameterised ELBO gradients with Adam, and IVON (Shen et al., 2024) takes natural-
gradient steps with an estimated Hessian.

The update is variational continual learning: the current posterior becomes the prior, and the
objective sees only the new points, for a tuned number of steps. The predictive is the mixture
over `A` atoms `loc + scale * eps_a`, with `eps_a` drawn once and shared by every predictive. The
optimiser's noise is shared by an environment's updates; `redrawn` gives an independent draw.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from pc_experiments.neural_testbed.environment import Setting
from pc_experiments.neural_testbed.models.base import (
    NetworkMixture,
    draw_normal,
    per_environment,
)
from pc_experiments.neural_testbed.networks import Architecture


@dataclass(frozen=True)
class BBBConfig:
    """
    Attributes:
        prior_scale: multiplies the standard deviations of the environments' prior.
        kl_weight: multiplies the divergence from the prior, at `fit` and `update` alike; at one
            the objective is the ELBO, under which a few dozen points barely move the posterior.
        init_std_factor: the initial `scale` at `fit`, as a fraction of the prior's.
        num_atoms: `A`.
        elbo_samples: parameter draws per step.
        lr, fit_steps: Adam at `fit`.
        update_lr, update_steps: Adam at `update`.
    """

    prior_scale: float
    kl_weight: float
    init_std_factor: float
    num_atoms: int
    elbo_samples: int
    lr: float
    fit_steps: int
    update_lr: float
    update_steps: int


@dataclass(frozen=True)
class IVONConfig:
    """
    Attributes:
        prior_scale, kl_weight: as for `BBBConfig`.
        num_atoms: `A`.
        mc_samples: parameter draws per step.
        hess_init: the likelihood's share of the precision, per point, when `fit` starts; an
            update starts from none.
        beta1, beta2: momentum of the gradient and of the Hessian estimate.
        lr, fit_steps: step size and number of steps at `fit`.
        update_lr, update_steps: at `update`.
    """

    prior_scale: float
    kl_weight: float
    num_atoms: int
    mc_samples: int
    hess_init: float
    beta1: float
    beta2: float
    lr: float
    fit_steps: int
    update_lr: float
    update_steps: int


class MeanField[Config: (BBBConfig, IVONConfig)](NetworkMixture):
    """Factorised Gaussians over a network's parameters, for a chunk of environments.

    Attributes:
        loc, scale: the posterior, shape `(E, P)`, or `(E, U, P)` once updated.
        bank: the standard normal draws behind the atoms, shape `(E, A, P)`.
    """

    name: str
    random_update = True

    def __init__(
        self,
        config: Config,
        architecture: Architecture,
        seeds: tuple[int, ...],
        loc: Tensor,
        scale: Tensor,
        bank: Tensor,
    ) -> None:
        self.config = config
        self.architecture = architecture
        self.num_atoms = config.num_atoms
        self.seeds = seeds
        self.loc, self.scale, self.bank = loc, scale, bank

    @classmethod
    def fit(
        cls,
        config: Config,
        setting: Setting,
        x: Tensor,
        y: Tensor,
        seeds: tuple[int, ...],
    ) -> Self:
        """Fit to inputs `(E, n, d)` and labels `(E, n)`, from the prior."""
        architecture, device = setting.architecture, x.device
        prior_scale = (config.prior_scale * architecture.prior_std).to(device)
        prior_scale = prior_scale.expand(len(seeds), 1, 1, -1)
        start = torch.stack(
            [
                architecture.sample_gaussian_prior(1, rng)
                for rng in per_environment(seeds, cls.name, "init")
            ]
        ).to(device)
        bank = draw_normal(
            per_environment(seeds, cls.name, "atoms", device=device),
            config.num_atoms,
            architecture.num_parameters,
        )
        model = cls(config, architecture, seeds, start[:, 0], prior_scale[:, 0, 0], bank)
        loc, scale = model._optimise(
            start=start.unsqueeze(1),
            prior=(torch.zeros_like(prior_scale), prior_scale),
            x=x.unsqueeze(1),
            y=y[:, None, None],
            fitting=True,
        )
        model.loc, model.scale = loc[:, 0, 0], scale[:, 0, 0]
        return model

    def update(self, x: Tensor, y: Tensor) -> Self:
        prior = (self.loc[:, None, None], self.scale[:, None, None])
        start = prior[0].expand(*y.shape[:3], -1)
        loc, scale = self._optimise(start, prior, x, y, fitting=False)
        updated = type(self)(
            self.config,
            self.architecture,
            self.seeds,
            loc.flatten(1, 2),
            scale.flatten(1, 2),
            self.bank,
        )
        updated.update_shape = tuple(y.shape[1:3])
        return updated

    def atom_logits(self, x: Tensor, updates: slice) -> Tensor:
        if not self.update_shape:
            atoms = self.loc.unsqueeze(1) + self.scale.unsqueeze(1) * self.bank
            return self.architecture.logits(x.unsqueeze(1), atoms)
        loc, scale = self.loc[:, updates, None], self.scale[:, updates, None]
        return self.architecture.logits(x[:, None, None], loc + scale * self.bank.unsqueeze(1))

    def _log_likelihood(self, x: Tensor, y: Tensor, theta: Tensor) -> Tensor:
        """Log likelihood of labels `(E, N, T, n)` at inputs `(E, N, n, d)` under parameters
        `(E, N, T, S, P)`, summed over points, shape `(E, N, T, S)`."""
        log_probs = torch.log_softmax(self.architecture.logits(x[:, :, None, None], theta), -1)
        labels = y[:, :, :, None, :, None].expand(*log_probs.shape[:-1], 1)
        return log_probs.gather(-1, labels).squeeze(-1).sum(-1)

    def _noise(self, fitting: bool) -> list[torch.Generator]:
        purpose = ("fit",) if fitting else ("update", *self.update_stream)
        return per_environment(self.seeds, self.name, *purpose, device=self.bank.device)

    @abstractmethod
    def _optimise(
        self, start: Tensor, prior: tuple[Tensor, Tensor], x: Tensor, y: Tensor, fitting: bool
    ) -> tuple[Tensor, Tensor]:
        """The posterior's `(loc, scale)`, each `(E, N, T, P)`, from `loc` at `start` under a prior
        `(E, 1, 1, P)`, given inputs `(E, N, n, d)` and labels `(E, N, T, n)`, at `fit` or update."""


class BBB(MeanField[BBBConfig]):
    """Bayes by backprop: Adam on reparameterised gradients of the ELBO."""

    name = "bbb"

    def _optimise(
        self, start: Tensor, prior: tuple[Tensor, Tensor], x: Tensor, y: Tensor, fitting: bool
    ) -> tuple[Tensor, Tensor]:
        config = self.config
        prior_loc, prior_scale = prior
        # An update starts at the posterior it conditions; a fit starts narrower than the prior.
        start_scale = prior_scale * (config.init_std_factor if fitting else 1.0)
        loc = start.clone().requires_grad_(True)
        log_scale = start_scale.log().expand_as(start).clone().requires_grad_(True)
        lr, steps = (
            (config.lr, config.fit_steps) if fitting else (config.update_lr, config.update_steps)
        )
        optimiser = torch.optim.Adam([loc, log_scale], lr=lr)
        generators = self._noise(fitting)
        for _ in range(steps):
            optimiser.zero_grad()
            eps = draw_normal(generators, config.elbo_samples, loc.shape[-1])[:, None, None]
            theta = loc.unsqueeze(-2) + log_scale.exp().unsqueeze(-2) * eps
            log_likelihood = self._log_likelihood(x, y, theta).mean(-1)
            ratio = (log_scale.exp() / prior_scale).square()
            shift = ((loc - prior_loc) / prior_scale).square()
            kl = (ratio + shift - 1 - ratio.log()).sum(-1) / 2
            (config.kl_weight * kl - log_likelihood).sum().backward()
            optimiser.step()
        return loc.detach(), log_scale.detach().exp()


class IVON(MeanField[IVONConfig]):
    """Improved variational online Newton.

    With the likelihood summed over points, the posterior precision is the prior's `p0` plus a
    Hessian estimate `h`. Each step draws `theta = loc + eps / sqrt(p0 + h)` and, from the
    gradient `g` of the negative log likelihood there,

        h_hat = g * (theta - loc) * (p0 + h)
        g_bar <- beta1 g_bar + (1 - beta1) g
        h     <- beta2 h + (1 - beta2) h_hat + (1 - beta2)^2 (h - h_hat)^2 / (2 (h + p0))
        loc   <- loc - lr (g_bar / (1 - beta1^t) + p0 (loc - loc0)) / (p0 + h).

    The prior enters exactly, as IVON's weight decay generalised to any diagonal Gaussian, and
    `kl_weight` divides the likelihood's gradient and Hessian. `h` is kept from falling below
    zero, so that data never lowers the precision.
    """

    name = "ivon"

    def _optimise(
        self, start: Tensor, prior: tuple[Tensor, Tensor], x: Tensor, y: Tensor, fitting: bool
    ) -> tuple[Tensor, Tensor]:
        config = self.config
        prior_loc, prior_precision = prior[0], prior[1].pow(-2)
        loc = start.clone()
        hessian = torch.full_like(loc, config.hess_init * x.shape[-2] if fitting else 0.0)
        momentum = torch.zeros_like(loc)
        lr, steps = (
            (config.lr, config.fit_steps) if fitting else (config.update_lr, config.update_steps)
        )
        generators = self._noise(fitting)
        for step in range(1, steps + 1):
            precision = prior_precision + hessian
            eps = draw_normal(generators, config.mc_samples, loc.shape[-1])[:, None, None]
            theta = (loc.unsqueeze(-2) + eps * precision.rsqrt().unsqueeze(-2)).requires_grad_(True)
            (gradient,) = torch.autograd.grad(-self._log_likelihood(x, y, theta).sum(), theta)
            gradient = gradient / config.kl_weight
            hessian_hat = (gradient * eps).mean(-2) * precision.sqrt()
            momentum = config.beta1 * momentum + (1 - config.beta1) * gradient.mean(-2)
            hessian = (
                config.beta2 * hessian
                + (1 - config.beta2) * hessian_hat
                + (1 - config.beta2) ** 2 * (hessian - hessian_hat).square() / (2 * precision)
            ).clamp_min(0)
            pull = momentum / (1 - config.beta1**step) + prior_precision * (loc - prior_loc)
            loc = loc - lr * pull / (prior_precision + hessian)
        return loc, (prior_precision + hessian).rsqrt()
