"""Deep ensembles and Monte Carlo dropout: point estimates, updated by training on.

An update replays: it carries on training from the current weights on all the data seen, the
new points included. Both are penalised maximum likelihood with an explicit penalty on the squared
weights, which, unlike an adaptive optimiser's weight decay, is the MAP objective of a Gaussian
prior.

An ensemble's atoms are its members (Lakshminarayanan et al., 2017); its full-batch update draws
no randomness. A dropout network's atoms are `A` masks drawn once (Gal and Ghahramani, 2016),
which an update keeps; training draws fresh masks, so its update is random, from a stream shared
by an environment's updates, and `redrawn` gives an independent one.
"""

import copy
from abc import abstractmethod
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from pc_experiments.neural_testbed.environment import Setting
from pc_experiments.neural_testbed.models.base import (
    NetworkMixture,
    draw_uniform,
    per_environment,
)
from pc_experiments.neural_testbed.networks import Architecture


@dataclass(frozen=True)
class EnsembleConfig:
    """
    Attributes:
        num_members: the atoms.
        weight_decay: coefficient of the sum of squared weights.
        lr, fit_steps: Adam at `fit`.
        update_lr, update_steps: Adam at `update`.
    """

    num_members: int
    weight_decay: float
    lr: float
    fit_steps: int
    update_lr: float
    update_steps: int


@dataclass(frozen=True)
class DropoutConfig:
    """
    Attributes:
        rate: probability with which a hidden unit is dropped.
        num_atoms: masks the predictive mixes over.
        weight_decay, lr, fit_steps, update_lr, update_steps: as for `EnsembleConfig`.
    """

    rate: float
    num_atoms: int
    weight_decay: float
    lr: float
    fit_steps: int
    update_lr: float
    update_steps: int


class Replay[Config: (EnsembleConfig, DropoutConfig)](NetworkMixture):
    """Networks that keep their training data and are updated by training on.

    Attributes:
        theta: parameters with a trailing `(..., P)`, led by `(E,)`, or by `(E, U)` once updated.
        x, y: the data seen when fitted, shapes `(E, n, d)` and `(E, n)`.
    """

    name: str

    def __init__(
        self,
        config: Config,
        architecture: Architecture,
        seeds: tuple[int, ...],
        x: Tensor,
        y: Tensor,
    ) -> None:
        self.config = config
        self.architecture = architecture
        self.seeds = seeds
        self.x, self.y = x, y
        self.theta: Tensor

    def _train(self, theta: Tensor, x: Tensor, y: Tensor, fitting: bool) -> Tensor:
        """Train parameters `(E, N, T, A, P)`, for `A` members trained at once, on inputs
        `(E, N, n, d)` and labels `(E, N, T, n)`."""
        config = self.config
        theta = theta.clone().requires_grad_(True)
        lr, steps = (
            (config.lr, config.fit_steps) if fitting else (config.update_lr, config.update_steps)
        )
        optimiser = torch.optim.Adam([theta], lr=lr)
        is_weight = self.architecture.is_weight.to(theta.device)
        purpose = ("fit",) if fitting else ("update", *self.update_stream)
        generators = per_environment(self.seeds, self.name, *purpose, device=theta.device)
        for _ in range(steps):
            optimiser.zero_grad()
            log_probs = torch.log_softmax(self._training_logits(x, theta, generators), dim=-1)
            labels = y[..., None, :, None]  # one for all members, shape `(E, N, T, 1, n, 1)`
            log_likelihood = log_probs.gather(-1, labels.expand(*log_probs.shape[:-1], 1)).sum()
            penalty = config.weight_decay * (theta.square() * is_weight).sum()
            (penalty - log_likelihood).backward()
            optimiser.step()
        return theta.detach()

    @abstractmethod
    def _training_logits(
        self, x: Tensor, theta: Tensor, generators: list[torch.Generator]
    ) -> Tensor:
        """Logits `(E, N, T, A, n, K)` for one step of training."""

    def update(self, x: Tensor, y: Tensor) -> Self:
        num_batches, num_labellings = y.shape[1:3]
        # Everything seen so far, and then the batch.
        inputs = torch.cat([self.x.unsqueeze(1).expand(-1, num_batches, -1, -1), x], dim=2)
        labels = torch.cat(
            [self.y[:, None, None].expand(-1, num_batches, num_labellings, -1), y], dim=3
        )
        start = self.theta[:, None, None].expand(
            -1, num_batches, num_labellings, *self.theta.shape[1:]
        )
        updated = copy.copy(self)
        updated.theta = self._train(start, inputs, labels, fitting=False).flatten(1, 2)
        updated.update_shape = (num_batches, num_labellings)
        return updated


class Ensemble(Replay[EnsembleConfig]):
    """Independently initialised networks trained on the same data. `theta`: `(E, [U,] M, P)`."""

    name = "ensemble"

    @classmethod
    def fit(
        cls,
        config: EnsembleConfig,
        setting: Setting,
        x: Tensor,
        y: Tensor,
        seeds: tuple[int, ...],
    ) -> Self:
        """Fit to inputs `(E, n, d)` and labels `(E, n)`."""
        architecture = setting.architecture
        model = cls(config, architecture, seeds, x, y)
        model.num_atoms = config.num_members
        start = torch.stack(
            [
                architecture.sample_gaussian_prior(config.num_members, rng)
                for rng in per_environment(seeds, cls.name, "init")
            ]
        ).to(x.device)
        model.theta = model._train(start[:, None, None], x.unsqueeze(1), y[:, None, None], True)[
            :, 0, 0
        ]
        return model

    def _training_logits(
        self, x: Tensor, theta: Tensor, generators: list[torch.Generator]
    ) -> Tensor:
        return self.architecture.logits(x[:, :, None, None], theta)

    def atom_logits(self, x: Tensor, updates: slice) -> Tensor:
        if not self.update_shape:
            return self.architecture.logits(x.unsqueeze(1), self.theta)
        return self.architecture.logits(x[:, None, None], self.theta[:, updates])


class Dropout(Replay[DropoutConfig]):
    """One network under `A` fixed masks of its hidden units. `theta`: `(E, [U,] P)`.

    Attributes:
        masks: for each hidden layer, what each atom multiplies its units by, shape `(E, A, H)`:
            zero for a dropped unit and `1 / (1 - rate)` for a kept one.
    """

    name = "dropout"
    random_update = True
    masks: list[Tensor]

    @classmethod
    def fit(
        cls,
        config: DropoutConfig,
        setting: Setting,
        x: Tensor,
        y: Tensor,
        seeds: tuple[int, ...],
    ) -> Self:
        """Fit to inputs `(E, n, d)` and labels `(E, n)`."""
        architecture = setting.architecture
        model = cls(config, architecture, seeds, x, y)
        model.num_atoms = config.num_atoms
        generators = per_environment(seeds, cls.name, "atoms", device=x.device)
        model.masks = model._draw_masks(generators, config.num_atoms)
        start = torch.stack(
            [
                architecture.sample_gaussian_prior(1, rng)[0]
                for rng in per_environment(seeds, cls.name, "init")
            ]
        ).to(x.device)
        model.theta = model._train(start[:, None, None], x.unsqueeze(1), y[:, None, None], True)[
            :, 0, 0
        ]
        return model

    def _draw_masks(self, generators: list[torch.Generator], num: int) -> list[Tensor]:
        """For each hidden layer, `num` masks for each environment, shape `(E, num, H)`."""
        keep = 1 - self.config.rate
        hidden = self.architecture.hidden_width
        uniform = draw_uniform(generators, self.architecture.num_hidden_layers, num, hidden)
        return list(((uniform < keep).to(uniform.dtype) / keep).unbind(1))

    def _training_logits(
        self, x: Tensor, theta: Tensor, generators: list[torch.Generator]
    ) -> Tensor:
        masks = [mask[:, None, None, None] for mask in self._draw_masks(generators, x.shape[-2])]
        return self.architecture.logits(x[:, :, None, None], theta.unsqueeze(-2), masks)

    def atom_logits(self, x: Tensor, updates: slice) -> Tensor:
        if not self.update_shape:
            masks = [mask.unsqueeze(2) for mask in self.masks]
            return self.architecture.logits(x.unsqueeze(1), self.theta.unsqueeze(1), masks)
        masks = [mask[:, None, :, None] for mask in self.masks]
        return self.architecture.logits(x[:, None, None], self.theta[:, updates, None], masks)
