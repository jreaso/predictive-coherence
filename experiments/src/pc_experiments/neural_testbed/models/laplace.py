"""Linearised Laplace approximations, with full, Kronecker-factored and diagonal curvature.

`fit` finds a MAP estimate `theta0` and the generalised Gauss-Newton matrix there; from then on
the model is the linearised one,

    f(x; theta) = f(x; theta0) + J(x) (theta - theta0),    theta ~ N(mu, Sigma),

and the linearisation point never moves. Its predictive is an equal mixture over `A` atoms
`theta_a = mu + Sigma^(1/2) eps_a`, with the standard normal `eps_a` drawn once and shared by the
current predictive and every updated one.

An update conditions the linearised model on new points alone, with the current Gaussian as the
prior: Newton's method to the mode, and a Laplace approximation there. For a batch with stacked
Jacobian `U` of `r = k K` rows the mode lies in the span of `Sigma U^T`, so Newton runs on `r`
coordinates `beta`, with `S = U Sigma U^T`:

    mu'     = mu + Sigma U^T beta
    Lambda' = Lambda + U^T W U,    W the softmax curvature at the mode.

`full` absorbs `U^T W U` exactly, as a low-rank downdate of `Sigma^(1/2)`; `diag` keeps its
diagonal; `kfac` adds it to per-layer Kronecker factors.
"""

import copy
import math
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from pc_experiments.neural_testbed.environment import Setting
from pc_experiments.neural_testbed.models.base import (
    MAX_ELEMENTS,
    Mixture,
    draw_normal,
    log_mixture,
    per_environment,
    point_chunks,
)
from pc_experiments.neural_testbed.networks import DTYPE, Architecture, batched_eigh

CURVATURES = ("full", "kfac", "diag")

# Newton steps per update, damped by backtracking: a fixed number, which reaches the mode of the
# convex objective to rounding.
NEWTON_STEPS = 16
HALVINGS = 30
JITTER = 1e-12


@dataclass(frozen=True)
class LaplaceConfig:
    """
    Attributes:
        curvature: `full`, `kfac` or `diag`.
        prior_scale: multiplies the standard deviations of the environments' prior, which
            suits the models wider since they lack the environments' temperature.
        num_atoms: `A`.
        lr, fit_steps: Adam, for the MAP estimate.
    """

    curvature: str
    prior_scale: float
    num_atoms: int
    lr: float
    fit_steps: int


def softmax_curvature(log_probs: Tensor) -> Tensor:
    """`diag(p) - p p^T`, shape `(..., K, K)`, from log probabilities `(..., K)`."""
    p = log_probs.exp()
    return torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)


def psd_sqrt(matrix: Tensor) -> Tensor:
    """Symmetric square root of positive semi-definite matrices `(..., n, n)`."""
    values, vectors = batched_eigh(matrix)
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.mT


def inverse_cholesky_transpose(matrix: Tensor) -> Tensor:
    """`R^{-T}` for the Cholesky factor `R` of `matrix`: a square root of its inverse."""
    factor = torch.linalg.cholesky(matrix)
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    return torch.linalg.solve_triangular(factor, identity.expand_as(matrix), upper=False).mT


class Laplace(Mixture):
    """Linearised Laplace models for a chunk of environments. An updated model keeps the fitted
    covariance and adds what each update changed.

    Attributes:
        theta0: the linearisation point, which is also the fitted mean, shape `(E, P)`.
        bank: the standard normal draws behind the atoms, shape `(E, A, P)`.
        root: `full`: a square root of the fitted covariance, shape `(E, P, P)`.
        sigma: `diag`: fitted standard deviations, shape `(E, P)`.
        a, g, seen: `kfac`: for each layer, sums of outer products of its inputs, shape
            `(E, in + 1, in + 1)`, and of backpropagated curvature, shape `(E, out, out)`, over
            the `seen` points fitted on.
        damping: `kfac`: what the prior adds to each layer's factors.
        projection: `Sigma^(T/2) U^T` for each batch updated on, shape `(E, N, P, r)`.
        beta: the Newton coordinates of each updated mean, shape `(E, N, T, r)`.
        mix: `full`: the downdate of each update in those coordinates, shape `(E, N, T, r, r)`.
        new_sigma: `diag`: updated standard deviations, shape `(E, N, T, P)`.
        new_a, new_g: `kfac`: updated sums, shapes `(E, N, in + 1, in + 1)` and
            `(E, N, T, out, out)`, over `seen` and a batch's points.
    """

    root: Tensor
    sigma: Tensor
    a: list[Tensor]
    g: list[Tensor]
    seen: int
    damping: list[float]
    projection: Tensor
    beta: Tensor
    mix: Tensor
    new_sigma: Tensor
    new_a: list[Tensor]
    new_g: list[Tensor]

    def __init__(
        self, config: LaplaceConfig, architecture: Architecture, theta0: Tensor, bank: Tensor
    ) -> None:
        if config.curvature not in CURVATURES:
            raise ValueError(f"curvature must be one of {CURVATURES}, not {config.curvature!r}")
        self.config = config
        self.architecture = architecture
        self.theta0 = theta0
        self.bank = bank

    # ----- fitting --------------------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        config: LaplaceConfig,
        setting: Setting,
        x: Tensor,
        y: Tensor,
        seeds: tuple[int, ...],
        stream: str = "laplace",
    ) -> Self:
        """Fit to inputs `(E, n, d)` and labels `(E, n)`. The three curvatures share their MAP
        estimate and atoms; a model fitted for another purpose names a `stream` of its own."""
        architecture = setting.architecture
        prior_std = (config.prior_scale * architecture.prior_std).to(x.device)
        start = torch.stack(
            [
                architecture.sample_gaussian_prior(1, rng)[0]
                for rng in per_environment(seeds, stream, "init")
            ]
        ).to(x.device)
        theta0 = _map_estimate(architecture, start, prior_std, x, y, config)
        bank = draw_normal(
            per_environment(seeds, stream, "atoms", device=x.device),
            config.num_atoms,
            architecture.num_parameters,
        )
        model = cls(config, architecture, theta0, bank)

        logits, jacobian = architecture.jacobian(x, theta0)
        curvature = softmax_curvature(torch.log_softmax(logits, dim=-1))  # (E, n, K, K)
        precision = prior_std.pow(-2)
        if config.curvature == "full":
            half = (psd_sqrt(curvature) @ jacobian).flatten(1, 2)
            model.root = inverse_cholesky_transpose(half.mT @ half + torch.diag(precision))
        elif config.curvature == "diag":
            ggn = torch.einsum("enkp,enkl,enlp->ep", jacobian, curvature, jacobian)
            model.sigma = (ggn + precision).rsqrt()
        else:
            model.a, model.g = model._kronecker_sums(x, curvature)
            model.seen = x.shape[1]
            # The prior enters a layer's factors as damping, at its mean precision over the layer.
            model.damping, offset = [], 0
            for fan_in, fan_out in architecture.layers:
                size = fan_in * fan_out + fan_out
                model.damping.append(math.sqrt(precision[offset : offset + size].mean().item()))
                offset += size
        return model

    def _kronecker_sums(self, x: Tensor, curvature: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """Each layer's Kronecker factors summed over inputs `(E, [N, k,] n, d)` with curvature
        `(E, [N, T,] n, K, K)`: `a` `(E, [N,] in + 1, in + 1)` and `g` `(E, [N, T,] out, out)`.

        A point's curvature in a layer is exactly `a a^T (x) B^T W B`, with `a` the layer's input
        and a one, and `B` the logits' derivative in its pre-activations; the approximation takes
        the product of the sums for the sum of the products.
        """
        theta = self.theta0 if x.dim() == 3 else self.theta0.unsqueeze(1)
        _, jacobian = self.architecture.jacobian(x, theta)
        a_sums, g_sums, offset, h = [], [], 0, x
        layers = self.architecture.unflatten(theta)
        for index, ((fan_in, fan_out), (weight, bias)) in enumerate(
            zip(self.architecture.layers, layers, strict=True)
        ):
            augmented = torch.cat([h, torch.ones_like(h[..., :1])], dim=-1)
            a_sums.append(augmented.mT @ augmented)
            # The derivative in a layer's biases, which close its block, is `B` itself.
            offset += fan_in * fan_out + fan_out
            backprop = jacobian[..., offset - fan_out : offset]
            if curvature.dim() > backprop.dim():
                backprop = backprop.unsqueeze(-4)  # shared by a batch's labellings
            g_sums.append((backprop.mT @ curvature @ backprop).sum(-3))
            if index < len(layers) - 1:
                h = torch.relu(h @ weight.mT + bias.unsqueeze(-2))
        return a_sums, g_sums

    # ----- the covariance's square root ----------------------------------------------------------

    def _kronecker_roots(
        self, a: list[Tensor], g: list[Tensor], seen: int
    ) -> list[tuple[Tensor, Tensor]]:
        """For each layer `(P_l, Q_l)` with `P_l P_l^T` and `Q_l Q_l^T` the inverses of its damped
        factors, which share the `1 / seen` of the product of sums between them."""
        roots = []
        for a_l, g_l, damping in zip(a, g, self.damping, strict=True):
            eye_a = torch.eye(a_l.shape[-1], dtype=DTYPE, device=a_l.device)
            eye_g = torch.eye(g_l.shape[-1], dtype=DTYPE, device=g_l.device)
            roots.append(
                (
                    inverse_cholesky_transpose(a_l / math.sqrt(seen) + damping * eye_a),
                    inverse_cholesky_transpose(g_l / math.sqrt(seen) + damping * eye_g),
                )
            )
        return roots

    def _by_layer(self, vectors: Tensor) -> list[Tensor]:
        """Parameter vectors `(..., P)` as each layer's matrix `(..., out, in + 1)`, biases last."""
        return [
            torch.cat([weight, bias.unsqueeze(-1)], dim=-1)
            for weight, bias in self.architecture.unflatten(vectors)
        ]

    @staticmethod
    def _flat(layers: list[Tensor]) -> Tensor:
        """The inverse of `_by_layer`."""
        pieces = []
        for layer in layers:
            pieces += [layer[..., :-1].flatten(-2), layer[..., -1]]
        return torch.cat(pieces, dim=-1)

    def _root_times(self, vectors: Tensor, transpose: bool = False) -> Tensor:
        """`Sigma^(1/2) v`, or `Sigma^(T/2) v`, for each row `v` of `(E, ..., m, P)`, under the
        fitted covariance."""
        lead = (slice(None), *[None] * (vectors.dim() - 3))
        if self.config.curvature == "full":
            # Rows times the transpose of what would multiply columns.
            return vectors @ (self.root if transpose else self.root.mT)[lead]
        if self.config.curvature == "diag":
            return vectors * self.sigma[(*lead, None)]
        out = []
        roots = self._kronecker_roots(self.a, self.g, self.seen)
        for (p, q), layer in zip(roots, self._by_layer(vectors), strict=True):
            p, q = p[(*lead, None)], q[(*lead, None)]
            out.append(q.mT @ layer @ p if transpose else q @ layer @ p.mT)
        return self._flat(out)

    # ----- the fitted model's predictive ---------------------------------------------------------

    def atom_log_probs(self, x: Tensor) -> Tensor:
        # The fitted mean is the linearisation point, so an atom's logits are those at `theta0`
        # plus `J Sigma^(1/2) eps_a`.
        logits, jacobian = self.architecture.jacobian(x, self.theta0)
        shifts = torch.einsum("emkp,eap->eamk", jacobian, self._root_times(self.bank))
        return torch.log_softmax(logits.unsqueeze(1) + shifts, dim=-1)

    # ----- updating -------------------------------------------------------------------------------

    def update(self, x: Tensor, y: Tensor) -> Self:
        size, num_classes = x.shape[2], self.architecture.num_classes
        logits, jacobian = self.architecture.jacobian(x, self.theta0.unsqueeze(1))
        design = jacobian.flatten(2, 3)  # (E, N, r, P): the stacked Jacobian U
        projection = self._root_times(design, transpose=True).mT
        gram = projection.mT @ projection
        gram = gram + JITTER * torch.eye(gram.shape[-1], dtype=DTYPE, device=x.device)

        one_hot = torch.nn.functional.one_hot(y, num_classes).to(DTYPE)  # (E, N, T, k, K)
        beta = _newton(logits, gram, one_hot)
        mode = logits.unsqueeze(2) + (beta @ gram).unflatten(-1, (size, num_classes))
        mode_log_probs = torch.log_softmax(mode, dim=-1)  # (E, N, T, k, K)

        updated = copy.copy(self)
        updated.update_shape = tuple(y.shape[1:3])
        updated.projection, updated.beta = projection, beta
        if self.config.curvature == "full":
            # Sigma' = L (I - B B^T) L^T with L the fitted root, B = C kappa, C the projection and
            # kappa kappa^T = (I + W S)^{-1} W in its symmetric form. A square root of it is
            # L (I - C mix C^T), where `mix` follows from the eigenvalues of B^T B.
            gram = gram.unsqueeze(2)
            w_half = _block_diagonal(psd_sqrt(softmax_curvature(mode_log_probs)))
            eye = torch.eye(gram.shape[-1], dtype=DTYPE, device=x.device)
            kappa = w_half @ psd_sqrt(torch.linalg.inv(eye + w_half @ gram @ w_half))
            values, vectors = batched_eigh(kappa.mT @ gram @ kappa)
            values = values.clamp(0, 1)
            gamma = torch.where(
                values > 1e-12, (1 - (1 - values).sqrt()) / values.clamp_min(1e-12), 0.5
            )
            updated.mix = kappa @ (vectors * gamma.unsqueeze(-2)) @ vectors.mT @ kappa.mT
        elif self.config.curvature == "diag":
            # The diagonal of U^T W U with W = diag(p) - p p^T, one input of the batch at a time.
            p = mode_log_probs.exp()
            precision = self.sigma.pow(-2)[:, None, None].expand(*beta.shape[:-1], -1).clone()
            for j in range(size):
                rows = jacobian[:, :, j]  # (E, N, K, P)
                precision += p[..., j, :] @ rows.square() - (p[..., j, :] @ rows).square()
            updated.new_sigma = precision.rsqrt()
        else:
            a, g = self._kronecker_sums(x, softmax_curvature(mode_log_probs))
            updated.new_a = [old.unsqueeze(1) + new for old, new in zip(self.a, a, strict=True)]
            updated.new_g = [old[:, None, None] + new for old, new in zip(self.g, g, strict=True)]
        return updated

    # ----- an updated model's predictive ---------------------------------------------------------

    def predict(self, x: Tensor) -> Tensor:
        num_envs, num_points = x.shape[:2]
        if not self.update_shape:
            # The Jacobian, `(E, m, K, P)`, is the largest thing formed.
            per_point = num_envs * self.architecture.num_classes * self.architecture.num_parameters
            chunks = [x[:, chunk] for chunk in point_chunks(num_points, per_point)]
            return torch.cat([log_mixture(self.atom_log_probs(c)) for c in chunks], dim=1)
        num_batches, num_labellings = self.update_shape
        num_atoms, num_classes = self.config.num_atoms, self.architecture.num_classes
        # `full` works in the batch's few coordinates. The others form every atom's parameters.
        per_labelling = num_envs * num_atoms * self.architecture.num_parameters
        labellings = (
            num_labellings
            if self.config.curvature == "full"
            else max(1, MAX_ELEMENTS // per_labelling)
        )
        points = max(1, MAX_ELEMENTS // (num_envs * labellings * num_atoms * num_classes))

        rows: list[list[Tensor]] = [[] for _ in range(num_batches)]
        for start in range(0, num_points, points):
            logits, jacobian = self.architecture.jacobian(x[:, start : start + points], self.theta0)
            jacobian = jacobian.flatten(1, 2)  # (E, m K, P)
            through_root = self._root_times(jacobian, transpose=True)  # J Sigma^(1/2)
            for batch in range(num_batches):
                blocks = [
                    self._predict(logits, jacobian, through_root, batch, slice(t, t + labellings))
                    for t in range(0, num_labellings, labellings)
                ]
                rows[batch].append(torch.cat(blocks, dim=1))
        return torch.stack([torch.cat(row, dim=-2) for row in rows], dim=1)

    def _predict(
        self, logits: Tensor, jacobian: Tensor, through_root: Tensor, batch: int, chosen: slice
    ) -> Tensor:
        """Log probabilities `(E, t, m, K)` of the models updated on some labellings of a batch."""
        projection, beta = self.projection[:, batch], self.beta[:, batch, chosen]
        # The mean moves by Sigma U^T beta = Sigma^(1/2) C beta.
        across = through_root @ projection  # (E, m K, r)
        mean = logits.flatten(1, 2).unsqueeze(1) + beta @ across.mT  # (E, t, m K)
        if self.config.curvature == "full":
            correction = self.mix[:, batch, chosen] @ (projection.mT @ self.bank.mT).unsqueeze(1)
            shifts = (through_root @ self.bank.mT).unsqueeze(1) - torch.einsum(
                "emr,etra->etma", across, correction
            )
        elif self.config.curvature == "diag":
            atoms = self.bank.unsqueeze(1) * self.new_sigma[:, batch, chosen].unsqueeze(2)
            shifts = torch.einsum("emp,etap->etma", jacobian, atoms)
        else:
            roots = self._kronecker_roots(
                [a_l[:, batch] for a_l in self.new_a],
                [g_l[:, batch, chosen] for g_l in self.new_g],
                self.seen + self.projection.shape[-1] // self.architecture.num_classes,
            )
            atoms = self._flat(
                [
                    q.unsqueeze(2) @ layer.unsqueeze(1) @ p.mT[:, None, None]
                    for (p, q), layer in zip(roots, self._by_layer(self.bank), strict=True)
                ]
            )
            shifts = torch.einsum("emp,etap->etma", jacobian, atoms)
        atom_logits = (mean.unsqueeze(-1) + shifts).mT.unflatten(-1, logits.shape[-2:])
        return log_mixture(torch.log_softmax(atom_logits, dim=-1))


def _map_estimate(
    architecture: Architecture,
    start: Tensor,
    prior_std: Tensor,
    x: Tensor,
    y: Tensor,
    config: LaplaceConfig,
) -> Tensor:
    """Maximise the log likelihood, summed over points, plus the Gaussian log prior."""
    theta = start.clone().requires_grad_(True)
    optimiser = torch.optim.Adam([theta], lr=config.lr)
    for _ in range(config.fit_steps):
        optimiser.zero_grad()
        log_probs = torch.log_softmax(architecture.logits(x, theta), dim=-1)
        log_likelihood = log_probs.gather(-1, y.unsqueeze(-1)).sum()
        penalty = (theta / prior_std).square().sum() / 2
        (penalty - log_likelihood).backward()
        optimiser.step()
    return theta.detach()


def _block_diagonal(blocks: Tensor) -> Tensor:
    """Blocks `(..., k, K, K)` along the diagonal of matrices `(..., k K, k K)`."""
    size, num_classes = blocks.shape[-3:-1]
    out = blocks.new_zeros(*blocks.shape[:-3], size * num_classes, size * num_classes)
    for j in range(size):
        span = slice(j * num_classes, (j + 1) * num_classes)
        out[..., span, span] = blocks[..., j, :, :]
    return out


def _newton(logits: Tensor, gram: Tensor, one_hot: Tensor) -> Tensor:
    """The mode of each update, in the coordinates `beta` of `mu' = mu + Sigma U^T beta`.

    Minimises `beta^T S beta / 2 - sum_j log softmax(logits + S beta)_j[y_j]`, whose gradient is
    `S (beta + softmax - onehot)`, by Newton steps `-(I + W S)^{-1} (beta + softmax - onehot)`.
    Steps are halved until they decrease the objective enough, since a full step can overshoot
    into a saturated softmax. From the current logits `(..., k, K)`, `S` `(..., r, r)` and one-hot
    labels `(..., T, k, K)`; shape `(..., T, r)`.
    """
    shape = one_hot.shape[-2:]
    logits = logits.unsqueeze(-3)
    identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    blocks = gram.unflatten(-2, shape).unsqueeze(-4)  # (..., 1, k, K, r)

    def objective(beta: Tensor) -> Tensor:
        moved = beta @ gram
        log_probs = torch.log_softmax(logits + moved.unflatten(-1, shape), dim=-1)
        return (moved * beta).sum(-1) / 2 - (log_probs * one_hot).sum((-1, -2))

    beta = one_hot.new_zeros(*one_hot.shape[:-2], gram.shape[-1])
    for _ in range(NEWTON_STEPS):
        log_probs = torch.log_softmax(logits + (beta @ gram).unflatten(-1, shape), dim=-1)
        residual = beta + (log_probs.exp() - one_hot).flatten(-2)
        curvature_gram = (softmax_curvature(log_probs) @ blocks).flatten(-3, -2)
        direction = torch.linalg.solve(identity + curvature_gram, residual.unsqueeze(-1))
        direction = direction.squeeze(-1)
        slope = ((residual @ gram) * direction).sum(-1).clamp_min(0)
        current = objective(beta)
        step = torch.ones_like(current)
        enough = torch.zeros_like(current, dtype=torch.bool)
        for _ in range(HALVINGS):
            moved = objective(beta - step.unsqueeze(-1) * direction)
            enough = enough | (moved <= current - 1e-4 * step * slope)
            if enough.all():
                break
            step = torch.where(enough, step, step / 2)
        beta = beta - step.unsqueeze(-1) * direction
    return beta
