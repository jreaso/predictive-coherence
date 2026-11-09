"""What the testbed needs of a model, and what mixtures over networks share.

A model is a batch of classifiers: one per environment `E` once fitted, and one per environment,
candidate `N` and labelling `T` once updated. Its joint over a batch's labels is part of the model,
since incoherence is measured against it. An update is a deterministic function of the model, the
data and, if it is random, one draw from the model's own stream; `redrawn` gives an independent one.
"""

import copy
import math
from abc import ABC, abstractmethod
from typing import ClassVar, Self

import torch
from torch import Tensor

from pc_experiments.neural_testbed.environment import generator
from pc_experiments.neural_testbed.networks import DTYPE, Architecture
from pc_experiments.neural_testbed.scoring import mixture_log_joint

# Largest intermediate tensor a prediction may build, in elements: a gibibyte in float64.
MAX_ELEMENTS = 2**27


class Model(ABC):
    """Classifiers for a chunk of `E` environments, all fitted or all updated.

    Attributes:
        update_shape: `()` for fitted models, `(N, T)` for updated ones.
        random_update: whether an update draws randomness of its own, beyond what the fit drew.
        update_stream: keys added to an update's stream: none for the one decisions use.
    """

    update_shape: tuple[int, ...] = ()
    random_update: ClassVar[bool] = False
    update_stream: tuple[str, ...] = ()

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """Log class probabilities at inputs `(E, m, d)`: `(E, m, K)` from fitted models,
        `(E, N, T, m, K)` from updated ones."""

    @abstractmethod
    def log_joint(self, x: Tensor, labellings: Tensor) -> Tensor:
        """Log probability `(E, N, Z)` of each labelling `(Z, k)` of each batch `(E, N, k, d)`,
        under a fitted model's own joint."""

    @abstractmethod
    def update(self, x: Tensor, y: Tensor) -> Self:
        """Fitted models updated on each labelling `(E, N, T, k)` of each batch `(E, N, k, d)`,
        with `update_shape` `(N, T)`."""

    def redrawn(self, stream: str) -> Self:
        """These fitted models, with their updates' randomness drawn from the stream `stream`,
        independent of the one they draw from by default. Themselves if an update draws none."""
        if not self.random_update:
            return self
        other = copy.copy(self)
        other.update_stream = (stream,)
        return other


def point_chunks(num_points: int, elements_per_point: int) -> list[slice]:
    """Slices of the inputs, each within `MAX_ELEMENTS` at `elements_per_point` apiece."""
    size = max(1, MAX_ELEMENTS // max(1, elements_per_point))
    return [slice(start, start + size) for start in range(0, num_points, size)]


def log_mixture(atom_log_probs: Tensor, log_weights: Tensor | None = None) -> Tensor:
    """Log of the mixture `(..., m, K)` of atoms' class probabilities `(..., A, m, K)`, with log
    weights `(..., A)` or equal ones."""
    if log_weights is None:
        return torch.logsumexp(atom_log_probs, dim=-3) - math.log(atom_log_probs.shape[-3])
    return torch.logsumexp(atom_log_probs + log_weights[..., None, None], dim=-3)


class Mixture(Model):
    """A mixture over `A` fixed atoms, each labelling inputs independently. The joint draws one
    atom, so its marginals are the predictive exactly and it can be enumerated."""

    @abstractmethod
    def atom_log_probs(self, x: Tensor) -> Tensor:
        """Each atom's log class probabilities `(E, A, m, K)` at inputs `(E, m, d)`."""

    def log_weights(self) -> Tensor | None:
        """Log weight of each atom under fitted models, shape `(E, A)`. `None` if equal."""
        return None

    def log_joint(self, x: Tensor, labellings: Tensor) -> Tensor:
        num_batches, size = x.shape[1:3]
        log_probs = self.atom_log_probs(x.flatten(1, 2)).unflatten(-2, (num_batches, size))
        log_weights = self.log_weights()
        if log_weights is None:
            num_atoms = log_probs.shape[1]
            log_weights = log_probs.new_full((1, 1, num_atoms), -math.log(num_atoms))
        else:
            log_weights = log_weights.unsqueeze(1)
        # Atoms lead the batches in `log_probs`, and must trail them for the mixture.
        return mixture_log_joint(log_weights, log_probs.transpose(1, 2), labellings)


class NetworkMixture(Mixture):
    """An equal mixture over `A` networks of one architecture, the atoms of a posterior."""

    architecture: Architecture
    num_atoms: int

    @abstractmethod
    def atom_logits(self, x: Tensor, updates: slice) -> Tensor:
        """Every atom's logits at inputs `(E, m, d)`: `(E, A, m, K)` from fitted models, or
        `(E, u, A, m, K)` for the flattened `updates`."""

    def predict(self, x: Tensor) -> Tensor:
        if not self.update_shape:
            return self._predict(x, slice(None))
        num_updates = math.prod(self.update_shape)
        per_update = self.num_atoms * x.shape[0] * self.architecture.hidden_width
        # As many updates at a time as fit, each over as many inputs as fit.
        size = max(1, MAX_ELEMENTS // (per_update * x.shape[-2]))
        blocks = [
            self._predict(x, slice(start, start + size)) for start in range(0, num_updates, size)
        ]
        return torch.cat(blocks, dim=1).unflatten(1, self.update_shape)

    def _predict(self, x: Tensor, updates: slice) -> Tensor:
        """Shape `(E, m, K)` from fitted models, `(E, u, m, K)` from updated ones."""
        num_updates = len(range(*updates.indices(math.prod(self.update_shape))))
        per_point = x.shape[0] * num_updates * self.num_atoms * self.architecture.hidden_width
        chunks = [
            log_mixture(torch.log_softmax(self.atom_logits(x[:, c], updates), dim=-1))
            for c in point_chunks(x.shape[-2], per_point)
        ]
        return torch.cat(chunks, dim=-2)

    def atom_log_probs(self, x: Tensor) -> Tensor:
        return torch.log_softmax(self.atom_logits(x, slice(None)), dim=-1)


def per_environment(
    seeds: tuple[int, ...], *keys: object, device: torch.device | str = "cpu"
) -> list[torch.Generator]:
    """One generator on `device` per environment, so that an environment's results do not
    depend on its chunk. A seed gives different values on different devices."""
    return [generator(seed, *keys, device=device) for seed in seeds]


def draw_normal(generators: list[torch.Generator], *shape: int) -> Tensor:
    """Standard normal draws of shape `(E, *shape)`, row `e` from generator `e`."""
    return torch.stack(
        [torch.randn(*shape, generator=g, dtype=DTYPE, device=g.device) for g in generators]
    )


def draw_uniform(generators: list[torch.Generator], *shape: int) -> Tensor:
    """Uniform draws on `[0, 1)` of shape `(E, *shape)`, row `e` from generator `e`."""
    return torch.stack(
        [torch.rand(*shape, generator=g, dtype=DTYPE, device=g.device) for g in generators]
    )
