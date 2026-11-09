"""Environments: a network drawn from a prior, whose class probabilities everything is scored
against, with inputs from four Gaussian clusters at `(+-c, +-c)`."""

import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from pc_experiments.neural_testbed.networks import DTYPE, Architecture

NUM_CLUSTERS = 4

# Classes are balanced over a grid on `[-BALANCE_LIMIT, BALANCE_LIMIT]^2`, covering the clusters,
# by a fixed number of proportional-fitting steps.
BALANCE_LIMIT = 6.0
BALANCE_RESOLUTION = 48
BALANCE_ITERATIONS = 60


@dataclass(frozen=True)
class Setting:
    """How environments are drawn, and the acquisition decision posed in each.

    Attributes:
        architecture: the network an environment is, which the models share.
        temperature: divides the logits; small values give sharp class boundaries.
        center: `c` in the cluster centres `(-c, -c), (-c, c), (c, -c), (c, c)`, in that order.
        train_std, pool_std, target_std: spread of each role's inputs; test inputs are drawn as
            target inputs.
        train_clusters, target_clusters: the clusters those roles use; the pool uses all four.
        min_class_mass: an environment is valid if every class has at least this share of the
            expected labels at the test inputs.
    """

    architecture: Architecture
    temperature: float
    center: float
    train_std: float
    pool_std: float
    target_std: float
    train_clusters: tuple[int, ...]
    target_clusters: tuple[int, ...]
    num_train: int
    num_pool: int
    num_target: int
    num_test: int
    num_candidates: int
    candidate_size: int
    min_class_mass: float


def generator(seed: int, *keys: object, device: torch.device | str = "cpu") -> torch.Generator:
    """A generator for one purpose within one seed, from a checksum of the seed and the keys, so
    that how much one purpose draws leaves the others alone."""
    digest = hashlib.blake2b(":".join(map(str, (seed, *keys))).encode(), digest_size=8).digest()
    return torch.Generator(device).manual_seed(int.from_bytes(digest, "big") >> 1)


def class_log_probs(
    architecture: Architecture, x: Tensor, theta: Tensor, offset: Tensor, temperature: float
) -> Tensor:
    """`log_softmax((f(x) - offset) / temperature)` for inputs `(..., n, d)`, parameters
    `(..., P)` and offsets `(..., K)`, shape `(..., n, K)`."""
    logits = architecture.logits(x, theta) - offset.unsqueeze(-2)
    return torch.log_softmax(logits / temperature, dim=-1)


def balance_grid(device: torch.device | str = "cpu") -> Tensor:
    """The inputs over which classes are balanced, shape `(BALANCE_RESOLUTION^2, 2)`."""
    axis = torch.linspace(
        -BALANCE_LIMIT, BALANCE_LIMIT, BALANCE_RESOLUTION, dtype=DTYPE, device=device
    )
    return torch.cartesian_prod(axis, axis)


def balance_offsets(logits: Tensor, temperature: float) -> Tensor:
    """Offsets `(..., K)`, summing to zero, under which `softmax((logits - o) / temperature)`
    averages `1 / K` for every class over the inputs of `logits` `(..., n, K)`, by proportional
    fitting. An offset is a shift of the output biases, so the network stays in the architecture.
    """
    num_classes = logits.shape[-1]
    offset = torch.zeros_like(logits[..., 0, :])
    for _ in range(BALANCE_ITERATIONS):
        mass = torch.softmax((logits - offset.unsqueeze(-2)) / temperature, dim=-1).mean(-2)
        offset = offset + temperature * (mass.clamp_min(1e-300).log() + math.log(num_classes))
    return offset - offset.mean(-1, keepdim=True)


@dataclass(frozen=True)
class Split:
    """Inputs in one role `(n, 2)`, labels drawn once from the truth `(n,)`, the true log class
    probabilities `(n, K)`, and each input's cluster `(n,)`. A chunk adds a leading `E`."""

    x: Tensor
    y: Tensor
    log_p: Tensor
    cluster: Tensor

    def __len__(self) -> int:
        return self.x.shape[-2]


@dataclass(frozen=True)
class Environment:
    """A network, parameters `(P,)` and offsets `(K,)`, with the inputs of each role: `train`,
    `pool` for candidates, `target` for objectives and incoherence, and `test` for scoring. A chunk
    stacks environments, with one seed each."""

    seeds: tuple[int, ...]
    theta: Tensor
    offset: Tensor
    train: Split
    pool: Split
    target: Split
    test: Split

    @property
    def seed(self) -> int:
        (seed,) = self.seeds
        return seed


def cluster_centers(center: float) -> Tensor:
    """Shape `(4, 2)`."""
    c = center
    return torch.tensor([[-c, -c], [-c, c], [c, -c], [c, c]], dtype=DTYPE)


def sample_inputs(
    num: int, clusters: Sequence[int], center: float, std: float, rng: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Inputs `(num, 2)` from an equal mixture of Gaussians at `clusters`, and each one's cluster."""
    cluster = torch.tensor(clusters)[torch.randint(len(clusters), (num,), generator=rng)]
    noise = torch.randn(num, 2, generator=rng, dtype=DTYPE)
    return cluster_centers(center)[cluster] + std * noise, cluster


def sample_environment(setting: Setting, seed: int) -> Environment:
    """The environment of `seed`. Each role has its own stream, so that the size of one does
    not change another."""
    architecture = setting.architecture
    theta = architecture.sample_prior(1, generator(seed, "network"))[0]
    offset = balance_offsets(architecture.logits(balance_grid(), theta), setting.temperature)

    def split(role: str, num: int, clusters: Sequence[int], std: float) -> Split:
        x, cluster = sample_inputs(num, clusters, setting.center, std, generator(seed, role, "x"))
        log_p = class_log_probs(architecture, x, theta, offset, setting.temperature)
        y = torch.multinomial(log_p.exp(), 1, generator=generator(seed, role, "y")).squeeze(-1)
        return Split(x, y, log_p, cluster)

    everywhere = tuple(range(NUM_CLUSTERS))
    return Environment(
        seeds=(seed,),
        theta=theta,
        offset=offset,
        train=split("train", setting.num_train, setting.train_clusters, setting.train_std),
        pool=split("pool", setting.num_pool, everywhere, setting.pool_std),
        target=split("target", setting.num_target, setting.target_clusters, setting.target_std),
        test=split("test", setting.num_test, setting.target_clusters, setting.target_std),
    )


def class_masses(environment: Environment) -> Tensor:
    """Expected share of each class among labels at the test inputs, shape `(K,)`."""
    return environment.test.log_p.exp().mean(-2)


def is_valid(environment: Environment, setting: Setting) -> bool:
    """Whether every class is common enough for the environment to be a `K`-class problem."""
    return bool(class_masses(environment).min() >= setting.min_class_mass)


def valid_environments(setting: Setting, first_seed: int = 0) -> Iterator[Environment]:
    """Valid environments by increasing seed, from `first_seed`. Invalid ones are skipped, not
    repaired."""
    seed = first_seed
    while True:
        environment = sample_environment(setting, seed)
        if is_valid(environment, setting):
            yield environment
        seed += 1


def stack(environments: Sequence[Environment], device: torch.device | str = "cpu") -> Environment:
    """A chunk: every tensor stacked along a new leading dimension, on `device`."""

    def tensors(values: Sequence[Tensor]) -> Tensor:
        return torch.stack(list(values)).to(device)

    def split(role: str) -> Split:
        splits = [getattr(environment, role) for environment in environments]
        return Split(
            *(tensors([getattr(s, f) for s in splits]) for f in Split.__dataclass_fields__)
        )

    return Environment(
        seeds=tuple(environment.seed for environment in environments),
        theta=tensors([environment.theta for environment in environments]),
        offset=tensors([environment.offset for environment in environments]),
        **{role: split(role) for role in ("train", "pool", "target", "test")},
    )
