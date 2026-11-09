"""Coherence and accuracy of regression models on functions drawn from a Gaussian process.

An environment is one function, observed at a pool of inputs; a task is a subsample of the pool
into a context, a block and evaluation points. Every arm is fitted on the context, scored, and
updated on blocks of its own outputs for rounds shared by the estimate and the test.
"""

import math
import multiprocessing
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache, partial
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.distributions import kl_divergence

from pc_experiments.gp import (
    ExactGP,
    GaussianModel,
    ReselectedSparseGP,
    SparseGP,
    gaussian_coherence,
    greedy_variance,
    sample_inputs,
)
from pc_experiments.gp.prior import Prior
from pc_experiments.inctnp import IncTNPModel, load_network
from pc_experiments.outputs import load_run
from predictive_coherence import (
    CoherenceEstimate,
    CoherenceEstimator,
    PredictiveModel,
    RFFKernel,
    SequentialTest,
    iter_rounds,
)

# ----- settings, arms and tasks -------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """Environments, and how each model is measured on them.

    Attributes:
        prior: what the models assume, and the incTNP was trained on.
        truth: what the environments are drawn from; `prior` when well specified.
        num_pool: inputs each environment is observed at, from which tasks are subsampled.
        num_design: points in a block.
        estimates_at: rounds after which the estimate of Ψ is also recorded.
        psi_lengthscale, psi_features: the random Fourier kernel on outputs.
    """

    prior: Prior
    truth: Prior
    input_range: tuple[float, float]
    num_pool: int
    num_context: int
    num_design: int
    num_eval: int
    num_rounds: int
    estimates_at: tuple[int, ...]
    psi_lengthscale: float
    psi_features: int


LADDERS = ("sparse", "reselected")  # the kinds of arm that come in rungs of inducing inputs


@dataclass(frozen=True)
class Arm:
    """A model: `exact`, `inctnp`, or a sparse Gaussian process with `num_inducing` inducing
    inputs, `sparse` holding them fixed through an update and `reselected` choosing them afresh
    from the enlarged data."""

    kind: str
    num_inducing: int = 0

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.num_inducing}" if self.kind in LADDERS else self.kind


@dataclass(frozen=True)
class Environment:
    """One function from the truth, observed at a pool of inputs."""

    x: Tensor  # (num_pool, 1)
    y: Tensor  # (num_pool,)


@dataclass(frozen=True)
class Task:
    """A context, a block, and evaluation points. The block's true outputs are used only for
    accuracy after an update: the rounds draw the block's outputs from the model."""

    x: Tensor  # (num_context, 1)
    y: Tensor  # (num_context,)
    x_design: Tensor  # (num_design, 1)
    x_eval: Tensor  # (num_eval, 1)
    y_eval: Tensor  # (num_eval,)
    y_design: Tensor  # (num_design,)


def sample_environment(setting: Setting, seed: int) -> Environment:
    generator = torch.Generator().manual_seed(seed)
    # Inputs before outputs, so that settings differing only in their truth share their inputs.
    x = sample_inputs((setting.num_pool,), setting.input_range, generator)
    return Environment(x, setting.truth.sample(x, generator))


def sample_task(setting: Setting, environment: int, index: int) -> Task:
    """Subsample `index` of environment `environment`, disjoint and without replacement. The split
    does not depend on the truth."""
    pool = sample_environment(setting, environment)
    # A checksum rather than `hash`, which Python salts differently in every process.
    generator = torch.Generator().manual_seed(zlib.crc32(f"{environment}:{index}".encode()))
    sizes = [setting.num_context, setting.num_design, setting.num_eval]
    order = torch.randperm(setting.num_pool, generator=generator)
    context, design, held_out = order[: sum(sizes)].split(sizes)
    return Task(
        pool.x[context],
        pool.y[context],
        pool.x[design],
        pool.x[held_out],
        pool.y[held_out],
        pool.y[design],
    )


def check_training(checkpoint: dict[str, Any], setting: Setting) -> None:
    """Raise unless the incTNP was trained on the setting's prior and on long enough contexts."""
    tasks = checkpoint["tasks"]
    trained = (tasks["prior"], tuple(tasks["input_range"]))
    if trained != (vars(setting.prior), tuple(setting.input_range)):
        raise ValueError(f"the incTNP was trained on {trained}, not on this setting's prior")
    # Every point of a sequence is predicted from the points before it.
    trained_context, needed = tasks["length"] - 1, setting.num_context + setting.num_design
    if trained_context < needed:
        raise ValueError(
            f"the incTNP was trained on contexts of up to {trained_context} points, "
            f"but this setting needs {needed}"
        )


# ----- running ------------------------------------------------------------------------------------


@cache
def _network(checkpoint: Path):
    """Loaded once in each worker."""
    return load_network(checkpoint)


def _fit(arm: Arm, task: Task, setting: Setting, checkpoint: Path | None) -> PredictiveModel:
    if arm.kind == "exact":
        return ExactGP(setting.prior, task.x, task.y)
    if arm.kind == "sparse":
        # A rung near the size of the context takes every input distinct enough to induce on,
        # which on a few contexts is one or two fewer than it asks for.
        z = greedy_variance(setting.prior, task.x, arm.num_inducing)
        return SparseGP.fit(setting.prior, z, task.x, task.y)
    if arm.kind == "reselected":
        return ReselectedSparseGP(setting.prior, task.x, task.y, arm.num_inducing)
    if arm.kind == "inctnp" and checkpoint is not None:
        return IncTNPModel.fit(_network(checkpoint), task.x, task.y)
    raise ValueError(f"cannot fit {arm}")


def run_task(
    setting: Setting, arms: Sequence[Arm], checkpoint: Path | None, index: tuple[int, int]
) -> dict[str, Tensor]:
    """Measure every arm on subsample `index[1]` of environment `index[0]`, shapes `(arms, ...)`.

    `nll`: of the held-out outputs; `excess_nll`: less the true predictive's.
    `kl`: from the true predictive, averaged over evaluation inputs.
    `kl_updated`: the same after an update on the block's true outputs, against the true
        predictive given the context and the block.
    `c`, `v`, `psi`, `psi_se`: estimates pooled over evaluation inputs.
    `psi_at`, `psi_se_at`: the same after each of `setting.estimates_at` rounds.
    `c_exact`, `v_exact`: closed forms, NaN for an arm that is not Gaussian.
    `log_wealth`: the test's after each round. `round_seconds`: wall time per round.
    """
    environment, subsample = index
    task = sample_task(setting, environment, subsample)
    kernel = RFFKernel(setting.psi_lengthscale, setting.psi_features)
    truth = ExactGP(setting.truth, task.x, task.y).predict(task.x_eval)
    with_block = torch.cat([task.x, task.x_design]), torch.cat([task.y, task.y_design])
    truth_after = ExactGP(setting.truth, *with_block).predict(task.x_eval)
    true_nll = -truth.log_prob(task.y_eval).mean()
    per_arm = []
    for arm in arms:
        # Each arm's randomness depends on the task and the arm, not on which arms ran before.
        torch.manual_seed(zlib.crc32(f"{environment}:{subsample}:{arm.name}".encode()))
        model = _fit(arm, task, setting, checkpoint)
        predictive = model.predict(task.x_eval)
        nll = -predictive.log_prob(task.y_eval).mean()

        estimator, test = CoherenceEstimator(), SequentialTest()
        at: list[CoherenceEstimate] = []
        start = time.perf_counter()
        rounds = iter_rounds(model, kernel, task.x_design, task.x_eval)
        for done, round_ in enumerate(islice(rounds, setting.num_rounds), start=1):
            estimator.update(*round_)
            test.update(round_.mu_p, round_.mu_r)
            if done in setting.estimates_at:
                at.append(estimator.estimate())
        round_seconds = (time.perf_counter() - start) / setting.num_rounds
        estimate = estimator.estimate()

        # After the rounds, so that their randomness is untouched: the update is deterministic.
        updated = model.update(task.x_design, task.y_design).predict(task.x_eval)
        kl_updated = kl_divergence(truth_after, updated).mean()

        c_exact = v_exact = torch.tensor(torch.nan)
        if isinstance(model, GaussianModel):
            c_exact, v_exact = (
                t.sum() for t in gaussian_coherence(model, kernel, task.x_design, task.x_eval)
            )
        per_arm.append(
            {
                "nll": nll,
                "excess_nll": nll - true_nll,
                "kl": kl_divergence(truth, predictive).mean(),
                "kl_updated": kl_updated,
                "c": estimate.mmd_sq,
                "v": estimate.v,
                "psi": estimate.psi,
                "psi_se": estimate.standard_error(),
                "psi_at": torch.stack([e.psi for e in at]),
                "psi_se_at": torch.stack([e.standard_error() for e in at]),
                "c_exact": c_exact,
                "v_exact": v_exact,
                "log_wealth": torch.tensor(test.log_wealth),
                "round_seconds": torch.tensor(round_seconds),
            }
        )
    return {key: torch.stack([result[key] for result in per_arm]) for key in per_arm[0]}


def run_setting(
    setting: Setting,
    arms: Sequence[Arm],
    checkpoint: Path | None,
    num_environments: int,
    num_subsamples: int,
    workers: int,
) -> dict[str, Tensor]:
    """`run_task` on `num_subsamples` tasks of each environment, `workers` at a time. Shapes
    `(arms, environments, subsamples, ...)`."""
    indices = [(e, s) for e in range(num_environments) for s in range(num_subsamples)]
    run = partial(run_task, setting, arms, checkpoint)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_worker) as pool:
        per_task = list(pool.map(run, indices))
    stacked = {key: torch.stack([result[key] for result in per_task], dim=1) for key in per_task[0]}
    return {
        key: values.unflatten(1, (num_environments, num_subsamples))
        for key, values in stacked.items()
    }


def _worker() -> None:
    """One thread and float64 in every worker, which inherits neither."""
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)


# ----- reading results back -----------------------------------------------------------------------


@dataclass(frozen=True)
class Results:
    """A saved run. `values[key]` has shape `(arms, environments, subsamples, ...)`, and
    `training` holds the incTNP's `steps` and final `kl`."""

    arms: list[Arm]
    config: dict[str, Any]
    training: dict[str, float]
    values: dict[str, Tensor]

    @classmethod
    def load(cls, path: Path) -> "Results":
        run = load_run(path)
        arms = [Arm(**arm) for arm in run["arms"]]
        return cls(arms, run["manifest"]["config"], run["training"], run["results"])

    def of(self, key: str, arm: Arm) -> Tensor:
        """One arm's values, shape `(environments, subsamples, ...)`."""
        return self.values[key][self.arms.index(arm)]

    def rungs(self, kind: str = "sparse") -> list[Arm]:
        """One ladder's arms, by number of inducing inputs."""
        return sorted((a for a in self.arms if a.kind == kind), key=lambda a: a.num_inducing)

    def along_ladder(self, key: str, kind: str = "sparse") -> Tensor:
        """Every rung's values, `(environments, subsamples, rungs)`."""
        return torch.stack([self.of(key, arm) for arm in self.rungs(kind)], dim=-1)

    def rejection_round(self, arm: Arm, alpha: float) -> Tensor:
        """The round each task's test first rejected at, or NaN, `(environments, subsamples)`."""
        rejected = self.of("log_wealth", arm) >= -math.log(alpha)
        first = rejected.double().argmax(-1) + 1.0  # `argmax` takes the first of equal values
        return torch.where(rejected.any(-1), first, torch.nan)

    def rejection_rate(self, arm: Arm, alpha: float) -> Tensor:
        """The share of tasks rejected by each round, `(num_rounds,)`."""
        first = self.rejection_round(arm, alpha).flatten()
        rounds = torch.arange(1, self.values["log_wealth"].shape[-1] + 1)
        return (first.unsqueeze(-1) <= rounds).double().mean(0)
