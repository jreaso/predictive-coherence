"""Coherence of predictive models on real tabular datasets.

An arm is fitted on a split's training set, scored on its test set, and updated in each round on
a block of pool inputs with outputs drawn from itself; the rounds are shared by the estimate and
the test. Each arm's result on each split is saved on its own, so a run picks up where it stopped.
"""

import dataclasses
import json
import math
import multiprocessing
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from threadpoolctl import threadpool_limits
from torch import Tensor
from torch.distributions import Categorical, Distribution

from pc_experiments.gp import ExactGP, SparseGP, fit_ard_prior, to_noise_level
from pc_experiments.outputs import load_run, manifest, save_run
from pc_experiments.tabular import classification, regression
from pc_experiments.tabular.datasets import Sizes, Task, checksum, load, make_task
from pc_experiments.tabular.predictives import embed
from pc_experiments.tabular.refitted import Autoregressive
from pc_experiments.tabular.tabpfn import TabPFN
from predictive_coherence import (
    DiracKernel,
    Kernel,
    PredictiveModel,
    RFFKernel,
    SequentialTest,
    estimate_coherence,
)

# ----- arms ---------------------------------------------------------------------------------------

REFITTED_REGRESSORS = {
    "bart": regression.BART,
    "xbart": regression.XBART,
    "kernel_ridge": regression.KernelRidge,
    "neighbours": regression.NearestNeighbours,
    "catboost": regression.CatBoost,
}
# The probability a label a model rules out is charged at, rather than an infinite log loss.
PROBABILITY_FLOOR = 1e-16

# What each approximate logistic arm keeps of the posterior's covariance.
LOGISTIC_COVARIANCE = {
    "logistic_laplace": "full",
    "logistic_diagonal": "diagonal",
    "logistic": "none",
}


@dataclass(frozen=True)
class Arm:
    """A model, by kind. An `autoregressive` arm draws a block's outputs one input at a time, each
    from the model updated on those before it (`refitted.Autoregressive`)."""

    kind: str
    autoregressive: bool = False

    @property
    def name(self) -> str:
        return self.kind + ("_autoregressive" if self.autoregressive else "")

    @property
    def tuned_as(self) -> str:
        return "logistic" if self.kind.startswith("logistic") else self.kind


def fit_arm(arm: Arm, task: Task, options: dict[str, Any]) -> PredictiveModel:
    """Fit an arm to a task's training set, with its tuned `options`."""
    model = _fit_model(arm, task, options)
    return Autoregressive(model) if arm.autoregressive else model


def _fit_model(arm: Arm, task: Task, options: dict[str, Any]) -> PredictiveModel:
    x, y, num_classes = task.x, task.y, task.num_classes
    if arm.kind in ("tabpfn", "tabpfn_preprocessing"):
        return TabPFN(x, y, num_classes, preprocessing=arm.kind == "tabpfn_preprocessing")
    if num_classes:
        if arm.kind == "logistic_exact":
            return classification.ExactLogistic.fit(x, y, num_classes, **options)
        if arm.kind in LOGISTIC_COVARIANCE:
            covariance = LOGISTIC_COVARIANCE[arm.kind]
            return classification.ApproximateLogistic(
                x, y, num_classes, covariance=covariance, **options
            )
        if arm.kind in classification.LEARNERS:
            return classification.RefittedClassifier(arm.kind, x, y, num_classes, **options)
    elif arm.kind in REFITTED_REGRESSORS:
        return REFITTED_REGRESSORS[arm.kind](x, y, **options)
    elif arm.kind in ("gp", "gp_refit", "sparse_gp"):
        # Hyperparameters by marginal likelihood, shared by the three.
        prior = fit_ard_prior(x, y)
        if arm.kind == "gp":
            return ExactGP(prior, x, y)
        if arm.kind == "gp_refit":
            return regression.RefitGP(prior, x, y)
        return SparseGP.fit(prior, to_noise_level(prior, x), x, y)
    raise ValueError(f"no arm of kind {arm.kind!r} for {task.dataset}")


def score(predictive: Distribution, y: Tensor) -> dict[str, Tensor]:
    """Mean negative log likelihood of outputs `(m,)`, and RMSE or error rate."""
    if isinstance(predictive, Categorical):
        probs = predictive.probs.clamp_min(PROBABILITY_FLOOR)
        nll = -probs.gather(-1, y.unsqueeze(-1)).log().mean()
        return {"nll": nll, "error": (probs.argmax(-1) != y).double().mean()}
    error = (predictive.mean - y).square().mean().sqrt()
    return {"nll": -predictive.log_prob(y).mean(), "error": error}


def cross_validate(
    arm: Arm, task: Task, options: dict[str, Any], num_folds: int = 5
) -> dict[str, float]:
    """An arm's `score` on held-out folds of the training set, averaged over the folds."""
    generator = torch.Generator().manual_seed(checksum(f"{task.dataset}:{task.seed}:folds"))
    folds = torch.tensor_split(torch.randperm(len(task.x), generator=generator), num_folds)
    scores: list[dict[str, Tensor]] = []
    for index, held_out in enumerate(folds):
        kept = torch.cat([fold for other, fold in enumerate(folds) if other != index])
        model = fit_arm(arm, dataclasses.replace(task, x=task.x[kept], y=task.y[kept]), options)
        scores.append(score(model.predict(task.x[held_out]), task.y[held_out]))
    return {key: torch.stack([s[key] for s in scores]).mean().item() for key in scores[0]}


# ----- running ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """How every arm is measured.

    Attributes:
        num_features: random Fourier frequencies in the kernel on real outputs.
        lengthscale_multiples: that kernel's lengthscales in noise scales; the test uses the first.
        fixed_lengthscales: further lengthscales, in units of the training outputs' spread.
    """

    sizes: Sizes
    num_rounds: int
    num_features: int
    lengthscale_multiples: tuple[float, ...]
    fixed_lengthscales: tuple[float, ...]

    def kernels(self, task: Task, noise_scale: float | None) -> list[Kernel]:
        """The kernels on a task's outputs: one for labels, one per lengthscale for real outputs."""
        if task.num_classes:
            return [DiracKernel(task.num_classes)]
        assert noise_scale is not None
        scales = [m * noise_scale for m in self.lengthscale_multiples] + [*self.fixed_lengthscales]
        return [RFFKernel(scale, self.num_features) for scale in scales]


def updated_predictives(
    model: PredictiveModel, blocks: Tensor, x: Tensor
) -> Iterator[Distribution]:
    """The predictive at `x` `(m, d)` after an update on each block `(R, k, d)`, with outputs drawn
    from the model."""
    if isinstance(model, TabPFN):
        return model.updated(blocks, x)
    return (model.update(block, model.joint_sample(block)).predict(x) for block in blocks)


def run_arm(
    arm: Arm,
    task: Task,
    options: dict[str, Any],
    kernels: Sequence[Kernel],
    num_rounds: int,
    stream: str = "",
) -> dict[str, Tensor]:
    """Fit, score and measure one arm on one task. Its randomness is seeded by the task and the
    arm, or afresh with a `stream`, for repeats of one job.

    Returns:
        `nll`, `error`: `score` on the test outputs.
        `c`, `v`: estimates at each test input under each kernel, shape `(kernels, M)`.
        `psi`, `psi_se`: the ratio pooled over test inputs and its standard error, `(kernels,)`.
        `log_wealth`: the sequential test's under the first kernel, after each round, `(R,)`.
        `fit_seconds`, `round_seconds`: wall time to fit, and per round.

    The embeddings of every round are held until the end, since the estimates leave each round
    out in turn: about `8 * kernels * num_rounds * M * dim` bytes, which is a gigabyte at the
    largest setting here.
    """
    torch.manual_seed(checksum(f"{task.dataset}:{task.seed}:{arm.name}{stream}"))
    start = time.perf_counter()
    model = fit_arm(arm, task, options)
    current = model.predict(task.x_test)
    fit_seconds = time.perf_counter() - start

    mu_p = [embed(kernel, current) for kernel in kernels]
    mu_r: list[list[Tensor]] = [[] for _ in kernels]
    test = SequentialTest()
    start = time.perf_counter()
    for predictive in updated_predictives(model, task.blocks(num_rounds), task.x_test):
        for rounds, kernel in zip(mu_r, kernels, strict=True):
            rounds.append(embed(kernel, predictive))
        test.update(mu_p[0], mu_r[0][-1])
    round_seconds = (time.perf_counter() - start) / num_rounds

    # One kernel at a time, discarding its rounds once they are summarised: the embeddings are
    # `kernels * num_rounds * M * dim` numbers, and stacking them all at once would double that.
    estimates = []
    for current_embedding, rounds in zip(mu_p, mu_r, strict=True):
        estimates.append(estimate_coherence(current_embedding, torch.stack(rounds)))
        rounds.clear()
    return {
        **score(current, task.y_test),
        "c": torch.stack([e.mmd_sq_per_point for e in estimates]),
        "v": torch.stack([e.v_per_point for e in estimates]),
        "psi": torch.stack([e.psi for e in estimates]),
        "psi_se": torch.stack([e.standard_error() for e in estimates]),
        "log_wealth": torch.tensor(test.log_wealth),
        "fit_seconds": torch.tensor(fit_seconds),
        "round_seconds": torch.tensor(round_seconds),
    }


def load_tuning(directory: Path, dataset: str) -> dict[str, Any]:
    """What `tune.py` chose for a dataset: `options` for each tuned arm, and the `noise_scale`."""
    return json.loads((directory / f"{dataset}.json").read_text())


Job = tuple[str, int, Arm]  # a dataset, the seed of a split, and an arm


def result_path(directory: Path, job: Job) -> Path:
    dataset, seed, arm = job
    return directory / dataset / f"seed{seed}" / f"{arm.name}.pt"


def job_config(setting: Setting, tuning: Path, job: Job) -> dict[str, Any]:
    """What a job's result depends on besides the code."""
    dataset, seed, arm = job
    tuned = load_tuning(tuning, dataset)
    return {
        "dataset": dataset,
        "seed": seed,
        **dataclasses.asdict(setting),
        "noise_scale": tuned["noise_scale"],
        "options": tuned["options"].get(arm.tuned_as, {}),
    }


def run_job(setting: Setting, tuning: Path, directory: Path, job: Job) -> str:
    """Run one arm on one split, save it, and return a line for the log."""
    dataset, seed, arm = job
    config = job_config(setting, tuning, job)
    provenance = manifest(config)
    task = make_task(load(dataset), setting.sizes, seed)
    kernels = setting.kernels(task, config["noise_scale"])
    values = run_arm(arm, task, config["options"], kernels, setting.num_rounds)
    path = result_path(directory, job)
    save_run(
        path.parent,
        path.stem,
        {"manifest": provenance, "arm": dataclasses.asdict(arm), "values": values},
    )
    return (
        f"{dataset} seed {seed} {arm.name}: psi {values['psi'][0]:.4f} "
        f"± {values['psi_se'][0]:.4f}, {values['round_seconds']:.2f} s per round"
    )


def is_saved(setting: Setting, tuning: Path, directory: Path, job: Job) -> bool:
    """Whether this job has a result made under the same `job_config`, so that changing the
    setting or the tuning reruns it."""
    manifest_path = result_path(directory, job).with_suffix(".json")
    if not manifest_path.exists():
        return False
    saved = json.loads(manifest_path.read_text())["config"]
    # As it was written, with tuples as lists.
    return saved == json.loads(json.dumps(job_config(setting, tuning, job)))


def run_jobs(
    setting: Setting,
    tuning: Path,
    directory: Path,
    jobs: Sequence[Job],
    workers: int = 1,
    overwrite: bool = False,
) -> None:
    """Run the jobs not already saved, or all of them with `overwrite`, `workers` at a time."""
    pending = [job for job in jobs if overwrite or not is_saved(setting, tuning, directory, job)]
    if len(pending) < len(jobs):
        print(f"{len(jobs) - len(pending)} of {len(jobs)} already saved; running the other "
              f"{len(pending)}", flush=True)  # fmt: skip
    jobs = pending
    run = partial(_attempt, setting, tuning, directory)
    lines = []
    if workers <= 1:
        worker_setup()
        for job in jobs:
            lines.append(run(job))
            print(lines[-1], flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context, initializer=worker_setup) as pool:
            for line in pool.map(run, jobs):
                lines.append(line)
                print(line, flush=True)
    failed = [line for line in lines if FAILED in line]
    if failed:
        print(f"\n{len(failed)} of {len(jobs)} failed and were not saved:", flush=True)
        print("\n".join(f"  {line}" for line in failed), flush=True)
        print("Running again retries them and leaves the rest alone.", flush=True)


FAILED = "FAILED"


def _attempt(setting: Setting, tuning: Path, directory: Path, job: Job) -> str:
    """Run a job, turning a failure into a line of the log so the other jobs carry on. A missing
    tuning file still raises, since every job on that dataset would fail."""
    try:
        return run_job(setting, tuning, directory, job)
    except FileNotFoundError:
        raise
    except Exception as error:  # noqa: BLE001 - what one arm does must not stop the rest
        dataset, seed, arm = job
        return f"{dataset} seed {seed} {arm.name}: {FAILED}, {type(error).__name__}: {error}"


def worker_setup() -> None:
    """One thread and float64 in a worker, which inherits neither."""
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    threadpool_limits(1)  # scikit-learn's learners thread through OpenMP


# ----- reading results back -----------------------------------------------------------------------


@dataclass(frozen=True)
class Results:
    """The saved results of a run, one per dataset, split and arm.

    Attributes:
        values: from `(dataset, arm name)` to what `run_arm` returned on each split it was run
            on, by the split's seed. Arms need not have been run on the same splits.
    """

    values: dict[tuple[str, str], dict[int, dict[str, Tensor]]]

    @classmethod
    def load(cls, directory: Path, datasets: Sequence[str] | None = None) -> "Results":
        """Every result under `directory`, with its datasets in alphabetical order, or those of
        `datasets` alone, in their order, if they are given."""
        values: dict[tuple[str, str], dict[int, dict[str, Tensor]]] = {}
        if datasets is None:
            datasets = sorted(path.name for path in directory.glob("*") if path.is_dir())
        paths = [path for d in datasets for path in sorted(directory.glob(f"{d}/seed*/*.pt"))]
        for path in paths:
            arm = values.setdefault((path.parent.parent.name, path.stem), {})
            arm[int(path.parent.name.removeprefix("seed"))] = load_run(path)["values"]
        return cls(values)

    def datasets(self) -> list[str]:
        """The datasets with results, in the order they were read."""
        return list(dict.fromkeys(dataset for dataset, _ in self.values))

    def of(self, key: str, dataset: str, arm: str) -> Tensor:
        """One arm's values on one dataset, by increasing seed, shape `(splits, ...)`."""
        splits = self.values[dataset, arm]
        return torch.stack([splits[seed][key] for seed in sorted(splits)])

    def arms(self, dataset: str) -> list[str]:
        return [arm for other, arm in self.values if other == dataset]

    def psi(self, dataset: str, arm: str, kernel: int = 0) -> tuple[float, float]:
        """Pooled ratio under one kernel: its mean over splits, and the standard error of that
        mean, or of the one estimate where there is a single split."""
        psi = self.of("psi", dataset, arm)[:, kernel]
        if len(psi) == 1:
            return psi.item(), self.of("psi_se", dataset, arm)[0, kernel].item()
        return psi.mean().item(), (psi.std() / math.sqrt(len(psi))).item()

    def rejection_round(self, dataset: str, arm: str, alpha: float) -> Tensor:
        """The round at which each split's test first rejected, NaN if it never did: `(splits,)`."""
        rejected = self.of("log_wealth", dataset, arm) >= -math.log(alpha)
        first = rejected.double().argmax(-1) + 1.0  # `argmax` takes the first of equal values
        return torch.where(rejected.any(-1), first, torch.nan)
