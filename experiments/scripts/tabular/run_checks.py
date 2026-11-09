"""Run the three checks behind statements of Appendix C.2 that the main run cannot answer.

- `replicates`: exact logistic regression on one split of cleveland, run again with fresh
  randomness `REPLICATES` times, to see where its estimate of Ψ is centred.
- `bart_null`: BART and XBART on the first split of each regression dataset, rerun on the
  training set alone with a fresh seed and no block, a "null update", set against the main run's
  updates on the same fit.
- `inducing`: how many inducing inputs the sparse GP takes on each split of each regression
  dataset.

`tune.py` and `run.py` must be run first. Takes about twenty minutes of compute, spread over
`--workers`.

    uv run experiments/scripts/tabular/run_checks.py [--checks C ...] [--workers N]

Writes `experiments/outputs/tabular/checks/<check>.pt`, which `report.py` reads.
"""

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import torch
from run import RUN, SETTING, SPLITS, results_dir, tuning_dir

from pc_experiments.gp import SparseGP
from pc_experiments.outputs import load_run, manifest, output_dir, save_run, timed
from pc_experiments.tabular import (
    REGRESSION,
    Arm,
    fit_arm,
    load,
    make_task,
    result_path,
    run_arm,
    worker_setup,
)
from pc_experiments.tabular.datasets import checksum
from pc_experiments.tabular.experiment import load_tuning
from pc_experiments.tabular.predictives import embed
from pc_experiments.tabular.regression import BART
from predictive_coherence import estimate_coherence

NAME = "tabular"
REPLICATED = ("cleveland", 1, "logistic_exact")  # a dataset, a split and an arm
REPLICATES = 60
NULL_ARMS = ("bart", "xbart")
NULL_ROUNDS = 50
CHECKS = ("replicates", "bart_null", "inducing")


def options(dataset: str, arm: Arm) -> dict[str, Any]:
    return load_tuning(tuning_dir(), dataset)["options"].get(arm.tuned_as, {})


def replicate(index: int) -> dict[str, float]:
    """The replicated job, run again from a stream of its own."""
    dataset, seed, name = REPLICATED
    arm, task = Arm(name), make_task(load(dataset), SETTING.sizes, seed)
    kernels = SETTING.kernels(task, load_tuning(tuning_dir(), dataset)["noise_scale"])[:1]
    values = run_arm(arm, task, options(dataset, arm), kernels, SETTING.num_rounds, f":{index}")
    return {
        "psi": values["psi"][0].item(),
        "psi_se": values["psi_se"][0].item(),
        "c": values["c"][0].sum().item(),
        "v": values["v"][0].sum().item(),
    }


def null_update(job: tuple[str, str]) -> dict[str, Any]:
    """Refit a sampler on the training set alone, `NULL_ROUNDS` times, after the fit the main run
    made on the first split, and measure how far that moves the predictive.

    Returns ‖Δ‖² and V of the null updates and of the main run's real ones, pooled over the test
    inputs under the test's kernel, and the spread of the function at the test inputs within one
    run's draws and of the run's mean across runs.
    """
    dataset, name = job
    arm, task = Arm(name), make_task(load(dataset), SETTING.sizes, 0)
    kernel = SETTING.kernels(task, load_tuning(tuning_dir(), dataset)["noise_scale"])[0]
    # Seeded as the main run seeds this job, so that the fit is the one it measured.
    torch.manual_seed(checksum(f"{task.dataset}:{task.seed}:{arm.name}"))
    model = fit_arm(arm, task, options(dataset, arm))
    assert isinstance(model, BART)
    current = embed(kernel, model.predict(task.x_test))
    null_rounds, run_means = [], []
    for _ in range(NULL_ROUNDS):
        refit = model.update(task.x[:0], task.y[:0])  # no block: the sampler again, a new seed
        assert isinstance(refit, BART)
        null_rounds.append(embed(kernel, refit.predict(task.x_test)))
        run_means.append(refit.draws(task.x_test).mean(-1))
    null = estimate_coherence(current, torch.stack(null_rounds))
    real = load_run(result_path(results_dir(), (dataset, 0, arm)))["values"]
    return {
        "dataset": dataset,
        "arm": name,
        "c_null": null.mmd_sq.item(),
        "v_null": null.v.item(),
        "c_real": real["c"][0].sum().item(),
        "v_real": real["v"][0].sum().item(),
        "within": model.draws(task.x_test).std(-1).square().mean().sqrt().item(),
        "across": torch.stack(run_means).std(0).square().mean().sqrt().item(),
    }


def inducing(job: tuple[str, int]) -> dict[str, Any]:
    """How many inducing inputs the sparse GP takes on one split."""
    dataset, seed = job
    arm, task = Arm("sparse_gp"), make_task(load(dataset), SETTING.sizes, seed)
    torch.manual_seed(checksum(f"{task.dataset}:{task.seed}:{arm.name}"))
    model = fit_arm(arm, task, options(dataset, arm))
    assert isinstance(model, SparseGP)
    return {"dataset": dataset, "seed": seed, "inducing": len(model.z), "train": len(task.x)}


def run_check(check: str, workers: int) -> None:
    """Run one check's jobs, `workers` at a time, and save what they found."""
    jobs: dict[str, tuple[Any, list[Any]]] = {
        "replicates": (replicate, list(range(1, REPLICATES + 1))),
        "bart_null": (null_update, [(d, arm) for d in REGRESSION for arm in NULL_ARMS]),
        "inducing": (inducing, [(d, seed) for d in REGRESSION for seed in range(SPLITS)]),
    }
    function, inputs = jobs[check]
    config = {"check": check, "run": RUN, "replicated": REPLICATED, "replicates": REPLICATES,
              "null_arms": NULL_ARMS, "null_rounds": NULL_ROUNDS}  # fmt: skip
    provenance = manifest(config)
    context = multiprocessing.get_context("spawn")
    with (
        timed(check),
        ProcessPoolExecutor(workers, mp_context=context, initializer=worker_setup) as pool,
    ):
        found = list(pool.map(function, inputs))
    save_run(output_dir(NAME) / "checks", check, {"manifest": provenance, "found": found})


def require(checks: list[str]) -> None:
    """Stop before any work if a check lacks the tuning, or the main run, that it reads."""
    datasets = {REPLICATED[0]} if "replicates" in checks else set()
    if {"bart_null", "inducing"} & set(checks):
        datasets |= set(REGRESSION)
    untuned = sorted(d for d in datasets if not (tuning_dir() / f"{d}.json").exists())
    if untuned:
        raise SystemExit(
            f"Not tuned: {', '.join(untuned)}.\n"
            "Tune them first: uv run experiments/scripts/tabular/tune.py --datasets "
            + " ".join(untuned)
        )
    if "bart_null" in checks:
        unrun = [
            f"{name} on {d}"
            for d in REGRESSION
            for name in NULL_ARMS
            if not result_path(results_dir(), (d, 0, Arm(name))).exists()
        ]
        if unrun:
            raise SystemExit(
                f"No main run on the first split for {', '.join(unrun)}.\n"
                "Run first: uv run experiments/scripts/tabular/run.py --arms " + " ".join(NULL_ARMS)
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checks", nargs="+", default=list(CHECKS), choices=CHECKS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    require(args.checks)
    torch.set_default_dtype(torch.float64)
    for check in args.checks:
        run_check(check, args.workers)


if __name__ == "__main__":
    main()
