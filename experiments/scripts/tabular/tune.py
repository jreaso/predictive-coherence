"""Choose each arm's hyperparameters for the tabular experiment, by how well it predicts.

A prerequisite of the run scripts, launched by hand and never by them. Hyperparameters are
chosen once for each dataset, on the training set of its first split, by the log likelihood of
held-out folds, and then kept for every split and every update. Nothing here looks at
incoherence. Arms whose hyperparameters are fitted by marginal likelihood, and TabPFN, which has
none, are not tuned.

The dataset's noise scale is recorded too: the smallest cross-validated RMSE among the regression
arms, in units of the training outputs' spread. The kernel on outputs takes its lengthscale from
it, so that it resolves the predictives it compares.

    uv run experiments/scripts/tabular/tune.py [--datasets D ...] [--workers N] [--overwrite]

What is chosen for a dataset is written to `experiments/outputs/tabular/tuning/<dataset>.json`.
"""

import argparse
import itertools
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import torch
from run import CLASSIFICATION_ARMS, DATASETS, REGRESSION_ARMS, SIZES, tuning_dir

from pc_experiments.outputs import manifest, timed
from pc_experiments.tabular import (
    CLASSIFICATION,
    Arm,
    cross_validate,
    load,
    make_task,
    worker_setup,
)

# Values tried for each arm, every combination of them. Kernel widths are given as multiples of
# `1 / d`, the scale of the squared distance between standardised inputs in `d` dimensions.
GRIDS: dict[str, dict[str, tuple[Any, ...]]] = {
    "kernel_ridge": {"alpha": (0.001, 0.01, 0.1, 1.0), "gamma": (0.03, 0.1, 0.3, 1.0, 3.0)},
    "logistic": {"prior_sd": (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)},
    "neighbours": {"n_neighbors": (1, 2, 3, 5, 8, 12, 20, 30)},
    "catboost": {"depth": (2, 4, 6), "iterations": (100, 300)},
    # These fit their own hyperparameters or keep their defaults; they are here to be scored, for
    # the noise scale.
    "gp": {},
    "bart": {},
    "xbart": {},
}


def tune(dataset: str) -> dict[str, Any]:
    """The options with the best held-out log likelihood for each tuned arm of a dataset."""
    task = make_task(load(dataset), SIZES, seed=0)
    arms = CLASSIFICATION_ARMS if dataset in CLASSIFICATION else REGRESSION_ARMS
    options: dict[str, dict[str, Any]] = {}
    scores: dict[str, dict[str, float]] = {}
    for kind in dict.fromkeys(arm.tuned_as for arm in arms):
        if kind not in GRIDS:
            continue
        grid = GRIDS[kind]
        tried: list[tuple[dict[str, float], dict[str, Any]]] = []
        for values in itertools.product(*grid.values()):  # an empty grid has one empty candidate
            candidate = dict(zip(grid, values, strict=True))
            if "gamma" in candidate:
                candidate["gamma"] /= task.x.shape[1]
            torch.manual_seed(0)
            tried.append((cross_validate(Arm(kind), task, candidate), candidate))
        scores[kind], options[kind] = min(tried, key=lambda pair: pair[0]["nll"])
        print(f"{dataset} {kind}: {options[kind]} nll {scores[kind]['nll']:.3f}", flush=True)
    noise_scale = None if task.num_classes else min(s["error"] for s in scores.values())
    return {"options": options, "scores": scores, "noise_scale": noise_scale}


def tune_and_save(dataset: str) -> None:
    provenance = manifest({"dataset": dataset, "grids": GRIDS, "sizes": vars(SIZES)})
    chosen = {"manifest": provenance, **tune(dataset)}
    tuning_dir().mkdir(parents=True, exist_ok=True)
    (tuning_dir() / f"{dataset}.json").write_text(json.dumps(chosen, indent=2) + "\n")


def main() -> None:
    parsed = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parsed.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parsed.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parsed.add_argument("--overwrite", action="store_true", help="tune datasets already tuned")
    args = parsed.parse_args()

    datasets = [
        d for d in args.datasets if args.overwrite or not (tuning_dir() / f"{d}.json").exists()
    ]
    context = multiprocessing.get_context("spawn")
    with (
        timed("Tuning"),
        ProcessPoolExecutor(args.workers, mp_context=context, initializer=worker_setup) as pool,
    ):
        list(pool.map(tune_and_save, datasets))


if __name__ == "__main__":
    main()
