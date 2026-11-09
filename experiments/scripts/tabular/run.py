"""Run the tabular experiment: incoherence of many kinds of model on real datasets.

The design lives here, and the other scripts here take it from this one; the machinery is in
`pc_experiments.tabular`. `tune.py` must be run first, and TabPFN is left to `run_tabpfn.py`.
Each arm's result on each split is saved on its own, under
`experiments/outputs/tabular/results/baseline/`, so a run that stops, or is divided between
machines with `--datasets` or `--arms`, picks up where it left off.

    uv run experiments/scripts/tabular/run.py [--datasets D ...] [--arms A ...] [--workers N]
"""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import torch

from pc_experiments.outputs import duration, load_run, output_dir, timed
from pc_experiments.tabular import (
    CLASSIFICATION,
    REGRESSION,
    Arm,
    Job,
    Results,
    Setting,
    Sizes,
    result_path,
    run_jobs,
)

NAME = "tabular"
RUN = "baseline"
DATASETS = (*REGRESSION, *CLASSIFICATION)

# A split trains on half the rows, tests at 15% of them, and draws blocks from the rest. A block
# is a quarter of the training set, which resolves roughly a fifth of what a model has yet to
# learn whatever the dataset. The caps only bound the cost on large datasets.
SIZES = Sizes(
    train_share=0.5, train_cap=1000, test_share=0.15, test_cap=100, pool_cap=2000, block_share=0.25
)
# Real outputs are compared under a Gaussian kernel at the dataset's noise scale, which `tune.py`
# measures, and at half, twice and one unit of the training outputs' spread as checks. The first
# is the one the sequential test uses.
SETTING = Setting(
    SIZES,
    num_rounds=500,
    num_features=256,
    lengthscale_multiples=(1.0, 0.5, 2.0),
    fixed_lengthscales=(1.0,),
)
SPLITS = 10

# Exactly Bayesian; approximately Bayesian; not Bayesian by construction but with a fit that is
# a posterior mean or mode; and not Bayesian at all. The sparse GP takes inducing inputs until
# they describe the function to within the noise, and BART is run by two samplers, one of them
# further from its posterior.
REGRESSION_ARMS = (
    Arm("gp"),
    Arm("sparse_gp"),
    Arm("gp_refit"),
    Arm("bart"),
    Arm("xbart"),
    Arm("kernel_ridge"),
    Arm("neighbours"),
    Arm("catboost"),
)
CLASSIFICATION_ARMS = (
    Arm("logistic_exact"),
    Arm("logistic_laplace"),
    Arm("logistic_diagonal"),
    Arm("logistic"),
    Arm("neighbours"),
    Arm("catboost"),
)
# The plug-in arms again with an autoregressive joint, for the learners that refit in well under
# a second: CatBoost would take hours to days a split.
AUTOREGRESSIVE_REGRESSION = ("kernel_ridge", "neighbours")
AUTOREGRESSIVE_CLASSIFICATION = ("logistic", "neighbours")
REGRESSION_ARMS += tuple(Arm(kind, autoregressive=True) for kind in AUTOREGRESSIVE_REGRESSION)
CLASSIFICATION_ARMS += tuple(
    Arm(kind, autoregressive=True) for kind in AUTOREGRESSIVE_CLASSIFICATION
)
TABPFN_ARMS = (Arm("tabpfn"), Arm("tabpfn_preprocessing"))


def tuning_dir() -> Path:
    """Where `tune.py` leaves what it chose for each dataset."""
    return output_dir(NAME) / "tuning"


def results_dir() -> Path:
    return output_dir(NAME) / "results" / RUN


def load_results() -> Results:
    """The saved results on the datasets of this design."""
    return Results.load(results_dir(), DATASETS)


def jobs(
    datasets: Sequence[str],
    num_splits: int,
    regression_arms: Sequence[Arm],
    classification_arms: Sequence[Arm],
    only: Sequence[str] | None = None,
) -> list[Job]:
    """Every arm that applies to each dataset, on splits `0, ..., num_splits - 1`, restricted to
    the arms named in `only` if it is given."""
    return [
        (dataset, seed, arm)
        for dataset in datasets
        for seed in range(num_splits)
        for arm in (classification_arms if dataset in CLASSIFICATION else regression_arms)
        if only is None or arm.name in only
    ]


def gigabytes_per_worker(setting: Setting) -> float:
    """Roughly what a worker holds at once on a regression dataset: every round's embedding at
    every test input under every kernel, which the estimates need until the last round."""
    kernels = len(setting.lengthscale_multiples) + len(setting.fixed_lengthscales)
    numbers = kernels * setting.num_rounds * setting.sizes.test_cap * 2 * setting.num_features
    return 2.5 * 8 * numbers / 2**30  # and about as much again in the estimates' working space


def compute(setting: Setting, to_run: Sequence[Job], workers: int, overwrite: bool) -> None:
    """Run jobs under `setting` and save each under `results/baseline/`."""
    missing = sorted({d for d, _, _ in to_run if not (tuning_dir() / f"{d}.json").exists()})
    if missing:
        raise SystemExit(
            f"Not tuned: {', '.join(missing)}.\n"
            "Tune them first: uv run experiments/scripts/tabular/tune.py --datasets "
            + " ".join(missing)
        )
    print(
        f"{len(to_run)} job{'s' * (len(to_run) != 1)}, {setting.num_rounds} rounds "
        f"each, {workers} at a time at up to {gigabytes_per_worker(setting):.1f} GB each",
        flush=True,
    )
    with timed("Jobs"):
        run_jobs(setting, tuning_dir(), results_dir(), to_run, workers, overwrite)
    print(f"Compute of these jobs, summed over workers: {duration(seconds(to_run))}", flush=True)


def seconds(jobs: Sequence[Job]) -> float:
    """What the saved jobs took to fit and to run their rounds, summed over the jobs."""
    total = 0.0
    for job in jobs:
        path = result_path(results_dir(), job)
        if path.exists():
            values = load_run(path)["values"]
            rounds = len(values["log_wealth"])
            total += (values["fit_seconds"] + rounds * values["round_seconds"]).item()
    return total


def parser(description: str) -> argparse.ArgumentParser:
    """The options both run scripts take."""
    parsed = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parsed.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parsed.add_argument("--arms", nargs="+", help="names of the arms to run, by default all")
    parsed.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="arms at a time; each needs a gigabyte or two on the regression datasets",
    )
    parsed.add_argument("--overwrite", action="store_true", help="redo results already saved")
    return parsed


def main() -> None:
    args = parser(__doc__ or "").parse_args()
    torch.set_default_dtype(torch.float64)
    to_run = jobs(args.datasets, SPLITS, REGRESSION_ARMS, CLASSIFICATION_ARMS, args.arms)
    compute(SETTING, to_run, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
