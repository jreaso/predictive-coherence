"""Run the synthetic Gaussian process experiment: sparse Gaussian processes against an incTNP.

The design lives here; the machinery in `pc_experiments.synthetic_gp`. `train_inctnp.py` must be
run first.

    uv run experiments/scripts/synthetic_gp/run.py [--environments N] [--subsamples N] [--workers N]

Writes `experiments/outputs/synthetic_gp/results/baseline.pt`, which `report.py` reads. A finished
run is not replaced without `--overwrite`.
"""

import argparse
import dataclasses
import os
from pathlib import Path

import torch

from pc_experiments.gp import GaussianKernelPrior
from pc_experiments.outputs import duration, load_run, manifest, output_dir, save_run, timed
from pc_experiments.synthetic_gp import Arm, Setting, check_training, run_setting

NAME = "synthetic_gp"
RUN = "baseline"

# An environment is one function; a task is a subsample of the points it is observed at.
ENVIRONMENTS = 20
SUBSAMPLES = 10

# What the models assume: the Gaussian process arms use this prior, and the incTNP is trained on
# it. Inputs are uniform on `INPUT_RANGE`.
PRIOR = GaussianKernelPrior(lengthscale=0.25, outputscale=1.0, noise_std=0.1)
INPUT_RANGE = (-2.0, 2.0)

SETTING = Setting(
    prior=PRIOR,
    truth=PRIOR,  # well specified: the environments are drawn from the prior the models assume
    input_range=INPUT_RANGE,
    num_pool=512,
    # Few enough points that the function is still uncertain between them, so that an update has
    # something to resolve. A block is a third of the context.
    num_context=24,
    num_design=8,
    num_eval=64,
    num_rounds=1000,
    # The estimate is also kept at smaller budgets, to show how precise it is at each.
    estimates_at=(10, 25, 50, 100, 250, 1000),
    psi_lengthscale=PRIOR.noise_std,
    psi_features=256,
)

# Two ladders of inducing inputs chosen greedily from the data: held fixed through an update, up to
# the whole context, or chosen afresh, up to the context and the block, where the model is exact.
# Near the size of the data a rung takes fewer where inputs are too close to induce on.
FIXED = (6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
RESELECTED = (*FIXED, 26, 28, 30, 32)

ARMS = (
    Arm("exact"),
    Arm("inctnp"),
    *(Arm("sparse", m) for m in FIXED),
    *(Arm("reselected", m) for m in RESELECTED),
)


def checkpoint_path() -> Path:
    """Where `train_inctnp.py` leaves the trained network, once training has finished."""
    return output_dir(NAME) / "checkpoints" / "inctnp.pt"


def compute(
    setting: Setting,
    checkpoint: Path,
    num_environments: int,
    num_subsamples: int,
    workers: int,
    overwrite: bool,
) -> None:
    """Run every arm in `setting` with the network at `checkpoint`, and save the results with
    their provenance as `results/baseline.pt`."""
    finished = results_path()
    if finished.exists() and not overwrite:
        raise SystemExit(f"{finished} is already finished. Pass --overwrite to replace it.")
    if not checkpoint.exists():
        raise SystemExit(
            f"No trained incTNP at {checkpoint}.\n"
            "Train one first: uv run experiments/scripts/synthetic_gp/train_inctnp.py"
        )
    trained = load_run(checkpoint)
    check_training(trained, setting)
    training = {"steps": trained["history"][-1]["step"], **trained["convergence"]}
    print(
        f"incTNP: {training['steps']} steps, "
        f"{training['kl']:.4f} nats per point from the exact predictive",
        flush=True,
    )

    config = {
        "setting": RUN,
        "num_environments": num_environments,
        "num_subsamples": num_subsamples,
        **dataclasses.asdict(setting),
        "arms": [arm.name for arm in ARMS],
    }
    provenance = manifest(config)  # of the code as it is now, not as it is when the run ends
    print(f"{num_environments} environments of {num_subsamples} subsamples", flush=True)
    with timed("Rounds of every arm on every task"):
        results = run_setting(setting, ARMS, checkpoint, num_environments, num_subsamples, workers)
    rounds = results["round_seconds"].sum().item() * setting.num_rounds
    print(f"Rounds alone, summed over workers: {duration(rounds)}", flush=True)
    results["psi_at_rounds"] = torch.tensor(setting.estimates_at)
    save_run(
        finished.parent,
        RUN,
        {
            "manifest": provenance,
            "arms": [dataclasses.asdict(arm) for arm in ARMS],
            "training": training,
            "results": results,
        },
    )


def results_path() -> Path:
    """Where the run's results are saved, and `report.py` reads them."""
    return output_dir(NAME) / "results" / f"{RUN}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--environments", type=int, default=ENVIRONMENTS)
    parser.add_argument("--subsamples", type=int, default=SUBSAMPLES)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--overwrite", action="store_true", help="replace a finished run")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    compute(SETTING, checkpoint_path(), args.environments, args.subsamples, args.workers,
            args.overwrite)  # fmt: skip


if __name__ == "__main__":
    main()
