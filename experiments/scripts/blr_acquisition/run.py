"""Run the Bayesian linear regression acquisition experiment, in its baseline and its four
robustness settings.

The design lives here; the machinery in `pc_experiments.blr`. Each setting takes about a minute.

    uv run experiments/scripts/blr_acquisition/run.py [--settings S ...] [--envs N] [--workers N]

Writes `experiments/outputs/blr_acquisition/results/<setting>.pt`, which `report.py` reads.
"""

import argparse
import dataclasses
import math
import os
from dataclasses import replace

import torch

from pc_experiments.blr import Acquirer, Setting, run_setting
from pc_experiments.outputs import manifest, output_dir, save_run, timed

NAME = "blr_acquisition"
ENVS = 100

# One environment: random Fourier features of a Gaussian kernel on a scalar input, a pool to
# acquire from, and targets to be scored on.
SETTING = Setting(
    num_features=100,
    lengthscale=0.1,
    prior_var=3.0,
    noise_var=0.8,
    num_init=25,
    num_pool=1000,
    num_targets=100,
    num_steps=100,
    input_range=(-10.0, 10.0),
    init_range=(-1.0, 1.0),
    target_range=(-10.0, 10.0),
    misspecification=0.0,
    checkpoints=(0, 10, 25, 50, 100),
    psi_lengthscale=math.sqrt(0.8),
)

# Each robustness setting changes one thing about the baseline.
SETTINGS = {
    "baseline": SETTING,
    "noise_low": replace(SETTING, noise_var=0.1),
    "noise_high": replace(SETTING, noise_var=3.0),
    "targets_far": replace(SETTING, target_range=(4.0, 8.0)),
    "misspecified": replace(SETTING, misspecification=1.0),
}

# Shares of the remaining pool that the coherentised predictives average over. Zero averages over
# nothing, so it is the mean-field model's own EPIG.
SHARES = (0.0, 0.01, 0.025, 0.05, 0.1, 1.0)

ACQUIRERS = (
    Acquirer("random"),
    Acquirer("exact"),
    Acquirer("alpha", 0.0),  # mean-field EPIG
    Acquirer("alpha", 1.0),  # mean-field EMI
    *(Acquirer("subsample", share) for share in SHARES),
)


def compute(name: str, setting: Setting, num_envs: int, workers: int) -> None:
    """Run every acquirer in `setting` and save the results with their provenance."""
    config = {
        "setting": name,
        "num_envs": num_envs,
        "seeds": list(range(num_envs)),
        **dataclasses.asdict(setting),
    }
    print(f"{name}: {num_envs} environments", flush=True)
    results = run_setting(setting, ACQUIRERS, num_envs, workers)
    save_run(
        output_dir(NAME) / "results",
        name,
        {
            "manifest": manifest(config | {"acquirers": [a.name for a in ACQUIRERS]}),
            "acquirers": [dataclasses.asdict(acquirer) for acquirer in ACQUIRERS],
            "checkpoints": setting.recorded,
            "results": results,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    parser.add_argument("--envs", type=int, default=ENVS, help=f"environments (default {ENVS})")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    for name in args.settings:
        with timed(name):
            compute(name, SETTINGS[name], args.envs, args.workers)


if __name__ == "__main__":
    main()
