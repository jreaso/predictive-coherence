"""Choose each family's hyperparameters for the neural testbed, by how well it predicts.

A prerequisite of `run.py`, launched by hand and never by it. A family's hyperparameters are
chosen in two steps, on environments that `run.py` never sees. Those that shape the fit are
chosen by the divergence from the truth to the fitted predictive. With those in place, those that
shape the update are chosen by the same divergence after an update on a candidate batch and its
true labels. Divergences are measured at held-out inputs like the data the model has seen, as
`pc_experiments.neural_testbed.tuning` explains, not at the test inputs. Nothing here looks at
incoherence or at an acquisition objective.

    uv run experiments/scripts/neural_testbed/tune.py [--families F ...] [--device D] [--overwrite]
    uv run experiments/scripts/neural_testbed/tune.py --smoke   # a few environments, into a temporary directory

What is chosen for a family is written to `experiments/outputs/neural_testbed/tuning/<family>.pt`, with the
whole table beside it in `<family>.md`.
"""

import argparse
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from run import CHUNK, FAMILIES, NAME, SETTING, TUNED, default_device, tuning_path

from pc_experiments.neural_testbed import (
    candidates_for,
    family_scores,
    grid_points,
    stack,
    valid_environments,
)
from pc_experiments.outputs import manifest, save_run, timed
from pc_experiments.plotting import mean_se

# Environments to tune on, from seeds far above any that `run.py` reaches.
ENVIRONMENTS = 64
FIRST_SEED = 1_000_000

# For each family, the values tried of the hyperparameters that shape the fit, and then of those
# that shape the update. Every combination is tried; what is not listed stays as `run.py` has it.
GRIDS: dict[str, dict[str, dict[str, tuple[Any, ...]]]] = {
    "laplace": {
        "fit": {"prior_scale": (1.0, 2.0, 4.0, 8.0), "fit_steps": (200, 1000)},
        "update": {},
    },
    # `kl_weight` 1 is the ELBO, kept in the grid so that the table shows what it costs.
    "bbb": {
        "fit": {
            "prior_scale": (1.0, 2.0),
            "kl_weight": (1.0, 0.3, 0.1, 0.03),
            "lr": (0.003, 0.01),
        },
        # Small updates: under this criterion a converged update forgets more than it learns,
        # and the best is stopped early.
        "update": {"update_lr": (0.0001, 0.0003, 0.001), "update_steps": (30, 100, 300, 1000)},
    },
    "ivon": {
        "fit": {
            "prior_scale": (1.0, 2.0),
            "kl_weight": (1.0, 0.3, 0.1, 0.03),
            "lr": (0.03, 0.1),
        },
        "update": {"update_lr": (0.001, 0.003, 0.01), "update_steps": (10, 30, 100, 300)},
    },
    "ensemble": {
        "fit": {"weight_decay": (0.3, 1.0, 3.0, 10.0), "fit_steps": (200, 1000)},
        "update": {"update_lr": (0.005, 0.05), "update_steps": (50, 200, 1000)},
    },
    "dropout": {
        "fit": {"weight_decay": (0.3, 1.0, 3.0), "rate": (0.2, 0.5), "fit_steps": (500, 2000)},
        "update": {"update_lr": (0.005, 0.05), "update_steps": (50, 200, 1000)},
    },
}


def tune(family_name: str, chunks: list, log: list[str], smoke: bool) -> dict[str, Any]:
    """Choose a family's hyperparameters, a step at a time, and record every value tried."""
    family = FAMILIES[family_name]
    chosen: dict[str, Any] = {}
    for step, updated in (("fit", False), ("update", True)):
        points = grid_points(GRIDS[family_name][step])[: 2 if smoke else None]
        if len(points) == 1:  # nothing to choose between, but maybe something to set
            chosen |= points[0]
            continue
        criterion = "after an update" if updated else "fitted"
        log += [f"\n## {family_name}: {step}, by the divergence {criterion}\n"]
        log += ["| values | nats |", "|---|---|"]
        scored = []
        for point in points:
            start = time.perf_counter()
            scores = family_scores(family, {**chosen, **point}, SETTING, chunks, updated)
            scored.append((scores.mean().item(), point))
            mean, se = mean_se(scores)
            log.append(f"| `{point}` | {mean:.4f} ± {se:.4f} |")
            seconds = time.perf_counter() - start
            print(f"{family_name} {point}: {scored[-1][0]:.4f} ({seconds:.0f}s)", flush=True)
        # A NaN marks a fit that diverged, which is not the best of anything.
        finite = [pair for pair in scored if pair[0] == pair[0]]
        if not finite:
            raise SystemExit(f"Every {step} setting of {family_name} diverged.")
        best = min(finite, key=lambda pair: pair[0])[1]
        log.append(f"\nChosen: `{best}`")
        chosen |= best
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--families", nargs="+", default=list(TUNED), choices=TUNED)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--overwrite", action="store_true", help="replace what was chosen before")
    parser.add_argument("--smoke", action="store_true", help="a short run that checks the machine")
    args = parser.parse_args()

    num_environments = 2 if args.smoke else ENVIRONMENTS
    paths = {family: tuning_path(family) for family in args.families}
    if args.smoke:
        directory = Path(tempfile.mkdtemp(prefix=f"{NAME}_tuning_smoke_"))
        paths = {family: directory / f"{family}.pt" for family in args.families}
    for path in paths.values():
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} is already chosen. Pass --overwrite to replace it.")

    source = valid_environments(SETTING, FIRST_SEED)
    chunks = []
    for start in range(0, num_environments, CHUNK):
        environments = [next(source) for _ in range(min(CHUNK, num_environments - start))]
        chunk = stack(environments, args.device)
        chunks.append((chunk, candidates_for(SETTING, chunk, environments)[0]))

    for family, path in paths.items():
        config = {
            "family": family,
            # What the choices were made for, which `run.py` checks before using them.
            "setting": asdict(SETTING),
            "start": FAMILIES[family].hyperparameters,
            "grid": GRIDS[family],
            "num_environments": num_environments,
            "first_seed": FIRST_SEED,
            "device": args.device,
        }
        provenance = manifest(config)
        log = [f"# Tuning of {family}", "", f"{num_environments} environments."]
        with timed(f"Tuning {family}"):
            chosen = tune(family, chunks, log, args.smoke)
        save_run(path.parent, path.stem, {"manifest": provenance, "chosen": chosen})
        path.with_suffix(".md").write_text("\n".join(log) + "\n")
        print(f"{family}: {chosen}\nSaved {path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
