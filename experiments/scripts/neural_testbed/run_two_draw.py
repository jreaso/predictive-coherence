"""Two draws of every random update: ‖Δ‖², V and Ψ as the paper defines them, on the
baseline's environments.

Bayes by backprop, IVON and MC dropout update with randomness of their own, which the saved
baseline drew once per environment, so its ‖Δ‖², V and Ψ for them are conditional on that draw.
This run updates each of them twice: from the baseline's stream, which the decisions use, and from
an independent one, with which ‖Δ‖², V and Ψ average over the randomness.

On the baseline's device and chunk size the first draw is the baseline's own, which `--check`
confirms, so the baseline's losses stand. Elsewhere a re-score needs `--test-losses` of its own.

    uv run experiments/scripts/neural_testbed/run_two_draw.py [--environments N] [--chunk E]
        [--resume]
    uv run experiments/scripts/neural_testbed/run_two_draw.py --workers 6 --threads 2 --chunk 1
    uv run experiments/scripts/neural_testbed/run_two_draw.py --check   # the baseline's first draw?

The second form is for a laptop's CPU. Needs `tune.py`'s output, as `run.py` does, and refuses
arms or a setting other than the baseline's. Writes
`experiments/outputs/neural_testbed/two_draw/results/full.pt`, which `report.py` reads with the
baseline.
"""

import argparse
from dataclasses import asdict
from functools import partial
from pathlib import Path

from run import (
    CHUNK,
    ENVIRONMENTS,
    NAME,
    RUN,
    SETTING,
    build_arms,
    default_device,
    diagnostic_arms,
    results_dir,
    run_in_chunks,
)

from pc_experiments.neural_testbed import OBJECTIVES, Arm, Results, describe, run_chunk
from pc_experiments.outputs import load_run, output_dir

# The re-score `report.py` reads: every environment of the baseline, in chunks of its size.
RESCORE = "full"
BASELINE = RUN
# How far a first draw may differ from the baseline's and still be its own, relative to each
# quantity's largest value: rounding, at most.
PARITY_TOLERANCE = 1e-4
# What the first draw is compared by: this run's key, and the baseline's, which is one draw's.
PARITY_KEYS = {
    "initial_kl": "initial_kl",
    "joint_kl": "joint_kl",
    "epig": "epig",
    "epig_kl": "epig_kl",
    "emi": "emi",
    "loss": "loss",
    "c_plugin": "c",
    "v_plugin": "v",
    "psi_plugin": "psi",
}


def two_draw_dir() -> Path:
    """Where the re-scores are written, under `results/`."""
    return output_dir(NAME) / "two_draw"


def random_arms() -> list[str]:
    """The arms whose updates draw randomness of their own."""
    return [name for name, arm in diagnostic_arms().items() if arm.random_update]


def check_against_baseline(arms: list[Arm], num_environments: int) -> None:
    """Refuse a setting, arms or a number of environments the baseline does not have."""
    baseline = load_run(results_dir() / f"{BASELINE}.pt")
    config = baseline["manifest"]["config"]
    if {key: config[key] for key in asdict(SETTING)} != asdict(SETTING):
        raise SystemExit(f"The {BASELINE} run is of another setting.")
    if num_environments > config["num_environments"]:
        raise SystemExit(f"The {BASELINE} run has {config['num_environments']} environments.")
    ran = {arm["name"]: arm for arm in baseline["arms"]}
    for arm in arms:
        if describe(arm) != ran.get(arm.name):
            raise SystemExit(f"{arm.name} is not as the {BASELINE} run had it: {ran.get(arm.name)}")


def parity(name: str) -> tuple[bool, list[str]]:
    """Whether a finished run's first draw is the baseline's, every objective choosing the same and
    every value of `PARITY_KEYS` within `PARITY_TOLERANCE`, with a line for each arm."""
    run = Results.load(two_draw_dir() / "results" / f"{name}.pt")
    full = Results.load(results_dir() / f"{BASELINE}.pt")
    baseline = full.head(len(run.seeds))
    if run.seeds != baseline.seeds:
        return False, [f"{name} is not of the {BASELINE} run's first environments."]
    devices = (run.config["device"], full.config["device"])
    if devices[0] != devices[1]:
        return False, [f"{name} ran on {devices[0]} and the {BASELINE} on {devices[1]}."]
    same, lines = True, []
    for arm in run.arms:
        worst = 0.0
        for ours, theirs in PARITY_KEYS.items():
            reference = baseline.of(theirs, arm)
            difference = (run.of(ours, arm) - reference).abs().nan_to_num(0).max()
            worst = max(worst, (difference / reference.abs().nan_to_num(0).max()).item())
        decisions = all(
            bool((run.chosen(objective, arm) == baseline.chosen(objective, arm)).all())
            for objective in OBJECTIVES
        )
        same &= decisions and worst <= PARITY_TOLERANCE
        lines.append(
            f"{arm}: decisions {'the same' if decisions else 'DIFFER'} in all "
            f"{len(run.seeds)} environments; largest relative difference {worst:.1e}"
        )
    return same, lines


def compute(
    name: str,
    arms: list[Arm],
    num_environments: int,
    chunk: int,
    device: str,
    resume: bool,
    test_losses: bool = False,
    workers: int = 1,
    threads: int = 1,
) -> None:
    """Run the arms on the first valid environments, and save the results with their provenance
    as `two_draw/results/<name>.pt`. How many processes run them is left out of the
    configuration, since the results do not depend on it."""
    config = {
        "run": name,
        "baseline": BASELINE,
        "num_environments": num_environments,
        "chunk": chunk,
        "device": device,
        "test_losses": test_losses,
        **asdict(SETTING),
        "arms": [describe(arm) for arm in arms],
    }
    run_in_chunks(
        name,
        config,
        [describe(arm) for arm in arms],
        partial(run_chunk, SETTING, arms, device=device, test_losses=test_losses),
        num_environments,
        chunk,
        resume,
        two_draw_dir() / "results",
        workers,
        threads,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--environments", type=int, default=ENVIRONMENTS)
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--test-losses", action="store_true", help="losses at the test inputs too")
    parser.add_argument("--workers", type=int, default=1, help="processes running chunks at once")
    parser.add_argument("--threads", type=int, default=1, help="threads in each worker process")
    parser.add_argument("--resume", action="store_true", help="carry on from a stopped run")
    parser.add_argument("--overwrite", action="store_true", help="replace a finished run")
    parser.add_argument(
        "--check", action="store_true", help="only check the first draw is the baseline's"
    )
    args = parser.parse_args()

    if args.check:
        same, lines = parity(RESCORE)
        print("\n".join(lines))
        print(f"{RESCORE}: the first draw {'is' if same else 'is NOT'} the {BASELINE}'s.")
        raise SystemExit(0 if same else 1)

    finished = two_draw_dir() / "results" / f"{RESCORE}.pt"
    if finished.exists() and not args.overwrite:
        raise SystemExit(f"{finished} is already finished. Pass --overwrite to replace it.")
    arms = build_arms(random_arms())
    check_against_baseline(arms, args.environments)
    workers = f", {args.workers} workers of {args.threads} threads" if args.workers > 1 else ""
    losses = ", with test losses" if args.test_losses else ""
    print(f"{[arm.name for arm in arms]} drawn twice on {args.device}{workers}{losses}")
    compute(
        RESCORE,
        arms,
        args.environments,
        args.chunk,
        args.device,
        args.resume,
        args.test_losses,
        args.workers,
        args.threads,
    )
    print("\n".join(parity(RESCORE)[1]))


if __name__ == "__main__":
    main()
