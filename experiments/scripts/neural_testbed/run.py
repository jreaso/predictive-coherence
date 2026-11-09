"""Run the neural testbed: one-step batch acquisition on sampled networks, scored exactly.

The design lives here; the machinery in `pc_experiments.neural_testbed`. `tune.py` and
`train_inctnp.py` must be run first.

    uv run experiments/scripts/neural_testbed/run.py [--environments N] [--chunk E] [--device D]
        [--resume]

Writes `experiments/outputs/neural_testbed/results/baseline.pt`, saving after every chunk so that
`--resume` carries on a stopped run. The decisions depend on the device: elsewhere than the
paper's GPU stack the run reproduces it in distribution only.
"""

import argparse
import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from pc_experiments.neural_testbed import (
    Architecture,
    Arm,
    Environment,
    Family,
    Setting,
    chunks_of,
    describe,
    merge_chunks,
    run_chunk,
)
from pc_experiments.outputs import duration, load_run, manifest, output_dir, save_run

NAME = "neural_testbed"
RUN = "baseline"

# The paper's run: a thousand environments, about five hours on one H100.
ENVIRONMENTS = 1000
# Environments that share the device at a time. Lower it if memory runs out.
CHUNK = 16

# An environment is a network of this architecture, which every model shares, so that whatever
# incoherence appears comes from inference and not from misspecification.
ARCHITECTURE = Architecture(input_dim=2, hidden_width=32, num_hidden_layers=2, num_classes=3)

SETTING = Setting(
    architecture=ARCHITECTURE,
    # Sharp, balanced class boundaries, so that uncertainty is mostly about where they are.
    temperature=0.35,
    center=3.0,
    # Tight training clusters, and targets spread across class boundaries. The pool reaches
    # everywhere, so that candidates differ in what they can teach.
    train_std=0.5,
    pool_std=1.5,
    target_std=1.5,
    # Training and targets overlap on one cluster only, so that predictions must extrapolate.
    train_clusters=(0, 1, 2),
    target_clusters=(2, 3),
    num_train=24,
    num_pool=512,
    num_target=64,
    num_test=4096,
    # Ten candidate batches of five inputs: with three classes a batch has 243 labellings, few
    # enough for every expectation over them to be a sum.
    num_candidates=10,
    candidate_size=5,
    min_class_mass=0.10,
)

ATOMS = 128

# Arms that share hyperparameters, with the values `tune.py` starts from; a tuned family runs with
# what it chose. At `kl_weight` 1, the ELBO, the mean-field models barely move from their prior.
FAMILIES = {
    "exact": Family("prior_bank", {"num_atoms": 8192}, {"exact": {}}),
    "laplace": Family(
        "laplace",
        {"prior_scale": 4.0, "num_atoms": ATOMS, "lr": 0.05, "fit_steps": 200},
        {
            f"laplace_{curvature}": {"curvature": curvature}
            for curvature in ("full", "kfac", "diag")
        },
    ),
    "bbb": Family(
        "bbb",
        {
            "prior_scale": 1.0,
            "kl_weight": 0.1,
            "init_std_factor": 0.1,
            "num_atoms": ATOMS,
            "elbo_samples": 8,
            "lr": 0.01,
            "fit_steps": 2000,
            "update_lr": 0.0003,
            "update_steps": 100,
        },
        {"bbb": {}},
    ),
    "ivon": Family(
        "ivon",
        {
            "prior_scale": 1.0,
            "kl_weight": 0.1,
            "num_atoms": ATOMS,
            "mc_samples": 4,
            "hess_init": 0.5,
            "beta1": 0.9,
            # The Hessian estimate settles over about a thousand steps, within the fit's budget.
            "beta2": 0.999,
            "lr": 0.1,
            "fit_steps": 2000,
            "update_lr": 0.001,
            "update_steps": 100,
        },
        {"ivon": {}},
    ),
    "ensemble": Family(
        "ensemble",
        {
            "num_members": 8,
            "weight_decay": 3.0,
            "lr": 0.05,
            "fit_steps": 200,
            "update_lr": 0.05,
            "update_steps": 100,
        },
        {"ensemble": {}},
    ),
    "dropout": Family(
        "dropout",
        {
            "rate": 0.5,
            "num_atoms": ATOMS,
            "weight_decay": 1.0,
            "lr": 0.05,
            "fit_steps": 500,
            "update_lr": 0.05,
            "update_steps": 100,
        },
        {"dropout": {}},
    ),
    "inctnp": Family("inctnp", {}, {"inctnp": {}}),
}

# Families whose hyperparameters `tune.py` chooses. The others have none to choose.
TUNED = ("laplace", "bbb", "ivon", "ensemble", "dropout")


def checkpoint_path() -> Path:
    """Where `train_inctnp.py` leaves the trained network, once training has finished."""
    return output_dir(NAME) / "checkpoints" / "inctnp.pt"


def results_dir() -> Path:
    """Where runs are saved, `<name>.pt` for each."""
    return output_dir(NAME) / "results"


def tuning_path(family: str) -> Path:
    """Where `tune.py` leaves what it chose for a family."""
    return output_dir(NAME) / "tuning" / f"{family}.pt"


def default_device() -> str:
    # Not an Apple GPU, which has no float64.
    return "cuda" if torch.cuda.is_available() else "cpu"


def tuned(family: str, required: bool) -> dict[str, Any]:
    """What `tune.py` chose for a family, if it tunes one. Choices made for another setting, or
    from other starting values, are refused rather than used."""
    path = tuning_path(family)
    if family not in TUNED or not (required or path.exists()):
        return {}
    if not path.exists():
        raise SystemExit(
            f"No tuned hyperparameters for {family} at {path}.\n"
            "Tune first: uv run experiments/scripts/neural_testbed/tune.py"
        )
    saved = load_run(path)
    made_for = saved["manifest"]["config"]
    expected = (asdict(SETTING), FAMILIES[family].hyperparameters)
    if (made_for["setting"], made_for["start"]) != expected:
        raise SystemExit(f"{path} was tuned for another setting or from other values: tune again.")
    return saved["chosen"]


def trained_checkpoint() -> str:
    """The incTNP's checkpoint, refused if it was trained for another setting."""
    path = checkpoint_path()
    if not path.exists():
        raise SystemExit(
            f"No trained incTNP at {path}.\n"
            "Train one first: uv run experiments/scripts/neural_testbed/train_inctnp.py"
        )
    if load_run(path)["tasks"]["setting"] != asdict(SETTING):
        raise SystemExit(f"{path} was trained for another setting: train again.")
    return str(path)


def diagnostic_arms() -> dict[str, Arm]:
    """Every arm but the incTNP, by name, with tuned values where there are any. For checks,
    which unlike a run should work before anything is tuned."""
    return {
        arm.name: arm
        for name, family in FAMILIES.items()
        if name != "inctnp"
        for arm in family.build(tuned(name, required=False))
    }


def build_arms(names: Sequence[str] | None = None) -> list[Arm]:
    """The arms to run, every one unless `names` are given, with the tuned values of each family."""
    arms = []
    for family_name, family in FAMILIES.items():
        wanted = [name for name in family.arms if names is None or name in names]
        if not wanted:
            continue
        if family_name == "inctnp":
            chosen = {"checkpoint": trained_checkpoint()}
        else:
            chosen = tuned(family_name, required=True)
        arms += [arm for arm in family.build(chosen) if arm.name in wanted]
    unknown = set(names or ()) - {arm.name for arm in arms}
    if unknown:
        raise SystemExit(f"Unknown arms: {sorted(unknown)}")
    return arms


def compute(
    name: str, arms: Sequence[Arm], num_environments: int, chunk: int, device: str, resume: bool
) -> None:
    """Run every arm on the first valid environments, and save the results with their provenance
    as `results/<name>.pt`."""
    config = {
        "run": name,
        "num_environments": num_environments,
        "chunk": chunk,
        "device": device,
        **asdict(SETTING),
        "arms": [describe(arm) for arm in arms],
    }
    run_in_chunks(
        name,
        config,
        [describe(arm) for arm in arms],
        lambda environments: run_chunk(SETTING, arms, environments, device),
        num_environments,
        chunk,
        resume,
    )


def run_in_chunks(
    name: str,
    config: dict[str, Any],
    arms: list[dict[str, Any]],
    run_one: Callable[[list[Environment]], dict[str, Any]],
    num_environments: int,
    chunk: int,
    resume: bool,
    directory: Path | None = None,
    workers: int = 1,
    threads: int = 1,
) -> None:
    """`run_one` on the first valid environments, `chunk` at a time, saved as `<name>.pt` in
    `directory`, by default `results/`.

    Each finished chunk goes to `<name>_partial.pt`, which `resume` carries on from under the same
    `config`, recording every session's provenance. With `workers` above one, chunks run in
    processes of `threads` threads, for a CPU, and are saved in order.
    """
    directory = directory or results_dir()
    partial = directory / f"{name}_partial.pt"
    finished: list[dict[str, Any]] = []
    provenance = manifest(config)  # of the code as it is now, not as it is when the run ends
    if partial.exists():
        if not resume:
            raise SystemExit(
                f"{partial} holds an unfinished run, which a new one would overwrite.\n"
                "Pass --resume to carry on from it, or delete it to start again."
            )
        state = torch.load(partial, weights_only=False)
        if state["config"] != config:
            raise SystemExit(f"{partial} was run under a different configuration.")
        finished = state["chunks"]
        resumed = [*state["manifest"].get("resumed", []), provenance]
        provenance = state["manifest"] | {"resumed": resumed}

    start = time.perf_counter()
    pending = list(enumerate(chunks_of(SETTING, num_environments, chunk)))[len(finished) :]

    def keep(result: dict[str, Any]) -> None:
        finished.append(result)
        directory.mkdir(parents=True, exist_ok=True)
        scratch = partial.with_name(partial.name + ".tmp")
        torch.save({"config": config, "manifest": provenance, "chunks": finished}, scratch)
        scratch.replace(partial)
        done = sum(len(chunk_["seeds"]) for chunk_ in finished)
        print(
            f"{done}/{num_environments} environments, {time.perf_counter() - start:.0f}s",
            flush=True,
        )

    if workers <= 1:
        for _, environments in pending:
            keep(run_one(environments))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            workers, mp_context=context, initializer=torch.set_num_threads, initargs=(threads,)
        ) as pool:
            futures = {pool.submit(run_one, environments): index for index, environments in pending}
            out_of_order: dict[int, dict[str, Any]] = {}
            try:
                for future in as_completed(futures):
                    out_of_order[futures[future]] = future.result()
                    while len(finished) in out_of_order:
                        keep(out_of_order.pop(len(finished)))
            except BaseException:
                pool.shutdown(cancel_futures=True)
                raise

    save_run(
        directory, name, {"manifest": provenance, "arms": arms, "results": merge_chunks(finished)}
    )
    partial.unlink()
    print(f"Saved {directory / name}.pt after {duration(time.perf_counter() - start)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--environments", type=int, default=ENVIRONMENTS)
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--resume", action="store_true", help="carry on from a stopped run")
    parser.add_argument("--overwrite", action="store_true", help="replace a finished run")
    args = parser.parse_args()

    finished = results_dir() / f"{RUN}.pt"
    if finished.exists() and not args.overwrite:
        raise SystemExit(f"{finished} is already finished. Pass --overwrite to replace it.")
    arms = build_arms()
    print(f"{[arm.name for arm in arms]} on {args.device}", flush=True)
    compute(RUN, arms, args.environments, args.chunk, args.device, args.resume)


if __name__ == "__main__":
    main()
