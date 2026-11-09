"""Train the incTNP that the synthetic Gaussian process experiment compares against.

A prerequisite of `run.py`, trained on the prior the experiment's models assume. Each validation
is printed, and the whole history is saved with the checkpoint.

    uv run experiments/scripts/synthetic_gp/train_inctnp.py --smoke   # a minute: is this machine ready?
    uv run experiments/scripts/synthetic_gp/train_inctnp.py [--device cuda] [--resume] [--overwrite]

`--smoke` does everything a full run does into a temporary directory, and says how long a full
run would take. The checkpoint `run.py` reads is written when training finishes; until then a
partial one beside it is what `--resume` carries on from.

The recipe follows the incTNP authors' one-dimensional Gaussian process configuration, except for
the optimiser (Muon on hidden weight matrices, AdamW on the rest) and shorter sequences in larger
batches.
"""

import dataclasses
import time

import torch
from run import INPUT_RANGE, PRIOR, SETTING, checkpoint_path

from pc_experiments.inctnp import (
    Architecture,
    Optimisation,
    SequenceTasks,
    build_network,
    convergence,
    evaluate,
    network_payload,
    train,
)
from pc_experiments.inctnp.launch import (
    checkpoint_paths,
    describe,
    print_validation,
    report_speed,
    training_parser,
)
from pc_experiments.outputs import duration, manifest, save_run

SEED = 0

ARCHITECTURE = Architecture(
    embed_dim=128,
    num_heads=8,
    head_dim=16,
    num_layers=5,
    norm_first=True,
)

# Sequences train contexts of up to one point fewer: `run.py`'s context and block, and some room.
# Every context size weighs the same, so longer sequences would spend training on dense contexts.
TASKS = SequenceTasks(PRIOR, INPUT_RANGE, length=49)

OPTIMISATION = Optimisation(
    steps=400_000,  # the paper's network
    batch_size=32,  # shorter sequences than the authors' 16, so a step sees as many points
    lr=5e-4,
    min_lr=1e-6,
    warmup_fraction=0.1,
    weight_decay=0.01,
    muon_momentum=0.95,
    grad_clip=0.5,
)

# Held-out sequences, fixed by their own seed, on which the network is compared with the exact
# predictive of the prior.
VALIDATION_TASKS = 4096
VALIDATION_SEED = 1
VALIDATE_EVERY = 2000
LOG_EVERY = 100

SMOKE_STEPS = 300


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    args = training_parser(__doc__ or "", default_device()).parse_args()
    needed = SETTING.num_context + SETTING.num_design
    if TASKS.max_context < needed:
        raise SystemExit(f"run.py needs contexts of {needed}, more than {TASKS.max_context}")
    final, partial = checkpoint_paths(checkpoint_path(), args)
    steps = SMOKE_STEPS if args.smoke else args.steps or OPTIMISATION.steps
    optimisation = dataclasses.replace(OPTIMISATION, steps=steps)
    validation_tasks, validate_every = (
        (256, SMOKE_STEPS // 3) if args.smoke else (VALIDATION_TASKS, VALIDATE_EVERY)
    )

    print(f"Training on {args.device} ({describe(args.device)}) for {optimisation.steps} steps")
    config = {
        "architecture": dataclasses.asdict(ARCHITECTURE),
        "tasks": dataclasses.asdict(TASKS),
        "optimisation": dataclasses.asdict(optimisation),
        "validation_tasks": validation_tasks,
        "seed": SEED,
        "device": args.device,
        "device_name": describe(args.device),
    }
    provenance = manifest(config)  # of the code as it is now, not as it is when training ends
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    network = build_network(ARCHITECTURE).to(args.device)
    validation = TASKS.sample(validation_tasks, torch.Generator().manual_seed(VALIDATION_SEED))

    start = time.perf_counter()
    history = train(
        network,
        TASKS,
        optimisation,
        validation,
        log=print_validation,
        log_every=LOG_EVERY,
        validate_every=validate_every,
        partial_path=partial,
        resume=args.resume,
    )
    seconds = time.perf_counter() - start

    # The checkpoint first, so that nothing that follows can lose it.
    converged = convergence(history)
    save_run(
        final.parent,
        final.stem,
        {
            "manifest": provenance,
            **network_payload(network, ARCHITECTURE),
            "tasks": dataclasses.asdict(TASKS),
            "history": history,
            "convergence": converged,
        },
    )
    # Kept, not deleted: it holds the optimisers' state, which is what lets a later run carry
    # this one on for longer with --resume --steps.

    print(f"Saved {final}, after {duration(seconds)} of training on the wall clock")
    if args.smoke:
        report_speed(seconds, optimisation.steps, OPTIMISATION.steps)
        return
    print(
        f"Divergence from the exact predictive: {converged['kl']:.4f} nats per point, having "
        f"fallen by {converged['kl_tail_drop']:.1%} of that over the last quarter of training."
    )
    # By context size, which says where the network is weakest. `run.py` uses contexts from
    # `num_context` to `num_context + num_design`.
    kl = evaluate(network, PRIOR, *validation)["kl"]
    for context in sorted({1, 8, 32, SETTING.num_context, needed, TASKS.max_context}):
        print(f"  context of {context:3d}: {kl[context - 1]:.4f}")


if __name__ == "__main__":
    main()
