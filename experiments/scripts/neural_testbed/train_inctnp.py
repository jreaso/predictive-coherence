"""Train the incTNP that the neural testbed evaluates.

A prerequisite of `run.py`, trained on fresh tasks shaped like the experiment's: a training set,
a candidate batch appended to it, and targets. It is the synthetic Gaussian process experiment's
network, larger and with class logits for a head. Each validation is printed, and the whole
history is saved with the checkpoint.

    uv run experiments/scripts/neural_testbed/train_inctnp.py --smoke   # a minute: is this machine ready?
    uv run experiments/scripts/neural_testbed/train_inctnp.py [--device cuda] [--resume] [--overwrite]

`--smoke` does everything a full run does into a temporary directory, and says how long a full
run would take. The checkpoint `run.py` reads is written when training finishes; until then a
partial one beside it is what `--resume` carries on from.
"""

import dataclasses
import time

import torch
from run import SETTING, checkpoint_path

from pc_experiments.inctnp import Architecture, Optimisation, convergence, train_network
from pc_experiments.inctnp.launch import (
    checkpoint_paths,
    describe,
    print_validation,
    report_speed,
    training_parser,
)
from pc_experiments.neural_testbed import Tasks, evaluate, task_loss
from pc_experiments.neural_testbed.models import build_classifier, classifier_payload
from pc_experiments.outputs import duration, manifest, save_run

SEED = 0

ARCHITECTURE = Architecture(
    dim_x=SETTING.architecture.input_dim,
    dim_y=SETTING.architecture.num_classes,
    embed_dim=256,
    num_heads=8,
    head_dim=32,
    num_layers=8,
    norm_first=True,
)

TASKS = Tasks(
    SETTING,
    num_targets=SETTING.num_target,
    num_validity=1024,
    # Half the batches are clumps about one pool input, as tight as neighbours in the pool are,
    # because some of the experiment's designed candidates are.
    clump_probability=0.5,
    clump_std=0.25,
)

OPTIMISATION = Optimisation(
    steps=90_000,  # the paper's network
    batch_size=128,
    lr=5e-4,
    min_lr=1e-6,
    warmup_fraction=0.05,
    weight_decay=0.01,
    muon_momentum=0.95,
    grad_clip=0.5,
)

# Held-out tasks, fixed by their own seed, on which the network is compared with the truth.
VALIDATION_TASKS = 2048
VALIDATION_SEED = 1
VALIDATE_EVERY = 2000
LOG_EVERY = 100

SMOKE_STEPS = 50


def default_device() -> str:
    # Tasks are drawn in float64, which an Apple GPU does not have.
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = training_parser(__doc__ or "", default_device()).parse_args()
    final, partial = checkpoint_paths(checkpoint_path(), args)
    steps = SMOKE_STEPS if args.smoke else args.steps or OPTIMISATION.steps
    optimisation = dataclasses.replace(OPTIMISATION, steps=steps)
    validation_tasks, validate_every = (
        (128, SMOKE_STEPS // 2) if args.smoke else (VALIDATION_TASKS, VALIDATE_EVERY)
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
    torch.manual_seed(VALIDATION_SEED)
    validation = TASKS.sample(validation_tasks, args.device)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    network = build_classifier(ARCHITECTURE).float().to(args.device)

    start = time.perf_counter()
    history = train_network(
        network,
        optimisation,
        loss=lambda: task_loss(network, TASKS, TASKS.sample(optimisation.batch_size, args.device)),
        validate=lambda: evaluate(network, TASKS, validation),
        trained_under={"tasks": config["tasks"], "optimisation": config["optimisation"]},
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
            **classifier_payload(network, ARCHITECTURE),
            "tasks": config["tasks"],
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
        f"Divergence from the truth at the targets: {history[-1]['kl']:.4f} nats given the "
        f"training set and {history[-1]['kl_updated']:.4f} given a batch as well, having fallen "
        f"by {converged['kl_tail_drop']:.1%} over the last quarter of training."
    )


if __name__ == "__main__":
    main()
