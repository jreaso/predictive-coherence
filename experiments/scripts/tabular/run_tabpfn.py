"""Run the TabPFN arms of the tabular experiment: the network, and the network with the
preprocessing it is shipped with.

The design is `run.py`'s, and the results join its own. It is a script of its own because it is
slow: a round costs one forward pass for every input of its block, so these arms get fewer rounds
and one split. Launched by hand; `--datasets` divides it between machines. The weights are
gated: accept their licence when the package asks, or copy them to `TABPFN_MODEL_CACHE_DIR`.

    uv run experiments/scripts/tabular/run_tabpfn.py [--datasets D ...] [--arms A ...]
"""

from dataclasses import replace

import torch
from run import SETTING, TABPFN_ARMS, compute, jobs, parser

from pc_experiments.tabular import Setting, load, make_task
from pc_experiments.tabular.tabpfn import MIN_CONTEXT

# A forward pass for every input of a block makes a round cost seconds to minutes on a CPU, which
# is what bounds these arms to one split. At 128 rounds the standard error of Ψ is a few hundredths.
ROUNDS = 128
SPLITS = 1


def check_contexts(setting: Setting, datasets: list[str], num_splits: int) -> None:
    """Stop unless every split trains on more rows than the package needs to read the inputs."""
    small = {
        dataset: len(make_task(load(dataset), setting.sizes, seed).x)
        for dataset in datasets
        for seed in range(num_splits)
        if len(make_task(load(dataset), setting.sizes, seed).x) <= MIN_CONTEXT
    }
    if small:
        raise SystemExit(
            f"These train on {MIN_CONTEXT} rows or fewer, where the package stops inferring "
            f"which inputs are categorical and an update changes how they are read: {small}."
        )


def main() -> None:
    parsed = parser(__doc__ or "")
    parsed.set_defaults(workers=1)  # the rounds of an arm are batched instead
    args = parsed.parse_args()
    torch.set_default_dtype(torch.float64)
    setting = replace(SETTING, num_rounds=ROUNDS)
    check_contexts(setting, args.datasets, SPLITS)
    to_run = jobs(args.datasets, SPLITS, TABPFN_ARMS, TABPFN_ARMS, args.arms)
    compute(setting, to_run, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
