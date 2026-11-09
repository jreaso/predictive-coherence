"""What a script that trains an incTNP needs around the training. A script refuses to replace a
finished or unfinished checkpoint unless told to, and `--smoke` trains briefly into a temporary
directory."""

import argparse
import tempfile
from pathlib import Path

import torch


def training_parser(description: str, device: str) -> argparse.ArgumentParser:
    """The options every training script takes, with `--device` defaulting to `device`."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default=device)
    parser.add_argument("--resume", action="store_true", help="carry on from a partial checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="replace a finished checkpoint")
    parser.add_argument("--smoke", action="store_true", help="a short run that checks the machine")
    parser.add_argument(
        "--steps",
        type=int,
        help="train for this many steps instead of the recipe's, which with --resume carries a "
        "finished run on for longer, or brings one to an end sooner",
    )
    return parser


def print_validation(step: int, metrics: dict[str, float]) -> None:
    """What training reports at `step`: each validation's results, and nothing for the training
    loss recorded between them."""
    if any(key.startswith("val/") for key in metrics):
        shown = ", ".join(
            f"{key.removeprefix('val/')} {value:.4f}" for key, value in metrics.items()
        )
        print(f"step {step}: {shown}", flush=True)


def describe(device: str) -> str:
    """What a device is, to say what a run trained on."""
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(device)
    return "Apple GPU" if device == "mps" else "no GPU: this will be slow"


def checkpoint_paths(final: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    """The finished and partial checkpoints' paths: beside `final`, or in a temporary directory for
    a smoke run. Stops a run that would replace either unasked, or resume from nothing."""
    if args.smoke:
        final = Path(tempfile.mkdtemp(prefix="inctnp_smoke_")) / final.name
        return final, final.with_name(f"{final.stem}_partial.pt")
    partial = final.with_name(f"{final.stem}_partial.pt")
    if final.exists() and not (args.overwrite or args.resume):
        raise SystemExit(
            f"{final} is already trained. Pass --overwrite to replace it, or --resume with "
            "--steps to carry it on for longer."
        )
    if partial.exists() and not final.exists() and not args.resume:
        raise SystemExit(
            f"{partial} holds an unfinished run, which a new one would overwrite.\n"
            "Pass --resume to carry on from it, or delete it to start again."
        )
    if args.resume and not partial.exists():
        raise SystemExit(f"Nothing to carry on from: there is no {partial}.")
    return final, partial


def report_speed(seconds: float, steps: int, full_steps: int) -> None:
    """How long a full run of `full_steps` would take, from a short run of `steps`."""
    hours = seconds / steps * full_steps / 3600
    print(
        f"{1000 * seconds / steps:.1f} ms per step, validations included: about {hours:.1f} "
        f"hours for the {full_steps} steps of a full run on this machine."
    )
