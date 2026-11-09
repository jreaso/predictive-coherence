"""Where experiment scripts read and write, and the manifest each run records.

Paths are under `experiments/`, found from this file rather than from git, so the scripts also
run from a downloaded archive.

    experiments/figures/<experiment>.pdf
    experiments/tables/<experiment>_<table>.csv
    experiments/outputs/<experiment>/   results, checkpoints and tuning
    experiments/data/                   downloaded datasets
"""

import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

PACKAGES = (
    "torch",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "catboost",
    "stochtree",
    "tabpfn",
    "predictive-coherence",
    "pc-experiments",
)

# `experiments/`, above `src/pc_experiments/`. Needs the editable install that `uv sync` makes.
EXPERIMENTS = Path(__file__).resolve().parents[2]


def _created(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(name: str) -> Path:
    """`experiments/outputs/<name>/`, created if needed."""
    return _created(EXPERIMENTS / "outputs" / name)


def figure_path(name: str) -> Path:
    return _created(EXPERIMENTS / "figures") / f"{name}.pdf"


def table_path(experiment: str, name: str) -> Path:
    return _created(EXPERIMENTS / "tables") / f"{experiment}_{name}.csv"


def data_dir() -> Path:
    """`experiments/data/`, created if needed."""
    return _created(EXPERIMENTS / "data")


def save_run(directory: Path, name: str, payload: dict[str, Any]) -> None:
    """Write a run's results, and its manifest beside them as JSON."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(payload, directory / f"{name}.pt")
    (directory / f"{name}.json").write_text(json.dumps(payload["manifest"], indent=2) + "\n")


def load_run(path: Path) -> dict[str, Any]:
    return torch.load(path, weights_only=False)


def _git(*args: str) -> str | None:
    """Git's output, or `None` outside a clone."""
    try:
        done = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=EXPERIMENTS
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip()


def manifest(config: dict[str, Any]) -> dict[str, Any]:
    """A run's provenance: commit, platform, package versions and configuration. The commit is
    `None` outside a git clone."""
    status = _git("status", "--porcelain")
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": sys.argv,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "config": config,
    }


def duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


@contextmanager
def timed(what: str) -> Iterator[None]:
    """Print the body's wall-clock time when it ends."""
    start = time.perf_counter()
    yield
    print(f"{what}: {duration(time.perf_counter() - start)} on the wall clock", flush=True)
