"""Incremental transformer neural processes (incTNP), as predictive models.

`attention`, `transformer` and `network` are adapted from the authors' implementation of
"Incremental Transformer Neural Processes" (Mortimer et al., ICML 2026); each file says what it
took and what it changed.
"""

from pc_experiments.inctnp.model import IncTNPModel, load_network, network_payload
from pc_experiments.inctnp.network import Architecture, build_network
from pc_experiments.inctnp.training import (
    Optimisation,
    SequenceTasks,
    convergence,
    evaluate,
    train,
    train_network,
)

__all__ = [
    "Architecture",
    "IncTNPModel",
    "Optimisation",
    "SequenceTasks",
    "build_network",
    "convergence",
    "evaluate",
    "load_network",
    "network_payload",
    "train",
    "train_network",
]
