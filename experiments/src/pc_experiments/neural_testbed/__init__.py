"""The neural testbed: models choose between candidate batches in environments that are sampled
networks, and every quantity is exact because a batch's labellings can be enumerated."""

from pc_experiments.neural_testbed.candidates import (
    RULES,
    candidates_for,
    design_candidates,
    designer_views,
)
from pc_experiments.neural_testbed.environment import (
    Environment,
    Setting,
    Split,
    class_masses,
    is_valid,
    sample_environment,
    stack,
    valid_environments,
)
from pc_experiments.neural_testbed.experiment import (
    OBJECTIVES,
    Arm,
    Results,
    chunks_of,
    describe,
    gather_inputs,
    merge_chunks,
    run_arm,
    run_chunk,
)
from pc_experiments.neural_testbed.networks import DTYPE, Architecture
from pc_experiments.neural_testbed.pretraining import Tasks, evaluate, task_loss
from pc_experiments.neural_testbed.tuning import (
    Family,
    family_scores,
    grid_points,
    prediction_scores,
)

__all__ = [
    "DTYPE",
    "OBJECTIVES",
    "RULES",
    "Architecture",
    "Arm",
    "Environment",
    "Family",
    "Results",
    "Setting",
    "Split",
    "Tasks",
    "candidates_for",
    "chunks_of",
    "class_masses",
    "describe",
    "design_candidates",
    "designer_views",
    "evaluate",
    "family_scores",
    "gather_inputs",
    "grid_points",
    "is_valid",
    "merge_chunks",
    "prediction_scores",
    "run_arm",
    "run_chunk",
    "sample_environment",
    "stack",
    "task_loss",
    "valid_environments",
]
