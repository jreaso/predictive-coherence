"""Coherence of predictive models on real tabular datasets."""

from pc_experiments.tabular.datasets import (
    CLASSIFICATION,
    REGRESSION,
    Sizes,
    Task,
    load,
    make_task,
    split_rows,
)
from pc_experiments.tabular.experiment import (
    Arm,
    Job,
    Results,
    Setting,
    cross_validate,
    fit_arm,
    result_path,
    run_arm,
    run_jobs,
    score,
    worker_setup,
)

__all__ = [
    "CLASSIFICATION",
    "REGRESSION",
    "Arm",
    "Job",
    "Results",
    "Setting",
    "Sizes",
    "Task",
    "cross_validate",
    "fit_arm",
    "load",
    "make_task",
    "split_rows",
    "result_path",
    "run_arm",
    "run_jobs",
    "score",
    "worker_setup",
]
