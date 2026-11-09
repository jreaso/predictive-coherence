"""Choosing hyperparameters by how well a model predicts, never by incoherence or an objective.

A family's fit is chosen by the divergence from the truth to the fitted predictive, and its update
by the same divergence after an update on a candidate's true labels. Both are measured at held-out
inputs like the data seen, not at the test inputs, which lie mostly where no data has been and
where a model that learned nothing scores as well as one that did. An update is scored around the
training inputs and around the batch, so that neither forgetting nor not learning is free.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from pc_experiments.neural_testbed.environment import (
    Environment,
    Setting,
    class_log_probs,
    generator,
    sample_inputs,
)
from pc_experiments.neural_testbed.experiment import Arm, gather_inputs
from pc_experiments.neural_testbed.networks import DTYPE
from pc_experiments.neural_testbed.scoring import kl_divergence

# Held-out inputs drawn as the training inputs are, for each environment.
HELD_OUT = 512
# Held-out inputs around each input of a candidate batch.
AROUND_BATCH = 16


@dataclass(frozen=True)
class Family:
    """Arms of one `kind` that share `hyperparameters`, with what is each arm's own in `arms`."""

    kind: str
    hyperparameters: dict[str, Any]
    arms: dict[str, dict[str, Any]]

    def build(self, tuned: Mapping[str, Any] | None = None) -> list[Arm]:
        """The family's arms, with `tuned` hyperparameters in place of the shared ones."""
        shared = {**self.hyperparameters, **(tuned or {})}
        return [Arm(name, self.kind, {**shared, **own}) for name, own in self.arms.items()]


def grid_points(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Every combination of a grid's values. One empty combination for an empty grid."""
    return [dict(zip(grid, values, strict=True)) for values in itertools.product(*grid.values())]


def held_out_inputs(
    setting: Setting, chunk: Environment, candidates: Tensor
) -> tuple[Tensor, Tensor]:
    """Held-out inputs drawn as the training inputs are, `(E, HELD_OUT, d)`, and `AROUND_BATCH`
    about each input of each candidate `(E, N, k)`, `(E, N, k * AROUND_BATCH, d)`."""
    device = chunk.pool.x.device
    like_training = [
        sample_inputs(
            HELD_OUT,
            setting.train_clusters,
            setting.center,
            setting.train_std,
            generator(seed, "held_out"),
        )[0]
        for seed in chunk.seeds
    ]
    batch = gather_inputs(chunk.pool.x, candidates)  # (E, N, k, d)
    shape = (*batch.shape[1:3], AROUND_BATCH, batch.shape[-1])
    noise = [
        torch.randn(shape, generator=generator(seed, "around_batch"), dtype=DTYPE)
        for seed in chunk.seeds
    ]
    around = batch.unsqueeze(-2) + setting.train_std * torch.stack(noise).to(device)
    return torch.stack(like_training).to(device), around.flatten(2, 3)


@torch.no_grad()
def prediction_scores(
    arm: Arm, setting: Setting, chunk: Environment, candidates: Tensor, updated: bool
) -> Tensor:
    """An arm's divergence from the truth at held-out inputs like the training data, `(E,)`; or,
    if `updated`, after an update on each candidate's true labels, there and around the batch,
    weighted by their numbers of points and averaged over candidates."""

    def truth(x: Tensor) -> Tensor:
        return class_log_probs(
            setting.architecture, x, chunk.theta, chunk.offset, setting.temperature
        )

    like_training, around = held_out_inputs(setting, chunk, candidates)
    with torch.enable_grad():
        model = arm.fit(setting, chunk)
        if not updated:
            return kl_divergence(truth(like_training), model.predict(like_training)).mean(-1).cpu()
        x = gather_inputs(chunk.pool.x, candidates)
        model = model.update(x, gather_inputs(chunk.pool.y, candidates).unsqueeze(2))
    predicted = model.predict(like_training)[:, :, 0]  # (E, N, m, K)
    at_training = kl_divergence(truth(like_training).unsqueeze(1), predicted).mean(-1)
    # Every updated model predicts around every candidate; each is scored around its own.
    num_candidates = candidates.shape[1]
    everywhere = around.flatten(1, 2)
    divergence = kl_divergence(truth(everywhere).unsqueeze(1), model.predict(everywhere)[:, :, 0])
    at_batch = divergence.unflatten(-1, (num_candidates, -1)).diagonal(dim1=1, dim2=2).mean(1)
    seen, new = setting.num_train, setting.candidate_size
    return ((seen * at_training + new * at_batch) / (seen + new)).mean(-1).cpu()


def family_scores(
    family: Family,
    tuned: Mapping[str, Any],
    setting: Setting,
    chunks: Sequence[tuple[Environment, Tensor]],
    updated: bool,
) -> Tensor:
    """A family's `prediction_scores` at some hyperparameters, averaged over its arms."""
    per_arm = []
    for arm in family.build(tuned):
        scores = [prediction_scores(arm, setting, *chunk, updated) for chunk in chunks]
        per_arm.append(torch.cat(scores))
    return torch.stack(per_arm).mean(0)
