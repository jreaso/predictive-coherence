"""One-step batch acquisition on sampled networks, with every quantity exact.

A model is fitted in each environment and offered candidate batches. Every labelling of every
candidate is enumerated and the model updated on each, giving the objectives, the incoherence of
each update, and, weighting labellings by their true probability, the loss each choice leads to.
Environments run in chunks, in which a model is a batch of classifiers.
"""

import dataclasses
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from pc_experiments.neural_testbed.candidates import candidates_for
from pc_experiments.neural_testbed.environment import (
    Environment,
    Setting,
    stack,
    valid_environments,
)
from pc_experiments.neural_testbed.models import (
    BBB,
    IVON,
    BBBConfig,
    Dropout,
    DropoutConfig,
    Ensemble,
    EnsembleConfig,
    IncTNPClassifier,
    IVONConfig,
    Laplace,
    LaplaceConfig,
    Model,
    PriorBank,
    PriorBankConfig,
    load_classifier,
)
from pc_experiments.neural_testbed.scoring import (
    coherence_of_random_updates,
    entropy,
    enumerate_labellings,
    expected_loss,
    joint_kl,
    kl_divergence,
    labelling_log_probs,
    score_candidates,
)
from pc_experiments.outputs import load_run

# The acquisition objectives, which are also keys of the results.
OBJECTIVES = ("epig", "epig_kl", "emi")
# Measured from two independent draws of a random update; the first draw's own carry `ONE_DRAW`.
INCOHERENCE = ("c", "v", "psi")
ONE_DRAW = "_plugin"
# The stream a random update's second draw comes from; decisions use the first.
SECOND_DRAW = "second"

KINDS: dict[str, tuple[Any, Any]] = {
    "prior_bank": (PriorBank, PriorBankConfig),
    "laplace": (Laplace, LaplaceConfig),
    "bbb": (BBB, BBBConfig),
    "ivon": (IVON, IVONConfig),
    "ensemble": (Ensemble, EnsembleConfig),
    "dropout": (Dropout, DropoutConfig),
}


@dataclass(frozen=True)
class Arm:
    """A model under evaluation: a `kind` from `KINDS` or `inctnp`, with its configuration's
    fields as `hyperparameters`, or for `inctnp` its `checkpoint`."""

    name: str
    kind: str
    hyperparameters: dict[str, Any]

    def fit(self, setting: Setting, chunk: Environment) -> Model:
        """The arm fitted to the training data of each environment in a chunk."""
        x, y = chunk.train.x, chunk.train.y
        if self.kind == "inctnp":
            network = _classifier(Path(self.hyperparameters["checkpoint"]), str(x.device))
            return IncTNPClassifier.fit(network, x, y)
        model, config = KINDS[self.kind]
        return model.fit(config(**self.hyperparameters), setting, x, y, chunk.seeds)

    @property
    def random_update(self) -> bool:
        """Whether the arm's update draws randomness of its own."""
        model = IncTNPClassifier if self.kind == "inctnp" else KINDS[self.kind][0]
        return model.random_update


@cache
def _classifier(checkpoint: Path, device: str):
    """Loaded once for a run."""
    return load_classifier(checkpoint).to(device)


# ----- running ------------------------------------------------------------------------------------


def gather_inputs(values: Tensor, indices: Tensor) -> Tensor:
    """Rows `indices` `(E, N, k)` of per-environment `values` `(E, m, ...)`, `(E, N, k, ...)`."""
    environments = torch.arange(len(values), device=values.device)[:, None, None]
    return values[environments, indices]


def expected_losses(
    model: Model, x: Tensor, log_truth: Tensor, log_joint_truth: Tensor, points: int = 512
) -> Tensor:
    """`sum_z p*(z) mean_m KL(p*(. | x_m) || q_z(. | x_m))` for each candidate, `(E, N)`, from
    updated models of update shape `(N, Z)`, inputs `(E, m, d)`, the truth there `(E, m, K)` and
    each labelling's true log probability `(E, N, Z)`, `points` inputs at a time."""
    total = torch.zeros_like(log_joint_truth[..., 0])
    for start in range(0, x.shape[1], points):
        chosen = slice(start, start + points)
        log_q = model.predict(x[:, chosen])
        block = expected_loss(log_truth[:, None, chosen], log_q, log_joint_truth)
        total += block * log_q.shape[-2]
    return total / x.shape[1]


@torch.no_grad()
def run_arm(
    arm: Arm,
    setting: Setting,
    chunk: Environment,
    candidates: Tensor,
    labellings: Tensor,
    test_losses: bool = True,
) -> dict[str, Tensor]:
    """Measure one arm on a chunk of environments, at candidates `(E, N, k)` with every
    labelling `(Z, k)`.

    The objectives, choices and losses come from one draw of an update's randomness, if it has
    any. A random update is run again from an independent stream, so that `c`, `v` and `psi`
    average over its randomness (`coherence_of_random_updates`); the first draw's own are kept
    as `c_plugin`, `v_plugin` and `psi_plugin`.

    Returns:
        Of shape `(E,)`: `initial_kl` and `initial_kl_target`, the fitted predictive's divergence
        from the truth at test and target inputs, and the wall time of each stage.

        Of shape `(E, N)`: what `score_candidates` returns; `loss` and, with `test_losses`,
        `loss_test`, the expected divergence from the truth after an update on the candidate's
        true labels; and `joint_kl`, from the labels' true distribution to the model's joint.
    """
    clock = _Clock(chunk.theta.device)
    with torch.enable_grad():
        model = arm.fit(setting, chunk)
    fit_seconds = clock.lap()

    x = gather_inputs(chunk.pool.x, candidates)
    log_joint_truth = labelling_log_probs(gather_inputs(chunk.pool.log_p, candidates), labellings)
    log_p = model.predict(chunk.target.x)
    log_joint = model.log_joint(x, labellings)
    results = {
        "initial_kl": kl_divergence(chunk.test.log_p, model.predict(chunk.test.x)).mean(-1),
        "initial_kl_target": kl_divergence(chunk.target.log_p, log_p).mean(-1),
        "joint_kl": joint_kl(log_joint_truth, log_joint),
    }
    clock.lap()
    y = labellings.expand(*candidates.shape[:2], -1, -1)
    with torch.enable_grad():
        updated = model.update(x, y)
    update_seconds = clock.lap()

    log_q = updated.predict(chunk.target.x)
    results |= score_candidates(log_p.unsqueeze(1), log_q, log_joint)
    results |= {key + ONE_DRAW: results[key] for key in INCOHERENCE}
    results["loss"] = expected_loss(chunk.target.log_p.unsqueeze(1), log_q, log_joint_truth)
    if test_losses:
        results["loss_test"] = expected_losses(
            updated, chunk.test.x, chunk.test.log_p, log_joint_truth
        )
    score_seconds = clock.lap()

    if model.random_update:
        with torch.enable_grad():
            again = model.redrawn(SECOND_DRAW).update(x, y)
        log_q_again = again.predict(chunk.target.x)
        results |= coherence_of_random_updates(log_p.unsqueeze(1), log_q, log_q_again, log_joint)
    seconds = {
        "fit": fit_seconds,
        "update": update_seconds,
        "score": score_seconds,
        "redraw": clock.lap() if model.random_update else 0.0,
    }
    for stage, elapsed in seconds.items():
        results[f"{stage}_seconds"] = torch.full_like(results["initial_kl"], elapsed / len(log_p))
    return {key: value.cpu() for key, value in results.items()}


class _Clock:
    """Wall time between laps, waiting for a GPU to finish what it was given."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.last = self._now()

    def _now(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def lap(self) -> float:
        now = self._now()
        elapsed, self.last = now - self.last, now
        return elapsed


def run_chunk(
    setting: Setting,
    arms: Sequence[Arm],
    environments: Sequence[Environment],
    device: str,
    test_losses: bool = True,
) -> dict[str, Any]:
    """Measure every arm on a chunk of environments: their `seeds`, `candidates` `(E, N, k)` and
    the `rules` behind them, `bayes_entropy`, the truth's entropy at the test inputs `(E,)`, and
    `values`, what `run_arm` returns stacked over arms."""
    chunk = stack(environments, device)
    candidates, rules = candidates_for(setting, chunk, environments)
    labellings = enumerate_labellings(
        setting.candidate_size, setting.architecture.num_classes, device
    )
    per_arm = [run_arm(arm, setting, chunk, candidates, labellings, test_losses) for arm in arms]
    return {
        "seeds": list(chunk.seeds),
        "candidates": candidates.cpu(),
        "rules": rules,
        "bayes_entropy": entropy(chunk.test.log_p).mean(-1).cpu(),
        "values": {key: torch.stack([values[key] for values in per_arm]) for key in per_arm[0]},
    }


def chunks_of(setting: Setting, num_environments: int, size: int) -> Iterator[list[Environment]]:
    """The first `num_environments` valid environments, `size` at a time."""
    source = valid_environments(setting)
    for start in range(0, num_environments, size):
        yield [next(source) for _ in range(min(size, num_environments - start))]


def merge_chunks(chunks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Chunks from `run_chunk`, joined along their environments."""
    return {
        "seeds": [seed for chunk in chunks for seed in chunk["seeds"]],
        "candidates": torch.cat([chunk["candidates"] for chunk in chunks]),
        "rules": [names for chunk in chunks for names in chunk["rules"]],
        "bayes_entropy": torch.cat([chunk["bayes_entropy"] for chunk in chunks]),
        "values": {
            key: torch.cat([chunk["values"][key] for chunk in chunks], dim=1)
            for key in chunks[0]["values"]
        },
    }


# ----- reading results back -----------------------------------------------------------------------


@dataclass(frozen=True)
class Results:
    """A saved run. `values[key]` has shape `(arms, environments)`, or
    `(arms, environments, candidates)` for what is measured at each candidate."""

    arms: list[str]
    config: dict[str, Any]
    seeds: list[int]
    rules: list[list[str]]
    bayes_entropy: Tensor
    values: dict[str, Tensor]

    @classmethod
    def load(cls, path: Path) -> "Results":
        run = load_run(path)
        results = run["results"]
        return cls(
            [arm["name"] for arm in run["arms"]],
            run["manifest"]["config"],
            results["seeds"],
            results["rules"],
            results["bayes_entropy"],
            results["values"],
        )

    def of(self, key: str, arm: str) -> Tensor:
        return self.values[key][self.arms.index(arm)]

    @property
    def drawn_twice(self) -> bool:
        """Whether the run drew random updates twice; one that did not needs `rescored`."""
        return all(key + ONE_DRAW in self.values for key in INCOHERENCE)

    def head(self, num_environments: int) -> "Results":
        """The run's first `num_environments` environments."""
        return dataclasses.replace(
            self,
            config=self.config | {"num_environments": num_environments},
            seeds=self.seeds[:num_environments],
            rules=self.rules[:num_environments],
            bayes_entropy=self.bayes_entropy[:num_environments],
            values={key: value[:, :num_environments] for key, value in self.values.items()},
        )

    def rescored(self, other: "Results") -> "Results":
        """This run's first environments, with every value `other`, a re-score of some of its
        arms by `run_two_draw.py`, holds in place of this run's. What `other` did not measure is
        kept, which is sound only where its first draw is this run's (`run_two_draw.py --check`).
        """
        if other.seeds != self.seeds[: len(other.seeds)]:
            raise ValueError("a re-score must be of this run's first environments")
        unknown = set(other.arms) - set(self.arms)
        if unknown:
            raise ValueError(f"a re-score of arms this run does not have: {sorted(unknown)}")
        head = self.head(len(other.seeds))
        values = dict(head.values)
        for key in INCOHERENCE:
            values.setdefault(key + ONE_DRAW, values[key])
        values = {key: value.clone() for key, value in values.items()}
        for arm in other.arms:
            for key in values.keys() & other.values.keys():
                values[key][head.arms.index(arm)] = other.of(key, arm)
        run = f"{self.config.get('run', '')}, re-scored by {other.config.get('run', '')}"
        return dataclasses.replace(head, config=head.config | {"run": run}, values=values)

    def median(self, key: str, arm: str) -> Tensor:
        """Each environment's median over its candidates, `(environments,)`."""
        return self.of(key, arm).nanquantile(0.5, dim=-1)

    def chosen(self, objective: str, arm: str) -> Tensor:
        return self.of(objective, arm).argmax(-1)

    def loss(self, objective: str, arm: str, split: str = "target") -> Tensor:
        """The expected loss an objective's choice leads to at the `target` or `test` inputs."""
        losses = self.of({"target": "loss", "test": "loss_test"}[split], arm)
        return losses.gather(-1, self.chosen(objective, arm).unsqueeze(-1)).squeeze(-1)

    def regret_gap(self, arm: str, objective: str = "epig", split: str = "target") -> Tensor:
        """How much more an objective's choice loses than EMI's, `(environments,)`."""
        return self.loss(objective, arm, split) - self.loss("emi", arm, split)

    def disagreement(self, arm: str, first: str = "epig", second: str = "emi") -> Tensor:
        """Whether two objectives pick different candidates, `(environments,)`."""
        return self.chosen(first, arm) != self.chosen(second, arm)

    def distortion(self, arm: str) -> Tensor:
        """How far incoherence moves the objective: the standard deviation over candidates of
        entropy-form EPIG minus EMI, which coincide under coherence, `(environments,)`."""
        return (self.of("epig", arm) - self.of("emi", arm)).std(-1)

    def psi(self, arm: str) -> Tensor:
        """Each environment's Ψ, its median over candidates, `(environments,)`."""
        return self.median("psi", arm)


def describe(arm: Arm) -> dict[str, Any]:
    """An arm as saved with a run."""
    return dataclasses.asdict(arm)
