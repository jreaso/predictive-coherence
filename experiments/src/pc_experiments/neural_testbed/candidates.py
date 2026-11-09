"""Candidate batches that are good in different ways, so that objectives can disagree.

Most are designed, each by its own rule (the most informative inputs, the same within the target
or training clusters, a redundant clump, coverage, the decision boundary, a mix), and the rest
sampled. They depend on the environment alone, so every model faces the same decision; rules that
need a model's view of the pool use a fixed designer fitted to the training data.
"""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from pc_experiments.neural_testbed.environment import (
    NUM_CLUSTERS,
    Environment,
    Setting,
    generator,
)
from pc_experiments.neural_testbed.models.laplace import Laplace, LaplaceConfig
from pc_experiments.neural_testbed.scoring import entropy

DESIGNER = LaplaceConfig(curvature="diag", prior_scale=4.0, num_atoms=128, lr=0.05, fit_steps=200)

# In order of how well they tell objectives apart, so that fewer candidates keep the most telling.
# Beyond these, candidates are sampled.
RULES = (
    "informative",
    "informative_target",
    "redundant",
    "coverage_target",
    "informative_train",
    "coverage",
    "boundary",
    "mixed",
)


def designer_views(setting: Setting, chunk: Environment) -> tuple[Tensor, Tensor]:
    """For each pool input, `(E, num_pool)` on the CPU: the mutual information between its label
    and the designer's parameters, and the margin between the designer's two likeliest classes."""
    # Its own streams, so that it is not the Laplace arm evaluated with the same settings.
    designer = Laplace.fit(
        DESIGNER, setting, chunk.train.x, chunk.train.y, chunk.seeds, stream="designer"
    )
    atom_log_probs = designer.atom_log_probs(chunk.pool.x)  # (E, A, m, K)
    mixture = atom_log_probs.exp().mean(1)
    information = entropy(mixture.clamp_min(1e-300).log()) - entropy(atom_log_probs).mean(1)
    top = mixture.topk(2, dim=-1).values
    return information.cpu(), (top[..., 0] - top[..., 1]).cpu()


def design_candidates(
    setting: Setting, environment: Environment, information: Tensor, margin: Tensor
) -> tuple[Tensor, list[str]]:
    """One environment's candidates as pool indices `(num_candidates, candidate_size)`, and the
    rule behind each. A designed candidate that repeats an earlier one is replaced by a sampled
    one, named `sampled_for_<rule>`."""
    x, cluster, size = environment.pool.x, environment.pool.cluster, setting.candidate_size
    everywhere = torch.arange(len(x))
    on_target = everywhere[torch.isin(cluster, torch.tensor(setting.target_clusters))]
    on_train = everywhere[torch.isin(cluster, torch.tensor(setting.train_clusters))]
    # Chosen inputs are kept this far apart, where there are enough of them to allow it.
    apart = setting.train_std

    def redundant() -> Tensor:
        """The most informative input and its nearest neighbours: worth a lot one at a time and
        little together, which is where objectives disagree most."""
        distance = (x - x[information.argmax()]).square().sum(-1)
        return distance.argsort(stable=True)[:size]

    def mixed() -> Tensor:
        """The most informative inputs of the target and training clusters, taken in turn."""
        by_target = on_target[information[on_target].argsort(descending=True, stable=True)]
        by_train = on_train[information[on_train].argsort(descending=True, stable=True)]
        taken: list[int] = []
        for pair in zip(by_target.tolist(), by_train.tolist(), strict=False):
            for index in pair:  # one at a time: a cluster can be both a target and a training one
                if index not in taken:
                    taken.append(index)
        return torch.tensor(taken[:size])

    rules: dict[str, Callable[[], Tensor]] = {
        "informative": lambda: _spaced_top(information, x, everywhere, size, apart),
        "informative_target": lambda: _spaced_top(information, x, on_target, size, apart),
        "redundant": redundant,
        "coverage_target": lambda: _farthest_points(x, on_target, size),
        "informative_train": lambda: _spaced_top(information, x, on_train, size, apart),
        "coverage": lambda: _farthest_points(x, everywhere, size),
        "boundary": lambda: _spaced_top(-margin, x, everywhere, size, apart),
        "mixed": mixed,
    }

    rng = generator(environment.seed, "candidates")
    names = [*RULES, *["sampled"] * setting.num_candidates][: setting.num_candidates]
    batches: list[Tensor] = []
    seen: set[tuple[int, ...]] = set()
    for position, name in enumerate(names):
        batch = _sample_batch(cluster, size, rng) if name == "sampled" else rules[name]()
        while tuple(sorted(batch.tolist())) in seen:
            batch = _sample_batch(cluster, size, rng)
            names[position] = name if name == "sampled" else f"sampled_for_{name}"
        seen.add(tuple(sorted(batch.tolist())))
        batches.append(batch)
    return torch.stack(batches), names


def _spaced_top(scores: Tensor, x: Tensor, members: Tensor, size: int, apart: float) -> Tensor:
    """The `size` highest-scoring of `members`, no two closer than `apart`, which is halved
    whenever a pass cannot fill the batch."""
    order = members[scores[members].argsort(descending=True, stable=True)].tolist()
    chosen: list[int] = []
    while len(chosen) < min(size, len(order)):
        for index in order:
            if len(chosen) == size:
                break
            if index not in chosen and all((x[index] - x[c]).norm() >= apart for c in chosen):
                chosen.append(index)
        apart /= 2
    return torch.tensor(chosen)


def _farthest_points(x: Tensor, members: Tensor, size: int) -> Tensor:
    """`size` of `members` spread out: from the one nearest their centroid, repeatedly the one
    farthest from all chosen so far."""
    points = x[members]
    chosen = [int((points - points.mean(0)).square().sum(-1).argmin())]
    distance = (points - points[chosen[0]]).square().sum(-1)
    while len(chosen) < min(size, len(members)):
        chosen.append(int(distance.argmax()))
        distance = torch.minimum(distance, (points - points[chosen[-1]]).square().sum(-1))
    return members[torch.tensor(chosen)]


def _sample_batch(cluster: Tensor, size: int, rng: torch.Generator) -> Tensor:
    """A batch with a random mix of clusters, its shares from a Dirichlet(1/2), drawn without
    replacement within clusters and from the rest of the pool where a cluster runs out."""
    shares = torch.randn(NUM_CLUSTERS, generator=rng, dtype=torch.float64).square()
    counts = torch.multinomial(shares / shares.sum(), size, replacement=True, generator=rng)
    chosen: list[int] = []
    for index, count in enumerate(counts.bincount(minlength=NUM_CLUSTERS).tolist()):
        members = (cluster == index).nonzero().squeeze(-1)
        order = torch.randperm(len(members), generator=rng)[:count]
        chosen += members[order].tolist()
    if len(chosen) < size:
        rest = torch.tensor([i for i in range(len(cluster)) if i not in chosen])
        chosen += rest[torch.randperm(len(rest), generator=rng)[: size - len(chosen)]].tolist()
    return torch.tensor(chosen)


def candidates_for(
    setting: Setting, chunk: Environment, environments: Sequence[Environment]
) -> tuple[Tensor, list[list[str]]]:
    """Candidates for every environment of a chunk, `(E, num_candidates, candidate_size)` on the
    chunk's device, with their rules. `environments` are the chunk's, one at a time on the CPU."""
    information, margin = designer_views(setting, chunk)
    designed = [
        design_candidates(setting, environment, information[e], margin[e])
        for e, environment in enumerate(environments)
    ]
    indices = torch.stack([batch for batch, _ in designed]).to(chunk.pool.x.device)
    return indices, [names for _, names in designed]
