"""Training an incTNP classifier on fresh tasks drawn as the experiment's are: a training set,
a batch appended to it, and targets. It predicts every context point from those before it, and
the targets from the training set and part of the batch, and is judged by its divergence from
the truth at the targets."""

from dataclasses import dataclass

import torch
from torch import Tensor

from pc_experiments.inctnp.network import IncTNPSeq
from pc_experiments.neural_testbed.environment import (
    NUM_CLUSTERS,
    Setting,
    balance_grid,
    balance_offsets,
    class_log_probs,
    cluster_centers,
)
from pc_experiments.neural_testbed.models.inctnp import prefix_logits
from pc_experiments.neural_testbed.networks import DTYPE
from pc_experiments.neural_testbed.scoring import kl_divergence


@dataclass(frozen=True)
class Tasks:
    """Tasks shaped like the experiment's, from its valid environments. A batch is a sample of
    the pool or, with probability `clump_probability`, a clump of spread `clump_std` about one
    pool input, as some designed candidates are. `num_validity` inputs check a task's classes."""

    setting: Setting
    num_targets: int
    num_validity: int
    clump_probability: float
    clump_std: float

    @property
    def max_context(self) -> int:
        """The longest context a task trains: the training set and a whole batch."""
        return self.setting.num_train + self.setting.candidate_size

    def sample(self, num_tasks: int, device: torch.device | str = "cpu") -> dict[str, Tensor]:
        """The valid tasks among `num_tasks` drawn: `x_context` `(m, max_context, 2)`, the
        training set then the batch, `y_context`, `x_target` `(m, num_targets, 2)`, `y_target`,
        and the true `log_p_target` `(m, num_targets, K)`."""
        setting = self.setting
        architecture = setting.architecture
        theta = architecture.sample_prior(num_tasks, device=device)
        logits = architecture.logits(balance_grid(device), theta)
        offset = balance_offsets(logits, setting.temperature)

        def inputs(num: int, clusters: tuple[int, ...], std: float) -> Tensor:
            chosen = torch.randint(len(clusters), (num_tasks, num), device=device)
            centers = cluster_centers(setting.center).to(device)[torch.tensor(clusters).to(device)]
            noise = torch.randn(num_tasks, num, 2, dtype=DTYPE, device=device)
            return centers[chosen] + std * noise

        def log_probs(x: Tensor) -> Tensor:
            return class_log_probs(architecture, x, theta, offset, setting.temperature)

        def labels(log_p: Tensor) -> Tensor:
            return torch.multinomial(log_p.exp().flatten(0, 1), 1).unflatten(0, log_p.shape[:2])[
                ..., 0
            ]

        everywhere = tuple(range(NUM_CLUSTERS))
        batch = inputs(setting.candidate_size, everywhere, setting.pool_std)
        clump = batch[:, :1] + self.clump_std * torch.randn_like(batch)
        clumped = torch.rand(num_tasks, 1, 1, device=device) < self.clump_probability
        x_context = torch.cat(
            [
                inputs(setting.num_train, setting.train_clusters, setting.train_std),
                torch.where(clumped, clump, batch),
            ],
            dim=1,
        )
        x_target = inputs(self.num_targets, setting.target_clusters, setting.target_std)
        x_validity = inputs(self.num_validity, setting.target_clusters, setting.target_std)
        valid = log_probs(x_validity).exp().mean(1).amin(-1) >= setting.min_class_mass

        log_p_target = log_probs(x_target)
        task = {
            "x_context": x_context,
            "y_context": labels(log_probs(x_context)),
            "x_target": x_target,
            "y_target": labels(log_p_target),
            "log_p_target": log_p_target,
        }
        return {key: value[valid] for key, value in task.items()}


def task_loss(network: IncTNPSeq, tasks: Tasks, task: dict[str, Tensor]) -> Tensor:
    """Mean negative log likelihood of a batch of tasks' labels: each context point from those
    before it, and each target from the training set and a random number of the batch's points."""
    parameter = next(network.parameters())
    device = parameter.device
    length, num_train = tasks.max_context, tasks.setting.num_train
    prefix = torch.cat(
        [
            torch.arange(1, length, device=device),
            torch.randint(num_train, length + 1, (tasks.num_targets,), device=device),
        ]
    )
    x = torch.cat([task["x_context"][:, 1:], task["x_target"]], dim=1).to(parameter)
    y = torch.cat([task["y_context"][:, 1:], task["y_target"]], dim=1)
    logits = prefix_logits(network, task["x_context"].to(parameter), task["y_context"], x, prefix)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())


@torch.no_grad()
def evaluate(
    network: IncTNPSeq, tasks: Tasks, task: dict[str, Tensor], batch_size: int = 256
) -> dict[str, float]:
    """Divergence from the truth at held-out tasks' targets: `kl` given the training set, and
    `kl_updated` given the batch and its true labels too."""
    parameter = next(network.parameters())
    num_targets = task["x_target"].shape[1]
    totals = {"kl": 0.0, "kl_updated": 0.0}
    for key, length in (("kl", tasks.setting.num_train), ("kl_updated", tasks.max_context)):
        prefix = torch.full((num_targets,), length, device=parameter.device)
        for start in range(0, len(task["x_target"]), batch_size):
            rows = slice(start, start + batch_size)
            logits = prefix_logits(
                network,
                task["x_context"][rows].to(parameter),
                task["y_context"][rows].to(parameter.device),
                task["x_target"][rows].to(parameter),
                prefix,
            )
            log_q = torch.log_softmax(logits.to(DTYPE), dim=-1).to(task["log_p_target"].device)
            totals[key] += kl_divergence(task["log_p_target"][rows], log_q).sum().item()
    count = task["log_p_target"][..., 0].numel()
    return {key: total / count for key, total in totals.items()}
