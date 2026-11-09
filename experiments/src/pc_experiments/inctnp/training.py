"""Training an incTNP on sequences from a Gaussian process prior, judged by its divergence from
the exact one-step-ahead predictive."""

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Normal, kl_divergence

from pc_experiments.gp.prior import Prior, sample_inputs
from pc_experiments.inctnp.model import to_caller
from pc_experiments.inctnp.network import IncTNPSeq, sequence_nll


@dataclass(frozen=True)
class SequenceTasks:
    """Sequences of `length` points, inputs uniform on `input_range` and outputs from `prior`;
    each point after the first is predicted from those before it."""

    prior: Prior
    input_range: tuple[float, float]
    length: int

    @property
    def max_context(self) -> int:
        return self.length - 1

    def sample(
        self, num_tasks: int, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Inputs of shape `(num_tasks, length, 1)` and outputs `(num_tasks, length)`, float64."""
        x = sample_inputs((num_tasks, self.length), self.input_range, generator)
        return x, self.prior.sample(x, generator)


@dataclass(frozen=True)
class Optimisation:
    """Muon on the hidden weight matrices and AdamW on the rest, sharing `lr` and `weight_decay`:
    a linear warm-up over `warmup_fraction` of the steps, then a cosine to `min_lr`."""

    steps: int
    batch_size: int
    lr: float
    min_lr: float
    warmup_fraction: float
    weight_decay: float
    muon_momentum: float
    grad_clip: float

    def lr_at(self, step: int) -> float:
        warmup = int(self.warmup_fraction * self.steps)
        if step < warmup:
            return self.lr * step / warmup
        progress = (step - warmup) / max(1, self.steps - warmup)
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (1 + math.cos(math.pi * progress))


def split_parameters(network: IncTNPSeq) -> tuple[list[Tensor], list[Tensor]]:
    """The hidden weight matrices, and every other parameter, the input and output layers among
    them."""
    first = network.encoder.xy_encoder.net[0].weight
    last = network.decoder.z_decoder.net[-1].weight
    hidden, other = [], []
    for parameter in network.parameters():
        is_hidden = parameter.ndim == 2 and parameter is not first and parameter is not last
        (hidden if is_hidden else other).append(parameter)
    return hidden, other


def make_optimisers(network: IncTNPSeq, optimisation: Optimisation) -> list[torch.optim.Optimizer]:
    hidden, other = split_parameters(network)
    lr, weight_decay = optimisation.lr, optimisation.weight_decay
    return [
        torch.optim.Muon(
            hidden,
            lr=lr,
            weight_decay=weight_decay,
            momentum=optimisation.muon_momentum,
            adjust_lr_fn="match_rms_adamw",
        ),
        torch.optim.AdamW(other, lr=lr, weight_decay=weight_decay),
    ]


@torch.no_grad()
def evaluate(
    network: IncTNPSeq, prior: Prior, x: Tensor, y: Tensor, batch_size: int = 256
) -> dict[str, Tensor]:
    """The network against the exact predictive on held-out sequences `(T, n, 1)`, `(T, n)`, by
    context size `(n - 1,)`: its `nll`, the exact `oracle_nll`, and the `kl` between them."""
    parameter = next(network.parameters())
    locs, scales = [], []
    for start in range(0, len(x), batch_size):
        stop = start + batch_size
        predictive = network.forward_sequence(
            x[start:stop].to(parameter), y[start:stop].unsqueeze(-1).to(parameter)
        )
        locs.append(to_caller(predictive.loc[..., 0], y))
        scales.append(to_caller(predictive.scale[..., 0], y))
    model = Normal(torch.cat(locs), torch.cat(scales))
    oracle = prior.sequential_predictive(x, y)
    return {
        "nll": -model.log_prob(y[:, 1:]).mean(0),
        "oracle_nll": -oracle.log_prob(y[:, 1:]).mean(0),
        "kl": kl_divergence(oracle, model).mean(0),
    }


def train(
    network: IncTNPSeq,
    tasks: SequenceTasks,
    optimisation: Optimisation,
    validation: tuple[Tensor, Tensor],
    log: Callable[[int, dict[str, float]], None],
    log_every: int,
    validate_every: int,
    partial_path: Path,
    resume: bool = False,
) -> list[dict[str, float]]:
    """`train_network` on sequences from `tasks`, validated on fixed held-out `validation`
    sequences by `nll`, `oracle_nll` and `kl` averaged over context sizes."""
    parameter = next(network.parameters())

    def loss() -> Tensor:
        x, y = tasks.sample(optimisation.batch_size)
        return sequence_nll(network, x.to(parameter), y.unsqueeze(-1).to(parameter))

    def validate() -> dict[str, float]:
        metrics = evaluate(network, tasks.prior, *validation)
        return {key: value.mean().item() for key, value in metrics.items()}

    trained_under = {"tasks": asdict(tasks), "optimisation": asdict(optimisation)}
    return train_network(
        network,
        optimisation,
        loss,
        validate,
        trained_under,
        log,
        log_every,
        validate_every,
        partial_path,
        resume,
    )


def train_network(
    network: IncTNPSeq,
    optimisation: Optimisation,
    loss: Callable[[], Tensor],
    validate: Callable[[], dict[str, float]],
    trained_under: dict[str, Any],
    log: Callable[[int, dict[str, float]], None],
    log_every: int,
    validate_every: int,
    partial_path: Path,
    resume: bool = False,
) -> list[dict[str, float]]:
    """Train `network` in place and return its validation history.

    `loss` is drawn on a fresh batch each step, and `validate` every `validate_every` steps, when
    `partial_path` is also overwritten, which `resume` carries on from if it was `trained_under` the
    same settings. `log` receives the training loss every `log_every` steps and each validation.
    """
    parameter = next(network.parameters())
    optimisers = make_optimisers(network, optimisation)
    history: list[dict[str, float]] = []
    first_step = 0
    if resume and partial_path.exists():
        # Onto the CPU: the optimisers move what belongs with the parameters themselves.
        state = torch.load(partial_path, map_location="cpu", weights_only=False)
        if _without_budget(state["trained_under"]) != _without_budget(trained_under):
            raise ValueError(f"{partial_path} was trained under a different recipe or prior")
        network.load_state_dict(state["network"])
        for optimiser, saved in zip(optimisers, state["optimisers"], strict=True):
            optimiser.load_state_dict(saved)
        history, first_step = state["history"], state["step"]
        # Carry on with sequences the network has not seen, rather than replay the first ones.
        torch.manual_seed(torch.initial_seed() + first_step)

    network.train()
    running_loss = torch.zeros((), device=parameter.device)
    for step in range(first_step, optimisation.steps):
        lr = optimisation.lr_at(step)
        for optimiser in optimisers:
            for group in optimiser.param_groups:
                group["lr"] = lr
            optimiser.zero_grad(set_to_none=True)

        batch_loss = loss()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), optimisation.grad_clip)
        for optimiser in optimisers:
            optimiser.step()
        running_loss += batch_loss.detach()

        done = step + 1
        if done % log_every == 0:
            log(done, {"train/loss": running_loss.item() / log_every, "train/lr": lr})
            running_loss.zero_()
        if done % validate_every == 0 or done == optimisation.steps:
            network.eval()
            history.append({"step": done, **validate()})
            network.train()
            log(done, {f"val/{k}": v for k, v in history[-1].items() if k != "step"})
            state = {
                "network": network.state_dict(),
                "optimisers": [optimiser.state_dict() for optimiser in optimisers],
                "history": history,
                "step": done,
                "trained_under": trained_under,
            }
            # Written beside the old one and then swapped in, so that being stopped part way
            # through leaves the old one to resume from.
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            scratch = partial_path.with_name(partial_path.name + ".tmp")
            torch.save(state, scratch)
            scratch.replace(partial_path)
    network.eval()
    return history


def _without_budget(trained_under: dict[str, Any]) -> dict[str, Any]:
    """What a partial checkpoint must match to be carried on from: everything but the number of
    steps, so that a run can be lengthened or shortened. The cosine follows whatever budget is in
    force, so changing it is a warm restart."""
    optimisation = {k: v for k, v in trained_under.get("optimisation", {}).items() if k != "steps"}
    return {**trained_under, "optimisation": optimisation}


def convergence(history: list[dict[str, float]]) -> dict[str, float]:
    """The final `kl` from the exact predictive, in nats per point, and `kl_tail_drop`, how much
    it fell over the last quarter of training as a fraction of the final value."""
    final = history[-1]["kl"]
    # Entry `i` was recorded after `i + 1` validations' worth of steps.
    earlier = history[max(0, round(0.75 * len(history)) - 1)]["kl"]
    return {"kl": final, "kl_tail_drop": (earlier - final) / final}
