"""A trained incTNP as a predictive model, updated by extending its key-value cache."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor
from torch.distributions import Normal

from pc_experiments.inctnp.attention import Cache
from pc_experiments.inctnp.network import Architecture, IncTNPSeq, build_network
from pc_experiments.outputs import load_run
from predictive_coherence import PredictiveModel


class IncTNPModel(PredictiveModel):
    """An incTNP conditioned on a context, held as the network's key-value cache.

    `update` encodes only the new points, attending to the cache, and appends them to a copy of it.
    It appends a block in the order given, and `joint_sample` draws in that order, so updating on a
    draw reaches the context the draw was made from. Inputs and outputs take the caller's dtype and
    device; the network keeps its own.
    """

    def __init__(self, network: IncTNPSeq) -> None:
        self.network = network
        self.cache: Cache = {}
        self.num_context = 0

    @classmethod
    def fit(cls, network: IncTNPSeq, x: Tensor, y: Tensor) -> Self:
        """Condition on a first context: inputs `(n, dim_x)` or `(n,)`, outputs `(n,)`."""
        return cls(network).update(x, y)

    def update(self, x: Tensor, y: Tensor) -> Self:
        updated = type(self)(self.network)
        # Encoding replaces the cache's entries rather than writing into them, so a shallow copy
        # is enough to leave this model's context as it is.
        updated.cache = dict(self.cache)
        updated.num_context = self.num_context + len(x)
        with torch.no_grad():
            self.network.encode_context(self._to_network(x), self._to_network(y), updated.cache)
        return updated

    def predict(self, x: Tensor) -> Normal:
        if self.num_context == 0:
            raise ValueError("an incTNP has no predictive for an empty context")
        with torch.no_grad():
            predictive = self.network.query(self._to_network(x), self.cache)
        return Normal(
            to_caller(predictive.loc[0, :, 0], x), to_caller(predictive.scale[0, :, 0], x)
        )

    def sample(self, x: Tensor, num_samples: int) -> Tensor:
        return self.predict(x).sample((num_samples,))

    def joint_sample(self, x: Tensor) -> Tensor:
        """Draw outputs one input at a time in the order given, each given the draws before it."""
        model, draws = self, []
        for i in range(len(x)):
            draws.append(model.sample(x[i : i + 1], 1)[0])
            if i < len(x) - 1:
                model = model.update(x[i : i + 1], draws[-1])
        return torch.cat(draws)

    def _to_network(self, values: Tensor) -> Tensor:
        """From `(n,)` or `(n, d)` to a batch of one task, `(1, n, d)`, as the network holds it."""
        return values.reshape(1, len(values), -1).to(next(self.network.parameters()))


def to_caller(values: Tensor, reference: Tensor) -> Tensor:
    """Move the network's output to `reference`'s device, then its dtype: in one step, MPS to a
    float64 CPU tensor returns zeros (torch 2.14)."""
    return values.to(reference.device).to(reference.dtype)


def network_payload(network: IncTNPSeq, architecture: Architecture) -> dict[str, Any]:
    """What `load_network` needs of a checkpoint: how to build the network, and its weights."""
    weights = {name: value.detach().cpu() for name, value in network.state_dict().items()}
    return {"architecture": asdict(architecture), "state_dict": weights}


def load_network(path: Path) -> IncTNPSeq:
    """The network saved at `path`, on the CPU and in the precision it was trained in."""
    checkpoint = load_run(path)
    network = build_network(Architecture(**checkpoint["architecture"]))
    # Built in the default dtype, which need not be the one the weights were trained in.
    network.to(next(iter(checkpoint["state_dict"].values())).dtype)
    network.load_state_dict(checkpoint["state_dict"])
    return network.eval()
