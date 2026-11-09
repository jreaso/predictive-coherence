"""An incTNP trained on the testbed's own environments, as a classifier.

The network of `pc_experiments.inctnp` with class logits for a head and one-hot labels in its
tokens. `update` appends a batch to the context in the order given, and the joint over a batch's
labels is autoregressive in that order,

    p(z) = prod_j p(z_j | context, (x_1, z_1), ..., (x_{j-1}, z_{j-1})),

so updating on a labelling reaches the context that gave it its probability. The network keeps
its training precision; its logits are cast to float64.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn

from pc_experiments.inctnp.attention import Cache
from pc_experiments.inctnp.network import Architecture, IncTNPSeq, build_network
from pc_experiments.neural_testbed.models.base import MAX_ELEMENTS, Model
from pc_experiments.neural_testbed.networks import DTYPE
from pc_experiments.outputs import load_run


class ClassLogits(nn.Module):
    """The head: what the decoder returns is the logits of a categorical predictive."""

    def forward(self, x: Tensor) -> Tensor:
        return x


def build_classifier(architecture: Architecture) -> IncTNPSeq:
    """An incTNP over `architecture.dim_y` classes: tokens are `(x, one-hot y, flag)`, and the
    decoder returns one logit for each class."""
    return build_network(architecture, ClassLogits(), per_output=1)


def prefix_logits(
    network: IncTNPSeq, x_context: Tensor, y_context: Tensor, x_target: Tensor, prefix: Tensor
) -> Tensor:
    """Class logits `(m, nt, K)` at targets `(m, nt, dim_x)`, each given the first `prefix`
    `(nt,)` points of the context `(m, nc, dim_x)`, `(m, nc)`, in one causal pass."""
    one_hot = nn.functional.one_hot(y_context, network.dim_y).to(x_context.dtype)
    zc = network.encoder.preprocess_context(x_context, one_hot)
    zt = network.encoder.preprocess_targets(x_target, network.dim_y)
    mask = torch.arange(x_context.shape[1], device=prefix.device) < prefix.unsqueeze(-1)
    hidden = network.encoder.transformer_encoder(zc, zt, mask_ca=mask, use_causal=True)
    return network.decoder(hidden)


class IncTNPClassifier(Model):
    """A trained network conditioned on a context, for a chunk of environments.

    Attributes:
        cache: the fitted context of each environment, as the network's key-value cache.
        blocks: once updated, each layer's representations of the batch that each update
            appended, shape `(E, U, k, dz)`. The fitted context's are shared by an environment's
            updates and stay in `cache`.
    """

    def __init__(self, network: IncTNPSeq, cache: Cache, num_context: int) -> None:
        self.network = network
        self.cache = cache
        self.num_context = num_context
        self.blocks: list[Tensor] = []

    @classmethod
    @torch.no_grad()
    def fit(cls, network: IncTNPSeq, x: Tensor, y: Tensor) -> Self:
        """Condition on inputs `(E, n, d)` and labels `(E, n)`, in the order given."""
        model = cls(network, {}, x.shape[1])
        one_hot = nn.functional.one_hot(y, network.dim_y)
        network.encode_context(model._cast(x), model._cast(one_hot), model.cache)
        return model

    def _cast(self, values: Tensor) -> Tensor:
        return values.to(next(self.network.parameters()).dtype)

    def _layers(self) -> nn.ModuleList:
        return self.network.encoder.transformer_encoder.mhca_layers

    @torch.no_grad()
    def _append(self, x: Tensor, y: Tensor) -> Cache:
        """The cache of each context extended by its environment's batch `(E, k, d)` under each
        labelling `(E, T, k)`: `E * T` contexts."""
        repeats = y.shape[1]
        cache: Cache = {
            tag: tuple(part.repeat_interleave(repeats, dim=0) for part in entry)
            if isinstance(entry, tuple)
            else entry.repeat_interleave(repeats, dim=0)
            for tag, entry in self.cache.items()
        }  # type: ignore[assignment]
        one_hot = nn.functional.one_hot(y.flatten(0, 1), self.network.dim_y)
        inputs = x.repeat_interleave(repeats, dim=0)
        self.network.encode_context(self._cast(inputs), self._cast(one_hot), cache)
        return cache

    @torch.no_grad()
    def log_joint(self, x: Tensor, labellings: Tensor) -> Tensor:
        num_envs, num_batches, size, _ = x.shape
        num_classes = self.network.dim_y
        # Input `j` of a batch is predicted from the fitted context and the batch before it.
        positions = torch.arange(self.num_context + size, device=x.device)
        visible = self.num_context + torch.arange(size, device=x.device)
        mask = positions < visible.unsqueeze(-1)
        rows = []
        for batch in range(num_batches):
            y = labellings.expand(num_envs, -1, -1)
            cache = self._append(x[:, batch], y)
            inputs = self._cast(x[:, batch].repeat_interleave(len(labellings), dim=0))
            hidden = self.network.encoder.preprocess_targets(inputs, num_classes)
            for index, layer in enumerate(self._layers()):
                hidden = layer(hidden, cache[f"context_layer_{index}"], mask=mask)
            log_probs = torch.log_softmax(self.network.decoder(hidden).to(DTYPE), dim=-1)
            chosen = log_probs.gather(-1, y.flatten(0, 1).unsqueeze(-1)).squeeze(-1).sum(-1)
            rows.append(chosen.unflatten(0, (num_envs, -1)))
        return torch.stack(rows, dim=1)

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor) -> Self:
        updated = type(self)(self.network, self.cache, self.num_context)
        updated.update_shape = tuple(y.shape[1:3])
        per_layer: list[list[Tensor]] = [[] for _ in self._layers()]
        for batch in range(x.shape[1]):
            cache = self._append(x[:, batch], y[:, batch])
            for index, blocks in enumerate(per_layer):
                context: Tensor = cache[f"context_layer_{index}"]  # type: ignore[assignment]
                blocks.append(context[:, self.num_context :].unflatten(0, (len(x), -1)))
        updated.blocks = [torch.cat(blocks, dim=1) for blocks in per_layer]
        return updated

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        num_classes = self.network.dim_y
        if not self.update_shape:
            hidden = self.network.encoder.preprocess_targets(self._cast(x), num_classes)
            for index, layer in enumerate(self._layers()):
                hidden = layer(hidden, self.cache[f"context_layer_{index}"])
            return torch.log_softmax(self.network.decoder(hidden).to(DTYPE), dim=-1)

        num_envs, num_points = x.shape[:2]
        num_updates, width = self.blocks[0].shape[1], self.blocks[0].shape[-1]
        # Twice a token's width, to leave room for a layer's intermediates.
        size = max(1, MAX_ELEMENTS // (num_envs * num_points * 2 * width))
        targets = self.network.encoder.preprocess_targets(self._cast(x), num_classes)
        chunks = []
        for start in range(0, num_updates, size):
            chosen = slice(start, start + size)
            count = self.blocks[0][:, chosen].shape[1]
            hidden = targets.repeat_interleave(count, dim=0)
            for index, layer in enumerate(self._layers()):
                fitted: Tensor = self.cache[f"context_layer_{index}"]  # type: ignore[assignment]
                context = torch.cat(
                    [fitted.unsqueeze(1).expand(-1, count, -1, -1), self.blocks[index][:, chosen]],
                    dim=2,
                )
                hidden = layer(hidden, context.flatten(0, 1))
            logits = self.network.decoder(hidden).to(DTYPE).unflatten(0, (num_envs, count))
            chunks.append(torch.log_softmax(logits, dim=-1))
        return torch.cat(chunks, dim=1).unflatten(1, self.update_shape)


def classifier_payload(network: IncTNPSeq, architecture: Architecture) -> dict[str, Any]:
    """What `load_classifier` needs of a checkpoint: how to build the network, and its weights."""
    weights = {name: value.detach().cpu() for name, value in network.state_dict().items()}
    return {"architecture": asdict(architecture), "state_dict": weights}


def load_classifier(path: Path) -> IncTNPSeq:
    """The classifier saved at `path`, on the CPU and in the precision it was trained in."""
    checkpoint = load_run(path)
    network = build_classifier(Architecture(**checkpoint["architecture"]))
    # Built in the default dtype, which need not be the one the weights were trained in.
    network.to(next(iter(checkpoint["state_dict"].values())).dtype)
    network.load_state_dict(checkpoint["state_dict"])
    return network.eval()
