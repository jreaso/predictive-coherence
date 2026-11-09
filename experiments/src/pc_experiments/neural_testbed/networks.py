"""A small classifier network as a function of flat parameters `(..., P)`, rather than an
`nn.Module`, so that many networks run at once, with inputs broadcasting against them."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor

# Everything outside a trained transformer is float64. An exactly coherent model's incoherence is
# a difference of nearly equal quantities, and in float32 its zero is lost to rounding.
DTYPE = torch.float64


# Matrices factorised at a time by `batched_eigh`, which bounds its scratch to a gigabyte or two.
EIGH_SLICE = 4096


def batched_eigh(matrix: Tensor, slice_size: int = EIGH_SLICE) -> tuple[Tensor, Tensor]:
    """`torch.linalg.eigh` of symmetric matrices `(..., n, n)`, a slice of the batch at a time: a
    GPU reserves scratch for every matrix of a batched decomposition, however small."""
    count = matrix.numel() // matrix.shape[-1] ** 2
    if count <= slice_size:
        return torch.linalg.eigh(matrix)
    flat = matrix.reshape(count, *matrix.shape[-2:])
    factorised = [torch.linalg.eigh(part) for part in flat.split(slice_size)]
    values = torch.cat([value for value, _ in factorised]).reshape(matrix.shape[:-1])
    vectors = torch.cat([vector for _, vector in factorised]).reshape(matrix.shape)
    return values, vectors


@dataclass(frozen=True)
class Architecture:
    """A ReLU network from `input_dim` inputs to `num_classes` logits."""

    input_dim: int
    hidden_width: int
    num_hidden_layers: int
    num_classes: int

    @cached_property
    def layers(self) -> list[tuple[int, int]]:
        """`(fan_in, fan_out)` of each layer."""
        widths = [self.input_dim, *[self.hidden_width] * self.num_hidden_layers, self.num_classes]
        return list(zip(widths[:-1], widths[1:], strict=True))

    @cached_property
    def num_parameters(self) -> int:
        return sum(fan_in * fan_out + fan_out for fan_in, fan_out in self.layers)

    def unflatten(self, theta: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Each layer's weights `(..., fan_out, fan_in)` and biases `(..., fan_out)`, as views of
        parameters `(..., P)`. A layer's weights come before its biases, row by row."""
        layers, start = [], 0
        for fan_in, fan_out in self.layers:
            weight = theta[..., start : start + fan_in * fan_out]
            start += fan_in * fan_out
            bias = theta[..., start : start + fan_out]
            start += fan_out
            layers.append((weight.unflatten(-1, (fan_out, fan_in)), bias))
        return layers

    def logits(self, x: Tensor, theta: Tensor, masks: Sequence[Tensor] | None = None) -> Tensor:
        """Logits `(..., n, num_classes)` of inputs `(..., n, input_dim)` under parameters
        `(..., P)`, multiplying each hidden layer's units by `masks` for dropout."""
        h = x
        for index, (weight, bias) in enumerate(self.unflatten(theta)):
            h = h @ weight.transpose(-1, -2) + bias.unsqueeze(-2)
            if index < len(self.layers) - 1:
                h = torch.relu(h)
                if masks is not None:
                    h = h * masks[index]
        return h

    def jacobian(self, x: Tensor, theta: Tensor) -> tuple[Tensor, Tensor]:
        """Logits `(..., n, K)` and, in closed form, their Jacobian `(..., n, K, P)`."""
        layers = self.unflatten(theta)
        inputs, active, h = [], [], x
        for index, (weight, bias) in enumerate(layers):
            inputs.append(h)
            h = h @ weight.transpose(-1, -2) + bias.unsqueeze(-2)
            if index < len(layers) - 1:
                active.append((h > 0).to(h.dtype))
                h = torch.relu(h)
        # Sensitivity of each logit to each layer's pre-activations, from the last layer back.
        backprop = torch.eye(self.num_classes, dtype=h.dtype, device=h.device).expand(
            *h.shape[:-1], -1, -1
        )
        pieces: list[Tensor] = []
        for index in range(len(layers) - 1, -1, -1):
            by_weight = backprop.unsqueeze(-1) * inputs[index][..., None, None, :]
            pieces = [by_weight.flatten(-2), backprop, *pieces]
            if index > 0:
                weight = layers[index][0].unsqueeze(-3)
                backprop = (backprop @ weight) * active[index - 1].unsqueeze(-2)
        return h, torch.cat(pieces, dim=-1)

    @cached_property
    def is_weight(self) -> Tensor:
        """Which parameters are weights rather than biases, shape `(P,)`, bool."""
        pieces = []
        for fan_in, fan_out in self.layers:
            pieces += [torch.ones(fan_in * fan_out, dtype=torch.bool), torch.zeros(fan_out).bool()]
        return torch.cat(pieces)

    @cached_property
    def prior_std(self) -> Tensor:
        """Each parameter's standard deviation under `sample_prior`, `(P,)`. Biases are uniform,
        so a Gaussian with these matches only their first two moments."""
        pieces = []
        for fan_in, fan_out in self.layers:
            pieces.append(
                torch.full((fan_in * fan_out,), math.sqrt(2 / (fan_in + fan_out)), dtype=DTYPE)
            )
            pieces.append(torch.full((fan_out,), math.sqrt(1 / (3 * fan_in)), dtype=DTYPE))
        return torch.cat(pieces)

    def sample_prior(
        self,
        num: int,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Parameters of `num` networks, shape `(num, P)`: Xavier-normal weights, and biases
        uniform on `+-1 / sqrt(fan_in)`. Without a generator, from the global one of `device`."""
        pieces = []
        for fan_in, fan_out in self.layers:
            std = math.sqrt(2 / (fan_in + fan_out))
            shape = (num, fan_in * fan_out)
            pieces.append(std * torch.randn(shape, generator=generator, dtype=DTYPE, device=device))
            uniform = torch.rand(num, fan_out, generator=generator, dtype=DTYPE, device=device)
            pieces.append((2 * uniform - 1) / math.sqrt(fan_in))
        return torch.cat(pieces, dim=-1)

    def sample_gaussian_prior(self, num: int, generator: torch.Generator) -> Tensor:
        """As `sample_prior`, with Gaussian biases of the same variance. Shape `(num, P)`."""
        noise = torch.randn(num, self.num_parameters, generator=generator, dtype=DTYPE)
        return noise * self.prior_std
