"""Normalised kernels on outputs with explicit finite-dimensional feature maps."""

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.distributions import Categorical, Distribution, Normal


class Kernel(ABC):
    """Kernel `k(y, y') = <phi(y), phi(y')>` with `k(y, y) = 1`.

    Normalisation keeps every mean embedding in the unit ball, which bounds the payoffs of the
    sequential test.
    """

    dim: int

    @abstractmethod
    def features(self, y: Tensor) -> Tensor:
        """Feature map `phi`, from shape `(...)` to `(..., dim)`."""

    @abstractmethod
    def embed(self, dist: Distribution) -> Tensor:
        """Exact mean embedding `E[phi(Y)]`, shape `(*dist.batch_shape, dim)`."""

    def embed_samples(self, samples: Tensor) -> tuple[Tensor, Tensor]:
        """Monte Carlo mean embedding from `S >= 2` samples.

        Args:
            samples: shape `(S, ...)`.

        Returns:
            The sample mean of `phi`, shape `(..., dim)`, and an unbiased estimate of its
            variance `E||mean - E[phi(Y)]||^2`, shape `(...)`.
        """
        num_samples = samples.shape[0]
        if num_samples < 2:
            raise ValueError(f"need at least 2 samples, got {num_samples}")
        phi = self.features(samples)
        mean = phi.mean(0)
        variance = (phi.square().sum(-1).mean(0) - mean.square().sum(-1)) / (num_samples - 1)
        return mean, variance


class DiracKernel(Kernel):
    """Kernel `k(y, y') = 1[y = y']` on class labels `0, ..., num_classes - 1`."""

    def __init__(self, num_classes: int) -> None:
        self.dim = num_classes

    def features(self, y: Tensor) -> Tensor:
        return torch.nn.functional.one_hot(y.long(), self.dim).to(torch.get_default_dtype())

    def embed(self, dist: Distribution) -> Tensor:
        if not isinstance(dist, Categorical):
            raise TypeError(f"DiracKernel embeds Categorical, not {type(dist).__name__}")
        return dist.probs


class RFFKernel(Kernel):
    """Random Fourier feature approximation of the Gaussian kernel on scalar outputs.

    Approximates `exp(-(y - y')^2 / (2 lengthscale^2))` with
    `phi(y) = [cos(w y), sin(w y)] / sqrt(num_features)` and frequencies
    `w ~ N(0, 1 / lengthscale^2)` drawn once from `seed`.
    """

    def __init__(self, lengthscale: float = 1.0, num_features: int = 256, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.frequencies = torch.randn(num_features, generator=generator, dtype=torch.float64)
        self.frequencies /= lengthscale
        self.dim = 2 * num_features

    def features(self, y: Tensor) -> Tensor:
        return self._fourier(y, torch.ones((), dtype=y.dtype, device=y.device))

    def embed(self, dist: Distribution) -> Tensor:
        if not isinstance(dist, Normal):
            raise TypeError(f"RFFKernel embeds Normal, not {type(dist).__name__}")
        # E[exp(i w Y)] = exp(i w mu - (w sigma)^2 / 2) for Y ~ N(mu, sigma^2).
        w = self.frequencies.to(dist.scale)
        return self._fourier(dist.loc, torch.exp(-0.5 * (dist.scale.unsqueeze(-1) * w).square()))

    def _fourier(self, y: Tensor, damping: Tensor) -> Tensor:
        angle = y.unsqueeze(-1) * self.frequencies.to(y)
        scale = damping / math.sqrt(self.frequencies.numel())
        return torch.cat([scale * angle.cos(), scale * angle.sin()], dim=-1)
