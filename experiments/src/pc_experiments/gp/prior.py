"""Gaussian process priors with fixed kernels."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Normal


class Prior(ABC):
    """Zero-mean Gaussian process `f`, observed as `y = f(x) + noise_std * eps`."""

    noise_std: float

    @abstractmethod
    def kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Prior covariance of `f` between inputs `(..., n1, d)` and `(..., n2, d)`, `(..., n1, n2)`."""

    def covariance(self, x: Tensor) -> Tensor:
        """Prior covariance of the outputs at inputs `(..., n, d)`, `(..., n, n)`."""
        return self.kernel(x, x) + self.noise_std**2 * torch.eye(x.shape[-2]).to(x)

    def sample(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Outputs drawn jointly from the prior at inputs `(..., n, d)`, `(..., n)`."""
        # In the inputs' dtype, not the default one: a seed gives a different stream in each.
        noise = torch.randn(x.shape[:-1], generator=generator, dtype=x.dtype)
        return (torch.linalg.cholesky(self.covariance(x)) @ noise.unsqueeze(-1)).squeeze(-1)

    def sequential_predictive(self, x: Tensor, y: Tensor) -> Normal:
        """Exact predictive of each `y[..., i]`, `i >= 1`, given the points before it, from inputs
        `(..., n, d)` and outputs `(..., n)`; `batch_shape` `(..., n - 1)`.

        With `L` the Cholesky factor of the outputs' covariance and `z = L^{-1} y`, point `i` has
        standard deviation `L_ii` and mean `y_i - L_ii z_i`.
        """
        factor = torch.linalg.cholesky(self.covariance(x))
        z = torch.linalg.solve_triangular(factor, y.unsqueeze(-1), upper=False).squeeze(-1)
        scale = factor.diagonal(dim1=-2, dim2=-1)
        return Normal((y - scale * z)[..., 1:], scale[..., 1:])


@dataclass(frozen=True)
class GaussianKernelPrior(Prior):
    """`k(x, x') = outputscale^2 exp(-||x - x'||^2 / (2 lengthscale^2))`."""

    lengthscale: float
    outputscale: float
    noise_std: float

    def kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.outputscale**2 * torch.exp(-0.5 * _sq_dist(x1, x2) / self.lengthscale**2)


@dataclass(frozen=True)
class ARDGaussianPrior(Prior):
    """`k(x, x') = outputscale^2 exp(-sum_j (x_j - x'_j)^2 / (2 lengthscales_j^2))`."""

    lengthscales: tuple[float, ...]
    outputscale: float
    noise_std: float

    def kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        scale = torch.tensor(self.lengthscales).to(x1)
        return self.outputscale**2 * torch.exp(-0.5 * squared_distances(x1 / scale, x2 / scale))


def squared_distances(x1: Tensor, x2: Tensor) -> Tensor:
    """Squared distances between inputs `(..., n1, d)` and `(..., n2, d)`, `(..., n1, n2)`, from
    inner products: memory does not grow with `d`, but nearly equal inputs lose precision."""
    norms1, norms2 = x1.square().sum(-1, keepdim=True), x2.square().sum(-1).unsqueeze(-2)
    return (norms1 - 2 * x1 @ x2.mT + norms2).clamp_min(0)


def _sq_dist(x1: Tensor, x2: Tensor) -> Tensor:
    """As `squared_distances`, from differences: exact for nearly equal inputs, at memory that
    grows with `d`."""
    return (x1.unsqueeze(-2) - x2.unsqueeze(-3)).square().sum(-1)


def sample_inputs(
    shape: tuple[int, ...],
    input_range: tuple[float, float],
    generator: torch.Generator | None = None,
) -> Tensor:
    """Scalar inputs uniform on `input_range`, `(*shape, 1)`, in float64 whatever the default
    dtype, so that a seed always gives the same inputs."""
    low, high = input_range
    uniform = torch.rand(*shape, 1, generator=generator, dtype=torch.float64)
    return low + (high - low) * uniform
