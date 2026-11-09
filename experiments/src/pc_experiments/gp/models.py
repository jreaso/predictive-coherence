"""Gaussian process regression models under a fixed prior."""

from abc import abstractmethod
from typing import Self

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal

from pc_experiments.gp.prior import Prior
from predictive_coherence import PredictiveModel

# Added to the diagonal of `K_zz`, relative to its mean, before it is factorised.
_JITTER = 1e-8


class GaussianModel(PredictiveModel):
    """A model whose outputs at any inputs are jointly Gaussian."""

    @abstractmethod
    def joint_predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean `(m,)` and covariance `(m, m)` of the outputs at inputs `(m, d)`."""

    def predict(self, x: Tensor) -> Normal:
        mean, cov = self.joint_predict(x)
        return Normal(mean, cov.diagonal().sqrt())

    def sample(self, x: Tensor, num_samples: int) -> Tensor:
        return self.predict(x).sample((num_samples,))

    def joint_sample(self, x: Tensor) -> Tensor:
        return MultivariateNormal(*self.joint_predict(x)).sample()


class ExactGP(GaussianModel):
    """Exact Gaussian process regression on inputs `(n, d)` and outputs `(n,)`: coherent."""

    def __init__(self, prior: Prior, x: Tensor, y: Tensor) -> None:
        self.prior, self.x, self.y = prior, x, y
        self._factor = torch.linalg.cholesky(prior.covariance(x))
        self._weights = torch.cholesky_solve(y.unsqueeze(-1), self._factor).squeeze(-1)

    def update(self, x: Tensor, y: Tensor) -> Self:
        return type(self)(self.prior, torch.cat([self.x, x]), torch.cat([self.y, y]))

    def joint_predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        cross = self.prior.kernel(x, self.x)
        solved = torch.linalg.solve_triangular(self._factor, cross.T, upper=False)
        return cross @ self._weights, self.prior.covariance(x) - solved.T @ solved


class SparseGP(GaussianModel):
    """Sparse Gaussian process regression with its inducing inputs `z` `(M, d)` held fixed.

    `q(u)`, over the function at `z`, is the optimum of Titsias' bound, which is the exact
    posterior under the likelihood `y | u ~ N(K_xz K_zz^{-1} u, noise_std^2 I)`: an update adds the
    new points to two sufficient statistics. The predictive restores the variance `K - Q` that this
    likelihood leaves out, so outputs are drawn with it but absorbed without it, and the model is
    incoherent, the less so the better `z` summarises the function.

    The belief is over `v = L^{-1} u`, with `K_zz = L L^T`, held as its precision `(M, M)` and
    `shift`, the precision times its mean `(M,)`; the prior's when `None`.
    """

    def __init__(
        self,
        prior: Prior,
        z: Tensor,
        precision: Tensor | None = None,
        shift: Tensor | None = None,
    ) -> None:
        self.prior, self.z = prior, z
        self.precision = torch.eye(len(z)).to(z) if precision is None else precision
        self.shift = z.new_zeros(len(z)) if shift is None else shift
        k_zz = prior.kernel(z, z)
        jitter = _JITTER * k_zz.diagonal().mean() * torch.eye(len(z)).to(z)
        self._factor = torch.linalg.cholesky(k_zz + jitter)  # L

    @classmethod
    def fit(cls, prior: Prior, z: Tensor, x: Tensor, y: Tensor) -> Self:
        return cls(prior, z).update(x, y)

    def update(self, x: Tensor, y: Tensor) -> Self:
        basis = self._basis(x) / self.prior.noise_std
        return type(self)(
            self.prior,
            self.z,
            self.precision + basis.T @ basis,
            self.shift + basis.T @ y / self.prior.noise_std,
        )

    def joint_predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        basis = self._basis(x)
        # With `P = R R^T`, `basis P^{-1} basis^T = half half^T` and the mean of `v` is
        # `R^{-T} R^{-1} shift`.
        root = torch.linalg.cholesky(self.precision)
        half = torch.linalg.solve_triangular(root, basis.T, upper=False).T
        mean = half @ torch.linalg.solve_triangular(root, self.shift.unsqueeze(-1), upper=False)
        return mean.squeeze(-1), self.prior.covariance(x) - basis @ basis.T + half @ half.T

    def _basis(self, x: Tensor) -> Tensor:
        """`K_xz L^{-T}`, `(n, M)`: the function at `x` has mean `basis @ v`."""
        cross = self.prior.kernel(self.z, x)
        return torch.linalg.solve_triangular(self._factor, cross, upper=False).T


class ReselectedSparseGP(SparseGP):
    """A sparse Gaussian process that chooses up to `num_inducing` inducing inputs greedily from
    the data `(n, d)`, `(n,)`, and chooses them afresh and refits whenever data arrive.

    The choice depends on the inputs alone, so the post-update predictive is still affine in
    a block's outputs.
    """

    def __init__(self, prior: Prior, x: Tensor, y: Tensor, num_inducing: int) -> None:
        z = greedy_variance(prior, x, num_inducing)
        fitted = SparseGP.fit(prior, z, x, y)
        super().__init__(prior, z, fitted.precision, fitted.shift)
        self.x, self.y, self.num_inducing = x, y, num_inducing

    def update(self, x: Tensor, y: Tensor) -> Self:
        return type(self)(
            self.prior, torch.cat([self.x, x]), torch.cat([self.y, y]), self.num_inducing
        )


def greedy_variance(prior: Prior, x: Tensor, num_inducing: int, until: float = 0.0) -> Tensor:
    """Up to `num_inducing` inducing inputs from `x` `(n, d)`, by a pivoted Cholesky factorisation:
    each step takes the input with the most prior variance left unexplained.

    Stops early once the inputs taken explain the rest to within rounding, or once the variance
    they leave, `tr(K - Q)` summed over `x`, is at most `until`.
    """
    remaining = prior.kernel(x, x).diagonal().clone()
    floor = 1e-10 * remaining.max()
    columns: list[Tensor] = []
    chosen: list[int] = []
    for _ in range(num_inducing):
        if remaining.sum() <= until:
            break
        pivot = int(remaining.argmax())
        if remaining[pivot] <= floor:
            break
        column = prior.kernel(x, x[pivot : pivot + 1]).squeeze(-1)
        for earlier in columns:
            column = column - earlier * earlier[pivot]
        column = column / column[pivot].sqrt()
        columns.append(column)
        chosen.append(pivot)
        remaining = (remaining - column.square()).clamp_min(0)
    return x[chosen]


def to_noise_level(prior: Prior, x: Tensor) -> Tensor:
    """Inducing inputs from `x` `(n, d)`, greedily, until the prior variance they leave is on
    average at most the noise variance. That average is the trace term of the collapsed bound,
    which controls how far it falls short of the evidence (Burt et al., 2019)."""
    return greedy_variance(prior, x, len(x), until=len(x) * prior.noise_std**2)
