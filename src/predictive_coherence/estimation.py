"""Estimating the incoherence ratio from rounds of simulated updates."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


def _pooled_psi(mmd_sq: Tensor, v: Tensor) -> Tensor:
    """Incoherence ratio pooled over evaluation points (the last dimension), `sum ||Delta||^2 /
    sum V`."""
    return mmd_sq.sum(-1) / v.sum(-1)


@dataclass(frozen=True)
class CoherenceEstimate:
    """Estimated bias and spread of a model's updates at `L` evaluation points.

    Round `r` updates the model on a block drawn from its own predictive. With `mu_p` the mean
    embedding of the current predictive `P_Y` and `mu_r` that of the round's post-update
    predictive, the displacement is `delta_r = mu_r - mu_p`, and with `Q_Y` the expected future
    predictive, whose embedding is `E[mu_r]`,

        ||Delta||^2 = ||E[delta_r]||^2     = MMD^2(P_Y, Q_Y),
        V = E||delta_r - E[delta_r]||^2    the spread of the updates,
        psi = ||Delta||^2 / V              the incoherence ratio.

    A coherent model has `||Delta||^2 = 0`. By default evaluation points are pooled: `mmd_sq`,
    `v` and `psi` are the quantities under the direct-sum kernel over points, so `mmd_sq` and `v`
    are sums over points. Per-point values localise incoherence.

    Attributes:
        mmd_sq_per_point: unbiased estimate of `||Delta||^2` at each point, shape `(L,)`. Can be
            negative near zero.
        v_per_point: unbiased estimate of `V` at each point, shape `(L,)`.
        mmd_sq_loo, v_loo: the same leaving out each round in turn, shape `(R, L)`.
    """

    mmd_sq_per_point: Tensor
    v_per_point: Tensor
    mmd_sq_loo: Tensor
    v_loo: Tensor

    @property
    def mmd_sq(self) -> Tensor:
        """`||Delta||^2` pooled over evaluation points."""
        return self.mmd_sq_per_point.sum(-1)

    @property
    def v(self) -> Tensor:
        """`V` pooled over evaluation points."""
        return self.v_per_point.sum(-1)

    @property
    def psi(self) -> Tensor:
        """Incoherence ratio pooled over evaluation points, `mmd_sq / v`."""
        return _pooled_psi(self.mmd_sq_per_point, self.v_per_point)

    @property
    def psi_per_point(self) -> Tensor:
        """Incoherence ratio at each evaluation point, shape `(L,)`."""
        return self.mmd_sq_per_point / self.v_per_point

    def standard_error(self, statistic: Callable[[Tensor, Tensor], Tensor] = _pooled_psi) -> Tensor:
        """Leave-one-round-out jackknife standard error of
        `statistic(mmd_sq_per_point, v_per_point)`, by default the pooled `psi`. `statistic` takes
        the evaluation points on the last dimension, e.g. `lambda mmd_sq, v: mmd_sq.sum(-1)` for
        the pooled `||Delta||^2` or `lambda mmd_sq, v: mmd_sq / v` per point."""
        replicates = statistic(self.mmd_sq_loo, self.v_loo)
        num_rounds = replicates.shape[0]
        deviations = replicates - replicates.mean(0)
        return ((num_rounds - 1) / num_rounds * deviations.square().sum(0)).sqrt()


def estimate_coherence(
    mu_p: Tensor,
    mu_r: Tensor,
    var_p: Tensor | None = None,
    var_r: Tensor | None = None,
) -> CoherenceEstimate:
    """Estimate `||Delta||^2` and `V` from `R >= 3` independent rounds.

    Both estimates are unbiased U-statistics over rounds. Monte Carlo noise in the embeddings
    is removed using its estimated variance, which excludes each sample's pairing with itself.

    Args:
        mu_p: mean embedding of the current predictive, shape `(L, d)`. Either exact, or a
            Monte Carlo estimate independent of the rounds.
        mu_r: mean embeddings of each round's post-update predictive, shape `(R, L, d)`.
        var_p: Monte Carlo variance of `mu_p`, shape `(L,)`. `None` if exact.
        var_r: Monte Carlo variance of each embedding in `mu_r`, shape `(R, L)`. `None` if exact.
    """
    num_rounds = mu_r.shape[0]
    if num_rounds < 3:
        raise ValueError(f"need at least 3 rounds, got {num_rounds}")

    # The spread of the displacements is the spread of the post-update embeddings: every round
    # subtracts the same current embedding, and a variance does not see a shift.
    mean_sq, mean_sq_loo, v, v_loo = _moments(mu_r - mu_p)

    # ||mean(delta)||^2 overestimates ||Delta||^2 by the variance of mean(delta): s^2(delta) / R
    # across rounds, plus the Monte Carlo variance of mu_p, which every round shares.
    mmd_sq = mean_sq - v / num_rounds
    mmd_sq_loo = mean_sq_loo - v_loo / (num_rounds - 1)
    if var_p is not None:
        mmd_sq, mmd_sq_loo = mmd_sq - var_p, mmd_sq_loo - var_p

    # s^2(mu_r) overestimates V by the average Monte Carlo variance within rounds.
    if var_r is not None:
        total = var_r.sum(0)
        v = v - total / num_rounds
        v_loo = v_loo - (total - var_r) / (num_rounds - 1)

    return CoherenceEstimate(mmd_sq, v, mmd_sq_loo, v_loo)


class CoherenceEstimator:
    """Estimate `||Delta||^2` and `V` from rounds as they arrive, which can be shared with a
    `SequentialTest`.

    The current predictive's embeddings are pooled over rounds, so Monte Carlo estimates of it
    must be independent between rounds. Every post-update embedding is kept for the jackknife,
    so memory grows as `O(R L d)`.
    """

    def __init__(self) -> None:
        # Sums of the current predictive's embeddings and their variances; 0-d until updated.
        self._mu_p_sum = torch.zeros(())
        self._var_p_sum = torch.zeros(())
        self._mu_r: list[Tensor] = []
        self._var_r: list[Tensor] = []

    @property
    def num_rounds(self) -> int:
        return len(self._mu_r)

    def update(
        self,
        mu_p: Tensor,
        mu_r: Tensor,
        var_p: Tensor | None = None,
        var_r: Tensor | None = None,
    ) -> None:
        """Add one round: embeddings `(L, d)`, and Monte Carlo variances `(L,)` or `None` if exact."""
        self._mu_p_sum = self._mu_p_sum + mu_p
        if var_p is not None:
            self._var_p_sum = self._var_p_sum + var_p
        self._mu_r.append(mu_r)
        self._var_r.append(mu_r.new_zeros(mu_r.shape[:-1]) if var_r is None else var_r)

    def estimate(self) -> CoherenceEstimate:
        n = self.num_rounds
        # The pooled mean of n independent estimates has variance sum(var) / n^2.
        return estimate_coherence(
            self._mu_p_sum / n,
            torch.stack(self._mu_r),
            self._var_p_sum / n**2,
            torch.stack(self._var_r),
        )


def _moments(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Squared norm of the mean and trace of the sample covariance over the first dimension.

    Args:
        x: shape `(R, L, d)`.

    Returns:
        `||mean||^2` and `s^2`, shape `(L,)`, and the same leaving out each row, shape `(R, L)`.
        Computed from residuals about the mean to avoid cancellation.
    """
    n = x.shape[0]
    mean = x.mean(0)
    residual = x - mean
    sq_residual = residual.square().sum(-1)
    total_sq_residual = sq_residual.sum(0)

    mean_sq = mean.square().sum(-1)
    # The mean without row r is mean - residual_r / (n - 1).
    mean_sq_loo = (
        mean_sq - 2 * torch.linalg.vecdot(residual, mean) / (n - 1) + sq_residual / (n - 1) ** 2
    )
    var = total_sq_residual / (n - 1)
    var_loo = (total_sq_residual - sq_residual * n / (n - 1)) / (n - 2)
    return mean_sq, mean_sq_loo, var, var_loo
