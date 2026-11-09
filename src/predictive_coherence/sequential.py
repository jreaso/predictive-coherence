"""Anytime-valid sequential test of coherence by betting."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# Online Newton step on the log wealth, with stakes capped at 1/2.
_ONS_SCALE = 2 / (2 - math.log(3))
_MAX_STAKE = 0.5


@dataclass
class _Wealth:
    """Betting state at each of `L` evaluation points."""

    displacement_sum: Tensor  # (L, d)
    stake: Tensor  # (L,)
    curvature: Tensor  # (L,), one plus the sum of squared ONS gradients
    log_wealth: Tensor  # (L,)


class SequentialTest:
    """Test that a model's updates are coherent at `L` evaluation points, one update at a time.

    Each round observes `delta_r = mu_r - mu_p`, an unbiased estimate of the displacement that
    one update induces in the mean embedding at each evaluation point. Under the null its
    conditional mean given earlier rounds is zero. At each point:

    - The witness `u_r` is the unit vector along the mean of earlier displacements, or zero
      while that mean has norm at most `noise_floor`.
    - The bet is `b_r = <u_r, delta_r> / (2 sqrt(kappa))`, which lies in `[-1, 1]`.
    - The stake `lambda_r` in `[0, 1/2]` is set by online Newton steps on the log wealth.
    - The wealth is multiplied by `1 + lambda_r b_r`.

    The test's wealth is the average of the per-point wealths, a nonnegative martingale under
    the null. By Ville's inequality it ever reaches `1 / alpha` with probability at most
    `alpha`, so the test rejects at the first round it does. Incoherence confined to one point
    costs at most `log L` in log wealth compared with testing that point alone.

    The test never stops by itself: stop at the rejection round to save updates, or keep going
    to share the rounds with a `CoherenceEstimator`. Every round needs a fresh update, and a fresh
    estimate of `mu_p` unless it is exact.

    Args:
        kernel_bound: `kappa` such that `||phi(y)||^2 <= kappa`, which is 1 for the kernels in
            this package.
        noise_floor: no bet is placed at a point while the mean displacement has norm at most
            this.

    Attributes:
        log_wealth: log wealth after each round.
    """

    def __init__(self, kernel_bound: float = 1.0, noise_floor: float = 1e-8) -> None:
        self.kernel_bound = kernel_bound
        self.noise_floor = noise_floor
        self.log_wealth: list[float] = []
        self._wealth: _Wealth | None = None

    def update(self, mu_p: Tensor, mu_r: Tensor) -> float:
        """Bet on one round's post-update embeddings `mu_r`, `(L, d)`, or several rounds in order,
        `(R, L, d)`, against the current `mu_p`, and return the log wealth."""
        displacements = mu_r - mu_p
        max_norm = 2 * math.sqrt(self.kernel_bound)
        if displacements.norm(dim=-1).max() > max_norm * (1 + 1e-6):
            raise ValueError(f"displacement norm exceeds 2 sqrt(kernel_bound) = {max_norm}")
        for displacement in displacements.reshape(-1, *displacements.shape[-2:]):
            self._round(displacement)
        return self.log_wealth[-1]

    def rejection_round(self, alpha: float) -> int | None:
        """The first round, from 1, at which wealth reached `1 / alpha`, or `None`."""
        threshold = -math.log(alpha)
        return next((r for r, w in enumerate(self.log_wealth, 1) if w >= threshold), None)

    def rejects(self, alpha: float) -> bool:
        """Whether wealth has reached `1 / alpha` after any round so far."""
        return self.rejection_round(alpha) is not None

    def _round(self, displacement: Tensor) -> None:
        if self._wealth is None:
            zeros = displacement.new_zeros(displacement.shape[:-1])
            self._wealth = _Wealth(torch.zeros_like(displacement), zeros, zeros + 1, zeros.clone())
        wealth = self._wealth

        mean = wealth.displacement_sum / max(len(self.log_wealth), 1)
        norm = mean.norm(dim=-1, keepdim=True)
        witness = torch.where(norm > self.noise_floor, mean / norm, 0.0)
        bet = torch.linalg.vecdot(witness, displacement) / (2 * math.sqrt(self.kernel_bound))

        wealth.log_wealth += torch.log1p(wealth.stake * bet)
        average = wealth.log_wealth.logsumexp(0) - math.log(len(bet))
        self.log_wealth.append(average.item())

        gradient = bet / (1 + wealth.stake * bet)
        wealth.curvature += gradient.square()
        wealth.stake += _ONS_SCALE * gradient / wealth.curvature
        wealth.stake.clamp_(0, _MAX_STAKE)
        wealth.displacement_sum += displacement
