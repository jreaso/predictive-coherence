"""When the sequential test rejects, observed and as the estimates of ||Delta||^2 and V predict it.

Once the witness has aligned with the displacement, which takes about `1 / Psi_l` rounds, log
wealth at evaluation point `l` grows at about `||Delta_l|| / (4 sqrt(kappa))`, with the stake at
its cap.
"""

import math

import torch
from torch import Tensor


def first_rejection(log_wealth: Tensor, alpha: float) -> Tensor:
    """The round, from 1, at which log wealth `(..., R)` first reaches `log(1 / alpha)`, or NaN.
    Shape `(...)`."""
    rejected = log_wealth >= -math.log(alpha)
    first = rejected.double().argmax(-1) + 1.0
    return torch.where(rejected.any(-1), first, torch.nan)


def saturated_rate(mmd_sq_per_point: Tensor, kernel_bound: float = 1.0) -> Tensor:
    """`||Delta_l|| / (4 sqrt(kappa))` from estimates of `||Delta_l||^2` `(..., L)`, NaN where
    the estimate is not positive."""
    positive = mmd_sq_per_point > 0
    rate = mmd_sq_per_point.clamp_min(0).sqrt() / (4 * math.sqrt(kernel_bound))
    return torch.where(positive, rate, torch.nan)


def predicted_round(
    mmd_sq_per_point: Tensor, v_per_point: Tensor, alpha: float, kernel_bound: float = 1.0
) -> Tensor:
    """The round the account above predicts a rejection at, from estimates at each point
    `(..., L)`: the earliest at which any point's wealth reaches `L / alpha`, so that the average
    reaches `1 / alpha`. Shape `(...)`, infinite where no estimate of `||Delta_l||^2` is
    positive."""
    num_points = mmd_sq_per_point.shape[-1]
    rate = saturated_rate(mmd_sq_per_point, kernel_bound)
    alignment = v_per_point / mmd_sq_per_point.clamp_min(1e-300)
    rounds = alignment + math.log(num_points / alpha) / rate
    return rounds.nan_to_num(math.inf).amin(-1)
