"""Rank statistics for reading results across arms, datasets and environments."""

import torch
from torch import Tensor


def ranks(values: Tensor) -> Tensor:
    """Ranks along the last dimension, from 1, with ties given their average rank."""
    order = values.argsort(-1)
    first = order.argsort(-1).double() + 1
    # Ties share their average rank.
    sorted_values = values.gather(-1, order)
    result = first.clone()
    for index in range(values.shape[-1]):
        value = sorted_values[..., index : index + 1]
        equal = values == value
        result = torch.where(
            equal, (first * equal).sum(-1, keepdim=True) / equal.sum(-1, keepdim=True), result
        )
    return result


def spearman(x: Tensor, y: Tensor) -> Tensor:
    """Spearman's rank correlation along the last dimension of `x` and `y`, shape `(...)`."""
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(-1, keepdim=True), ry - ry.mean(-1, keepdim=True)
    return (rx * ry).sum(-1) / (rx.norm(dim=-1) * ry.norm(dim=-1)).clamp_min(1e-300)


def kendall_w(values: Tensor) -> Tensor:
    """Kendall's coefficient of concordance among the rankings of `values` `(judges, items)`:
    one when every judge ranks the items alike, zero when the rankings are unrelated."""
    judges, items = values.shape
    sums = ranks(values).sum(0)
    spread = ((sums - sums.mean()) ** 2).sum()
    return 12 * spread / (judges**2 * (items**3 - items))
