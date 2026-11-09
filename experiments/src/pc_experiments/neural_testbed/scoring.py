"""Exact acquisition objectives, incoherence and losses, by enumerating a batch's `K^k`
labellings `z`, each with the model's probability `p(z)` and post-update predictive `q_z`.

With `p` the current predictive and `q_bar = sum_z p(z) q_z` the expected future predictive, the
objectives are, averaged over target inputs,

    EPIG, entropy form   H[p] - E_z H[q_z]
    EPIG, KL form        E_z KL(q_z || p)
    EMI                  H[q_bar] - E_z H[q_z] = E_z KL(q_z || q_bar)

which agree when the model is coherent, `q_bar = p`. Leading dimensions broadcast throughout.
"""

import torch
from torch import Tensor

# A pooled spread of the updates below this is rounding: the model did not move, and has no ratio.
UNMOVED = 1e-20
TINY = 1e-300


def enumerate_labellings(size: int, num_classes: int, device: torch.device | str = "cpu") -> Tensor:
    """Every labelling of `size` inputs, shape `(num_classes^size, size)`, int64.

    Row `i` holds the digits of `i` in base `num_classes`, the first input's label leading.
    """
    powers = num_classes ** torch.arange(size - 1, -1, -1, device=device)
    return (torch.arange(num_classes**size, device=device).unsqueeze(-1) // powers) % num_classes


def labelling_log_probs(log_probs: Tensor, labellings: Tensor) -> Tensor:
    """Log probability `(..., Z)` of each labelling `(Z, k)` under independent labels with log
    class probabilities `(..., k, K)`."""
    inputs = torch.arange(labellings.shape[-1], device=labellings.device)
    return log_probs[..., inputs, labellings].sum(-1)


def mixture_log_joint(log_weights: Tensor, atom_log_probs: Tensor, labellings: Tensor) -> Tensor:
    """`log sum_a w_a prod_j p_a(z_j | x_j)`, `(..., Z)`, for atoms with log weights `(..., A)` and
    log class probabilities `(..., A, k, K)`."""
    per_atom = labelling_log_probs(atom_log_probs, labellings)
    return torch.logsumexp(log_weights.unsqueeze(-1) + per_atom, dim=-2)


def entropy(log_probs: Tensor) -> Tensor:
    """Entropy in nats from log probabilities, over the last dimension."""
    return -(log_probs.exp() * log_probs).sum(-1)


def kl_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    """`KL(p || q)` in nats from log probabilities, over the last dimension."""
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def score_candidates(log_p: Tensor, log_q: Tensor, log_joint: Tensor) -> dict[str, Tensor]:
    """Objectives and incoherence of each candidate, each `(...)`, from the current predictive at
    the targets `(..., M, K)`, the post-update ones `(..., Z, M, K)` and the joint `(..., Z)`.

    `epig`, `epig_kl`, `emi`: the objectives above.
    `c`: `||Delta||^2 = sum_m ||q_bar - p||^2`, the squared MMD under the Dirac kernel.
    `v`: `sum_m E_z ||q_z - q_bar||^2`. `psi`: `c / v`, NaN where the updates never move.
    """
    weights = log_joint.exp()
    p, q = log_p.exp(), log_q.exp()
    q_bar = torch.einsum("...z,...zmk->...mk", weights, q)
    log_q_bar = q_bar.clamp_min(TINY).log()

    expected_entropy = (weights * entropy(log_q).mean(-1)).sum(-1)
    current_entropy = entropy(log_p).mean(-1)
    cross_entropy = -(q_bar * log_p).sum(-1).mean(-1)

    bias = q_bar - p
    centred = q - q_bar.unsqueeze(-3)
    c = bias.square().sum((-1, -2))
    v = (weights * centred.square().sum((-1, -2))).sum(-1)
    return {
        "epig": current_entropy - expected_entropy,
        "epig_kl": cross_entropy - expected_entropy,
        "emi": entropy(log_q_bar).mean(-1) - expected_entropy,
        "c": c,
        "v": v,
        "psi": torch.where(v > UNMOVED, c / v, torch.nan),
    }


def coherence_of_random_updates(
    log_p: Tensor, log_q: Tensor, log_q_again: Tensor, log_joint: Tensor
) -> dict[str, Tensor]:
    """Incoherence of an update that is random given its labels, from two independent draws of
    it, `log_q` and `log_q_again` `(..., Z, M, K)`, each `(...)` as in `score_candidates`.

    From one draw, `||q_bar - p||^2` is biased upward by the variance of `q_bar` between draws,
    and the spread downward by as much. The product of two independent draws' displacements is
    unbiased:

    `c`: `<q_bar - p, q_bar' - p>`, negative at times when the displacement is small.
    `v`: `E_z (||q_z - p||^2 + ||q'_z - p||^2) / 2 - c`. `psi`: `c / v`.
    """
    weights = log_joint.exp()
    p = log_p.exp().unsqueeze(-3)
    q, q_again = log_q.exp(), log_q_again.exp()
    bias = torch.einsum("...z,...zmk->...mk", weights, q - p)
    bias_again = torch.einsum("...z,...zmk->...mk", weights, q_again - p)
    c = (bias * bias_again).sum((-1, -2))
    second = (q - p).square().sum((-1, -2)) + (q_again - p).square().sum((-1, -2))
    v = (weights * second).sum(-1) / 2 - c
    return {"c": c, "v": v, "psi": torch.where(v > UNMOVED, c / v, torch.nan)}


def expected_loss(log_truth: Tensor, log_q: Tensor, log_joint_truth: Tensor) -> Tensor:
    """`sum_z p*(z) mean_m KL(p*(. | x_m) || q_z(. | x_m))`, `(...)`, from the truth at the
    scored inputs `(..., M, K)`, the post-update predictives `(..., Z, M, K)` and each
    labelling's true log probability `(..., Z)`."""
    per_labelling = kl_divergence(log_truth.unsqueeze(-3), log_q).mean(-1)
    return (log_joint_truth.exp() * per_labelling).sum(-1)


def joint_kl(log_joint_truth: Tensor, log_joint: Tensor) -> Tensor:
    """`KL(p* || p)` between the true and the model's distribution over a batch's labellings."""
    weights = log_joint_truth.exp()
    # A labelling the truth rules out contributes nothing, whatever the model gives it.
    terms = torch.where(weights > 0, weights * (log_joint_truth - log_joint), 0.0)
    return terms.sum(-1)
