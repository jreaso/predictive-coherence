"""Closed-form Gaussian quantities for acquisition in Bayesian linear regression.

Weights have a Gaussian belief and outputs are `features @ weights` plus Gaussian noise, so every
predictive is Gaussian. Mean-field VI recovers the exact posterior mean, so every predictive shares
it and they differ only in covariances. The predictive after an update has a mean that is affine
in the observed outcomes, so its average over outcomes drawn from the current predictive is
Gaussian with the post-update variance plus the variance of the mean shift.
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import torch
from torch import Tensor


def quadratic(features: Tensor, cov: Tensor) -> Tensor:
    """`diag(features @ cov @ features.T)`, shape `(n,)`."""
    return ((features @ cov) * features).sum(-1)


@dataclass(frozen=True)
class Posterior:
    """Exact Gaussian posterior over the weights."""

    precision: Tensor
    cov: Tensor
    mean: Tensor
    noise_var: float

    @classmethod
    def fit(
        cls, features: Tensor, labels: Tensor, prior_var: float, noise_var: float
    ) -> "Posterior":
        eye = torch.eye(features.shape[1], dtype=features.dtype)
        precision = eye / prior_var + features.T @ features / noise_var
        cov = torch.linalg.inv(precision)
        return cls(precision, cov, cov @ features.T @ labels / noise_var, noise_var)

    def update(self, feature: Tensor, label: Tensor) -> "Posterior":
        """Condition on one observation."""
        shared = self.cov @ feature
        denominator = self.noise_var + feature @ shared
        return Posterior(
            self.precision + torch.outer(feature, feature) / self.noise_var,
            self.cov - torch.outer(shared, shared) / denominator,
            self.mean + shared * (label - feature @ self.mean) / denominator,
            self.noise_var,
        )

    @property
    def mean_field_var(self) -> Tensor:
        """Variances of the mean-field approximation, the reciprocal diagonal of the precision."""
        return self.precision.diagonal().reciprocal()


class Moments(NamedTuple):
    """Predictive variances at each target under an anticipated update.

    Attributes:
        current: variance of the current predictive, shape `(M,)` or shaped like `updated`.
        updated: variance of the predictive after the update, `(N, M)` for `N` single-candidate
            updates or `(M,)` for one block update.
        shift: variance of the updated mean over outcomes drawn from the current predictive,
            shaped like `updated`.
    """

    current: Tensor
    updated: Tensor
    shift: Tensor

    def epig(self) -> Tensor:
        """Expected predictive information gain `H[p] - E H[q_z]`, averaged over targets."""
        return 0.5 * torch.log(self.current / self.updated).mean(-1)

    def emi(self) -> Tensor:
        """Expected mutual information `H[q_bar] - E H[q_z]`, averaged over targets."""
        return 0.5 * torch.log1p(self.shift / self.updated).mean(-1)


def conditioning(
    cross: Tensor, candidate_var: Tensor, target_var: Tensor, noise_var: float
) -> Moments:
    """Moments when a Gaussian belief is conditioned on each candidate's outcome.

    Args:
        cross: covariance of noiseless outputs at candidates and targets, shape `(N, M)`.
        candidate_var: variance of noiseless outputs at candidates, shape `(N,)`.
        target_var: variance of noiseless outputs at targets, shape `(M,)`.
    """
    current = target_var + noise_var
    shift = cross.square() / (candidate_var + noise_var)[:, None]
    return Moments(current, current - shift, shift)


class View:
    """Exact and mean-field quantities of one posterior at every candidate and target.

    Candidates have shape `(N, d)` and targets `(M, d)`. Quantities are computed on first use.
    """

    def __init__(self, posterior: Posterior, candidates: Tensor, targets: Tensor) -> None:
        self.posterior = posterior
        self.candidates = candidates
        self.targets = targets

    @cached_property
    def cross(self) -> Tensor:
        return self.candidates @ (self.posterior.cov @ self.targets.T)

    @cached_property
    def candidate_var(self) -> Tensor:
        return quadratic(self.candidates, self.posterior.cov)

    @cached_property
    def target_var(self) -> Tensor:
        return quadratic(self.targets, self.posterior.cov)

    @cached_property
    def mf_candidate_var(self) -> Tensor:
        return self.candidates.square() @ self.posterior.mean_field_var

    @cached_property
    def mf_target_var(self) -> Tensor:
        return self.targets.square() @ self.posterior.mean_field_var

    @cached_property
    def mf_updated_target_var(self) -> Tensor:
        """Mean-field target variances after refitting on each candidate, shape `(N, M)`."""
        noise_var = self.posterior.noise_var
        precision = self.posterior.precision.diagonal() + self.candidates.square() / noise_var
        return precision.reciprocal() @ self.targets.square().T

    @cached_property
    def exact_information(self) -> Tensor:
        """Expected information gain of each candidate under the exact posterior, shape `(N,)`."""
        noise_var = self.posterior.noise_var
        return conditioning(self.cross, self.candidate_var, self.target_var, noise_var).epig()


def mean_field_refit(view: View) -> Moments:
    """Moments when the mean-field model refits on each single candidate.

    Outcomes are drawn from the mean-field predictive, the mean moves by the exact gain, and the
    refit reports mean-field variances.
    """
    noise_var = view.posterior.noise_var
    gain = view.cross / (view.candidate_var + noise_var)[:, None]
    shift = gain.square() * (view.mf_candidate_var + noise_var)[:, None]
    return Moments(view.mf_target_var + noise_var, view.mf_updated_target_var + noise_var, shift)


def mean_field_block_refit(
    posterior: Posterior, cov: Tensor, block: Tensor, targets: Tensor
) -> Moments:
    """Moments when the mean-field model refits on all of `block` at once.

    The current predictive has weight covariance `cov`, and outcomes are drawn from it.
    """
    noise_var = posterior.noise_var
    gram = block.T @ block / noise_var
    exact_updated = torch.linalg.inv(posterior.precision + gram)
    # The mean moves by exact_updated @ block^T (z - block mean) / noise_var.
    shift = quadratic(targets, exact_updated @ (gram @ cov @ gram + gram) @ exact_updated)
    updated = targets.square() @ (posterior.precision.diagonal() + gram.diagonal()).reciprocal()
    return Moments(quadratic(targets, cov) + noise_var, updated + noise_var, shift)


def coherent_refit_information(view: View, reference: Tensor, cov: Tensor) -> Tensor:
    """Expected information gain of a predictive that refits on the candidate and re-averages.

    The acquirer holds the predictive with weight covariance `cov`, which averages the mean-field
    refit over outcomes at `reference`. Observing a candidate refits that model, and the average
    over the block carries on, taken over outcomes drawn from `cov` and conditioned on what the
    candidate returned. This is one step of the map `coherent_mean_field_cov` solves, evaluated
    after the candidate. With no reference points it is the mean-field model's own EPIG.

    Returns:
        The score of each candidate, shape `(N,)`.
    """
    posterior, candidates, targets = view.posterior, view.candidates, view.targets
    noise_var = posterior.noise_var
    gram = reference.T @ reference / noise_var
    block_precision = posterior.precision + gram
    block_cov = torch.linalg.inv(block_precision)
    spread = block_cov @ ((gram @ cov) @ gram + gram) @ block_cov
    gain = (block_cov @ gram) @ cov

    aligned = candidates @ block_cov
    weight = (aligned @ targets.T) / (noise_var + (aligned * candidates).sum(-1))[:, None]
    correction = (targets @ gain @ candidates.T).T - weight * quadratic(candidates, gain)[:, None]
    mean_var = (
        quadratic(targets, spread)
        - 2 * weight * (candidates @ spread @ targets.T)
        + weight.square() * quadratic(candidates, spread)[:, None]
        - correction.square() / (quadratic(candidates, cov) + noise_var)[:, None]
    )

    precision_diag = block_precision.diagonal() + candidates.square() / noise_var
    updated = precision_diag.reciprocal() @ targets.square().T + noise_var + mean_var
    return 0.5 * torch.log((quadratic(targets, cov) + noise_var) / updated).mean(-1)


# Terms of the series below are negligible once the transition matrix has decayed this far.
SERIES_TOLERANCE = 1e-9
SERIES_STEPS = 64


def coherent_mean_field_cov(precision: Tensor, reference: Tensor, noise_var: float) -> Tensor:
    """Covariance of the Gaussian predictive under which the mean-field refit is coherent.

    Coherence is over outcomes at `reference`, drawn from the predictive itself. Solves the Stein
    equation `V = D + S B S + G V G^T`, where `B = reference^T reference / noise_var`, `S` is the
    exact covariance after observing `reference`, `G = S B`, and `D` is the mean-field covariance
    after observing `reference`. With no reference points it is the mean-field covariance.

    The solution is the series `sum_n G^n (D + S B S) (G^n)^T`, which converges because every
    eigenvalue of `G` is below one. Squaring `G` doubles the number of terms summed per step.
    """
    gram = reference.T @ reference / noise_var
    updated_precision = precision + gram
    updated_cov = torch.linalg.inv(updated_precision)
    floor = torch.diag(updated_precision.diagonal().reciprocal())

    cov = floor + updated_cov @ gram @ updated_cov
    power = updated_cov @ gram
    for _ in range(SERIES_STEPS):
        if power.abs().max() < SERIES_TOLERANCE:
            return (cov + cov.T) / 2
        cov = cov + power @ cov @ power.T
        power = power @ power
    raise RuntimeError(f"the series did not converge in {SERIES_STEPS} steps")


def incoherence(moments: Moments, lengthscale: float) -> tuple[Tensor, Tensor]:
    """`||Delta||^2` and `V` under the Gaussian kernel with `lengthscale`, summed over targets.

    Every predictive at a target shares its mean, and the expected future predictive is Gaussian
    with variance `updated + shift`. For independent `Y ~ N(m, a)` and `Y' ~ N(m, b)`,
    `E k(Y, Y') = lengthscale / sqrt(lengthscale^2 + a + b)`.

    Returns:
        `||Delta||^2` and `V`, each shaped like `moments.updated` without its last dimension.
    """

    def overlap(a: Tensor, b: Tensor) -> Tensor:
        return lengthscale / torch.sqrt(lengthscale**2 + a + b)

    current, updated = moments.current, moments.updated
    average = updated + moments.shift
    c = overlap(average, average) - 2 * overlap(average, current) + overlap(current, current)
    v = overlap(updated, updated) - overlap(average, average)
    return c.sum(-1), v.sum(-1)


def gaussian_kl(var_p: Tensor, var_q: Tensor) -> Tensor:
    """`KL(N(0, var_p) || N(0, var_q))`: the expected excess log loss of `q` when `p` is right."""
    return 0.5 * (torch.log(var_q / var_p) + var_p / var_q - 1)


def expected_nll(mean: Tensor, var: Tensor, truth: Tensor, noise_var: float) -> Tensor:
    """Expected negative log likelihood of a noisy observation of `truth` under `N(mean, var)`."""
    return 0.5 * torch.log(2 * math.pi * var) + ((mean - truth).square() + noise_var) / (2 * var)
