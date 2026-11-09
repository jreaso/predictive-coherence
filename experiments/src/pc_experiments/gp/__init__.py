"""Gaussian process regression: priors, exact and sparse models, and fitting a prior."""

from pc_experiments.gp.coherence import gaussian_coherence
from pc_experiments.gp.evidence import fit_ard_prior
from pc_experiments.gp.models import (
    ExactGP,
    GaussianModel,
    ReselectedSparseGP,
    SparseGP,
    greedy_variance,
    to_noise_level,
)
from pc_experiments.gp.prior import (
    ARDGaussianPrior,
    GaussianKernelPrior,
    sample_inputs,
)

__all__ = [
    "ARDGaussianPrior",
    "ExactGP",
    "GaussianKernelPrior",
    "GaussianModel",
    "ReselectedSparseGP",
    "SparseGP",
    "fit_ard_prior",
    "gaussian_coherence",
    "greedy_variance",
    "to_noise_level",
    "sample_inputs",
]
