"""Fitting a prior's hyperparameters by maximising the marginal likelihood."""

import math

import torch
from torch import Tensor

from pc_experiments.gp.prior import ARDGaussianPrior, squared_distances

# Added to the noise variance during the search, as a share of the prior variance, so that a line
# search step that sends the noise to zero leaves the covariance factorisable and is backed away
# from. A share, since the output scale ranges over orders of magnitude during a search.
NOISE_FLOOR = 1e-10


def fit_ard_prior(
    x: Tensor, y: Tensor, start: ARDGaussianPrior | None = None, max_iter: int = 100
) -> ARDGaussianPrior:
    """The ARD Gaussian prior maximising the evidence of inputs `(n, d)` and outputs `(n,)`, by
    L-BFGS on the logs of the hyperparameters. By default the search starts at lengthscales of
    `sqrt(d)`, unit output scale and noise 0.3."""
    num_inputs = x.shape[-1]
    if start is None:
        start = ARDGaussianPrior((math.sqrt(num_inputs),) * num_inputs, 1.0, 0.3)
    initial = torch.tensor([*start.lengthscales, start.outputscale, start.noise_std]).to(x).log()
    log_values = initial.clone().requires_grad_()
    optimiser = torch.optim.LBFGS([log_values], max_iter=max_iter, line_search_fn="strong_wolfe")
    identity = torch.eye(len(y)).to(y)

    def negative_log_evidence() -> Tensor:
        optimiser.zero_grad()
        values = log_values.exp()
        scaled = x / values[:-2]
        covariance = values[-2] ** 2 * torch.exp(-0.5 * squared_distances(scaled, scaled))
        noise_var = values[-1] ** 2 + NOISE_FLOOR * covariance.diagonal().mean()
        factor = torch.linalg.cholesky(covariance + noise_var * identity)
        whitened = torch.linalg.solve_triangular(factor, y.unsqueeze(-1), upper=False)
        loss = 0.5 * whitened.square().sum() + factor.diagonal().log().sum()
        loss.backward()
        return loss

    optimiser.step(negative_log_evidence)
    *lengthscales, outputscale, noise_std = log_values.detach().exp().tolist()
    return ARDGaussianPrior(tuple(lengthscales), outputscale, noise_std)
