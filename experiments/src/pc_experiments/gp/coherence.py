"""Exact incoherence of Gaussian models whose updates are linear in the outputs."""

import torch
from torch import Tensor
from torch.distributions import Normal

from pc_experiments.gp.models import GaussianModel
from predictive_coherence import RFFKernel


def gaussian_coherence(
    model: GaussianModel, kernel: RFFKernel, x_design: Tensor, x_eval: Tensor
) -> tuple[Tensor, Tensor]:
    """`||Delta||^2` and `V`, each `(L,)`, at evaluation inputs `(L, d)` for a block at `(n, d)`.

    The post-update predictive is Gaussian, with a variance that does not depend on the block's
    outputs and a mean affine in them, so the expected future predictive is Gaussian too, with
    the post-update variance plus the variance of the post-update mean.
    """
    design_mean, design_cov = model.joint_predict(x_design)
    current = model.predict(x_eval)
    centre = model.update(x_design, design_mean).predict(x_eval)
    # The updated mean is affine in the outputs, so unit perturbations recover its gain exactly.
    gain = torch.stack(
        [
            model.update(x_design, design_mean + unit).predict(x_eval).loc - centre.loc
            for unit in torch.eye(len(x_design)).to(design_mean)
        ],
        dim=1,
    )
    mean_var = (gain @ design_cov @ gain.T).diagonal()

    average = kernel.embed(Normal(centre.loc, (centre.scale.square() + mean_var).sqrt()))
    c = (average - kernel.embed(current)).square().sum(-1)
    # The kernel is translation invariant, so `||mu_r||^2` does not depend on the updated mean.
    v = kernel.embed(Normal(torch.zeros_like(centre.loc), centre.scale)).square().sum(-1)
    return c, v - average.square().sum(-1)
