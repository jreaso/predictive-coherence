"""The Gaussian process priors and the models under them."""

import dataclasses
import math
from itertools import islice

import pytest
import torch

from pc_experiments.gp import (
    ARDGaussianPrior,
    ExactGP,
    GaussianKernelPrior,
    ReselectedSparseGP,
    SparseGP,
    fit_ard_prior,
    gaussian_coherence,
    greedy_variance,
    sample_inputs,
    to_noise_level,
)
from predictive_coherence import CoherenceEstimator, RFFKernel, SequentialTest, iter_rounds

PRIOR = GaussianKernelPrior(lengthscale=0.5, outputscale=1.2, noise_std=0.3)
NARROW = GaussianKernelPrior(lengthscale=0.2, outputscale=0.7, noise_std=0.1)


def problem(num_context=24, num_design=6, num_eval=16):
    """A context with its outputs, a block and evaluation inputs."""
    x, x_design, x_eval = (
        sample_inputs((n,), (-2.0, 2.0)) for n in (num_context, num_design, num_eval)
    )
    return x, PRIOR.sample(x), x_design, x_eval


# ----- priors -------------------------------------------------------------------------------------


@pytest.mark.parametrize("prior", [PRIOR, NARROW])
def test_prior_draws_have_the_prior_covariance(prior):
    x = sample_inputs((6,), (-2.0, 2.0))
    draws = prior.sample(x.expand(40_000, 6, 1))
    assert torch.allclose(draws.mean(0), torch.zeros(6), atol=0.03)
    assert torch.allclose(torch.cov(draws.T), prior.covariance(x), atol=0.06)


def test_kernels_match_their_closed_forms():
    x = sample_inputs((6,), (-2.0, 2.0))
    assert torch.allclose(PRIOR.covariance(x).diagonal(), torch.full((6,), 1.2**2 + 0.3**2))

    pair = torch.tensor([[0.2], [0.5]])  # a distance of 0.3 apart
    gaussian = 1.2**2 * math.exp(-0.5 * (0.3 / 0.5) ** 2)
    assert PRIOR.kernel(pair, pair)[0, 1] == pytest.approx(gaussian)


def test_inputs_do_not_depend_on_the_default_dtype():
    seeded = [torch.Generator().manual_seed(3) for _ in range(2)]
    in_double = sample_inputs((5,), (-2.0, 2.0), seeded[0])
    torch.set_default_dtype(torch.float32)  # the fixture restores it
    in_single = sample_inputs((5,), (-2.0, 2.0), seeded[1])
    assert in_single.dtype == torch.float64 and torch.equal(in_single, in_double)
    assert PRIOR.sample(in_single, seeded[1]).dtype == torch.float64


@pytest.mark.parametrize("prior", [PRIOR, NARROW])
def test_sequential_predictive_conditions_on_each_prefix(prior):
    x = sample_inputs((2, 7), (-2.0, 2.0))
    y = prior.sample(x)
    sequential = prior.sequential_predictive(x, y)
    assert sequential.batch_shape == (2, 6)
    for task in range(2):
        for i in range(1, 7):
            exact = ExactGP(prior, x[task, :i], y[task, :i]).predict(x[task, i : i + 1])
            assert torch.allclose(sequential.loc[task, i - 1], exact.loc[0])
            assert torch.allclose(sequential.scale[task, i - 1], exact.scale[0])


# ----- the exact model ----------------------------------------------------------------------------


def test_update_conditions_exactly():
    x, y, x_new, x_test = problem(num_context=8, num_design=3, num_eval=5)
    y_new = PRIOR.sample(x_new)
    model = ExactGP(PRIOR, x, y)
    updated = model.update(x_new, y_new).predict(x_test)
    # Conditioning the joint predictive at the new and the test inputs on the new outputs.
    mean, cov = model.joint_predict(torch.cat([x_new, x_test]))
    gain = torch.linalg.solve(cov[:3, :3], cov[:3, 3:]).T
    assert torch.allclose(updated.loc, mean[3:] + gain @ (y_new - mean[:3]))
    assert torch.allclose(updated.scale.square(), (cov[3:, 3:] - gain @ cov[:3, 3:]).diagonal())
    assert len(model.x) == 8


def test_exact_gp_is_coherent():
    x, y, x_design, x_eval = problem(num_context=10, num_design=4)
    model = ExactGP(PRIOR, x, y)
    estimator, test = CoherenceEstimator(), SequentialTest()
    for round_ in islice(iter_rounds(model, RFFKernel(lengthscale=0.5), x_design, x_eval), 300):
        estimator.update(*round_)
        test.update(round_.mu_p, round_.mu_r)
    estimate = estimator.estimate()
    assert estimate.mmd_sq < 3 * estimate.standard_error(lambda mmd_sq, v: mmd_sq.sum(-1))
    assert estimate.psi < 0.02
    assert not test.rejects(alpha=0.01)


# ----- the sparse model ---------------------------------------------------------------------------


def test_sparse_gp_with_every_input_inducing_is_exact():
    # Inputs a lengthscale apart keep `K_zz` well conditioned, so its jitter is immaterial.
    x, x_eval = torch.linspace(-2, 2, 9).unsqueeze(-1), sample_inputs((16,), (-2.0, 2.0))
    y = PRIOR.sample(x)
    sparse, exact = SparseGP.fit(PRIOR, x, x, y), ExactGP(PRIOR, x, y)
    for ours, theirs in zip(sparse.joint_predict(x_eval), exact.joint_predict(x_eval), strict=True):
        assert torch.allclose(ours, theirs, atol=1e-5)


def test_sparse_update_refits_the_belief_and_leaves_the_model_unchanged():
    x, y, x_new, x_eval = problem()
    y_new = PRIOR.sample(x_new)
    model = SparseGP.fit(PRIOR, greedy_variance(PRIOR, x, 6), x, y)
    before = model.predict(x_eval)
    updated = model.update(x_new, y_new).predict(x_eval)
    refit = SparseGP.fit(PRIOR, model.z, torch.cat([x, x_new]), torch.cat([y, y_new]))
    assert torch.allclose(updated.loc, refit.predict(x_eval).loc)
    assert torch.allclose(updated.scale, refit.predict(x_eval).scale)
    assert torch.equal(model.predict(x_eval).loc, before.loc)
    assert not torch.allclose(updated.loc, before.loc)


def test_greedy_selection_is_nested_and_spreads_out():
    x, _, _, _ = problem(num_context=40)
    greedy = greedy_variance(PRIOR, x, 6)
    assert len(greedy.unique(dim=0)) == 6
    assert torch.equal(greedy[:3], greedy_variance(PRIOR, x, 3))
    # Each rung leaves less of the prior variance at the context unexplained than the one before.
    left = [unexplained(PRIOR, x, greedy_variance(PRIOR, x, m)) for m in (2, 4, 8, 16)]
    assert all(more > less for more, less in zip(left, left[1:], strict=False))


def unexplained(prior, x, z):
    """The prior variance at inputs `x` that inducing inputs `z` leave unexplained, summed."""
    cross = prior.kernel(x, z)
    explained = cross @ torch.linalg.solve(prior.kernel(z, z) + 1e-8 * torch.eye(len(z)), cross.T)
    return (prior.kernel(x, x) - explained).diagonal().sum()


def test_inducing_to_the_noise_level_takes_the_fewest_that_explain_the_function_within_it():
    x, _, _, _ = problem(num_context=60)
    z = to_noise_level(PRIOR, x)
    assert 1 < len(z) < len(x)
    assert torch.equal(z, greedy_variance(PRIOR, x, len(z)))  # the greedy order, stopped early
    noise = len(x) * PRIOR.noise_std**2
    assert unexplained(PRIOR, x, z) <= noise < unexplained(PRIOR, x, z[:-1])
    # The noisier the observations, the fewer inputs it takes to explain the function within them.
    noisier = dataclasses.replace(PRIOR, noise_std=2 * PRIOR.noise_std)
    assert len(to_noise_level(noisier, x)) < len(z)


def test_reselected_sparse_gp_with_every_input_inducing_is_exact_after_an_update():
    x, y, x_new, x_eval = problem()
    y_new = PRIOR.sample(x_new)
    model = ReselectedSparseGP(PRIOR, x, y, len(x) + len(x_new))
    updated = model.update(x_new, y_new)
    # It takes every input that adds anything: the kernel matrix's numerical rank, at most all,
    # and the rest are determined by those to within rounding.
    assert len(updated.z) <= len(x) + len(x_new)
    assert all(any(torch.equal(row, point) for point in updated.x) for row in updated.z)
    exact = ExactGP(PRIOR, torch.cat([x, x_new]), torch.cat([y, y_new])).predict(x_eval)
    # Thirty inputs on eight lengthscales make `K_zz` ill conditioned, so its jitter shows at the
    # fifth decimal of the mean; that is the whole difference from the exact posterior.
    assert torch.allclose(updated.predict(x_eval).loc, exact.loc, atol=1e-4)
    assert torch.allclose(updated.predict(x_eval).scale, exact.scale, atol=1e-5)
    # Fewer inducing inputs than data: the update chooses afresh, and may drop a context input.
    coarse = ReselectedSparseGP(PRIOR, x, y, 4).update(x_new, y_new)
    assert len(coarse.z) == 4 and coarse.x.shape[0] == len(x) + len(x_new)


def test_greedy_selection_stops_at_the_distinct_inputs():
    x, _, _, _ = problem(num_context=6)
    assert len(greedy_variance(PRIOR, torch.cat([x, x]), 12)) == 6
    x = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    assert len(greedy_variance(PRIOR, x, 2).unique(dim=0)) == 2
    assert len(greedy_variance(PRIOR, x, 3)) == 2


# ----- incoherence in closed form -----------------------------------------------------------------


def test_enough_inducing_inputs_make_the_model_coherent():
    x, y, x_design, x_eval = problem(num_context=40)
    kernel = RFFKernel(lengthscale=0.5)

    def psi(model):
        c, v = gaussian_coherence(model, kernel, x_design, x_eval)
        return c.sum() / v.sum()

    # Not monotone task by task among coarse models, whose updates barely move: both
    # `||Delta||^2` and `V` are then small. But every coarse model is far less coherent than a fine one.
    *coarse, fine = [
        psi(SparseGP.fit(PRIOR, greedy_variance(PRIOR, x, m), x, y)) for m in (2, 4, 8, 20)
    ]
    assert min(coarse) > 100 * fine
    assert psi(ExactGP(PRIOR, x, y)).abs() < 1e-12


@pytest.mark.slow
def test_estimates_match_closed_form_for_a_sparse_gp():
    x, y, x_design, x_eval = problem()
    kernel = RFFKernel(lengthscale=0.5)
    model = SparseGP.fit(PRIOR, greedy_variance(PRIOR, x, 4), x, y)
    c, v = gaussian_coherence(model, kernel, x_design, x_eval)

    # With few rounds the estimate of `||Delta||^2` is skewed and its standard error moves with it, so a
    # low estimate looks more certain than it is. Rounds are cheap here, so use plenty.
    estimator = CoherenceEstimator()
    for round_ in islice(iter_rounds(model, kernel, x_design, x_eval), 4000):
        estimator.update(*round_)
    estimate = estimator.estimate()
    assert abs(estimate.mmd_sq - c.sum()) < 4 * estimate.standard_error(
        lambda mmd_sq, v: mmd_sq.sum(-1)
    )
    assert abs(estimate.v - v.sum()) < 4 * estimate.standard_error(lambda c, v: v.sum(-1))


# ----- priors over several inputs, and fitting them -----------------------------------------------


def test_ard_kernel_matches_its_closed_form_and_the_isotropic_one():
    x1, x2 = torch.randn(5, 3), torch.randn(4, 3)
    prior = ARDGaussianPrior((0.5, 2.0, 1.0), outputscale=1.3, noise_std=0.1)
    scaled = (x1.unsqueeze(1) - x2) / torch.tensor([0.5, 2.0, 1.0])
    torch.testing.assert_close(
        prior.kernel(x1, x2), 1.3**2 * (-0.5 * scaled.square().sum(-1)).exp()
    )
    same = ARDGaussianPrior((0.7, 0.7, 0.7), 1.3, 0.1).kernel(x1, x2)
    torch.testing.assert_close(same, GaussianKernelPrior(0.7, 1.3, 0.1).kernel(x1, x2))


def test_fitting_an_ard_prior_finds_the_noise_and_the_input_that_does_not_matter():
    x = torch.randn(150, 2)
    y = torch.sin(2 * x[:, 0]) + 0.1 * torch.randn(150)
    prior = fit_ard_prior(x, y)
    assert prior.noise_std == pytest.approx(0.1, rel=0.3)
    assert prior.lengthscales[1] > 5 * prior.lengthscales[0]
    # A search from the optimum stays there, and a short one from elsewhere moves towards it.
    again = fit_ard_prior(x, y, start=prior, max_iter=5)
    assert again.lengthscales[0] == pytest.approx(prior.lengthscales[0], rel=1e-3)
    far = ARDGaussianPrior((3.0, 3.0), 1.0, 0.5)
    assert fit_ard_prior(x, y, start=far, max_iter=5).noise_std < 0.5


def test_fitting_a_prior_survives_outputs_with_no_noise_to_find():
    """The search works on the logs of the hyperparameters, so on noiseless outputs it drives the
    noise towards zero until it underflows. Without a floor on the noise variance the covariance
    is singular there and the factorisation fails, taking the fit with it."""
    x = torch.randn(60, 2)
    exact = x[:, 0] - 0.5 * x[:, 1]  # no noise at all, and a duplicated input for good measure
    x, exact = torch.cat([x, x[:1]]), torch.cat([exact, exact[:1]])
    prior = fit_ard_prior(x, exact)
    assert prior.noise_std >= 0  # it got there at all
    assert math.isfinite(prior.outputscale)
