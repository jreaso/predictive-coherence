import itertools
import math

import pytest
import torch

from predictive_coherence import CoherenceEstimator, estimate_coherence


def test_matches_definitions():
    mu_p, mu_r = torch.randn(3, 4), torch.randn(6, 3, 4)
    estimate = estimate_coherence(mu_p, mu_r)

    displacement = mu_r - mu_p
    pairs = itertools.permutations(range(6), 2)
    mmd_sq = torch.stack([(displacement[r] * displacement[s]).sum(-1) for r, s in pairs]).mean(0)
    v = torch.stack([torch.cov(mu_r[:, m].T).trace() for m in range(3)])

    assert torch.allclose(estimate.mmd_sq_per_point, mmd_sq)
    assert torch.allclose(estimate.v_per_point, v)
    assert torch.allclose(estimate.psi_per_point, mmd_sq / v)
    assert torch.allclose(estimate.mmd_sq, mmd_sq.sum())
    assert torch.allclose(estimate.v, v.sum())
    assert torch.allclose(estimate.psi, mmd_sq.sum() / v.sum())


def test_leave_one_out_matches_recomputation():
    mu_p, var_p, mu_r, var_r = (
        torch.randn(3, 4),
        torch.rand(3),
        torch.randn(6, 3, 4),
        torch.rand(6, 3),
    )
    estimate = estimate_coherence(mu_p, mu_r, var_p, var_r)
    for r in range(6):
        keep = [s for s in range(6) if s != r]
        subset = estimate_coherence(mu_p, mu_r[keep], var_p, var_r[keep])
        assert torch.allclose(estimate.mmd_sq_loo[r], subset.mmd_sq_per_point)
        assert torch.allclose(estimate.v_loo[r], subset.v_per_point)


@pytest.mark.parametrize("bias", [0.0, 0.3])
def test_unbiased_with_conservative_standard_errors(bias):
    # 2000 independent replications of 50 rounds, laid out along the evaluation-point dimension.
    scales = torch.linspace(0.1, 1.0, 10)
    delta = torch.zeros(10)
    delta[0] = bias
    mu_r = delta + scales * torch.randn(50, 2000, 10)
    estimate = estimate_coherence(torch.zeros(2000, 10), mu_r)

    assert math.isclose(estimate.mmd_sq_per_point.mean(), bias**2, abs_tol=5e-3)
    assert math.isclose(estimate.v_per_point.mean(), scales.square().sum(), rel_tol=0.01)
    standard_error = estimate.standard_error(lambda mmd_sq, v: mmd_sq)
    assert 1.0 <= standard_error.mean() / estimate.mmd_sq_per_point.std() <= 1.6


def test_removes_monte_carlo_noise():
    # As above, with each embedding observed through independent Monte Carlo noise.
    scales = torch.linspace(0.1, 1.0, 10)
    delta = torch.zeros(10)
    delta[0] = 0.3
    noise_p, noise_q = 0.1, 0.4
    mu_p = noise_p * torch.randn(2000, 10)
    mu_r = delta + scales * torch.randn(50, 2000, 10) + noise_q * torch.randn(50, 2000, 10)
    var_p, var_r = torch.full((2000,), 10 * noise_p**2), torch.full((50, 2000), 10 * noise_q**2)
    estimate = estimate_coherence(mu_p, mu_r, var_p, var_r)

    assert math.isclose(estimate.mmd_sq_per_point.mean(), 0.3**2, abs_tol=0.01)
    assert math.isclose(estimate.v_per_point.mean(), scales.square().sum(), rel_tol=0.02)


def test_estimator_pools_the_current_predictive_over_rounds():
    mu_p, var_p, mu_r, var_r = (
        torch.randn(10, 3, 4),
        torch.rand(10, 3),
        torch.randn(10, 3, 4),
        torch.rand(10, 3),
    )
    sampled, exact = CoherenceEstimator(), CoherenceEstimator()
    for r in range(10):
        sampled.update(mu_p[r], mu_r[r], var_p[r], var_r[r])
        exact.update(mu_p[0], mu_r[r])

    expected = estimate_coherence(mu_p.mean(0), mu_r, var_p.sum(0) / 10**2, var_r)
    for name in ("mmd_sq_per_point", "v_per_point", "mmd_sq_loo", "v_loo"):
        assert torch.allclose(getattr(sampled.estimate(), name), getattr(expected, name))
    assert torch.allclose(
        exact.estimate().mmd_sq_per_point, estimate_coherence(mu_p[0], mu_r).mmd_sq_per_point
    )
    assert sampled.num_rounds == 10


def test_standard_error_defaults_to_pooled_psi():
    estimate = estimate_coherence(torch.randn(3, 4), torch.randn(20, 3, 4))
    explicit = estimate.standard_error(lambda mmd_sq, v: mmd_sq.sum(-1) / v.sum(-1))
    assert torch.equal(estimate.standard_error(), explicit)
    assert estimate.standard_error().shape == ()


def test_requires_three_rounds():
    with pytest.raises(ValueError):
        estimate_coherence(torch.zeros(3, 4), torch.zeros(2, 3, 4))
