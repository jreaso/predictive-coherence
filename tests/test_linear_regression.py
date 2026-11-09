"""End-to-end checks on Bayesian linear regression, where coherence is known in closed form."""

from dataclasses import dataclass, replace
from itertools import islice

import pytest
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal

from predictive_coherence import (
    CoherenceEstimator,
    PredictiveModel,
    RFFKernel,
    SequentialTest,
    iter_rounds,
)


@dataclass(frozen=True)
class LinearRegression(PredictiveModel):
    """Linear regression `y = x @ w + noise` with a Gaussian belief `w ~ N(mean, cov)`.

    Updates condition the belief exactly, but predictions scale its covariance by `uncertainty`.
    The model is coherent only when `uncertainty = 1`; `uncertainty = 0` plugs in the mean.
    """

    mean: Tensor
    cov: Tensor
    noise_std: float
    uncertainty: float = 1.0

    def output_cov(self, x: Tensor) -> Tensor:
        return self.uncertainty * x @ self.cov @ x.T + self.noise_std**2 * torch.eye(len(x))

    def gain(self, x: Tensor) -> Tensor:
        innovation = x @ self.cov @ x.T + self.noise_std**2 * torch.eye(len(x))
        return torch.linalg.solve(innovation, x @ self.cov).T

    def predict(self, x: Tensor) -> Normal:
        variance = self.uncertainty * torch.einsum("ni,ij,nj->n", x, self.cov, x)
        return Normal(x @ self.mean, (variance + self.noise_std**2).sqrt())

    def sample(self, x: Tensor, num_samples: int) -> Tensor:
        return self.predict(x).sample((num_samples,))

    def joint_sample(self, x: Tensor) -> Tensor:
        return MultivariateNormal(x @ self.mean, self.output_cov(x)).sample()

    def update(self, x: Tensor, y: Tensor) -> "LinearRegression":
        gain = self.gain(x)
        mean = self.mean + gain @ (y - x @ self.mean)
        return replace(self, mean=mean, cov=self.cov - gain @ x @ self.cov)


def exact_coherence(model, kernel, x_design, x_eval):
    """Pooled `||Delta||^2` and `V`, using that only the post-update mean is random, with a
    Gaussian law."""
    gain = model.gain(x_design)
    shift_var = torch.einsum(
        "mi,ij,mj->m", x_eval, gain @ model.output_cov(x_design) @ gain.T, x_eval
    )
    p = model.predict(x_eval)
    q_scale = model.update(x_design, torch.zeros(len(x_design))).predict(x_eval).scale
    average_q = kernel.embed(Normal(p.loc, (q_scale.square() + shift_var).sqrt()))
    mmd_sq = (average_q - kernel.embed(p)).square().sum()
    # The kernel is translation invariant, so ||mu_r||^2 does not depend on the updated mean.
    v = kernel.embed(Normal(0.0, q_scale)).square().sum() - average_q.square().sum()
    return mmd_sq, v


@pytest.fixture
def problem():
    """Few, small design inputs, so plugging in the mean badly understates the update's spread."""
    x_design, x_eval = 0.1 * torch.randn(3, 2), torch.randn(16, 2)
    return RFFKernel(lengthscale=0.2, num_features=64), x_design, x_eval


def model(uncertainty: float) -> LinearRegression:
    return LinearRegression(torch.zeros(2), torch.eye(2), noise_std=0.1, uncertainty=uncertainty)


def run(model, problem, num_rounds, num_samples=None):
    """Share every round between an estimator and a sequential test, without stopping early."""
    kernel, x_design, x_eval = problem
    estimator, test = CoherenceEstimator(), SequentialTest()
    for round_ in islice(iter_rounds(model, kernel, x_design, x_eval, num_samples), num_rounds):
        estimator.update(*round_)
        test.update(round_.mu_p, round_.mu_r)
    return estimator.estimate(), test


def rounds_to_reject(model, problem, num_samples=None, alpha=0.01, max_rounds=1000):
    """Test one update at a time, stopping at the first rejection."""
    kernel, x_design, x_eval = problem
    test = SequentialTest()
    for round_ in islice(iter_rounds(model, kernel, x_design, x_eval, num_samples), max_rounds):
        test.update(round_.mu_p, round_.mu_r)
        if test.rejects(alpha):
            return len(test.log_wealth)
    return None


@pytest.mark.parametrize("uncertainty", [1.0, 0.25, 0.0])
@pytest.mark.parametrize("num_samples", [None, 64])
def test_estimates_match_closed_form(problem, uncertainty, num_samples):
    kernel, x_design, x_eval = problem
    mmd_sq, v = exact_coherence(model(uncertainty), kernel, x_design, x_eval)
    estimate, _ = run(model(uncertainty), problem, 300, num_samples)
    mmd_sq_se = estimate.standard_error(lambda mmd_sq, v: mmd_sq.sum(-1))
    v_se = estimate.standard_error(lambda mmd_sq, v: v.sum(-1))
    assert abs(estimate.mmd_sq - mmd_sq) < 4 * mmd_sq_se
    assert abs(estimate.v - v) < 4 * v_se


def test_exact_bayes_is_coherent(problem):
    kernel, x_design, x_eval = problem
    mmd_sq, _ = exact_coherence(model(1.0), kernel, x_design, x_eval)
    assert abs(mmd_sq) < 1e-12
    assert rounds_to_reject(model(1.0), problem) is None


@pytest.mark.parametrize("num_samples", [None, 64])
def test_stops_sooner_for_stronger_incoherence(problem, num_samples):
    plug_in = rounds_to_reject(model(0.0), problem, num_samples)
    weak = rounds_to_reject(model(0.25), problem, num_samples)
    assert plug_in is not None and plug_in <= 100
    assert weak is not None and plug_in < weak


def test_rounds_are_shared_beyond_rejection(problem):
    estimate, test = run(model(0.0), problem, 300, num_samples=64)
    rejection = test.rejection_round(alpha=0.01)
    assert rejection is not None and rejection <= 100
    assert len(test.log_wealth) == 300
    assert estimate.psi > 10 * estimate.standard_error()


def test_default_joint_sample_conditions_sequentially():
    class Sequential(LinearRegression):
        joint_sample = PredictiveModel.joint_sample

    coherent = Sequential(torch.zeros(2), torch.eye(2), noise_std=0.5)
    x = torch.randn(3, 2)
    draws = torch.stack([coherent.joint_sample(x) for _ in range(4000)])
    assert torch.allclose(draws.mean(0), torch.zeros(3), atol=0.1)
    assert torch.allclose(torch.cov(draws.T), coherent.output_cov(x), rtol=0.1, atol=0.1)


def test_predict_is_optional():
    class SamplesOnly(PredictiveModel):
        def sample(self, x, num_samples):
            return torch.zeros(num_samples, len(x))

        def update(self, x, y):
            return self

    with pytest.raises(NotImplementedError):
        SamplesOnly().predict(torch.zeros(1, 1))
