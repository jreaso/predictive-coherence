from dataclasses import dataclass, replace
from itertools import islice

import pytest
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal

from pc_experiments.blr import Acquirer, Setting, run_environment, sample_environment
from pc_experiments.blr.gaussian import (
    Moments,
    Posterior,
    View,
    coherent_mean_field_cov,
    conditioning,
    incoherence,
    mean_field_block_refit,
    mean_field_refit,
    quadratic,
)
from predictive_coherence import CoherenceEstimator, Kernel, PredictiveModel, RFFKernel, iter_rounds

TINY = Setting(
    num_features=6,
    num_init=4,
    num_pool=12,
    num_targets=5,
    num_steps=6,
    lengthscale=1.0,
)
EXACT = Acquirer("exact")
ALPHAS = (0.0, 1.0)
SHARES = (0.0, 0.5, 1.0)
ACQUIRERS = (
    Acquirer("random"),
    EXACT,
    *(Acquirer("alpha", alpha) for alpha in ALPHAS),
    *(Acquirer("subsample", share) for share in SHARES),
)


def one_candidate(moments: Moments, index: int) -> Moments:
    """The moments of candidate `index` alone."""
    return Moments(moments.current, moments.updated[index], moments.shift[index])


@pytest.fixture
def problem():
    env = sample_environment(replace(TINY, num_pool=40), seed=3)
    posterior = Posterior.fit(env.init_features, env.init_labels, TINY.prior_var, TINY.noise_var)
    return env, posterior, View(posterior, env.pool_features, env.target_features)


def test_mean_field_refit_matches_direct_updates(problem):
    env, posterior, view = problem
    targets, noise_var = env.target_features, TINY.noise_var
    mean_field_var = posterior.mean_field_var
    refit = mean_field_refit(view)
    assert torch.allclose(refit.current, targets.square() @ mean_field_var + noise_var)

    for i in range(3):
        feature = env.pool_features[i]
        precision = posterior.precision.diagonal() + feature.square() / noise_var
        updated = targets.square() @ precision.reciprocal() + noise_var
        assert torch.allclose(refit.updated[i], updated)

        # The shift is the variance over outcomes of the updated exact posterior mean.
        outcome_var = feature.square() @ mean_field_var + noise_var
        outcome = Normal(feature @ posterior.mean, outcome_var.sqrt())
        means = torch.stack(
            [targets @ posterior.update(feature, z).mean for z in outcome.sample((20_000,))]
        )
        assert torch.allclose(means.var(0), refit.shift[i], rtol=0.05)


def test_block_of_one_matches_a_single_candidate(problem):
    env, posterior, view = problem
    cov = torch.diag(posterior.mean_field_var)
    block = mean_field_block_refit(posterior, cov, env.pool_features[2:3], env.target_features)
    single = one_candidate(mean_field_refit(view), 2)
    for name in ("current", "updated", "shift"):
        assert torch.allclose(getattr(block, name), getattr(single, name))


def test_alpha_interpolates_between_epig_and_emi(problem):
    env, posterior, view = problem
    cov = torch.diag(posterior.mean_field_var)
    refit = mean_field_refit(view)
    pool = env.pool_features
    assert torch.allclose(Acquirer("alpha", 0.0).scores(view, cov, pool), refit.epig())
    assert torch.allclose(Acquirer("alpha", 1.0).scores(view, cov, pool), refit.emi())

    mixed = Acquirer("alpha", 0.25).anticipated(view, cov, env.pool_features)
    average = refit.updated + refit.shift
    assert torch.allclose(mixed.current, 0.75 * refit.current + 0.25 * average)
    assert torch.allclose(mixed.updated, refit.updated)


def test_alpha_removes_incoherence(problem):
    env, posterior, view = problem
    cov = torch.diag(posterior.mean_field_var)

    def stats(alpha: float) -> tuple[Tensor, Tensor]:
        moments = Acquirer("alpha", alpha).anticipated(view, cov, env.pool_features)
        return incoherence(moments, TINY.psi_lengthscale)

    incoherent, variance = stats(0.0)
    assert torch.all(incoherent > 1e-6 * variance)
    previous = incoherent
    for alpha in (0.25, 0.5, 0.75):
        mixed, _ = stats(alpha)
        assert torch.all(mixed < previous)
        # The planning predictive is (1 - alpha) of the way from q_bar to p, and ||Delta||^2 is
        # close to quadratic in that distance.
        assert torch.allclose(mixed / incoherent, torch.tensor((1 - alpha) ** 2), rtol=0.25)
        previous = mixed
    coherent, variance = stats(1.0)
    assert torch.all(coherent.abs() < 1e-12 * variance)


def test_subsample_covariances_at_the_endpoints(problem):
    env, posterior, _ = problem
    pool, targets = env.pool_features, env.target_features
    mean_field = Acquirer("subsample", 0.0).covariance(posterior, pool)
    cov = Acquirer("subsample", 1.0).covariance(posterior, pool)
    assert mean_field is not None and cov is not None
    assert torch.allclose(mean_field, torch.diag(posterior.mean_field_var))

    # Over the whole pool the predictive is the one the mean-field refit cannot move on average.
    block = mean_field_block_refit(posterior, cov, pool, targets)
    assert torch.allclose(block.updated + block.shift, block.current)
    incoherent, _ = incoherence(mean_field_block_refit(posterior, mean_field, pool, targets), 1.0)
    assert incoherent > 0


@pytest.mark.parametrize("fraction", [0.0, 0.25, 1.0])
def test_conditioning_makes_epig_and_emi_agree(problem, fraction):
    """Conditioning is coherent with respect to the predictive it starts from, so the two
    objectives coincide however that predictive was built."""
    env, posterior, view = problem
    acquirer = Acquirer("subsample", fraction)
    cov = acquirer.covariance(posterior, env.pool_features)
    assert cov is not None
    cross = view.candidates @ (cov @ view.targets.T)
    candidate_var, target_var = quadratic(view.candidates, cov), quadratic(view.targets, cov)
    moments = conditioning(cross, candidate_var, target_var, view.posterior.noise_var)
    assert torch.allclose(
        moments.updated + moments.shift, moments.current.expand_as(moments.updated)
    )
    assert torch.allclose(moments.epig(), moments.emi())


def test_averaging_over_nothing_is_the_models_own_epig(problem):
    """With an empty block the coherentised acquirer is the mean-field model scoring with EPIG."""
    env, posterior, view = problem
    acquirer = Acquirer("subsample", 0.0)
    cov = acquirer.covariance(posterior, env.pool_features)
    assert cov is not None
    assert torch.allclose(cov, torch.diag(posterior.mean_field_var))
    scores = acquirer.scores(view, cov, env.pool_features)
    assert torch.allclose(scores, mean_field_refit(view).epig())


def test_coherent_refit_information_matches_an_explicit_construction(problem):
    """The closed form against the block posterior and outcome covariance written out."""
    env, posterior, view = problem
    targets, noise_var = env.target_features, TINY.noise_var
    acquirer = Acquirer("subsample", 0.5)
    reference = acquirer.reference(env.pool_features)
    cov = acquirer.covariance(posterior, env.pool_features)
    assert cov is not None
    scores = acquirer.scores(view, cov, env.pool_features)

    gram = reference.T @ reference / noise_var
    block_precision = posterior.precision + gram
    current = quadratic(targets, cov) + noise_var
    for index in (0, 5, 9):
        candidate = env.pool_features[index]
        precision = block_precision + torch.outer(candidate, candidate) / noise_var
        block_cov = torch.linalg.inv(precision)
        # Outcomes at the block, drawn from the predictive held and conditioned on the candidate.
        outcomes = reference @ cov @ reference.T + noise_var * torch.eye(len(reference))
        joint = reference @ cov @ candidate
        conditional = outcomes - torch.outer(joint, joint) / (
            candidate @ cov @ candidate + noise_var
        )
        moved = block_cov @ reference.T @ conditional @ reference @ block_cov / noise_var**2
        updated = (
            targets.square() @ precision.diagonal().reciprocal()
            + noise_var
            + quadratic(targets, moved)
        )
        assert torch.allclose(scores[index], 0.5 * torch.log(current / updated).mean())


@pytest.mark.parametrize("num_reference", [0, 2, 40])
def test_coherent_covariance_solves_its_stein_equation(problem, num_reference):
    env, posterior, _ = problem
    reference = env.pool_features[:num_reference]
    cov = coherent_mean_field_cov(posterior.precision, reference, TINY.noise_var)

    gram = reference.T @ reference / TINY.noise_var
    updated_precision = posterior.precision + gram
    updated_cov = torch.linalg.inv(updated_precision)
    transition = updated_cov @ gram
    floor = torch.diag(updated_precision.diagonal().reciprocal())
    assert torch.allclose(cov, floor + transition @ updated_cov + transition @ cov @ transition.T)


def test_exact_acquisition_is_coherent(problem):
    env, posterior, view = problem
    moments = EXACT.anticipated(view, posterior.cov, env.pool_features)
    assert torch.allclose(
        moments.updated + moments.shift, moments.current.expand_as(moments.updated)
    )
    assert torch.allclose(moments.epig(), moments.emi())
    assert torch.allclose(EXACT.scores(view, posterior.cov, env.pool_features), moments.epig())
    c, v = incoherence(moments, TINY.psi_lengthscale)
    assert torch.all(c.abs() < 1e-12 * v)


def embedded_incoherence(kernel: Kernel, moments: Moments) -> tuple[Tensor, Tensor]:
    """`||Delta||^2` and `V` from explicit kernel embeddings, which are centred because the means
    agree."""

    def embed(var: Tensor) -> Tensor:
        return kernel.embed(Normal(torch.zeros_like(var), var.sqrt()))

    average = embed(moments.updated + moments.shift)
    c = (average - embed(moments.current)).square().sum(-1)
    v = embed(moments.updated).square().sum(-1) - average.square().sum(-1)
    return c.sum(-1), v.sum(-1)


def test_incoherence_matches_a_kernel_embedding(problem):
    _, _, view = problem
    moments = one_candidate(mean_field_refit(view), 0)
    c, v = incoherence(moments, lengthscale=1.0)
    approximate = embedded_incoherence(RFFKernel(lengthscale=1.0, num_features=100_000), moments)
    assert torch.allclose(c, approximate[0], rtol=0.05)
    assert torch.allclose(v, approximate[1], rtol=0.05)


@dataclass(frozen=True)
class MeanFieldModel(PredictiveModel):
    """Bayesian linear regression predicting with mean-field variances and the exact mean."""

    posterior: Posterior

    def cov(self) -> Tensor:
        return torch.diag(self.posterior.mean_field_var)

    def predict(self, x: Tensor) -> Normal:
        var = x.square() @ self.posterior.mean_field_var + self.posterior.noise_var
        return Normal(x @ self.posterior.mean, var.sqrt())

    def sample(self, x: Tensor, num_samples: int) -> Tensor:
        return self.predict(x).sample((num_samples,))

    def joint_sample(self, x: Tensor) -> Tensor:
        cov = x @ self.cov() @ x.T + self.posterior.noise_var * torch.eye(len(x))
        return MultivariateNormal(x @ self.posterior.mean, cov).sample()

    def update(self, x: Tensor, y: Tensor) -> "MeanFieldModel":
        return replace(self, posterior=self.posterior.update(x[0], y[0]))


def test_incoherence_matches_the_library_estimator(problem):
    env, posterior, view = problem
    kernel = RFFKernel(lengthscale=1.0, num_features=128)
    c, v = embedded_incoherence(kernel, one_candidate(mean_field_refit(view), 0))

    estimator = CoherenceEstimator()
    rounds = iter_rounds(
        MeanFieldModel(posterior), kernel, env.pool_features[:1], env.target_features
    )
    for round_ in islice(rounds, 4000):
        estimator.update(*round_)
    estimate = estimator.estimate()
    assert abs(estimate.mmd_sq - c) < 4 * estimate.standard_error(lambda mmd_sq, v: mmd_sq.sum(-1))
    assert abs(estimate.v - v) < 4 * estimate.standard_error(lambda c, v: v.sum(-1))
    assert c > 10 * estimate.standard_error(lambda mmd_sq, v: mmd_sq.sum(-1))


def test_run_environment_gates():
    results = run_environment(TINY, ACQUIRERS, seed=0)
    names = [acquirer.name for acquirer in ACQUIRERS]

    def get(key: str, name: str) -> Tensor:
        return results[key][names.index(name)]

    # The exact acquirer plans with the exact predictive and is coherent; the random one plans
    # with no predictive and picks in the pool's order.
    assert torch.all(get("excess_nll", "exact").abs() < 1e-12)
    assert torch.all(get("c", "exact").abs() < 1e-10 * get("v", "exact"))
    assert get("excess_nll", "random").isnan().all()
    env = sample_environment(TINY, seed=0)
    assert torch.allclose(get("pick", "random"), env.pool_inputs[env.pool_order[: TINY.num_steps]])

    # Every other acquirer starts from the mean-field predictive, exactly so at both endpoints.
    mean_field = get("excess_nll", "alpha0")[0]
    assert mean_field > 0
    for name in [f"alpha{alpha:g}" for alpha in ALPHAS] + ["subsample0"]:
        assert torch.isclose(get("excess_nll", name)[0], mean_field)
    assert torch.all(get("c", "subsample0") > 0)
    assert torch.all(get("c", "alpha0") > 0)
    assert torch.all(get("c", "alpha1").abs() < 1e-10 * get("v", "alpha1"))

    # Acquisition is without replacement, and incoherence is recorded at every checkpoint the
    # horizon reaches.
    for name in names:
        picks = get("pick", name)
        assert len(picks.unique()) == TINY.num_steps
        assert not get("c", name).isnan().any() or name == "random"
    assert results["c"].shape[1] == len(TINY.recorded)
