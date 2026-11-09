"""The tabular experiment's models, against closed forms and scikit-learn."""

import math

import pytest
import torch
from sklearn import kernel_ridge, linear_model, neighbors

from pc_experiments.gp import ARDGaussianPrior, ExactGP
from pc_experiments.tabular import regression
from pc_experiments.tabular.classification import (
    ApproximateLogistic,
    ExactLogistic,
    RefittedClassifier,
    laplace,
    log_likelihood,
    log_posterior,
    prior_precision,
)


def regression_data(num_rows=40):
    x = torch.randn(num_rows, 3)
    return x, torch.sin(x[:, 0]) + x[:, 1] + 0.2 * torch.randn(num_rows), torch.randn(7, 3)


def labelled_data(num_rows=200, num_classes=3):
    x = torch.randn(num_rows, 2)
    scores = torch.stack([1.5 * x[:, 0], 1.5 * x[:, 1], -x.sum(-1)], -1)[:, :num_classes]
    return x, torch.distributions.Categorical(logits=scores).sample()


def leave_one_out_scale(model_class, x, y, **options):
    """Root mean square error of each output predicted by the model fitted without it."""
    errors = []
    for i in range(len(x)):
        kept = torch.arange(len(x)) != i
        errors.append(model_class(x[kept], y[kept], **options).mean(x[i : i + 1]) - y[i])
    return torch.cat(errors).square().mean().sqrt()


# ----- plug-in regression -------------------------------------------------------------------------


def test_kernel_ridge_is_scikit_learns_and_its_scale_leaves_each_point_out():
    x, y, x_new = regression_data()
    model = regression.KernelRidge(x, y, alpha=0.3, gamma=0.2)
    theirs = kernel_ridge.KernelRidge(kernel="rbf", gamma=0.2).set_params(alpha=0.3)
    theirs.fit(x.numpy(), y.numpy())
    torch.testing.assert_close(model.mean(x_new), torch.tensor(theirs.predict(x_new.numpy())))
    torch.testing.assert_close(
        model.scale, leave_one_out_scale(regression.KernelRidge, x, y, alpha=0.3, gamma=0.2)
    )


def test_nearest_neighbours_is_scikit_learns_and_its_scale_leaves_each_point_out():
    x, y, x_new = regression_data()
    model = regression.NearestNeighbours(x, y, n_neighbors=4)
    theirs = neighbors.KNeighborsRegressor(4).fit(x.numpy(), y.numpy())
    torch.testing.assert_close(model.mean(x_new), torch.tensor(theirs.predict(x_new.numpy())))
    torch.testing.assert_close(
        model.scale, leave_one_out_scale(regression.NearestNeighbours, x, y, n_neighbors=4)
    )


def test_a_plug_in_has_one_scale_everywhere_and_refits_on_everything():
    x, y, x_new = regression_data()
    model = regression.KernelRidge(x, y, alpha=0.3, gamma=0.2)
    predictive = model.predict(x_new)
    torch.testing.assert_close(predictive.stddev, model.scale.expand(7))
    updated = model.update(x_new, predictive.mean)
    refitted = regression.KernelRidge(
        torch.cat([x, x_new]), torch.cat([y, predictive.mean]), alpha=0.3, gamma=0.2
    )
    torch.testing.assert_close(updated.mean(x_new), refitted.mean(x_new))
    assert len(model.x) == 40 and updated.options == model.options


def test_catboost_predicts_a_mean_and_a_variance_in_that_order():
    # A step in the mean and a step in the noise, in different places: each column must follow
    # the one it belongs to, which tells the two apart.
    x = torch.rand(2000, 1) * 4 - 2
    y = torch.where(x[:, 0] > 0, 5.0, 0.0) + torch.where(
        x[:, 0].abs() > 1, 1.0, 0.05
    ) * torch.randn(2000)
    predictive = regression.CatBoost(x, y, iterations=200, depth=3).predict(x)
    near, far = x[:, 0].abs() < 0.8, x[:, 0].abs() > 1.2
    above, below = x[:, 0] > 0.2, x[:, 0] < -0.2
    assert predictive.mean[above].mean() > 4 and predictive.mean[below].mean().abs() < 1
    assert predictive.stddev[near].mean() < 0.5 < predictive.stddev[far].mean()


def test_bart_takes_fresh_randomness_from_torchs_seed():
    x, y, x_new = regression_data(80)
    torch.manual_seed(1)
    first = regression.BART(x, y)
    second = regression.BART(x, y)
    torch.manual_seed(1)
    again = regression.BART(x, y)
    torch.testing.assert_close(first.predict(x_new).mean, again.predict(x_new).mean)
    assert not torch.equal(first.predict(x_new).mean, second.predict(x_new).mean)


@pytest.mark.parametrize("sampler", [regression.BART, regression.XBART], ids=["bart", "xbart"])
def test_bart_predicts_the_mixture_of_its_draws_and_draws_a_block_from_one(sampler):
    x, y, x_new = regression_data(200)
    model = sampler(x, y)
    # A warm start keeps only the Markov chain's draws; sweeps alone keep the sweeps.
    retained = (sampler.num_mcmc or sampler.num_gfr) - sampler.num_discarded
    means = model.draws(x_new)
    assert means.shape == (7, retained)
    predictive = model.predict(x_new)
    torch.testing.assert_close(predictive.mean, means.mean(-1))
    assert (model.predict(x).mean - y).square().mean() < 0.2 * y.var()  # it fits
    # A block of two copies of one input: its outputs share a draw, so they are correlated by
    # the spread of the draws' means there, and each is distributed as the predictive.
    pair = x_new[:1].repeat(2, 1)
    blocks = torch.stack([model.joint_sample(pair) for _ in range(4000)])
    spread = means[0].var(correction=0)
    torch.testing.assert_close(torch.cov(blocks.T)[0, 1], spread, atol=0.02, rtol=0.25)
    torch.testing.assert_close(blocks.mean(0), predictive.mean[:1].expand(2), atol=0.05, rtol=0)
    torch.testing.assert_close(
        blocks.var(0), predictive.variance[:1].expand(2), atol=0.02, rtol=0.15
    )


@pytest.mark.parametrize("sampler", [regression.BART, regression.XBART], ids=["bart", "xbart"])
def test_bart_splits_on_inputs_with_many_ties(sampler):
    x = torch.randint(0, 6, (300, 3)).double()  # six values each, every one repeated
    y = torch.sin(x[:, 0]) + x[:, 1] / 3 + 0.1 * torch.randn(300)
    model = sampler(x, y)
    assert (model.predict(x).mean - y).square().mean() < 0.1 * y.var()


def test_bart_takes_in_a_block_by_sampling_again_from_all_the_data():
    x, y, x_new = regression_data(60)
    model = regression.BART(x, y)
    updated = model.update(x_new, model.joint_sample(x_new))
    assert len(updated.x) == 67 and len(model.x) == 60 and updated.options == model.options


def test_a_refitted_gp_draws_from_its_hyperparameters_and_changes_them_when_updated():
    x, x_new = torch.randn(60, 2), torch.randn(5, 2)
    y = torch.sin(2 * x[:, 0]) + 0.1 * torch.randn(60)
    prior = ARDGaussianPrior((1.0, 1.0), 1.0, 0.3)  # not the best, so that a refit moves it
    model = regression.RefitGP(prior, x, y)
    fixed = ExactGP(prior, x, y).joint_predict(x_new)
    for ours, theirs in zip(model.joint_predict(x_new), fixed, strict=True):
        torch.testing.assert_close(ours, theirs)

    updated = model.update(x_new, model.joint_sample(x_new))
    assert updated.prior != prior and model.prior == prior
    assert updated.prior.noise_std < prior.noise_std  # towards the noise the data have
    assert len(updated.x) == 65


# ----- logistic regression ------------------------------------------------------------------------


def test_the_mode_is_scikit_learns_logistic_regression():
    x, y = labelled_data()
    mode, _ = laplace(x, y, num_classes=3, prior_sd=0.7)
    theirs = linear_model.LogisticRegression(C=0.7**2, tol=1e-10, max_iter=1000)
    theirs.fit(x.numpy(), y.numpy())
    ours = ApproximateLogistic(x, y, 3, prior_sd=0.7, covariance="none").predict(x).probs
    # Theirs leaves the intercepts unpenalised, where ours have a wide prior.
    torch.testing.assert_close(
        ours, torch.tensor(theirs.predict_proba(x.numpy())), atol=2e-3, rtol=0
    )
    assert mode.shape == (9,)


def test_the_precision_is_the_hessian_of_the_negative_log_posterior():
    x, y = labelled_data(60)
    mode, precision = laplace(x, y, num_classes=3, prior_sd=0.7)

    def negative_log_posterior(weights):
        return -log_posterior(weights, x, y, num_classes=3, prior_sd=0.7)

    torch.testing.assert_close(
        precision, torch.autograd.functional.hessian(negative_log_posterior, mode)
    )
    gradient = torch.autograd.functional.jacobian(negative_log_posterior, mode)
    assert gradient.abs().max() < 1e-5


def test_the_log_posterior_is_the_log_likelihood_less_a_penalty_per_member():
    x, y = labelled_data(30, num_classes=2)
    weights = torch.randn(4, 6)
    prior = prior_precision(x, 2, 0.7)
    expected = log_likelihood(weights.unflatten(-1, (2, -1)), x, y) - 0.5 * (
        prior * weights.square()
    ).sum(-1)
    torch.testing.assert_close(log_posterior(weights, x, y, 2, 0.7), expected)
    assert expected.shape == (4,)


@pytest.mark.parametrize("covariance", ["full", "diagonal"])
def test_an_approximations_points_have_its_mean_and_covariance(covariance):
    x, y = labelled_data(80, num_classes=2)
    mode, precision = laplace(x, y, num_classes=2, prior_sd=1.0)
    model = ApproximateLogistic(x, y, 2, prior_sd=1.0, covariance=covariance)
    points = model.weights.flatten(1)
    expected = torch.linalg.inv(precision)
    if covariance == "diagonal":
        expected = torch.diag(1 / precision.diagonal())
    scale = expected.diagonal().max()
    assert (points.mean(0) - mode).abs().max() < 0.15 * scale.sqrt()
    assert (torch.cov(points.T) - expected).abs().max() < 0.2 * scale


def test_a_block_is_drawn_from_one_member_of_the_mixture():
    # Two members that disagree at every input: within a draw every label comes from one of them,
    # so the labels agree with each other, which the product of the marginals would not give.
    weights = torch.zeros(2, 2, 3)
    weights[0, 0, -1], weights[1, 1, -1] = 20.0, 20.0
    model = ExactLogistic(weights, torch.zeros(2))
    x_new = torch.randn(6, 2)
    draws = torch.stack([model.joint_sample(x_new) for _ in range(200)])
    assert ((draws == 0).all(-1) | (draws == 1).all(-1)).all()
    assert 0 < (draws[:, 0] == 0).double().mean() < 1
    torch.testing.assert_close(
        model.predict(x_new).probs, torch.full((6, 2), 0.5), atol=1e-8, rtol=0
    )


def test_exact_logistic_updates_by_bayes_rule():
    weights = torch.randn(5, 2, 3)
    model = ExactLogistic(weights, torch.zeros(5))
    x, y = torch.randn(4, 2), torch.tensor([0, 1, 1, 0])
    updated = model.update(x, y)
    posterior = log_likelihood(weights, x, y).softmax(0)  # from a uniform prior
    torch.testing.assert_close(updated.log_mixture.exp(), posterior)
    assert updated.weights is weights
    torch.testing.assert_close(model.log_mixture.exp(), torch.full((5,), 0.2))
    twice = model.update(x[:2], y[:2]).update(x[2:], y[2:])
    torch.testing.assert_close(twice.log_mixture, updated.log_mixture)


def test_posterior_draws_have_the_posteriors_mean_and_spread():
    # With many labels the posterior is close to its Laplace approximation.
    x, y = labelled_data(600, num_classes=2)
    mode, precision = laplace(x, y, num_classes=2, prior_sd=1.0)
    draws = ExactLogistic.fit(x, y, 2, prior_sd=1.0, num_draws=4096).weights.flatten(1)
    assert draws.shape == (4096, 6)
    # The two classes' weights are tied only by the prior, so compare what the labels inform:
    # the difference between them.
    contrast = torch.cat([torch.eye(3), -torch.eye(3)], -1)
    expected = contrast @ torch.linalg.inv(precision) @ contrast.T
    difference = draws @ contrast.T
    spread = expected.diagonal().sqrt()
    assert ((difference.mean(0) - contrast @ mode).abs() < 0.2 * spread).all()
    assert (difference.std(0) / spread - 1).abs().max() < 0.15


def test_a_class_absent_from_the_data_is_given_no_probability():
    x, y = labelled_data(60)
    model = RefittedClassifier("neighbours", x[y != 1], y[y != 1], 4, n_neighbors=5)
    probs = model.predict(x[:6]).probs
    assert probs.shape == (6, 4) and (probs[:, [1, 3]] == 0).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(6))
    assert math.isclose(probs[0, 0].item() * 5, round(probs[0, 0].item() * 5))
