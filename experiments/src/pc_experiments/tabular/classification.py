"""Classifiers: multinomial logistic regression from exact Bayes to a plug-in, and refitted
learners.

The logistic models share the class scores `W [x, 1]`, with weights `N(0, prior_sd^2)` and
intercepts `N(0, INTERCEPT_SD^2)`, and differ in what they keep of the posterior over `W`, held
flattened to `(K (d + 1),)`.
"""

import math
from functools import cache
from typing import Any, Self

import catboost
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor
from torch.distributions import Categorical

from pc_experiments.tabular.refitted import CATBOOST_SEED, Refitted
from predictive_coherence import PredictiveModel

INTERCEPT_SD = 10.0
# Points standing for a Gaussian approximation, which the predictive averages over.
NUM_POINTS = 1024
# Hamiltonian Monte Carlo: chains, warm-up iterations, leapfrog steps, and the acceptance rate
# the step size is tuned towards.
NUM_CHAINS, NUM_WARMUP, NUM_LEAPFROG, TARGET_ACCEPTANCE = 128, 150, 8, 0.8


def design(x: Tensor) -> Tensor:
    """Inputs `(n, d)` with a column of ones for the intercept, shape `(n, d + 1)`."""
    return torch.cat([x, x.new_ones(len(x), 1)], -1)


def log_likelihood(weights: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Log likelihood of labels `(n,)` at inputs `(n, d)` under weights `(..., K, d + 1)`."""
    scores = (weights @ design(x).T).transpose(-1, -2)  # (..., n, K)
    return Categorical(logits=scores).log_prob(y).sum(-1)


def prior_precision(x: Tensor, num_classes: int, prior_sd: float) -> Tensor:
    """Of the flattened weights, `(K (d + 1),)`, each class's intercept last."""
    per_class = x.new_full((x.shape[1] + 1,), prior_sd**-2)
    per_class[-1] = INTERCEPT_SD**-2
    return per_class.repeat(num_classes)


def log_posterior(
    weights: Tensor, x: Tensor, y: Tensor, num_classes: int, prior_sd: float
) -> Tensor:
    """Log posterior density of flattened weights `(..., K (d + 1))`, up to a constant, `(...)`."""
    prior = prior_precision(x, num_classes, prior_sd)
    penalty = 0.5 * (prior * weights.square()).sum(-1)
    return log_likelihood(weights.unflatten(-1, (num_classes, -1)), x, y) - penalty


class _Logistic(PredictiveModel):
    """A mixture over weight matrices `(S, K, d + 1)` with log probabilities `log_mixture`."""

    weights: Tensor
    log_mixture: Tensor

    def predict(self, x: Tensor) -> Categorical:
        scores = (self.weights @ design(x).T).transpose(-1, -2)  # (S, m, K)
        return Categorical(probs=self.log_mixture.exp() @ scores.softmax(-1).transpose(0, 1))

    def sample(self, x: Tensor, num_samples: int) -> Tensor:
        return self.predict(x).sample((num_samples,))

    def joint_sample(self, x: Tensor) -> Tensor:
        member = Categorical(logits=self.log_mixture).sample()
        return Categorical(logits=design(x) @ self.weights[member].T).sample()


class ApproximateLogistic(_Logistic):
    """A Gaussian at the posterior mode, found again at every update: the Laplace approximation
    for `covariance` `"full"`, its precision's diagonal for `"diagonal"`, and the mode alone, the
    plug-in model, for `"none"`."""

    def __init__(
        self, x: Tensor, y: Tensor, num_classes: int, prior_sd: float, covariance: str
    ) -> None:
        self.x, self.y = x, y
        self.options: dict[str, Any] = dict(
            num_classes=num_classes, prior_sd=prior_sd, covariance=covariance
        )
        mode, precision = laplace(x, y, num_classes, prior_sd)
        noise = _standard_points(mode.numel()).to(mode)
        if covariance == "none":
            points = mode.unsqueeze(0)
        elif covariance == "diagonal":
            points = mode + noise / precision.diagonal().sqrt()
        else:
            # If P = R R^T then R^{-T} z has covariance P^{-1}.
            root = torch.linalg.cholesky(precision)
            points = mode + torch.linalg.solve_triangular(root.T, noise.T, upper=True).T
        self.weights = points.unflatten(-1, (num_classes, -1))
        self.log_mixture = x.new_full((len(points),), -math.log(len(points)))

    def update(self, x: Tensor, y: Tensor) -> Self:
        return type(self)(torch.cat([self.x, x]), torch.cat([self.y, y]), **self.options)


class ExactLogistic(_Logistic):
    """Exact Bayes over a fixed set of weight matrices, which an update reweights. `fit` draws the
    set by Hamiltonian Monte Carlo; whatever the sampler did, the model is then exactly coherent,
    since conditioning a discrete prior is exact."""

    def __init__(self, weights: Tensor, log_mixture: Tensor) -> None:
        self.weights, self.log_mixture = weights, log_mixture.log_softmax(0)

    @classmethod
    def fit(
        cls, x: Tensor, y: Tensor, num_classes: int, prior_sd: float, num_draws: int = 8192
    ) -> Self:
        draws = posterior_draws(x, y, num_classes, prior_sd, num_draws)
        return cls(draws.unflatten(-1, (num_classes, -1)), x.new_zeros(len(draws)))

    def update(self, x: Tensor, y: Tensor) -> Self:
        return type(self)(self.weights, self.log_mixture + log_likelihood(self.weights, x, y))


@cache
def _standard_points(dim: int, seed: int = 0) -> Tensor:
    """`NUM_POINTS` fixed standard normal draws in `dim` dimensions, so that an approximation and
    its update differ only in where the points are placed."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_POINTS, dim, generator=generator, dtype=torch.float64)


def laplace(x: Tensor, y: Tensor, num_classes: int, prior_sd: float) -> tuple[Tensor, Tensor]:
    """The posterior mode of the flattened weights, `(K (d + 1),)`, and the precision there."""
    prior = prior_precision(x, num_classes, prior_sd)
    weights = x.new_zeros(len(prior), requires_grad=True)
    optimiser = torch.optim.LBFGS(
        [weights], max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )  # fmt: skip

    def negative_log_posterior() -> Tensor:
        optimiser.zero_grad()
        loss = -log_posterior(weights, x, y, num_classes, prior_sd)
        loss.backward()
        return loss

    optimiser.step(negative_log_posterior)
    mode = weights.detach()

    # The Hessian of the negative log likelihood is sum_i (diag(p_i) - p_i p_i^T) ⊗ phi_i phi_i^T.
    basis = design(x)
    probs = (basis @ mode.unflatten(-1, (num_classes, -1)).T).softmax(-1)  # (n, K)
    curvature = torch.diag_embed(probs) - probs.unsqueeze(-1) * probs.unsqueeze(-2)
    hessian = torch.einsum("nkl,nd,ne->kdle", curvature, basis, basis)
    return mode, hessian.reshape(len(prior), len(prior)) + torch.diag(prior)


def posterior_draws(
    x: Tensor, y: Tensor, num_classes: int, prior_sd: float, num_draws: int
) -> Tensor:
    """Hamiltonian Monte Carlo on the flattened weights, `(num_draws, K (d + 1))`, in coordinates
    where the Laplace approximation is standard normal, `weights = mode + R^{-T} z` with `R R^T`
    the precision at the mode, so that unit masses and one step size serve every direction."""
    mode, precision = laplace(x, y, num_classes, prior_sd)
    root = torch.linalg.cholesky(precision)

    def to_weights(z: Tensor) -> Tensor:
        return mode + torch.linalg.solve_triangular(root.T, z.T, upper=True).T

    def value_and_gradient(z: Tensor) -> tuple[Tensor, Tensor]:
        z = z.detach().requires_grad_()
        value = log_posterior(to_weights(z), x, y, num_classes, prior_sd)
        (gradient,) = torch.autograd.grad(value.sum(), z)  # chains are independent
        return value.detach(), gradient

    z = torch.randn(NUM_CHAINS, len(mode)).to(x)
    value, gradient = value_and_gradient(z)
    step_size = len(mode) ** -0.25
    kept: list[Tensor] = []
    for iteration in range(NUM_WARMUP + -(-num_draws // NUM_CHAINS)):
        momentum = torch.randn_like(z)
        proposal, new_momentum = z, momentum
        new_value, new_gradient = value, gradient
        for _ in range(NUM_LEAPFROG):
            new_momentum = new_momentum + 0.5 * step_size * new_gradient
            proposal = proposal + step_size * new_momentum
            new_value, new_gradient = value_and_gradient(proposal)
            new_momentum = new_momentum + 0.5 * step_size * new_gradient
        log_ratio = (
            new_value - 0.5 * new_momentum.square().sum(-1)
            - value + 0.5 * momentum.square().sum(-1)
        )  # fmt: skip
        accept = torch.rand(NUM_CHAINS).to(x).log() < log_ratio
        z = torch.where(accept.unsqueeze(-1), proposal, z)
        gradient = torch.where(accept.unsqueeze(-1), new_gradient, gradient)
        value = torch.where(accept, new_value, value)
        if iteration < NUM_WARMUP:
            step_size *= 1.1 if accept.double().mean() > TARGET_ACCEPTANCE else 0.9
        else:
            kept.append(to_weights(z))
    return torch.cat(kept)[:num_draws]


def _catboost(**options: Any) -> catboost.CatBoostClassifier:
    return catboost.CatBoostClassifier(
        random_seed=CATBOOST_SEED, thread_count=1, verbose=False, allow_writing_files=False,
        **options,
    )  # fmt: skip


# Learners that give class probabilities, by the name of the arm that uses them.
LEARNERS = {
    "neighbours": KNeighborsClassifier,
    "catboost": _catboost,
}


class RefittedClassifier(Refitted):
    """A learner from `LEARNERS`, refitted on all the data at an update. Neither has randomness
    of its own."""

    def __init__(self, kind: str, x: Tensor, y: Tensor, num_classes: int, **options: Any) -> None:
        self.kind, self.num_classes = kind, num_classes
        super().__init__(x, y, **options)

    def _fit(self) -> None:
        self._fitted = LEARNERS[self.kind](**self.options).fit(self.x.numpy(), self.y.numpy())

    def predict(self, x: Tensor) -> Categorical:
        probs = torch.tensor(self._fitted.predict_proba(x.numpy())).to(x)
        # A class the learner never saw gets probability zero.
        over_all = probs.new_zeros(len(probs), self.num_classes)
        over_all[:, torch.tensor(self._fitted.classes_)] = probs
        return Categorical(probs=over_all)

    def update(self, x: Tensor, y: Tensor) -> Self:
        return type(self)(
            self.kind, torch.cat([self.x, x]), torch.cat([self.y, y]), self.num_classes,
            **self.options,
        )  # fmt: skip
