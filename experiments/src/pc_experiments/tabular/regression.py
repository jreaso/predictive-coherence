"""Regression models other than a Gaussian process with a fixed prior. A plug-in model's scale is
its held-out error; BART is represented by its sampler's draws, taken afresh at every update."""

import warnings
from abc import abstractmethod
from typing import Self

import torch
from catboost import CatBoostRegressor
from stochtree import BARTModel
from torch import Tensor
from torch.distributions import Categorical, MixtureSameFamily, Normal

from pc_experiments.gp import ARDGaussianPrior, ExactGP, fit_ard_prior
from pc_experiments.gp.prior import squared_distances
from pc_experiments.tabular.refitted import CATBOOST_SEED, Refitted, fresh_seed

# L-BFGS iterations when the hyperparameters are refitted from their last values.
REFIT_ITERATIONS = 10


class RefitGP(ExactGP):
    """A Gaussian process whose hyperparameters are refitted when it takes in a block, which is
    drawn under the hyperparameters as they stand."""

    prior: ARDGaussianPrior

    def update(self, x: Tensor, y: Tensor) -> Self:
        x, y = torch.cat([self.x, x]), torch.cat([self.y, y])
        return type(self)(fit_ard_prior(x, y, self.prior, REFIT_ITERATIONS), x, y)


class PlugIn(Refitted):
    """`N(mean(x), scale^2)` at each input, `_fit` setting `scale` from held-out residuals."""

    scale: Tensor

    @abstractmethod
    def mean(self, x: Tensor) -> Tensor:
        """The fitted mean at inputs `(m, d)`, `(m,)`."""

    def predict(self, x: Tensor) -> Normal:
        return Normal(self.mean(x), self.scale.expand(len(x)))


def _root_mean_square(residuals: Tensor) -> Tensor:
    return residuals.square().mean().sqrt()


class KernelRidge(PlugIn):
    """Kernel ridge regression with the kernel `exp(-gamma ||x - x'||^2)` and penalty `alpha`."""

    def _kernel(self, x: Tensor) -> Tensor:
        return torch.exp(-self.options["gamma"] * squared_distances(x, self.x))

    def _fit(self) -> None:
        kernel = self._kernel(self.x)
        smoother = kernel + self.options["alpha"] * torch.eye(len(self.y)).to(kernel)
        self._weights = torch.linalg.solve(smoother, self.y)
        leverage = torch.linalg.solve(smoother, kernel).diagonal()
        # Leaving a point out of a linear smoother divides its residual by `1 - leverage`.
        self.scale = _root_mean_square((self.y - kernel @ self._weights) / (1 - leverage))

    def mean(self, x: Tensor) -> Tensor:
        return self._kernel(x) @ self._weights


class NearestNeighbours(PlugIn):
    """The mean output of the `n_neighbors` nearest training inputs."""

    def _neighbour_mean(self, distances: Tensor) -> Tensor:
        nearest = distances.topk(self.options["n_neighbors"], largest=False).indices
        return self.y[nearest].mean(-1)

    def _fit(self) -> None:
        apart = squared_distances(self.x, self.x).fill_diagonal_(torch.inf)  # leave each one out
        self.scale = _root_mean_square(self.y - self._neighbour_mean(apart))

    def mean(self, x: Tensor) -> Tensor:
        return self._neighbour_mean(squared_distances(x, self.x))


class CatBoost(Refitted):
    """CatBoost with a Gaussian likelihood, so that it predicts a variance, and a fixed seed."""

    def _fit(self) -> None:
        self._trees = CatBoostRegressor(
            loss_function="RMSEWithUncertainty",
            random_seed=CATBOOST_SEED,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            **self.options,
        ).fit(self.x.numpy(), self.y.numpy())

    def predict(self, x: Tensor) -> Normal:
        mean, variance = torch.tensor(self._trees.predict(x.numpy())).unbind(-1)
        return Normal(mean, variance.sqrt())


class BART(Refitted):
    """Bayesian additive regression trees, by MCMC started from a few grow-from-root sweeps as
    `stochtree` recommends. The predictive is the mixture over the retained draws and a block comes
    from one draw, so the incoherence is all in running the sampler afresh at each update.

    Every distinct input value is a candidate cutpoint: with `stochtree`'s default grid of 100,
    the sweeps never split on inputs with many ties, as every regression dataset here has.
    """

    num_gfr, num_mcmc, num_discarded = 5, 100, 0

    def _fit(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Covariate", UserWarning)  # about ties, handled
            self._sampler = BARTModel()
            self._sampler.sample(
                self.x.numpy(),
                self.y.numpy(),
                num_gfr=self.num_gfr,
                num_mcmc=self.num_mcmc,
                general_params={
                    "random_seed": fresh_seed(),
                    "num_threads": 1,
                    "cutpoint_grid_size": len(self.x),
                },
            )
        variances = torch.tensor(self._sampler.global_var_samples)[self.num_discarded :]
        self._noise_std = variances.sqrt()

    def draws(self, x: Tensor) -> Tensor:
        """The sum of trees at inputs `(m, d)` under each retained draw, `(m, S)`."""
        means = self._sampler.predict(x.numpy(), type="posterior", terms="y_hat")
        return torch.tensor(means)[:, self.num_discarded :].to(x)

    def predict(self, x: Tensor) -> MixtureSameFamily:
        means = self.draws(x)
        weights = Categorical(probs=torch.ones_like(means) / means.shape[-1])
        return MixtureSameFamily(weights, Normal(means, self._noise_std.to(x).expand_as(means)))

    def joint_sample(self, x: Tensor) -> Tensor:
        draw = int(torch.randint(len(self._noise_std), ()))
        return self.draws(x)[:, draw] + self._noise_std[draw] * torch.randn(len(x)).to(x)


class XBART(BART):
    """BART by grow-from-root sweeps alone (He and Hahn, 2023), which are fast but do not sample
    the posterior: forty, the first fifteen discarded."""

    num_gfr, num_mcmc, num_discarded = 40, 0, 15
