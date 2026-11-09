"""Models with no belief about their fit, which take in new data by being fitted again."""

from abc import abstractmethod
from typing import Any, Self

import torch
from torch import Tensor
from torch.distributions import Distribution

from predictive_coherence import PredictiveModel

# CatBoost's default seed, kept at every refit so that it has no randomness of its own.
CATBOOST_SEED = 0


def fresh_seed() -> int:
    """A new seed for a learner, drawn from torch's global generator."""
    return int(torch.randint(2**31 - 1, ()))


class Refitted(PredictiveModel):
    """A learner fitted to all the data so far `(n, d)`, `(n,)`, with fixed hyperparameters
    `options`. With no uncertainty about its fit, it draws a block's outputs independently."""

    def __init__(self, x: Tensor, y: Tensor, **options: Any) -> None:
        self.x, self.y, self.options = x, y, options
        self._fit()

    @abstractmethod
    def _fit(self) -> None:
        """Fit to `self.x` and `self.y`."""

    def sample(self, x: Tensor, num_samples: int) -> Tensor:
        return self.predict(x).sample((num_samples,))

    def joint_sample(self, x: Tensor) -> Tensor:
        return self.predict(x).sample()

    def update(self, x: Tensor, y: Tensor) -> Self:
        return type(self)(torch.cat([self.x, x]), torch.cat([self.y, y]), **self.options)


class Autoregressive(PredictiveModel):
    """`model` with the joint its own update rule implies: a block's outputs are drawn one at a
    time, in random order, each from the model updated on those before it. For an exactly
    Bayesian model this is its joint; its predictive and update are unchanged."""

    def __init__(self, model: PredictiveModel) -> None:
        self.model = model

    def predict(self, x: Tensor) -> Distribution:
        return self.model.predict(x)

    def sample(self, x: Tensor, num_samples: int) -> Tensor:
        return self.model.sample(x, num_samples)

    def joint_sample(self, x: Tensor) -> Tensor:
        return PredictiveModel.joint_sample(self.model, x)

    def update(self, x: Tensor, y: Tensor) -> Self:
        return type(self)(self.model.update(x, y))
