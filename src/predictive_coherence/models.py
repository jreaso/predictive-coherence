"""Interface for predictive models, and simulation of their updates."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import NamedTuple, Self

import torch
from torch import Tensor
from torch.distributions import Distribution

from predictive_coherence.kernels import Kernel


class PredictiveModel(ABC):
    """A predictive model over scalar outputs that can be sampled and conditioned on data.

    Subclasses implement `sample` and `update`. Implement `predict` when the kernel can embed
    the predictive exactly, and override `joint_sample` when it can be sampled natively.
    """

    @abstractmethod
    def sample(self, x: Tensor, num_samples: int) -> Tensor:
        """Independent draws from the predictive at each input `(n, ...)`, shape `(num_samples, n)`."""

    @abstractmethod
    def update(self, x: Tensor, y: Tensor) -> Self:
        """The model conditioned on further data `(x, y)`. `self` is left unchanged."""

    def predict(self, x: Tensor) -> Distribution:
        """Marginal predictive at each input, with `batch_shape` `(n,)`."""
        raise NotImplementedError(f"{type(self).__name__} has no closed-form predictive")

    def joint_sample(self, x: Tensor) -> Tensor:
        """Outputs drawn jointly at every input, shape `(n,)`. By default one input at a time, in
        random order, each from the model updated on those drawn before it."""
        order = torch.randperm(len(x)).tolist()
        model, draws = self, []
        for step, i in enumerate(order):
            draws.append(model.sample(x[i : i + 1], 1)[0])
            if step < len(order) - 1:
                model = model.update(x[i : i + 1], draws[-1])
        return torch.cat(draws)[torch.tensor(order).argsort()]


class Round(NamedTuple):
    """Embeddings from one round, in the argument order of `CoherenceEstimator.update`.

    Attributes:
        mu_p: the current predictive at each evaluation point, shape `(L, d)`.
        mu_r: the round's post-update predictive, shape `(L, d)`.
        var_p: Monte Carlo variance of `mu_p`, shape `(L,)`, zero if exact.
        var_r: Monte Carlo variance of `mu_r`, shape `(L,)`, zero if exact.
    """

    mu_p: Tensor
    mu_r: Tensor
    var_p: Tensor
    var_r: Tensor


def iter_rounds(
    model: PredictiveModel,
    kernel: Kernel,
    x_design: Tensor,
    x_eval: Tensor,
    num_samples: int | None = None,
) -> Iterator[Round]:
    """Simulate independent updates of `model`, one round at a time, without end.

    Each round draws outputs jointly at `x_design` from the model, updates a copy of it on them,
    and embeds the current and post-update predictives at `x_eval`.

    Args:
        x_design: where outputs are drawn, shape `(n, ...)`.
        x_eval: where predictives are compared, shape `(L, ...)`.
        num_samples: `None` to embed exactly through `predict`; otherwise the number of samples
            each predictive is embedded from, drawn afresh every round.
    """
    if num_samples is None:
        mu_p = kernel.embed(model.predict(x_eval))
        exact = mu_p.new_zeros(mu_p.shape[:-1])
        while True:
            updated = model.update(x_design, model.joint_sample(x_design))
            yield Round(mu_p, kernel.embed(updated.predict(x_eval)), exact, exact)

    while True:
        updated = model.update(x_design, model.joint_sample(x_design))
        mu_p, var_p = kernel.embed_samples(model.sample(x_eval, num_samples))
        mu_r, var_r = kernel.embed_samples(updated.sample(x_eval, num_samples))
        yield Round(mu_p, mu_r, var_p, var_r)
