"""A piecewise-uniform predictive, and exact mean embeddings of every predictive used here."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.distributions import Categorical, Distribution, MixtureSameFamily
from torch.distributions.constraints import Constraint

from predictive_coherence import Kernel, RFFKernel


class Bars(Distribution):
    """A density constant between consecutive `borders` `(B + 1,)`, with bar `logits` `(..., B)`:
    TabPFN's regression head.

    TabPFN's outermost bars are really half-normal tails; here they are uniform like the rest,
    which misplaces at most about `1e-5` of the mass on the datasets here.
    """

    def __init__(self, logits: Tensor, borders: Tensor) -> None:
        self.probs = torch.softmax(logits.double(), -1)
        self.borders = borders.double()
        super().__init__(batch_shape=logits.shape[:-1], validate_args=False)

    @property
    def arg_constraints(self) -> dict[str, Constraint]:
        return {}

    @property
    def mean(self) -> Tensor:
        return self.probs @ ((self.borders[:-1] + self.borders[1:]) / 2)

    def sample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        bars = Categorical(probs=self.probs).sample(torch.Size(sample_shape))
        left, right = self.borders[bars], self.borders[bars + 1]
        return left + (right - left) * torch.rand(bars.shape, dtype=left.dtype)

    def log_prob(self, value: Tensor) -> Tensor:
        bars = torch.searchsorted(self.borders, value.double()) - 1
        bars = bars.clamp(0, self.probs.shape[-1] - 1)  # a value beyond the borders: the end bar
        probs = self.probs.expand(*value.shape, -1).gather(-1, bars.unsqueeze(-1)).squeeze(-1)
        return probs.log() - (self.borders[1:] - self.borders[:-1])[bars].log()


def embed(kernel: Kernel, predictive: Distribution) -> Tensor:
    """Exact mean embedding of a predictive, a mixture included, `(*batch_shape, dim)`."""
    if isinstance(predictive, MixtureSameFamily):
        components = embed(kernel, predictive.component_distribution)  # (..., S, dim)
        weights = predictive.mixture_distribution.probs.to(components)
        return (weights.unsqueeze(-1) * components).sum(-2)
    if not isinstance(predictive, Bars):
        return kernel.embed(predictive)
    assert isinstance(kernel, RFFKernel)
    # The mean of exp(i w y) over a bar [a, b] is (exp(i w b) - exp(i w a)) / (i w (b - a)).
    frequencies = kernel.frequencies.unsqueeze(-1)
    edges = frequencies * predictive.borders  # (F, B + 1)
    widths = frequencies * (predictive.borders[1:] - predictive.borders[:-1])
    cos = (edges[:, 1:].sin() - edges[:, :-1].sin()) / widths
    sin = -(edges[:, 1:].cos() - edges[:, :-1].cos()) / widths
    # A bar of zero width carries no mass, and its 0 / 0 is never used.
    features = torch.cat([cos, sin]).nan_to_num(0.0, 0.0, 0.0).T  # (B, 2F)
    return predictive.probs @ features / math.sqrt(kernel.frequencies.numel())
