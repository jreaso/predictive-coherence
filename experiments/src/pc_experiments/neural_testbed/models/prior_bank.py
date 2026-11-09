"""Exact Bayes over a finite bank of networks drawn from the environments' own prior.

The coherent control: `A` networks drawn as an environment is, with equal prior weight, and
conditioning by Bayes' rule on that finite set,

    w_a  <-  w_a prod_i p_a(y_i | x_i),   renormalised.

With `p(z) = sum_a w_a p_a(z)` the joint over a batch's labels and `w_a(z)` the weights given `z`,

    E_z[q_z(y | x)] = sum_z p(z) sum_a w_a(z) p_a(y | x)
                    = sum_a w_a p_a(y | x) sum_z p_a(z) = p(y | x),

so the model is exactly coherent with any number of atoms, whether or not its prior is right.
"""

import math
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from pc_experiments.neural_testbed.environment import (
    Setting,
    balance_grid,
    balance_offsets,
    class_log_probs,
    generator,
)
from pc_experiments.neural_testbed.models.base import MAX_ELEMENTS, Mixture, point_chunks


@dataclass(frozen=True)
class PriorBankConfig:
    num_atoms: int


class PriorBank(Mixture):
    """Posterior weights over each environment's own bank of prior networks.

    Attributes:
        theta: the atoms' parameters, shape `(E, A, P)`.
        offset: their class-balancing offsets, shape `(E, A, K)`.
        log_w: log posterior weights, shape `(E, A)`, or `(E, U, A)` once updated.
    """

    def __init__(self, setting: Setting, theta: Tensor, offset: Tensor, log_w: Tensor) -> None:
        self.setting = setting
        self.theta = theta
        self.offset = offset
        self.log_w = log_w

    @classmethod
    def fit(
        cls, config: PriorBankConfig, setting: Setting, x: Tensor, y: Tensor, seeds: tuple[int, ...]
    ) -> Self:
        """Condition an equal prior over fresh atoms on inputs `(E, n, d)` and labels `(E, n)`."""
        architecture = setting.architecture
        theta = torch.stack(
            [
                architecture.sample_prior(config.num_atoms, generator(seed, "prior_bank"))
                for seed in seeds
            ]
        ).to(x.device)
        # Each atom is balanced as an environment is, so that the bank is a sample of the very
        # distribution environments come from.
        grid = balance_grid(x.device)
        size = max(1, MAX_ELEMENTS // (len(grid) * architecture.hidden_width * len(seeds)))
        offset = torch.cat(
            [
                balance_offsets(architecture.logits(grid, chunk), setting.temperature)
                for chunk in theta.split(size, dim=1)
            ],
            dim=1,
        )
        prior = theta.new_full(theta.shape[:2], -math.log(config.num_atoms))
        model = cls(setting, theta, offset, prior)
        model.log_w = model._condition(x.unsqueeze(1), y[:, None, None])[:, 0]
        return model

    def atom_log_probs(self, x: Tensor) -> Tensor:
        return class_log_probs(
            self.setting.architecture,
            x.unsqueeze(1),
            self.theta,
            self.offset,
            self.setting.temperature,
        )

    def log_weights(self) -> Tensor:
        return self.log_w

    def _condition(self, x: Tensor, y: Tensor) -> Tensor:
        """Log weights `(E, N * T, A)` given labels `(E, N, T, k)` at inputs `(E, N, k, d)`."""
        num_batches, size = x.shape[1:3]
        log_probs = self.atom_log_probs(x.flatten(1, 2)).unflatten(-2, (num_batches, size))
        one_hot = torch.nn.functional.one_hot(y, log_probs.shape[-1]).to(log_probs.dtype)
        log_likelihood = torch.einsum("entjk,eanjk->enta", one_hot, log_probs)
        return torch.log_softmax(self.log_w[:, None, None] + log_likelihood, dim=-1).flatten(1, 2)

    def update(self, x: Tensor, y: Tensor) -> Self:
        updated = type(self)(self.setting, self.theta, self.offset, self._condition(x, y))
        updated.update_shape = tuple(y.shape[1:3])
        return updated

    def predict(self, x: Tensor) -> Tensor:
        weights = self.log_w.exp() if self.update_shape else self.log_w.exp().unsqueeze(1)
        per_point = self.theta[..., 0].numel() * self.setting.architecture.hidden_width
        chunks = []
        for chunk in point_chunks(x.shape[-2], per_point):
            probs = self.atom_log_probs(x[:, chunk]).exp()
            # The mixture as a matrix product, which makes thousands of atoms affordable.
            chunks.append((weights @ probs.flatten(-2)).unflatten(-1, probs.shape[-2:]))
        log_probs = torch.cat(chunks, dim=-2).clamp_min(1e-300).log()
        return log_probs.unflatten(1, self.update_shape) if self.update_shape else log_probs[:, 0]
