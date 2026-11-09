"""The Bayesian linear regression acquisition experiment: environments, acquirers and the runner.
Every acquirer follows its own trajectory in each environment."""

import math
import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch import Tensor

from pc_experiments.blr.gaussian import (
    Moments,
    Posterior,
    View,
    coherent_mean_field_cov,
    coherent_refit_information,
    conditioning,
    expected_nll,
    gaussian_kl,
    incoherence,
    mean_field_block_refit,
    mean_field_refit,
    quadratic,
)
from pc_experiments.outputs import load_run

DTYPE = torch.float64


# ----- environments ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """One configuration of the sequential acquisition problem.

    Attributes:
        misspecification: share of the truth's prior variance carried by random features the
            model does not have. Zero is well specified.
        checkpoints: steps at which incoherence and marginal accuracy are recorded.
        psi_lengthscale: bandwidth of the Gaussian kernel that incoherence is measured under.
    """

    num_features: int = 100
    lengthscale: float = 0.1
    prior_var: float = 3.0
    noise_var: float = 0.8
    num_init: int = 25
    num_pool: int = 1000
    num_targets: int = 100
    num_steps: int = 100
    input_range: tuple[float, float] = (-10.0, 10.0)
    init_range: tuple[float, float] = (-1.0, 1.0)
    target_range: tuple[float, float] = (-10.0, 10.0)
    misspecification: float = 0.0
    checkpoints: tuple[int, ...] = (0, 10, 25, 50, 100)
    psi_lengthscale: float = math.sqrt(0.8)

    @property
    def recorded(self) -> tuple[int, ...]:
        return tuple(step for step in self.checkpoints if step <= self.num_steps)


@dataclass(frozen=True)
class Environment:
    """One environment. Pool labels are drawn once, so acquirers picking a point see the same
    label. `pool_order` is a random order of the pool, which random acquisition follows and from
    which subsampling acquirers take their share."""

    init_features: Tensor
    init_labels: Tensor
    pool_inputs: Tensor
    pool_features: Tensor
    pool_labels: Tensor
    target_inputs: Tensor
    target_features: Tensor
    target_truth: Tensor
    pool_order: Tensor


def sample_environment(setting: Setting, seed: int) -> Environment:
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator, dtype=DTYPE)

    def uniform(size: int, bounds: tuple[float, float]) -> Tensor:
        low, high = bounds
        return low + (high - low) * torch.rand(size, generator=generator, dtype=DTYPE)

    features = _random_features(setting.num_features, setting.lengthscale, generator)
    outside_features = _random_features(setting.num_features, setting.lengthscale, generator)
    weights, outside_weights = math.sqrt(setting.prior_var) * normal(2, setting.num_features)
    inside_share = math.sqrt(1 - setting.misspecification)
    outside_share = math.sqrt(setting.misspecification)

    def truth(inputs: Tensor) -> Tensor:
        inside = features(inputs) @ weights
        return inside_share * inside + outside_share * outside_features(inputs) @ outside_weights

    noise_std = math.sqrt(setting.noise_var)
    init_inputs = uniform(setting.num_init, setting.init_range)
    pool_inputs = uniform(setting.num_pool, setting.input_range)
    target_inputs = uniform(setting.num_targets, setting.target_range)
    return Environment(
        init_features=features(init_inputs),
        init_labels=truth(init_inputs) + noise_std * normal(setting.num_init),
        pool_inputs=pool_inputs,
        pool_features=features(pool_inputs),
        pool_labels=truth(pool_inputs) + noise_std * normal(setting.num_pool),
        target_inputs=target_inputs,
        target_features=features(target_inputs),
        target_truth=truth(target_inputs),
        pool_order=torch.randperm(setting.num_pool, generator=generator),
    )


def _random_features(
    num_features: int, lengthscale: float, generator: torch.Generator
) -> Callable[[Tensor], Tensor]:
    """Random Fourier features of the Gaussian kernel on scalar inputs."""
    frequencies = torch.randn(num_features, generator=generator, dtype=DTYPE) / lengthscale
    phases = 2 * math.pi * torch.rand(num_features, generator=generator, dtype=DTYPE)
    scale = math.sqrt(2 / num_features)
    return lambda inputs: scale * torch.cos(inputs[:, None] * frequencies + phases)


# ----- acquirers ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Acquirer:
    """A way of scoring candidates.

    Attributes:
        kind:
            - `exact`: expected information gain under the exact posterior.
            - `alpha`: EPIG of the mean-field model with its current predictive `p` replaced by the
              moment-matched mixture `(1 - param) p + param q_bar`, where `q_bar` averages the
              model's refits on the candidate. `param = 0` is EPIG and `param = 1` is EMI.
            - `subsample`: expected information gain under the Gaussian predictive for which the
              mean-field refit is coherent over the first `param` share of the remaining pool, in
              the environment's pool order. The candidate updates it by the model's own refit and
              the average over the block carries on, so `param = 0` is the mean-field model's own
              EPIG.
            - `random`: picks the remaining pool in the environment's pool order.
        param: the mixture weight `alpha`, or the share of the remaining pool.
    """

    kind: str
    param: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.kind}{self.param:g}" if self.kind in ("alpha", "subsample") else self.kind

    def reference(self, remaining: Tensor) -> Tensor:
        """The block whose outcomes are averaged over: the first `param` share of `remaining`."""
        return remaining[: round(self.param * len(remaining))]

    def covariance(self, posterior: Posterior, remaining: Tensor) -> Tensor | None:
        """Weight covariance of the predictive planned with, or `None` for random picks, given the
        remaining pool's features `(N, d)` in the environment's pool order."""
        match self.kind:
            case "exact":
                return posterior.cov
            case "alpha":
                return torch.diag(posterior.mean_field_var)
            case "subsample":
                reference = self.reference(remaining)
                return coherent_mean_field_cov(posterior.precision, reference, posterior.noise_var)
        return None

    def scores(self, view: View, cov: Tensor, remaining: Tensor) -> Tensor:
        """Score of every candidate in `view`, shape `(N,)`."""
        if self.kind == "exact":
            return view.exact_information
        if self.kind == "alpha":
            return self._mixed_refit(view).epig()
        return coherent_refit_information(view, self.reference(remaining), cov)

    def anticipated(self, view: View, cov: Tensor, remaining: Tensor) -> Moments:
        """The planning predictive and the update its incoherence is measured against.

        - `exact`: the exact posterior, conditioned on each single candidate.
        - `alpha`: the mixed predictive, with the mean-field refit on each single candidate. At
          `param` 1 the two agree by construction, so the incoherence is exactly zero.
        - `subsample`: its predictive, with the mean-field refit on the whole remaining pool. That
          is the model's own update, which the acquirer only partly anticipates: it averages over
          a block, while the incoherence is measured over everything still to be picked.
        """
        if self.kind == "exact":
            noise_var = view.posterior.noise_var
            return conditioning(view.cross, view.candidate_var, view.target_var, noise_var)
        if self.kind == "alpha":
            return self._mixed_refit(view)
        return mean_field_block_refit(view.posterior, cov, remaining, view.targets)

    def _mixed_refit(self, view: View) -> Moments:
        """The mean-field refit, its current predictive mixed towards its average by moment
        matching, exact at `param` 0 and 1."""
        refit = mean_field_refit(view)
        average = refit.updated + refit.shift
        mixed = (1 - self.param) * refit.current + self.param * average
        return Moments(mixed, refit.updated, refit.shift)


# ----- running -----------------------------------------------------------------------------------


def run_setting(
    setting: Setting, acquirers: Sequence[Acquirer], num_envs: int, workers: int
) -> dict[str, Tensor]:
    """Run every acquirer in `num_envs` environments. Tensors have shape `(acquirers, envs, ...)`."""
    run = partial(run_environment, setting, acquirers)
    _single_thread()  # the pool inherits nothing, and direct calls should match it
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_single_thread) as pool:
        per_env = list(pool.map(run, range(num_envs)))
    return {key: torch.stack([result[key] for result in per_env], dim=1) for key in per_env[0]}


def run_environment(
    setting: Setting, acquirers: Sequence[Acquirer], seed: int
) -> dict[str, Tensor]:
    """Run every acquirer in one environment. Tensors have shape `(acquirers, ...)`.

    `nll_mf`, `(acquirers, T + 1)`: the expected test log loss, after each step, of a mean-field
        learner trained on the data the acquirer picked.
    `pick`, `(acquirers, T)`: the input picked at each step.
    At `setting.recorded`, NaN for `random`:
        `c`, `v`: incoherence of `Acquirer.anticipated`, summed over targets and, for
            single-candidate updates, over the remaining candidates.
        `excess_nll`: the expected excess log loss of the planning predictive over the exact one.
    """
    env = sample_environment(setting, seed)
    trajectories = [_trajectory(acquirer, env, setting) for acquirer in acquirers]
    return {key: torch.stack([out[key] for out in trajectories]) for key in trajectories[0]}


def _trajectory(acquirer: Acquirer, env: Environment, setting: Setting) -> dict[str, Tensor]:
    steps, noise_var = setting.num_steps, setting.noise_var
    recorded = setting.recorded

    def empty(size: int) -> Tensor:
        return torch.full((size,), math.nan, dtype=DTYPE)

    out = {"nll_mf": empty(steps + 1), "pick": empty(steps)}
    out |= {key: empty(len(recorded)) for key in ("c", "v", "excess_nll")}

    posterior = Posterior.fit(env.init_features, env.init_labels, setting.prior_var, noise_var)
    available = torch.ones(setting.num_pool, dtype=torch.bool)
    targets = env.target_features
    # Earlier positions in the pool order score higher, so random picks follow it.
    order_scores = -torch.argsort(env.pool_order).to(DTYPE)
    for t in range(steps + 1):
        ordered = env.pool_order[available[env.pool_order]]
        remaining = env.pool_features[ordered]
        cov = acquirer.covariance(posterior, remaining)
        view = View(posterior, env.pool_features, targets)

        mean = targets @ posterior.mean
        exact_var = quadratic(targets, posterior.cov) + noise_var
        mf_var = targets.square() @ posterior.mean_field_var + noise_var
        out["nll_mf"][t] = expected_nll(mean, mf_var, env.target_truth, noise_var).mean()
        if cov is not None and t in recorded:
            index = recorded.index(t)
            own_var = quadratic(targets, cov) + noise_var
            out["excess_nll"][index] = gaussian_kl(exact_var, own_var).mean()
            moments = acquirer.anticipated(view, cov, remaining)
            c, v = incoherence(moments, setting.psi_lengthscale)
            if c.ndim:
                c, v = c[available].sum(), v[available].sum()
            out["c"][index], out["v"][index] = c, v
        if t == steps:
            break

        scores = order_scores if cov is None else acquirer.scores(view, cov, remaining)
        pick = int(torch.where(available, scores, -math.inf).argmax())
        out["pick"][t] = env.pool_inputs[pick]
        posterior = posterior.update(env.pool_features[pick], env.pool_labels[pick])
        available[pick] = False
    return out


def _single_thread() -> None:
    torch.set_num_threads(1)


# ----- reading results back ----------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """Results for one setting, with tensors of shape `(acquirers, envs, ...)`."""

    acquirers: list[Acquirer]
    checkpoints: tuple[int, ...]
    results: dict[str, Tensor]

    @classmethod
    def load(cls, path: Path) -> "Run":
        payload = load_run(path)
        acquirers = [Acquirer(**acquirer) for acquirer in payload["acquirers"]]
        return cls(acquirers, tuple(payload["checkpoints"]), payload["results"])

    @property
    def names(self) -> list[str]:
        return [acquirer.name for acquirer in self.acquirers]

    @property
    def shares(self) -> list[float]:
        """The subsampling acquirers' shares."""
        return [acquirer.param for acquirer in self.acquirers if acquirer.kind == "subsample"]

    @property
    def num_steps(self) -> int:
        return self.results["nll_mf"].shape[2] - 1

    def get(self, key: str, name: str) -> Tensor:
        return self.results[key][self.names.index(name)]

    def auc(self, key: str, name: str) -> Tensor:
        """The mean over steps `1..T`, `(envs,)`."""
        return self.get(key, name)[:, 1:].mean(-1)

    def loss(self, name: str, learner: str = "nll_mf") -> Tensor:
        """The loss less the exact acquirer's, `(envs,)`."""
        return self.auc(learner, name) - self.auc(learner, "exact")

    def at_checkpoint(self, key: str, name: str, checkpoint: int = 0) -> Tensor:
        return self.get(key, name)[:, self.checkpoints.index(checkpoint)]

    def psi(self, name: str, checkpoint: int = 0) -> Tensor:
        """Ψ in each environment, `(envs,)`."""
        c = self.at_checkpoint("c", name, checkpoint)
        return c / self.at_checkpoint("v", name, checkpoint)
