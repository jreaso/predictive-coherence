import math

import pytest
import torch

from predictive_coherence import SequentialTest


def one_hot_draws(probs: torch.Tensor, num_rounds: int) -> torch.Tensor:
    """Point masses drawn from `probs`, whose embeddings average exactly to `probs`."""
    labels = torch.distributions.Categorical(probs=probs).sample((num_rounds,))
    return torch.nn.functional.one_hot(labels, probs.shape[-1]).to(probs)


PROBS = torch.tensor(
    [[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]], dtype=torch.float64
)


def test_valid_under_the_null():
    rejections = 0
    for _ in range(200):
        test = SequentialTest()
        test.update(PROBS, one_hot_draws(PROBS, 200))
        rejections += test.rejects(alpha=0.1)
    assert rejections / 200 <= 0.1


def test_rejects_under_the_alternative_at_the_first_crossing():
    biased = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]])
    test = SequentialTest()
    test.update(PROBS, one_hot_draws(biased, 500))

    first = test.rejection_round(alpha=0.01)
    assert test.rejects(alpha=0.01) and first is not None
    assert test.log_wealth[first - 1] >= math.log(100) > max(test.log_wealth[: first - 1])
    assert len(test.log_wealth) == 500


def test_wealth_is_the_average_over_points():
    mu_r = one_hot_draws(PROBS, 300)
    joint = SequentialTest()
    joint.update(PROBS, mu_r)

    single = torch.zeros(300, 4)
    for m in range(4):
        test = SequentialTest()
        test.update(PROBS[m : m + 1], mu_r[:, m : m + 1])
        single[:, m] = torch.tensor(test.log_wealth)

    average = single.logsumexp(-1) - math.log(4)
    assert joint.log_wealth == pytest.approx(average.tolist())


def test_batch_update_matches_round_by_round_updates():
    mu_r = one_hot_draws(PROBS, 50)
    batched, sequential = SequentialTest(), SequentialTest()
    batched.update(PROBS, mu_r)
    for round_ in mu_r:
        sequential.update(PROBS, round_)
    assert batched.log_wealth == pytest.approx(sequential.log_wealth)
    assert batched.log_wealth[0] == 0.0


def test_no_bets_below_the_noise_floor():
    test = SequentialTest(noise_floor=2.0)
    test.update(PROBS, one_hot_draws(PROBS, 50))
    assert test.log_wealth == [0.0] * 50


def test_rejects_displacements_beyond_the_kernel_bound():
    with pytest.raises(ValueError):
        SequentialTest(kernel_bound=0.1).update(PROBS, one_hot_draws(PROBS, 5))
