"""When the test rejects, against what the estimates predict."""

import math

import torch

from pc_experiments.detectability import first_rejection, predicted_round, saturated_rate
from predictive_coherence import SequentialTest


def test_first_rejection_and_the_predicted_round():
    path = torch.stack([torch.linspace(0, 6, 60), torch.zeros(60)])
    assert first_rejection(path, 0.05)[0] == 31 and first_rejection(path, 0.05)[1].isnan()
    mmd_sq, v = torch.tensor([[0.04, 0.0]]), torch.tensor([[0.4, 1.0]])
    assert torch.allclose(saturated_rate(mmd_sq)[0, 0], torch.tensor(0.05))
    assert saturated_rate(mmd_sq)[0, 1].isnan()
    expected = 0.4 / 0.04 + math.log(2 / 0.05) / 0.05
    assert torch.isclose(predicted_round(mmd_sq, v, 0.05)[0], torch.tensor(expected))
    assert predicted_round(torch.zeros(1, 2), v, 0.05)[0] == math.inf


def test_the_round_of_rejection_is_near_the_prediction():
    """A fixed displacement with noise around it, which the test rejects near the predicted round."""
    delta, noise, rounds, m = 0.08, 0.05, 3000, 3
    direction = torch.zeros(m, 8)
    direction[:, 0] = 1
    test = SequentialTest()
    for _ in range(rounds):
        displacement = delta * direction + noise * (2 * torch.rand(m, 8) - 1)
        test.update(torch.zeros(m, 8), displacement)
    path = torch.tensor(test.log_wealth)
    mmd_sq = torch.full((m,), delta**2)
    v = torch.full((m,), 8 * noise**2 / 3)
    rejected, predicted = (
        first_rejection(path, 0.05).item(),
        predicted_round(mmd_sq, v, 0.05).item(),
    )
    assert 0.5 < rejected / predicted < 2
