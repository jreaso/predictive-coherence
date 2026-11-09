import math

import pytest
import torch
from torch.distributions import Categorical, Normal

from predictive_coherence import DiracKernel, RFFKernel


def test_features_have_unit_norm():
    assert torch.allclose(RFFKernel().features(torch.randn(10)).norm(dim=-1), torch.ones(10))
    labels = torch.randint(3, (10,))
    assert torch.allclose(DiracKernel(3).features(labels).norm(dim=-1), torch.ones(10))


def test_rff_approximates_gaussian_kernel():
    kernel = RFFKernel(lengthscale=0.5, num_features=20_000)
    y = torch.linspace(-1, 1, 5)
    gram = kernel.features(y) @ kernel.features(y).T
    exact = torch.exp(-((y[:, None] - y[None, :]) ** 2) / (2 * 0.5**2))
    assert torch.allclose(gram, exact, atol=0.03)


def test_rff_embedding_matches_monte_carlo():
    kernel = RFFKernel(lengthscale=0.7, num_features=32)
    dist = Normal(torch.tensor([-1.0, 0.0, 2.0]), torch.tensor([0.1, 1.0, 0.5]))
    mean, _ = kernel.embed_samples(dist.sample((200_000,)))
    assert torch.allclose(kernel.embed(dist), mean, atol=5e-3)


def test_dirac_embedding_is_probabilities():
    dist = Categorical(probs=torch.tensor([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]]))
    kernel = DiracKernel(3)
    assert torch.equal(kernel.embed(dist), dist.probs)
    mean, _ = kernel.embed_samples(dist.sample((100_000,)))
    assert torch.allclose(mean, dist.probs, atol=5e-3)


def test_sample_embedding_variance_is_unbiased():
    kernel = RFFKernel(lengthscale=0.5, num_features=16)
    dist = Normal(torch.zeros(20_000), torch.ones(20_000))
    mean, variance = kernel.embed_samples(dist.sample((8,)))
    squared_error = (mean - kernel.embed(dist)).square().sum(-1)
    assert math.isclose(squared_error.mean(), variance.mean(), rel_tol=0.02)


def test_embed_rejects_unsupported_distributions():
    with pytest.raises(TypeError):
        RFFKernel().embed(Categorical(probs=torch.ones(3) / 3))
    with pytest.raises(TypeError):
        DiracKernel(3).embed(Normal(0.0, 1.0))
