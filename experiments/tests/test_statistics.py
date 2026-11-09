"""Rank statistics."""

import torch

from pc_experiments.statistics import kendall_w, ranks, spearman


def test_ranks_average_ties():
    assert ranks(torch.tensor([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_spearman_is_one_for_any_monotone_relation_and_batched():
    x = torch.rand(4, 20)
    assert torch.allclose(spearman(x, x.exp()), torch.ones(4))
    assert torch.allclose(spearman(x, -x), -torch.ones(4))


def test_concordance_is_one_for_agreeing_judges_and_small_for_unrelated_ones():
    agreeing = torch.arange(6.0).repeat(5, 1) + torch.rand(5, 1)
    assert torch.isclose(kendall_w(agreeing), torch.tensor(1.0))
    assert kendall_w(torch.rand(200, 6)) < 0.05
