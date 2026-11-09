"""The neural testbed: environments, exact scoring, candidates, the runner and saved results."""

import dataclasses
import itertools
import math

import pytest
import torch

from pc_experiments.neural_testbed import (
    RULES,
    Architecture,
    Arm,
    Family,
    Results,
    Setting,
    Tasks,
    candidates_for,
    chunks_of,
    class_masses,
    evaluate,
    gather_inputs,
    grid_points,
    is_valid,
    merge_chunks,
    prediction_scores,
    run_chunk,
    sample_environment,
    stack,
    task_loss,
    valid_environments,
)
from pc_experiments.neural_testbed.environment import (
    balance_grid,
    balance_offsets,
    class_log_probs,
    generator,
)
from pc_experiments.neural_testbed.networks import batched_eigh
from pc_experiments.neural_testbed.scoring import (
    coherence_of_random_updates,
    entropy,
    enumerate_labellings,
    expected_loss,
    joint_kl,
    kl_divergence,
    labelling_log_probs,
    mixture_log_joint,
    score_candidates,
)
from pc_experiments.neural_testbed.tuning import AROUND_BATCH, HELD_OUT, held_out_inputs

ARCHITECTURE = Architecture(input_dim=2, hidden_width=8, num_hidden_layers=2, num_classes=3)
SETTING = Setting(
    architecture=ARCHITECTURE,
    temperature=0.35,
    center=3.0,
    train_std=0.5,
    pool_std=1.5,
    target_std=1.5,
    train_clusters=(0, 1, 2),
    target_clusters=(2, 3),
    num_train=12,
    num_pool=64,
    num_target=10,
    num_test=200,
    num_candidates=10,
    candidate_size=3,
    min_class_mass=0.10,
)
ARMS = (
    Arm("exact", "prior_bank", {"num_atoms": 256}),
    Arm(
        "laplace_diag",
        "laplace",
        {"curvature": "diag", "prior_scale": 4.0, "num_atoms": 16, "lr": 0.05, "fit_steps": 100},
    ),
)


# ----- networks -----------------------------------------------------------------------------------


def test_the_jacobian_is_the_derivative_of_the_logits():
    x = torch.randn(5, 2)
    theta = ARCHITECTURE.sample_prior(1, generator(0, "theta"))[0]
    logits, jacobian = ARCHITECTURE.jacobian(x, theta)
    by_autograd = torch.autograd.functional.jacobian(lambda t: ARCHITECTURE.logits(x, t), theta)
    torch.testing.assert_close(logits, ARCHITECTURE.logits(x, theta))
    torch.testing.assert_close(jacobian, by_autograd)


def test_slicing_a_batch_of_eigendecompositions_changes_nothing():
    """A GPU's scratch for a batched eigendecomposition grows with the number of matrices and not
    their size, so large batches are taken in slices. The answer must not depend on the slice."""
    p = torch.rand(40, 7, 3).softmax(-1)
    matrix = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    values, vectors = batched_eigh(matrix, slice_size=len(matrix) * 7)  # one call
    for slice_size in (1, 7, 43):
        sliced_values, sliced_vectors = batched_eigh(matrix, slice_size=slice_size)
        torch.testing.assert_close(sliced_values, values)
        torch.testing.assert_close(sliced_vectors, vectors)
    assert values.shape == (40, 7, 3) and vectors.shape == (40, 7, 3, 3)
    # And it is the square root it is used for: V sqrt(L) V^T squares back to the matrix.
    root = (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.mT
    torch.testing.assert_close(root @ root, matrix)


def test_networks_broadcast_over_leading_dimensions():
    theta = ARCHITECTURE.sample_prior(6, generator(0, "theta")).unflatten(0, (2, 3))
    x = torch.randn(2, 1, 4, 2)
    logits = ARCHITECTURE.logits(x, theta)
    assert logits.shape == (2, 3, 4, 3)
    torch.testing.assert_close(logits[1, 2], ARCHITECTURE.logits(x[1, 0], theta[1, 2]))
    assert ARCHITECTURE.num_parameters == 2 * 8 + 8 + 8 * 8 + 8 + 8 * 3 + 3
    assert ARCHITECTURE.is_weight.sum() == 2 * 8 + 8 * 8 + 8 * 3


def test_the_gaussian_prior_matches_the_moments_of_the_prior():
    draws = ARCHITECTURE.sample_prior(20000, generator(0, "prior"))
    torch.testing.assert_close(draws.std(0), ARCHITECTURE.prior_std, rtol=0.05, atol=0)
    assert draws.mean(0).abs().max() < 0.02


# ----- environments -------------------------------------------------------------------------------


def test_an_environment_is_a_function_of_its_seed_with_a_stream_for_each_role():
    environment, again = sample_environment(SETTING, 3), sample_environment(SETTING, 3)
    assert torch.equal(environment.pool.x, again.pool.x)
    assert torch.equal(environment.train.y, again.train.y)
    assert not torch.equal(environment.theta, sample_environment(SETTING, 4).theta)
    assert environment.train.x.shape == (12, 2) and environment.test.log_p.shape == (200, 3)
    assert environment.pool.y.dtype == torch.int64

    # Drawing more test inputs leaves every other role as it was.
    larger = sample_environment(dataclasses.replace(SETTING, num_test=300), 3)
    assert torch.equal(larger.train.x, environment.train.x)
    assert torch.equal(larger.pool.y, environment.pool.y)
    assert torch.equal(larger.theta, environment.theta)


def test_roles_are_drawn_from_their_own_clusters():
    environment = sample_environment(SETTING, 0)
    assert set(environment.train.cluster.tolist()) <= {0, 1, 2}
    assert set(environment.target.cluster.tolist()) <= {2, 3}
    assert set(environment.pool.cluster.tolist()) == {0, 1, 2, 3}
    # Tight training clusters sit near their centres.
    centres = torch.tensor([[-3.0, -3.0], [-3.0, 3.0], [3.0, -3.0], [3.0, 3.0]])
    offsets = environment.train.x - centres[environment.train.cluster]
    assert offsets.abs().max() < 2.5


def test_classes_are_balanced_over_the_grid_and_the_truth_is_a_distribution():
    environment = sample_environment(SETTING, 1)
    log_probs = class_log_probs(
        ARCHITECTURE, balance_grid(), environment.theta, environment.offset, SETTING.temperature
    )
    torch.testing.assert_close(log_probs.exp().mean(0), torch.full((3,), 1 / 3))
    torch.testing.assert_close(environment.offset.sum(), torch.tensor(0.0), atol=1e-12, rtol=0)
    torch.testing.assert_close(environment.test.log_p.exp().sum(-1), torch.ones(200))

    unbalanced = ARCHITECTURE.logits(balance_grid(), environment.theta)
    skewed = torch.softmax(unbalanced / SETTING.temperature, -1).mean(0)
    assert (skewed - 1 / 3).abs().max() > 0.05
    assert balance_offsets(unbalanced, SETTING.temperature).shape == (3,)


def test_invalid_environments_are_skipped_and_chunks_stack():
    strict = dataclasses.replace(SETTING, min_class_mass=0.25)
    source = valid_environments(strict)
    environments = [next(source) for _ in range(3)]
    seeds = [environment.seed for environment in environments]
    assert seeds == sorted(seeds) and seeds[-1] > 2  # some seeds were left out
    assert all(class_masses(e).min() >= 0.25 for e in environments)
    skipped = set(range(seeds[-1])) - set(seeds)
    assert not any(is_valid(sample_environment(strict, seed), strict) for seed in skipped)

    chunk = stack(environments)
    assert chunk.seeds == tuple(seeds)
    assert chunk.pool.x.shape == (3, 64, 2)
    assert chunk.theta.shape == (3, ARCHITECTURE.num_parameters)
    assert torch.equal(chunk.target.log_p[1], environments[1].target.log_p)


# ----- scoring ------------------------------------------------------------------------------------


def test_labellings_are_enumerated_in_base_k_with_the_first_input_leading():
    labellings = enumerate_labellings(3, 3)
    assert labellings.shape == (27, 3)
    assert labellings[0].tolist() == [0, 0, 0] and labellings[5].tolist() == [0, 1, 2]
    assert len({tuple(row) for row in labellings.tolist()}) == 27

    log_probs = torch.log_softmax(torch.randn(4, 3, 3), -1)
    joint = labelling_log_probs(log_probs, labellings)
    torch.testing.assert_close(joint.exp().sum(-1), torch.ones(4))
    torch.testing.assert_close(
        joint[2, 5], log_probs[2, 0, 0] + log_probs[2, 1, 1] + log_probs[2, 2, 2]
    )


def _coherent_example(num_atoms=5, size=2, num_targets=4):
    """A mixture updated by Bayes' rule: weights, the joint, and the predictives before and after."""
    labellings = enumerate_labellings(size, 3)
    log_w = torch.log_softmax(torch.randn(num_atoms), -1)
    at_batch = torch.log_softmax(torch.randn(num_atoms, size, 3), -1)
    at_targets = torch.log_softmax(torch.randn(num_atoms, num_targets, 3), -1)
    log_joint = mixture_log_joint(log_w, at_batch, labellings)
    posterior = torch.log_softmax(
        log_w.unsqueeze(-1) + labelling_log_probs(at_batch, labellings), dim=0
    )  # (A, Z)
    log_p = torch.logsumexp(log_w[:, None, None] + at_targets, 0)
    log_q = torch.logsumexp(posterior.T[:, :, None, None] + at_targets, 1)
    return log_p, log_q, log_joint


def test_a_bayesian_update_is_coherent_and_its_objectives_agree():
    log_p, log_q, log_joint = _coherent_example()
    torch.testing.assert_close(log_joint.exp().sum(), torch.tensor(1.0))
    scores = score_candidates(log_p, log_q, log_joint)
    assert scores["c"] < 1e-25 and scores["psi"] < 1e-25
    assert scores["v"] > 0 and scores["epig"] > 0
    torch.testing.assert_close(scores["epig"], scores["emi"])
    torch.testing.assert_close(scores["epig_kl"], scores["emi"])


def test_the_objectives_differ_by_what_incoherence_adds():
    log_p, log_q, log_joint = _coherent_example()
    log_q = torch.log_softmax(log_q + 0.3 * torch.randn_like(log_q), -1)  # no longer Bayes' rule
    scores = score_candidates(log_p, log_q, log_joint)
    weights = log_joint.exp()
    q_bar = torch.einsum("z,zmk->mk", weights, log_q.exp())

    torch.testing.assert_close(
        scores["epig"] - scores["emi"], (entropy(log_p) - entropy(q_bar.log())).mean()
    )
    torch.testing.assert_close(
        scores["epig_kl"] - scores["emi"], kl_divergence(q_bar.log(), log_p).mean()
    )
    by_definition = (weights * kl_divergence(log_q, log_p.unsqueeze(0)).mean(-1)).sum()
    torch.testing.assert_close(scores["epig_kl"], by_definition)

    bias = q_bar - log_p.exp()
    spread = (weights[:, None, None] * (log_q.exp() - q_bar).square()).sum()
    torch.testing.assert_close(scores["c"], bias.square().sum())
    torch.testing.assert_close(scores["v"], spread)
    torch.testing.assert_close(scores["psi"], scores["c"] / scores["v"])


def test_two_draws_of_a_random_update_measure_its_incoherence_without_bias():
    # Each labelling's update is one of two predictives at even odds, so every pair of draws can
    # be enumerated and the expected future predictive averages over both.
    log_p, log_q, log_joint = _coherent_example(size=1)
    num_labellings = len(log_joint)
    outcomes = torch.log_softmax(log_q + 0.5 * torch.randn(2, *log_q.shape), -1)  # (2, Z, M, K)
    weights = log_joint.exp()
    q_bar = torch.einsum("z,zmk->mk", weights, outcomes.exp().mean(0))
    exact_c = (q_bar - log_p.exp()).square().sum()
    exact_v = (weights * (outcomes.exp() - q_bar).square().sum((-1, -2)).mean(0)).sum()

    def drawn(which):
        return outcomes[torch.tensor(which), torch.arange(num_labellings)]

    draws = list(itertools.product(range(2), repeat=num_labellings))
    pairs = [
        coherence_of_random_updates(log_p, drawn(first), drawn(again), log_joint)
        for first in draws
        for again in draws
    ]
    torch.testing.assert_close(torch.stack([pair["c"] for pair in pairs]).mean(), exact_c)
    torch.testing.assert_close(torch.stack([pair["v"] for pair in pairs]).mean(), exact_v)

    # One draw alone overstates the displacement by its variance, weighted by sum_z p(z)^2.
    plugin = torch.stack([score_candidates(log_p, drawn(d), log_joint)["c"] for d in draws])
    variance = (outcomes[0].exp() - outcomes[1].exp()).square().sum((-1, -2)) / 4
    torch.testing.assert_close(plugin.mean(), exact_c + (weights.square() * variance).sum())

    # Two equal draws are a deterministic update, and give what `score_candidates` gives.
    same = coherence_of_random_updates(log_p, drawn(draws[3]), drawn(draws[3]), log_joint)
    scores = score_candidates(log_p, drawn(draws[3]), log_joint)
    for key in ("c", "v", "psi"):
        torch.testing.assert_close(same[key], scores[key])


def test_two_draws_shared_by_every_labelling_measure_incoherence_without_bias():
    # A coherent update thrown off by a seed that every labelling shares: at even odds it tilts
    # every post-update predictive one way or the other, so the seed's variance is known exactly and
    # every pair of draws can be enumerated.
    log_p, log_q, log_joint = _coherent_example()
    tilt = torch.randn(log_q.shape[-2:])  # (M, K), the same for every labelling
    noise = 0.1 * torch.randn(2, *log_q.shape)
    seeds = [
        torch.log_softmax(log_q + sign * tilt + noise[i], -1) for i, sign in enumerate((1, -1))
    ]
    weights = log_joint.exp()
    per_seed = [torch.einsum("z,zmk->mk", weights, seed.exp()) for seed in seeds]
    q_bar = (per_seed[0] + per_seed[1]) / 2
    exact_c = (q_bar - log_p.exp()).square().sum()
    exact_v = (
        sum((weights * (seed.exp() - q_bar).square().sum((-1, -2))).sum() for seed in seeds) / 2
    )
    seed_variance = (per_seed[0] - per_seed[1]).square().sum() / 4

    pairs = [
        coherence_of_random_updates(log_p, first, again, log_joint)
        for first in seeds
        for again in seeds
    ]
    torch.testing.assert_close(torch.stack([pair["c"] for pair in pairs]).mean(), exact_c)
    torch.testing.assert_close(torch.stack([pair["v"] for pair in pairs]).mean(), exact_v)

    # One draw counts all of the seed's variance as displacement, and leaves it out of the spread.
    plugin = [score_candidates(log_p, seed, log_joint) for seed in seeds]
    one_c = torch.stack([scores["c"] for scores in plugin]).mean()
    one_v = torch.stack([scores["v"] for scores in plugin]).mean()
    torch.testing.assert_close(one_c, exact_c + seed_variance)
    torch.testing.assert_close(one_v, exact_v - seed_variance)
    assert seed_variance > 1e-3


def test_scores_broadcast_over_environments_and_candidates():
    examples = [_coherent_example() for _ in range(6)]
    log_p, log_q, log_joint = (
        torch.stack([example[i] for example in examples]).unflatten(0, (2, 3)) for i in range(3)
    )
    log_q = torch.log_softmax(log_q + 0.2 * torch.randn_like(log_q), -1)
    scores = score_candidates(log_p, log_q, log_joint)
    alone = score_candidates(log_p[1, 2], log_q[1, 2], log_joint[1, 2])
    for key, value in scores.items():
        assert value.shape == (2, 3)
        torch.testing.assert_close(value[1, 2], alone[key])


def test_a_model_that_never_moves_has_no_ratio():
    log_p = torch.log_softmax(torch.randn(4, 3), -1)
    log_joint = torch.log_softmax(torch.randn(9), -1)
    scores = score_candidates(log_p, log_p.expand(9, -1, -1), log_joint)
    assert scores["v"] < 1e-25 and scores["psi"].isnan()


def test_losses_are_expectations_over_the_true_labels():
    log_p, log_q, log_joint = _coherent_example()
    log_truth = torch.log_softmax(torch.randn(4, 3), -1)
    log_joint_truth = torch.log_softmax(torch.randn(9), -1)
    by_hand = sum(
        log_joint_truth[z].exp() * kl_divergence(log_truth, log_q[z]).mean() for z in range(9)
    )
    torch.testing.assert_close(expected_loss(log_truth, log_q, log_joint_truth), by_hand)
    torch.testing.assert_close(joint_kl(log_joint, log_joint), torch.tensor(0.0))
    assert joint_kl(log_joint_truth, log_joint) > 0


# ----- candidates ---------------------------------------------------------------------------------


def test_candidates_are_distinct_batches_of_the_pool_fixed_by_the_environment():
    environments = [sample_environment(SETTING, seed) for seed in (0, 1)]
    candidates, names = candidates_for(SETTING, stack(environments), environments)
    assert candidates.shape == (2, 10, 3)
    assert candidates.min() >= 0 and candidates.max() < 64
    for batches, rules in zip(candidates.tolist(), names, strict=True):
        assert all(len(set(batch)) == 3 for batch in batches)
        assert len({tuple(sorted(batch)) for batch in batches}) == 10
        assert all(rule.removeprefix("sampled_for_") in {*RULES, "sampled"} for rule in rules)
        assert rules[-2:] == ["sampled", "sampled"]

    # The same whatever they are chunked with.
    alone, _ = candidates_for(SETTING, stack(environments[1:]), environments[1:])
    assert torch.equal(alone[0], candidates[1])

    # Inputs restricted to the target clusters come from them.
    target = names[0].index("coverage_target")
    chosen = environments[0].pool.cluster[candidates[0, target]]
    assert set(chosen.tolist()) <= {2, 3}

    x = gather_inputs(stack(environments).pool.x, candidates)
    assert torch.equal(x[1, 4, 2], environments[1].pool.x[candidates[1, 4, 2]])


# ----- the runner and its results -----------------------------------------------------------------


@pytest.fixture(scope="module")
def run():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    chunks = [run_chunk(SETTING, ARMS, envs, "cpu") for envs in chunks_of(SETTING, 3, 2)]
    torch.set_default_dtype(dtype)
    return merge_chunks(chunks)


def test_a_run_measures_every_arm_in_every_environment(run):
    assert len(run["seeds"]) == 3 and run["candidates"].shape == (3, 10, 3)
    assert run["bayes_entropy"].shape == (3,) and (run["bayes_entropy"] < math.log(3)).all()
    values = run["values"]
    assert values["initial_kl"].shape == (2, 3) and values["psi"].shape == (2, 3, 10)
    assert all(value.isfinite().all() for value in values.values())
    assert (values["loss"] > 0).all() and (values["loss_test"] > 0).all()
    assert (values["joint_kl"] >= 0).all()
    assert (values["update_seconds"] >= 0).all()


def test_the_exact_arm_is_coherent_and_the_approximation_is_not(run):
    arms = ["exact", "laplace_diag"]
    results = Results(arms, {}, run["seeds"], run["rules"], run["bayes_entropy"], run["values"])
    assert results.of("c", "exact").max() < 1e-25
    assert not results.disagreement("exact").any()
    assert not results.disagreement("exact", "epig_kl", "emi").any()
    assert results.regret_gap("exact").abs().max() == 0
    assert results.psi("exact").max() < 1e-20

    assert results.psi("laplace_diag").min() > 1e-6
    gap = results.of("epig", "laplace_diag") - results.of("emi", "laplace_diag")
    assert gap.abs().min() > 1e-8
    # The entropy form can fall below EMI, but the KL form exceeds it by a divergence.
    assert (results.of("epig_kl", "laplace_diag") >= results.of("emi", "laplace_diag")).all()


def test_a_random_update_is_measured_from_two_draws_and_decided_from_one(run):
    dropout = Arm(
        "dropout",
        "dropout",
        dict(rate=0.5, num_atoms=8, weight_decay=0.5, lr=0.02, fit_steps=30, update_lr=0.02,
             update_steps=10),
    )  # fmt: skip
    arms = [*ARMS, dropout]
    chunks = [
        run_chunk(SETTING, arms, envs, "cpu", test_losses=False)
        for envs in chunks_of(SETTING, 3, 2)
    ]
    merged = merge_chunks(chunks)
    values = merged["values"]
    assert "loss_test" not in values

    # Deterministic arms are as they were, and two draws of them are the one they have.
    for key, value in run["values"].items():
        if key != "loss_test" and not key.endswith("_seconds"):
            assert torch.equal(values[key][:2], value), key
    for key in ("c", "v", "psi"):
        assert torch.equal(values[f"{key}_plugin"][:2], values[key][:2])
    assert (values["redraw_seconds"][:2] == 0).all() and (values["redraw_seconds"][2] > 0).all()

    # The random arm decides from its first draw, and measures incoherence from both.
    results = Results([arm.name for arm in arms], {}, merged["seeds"], merged["rules"],
                      merged["bayes_entropy"], values)  # fmt: skip
    assert results.drawn_twice
    assert not torch.equal(results.of("c", "dropout"), results.of("c_plugin", "dropout"))
    assert results.of("v", "dropout").min() > 0
    alone = run_chunk(SETTING, [dropout], [sample_environment(SETTING, run["seeds"][2])], "cpu")
    for key in ("epig", "emi", "loss", "c", "c_plugin", "psi", "psi_plugin"):
        torch.testing.assert_close(alone["values"][key][0, 0], values[key][2, 2])
    torch.testing.assert_close(results.psi("dropout"), results.median("psi", "dropout"))

    head = results.head(2)
    assert head.of("psi", "dropout").shape == (2, 10)
    assert torch.equal(head.of("c", "dropout"), results.of("c", "dropout")[:2])
    assert head.config["num_environments"] == 2


def test_a_re_score_replaces_what_it_measured_and_keeps_the_rest():
    dropout = Arm(
        "dropout",
        "dropout",
        dict(rate=0.5, num_atoms=8, weight_decay=0.5, lr=0.02, fit_steps=30, update_lr=0.02,
             update_steps=10),
    )  # fmt: skip
    arms = [*ARMS, dropout]
    names = [arm.name for arm in arms]
    full = merge_chunks(
        [run_chunk(SETTING, arms, envs, "cpu") for envs in chunks_of(SETTING, 3, 2)]
    )
    # A run from before random updates were drawn twice: one draw's values, and no `_plugin`.
    old = {key: value.clone() for key, value in full["values"].items() if "_plugin" not in key}
    for key in ("c", "v", "psi"):
        old[key][2] = full["values"][f"{key}_plugin"][2]
    baseline = Results(names, {"run": "old"}, full["seeds"], full["rules"],
                       full["bayes_entropy"], old)  # fmt: skip
    (environments,) = chunks_of(SETTING, 2, 2)
    again = run_chunk(SETTING, [dropout], environments, "cpu", test_losses=False)
    rescore = Results(["dropout"], {"run": "again"}, again["seeds"], again["rules"],
                      again["bayes_entropy"], again["values"])  # fmt: skip

    assert not baseline.drawn_twice
    rescored = baseline.rescored(rescore)
    assert rescored.drawn_twice
    assert rescored.seeds == full["seeds"][:2] and rescored.config["num_environments"] == 2
    for key in ("c", "v", "psi", "c_plugin", "psi_plugin", "epig", "emi", "loss"):
        torch.testing.assert_close(rescored.of(key, "dropout"), full["values"][key][2, :2])
    # What the re-score did not measure is kept, and it is what a run of both would have had.
    assert torch.equal(rescored.of("loss_test", "dropout"), full["values"]["loss_test"][2, :2])
    for arm in ("exact", "laplace_diag"):
        assert torch.equal(rescored.of("c_plugin", arm), rescored.of("c", arm))
        assert torch.equal(rescored.of("c", arm), baseline.of("c", arm)[:2])

    with pytest.raises(ValueError, match="first environments"):
        baseline.head(1).rescored(rescore)


def test_results_read_choices_from_scores():
    values = {
        "epig": torch.tensor([[[3.0, 1.0, 2.0], [0.0, 5.0, 1.0]]]),
        "epig_kl": torch.tensor([[[1.0, 3.0, 2.0], [0.0, 5.0, 1.0]]]),
        "emi": torch.tensor([[[1.0, 2.0, 3.0], [0.0, 5.0, 1.0]]]),
        "loss": torch.tensor([[[0.5, 0.4, 0.1], [0.2, 0.3, 0.7]]]),
        "loss_test": torch.tensor([[[0.6, 0.2, 0.3], [0.2, 0.3, 0.7]]]),
        "psi": torch.tensor([[[0.1, 0.2, 9.0], [0.3, float("nan"), 0.5]]]),
    }
    results = Results(["arm"], {}, [0, 1], [[], []], torch.zeros(2), values)
    assert results.chosen("epig", "arm").tolist() == [0, 1]
    assert results.chosen("emi", "arm").tolist() == [2, 1]
    assert results.disagreement("arm").tolist() == [True, False]
    assert results.disagreement("arm", "epig", "epig_kl").tolist() == [True, False]
    torch.testing.assert_close(results.loss("epig", "arm"), torch.tensor([0.5, 0.3]))
    torch.testing.assert_close(results.regret_gap("arm"), torch.tensor([0.4, 0.0]))
    torch.testing.assert_close(results.regret_gap("arm", "epig_kl"), torch.tensor([0.3, 0.0]))
    torch.testing.assert_close(results.regret_gap("arm", split="test"), torch.tensor([0.3, 0.0]))
    # A median over candidates, which one extreme candidate does not move and NaNs do not enter.
    torch.testing.assert_close(results.psi("arm"), torch.tensor([0.2, 0.4]))


def test_the_inctnp_arm_runs_from_a_checkpoint(tmp_path):
    from pc_experiments.inctnp import Architecture as NetworkArchitecture
    from pc_experiments.neural_testbed.models import build_classifier, classifier_payload
    from pc_experiments.outputs import save_run

    architecture = NetworkArchitecture(
        dim_x=2, dim_y=3, embed_dim=16, num_heads=2, head_dim=8, num_layers=1
    )
    network = build_classifier(architecture).float()
    save_run(tmp_path, "inctnp", {"manifest": {}, **classifier_payload(network, architecture)})
    arm = Arm("inctnp", "inctnp", {"checkpoint": str(tmp_path / "inctnp.pt")})
    environments = [sample_environment(SETTING, seed) for seed in (0, 1)]
    values = run_chunk(SETTING, [arm], environments, "cpu")["values"]
    assert values["psi"].shape == (1, 2, 10) and values["loss_test"].isfinite().all()
    assert values["initial_kl"].dtype == torch.float64
    # An untrained network's joint is no mixture, and its update no conditioning.
    assert (values["c"] > 0).all()


# ----- tuning -------------------------------------------------------------------------------------


def test_a_family_shares_tuned_hyperparameters_between_its_arms():
    family = Family("laplace", {"a": 1, "b": 2}, {"x": {"c": 3}, "y": {"c": 4}})
    arms = family.build({"b": 5})
    assert [arm.name for arm in arms] == ["x", "y"] and arms[0].kind == "laplace"
    assert arms[1].hyperparameters == {"a": 1, "b": 5, "c": 4}
    assert family.build()[0].hyperparameters == {"a": 1, "b": 2, "c": 3}
    assert grid_points({"a": (1, 2), "b": ("u", "v", "w")})[4] == {"a": 2, "b": "v"}
    assert grid_points({}) == [{}]


def test_models_are_tuned_at_held_out_inputs_like_the_data_they_have_seen():
    environments = [sample_environment(SETTING, seed) for seed in (0, 1)]
    chunk = stack(environments)
    candidates, _ = candidates_for(SETTING, chunk, environments)
    like_training, around = held_out_inputs(SETTING, chunk, candidates)
    assert like_training.shape == (2, HELD_OUT, 2)
    assert around.shape == (2, 10, 3 * AROUND_BATCH, 2)
    # Around each candidate's own inputs, at the training spread.
    batch = gather_inputs(chunk.pool.x, candidates)
    offsets = around.unflatten(2, (3, AROUND_BATCH)) - batch.unsqueeze(-2)
    assert abs(offsets.std().item() - SETTING.train_std) < 0.05
    # Drawn from the training clusters, whatever else the environment was chunked with.
    alone, _ = held_out_inputs(SETTING, stack(environments[1:]), candidates[1:])
    torch.testing.assert_close(alone[0], like_training[1])
    centres = torch.tensor([[-3.0, -3.0], [-3.0, 3.0], [3.0, -3.0]])
    nearest = torch.cdist(like_training, centres).amin(-1)
    assert nearest.median() < 2 * SETTING.train_std

    fitted, updated = (
        prediction_scores(ARMS[0], SETTING, chunk, candidates, updated) for updated in (False, True)
    )
    assert fitted.shape == (2,) and updated.shape == (2,)
    assert (fitted > 0).all() and (updated > 0).all()


# ----- training the incTNP ------------------------------------------------------------------------


def test_training_tasks_are_shaped_like_the_experiments():
    from pc_experiments.inctnp import Architecture as NetworkArchitecture
    from pc_experiments.neural_testbed.models import build_classifier

    tasks = Tasks(SETTING, num_targets=6, num_validity=256, clump_probability=0.5, clump_std=0.25)
    assert tasks.max_context == 15
    torch.manual_seed(0)
    task = tasks.sample(64)
    kept = len(task["x_context"])
    assert 32 < kept < 64  # invalid environments are left out
    assert task["x_context"].shape == (kept, 15, 2) and task["y_context"].shape == (kept, 15)
    assert task["log_p_target"].shape == (kept, 6, 3) and task["y_target"].max() <= 2
    # Half the batches are clumps about their first input.
    spread = (task["x_context"][:, 12:] - task["x_context"][:, 12:13]).norm(dim=-1).amax(-1)
    assert 0.2 < (spread < 1.0).double().mean() < 0.8

    network = build_classifier(
        NetworkArchitecture(dim_x=2, dim_y=3, embed_dim=16, num_heads=2, head_dim=8, num_layers=1)
    ).float()
    loss = task_loss(network, tasks, task)
    assert loss.isfinite() and abs(loss.item() - math.log(3)) < 0.5
    loss.backward()
    metrics = evaluate(network, tasks, task)
    assert set(metrics) == {"kl", "kl_updated"} and metrics["kl"] > 0
