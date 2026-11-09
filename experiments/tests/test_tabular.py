"""The tabular experiment: datasets and splits, predictives, the runner and its results."""

import dataclasses
import json
import math
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal

from pc_experiments.gp import ExactGP
from pc_experiments.tabular import (
    Arm,
    Results,
    Setting,
    Sizes,
    cross_validate,
    fit_arm,
    make_task,
    result_path,
    run_arm,
    run_jobs,
    score,
)
from pc_experiments.tabular import experiment as experiment_module
from pc_experiments.tabular.datasets import Dataset, encode
from pc_experiments.tabular.predictives import Bars, embed
from pc_experiments.tabular.refitted import Autoregressive
from pc_experiments.tabular.tabpfn import ROUNDS_PER_CALL, TabPFN
from predictive_coherence import DiracKernel, RFFKernel

SIZES = Sizes(
    train_share=0.5, train_cap=40, test_share=0.2, test_cap=10, pool_cap=30, block_share=0.25
)
LENGTHSCALES: tuple[tuple[float, ...], tuple[float, ...]] = ((1.0, 0.5, 2.0), (1.0,))


def setting(num_rounds: int, num_features: int = 8) -> Setting:
    return Setting(SIZES, num_rounds, num_features, *LENGTHSCALES)


def regression_dataset(num_rows=120, num_inputs=3):
    x = torch.randn(num_rows, num_inputs)
    y = 5 + 3 * x[:, 0] - 2 * x[:, 1] + 0.3 * torch.randn(num_rows)
    return Dataset("synthetic", x, y, 0)


def probabilities(model, x):
    predictive = model.predict(x)
    assert isinstance(predictive, Categorical)
    return predictive.probs


def classification_dataset(num_rows=160, num_classes=3):
    x = torch.randn(num_rows, 2)
    scores = torch.stack([2 * x[:, 0], 2 * x[:, 1], -x.sum(-1)], -1)[:, :num_classes]
    return Dataset("labels", x, Categorical(logits=scores).sample(), num_classes)


# ----- datasets -----------------------------------------------------------------------------------


def test_encode_drops_incomplete_rows_and_one_hot_encodes():
    inputs = pd.DataFrame(
        {
            "size": [1.0, 2.0, np.nan, 4.0],
            "colour": pd.Categorical(["red", "blue", "red", "green"]),
        }
    )
    dataset = encode("toy", inputs, pd.Series(["no", "yes", "yes", "no"]), classify=True)
    assert dataset.x.shape == (3, 4)  # a column for the size and one for each colour
    assert dataset.x[:, 1:].sum(-1).tolist() == [1, 1, 1]
    assert dataset.y.tolist() == [0, 1, 0] and dataset.num_classes == 2

    sizes = inputs.drop(columns="colour")
    real = encode("toy", sizes, pd.Series([1.5, np.nan, 2.5, 3.5]), classify=False)
    assert real.y.tolist() == [1.5, 3.5] and real.num_classes == 0


def test_split_sizes_follow_shares_until_a_cap_binds():
    task = make_task(regression_dataset(60), SIZES, 0)
    assert (len(task.x), len(task.x_test), len(task.x_pool), task.block_size) == (30, 10, 20, 7)
    capped = make_task(regression_dataset(400), SIZES, 0)
    assert (len(capped.x), len(capped.x_test), len(capped.x_pool)) == (40, 10, 30)
    with pytest.raises(ValueError, match="a block of 30 from a pool of 20"):
        make_task(regression_dataset(60), dataclasses.replace(SIZES, block_share=1.0), 0)


def test_split_is_disjoint_seeded_and_standardised_on_the_training_set():
    dataset = regression_dataset()
    task, again, other = (make_task(dataset, SIZES, seed) for seed in (0, 0, 1))
    assert torch.equal(task.x, again.x) and not torch.equal(task.x, other.x)
    rows = torch.cat([task.x, task.x_test, task.x_pool])
    assert len(rows.unique(dim=0)) == len(rows)
    torch.testing.assert_close(task.x.mean(0), torch.zeros(3))
    torch.testing.assert_close(task.x.std(0), torch.ones(3))
    torch.testing.assert_close(task.y.std(), torch.tensor(1.0))
    assert task.y_test.dtype == torch.float64


def test_a_column_constant_in_training_stays_finite():
    dataset = regression_dataset()
    x = torch.cat([dataset.x, torch.zeros(len(dataset.x), 1)], -1)
    task = make_task(dataclasses.replace(dataset, x=x), SIZES, 0)
    assert task.x.isfinite().all() and (task.x[:, -1] == 0).all()


def test_labels_are_left_as_they_are():
    task = make_task(classification_dataset(), SIZES, 0)
    assert task.y.dtype == torch.int64 and task.num_classes == 3


def test_blocks_are_drawn_from_the_pool_without_replacement_and_shared_by_arms():
    task = make_task(regression_dataset(), SIZES, 0)
    blocks = task.blocks(5)
    assert blocks.shape == (5, task.block_size, 3)
    assert torch.equal(blocks, task.blocks(5))
    pool = {tuple(row.tolist()) for row in task.x_pool}
    for block in blocks:
        rows = [tuple(row.tolist()) for row in block]
        assert len(set(rows)) == len(rows) and set(rows) <= pool
    assert not torch.equal(blocks[0], blocks[1])


# ----- predictives --------------------------------------------------------------------------------


def test_bars_embed_as_their_samples_do():
    borders = torch.tensor([-2.0, -0.5, -0.5, 0.0, 0.4, 3.0])  # one bar of zero width
    logits = torch.tensor([[0.0, -math.inf, 1.0, 0.5, -1.0], [2.0, -math.inf, 0.0, 0.0, 1.0]])
    bars = Bars(logits, borders)
    kernel = RFFKernel(lengthscale=0.7, num_features=16)
    samples = bars.sample((200_000,))
    assert samples.shape == (200_000, 2)
    torch.testing.assert_close(
        embed(kernel, bars), kernel.features(samples).mean(0), atol=5e-3, rtol=0
    )
    torch.testing.assert_close(bars.mean, samples.mean(0), atol=1e-2, rtol=0)


def test_bar_density_integrates_to_one():
    bars = Bars(torch.tensor([0.3, -1.0, 2.0]), torch.tensor([0.0, 1.0, 1.5, 4.0]))
    grid = torch.linspace(0, 4, 400_001)[:-1] + 5e-6
    torch.testing.assert_close(
        bars.log_prob(grid).exp().mean() * 4, torch.tensor(1.0), atol=1e-4, rtol=0
    )


def test_a_mixture_embeds_as_the_weighted_mean_of_its_components():
    kernel = RFFKernel(lengthscale=0.7, num_features=16)
    weights = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
    components = Normal(torch.tensor([[0.0, 2.0], [-1.0, 1.0]]), torch.tensor([0.5, 1.0]))
    mixture = MixtureSameFamily(Categorical(probs=weights), components)
    embedded = kernel.embed(components)  # (2, 2, dim)
    expected = weights[:, :1] * embedded[:, 0] + weights[:, 1:] * embedded[:, 1]
    torch.testing.assert_close(embed(kernel, mixture), expected)
    samples = mixture.sample((200_000,))
    torch.testing.assert_close(
        embed(kernel, mixture), kernel.features(samples).mean(0), atol=5e-3, rtol=0
    )


def test_other_predictives_are_embedded_by_the_kernel():
    kernel = RFFKernel(lengthscale=1.0, num_features=8)
    normal = Normal(torch.zeros(3), torch.ones(3))
    assert torch.equal(embed(kernel, normal), kernel.embed(normal))


# ----- the runner ---------------------------------------------------------------------------------

REGRESSION_OPTIONS = {
    "kernel_ridge": {"alpha": 0.1, "gamma": 0.3},
    "neighbours": {"n_neighbors": 3},
    "catboost": {"iterations": 20, "depth": 2},
}
CLASSIFICATION_OPTIONS = {
    "logistic": {"prior_sd": 1.0},
    "neighbours": {"n_neighbors": 5},
    "catboost": {"iterations": 20, "depth": 2},
}
REGRESSION_ARMS = [
    Arm("gp"),
    Arm("sparse_gp"),
    Arm("gp_refit"),
    Arm("bart"),
    Arm("xbart"),
    *(Arm(kind) for kind in REGRESSION_OPTIONS),
    Arm("kernel_ridge", autoregressive=True),
]
CLASSIFICATION_ARMS = [
    Arm("logistic_exact"),
    Arm("logistic_laplace"),
    Arm("logistic_diagonal"),
    *(Arm(kind) for kind in CLASSIFICATION_OPTIONS),
    Arm("logistic", autoregressive=True),
]


@pytest.mark.parametrize("arm", REGRESSION_ARMS, ids=lambda arm: arm.name)
def test_every_regression_arm_predicts_draws_and_updates(arm):
    task = make_task(regression_dataset(), SIZES, 0)
    model = fit_arm(arm, task, REGRESSION_OPTIONS.get(arm.kind, {}))
    before = model.predict(task.x_test)
    assert before.mean.shape == (10,) and (before.stddev > 0).all()
    block = task.x_pool[: task.block_size]
    outputs = model.joint_sample(block)
    assert outputs.shape == (task.block_size,)
    updated = model.update(block, outputs + 3)
    assert not torch.allclose(updated.predict(task.x_test).mean, before.mean)
    torch.testing.assert_close(model.predict(task.x_test).mean, before.mean)  # left as it was


@pytest.mark.parametrize("arm", CLASSIFICATION_ARMS, ids=lambda arm: arm.name)
def test_every_classification_arm_predicts_draws_and_updates(arm):
    task = make_task(classification_dataset(), SIZES, 0)
    model = fit_arm(arm, task, CLASSIFICATION_OPTIONS.get(arm.tuned_as, {}))
    before = probabilities(model, task.x_test)
    assert before.shape == (10, 3)
    torch.testing.assert_close(before.sum(-1), torch.ones(10))
    block = task.x_pool[: task.block_size]
    labels = model.joint_sample(block)
    assert labels.shape == (task.block_size,) and labels.dtype == torch.int64
    updated = model.update(block, torch.zeros_like(labels))
    assert probabilities(updated, task.x_test)[:, 0].mean() > before[:, 0].mean()
    torch.testing.assert_close(probabilities(model, task.x_test), before)


def test_an_arm_unknown_for_the_task_is_refused():
    with pytest.raises(ValueError, match="no arm of kind 'logistic'"):
        fit_arm(Arm("logistic"), make_task(regression_dataset(), SIZES, 0), {})


def test_arm_names_and_whose_tuning_they_use():
    assert Arm("sparse_gp").name == "sparse_gp" and Arm("gp").name == "gp"
    assert Arm("logistic_laplace").tuned_as == "logistic" and Arm("bart").tuned_as == "bart"
    rolled = Arm("neighbours", autoregressive=True)
    assert rolled.name == "neighbours_autoregressive" and rolled.tuned_as == "neighbours"


def test_an_autoregressive_joint_is_the_exact_joint_of_an_exact_model():
    task = make_task(regression_dataset(), SIZES, 0)
    model = fit_arm(Arm("gp", autoregressive=True), task, {})
    assert isinstance(model, Autoregressive) and isinstance(model.model, ExactGP)
    # Close together and far from the data, where the function's uncertainty correlates them.
    x = 20 * task.x_pool[:1] + 0.1 * task.x_pool[1:4]
    mean, covariance = model.model.joint_predict(x)
    draws = torch.stack([model.joint_sample(x) for _ in range(4000)])
    scale = covariance.diagonal().sqrt()
    torch.testing.assert_close(draws.mean(0) / scale, mean / scale, atol=0.1, rtol=0)
    correlation = covariance / scale.outer(scale)
    empirical = torch.cov(draws.T) / scale.outer(scale)
    torch.testing.assert_close(empirical, correlation, atol=0.1, rtol=0)
    assert (correlation - torch.eye(3)).abs().max() > 0.3  # the check has an off-diagonal to find


@pytest.mark.parametrize("kind", ["kernel_ridge", "neighbours", "catboost"])
def test_an_autoregressive_arm_differs_from_its_twin_in_the_joint_alone(kind):
    task = make_task(regression_dataset(), SIZES, 0)
    options = REGRESSION_OPTIONS.get(kind, {})
    torch.manual_seed(1)
    twin = fit_arm(Arm(kind), task, options)
    torch.manual_seed(1)
    rolled = fit_arm(Arm(kind, autoregressive=True), task, options)
    torch.testing.assert_close(rolled.predict(task.x_test).mean, twin.predict(task.x_test).mean)
    one = task.x_pool[:1]
    torch.manual_seed(2)
    single = twin.joint_sample(one)
    torch.manual_seed(2)
    torch.testing.assert_close(rolled.joint_sample(one), single)  # a block of one is the same
    block, outputs = task.x_pool[:4], torch.randn(4)
    torch.manual_seed(3)
    expected = twin.update(block, outputs).predict(task.x_test).mean
    torch.manual_seed(3)
    updated = rolled.update(block, outputs)
    assert isinstance(updated, Autoregressive)
    torch.testing.assert_close(updated.predict(task.x_test).mean, expected)


@pytest.mark.parametrize("classify", [False, True], ids=["regression", "classification"])
def test_catboost_takes_in_a_block_with_no_randomness_of_its_own(classify):
    dataset = classification_dataset() if classify else regression_dataset()
    options = (CLASSIFICATION_OPTIONS if classify else REGRESSION_OPTIONS)["catboost"]
    task = make_task(dataset, SIZES, 0)
    block = task.x_pool[: task.block_size]
    outputs = torch.zeros(task.block_size, dtype=torch.int64 if classify else torch.float64)
    predicted = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        updated = fit_arm(Arm("catboost"), task, options).update(block, outputs)
        x = task.x_test
        predicted.append(probabilities(updated, x) if classify else updated.predict(x).mean)
    torch.testing.assert_close(predicted[0], predicted[1], rtol=0, atol=0)


def test_an_autoregressive_plug_in_correlates_a_block():
    task = make_task(regression_dataset(), SIZES, 0)
    model = fit_arm(Arm("neighbours", autoregressive=True), task, {"n_neighbors": 1})
    assert isinstance(model, Autoregressive)
    x = task.x_pool[:1].repeat(2, 1)  # the same input twice: the second draw sees the first
    draws = torch.stack([model.joint_sample(x) for _ in range(2000)])
    assert torch.corrcoef(draws.T)[0, 1] > 0.3
    independent = torch.stack([model.model.joint_sample(x) for _ in range(2000)])
    assert torch.corrcoef(independent.T)[0, 1].abs() < 0.1


def test_score_is_log_loss_with_rmse_or_error_rate():
    real = score(Normal(torch.tensor([0.0, 1.0]), torch.ones(2)), torch.tensor([0.0, 3.0]))
    torch.testing.assert_close(real["error"], torch.tensor(math.sqrt(2.0)))
    torch.testing.assert_close(real["nll"], torch.tensor(0.5 * math.log(2 * math.pi) + 1.0))

    probs = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.2, 0.8]])
    labels = score(Categorical(probs=probs), torch.tensor([0, 0]))
    torch.testing.assert_close(labels["error"], torch.tensor(0.5))
    assert labels["nll"].isfinite() and labels["nll"] > 10  # an impossible label costs a lot


def test_cross_validation_prefers_the_better_hyperparameters():
    task = make_task(regression_dataset(), SIZES, 0)
    good = cross_validate(Arm("kernel_ridge"), task, {"alpha": 0.1, "gamma": 0.3})
    bad = cross_validate(Arm("kernel_ridge"), task, {"alpha": 1e4, "gamma": 0.3})
    assert good["nll"] < bad["nll"] and good["error"] < bad["error"]
    assert good == cross_validate(Arm("kernel_ridge"), task, {"alpha": 0.1, "gamma": 0.3})


def test_setting_gives_a_kernel_per_lengthscale_or_one_for_labels():
    real = Setting(
        SIZES, 10, num_features=4, lengthscale_multiples=(1.0, 3.0), fixed_lengthscales=(0.5,)
    )
    kernels = real.kernels(make_task(regression_dataset(), SIZES, 0), noise_scale=0.2)
    expected = [RFFKernel(lengthscale, num_features=4) for lengthscale in (0.2, 0.6, 0.5)]
    assert len(kernels) == 3
    for kernel, wanted in zip(kernels, expected, strict=True):
        assert isinstance(kernel, RFFKernel)
        torch.testing.assert_close(kernel.frequencies, wanted.frequencies)
    labels = real.kernels(make_task(classification_dataset(), SIZES, 0), None)
    assert len(labels) == 1 and labels[0].dim == 3


def test_coherent_arms_have_no_incoherence_and_a_plug_in_neighbour_has():
    task = make_task(regression_dataset(), SIZES, 0)
    kernels = setting(0, num_features=32).kernels(task, noise_scale=0.3)
    values = run_arm(Arm("gp"), task, {}, kernels, num_rounds=150)
    assert values["c"].shape == values["v"].shape == (4, 10)
    assert values["log_wealth"].shape == (150,)
    assert (values["psi"] < 3 * values["psi_se"] + 1e-3).all()
    assert values["log_wealth"].max() < -math.log(0.05)
    neighbours = run_arm(Arm("neighbours"), task, {"n_neighbors": 3}, kernels, num_rounds=150)
    assert neighbours["psi"][0] > 5 * neighbours["psi_se"][0]


def test_exact_logistic_is_coherent_and_the_plug_in_neighbours_are_not():
    task = make_task(classification_dataset(), SIZES, 0)
    kernels = [DiracKernel(3)]
    exact = run_arm(Arm("logistic_exact"), task, {"prior_sd": 1.0}, kernels, num_rounds=150)
    assert exact["psi"].shape == (1,) and exact["psi"] < 3 * exact["psi_se"] + 1e-3
    neighbours = run_arm(Arm("neighbours"), task, {"n_neighbors": 5}, kernels, num_rounds=150)
    assert neighbours["psi"] > 5 * neighbours["psi_se"]


def test_a_run_is_the_same_whatever_ran_before_it():
    task = make_task(regression_dataset(), SIZES, 0)
    kernels = setting(0).kernels(task, noise_scale=0.3)
    first = run_arm(Arm("bart"), task, {}, kernels, num_rounds=5)
    torch.manual_seed(123)
    second = run_arm(Arm("bart"), task, {}, kernels, num_rounds=5)
    torch.testing.assert_close(first["psi"], second["psi"])


def test_jobs_are_saved_one_by_one_skipped_once_done_and_read_back(tmp_path, monkeypatch):
    dataset = regression_dataset()
    monkeypatch.setattr(experiment_module, "load", lambda name: dataset)
    tuning, results = tmp_path / "tuning", tmp_path / "results"
    tuning.mkdir()
    tuned = {"options": {"kernel_ridge": {"alpha": 0.1, "gamma": 0.3}}, "noise_scale": 0.3}
    (tuning / "synthetic.json").write_text(json.dumps(tuned))
    jobs = [("synthetic", seed, arm) for seed in (0, 1) for arm in (Arm("kernel_ridge"), Arm("gp"))]

    run_jobs(setting(20), tuning, results, jobs)
    saved = result_path(results, jobs[0])
    assert saved == results / "synthetic" / "seed0" / "kernel_ridge.pt" and saved.exists()
    assert json.loads(saved.with_suffix(".json").read_text())["config"]["num_rounds"] == 20
    written = saved.stat().st_mtime_ns
    run_jobs(setting(20), tuning, results, jobs)
    assert saved.stat().st_mtime_ns == written
    run_jobs(setting(20), tuning, results, jobs[:1], overwrite=True)
    assert saved.stat().st_mtime_ns > written

    read = Results.load(results)
    assert read.datasets() == ["synthetic"]
    assert sorted(read.arms("synthetic")) == ["gp", "kernel_ridge"]
    assert not Results.load(results, ["other"]).values  # a dataset left out is not read
    shutil.copytree(results / "synthetic", results / "another")
    assert Results.load(results).datasets() == ["another", "synthetic"]
    assert Results.load(results, ["synthetic", "another"]).datasets() == ["synthetic", "another"]
    assert read.of("psi", "synthetic", "gp").shape == (2, 4)
    assert read.of("c", "synthetic", "kernel_ridge").shape == (2, 4, 10)
    assert read.rejection_round("synthetic", "gp", 0.05).isnan().all()
    # What was saved for a split is what running that arm on that split gives.
    task = make_task(dataset, SIZES, 1)
    expected = run_arm(Arm("gp"), task, {}, setting(20).kernels(task, 0.3), 20)
    torch.testing.assert_close(read.of("psi", "synthetic", "gp")[1], expected["psi"])
    mean, se = read.psi("synthetic", "gp")
    assert mean == pytest.approx(read.of("psi", "synthetic", "gp")[:, 0].mean().item()) and se > 0

    with pytest.raises(FileNotFoundError):
        run_jobs(setting(20), tuning, results, [("untuned", 0, Arm("gp"))])


def test_a_job_that_fails_costs_only_itself(tmp_path, monkeypatch, capsys):
    """The jobs are independent and each is saved on its own, so one arm that cannot be fitted
    must not take the rest of the run with it. Running again retries what is missing."""
    dataset = regression_dataset()
    monkeypatch.setattr(experiment_module, "load", lambda name: dataset)
    tuning, results = tmp_path / "tuning", tmp_path / "results"
    tuning.mkdir()
    tuned = {"options": {"kernel_ridge": {"alpha": 0.1, "gamma": 0.3}}, "noise_scale": 0.3}
    (tuning / "synthetic.json").write_text(json.dumps(tuned))
    jobs = [("synthetic", 0, Arm(kind)) for kind in ("gp", "kernel_ridge", "gp_refit")]

    fit = experiment_module.fit_arm

    def fail_on_kernel_ridge(arm, task, options):
        if arm.kind == "kernel_ridge":
            raise RuntimeError("no fit for you")
        return fit(arm, task, options)

    monkeypatch.setattr(experiment_module, "fit_arm", fail_on_kernel_ridge)
    run_jobs(setting(20), tuning, results, jobs)
    printed = capsys.readouterr().out
    assert "FAILED, RuntimeError: no fit for you" in printed and "1 of 3 failed" in printed
    # The others ran and were saved; the one that failed was not.
    assert result_path(results, jobs[0]).exists() and result_path(results, jobs[2]).exists()
    assert not result_path(results, jobs[1]).exists()

    # With the fit working again, running the same jobs fills in only the one that is missing.
    monkeypatch.setattr(experiment_module, "fit_arm", fit)
    done = result_path(results, jobs[0]).stat().st_mtime_ns
    run_jobs(setting(20), tuning, results, jobs)
    assert result_path(results, jobs[1]).exists()
    assert result_path(results, jobs[0]).stat().st_mtime_ns == done


def test_a_result_from_another_setting_or_tuning_is_run_again(tmp_path, monkeypatch):
    dataset = regression_dataset()
    monkeypatch.setattr(experiment_module, "load", lambda name: dataset)
    tuning, results = tmp_path / "tuning", tmp_path / "results"
    tuning.mkdir()
    (tuning / "synthetic.json").write_text(json.dumps({"options": {}, "noise_scale": 0.3}))
    job = ("synthetic", 0, Arm("gp"))

    run_jobs(setting(20), tuning, results, [job])
    written = result_path(results, job).stat().st_mtime_ns
    run_jobs(setting(20), tuning, results, [job])
    assert result_path(results, job).stat().st_mtime_ns == written  # the same setting: left alone
    run_jobs(setting(30), tuning, results, [job])
    assert result_path(results, job).stat().st_mtime_ns > written
    assert Results.load(results).of("log_wealth", "synthetic", "gp").shape == (1, 30)

    # Tuned again, to another noise scale: the kernel has changed, so the result is redone.
    written = result_path(results, job).stat().st_mtime_ns
    (tuning / "synthetic.json").write_text(json.dumps({"options": {}, "noise_scale": 0.4}))
    run_jobs(setting(30), tuning, results, [job])
    assert result_path(results, job).stat().st_mtime_ns > written
    saved = json.loads(result_path(results, job).with_suffix(".json").read_text())["config"]
    assert saved["noise_scale"] == 0.4 and saved["options"] == {}


def test_the_first_rejection_is_found_and_a_single_split_reports_its_own_error():
    wealth = torch.tensor([0.0, 1.0, 3.5, 2.0, 4.0])
    values = {"log_wealth": wealth, "psi": torch.tensor([0.2]), "psi_se": torch.tensor([0.05])}
    results = Results({("d", "a"): {0: values}})
    assert results.rejection_round("d", "a", 0.05).tolist() == [3.0]
    assert results.psi("d", "a") == pytest.approx((0.2, 0.05))


# ----- TabPFN's bookkeeping, with a stand-in for the network --------------------------------------


class _Echo:
    """A stand-in for the network that records the size of every context it is given."""

    def __init__(self):
        self.contexts = []

    def _seen(self, contexts, outputs, inputs):
        # The package fuses the contexts into one forward pass, so it takes only sets of them
        # that share a shape, and a classifier takes only sets that share their classes.
        assert len({c.shape for c in contexts}) == 1 and len({x.shape for x in inputs}) == 1
        assert all(len(c) == len(o) for c, o in zip(contexts, outputs, strict=True))
        self.contexts.append([len(c) for c in contexts])
        return [len(x) for x in inputs]

    def predict_batched(self, contexts, outputs, inputs, output_type):
        assert output_type == "full"
        borders = torch.linspace(-4, 4, 9)
        logits = torch.tensor([1.0, 2.0, 0.0, -1.0, 0.5, 0.0, 1.5, -0.5])
        return [
            {"logits": logits.expand(m, 8), "criterion": _Borders(borders)}
            for m in self._seen(contexts, outputs, inputs)
        ]

    def predict_proba_batched(self, contexts, outputs, inputs):
        assert len({tuple(np.unique(o)) for o in outputs}) == 1  # one class set for the batch
        return [
            np.full((m, len(np.unique(o))), 1 / len(np.unique(o)))
            for o, m in zip(outputs, self._seen(contexts, outputs, inputs), strict=True)
        ]


@dataclasses.dataclass
class _Borders:
    borders: torch.Tensor


def test_tabpfn_rolls_each_block_out_one_input_at_a_time(monkeypatch):
    echo, asked = _Echo(), []

    def network(classify, preprocessing):
        asked.append((classify, preprocessing))
        return echo

    monkeypatch.setattr("pc_experiments.tabular.tabpfn._network", network)
    task = make_task(regression_dataset(), SIZES, 0)
    model = fit_arm(Arm("tabpfn_preprocessing"), task, {})
    network_only = fit_arm(Arm("tabpfn"), task, {})
    assert isinstance(model, TabPFN) and model.preprocessing
    assert isinstance(network_only, TabPFN) and not network_only.preprocessing

    rounds, n, k = 40, len(task.x), task.block_size  # more rounds than one call takes
    predictives = list(model.updated(task.blocks(rounds), task.x_test))
    assert len(predictives) == rounds and asked[0] == (False, True)
    assert all(isinstance(p, Bars) and p.batch_shape == (10,) for p in predictives)
    # Each block is rolled out one input at a time, then the whole block is in the context. The
    # rounds are scored in chunks, so a chunk's contexts all have the same size.
    chunks = [ROUNDS_PER_CALL, rounds - ROUNDS_PER_CALL]
    sizes = [[n + step] * chunk for chunk in chunks for step in [*range(k), k]]
    assert echo.contexts == sizes

    current = model.predict(task.x_test)
    assert isinstance(current, Bars) and current.probs.shape == (10, 8)
    grown = model.update(task.blocks(1)[0], torch.zeros(k))
    assert len(grown.x) == n + k and len(model.x) == n


def test_tabpfn_gives_a_class_missing_from_its_context_no_probability(monkeypatch):
    echo = _Echo()
    monkeypatch.setattr("pc_experiments.tabular.tabpfn._network", lambda *_: echo)
    task = make_task(classification_dataset(), SIZES, 0)
    seen = task.y != 1
    model = TabPFN(task.x[seen], task.y[seen], num_classes=3, preprocessing=False)
    probs = probabilities(model, task.x_test)
    torch.testing.assert_close(probs, torch.tensor([0.5, 0.0, 0.5]).expand(10, 3))
    labels = [p.sample() for p in model.updated(task.blocks(3), task.x_test)]
    assert all((draw != 1).all() for draw in labels)
