"""The synthetic Gaussian process experiment: environments, tasks, arms and results."""

import dataclasses

import pytest
import torch

from pc_experiments.gp import GaussianKernelPrior
from pc_experiments.inctnp import Architecture, SequenceTasks, build_network, network_payload
from pc_experiments.outputs import save_run
from pc_experiments.synthetic_gp import (
    Arm,
    Results,
    Setting,
    check_training,
    run_setting,
    run_task,
    sample_environment,
    sample_task,
)

PRIOR = GaussianKernelPrior(lengthscale=0.5, outputscale=1.0, noise_std=0.1)
SETTING = Setting(
    prior=PRIOR,
    truth=PRIOR,
    input_range=(-2.0, 2.0),
    num_pool=40,
    num_context=16,
    num_design=3,
    num_eval=8,
    num_rounds=40,
    estimates_at=(5, 40),
    psi_lengthscale=0.1,
    psi_features=32,
)
MISSPECIFIED = dataclasses.replace(
    SETTING, truth=GaussianKernelPrior(lengthscale=0.15, outputscale=0.8, noise_std=0.1)
)
GP_ARMS = (Arm("exact"), Arm("sparse", 4), Arm("sparse", 12))


def training(length=32, prior=PRIOR):
    return {"tasks": dataclasses.asdict(SequenceTasks(prior, (-2.0, 2.0), length))}


def test_tasks_are_disjoint_subsamples_of_one_function():
    environment = sample_environment(SETTING, 3)
    assert environment.x.shape == (40, 1) and environment.y.shape == (40,)
    task, again, other = (sample_task(SETTING, 3, index) for index in (0, 0, 1))
    assert task.x.shape == (16, 1) and task.x_design.shape == (3, 1)
    assert task.y.shape == (16,) and task.y_eval.shape == (8,)
    assert torch.equal(task.y, again.y) and torch.equal(task.x_eval, again.x_eval)
    assert not torch.equal(task.x, other.x)

    inputs = torch.cat([task.x, task.x_design, task.x_eval])
    assert len(inputs.unique()) == 27
    # Every point is one of the pool's, with the pool's output there.
    pool = dict(zip(environment.x[:, 0].tolist(), environment.y.tolist(), strict=True))
    assert [pool[x] for x in task.x[:, 0].tolist()] == task.y.tolist()
    assert [pool[x] for x in task.x_eval[:, 0].tolist()] == task.y_eval.tolist()
    assert set(other.x[:, 0].tolist()) <= set(pool)


def test_settings_that_differ_in_their_truth_share_their_inputs():
    well, mis = sample_task(SETTING, 2, 5), sample_task(MISSPECIFIED, 2, 5)
    for ours, theirs in [(well.x, mis.x), (well.x_design, mis.x_design), (well.x_eval, mis.x_eval)]:
        assert torch.equal(ours, theirs)
    assert not torch.equal(well.y, mis.y)


def test_network_must_be_trained_for_the_prior_but_not_the_truth():
    check_training(training(), SETTING)
    check_training(training(), MISSPECIFIED)
    with pytest.raises(ValueError, match="not on this setting"):
        check_training(training(prior=dataclasses.replace(PRIOR, lengthscale=1.0)), SETTING)
    with pytest.raises(ValueError, match="needs 19"):
        check_training(training(length=19), SETTING)
    check_training(training(length=20), SETTING)


def test_what_is_recorded_for_each_arm():
    results = run_task(SETTING, GP_ARMS, None, (0, 0))
    assert results["psi"].shape == (3,) and results["log_wealth"].shape == (3, 40)
    # The exact arm is the truth here, and coherent; the sparse arms are neither.
    assert results["kl"][0].abs() < 1e-12 and (results["kl"][1:] > 0).all()
    # After the block, the exact arm is still the truth and the sparse arms still are not.
    assert results["kl_updated"][0].abs() < 1e-12 and (results["kl_updated"][1:] > 0).all()
    assert results["excess_nll"][0].abs() < 1e-12
    assert torch.allclose(results["excess_nll"], results["nll"] - results["nll"][0])
    assert results["c_exact"][0].abs() < 1e-12 and (results["c_exact"][1:] > 0).all()


def test_the_estimates_kept_along_the_way_are_those_of_the_rounds_so_far():
    full = run_task(SETTING, GP_ARMS, None, (0, 0))
    assert full["psi_at"].shape == (len(GP_ARMS), 2)
    torch.testing.assert_close(full["psi_at"][:, -1], full["psi"])
    torch.testing.assert_close(full["psi_se_at"][:, -1], full["psi_se"])
    # The first five rounds are the same rounds whether or not more follow.
    short = run_task(dataclasses.replace(SETTING, num_rounds=5, estimates_at=(5,)), GP_ARMS, None,
                     (0, 0))  # fmt: skip
    torch.testing.assert_close(full["psi_at"][:, 0], short["psi"])
    torch.testing.assert_close(full["psi_se_at"][:, 0], short["psi_se"])


def test_an_arm_does_not_depend_on_which_others_are_run():
    together = run_task(SETTING, GP_ARMS, None, (0, 0))
    alone = run_task(SETTING, GP_ARMS[1:2], None, (0, 0))
    for key in ("psi", "nll", "log_wealth", "c_exact"):
        assert torch.equal(alone[key][0], together[key][1])


@pytest.mark.slow
def test_tasks_land_at_their_environment_and_subsample():
    results = run_setting(
        SETTING, GP_ARMS[:2], None, num_environments=2, num_subsamples=3, workers=2
    )
    assert results["psi"].shape == (2, 2, 3) and results["log_wealth"].shape == (2, 2, 3, 40)
    for index in [(0, 0), (0, 2), (1, 1)]:
        task = run_task(SETTING, GP_ARMS[:2], None, index)
        for key in ("psi", "kl", "c_exact"):
            assert torch.equal(results[key][:, index[0], index[1]], task[key])


def test_misspecification_moves_accuracy_but_not_the_incoherence_of_gaussian_arms():
    well = run_task(SETTING, GP_ARMS, None, (1, 2))
    mis = run_task(MISSPECIFIED, GP_ARMS, None, (1, 2))
    # Incoherence is a property of a model's own predictive and update. With inducing inputs
    # chosen from the inputs alone, neither looks at the outputs beyond a shared shift.
    for key in ("c_exact", "v_exact"):
        assert torch.allclose(well[key], mis[key], rtol=1e-9, atol=1e-15)
    assert torch.allclose(well["psi"], mis["psi"], rtol=1e-6, atol=1e-9)
    # The exact arm is the truth when well specified, and merely coherent when not.
    assert well["kl"][0].abs() < 1e-12
    assert mis["kl"][0] > 1e-3 and mis["c_exact"][0].abs() < 1e-12


def test_the_inctnp_arm_runs_from_a_checkpoint(tmp_path):
    architecture = Architecture(embed_dim=32, num_heads=4, head_dim=8, num_layers=2)
    network = build_network(architecture).float()
    payload = {"manifest": {}, **network_payload(network, architecture), **training()}
    save_run(tmp_path, "inctnp", payload)
    results = run_task(SETTING, (Arm("inctnp"),), tmp_path / "inctnp.pt", (0, 0))
    assert results["psi"].isfinite().all() and results["kl"] > 0
    assert results["c_exact"].isnan().all()
    with pytest.raises(FileNotFoundError):
        run_task(SETTING, (Arm("inctnp"),), tmp_path / "missing.pt", (0, 0))


def test_results_accessors():
    arms = [Arm("sparse", 8), Arm("exact"), Arm("sparse", 4)]
    # (arms, environments, subsamples, rounds): two environments of one subsample each.
    log_wealth = torch.tensor(
        [
            [[[0.0, 4.0, 1.0]], [[0.0, 0.0, 0.0]]],  # rejects once, at round 2, and stays rejected
            [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
            [[[5.0, 0.0, 0.0]], [[0.0, 0.0, 3.5]]],
        ]
    )
    results = Results(arms, {}, {}, {"log_wealth": log_wealth, "kl": torch.ones(3, 2, 1)})
    assert [arm.num_inducing for arm in results.rungs("sparse")] == [4, 8]
    assert results.of("log_wealth", Arm("exact")).shape == (2, 1, 3)
    assert results.along_ladder("kl").shape == (2, 1, 2)
    assert results.rejection_round(arms[2], alpha=0.05).tolist() == [[1.0], [3.0]]
    assert results.rejection_round(arms[1], alpha=0.05).isnan().all()
    assert results.rejection_rate(arms[0], alpha=0.05).tolist() == [0.0, 0.5, 0.5]
    assert results.rejection_rate(arms[2], alpha=0.05).tolist() == [0.5, 0.5, 1.0]
    assert Arm("sparse", 8).name == "sparse_8" and Arm("exact").name == "exact"
    assert Arm("reselected", 32).name == "reselected_32"
