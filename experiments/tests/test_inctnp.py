"""The incTNP network's cache, and the predictive model built on it."""

import copy
import dataclasses
from itertools import islice

import pytest
import torch

from pc_experiments.gp import GaussianKernelPrior
from pc_experiments.inctnp import (
    Architecture,
    IncTNPModel,
    Optimisation,
    SequenceTasks,
    build_network,
    convergence,
    evaluate,
    load_network,
    network_payload,
    train,
)
from pc_experiments.inctnp.launch import checkpoint_paths, training_parser
from pc_experiments.inctnp.training import split_parameters
from pc_experiments.outputs import save_run
from predictive_coherence import CoherenceEstimator, RFFKernel, SequentialTest, iter_rounds

SMALL = Architecture(embed_dim=32, num_heads=4, head_dim=8, num_layers=2)


@pytest.fixture
def network():
    """Untrained, in float64: these tests are about the computation, not what was learned."""
    return build_network(SMALL).double().eval()


def data(n):
    return torch.randn(n, 1), torch.randn(n)


def test_cache_reproduces_one_pass(network):
    (xc, yc), (xt, _) = data(12), data(7)
    cache = {}
    with torch.no_grad():
        for lo, hi in [(0, 5), (5, 6), (6, 10), (10, 11), (11, 12)]:
            network.encode_context(xc[None, lo:hi], yc[None, lo:hi, None], cache)
        cached = network.query(xt[None], cache)
        one_pass = network(xc[None], yc[None, :, None], xt[None])
    assert torch.allclose(cached.loc, one_pass.loc, atol=1e-12)
    assert torch.allclose(cached.scale, one_pass.scale, atol=1e-12)


def test_dense_training_pass_predicts_each_point_from_its_prefix(network):
    x, y = data(9)
    with torch.no_grad():
        dense = network.forward_sequence(x[None], y[None, :, None])
        for i in range(1, 9):
            prefix = network(x[None, :i], y[None, :i, None], x[None, i : i + 1])
            assert torch.allclose(dense.loc[0, i - 1], prefix.loc[0, 0], atol=1e-12)
            assert torch.allclose(dense.scale[0, i - 1], prefix.scale[0, 0], atol=1e-12)


def test_update_extends_the_context_and_leaves_the_model_unchanged(network):
    (x0, y0), (x1, y1), (x2, y2), (xt, _) = data(6), data(3), data(3), data(5)
    model = IncTNPModel.fit(network, x0, y0)
    before = model.predict(xt)

    first, second = model.update(x1, y1), model.update(x2, y2)

    after = model.predict(xt)
    assert torch.equal(before.loc, after.loc) and torch.equal(before.scale, after.scale)
    assert model.num_context == 6 and first.num_context == 9
    # Each update is the model refitted on the longer context, whatever else was updated from it.
    for updated, x, y in [(first, x1, y1), (second, x2, y2)]:
        refit = IncTNPModel.fit(network, torch.cat([x0, x]), torch.cat([y0, y])).predict(xt)
        assert torch.allclose(updated.predict(xt).loc, refit.loc, atol=1e-12)
        assert torch.allclose(updated.predict(xt).scale, refit.scale, atol=1e-12)
    assert not torch.allclose(first.predict(xt).loc, after.loc)


def test_joint_draws_come_from_the_context_the_update_reaches(network):
    (x0, y0), (x, _) = data(6), data(4)
    model = IncTNPModel.fit(network, x0, y0)
    torch.manual_seed(1)
    y = model.joint_sample(x)
    # Replay the draws: each is the model's own sample given the block's earlier outputs,
    # whether those entered the context one at a time or as a block.
    torch.manual_seed(1)
    for i in range(4):
        prefix = model.update(x[:i], y[:i]) if i else model
        assert torch.allclose(prefix.sample(x[i : i + 1], 1)[0], y[i : i + 1], atol=1e-10)
    assert model.num_context == 6


def test_predictions_take_the_callers_dtype():
    (x0, y0), (xt, _) = data(6), data(5)
    single = IncTNPModel.fit(build_network(SMALL).float().eval(), x0, y0)
    assert single.predict(xt).loc.dtype == torch.float64
    assert single.predict(xt.float()).loc.dtype == torch.float32
    assert single.sample(xt, 3).shape == (3, 5)
    assert single.joint_sample(xt).shape == (5,)


def test_an_empty_context_has_no_predictive(network):
    with pytest.raises(ValueError, match="empty context"):
        IncTNPModel(network).predict(torch.zeros(3, 1))


def test_rounds_feed_the_estimator_and_the_test(network):
    (x0, y0), (x_design, _), (x_eval, _) = data(6), data(3), data(4)
    model = IncTNPModel.fit(network, x0, y0)
    estimator, test = CoherenceEstimator(), SequentialTest()
    for round_ in islice(iter_rounds(model, RFFKernel(num_features=16), x_design, x_eval), 5):
        assert round_.mu_r.shape == (4, 32)
        estimator.update(*round_)
        test.update(round_.mu_p, round_.mu_r)
    assert estimator.estimate().v > 0
    assert model.num_context == 6


ACCELERATORS = [
    name
    for name, available in [
        ("cuda", torch.cuda.is_available),
        ("mps", torch.backends.mps.is_available),
    ]
    if available()
]


@pytest.mark.skipif(not ACCELERATORS, reason="needs a GPU")
@pytest.mark.parametrize("device", ACCELERATORS)
def test_devices_agree(device):
    (x0, y0), (xt, _) = data(6), data(5)
    network = build_network(SMALL).float().eval()
    on_cpu = IncTNPModel.fit(network, x0, y0).predict(xt)
    on_gpu = IncTNPModel.fit(copy.deepcopy(network).to(device), x0, y0).predict(xt)
    assert on_gpu.loc.device.type == "cpu" and on_gpu.loc.dtype == torch.float64
    assert torch.allclose(on_gpu.loc, on_cpu.loc, atol=1e-4)
    assert torch.allclose(on_gpu.scale, on_cpu.scale, atol=1e-4)


def test_muon_takes_only_the_hidden_matrices():
    network = build_network(SMALL)
    hidden, other = split_parameters(network)
    assert len(hidden) + len(other) == len(list(network.parameters()))
    assert all(p.shape == (32, 32) for p in hidden)
    # Four attention matrices and two feedforward matrices in each of the four layers, and the
    # two hidden matrices of each of the embedding and the decoder.
    assert len(hidden) == 4 * 6 + 4
    shapes = {tuple(p.shape) for p in other if p.ndim == 2}
    assert shapes == {(32, 3), (2, 32)}


OPTIMISATION = Optimisation(
    steps=300,
    batch_size=8,
    lr=3e-3,
    min_lr=1e-5,
    warmup_fraction=0.1,
    weight_decay=0.01,
    muon_momentum=0.95,
    grad_clip=0.5,
)
PRIOR = GaussianKernelPrior(lengthscale=0.5, outputscale=1.0, noise_std=0.1)
TASKS = SequenceTasks(PRIOR, (-2.0, 2.0), 12)


def validation():
    return TASKS.sample(64, torch.Generator().manual_seed(1))


def run_training(network, tmp_path, log=lambda step, metrics: None, optimisation=None, **kwargs):
    return train(
        network,
        TASKS,
        optimisation or OPTIMISATION,
        validation(),
        log,
        log_every=100,
        validate_every=100,
        partial_path=tmp_path / "partial.pt",
        **kwargs,
    )


def interrupt_at(stop):
    """A `log` that stops the run at step `stop`, before that step's validation and save."""

    def log(step, metrics):
        if step == stop:
            raise KeyboardInterrupt

    return log


def test_learning_rate_warms_up_and_then_decays():
    assert OPTIMISATION.lr_at(0) == 0
    assert OPTIMISATION.lr_at(30) == pytest.approx(3e-3)  # the end of a tenth of 300 steps
    assert OPTIMISATION.lr_at(15) == pytest.approx(1.5e-3)
    assert OPTIMISATION.lr_at(299) == pytest.approx(1e-5, rel=0.05)  # all but at the floor
    rates = [OPTIMISATION.lr_at(step) for step in range(30, 300)]
    assert all(later < earlier for earlier, later in zip(rates, rates[1:], strict=False))


def test_convergence_looks_back_over_the_last_quarter():
    history = [{"kl": kl} for kl in (8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)]
    # Eight validations: the last quarter runs from the sixth, where the divergence was 3.
    assert convergence(history) == {"kl": 1.0, "kl_tail_drop": 2.0}
    assert convergence(history[:1]) == {"kl": 8.0, "kl_tail_drop": 0.0}


@pytest.mark.slow
def test_training_approaches_the_exact_predictive(tmp_path):
    network = build_network(SMALL)
    untrained = evaluate(network, PRIOR, *validation())["kl"].mean().item()
    logged = {}
    history = run_training(network, tmp_path, lambda step, metrics: logged.update(metrics))
    assert [entry["step"] for entry in history] == [100, 200, 300]
    assert 0 < history[-1]["kl"] < 0.9 * untrained
    assert abs(history[-1]["nll"] - history[-1]["oracle_nll"] - history[-1]["kl"]) < 0.5
    assert {"train/loss", "train/lr", "val/kl"} <= set(logged)
    # The partial checkpoint is swapped in whole, leaving nothing half written beside it.
    assert [path.name for path in tmp_path.iterdir()] == ["partial.pt"]

    # Resuming a finished run trains no further and returns its history.
    assert run_training(build_network(SMALL), tmp_path, resume=True) == history


@pytest.mark.slow
def test_an_interrupted_run_resumes_where_it_stopped(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        run_training(build_network(SMALL), tmp_path, interrupt_at(300))

    # A new process starts from the script's seed; the run must not replay its first sequences.
    torch.manual_seed(0)
    resumed = run_training(build_network(SMALL), tmp_path, resume=True)
    assert [entry["step"] for entry in resumed] == [100, 200, 300]
    assert torch.initial_seed() == 200
    assert resumed[-1]["kl"] < resumed[0]["kl"]
    # The optimisers carried on too: AdamW has counted every step, not only those since.
    adamw = torch.load(tmp_path / "partial.pt", weights_only=False)["optimisers"][1]
    assert {int(state["step"]) for state in adamw["state"].values()} == {300}


@pytest.mark.slow
def test_a_finished_run_carries_on_under_a_longer_budget(tmp_path):
    """Training is a fixed budget with a cosine that anneals to its end, so a run that turns out
    not to have converged is carried on by giving it more steps. Everything but the budget must
    still match."""
    history = run_training(build_network(SMALL), tmp_path)
    assert [entry["step"] for entry in history] == [100, 200, 300]

    longer = dataclasses.replace(OPTIMISATION, steps=500)
    extended = run_training(build_network(SMALL), tmp_path, optimisation=longer, resume=True)
    assert [entry["step"] for entry in extended] == [100, 200, 300, 400, 500]
    assert extended[:3] == history  # the earlier history is kept, not replayed
    # The optimisers carried on rather than restarting from nothing.
    adamw = torch.load(tmp_path / "partial.pt", weights_only=False)["optimisers"][1]
    assert {int(state["step"]) for state in adamw["state"].values()} == {500}
    # A budget shorter than what is done leaves the network alone and asks nothing more of it.
    shorter = dataclasses.replace(OPTIMISATION, steps=200)
    assert (
        run_training(build_network(SMALL), tmp_path, optimisation=shorter, resume=True) == extended
    )


def test_a_partial_checkpoint_is_not_resumed_under_another_recipe(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        run_training(build_network(SMALL), tmp_path, interrupt_at(200))
    longer = SequenceTasks(PRIOR, (-2.0, 2.0), 16)
    with pytest.raises(ValueError, match="different recipe or prior"):
        train(
            build_network(SMALL),
            longer,
            OPTIMISATION,
            longer.sample(8),
            log=lambda step, metrics: None,
            log_every=100,
            validate_every=100,
            partial_path=tmp_path / "partial.pt",
            resume=True,
        )


def test_checkpoint_round_trip(tmp_path):
    (x0, y0), (xt, _) = data(6), data(5)
    network = build_network(SMALL).float().eval()
    save_run(tmp_path, "inctnp", {"manifest": {}, **network_payload(network, SMALL)})
    loaded = load_network(tmp_path / "inctnp.pt")
    assert next(loaded.parameters()).dtype == torch.float32
    expected = IncTNPModel.fit(network, x0, y0).predict(xt)
    assert torch.equal(IncTNPModel.fit(loaded, x0, y0).predict(xt).loc, expected.loc)


def test_training_refuses_to_replace_a_checkpoint_it_was_not_told_to(tmp_path):
    final = tmp_path / "inctnp.pt"
    options = training_parser("", "cpu")

    def paths(*flags):
        return checkpoint_paths(final, options.parse_args(list(flags)))

    assert paths() == (final, tmp_path / "inctnp_partial.pt")
    with pytest.raises(SystemExit, match="Nothing to carry on from"):
        paths("--resume")
    (tmp_path / "inctnp_partial.pt").touch()
    with pytest.raises(SystemExit, match="unfinished run"):
        paths()
    assert paths("--resume")[1].exists()
    final.touch()
    with pytest.raises(SystemExit, match="already trained"):
        paths()
    assert paths("--overwrite")[0] == final
    # A smoke run goes elsewhere, whatever is here.
    smoke, smoke_partial = paths("--smoke")
    assert smoke.parent != tmp_path and smoke.name == "inctnp.pt" and not smoke.exists()
    assert smoke_partial == smoke.with_name("inctnp_partial.pt")
