"""The neural testbed's models: what every model must satisfy, and what each one is."""

import math

import pytest
import torch

from pc_experiments.inctnp import Architecture as NetworkArchitecture
from pc_experiments.neural_testbed import Architecture, Arm, Setting, sample_environment, stack
from pc_experiments.neural_testbed.environment import generator
from pc_experiments.neural_testbed.models import (
    CURVATURES,
    Dropout,
    Ensemble,
    IncTNPClassifier,
    Laplace,
    LaplaceConfig,
    Mixture,
    Model,
    PriorBank,
    base,
    build_classifier,
    prefix_logits,
)
from pc_experiments.neural_testbed.models.laplace import softmax_curvature
from pc_experiments.neural_testbed.models.mean_field import MeanField
from pc_experiments.neural_testbed.scoring import (
    coherence_of_random_updates,
    enumerate_labellings,
    labelling_log_probs,
    score_candidates,
)

ARCHITECTURE = Architecture(input_dim=2, hidden_width=6, num_hidden_layers=2, num_classes=3)
SETTING = Setting(
    architecture=ARCHITECTURE,
    temperature=0.35,
    center=3.0,
    train_std=0.5,
    pool_std=1.5,
    target_std=1.5,
    train_clusters=(0, 1, 2),
    target_clusters=(2, 3),
    num_train=10,
    num_pool=40,
    num_target=6,
    num_test=50,
    num_candidates=3,
    candidate_size=2,
    min_class_mass=0.0,
)
STEPS = {"lr": 0.02, "fit_steps": 30, "update_lr": 0.02, "update_steps": 10}
ARMS = {
    "exact": Arm("exact", "prior_bank", {"num_atoms": 64}),
    **{
        f"laplace_{curvature}": Arm(
            f"laplace_{curvature}",
            "laplace",
            dict(curvature=curvature, prior_scale=2.0, num_atoms=8, lr=0.05, fit_steps=30),
        )
        for curvature in CURVATURES
    },
    "bbb": Arm(
        "bbb",
        "bbb",
        dict(
            prior_scale=2.0,
            kl_weight=0.5,
            init_std_factor=0.1,
            num_atoms=8,
            elbo_samples=2,
            **STEPS,
        ),
    ),
    "ivon": Arm(
        "ivon",
        "ivon",
        dict(
            prior_scale=2.0,
            kl_weight=0.5,
            num_atoms=8,
            mc_samples=2,
            hess_init=0.1,
            beta1=0.9,
            beta2=0.99,
        )
        | {**STEPS, "lr": 0.1, "update_lr": 0.1},
    ),
    "ensemble": Arm("ensemble", "ensemble", dict(num_members=3, weight_decay=0.5, **STEPS)),
    "dropout": Arm("dropout", "dropout", dict(rate=0.5, num_atoms=8, weight_decay=0.5, **STEPS)),
}
LABELLINGS = enumerate_labellings(2, 3)


def chunk_of(*seeds):
    return stack([sample_environment(SETTING, seed) for seed in seeds])


def fit[M: Model](kind: type[M], name, chunk) -> M:
    """An arm fitted to a chunk, as the class it is."""
    model = ARMS[name].fit(SETTING, chunk)
    assert isinstance(model, kind)
    return model


def batches(chunk):
    """Three batches of two pool inputs for each environment, shape `(E, 3, 2, 2)`."""
    return chunk.pool.x[:, :6].unflatten(1, (3, 2))


# ----- what every model must satisfy ---------------------------------------------------------------


@pytest.mark.parametrize("name", ARMS)
def test_the_joint_is_a_distribution_whose_marginals_are_the_predictive(name):
    chunk = chunk_of(0, 1)
    model = ARMS[name].fit(SETTING, chunk)
    x = batches(chunk)
    log_joint = model.log_joint(x, LABELLINGS)
    assert log_joint.shape == (2, 3, 9)
    torch.testing.assert_close(log_joint.exp().sum(-1), torch.ones(2, 3))

    assert isinstance(model, Mixture)
    marginal = log_joint.exp().unflatten(-1, (3, 3)).sum(-1)  # of each batch's first input
    torch.testing.assert_close(marginal, model.predict(x[:, :, 0]).exp())


@pytest.mark.parametrize("name", ARMS)
def test_an_update_leaves_the_model_as_it_was_and_batches_like_one_alone(name):
    chunk = chunk_of(0, 1)
    model = ARMS[name].fit(SETTING, chunk)
    before = model.predict(chunk.target.x)
    assert before.shape == (2, 6, 3)
    torch.testing.assert_close(before.exp().sum(-1), torch.ones(2, 6))

    x = batches(chunk)
    updated = model.update(x, LABELLINGS.expand(2, 3, -1, -1))
    after = updated.predict(chunk.target.x)
    assert updated.update_shape == (3, 9) and after.shape == (2, 3, 9, 6, 3)
    torch.testing.assert_close(after.exp().sum(-1), torch.ones(2, 3, 9, 6))
    assert torch.equal(model.predict(chunk.target.x), before)
    assert (after - before[:, None, None]).abs().max() > 1e-4

    # One update alone is the same update among many: a deterministic function of its inputs.
    alone = model.update(x[:, 1:2], LABELLINGS[5].expand(2, 1, 1, -1)).predict(chunk.target.x)
    torch.testing.assert_close(alone[:, 0, 0], after[:, 1, 5], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", ARMS)
def test_an_environment_does_not_depend_on_what_it_is_chunked_with(name):
    def views(chunk, e):
        model = ARMS[name].fit(SETTING, chunk)
        x = batches(chunk)
        updated = model.update(x, LABELLINGS.expand(len(x), 3, -1, -1))
        return [
            model.predict(chunk.target.x)[e],
            model.log_joint(x, LABELLINGS)[e],
            updated.predict(chunk.target.x)[e],
        ]

    for together, apart in zip(views(chunk_of(0, 1), 1), views(chunk_of(1), 0), strict=True):
        torch.testing.assert_close(together, apart, rtol=1e-6, atol=1e-8)


RANDOM_UPDATES = {"bbb", "ivon", "dropout"}


@pytest.mark.parametrize("name", ARMS)
def test_an_update_is_random_only_where_the_model_says_so(name, monkeypatch):
    chunk = chunk_of(0, 1)
    model = ARMS[name].fit(SETTING, chunk)
    assert model.random_update == ARMS[name].random_update == (name in RANDOM_UPDATES)
    x, y = batches(chunk), LABELLINGS.expand(2, 3, -1, -1)
    first = model.update(x, y).predict(chunk.target.x)

    # Every stream an update could draw from, moved: only a random update notices.
    drawn = []

    def moved(seed, *keys, device="cpu"):
        drawn.append(keys)
        return generator(seed + 1, *keys, device=device)

    monkeypatch.setattr(base, "generator", moved)
    assert torch.equal(model.update(x, y).predict(chunk.target.x), first) != model.random_update
    model.redrawn("second").update(x, y)
    monkeypatch.undo()

    again = model.redrawn("second").update(x, y).predict(chunk.target.x)
    if not model.random_update:
        assert model.redrawn("second") is model and torch.equal(again, first)
        return
    # The first draw comes from the stream it always has, so decisions do not move; the second
    # from another, which leaves the model as it was.
    assert set(drawn) == {(name, "update"), (name, "update", "second")}
    assert not torch.equal(again, first)
    assert model.update_stream == ()
    assert torch.equal(model.update(x, y).predict(chunk.target.x), first)


@pytest.mark.parametrize("name", ARMS)
def test_two_draws_of_a_deterministic_update_measure_what_one_does(name):
    chunk = chunk_of(0, 1)
    model = ARMS[name].fit(SETTING, chunk)
    x, y = batches(chunk), LABELLINGS.expand(2, 3, -1, -1)
    log_p = model.predict(chunk.target.x).unsqueeze(1)
    log_joint = model.log_joint(x, LABELLINGS)
    log_q = model.update(x, y).predict(chunk.target.x)
    log_q_again = model.redrawn("second").update(x, y).predict(chunk.target.x)
    one = score_candidates(log_p, log_q, log_joint)
    two = coherence_of_random_updates(log_p, log_q, log_q_again, log_joint)
    if model.random_update:
        assert not torch.allclose(two["c"], one["c"])
    else:
        for key in ("c", "v", "psi"):
            torch.testing.assert_close(two[key], one[key])


# ----- exact inference over a bank of prior networks -----------------------------------------------


def test_the_prior_bank_is_bayes_rule_and_exactly_coherent():
    chunk = chunk_of(2, 3)
    model = fit(PriorBank, "exact", chunk)
    # The posterior weights, by hand, for the first environment.
    log_likelihood = (
        model.atom_log_probs(chunk.train.x)[0]
        .gather(-1, chunk.train.y[0].expand(64, -1).unsqueeze(-1))
        .sum((-1, -2))
    )
    torch.testing.assert_close(model.log_w[0], torch.log_softmax(log_likelihood, 0))

    x = batches(chunk)
    log_joint = model.log_joint(x, LABELLINGS)
    log_q = model.update(x, LABELLINGS.expand(2, 3, -1, -1)).predict(chunk.target.x)
    scores = score_candidates(model.predict(chunk.target.x)[:, None], log_q, log_joint)
    assert scores["c"].max() < 1e-25
    torch.testing.assert_close(scores["epig"], scores["emi"])
    torch.testing.assert_close(scores["epig_kl"], scores["emi"])
    assert scores["v"].min() > 1e-6


# ----- linearised Laplace --------------------------------------------------------------------------


@pytest.mark.parametrize("curvature", CURVATURES)
def test_laplace_matches_a_dense_reference(curvature):
    architecture, num_atoms = ARCHITECTURE, 5
    num_parameters = architecture.num_parameters
    rng = generator(0, "laplace-test")
    x = torch.randn(2, 12, 2, generator=rng)
    y = torch.randint(3, (2, 12), generator=rng)
    x_batch = torch.randn(2, 3, 2, 2, generator=rng)
    y_batch = torch.randint(3, (2, 3, 4, 2), generator=rng)
    x_eval = torch.randn(2, 7, 2, generator=rng)
    config = LaplaceConfig(curvature, 2.0, num_atoms, 0.05, 200)
    model = Laplace.fit(config, SETTING, x, y, (0, 1))
    updated = model.update(x_batch, y_batch)
    predicted = updated.predict(x_eval)

    e, b, t = 1, 2, 3  # the environment, batch and labelling checked
    eye = torch.eye(num_parameters)
    theta0, bank = model.theta0[e], model.bank[e]
    prior_precision = (2.0 * architecture.prior_std).pow(-2)
    logits, jacobian = architecture.jacobian(x[e], theta0)
    jacobian = jacobian.flatten(0, 1)
    curvatures = torch.block_diag(*softmax_curvature(torch.log_softmax(logits, -1)))
    ggn = jacobian.T @ curvatures @ jacobian
    # Parameters in the order `_by_layer` holds them, for the Kronecker products.
    order = torch.cat([layer.flatten(1) for layer in model._by_layer(eye)], dim=1)

    def kronecker_covariance(roots):
        blocks = [torch.kron(q[e] @ q[e].T, p[e] @ p[e].T) for p, q in roots]
        return order @ torch.block_diag(*blocks) @ order.T

    if curvature == "full":
        covariance = torch.linalg.inv(ggn + torch.diag(prior_precision))
    elif curvature == "diag":
        covariance = torch.diag(1 / (ggn.diagonal() + prior_precision))
    else:
        covariance = kronecker_covariance(model._kronecker_roots(model.a, model.g, model.seen))
        first_input = torch.cat([x[e], torch.ones(12, 1)], -1)
        torch.testing.assert_close(model.a[0][e], first_input.T @ first_input)
    root = model._root_times(eye.expand(2, -1, -1))[e].T
    torch.testing.assert_close(root @ root.T, covariance)
    torch.testing.assert_close(model._root_times(eye.expand(2, -1, -1), transpose=True)[e], root)

    # The mode of the linearised posterior, by damped Newton steps on every parameter.
    batch_logits, design = architecture.jacobian(x_batch[e, b], theta0)
    design = design.flatten(0, 1)
    one_hot = torch.nn.functional.one_hot(y_batch[e, b, t], 3).double()
    precision = torch.linalg.inv(covariance)
    mean = theta0.clone()
    for _ in range(100):
        moved = batch_logits + (design @ (mean - theta0)).unflatten(0, (2, 3))
        log_probs = torch.log_softmax(moved, -1)
        gradient = design.T @ (log_probs.exp() - one_hot).flatten() + precision @ (mean - theta0)
        hessian = design.T @ torch.block_diag(*softmax_curvature(log_probs)) @ design + precision
        mean = mean - 0.5 * torch.linalg.solve(hessian, gradient)
    ours = theta0 + root @ (updated.projection[e, b] @ updated.beta[e, b, t])
    torch.testing.assert_close(ours, mean, rtol=0, atol=1e-7)

    moved = batch_logits + (design @ (mean - theta0)).unflatten(0, (2, 3))
    at_mode = torch.block_diag(*softmax_curvature(torch.log_softmax(moved, -1)))
    if curvature == "full":
        projection = updated.projection[e, b]
        new_root = root @ (eye - projection @ updated.mix[e, b, t] @ projection.T)
        expected = torch.linalg.inv(precision + design.T @ at_mode @ design)
        torch.testing.assert_close(new_root @ new_root.T, expected, rtol=0, atol=1e-9)
        atoms = ours + (new_root @ bank.T).T
    elif curvature == "diag":
        new_precision = model.sigma[e].pow(-2) + (design.T @ at_mode @ design).diagonal()
        torch.testing.assert_close(updated.new_sigma[e, b, t].pow(-2), new_precision)
        atoms = ours + bank * updated.new_sigma[e, b, t]
    else:
        roots = updated._kronecker_roots(
            [a[:, b] for a in updated.new_a], [g[:, b, t] for g in updated.new_g], model.seen + 2
        )
        shifts = [
            q[e] @ layer @ p[e].T
            for (p, q), layer in zip(roots, model._by_layer(bank), strict=True)
        ]
        atoms = ours + model._flat(shifts)

    eval_logits, eval_jacobian = architecture.jacobian(x_eval[e], theta0)

    def mixture_at_eval(atoms):
        atom_logits = eval_logits + torch.einsum("mkp,ap->amk", eval_jacobian, atoms - theta0)
        return torch.logsumexp(torch.log_softmax(atom_logits, -1), 0) - math.log(num_atoms)

    torch.testing.assert_close(predicted[e, b, t], mixture_at_eval(atoms))


def test_the_curvatures_share_their_map_estimate_and_their_atoms():
    chunk = chunk_of(0)
    models = [fit(Laplace, f"laplace_{c}", chunk) for c in CURVATURES]
    assert all(torch.equal(model.theta0, models[0].theta0) for model in models)
    assert all(torch.equal(model.bank, models[0].bank) for model in models)
    predictions = [model.predict(chunk.target.x) for model in models]
    assert not torch.allclose(predictions[0], predictions[2])


# ----- variational and replayed updates ------------------------------------------------------------


def test_a_variational_update_conditions_on_the_new_points_alone():
    chunk = chunk_of(0)
    for name in ("bbb", "ivon"):
        model = fit(MeanField, name, chunk)
        assert model.loc.shape == (1, ARCHITECTURE.num_parameters) and (model.scale > 0).all()
        x = batches(chunk)
        updated = model.update(x, LABELLINGS.expand(1, 3, -1, -1))
        assert updated.loc.shape == (1, 27, ARCHITECTURE.num_parameters)
        assert torch.equal(updated.bank, model.bank)  # the atoms' draws are shared
        # The labels it was given become more probable, on average over labellings.
        log_probs = updated.predict(x[:, 0])[0, 0]  # (9, 2, 3), at the first batch's inputs
        given = log_probs.gather(-1, LABELLINGS.unsqueeze(-1)).squeeze(-1).sum(-1)
        before = labelling_log_probs(model.predict(x[:, 0])[0], LABELLINGS)
        assert (given - before).mean() > 0.05
    # IVON never lets data lower the precision.
    assert (updated.scale <= model.scale[:, None] + 1e-12).all()


def test_replayed_updates_keep_their_atoms_and_train_on_everything_seen():
    chunk = chunk_of(0)
    ensemble = fit(Ensemble, "ensemble", chunk)
    assert ensemble.theta.shape == (1, 3, ARCHITECTURE.num_parameters)
    assert not torch.allclose(ensemble.theta[0, 0], ensemble.theta[0, 1])

    dropout = fit(Dropout, "dropout", chunk)
    assert [mask.shape for mask in dropout.masks] == [(1, 8, 6), (1, 8, 6)]
    assert set(dropout.masks[0].unique().tolist()) <= {0.0, 2.0}
    updated = dropout.update(batches(chunk), LABELLINGS.expand(1, 3, -1, -1))
    assert updated.masks is dropout.masks
    assert updated.theta.shape == (1, 27, ARCHITECTURE.num_parameters)
    assert torch.equal(updated.x, chunk.train.x)


# ----- the incTNP classifier -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def network():
    torch.manual_seed(0)
    architecture = NetworkArchitecture(
        dim_x=2, dim_y=3, embed_dim=16, num_heads=2, head_dim=8, num_layers=2
    )
    return build_classifier(architecture).float().eval()


def test_the_classifiers_cache_reproduces_a_single_pass(network):
    chunk = chunk_of(0, 1)
    model = IncTNPClassifier.fit(network, chunk.train.x, chunk.train.y)
    x = batches(chunk)
    updated = model.update(x, LABELLINGS.expand(2, 3, -1, -1))
    after = updated.predict(chunk.target.x)
    assert after.shape == (2, 3, 9, 6, 3) and after.dtype == torch.float64

    e, b, z = 1, 2, 5
    context_x = torch.cat([chunk.train.x[e], x[e, b]])[None].float()
    context_y = torch.cat([chunk.train.y[e], LABELLINGS[z]])[None]
    targets = chunk.target.x[e][None].float()
    whole = prefix_logits(network, context_x, context_y, targets, torch.full((6,), 12))
    torch.testing.assert_close(
        after[e, b, z], torch.log_softmax(whole[0].double(), -1), rtol=0, atol=1e-5
    )
    fitted = prefix_logits(network, context_x, context_y, targets, torch.full((6,), 10))
    torch.testing.assert_close(
        model.predict(chunk.target.x)[e],
        torch.log_softmax(fitted[0].double(), -1),
        rtol=0,
        atol=1e-5,
    )


def test_the_classifiers_joint_is_the_run_of_one_point_updates(network):
    chunk = chunk_of(0, 1)
    model = IncTNPClassifier.fit(network, chunk.train.x, chunk.train.y)
    x = batches(chunk)
    log_joint = model.log_joint(x, LABELLINGS)
    torch.testing.assert_close(log_joint.exp().sum(-1), torch.ones(2, 3), rtol=0, atol=1e-5)

    e, b, z = 0, 1, 7
    total, context_x, context_y = 0.0, chunk.train.x[e], chunk.train.y[e]
    for j in range(2):
        step = IncTNPClassifier.fit(network, context_x[None], context_y[None])
        total += step.predict(x[e, b, j][None, None])[0, 0, LABELLINGS[z, j]]
        context_x = torch.cat([context_x, x[e, b, j][None]])
        context_y = torch.cat([context_y, LABELLINGS[z, j][None]])
    torch.testing.assert_close(log_joint[e, b, z], total, rtol=0, atol=1e-5)
