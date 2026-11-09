"""Tables 3 to 7, and the numbers the paper quotes, for the tabular experiment.

Reads what `tune.py`, `run.py`, `run_tabpfn.py` and `run_checks.py` saved. Writes the tables to
`experiments/tables/` and prints the numbers Section 6 and Appendix C.2 quote. The experiment has
no figure.

    uv run experiments/scripts/tabular/report.py
"""

import json
from itertools import combinations
from typing import Any

import torch
from run import DATASETS, NAME, SIZES, SPLITS, load_results, results_dir, tuning_dir
from run_checks import REPLICATED
from torch import Tensor

from pc_experiments.detectability import first_rejection, predicted_round
from pc_experiments.outputs import load_run, output_dir
from pc_experiments.plotting import mean_se
from pc_experiments.quotes import quote, section
from pc_experiments.statistics import kendall_w, ranks, spearman
from pc_experiments.tables import Table, write_tables
from pc_experiments.tabular import CLASSIFICATION, REGRESSION, Results, load, make_task, split_rows

ALPHA = 0.05

# The models of the paper's tables, in their order. The autoregressive joints of the plug-in
# models appear only in Appendix C.2's text.
MODELS = {
    "regression": ("gp", "sparse_gp", "gp_refit", "bart", "xbart", "kernel_ridge", "neighbours",
                   "catboost", "tabpfn", "tabpfn_preprocessing"),
    "classification": ("logistic_exact", "logistic_laplace", "logistic_diagonal", "logistic",
                       "neighbours", "catboost", "tabpfn", "tabpfn_preprocessing"),
}  # fmt: skip
EXACT = ("gp", "logistic_exact")
TABPFN = ("tabpfn", "tabpfn_preprocessing")
LABELS = {
    "gp": "GP",
    "sparse_gp": "Sparse GP",
    "gp_refit": "GP, refitted",
    "bart": "BART",
    "xbart": "XBART",
    "kernel_ridge": "Kernel ridge",
    "neighbours": "Nearest neighbours",
    "catboost": "CatBoost",
    "tabpfn": "TabPFN",
    "tabpfn_preprocessing": "TabPFN, preproc.",
    "logistic_exact": "Logistic, exact",
    "logistic_laplace": "Laplace",
    "logistic_diagonal": "Diagonal Laplace",
    "logistic": "Logistic regression",
}
# Kernels each regression cell was measured under, in the order `run.py` sets them.
NOISE_SCALE, HALF, TWICE, UNIT = range(4)


def task_of(dataset: str) -> str:
    return "classification" if dataset in CLASSIFICATION else "regression"


def datasets_of(task: str) -> list[str]:
    return [d for d in DATASETS if task_of(d) == task]


def psi(results: Results, dataset: str, model: str, kernel: int = NOISE_SCALE) -> float:
    """The model's Ψ on a dataset: its mean over splits."""
    return results.psi(dataset, model, kernel)[0]


def log_loss(results: Results, dataset: str, model: str) -> float:
    return results.of("nll", dataset, model).mean().item()


def median(values: list[float]) -> float:
    return torch.tensor(values).quantile(0.5).item()


# ----- tables ------------------------------------------------------------------------------------


def datasets_table() -> Table:
    """Table 3: each dataset and the sizes of a split."""
    rows = []
    for name in DATASETS:
        dataset = load(name)
        train, test, pool = split_rows(dataset, SIZES, 0)
        rows.append([
            name,
            task_of(name),
            {**REGRESSION, **CLASSIFICATION}[name],
            len(dataset.x),
            dataset.x.shape[1],
            dataset.num_classes or None,
            len(train),
            len(test),
            len(pool),
            make_task(dataset, SIZES, 0).block_size,
        ])  # fmt: skip
    columns = ["dataset", "task", "openml_id", "rows", "inputs", "classes", "train", "test",
               "pool", "block"]  # fmt: skip
    return Table("datasets", columns, rows)


def tuning(dataset: str) -> dict[str, Any]:
    return json.loads((tuning_dir() / f"{dataset}.json").read_text())


def hyperparameters_table() -> Table:
    """Table 4: what tuning chose for each dataset, and the dataset's noise scale. Kernel ridge's
    `gamma` is as used, on the standardised inputs, not as a multiple of one over their number."""
    rows = []
    for name in DATASETS:
        chosen = tuning(name)
        options = chosen["options"]
        ridge, catboost = options.get("kernel_ridge", {}), options["catboost"]
        rows.append([
            name,
            ridge.get("alpha"),
            ridge.get("gamma"),
            options["neighbours"]["n_neighbors"],
            catboost["depth"],
            catboost["iterations"],
            options.get("logistic", {}).get("prior_sd"),
            chosen["noise_scale"],
        ])  # fmt: skip
    columns = ["dataset", "kernel_ridge_alpha", "kernel_ridge_gamma", "neighbours",
               "catboost_depth", "catboost_trees", "logistic_prior_sd", "noise_scale"]  # fmt: skip
    return Table("hyperparameters", columns, rows)


def log_loss_ranks(results: Results, task: str) -> dict[str, list[float]]:
    """Each model's rank by test log loss among the models of its task, on each dataset, from 1
    for the lowest, with ties given their average rank."""
    models = MODELS[task]
    by_dataset = torch.tensor(
        [[log_loss(results, d, m) for m in models] for d in datasets_of(task)]
    )
    return {model: ranks(by_dataset)[:, i].tolist() for i, model in enumerate(models)}


def rejected_share(results: Results, model: str, task: str) -> float:
    """The share of the model's splits, over the datasets of its task, rejected at `ALPHA`."""
    rounds = torch.cat([results.rejection_round(d, model, ALPHA) for d in datasets_of(task)])
    return (~rounds.isnan()).double().mean().item()


def summary_table(results: Results) -> Table:
    """Table 5: each model's median Ψ over datasets with its range, its median rank by test log
    loss, and the share of its splits rejected. TabPFN's single split at fewer rounds is not
    comparable for rejections, so its share is left out."""
    rows = []
    for task, models in MODELS.items():
        ranked = log_loss_ranks(results, task)
        for model in models:
            values = [psi(results, d, model) for d in datasets_of(task)]
            rejected = None if model in TABPFN else rejected_share(results, model, task)
            rows.append([task, model, median(values), min(values), max(values),
                         median(ranked[model]), rejected])  # fmt: skip
    columns = ["task", "model", "median_psi", "min_psi", "max_psi", "median_rank", "rejected"]
    return Table("summary", columns, rows)


def psi_table(results: Results) -> Table:
    """Table 6: Ψ on each dataset, its mean over splits with the standard error over splits, or
    for a model run on one split, that split's leave-one-round-out standard error."""
    rows = [
        [task, d, model, len(results.values[d, model]), *results.psi(d, model)]
        for task, models in MODELS.items()
        for model in models
        for d in datasets_of(task)
    ]
    return Table("psi", ["task", "dataset", "model", "splits", "psi", "psi_se"], rows)


def log_loss_table(results: Results) -> Table:
    """Table 7: test log loss in nats per output, the mean over splits."""
    rows = [
        [task, d, model, len(results.values[d, model]), log_loss(results, d, model)]
        for task, models in MODELS.items()
        for model in models
        for d in datasets_of(task)
    ]
    return Table("log_loss", ["task", "dataset", "model", "splits", "log_loss"], rows)


# ----- numbers in the text -----------------------------------------------------------------------


def largest_swap(results: Results, kernel: int) -> tuple[float, float]:
    """Over pairs of regression models that change order between the noise scale's kernel and
    `kernel` on some dataset, the largest gap between them at the noise scale, relative to the
    larger of the two and to the smaller."""
    largest = (0.0, 0.0)
    for d in datasets_of("regression"):
        for a, b in combinations(MODELS["regression"], 2):
            first = psi(results, d, a) - psi(results, d, b)
            then = psi(results, d, a, kernel) - psi(results, d, b, kernel)
            if first * then < 0:
                low, high = sorted((psi(results, d, a), psi(results, d, b)))
                largest = max(largest, ((high - low) / high, (high - low) / low))
    return largest


def checks(name: str) -> list[dict[str, Any]]:
    path = output_dir(NAME) / "checks" / f"{name}.pt"
    if not path.exists():
        raise SystemExit(
            f"No check {name} at {path}.\n"
            f"Run first: uv run experiments/scripts/{NAME}/run_checks.py --checks {name}"
        )
    return load_run(path)["found"]


def projected(results: Results, model: str) -> tuple[Tensor, Tensor, Tensor]:
    """For each of a model's cells, a split of one dataset: whether its Ψ is more than two
    standard errors above zero, the round its test rejected at, NaN if it did not, and the round
    the estimates of ‖Δ‖² and V at each test input predict."""
    significant, rejection, predicted = [], [], []
    for d in DATASETS:
        if (d, model) not in results.values:
            continue
        values, errors = results.of("psi", d, model)[:, 0], results.of("psi_se", d, model)[:, 0]
        c, v = (results.of(key, d, model)[:, 0] for key in ("c", "v"))
        significant.append(values > 2 * errors)
        rejection.append(first_rejection(results.of("log_wealth", d, model), ALPHA))
        predicted.append(predicted_round(c, v, ALPHA))
    return torch.cat(significant), torch.cat(rejection), torch.cat(predicted)


def main_text(results: Results) -> None:
    section("Section 6, incoherence across models on tabular data (Tables 5 to 7)")
    exact = [(d, m) for m in EXACT for d in DATASETS if (d, m) in results.values]
    largest = max(abs(psi(results, d, m)) for d, m in exact)
    rejected = sum(int((~results.rejection_round(d, m, ALPHA).isnan()).sum()) for d, m in exact)
    splits = sum(len(results.values[d, m]) for d, m in exact)
    quote("Exact models: largest |Ψ| on a dataset, and splits rejected",
          f"{largest:.6f}, {rejected} of {splits}")  # fmt: skip
    quote("Halving or doubling the kernel's lengthscale: the largest gap between models that swap, "
          "relative to the larger and to the smaller",
          ", ".join(f"{max(largest_swap(results, k)[i] for k in (HALF, TWICE)):.3f}"
                    for i in range(2)))  # fmt: skip
    same = all(
        torch.equal(results.of("nll", d, "gp_refit"), results.of("nll", d, "gp"))
        for d in datasets_of("regression")
    )
    quote("Refitted GP: test log loss the same as the GP's on every split", "yes" if same else "no")
    quote("Refitted GP: Ψ on each regression dataset",
          ", ".join("{} {:.5f} ± {:.5f}".format(d, *results.psi(d, "gp_refit"))
                    for d in datasets_of("regression")))  # fmt: skip
    ratios = {d: psi(results, d, "tabpfn_preprocessing") / psi(results, d, "tabpfn")
              for d in DATASETS}  # fmt: skip
    widest = max(ratios, key=lambda k: ratios[k])
    quote("TabPFN with preprocessing over without: smallest and largest ratio of Ψ",
          f"{min(ratios.values()):.2f} ({min(ratios, key=lambda k: ratios[k])}) to "
          f"{ratios[widest]:.2f} ({widest})")  # fmt: skip
    quote(f"TabPFN without and with preprocessing: test log loss on {widest}",
          f"{log_loss(results, widest, 'tabpfn'):.4f}, "
          f"{log_loss(results, widest, 'tabpfn_preprocessing'):.4f}")  # fmt: skip
    for task, models in (("regression", ("bart", "neighbours", "catboost", "kernel_ridge")),
                         ("classification", ("logistic", "logistic_laplace",
                                             "logistic_diagonal"))):  # fmt: skip
        for model in models:
            values = {d: psi(results, d, model) for d in datasets_of(task)}
            quote(f"{LABELS[model]} ({task}): median Ψ, smallest and largest",
                  f"{median(list(values.values())):.4f}, "
                  f"{min(values.values()):.4f} ({min(values, key=lambda k: values[k])}) to "
                  f"{max(values.values()):.4f} ({max(values, key=lambda k: values[k])})")  # fmt: skip
    for task, models in MODELS.items():
        table = torch.tensor([[psi(results, d, m) for m in models] for d in datasets_of(task)])
        pairs = [spearman(table[i], table[j]).item()
                 for i, j in combinations(range(len(table)), 2)]  # fmt: skip
        quote(f"Rank correlation of the models' Ψ between {task} datasets: smallest pair, "
              "and Kendall's W", f"{min(pairs):.3f}, {kendall_w(table):.3f}")  # fmt: skip
    lowest = {d: min(MODELS[task_of(d)], key=lambda m, d=d: log_loss(results, d, m))
              for d in DATASETS}  # fmt: skip
    others = [f"{d} ({LABELS[m]})" for d, m in lowest.items() if m not in TABPFN]
    quote("Datasets on which TabPFN has the lowest test log loss",
          f"{len(DATASETS) - len(others)} of {len(DATASETS)}; otherwise {', '.join(others)}")  # fmt: skip
    classification = datasets_of("classification")
    over = {(d, m): psi(results, d, m) / psi(results, d, "logistic_laplace")
            for d in classification for m in TABPFN}  # fmt: skip
    smallest = min(over, key=lambda k: over[k])
    quote("TabPFN's Ψ over the full Laplace approximation's, smallest over datasets and variants",
          f"{over[smallest]:.1f} ({smallest[0]}, {LABELS[smallest[1]]})")  # fmt: skip
    better = [d for d in classification
              if all(log_loss(results, d, "logistic_laplace") < log_loss(results, d, m)
                     for m in TABPFN)]  # fmt: skip
    quote("Datasets on which the full Laplace approximation has a lower test log loss than both "
          "TabPFN variants", ", ".join(better) or "none")  # fmt: skip


def appendix(results: Results) -> None:
    section("Appendix C.2, tabular data (Tables 3 to 7)")
    inducing = checks("inducing")
    for d in datasets_of("regression"):
        counts = [found["inducing"] for found in inducing if found["dataset"] == d]
        quote(
            f"Sparse GP inducing points on {d}, over its splits", f"{min(counts)} to {max(counts)}"
        )
    for d in DATASETS:
        dataset = load(d)
        first = set(split_rows(dataset, SIZES, 0)[0].tolist())
        shares = [len(first & set(split_rows(dataset, SIZES, k)[1].tolist()))
                  / len(split_rows(dataset, SIZES, k)[1]) for k in range(1, SPLITS)]  # fmt: skip
        quote(f"Share of a later split's test rows in the first split's training set, {d}",
              f"{sum(shares) / len(shares):.2f}")  # fmt: skip
    z = {d: results.psi(d, "logistic_exact")[0] / results.psi(d, "logistic_exact")[1]
         for d in datasets_of("classification")}  # fmt: skip
    lowest = min(z, key=lambda k: z[k])
    quote("Exact logistic regression: furthest from zero, in standard errors over splits",
          f"{z[lowest]:.2f} ({lowest})")  # fmt: skip
    found = checks("replicates")
    dataset, split, model = REPLICATED
    for key, name in (("psi", "Ψ"), ("c", "‖Δ‖²")):
        values = torch.tensor([run[key] for run in found])
        quote(f"{LABELS[model]} on {dataset}, split {split}, {len(found)} reruns: mean {name} and "
              "its standard error, share negative",
              "{:.2e} ± {:.2e}, ".format(*mean_se(values))
              + f"{(values < 0).double().mean():.2f}")  # fmt: skip
    spread = {}
    for task, models in MODELS.items():
        for model in models:
            if model in TABPFN:
                continue
            ratios = []
            for d in datasets_of(task):
                values = results.of("psi", d, model)[:, 0]
                errors = results.of("psi_se", d, model)[:, 0]
                ratios.append((values.std() / errors.mean()).item())
            spread[model] = median(ratios)
    inexact = {m: r for m, r in spread.items() if m not in EXACT}
    quote("Spread of Ψ over splits over the within-split standard error, median over datasets, "
          "smallest and largest over inexact models",
          f"{min(inexact.values()):.2f} ({LABELS[min(inexact, key=lambda k: inexact[k])]}) to "
          f"{max(inexact.values()):.2f} ({LABELS[max(inexact, key=lambda k: inexact[k])]})")  # fmt: skip
    noise = [tuning(d)["noise_scale"] for d in datasets_of("regression")]
    quote("A lengthscale of one, in noise scales", f"{1 / max(noise):.1f} to {1 / min(noise):.1f}")
    quote("A lengthscale of one: the largest gap between models that swap, relative to the larger "
          "and to the smaller", ", ".join(f"{g:.3f}" for g in largest_swap(results, UNIT)))  # fmt: skip
    quote("Kernel ridge on yacht: Ψ with independent and with autoregressive outputs",
          f"{psi(results, 'yacht', 'kernel_ridge'):.4f}, "
          f"{psi(results, 'yacht', 'kernel_ridge_autoregressive'):.4f}")  # fmt: skip
    for null in checks("bart_null"):
        draws = (null["within"] / null["across"]) ** 2
        quote(f"{LABELS[null['arm']]} on {null['dataset']}, split 0: ‖Δ‖² of a null rerun over "
              "‖Δ‖² of an update, and effective independent draws in a run",
              f"{null['c_null'] / null['c_real']:.2f}, {draws:.1f}")  # fmt: skip
    significant, rejection, predicted = projected(results, "sparse_gp")
    quote("Sparse GP: cells with Ψ above two standard errors, cells rejected, median predicted "
          "round of the former", f"{int(significant.sum())} of {len(significant)}, "
          f"{int((~rejection.isnan()).sum())}, {predicted[significant].quantile(0.5):.0f}")  # fmt: skip
    cells = [projected(results, model) for model in TABPFN]
    rounds = torch.cat([predicted[significant] for significant, _, predicted in cells])
    quote(f"TabPFN: predicted rounds, smallest and largest over its {len(rounds)} cells with Ψ "
          "above two standard errors",
          f"{rounds.min():.0f} to {rounds.max():.0f}")  # fmt: skip


def main() -> None:
    results = load_results()
    if not results.values:
        raise SystemExit(f"No results in {results_dir()}.\n"
                         f"Run first: uv run experiments/scripts/{NAME}/run.py")  # fmt: skip
    write_tables(NAME, [
        datasets_table(),
        hyperparameters_table(),
        summary_table(results),
        psi_table(results),
        log_loss_table(results),
    ])  # fmt: skip
    main_text(results)
    appendix(results)


if __name__ == "__main__":
    main()
