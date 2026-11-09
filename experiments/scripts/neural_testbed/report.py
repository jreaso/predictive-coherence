"""Figure 4, Table 10, and the numbers the paper quotes for the neural testbed.

Reads the baseline `run.py` saved. The baseline drew each random update once, so its random arms
are read from their re-score by `run_two_draw.py`, whose ‖Δ‖², V and Ψ average over two independent
draws, as the paper defines them; its decisions and losses are the baseline's, which the
re-score's first draw reproduces exactly. A baseline run again by today's `run.py`, which draws
twice itself, needs no re-score. Writes the figure to `experiments/figures/neural_testbed.pdf`,
the table to `experiments/tables/`, and prints the numbers Section 6 and Appendix C.4 quote.

    uv run experiments/scripts/neural_testbed/report.py [--png]
"""

import argparse
import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from run import NAME, RUN, checkpoint_path, results_dir, tuning_path
from run_two_draw import RESCORE, parity, two_draw_dir
from torch import Tensor
from tune import GRIDS

from pc_experiments.neural_testbed import Results
from pc_experiments.outputs import load_run
from pc_experiments.plotting import (
    BASELINE,
    INK,
    TEXT_WIDTH,
    mean_se,
    panel_label,
    save_figure,
    use_paper_style,
)
from pc_experiments.quotes import quote, section
from pc_experiments.statistics import spearman
from pc_experiments.tables import Table, write_tables

LABELS = {
    "exact": "Exact Bayes",
    "laplace_full": "Laplace, full",
    "laplace_kfac": "Laplace, KFAC",
    "laplace_diag": "Laplace, diagonal",
    "bbb": "Bayes by backprop",
    "ivon": "IVON",
    "ensemble": "Ensemble",
    "dropout": "MC dropout",
    "inctnp": "incTNP",
}


# A ratio below this is drawn at it and marked as zero: an exactly coherent model's is rounding.
PSI_FLOOR = 1e-5

# The main figure. A name, a marker and its size for each arm: the three Laplace curvatures
# share the triangle family, and colour runs dark to light with the arm's incoherence, so that
# neither channel has to be read on its own.
ARM_STYLE = {
    "laplace_kfac": ("LL-KFAC", "v", 5.0),
    "ivon": ("IVON", "P", 5.4),
    "inctnp": ("incTNP", "*", 7.4),
    "bbb": ("BBB", "o", 5.0),
    "ensemble": ("Ensemble", "X", 5.4),
    "laplace_full": ("LL-Full", ">", 5.0),
    "laplace_diag": ("LL-Diag", "^", 5.0),
    "dropout": ("Dropout", "s", 4.6),
}
# Bandwidths chosen by five-fold cross-validation; the held-out error is nearly flat in them.
BANDWIDTH_PSI, BANDWIDTH_ACCURACY = 0.16, 0.06
# The fit is drawn over exactly this range of log10 Ψ, and panel (a)'s axis is set to match.
PSI_RANGE = (-2.5, -0.4)
ACCURACY_RANGE = (0.10, 0.88)


def local_linear(x: Tensor, y: Tensor, grid: Tensor, bandwidth: float) -> Tensor:
    """Gaussian-kernel local linear regression of `y` on `x`, both `(n,)`, at `grid` `(g,)`.
    Unlike a local mean it is unbiased at the ends of the range."""
    u = (x[None, :] - grid[:, None]) / bandwidth
    w = torch.exp(-0.5 * u * u)
    s0, s1, s2 = w.sum(1), (w * u).sum(1), (w * u * u).sum(1)
    t0, t1 = (w * y[None, :]).sum(1), (w * u * y[None, :]).sum(1)
    denominator = s0 * s2 - s1 * s1
    return torch.where(denominator.abs() > 1e-12, (s2 * t0 - s1 * t1) / denominator,
                       t0 / s0.clamp_min(1e-12))  # fmt: skip


def local_quantiles(
    x: Tensor, y: Tensor, grid: Tensor, bandwidth: float, levels: Sequence[float] = (0.1, 0.9)
) -> list[Tensor]:
    """Kernel-weighted quantiles of `y` at each point of `grid`, each `(g,)`: the scatter about
    the fit, which its standard error does not show."""
    w = torch.exp(-0.5 * ((grid[:, None] - x[None, :]) / bandwidth) ** 2)
    order = y.argsort()
    y_sorted, w_sorted = y[order], w[:, order]
    cumulative = w_sorted.cumsum(1) / w_sorted.sum(1, keepdim=True)
    return [y_sorted[(cumulative < level).sum(1).clamp(max=len(y) - 1)] for level in levels]


def local_error(x: Tensor, y: Tensor, grid: Tensor, bandwidth: float, fitted: Tensor) -> Tensor:
    """Two standard errors of a local fit at each point of `grid`, `(g,)`, taking the kernel's
    effective sample size as the number of observations."""
    w = torch.exp(-0.5 * ((grid[:, None] - x[None, :]) / bandwidth) ** 2)
    effective = w.sum(1) ** 2 / (w * w).sum(1)
    variance = (w * (y[None, :] - fitted[:, None]) ** 2).sum(1) / w.sum(1)
    return 2 * (variance / effective).sqrt()


def label(arm: str) -> str:
    return LABELS.get(arm, arm)


def quartiles(values: Tensor) -> tuple[float, float, float]:
    """The median and the quartiles about it, ignoring NaNs."""
    values = values[~values.isnan()]
    if not len(values):
        return math.nan, math.nan, math.nan
    low, median, high = values.quantile(torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype))
    return median.item(), low.item(), high.item()


def by_incoherence(results: Results) -> list[str]:
    """Arms from the least to the most incoherent. One whose updates never move has no ratio,
    and goes last."""

    def key(arm: str) -> tuple[bool, float]:
        median = quartiles(results.psi(arm))[0]
        return math.isnan(median), median

    return sorted(results.arms, key=key)


def arm_colours(arms: list[str]) -> dict[str, tuple[float, ...]]:
    """A colour per arm, dark to light as its median incoherence rises."""
    return {
        arm: plt.get_cmap("viridis")(0.05 + 0.80 * index / (len(arms) - 1))
        for index, arm in enumerate(arms)
    }


def decades(axis: Axis) -> None:
    """Major ticks every decade, labelled as a power of ten, with minor ticks left unlabelled."""
    axis.set_major_locator(LogLocator(base=10))
    axis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    axis.set_minor_formatter(NullFormatter())


def draw_arm(ax: Axes, arm: str, x: float, y: float, colour: tuple[float, ...]) -> None:
    _, marker, size = ARM_STYLE[arm]
    ax.plot(x, y, marker=marker, markersize=size, color=colour, linestyle="none",
            markeredgecolor="white", markeredgewidth=0.5, zorder=5)  # fmt: skip


def susceptibility(ax: Axes, results: Results, arms: list[str], palette) -> None:
    """Ψ against how far incoherence moves the objective, one point per environment."""
    psi = torch.cat([results.psi(arm) for arm in arms])
    y = torch.cat([results.distortion(arm) for arm in arms])
    # An estimate of Ψ at or below zero, which two draws of a random update allow, sits at
    # log Ψ = -inf, where the kernel gives it no weight: leaving it out changes nothing.
    x, y = psi[psi > 0].log10(), y[psi > 0]
    grid = torch.linspace(*PSI_RANGE, 140, dtype=x.dtype)
    low, high = local_quantiles(x, y, grid, BANDWIDTH_PSI)
    ax.fill_between(10**grid, low, high, color=BASELINE, alpha=0.35, linewidth=0, zorder=1)
    ax.plot(10**grid, local_linear(x, y, grid, BANDWIDTH_PSI), color=INK, linewidth=1.6, zorder=3)
    for arm in arms:
        draw_arm(ax, arm, results.psi(arm).mean().item(),
                 results.distortion(arm).mean().item(), palette[arm])  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** PSI_RANGE[0], 10 ** PSI_RANGE[1])
    ax.set_ylim(1e-3, 4.5e-2)
    decades(ax.xaxis)
    decades(ax.yaxis)
    # Two decades of range leave a single horizontal rule, which reads as a stray line, so this
    # panel opts out of the shared grid.
    ax.grid(visible=False)
    ax.set_xlabel("Incoherence ratio Ψ")
    ax.set_ylabel("Objective distortion (nats)")
    handles = [
        Line2D([], [], color=palette[arm], marker=ARM_STYLE[arm][1], linestyle="none",
               markersize=ARM_STYLE[arm][2] * 0.8, markeredgecolor="white",
               markeredgewidth=0.4, label=ARM_STYLE[arm][0])
        for arm in arms
    ]  # fmt: skip
    ax.legend(handles=handles, loc="lower right", ncols=2,
              handletextpad=0.2, columnspacing=0.7, labelspacing=0.28, borderpad=0.35)  # fmt: skip


def severity(ax: Axes, results: Results, arms: list[str], palette) -> None:
    """Where the two objectives disagreed, what the disagreement cost, against accuracy."""
    differs = {arm: results.disagreement(arm) for arm in arms}
    x = torch.cat([results.of("initial_kl", arm)[differs[arm]] for arm in arms]).double()
    y = torch.cat([results.regret_gap(arm, "epig", "test")[differs[arm]] for arm in arms])
    grid = torch.linspace(*ACCURACY_RANGE, 140, dtype=x.dtype)
    fitted = local_linear(x, y, grid, BANDWIDTH_ACCURACY)
    error = local_error(x, y, grid, BANDWIDTH_ACCURACY, fitted)
    ax.fill_between(grid, fitted - error, fitted + error, color=BASELINE, alpha=0.35, linewidth=0)
    ax.plot(grid, fitted, color=INK, linewidth=1.6, zorder=3)
    for arm in arms:
        chosen = differs[arm]
        draw_arm(ax, arm, results.of("initial_kl", arm)[chosen].mean().item(),
                 results.regret_gap(arm, "epig", "test")[chosen].mean().item(),
                 palette[arm])  # fmt: skip
    ax.set_xlim(*ACCURACY_RANGE)
    ax.set_ylim(0, None)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("KL from the truth (nats)")
    ax.set_ylabel("Regret gap (nats)")


def main_figure(results: Results) -> Figure:
    """Ψ sets how far incoherence moves the objective; accuracy sets what a changed choice costs."""
    arms = [arm for arm in by_incoherence(results) if arm != "exact"]
    palette = arm_colours(arms)
    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.8), layout="constrained")
    susceptibility(axes[0], results, arms, palette)
    severity(axes[1], results, arms, palette)
    for ax, letter in zip(axes, "ab", strict=True):
        panel_label(ax, letter)
    return figure


# ----- table --------------------------------------------------------------------------------------


def signed_root(values: Tensor) -> Tensor:
    """‖Δ‖ from estimates of ‖Δ‖², keeping their sign: a two-draw estimate near zero can be
    negative."""
    return values.sign() * values.abs().sqrt()


def results_table(results: Results) -> Table:
    """Table 10: each model's incoherence, how often each form of EPIG disagrees with EMI, what
    the entropy form's choice costs over all decisions, and the model's KL from the truth before
    the update. Ψ and ‖Δ‖ are medians over environments of each one's median over candidates;
    the rest are means over environments, two with their standard errors."""
    rows = []
    for arm in by_incoherence(results):
        delta = signed_root(results.of("c", arm)).nanquantile(0.5, dim=-1)
        rows.append([
            arm,
            quartiles(results.psi(arm))[0],
            quartiles(delta)[0],
            results.disagreement(arm, "epig", "emi").double().mean().item(),
            results.disagreement(arm, "epig_kl", "emi").double().mean().item(),
            *mean_se(results.regret_gap(arm, "epig", "test")),
            *mean_se(results.of("initial_kl", arm)),
        ])  # fmt: skip
    columns = ["model", "psi", "delta_norm", "disagree_epig_h", "disagree_epig_kl", "cost",
               "cost_se", "kl", "kl_se"]  # fmt: skip
    return Table("results", columns, rows)


# ----- numbers in the text -----------------------------------------------------------------------


def approximate(results: Results) -> list[str]:
    """The arms with a ratio to speak of: every one but the exactly coherent control."""
    return [arm for arm in by_incoherence(results) if quartiles(results.psi(arm))[0] >= PSI_FLOOR]


def margin(results: Results, arm: str) -> Tensor:
    """How far EMI's best candidate is ahead of its second best, shape `(environments,)`."""
    top = results.of("emi", arm).topk(2, dim=-1).values
    return top[..., 0] - top[..., 1]


def quintile_rates(score: Tensor, flag: Tensor) -> list[float]:
    """The mean of `flag` within each fifth of `score`, from the lowest fifth to the highest."""
    order = score.argsort()
    return [part.double().mean().item() for part in flag[order].tensor_split(5)]


def tuned_at_edges() -> list[str]:
    """Each tuned hyperparameter whose chosen value is the smallest or largest of its grid."""
    edges = []
    for family, stages in GRIDS.items():
        chosen = load_run(tuning_path(family))["chosen"]
        for grid in stages.values():
            for key, values in grid.items():
                if len(values) > 1 and chosen[key] in (min(values), max(values)):
                    edges.append(f"{family} {key} {chosen[key]}")
    return edges


def report(results: Results) -> None:
    arms = approximate(results)
    psi = torch.stack([results.psi(arm) for arm in arms])  # (arms, environments)
    distortion = torch.stack([results.distortion(arm) for arm in arms])
    differ = torch.stack([results.disagreement(arm) for arm in arms])
    gap = torch.stack([results.regret_gap(arm, "epig", "test") for arm in arms])
    accuracy = torch.stack([results.of("initial_kl", arm) for arm in arms])

    section("Section 6, incoherence and acquisition across models (Figure 4)")
    exact = [results.disagreement("exact", objective, "emi").double().mean().item()
             for objective in ("epig", "epig_kl")]  # fmt: skip
    quote("Exact Bayes: share of decisions where EPIG_H, and EPIG_KL, disagree with EMI",
          f"{exact[0]:.3f}, {exact[1]:.3f}")  # fmt: skip
    for objective, text in (("epig", "EPIG_H"), ("epig_kl", "EPIG_KL")):
        rates = {arm: results.disagreement(arm, objective, "emi").double().mean().item()
                 for arm in arms}  # fmt: skip
        low, high = min(rates, key=lambda k: rates[k]), max(rates, key=lambda k: rates[k])
        quote(f"{text} disagrees with EMI, lowest and highest over the approximate models",
              f"{rates[low]:.3f} ({label(low)}) to {rates[high]:.3f} ({label(high)})")  # fmt: skip
    positive = sum(mean_se(results.regret_gap(arm, "epig", "test"))[0] > 0 for arm in arms)
    quote("Approximate models whose EPIG_H choice loses more than EMI's on average",
          f"{positive} of {len(arms)}")  # fmt: skip
    within = spearman(psi.T, distortion.T)
    quote("Within-environment rank correlation of Ψ with distortion: mean, median, share positive",
          f"{within.mean():.3f}, {within.median():.3f}, {(within > 0).double().mean():.3f}")  # fmt: skip
    for arm in ("bbb", "inctnp"):
        cost = results.regret_gap(arm, "epig", "test")
        quote(f"{label(arm)}: median Ψ, EPIG_H disagreement, cost over all decisions and over "
              "disagreements alone (nats)",
              f"{quartiles(results.psi(arm))[0]:.4f}, "
              f"{results.disagreement(arm).double().mean():.3f}, "
              "{:.4f} ± {:.4f}, ".format(*mean_se(cost))
              + "{:.4f} ± {:.4f}".format(*mean_se(cost[results.disagreement(arm)])))  # fmt: skip

    section("Appendix C.4, neural testbed (Table 10)")
    quote("Environments kept, of those drawn", f"{len(results.seeds)} of {max(results.seeds) + 1}")
    sampled = torch.tensor([float(sum(rule.startswith("sampled") for rule in rules))
                            for rules in results.rules])  # fmt: skip
    quote(f"Sampled batches per environment, of {results.config['num_candidates']}",
          f"{sampled.mean():.2f}")  # fmt: skip
    quote("incTNP training steps", f"{load_run(checkpoint_path())['history'][-1]['step']}")
    quote("Tuned values at the edge of their grid", "; ".join(tuned_at_edges()))
    counts = {arm: int((results.psi(arm) <= 0).sum()) for arm in results.arms}
    quote(f"Environments whose two-draw Ψ is not positive, of {len(results.seeds)}",
          ", ".join(f"{label(arm)} {n}" for arm, n in counts.items() if n))  # fmt: skip
    mean, se = mean_se(results.regret_gap("inctnp", "epig", "test"))
    quote("incTNP: cost over all decisions, and its distance from zero in standard errors",
          f"{mean:.4f} ± {se:.4f}, {mean / se:.2f}")  # fmt: skip
    better = [(gap[i][differ[i]] < 0).double().mean().item() for i in range(len(arms))]
    quote("Share of disagreements in which EPIG_H's choice is the better, over the models",
          f"{min(better):.3f} to {max(better):.3f}")  # fmt: skip
    ratio = torch.stack([margin(results, arm) for arm in arms]) / distortion
    quote("Disagreement rate by fifth of EMI's top-two margin over the distortion, pooled",
          ", ".join(f"{rate:.2f}" for rate in quintile_rates(ratio.flatten(), differ.flatten())))  # fmt: skip
    cost, kl = gap[differ], accuracy[differ]
    psi_differ = psi[differ]
    quote("Rank correlation of the cost of a disagreement with Ψ, and with the KL, pooled",
          f"{spearman(cost, psi_differ):.3f}, {spearman(cost, kl):.3f}")  # fmt: skip
    # Each model summarised as Table 10 summarises it: Ψ by its median, the KL by its mean.
    conditional = torch.stack([gap[i][differ[i]].mean() for i in range(len(arms))])
    quote("Rank correlation of a model's mean cost of a disagreement with its Ψ, and with its KL, "
          "across the models",
          f"{spearman(conditional, psi.median(1).values):.4f}, "
          f"{spearman(conditional, accuracy.mean(1)):.4f}")  # fmt: skip


def with_rescore(results: Results, name: str = RESCORE) -> Results:
    """The baseline with its random arms read from a re-score by `run_two_draw.py`, if it drew
    each random update once; a baseline that drew them twice, as `run.py` now does, as it is.

    Refused unless the re-score's losses are of the models its incoherence was measured on: its
    own, or the baseline's where its first draw is the baseline's (`run_two_draw.py --check`).
    """
    if results.drawn_twice:
        return results
    path = two_draw_dir() / "results" / f"{name}.pt"
    if not path.exists():
        raise SystemExit(
            f"No re-score at {path}.\nRun first: uv run experiments/scripts/{NAME}/run_two_draw.py, "
            "on the device the baseline ran on."
        )
    rescore = Results.load(path)
    if "loss_test" not in rescore.values:
        same, reasons = parity(name)
        if not same:
            raise SystemExit(
                "\n".join(reasons) + f"\n{name} has no test losses of its own and its first draw "
                "is not the baseline's, so its incoherence and the baseline's losses would be of "
                "different models. Re-score with --test-losses."
            )
    return results.rescored(rescore)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--png", action="store_true", help="also write a PNG beside the PDF")
    args = parser.parse_args()

    path = results_dir() / f"{RUN}.pt"
    if not path.exists():
        raise SystemExit(
            f"No results at {path}.\nRun first: uv run experiments/scripts/{NAME}/run.py"
        )
    use_paper_style()
    results = with_rescore(Results.load(path))
    save_figure(main_figure(results), NAME, args.png)
    write_tables(NAME, [results_table(results)])
    report(results)


if __name__ == "__main__":
    main()
