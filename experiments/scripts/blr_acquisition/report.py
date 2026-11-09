"""Figure 3, Tables 8 and 9, and the numbers the paper quotes for the Bayesian linear regression
acquisition experiment.

Reads the results `run.py` saved for every setting. Writes the figure from the baseline to
`experiments/figures/blr_acquisition.pdf`, the tables to `experiments/tables/`, and prints the
numbers Section 6 and Appendix C.3 quote. How the figure looks lives here; the drawing helpers it
uses live in `pc_experiments.plotting`.

    uv run experiments/scripts/blr_acquisition/report.py [--png]
"""

import argparse
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from run import SETTINGS
from torch import Tensor

from pc_experiments.blr import Run
from pc_experiments.outputs import output_dir
from pc_experiments.plotting import (
    BASELINE,
    INK,
    NEUTRAL,
    ORANGE,
    SEQUENTIAL,
    TEXT_WIDTH,
    annotate,
    band,
    mean_se,
    panel_label,
    reference_line,
    save_figure,
    use_paper_style,
)
from pc_experiments.quotes import quote, section
from pc_experiments.tables import Table, write_tables

NAME = "blr_acquisition"

EXACT, RANDOM = "exact", "random"
EPIG, EMI = "alpha0", "alpha1"
CURVE_SHARES = (0.0, 0.05, 1.0)  # shares drawn as curves; every share appears against Ψ
PSI_UNIT, PSI_UNIT_LABEL = 1e-3, "$\\times 10^{-3}$"
# Dotted for the reference the acquirers are measured against, dashed for the floor.
CURVE_STYLE = {EXACT: (":", 1.6), RANDOM: ("--", 1.5)}
POINT_LABEL_OFFSET = (-4, 5)  # points, up and to the left of the marker it names

# ----- names and colours -------------------------------------------------------------------------


def results_dir() -> Path:
    return output_dir(NAME) / "results"


def subsample_name(fraction: float) -> str:
    return f"subsample{fraction:g}"


def share(fraction: float) -> str:
    return f"{100 * fraction:g}%"


def label(name: str) -> str:
    """Readable name of an acquirer.

    A rung of the ladder is named by the share it averages over, and the share-zero rung is the
    mean-field model itself, since averaging over nothing leaves it unchanged.
    """
    fixed = {
        EXACT: "Exact Bayes",
        RANDOM: "Random",
        EPIG: "Mean-field, EPIG objective",
        EMI: "Mean-field, EMI objective",
    }
    if name in fixed:
        return fixed[name]
    fraction = float(name.removeprefix("subsample"))
    coherentised = "Mean-field" if fraction == 0 else "Coherentised"
    return f"{coherentised} ($\\rho$ = {share(fraction)})"


def plain(name: str) -> str:
    """`label` without mathtext, for tables."""
    return label(name).replace("$\\rho$", "ρ")


def colors(run: Run) -> dict[str, str]:
    """Colour per acquirer: shares along the ramp by rank, darkest at the smallest share."""
    shares = sorted(run.shares)
    palette = {
        subsample_name(fraction): to_hex(SEQUENTIAL(1 - index / (len(shares) - 1)))
        for index, fraction in enumerate(shares)
    }
    return palette | {EXACT: INK, RANDOM: NEUTRAL, EPIG: ORANGE, EMI: ORANGE}


def incoherence(run: Run, names: list[str]) -> list[float]:
    """Mean incoherence of each acquirer at step 0, in units of `PSI_UNIT`."""
    return [max(mean_se(run.psi(name))[0], 0.0) / PSI_UNIT for name in names]


# ----- the figure --------------------------------------------------------------------------------


def main_figure(run: Run) -> Figure:
    """Learning curves beside loss against the incoherence of the planning predictive."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.8), layout="constrained")
    palette = colors(run)

    steps = torch.arange(run.num_steps + 1)
    ladder = [subsample_name(fraction) for fraction in CURVE_SHARES]
    # Drawn from the floor upwards, ending on the exact reference, so that it lies over the
    # acquirers that reach it; it is dotted, so they stay visible underneath. EPIG is the
    # share-zero rung of the ladder, so it is drawn once, there.
    curves = {
        name: band(axes[0], steps, run.get("nll_mf", name), palette[name], label(name),
                   *CURVE_STYLE.get(name, ("-", 1.5)))
        for name in [RANDOM, EMI, *ladder, EXACT]
    }  # fmt: skip
    panel_label(axes[0], "a")
    axes[0].set_xlabel("Acquisition step")
    axes[0].set_ylabel("Log loss (nats)")
    # Every curve falls away to the right, which leaves the bottom left free for the legend once
    # the axis has room below.
    bottom, top = axes[0].get_ylim()
    axes[0].set_ylim(bottom - 0.42 * (top - bottom), top)
    # The legend reads down from the target to the floor: exact Bayes, the ladder as its planning
    # predictive is coherentised, the objective that arrives at the same place, and random. In two
    # columns it is half as tall, and fits the band the curves leave free along the bottom.
    legend_order = [EXACT, *ladder, EMI, RANDOM]
    axes[0].legend([curves[name] for name in legend_order],
                   [label(name) for name in legend_order], loc="lower left", ncols=2,
                   labelspacing=0.3, columnspacing=1.0, handlelength=1.6,
                   borderpad=0.35)  # fmt: skip

    loss_against_incoherence(axes[1], run, palette)
    panel_label(axes[1], "b")
    return fig


def loss_against_incoherence(ax: Axes, run: Run, palette: dict[str, str]) -> None:
    """Loss relative to exact against the incoherence of each rung, with EMI beside the ladder."""
    ax.axhline(0, color=BASELINE, linewidth=1)
    reference_line(ax, mean_se(run.loss(RANDOM)), label(RANDOM))

    shares = sorted(run.shares)
    names = [subsample_name(fraction) for fraction in shares]
    x = incoherence(run, names)
    y = [mean_se(run.loss(name)) for name in names]

    ax.plot(x, [mean for mean, _ in y], color=BASELINE, linewidth=1, zorder=1)
    for fraction, name, position, (mean, se) in zip(shares, names, x, y, strict=True):
        ax.errorbar(position, mean, yerr=2 * se, color=palette[name], marker="o", markersize=5,
                    capsize=2, linestyle="none", zorder=3)  # fmt: skip
        # Labels sit above and to the left, where the rising curve leaves room.
        text = share(fraction) if fraction else f"(mean-field) {share(fraction)}"
        annotate(ax, position, mean, text, POINT_LABEL_OFFSET)

    # EMI plans with the mean-field predictive, so it shares the incoherence of the share-zero
    # rung and differs only in the objective.
    mean, se = mean_se(run.loss(EMI))
    ax.errorbar(max(x), mean, yerr=2 * se, color=palette[EMI], marker="o", markersize=5,
                capsize=2, linestyle="none", zorder=3, label=label(EMI))  # fmt: skip
    annotate(ax, max(x), mean, "EMI", POINT_LABEL_OFFSET)
    # The most coherent point sits at zero, so the axis is padded to hold its label.
    ax.set_xlim(-0.16 * max(x), 1.06 * max(x))
    ax.set_xlabel(f"Incoherence ratio $\\Psi$ ({PSI_UNIT_LABEL})")
    ax.set_ylabel("Excess log loss (nats)")


# ----- tables ------------------------------------------------------------------------------------


def ladder(run: Run) -> list[str]:
    """The rungs of the ladder, by increasing share; the first is the mean-field model's EPIG."""
    return [subsample_name(fraction) for fraction in sorted(run.shares)]


def results_table(run: Run) -> Table:
    """Table 8: each acquirer's excess loss over the exact acquirer, averaged over the steps, and
    the incoherence of the predictive it plans with, for an update on the whole remaining pool.

    `acquirer` is `epig` along the ladder, planning with the predictive made coherent over a
    share `rho` of the pool, so that `rho` 0 is the mean-field model itself; `emi` is the
    mean-field model with EMI, and `random` takes the pool in a random order.
    """
    rows = []
    for name, fraction in zip(ladder(run), sorted(run.shares), strict=True):
        rows.append(["epig", fraction, *mean_se(run.loss(name)), *mean_se(run.psi(name))])
    for name, key in ((EMI, "emi"), (RANDOM, "random")):
        rows.append([key, None, *mean_se(run.loss(name)), None, None])
    columns = ["acquirer", "rho", "excess_loss", "excess_loss_se", "psi", "psi_se"]
    return Table("results", columns, rows)


ROBUSTNESS_ACQUIRERS = (
    ("epig", 0.0),
    ("emi", None),
    ("random", None),
    ("epig", 0.05),
    ("epig", 1.0),
)


def robustness_table(runs: dict[str, Run]) -> Table:
    """Table 9: the excess loss of five acquirers in the baseline and each robustness setting."""
    names = {"emi": EMI, "random": RANDOM}
    rows = []
    for setting, run in runs.items():
        for acquirer, fraction in ROBUSTNESS_ACQUIRERS:
            name = subsample_name(fraction) if fraction is not None else names[acquirer]
            rows.append([setting, acquirer, fraction, *mean_se(run.loss(name))])
    return Table(
        "robustness", ["setting", "acquirer", "rho", "excess_loss", "excess_loss_se"], rows
    )


# ----- numbers in the text -----------------------------------------------------------------------


def overlap(run: Run, first: str, second: str) -> Tensor:
    """How many points two acquirers both acquired, in each environment, shape `(envs,)`."""
    picks = zip(run.get("pick", first).tolist(), run.get("pick", second).tolist(), strict=True)
    return torch.tensor([float(len(set(a) & set(b))) for a, b in picks])


def report(runs: dict[str, Run]) -> None:
    run = runs["baseline"]
    rungs = ladder(run)
    epig = mean_se(run.loss(EPIG))[0]

    section("Section 6, incoherence and acquisition within mean-field BLR (Figure 3)")
    for name in (EPIG, RANDOM, EMI):
        quote(
            f"Excess loss, {plain(name)} (nats)", "{:.4f} ± {:.4f}".format(*mean_se(run.loss(name)))
        )
    losses = [mean_se(run.loss(name))[0] for name in rungs]
    psis = [mean_se(run.psi(name))[0] for name in rungs]
    falls = all(a > b for a, b in zip(losses, losses[1:], strict=False))
    falls &= all(a > b for a, b in zip(psis, psis[1:], strict=False))
    quote("Excess loss falls with Ψ at every step along the ladder", "yes" if falls else "no")

    section("Appendix C.3, mean-field Bayesian linear regression (Table 8)")
    separations = []
    for coarse, fine in zip(rungs, rungs[1:], strict=False):
        mean, se = mean_se(run.loss(coarse) - run.loss(fine))
        separations.append(mean / se)
    quote("Smallest separation in loss of adjacent rungs (paired standard errors)",
          f"{min(separations):.1f}")  # fmt: skip
    for name, fraction in zip(rungs[1:], sorted(run.shares)[1:], strict=False):
        if fraction < 1:
            recovered = 1 - mean_se(run.loss(name))[0] / epig
            quote(f"Share of mean-field EPIG's excess recovered at ρ = {share(fraction)}",
                  f"{recovered:.3f}")  # fmt: skip
    first = (run.get("pick", EPIG)[:, 0] != run.get("pick", EMI)[:, 0]).double().mean()
    quote("Share of environments where EPIG and EMI pick differently at the first step",
          f"{first:.2f}")  # fmt: skip
    shared = overlap(run, EPIG, EMI)
    quote(f"Points both acquired, of {run.num_steps}: mean, range",
          f"{shared.mean():.1f}, {shared.min():.0f} to {shared.max():.0f}")  # fmt: skip
    quote("Ψ of the mean-field model, update on the remaining pool (ρ = 0 rung)",
          "{:.3e} ± {:.1e}".format(*mean_se(run.psi(rungs[0]))))  # fmt: skip
    quote("Ψ of the mean-field model, update on a single candidate",
          "{:.4f} ± {:.4f}".format(*mean_se(run.psi(EPIG))))  # fmt: skip

    section("Appendix C.3, robustness (Table 9)")
    worse = [setting for setting, other in runs.items()
             if mean_se(other.loss(EPIG))[0] > mean_se(other.loss(RANDOM))[0]]  # fmt: skip
    quote("Settings where mean-field EPIG does worse than random",
          f"{len(worse)} of {len(runs)}: {', '.join(worse)}")  # fmt: skip
    for setting in ("baseline", "noise_low"):
        before = mean_se(runs[setting].at_checkpoint("excess_nll", EPIG))[0]
        quote(f"Mean-field model's excess log loss before acquisition, {setting} (nats)",
              f"{before:.4f}")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--png", action="store_true", help="also write a PNG beside the PDF")
    args = parser.parse_args()

    use_paper_style()
    paths = {setting: results_dir() / f"{setting}.pt" for setting in SETTINGS}
    missing = [setting for setting, path in paths.items() if not path.exists()]
    if missing:
        raise SystemExit(
            f"No results for {', '.join(missing)} in {results_dir()}.\n"
            f"Run first: uv run experiments/scripts/{NAME}/run.py"
        )
    runs = {setting: Run.load(path) for setting, path in paths.items()}
    save_figure(main_figure(runs["baseline"]), NAME, args.png)
    write_tables(NAME, [results_table(runs["baseline"]), robustness_table(runs)])
    report(runs)


if __name__ == "__main__":
    main()
