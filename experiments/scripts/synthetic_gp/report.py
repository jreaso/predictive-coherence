"""Figure 2, Tables 1 and 2, and the numbers the paper quotes for the sparse GP experiment.

Reads the results `run.py` saved. Writes the figure to `experiments/figures/synthetic_gp.pdf`, the
tables to `experiments/tables/`, and prints the numbers Section 6 and Appendix C.1 quote. How the
figure looks lives here; the drawing helpers it uses live in `pc_experiments.plotting`.

    uv run experiments/scripts/synthetic_gp/report.py [--png]
"""

import argparse
import math
from collections import Counter
from typing import Any

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from run import FIXED, SETTING, results_path
from torch import Tensor

from pc_experiments.gp import greedy_variance
from pc_experiments.plotting import (
    AQUA,
    BASELINE,
    INK,
    ORANGE,
    PANEL_PAD,
    RAMP,
    SEQUENTIAL,
    TEXT_WIDTH,
    annotate,
    curve_mean_se,
    mean_se,
    panel_label,
    save_figure,
    use_paper_style,
)
from pc_experiments.quotes import quote, section
from pc_experiments.synthetic_gp import LADDERS, Arm, Results, sample_task
from pc_experiments.tables import Table, write_tables

NAME = "synthetic_gp"
ALPHA = 0.05
# An exact model's KL from the truth is zero to rounding; on a log axis it is drawn here.
KL_FLOOR = 1e-10
# A closed-form Ψ of exactly zero has no place on a log axis; it is put here, below any frame.
PSI_ZERO = 1e-12

EXACT, INCTNP = Arm("exact"), Arm("inctnp")
LABELS = {
    "exact": "Exact GP",
    "inctnp": "incTNP",
    "sparse": "Sparse GP, inducing inputs fixed",
    "reselected": "Sparse GP, inducing inputs re-selected",
}
# The two ladders: inducing inputs held fixed through an update, and chosen afresh from the
# enlarged data at each update. The short names are for legends inside a narrow panel.
LADDER_COLOUR = {"sparse": RAMP[2], "reselected": AQUA}
SHORT_LABELS = {"sparse": "SGP, fixed", "reselected": "SGP, re-selected"}
# Drawn above the reference levels, the fixed ladder above the re-selected one.
LADDER_ZORDER = {"sparse": 4, "reselected": 3}
# The floor of the main figure's Ψ axes. What lies beneath leaves the frame: the re-selected
# ladder on its way to the exact model, whose ratio is zero.
PSI_FLOOR = 1e-4
MARKER: dict[str, Any] = dict(
    marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.6
)


def closed_form_psi(results: Results, kind: str = "sparse") -> Tensor:
    """Exact Ψ of every rung of a ladder on every task, `(environments, subsamples, rungs)`."""
    return results.along_ladder("c_exact", kind) / results.along_ladder("v_exact", kind)


def ladders_present(results: Results) -> list[str]:
    """The kinds of ladder the run has, fixed first."""
    return [kind for kind in LADDERS if results.rungs(kind)]


def centre_band(values: Tensor, log: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """The centre over environments of `(environments, subsamples, ...)`, and two standard errors
    either side of it.

    Subsamples of one environment share its function, so they are averaged first and the standard
    error is over environments. On a log axis the centre is the geometric mean and the band is
    symmetric in the logarithm, since an arithmetic band would reach below zero.
    """
    mean, se = curve_mean_se((values.log10() if log else values).mean(1))
    band = (mean, mean - 2 * se, mean + 2 * se)
    return (10 ** band[0], 10 ** band[1], 10 ** band[2]) if log else band


# ----- figures ------------------------------------------------------------------------------------


def psi_axis(ax: Axes) -> None:
    """Ψ on a log axis from `PSI_FLOOR`. A ladder heading for zero leaves the frame at the
    bottom, which the caption says; nothing is bent to keep it in."""
    ax.set_yscale("log")
    ax.set_ylim(PSI_FLOOR, 2)
    ax.set_ylabel("Incoherence ratio $\\Psi$")


# One ramp per ladder for its rungs, darkest at the fewest inducing points, as the test panel.
LADDER_RAMP = {
    "sparse": SEQUENTIAL,
    "reselected": LinearSegmentedColormap.from_list("reselected", ["#9fe3c9", AQUA, "#0b6b48"]),
}


def rung_colours(kind: str, count: int) -> list[Any]:
    """The ramp of one ladder, darkest at the coarsest rung."""
    return [LADDER_RAMP[kind](1 - rank / max(1, count - 1)) for rank in range(count)]


def shown_rungs(results: Results, kind: str, min_inducing: int | None) -> list[int]:
    """The ranks, along one ladder, of the rungs with at least `min_inducing` inducing points."""
    rungs = results.rungs(kind)
    return [rank for rank, arm in enumerate(rungs) if arm.num_inducing >= (min_inducing or 0)]


def arms_panel(
    ax: Axes,
    results: Results,
    accuracy: str,
    min_inducing: int | None = None,
    numbered: bool = False,
    up_left: frozenset[int] = frozenset(),
    legend: bool = True,
) -> None:
    """Exact Ψ against accuracy, one point to an arm: each ladder a curve through its rungs,
    and the incTNP where it falls.

    `accuracy` is `kl`, the fitted model's divergence from the exact GP, or `kl_updated`, its
    divergence after taking in the block with its true outputs. Each point is the mean over
    environments of the mean over subsamples, the KL's on a log scale, with a band of two
    standard errors over environments in Ψ. The rungs are coloured by their number of inducing
    points, darkest at the coarsest. With `numbered`, a light line joins each rung's two arms and
    the rung's number sits directly above the more incoherent, or up and to the left of it for
    the rungs in `up_left`. The exact GP, at Ψ = 0, is off the log axis.
    """
    positions: dict[str, dict[int, tuple[float, float]]] = {}
    for kind in ladders_present(results):
        ranks = shown_rungs(results, kind, min_inducing)
        rungs = [results.rungs(kind)[rank] for rank in ranks]
        colour, order = LADDER_COLOUR[kind], LADDER_ZORDER[kind]
        kl = results.along_ladder(accuracy, kind)[..., ranks].clamp_min(KL_FLOOR)
        psi = closed_form_psi(results, kind).clamp_min(PSI_ZERO)[..., ranks]
        kl_centre = centre_band(kl, log=True)[0]
        psi_centre, psi_low, psi_high = centre_band(psi)
        ax.plot(kl_centre, psi_centre, color=colour, label=SHORT_LABELS[kind], zorder=order,
                **MARKER)  # fmt: skip
        # The rungs coloured by their number of inducing points, so that the ladder reads along
        # the curve as it does along the test panel's lines.
        rung_marker: dict[str, Any] = {**MARKER, "markersize": 4.2}
        for rank, shade in enumerate(rung_colours(kind, len(rungs))):
            ax.plot([kl_centre[rank]], [psi_centre[rank]], color=shade, linestyle="",
                    zorder=order + 0.5, **rung_marker)  # fmt: skip
        # Uncertainty in Ψ only, as a band; accuracy is a position.
        ax.fill_between(kl_centre, psi_low, psi_high, color=colour, alpha=0.12, linewidth=0,
                        zorder=order - 2)  # fmt: skip
        positions[kind] = {
            rung.num_inducing: (kl_centre[rank].item(), psi_centre[rank].item())
            for rank, rung in enumerate(rungs)
        }

    if numbered:
        # The line reaches the floor where an arm has left the frame. Before the update a rung's
        # two arms fit the same way and share their accuracy, so it is vertical.
        for m in sorted(set.intersection(*(set(p) for p in positions.values()))):
            points = [positions[kind][m] for kind in positions]
            ax.plot([x for x, _ in points], [max(y, PSI_FLOOR) for _, y in points],
                    color=BASELINE, linewidth=0.8, zorder=0)  # fmt: skip
            x, y = max(points, key=lambda point: point[1])
            if m in up_left:
                annotate(ax, x, y, f"{m}", (-3, 3), zorder=6)
            else:
                annotate(ax, x, y, f"{m}", (0, 5), ha="center", zorder=6)

    kl_i = results.of(accuracy, INCTNP).clamp_min(KL_FLOOR)
    x, y = centre_band(kl_i, log=True)[0], centre_band(results.of("psi", INCTNP))[0]
    ax.plot([x], [y], color=ORANGE, marker="*", markersize=8, markeredgecolor="white",
            markeredgewidth=0.6, linestyle="", label=LABELS["inctnp"], zorder=5)  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("KL from exact GP (nats)")
    psi_axis(ax)
    if legend:
        ax.legend(loc="lower right")


def test_panel(
    ax: Axes, results: Results, kind: str = "sparse", min_inducing: int | None = None
) -> None:
    """The share of tasks on which the test has rejected by each round, for one ladder's rungs
    from `min_inducing` in its ramp from dark to light, the incTNP, and the exact GP in black
    beneath them all, which stays at zero throughout, every line solid. The level is left to
    the caption."""
    rounds = torch.arange(1, results.config["num_rounds"] + 1)
    rungs = [results.rungs(kind)[rank] for rank in shown_rungs(results, kind, min_inducing)]
    shades = rung_colours(kind, len(rungs))
    # Along zero, above the gridline there, which would otherwise run through it, and beneath
    # the other lines.
    ax.plot(rounds, results.rejection_rate(EXACT, ALPHA), color=INK, linewidth=1.4, zorder=1.9)
    for arm, colour in zip(rungs, shades, strict=True):
        ax.plot(rounds, results.rejection_rate(arm, ALPHA), color=colour, linewidth=1.1, zorder=2)
    ax.plot(rounds, results.rejection_rate(INCTNP, ALPHA), color=ORANGE, linewidth=1.4, zorder=3)
    first, last = rungs[0].num_inducing, rungs[-1].num_inducing
    short = SHORT_LABELS[kind]
    handles = [
        Line2D([], [], color=shades[0], linewidth=1.1, label=f"{short} (M={first})"),
        Line2D([], [], color=shades[-1], linewidth=1.1, label=f"{short} (M={last})"),
        Line2D([], [], color=ORANGE, linewidth=1.4, label=LABELS["inctnp"]),
        Line2D([], [], color=INK, linewidth=1.4, label=LABELS["exact"]),
    ]
    # Compact, and lifted clear of the exact GP's line.
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 0.05), labelspacing=0.25,
              borderpad=0.3, handlelength=1.4)  # fmt: skip
    ax.set_xlabel("Rounds of updates")
    ax.set_ylabel("Fraction rejected")
    ax.set_xlim(0, results.config["num_rounds"])
    # A little below zero, so that the exact GP's line shows, with ticks only where a fraction
    # can be.
    ax.set_ylim(-0.03, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])


# The rungs the main figure shows. Below this a sparse model barely moves when updated, and the
# coarsest rungs would crowd the top of the plane.
MAIN_MIN_INDUCING = 10
# The main figure's two planes, by accuracy: before the update wide enough for the finest fixed
# rung, fitted exactly to about 1e-10; after it, the re-selected rung of 24 lies below the frame.
KL_LIMITS = {"kl": (2e-11, 2), "kl_updated": (3e-6, 2)}
PLANES = (("kl", "Pre-update"), ("kl_updated", "Post-update"))
# Numbered up and to the left in (a), where above would crowd the rising re-selected ladder.
UP_LEFT = frozenset({12, 14, 16})


def main_figure(results: Results) -> Figure:
    """Exact Ψ against the KL from the exact GP (a) before the update and (b) after it, both
    ladders and the incTNP, each rung's two arms joined and numbered in (a); (c) the sequential
    test's rejections by round. The rungs are coloured by their number of inducing points
    throughout. The two Ψ axes have the same limits, so a point keeps its height from (a) to
    (b)."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.8), layout="constrained")
    for ax, (accuracy, title) in zip(axes, PLANES, strict=False):
        first = accuracy == "kl"
        arms_panel(ax, results, accuracy, MAIN_MIN_INDUCING, numbered=first,
                   up_left=UP_LEFT if first else frozenset(), legend=first)  # fmt: skip
        ax.set_xlim(*KL_LIMITS[accuracy])
        ax.set_title(title, loc="center", pad=PANEL_PAD)
    test_panel(axes[2], results, min_inducing=MAIN_MIN_INDUCING)
    for ax, panel in zip(axes, "abc", strict=True):
        panel_label(ax, panel)
    return fig


# ----- tables ------------------------------------------------------------------------------------

# The rungs the tables give, as the figure.
TABLE_RUNGS = [m for m in FIXED if m >= MAIN_MIN_INDUCING]
# Arms whose estimates Table 2 follows over the rounds.
ROUNDS_ARMS = (EXACT, INCTNP, Arm("sparse", 10), Arm("sparse", 24), Arm("reselected", 20))
ROUNDS_SHOWN = (10, 100, 1000)
# What the tables call each kind of arm.
MODELS = {"exact": "exact", "inctnp": "inctnp", "sparse": "sparse_fixed",
          "reselected": "sparse_reselected"}  # fmt: skip


def table_arms() -> list[Arm]:
    return [EXACT, INCTNP, *(Arm(kind, m) for kind in LADDERS for m in TABLE_RUNGS)]


def has_closed_form(results: Results, arm: Arm) -> bool:
    return not results.of("c_exact", arm).isnan().any().item()


def psi_of(results: Results, arm: Arm) -> Tensor:
    """Ψ on every task, `(environments, subsamples)`: the closed form where the arm has one, and
    the estimate from every round otherwise."""
    if has_closed_form(results, arm):
        return results.of("c_exact", arm) / results.of("v_exact", arm)
    return results.of("psi", arm)


def geometric_kl(results: Results, key: str, arm: Arm) -> float:
    """The geometric mean over tasks of a divergence, after clamping at `KL_FLOOR`, as the figure
    places it; zero where it is zero on every task."""
    kl = results.of(key, arm)
    if (kl == 0).all():
        return 0.0
    return kl.clamp_min(KL_FLOOR).log().mean().exp().item()


def results_table(results: Results) -> Table:
    """Table 1: each arm's Ψ, its accuracy before and after the update, and the share of tasks
    rejected. Ψ is the mean over tasks, with its standard error over environments."""
    rows = []
    for arm in table_arms():
        psi, se = mean_se(psi_of(results, arm).mean(1))
        rows.append([
            MODELS[arm.kind],
            arm.num_inducing if arm.kind in LADDERS else None,
            psi,
            se,
            "closed_form" if has_closed_form(results, arm) else "estimate",
            geometric_kl(results, "kl", arm),
            geometric_kl(results, "kl_updated", arm),
            results.rejection_rate(arm, ALPHA)[-1].item(),
        ])  # fmt: skip
    columns = ["model", "inducing_points", "psi", "psi_se", "psi_source", "kl_before", "kl_after",
               "rejected"]  # fmt: skip
    return Table("results", columns, rows)


def rounds_table(results: Results) -> Table:
    """Table 2: the estimate of Ψ after `R` rounds, as the mean over tasks and the root mean
    square of the tasks' leave-one-round-out standard errors. The budgets are prefixes of the
    same rounds."""
    budgets = results.values["psi_at_rounds"].tolist()
    rows = []
    for arm in ROUNDS_ARMS:
        closed = psi_of(results, arm).mean().item() if has_closed_form(results, arm) else None
        estimates, errors = results.of("psi_at", arm), results.of("psi_se_at", arm)
        for budget in ROUNDS_SHOWN:
            index = budgets.index(budget)
            rows.append([
                MODELS[arm.kind],
                arm.num_inducing if arm.kind in LADDERS else None,
                closed,
                budget,
                estimates[..., index].mean().item(),
                errors[..., index].square().mean().sqrt().item(),
            ])  # fmt: skip
    columns = ["model", "inducing_points", "psi_closed_form", "rounds", "psi", "psi_se_rms"]
    return Table("rounds", columns, rows)


# ----- numbers in the text -----------------------------------------------------------------------


def name(arm: Arm) -> str:
    return LABELS[arm.kind] + (f", M = {arm.num_inducing}" if arm.kind in LADDERS else "")


def inducing_taken(results: Results, m: int) -> Counter[int]:
    """How many tasks the fixed rung of `m` takes each number of inducing inputs on. Selection
    stops early once those taken explain the rest to within rounding."""
    environments, subsamples = results.of("psi", EXACT).shape
    taken = Counter()
    for e in range(environments):
        for s in range(subsamples):
            task = sample_task(SETTING, e, s)
            taken[len(greedy_variance(SETTING.prior, task.x, m))] += 1
    return taken


def falls(values: list[float]) -> bool:
    return all(a > b for a, b in zip(values, values[1:], strict=False))


def report(results: Results) -> None:
    section("Section 6, incoherence within sparse GPs (Figure 2)")
    for kind in LADDERS:
        arms = [Arm(kind, m) for m in TABLE_RUNGS]
        psi = [psi_of(results, arm).mean().item() for arm in arms]
        kl = [geometric_kl(results, "kl", arm) for arm in arms]
        quote(f"{LABELS[kind]}: Ψ and pre-update KL fall at every rung from M = "
              f"{TABLE_RUNGS[0]}", "yes" if falls(psi) and falls(kl) else "no")  # fmt: skip
    fixed = {m: psi_of(results, Arm("sparse", m)).mean().item() for m in TABLE_RUNGS}
    reselected = {m: psi_of(results, Arm("reselected", m)).mean().item() for m in TABLE_RUNGS}
    quote("Rungs where re-selection is the less coherent",
          ", ".join(str(m) for m in TABLE_RUNGS if reselected[m] > fixed[m]))  # fmt: skip
    after = {kind: {m: geometric_kl(results, "kl_updated", Arm(kind, m)) for m in TABLE_RUNGS}
             for kind in LADDERS}  # fmt: skip
    quote("Post-update KL, re-selected over fixed, by rung",
          ", ".join(f"{m}: {after['reselected'][m] / after['sparse'][m]:.3f}" for m in TABLE_RUNGS))  # fmt: skip
    inctnp_after = geometric_kl(results, "kl_updated", INCTNP)
    quote("Post-update KL of the incTNP, and the best fixed rung's",
          f"{inctnp_after:.2e}, {min(after['sparse'].values()):.2e}")  # fmt: skip
    threshold = -math.log(ALPHA)
    wealth = results.of("log_wealth", EXACT)
    quote("Exact GP: share of tasks rejected, and its largest log wealth against log(1/α)",
          f"{results.rejection_rate(EXACT, ALPHA)[-1]:.2f}, {wealth.max():.2f} against "
          f"{threshold:.2f}")  # fmt: skip
    for arm in (INCTNP, *(Arm("sparse", m) for m in (20, 22, 24))):
        rounds = results.rejection_round(arm, ALPHA)
        rejected = rounds[~rounds.isnan()]
        quote(f"{name(arm)}: Ψ, share rejected by round {results.config['num_rounds']}, "
              "first rejection, median round among rejected",
              f"{psi_of(results, arm).mean():.4f}, {len(rejected) / rounds.numel():.3f}, "
              f"{rejected.min():.0f}, {rejected.median():.0f}")  # fmt: skip

    section("Appendix C.1, sparse GPs (Tables 1 and 2)")
    quote("incTNP training steps", f"{results.training['steps']}")
    for m in (22, 24):
        taken = inducing_taken(results, m)
        quote(f"Fixed rung of {m}: tasks by inducing inputs taken",
              ", ".join(f"{k}: {taken[k]}" for k in sorted(taken, reverse=True)))  # fmt: skip
    gaps = []
    for arm in table_arms()[2:]:
        estimate, se = mean_se(results.of("psi", arm).mean(1))
        gaps.append(abs(estimate - psi_of(results, arm).mean().item()) / se)
    gaps = torch.tensor(gaps)
    quote(f"Estimate against closed form over the {len(gaps)} sparse arms of Table 1: within one "
          "standard error, within two, largest gap",
          f"{int((gaps <= 1).sum())}, {int((gaps <= 2).sum())}, {gaps.max():.2f}")  # fmt: skip
    budgets = results.values["psi_at_rounds"].tolist()
    for arm in ROUNDS_ARMS:
        errors = results.of("psi_se_at", arm).square().mean((0, 1)).sqrt()
        ratio = errors[budgets.index(100)] / errors[budgets.index(1000)]
        quote(f"{name(arm)}: standard error at 100 rounds over that at 1000 "
              "(3.2 for R^-1/2, 10 for R^-1)", f"{ratio:.1f}")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--png", action="store_true", help="also write a PNG beside the PDF")
    args = parser.parse_args()

    if not results_path().exists():
        raise SystemExit(f"No results at {results_path()}.\n"
                         f"Run first: uv run experiments/scripts/{NAME}/run.py")  # fmt: skip
    use_paper_style()
    results = Results.load(results_path())
    save_figure(main_figure(results), NAME, args.png)
    write_tables(NAME, [results_table(results), rounds_table(results)])
    report(results)


if __name__ == "__main__":
    main()
