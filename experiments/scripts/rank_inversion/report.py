"""Figure 1: EPIG_H and EMI rank two designs in opposite orders under mean-field inference.

Bayesian linear regression `y = w_0 + w_1 x + noise` is fitted to five points far from the
origin, so that intercept and slope are strongly correlated, by exact conditioning and by mean-field
variational inference. At one target, EPIG_H is `H[P_Y] - E_Z H[P_Y^Z]` and EMI is
`H[Q_Y] - E_Z H[P_Y^Z]`, with `P_Y^Z` the predictive after updating on the outcome `Z` and `Q_Y`,
the expected future predictive, its average over outcomes; they coincide when `Q_Y = P_Y`, which
the mean-field refit breaks. Everything is in closed form.

Writes `experiments/figures/rank_inversion.pdf` and prints the values Appendix C.3 quotes. There
is no run to make first.

    uv run experiments/scripts/rank_inversion/report.py [--png]
"""

import argparse
import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

from pc_experiments.blr.gaussian import Moments, Posterior, View, conditioning, mean_field_refit
from pc_experiments.plotting import (
    ANNOTATION_SIZE,
    INK,
    INK_SECONDARY,
    MUTED,
    ORANGE,
    RAMP,
    TEXT_WIDTH,
    annotate,
    panel_label,
    save_figure,
    use_paper_style,
)
from pc_experiments.quotes import quote, section

NAME = "rank_inversion"

# ----- the setting -------------------------------------------------------------------------------

NOISE_VAR = 1.0
PRIOR_VAR = 4.0
SEED = 42
NUM_POINTS = 5
INPUT_RANGE = (4.0, 6.0)  # far from the origin, so that intercept and slope trade off
TRUE_WEIGHTS = (0.0, 0.2)  # intercept and slope of the line the data come from
TARGET = 3.5
DESIGNS = torch.linspace(-10.0, 30.0, 401, dtype=torch.float64)
DESIGN_RANGE = (0.0, 18.0)  # the part of the design axis drawn
MARKED = {"A": 2.0, "B": 16.0}  # the two designs whose order the objectives disagree on

# ----- the figure --------------------------------------------------------------------------------

HEIGHT = 1.8  # as the other two-panel figures
DENSITY_POINTS = 600
# Three blues far enough apart to tell three densities apart where they nearly coincide, from the
# narrowest (darkest) to the widest (lightest), and the fill between them.
SHADES = ("#0a2a5e", "#3987e5", "#a9c9f2")
FILL = "#3987e5"
# The mean-field objectives: EMI in orange, as in the BLR acquisition figure, and EPIG_H in a mid
# blue, clear of exact Bayes's black.
BLUE = RAMP[2]
ZOOM = (0.49, 0.31, 0.49)  # left, bottom and width of the zoom, as fractions of the axes
MARKER_AREA = 20


def make_data() -> tuple[Tensor, Tensor]:
    """Inputs clustered far from the origin and noisy outputs from a line, shapes `(n,)`."""
    rng = np.random.default_rng(SEED)
    x = rng.uniform(*INPUT_RANGE, size=NUM_POINTS)
    noise = rng.normal(0.0, math.sqrt(NOISE_VAR), size=NUM_POINTS)
    y = TRUE_WEIGHTS[0] + TRUE_WEIGHTS[1] * x + noise
    return torch.from_numpy(x), torch.from_numpy(y)


def features(x: Tensor) -> Tensor:
    """Intercept and slope features of inputs `(n,)`, shape `(n, 2)`."""
    return torch.stack([torch.ones_like(x), x], dim=-1)


def anticipate(x: Tensor, y: Tensor) -> tuple[Moments, Moments, float]:
    """The moments at the target under exact conditioning and the mean-field refit at every
    design, and the predictive mean there, which every predictive shares."""
    posterior = Posterior.fit(features(x), y, PRIOR_VAR, NOISE_VAR)
    target = features(torch.tensor([TARGET], dtype=torch.float64))
    view = View(posterior, features(DESIGNS), target)
    exact = conditioning(view.cross, view.candidate_var, view.target_var, NOISE_VAR)
    return exact, mean_field_refit(view), (target @ posterior.mean).item()


def design_index(value: float) -> int:
    """The position in `DESIGNS` nearest `value`."""
    return int((DESIGNS - value).abs().argmin())


def gaussian_density(grid: Tensor, mean: float, var: float) -> Tensor:
    return torch.exp(-0.5 * (grid - mean) ** 2 / var) / math.sqrt(2 * math.pi * var)


def predictive_panel(ax: Axes, mean: float, moments: Moments) -> None:
    """The mean-field predictive at the target beside its average update at each marked design."""
    averaged = (moments.updated + moments.shift)[:, 0]
    cases = [  # narrowest first
        ("$Q_Y^B$", averaged[design_index(MARKED["B"])].item(), SHADES[0], "-"),
        ("$P_Y$", moments.current[0].item(), SHADES[1], "--"),
        ("$Q_Y^A$", averaged[design_index(MARKED["A"])].item(), SHADES[2], "-"),
    ]
    widest = math.sqrt(max(var for _, var, _, _ in cases))
    low, high = mean - 4 * widest + 1.5, mean + 4 * widest  # the left tail is cut short
    grid = torch.linspace(low, high, DENSITY_POINTS, dtype=torch.float64)
    densities = torch.stack([gaussian_density(grid, mean, var) for _, var, _, _ in cases])

    def draw(target: Axes, weight: float) -> None:
        target.fill_between(grid, densities.min(0).values, densities.max(0).values,
                            color=FILL, alpha=0.15, linewidth=0)  # fmt: skip
        for (_, _, color, style), density in zip(cases, densities, strict=True):
            target.plot(grid, density, color=color, linestyle=style, linewidth=weight)

    draw(ax, 1.5)
    ax.set_xlim(low, high)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Output at the target, $y_*$")
    ax.set_ylabel("Predictive density")

    # A zoom on the peaks, where the three densities separate. It enlarges a region half a unit
    # either side of the shared mean, from below the lowest peak to above the highest.
    peaks = [1 / math.sqrt(2 * math.pi * var) for _, var, _, _ in cases]
    pad = 0.6 * (max(peaks) - min(peaks))
    x1, x2 = mean - 0.5, mean + 0.5
    y1, y2 = min(peaks) - 3 * pad, max(peaks) + pad
    # The zoom keeps the shape of the region it enlarges, so its height follows from its width.
    left, bottom, width = ZOOM
    _, top = ax.get_ylim()
    height = width * ((y2 - y1) / top) / ((x2 - x1) / (high - low))
    zoom = ax.inset_axes((left, bottom, width, height))
    draw(zoom, 1.2)
    zoom.set_xlim(x1, x2)
    zoom.set_ylim(y1, y2)
    zoom.set_xticks([])
    zoom.set_yticks([])
    zoom.grid(visible=False)
    for spine in zoom.spines.values():
        spine.set_visible(True)
        spine.set_color(MUTED)
    # Each density is named in the zoom, the narrowest above its peak and the others under theirs.
    label_pad = 0.02 * (y2 - y1)
    for (text, *_), peak, above in zip(cases, peaks, (True, False, False), strict=True):
        zoom.text(mean, peak + label_pad if above else peak - label_pad, text, ha="center",
                  va="bottom" if above else "top", fontsize=ANNOTATION_SIZE,
                  color=INK_SECONDARY)  # fmt: skip
    ax.indicate_inset_zoom(zoom, edgecolor=MUTED, alpha=0.8)


def objective_panel(ax: Axes, exact: Moments, mean_field: Moments) -> None:
    """The objectives along the design axis, with the marked designs."""
    curves = [
        ("Exact Bayes, EPIG$_H$ = EMI", exact.epig(), INK),
        ("Mean-field, EPIG$_H$", mean_field.epig(), BLUE),
        ("Mean-field, EMI", mean_field.emi(), ORANGE),
    ]
    for text, values, color in curves:
        ax.plot(DESIGNS, values, color=color, label=text)
    for name, value in MARKED.items():
        index = design_index(value)
        design = DESIGNS[index].item()
        heights = [values[index].item() for _, values, _ in curves]
        for height, (_, _, color) in zip(heights, curves, strict=True):
            ax.scatter(design, height, s=MARKER_AREA, color=color, edgecolors="white",
                       linewidths=0.6, zorder=5)  # fmt: skip
        annotate(ax, design, max(heights), name, offset=(0, 5), ha="center")
    ax.set_xlim(*DESIGN_RANGE)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Candidate design $a$")
    ax.set_ylabel("Objective (nats)")
    ax.legend(loc="upper center")


def figure(exact: Moments, mean_field: Moments, mean: float) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, HEIGHT), layout="constrained")
    predictive_panel(axes[0], mean, mean_field)
    panel_label(axes[0], "a")
    objective_panel(axes[1], exact, mean_field)
    panel_label(axes[1], "b")
    return fig


def report(x: Tensor, y: Tensor, exact: Moments, mean_field: Moments) -> None:
    """What Appendix C.3 says of the example: the posterior correlation the mean-field
    approximation drops, and each objective at the two marked designs."""
    cov = Posterior.fit(features(x), y, PRIOR_VAR, NOISE_VAR).cov
    correlation = cov[0, 1] / (cov[0, 0] * cov[1, 1]).sqrt()
    a, b = design_index(MARKED["A"]), design_index(MARKED["B"])
    section("Appendix C.3, the two-parameter example (Figure 1)")
    quote("Exact posterior correlation of intercept and slope", f"{correlation:.3f}")
    for text, values in [
        ("Exact Bayes, EPIG_H", exact.epig()),
        ("Exact Bayes, EMI", exact.emi()),
        ("Mean-field, EPIG_H", mean_field.epig()),
        ("Mean-field, EMI", mean_field.emi()),
    ]:
        quote(f"{text} at A and B (nats)", f"{values[a]:.4f}, {values[b]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--png", action="store_true", help="also write a PNG beside the PDF")
    args = parser.parse_args()

    use_paper_style()
    x, y = make_data()
    exact, mean_field, mean = anticipate(x, y)
    save_figure(figure(exact, mean_field, mean), NAME, args.png)
    report(x, y, exact, mean_field)


if __name__ == "__main__":
    main()
