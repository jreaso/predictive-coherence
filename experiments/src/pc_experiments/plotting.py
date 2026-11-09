"""The figures' shared style and drawing helpers. Call `use_paper_style()` before drawing."""

import io
import math
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from torch import Tensor

from pc_experiments.outputs import figure_path

# Ink, two accents, and a blue ramp for ordered families, distinguishable under the common
# colour deficiencies.
ORANGE, AQUA = "#eb6834", "#1baf7a"
RAMP = ("#86b6ef", "#3987e5", "#256abf", "#104281")
SEQUENTIAL = LinearSegmentedColormap.from_list("ordered", RAMP)
INK, INK_SECONDARY, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NEUTRAL = "#a8a7a0"  # for a reference series such as random acquisition

# The paper's text width in inches: figures are drawn at it and typeset unscaled.
TEXT_WIDTH = 5.5

STYLE: dict[Any, Any] = {
    "font.size": 6.5,
    "axes.titlesize": 7,
    "axes.labelsize": 6.5,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.edgecolor": "0.8",
    "legend.facecolor": "white",
    "legend.framealpha": 0.8,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.35,
    "legend.borderaxespad": 0.4,
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "savefig.dpi": 200,
    "figure.facecolor": "white",
    # Laid out within the canvas, so a figure is exactly the size it asks for.
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

ANNOTATION_SIZE = 6
PANEL_OFFSET = 26.0  # points left of the axes, clear of the y-axis label
PANEL_PAD = 3.0


def use_paper_style() -> None:
    """Switch to a non-interactive backend and apply `STYLE`."""
    matplotlib.use("Agg")
    plt.rcParams.update(STYLE)


def mean_se(values: Tensor) -> tuple[float, float]:
    """Mean and standard error over the only dimension of `values`, ignoring NaNs."""
    values = values[~values.isnan()]
    if len(values) < 2:
        return math.nan, math.nan
    return values.mean().item(), (values.std() / math.sqrt(len(values))).item()


def curve_mean_se(values: Tensor) -> tuple[Tensor, Tensor]:
    """Mean and standard error over the first dimension of `(n, steps)`, each shape `(steps,)`."""
    return values.mean(0), values.std(0) / math.sqrt(values.shape[0])


def band(
    ax: Axes,
    x: Tensor,
    values: Tensor,
    color: str,
    text: str | None = None,
    linestyle: str = "-",
    weight: float = 1.5,
) -> Line2D:
    """The mean of `(n, steps)` with two standard errors shaded, returning the mean's line."""
    mean, se = curve_mean_se(values)
    (line,) = ax.plot(x, mean, color=color, label=text, linewidth=weight, linestyle=linestyle)
    ax.fill_between(x, mean - 2 * se, mean + 2 * se, color=color, alpha=0.12, linewidth=0)
    return line


def reference_line(ax: Axes, value: tuple[float, float], text: str | None = None) -> None:
    """A horizontal line at a mean, with two standard errors shaded."""
    mean, se = value
    ax.axhspan(mean - 2 * se, mean + 2 * se, color=NEUTRAL, alpha=0.15, linewidth=0)
    ax.axhline(mean, color=NEUTRAL, linewidth=1, linestyle=":", label=text)


def panel_label(ax: Axes, letter: str) -> None:
    """Mark a panel `(letter)` at its outer top-left."""
    ax.set_title(" ", loc="left", pad=PANEL_PAD)
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                xytext=(-PANEL_OFFSET, PANEL_PAD), textcoords="offset points", ha="left",
                va="baseline", color=INK, fontsize=plt.rcParams["axes.titlesize"])  # fmt: skip


def annotate(
    ax: Axes,
    position: float,
    value: float,
    text: str,
    offset: tuple[float, float] = (-6, 5),
    ha: str | None = None,
    zorder: float | None = None,
) -> None:
    """Label a point, above and to its left by default, the text aligned away from it."""
    ax.annotate(text, (position, value), textcoords="offset points", xytext=offset,
                fontsize=ANNOTATION_SIZE, color=INK_SECONDARY,
                ha=ha or ("right" if offset[0] < 0 else "left"), zorder=zorder)  # fmt: skip


def save_figure(figure: Figure, name: str, png: bool = False) -> None:
    """Write `figure` to `experiments/figures/<name>.pdf`, and a PNG beside it if `png`."""
    path = figure_path(name)
    # Constrained layout settles over successive draws, so the raster is always drawn first, kept
    # or not: the PDF is then the same file every time, as it is without its creation date.
    figure.savefig(path.with_suffix(".png") if png else io.BytesIO(), format="png")
    figure.savefig(path, metadata={"CreationDate": None})
    plt.close(figure)
    print(f"  {path.relative_to(path.parents[1])}", flush=True)
