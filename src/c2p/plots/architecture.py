"""Schematic of how Chronos-2 turns a group of series into quantile forecasts (article 1).

Unlike every other figure of the series, this one draws a *description*, not a measurement: the
stages come from the Chronos-2 technical report (Ansari et al., 2025,
`arXiv:2510.15821 <https://arxiv.org/abs/2510.15821>`_, section 3) and the published
``config.json`` of ``amazon/chronos-2``. The sizes it prints (context length, horizon, quantile
levels, patch size, encoder shape) are *arguments*, so a notebook can pass the values read from
the loaded pipeline and the drawing cannot quietly disagree with the checkpoint::

    from c2p.plots.architecture import chronos2_sizes, plot_architecture

    fig = plot_architecture(sizes=chronos2_sizes(pipeline))

The defaults describe ``amazon/chronos-2`` as published, for use without a loaded model.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from c2p.plotting import AXIS_COLOR, BLUE, INK_PRIMARY, INK_SECONDARY, SURFACE, style_context

if TYPE_CHECKING:
    from chronos import Chronos2Pipeline

__all__ = ["Chronos2Sizes", "Stage", "chronos2_sizes", "plot_architecture"]

# --- Geometry (inches; the axes maps one data unit to one inch) ----------------------------

FIGURE_WIDTH = 8.0
_MARGIN = 0.1
_PAD_X = 0.22
_PAD_Y = 0.13
_TITLE_HEIGHT = 0.27
_LINE_HEIGHT = 0.19
_SUB_GAP = 0.07
_ARROW_GAP = 0.2
_NUMBER_WIDTH = 0.34  # left strip holding the step number

TITLE_SIZE = 10.5
DETAIL_SIZE = 9.3
BADGE_SIZE = 8.2

_HIGHLIGHT_FILL = "#eef4fd"  # lightest step of the blue ramp, for the new encoder block
_STAGE_FILL = "#f6f5f1"  # neutral surface, a shade off white
_SUB_FILL = SURFACE
# Characters per inch at a given point size, measured on the style's sans-serif faces; used
# only to wrap text, so a small error costs a line break, never a wrong number.
_CHARS_PER_POINT_INCH = 72 / 0.52


@dataclass(frozen=True)
class Chronos2Sizes:
    """The sizes the diagram prints. Defaults describe ``amazon/chronos-2`` as published."""

    context_length: int = 8192
    prediction_length: int = 1024
    n_quantiles: int = 21
    quantile_low: float = 0.01
    quantile_high: float = 0.99
    patch_size: int = 16
    n_blocks: int = 12
    d_model: int = 768

    @property
    def n_output_patches(self) -> int:
        """Output patches behind the maximum horizon, e.g. 1,024 / 16 = 64."""
        patches, remainder = divmod(self.prediction_length, self.patch_size)
        if remainder:
            raise ValueError(
                f"prediction_length={self.prediction_length} is not a whole number of "
                f"patch_size={self.patch_size} steps."
            )
        return patches


def chronos2_sizes(pipeline: Chronos2Pipeline) -> Chronos2Sizes:
    """Read the diagram's sizes from a loaded pipeline, so the figure follows the checkpoint.

    Values the pipeline exposes directly (context length, horizon, patch size, quantile levels)
    are taken from it; the encoder shape comes from the inner model's ``config``.
    """
    config = pipeline.inner_model.config
    quantiles = sorted(float(level) for level in pipeline.quantiles)
    return Chronos2Sizes(
        context_length=int(pipeline.model_context_length),
        prediction_length=int(pipeline.model_prediction_length),
        n_quantiles=len(quantiles),
        quantile_low=quantiles[0],
        quantile_high=quantiles[-1],
        patch_size=int(pipeline.model_output_patch_size),
        n_blocks=int(config.num_layers),
        d_model=int(config.d_model),
    )


@dataclass(frozen=True)
class Stage:
    """One numbered box of the diagram: a title, a sentence and optional nested boxes."""

    title: str
    detail: str
    sub: Sequence[tuple[str, str]] = field(default_factory=tuple)
    badge: str | None = None
    highlight: bool = False


def _stages(sizes: Chronos2Sizes) -> list[Stage]:
    """The six stages of a forward pass, with the sizes filled in."""
    return [
        Stage(
            "Grouped inputs",
            "The target series, past covariates and known-future covariates. Every series "
            "carries a group ID, and that ID is the whole trick: it decides which other series "
            "a series is allowed to see.",
        ),
        Stage(
            "Robust scaling",
            "Each series is standardised with its own mean and standard deviation, then passed "
            "through arcsinh, which compresses extremes. Missing values become 0 plus an "
            "observed / not-observed mask, and never enter the mean.",
        ),
        Stage(
            f"Patches of {sizes.patch_size} steps, then embedding",
            f"Values, the time index and the mask go through a residual network into "
            f"{sizes.d_model:,} dimensions. A REG token separates the context, at most "
            f"{sizes.context_length:,} steps, from the patches that stand for the future.",
        ),
        Stage(
            f"T5-style encoder: {sizes.n_blocks} blocks, d_model {sizes.d_model:,}",
            "Every block runs both kinds of attention, one after the other:",
            sub=(
                (
                    "Temporal attention",
                    "across the patches of one series, positions encoded with RoPE",
                ),
                (
                    "Group attention",
                    "across the series of one group, at each patch index in turn",
                ),
            ),
            badge="New: group attention",
            highlight=True,
        ),
        Stage(
            "Quantile head",
            f"{sizes.n_quantiles} quantile levels, {sizes.quantile_low:g} to "
            f"{sizes.quantile_high:g}, for every step of up to {sizes.n_output_patches} output "
            f"patches: a horizon of at most {sizes.prediction_length:,} steps in one pass.",
        ),
        Stage(
            "Un-scaling, and a probabilistic forecast",
            "The inverse transform returns every quantile to the original scale of its own "
            "series. Nothing is sampled: the quantiles are predicted directly.",
        ),
    ]


# --- Drawing -------------------------------------------------------------------------------


def _wrap(text: str, *, fontsize: float, width: float) -> list[str]:
    """Split ``text`` into lines that fit ``width`` inches at ``fontsize`` points."""
    return textwrap.wrap(text, width=max(8, int(width * _CHARS_PER_POINT_INCH / fontsize)))


def _sub_lines(label: str, detail: str, width: float) -> list[str]:
    return _wrap(f"{label} - {detail}", fontsize=DETAIL_SIZE, width=width)


def _stage_height(stage: Stage, *, text_width: float) -> float:
    """Height in inches of one stage box, including its nested boxes."""
    lines = len(_wrap(stage.detail, fontsize=DETAIL_SIZE, width=text_width))
    height = 2 * _PAD_Y + _TITLE_HEIGHT + lines * _LINE_HEIGHT
    for label, detail in stage.sub:
        sub_lines = _sub_lines(label, detail, text_width - 2 * _PAD_X)
        height += _SUB_GAP + 2 * _PAD_Y + len(sub_lines) * _LINE_HEIGHT
    return height


def _box(ax: Axes, x: float, y: float, width: float, height: float, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=color,
            edgecolor="none",
            zorder=1,
        )
    )


def _draw_stage(ax: Axes, stage: Stage, number: int, top: float, *, text_width: float) -> None:
    """Draw one stage with its top edge at ``top`` (inches from the bottom of the axes)."""
    height = _stage_height(stage, text_width=text_width)
    bottom = top - height
    _box(
        ax,
        _MARGIN,
        bottom,
        FIGURE_WIDTH - 2 * _MARGIN,
        height,
        _HIGHLIGHT_FILL if stage.highlight else _STAGE_FILL,
    )
    if stage.highlight:
        ax.add_patch(
            Rectangle((_MARGIN, bottom), 0.055, height, facecolor=BLUE, edgecolor="none", zorder=2)
        )
    left = _MARGIN + _PAD_X
    ax.text(
        left,
        top - _PAD_Y - _TITLE_HEIGHT / 2,
        str(number),
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=BLUE,
        va="center",
        ha="left",
        zorder=3,
    )
    ax.text(
        left + _NUMBER_WIDTH,
        top - _PAD_Y - _TITLE_HEIGHT / 2,
        stage.title,
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=INK_PRIMARY,
        va="center",
        ha="left",
        zorder=3,
    )
    if stage.badge is not None:
        _draw_badge(ax, stage.badge, top - _PAD_Y - _TITLE_HEIGHT / 2)
    y = top - _PAD_Y - _TITLE_HEIGHT
    y = _draw_lines(ax, _wrap(stage.detail, fontsize=DETAIL_SIZE, width=text_width), left, y)
    for label, detail in stage.sub:
        y = _draw_sub(ax, label, detail, y - _SUB_GAP, text_width=text_width)


def _draw_lines(ax: Axes, lines: Sequence[str], x: float, y: float) -> float:
    """Draw wrapped detail lines downwards from ``y``; return the y below the last one."""
    for line in lines:
        ax.text(
            x,
            y - _LINE_HEIGHT / 2,
            line,
            fontsize=DETAIL_SIZE,
            color=INK_SECONDARY,
            va="center",
            ha="left",
            zorder=3,
        )
        y -= _LINE_HEIGHT
    return y


def _draw_sub(ax: Axes, label: str, detail: str, top: float, *, text_width: float) -> float:
    """Draw one nested box of the encoder stage; return the y below it."""
    width = text_width - 2 * _PAD_X
    lines = _sub_lines(label, detail, width)
    height = 2 * _PAD_Y + len(lines) * _LINE_HEIGHT
    x = _MARGIN + _PAD_X + _NUMBER_WIDTH
    _box(ax, x, top - height, FIGURE_WIDTH - _MARGIN - _PAD_X - x, height, _SUB_FILL)
    y = top - _PAD_Y
    for index, line in enumerate(lines):
        bold, _, rest = line.partition(" - ") if index == 0 else ("", "", line)
        text = ax.text(
            x + _PAD_X,
            y - _LINE_HEIGHT / 2,
            bold,
            fontsize=DETAIL_SIZE,
            fontweight="bold",
            color=INK_PRIMARY,
            va="center",
            ha="left",
            zorder=3,
        )
        ax.annotate(
            f"  {rest}" if bold else rest,
            xycoords=text,
            xy=(1.0, 0.5),
            fontsize=DETAIL_SIZE,
            color=INK_SECONDARY,
            va="center",
            ha="left",
            zorder=3,
        )
        y -= _LINE_HEIGHT
    return top - height


def _draw_badge(ax: Axes, text: str, y: float) -> None:
    """Small filled label at the right edge of a stage title."""
    right = FIGURE_WIDTH - _MARGIN - _PAD_X
    ax.text(
        right,
        y,
        text,
        fontsize=BADGE_SIZE,
        fontweight="bold",
        color=SURFACE,
        va="center",
        ha="right",
        zorder=4,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": BLUE, "edgecolor": "none"},
    )


def _draw_arrow(ax: Axes, top: float) -> None:
    """Short downward arrow filling the gap between two stages."""
    x = _MARGIN + _PAD_X + _NUMBER_WIDTH / 2
    ax.add_patch(
        FancyArrowPatch(
            (x, top),
            (x, top - _ARROW_GAP),
            arrowstyle="-|>",
            mutation_scale=9,
            color=AXIS_COLOR,
            linewidth=1.0,
            shrinkA=1.0,
            shrinkB=1.0,
            zorder=2,
        )
    )


DEFAULT_TITLE = "One forward pass: Chronos-2 reads a whole group of series at once"


def plot_architecture(
    *, sizes: Chronos2Sizes | None = None, title: str | None = DEFAULT_TITLE
) -> Figure:
    """Draw the six stages of a Chronos-2 forward pass as a top-to-bottom flow.

    Parameters
    ----------
    sizes
        Sizes printed in the boxes; ``None`` uses the published ``amazon/chronos-2`` values.
        Pass ``chronos2_sizes(pipeline)` to read them from a loaded checkpoint instead.
    title
        Figure title, or ``None`` for no title.
    """
    sizes = sizes or Chronos2Sizes()
    stages = _stages(sizes)
    text_width = FIGURE_WIDTH - 2 * (_MARGIN + _PAD_X) - _NUMBER_WIDTH
    heights = [_stage_height(stage, text_width=text_width) for stage in stages]
    title_space = 0.42 if title is not None else 0.0
    height = sum(heights) + _ARROW_GAP * (len(stages) - 1) + title_space + 2 * _MARGIN
    with style_context():
        fig = plt.figure(figsize=(FIGURE_WIDTH, height))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0.0, FIGURE_WIDTH)
        ax.set_ylim(0.0, height)
        ax.set_aspect("equal")
        ax.axis("off")
        top = height - _MARGIN - title_space
        for number, (stage, stage_height) in enumerate(zip(stages, heights, strict=True), start=1):
            _draw_stage(ax, stage, number, top, text_width=text_width)
            top -= stage_height
            if number < len(stages):
                _draw_arrow(ax, top)
                top -= _ARROW_GAP
        if title is not None:
            ax.text(
                _MARGIN,
                height - _MARGIN,
                title,
                fontsize=13,
                fontweight="bold",
                color=INK_PRIMARY,
                va="top",
                ha="left",
            )
    return fig
