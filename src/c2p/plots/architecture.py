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

The drawing imitates an Excalidraw sketch: every outline is a wobbly hand-drawn stroke
(Matplotlib's sketch path filter, which is deterministic) and all text is set in Excalifont,
the font of Excalidraw, shipped with the package under the SIL Open Font License (see
``c2p/plots/fonts/README.md``). The column of stages is centred in the canvas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.textpath import TextPath

from c2p.plotting import style_context

if TYPE_CHECKING:
    from chronos import Chronos2Pipeline

__all__ = [
    "FONT_PATH",
    "Chronos2Sizes",
    "Stage",
    "chronos2_sizes",
    "decision_stages",
    "hand_font",
    "plot_architecture",
    "plot_decision_path",
    "plot_stages",
]

# --- Hand-drawn look ---------------------------------------------------------------------

FONT_PATH: Path = Path(str(resources.files("c2p.plots").joinpath("fonts/Excalifont-Regular.ttf")))

# Excalidraw's own palette: near-black ink, the light blue and yellow of its colour picker.
INK = "#1e1e1e"
INK_SOFT = "#4a4d52"  # detail sentences, a step lighter than the ink so titles stand out
STROKE_BLUE = "#1971c2"
FILL_BLUE = "#a5d8ff"
FILL_YELLOW = "#ffec99"
FILL_PAPER = "#ffffff"

# Matplotlib sketch filter: wobble amplitude in points, wavelength in points, randomness.
_SKETCH = (0.9, 120.0, 5.0)
_STROKE_WIDTH = 1.5

# --- Geometry (inches; the axes maps one data unit to one inch) ----------------------------

FIGURE_WIDTH = 8.0
COLUMN_WIDTH = 6.6  # the stage boxes, centred in the figure
_MARGIN = 0.18
_PAD_X = 0.24
_PAD_Y = 0.16
_TITLE_HEIGHT = 0.3
_LINE_HEIGHT = 0.21
_SUB_GAP = 0.09
_ARROW_GAP = 0.34
_NUMBER_WIDTH = 0.44  # left strip holding the numbered circle
_NUMBER_RADIUS = 0.14
_FIGURE_TITLE_SPACE = 0.55

FIGURE_TITLE_SIZE = 14.0
TITLE_SIZE = 12.0
DETAIL_SIZE = 10.2
BADGE_SIZE = 9.2


def hand_font(size: float) -> FontProperties:
    """Excalifont at ``size`` points, loaded from the file shipped with the package."""
    return FontProperties(fname=str(FONT_PATH), size=size)


def _bold() -> list[patheffects.AbstractPathEffect]:
    """Excalifont has one weight; a thin stroke around the glyphs stands in for bold."""
    return [patheffects.withStroke(linewidth=0.75, foreground=INK)]


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
                    "Time attention",
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


# --- Text measurement and wrapping ---------------------------------------------------------


@lru_cache(maxsize=4096)
def _glyph_run_width(text: str, fontsize: float) -> float:
    """Width in inches of one word (or one character) set in Excalifont at ``fontsize``."""
    if not text:
        return 0.0
    path = TextPath((0, 0), text, size=fontsize, prop=hand_font(fontsize))
    return float(path.get_extents().width) / 72.0


def _text_width(text: str, fontsize: float) -> float:
    """Width in inches of ``text``: its words measured on the font, plus the spaces between.

    Summing per-word widths ignores kerning across a space, an error far below one glyph, and
    keeps the measurement cache small enough that wrapping costs a few dozen font lookups.
    """
    words = text.split(" ")
    space = _glyph_run_width("n n", fontsize) - 2 * _glyph_run_width("n", fontsize)
    return sum(_glyph_run_width(word, fontsize) for word in words) + space * (len(words) - 1)


def _wrap(text: str, *, fontsize: float, width: float) -> list[str]:
    """Greedy word wrap of ``text`` into lines no wider than ``width`` inches.

    Widths are measured on the actual font, so the wrapping follows the glyphs rather than a
    characters-per-inch estimate. A single word wider than ``width`` gets its own line.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and _text_width(candidate, fontsize) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


# --- Drawing -------------------------------------------------------------------------------


def _column_left() -> float:
    return (FIGURE_WIDTH - COLUMN_WIDTH) / 2


def _sketchy(patch: Patch) -> Patch:
    """Give a patch the wobbly outline of a hand-drawn shape."""
    patch.set_sketch_params(*_SKETCH)
    return patch


def _box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = INK,
    rounding: float = 0.1,
) -> None:
    ax.add_patch(
        _sketchy(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle=f"round,pad=0,rounding_size={rounding}",
                facecolor=fill,
                edgecolor=stroke,
                linewidth=_STROKE_WIDTH,
                zorder=1,
            )
        )
    )


def _sub_text(label: str, detail: str) -> str:
    return f"{label}  {detail}"


def _stage_height(stage: Stage, *, text_width: float) -> float:
    """Height in inches of one stage box, including its nested boxes."""
    lines = len(_wrap(stage.detail, fontsize=DETAIL_SIZE, width=text_width))
    height = 2 * _PAD_Y + _TITLE_HEIGHT + lines * _LINE_HEIGHT
    for label, detail in stage.sub:
        sub_lines = _wrap(
            _sub_text(label, detail), fontsize=DETAIL_SIZE, width=text_width - 2 * _PAD_X
        )
        height += _SUB_GAP + 2 * _PAD_Y + len(sub_lines) * _LINE_HEIGHT
    return height


def _draw_stage(ax: Axes, stage: Stage, number: int, top: float, *, text_width: float) -> None:
    """Draw one stage with its top edge at ``top`` (inches from the bottom of the axes)."""
    height = _stage_height(stage, text_width=text_width)
    bottom = top - height
    x0 = _column_left()
    _box(
        ax,
        x0,
        bottom,
        COLUMN_WIDTH,
        height,
        fill=FILL_BLUE if stage.highlight else FILL_PAPER,
        stroke=STROKE_BLUE if stage.highlight else INK,
    )
    left = x0 + _PAD_X
    title_y = top - _PAD_Y - _TITLE_HEIGHT / 2
    _draw_number(ax, number, left + _NUMBER_RADIUS, title_y)
    ax.text(
        left + _NUMBER_WIDTH,
        title_y,
        stage.title,
        fontproperties=hand_font(TITLE_SIZE),
        color=INK,
        path_effects=_bold(),
        va="center",
        ha="left",
        zorder=3,
    )
    if stage.badge is not None:
        _draw_badge(ax, stage.badge, title_y)
    y = top - _PAD_Y - _TITLE_HEIGHT
    y = _draw_lines(
        ax, _wrap(stage.detail, fontsize=DETAIL_SIZE, width=text_width), left + _NUMBER_WIDTH, y
    )
    for label, detail in stage.sub:
        y = _draw_sub(ax, label, detail, y - _SUB_GAP, text_width=text_width)


def _draw_number(ax: Axes, number: int, cx: float, cy: float) -> None:
    """The step number inside a hand-drawn circle."""
    ax.add_patch(
        _sketchy(
            Ellipse(
                (cx, cy),
                2 * _NUMBER_RADIUS,
                2 * _NUMBER_RADIUS,
                facecolor=FILL_YELLOW,
                edgecolor=INK,
                linewidth=_STROKE_WIDTH,
                zorder=2,
            )
        )
    )
    ax.text(
        cx,
        cy - 0.01,
        str(number),
        fontproperties=hand_font(TITLE_SIZE - 1),
        color=INK,
        path_effects=_bold(),
        va="center",
        ha="center",
        zorder=3,
    )


def _draw_lines(ax: Axes, lines: Sequence[str], x: float, y: float) -> float:
    """Draw wrapped detail lines downwards from ``y``; return the y below the last one."""
    for line in lines:
        ax.text(
            x,
            y - _LINE_HEIGHT / 2,
            line,
            fontproperties=hand_font(DETAIL_SIZE),
            color=INK_SOFT,
            va="center",
            ha="left",
            zorder=3,
        )
        y -= _LINE_HEIGHT
    return y


def _draw_sub(ax: Axes, label: str, detail: str, top: float, *, text_width: float) -> float:
    """Draw one nested box of the encoder stage; return the y below it."""
    width = text_width - 2 * _PAD_X
    lines = _wrap(_sub_text(label, detail), fontsize=DETAIL_SIZE, width=width)
    height = 2 * _PAD_Y + len(lines) * _LINE_HEIGHT
    x = _column_left() + _PAD_X + _NUMBER_WIDTH
    _box(
        ax,
        x,
        top - height,
        COLUMN_WIDTH - 2 * _PAD_X - _NUMBER_WIDTH,
        height,
        fill=FILL_PAPER,
        rounding=0.07,
    )
    y = top - _PAD_Y
    for index, line in enumerate(lines):
        if index == 0 and line.startswith(label):
            bold, rest = label, line[len(label) :]
        else:
            bold, rest = "", line
        text = ax.text(
            x + _PAD_X,
            y - _LINE_HEIGHT / 2,
            bold,
            fontproperties=hand_font(DETAIL_SIZE),
            color=INK,
            path_effects=_bold(),
            va="center",
            ha="left",
            zorder=3,
        )
        ax.annotate(
            rest,
            xycoords=text,
            xy=(1.0, 0.5),
            fontproperties=hand_font(DETAIL_SIZE),
            color=INK_SOFT,
            va="center",
            ha="left",
            zorder=3,
        )
        y -= _LINE_HEIGHT
    return top - height


def _draw_badge(ax: Axes, text: str, y: float) -> None:
    """Small sticky-note label at the right edge of a stage title."""
    right = _column_left() + COLUMN_WIDTH - _PAD_X
    width = _text_width(text, BADGE_SIZE) + 0.22
    height = 0.27
    _box(ax, right - width, y - height / 2, width, height, fill=FILL_YELLOW, rounding=0.05)
    ax.text(
        right - width / 2,
        y,
        text,
        fontproperties=hand_font(BADGE_SIZE),
        color=INK,
        path_effects=_bold(),
        va="center",
        ha="center",
        zorder=4,
    )


def _draw_arrow(ax: Axes, top: float) -> None:
    """Hand-drawn downward arrow in the middle of the gap between two stages."""
    x = FIGURE_WIDTH / 2
    ax.add_patch(
        _sketchy(
            FancyArrowPatch(
                (x, top),
                (x, top - _ARROW_GAP),
                arrowstyle="-|>,head_length=4,head_width=3",
                color=INK,
                linewidth=_STROKE_WIDTH,
                shrinkA=2.0,
                shrinkB=2.0,
                zorder=2,
            )
        )
    )


DEFAULT_TITLE = "One forward pass: Chronos-2 reads a whole group of series at once"


def plot_architecture(
    *, sizes: Chronos2Sizes | None = None, title: str | None = DEFAULT_TITLE
) -> Figure:
    """Draw the six stages of a Chronos-2 forward pass as a centred top-to-bottom sketch.

    Parameters
    ----------
    sizes
        Sizes printed in the boxes; ``None`` uses the published ``amazon/chronos-2`` values.
        Pass ``chronos2_sizes(pipeline)`` to read them from a loaded checkpoint instead.
    title
        Figure title, centred above the column, or ``None`` for no title.
    """
    return plot_stages(_stages(sizes or Chronos2Sizes()), title=title)


def plot_stages(stages: Sequence[Stage], *, title: str | None = None) -> Figure:
    """Draw numbered stages top to bottom, joined by arrows, in the hand-drawn style.

    The generic form of :func:`plot_architecture`: any sequence of :class:`Stage` becomes a
    column of numbered boxes, each with its title, a sentence, optional nested boxes
    (``sub``), an optional ``badge`` next to the title and an optional ``highlight`` fill.
    """
    if not stages:
        raise ValueError("A diagram needs at least one stage.")
    text_width = COLUMN_WIDTH - 2 * _PAD_X - _NUMBER_WIDTH
    heights = [_stage_height(stage, text_width=text_width) for stage in stages]
    title_space = _FIGURE_TITLE_SPACE if title is not None else 0.0
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
                FIGURE_WIDTH / 2,
                height - _MARGIN - 0.1,
                title,
                fontproperties=hand_font(FIGURE_TITLE_SIZE),
                color=INK,
                path_effects=_bold(),
                va="top",
                ha="center",
            )
    return fig


# --- The decision path of the series -------------------------------------------------------

DECISION_TITLE = "Chronos-2 on your data: the order in which to pull the levers"


def decision_stages(measured: Mapping[str, str]) -> list[Stage]:
    """The eight steps the series arrives at, each with the number that was measured for it.

    ``measured`` maps a step key to the sentence printed in that step's "Measured" box, so the
    drawing states what the results files say and nothing else: ``data``, ``baseline``,
    ``covariates``, ``context``, ``finetune``, ``calibration``, ``serving`` and
    ``monitoring``. ``scripts/series_summary.py`` builds it from ``results/``.
    """
    steps = (
        (
            "data",
            "Put the series on a regular grid",
            "A missing row shifts everything before it and nothing warns you. Reindex to the "
            "frequency, leave gaps as NaN, never fill with 0, and do not pass freq unless you "
            "have checked the grid yourself.",
        ),
        (
            "baseline",
            "Measure before you change anything",
            "A rolling-origin backtest against the seasonal naive, a competent classical model "
            "and any professional forecast you have. Sample every weekday, and resample "
            "consecutive days in blocks.",
        ),
        (
            "covariates",
            "Give it what is known about the future",
            "A covariate pays when you know its future values: published forecasts, prices, "
            "promotions, a holiday flag. Skip the periodic calendar. If a professional "
            "forecast of the target exists, forecast its errors instead.",
        ),
        (
            "context",
            "Set the context length once",
            "Sweep it on your own series. 2,048 steps is a good default: the gain flattens "
            "after about a thousand, and every doubling costs throughput.",
        ),
        (
            "finetune",
            "Fine-tune last, with a validation period and several seeds",
            "LoRA first. Choose the budget on a validation backtest, not on the training loss, "
            "and repeat the run with other seeds before you believe a gain of a few percent.",
        ),
        (
            "calibration",
            "Calibrate the intervals, and keep calibrating",
            "A conformal correction costs no training. Learned once, it fixes the yearly "
            "coverage; updated with each resolved forecast, it also holds the months it has "
            "not seen.",
        ),
        (
            "serving",
            "Serve it where the idle time is cheapest",
            "A batch job for a scheduled forecast, serverless for bursty low volume, a GPU "
            "endpoint for high steady volume. Batch the requests on a GPU.",
        ),
        (
            "monitoring",
            "Watch the error and the coverage",
            "Log every forecast with its input hash, join the actuals when they arrive, alert "
            "once per episode, and look at coverage by month.",
        ),
    )
    missing = [key for key, _, _ in steps if key not in measured]
    if missing:
        raise ValueError(f"measured lacks the steps {missing}.")
    return [
        Stage(title, detail, sub=(("Measured", measured[key]),)) for key, title, detail in steps
    ]


def plot_decision_path(
    measured: Mapping[str, str], *, title: str | None = DECISION_TITLE
) -> Figure:
    """Draw the decision path of the series: eight numbered steps with their measured effect."""
    return plot_stages(decision_stages(measured), title=title)
