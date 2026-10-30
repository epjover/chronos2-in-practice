"""Article figures: one consistent matplotlib style for static PNGs published on Medium.

Every figure in the series is drawn by a function of this module from DataFrames that follow
the shared contract in ``c2p.schema``, then written with :func:`save_figure`::

    from c2p.plotting import plot_fan, save_figure, set_style

    set_style()  # once per notebook, so inline previews match the saved PNGs
    ax = plot_fan(context, forecast, actuals=actuals, title="Chronos-2 tracks the daily cycle")
    save_figure(ax.figure, "fan_chart", PATHS.figures)

Style rules (light surface, read on desktop and phone):

- Categorical colours come from :data:`PALETTE` in fixed order and are never cycled past eight.
  Colour follows the entity: each model keeps its :data:`MODEL_COLORS` entry in every figure.
  Slots 3 (aqua) and 4 (yellow) have less than 3:1 contrast on white, so series drawn in them
  always carry a legend entry.
- Text uses ink colours only (never a series colour); grids and axes are recessive solid
  hairlines; one y-axis per panel, never dual axes.
- Prediction intervals are nested washes of one hue (blue), widest band lightest.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.rcsetup import cycler
from matplotlib.ticker import Formatter, NullLocator
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset

from c2p.schema import (
    ID_COL,
    INTERVALS,
    PREDICTION_COL,
    TARGET_COL,
    TARGET_NAME_COL,
    TIMESTAMP_COL,
    quantile_col,
)

__all__ = [
    "MODEL_COLORS",
    "PALETTE",
    "STYLE",
    "assign_colors",
    "format_compact",
    "plot_benchmarks",
    "plot_fan",
    "plot_gap_grid",
    "plot_metric_vs_param",
    "plot_scenarios",
    "save_figure",
    "set_style",
    "style_context",
]

# --- Colours ------------------------------------------------------------------------------

# Categorical slots in their validated order: blue, orange, aqua, yellow, magenta, green,
# violet, red. Adjacent pairs stay distinguishable under colour-vision deficiencies.
PALETTE: tuple[str, ...] = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)  # fmt: skip
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = PALETTE

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
SURFACE = "#ffffff"

# The forecaster names of ``Forecaster.name`` mapped to fixed slots, so a model is the same
# colour in every figure of the series. Chronos-2 takes the leading blue; the baselines take
# the next slots. New models take the next free slot (magenta, green, violet, red).
MODEL_COLORS: Mapping[str, str] = MappingProxyType(
    {
        "chronos-2": BLUE,
        "seasonal_naive": ORANGE,
        "auto_ets": AQUA,
        "chronos-bolt": YELLOW,
    }
)

NEUTRAL_BAR = AXIS_COLOR  # non-highlighted bars in emphasis charts
OBSERVED_CELL = "#dddcd4"  # light gray: a value is there
ISSUE_COLOR = RED  # data problems: missing cells, injected spikes
GAP_SPAN = "#eceae4"  # background span behind missing context

# Nested interval washes (narrowest first): colour and alpha from the sequential blue ramp.
_BAND_FILLS: Mapping[int, tuple[tuple[str, float], ...]] = MappingProxyType(
    {
        1: (("#86b6ef", 0.55),),
        2: (("#86b6ef", 0.55), ("#cde2fb", 0.7)),
        3: (("#6da7ec", 0.5), ("#9ec5f4", 0.55), ("#cde2fb", 0.7)),
    }
)

# --- Style --------------------------------------------------------------------------------

LINE_WIDTH = 1.6
MARKER_SIZE = 6.0
FONT_FAMILY: tuple[str, ...] = ("Segoe UI", "DejaVu Sans")  # first one found wins, no warning
LEGEND_FONT_SIZE = 9.5

STYLE: Mapping[str, object] = MappingProxyType(
    {
        "font.family": "sans-serif",
        "font.sans-serif": list(FONT_FAMILY),
        "font.size": 10.5,
        "text.color": INK_PRIMARY,
        "figure.facecolor": SURFACE,
        "figure.edgecolor": SURFACE,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "figure.labelsize": 10,
        "figure.labelweight": "normal",
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS_COLOR,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlelocation": "left",
        "axes.titlepad": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.xmargin": 0.0,
        "axes.ymargin": 0.05,
        "axes.prop_cycle": cycler(color=list(PALETTE)),
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.size": 0.0,
        "legend.frameon": False,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.labelcolor": INK_SECONDARY,
        "legend.handlelength": 1.6,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "patch.linewidth": 0.0,
        "date.converter": "concise",
        "savefig.dpi": 200,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    }
)


def set_style() -> None:
    """Apply the series style to matplotlib's global ``rcParams`` (idempotent)."""
    mpl.rcParams.update(STYLE)


@contextmanager
def style_context() -> Iterator[None]:
    """Apply the series style temporarily; ``rcParams`` are restored on exit."""
    with mpl.rc_context(dict(STYLE)):
        yield


def save_figure(
    fig: Figure,
    name: str,
    folder: Path,
    *,
    formats: Sequence[str] = ("png",),
    dpi: int = 200,
) -> list[Path]:
    """Write ``fig`` as ``folder/name.<format>`` for each format, then close it.

    ``name`` is a file stem made of letters, digits, ``_``, ``-`` and ``.``, without the
    extension (portable across Windows, macOS and Linux). Fonts are resolved while saving, so
    the files use the series style even when the global ``rcParams`` differ. The figure is
    left open only if the arguments are invalid.

    Returns
    -------
    list[Path]
        The written files, in the order of ``formats``.
    """
    _check_figure_name(fig, name)
    _check_formats(fig, formats)
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        with style_context():
            for fmt in formats:
                path = folder / f"{name}.{fmt}"
                fig.savefig(path, format=fmt, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
                paths.append(path)
    finally:
        plt.close(fig)
    return paths


_FIGURE_NAME = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*")
# Device names Windows reserves even with an extension ("con.png" cannot be created).
_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"{port}{i}" for port in ("COM", "LPT") for i in range(1, 10))}
)


def _check_figure_name(fig: Figure, name: str) -> None:
    if not _FIGURE_NAME.fullmatch(name):
        raise ValueError(
            f"Figure name {name!r} must be a file stem of letters, digits, '_', '-' and '.' "
            "(no spaces, ':' or folders); pass the folder separately."
        )
    if name.split(".", maxsplit=1)[0].upper() in _RESERVED_STEMS:
        raise ValueError(f"Figure name {name!r} is a reserved device name on Windows.")
    suffix = name.rpartition(".")[2].lower() if "." in name else ""
    if suffix in fig.canvas.get_supported_filetypes():
        raise ValueError(
            f"Figure name {name!r} already ends in '.{suffix}'; pass the stem and pick the "
            "file types with formats=."
        )


def _check_formats(fig: Figure, formats: Sequence[str]) -> None:
    if isinstance(formats, str) or not formats:
        raise ValueError(f"formats must be a non-empty sequence such as ('png',), got {formats!r}.")
    supported = fig.canvas.get_supported_filetypes()
    unknown = [fmt for fmt in formats if fmt not in supported]
    if unknown:
        raise ValueError(f"Unsupported figure formats {unknown}; choose from {sorted(supported)}.")


_UNITS: tuple[tuple[float, str], ...] = ((1e9, "B"), (1e6, "M"), (1e3, "K"))


def format_compact(value: float, _position: int | None = None) -> str:
    """Format a value compactly: ``25000 -> "25K"``, ``1.5e6 -> "1.5M"``, ``0.05 -> "0.05"``.

    Up to two decimals of the unit are kept (25,050 -> "25.05K"); a value that rounds up to
    1,000 of a unit moves to the next one (999,999 -> "1M"). The signature matches
    :class:`matplotlib.ticker.FuncFormatter`. The y axes of this module's charts switch to
    plain numbers (25,000 / 25,002) when two ticks would otherwise get the same label.
    """
    magnitude = abs(value)
    for index, (threshold, suffix) in enumerate(_UNITS):
        if magnitude < threshold:
            continue
        unit_threshold, unit_suffix = threshold, suffix
        text = f"{value / unit_threshold:.2f}"
        if abs(float(text)) >= 1000 and index > 0:  # 999,999 rounds up: show 1M, not 1000K
            unit_threshold, unit_suffix = _UNITS[index - 1]
            text = f"{value / unit_threshold:.2f}"
        return text.rstrip("0").rstrip(".") + unit_suffix
    return "0" if value == 0 else f"{value:g}"


def _distinct_plain_labels(values: Sequence[float]) -> list[str]:
    """Thousands-separated labels with the fewest decimals (up to 6) that tell ticks apart."""
    labels: list[str] = []
    for decimals in range(7):
        labels = [f"{round(value, decimals) + 0.0:,.{decimals}f}" for value in values]
        if len(set(labels)) == len(labels):
            break
    return labels


class _CompactFormatter(Formatter):
    """Tick labels from :func:`format_compact`, or plain numbers when compact ones repeat.

    A narrow y range such as 25,000-25,008 would otherwise label every tick "25K".
    """

    def __call__(self, x: float, pos: int | None = None) -> str:
        return format_compact(x)

    def format_ticks(self, values: Sequence[float]) -> list[str]:
        self.set_locs(values)
        labels = [format_compact(value) for value in values]
        if len(set(labels)) == len(labels):
            return labels
        return _distinct_plain_labels(values)


def assign_colors(
    names: Sequence[str], *, fixed: Mapping[str, str] = MODEL_COLORS
) -> dict[str, str]:
    """Map series names to colours so that colour follows the entity.

    Names found in ``fixed`` keep their colour, and the slots of ``fixed`` stay reserved even
    when none of its names is present, so a new model never borrows another model's colour.
    The other names take the free :data:`PALETTE` slots in palette order, by order of
    appearance; add a recurring model to :data:`MODEL_COLORS` to pin its colour. For series
    that are not models (cities, targets...), pass ``fixed={}`` to start at slot 1, blue.

    Raises
    ------
    ValueError
        When there are more names than free slots: fold the tail into "Other" or facet.
    """
    unique = list(dict.fromkeys(names))
    known = {name: fixed[name] for name in unique if name in fixed}
    reserved = set(fixed.values())
    free = [color for color in PALETTE if color not in reserved]
    unknown = [name for name in unique if name not in fixed]
    if len(unknown) > len(free):
        raise ValueError(
            f"{len(unknown)} series need a colour but only {len(free)} palette slots are free; "
            "fold the smallest series into 'Other' or use small multiples."
        )
    return {**known, **dict(zip(unknown, free, strict=False))}


# --- Shared helpers -----------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing columns {missing}; it has {list(frame.columns)}.")


def _check_timestamps(frame: pd.DataFrame, what: str) -> None:
    dtype = frame[TIMESTAMP_COL].dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        raise TypeError(
            f"{what}['{TIMESTAMP_COL}'] is timezone-aware ({dtype}); the schema uses "
            "timezone-naive UTC (run c2p.data.validate_and_prepare first)."
        )
    if not pd.api.types.is_datetime64_dtype(dtype):
        raise TypeError(f"{what}['{TIMESTAMP_COL}'] must be datetime64, got {dtype}.")


def _resolve_item(frame: pd.DataFrame, item_id: Hashable | None, what: str) -> Hashable:
    """The requested series id, or the only one; returned as stored (not cast to str)."""
    _require_columns(frame, [ID_COL], what)
    ids = list(pd.unique(frame[ID_COL]))
    if item_id is None:
        if len(ids) != 1:
            raise ValueError(f"{what} holds {len(ids)} series {ids[:5]}; pass item_id=.")
        return ids[0]
    if item_id not in ids:
        raise ValueError(f"item_id {item_id!r} not found in {what}; available: {ids[:10]}.")
    return item_id


def _item_rows(frame: pd.DataFrame, item_id: Hashable, what: str) -> pd.DataFrame:
    _require_columns(frame, [ID_COL, TIMESTAMP_COL], what)
    _check_timestamps(frame, what)
    rows = frame.loc[frame[ID_COL] == item_id]
    if rows.empty:
        raise ValueError(f"{what} has no rows for item_id {item_id!r}.")
    return rows


def _series(rows: pd.DataFrame, column: str, what: str) -> pd.Series:
    """Return ``column`` as a float64 series indexed by sorted, unique timestamps."""
    _require_columns(rows, [column], what)
    series = pd.Series(
        rows[column].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(rows[TIMESTAMP_COL]),
        name=column,
    ).sort_index()
    if series.index.has_duplicates:
        duplicated = series.index[series.index.duplicated()][:3].tolist()
        raise ValueError(f"{what} has duplicated timestamps, e.g. {duplicated}.")
    return series


def _grid_offset(index: pd.DatetimeIndex, freq: str | None, what: str) -> BaseOffset:
    """The grid frequency of ``index``: ``freq`` when given, else inferred; checked on ``index``."""
    offset = to_offset(freq) if freq is not None else _infer_offset(index, what)
    expected = pd.date_range(index.min(), index.max(), freq=offset)
    off_grid = index.difference(expected)
    if len(off_grid):
        raise ValueError(
            f"{what} has {len(off_grid)} timestamps off the {offset.freqstr!r} grid that starts "
            f"at {index.min()}, e.g. {off_grid[:3].tolist()}; pass the right freq= or "
            "resample first."
        )
    return offset


def _infer_offset(index: pd.DatetimeIndex, what: str) -> BaseOffset:
    """Infer the frequency of a complete grid; for one with missing rows, a sub-daily step.

    Steps of a day or more are ambiguous once rows are missing (daily or business days?
    month starts?), so they raise and ask for ``freq`` instead of guessing.
    """
    if len(index) < 2:
        raise ValueError(f"{what} needs at least two timestamps to infer its frequency.")
    inferred = pd.DatetimeIndex(index.to_numpy(), freq="infer").freq
    if inferred is not None:
        return inferred
    step = pd.Timedelta(pd.Series(index[1:] - index[:-1]).mode().iloc[0])
    if step >= pd.Timedelta(days=1):
        raise ValueError(
            f"Could not infer the frequency of {what}: it has missing rows (or too few to "
            f"tell) and its usual step is {step}, which fits several calendars; pass freq= "
            "(e.g. 'D', 'B' or 'MS')."
        )
    return to_offset(step)


def _fixed_step(freq: str) -> pd.Timedelta:
    try:
        return pd.Timedelta(to_offset(freq))
    except ValueError as error:
        raise ValueError(
            f"freq={freq!r} is not a fixed duration (use e.g. 'h', '30min' or 'D')."
        ) from error


def _percent(level: float) -> str:
    return f"{level * 100:g}%"


def _style_axes(ax: Axes, *, grid: Literal["x", "y", "none"]) -> None:
    """Apply the recessive chrome explicitly, so axes look right outside ``style_context``."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid != "none":
        ax.grid(True, axis=grid, color=GRID_COLOR, linewidth=0.6, linestyle="-")


def _format_time_axis(ax: Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


_TITLE_GAP_POINTS = 14.0  # between the top of the legend and the title baseline


def _legend_above(
    ax: Axes, handles: Sequence[Artist], labels: Sequence[str], title: str | None
) -> None:
    """Legend between the plot and the title, so it never covers data.

    Entries read left to right in as few rows as fit the axes width: a legend wider than
    the axes would make constrained layout squeeze the plot instead.
    """
    legend = _fit_legend_above(ax, handles, labels)
    if title is not None:
        height = legend.get_window_extent().height * 72 / ax.get_figure(root=True).dpi
        ax.set_title(title, loc="left", pad=height + _TITLE_GAP_POINTS, color=INK_PRIMARY)


def _fit_legend_above(ax: Axes, handles: Sequence[Artist], labels: Sequence[str]) -> Legend:
    """Draw the legend with the most columns whose width fits the laid-out axes."""
    if not handles:
        raise ValueError("A legend needs at least one entry.")
    ax.get_figure(root=True).draw_without_rendering()  # the axes' final width, without legend
    available = ax.get_window_extent().width
    for ncols in range(len(handles), 0, -1):
        legend = ax.legend(
            *_row_major(handles, labels, ncols),
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncols=ncols,
            borderaxespad=0.3,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            labelcolor=INK_SECONDARY,
            columnspacing=1.4,
        )
        if legend.get_window_extent().width <= available:
            break
    return legend


def _row_major(
    handles: Sequence[Artist], labels: Sequence[str], ncols: int
) -> tuple[list[Artist], list[str]]:
    """Reorder entries so matplotlib's column-by-column packing reads row by row.

    Matplotlib fills columns top to bottom, the first ``n % ncols`` columns one entry
    longer; taking every ``ncols``-th entry per column gives exactly those column sizes.
    """
    nrows = math.ceil(len(handles) / ncols)
    order = [
        row * ncols + col
        for col in range(ncols)
        for row in range(nrows)
        if row * ncols + col < len(handles)
    ]
    return [handles[i] for i in order], [labels[i] for i in order]


def _figure_header(
    fig: Figure, title: str | None, handles: Sequence[Artist], labels: Sequence[str]
) -> None:
    """Left-aligned figure title with a one-row legend underneath (constrained layout).

    The title is the legend's own title: constrained layout reserves the top margin for the
    larger of a suptitle and an outside legend, not their sum, so separate artists overlap.
    """
    if not handles:
        if title is not None:
            fig.suptitle(title, x=0.0, ha="left", color=INK_PRIMARY)
        return
    legend = fig.legend(
        handles,
        labels,
        loc="outside upper left",
        ncols=len(handles),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        labelcolor=INK_SECONDARY,
        columnspacing=1.4,
        title=title,
        alignment="left",
        title_fontproperties={"size": STYLE["figure.titlesize"], "weight": "bold"},
    )
    legend.get_title().set_color(INK_PRIMARY)


def _hide_unused_axes(axes: np.ndarray, n_used: int) -> None:
    """Hide empty grid slots; the panel above each gets back its shared x tick labels."""
    nrows, ncols = axes.shape
    for index in range(n_used, nrows * ncols):
        row, col = divmod(index, ncols)
        axes[row, col].set_visible(False)
        if row > 0:
            axes[row - 1, col].tick_params(axis="x", labelbottom=True)
            axes[row - 1, col].xaxis.offsetText.set_visible(True)


# --- Fan chart ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _FanData:
    """Everything one fan chart draws, already filtered to one series and target."""

    history: pd.Series
    median: pd.Series
    bands: dict[float, tuple[pd.Series, pd.Series]]
    actuals: pd.Series | None
    start: pd.Timestamp  # first grid step of the plotted window, even if its row is missing
    cutoff: pd.Timestamp  # last context timestamp
    offset: BaseOffset  # grid frequency of the context


def _prepare_fan(
    context: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    actuals: pd.DataFrame | None,
    target: str,
    item_id: Hashable | None,
    history: int | None,
    intervals: Mapping[float, tuple[float, float]],
    freq: str | None,
) -> _FanData:
    if history is not None and history <= 0:
        raise ValueError(f"history must be a positive number of steps or None, got {history}.")
    if len(intervals) > len(_BAND_FILLS):
        raise ValueError(f"At most {len(_BAND_FILLS)} nested intervals stay legible.")
    item = _resolve_item(context, item_id, "context")
    past = _series(_item_rows(context, item, "context"), target, "context")
    offset = _grid_offset(past.index, freq, "context")
    cutoff = past.index.max()
    start = past.index.min()
    if history is not None:
        start = max(start, cutoff - (history - 1) * offset)
        past = past.loc[past.index >= start]

    rows = _forecast_rows(forecast, item, target)
    median = _series(rows, PREDICTION_COL, "forecast")
    if median.index.min() <= cutoff:
        raise ValueError(
            f"forecast starts at {median.index.min()}, not after the context end {cutoff}."
        )
    bands = {
        level: (
            _series(rows, quantile_col(low), "forecast"),
            _series(rows, quantile_col(high), "forecast"),
        )
        for level, (low, high) in intervals.items()
    }
    observed = None
    if actuals is not None:
        future = _series(_item_rows(actuals, item, "actuals"), target, "actuals")
        observed = future.loc[(future.index > cutoff) & (future.index <= median.index.max())]
    return _FanData(past, median, bands, observed, start, cutoff, offset)


def _forecast_rows(forecast: pd.DataFrame, item_id: Hashable, target: str) -> pd.DataFrame:
    rows = _item_rows(forecast, item_id, "forecast")
    _require_columns(rows, [TARGET_NAME_COL], "forecast")
    rows = rows.loc[rows[TARGET_NAME_COL] == target]
    if rows.empty:
        names = list(pd.unique(forecast[TARGET_NAME_COL]))
        raise ValueError(f"forecast has no rows for target {target!r}; targets: {names}.")
    return rows


def _draw_fan(ax: Axes, data: _FanData) -> tuple[list[Artist], list[str]]:
    """Draw bands, history, actuals and median; return legend handles in reading order."""
    entries: list[tuple[Artist, str]] = []
    ordered = sorted(data.bands.items())  # narrowest first, matching _BAND_FILLS
    fills = _BAND_FILLS[len(ordered)] if ordered else ()
    band_entries: list[tuple[Artist, str]] = []
    for (level, (low, high)), (color, alpha) in reversed(list(zip(ordered, fills, strict=True))):
        band = ax.fill_between(
            low.index, low.to_numpy(), high.to_numpy(), color=color, alpha=alpha, lw=0, zorder=1
        )
        band_entries.append((band, f"{_percent(level)} interval"))
    ax.axvline(data.cutoff, color=AXIS_COLOR, linewidth=1.0, zorder=1.5)
    (past,) = ax.plot(data.history.index, data.history.to_numpy(), color=INK_PRIMARY, zorder=3)
    entries.append((past, "History"))
    if data.actuals is not None and not data.actuals.empty:
        # Start the actuals at the last context point so the observed line is continuous.
        joined = pd.concat([data.history.iloc[-1:], data.actuals])
        (actual,) = ax.plot(joined.index, joined.to_numpy(), color=INK_SECONDARY, zorder=3)
        entries.append((actual, "Actual"))
    (median,) = ax.plot(data.median.index, data.median.to_numpy(), color=BLUE, zorder=4)
    entries.append((median, "Forecast (median)"))
    entries.extend(reversed(band_entries))  # narrowest interval first in the legend
    handles, labels = zip(*entries, strict=True)
    return list(handles), list(labels)


def _finish_time_axes(ax: Axes, ylabel: str) -> None:
    _style_axes(ax, grid="y")
    _format_time_axis(ax)
    ax.yaxis.set_major_formatter(_CompactFormatter())
    ax.set_ylabel(ylabel)


def _time_span(data: Iterable[_FanData]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """From the earliest window start to the latest forecast step."""
    items = list(data)
    return min(d.start for d in items), max(d.median.index.max() for d in items)


def plot_fan(
    context: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    actuals: pd.DataFrame | None = None,
    target: str = TARGET_COL,
    item_id: Hashable | None = None,
    history: int | None = 336,
    freq: str | None = None,
    intervals: Mapping[float, tuple[float, float]] = INTERVALS,
    title: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Fan chart of one series: recent history, median forecast and nested intervals.

    Parameters
    ----------
    context, forecast
        A series frame and the forecast frame made from it (``c2p.schema``).
    actuals
        Optional frame with the true values; only the forecast horizon is drawn.
    target, item_id
        Which target and series to draw; ``item_id`` may be omitted for one series.
    history
        Grid steps shown before the forecast (by time, so gappy contexts line up);
        ``None`` shows the whole context.
    freq
        Grid frequency of the context, e.g. ``"h"``. Inferred when omitted; a context with
        missing rows and a step of a day or more (``"D"``, ``"B"``, ``"MS"``...) needs it.
    intervals
        Nominal level -> (lower, upper) quantile levels; the quantile columns must exist.
    title, ylabel
        Sentence-case takeaway title and y label (default: the target name).
    ax
        Axes to draw into; a new figure is created when omitted.
    """
    data = _prepare_fan(
        context,
        forecast,
        actuals=actuals,
        target=target,
        item_id=item_id,
        history=history,
        intervals=intervals,
        freq=freq,
    )
    with style_context():
        if ax is None:
            _, ax = plt.subplots(figsize=(8.0, 3.6), layout="constrained")
        handles, labels = _draw_fan(ax, data)
        _finish_time_axes(ax, ylabel if ylabel is not None else target)
        ax.set_xlim(*_time_span([data]))
        _legend_above(ax, handles, labels, title)
    return ax


# --- Scenario small multiples -------------------------------------------------------------


def _missing_spans(
    history: pd.Series, start: pd.Timestamp, end: pd.Timestamp, offset: BaseOffset
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Runs of grid steps in ``[start, end]`` that are absent or NaN, as (start, end) spans.

    The grid runs from the window start, so rows dropped at the very start of the window are
    found too. Each span reaches halfway to the neighbouring grid steps.
    """
    grid = pd.date_range(start, end, freq=offset)
    missing = history.reindex(grid).isna().to_numpy()
    if not missing.any():
        return []
    padded = pd.date_range(grid[0] - offset, grid[-1] + offset, freq=offset)
    edges = padded[:-1] + (padded[1:] - padded[:-1]) / 2  # edges[i] = left edge of grid[i]
    runs = np.flatnonzero(np.diff(np.concatenate(([0], missing.astype(np.int8), [0]))))
    return [(edges[first], edges[stop]) for first, stop in zip(runs[::2], runs[1::2], strict=True)]


_TIMESTAMP_TYPES = (dt.datetime, np.datetime64, str)  # pd.Timestamp is a datetime


def _as_timestamps(value: object, name: str) -> list[pd.Timestamp]:
    """One timestamp or an iterable of them as a list; anything else raises ``TypeError``."""
    if isinstance(value, _TIMESTAMP_TYPES):
        return [pd.Timestamp(value)]
    items = list(value) if isinstance(value, Iterable) else None
    if items is None or not all(isinstance(item, _TIMESTAMP_TYPES) for item in items):
        raise TypeError(
            f"spikes[{name!r}] must be a timestamp or a list of timestamps, got {value!r}; "
            f"turn a row position into a timestamp with context['{TIMESTAMP_COL}'].iloc[pos]."
        )
    return [pd.Timestamp(item) for item in items]


def _draw_spikes(ax: Axes, data: _FanData, at: Sequence[pd.Timestamp], name: str) -> Line2D:
    absent = [stamp for stamp in at if stamp not in data.history.index]
    if absent:
        raise ValueError(
            f"Spike timestamps {absent} of scenario {name!r} have no row in the plotted "
            f"history ({data.start} to {data.cutoff}); check them, or raise history= to "
            "show earlier spikes."
        )
    values = data.history.loc[list(at)].to_numpy()
    (marker,) = ax.plot(
        list(at),
        values,
        linestyle="none",
        marker="o",
        markersize=MARKER_SIZE + 1,
        markerfacecolor=ISSUE_COLOR,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        zorder=5,
    )
    return marker


def _check_scenario_keys(
    scenarios: Mapping[str, object], extra: Mapping[str, object] | None, what: str
) -> None:
    unknown = sorted(set(extra or {}) - set(scenarios))
    if unknown:
        raise ValueError(f"{what} keys {unknown} are not scenarios; choose from {list(scenarios)}.")


def plot_scenarios(
    scenarios: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    actuals: pd.DataFrame | None = None,
    target: str = TARGET_COL,
    item_id: Hashable | None = None,
    history: int | None = 168,
    freq: str | None = None,
    intervals: Mapping[float, tuple[float, float]] = INTERVALS,
    spikes: Mapping[str, pd.Timestamp | Sequence[pd.Timestamp]] | None = None,
    notes: Mapping[str, str] | None = None,
    ncols: int = 1,
    sharey: bool = True,
    title: str | None = None,
    ylabel: str | None = None,
) -> Figure:
    """Small multiples of fan charts, one panel per data-quality scenario.

    Each panel shows ``scenario -> (context, forecast)``. Grid steps of the last ``history``
    steps that are NaN or have no row at all (dropped rows) are shaded with a light gray
    span; ``spikes`` marks injected outliers (timestamps) with a red dot; ``notes`` adds a
    short right-aligned note per panel (for example an error metric). All panels share one
    time range, from the earliest window start to the latest forecast step, and the y scale
    by default, so damage is comparable. ``freq`` is the context grid frequency (see
    :func:`plot_fan`); pass it for calendar frequencies when rows are missing.
    """
    if not scenarios:
        raise ValueError("scenarios is empty; pass at least one name -> (context, forecast).")
    if ncols <= 0:
        raise ValueError(f"ncols must be positive, got {ncols}.")
    _check_scenario_keys(scenarios, spikes, "spikes")
    _check_scenario_keys(scenarios, notes, "notes")
    prepared = {
        name: _prepare_fan(
            context,
            forecast,
            actuals=actuals,
            target=target,
            item_id=item_id,
            history=history,
            intervals=intervals,
            freq=freq,
        )
        for name, (context, forecast) in scenarios.items()
    }
    nrows = math.ceil(len(prepared) / ncols)
    with style_context():
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(8.0 if ncols == 1 else 4.6 * ncols, 1.9 * nrows + 1.0),
            sharex=True,
            sharey=sharey,
            squeeze=False,
            layout="constrained",
        )
        handles, labels = _draw_scenario_panels(axes.flat, prepared, spikes or {}, notes or {})
        axes[0, 0].set_xlim(*_time_span(prepared.values()))  # shared by every panel
        _hide_unused_axes(axes, len(prepared))
        fig.supylabel(ylabel if ylabel is not None else target, color=INK_SECONDARY)
        _figure_header(fig, title, handles, labels)
    return fig


def _draw_scenario_panels(
    axes: Iterator[Axes],
    prepared: Mapping[str, _FanData],
    spikes: Mapping[str, pd.Timestamp | Sequence[pd.Timestamp]],
    notes: Mapping[str, str],
) -> tuple[list[Artist], list[str]]:
    handles: list[Artist] = []
    labels: list[str] = []
    gap_patch: Artist | None = None
    spike_marker: Artist | None = None
    for ax, (name, data) in zip(axes, prepared.items(), strict=False):
        panel_handles, panel_labels = _draw_fan(ax, data)
        if len(panel_handles) > len(handles):
            handles, labels = panel_handles, panel_labels
        for start, end in _missing_spans(data.history, data.start, data.cutoff, data.offset):
            gap_patch = ax.axvspan(start, end, color=GAP_SPAN, lw=0, zorder=0.4)
        if name in spikes:
            spike_marker = _draw_spikes(ax, data, _as_timestamps(spikes[name], name), name)
        _finish_time_axes(ax, "")
        ax.set_title(name, loc="left", fontsize=10.5, pad=4)
        if name in notes:
            ax.set_title(
                notes[name],
                loc="right",
                fontsize=9.5,
                fontweight="normal",
                pad=4,
                color=INK_SECONDARY,
            )
    if gap_patch is not None:
        handles, labels = [*handles, gap_patch], [*labels, "Missing context"]
    if spike_marker is not None:
        handles, labels = [*handles, spike_marker], [*labels, "Spike"]
    return handles, labels


# --- Gap grid -----------------------------------------------------------------------------

_OUTSIDE, _OBSERVED, _MISSING = 0, 1, 2
_STATE_CMAP = ListedColormap([SURFACE, OBSERVED_CELL, ISSUE_COLOR])
_STATE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _STATE_CMAP.N)


# Figure geometry: one day row is 0.2 in tall until the figure reaches 9 in; titles, legend
# and tick labels take about 1.6 in; each panel's cells are about 3.2 in wide.
_GAP_ROW_INCHES = 0.2
_GAP_FRAME_INCHES = 1.6
_GAP_MAX_INCHES = 9.0
_GAP_PANEL_INCHES = 3.2
# White cell separators (0.6 pt) are drawn only on cells at least this tall and wide;
# on smaller cells they would cover the cells themselves.
_CELL_EDGE_MIN_POINTS = 4.0


def _check_on_grid(
    index: pd.DatetimeIndex, origin: pd.Timestamp, step: pd.Timedelta, freq: str, what: str
) -> None:
    """Every timestamp must be a whole number of steps after ``origin`` (the series start)."""
    off_grid = index[(index - origin) % step != pd.Timedelta(0)]
    if len(off_grid):
        raise ValueError(
            f"{what} has {len(off_grid)} timestamps off the {freq!r} grid that starts at "
            f"{origin}, e.g. {off_grid[:3].tolist()}; resample or aggregate them first."
        )


def _gap_grid_geometry(n_days: int, slots: int) -> tuple[float, bool]:
    """Figure height in inches, and whether cells are big enough for white separators."""
    height = min(_GAP_MAX_INCHES, _GAP_FRAME_INCHES + _GAP_ROW_INCHES * n_days)
    row_points = 72 * (height - _GAP_FRAME_INCHES) / n_days
    column_points = 72 * _GAP_PANEL_INCHES / slots
    return height, min(row_points, column_points) >= _CELL_EDGE_MIN_POINTS


def _state_matrix(
    grid: pd.DatetimeIndex, present: pd.DatetimeIndex, span: tuple[pd.Timestamp, pd.Timestamp]
) -> np.ndarray:
    inside = (grid >= span[0]) & (grid <= span[1])
    states = np.where(grid.isin(present), _OBSERVED, _MISSING)
    return np.where(inside, states, _OUTSIDE)


def _time_of_day(offset: pd.Timedelta) -> str:
    components = offset.components
    return f"{components.hours:02d}:{components.minutes:02d}"


def _draw_state_panel(
    ax: Axes,
    matrix: np.ndarray,
    days: pd.DatetimeIndex,
    step: pd.Timedelta,
    phase: pd.Timedelta,
    *,
    edges: bool,
) -> None:
    ax.pcolormesh(
        matrix,
        cmap=_STATE_CMAP,
        norm=_STATE_NORM,
        edgecolors=SURFACE if edges else "none",
        linewidth=0.6 if edges else 0.0,
    )
    _style_axes(ax, grid="none")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    slots = matrix.shape[1]
    every = max(1, slots // 4)
    positions = range(0, slots, every)
    ax.set_xticks(
        [p + 0.5 for p in positions], labels=[_time_of_day(phase + p * step) for p in positions]
    )
    row_every = max(1, math.ceil(len(days) / 10))
    rows = np.arange(0, len(days), row_every)
    ax.set_yticks(rows + 0.5, labels=[days[r].strftime("%Y-%m-%d") for r in rows])
    ax.set_xlim(0, slots)
    ax.set_ylim(len(days), 0)  # earliest day at the top, like a calendar


def plot_gap_grid(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    column: str,
    freq: str = "h",
    item_id: Hashable | None = None,
    max_days: int = 400,
    title: str | None = None,
) -> Figure:
    """Day x time-of-day grid of one series, before and after reindexing to ``freq``.

    The left panel marks which grid timestamps have a row in ``before`` (dropped rows are
    invisible in a line chart). The right panel marks which values of ``column`` are
    observed in ``after``, where every gap has become an explicit NaN. Two states only:
    observed (light gray) and missing (red). Times are naive UTC, as in the schema; the
    grid starts at the first timestamp of either frame.

    One row per day fits about a year (``max_days=400``): beyond that, rows get thinner
    than about two screen pixels and a single missing hour could vanish. Slice a window
    (e.g. one year) for the article figure, or raise ``max_days`` for a dense overview
    saved at high dpi.
    """
    step = _fixed_step(freq)
    slots, remainder = divmod(pd.Timedelta(days=1), step)
    if slots < 1 or remainder != pd.Timedelta(0):
        raise ValueError(f"freq={freq!r} must divide one day evenly for a day x time grid.")
    if max_days <= 0:
        raise ValueError(f"max_days must be positive, got {max_days}.")
    item = _resolve_item(before, item_id, "before")
    before_index = pd.DatetimeIndex(_item_rows(before, item, "before")[TIMESTAMP_COL]).unique()
    after_series = _series(_item_rows(after, item, "after"), column, "after")
    first = min(before_index.min(), after_series.index.min())
    last = max(before_index.max(), after_series.index.max())
    _check_on_grid(before_index, first, step, freq, "before")
    _check_on_grid(after_series.index, first, step, freq, "after")

    days = pd.date_range(first.normalize(), last.normalize(), freq="D")
    if len(days) > max_days:
        raise ValueError(
            f"before/after span {len(days):,} days, more than max_days={max_days} legible "
            f"day rows; slice a window first, e.g. frame[frame['{TIMESTAMP_COL}'] >= "
            "'2018-01-01'], or raise max_days for a dense overview."
        )
    phase = (first - first.normalize()) % step  # e.g. 1h for a 2h grid on odd hours
    grid = pd.date_range(days[0] + phase, periods=len(days) * slots, freq=step)
    shape = (len(days), slots)
    panels = {
        "Before": _state_matrix(grid, before_index, (first, last)).reshape(shape),
        "After": _state_matrix(grid, after_series.dropna().index, (first, last)).reshape(shape),
    }
    height, edges = _gap_grid_geometry(len(days), slots)
    with style_context():
        fig, axes = plt.subplots(1, 2, figsize=(8.0, height), sharey=True, layout="constrained")
        subtitles = {"Before": "rows missing", "After": "values NaN on the grid"}
        for ax, (name, matrix) in zip(axes, panels.items(), strict=True):
            _draw_state_panel(ax, matrix, days, step, phase, edges=edges)
            n_missing = int((matrix == _MISSING).sum())
            ax.set_title(f"{name}: {n_missing:,} {subtitles[name]}", loc="left", fontsize=10.5)
        axes[0].set_ylabel("Day (UTC)")
        fig.supxlabel("Time of day (UTC)", color=INK_SECONDARY)
        handles = [Patch(color=OBSERVED_CELL), Patch(color=ISSUE_COLOR)]
        _figure_header(fig, title, handles, ["Observed", "Missing"])
    return fig


# --- Benchmark bars -----------------------------------------------------------------------


def _check_benchmarks(df: pd.DataFrame, value: str, highlight: str | None) -> None:
    _require_columns(df, ["model", "benchmark", value], "benchmarks")
    if df[value].isna().any():
        raise ValueError(f"benchmarks has NaN in {value!r}; drop or fill those rows first.")
    duplicated = df.duplicated(["model", "benchmark"])
    if duplicated.any():
        examples = df.loc[duplicated, ["model", "benchmark"]].head(3).to_numpy().tolist()
        raise ValueError(f"benchmarks has duplicated (model, benchmark) pairs, e.g. {examples}.")
    models = list(pd.unique(df["model"]))
    if highlight is not None and highlight not in models:
        raise ValueError(f"highlight {highlight!r} is not a model; models: {models}.")


def _draw_benchmark_panel(
    ax: Axes, rows: pd.DataFrame, value: str, highlight: str | None, label_offset: float
) -> None:
    rows = rows.sort_values(value)  # largest at the top
    positions = np.arange(len(rows))
    is_highlight = (rows["model"] == highlight).to_numpy()
    values = rows[value].to_numpy(dtype="float64")
    ax.barh(positions, values, height=0.62, color=np.where(is_highlight, BLUE, NEUTRAL_BAR))
    for y, v, strong in zip(positions, values, is_highlight, strict=True):
        ax.text(
            v + (label_offset if v >= 0 else -label_offset),
            y,
            f"{v:.1f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9.5,
            color=INK_PRIMARY if strong else INK_SECONDARY,
            fontweight="bold" if strong else "normal",
        )
    _style_axes(ax, grid="none")  # before the per-label emphasis, which it would reset
    ax.set_yticks(positions, labels=rows["model"].tolist())
    for tick_label, strong in zip(ax.get_yticklabels(), is_highlight, strict=True):
        tick_label.set_color(INK_PRIMARY if strong else INK_SECONDARY)
        tick_label.set_fontweight("bold" if strong else "normal")
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.axvline(0, color=AXIS_COLOR, linewidth=1.0)


def plot_benchmarks(
    df: pd.DataFrame,
    *,
    highlight: str | None = "Chronos-2",
    value: str = "skill_score_pct",
    ncols: int | None = None,
    xlabel: str = "Skill score vs Seasonal Naive (%), higher is better",
    title: str | None = None,
) -> Figure:
    """Horizontal bars per benchmark (small multiples), one model highlighted in blue.

    ``df`` is long: one row per (``model``, ``benchmark``) with the score in ``value``, as in
    ``data/reference/paper_benchmarks.csv``. Panels follow the order of first appearance of
    each benchmark; bars are sorted within each panel and labelled at the tip.
    """
    _check_benchmarks(df, value, highlight)
    if ncols is not None and ncols <= 0:
        raise ValueError(f"ncols must be positive or None, got {ncols}.")
    benchmarks = list(pd.unique(df["benchmark"]))
    ncols = ncols if ncols is not None else min(3, len(benchmarks))
    nrows = math.ceil(len(benchmarks) / ncols)
    n_models = int(df.groupby("benchmark")["model"].size().max())
    low = min(0.0, float(df[value].min()))
    high = max(0.0, float(df[value].max()))
    span = high - low
    with style_context():
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.2 * ncols, (0.34 * n_models + 0.5) * nrows + 0.9),
            sharex=True,
            squeeze=False,
            layout="constrained",
        )
        for ax, benchmark in zip(axes.flat, benchmarks, strict=False):
            rows = df.loc[df["benchmark"] == benchmark]
            _draw_benchmark_panel(ax, rows, value, highlight, label_offset=0.02 * span)
            ax.set_title(benchmark, loc="left", fontsize=10.5)
        _hide_unused_axes(axes, len(benchmarks))
        # Room for the value labels beyond the longest bar on each side of zero.
        axes.flat[0].set_xlim(low - 0.2 * span if low < 0 else 0.0, high + 0.2 * span)
        fig.supxlabel(xlabel, color=INK_SECONDARY)
        _figure_header(fig, title, [], [])
    return fig


# --- Metric vs parameter ------------------------------------------------------------------


def _set_value_ticks(ax: Axes, values: np.ndarray, *, logx: bool) -> None:
    """Tick exactly the parameter values tried (when few), e.g. context lengths."""
    if logx:
        ax.set_xscale("log", base=2)
    unique = np.unique(values)
    if np.issubdtype(unique.dtype, np.number) and len(unique) <= 12:
        ax.set_xticks(unique, labels=[f"{v:,.6g}" for v in unique])
        ax.xaxis.set_minor_locator(NullLocator())


def plot_metric_vs_param(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    band: tuple[str, str] | None = None,
    colors: Mapping[str, str] = MODEL_COLORS,
    logx: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Line chart of a metric against a parameter, one line per ``hue`` value.

    Typical use: error vs context length per model. ``band=(low_column, high_column)`` adds a
    light wash (e.g. a bootstrap interval). Colours come from :func:`assign_colors` with
    ``colors`` as the fixed mapping, so models keep their series colour and new models never
    take a known model's slot; pass ``colors={}`` when ``hue`` is not a model. A legend is
    drawn for two or more lines (wrapped into rows when wide); a single line is named by the
    title.
    """
    _require_columns(df, [x, y, hue, *(band or ())], "df")
    if df[[x, y, hue]].isna().any().any():
        raise ValueError(f"df has NaN in {x!r}, {y!r} or {hue!r}; drop those rows first.")
    groups = [(str(name), rows) for name, rows in df.groupby(hue, sort=False)]
    names = [name for name, _ in groups]
    palette = assign_colors(names, fixed=colors)
    with style_context():
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
        handles: list[Artist] = []
        for name, group in groups:
            rows = group.sort_values(x)
            color = palette[name]
            if band is not None:
                ax.fill_between(rows[x], rows[band[0]], rows[band[1]], color=color, alpha=0.12)
            (line,) = ax.plot(
                rows[x],
                rows[y],
                color=color,
                marker="o",
                markersize=MARKER_SIZE,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
            )
            handles.append(line)
        _style_axes(ax, grid="y")
        _set_value_ticks(ax, df[x].to_numpy(), logx=logx)
        ax.set_xlabel(xlabel if xlabel is not None else x)
        ax.set_ylabel(ylabel if ylabel is not None else y)
        if len(handles) >= 2:
            _legend_above(ax, handles, names, title)
        elif title is not None:
            ax.set_title(title, loc="left")
    return ax
