"""Figures of the evaluation article (part 2): window schedule, skill scores and coverage.

These three charts answer the three questions a backtest raises, in order: *which data did each
forecast see* (:func:`plot_backtest_windows`), *is the model better than the baselines, and by
enough to believe it* (:func:`plot_skill_scores`), and *are the prediction intervals honest*
(:func:`plot_coverage`).

They reuse the shared style of :mod:`c2p.plotting`, including its private axis helpers, so that
a figure drawn here is indistinguishable from one drawn there; the alternative, restating the
style rules, is how two figures in one article end up looking slightly different.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from c2p.plotting import (
    AXIS_COLOR,
    BLUE,
    INK_PRIMARY,
    INK_SECONDARY,
    MODEL_COLORS,
    NEUTRAL_BAR,
    SURFACE,
    _format_time_axis,
    _legend_above,
    _style_axes,
    assign_colors,
    style_context,
)

__all__ = [
    "coverage_long",
    "plot_backtest_windows",
    "plot_coverage",
    "plot_skill_scores",
    "window_schedule",
]

_CONTEXT_FILL = "#dddcd4"
_HORIZON_FILL = BLUE


# --- Window schedule ----------------------------------------------------------------------


def window_schedule(windows: Sequence, *, id_column: str = "item_id") -> pd.DataFrame:
    """Turn ``rolling_origin_windows`` output into a frame of spans, one row per window.

    Columns: ``window``, ``cutoff``, ``context_start``, ``horizon_end``, ``n_context``. Taking
    the spans from the windows themselves (rather than recomputing them from the config) is what
    makes the figure a picture of the backtest that actually ran.
    """
    if len(windows) == 0:
        raise ValueError("windows is empty.")
    rows = []
    for window in windows:
        context, actuals = window.context, window.actuals
        rows.append(
            {
                "window": window.index,
                "cutoff": pd.Timestamp(window.cutoff),
                "context_start": pd.Timestamp(context["timestamp"].min()),
                "horizon_end": pd.Timestamp(actuals["timestamp"].max()),
                "n_context": len(context) // max(1, context[id_column].nunique()),
            }
        )
    return pd.DataFrame(rows)


def plot_backtest_windows(
    schedule: pd.DataFrame,
    *,
    n_shown: int = 8,
    context_shown: pd.Timedelta | None = None,
    title: str | None = None,
    xlabel: str = "Time",
    ax: Axes | None = None,
) -> Axes:
    """Draw the first and last few windows of a rolling-origin backtest as context/horizon spans.

    Each row is one forecast origin: the grey bar is the context the model received, the blue
    bar the horizon it was scored on, and the tick between them the cutoff. When the backtest
    has more than ``n_shown`` windows, the first and last ``n_shown // 2`` are drawn with a
    gap row in between, labelled with the number of windows it stands for.

    ``context_shown`` draws only the most recent part of each context. A 24-hour horizon next
    to a 2,048-hour context is 1% of the row and vanishes, which hides the very thing the
    figure exists to show; truncating the context keeps the proportions readable. The legend
    then states that the context is cut, so the figure does not imply a shorter backtest.
    """
    for column in ("window", "cutoff", "context_start", "horizon_end"):
        if column not in schedule.columns:
            raise ValueError(f"schedule is missing the column {column!r}.")
    if n_shown < 2:
        raise ValueError(f"n_shown must be at least 2, got {n_shown}.")
    if context_shown is not None and context_shown <= pd.Timedelta(0):
        raise ValueError(f"context_shown must be a positive Timedelta, got {context_shown}.")
    rows, skipped = _rows_to_draw(schedule.sort_values("window"), n_shown)
    with style_context():
        if ax is None:
            _, ax = plt.subplots(figsize=(8.0, 0.32 * len(rows) + 1.5), layout="constrained")
        _draw_spans(ax, rows, skipped, context_shown)
        _style_axes(ax, grid="none")
        _format_time_axis(ax)
        ax.set_ylim(len(rows) - 0.5, -0.5)  # first window at the top
        ax.set_yticks(
            range(len(rows)),
            labels=[
                "…" if row is None else f"window {int(row['window'])}" for _, row in enumerate(rows)
            ],
        )
        ax.set_xlabel(xlabel)
        context_label = "Context (what the model sees)"
        if context_shown is not None:
            context_label = f"Context, last {_format_span(context_shown)} shown"
        handles = [Patch(color=_CONTEXT_FILL), Patch(color=_HORIZON_FILL)]
        labels = [context_label, "Horizon (what it is scored on)"]
        _legend_above(ax, handles, labels, title)
    return ax


def _rows_to_draw(schedule: pd.DataFrame, n_shown: int) -> tuple[list, int]:
    """The rows to draw, with ``None`` marking the gap, and how many windows the gap hides."""
    if len(schedule) <= n_shown:
        return [row for _, row in schedule.iterrows()], 0
    half = n_shown // 2
    head = [row for _, row in schedule.head(half).iterrows()]
    tail = [row for _, row in schedule.tail(n_shown - half).iterrows()]
    return [*head, None, *tail], len(schedule) - n_shown


def _format_span(span: pd.Timedelta) -> str:
    """``Timedelta('7 days')`` -> ``"168 h"``; sub-hourly spans keep their own unit."""
    hours = span / pd.Timedelta(hours=1)
    return f"{hours:,.0f} h" if hours >= 1 else str(span)


def _draw_spans(ax: Axes, rows: Sequence, skipped: int, context_shown: pd.Timedelta | None) -> None:
    for y, row in enumerate(rows):
        if row is None:
            ax.annotate(
                f"{skipped} more windows, one per day",
                (0.5, y),
                xycoords=("axes fraction", "data"),
                ha="center",
                va="center",
                fontsize=9,
                color=INK_SECONDARY,
            )
            continue
        start = row["context_start"]
        if context_shown is not None:
            start = max(start, row["cutoff"] - context_shown)
        ax.barh(y, row["cutoff"] - start, left=start, height=0.6, color=_CONTEXT_FILL)
        ax.barh(
            y,
            row["horizon_end"] - row["cutoff"],
            left=row["cutoff"],
            height=0.6,
            color=_HORIZON_FILL,
        )
        ax.plot([row["cutoff"]], [y], marker="|", markersize=11, color=INK_PRIMARY, zorder=3)


# --- Skill scores -------------------------------------------------------------------------


def plot_skill_scores(
    skill: pd.DataFrame,
    *,
    highlight: str = "chronos-2",
    drop_baseline: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Horizontal bars of skill score per model with their bootstrap intervals.

    ``skill`` is the output of ``c2p.eval.skill_table``: one row per model with ``skill``,
    ``ci_low``, ``ci_high``, ``win_rate`` and ``baseline``. Skills are drawn as percentages,
    sorted best first, with the win rate printed at the right. The baseline itself sits at
    exactly 0 by construction and is dropped by default.
    """
    required = ["model", "skill", "ci_low", "ci_high", "win_rate", "baseline"]
    missing = [column for column in required if column not in skill.columns]
    if missing:
        raise ValueError(f"skill is missing the columns {missing}.")
    baseline = str(skill["baseline"].iloc[0])
    rows = skill[skill["model"] != baseline] if drop_baseline else skill
    if rows.empty:
        raise ValueError("Nothing to plot: skill holds only the baseline.")
    rows = rows.sort_values("skill")  # best at the top
    with style_context():
        if ax is None:
            _, ax = plt.subplots(figsize=(8.0, 0.46 * len(rows) + 1.5), layout="constrained")
        _draw_skill_bars(ax, rows, highlight)
        _style_axes(ax, grid="none")
        ax.axvline(0, color=AXIS_COLOR, linewidth=1.0)
        ax.set_xlabel(
            xlabel if xlabel is not None else f"Skill score vs {baseline} (%), higher is better"
        )
        if title is not None:
            ax.set_title(title, loc="left", color=INK_PRIMARY)
    return ax


def _draw_skill_bars(ax: Axes, rows: pd.DataFrame, highlight: str) -> None:
    positions = np.arange(len(rows))
    skills = 100 * rows["skill"].to_numpy(dtype="float64")
    low = 100 * rows["ci_low"].to_numpy(dtype="float64")
    high = 100 * rows["ci_high"].to_numpy(dtype="float64")
    is_highlight = (rows["model"] == highlight).to_numpy()
    ax.barh(
        positions, skills, height=0.56, color=np.where(is_highlight, BLUE, NEUTRAL_BAR), zorder=2
    )
    ax.errorbar(
        skills,
        positions,
        xerr=[skills - low, high - skills],
        fmt="none",
        ecolor=INK_SECONDARY,
        elinewidth=1.1,
        capsize=3.5,
        zorder=3,
    )
    span = max(high.max() - min(0.0, low.min()), 1e-9)
    for y, value, hi, strong, win in zip(
        positions, skills, high, is_highlight, rows["win_rate"], strict=True
    ):
        ax.annotate(
            f"{value:.1f}  ({win:.0%} of windows)",
            (hi + 0.02 * span, y),
            va="center",
            ha="left",
            fontsize=9.5,
            color=INK_PRIMARY if strong else INK_SECONDARY,
            fontweight="bold" if strong else "normal",
        )
    ax.set_yticks(positions, labels=rows["model"].tolist())
    for label, strong in zip(ax.get_yticklabels(), is_highlight, strict=True):
        label.set_color(INK_PRIMARY if strong else INK_SECONDARY)
        label.set_fontweight("bold" if strong else "normal")
    ax.set_xlim(min(0.0, low.min()) - 0.05 * span, high.max() + 0.45 * span)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)


# --- Interval coverage --------------------------------------------------------------------


def plot_coverage(
    coverage: pd.DataFrame,
    *,
    colors: Mapping[str, str] = MODEL_COLORS,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Achieved coverage against the nominal level, one marker per model and interval.

    ``coverage`` is long: ``model``, ``nominal`` and ``achieved`` as fractions. The dashed line
    at each nominal level is the target; a marker to its left means the interval is too narrow
    (over-confident), to its right that it is too wide. Nominal levels are drawn as separate
    rows because an 80% and a 98% interval are different promises.
    """
    required = ["model", "nominal", "achieved"]
    missing = [column for column in required if column not in coverage.columns]
    if missing:
        raise ValueError(f"coverage is missing the columns {missing}.")
    if coverage.empty:
        raise ValueError("coverage is empty.")
    levels = sorted(coverage["nominal"].unique())
    models = list(dict.fromkeys(coverage["model"]))
    palette = assign_colors(models, fixed=colors)
    with style_context():
        if ax is None:
            _, ax = plt.subplots(
                figsize=(8.0, 0.5 * len(levels) * len(models) + 1.4), layout="constrained"
            )
        handles = _draw_coverage(ax, coverage, levels, models, palette)
        _style_axes(ax, grid="none")
        ax.set_xlabel("Share of actuals inside the interval")
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        _legend_above(ax, handles, models, title)
    return ax


def _draw_coverage(
    ax: Axes,
    coverage: pd.DataFrame,
    levels: Sequence[float],
    models: Sequence[str],
    palette: Mapping[str, str],
) -> list[Line2D]:
    handles: dict[str, Line2D] = {}
    y = 0.0
    ticks: list[float] = []
    labels: list[str] = []
    for level in levels:
        rows = coverage[coverage["nominal"] == level]
        block_top = y
        for model in models:
            match = rows[rows["model"] == model]
            if match.empty:
                continue
            (marker,) = ax.plot(
                [float(match["achieved"].iloc[0])],
                [y],
                marker="o",
                markersize=7.5,
                markerfacecolor=palette[model],
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                linestyle="none",
                zorder=3,
            )
            handles.setdefault(model, marker)
            y += 1.0
        ax.plot(
            [level, level],
            [block_top - 0.5, y - 0.5],
            linestyle=(0, (3, 3)),
            color=INK_SECONDARY,
            linewidth=1.0,
            zorder=2,
        )
        ticks.append((block_top + y - 1) / 2)
        labels.append(f"{level:.0%} interval")
        y += 0.8
    ax.set_yticks(ticks, labels=labels)
    ax.set_ylim(y - 1.3, -0.7)
    low = min(float(coverage["achieved"].min()), *levels)
    high = max(float(coverage["achieved"].max()), *levels)
    pad = max(0.02 * (high - low), 0.005)
    ax.set_xlim(low - pad, high + pad)
    return [handles[model] for model in models if model in handles]


def coverage_long(summary: pd.DataFrame, intervals: Mapping[float, object]) -> pd.DataFrame:
    """Reshape a ``c2p.eval.summarize`` frame into the long form :func:`plot_coverage` wants."""
    from c2p.eval import coverage_column

    rows = [
        {"model": row["model"], "nominal": level, "achieved": row[coverage_column(level)]}
        for _, row in summary.iterrows()
        for level in intervals
    ]
    return pd.DataFrame(rows)
