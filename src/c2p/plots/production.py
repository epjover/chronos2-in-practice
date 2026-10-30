"""Figures of the production article (part 5): throughput and cost per million forecasts.

Two charts, for the two questions that decide a deployment. :func:`plot_throughput` answers *how
many series per second does this machine sustain, and where does batching stop helping*: the
answer is a curve against batch size, one line per (device, checkpoint), on log axes because the
range spans three orders of magnitude. :func:`plot_cost_per_million` answers *what does that
cost*, by joining the same throughput with published hourly rates.

The cost chart carries a deliberate asymmetry. Rates are exact; throughput measured on a local
machine is not what a cloud instance would deliver. The figure therefore draws the rate-only
break-even ratio as an annotated reference line rather than burying it in the bars, so the
reader can separate the part that is published from the part that is an estimate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from c2p.plotting import (
    BLUE,
    INK_PRIMARY,
    INK_SECONDARY,
    NEUTRAL_BAR,
    _legend_above,
    _style_axes,
    style_context,
)

__all__ = ["device_label", "plot_cost_per_million", "plot_coverage_by_month", "plot_throughput"]

_CPU_COLOR = "#8c8b82"
_GPU_COLOR = BLUE
_SMALL_STYLE = (0, (4, 2))


def device_label(row: Mapping[str, object]) -> str:
    """Short series label: checkpoint name plus the hardware it ran on.

    GPU rows name the card; CPU rows name the processor (brand noise stripped) and the PyTorch
    thread count, because a CPU throughput without its thread count is not reproducible.
    """
    model = str(row["model"]).split("/")[-1]
    if str(row["device"]) == "cuda" and row.get("gpu"):
        return f"{model} on {row['gpu']}"
    cpu = re.sub(r"\(R\)|\(TM\)|\d+th Gen |Processor", "", str(row.get("cpu") or ""))
    cpu = re.sub(r"\s+", " ", cpu).strip() or "CPU"
    threads = row.get("torch_threads")
    suffix = f", {int(threads)} threads" if threads not in (None, "") else ""
    return f"{model} on {cpu}{suffix}"


def plot_throughput(
    bench: pd.DataFrame,
    *,
    context_length: int,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Series per second against batch size, one line per checkpoint and device.

    ``bench`` is ``results/bench_latency.csv`` from ``bench/bench_cpu_gpu.py``. One context
    length at a time: mixing them puts four lines of two different shapes on one axis and the
    saturation point of each becomes unreadable. When the frame carries
    ``series_per_second_p05`` and ``series_per_second_p95`` (the throughput at the slowest and
    fastest 5% of calls), each line gets a shaded band between them; on a log axis a band too
    thin to see is the honest picture of a stable measurement.
    """
    required = ["model", "device", "gpu", "context_length", "batch_size", "series_per_second"]
    missing = [column for column in required if column not in bench.columns]
    if missing:
        raise ValueError(f"bench is missing the columns {missing}.")
    rows = bench[bench["context_length"] == context_length]
    if rows.empty:
        available = sorted(bench["context_length"].unique())
        raise ValueError(f"No rows with context_length={context_length}; available: {available}.")
    with style_context():
        if ax is None:
            _, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
        for (model, device), group in rows.groupby(["model", "device"], sort=True):
            group = group.sort_values("batch_size")
            is_small = "small" in str(model)
            color = _GPU_COLOR if device == "cuda" else _CPU_COLOR
            if {"series_per_second_p05", "series_per_second_p95"} <= set(group.columns):
                ax.fill_between(
                    group["batch_size"],
                    group["series_per_second_p05"],
                    group["series_per_second_p95"],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=2,
                )
            ax.plot(
                group["batch_size"],
                group["series_per_second"],
                marker="o",
                markersize=5,
                linewidth=1.8,
                linestyle=_SMALL_STYLE if is_small else "solid",
                color=color,
                label=device_label(group.iloc[0]),
                zorder=3,
            )
            best = group.loc[group["series_per_second"].idxmax()]
            ax.annotate(
                f"{best['series_per_second']:,.0f}/s",
                (best["batch_size"], best["series_per_second"]),
                xytext=(6, 2),
                textcoords="offset points",
                fontsize=9,
                color=INK_PRIMARY if device == "cuda" else INK_SECONDARY,
            )
        batches = sorted(rows["batch_size"].unique())
        # Headroom on the right so the "N/s" label of the largest batch stays inside the axes.
        ax.set_xlim(batches[0] * 0.9, batches[-1] * 1.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(batches)
        ax.get_xaxis().set_major_formatter(lambda value, _: f"{value:g}")
        ax.set_xlabel("Series per request")
        ax.set_ylabel("Series per second")
        _style_axes(ax, grid="y")
        handles, labels = ax.get_legend_handles_labels()
        _legend_above(ax, handles, labels, title)
    return ax


def plot_latency_spread(
    bench: pd.DataFrame,
    *,
    context_length: int,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """How much each timed call deviates from its cell's median, as p5 and p95 in percent.

    One marker pair per (checkpoint, device, batch size) at one context length: the 5th and the
    95th percentile of call duration relative to the median, so a cell whose calls all take the
    same time sits on the zero line and a noisy one spreads out. The number an operator wants is
    the p95 side: how much slower than typical the slow calls are. Needs the ``seconds_median``,
    ``seconds_p05`` and ``seconds_p95`` columns of ``bench/bench_cpu_gpu.py``.
    """
    required = ["model", "device", "gpu", "context_length", "batch_size", "seconds_median"]
    required += ["seconds_p05", "seconds_p95"]
    missing = [column for column in required if column not in bench.columns]
    if missing:
        raise ValueError(f"bench is missing the columns {missing}.")
    rows = bench[bench["context_length"] == context_length]
    if rows.empty:
        available = sorted(bench["context_length"].unique())
        raise ValueError(f"No rows with context_length={context_length}; available: {available}.")
    with style_context():
        if ax is None:
            _, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
        groups = list(rows.groupby(["model", "device"], sort=True))
        batches = sorted(rows["batch_size"].unique())
        width = 0.8 / max(len(groups), 1)
        for offset, ((model, device), group) in enumerate(groups):
            group = group.sort_values("batch_size")
            is_small = "small" in str(model)
            color = _GPU_COLOR if device == "cuda" else _CPU_COLOR
            x = (
                np.array([batches.index(b) for b in group["batch_size"]])
                + (offset - (len(groups) - 1) / 2) * width
            )
            low = (group["seconds_p05"] / group["seconds_median"] - 1) * 100
            high = (group["seconds_p95"] / group["seconds_median"] - 1) * 100
            ax.vlines(x, low, high, color=color, linewidth=6, alpha=0.35 if is_small else 0.75)
            ax.scatter(
                x,
                high,
                marker="^" if is_small else "o",
                s=28,
                color=color,
                zorder=3,
                label=device_label(group.iloc[0]),
            )
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.set_xlim(-0.6, len(batches) - 0.4)
        ax.set_xticks(range(len(batches)), labels=[f"{b:g}" for b in batches])
        ax.set_xlabel("Series per request")
        ax.set_ylabel("Call duration vs. median (%)")
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:+.0f}%")
        _style_axes(ax, grid="y")
        handles, labels = ax.get_legend_handles_labels()
        _legend_above(ax, handles, labels, title)
    return ax


def plot_cost_per_million(
    costs: pd.DataFrame,
    *,
    model: str,
    context_length: int,
    instances: Sequence[str] | None = None,
    break_even: pd.DataFrame | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Cost per million forecasts per instance, cheapest first, GPU instances highlighted.

    ``costs`` is ``results/09_cost_per_million.csv``. Pass ``instances`` to restrict the chart
    to a comparable set: the CPU throughput behind these costs was measured on one machine, so a
    rate for an instance with fewer vCPUs than that machine would be flattered by it.

    Passing ``break_even`` (from
    ``results/09_break_even.csv``) adds the rate-only speed-up a GPU instance needs to be the
    cheaper choice, which is the one number in the figure that does not depend on a local
    measurement.
    """
    required = ["instance", "model", "context_length", "usd_per_million_forecasts", "device"]
    missing = [column for column in required if column not in costs.columns]
    if missing:
        raise ValueError(f"costs is missing the columns {missing}.")
    rows = costs[(costs["model"] == model) & (costs["context_length"] == context_length)]
    if instances is not None:
        rows = rows[rows["instance"].isin(list(instances))]
    if rows.empty:
        raise ValueError(f"No rows for model={model!r} and context_length={context_length}.")
    rows = rows.sort_values("usd_per_million_forecasts", ascending=False)
    is_gpu = (rows["device"] == "cuda").to_numpy()
    values = rows["usd_per_million_forecasts"].to_numpy(dtype="float64")
    positions = np.arange(len(rows))
    with style_context():
        if ax is None:
            _, ax = plt.subplots(figsize=(8.0, 0.42 * len(rows) + 1.6), layout="constrained")
        ax.barh(positions, values, height=0.6, color=np.where(is_gpu, BLUE, NEUTRAL_BAR), zorder=2)
        for y, value, gpu in zip(positions, values, is_gpu, strict=True):
            ax.annotate(
                f"${value:,.2f}",
                (value, y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=9.5,
                color=INK_PRIMARY if gpu else INK_SECONDARY,
                fontweight="bold" if gpu else "normal",
            )
        ax.set_yticks(positions, labels=rows["instance"].tolist())
        ax.set_xlim(0, values.max() * 1.22)
        ax.set_xlabel("USD per million 24-step forecasts (on-demand rates, us-east-1)")
        _style_axes(ax, grid="x")
        if break_even is not None and not break_even.empty:
            row = break_even.iloc[0]
            ax.annotate(
                f"Rates alone: {row['gpu_instance']} must be "
                f"{row['speedup_to_break_even']:.1f}x faster than {row['cpu_instance']}",
                (0.995, 0.98),  # top right: the GPU bars are short, so the corner is empty
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=9,
                color=INK_SECONDARY,
            )
        if title is not None:
            ax.set_title(title, loc="left", color=INK_PRIMARY)
    return ax


# --- Interval coverage by month --------------------------------------------------------------

_MODEL_STYLES = {
    "zero-shot": (INK_SECONDARY, "solid"),
    "conformal_additive": (BLUE, "solid"),
    "conformal_scaled": (BLUE, (0, (4, 2))),
}


def plot_coverage_by_month(
    by_month: pd.DataFrame,
    *,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Achieved coverage of one interval per calendar month, one line per model.

    ``by_month`` is the output of :func:`c2p.conformal.coverage_by_period`: ``model``,
    ``period`` (``YYYY-MM``), ``coverage``, ``n`` and ``nominal``. The dashed line is the
    nominal level; the vertical distance to it is the promise the band did not keep that month.
    """
    required = ["model", "period", "coverage", "nominal"]
    missing = [column for column in required if column not in by_month.columns]
    if missing:
        raise ValueError(f"by_month is missing the columns {missing}.")
    if by_month.empty:
        raise ValueError("by_month is empty.")
    nominal = float(by_month["nominal"].iloc[0])
    periods = sorted(by_month["period"].unique())
    with style_context():
        if ax is None:
            _, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
        ax.axhline(
            nominal,
            color=INK_SECONDARY,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=2,
            label=f"nominal {nominal:.0%}",
        )
        for model, group in by_month.groupby("model", sort=False):
            group = group.set_index("period").reindex(periods)
            color, style = _MODEL_STYLES.get(str(model), (NEUTRAL_BAR, "solid"))
            ax.plot(
                range(len(periods)),
                group["coverage"].to_numpy(),
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                color=color,
                linestyle=style,
                label=str(model),
                zorder=3,
            )
        ax.set_xticks(range(len(periods)), labels=periods, rotation=45, ha="right")
        ax.set_xlabel("Month")
        ax.set_ylabel("Share of hours inside the band")
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        _style_axes(ax, grid="y")
        handles, labels = ax.get_legend_handles_labels()
        _legend_above(ax, handles, labels, title)
    return ax
