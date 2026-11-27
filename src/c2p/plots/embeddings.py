"""Figures for the embedding experiments: analog days and a similarity tree of series."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.cluster.hierarchy import dendrogram, linkage

from c2p.plotting import (
    BLUE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    NEUTRAL_BAR,
    _legend_above,
    _style_axes,
    set_axes_title,
    style_context,
)

__all__ = ["plot_analogs", "plot_dendrogram"]


def plot_analogs(
    actual: Sequence[float],
    analogs: np.ndarray,
    forecast: Sequence[float],
    *,
    title: str | None = None,
    xlabel: str = "Hours ahead",
    ylabel: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """One day against the days the embedding retrieved for it.

    ``analogs`` has one row per retrieved day, already brought to the level and scale of the
    day being forecast; ``forecast`` is their average and ``actual`` what happened. Thin grey
    lines that hug the black one mean the neighbours in embedding space were days of the same
    shape.
    """
    analogs = np.asarray(analogs, dtype="float64")
    if analogs.ndim != 2 or analogs.shape[1] != len(actual) or len(forecast) != len(actual):
        raise ValueError(
            f"analogs must have shape (n_analogs, {len(actual)}), got {analogs.shape}; "
            f"forecast has {len(forecast)} steps."
        )
    steps = np.arange(1, len(actual) + 1)
    with style_context():
        if ax is None:
            _, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
        for row in analogs:
            (thin,) = ax.plot(steps, row, color=NEUTRAL_BAR, linewidth=1.0, zorder=2)
        (mean,) = ax.plot(steps, forecast, color=BLUE, linewidth=2.0, zorder=3)
        (truth,) = ax.plot(steps, actual, color=INK_PRIMARY, linewidth=1.6, zorder=4)
        _style_axes(ax, grid="y")
        ax.set_xlim(1, len(actual))
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        labels = ["Actual", "Average of the analogs", f"The {len(analogs)} analog days"]
        _legend_above(ax, [truth, mean, thin], labels, title)
    return ax


def plot_dendrogram(
    embeddings: np.ndarray,
    labels: Sequence[str],
    *,
    title: str | None = None,
    xlabel: str = "Cosine distance at which two groups merge",
    ax: Axes | None = None,
) -> Axes:
    """Tree of series by the similarity of their embeddings (average linkage, cosine distance).

    Two series that join near the left edge have nearly the same embedding; the further right
    two branches meet, the less alike they are. The tree is drawn sideways so that the series
    names stay readable.
    """
    embeddings = np.asarray(embeddings, dtype="float64")
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError(
            f"embeddings must have one row per label: got {embeddings.shape} and "
            f"{len(labels)} labels."
        )
    if len(labels) < 2:
        raise ValueError("A tree needs at least two series.")
    tree = linkage(embeddings, method="average", metric="cosine")
    with style_context():
        if ax is None:
            height = max(3.0, 0.26 * len(labels) + 1.2)
            _, ax = plt.subplots(figsize=(8.0, height), layout="constrained")
        dendrogram(
            tree,
            labels=list(labels),
            orientation="right",
            color_threshold=0.0,
            above_threshold_color=INK_SECONDARY,
            ax=ax,
        )
        _style_axes(ax, grid="x")
        ax.tick_params(axis="y", labelsize=9, labelcolor=INK_PRIMARY)
        ax.tick_params(axis="x", colors=INK_MUTED)
        ax.set_xlabel(xlabel)
        if title is not None:
            set_axes_title(ax, title)
    return ax
