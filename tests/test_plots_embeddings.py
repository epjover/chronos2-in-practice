"""Tests for ``c2p.plots.embeddings``: what is drawn, not how it looks."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from c2p.plots.embeddings import plot_analogs, plot_dendrogram


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_analogs_draws_every_analog_the_mean_and_the_actual() -> None:
    actual = np.arange(24.0)
    analogs = np.vstack([actual + shift for shift in (-1.0, 0.0, 1.0)])
    ax = plot_analogs(actual, analogs, analogs.mean(axis=0), title="Three days like this one")
    assert len(ax.lines) == 5
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["Actual", "Average of the analogs", "The 3 analog days"]
    assert ax.get_xlim() == (1.0, 24.0)


def test_plot_analogs_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="analogs must have shape"):
        plot_analogs(np.zeros(24), np.zeros((3, 12)), np.zeros(24))
    with pytest.raises(ValueError, match="forecast has"):
        plot_analogs(np.zeros(24), np.zeros((3, 24)), np.zeros(12))


def test_plot_dendrogram_puts_similar_series_on_adjacent_leaves() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.99, 0.05], [0.05, 0.99]])
    ax = plot_dendrogram(embeddings, ["solar", "wind", "solar forecast", "wind forecast"])
    leaves = [tick.get_text() for tick in ax.get_yticklabels()]
    assert sorted(leaves) == ["solar", "solar forecast", "wind", "wind forecast"]
    assert abs(leaves.index("solar") - leaves.index("solar forecast")) == 1
    assert abs(leaves.index("wind") - leaves.index("wind forecast")) == 1


def test_plot_dendrogram_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="one row per label"):
        plot_dendrogram(np.zeros((3, 2)), ["a", "b"])
    with pytest.raises(ValueError, match="at least two"):
        plot_dendrogram(np.ones((1, 2)), ["a"])
