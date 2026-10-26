"""Contract tests for ``c2p.plots.evaluation`` on synthetic frames (no model, no network)."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from c2p.plots.evaluation import (
    coverage_long,
    plot_backtest_windows,
    plot_coverage,
    plot_skill_scores,
    window_schedule,
)
from c2p.schema import INTERVALS


@pytest.fixture
def schedule() -> pd.DataFrame:
    cutoffs = pd.date_range("2018-01-01 22:00", periods=20, freq="24h")
    return pd.DataFrame(
        {
            "window": range(20),
            "cutoff": cutoffs,
            "context_start": cutoffs - pd.Timedelta(hours=2048),
            "horizon_end": cutoffs + pd.Timedelta(hours=24),
            "n_context": 2048,
        }
    )


@pytest.fixture
def skill() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["chronos-2", "chronos-bolt", "auto_ets", "seasonal_naive"],
            "baseline": "seasonal_naive",
            "metric": "WQL",
            "skill": [0.42, 0.31, -0.18, 0.0],
            "ci_low": [0.38, 0.27, -0.25, 0.0],
            "ci_high": [0.46, 0.35, -0.11, 0.0],
            "win_rate": [0.91, 0.78, 0.33, 0.5],
            "n_units": 365,
        }
    )


def test_window_schedule_reads_the_spans_from_the_windows() -> None:
    from c2p.eval import BacktestConfig, rolling_origin_windows

    timestamps = pd.date_range("2024-01-01", periods=400, freq="h")
    df = pd.DataFrame(
        {"item_id": "A", "timestamp": timestamps, "target": range(400), "price": range(400)}
    )
    config = BacktestConfig(prediction_length=24, n_windows=3, freq="h", season_length=24)
    windows = rolling_origin_windows(df, config)
    out = window_schedule(windows)
    assert list(out["window"]) == [0, 1, 2]
    assert out["horizon_end"].iloc[-1] == timestamps[-1]
    assert (out["cutoff"] < out["horizon_end"]).all()
    assert (out["context_start"] <= out["cutoff"]).all()
    # n_context counts steps per series, not rows across series.
    assert out["n_context"].iloc[0] == len(windows[0].context)


def test_window_schedule_rejects_an_empty_backtest() -> None:
    with pytest.raises(ValueError, match="windows is empty"):
        window_schedule([])


def test_backtest_windows_collapses_a_long_backtest(schedule: pd.DataFrame) -> None:
    ax = plot_backtest_windows(schedule, n_shown=6)
    try:
        labels = [text.get_text() for text in ax.get_yticklabels()]
        assert labels.count("…") == 1, labels
        assert len(labels) == 7  # 6 windows plus the gap row
        notes = [text.get_text() for text in ax.texts]
        assert any("14 more windows" in note for note in notes), notes
    finally:
        plt.close(ax.figure)


def test_backtest_windows_draws_every_window_when_they_fit(schedule: pd.DataFrame) -> None:
    ax = plot_backtest_windows(schedule.head(4), n_shown=8)
    try:
        assert [text.get_text() for text in ax.get_yticklabels()] == [
            f"window {i}" for i in range(4)
        ]
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"n_shown": 1}, "n_shown must be at least 2"), ({}, "missing the column")],
)
def test_backtest_windows_rejects_bad_input(
    schedule: pd.DataFrame, kwargs: dict, match: str
) -> None:
    frame = schedule if kwargs else schedule.drop(columns="cutoff")
    with pytest.raises(ValueError, match=match):
        plot_backtest_windows(frame, **kwargs)


def test_skill_scores_drops_the_baseline_and_labels_win_rates(skill: pd.DataFrame) -> None:
    ax = plot_skill_scores(skill)
    try:
        labels = [text.get_text() for text in ax.get_yticklabels()]
        assert "seasonal_naive" not in labels
        assert labels[-1] == "chronos-2", "best model goes on top"
        annotations = " ".join(text.get_text() for text in ax.texts)
        assert "42.0  (91% of windows)" in annotations
        assert "Skill score vs seasonal_naive" in ax.get_xlabel()
    finally:
        plt.close(ax.figure)


def test_skill_scores_needs_more_than_the_baseline(skill: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="only the baseline"):
        plot_skill_scores(skill[skill["model"] == "seasonal_naive"])


def test_coverage_long_matches_the_summary_columns() -> None:
    summary = pd.DataFrame(
        {
            "model": ["chronos-2", "auto_ets"],
            "coverage_80": [0.79, 0.68],
            "coverage_98": [0.97, 0.9],
        }
    )
    out = coverage_long(summary, INTERVALS)
    assert set(out.columns) == {"model", "nominal", "achieved"}
    assert len(out) == 4
    assert out.set_index(["model", "nominal"]).loc[("auto_ets", 0.98), "achieved"] == 0.9


def test_coverage_draws_one_marker_per_model_and_level() -> None:
    summary = pd.DataFrame(
        {
            "model": ["chronos-2", "auto_ets"],
            "coverage_80": [0.79, 0.68],
            "coverage_98": [0.97, 0.9],
        }
    )
    ax = plot_coverage(coverage_long(summary, INTERVALS))
    try:
        markers = [line for line in ax.lines if line.get_linestyle() == "None"]
        assert len(markers) == 4
        assert [text.get_text() for text in ax.get_yticklabels()] == [
            "80% interval",
            "98% interval",
        ]
    finally:
        plt.close(ax.figure)


def test_coverage_rejects_an_empty_frame() -> None:
    with pytest.raises(ValueError, match="coverage is empty"):
        plot_coverage(pd.DataFrame({"model": [], "nominal": [], "achieved": []}))


def test_backtest_windows_can_truncate_the_context(schedule: pd.DataFrame) -> None:
    """Only the most recent part of the context is drawn, and the legend says so."""
    ax = plot_backtest_windows(schedule.head(3), context_shown=pd.Timedelta(days=7))
    try:
        drawn = min(patch.get_x() for patch in ax.patches)
        earliest_allowed = mdates.date2num(schedule["cutoff"].iloc[0] - pd.Timedelta(days=7))
        assert drawn >= earliest_allowed - 1e-6, "context drawn earlier than allowed"
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert any("last 168 h shown" in label for label in labels), labels
    finally:
        plt.close(ax.figure)


def test_backtest_windows_rejects_a_non_positive_context_window(schedule: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="context_shown must be a positive"):
        plot_backtest_windows(schedule, context_shown=pd.Timedelta(0))
