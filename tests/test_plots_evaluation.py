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
    plot_error_by_step,
    plot_interval_bars,
    plot_reliability,
    plot_skill_scores,
    plot_what_if,
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


def what_if_frame() -> pd.DataFrame:
    times = pd.date_range("2018-12-30 23:00", periods=24, freq="h")
    frames = []
    for name, shift in (("Operator's forecast", 0.0), ("Wind x1.5", -0.8), ("Wind x0.5", 1.1)):
        median = 70.0 + shift + pd.Series(range(24)) % 6
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": times,
                    "predictions": median,
                    "0.1": median - 3.0,
                    "0.9": median + 3.0,
                    "scenario": name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_what_if_overlays_every_scenario_and_labels_its_mean() -> None:
    frame = what_if_frame()
    actuals = pd.DataFrame({"timestamp": frame["timestamp"].unique(), "price": 71.0})
    ax = plot_what_if(frame, baseline="Operator's forecast", actuals=actuals, target="price")
    try:
        # The legend may wrap into rows, which reorders its entries: compare as a set.
        labels = {text.get_text() for text in ax.get_legend().get_texts()}
        assert labels == {
            "Actual",
            "Operator's forecast (mean 72.50)",
            "Wind x1.5 (mean 71.70)",
            "Wind x0.5 (mean 73.60)",
            "80% interval, Operator's forecast",
        }
        assert len(ax.lines) == 4  # the actual and three medians, on one pair of axes
        assert len(ax.collections) == 1  # one band: the baseline's interval
    finally:
        plt.close(ax.figure)


def test_what_if_rejects_bad_input() -> None:
    frame = what_if_frame()
    with pytest.raises(ValueError, match="not a scenario"):
        plot_what_if(frame, baseline="Nope")
    with pytest.raises(ValueError, match="missing the columns"):
        plot_what_if(frame.drop(columns="0.9"), baseline="Operator's forecast")
    with pytest.raises(ValueError, match="together"):
        plot_what_if(frame, baseline="Operator's forecast", target="price")


def test_reliability_draws_the_gap_to_the_level_per_model() -> None:
    levels = [0.1, 0.5, 0.9]
    table = pd.DataFrame(
        {
            "model": ["chronos-2"] * 3 + ["seasonal_naive"] * 3,
            "level": levels * 2,
            "observed": [0.14, 0.5, 0.86, 0.1, 0.5, 0.9],
        }
    )
    ax = plot_reliability(table, title="Narrow intervals")
    narrow, calibrated = ax.get_lines()[:2]
    assert narrow.get_ydata().tolist() == pytest.approx([4.0, 0.0, -4.0])
    assert calibrated.get_ydata().tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "chronos-2",
        "seasonal_naive",
    ]
    assert ax.get_title(loc="left") == "Narrow intervals"
    plt.close(ax.figure)


def test_error_by_step_draws_one_line_per_model_from_zero() -> None:
    table = pd.DataFrame(
        {
            "model": ["chronos-2"] * 3 + ["mstl"] * 3,
            "step": [1, 2, 3] * 2,
            "WQL": [0.02, 0.03, 0.04, 0.05, 0.05, 0.06],
            "WAPE": [0.03, 0.04, 0.05, 0.06, 0.06, 0.07],
        }
    )
    ax = plot_error_by_step(table, metric="WAPE", xlabel="Hours ahead")
    assert len(ax.get_lines()) == 2
    assert ax.get_lines()[1].get_ydata().tolist() == [0.06, 0.06, 0.07]
    assert ax.get_ylim()[0] == 0.0
    assert ax.get_xlabel() == "Hours ahead"
    plt.close(ax.figure)


@pytest.mark.parametrize("plot", [plot_reliability, plot_error_by_step])
def test_diagnostic_plots_reject_bad_frames(plot) -> None:
    with pytest.raises(ValueError, match="missing the columns"):
        plot(pd.DataFrame({"model": ["a"]}))
    empty = pd.DataFrame({"model": [], "level": [], "observed": [], "step": [], "WQL": []})
    with pytest.raises(ValueError, match="is empty"):
        plot(empty)


def test_interval_bars_sort_by_value_and_write_the_interval() -> None:
    table = pd.DataFrame(
        {
            "player": ["calendar", "wind", "load"],
            "mean": [0.002, -0.001, 0.004],
            "ci_low": [-0.001, -0.003, 0.001],
            "ci_high": [0.005, 0.001, 0.007],
        }
    )
    ax = plot_interval_bars(table, label="player", highlight="load", xlabel="WQL removed")
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["wind", "calendar", "load"]
    annotations = [text.get_text() for text in ax.texts]
    assert annotations[-1] == "+0.0040  [+0.0010, +0.0070]"
    assert ax.get_yticklabels()[-1].get_fontweight() == "bold"
    assert ax.get_xlabel() == "WQL removed"
    plt.close(ax.figure)


def test_interval_bars_reject_bad_frames() -> None:
    with pytest.raises(ValueError, match="missing the columns"):
        plot_interval_bars(pd.DataFrame({"player": ["a"]}), label="player")
    empty = pd.DataFrame({"player": [], "mean": [], "ci_low": [], "ci_high": []})
    with pytest.raises(ValueError, match="is empty"):
        plot_interval_bars(empty, label="player")
