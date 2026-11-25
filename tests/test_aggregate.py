"""Tests for ``c2p.aggregate``: daily totals and the two ways to add up hourly quantiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from c2p.aggregate import aggregate_forecasts, daily_totals
from c2p.baselines import SeasonalNaive
from c2p.eval import (
    ACTUAL_COL,
    CUTOFF_COL,
    MODEL_COL,
    WINDOW_COL,
    BacktestConfig,
    backtest,
    evaluate,
)
from c2p.schema import (
    EVAL_QUANTILES,
    ID_COL,
    PREDICTION_COL,
    TARGET_NAME_COL,
    TIMESTAMP_COL,
    quantile_col,
)

LEVELS = (0.1, 0.5, 0.9)


def hourly_frame(days: int = 40, *, start: str = "2018-01-01 23:00") -> pd.DataFrame:
    """An hourly series whose days run from 23:00 to 22:00, like the day-ahead horizon."""
    stamps = pd.date_range(start, periods=days * 24, freq="h")
    hours = np.arange(len(stamps))
    values = 100.0 + 10.0 * np.sin(2 * np.pi * hours / 24) + 0.01 * hours
    return pd.DataFrame({ID_COL: "s", TIMESTAMP_COL: stamps, "y": values})


def hourly_forecast(spread: float = 2.0) -> pd.DataFrame:
    """One window of two days: every hour forecasts 10 with quantiles 10 -/+ ``spread``."""
    stamps = pd.date_range("2018-01-01 23:00", periods=48, freq="h")
    frame = pd.DataFrame(
        {
            ID_COL: "s",
            TIMESTAMP_COL: stamps,
            TARGET_NAME_COL: "y",
            PREDICTION_COL: 10.0,
            WINDOW_COL: 0,
            ACTUAL_COL: 11.0,
        }
    )
    frame[quantile_col(0.1)] = 10.0 - spread
    frame[quantile_col(0.5)] = 10.0
    frame[quantile_col(0.9)] = 10.0 + spread
    return frame


def test_daily_totals_keep_a_regular_grid_and_drop_the_partial_days_at_the_ends() -> None:
    frame = hourly_frame(3)
    tail = frame.iloc[:5].assign(
        **{TIMESTAMP_COL: frame[TIMESTAMP_COL].max() + pd.to_timedelta(np.arange(1, 6), "h")}
    )
    frame = pd.concat([frame, tail], ignore_index=True)
    frame.loc[30, "y"] = np.nan  # a missing hour in the second day
    totals = daily_totals(frame, "y")
    days = pd.date_range("2018-01-02", periods=3, freq="D")
    assert totals[TIMESTAMP_COL].tolist() == list(days)  # the five-hour fourth day is dropped
    assert totals["y"].iloc[0] == pytest.approx(hourly_frame(3)["y"].iloc[:24].sum())
    assert totals["y"].notna().tolist() == [True, False, True]  # no under-counted total


def test_comonotonic_totals_sum_every_quantile() -> None:
    totals = aggregate_forecasts(
        hourly_forecast(), quantile_levels=LEVELS, method="comonotonic", name="sum"
    )
    assert len(totals) == 2
    assert totals[TIMESTAMP_COL].tolist() == [
        pd.Timestamp("2018-01-02"),
        pd.Timestamp("2018-01-03"),
    ]
    assert (totals[CUTOFF_COL] == pd.Timestamp("2018-01-01")).all()
    assert (totals[MODEL_COL] == "sum").all()
    assert totals[PREDICTION_COL].tolist() == [240.0, 240.0]
    assert totals[quantile_col(0.1)].tolist() == [240.0 - 48.0] * 2
    assert totals[quantile_col(0.9)].tolist() == [240.0 + 48.0] * 2
    assert totals[ACTUAL_COL].tolist() == [264.0, 264.0]


def test_independent_totals_add_the_spread_in_quadrature() -> None:
    totals = aggregate_forecasts(
        hourly_forecast(), quantile_levels=LEVELS, method="independent", name="root"
    )
    half_width = 2.0 * np.sqrt(24)  # 24 hours, each 2 away from its median
    assert totals[quantile_col(0.5)].tolist() == [240.0, 240.0]
    assert totals[quantile_col(0.1)].iloc[0] == pytest.approx(240.0 - half_width)
    assert totals[quantile_col(0.9)].iloc[0] == pytest.approx(240.0 + half_width)


def test_aggregate_drops_partial_days_and_rejects_unknown_methods() -> None:
    forecast = hourly_forecast().iloc[:30]  # one full day and six hours of the next
    totals = aggregate_forecasts(forecast, quantile_levels=LEVELS, method="comonotonic", name="m")
    assert len(totals) == 1
    with pytest.raises(ValueError, match="method"):
        aggregate_forecasts(forecast, quantile_levels=LEVELS, method="copula", name="m")


def test_aggregated_forecasts_are_scored_by_the_same_harness_as_a_daily_model() -> None:
    hourly = hourly_frame(40)
    daily = daily_totals(hourly, "y")
    hourly_config = BacktestConfig(
        prediction_length=24, n_windows=5, freq="h", season_length=24, target="y"
    )
    daily_config = BacktestConfig(
        prediction_length=1, n_windows=5, freq="D", season_length=7, target="y"
    )
    forecasts = backtest(SeasonalNaive(season_length=24), hourly, hourly_config)
    summed = aggregate_forecasts(
        forecasts, quantile_levels=EVAL_QUANTILES, method="comonotonic", name="hourly, summed"
    )
    metrics = evaluate(summed, daily, daily_config)
    assert metrics[WINDOW_COL].tolist() == [0, 1, 2, 3, 4]
    direct = evaluate(
        backtest(SeasonalNaive(season_length=7), daily, daily_config), daily, daily_config
    )
    assert metrics[CUTOFF_COL].tolist() == direct[CUTOFF_COL].tolist()
    assert metrics["WAPE"].max() < 0.01  # a daily cycle plus a slow trend: nearly exact
