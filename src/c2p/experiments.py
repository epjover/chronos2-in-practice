"""Experiment protocol shared by the articles, so every number is measured on the same windows.

The test period is 2018, the last year of the Spanish dataset. Day-ahead windows forecast the
next 24 hours from 22:00 UTC (23:00 in Madrid in winter), one window per day. Data before 2018
is history: fine-tuning (article 4) trains on data up to :data:`TRAIN_END` and validates on
the second half of 2017, so the 2018 test windows are never seen during training.
"""

from __future__ import annotations

import pandas as pd

from c2p.eval import BacktestConfig

FREQ = "h"
DAILY_SEASON = 24
WEEKLY_SEASON = 168

LOAD = "load_actual"
PRICE = "price_actual"
# The grid operator's (TSO) day-ahead forecasts, published the day before: industrial
# baselines for LOAD and PRICE, and known-future covariates for horizons up to 24 hours.
TSO_LOAD_FORECAST = "load_forecast_tso"
TSO_PRICE_FORECAST = "price_day_ahead_tso"
TSO_COVARIATES: tuple[str, ...] = (
    "load_forecast_tso",
    "solar_forecast_tso",
    "wind_onshore_forecast_tso",
)

# Naive UTC timestamps. The data ends at 2018-12-31 22:00 UTC (23:00 in Madrid).
TRAIN_END = pd.Timestamp("2017-06-30 21:00")
VALIDATION_END = pd.Timestamp("2017-12-31 22:00")
TEST_DAYS = 365


def day_ahead_config(
    fast_mode: bool,
    *,
    target: str | tuple[str, ...] = LOAD,
    known_covariates: tuple[str, ...] = (),
    max_context: int | None = None,
) -> BacktestConfig:
    """Day-ahead backtest over 2018: 365 daily windows, or 12 monthly ones in fast mode."""
    return BacktestConfig(
        prediction_length=24,
        n_windows=12 if fast_mode else TEST_DAYS,
        step=24 * 30 if fast_mode else 24,
        freq=FREQ,
        season_length=DAILY_SEASON,
        target=target,
        known_covariates=known_covariates,
        max_context=max_context,
    )


def weekly_sample_config(
    fast_mode: bool,
    *,
    prediction_length: int = 24,
    target: str | tuple[str, ...] = LOAD,
    known_covariates: tuple[str, ...] = (),
    max_context: int | None = None,
) -> BacktestConfig:
    """Cheaper sample of 2018 for sweeps: one window per week (52), or 6 in fast mode."""
    n_windows = 6 if fast_mode else 52
    step = 24 * 7 * (8 if fast_mode else 1)
    return BacktestConfig(
        prediction_length=prediction_length,
        n_windows=n_windows,
        step=step,
        freq=FREQ,
        season_length=DAILY_SEASON,
        target=target,
        known_covariates=known_covariates,
        max_context=max_context,
    )
