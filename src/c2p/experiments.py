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
# The grid operator's (TSO) day-ahead forecasts, published the day before: known-future
# covariates for horizons up to 24 hours (article 3).
TSO_LOAD_FORECAST = "load_forecast_tso"
TSO_COVARIATES: tuple[str, ...] = (
    TSO_LOAD_FORECAST,
    "solar_forecast_tso",
    "wind_onshore_forecast_tso",
)

# Naive UTC timestamps. The data ends at 2018-12-31 22:00 UTC (23:00 in Madrid).
TRAIN_END = pd.Timestamp("2017-06-30 21:00")
VALIDATION_END = pd.Timestamp("2017-12-31 22:00")
TEST_DAYS = 365
VALIDATION_DAYS = 184  # 2017-07-01 to 2017-12-31, local days
# Sweeps run on a sample of the test year: one window every five days. Five and seven share no
# factor, so the forecast day walks through the week.
SAMPLE_STEP_DAYS = 5
# Day-ahead windows are one day apart and their errors are correlated from one day to the next
# (lag-1 autocorrelation of the log error ratio against the seasonal naive: 0.47). Confidence
# intervals over those windows resample runs of seven consecutive windows.
DAY_AHEAD_BLOCK = 7


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


def validation_config(
    fast_mode: bool,
    *,
    target: str | tuple[str, ...] = LOAD,
    known_covariates: tuple[str, ...] = (),
    max_context: int | None = None,
) -> BacktestConfig:
    """Day-ahead backtest over the validation period: the 184 days of the second half of 2017.

    Windows are counted back from the end of the frame, so pass a frame cut at
    :data:`VALIDATION_END` (``df[df["timestamp"] <= VALIDATION_END]``). This is where choices
    are made (a fine-tuning budget, a calibration): the 2018 windows of
    :func:`day_ahead_config` stay untouched until the choice is final. Fast mode keeps 12
    windows.
    """
    return BacktestConfig(
        prediction_length=24,
        n_windows=12 if fast_mode else VALIDATION_DAYS,
        step=24,
        freq=FREQ,
        season_length=DAILY_SEASON,
        target=target,
        known_covariates=known_covariates,
        max_context=max_context,
    )


def rotating_sample_config(
    fast_mode: bool,
    *,
    prediction_length: int = 24,
    step_days: int = SAMPLE_STEP_DAYS,
    target: str | tuple[str, ...] = LOAD,
    known_covariates: tuple[str, ...] = (),
    max_context: int | None = None,
) -> BacktestConfig:
    """Cheaper sample of 2018 for sweeps: one window every ``step_days`` days.

    The default of five days gives 73 windows, and each weekday is forecast ten or eleven times.
    ``step_days`` must not be a multiple of seven: a weekly step forecasts the same weekday in
    every window, and on Spanish load the skill against the seasonal naive runs from 53% on
    Tuesdays to 73% on Mondays and Saturdays, so a one-weekday sample measures that weekday.
    Use a step of at least ``prediction_length`` if the horizons must not overlap (eight days
    for a week-ahead horizon). Fast mode keeps 6 windows, 61 days apart.
    """
    if step_days < 1 or step_days % 7 == 0:
        raise ValueError(
            f"step_days must be a positive number of days that is not a multiple of 7, got "
            f"{step_days}: a weekly step would forecast the same weekday in every window."
        )
    n_windows = 6 if fast_mode else TEST_DAYS // step_days
    step = 24 * (61 if fast_mode else step_days)
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
