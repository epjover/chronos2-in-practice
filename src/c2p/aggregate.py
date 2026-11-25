"""From forecasts of the hours to a forecast of the day: what quantiles can and cannot give.

Chronos-2 returns quantiles for each step of the horizon, one step at a time. A decision often
needs something else: the energy of the whole day, the sales of the whole week. The point
forecast of a total is the sum of the point forecasts. The *interval* of a total is not
determined by the hourly quantiles, because it depends on how the hourly errors move together,
and marginal quantiles say nothing about that.

This module makes the two textbook assumptions explicit, so their coverage can be measured
instead of assumed:

- ``"comonotonic"``: the hourly errors move together perfectly. The quantile of the total is
  the sum of the hourly quantiles. This is the widest interval the marginals allow.
- ``"independent"``: the hourly errors are unrelated. The spread of each hour around its
  median adds in quadrature, as variances do. This is much narrower.

The truth lies between the two, and the alternative that needs no assumption is to forecast
the aggregated series directly (:func:`daily_totals` builds it).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from c2p.eval import ACTUAL_COL, CUTOFF_COL, MODEL_COL, WINDOW_COL
from c2p.schema import (
    ID_COL,
    PREDICTION_COL,
    TARGET_NAME_COL,
    TIMESTAMP_COL,
    quantile_col,
)

__all__ = ["DAY_OFFSET", "aggregate_forecasts", "daily_totals"]

# The day-ahead horizon of this series starts at 23:00 UTC (midnight in Madrid in winter).
# Shifting the timestamps by one hour before flooring makes those 24 hours one calendar day.
DAY_OFFSET = pd.Timedelta(hours=1)
STEPS_PER_DAY = 24

Method = Literal["comonotonic", "independent"]


def _day(timestamps: pd.Series, offset: pd.Timedelta) -> pd.Series:
    return (timestamps + offset).dt.floor("D")


def daily_totals(
    frame: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    offset: pd.Timedelta = DAY_OFFSET,
    steps_per_day: int = STEPS_PER_DAY,
) -> pd.DataFrame:
    """Sum the hourly ``columns`` of a series frame into days, on a regular daily grid.

    A day is the ``steps_per_day`` rows whose ``timestamp + offset`` fall on the same date, and
    its row is stamped with that date. A day with a missing value has a NaN total rather than
    an under-counted one. The partial days at the two ends of a series (fewer than
    ``steps_per_day`` rows) are dropped, so the result starts and ends on a complete day.
    """
    columns = [columns] if isinstance(columns, str) else list(columns)
    keyed = frame[[ID_COL, *columns]].assign(**{TIMESTAMP_COL: _day(frame[TIMESTAMP_COL], offset)})
    grouped = keyed.groupby([ID_COL, TIMESTAMP_COL], sort=True)
    totals = grouped[columns].sum(min_count=steps_per_day)
    totals["_rows"] = grouped[columns[0]].size()
    totals = totals.reset_index()
    full = totals["_rows"] == steps_per_day
    by_series = full.groupby(totals[ID_COL])
    # Keep from the first complete day to the last one of each series.
    inside = by_series.cummax() & by_series.transform(lambda day: day[::-1].cummax()[::-1])
    totals.loc[~full, columns] = np.nan
    return totals.loc[inside].drop(columns="_rows").reset_index(drop=True)


def aggregate_forecasts(
    forecasts: pd.DataFrame,
    *,
    quantile_levels: Sequence[float],
    method: Method,
    name: str,
    offset: pd.Timedelta = DAY_OFFSET,
    steps_per_day: int = STEPS_PER_DAY,
) -> pd.DataFrame:
    """Turn hourly backtest forecasts into forecasts of the daily totals.

    One row per (window, series, target, day), in the format of a backtest frame, so the
    result can be scored against :func:`daily_totals` with :func:`c2p.eval.evaluate`: the
    ``timestamp`` is the day, the ``cutoff`` is the day before the window's first day, the
    point forecast is the sum of the hourly point forecasts and the quantiles follow
    ``method`` (see the module docstring). Days of the horizon with fewer than
    ``steps_per_day`` rows are dropped. ``name`` is the model name of the result.
    """
    if method not in ("comonotonic", "independent"):
        raise ValueError(f"method must be 'comonotonic' or 'independent', got {method!r}.")
    levels = sorted(float(level) for level in quantile_levels)
    columns = [quantile_col(level) for level in levels]
    keys = [WINDOW_COL, ID_COL, TARGET_NAME_COL]
    frame = forecasts[[*keys, TIMESTAMP_COL, PREDICTION_COL, *columns]].copy()
    frame[TIMESTAMP_COL] = _day(frame[TIMESTAMP_COL], offset)
    median = frame[quantile_col(0.5)] if 0.5 in levels else frame[PREDICTION_COL]
    if method == "independent":
        # Each hour's distance from its median, squared: these add up like variances.
        for column in columns:
            frame[column] = (frame[column] - median) ** 2
    frame["_median"] = median
    grouped = frame.groupby([*keys, TIMESTAMP_COL], sort=True)
    totals = grouped[[PREDICTION_COL, "_median", *columns]].sum()
    totals = totals[grouped[PREDICTION_COL].size() == steps_per_day].reset_index()
    if method == "independent":
        for level, column in zip(levels, columns, strict=True):
            sign = -1.0 if level < 0.5 else 1.0
            totals[column] = totals["_median"] + sign * np.sqrt(totals[column])
    first_day = totals.groupby(keys, sort=False)[TIMESTAMP_COL].transform("min")
    totals[CUTOFF_COL] = first_day - pd.Timedelta(days=1)
    totals[MODEL_COL] = name
    if ACTUAL_COL in forecasts.columns:
        actual = forecasts[[*keys, ACTUAL_COL]].assign(
            **{TIMESTAMP_COL: _day(forecasts[TIMESTAMP_COL], offset)}
        )
        actual = actual.groupby([*keys, TIMESTAMP_COL], sort=True)[ACTUAL_COL].sum(
            min_count=steps_per_day
        )
        totals = totals.merge(actual.reset_index(), on=[*keys, TIMESTAMP_COL], how="left")
    ordered = [ID_COL, TIMESTAMP_COL, TARGET_NAME_COL, PREDICTION_COL, *columns]
    extra = [WINDOW_COL, CUTOFF_COL, MODEL_COL]
    extra += [ACTUAL_COL] if ACTUAL_COL in totals.columns else []
    return totals[[*ordered, *extra]]
