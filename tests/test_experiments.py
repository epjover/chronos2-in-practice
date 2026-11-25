from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from c2p.eval import rolling_origin_windows
from c2p.experiments import (
    LOAD,
    TEST_DAYS,
    TRAIN_END,
    VALIDATION_END,
    day_ahead_config,
    rotating_sample_config,
)
from c2p.schema import ID_COL, TIMESTAMP_COL


@pytest.fixture(scope="module")
def spain_grid() -> pd.DataFrame:
    """Same hourly UTC grid as the Spanish dataset (2014-12-31 23:00 to 2018-12-31 22:00)."""
    timestamps = pd.date_range("2014-12-31 23:00", "2018-12-31 22:00", freq="h")
    return pd.DataFrame(
        {ID_COL: "ES", TIMESTAMP_COL: timestamps, LOAD: np.arange(len(timestamps), dtype=float)}
    )


def test_day_ahead_windows_cover_2018_madrid_days(spain_grid: pd.DataFrame) -> None:
    windows = rolling_origin_windows(spain_grid, day_ahead_config(fast_mode=False))
    assert len(windows) == TEST_DAYS
    assert windows[0].cutoff == VALIDATION_END
    first_hour_madrid = (
        windows[0].actuals[TIMESTAMP_COL].iloc[0].tz_localize("UTC").tz_convert("Europe/Madrid")
    )
    assert first_hour_madrid == pd.Timestamp("2018-01-01 00:00", tz="Europe/Madrid")
    assert windows[-1].actuals[TIMESTAMP_COL].iloc[-1] == spain_grid[TIMESTAMP_COL].iloc[-1]


def test_training_period_ends_before_validation_and_test() -> None:
    assert TRAIN_END < VALIDATION_END
    assert TRAIN_END.tz_localize("UTC").tz_convert("Europe/Madrid").hour == 23


@pytest.mark.parametrize("config_fn", [day_ahead_config, rotating_sample_config])
def test_fast_mode_windows_stay_inside_2018(spain_grid: pd.DataFrame, config_fn) -> None:
    windows = rolling_origin_windows(spain_grid, config_fn(fast_mode=True))
    assert windows[0].cutoff >= VALIDATION_END
    assert len(windows) < 13


def _forecast_weekdays(windows: list) -> pd.Series:
    first_hours = pd.DatetimeIndex([window.actuals[TIMESTAMP_COL].iloc[0] for window in windows])
    return pd.Series(first_hours.tz_localize("UTC").tz_convert("Europe/Madrid").day_name())


def test_rotating_sample_spans_2018_and_every_weekday(spain_grid: pd.DataFrame) -> None:
    windows = rolling_origin_windows(spain_grid, rotating_sample_config(fast_mode=False))
    assert len(windows) == 73
    assert windows[0].cutoff >= VALIDATION_END
    assert {window.cutoff.hour for window in windows} == {22}
    counts = _forecast_weekdays(windows).value_counts()
    assert len(counts) == 7
    assert counts.min() >= 10 and counts.max() <= 11


def test_rotating_sample_with_week_ahead_horizon_does_not_overlap(spain_grid: pd.DataFrame) -> None:
    config = rotating_sample_config(fast_mode=False, prediction_length=168, step_days=8)
    windows = rolling_origin_windows(spain_grid, config)
    assert len(windows) == 45
    assert windows[0].cutoff >= VALIDATION_END
    assert _forecast_weekdays(windows).nunique() == 7
    assert all(
        earlier.actuals[TIMESTAMP_COL].iloc[-1] < later.actuals[TIMESTAMP_COL].iloc[0]
        for earlier, later in pairwise(windows)
    )


def test_fast_mode_sample_does_not_repeat_one_weekday(spain_grid: pd.DataFrame) -> None:
    windows = rolling_origin_windows(spain_grid, rotating_sample_config(fast_mode=True))
    assert len(windows) == 6
    assert _forecast_weekdays(windows).nunique() == 6


@pytest.mark.parametrize("step_days", [0, 7, 14])
def test_rotating_sample_rejects_a_weekly_step(step_days: int) -> None:
    with pytest.raises(ValueError, match="same weekday"):
        rotating_sample_config(fast_mode=False, step_days=step_days)
