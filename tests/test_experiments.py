from __future__ import annotations

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
    weekly_sample_config,
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


@pytest.mark.parametrize("config_fn", [day_ahead_config, weekly_sample_config])
def test_fast_mode_windows_stay_inside_2018(spain_grid: pd.DataFrame, config_fn) -> None:
    windows = rolling_origin_windows(spain_grid, config_fn(fast_mode=True))
    assert windows[0].cutoff >= VALIDATION_END
    assert len(windows) < 13


def test_weekly_sample_spans_2018(spain_grid: pd.DataFrame) -> None:
    windows = rolling_origin_windows(spain_grid, weekly_sample_config(fast_mode=False))
    assert len(windows) == 52
    assert windows[0].cutoff >= VALIDATION_END
