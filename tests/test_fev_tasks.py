"""Tests for ``c2p.fev_tasks``: the translation of a published fev task into our protocol.

The fast tests use a stand-in task object, so they run without network access; the slow test
loads the real EPF-DE task and checks that the translation still lands on its published cutoffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from c2p.fev_tasks import (
    EPF_DE,
    assert_cutoffs_match,
    task_backtest_config,
    task_series_frame,
)
from c2p.schema import EVAL_QUANTILES, ID_COL, TIMESTAMP_COL

HOURS = 200


@dataclass
class FakeTask:
    """The slice of ``fev.Task`` that ``c2p.fev_tasks`` uses."""

    task_name: str = "fake"
    target: str = "target"
    id_column: str = "id"
    timestamp_column: str = "timestamp"
    freq: str = "h"
    horizon: int = 24
    num_windows: int = 3
    window_step_size: int = 24
    seasonality: int = 24
    known_dynamic_columns: list[str] = field(default_factory=lambda: ["load_forecast"])
    past_dynamic_columns: list[str] = field(default_factory=list)
    cutoffs: list = field(default_factory=lambda: [-72, -48, -24])

    def load_full_dataset(self) -> list[dict]:
        timestamps = pd.date_range("2024-01-01", periods=HOURS, freq="h")
        return [
            {
                "id": "DE",
                "timestamp": list(timestamps),
                "target": list(range(HOURS)),
                "load_forecast": [float(value) for value in range(HOURS)],
            }
        ]


def test_series_frame_flattens_the_dataset() -> None:
    frame = task_series_frame(FakeTask())
    assert list(frame.columns) == [ID_COL, TIMESTAMP_COL, "target", "load_forecast"]
    assert len(frame) == HOURS
    assert frame[ID_COL].unique().tolist() == ["DE"]
    assert frame[TIMESTAMP_COL].is_monotonic_increasing
    assert frame[TIMESTAMP_COL].dt.tz is None, "the schema uses timezone-naive timestamps"


def test_series_frame_keeps_text_columns_as_categories() -> None:
    class HolidayTask(FakeTask):
        def load_full_dataset(self) -> list[dict]:
            (record,) = super().load_full_dataset()
            record["holiday"] = ["a" if hour % 24 == 0 else "0" for hour in range(HOURS)]
            return [record]

    task = HolidayTask(known_dynamic_columns=["load_forecast", "holiday"])
    frame = task_series_frame(task)
    assert frame["holiday"].dtype == "category"
    assert sorted(frame["holiday"].cat.categories) == ["0", "a"]
    assert frame["load_forecast"].dtype == "float64"


def test_series_frame_repeats_the_static_columns_that_are_asked_for() -> None:
    class StoreTask(FakeTask):
        def load_full_dataset(self) -> list[dict]:
            (record,) = super().load_full_dataset()
            record.update({"store_type": "c", "distance": np.float32(1270.0), "interval": None})
            return [record]

    assert "store_type" not in task_series_frame(StoreTask()).columns
    frame = task_series_frame(StoreTask(), static_columns=["store_type", "distance", "interval"])
    assert list(frame.columns)[-3:] == ["store_type", "distance", "interval"]
    assert frame["store_type"].dtype == "category" and (frame["store_type"] == "c").all()
    assert frame["distance"].dtype == "float64" and (frame["distance"] == 1270.0).all()
    assert (frame["interval"] == "missing").all()
    with pytest.raises(ValueError, match="no static columns"):
        task_series_frame(StoreTask(), static_columns=["region"])


def test_backtest_config_copies_the_published_spec() -> None:
    config = task_backtest_config(FakeTask())
    assert config.prediction_length == 24
    assert config.n_windows == 3
    assert config.step == 24
    assert config.season_length == 24
    assert config.target == "target"
    assert config.known_covariates == ("load_forecast",)
    assert config.quantile_levels == EVAL_QUANTILES


def test_backtest_config_can_drop_the_covariates() -> None:
    assert task_backtest_config(FakeTask(), with_covariates=False).known_covariates == ()


def test_cutoffs_match_the_published_offsets() -> None:
    task = FakeTask()
    frame = task_series_frame(task)
    assert_cutoffs_match(task, frame, task_backtest_config(task))


def test_cutoffs_mismatch_is_an_error() -> None:
    """Fewer windows than the task defines no longer covers the published cutoffs."""
    task = FakeTask()
    frame = task_series_frame(task)
    config = task_backtest_config(task, n_windows=2)
    with pytest.raises(AssertionError, match="Cutoffs do not match"):
        assert_cutoffs_match(task, frame, config)


def test_cutoffs_accept_timestamps_instead_of_offsets() -> None:
    """Some tasks state cutoffs as dates; both forms must resolve to the same windows."""
    task = FakeTask()
    frame = task_series_frame(task)
    end = frame[TIMESTAMP_COL].max()
    task.cutoffs = [end - pd.Timedelta(hours=hours) for hours in (72, 48, 24)]
    assert_cutoffs_match(task, frame, task_backtest_config(task))


@pytest.mark.slow
def test_real_epf_de_task_matches_its_published_cutoffs() -> None:
    from c2p.fev_tasks import load_fev_task

    task = load_fev_task(EPF_DE)
    frame = task_series_frame(task)
    config = task_backtest_config(task)
    assert config.prediction_length == 24
    assert config.n_windows == 20
    assert len(config.known_covariates) == 2
    assert_cutoffs_match(task, frame, config)
