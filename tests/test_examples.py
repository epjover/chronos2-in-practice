"""Tests for ``examples/forecast_your_series.py`` (the model is replaced by a seasonal naive)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from c2p.baselines import SeasonalNaive
from c2p.schema import ID_COL, TIMESTAMP_COL

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "forecast_your_series.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("forecast_your_series", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fys = _load_script()


def _write_csv(path: Path, *, n_days: int = 30, with_known: bool = False) -> Path:
    """Two 'stores' of daily sales with a weekly pattern, foreign column names, a gap and a 0."""
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(0)
    for store in ("north", "south"):
        base = 100.0 if store == "north" else 60.0
        for i, date in enumerate(dates):
            sales = base + 20 * (date.day_of_week >= 5) + rng.normal(0, 2)
            rows.append({"store": store, "date": date, "sales": sales, "promo": int(i % 7 == 0)})
    frame = pd.DataFrame(rows)
    frame = frame.drop(index=5)  # one missing day
    frame.loc[frame.index[-1], "sales"] = 0.0  # a sentinel
    if with_known:  # append the horizon rows with the target left empty
        future = pd.DataFrame(
            {
                "store": "north",
                "date": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=7, freq="D"),
                "sales": np.nan,
                "promo": 1,
            }
        )
        frame = pd.concat([frame, future], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


class _FakeChronos2:
    """Stands in for ``Chronos2Forecaster``: a seasonal naive that answers to the same calls."""

    def __init__(self, pipeline: object, context_length: int | None = None) -> None:
        self.inner = SeasonalNaive(season_length=7)
        self.name = "chronos-2"
        self.context_length = context_length

    def predict(self, *args: object, **kwargs: object) -> pd.DataFrame:
        return self.inner.predict(*args, **kwargs)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fys, "load_chronos2", lambda *a, **k: object())
    monkeypatch.setattr(fys, "Chronos2Forecaster", _FakeChronos2)


def test_load_frame_maps_foreign_columns_onto_the_schema(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "sales.csv")
    frame, report = fys.load_frame(
        csv,
        id_column="store",
        timestamp_column="date",
        target="sales",
        known=["promo"],
        freq="D",
        source_tz=None,
        sentinels=[0.0],
    )
    assert list(frame.columns[:2]) == [ID_COL, TIMESTAMP_COL]
    assert set(frame[ID_COL]) == {"north", "south"}
    assert frame.groupby(ID_COL).size().eq(30).all()  # the gap was filled with a row
    assert frame["sales"].isna().sum() == 2  # the gap and the sentinel
    assert "promo" in frame.columns
    assert "sentinel" in str(report)


def test_load_frame_single_series_uses_the_file_name(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "meter_7.csv")
    frame = pd.read_csv(csv).query("store == 'north'").drop(columns="store")
    frame.to_csv(csv, index=False)
    prepared, _ = fys.load_frame(
        csv,
        id_column=None,
        timestamp_column="date",
        target="sales",
        known=[],
        freq="D",
        source_tz=None,
        sentinels=[],
    )
    assert prepared[ID_COL].unique().tolist() == ["meter_7"]


def test_load_frame_rejects_missing_columns(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "sales.csv")
    with pytest.raises(SystemExit, match=r"'revenue' is not in sales\.csv"):
        fys.load_frame(
            csv,
            id_column="store",
            timestamp_column="date",
            target="revenue",
            known=[],
            freq="D",
            source_tz=None,
            sentinels=[],
        )


def test_split_future_separates_rows_after_the_last_target(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "sales.csv", with_known=True)
    frame, _ = fys.load_frame(
        csv,
        id_column="store",
        timestamp_column="date",
        target="sales",
        known=["promo"],
        freq="D",
        source_tz=None,
        sentinels=[0.0],
    )
    # the sentinel turned the last south value into NaN, so south has one "future" row only
    with pytest.raises(SystemExit, match=r"south: 1"):
        fys.split_future(frame, target="sales", known=["promo"], horizon=7)
    # with a shorter horizon every series has enough rows and the future is cut to the horizon
    history, future = fys.split_future(frame, target="sales", known=["promo"], horizon=1)
    assert future is not None
    assert list(future.columns) == [ID_COL, TIMESTAMP_COL, "promo"]
    assert future.groupby(ID_COL).size().to_dict() == {"north": 1, "south": 1}
    last = history.groupby(ID_COL)[TIMESTAMP_COL].max()
    assert last["north"] == pd.Timestamp("2024-01-30")
    assert last["south"] == pd.Timestamp("2024-01-29")


def test_split_future_without_known_returns_frame_unchanged() -> None:
    frame = pd.DataFrame({ID_COL: ["a"], TIMESTAMP_COL: [pd.Timestamp("2024-01-01")], "y": [1.0]})
    history, future = fys.split_future(frame, target="y", known=[], horizon=3)
    assert history is frame and future is None


def test_main_writes_forecast_figure_and_backtest(
    tmp_path: Path, fake_model: None, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = _write_csv(tmp_path / "sales.csv", n_days=60)
    out = tmp_path / "out"
    code = fys.main(
        [
            str(csv),
            "--id-column",
            "store",
            "--timestamp-column",
            "date",
            "--target",
            "sales",
            "--freq",
            "D",
            "--horizon",
            "7",
            "--sentinels",
            "0",
            "--backtest",
            "3",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    forecast = pd.read_csv(out / "sales_forecast.csv")
    assert len(forecast) == 2 * 7
    assert {"0.1", "0.5", "0.9"} <= set(forecast.columns)
    assert (out / "sales_fan.png").exists()
    metrics = pd.read_csv(out / "sales_backtest.csv")
    assert set(metrics["model"]) == {"chronos-2", "seasonal_naive"}
    assert metrics.groupby("model").size().eq(3).all()  # one row per window
    text = capsys.readouterr().out
    assert "DataReport" in text
    assert "WQL skill" in text
    assert "wins" in text


def test_main_warns_when_known_has_no_future_rows(
    tmp_path: Path, fake_model: None, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = _write_csv(tmp_path / "sales.csv")
    code = fys.main(
        [
            str(csv),
            "--id-column",
            "store",
            "--timestamp-column",
            "date",
            "--target",
            "sales",
            "--known",
            "promo",
            "--freq",
            "D",
            "--horizon",
            "7",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    assert "past-only" in capsys.readouterr().err
