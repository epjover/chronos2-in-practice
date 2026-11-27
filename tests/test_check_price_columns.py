"""Offline tests of ``scripts/check_price_columns.py``: parsing OMIE files and the comparison."""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from c2p.schema import TIMESTAMP_COL

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_price_columns.py"
spec = importlib.util.spec_from_file_location("check_price_columns", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def omie_archive(days: dict[str, list[float]]) -> bytes:
    """A zip like OMIE's: one file per day, one line per local hour, Portugal then Spain."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as files:
        for day, prices in days.items():
            year, month, dom = day.split("-")
            lines = ["MARGINALPDBC;"]
            lines += [
                f"{year};{month};{dom};{hour};{price + 1:.2f};{price:.2f};"
                for hour, price in enumerate(prices, start=1)
            ]
            files.writestr(f"marginalpdbc_{year}{month}{dom}.1", "\r\n".join([*lines, "*"]))
    return buffer.getvalue()


def test_parse_omie_puts_local_hours_on_the_utc_grid() -> None:
    summer = [float(hour) for hour in range(24)]
    table = check.parse_omie(omie_archive({"2018-06-13": summer}))
    assert len(table) == 24
    # Hour 1 of 13 June in Madrid (UTC+2) is 22:00 UTC on 12 June.
    assert table[TIMESTAMP_COL].iloc[0] == pd.Timestamp("2018-06-12 22:00")
    assert table[check.OMIE].tolist() == summer  # the Spanish column, not the Portuguese one


@pytest.mark.parametrize(("day", "hours"), [("2018-03-25", 23), ("2018-10-28", 25)])
def test_parse_omie_keeps_clock_change_days_aligned(day: str, hours: int) -> None:
    table = check.parse_omie(omie_archive({day: [10.0] * hours}))
    stamps = table[TIMESTAMP_COL]
    assert len(table) == hours
    assert stamps.is_unique
    assert (stamps.diff().dropna() == pd.Timedelta(hours=1)).all()
    # The next local midnight is one hour after the last row.
    next_day = pd.Timestamp(day) + pd.Timedelta(days=1)
    next_midnight = next_day.tz_localize("Europe/Madrid").tz_convert("UTC")
    assert stamps.iloc[-1] + pd.Timedelta(hours=1) == next_midnight.tz_localize(None)


def test_parse_omie_rejects_an_archive_without_prices() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as files:
        files.writestr("readme.txt", "nothing here")
    with pytest.raises(ValueError, match="no hourly price lines"):
        check.parse_omie(buffer.getvalue())


def test_compare_measures_the_premium_and_the_share_of_equal_hours() -> None:
    stamps = pd.date_range("2018-06-12 22:00", periods=48, freq="h")  # two local days
    omie = pd.DataFrame({TIMESTAMP_COL: stamps, check.OMIE: np.linspace(40.0, 60.0, 48)})
    day_ahead = omie[check.OMIE].to_numpy().copy()
    day_ahead[24:] = 99.0  # the second day holds something else
    dataset = pd.DataFrame(
        {
            TIMESTAMP_COL: stamps,
            check.ACTUAL: omie[check.OMIE] + 6.0,
            check.DAY_AHEAD: day_ahead,
        }
    )
    result = check.compare(dataset, omie).set_index("measure")["value"]
    assert result["hours_compared"] == 48
    assert result["actual_vs_omie_correlation"] == pytest.approx(1.0)
    assert result["actual_minus_omie_mean"] == pytest.approx(6.0)
    assert result["actual_equal_to_omie_share"] == 0.0
    assert result["day_ahead_column_equal_to_omie_share"] == 0.5
    assert result["days_compared"] == 2
    assert result["days_day_ahead_column_equal_all_hours"] == 1
    assert result["days_day_ahead_column_equal_no_hour"] == 1


def test_compare_rejects_frames_without_common_hours() -> None:
    dataset = pd.DataFrame(
        {
            TIMESTAMP_COL: pd.date_range("2017-01-01", periods=3, freq="h"),
            check.ACTUAL: 1.0,
            check.DAY_AHEAD: 1.0,
        }
    )
    omie = pd.DataFrame(
        {TIMESTAMP_COL: pd.date_range("2018-01-01", periods=3, freq="h"), check.OMIE: 1.0}
    )
    with pytest.raises(ValueError, match="share no hours"):
        check.compare(dataset, omie)
