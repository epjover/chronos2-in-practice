"""Tests for ``c2p.data``: synthetic frames only, except the one ``slow`` test on real data."""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import kagglehub
import numpy as np
import pandas as pd
import pytest

from c2p import data
from c2p.data import (
    CALENDAR_COLUMNS,
    ENERGY_COLUMNS,
    DataReport,
    add_calendar_features,
    check_future_frame,
    drop_rows,
    inject_spike,
    make_future_frame,
    mask_values,
    validate_and_prepare,
)
from c2p.schema import ID_COL, TARGET_COL, TIMESTAMP_COL

REPO_ROOT = Path(__file__).resolve().parents[1]


def hourly_frame(
    n: int = 48, *, item: str | int = "A", start: str = "2018-01-01", **columns: object
) -> pd.DataFrame:
    """Hourly series frame with target 0, 1, 2, ... and optional extra columns."""
    frame = pd.DataFrame(
        {
            ID_COL: item,
            TIMESTAMP_COL: pd.date_range(start, periods=n, freq="h"),
            TARGET_COL: np.arange(n, dtype="float64"),
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


def prepare(df: pd.DataFrame, **kwargs: object) -> tuple[pd.DataFrame, DataReport]:
    kwargs.setdefault("freq", "h")
    return validate_and_prepare(df, **kwargs)


# --- step 1: columns, copy -----------------------------------------------------------------


def test_missing_column_is_reported() -> None:
    with pytest.raises(ValueError, match="missing column"):
        prepare(hourly_frame(), target="load")


def test_target_must_be_numeric() -> None:
    df = hourly_frame()
    df[TARGET_COL] = df[TARGET_COL].astype(str)
    with pytest.raises(TypeError, match="must be numeric"):
        prepare(df)


def test_text_columns_must_be_declared_categorical() -> None:
    df = hourly_frame(4, weather=["sun", "rain", "sun", "fog"])
    with pytest.raises(TypeError, match="categorical="):
        prepare(df)
    prepared, _ = prepare(df, categorical=["weather"])
    assert isinstance(prepared["weather"].dtype, pd.CategoricalDtype)


def test_missing_ids_or_timestamps_are_rejected() -> None:
    df = hourly_frame(4)
    df.loc[1, TIMESTAMP_COL] = pd.NaT
    with pytest.raises(ValueError, match="missing 'timestamp'"):
        prepare(df)


def test_blank_or_unparseable_timestamp_strings_are_rejected() -> None:
    df = hourly_frame(3)
    df[TIMESTAMP_COL] = ["2018-01-01 00:00", " ", "2018-01-01 02:00"]
    with pytest.raises(ValueError, match="missing 'timestamp'"):
        prepare(df)
    df[TIMESTAMP_COL] = ["2018-01-01 00:00", "NaT", "2018-01-01 02:00"]
    with pytest.raises(ValueError, match="not dates"):
        prepare(df)


def test_empty_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        prepare(hourly_frame(0))


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates must be"):
        prepare(hourly_frame(), duplicates="median")
    with pytest.raises(ValueError, match="max_length"):
        prepare(hourly_frame(), max_length=0)
    with pytest.raises(ValueError, match="used twice"):
        prepare(hourly_frame(), categorical=[TARGET_COL])
    with pytest.raises(TypeError, match="sentinels must be a list"):
        prepare(hourly_frame(), sentinels=0)
    with pytest.raises(TypeError, match="sentinels must be a list"):
        prepare(hourly_frame(), sentinels=["NA"])


def test_a_single_categorical_name_is_one_column() -> None:
    df = hourly_frame(4, weather_main=["sun", "rain", "sun", "fog"])
    prepared, _ = prepare(df, categorical="weather_main")
    assert isinstance(prepared["weather_main"].dtype, pd.CategoricalDtype)


def test_input_is_never_mutated() -> None:
    df = hourly_frame(24, extra=np.arange(24))
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.strftime("%Y-%m-%d %H:%M:%S+01:00")
    df.loc[3, TARGET_COL] = -999.0
    df = pd.concat([df.drop(index=[5, 6]), df.iloc[[10]]]).iloc[::-1]
    original = df.copy(deep=True)

    prepare(df, sentinels=[-999], max_length=12, categorical=())

    pd.testing.assert_frame_equal(df, original)


# --- step 2: timestamps -> naive UTC ---------------------------------------------------------


@pytest.mark.parametrize("hour_only", [False, True], ids=["+hh:mm", "+hh"])
@pytest.mark.parametrize(
    "first_utc", ["2018-03-24 22:00", "2018-10-27 23:00"], ids=["spring", "autumn"]
)
def test_offset_strings_across_dst_become_regular_naive_utc(
    first_utc: str, hour_only: bool
) -> None:
    utc = pd.date_range(first_utc, periods=5, freq="h")
    local = utc.tz_localize("UTC").tz_convert("Europe/Madrid")
    strings = [ts.isoformat(sep=" ") for ts in local]  # e.g. "2018-10-28 02:00:00+01:00"
    if hour_only:  # PostgreSQL prints whole-hour offsets as "+01"
        strings = [text.removesuffix(":00") for text in strings]
    df = pd.DataFrame({ID_COL: "ES", TIMESTAMP_COL: strings, TARGET_COL: np.arange(5.0)})

    prepared, report = prepare(df)

    assert prepared[TIMESTAMP_COL].dt.tz is None
    assert prepared[TIMESTAMP_COL].tolist() == utc.tolist()
    assert prepared[TARGET_COL].tolist() == list(range(5))
    assert (report.n_duplicates, report.n_missing_timestamps) == (0, 0)
    assert report.notes == []


def test_date_only_strings_are_naive_not_offsets() -> None:
    df = pd.DataFrame({ID_COL: "A", TIMESTAMP_COL: ["12-30-2017", "12-31-2017"], TARGET_COL: 1.0})
    prepared, report = prepare(df, freq="D")
    assert prepared[TIMESTAMP_COL].tolist() == [
        pd.Timestamp("2017-12-30"),
        pd.Timestamp("2017-12-31"),
    ]
    assert any("assumed to be UTC" in note for note in report.notes)  # "-2017" is no offset


def test_tz_aware_timestamps_are_converted_to_utc() -> None:
    df = hourly_frame(3)
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.tz_localize("Europe/Madrid")
    prepared, report = prepare(df, source_tz="Europe/Madrid")
    assert prepared[TIMESTAMP_COL].iloc[0] == pd.Timestamp("2017-12-31 23:00")
    assert any("ignored" in note for note in report.notes)


@pytest.mark.parametrize(
    ("day", "n_local_hours", "first_utc"),
    [("2018-03-25", 23, "2018-03-24 23:00"), ("2018-10-28", 25, "2018-10-27 22:00")],
)
def test_naive_local_times_with_source_tz_handle_short_and_long_days(
    day: str, n_local_hours: int, first_utc: str
) -> None:
    utc = pd.date_range(first_utc, periods=n_local_hours, freq="h")
    local = utc.tz_localize("UTC").tz_convert("Europe/Madrid").tz_localize(None)
    assert (local.normalize() == pd.Timestamp(day)).all()  # one local calendar day
    df = pd.DataFrame(
        {ID_COL: "ES", TIMESTAMP_COL: local, TARGET_COL: np.arange(n_local_hours, dtype=float)}
    )

    prepared, report = prepare(df, source_tz="Europe/Madrid")

    assert prepared[TIMESTAMP_COL].tolist() == utc.tolist()
    assert prepared[TARGET_COL].tolist() == list(range(n_local_hours))
    assert (report.n_duplicates, report.n_missing_timestamps) == (0, 0)


@pytest.mark.parametrize("freq", ["15min", "30min"])
def test_sub_hourly_local_times_across_the_autumn_dst_change(freq: str) -> None:
    utc = pd.date_range("2018-10-27 22:00", "2018-10-28 02:45", freq=freq)
    local = utc.tz_localize("UTC").tz_convert("Europe/Madrid").tz_localize(None)
    assert local.duplicated().sum() == (4 if freq == "15min" else 2)  # 02:00-02:59 twice
    df = pd.DataFrame(
        {ID_COL: "ES", TIMESTAMP_COL: local, TARGET_COL: np.arange(len(utc), dtype=float)}
    )

    prepared, report = prepare(df, freq=freq, source_tz="Europe/Madrid")

    assert prepared[TIMESTAMP_COL].tolist() == utc.tolist()
    assert prepared[TARGET_COL].tolist() == list(range(len(utc)))
    assert (report.n_duplicates, report.n_missing_timestamps) == (0, 0)


def test_repeated_local_times_out_of_time_order_raise() -> None:
    utc = pd.date_range("2018-10-27 23:00", periods=5, freq="h")
    local = utc.tz_localize("UTC").tz_convert("Europe/Madrid").tz_localize(None)
    df = pd.DataFrame({ID_COL: "ES", TIMESTAMP_COL: local, TARGET_COL: np.arange(5.0)})
    with pytest.raises(ValueError, match="not in time order"):
        prepare(df.iloc[::-1], source_tz="Europe/Madrid")
    # Without repeated local times the row order does not matter.
    winter = hourly_frame(5).iloc[::-1]
    prepared, _ = prepare(winter, source_tz="Europe/Madrid")
    assert prepared[TIMESTAMP_COL].iloc[0] == pd.Timestamp("2017-12-31 23:00")


def test_unresolvable_ambiguous_local_time_raises() -> None:
    local = pd.to_datetime(["2018-10-28 01:00", "2018-10-28 02:00", "2018-10-28 03:00"])
    df = pd.DataFrame({ID_COL: "ES", TIMESTAMP_COL: local, TARGET_COL: [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="DST"):
        prepare(df, source_tz="Europe/Madrid")


def test_nonexistent_local_time_is_shifted_forward_and_noted() -> None:
    local = pd.to_datetime(["2018-03-25 01:00", "2018-03-25 02:00", "2018-03-25 03:00"])
    df = pd.DataFrame({ID_COL: "ES", TIMESTAMP_COL: local, TARGET_COL: [1.0, 2.0, 4.0]})
    prepared, report = prepare(df, source_tz="Europe/Madrid")
    # 02:00 does not exist in Madrid that day: it becomes 03:00 and collides with 03:00.
    assert report.n_duplicates == 1
    assert prepared[TARGET_COL].tolist() == [1.0, 3.0]
    assert any("shifted forward" in note for note in report.notes)


def test_naive_timestamps_without_source_tz_are_assumed_utc() -> None:
    prepared, report = prepare(hourly_frame(3))
    assert prepared[TIMESTAMP_COL].iloc[0] == pd.Timestamp("2018-01-01 00:00")
    assert any("assumed to be UTC" in note for note in report.notes)


def test_mixed_offset_and_naive_strings_raise() -> None:
    df = hourly_frame(2)
    df[TIMESTAMP_COL] = ["2018-01-01 00:00:00+01:00", "2018-01-01 01:00:00"]
    with pytest.raises(ValueError, match="lack a UTC offset"):
        prepare(df)


# --- step 3: sentinels -------------------------------------------------------------------------


def test_sentinels_become_nan_and_are_counted() -> None:
    df = hourly_frame(6, price=[10.0, -999.0, 12.0, 0.0, 14.0, 15.0], flag=[False] * 6)
    df.loc[2, TARGET_COL] = -999.0
    prepared, report = prepare(df, sentinels=[-999, 0])

    assert report.n_sentinels == 4  # target 0.0 at row 0, target -999, price -999 and 0
    assert report.n_sentinels_by_column == {TARGET_COL: 2, "price": 2}
    assert prepared[TARGET_COL].isna().tolist() == [True, False, True, False, False, False]
    assert prepared["price"].isna().sum() == 2
    assert prepared["flag"].tolist() == [0.0] * 6  # booleans are never sentinels


def test_sentinel_columns_limit_the_search() -> None:
    df = hourly_frame(4, solar=[0.0, 0.0, 5.0, 0.0])  # zeros are real readings at night
    _, everywhere = prepare(df, sentinels=[0])
    assert everywhere.n_sentinels_by_column == {TARGET_COL: 1, "solar": 3}
    assert f"4 ({TARGET_COL}=1, solar=3)" in str(everywhere)

    prepared, report = prepare(df, sentinels=[0], sentinel_columns=[TARGET_COL])

    assert report.n_sentinels == 1
    assert prepared["solar"].tolist() == [0.0, 0.0, 5.0, 0.0]
    assert f"n_sentinels[{TARGET_COL}]" in set(report.to_frame()["metric"])
    with pytest.raises(ValueError, match="sentinel_columns"):
        prepare(df, sentinels=[0], sentinel_columns=[TIMESTAMP_COL])


# --- step 4: duplicates ------------------------------------------------------------------------


def frame_with_duplicates() -> pd.DataFrame:
    df = hourly_frame(3, kind=["a", "b", "c"])
    repeat = df.iloc[[1]].assign(**{TARGET_COL: 3.0, "kind": "z"})
    return pd.concat([df, repeat], ignore_index=True)


@pytest.mark.parametrize(
    ("policy", "value", "kind"), [("mean", 2.0, "b"), ("first", 1.0, "b"), ("last", 3.0, "z")]
)
def test_duplicate_policies(policy: str, value: float, kind: str) -> None:
    prepared, report = prepare(frame_with_duplicates(), duplicates=policy, categorical=["kind"])
    assert report.n_duplicates == 1
    assert len(prepared) == 3
    assert prepared.loc[1, TARGET_COL] == value
    assert prepared.loc[1, "kind"] == kind


def test_duplicate_policy_raise_lists_examples() -> None:
    with pytest.raises(ValueError, match="duplicated"):
        prepare(frame_with_duplicates(), duplicates="raise", categorical=["kind"])


# --- step 5: off-grid timestamps ---------------------------------------------------------------


def test_off_grid_timestamp_is_rejected() -> None:
    df = hourly_frame(4)
    df.loc[2, TIMESTAMP_COL] = pd.Timestamp("2018-01-01 01:30")
    with pytest.raises(ValueError, match="not on the regular 'h' grid"):
        prepare(df)


def test_consistent_half_hour_phase_is_accepted() -> None:
    # Local hours in India (UTC+05:30) fall at :30 past every hour in UTC: still regular.
    prepared, report = prepare(hourly_frame(6), source_tz="Asia/Kolkata")
    assert prepared[TIMESTAMP_COL].iloc[0] == pd.Timestamp("2017-12-31 18:30")
    assert report.n_missing_timestamps == 0
    _, report_shifted = prepare(hourly_frame(4, start="2018-01-01 10:30"))
    assert report_shifted.n_rows_out == 4


def test_daily_dates_with_source_tz_across_dst_are_rejected_with_a_hint() -> None:
    days = pd.date_range("2018-03-22", periods=7, freq="D")
    df = pd.DataFrame({ID_COL: "A", TIMESTAMP_COL: days, TARGET_COL: 1.0})
    with pytest.raises(ValueError, match="should not get source_tz"):
        prepare(df, freq="D", source_tz="Europe/Madrid")
    prepared, _ = prepare(df, freq="D")
    assert prepared[TIMESTAMP_COL].tolist() == days.tolist()


def test_daily_grid_accepts_consistent_times() -> None:
    df = pd.DataFrame(
        {
            ID_COL: "A",
            TIMESTAMP_COL: pd.date_range("2018-01-01 23:00", periods=5, freq="D"),
            TARGET_COL: 1.0,
        }
    ).drop(index=2)
    prepared, report = prepare(df, freq="D")
    assert report.n_missing_timestamps == 1
    assert len(prepared) == 5


# --- step 6: gaps --------------------------------------------------------------------------------


def test_missing_rows_are_reinserted_as_nan_per_series() -> None:
    a = hourly_frame(10, item="A").drop(index=[3, 4, 7])
    b = hourly_frame(5, item="B", start="2018-01-01 02:00").drop(index=[1])
    prepared, report = prepare(pd.concat([b, a]))

    assert report.n_missing_timestamps == 4
    assert prepared.groupby(ID_COL).size().to_dict() == {"A": 10, "B": 5}
    a_out = prepared[prepared[ID_COL] == "A"]
    assert a_out[TARGET_COL].isna().tolist() == [i in {3, 4, 7} for i in range(10)]
    assert a_out[TIMESTAMP_COL].diff().dropna().eq(pd.Timedelta("1h")).all()


# --- steps 7 and 8: trimming and short series --------------------------------------------------


def test_max_length_keeps_most_recent_rows() -> None:
    df = pd.concat([hourly_frame(10, item="A"), hourly_frame(4, item="B")])
    prepared, report = prepare(df, max_length=6)
    assert report.n_trimmed == 4
    a_out = prepared[prepared[ID_COL] == "A"]
    assert a_out[TARGET_COL].tolist() == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_short_series_are_flagged_not_dropped() -> None:
    df = pd.concat([hourly_frame(10, item="A"), hourly_frame(4, item="B")])
    prepared, report = prepare(df, min_length=5)
    assert report.short_series == ["B"]
    assert set(prepared[ID_COL]) == {"A", "B"}
    _, report_default = prepare(df)
    assert report_default.short_series == []


# --- step 9: dtypes, order, report -------------------------------------------------------------


def test_dtypes_sorting_and_index() -> None:
    df = hourly_frame(6, count=np.arange(6), kind=list("aabbcc"))
    df[ID_COL] = 7
    shuffled = df.sample(frac=1.0, random_state=0)
    prepared, _ = prepare(shuffled, categorical=["kind"])

    assert prepared[TARGET_COL].dtype == "float64"
    assert prepared["count"].dtype == "float64"
    assert isinstance(prepared["kind"].dtype, pd.CategoricalDtype)
    assert prepared[ID_COL].iloc[0] == "7"
    assert prepared[TIMESTAMP_COL].is_monotonic_increasing
    assert prepared.index.equals(pd.RangeIndex(6))


def test_report_frame_and_text() -> None:
    df = hourly_frame(6).drop(index=[2])
    df.loc[0, TARGET_COL] = np.nan
    _, report = prepare(df, min_length=10)

    table = report.to_frame()
    assert table.columns.tolist() == ["metric", "value"]
    values = dict(zip(table["metric"], table["value"], strict=True))
    assert values["n_rows_in"] == 5
    assert values["n_rows_out"] == 6
    assert values["n_missing_timestamps"] == 1
    assert values[f"n_missing_values[{TARGET_COL}]"] == 2
    assert values["short_series"] == "A"
    text = str(report)
    assert "missing timestamps inserted: 1" in text
    assert f"{TARGET_COL}=2" in text


def test_multiple_targets() -> None:
    df = hourly_frame(4, other=["1", "2", "3", "4"])
    with pytest.raises(TypeError, match="must be numeric"):
        prepare(df, target=[TARGET_COL, "other"])
    df["other"] = [1.0, -1.0, 3.0, 4.0]
    _, report = prepare(df, target=[TARGET_COL, "other"], sentinels=[-1])
    assert report.n_missing_values == {TARGET_COL: 0, "other": 1}


# --- future frames -------------------------------------------------------------------------------


def two_series_context() -> pd.DataFrame:
    """Two series with a known covariate ``load_forecast``."""
    return pd.concat(
        [
            hourly_frame(10, item="A", load_forecast=2.0),
            hourly_frame(5, item="B", load_forecast=2.0),
        ],
        ignore_index=True,
    )


def test_make_future_frame_continues_each_series() -> None:
    future = make_future_frame(two_series_context(), prediction_length=3, freq="h")
    assert future.columns.tolist() == [ID_COL, TIMESTAMP_COL]
    by_item = future.groupby(ID_COL)[TIMESTAMP_COL].apply(list).to_dict()
    assert by_item["A"] == pd.date_range("2018-01-01 10:00", periods=3, freq="h").tolist()
    assert by_item["B"] == pd.date_range("2018-01-01 05:00", periods=3, freq="h").tolist()


def valid_future() -> pd.DataFrame:
    future = make_future_frame(two_series_context(), prediction_length=3, freq="h")
    return future.assign(load_forecast=1.0)


def test_check_future_frame_accepts_valid_frame() -> None:
    check_future_frame(two_series_context(), valid_future(), prediction_length=3, freq="h")


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda f: f[f[ID_COL] == "A"], "Series differ"),
        (lambda f: f.drop(index=0), "exactly prediction_length"),
        (lambda f: f.assign(**{TIMESTAMP_COL: f[TIMESTAMP_COL] + pd.Timedelta("1h")}), "steps"),
        (lambda f: f.assign(**{TARGET_COL: 0.0}), "target column"),
        (lambda f: f.assign(temperature=15.0), "context frame lacks"),
        (lambda f: f.assign(load_forecast=[np.nan, *[1.0] * 5]), "NaN"),
    ],
    ids=["ids", "row-count", "timestamps", "target-leak", "column-not-in-context", "nan"],
)
def test_check_future_frame_failure_modes(
    corrupt: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        check_future_frame(
            two_series_context(), corrupt(valid_future()), prediction_length=3, freq="h"
        )


def test_check_future_frame_compares_ids_with_their_type() -> None:
    context, _ = prepare(hourly_frame(10, item=1))  # validate_and_prepare makes ids str: "1"
    future = make_future_frame(context, prediction_length=3, freq="h")
    check_future_frame(context, future, prediction_length=3, freq="h")
    with pytest.raises(ValueError, match="differ only in type"):
        check_future_frame(context, future.assign(**{ID_COL: 1}), prediction_length=3, freq="h")


# --- calendar features ---------------------------------------------------------------------------


def test_calendar_features_use_local_time_and_spanish_holidays() -> None:
    utc = pd.to_datetime(
        [
            "2017-12-31 23:00",  # 2018-01-01 00:00 local: New Year
            "2018-03-25 00:00",  # 01:00 CET
            "2018-03-25 01:00",  # 03:00 CEST (02:00 skipped)
            "2018-10-11 22:00",  # 2018-10-12 00:00 CEST: Fiesta Nacional
            "2018-10-28 00:00",  # 02:00 CEST
            "2018-10-28 01:00",  # 02:00 CET (repeated hour)
            "2018-10-16 12:00",  # ordinary Tuesday
        ]
    )
    df = pd.DataFrame({ID_COL: "ES", TIMESTAMP_COL: utc})
    out = add_calendar_features(df)

    assert out["hour"].tolist() == [0.0, 1.0, 3.0, 0.0, 2.0, 2.0, 14.0]
    assert out["is_holiday"].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert out["is_weekend"].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    assert out["day_of_week"].tolist() == [0.0, 6.0, 6.0, 4.0, 6.0, 6.0, 1.0]
    assert out["month"].iloc[0] == 1.0
    assert all(out[column].dtype == "float64" for column in CALENDAR_COLUMNS)
    assert out[TIMESTAMP_COL].equals(df[TIMESTAMP_COL])  # stays naive UTC
    assert df.columns.tolist() == [ID_COL, TIMESTAMP_COL]  # input untouched


def test_calendar_features_reject_aware_timestamps_and_existing_columns() -> None:
    df = hourly_frame(3)
    with pytest.raises(ValueError, match="already exist"):
        add_calendar_features(df.assign(hour=1.0))
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-naive"):
        add_calendar_features(df)


def test_calendar_features_make_a_valid_future_frame() -> None:
    context = add_calendar_features(two_series_context())  # past values of the covariates
    future = add_calendar_features(make_future_frame(context, prediction_length=4, freq="h"))
    check_future_frame(context, future, prediction_length=4, freq="h")


# --- corruption helpers ------------------------------------------------------------------------


def test_drop_rows_and_mask_values_hit_the_same_positions() -> None:
    df = pd.concat([hourly_frame(60, item="A"), hourly_frame(40, item="B")], ignore_index=True)
    original = df.copy()
    gaps = {"n_gaps": 3, "gap_length": 4, "protect_last": 10}

    dropped = drop_rows(df, rng=np.random.default_rng(7), **gaps)
    masked = mask_values(df, rng=np.random.default_rng(7), **gaps)

    removed = df.index.difference(dropped.index)
    assert masked.index[masked[TARGET_COL].isna()].equals(removed)
    assert len(removed) == 2 * 3 * 4
    pd.testing.assert_frame_equal(df, original)
    for _, series in masked.groupby(ID_COL):
        is_gap = series[TARGET_COL].isna().to_numpy()
        assert not is_gap[0]
        assert not is_gap[-10:].any()
        starts = np.flatnonzero(is_gap & ~np.r_[False, is_gap[:-1]])
        assert len(starts) == 3  # three separate runs, each of four rows


def test_gaps_follow_time_order_whatever_the_row_order() -> None:
    df = hourly_frame(40)  # index labels follow time order
    gaps = {"n_gaps": 2, "gap_length": 5, "protect_last": 24}
    removed_sorted = df.index.difference(drop_rows(df, rng=np.random.default_rng(3), **gaps).index)
    assert removed_sorted.max() < 40 - 24  # the protected tail is untouched

    for unsorted in (df.iloc[::-1], df.sample(frac=1.0, random_state=0)):
        dropped = drop_rows(unsorted, rng=np.random.default_rng(3), **gaps)
        masked = mask_values(unsorted, rng=np.random.default_rng(3), **gaps)
        assert df.index.difference(dropped.index).equals(removed_sorted)
        assert masked.index[masked[TARGET_COL].isna()].sort_values().equals(removed_sorted)
    with pytest.raises(TypeError, match="datetimes"):
        drop_rows(df.astype({TIMESTAMP_COL: str}), rng=np.random.default_rng(3), **gaps)


def test_mask_values_with_sentinel_fill_and_too_many_gaps() -> None:
    df = hourly_frame(20)
    as_zero = mask_values(df, n_gaps=2, gap_length=3, rng=np.random.default_rng(0), fill=0.0)
    as_nan = mask_values(df, n_gaps=2, gap_length=3, rng=np.random.default_rng(0))

    changed = as_zero[TARGET_COL] != df[TARGET_COL]
    assert changed.sum() == 6
    assert (as_zero.loc[changed, TARGET_COL] == 0.0).all()
    assert changed.equals(as_nan[TARGET_COL].isna())
    with pytest.raises(ValueError, match="cannot hold"):
        drop_rows(hourly_frame(10), n_gaps=3, gap_length=3, rng=np.random.default_rng(0))


def test_drop_rows_then_prepare_restores_the_grid() -> None:
    df = hourly_frame(50)
    dropped = drop_rows(df, n_gaps=2, gap_length=5, rng=np.random.default_rng(1))
    prepared, report = prepare(dropped)
    assert report.n_missing_timestamps == 10
    assert len(prepared) == 50


def test_inject_spike_by_position_and_timestamp() -> None:
    df = hourly_frame(5)
    by_position = inject_spike(df, at=-1, factor=3.0)
    by_time = inject_spike(df, at=pd.Timestamp("2018-01-01 04:00"), factor=3.0)

    assert by_position[TARGET_COL].tolist() == [0.0, 1.0, 2.0, 3.0, 12.0]
    pd.testing.assert_frame_equal(by_position, by_time)
    assert df[TARGET_COL].iloc[-1] == 4.0
    with pytest.raises(ValueError, match="NaN"):
        inject_spike(df.assign(**{TARGET_COL: np.nan}), at=1, factor=2.0)
    with pytest.raises(ValueError, match="matches 0 rows"):
        inject_spike(df, at=pd.Timestamp("2019-01-01"), factor=2.0)
    with pytest.raises(IndexError):
        inject_spike(df, at=5, factor=2.0)


# --- download and Spanish frames (no network: kagglehub is faked) --------------------------------

ENERGY_CSV = (
    "time,total load actual,price actual,generation fossil brown coal/lignite,"
    "forecast wind offshore eday ahead\n"
    "2018-03-25 01:00:00+01:00,100.0,50.0,1.0,\n"
    "2018-03-25 03:00:00+02:00,,51.0,2.0,\n"
)
WEATHER_CSV = (
    "dt_iso,city_name,temp,temp_min,temp_max,pressure,humidity,wind_speed,wind_deg,"
    "rain_1h,rain_3h,snow_3h,clouds_all,weather_id,weather_main,weather_description,weather_icon\n"
    "2018-01-01 00:00:00+01:00, Barcelona,273.15,272.15,274.15,1010,80,2,90,0,0,0,10,800,clear,sky is clear,01n\n"  # noqa: E501
    "2018-01-01 00:00:00+01:00, Barcelona,283.15,282.15,284.15,1011,81,3,91,0,0,0,11,800,clear,sky is clear,01n\n"  # noqa: E501
    "2017-12-31 23:00:00+00:00,Madrid,280.15,279.15,281.15,1012,82,4,92,0,0,0,12,500,rain,light rain,10n\n"  # noqa: E501
)


@pytest.fixture
def fake_kaggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Serve synthetic CSVs from a fake kagglehub cache; return the force flag of each call."""
    cache = tmp_path / "kaggle_cache"
    cache.mkdir()
    contents = {data.ENERGY_FILE: ENERGY_CSV, data.WEATHER_FILE: WEATHER_CSV}
    for name, text in contents.items():
        (cache / name).write_text(text, encoding="utf-8")
    hashes = {name: hashlib.sha256((cache / name).read_bytes()).hexdigest() for name in contents}
    monkeypatch.setattr(data, "SPAIN_ENERGY_FILES", hashes)
    calls: list[bool] = []

    def fake_download(handle: str, path: str | None = None, *, force_download: bool = False) -> str:
        assert handle == data.SPAIN_ENERGY_HANDLE
        calls.append(force_download)
        return str(cache)

    monkeypatch.setattr(kagglehub, "dataset_download", fake_download)
    return calls


def test_download_copies_verifies_and_is_idempotent(tmp_path: Path, fake_kaggle) -> None:
    dest = tmp_path / "raw" / "spain_energy"
    assert data.download_spain_energy(dest) == dest
    assert sorted(p.name for p in dest.iterdir()) == sorted(data.SPAIN_ENERGY_FILES)
    data.download_spain_energy(dest)
    assert fake_kaggle == [False]  # second call found the files
    data.download_spain_energy(dest, force=True)
    assert fake_kaggle == [False, True]


def test_download_rejects_checksum_mismatch(tmp_path: Path, fake_kaggle) -> None:
    dest = data.download_spain_energy(tmp_path / "spain_energy")
    (dest / data.ENERGY_FILE).write_text("tampered", encoding="utf-8")
    expected = data.SPAIN_ENERGY_FILES[data.ENERGY_FILE]
    actual = hashlib.sha256(b"tampered").hexdigest()
    with pytest.raises(ValueError, match=f"expected SHA256 {expected}, got {actual}"):
        data.download_spain_energy(dest)


def test_spain_energy_frame_renames_and_drops_all_nan(tmp_path: Path, fake_kaggle) -> None:
    frame = data.spain_energy_frame(tmp_path)
    assert frame.columns.tolist() == [
        ID_COL,
        TIMESTAMP_COL,
        "load_actual",
        "price_actual",
        "gen_fossil_brown_coal_lignite",
    ]
    assert (frame[ID_COL] == "ES").all()
    assert frame[TIMESTAMP_COL].tolist() == [
        pd.Timestamp("2018-03-25 00:00"),
        pd.Timestamp("2018-03-25 01:00"),
    ]
    assert frame["load_actual"].isna().sum() == 1  # raw quirks are kept
    assert len(data.load_energy_raw(tmp_path)) == 2


def test_spain_weather_frame_cleans_names_units_and_keeps_duplicates(
    tmp_path: Path, fake_kaggle
) -> None:
    frame = data.spain_weather_frame(tmp_path)
    assert frame[ID_COL].tolist() == ["Barcelona", "Barcelona", "Madrid"]
    assert (frame[TIMESTAMP_COL] == pd.Timestamp("2017-12-31 23:00")).all()
    assert frame["temp_c"].tolist() == pytest.approx([0.0, 10.0, 7.0])
    assert {"temp", "weather_id", "weather_description", "weather_icon"}.isdisjoint(frame.columns)
    assert isinstance(frame["weather_main"].dtype, pd.CategoricalDtype)
    prepared, report = prepare(frame, target="temp_c", categorical=["weather_main"])
    assert report.n_duplicates == 1
    assert prepared["temp_c"].tolist() == pytest.approx([5.0, 7.0])


def test_energy_columns_mapping() -> None:
    assert ENERGY_COLUMNS["total load actual"] == "load_actual"
    assert ENERGY_COLUMNS["price day ahead"] == "price_day_ahead_tso"
    assert ENERGY_COLUMNS["generation hydro run-of-river and poundage"] == (
        "gen_hydro_run_of_river_and_poundage"
    )
    assert len(set(ENERGY_COLUMNS.values())) == len(ENERGY_COLUMNS)


def load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "download_cli", REPO_ROOT / "data" / "download.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_cli(tmp_path: Path, fake_kaggle, capsys: pytest.CaptureFixture[str]) -> None:
    assert load_cli().main(["--dest", str(tmp_path / "cli"), "--force"]) == 0
    assert fake_kaggle == [True]
    output = capsys.readouterr().out
    assert "ready in" in output
    assert data.ENERGY_FILE in output


# --- real data ---------------------------------------------------------------------------------


@pytest.mark.slow
def test_real_spanish_dataset() -> None:
    folder = data.download_spain_energy()  # verifies both SHA256 hashes
    assert folder == data.default_data_dir()
    assert data.load_energy_raw().shape == (35_064, 29)
    assert data.load_weather_raw().shape == (178_396, 17)

    energy, report = validate_and_prepare(data.spain_energy_frame(), freq="h", target="load_actual")
    assert len(energy) == 35_064
    assert (report.n_duplicates, report.n_missing_timestamps) == (0, 0)
    assert report.n_missing_values["load_actual"] == 36
    assert "wind_offshore_forecast_tso" not in energy.columns  # all-NaN column dropped

    weather, weather_report = validate_and_prepare(
        data.spain_weather_frame(), freq="h", target="temp_c", categorical=["weather_main"]
    )
    assert weather_report.n_duplicates == 3_076
    assert sorted(weather[ID_COL].unique()) == [
        "Barcelona",
        "Bilbao",
        "Madrid",
        "Seville",
        "Valencia",
    ]
    assert len(weather) == 5 * 35_064
