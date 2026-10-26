"""Tests for ``c2p.monitoring``: the prediction log, the delayed join and the alert rule.

The point of these tests is the production order of events. A forecast is logged before its
actual exists, so every test builds the log first and only then supplies actuals — including the
case where some never arrive.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from c2p.monitoring import (
    ACTUAL_COL,
    ALERT_COL,
    CUTOFF_COL,
    INPUT_HASH_COL,
    MODEL_VERSION_COL,
    append_log,
    context_hash,
    input_drift,
    join_actuals,
    log_predictions,
    log_summary,
    read_log,
    retraining_alerts,
    rolling_metrics,
    to_json_line,
)
from c2p.schema import ID_COL, PREDICTION_COL, TARGET_NAME_COL, TIMESTAMP_COL, quantile_col

HORIZON = 4
VERSION = "amazon/chronos-2@2.3.2"


def make_context(*, start: str = "2018-01-01", periods: int = 48, offset: float = 0.0):
    timestamps = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            ID_COL: "spain",
            TIMESTAMP_COL: timestamps,
            "load_actual": np.linspace(100.0, 200.0, periods) + offset,
        }
    )


def make_forecast(cutoff: pd.Timestamp, *, bias: float = 0.0, width: float = 20.0):
    timestamps = pd.date_range(cutoff + pd.Timedelta("1h"), periods=HORIZON, freq="h")
    median = np.full(HORIZON, 150.0) + bias
    return pd.DataFrame(
        {
            ID_COL: "spain",
            TIMESTAMP_COL: timestamps,
            TARGET_NAME_COL: "load_actual",
            PREDICTION_COL: median,
            quantile_col(0.1): median - width,
            quantile_col(0.5): median,
            quantile_col(0.9): median + width,
        }
    )


def make_log(n_cutoffs: int = 6, *, bias_from: int | None = None, bias: float = 60.0):
    """A log of ``n_cutoffs`` daily forecasts, optionally going wrong partway through."""
    rows = []
    for index in range(n_cutoffs):
        cutoff = pd.Timestamp("2018-01-02 22:00", tz="UTC") + pd.Timedelta(days=index)
        shift = bias if bias_from is not None and index >= bias_from else 0.0
        rows.append(
            log_predictions(
                make_forecast(cutoff, bias=shift),
                make_context(offset=index),
                model_version=VERSION,
                cutoff=cutoff,
            )
        )
    return pd.concat(rows, ignore_index=True)


def make_actuals(log: pd.DataFrame, *, value: float = 150.0):
    stamps = log[[ID_COL, TIMESTAMP_COL]].drop_duplicates()
    return stamps.assign(load_actual=value)


# --- the input hash -------------------------------------------------------------------------


def test_context_hash_is_stable_and_order_independent_of_column_order():
    context = make_context()
    reordered = context[[TIMESTAMP_COL, "load_actual", ID_COL]]
    assert context_hash(context) == context_hash(context.copy())
    assert context_hash(context) == context_hash(reordered)


def test_context_hash_changes_when_a_single_value_changes():
    context = make_context()
    edited = context.copy()
    edited.loc[0, "load_actual"] += 1e-6
    assert context_hash(edited) != context_hash(context)


def test_context_hash_distinguishes_a_gap_from_a_filled_gap():
    context = make_context()
    with_gap = context.copy()
    with_gap.loc[5, "load_actual"] = np.nan
    filled = with_gap.ffill()
    assert len({context_hash(context), context_hash(with_gap), context_hash(filled)}) == 3


def test_context_hash_rejects_an_empty_context():
    with pytest.raises(ValueError, match="empty context"):
        context_hash(make_context().head(0))


# --- logging --------------------------------------------------------------------------------


def test_log_predictions_has_no_actual_column():
    rows = log_predictions(
        make_forecast(pd.Timestamp("2018-01-02 22:00", tz="UTC")),
        make_context(),
        model_version=VERSION,
    )
    assert ACTUAL_COL not in rows.columns
    assert len(rows) == HORIZON
    assert set(rows[MODEL_VERSION_COL]) == {VERSION}
    assert len(set(rows[INPUT_HASH_COL])) == 1


def test_log_predictions_defaults_the_cutoff_to_the_last_context_timestamp():
    context = make_context()
    rows = log_predictions(
        make_forecast(context[TIMESTAMP_COL].max()), context, model_version=VERSION
    )
    assert rows[CUTOFF_COL].iloc[0] == context[TIMESTAMP_COL].max()


def test_log_predictions_carries_extra_fields_and_requires_a_version():
    rows = log_predictions(
        make_forecast(pd.Timestamp("2018-01-02 22:00", tz="UTC")),
        make_context(),
        model_version=VERSION,
        extra={"context_steps": 48},
    )
    assert set(rows["context_steps"]) == {48}
    with pytest.raises(ValueError, match="model_version"):
        log_predictions(
            make_forecast(pd.Timestamp("2018-01-02 22:00", tz="UTC")),
            make_context(),
            model_version="",
        )


def test_log_predictions_rejects_a_forecast_without_the_requested_quantiles():
    forecast = make_forecast(pd.Timestamp("2018-01-02 22:00", tz="UTC")).drop(
        columns=[quantile_col(0.9)]
    )
    with pytest.raises(ValueError, match="lacks the columns"):
        log_predictions(forecast, make_context(), model_version=VERSION)


def test_append_log_writes_one_header_and_read_log_parses_timestamps(tmp_path):
    path = tmp_path / "log.csv"
    first = make_log(2)
    append_log(first, path)
    append_log(make_log(2), path)
    text = path.read_text(encoding="utf-8")
    assert text.count(MODEL_VERSION_COL) == 1
    back = read_log(path)
    assert len(back) == 2 * len(first)
    assert pd.api.types.is_datetime64_any_dtype(back[CUTOFF_COL])
    assert pd.api.types.is_datetime64_any_dtype(back[TIMESTAMP_COL])


def test_read_log_fails_loudly_when_the_log_does_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="log a forecast first"):
        read_log(tmp_path / "absent.csv")


# --- the delayed join -----------------------------------------------------------------------


def test_join_actuals_leaves_unresolved_rows_as_nan():
    log = make_log(3)
    actuals = make_actuals(log).iloc[: 2 * HORIZON]
    joined = join_actuals(log, actuals, target="load_actual")
    assert len(joined) == len(log)
    assert joined[ACTUAL_COL].notna().sum() == 2 * HORIZON
    assert joined[ACTUAL_COL].isna().sum() == HORIZON


def test_join_actuals_rejects_actuals_without_the_target():
    log = make_log(2)
    with pytest.raises(ValueError, match="target column"):
        join_actuals(log, make_actuals(log), target="price_actual")


# --- rolling metrics ------------------------------------------------------------------------


def test_rolling_metrics_are_nan_before_the_window_fills():
    joined = join_actuals(make_log(6), make_actuals(make_log(6)), target="load_actual")
    rolling = rolling_metrics(joined, window=3)
    assert rolling["wape"].isna().sum() == 2
    assert rolling["wape"].notna().sum() == 4


def test_rolling_wape_is_zero_for_a_perfect_forecast_and_coverage_is_one():
    log = make_log(4)
    joined = join_actuals(log, make_actuals(log, value=150.0), target="load_actual")
    rolling = rolling_metrics(joined, window=2)
    assert rolling["wape"].dropna().eq(0.0).all()
    assert rolling["coverage"].dropna().eq(1.0).all()
    assert set(rolling["nominal"]) == {0.8}


def test_rolling_metrics_react_to_a_forecast_that_goes_wrong_partway_through():
    log = make_log(8, bias_from=4)
    joined = join_actuals(log, make_actuals(log), target="load_actual")
    rolling = rolling_metrics(joined, window=2).dropna(subset=["wape"])
    assert rolling["wape"].iloc[0] == pytest.approx(0.0)
    assert rolling["wape"].iloc[-1] > 0.3
    assert rolling["coverage"].iloc[-1] == 0.0


def test_rolling_metrics_ignore_unresolved_rows_but_need_at_least_one():
    log = make_log(3)
    joined = join_actuals(log, make_actuals(log).iloc[:HORIZON], target="load_actual")
    rolling = rolling_metrics(joined, window=1)
    assert len(rolling) == 1
    empty = join_actuals(log, make_actuals(log).iloc[:0], target="load_actual")
    with pytest.raises(ValueError, match="nothing to score"):
        rolling_metrics(empty, window=1)


def test_rolling_metrics_keep_model_versions_apart():
    log = pd.concat(
        [make_log(4), make_log(4, bias_from=0).assign(**{MODEL_VERSION_COL: "candidate"})],
        ignore_index=True,
    )
    joined = join_actuals(log, make_actuals(log), target="load_actual")
    rolling = rolling_metrics(joined, window=2)
    by_version = rolling.groupby(MODEL_VERSION_COL)["wape"].mean()
    assert by_version[VERSION] == pytest.approx(0.0)
    assert by_version["candidate"] > 0.3


# --- the alert rule -------------------------------------------------------------------------


def test_alerts_fire_once_per_episode_not_once_per_cutoff():
    log = make_log(8, bias_from=4)
    joined = join_actuals(log, make_actuals(log), target="load_actual")
    rolling = retraining_alerts(rolling_metrics(joined, window=1), wape_threshold=0.05)
    assert rolling[ALERT_COL].sum() == 4
    assert rolling["first_breach"].sum() == 1
    assert set(rolling["wape_threshold"]) == {0.05}


def test_alert_threshold_must_be_a_fraction():
    log = make_log(3)
    joined = join_actuals(log, make_actuals(log), target="load_actual")
    rolling = rolling_metrics(joined, window=1)
    with pytest.raises(ValueError, match="fraction"):
        retraining_alerts(rolling, wape_threshold=6.0)


# --- input drift ----------------------------------------------------------------------------


def test_input_drift_reports_no_shift_for_the_same_distribution():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"load_actual": rng.normal(100.0, 10.0, 4000)})
    other = pd.DataFrame({"load_actual": rng.normal(100.0, 10.0, 4000)})
    drift = input_drift(frame, other, columns=["load_actual"])
    assert abs(float(drift["z_shift"].iloc[0])) < 0.1
    assert float(drift["share_outside_reference_range"].iloc[0]) == pytest.approx(0.2, abs=0.05)


def test_input_drift_reports_a_shifted_mean_in_reference_standard_deviations():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"load_actual": rng.normal(100.0, 10.0, 4000)})
    shifted = pd.DataFrame({"load_actual": rng.normal(120.0, 10.0, 4000)})
    drift = input_drift(frame, shifted, columns=["load_actual"])
    assert float(drift["z_shift"].iloc[0]) == pytest.approx(2.0, abs=0.15)
    assert float(drift["share_outside_reference_range"].iloc[0]) > 0.5


def test_input_drift_rejects_a_column_absent_from_either_frame():
    frame = pd.DataFrame({"load_actual": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not in both frames"):
        input_drift(frame, frame, columns=["price_actual"])


# --- reporting helpers ----------------------------------------------------------------------


def test_log_summary_mentions_resolution_only_once_actuals_are_joined():
    log = make_log(3)
    assert "resolved" not in log_summary(log)
    joined = join_actuals(log, make_actuals(log), target="load_actual")
    summary = log_summary(joined)
    assert f"{len(log)} logged forecasts" in summary
    assert "resolved" in summary
    assert VERSION in summary


def test_to_json_line_is_valid_json_with_isoformat_timestamps():
    row = make_log(1).iloc[0].to_dict()
    payload = json.loads(to_json_line(row))
    assert payload[MODEL_VERSION_COL] == VERSION
    assert pd.Timestamp(payload[CUTOFF_COL]) == row[CUTOFF_COL]
    assert sorted(payload) == list(payload)
