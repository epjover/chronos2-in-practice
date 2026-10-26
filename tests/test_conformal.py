"""Tests for ``c2p.conformal``: CQR calibration, application and the wrapped forecaster.

The tests build a toy backtest frame whose intervals are too narrow by a known amount, so the
correct adjustment is known in advance, and check the guarantee CQR actually makes: on the
calibration data itself, coverage after calibration is at least the nominal level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from c2p.conformal import (
    STEP_COL,
    ConformalForecaster,
    calibrate,
    conformalize,
    coverage_by_period,
    horizon_step,
    symmetric_pairs,
)
from c2p.eval import ACTUAL_COL, MODEL_COL, WINDOW_COL
from c2p.schema import ID_COL, PREDICTION_COL, TARGET_NAME_COL, TIMESTAMP_COL, quantile_col

LEVELS = (0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99)
HORIZON = 4


def toy_backtest(n_windows: int = 60, *, width: float = 1.0, noise: float = 2.0, seed: int = 0):
    """Forecasts whose quantile bands have half-width ``width`` around a truth with sd ``noise``."""
    rng = np.random.default_rng(seed)
    rows = []
    for window in range(n_windows):
        cutoff = pd.Timestamp("2017-07-01") + pd.Timedelta(days=window)
        for step in range(1, HORIZON + 1):
            median = 100.0 + step
            row = {
                WINDOW_COL: window,
                ID_COL: "s",
                TARGET_NAME_COL: "y",
                TIMESTAMP_COL: cutoff + pd.Timedelta(hours=step),
                PREDICTION_COL: median,
                ACTUAL_COL: median + rng.normal(0.0, noise),
                MODEL_COL: "toy",
            }
            for level in LEVELS:
                row[quantile_col(level)] = median + (level - 0.5) * 2 * width
            rows.append(row)
    return pd.DataFrame(rows)


def coverage(frame: pd.DataFrame, lower: float, upper: float) -> float:
    inside = frame[ACTUAL_COL].between(frame[quantile_col(lower)], frame[quantile_col(upper)])
    return float(inside.mean())


# --- helpers --------------------------------------------------------------------------------


def test_symmetric_pairs_are_widest_first_and_skip_the_median():
    assert symmetric_pairs(LEVELS) == [(0.01, 0.99), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
    with pytest.raises(ValueError, match="symmetric"):
        symmetric_pairs((0.1, 0.5, 0.95))


def test_horizon_step_counts_from_one_within_each_window_and_series():
    frame = toy_backtest(3)
    steps = horizon_step(frame)
    assert steps.min() == 1 and steps.max() == HORIZON
    per_window = steps.groupby(frame[WINDOW_COL]).apply(list)
    assert all(values == [1, 2, 3, 4] for values in per_window)


# --- calibration ----------------------------------------------------------------------------


def test_calibration_widens_bands_that_were_too_narrow_until_coverage_is_honest():
    frame = toy_backtest(width=1.0, noise=2.0)
    assert coverage(frame, 0.1, 0.9) < 0.5  # the toy bands miss most of the time
    calibration = calibrate(frame, quantile_levels=LEVELS, mode="additive")
    assert (calibration.table["delta"] > 0).all()
    adjusted = conformalize(frame, calibration, quantile_levels=LEVELS)
    for lower, upper in symmetric_pairs(LEVELS):
        assert coverage(adjusted, lower, upper) >= upper - lower - 1e-9


def test_calibration_narrows_bands_that_were_too_wide():
    frame = toy_backtest(width=50.0, noise=2.0)
    calibration = calibrate(frame, quantile_levels=LEVELS, mode="additive")
    assert (calibration.table["delta"] < 0).all()
    adjusted = conformalize(frame, calibration, quantile_levels=LEVELS)
    assert coverage(adjusted, 0.1, 0.9) < coverage(frame, 0.1, 0.9)
    assert coverage(adjusted, 0.1, 0.9) >= 0.8 - 1e-9


def test_scaled_mode_learns_a_multiplier_not_a_shift():
    frame = toy_backtest(width=1.0, noise=2.0)
    scaled = calibrate(frame, quantile_levels=LEVELS, mode="scaled")
    additive = calibrate(frame, quantile_levels=LEVELS, mode="additive")
    pair = (0.1, 0.9)
    width = 2 * (0.9 - 0.5) * 2 * 1.0  # the toy interval width for this pair
    assert scaled.delta(*pair).to_numpy() * width == pytest.approx(additive.delta(*pair).to_numpy())
    adjusted = conformalize(frame, scaled, quantile_levels=LEVELS)
    assert coverage(adjusted, *pair) >= 0.8 - 1e-9


def test_calibration_is_per_step_by_default_and_global_on_request():
    frame = toy_backtest()
    per_step = calibrate(frame, quantile_levels=LEVELS)
    assert sorted(per_step.table[STEP_COL].unique()) == [1, 2, 3, 4]
    pooled = calibrate(frame, quantile_levels=LEVELS, by_step=False)
    assert sorted(pooled.table[STEP_COL].unique()) == [1]
    assert pooled.n_calibration_rows == len(frame)


def test_calibration_rejects_frames_without_actuals():
    frame = toy_backtest(5)
    with pytest.raises(ValueError, match="lacks the columns"):
        calibrate(frame.drop(columns=[ACTUAL_COL]), quantile_levels=LEVELS)
    with pytest.raises(ValueError, match="No calibration row"):
        calibrate(frame.assign(**{ACTUAL_COL: np.nan}), quantile_levels=LEVELS)


# --- application ----------------------------------------------------------------------------


def test_conformalize_keeps_quantiles_ordered_and_median_as_prediction():
    frame = toy_backtest(width=1.0, noise=2.0)
    calibration = calibrate(frame, quantile_levels=LEVELS)
    adjusted = conformalize(frame, calibration, quantile_levels=LEVELS)
    columns = [quantile_col(level) for level in LEVELS]
    values = adjusted[columns].to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()
    assert adjusted[PREDICTION_COL].to_numpy() == pytest.approx(
        adjusted[quantile_col(0.5)].to_numpy()
    )
    assert adjusted[quantile_col(0.5)].to_numpy() == pytest.approx(
        frame[quantile_col(0.5)].to_numpy()
    )


def test_conformalize_reuses_the_last_step_beyond_the_calibrated_horizon():
    frame = toy_backtest()
    calibration = calibrate(frame, quantile_levels=LEVELS)
    longer = toy_backtest(2)
    extra = longer[longer[WINDOW_COL] == 0].copy()
    extra[TIMESTAMP_COL] = extra[TIMESTAMP_COL] + pd.Timedelta(hours=HORIZON)
    longer = pd.concat([longer, extra], ignore_index=True)
    adjusted = conformalize(longer, calibration, quantile_levels=LEVELS)
    assert len(adjusted) == len(longer)
    assert adjusted[quantile_col(0.9)].notna().all()


def test_conformalize_does_not_mutate_its_input():
    frame = toy_backtest(5)
    before = frame.copy()
    conformalize(frame, calibrate(frame, quantile_levels=LEVELS), quantile_levels=LEVELS)
    pd.testing.assert_frame_equal(frame, before)


# --- the wrapped forecaster -----------------------------------------------------------------


class NarrowForecaster:
    name = "narrow"

    def predict(self, context, *, prediction_length, quantile_levels, freq, target, **_):
        last = context.sort_values(TIMESTAMP_COL).iloc[-1]
        stamps = pd.date_range(
            last[TIMESTAMP_COL] + pd.Timedelta(hours=1), periods=prediction_length, freq=freq
        )
        frame = pd.DataFrame(
            {ID_COL: "s", TIMESTAMP_COL: stamps, TARGET_NAME_COL: target, PREDICTION_COL: 100.0}
        )
        for level in quantile_levels:
            frame[quantile_col(level)] = 100.0 + (level - 0.5) * 2.0
        return frame


def test_conformal_forecaster_wraps_a_base_forecaster_and_names_itself():
    calibration = calibrate(toy_backtest(width=1.0, noise=2.0), quantile_levels=LEVELS)
    wrapped = ConformalForecaster(NarrowForecaster(), calibration)
    assert wrapped.name == "narrow+conformal_additive"
    context = pd.DataFrame(
        {
            ID_COL: "s",
            TIMESTAMP_COL: pd.date_range("2018-01-01", periods=48, freq="h"),
            "y": 100.0,
        }
    )
    out = wrapped.predict(
        context, prediction_length=HORIZON, quantile_levels=LEVELS, freq="h", target="y"
    )
    assert len(out) == HORIZON
    assert (out[quantile_col(0.9)] - out[quantile_col(0.1)] > 2 * 0.8).all()


# --- the seasonal diagnostic ----------------------------------------------------------------


def test_coverage_by_period_reports_one_row_per_model_and_month():
    frame = toy_backtest(90)
    out = coverage_by_period(frame, lower=0.1, upper=0.9, period="M")
    assert list(out.columns) == [MODEL_COL, "period", "coverage", "n", "nominal"]
    assert out["period"].tolist() == ["2017-07", "2017-08", "2017-09"]
    assert out["n"].sum() == len(frame)
    assert out["nominal"].to_numpy() == pytest.approx(0.8)
