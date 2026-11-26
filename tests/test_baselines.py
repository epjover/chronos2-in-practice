"""Tests for ``c2p.models`` and ``c2p.baselines``.

Fast tests use synthetic hourly series (daily sine + noise, two items, a couple of NaNs) and stub
pipelines; slow tests load the real Chronos-2 and Chronos-Bolt models on CPU.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Sequence
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest
from conftest import CHRONOS2_QUANTILES
from pandas.tseries.frequencies import to_offset
from scipy.stats import norm

from c2p.baselines import (
    MSTL,
    AutoETS,
    ChronosBolt,
    PublishedForecast,
    ResidualForecaster,
    SeasonalNaive,
    _from_statsforecast,
    _interval_level,
)
from c2p.models import (
    CHRONOS2,
    CHRONOS_BOLT_BASE,
    Chronos2Forecaster,
    assert_regular_grid,
    load_chronos2,
    load_chronos_bolt,
    parameter_count,
    validate_forecast_inputs,
)
from c2p.schema import (
    EVAL_QUANTILES,
    ID_COL,
    PREDICTION_COL,
    TARGET_COL,
    TARGET_NAME_COL,
    TIMESTAMP_COL,
    Forecaster,
    quantile_col,
)

FREQ = "h"
SEASON = 24
BOLT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def make_context(
    *,
    n_days: int = 14,
    items: Sequence[str] = ("A", "B"),
    targets: Sequence[str] = (TARGET_COL,),
    nan_positions: Sequence[int] = (30, 31),
    seed: int = 0,
) -> pd.DataFrame:
    """Hourly series frame: daily sine + noise per (item, target), NaN at ``nan_positions``."""
    rng = np.random.default_rng(seed)
    n = n_days * SEASON
    timestamps = pd.date_range("2024-03-01", periods=n, freq=FREQ)
    phase = 2 * np.pi * np.arange(n) / SEASON
    frames = []
    for i, item in enumerate(items):
        data: dict[str, object] = {ID_COL: item, TIMESTAMP_COL: timestamps}
        for j, target in enumerate(targets):
            values = 100 + 10 * i + (5 + 3 * j) * np.sin(phase + i) + rng.normal(0, 0.5, n)
            values[list(nan_positions)] = np.nan
            data[target] = values
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def assert_valid_forecast(
    forecast: pd.DataFrame,
    context: pd.DataFrame,
    *,
    targets: Sequence[str],
    prediction_length: int,
    quantile_levels: Sequence[float],
) -> None:
    """Check the forecast-frame contract of ``c2p.schema``."""
    quantile_columns = [quantile_col(q) for q in quantile_levels]
    assert list(forecast.columns) == [
        ID_COL,
        TIMESTAMP_COL,
        TARGET_NAME_COL,
        PREDICTION_COL,
        *quantile_columns,
    ]
    items = list(pd.unique(context[ID_COL]))
    assert len(forecast) == len(items) * len(targets) * prediction_length
    blocks = list(forecast.groupby([ID_COL, TARGET_NAME_COL], sort=False))
    assert [key for key, _ in blocks] == [(item, name) for item in items for name in targets]
    offset = to_offset(FREQ)
    for (item_id, _), block in blocks:
        last = context.loc[context[ID_COL] == item_id, TIMESTAMP_COL].max()
        expected = pd.date_range(last + offset, periods=prediction_length, freq=FREQ)
        assert block[TIMESTAMP_COL].tolist() == expected.tolist()
    values = forecast[[PREDICTION_COL, *quantile_columns]]
    assert (values.dtypes == "float64").all()
    assert not values.isna().any().any()
    ordered = forecast[[quantile_col(q) for q in sorted(quantile_levels)]].to_numpy()
    assert (np.diff(ordered, axis=1) >= 0).all(), "quantiles cross"


def item_values(frame: pd.DataFrame, item_id: str, column: str) -> np.ndarray:
    return frame.loc[frame[ID_COL] == item_id, column].to_numpy()


def observed_seasonal_sigma(values: np.ndarray) -> float:
    """RMS of the observed seasonal differences ``y[t] - y[t - SEASON]`` (NaN ones skipped)."""
    differences = values[SEASON:] - values[:-SEASON]
    return float(np.sqrt(np.nanmean(differences**2)))


def fake_predict_df(
    df: pd.DataFrame,
    future_df: pd.DataFrame | None = None,
    *,
    id_column: str,
    timestamp_column: str,
    target: str | list[str],
    prediction_length: int,
    quantile_levels: list[float],
    freq: str,
    **_: object,
) -> pd.DataFrame:
    """Stand-in for ``predict_df``: float32 output in Chronos' (item, target, step) layout."""
    targets = [target] if isinstance(target, str) else target
    blocks = []
    for item_id, group in df.groupby(id_column, sort=False):
        start = group[timestamp_column].iloc[-1] + to_offset(freq)
        timestamps = pd.date_range(start, periods=prediction_length, freq=freq)
        for name in targets:
            block = pd.DataFrame(
                {id_column: item_id, timestamp_column: timestamps, TARGET_NAME_COL: name}
            )
            block[PREDICTION_COL] = np.float32(0.5)
            for level in quantile_levels:
                block[str(level)] = np.float32(level)
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


@pytest.fixture
def chronos2_stub():
    from chronos import Chronos2Pipeline

    stub = create_autospec(Chronos2Pipeline, instance=True)
    stub.quantiles = list(CHRONOS2_QUANTILES)
    stub.predict_df.side_effect = fake_predict_df
    return stub


@pytest.fixture
def bolt_stub():
    from chronos import ChronosBoltPipeline

    stub = create_autospec(ChronosBoltPipeline, instance=True)
    stub.quantiles = BOLT_QUANTILES
    stub.predict_df.side_effect = fake_predict_df
    return stub


STAT_BASELINES = [SeasonalNaive(SEASON), AutoETS(SEASON), MSTL((SEASON, 2 * SEASON))]
STAT_IDS = ["seasonal_naive", "auto_ets", "mstl"]


# --------------------------------------------------------------------------------------------
# Shared input checks
# --------------------------------------------------------------------------------------------


def test_assert_regular_grid_accepts_regular_series_with_nan_values():
    assert_regular_grid(make_context(), FREQ)


def test_assert_regular_grid_reports_missing_rows():
    context = make_context()
    gappy = context.drop(index=context.index[(context[ID_COL] == "B")][50:53])
    with pytest.raises(ValueError, match=r"'B': expected 2024-03-03 02:00:00 after"):
        assert_regular_grid(gappy, FREQ)


def test_assert_regular_grid_reports_duplicates_and_unsorted_rows():
    context = make_context()
    with pytest.raises(ValueError, match="duplicated timestamps"):
        assert_regular_grid(pd.concat([context, context.iloc[[5]]]), FREQ)
    with pytest.raises(ValueError, match="not sorted"):
        assert_regular_grid(context.iloc[::-1], FREQ)


def test_assert_regular_grid_rejects_off_grid_start():
    frame = pd.DataFrame(
        {
            ID_COL: "A",
            TIMESTAMP_COL: pd.to_datetime(["2024-01-15", "2024-02-01", "2024-03-01"]),
            TARGET_COL: [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="not on the grid"):
        assert_regular_grid(frame, "MS")


@pytest.mark.parametrize(
    ("change", "kwargs", "error", "match"),
    [
        ("tz", {}, ValueError, "timezone-aware"),
        ("categorical_id", {}, ValueError, "'item_id' is categorical"),
        ("missing_id", {}, ValueError, "'item_id' has 2 missing values"),
        ("none", {"target": "missing"}, ValueError, "lacks the columns"),
        ("text_target", {}, ValueError, "must be numeric"),
        ("none", {"quantile_levels": [0.0, 0.5]}, ValueError, r"\(0, 1\)"),
        ("none", {"quantile_levels": [0.5, 0.5]}, ValueError, "Duplicated quantile"),
        ("none", {"quantile_levels": []}, ValueError, "At least one quantile"),
        ("none", {"target": []}, ValueError, "At least one target"),
        ("none", {"prediction_length": 0}, ValueError, "prediction_length"),
        ("none", {"prediction_length": 2.5}, TypeError, "prediction_length"),
        ("none", {"freq": "hourly"}, ValueError, "pandas frequency"),
    ],
)
def test_validate_forecast_inputs_errors(change, kwargs, error, match):
    context = make_context()
    if change == "tz":
        context[TIMESTAMP_COL] = context[TIMESTAMP_COL].dt.tz_localize("UTC")
    elif change == "text_target":
        context[TARGET_COL] = context[TARGET_COL].astype(str)
    elif change == "categorical_id":
        context[ID_COL] = context[ID_COL].astype("category")
    elif change == "missing_id":
        context.loc[[0, 1], ID_COL] = np.nan
    request = {
        "target": TARGET_COL,
        "quantile_levels": EVAL_QUANTILES,
        "prediction_length": 24,
        "freq": FREQ,
    }
    with pytest.raises(error, match=match):
        validate_forecast_inputs(context, **(request | kwargs))


def test_validate_forecast_inputs_returns_lists():
    targets, levels = validate_forecast_inputs(
        make_context(),
        target=TARGET_COL,
        quantile_levels=(0.1, 0.9),
        prediction_length=3,
        freq=FREQ,
    )
    assert targets == [TARGET_COL]
    assert levels == [0.1, 0.9]


# --------------------------------------------------------------------------------------------
# statsforecast conversion
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantile_level", "level"),
    [(0.1, 80), (0.9, 80), (0.01, 98), (0.99, 98), (0.15, 70), (0.45, 10), (0.025, 95)],
)
def test_interval_level_maps_quantiles_to_central_intervals(quantile_level, level):
    assert _interval_level(quantile_level) == level
    assert isinstance(_interval_level(quantile_level), int)


def test_interval_level_keeps_fractional_levels():
    assert _interval_level(0.0125) == 97.5


def test_from_statsforecast_sorts_crossing_quantiles():
    raw = pd.DataFrame(
        {
            ID_COL: ["A"],
            TIMESTAMP_COL: pd.to_datetime(["2024-01-01"]),
            "M": [1.0],
            "M-lo-80": [2.0],
            "M-hi-80": [0.0],
        }
    )
    out = _from_statsforecast(raw, alias="M", target="y", quantile_levels=[0.9, 0.1, 0.5])
    assert out[["0.1", "0.5", "0.9"]].to_numpy().tolist() == [[0.0, 1.0, 2.0]]
    assert out[PREDICTION_COL].tolist() == [1.0]
    assert out[TARGET_NAME_COL].tolist() == ["y"]


# --------------------------------------------------------------------------------------------
# Statistical baselines: shared contract
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("forecaster", STAT_BASELINES, ids=STAT_IDS)
def test_statistical_baseline_output_schema(forecaster):
    context = make_context()
    forecast = forecaster.predict(
        context, prediction_length=30, quantile_levels=EVAL_QUANTILES, freq=FREQ
    )
    assert isinstance(forecaster, Forecaster)
    assert_valid_forecast(
        forecast,
        context,
        targets=[TARGET_COL],
        prediction_length=30,
        quantile_levels=EVAL_QUANTILES,
    )


@pytest.mark.parametrize("forecaster", STAT_BASELINES, ids=STAT_IDS)
def test_statistical_baseline_keeps_requested_quantile_order(forecaster):
    context = make_context()
    levels = (0.9, 0.5, 0.1)
    forecast = forecaster.predict(context, prediction_length=5, quantile_levels=levels, freq=FREQ)
    assert_valid_forecast(
        forecast, context, targets=[TARGET_COL], prediction_length=5, quantile_levels=levels
    )


@pytest.mark.parametrize("forecaster", STAT_BASELINES, ids=STAT_IDS)
def test_statistical_baseline_keeps_the_context_item_order(forecaster):
    context = make_context(items=("B", "A"))
    forecast = forecaster.predict(
        context, prediction_length=4, quantile_levels=EVAL_QUANTILES, freq=FREQ
    )
    assert list(pd.unique(forecast[ID_COL])) == ["B", "A"]
    assert_valid_forecast(
        forecast, context, targets=[TARGET_COL], prediction_length=4, quantile_levels=EVAL_QUANTILES
    )


@pytest.mark.parametrize("forecaster", STAT_BASELINES, ids=STAT_IDS)
def test_statistical_baseline_loops_over_targets(forecaster):
    targets = ["load", "price"]
    context = make_context(targets=targets)
    forecast = forecaster.predict(
        context, prediction_length=12, quantile_levels=EVAL_QUANTILES, freq=FREQ, target=targets
    )
    assert_valid_forecast(
        forecast, context, targets=targets, prediction_length=12, quantile_levels=EVAL_QUANTILES
    )
    price_only = forecaster.predict(
        context, prediction_length=12, quantile_levels=EVAL_QUANTILES, freq=FREQ, target="price"
    )
    multi_price = forecast[forecast[TARGET_NAME_COL] == "price"].reset_index(drop=True)
    pd.testing.assert_frame_equal(multi_price, price_only)


@pytest.mark.parametrize("forecaster", STAT_BASELINES, ids=STAT_IDS)
def test_statistical_baseline_ignores_covariates(forecaster):
    context = make_context()
    with_covariate = context.assign(temperature=np.arange(len(context), dtype=float))
    future = pd.DataFrame({ID_COL: ["A"] * 6 + ["B"] * 6})  # never read by the baselines
    kwargs = {"prediction_length": 6, "quantile_levels": EVAL_QUANTILES, "freq": FREQ}
    pd.testing.assert_frame_equal(
        forecaster.predict(with_covariate, future_covariates=future, **kwargs),
        forecaster.predict(context, **kwargs),
    )


@pytest.mark.parametrize("forecaster", STAT_BASELINES, ids=STAT_IDS)
def test_statistical_baseline_rejects_gappy_grid(forecaster):
    context = make_context().drop(index=[100, 101])
    with pytest.raises(ValueError, match="regular 'h' grid"):
        forecaster.predict(context, prediction_length=6, quantile_levels=EVAL_QUANTILES, freq=FREQ)


# --------------------------------------------------------------------------------------------
# SeasonalNaive
# --------------------------------------------------------------------------------------------


def test_seasonal_naive_repeats_last_season_exactly():
    context = make_context()
    horizon = 2 * SEASON + 5
    forecast = SeasonalNaive(SEASON).predict(
        context, prediction_length=horizon, quantile_levels=EVAL_QUANTILES, freq=FREQ
    )
    for item_id in ("A", "B"):
        last_season = item_values(context, item_id, TARGET_COL)[-SEASON:]
        expected = np.tile(last_season, 3)[:horizon]
        np.testing.assert_array_equal(item_values(forecast, item_id, PREDICTION_COL), expected)


def test_seasonal_naive_median_equals_predictions():
    forecast = SeasonalNaive(SEASON).predict(
        make_context(), prediction_length=30, quantile_levels=EVAL_QUANTILES, freq=FREQ
    )
    np.testing.assert_array_equal(forecast[quantile_col(0.5)], forecast[PREDICTION_COL])


# ~40% NaN in the first 13 days plus one NaN in the last season (filled from the day before).
MANY_NANS = (
    *np.random.default_rng(3).choice(13 * SEASON, size=130, replace=False).tolist(),
    14 * SEASON - 3,
)


@pytest.mark.parametrize("nan_positions", [(), (30, 31), MANY_NANS], ids=["none", "two", "many"])
def test_seasonal_naive_quantiles_are_normal_intervals_of_observed_seasonal_differences(
    nan_positions,
):
    context = make_context(nan_positions=nan_positions)
    forecast = SeasonalNaive(SEASON).predict(
        context, prediction_length=2 * SEASON, quantile_levels=EVAL_QUANTILES, freq=FREQ
    )
    for item_id in ("A", "B"):
        # Missing differences (and the zeros the last-season fill creates) do not shrink sigma.
        sigma = observed_seasonal_sigma(item_values(context, item_id, TARGET_COL))
        point = item_values(forecast, item_id, PREDICTION_COL)
        for level in EVAL_QUANTILES:
            quantile = item_values(forecast, item_id, quantile_col(level))
            # First season: sigma; second season: sigma * sqrt(2).
            np.testing.assert_allclose(quantile[:SEASON], point[:SEASON] + norm.ppf(level) * sigma)
            np.testing.assert_allclose(
                quantile[SEASON:], point[SEASON:] + norm.ppf(level) * sigma * np.sqrt(2)
            )


def test_seasonal_naive_fills_nan_in_last_season_from_previous_season():
    n = 14 * SEASON
    context = make_context(nan_positions=(n - 3,))
    forecast = SeasonalNaive(SEASON).predict(
        context, prediction_length=SEASON, quantile_levels=EVAL_QUANTILES, freq=FREQ
    )
    values = item_values(context, "A", TARGET_COL)
    point = item_values(forecast, "A", PREDICTION_COL)
    assert point[SEASON - 3] == values[n - 3 - SEASON]
    np.testing.assert_array_equal(np.delete(point, SEASON - 3), np.delete(values[-SEASON:], -3))


@pytest.mark.parametrize(
    ("n_points", "nan_positions"),
    [(SEASON - 1, ()), (SEASON, ()), (2 * SEASON, tuple(range(SEASON, 2 * SEASON)))],
    ids=["shorter_than_a_season", "exactly_one_season", "second_season_missing"],
)
def test_seasonal_naive_rejects_context_without_observed_seasonal_differences(
    n_points, nan_positions
):
    # Without a single observed difference sigma would be 0: zero-width intervals.
    context = make_context(n_days=2, nan_positions=nan_positions).groupby(ID_COL).head(n_points)
    with pytest.raises(ValueError, match="'A'/'target' has no two observed values one season"):
        SeasonalNaive(SEASON).predict(
            context, prediction_length=3, quantile_levels=EVAL_QUANTILES, freq=FREQ
        )


def test_seasonal_naive_rejects_never_observed_season_position():
    n = 3 * SEASON
    context = make_context(n_days=3, nan_positions=(5, 5 + SEASON, 5 + 2 * SEASON))
    assert len(context) == 2 * n
    with pytest.raises(ValueError, match="never been observed at 1 of the 24"):
        SeasonalNaive(SEASON).predict(
            context, prediction_length=3, quantile_levels=EVAL_QUANTILES, freq=FREQ
        )


def test_seasonal_naive_validates_season_length():
    with pytest.raises(ValueError, match="season_length"):
        SeasonalNaive(0)


# --------------------------------------------------------------------------------------------
# AutoETS
# --------------------------------------------------------------------------------------------


def test_auto_ets_handles_nan_in_context_including_the_last_step():
    n = 14 * SEASON
    context = make_context(nan_positions=(10, 11, 12, 200, n - 1))
    forecast = AutoETS(SEASON).predict(
        context, prediction_length=SEASON, quantile_levels=EVAL_QUANTILES, freq=FREQ
    )
    assert_valid_forecast(
        forecast,
        context,
        targets=[TARGET_COL],
        prediction_length=SEASON,
        quantile_levels=EVAL_QUANTILES,
    )
    # The daily cycle is captured: the forecast tracks the next day's noise-free sine wave.
    phase = 2 * np.pi * np.arange(n, n + SEASON) / SEASON
    truth = 100 + 5 * np.sin(phase)
    assert np.abs(item_values(forecast, "A", PREDICTION_COL) - truth).mean() < 1.0


def test_auto_ets_uses_only_the_last_max_context_steps():
    context = make_context()
    forecaster = AutoETS(SEASON, max_context=7 * SEASON)
    kwargs = {"prediction_length": 12, "quantile_levels": EVAL_QUANTILES, "freq": FREQ}
    old = context.groupby(ID_COL).cumcount() < len(context) // 2 - 7 * SEASON
    corrupted = context.assign(**{TARGET_COL: context[TARGET_COL].mask(old, 1e6)})
    pd.testing.assert_frame_equal(
        forecaster.predict(corrupted, **kwargs), forecaster.predict(context, **kwargs)
    )


def test_auto_ets_rejects_all_nan_series():
    context = make_context()
    context.loc[context[ID_COL] == "B", TARGET_COL] = np.nan
    with pytest.raises(ValueError, match="'B'/'target' has no observed values"):
        AutoETS(SEASON).predict(
            context, prediction_length=3, quantile_levels=EVAL_QUANTILES, freq=FREQ
        )


@pytest.mark.parametrize(
    ("season_length", "n_points"),
    [(SEASON, SEASON), (SEASON, 2 * SEASON - 1), (1, 6)],
    ids=["one_season", "just_under_two_seasons", "tiny_non_seasonal"],
)
def test_auto_ets_rejects_contexts_shorter_than_min_context(season_length, n_points):
    # statsforecast would silently fit a non-seasonal model (or raise "tiny datasets").
    context = make_context(n_days=2, nan_positions=()).groupby(ID_COL).head(n_points)
    with pytest.raises(ValueError, match=f"'A'/'target' has {n_points} points, but AutoETS"):
        AutoETS(season_length).predict(
            context, prediction_length=3, quantile_levels=EVAL_QUANTILES, freq=FREQ
        )


def test_auto_ets_accepts_exactly_two_seasons():
    forecaster = AutoETS(SEASON)
    assert forecaster.min_context == 2 * SEASON
    context = make_context(n_days=2, nan_positions=())
    forecast = forecaster.predict(
        context, prediction_length=3, quantile_levels=EVAL_QUANTILES, freq=FREQ
    )
    assert_valid_forecast(
        forecast, context, targets=[TARGET_COL], prediction_length=3, quantile_levels=EVAL_QUANTILES
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"season_length": 168}, "season_length must be in"),
        ({"season_length": 0}, "season_length must be in"),
        ({"season_length": 24, "max_context": 47}, "max_context must be at least 48"),
        ({"season_length": 1, "max_context": 6}, "max_context must be at least 7"),
    ],
)
def test_auto_ets_validates_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AutoETS(**kwargs)


# --------------------------------------------------------------------------------------------
# Chronos wrappers with stub pipelines (no network)
# --------------------------------------------------------------------------------------------


def test_chronos2_forecaster_warns_on_cross_learning_over_many_series(chronos2_stub):
    kwargs = {"prediction_length": 6, "quantile_levels": EVAL_QUANTILES, "freq": FREQ}
    many = make_context(n_days=3, items=[f"s{i:03d}" for i in range(101)], nan_positions=())
    model = Chronos2Forecaster(chronos2_stub, cross_learning=True)
    with pytest.warns(UserWarning, match="depends on how the frame is cut"):
        model.predict(many, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.predict(many, **kwargs)  # the second window of a backtest does not repeat it
        Chronos2Forecaster(chronos2_stub, cross_learning=False).predict(many, **kwargs)
        Chronos2Forecaster(chronos2_stub, cross_learning=True).predict(make_context(), **kwargs)


def test_chronos2_forecaster_passes_future_df_only_with_covariates(chronos2_stub):
    forecaster = Chronos2Forecaster(chronos2_stub, context_length=128, cross_learning=True)
    context = make_context().assign(temperature=1.0, past_only=2.0)
    kwargs = {"prediction_length": 6, "quantile_levels": EVAL_QUANTILES, "freq": FREQ}

    forecast = forecaster.predict(context, **kwargs)
    call = chronos2_stub.predict_df.call_args
    assert call.kwargs["future_df"] is None
    assert call.args[0] is context  # covariate columns reach Chronos untouched
    assert call.kwargs["target"] == [TARGET_COL]
    assert call.kwargs["freq"] == FREQ
    assert call.kwargs["context_length"] == 128
    assert call.kwargs["cross_learning"] is True
    assert_valid_forecast(
        forecast,
        context,
        targets=[TARGET_COL],
        prediction_length=6,
        quantile_levels=EVAL_QUANTILES,
    )

    future = forecast[[ID_COL, TIMESTAMP_COL]].assign(temperature=1.0)
    forecaster.predict(context, future_covariates=future, **kwargs)
    assert chronos2_stub.predict_df.call_args.kwargs["future_df"] is future


def test_chronos2_forecaster_orders_multi_target_output(chronos2_stub):
    targets = ["price", "load"]
    context = make_context(targets=targets)
    forecast = Chronos2Forecaster(chronos2_stub).predict(
        context, prediction_length=4, quantile_levels=EVAL_QUANTILES, freq=FREQ, target=targets
    )
    assert_valid_forecast(
        forecast, context, targets=targets, prediction_length=4, quantile_levels=EVAL_QUANTILES
    )


def test_chronos2_forecaster_rejects_gappy_grid_before_calling_the_model(chronos2_stub):
    context = make_context().drop(index=[100])
    with pytest.raises(ValueError, match="regular 'h' grid"):
        Chronos2Forecaster(chronos2_stub).predict(
            context, prediction_length=6, quantile_levels=EVAL_QUANTILES, freq=FREQ
        )
    chronos2_stub.predict_df.assert_not_called()


def test_chronos2_forecaster_keeps_the_context_item_order(chronos2_stub):
    context = make_context(items=("B", "A"))
    forecast = Chronos2Forecaster(chronos2_stub).predict(
        context, prediction_length=4, quantile_levels=EVAL_QUANTILES, freq=FREQ
    )
    assert list(pd.unique(forecast[ID_COL])) == ["B", "A"]
    assert_valid_forecast(
        forecast, context, targets=[TARGET_COL], prediction_length=4, quantile_levels=EVAL_QUANTILES
    )


def test_chronos2_forecaster_rejects_categorical_ids_before_calling_the_model(chronos2_stub):
    # Chronos would count the unused category "B" as an empty series and misalign the others.
    context = make_context(items=("A", "B", "C")).astype({ID_COL: "category"})
    context = context[context[ID_COL] != "B"]
    with pytest.raises(ValueError, match="categorical"):
        Chronos2Forecaster(chronos2_stub).predict(
            context, prediction_length=2, quantile_levels=EVAL_QUANTILES, freq=FREQ
        )
    chronos2_stub.predict_df.assert_not_called()


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        (lambda raw: raw.drop(columns="0.9"), r"without the columns \['0.9'\]"),
        (
            lambda raw: raw.assign(**{"0.1": raw["0.1"].where(raw.index != 3)}),
            r"NaN forecasts for \(item, target\) pairs \[\('A', 'target'\)\]",
        ),
        (lambda raw: pd.concat([raw, raw.iloc[[0]]]), "duplicated"),
        (
            lambda raw: raw.assign(**{TIMESTAMP_COL: raw[TIMESTAMP_COL] + pd.Timedelta(hours=1)}),
            "2 expected .* rows are missing .* and 2 rows are unexpected",
        ),
        (lambda raw: raw[raw[ID_COL] != "B"], "6 expected .* rows are missing"),
    ],
    ids=["missing_column", "nan_quantile", "duplicated_rows", "shifted_timestamps", "lost_item"],
)
def test_chronos2_forecaster_rejects_malformed_model_output(chronos2_stub, corrupt, match):
    chronos2_stub.predict_df.side_effect = lambda *args, **kwargs: corrupt(
        fake_predict_df(*args, **kwargs)
    )
    with pytest.raises(ValueError, match=match):
        Chronos2Forecaster(chronos2_stub).predict(
            make_context(), prediction_length=6, quantile_levels=[0.1, 0.5, 0.9], freq=FREQ
        )


def test_chronos2_forecaster_rejects_other_pipelines(bolt_stub):
    with pytest.raises(TypeError, match="Expected a Chronos2Pipeline"):
        Chronos2Forecaster(bolt_stub)


def test_load_chronos2_rejects_an_unknown_precision_before_loading_weights():
    with pytest.raises(ValueError, match="dtype must be one of"):
        load_chronos2(dtype="float8")


def test_load_chronos2_rejects_other_families_before_loading_weights(tmp_path):
    config = {"model_type": "t5", "chronos_pipeline_class": "ChronosBoltPipeline"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(TypeError, match="ChronosBoltPipeline checkpoint, not a Chronos-2 one"):
        load_chronos2(str(tmp_path), device="cpu")


def test_load_chronos_bolt_rejects_other_families(monkeypatch, chronos2_stub):
    from chronos import BaseChronosPipeline

    monkeypatch.setattr(BaseChronosPipeline, "from_pretrained", lambda *_, **__: chronos2_stub)
    with pytest.raises(TypeError, match="not a ChronosBoltPipeline"):
        load_chronos_bolt(CHRONOS2, device="cpu")


def test_chronos_bolt_forecasts_each_target_from_its_own_column(bolt_stub):
    targets = ["load", "price"]
    context = make_context(targets=targets).assign(temperature=1.0)
    forecast = ChronosBolt(bolt_stub).predict(
        context, prediction_length=6, quantile_levels=BOLT_QUANTILES, freq=FREQ, target=targets
    )
    calls = bolt_stub.predict_df.call_args_list
    assert [call.kwargs["target"] for call in calls] == targets
    assert [list(call.args[0].columns) for call in calls] == [
        [ID_COL, TIMESTAMP_COL, name] for name in targets
    ]
    assert all(call.kwargs["freq"] == FREQ for call in calls)
    assert_valid_forecast(
        forecast, context, targets=targets, prediction_length=6, quantile_levels=BOLT_QUANTILES
    )


def test_chronos_bolt_warns_for_levels_outside_its_training_range(bolt_stub, caplog):
    model = ChronosBolt(bolt_stub)
    with caplog.at_level(logging.WARNING, logger="c2p.models"):
        for _ in range(3):  # three windows of a backtest: one warning
            model.predict(
                make_context(), prediction_length=3, quantile_levels=EVAL_QUANTILES, freq=FREQ
            )
    assert caplog.text.count("levels [0.01, 0.99] are clamped") == 1


def test_chronos_bolt_rejects_pipeline_with_loading_arguments(bolt_stub, chronos2_stub):
    with pytest.raises(ValueError, match="either a loaded pipeline"):
        ChronosBolt(bolt_stub, device="cpu")
    with pytest.raises(TypeError, match="Expected a ChronosBoltPipeline"):
        ChronosBolt(chronos2_stub)


# --------------------------------------------------------------------------------------------
# Slow tests: real models on CPU
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chronos2_pipeline():
    return load_chronos2(CHRONOS2, device="cpu")


@pytest.fixture(scope="module")
def chronos_bolt():
    return ChronosBolt(device="cpu")


def make_covariate_context(
    n: int, horizon: int, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Target = 50 + 10 * driver + noise; returns (context, future covariates, actuals)."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-03-01", periods=n + horizon, freq=FREQ)
    driver = rng.normal(0, 1, n + horizon)
    full = pd.DataFrame(
        {
            ID_COL: "A",
            TIMESTAMP_COL: timestamps,
            TARGET_COL: 50 + 10 * driver + rng.normal(0, 0.2, n + horizon),
            "driver": driver,
            "past_only": rng.normal(0, 1, n + horizon),
        }
    )
    context, horizon_rows = full.iloc[:n], full.iloc[n:].reset_index(drop=True)
    return context, horizon_rows[[ID_COL, TIMESTAMP_COL, "driver"]], horizon_rows[TARGET_COL]


@pytest.mark.slow
def test_chronos2_forecaster_real_model(chronos2_pipeline):
    assert 119e6 < parameter_count(chronos2_pipeline) < 120e6  # 119.5M parameters
    targets = [TARGET_COL, "other"]
    context = make_context(targets=targets)
    forecaster = Chronos2Forecaster(chronos2_pipeline)
    forecast = forecaster.predict(
        context, prediction_length=SEASON, quantile_levels=EVAL_QUANTILES, freq=FREQ, target=targets
    )
    assert_valid_forecast(
        forecast,
        context,
        targets=targets,
        prediction_length=SEASON,
        quantile_levels=EVAL_QUANTILES,
    )
    np.testing.assert_allclose(forecast[quantile_col(0.5)], forecast[PREDICTION_COL])
    n = 14 * SEASON
    truth = 100 + 5 * np.sin(2 * np.pi * np.arange(n, n + SEASON) / SEASON)
    point = forecast.loc[
        (forecast[ID_COL] == "A") & (forecast[TARGET_NAME_COL] == TARGET_COL), PREDICTION_COL
    ]
    assert np.abs(point.to_numpy() - truth).mean() < 1.0


@pytest.mark.slow
def test_chronos2_forecaster_uses_future_covariates(chronos2_pipeline):
    context, future, actuals = make_covariate_context(n=24 * 21, horizon=SEASON)
    forecaster = Chronos2Forecaster(chronos2_pipeline)
    kwargs = {"prediction_length": SEASON, "quantile_levels": EVAL_QUANTILES, "freq": FREQ}
    with_future = forecaster.predict(context, future_covariates=future, **kwargs)
    past_only = forecaster.predict(context, **kwargs)
    for forecast in (with_future, past_only):
        assert_valid_forecast(
            forecast,
            context,
            targets=[TARGET_COL],
            prediction_length=SEASON,
            quantile_levels=EVAL_QUANTILES,
        )
    error_with = np.abs(with_future[PREDICTION_COL] - actuals).mean()
    error_without = np.abs(past_only[PREDICTION_COL] - actuals).mean()
    assert error_with < 0.5 * error_without


@pytest.mark.slow
def test_chronos2_forecaster_real_model_rejects_gappy_grid(chronos2_pipeline):
    context = make_context().drop(index=[100, 101, 102])
    with pytest.raises(ValueError, match="regular 'h' grid"):
        Chronos2Forecaster(chronos2_pipeline).predict(
            context, prediction_length=SEASON, quantile_levels=EVAL_QUANTILES, freq=FREQ
        )


@pytest.mark.slow
def test_load_chronos2_rejects_a_chronos_bolt_model():
    with pytest.raises(TypeError, match="Use load_chronos_bolt"):
        load_chronos2(CHRONOS_BOLT_BASE, device="cpu")


@pytest.mark.slow
def test_chronos_bolt_real_model(chronos_bolt, caplog):
    targets = [TARGET_COL, "other"]
    context = make_context(targets=targets)
    with caplog.at_level(logging.WARNING, logger="c2p.models"):
        forecast = chronos_bolt.predict(
            context,
            prediction_length=SEASON,
            quantile_levels=EVAL_QUANTILES,
            freq=FREQ,
            target=targets,
        )
    assert "clamped" in caplog.text
    assert_valid_forecast(
        forecast,
        context,
        targets=targets,
        prediction_length=SEASON,
        quantile_levels=EVAL_QUANTILES,
    )
    # Levels outside [0.1, 0.9] are clamped by the library to the nearest trained level.
    np.testing.assert_allclose(forecast["0.01"], forecast["0.1"])
    np.testing.assert_allclose(forecast["0.99"], forecast["0.9"])
    np.testing.assert_allclose(forecast[quantile_col(0.5)], forecast[PREDICTION_COL])


# --------------------------------------------------------------------------------------------
# MSTL and the published forecast
# --------------------------------------------------------------------------------------------


def test_mstl_follows_a_series_with_two_cycles():
    steps = np.arange(28 * SEASON)
    daily = 10 * np.sin(2 * np.pi * steps / SEASON)
    weekly = 6 * np.sin(2 * np.pi * steps / (7 * SEASON))
    noise = np.random.default_rng(0).normal(0, 0.2, len(steps))
    values = 100 + daily + weekly + noise
    frame = pd.DataFrame(
        {
            ID_COL: "A",
            TIMESTAMP_COL: pd.date_range("2024-03-01", periods=len(steps), freq=FREQ),
            TARGET_COL: values,
        }
    )
    context, actual = frame.iloc[:-SEASON], values[-SEASON:]
    kwargs = {"prediction_length": SEASON, "quantile_levels": EVAL_QUANTILES, "freq": FREQ}
    two_cycles = MSTL((SEASON, 7 * SEASON)).predict(context, **kwargs)
    one_cycle = AutoETS(SEASON).predict(context, **kwargs)
    error_two = np.abs(two_cycles[PREDICTION_COL].to_numpy() - actual).mean()
    error_one = np.abs(one_cycle[PREDICTION_COL].to_numpy() - actual).mean()
    assert error_two < 1.0
    assert error_two < 0.5 * error_one


def test_mstl_uses_only_the_last_max_context_steps():
    context = make_context(n_days=20, items=("A",), nan_positions=())
    kwargs = {"prediction_length": 6, "quantile_levels": [0.1, 0.5, 0.9], "freq": FREQ}
    shifted = context.copy()
    shifted.loc[shifted.index[: 10 * SEASON], TARGET_COL] += 1000.0  # before the last 10 days
    model = MSTL((SEASON, 2 * SEASON), max_context=8 * SEASON)
    pd.testing.assert_frame_equal(
        model.predict(shifted, **kwargs), model.predict(context, **kwargs)
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"season_lengths": ()}, "at least 2"),
        ({"season_lengths": (24, 1)}, "at least 2"),
        ({"season_lengths": (24, 24)}, "repeated"),
        ({"season_lengths": (24, 168), "max_context": 200}, "at least 336"),
    ],
)
def test_mstl_validates_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MSTL(**kwargs)


def test_mstl_rejects_contexts_shorter_than_two_long_cycles():
    context = make_context(n_days=6, items=("A",), nan_positions=())
    with pytest.raises(ValueError, match="needs at least 336"):
        MSTL((SEASON, 7 * SEASON)).predict(
            context, prediction_length=6, quantile_levels=[0.5], freq=FREQ
        )


def make_published_future(context: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    frames = []
    for item, group in context.groupby(ID_COL, sort=False):
        start = group[TIMESTAMP_COL].max() + to_offset(FREQ)
        frames.append(
            pd.DataFrame(
                {
                    ID_COL: item,
                    TIMESTAMP_COL: pd.date_range(start, periods=prediction_length, freq=FREQ),
                    "operator": np.arange(prediction_length, dtype=float) + len(frames) * 100,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_published_forecast_replays_the_column_at_every_level():
    context = make_context()
    future = make_published_future(context, 6)
    model = PublishedForecast("operator", name="tso")
    forecast = model.predict(
        context,
        prediction_length=6,
        quantile_levels=EVAL_QUANTILES,
        freq=FREQ,
        future_covariates=future,
    )
    assert isinstance(model, Forecaster)
    assert_valid_forecast(
        forecast, context, targets=[TARGET_COL], prediction_length=6, quantile_levels=EVAL_QUANTILES
    )
    assert forecast[PREDICTION_COL].tolist() == future["operator"].tolist()
    for level in EVAL_QUANTILES:
        assert forecast[quantile_col(level)].tolist() == future["operator"].tolist()


def test_published_forecast_wql_equals_wape_on_the_backtest_metrics():
    from c2p.eval import BacktestConfig, backtest, evaluate

    context = make_context(n_days=10, items=("A",), nan_positions=())
    frame = context.assign(operator=context[TARGET_COL] + 1.5)
    config = BacktestConfig(
        prediction_length=12,
        n_windows=3,
        freq=FREQ,
        season_length=SEASON,
        known_covariates=("operator",),
    )
    metrics = evaluate(backtest(PublishedForecast("operator"), frame, config), frame, config)
    assert metrics["WQL"].to_numpy() == pytest.approx(metrics["WAPE"].to_numpy())
    assert (metrics["coverage_80"] == 0.0).all()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda future: None, "needs 'operator' in future_covariates"),
        (lambda future: future.drop(columns="operator"), "needs 'operator' in future_covariates"),
        (lambda future: future.assign(operator=np.nan), "missing values in the horizon"),
    ],
)
def test_published_forecast_rejects_a_missing_or_incomplete_column(change, match):
    context = make_context()
    future = change(make_published_future(context, 6))
    with pytest.raises(ValueError, match=match):
        PublishedForecast("operator").predict(
            context, prediction_length=6, quantile_levels=[0.5], freq=FREQ, future_covariates=future
        )


def test_published_forecast_rejects_several_targets():
    context = make_context(targets=("load", "price"))
    future = make_published_future(context, 6)
    with pytest.raises(ValueError, match="one target"):
        PublishedForecast("operator").predict(
            context,
            prediction_length=6,
            quantile_levels=[0.5],
            freq=FREQ,
            target=["load", "price"],
            future_covariates=future,
        )


def test_residual_forecaster_adds_the_published_forecast_back():
    context = make_context(nan_positions=())
    # The published forecast is the target minus a daily pattern the naive can repeat exactly.
    hours = context[TIMESTAMP_COL].dt.hour.to_numpy()
    context = context.assign(operator=context[TARGET_COL] - np.sin(hours))
    future = make_published_future(context, SEASON)
    model = ResidualForecaster(SeasonalNaive(SEASON), "operator")
    forecast = model.predict(
        context,
        prediction_length=SEASON,
        quantile_levels=[0.1, 0.5, 0.9],
        freq=FREQ,
        future_covariates=future,
    )
    assert model.name == "seasonal_naive_on_operator_error"
    assert_valid_forecast(
        forecast,
        context,
        targets=[TARGET_COL],
        prediction_length=SEASON,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    future_hours = future[TIMESTAMP_COL].dt.hour.to_numpy()
    expected = future["operator"].to_numpy() + np.sin(future_hours)
    assert forecast[PREDICTION_COL].to_numpy() == pytest.approx(expected, abs=1e-9)
    assert (forecast["0.9"] >= forecast["0.5"]).all() and (forecast["0.5"] >= forecast["0.1"]).all()


def test_residual_forecaster_hands_the_base_only_the_residual():
    seen = {}

    class Spy:
        name = "spy"

        def predict(self, context, **kwargs):
            seen["columns"] = list(context.columns)
            seen["future"] = kwargs.get("future_covariates")
            return SeasonalNaive(SEASON).predict(context, **kwargs)

    context = make_context(nan_positions=()).assign(operator=1.0, other=2.0)
    future = make_published_future(context, 6)
    ResidualForecaster(Spy(), "operator").predict(
        context, prediction_length=6, quantile_levels=[0.5], freq=FREQ, future_covariates=future
    )
    assert seen["columns"] == [ID_COL, TIMESTAMP_COL, TARGET_COL]
    assert seen["future"] is None


@pytest.mark.parametrize(
    ("context_change", "future_change", "match"),
    [
        (lambda c: c.drop(columns="operator"), lambda f: f, "in the context"),
        (lambda c: c, lambda f: None, "in future_covariates"),
        (lambda c: c, lambda f: f.assign(operator=np.nan), "missing values in the horizon"),
    ],
)
def test_residual_forecaster_rejects_incomplete_inputs(context_change, future_change, match):
    context = make_context(nan_positions=()).assign(operator=1.0)
    future = future_change(make_published_future(context, 6))
    with pytest.raises(ValueError, match=match):
        ResidualForecaster(SeasonalNaive(SEASON), "operator").predict(
            context_change(context),
            prediction_length=6,
            quantile_levels=[0.5],
            freq=FREQ,
            future_covariates=future,
        )
