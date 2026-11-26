"""Tests for the rolling-origin backtest harness (synthetic data only, no network)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import fev.metrics
import numpy as np
import pandas as pd
import pytest
import scipy.stats
from conftest import CHRONOS2_QUANTILES

from c2p.eval import (
    ACTUAL_COL,
    COVERAGE_COLUMNS,
    CUTOFF_COL,
    METRICS,
    MODEL_COL,
    N_SERIES_COL,
    SCORED_QUANTILES,
    SECONDS_COL,
    WINDOW_COL,
    BacktestConfig,
    Window,
    average_runs,
    backtest,
    error_by_horizon_block,
    error_by_series,
    error_by_step,
    evaluate,
    reliability_table,
    rolling_origin_windows,
    skill_score,
    skill_table,
    summarize,
    summarize_runs,
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

START = pd.Timestamp("2024-01-01")
STEP = pd.Timedelta(hours=1)
QUANTILE_COLUMNS = [quantile_col(q) for q in EVAL_QUANTILES]
FORECAST_COLUMNS = [ID_COL, TIMESTAMP_COL, TARGET_NAME_COL, PREDICTION_COL, *QUANTILE_COLUMNS]
# fev-bench scores WQL and SQL on the nine deciles; written out to check the module against.
FEV_BENCH_DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DECILE_COLUMNS = [quantile_col(q) for q in FEV_BENCH_DECILES]
FEV_BENCH_METRICS = {
    "WQL": fev.metrics.WQL(epsilon=1.0),
    "MASE": fev.metrics.MASE(),
    "WAPE": fev.metrics.WAPE(epsilon=1.0),
    "SQL": fev.metrics.SQL(),
}


# --------------------------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------------------------


def encoded_value(item_index: int, timestamp: pd.Timestamp) -> float:
    """Every value encodes where it comes from: 1000 * item + hours since START."""
    return 1000.0 * item_index + (timestamp - START) / STEP


def make_series(
    lengths: Sequence[int] = (60, 60),
    *,
    targets: Sequence[str] = (TARGET_COL,),
    end: pd.Timestamp = START + 59 * STEP,
) -> pd.DataFrame:
    """Hourly series frame ending at ``end``; one item per entry of ``lengths``.

    Targets and the past-only covariate ``past_cov`` encode their own timestamp, so a value
    seen by a forecaster tells exactly when it was observed. ``known_cov`` is a calendar-like
    known-future covariate.
    """
    frames = []
    for index, length in enumerate(lengths):
        timestamps = pd.date_range(end=end, periods=length, freq="h")
        codes = np.array([encoded_value(index, ts) for ts in timestamps])
        frame = pd.DataFrame({ID_COL: f"item_{index}", TIMESTAMP_COL: timestamps})
        for offset, target in enumerate(targets):
            frame[target] = codes + 0.25 * offset
        frame["past_cov"] = codes + 0.5
        frame["known_cov"] = timestamps.hour.to_numpy(dtype="float64")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def decode_timestamp(value: float) -> pd.Timestamp:
    return START + (np.floor(value) % 1000) * STEP


def with_random_walks(df: pd.DataFrame, columns: Sequence[str], *, seed: int) -> pd.DataFrame:
    """Replace ``columns`` with seeded random walks whose step size grows over time.

    Their seasonal differences change along the series, unlike the encoded values, so scaling
    MASE and SQL with the wrong history (future rows, or ignoring ``max_context``) changes
    the scores.
    """
    rng = np.random.default_rng(seed)
    amplitude = np.linspace(1.0, 10.0, len(df))
    return df.assign(
        **{
            column: 100.0 + np.cumsum(rng.normal(0.0, 1.0, len(df)) * amplitude)
            for column in columns
        }
    )


# --------------------------------------------------------------------------------------------
# Toy forecasters
# --------------------------------------------------------------------------------------------


def forecast_frame(
    context: pd.DataFrame,
    *,
    prediction_length: int,
    quantile_levels: Sequence[float],
    freq: str,
    target: str | Sequence[str],
    point: Callable[[pd.DataFrame, str], np.ndarray],
    offsets: Callable[[float], float],
) -> pd.DataFrame:
    """Forecast frame with ``point(series, target)`` as the median and fixed quantile offsets."""
    targets = [target] if isinstance(target, str) else list(target)
    frames = []
    for item_id, series in context.groupby(ID_COL, sort=True):
        last = series[TIMESTAMP_COL].iloc[-1]
        timestamps = pd.date_range(last, periods=prediction_length + 1, freq=freq)[1:]
        for name in targets:
            values = point(series, name)
            frame = pd.DataFrame(
                {
                    ID_COL: item_id,
                    TIMESTAMP_COL: timestamps,
                    TARGET_NAME_COL: name,
                    PREDICTION_COL: values,
                }
            )
            for level in quantile_levels:
                frame[quantile_col(level)] = values + offsets(level)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@dataclass
class LastValue:
    """Repeats the last observed value, with quantiles ``spread * (q - 0.5)`` around it."""

    name: str = "last_value"
    spread: float = 2.0
    calls: list[dict[str, object]] = field(default_factory=list)

    def predict(
        self,
        context: pd.DataFrame,
        *,
        prediction_length: int,
        quantile_levels: Sequence[float],
        freq: str,
        target: str | Sequence[str] = TARGET_COL,
        future_covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "context": context.copy(),
                "future_covariates": None
                if future_covariates is None
                else future_covariates.copy(),
                "target": target,
                "quantile_levels": list(quantile_levels),
            }
        )
        return forecast_frame(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            freq=freq,
            target=target,
            point=lambda series, name: np.full(prediction_length, series[name].dropna().iloc[-1]),
            offsets=lambda level: self.spread * (level - 0.5),
        )


@dataclass
class Fixed:
    """Ignores the data: the same point forecast and quantile values for every series."""

    point: Sequence[float]
    quantiles: dict[float, float]
    name: str = "fixed"

    def predict(self, context, *, prediction_length, quantile_levels, freq, target, **_):
        point = np.asarray(self.point, dtype="float64")
        return forecast_frame(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            freq=freq,
            target=target,
            point=lambda series, name: point,
            offsets=lambda level: self.quantiles[level] - point,
        )


@dataclass
class Noisy:
    """Deterministic but irregular forecasts (seeded by the context), to exercise metrics."""

    name: str = "noisy"

    def predict(self, context, *, prediction_length, quantile_levels, freq, target, **_):
        def point(series: pd.DataFrame, name: str) -> np.ndarray:
            rng = np.random.default_rng(len(series) + len(name))
            return series[name].dropna().iloc[-1] + rng.normal(0, 5, prediction_length)

        return forecast_frame(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            freq=freq,
            target=target,
            point=point,
            offsets=lambda level: 8.0 * scipy.stats.norm.ppf(level),
        )


@dataclass
class UsesEverything:
    """Output depends on every value it receives: any leaked value would change it."""

    name: str = "uses_everything"

    def predict(
        self, context, *, prediction_length, quantile_levels, freq, target, future_covariates=None
    ):
        def point(series: pd.DataFrame, name: str) -> np.ndarray:
            total = series.select_dtypes("number").to_numpy().sum()
            if future_covariates is not None:
                rows = future_covariates[future_covariates[ID_COL] == series[ID_COL].iloc[0]]
                total += rows.select_dtypes("number").to_numpy().sum()
            return total + np.arange(prediction_length, dtype="float64")

        return forecast_frame(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            freq=freq,
            target=target,
            point=point,
            offsets=lambda level: level - 0.5,
        )


@dataclass
class LeakageSpy:
    """Asserts on its own inputs that nothing from after the cutoff reaches it."""

    known: tuple[str, ...]
    prediction_length_seen: list[int] = field(default_factory=list)
    cutoffs_seen: list[pd.Timestamp] = field(default_factory=list)
    name: str = "spy"

    def predict(
        self, context, *, prediction_length, quantile_levels, freq, target, future_covariates=None
    ):
        cutoff = context[TIMESTAMP_COL].max()
        self.cutoffs_seen.append(cutoff)
        # Every target and past-covariate value decodes to a timestamp <= cutoff.
        for column in [TARGET_COL, "past_cov"]:
            observed_at = context[column].dropna().map(decode_timestamp)
            assert (observed_at <= cutoff).all(), f"{column} leaks values after {cutoff}"
            assert (observed_at == context[TIMESTAMP_COL]).all()
        # Future covariates: only the declared columns, only the next horizon steps.
        assert future_covariates is not None
        assert list(future_covariates.columns) == [ID_COL, TIMESTAMP_COL, *self.known]
        for _, rows in future_covariates.groupby(ID_COL):
            expected = pd.date_range(cutoff, periods=prediction_length + 1, freq=freq)[1:]
            assert np.array_equal(rows[TIMESTAMP_COL].to_numpy(), expected.to_numpy())
        return LastValue().predict(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            freq=freq,
            target=target,
        )


@dataclass
class Vandal:
    """Returns a valid forecast, then scribbles over the inputs it was given."""

    name: str = "vandal"

    def predict(self, context, *, future_covariates=None, **kwargs):
        forecast = LastValue().predict(context, **kwargs)
        context[TARGET_COL] = -1.0
        context.loc[context.index[:5], TIMESTAMP_COL] = pd.Timestamp("2100-01-01")
        if future_covariates is not None:
            future_covariates["known_cov"] = 1e9
        return forecast


@dataclass
class Broken:
    """Wraps ``LastValue`` and corrupts its output with ``mutate``."""

    mutate: Callable[[pd.DataFrame], object]
    name: str = "broken"

    def predict(self, context, **kwargs):
        return self.mutate(LastValue().predict(context, **kwargs))


def fev_direct(metric: fev.metrics.Metric, **arrays: np.ndarray) -> float:
    return metric.compute(**arrays, quantile_levels=list(FEV_BENCH_DECILES))


# --------------------------------------------------------------------------------------------
# BacktestConfig
# --------------------------------------------------------------------------------------------


def test_config_defaults_and_normalization() -> None:
    config = BacktestConfig(
        prediction_length=24,
        n_windows=3,
        freq="h",
        season_length=24,
        target=["a", "b"],  # type: ignore[arg-type]
        quantile_levels=[np.float64(q) for q in EVAL_QUANTILES],  # type: ignore[arg-type]
    )
    assert config.step is None
    assert config.stride == 24
    assert config.target == ("a", "b")
    assert config.targets == ("a", "b")
    assert config.quantile_columns == tuple(QUANTILE_COLUMNS)
    assert all(type(level) is float for level in config.quantile_levels)
    assert config.min_series_length == 24 + 1 + 24 + 2 * 24
    assert BacktestConfig(prediction_length=6, n_windows=1, freq="h", season_length=2).targets == (
        TARGET_COL,
    )
    assert SCORED_QUANTILES == FEV_BENCH_DECILES
    # Any superset of the scored deciles and interval bounds is accepted.
    BacktestConfig(
        prediction_length=6,
        n_windows=1,
        freq="h",
        season_length=2,
        quantile_levels=CHRONOS2_QUANTILES,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.n_windows = 5  # type: ignore[misc]


def test_default_step_follows_prediction_length_through_replace() -> None:
    config = BacktestConfig(prediction_length=24, n_windows=3, freq="h", season_length=24)
    longer = dataclasses.replace(config, prediction_length=48)
    assert longer.step is None
    assert longer.stride == 48
    assert longer == BacktestConfig(prediction_length=48, n_windows=3, freq="h", season_length=24)
    assert dataclasses.replace(config, step=12, prediction_length=48).stride == 12

    cutoffs = pd.Series([w.cutoff for w in rolling_origin_windows(make_series((200,)), longer)])
    assert (cutoffs.diff().dropna() == 48 * STEP).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prediction_length": 0}, "prediction_length"),
        ({"n_windows": -1}, "n_windows"),
        ({"season_length": 1.5}, "season_length"),
        ({"season_length": True}, "season_length"),
        ({"step": 0}, "step"),
        ({"freq": "H"}, "frequency"),
        ({"freq": ""}, "freq"),
        ({"quantile_levels": (0.01, 0.1, 0.9, 0.5, 0.99)}, "sorted"),
        ({"quantile_levels": (0.01, 0.1, 0.5, 0.5, 0.9, 0.99)}, "sorted"),
        ({"quantile_levels": (0.0, 0.01, 0.1, 0.5, 0.9, 0.99)}, r"\(0, 1\)"),
        ({"quantile_levels": (0.01, 0.1, 0.9, 0.99)}, r"missing \[0.2, 0.3, 0.4, 0.5, 0.6, 0.7"),
        (
            {"quantile_levels": (0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)},
            r"missing \[0.99\]",
        ),
        ({"quantile_levels": (0.1, 0.5, 0.9)}, r"missing \[0.01, 0.2, 0.3"),
        ({"quantile_levels": ()}, "empty"),
        ({"target": ()}, "target"),
        ({"target": ("a", "a")}, "duplicated"),
        ({"target": ACTUAL_COL}, r"cannot be named \['actual'\]"),
        ({"target": ("load", ID_COL)}, r"cannot be named \['item_id'\]"),
        ({"target": TIMESTAMP_COL}, r"cannot be named \['timestamp'\]"),
        ({"known_covariates": (TARGET_COL,)}, "cannot include"),
        ({"known_covariates": (TIMESTAMP_COL,)}, "cannot include"),
        ({"known_covariates": ("x", "x")}, "duplicated"),
        ({"max_context": 24}, "larger than season_length"),
        ({"max_context": 0}, "max_context"),
    ],
)
def test_config_rejects_invalid_settings(overrides: dict[str, object], message: str) -> None:
    settings = {"prediction_length": 24, "n_windows": 2, "freq": "h", "season_length": 24}
    with pytest.raises(ValueError, match=message):
        BacktestConfig(**(settings | overrides))  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("step", [None, 2, 10])
def test_window_boundaries(step: int | None) -> None:
    df = make_series((60, 45))
    config = BacktestConfig(prediction_length=6, n_windows=4, freq="h", season_length=4, step=step)
    windows = rolling_origin_windows(df, config)
    stride = config.stride

    assert [w.index for w in windows] == [0, 1, 2, 3]
    cutoffs = pd.DatetimeIndex([w.cutoff for w in windows])
    assert (cutoffs.to_series().diff().dropna() == stride * STEP).all()  # oldest -> newest
    last_timestamp = df[TIMESTAMP_COL].max()
    assert windows[-1].cutoff == last_timestamp - 6 * STEP
    assert windows[-1].actuals[TIMESTAMP_COL].max() == last_timestamp

    for window in windows:
        expected_context = df[df[TIMESTAMP_COL] <= window.cutoff]
        assert len(window.context) == len(expected_context)
        assert list(window.context.columns) == list(df.columns)
        assert window.future_covariates is None
        assert list(window.actuals.columns) == [ID_COL, TIMESTAMP_COL, TARGET_COL]
        horizon = pd.date_range(window.cutoff + STEP, periods=6, freq="h")
        for item_id, context in window.context.groupby(ID_COL):
            assert context[TIMESTAMP_COL].max() == window.cutoff
            actuals = window.actuals[window.actuals[ID_COL] == item_id]
            assert np.array_equal(actuals[TIMESTAMP_COL].to_numpy(), horizon.to_numpy())
            index = int(str(item_id).split("_")[1])
            expected = [encoded_value(index, ts) for ts in horizon]
            assert actuals[TARGET_COL].tolist() == expected


def test_windows_with_max_context_keep_the_most_recent_rows() -> None:
    df = make_series((80, 50))
    config = BacktestConfig(
        prediction_length=5, n_windows=3, freq="h", season_length=4, max_context=20
    )
    for window in rolling_origin_windows(df, config):
        for _, context in window.context.groupby(ID_COL):
            assert len(context) == 20
            assert context[TIMESTAMP_COL].max() == window.cutoff
            assert context[TIMESTAMP_COL].min() == window.cutoff - 19 * STEP


def test_future_covariates_contain_only_known_columns() -> None:
    df = make_series()
    config = BacktestConfig(
        prediction_length=6, n_windows=2, freq="h", season_length=4, known_covariates=("known_cov",)
    )
    for window in rolling_origin_windows(df, config):
        future = window.future_covariates
        assert future is not None
        assert list(future.columns) == [ID_COL, TIMESTAMP_COL, "known_cov"]
        assert TARGET_COL not in future and "past_cov" not in future
        assert future[TIMESTAMP_COL].min() == window.cutoff + STEP
        assert len(future) == 6 * 2
        assert future[[ID_COL, TIMESTAMP_COL]].equals(window.actuals[[ID_COL, TIMESTAMP_COL]])


def test_windows_leave_the_input_alone_and_carry_no_metadata() -> None:
    df = make_series()
    shuffled = df.sample(frac=1.0, random_state=0)
    shuffled.attrs = {"source": "synthetic"}
    before = shuffled.copy()
    config = BacktestConfig(
        prediction_length=6, n_windows=2, freq="h", season_length=4, known_covariates=("known_cov",)
    )
    windows = rolling_origin_windows(shuffled, config)
    pd.testing.assert_frame_equal(shuffled, before)  # not sorted or modified in place
    assert shuffled.attrs == {"source": "synthetic"}

    reference = rolling_origin_windows(df, config)  # sorting the input changes nothing
    for window, expected in zip(windows, reference, strict=True):
        pd.testing.assert_frame_equal(window.context, expected.context)
        pd.testing.assert_frame_equal(window.actuals, expected.actuals)
        for frame in (window.context, window.future_covariates, window.actuals):
            assert frame is not None
            assert frame.attrs == {}
            assert frame.index.equals(pd.RangeIndex(len(frame)))
    windows[0].context.loc[:, TARGET_COL] = np.nan
    pd.testing.assert_frame_equal(shuffled, before)
    pd.testing.assert_frame_equal(windows[1].context, reference[1].context)
    assert "rows" in repr(windows[0]) and isinstance(windows[0], Window)


def test_too_short_series_raises() -> None:
    df = make_series((60, 20))
    config = BacktestConfig(prediction_length=6, n_windows=3, freq="h", season_length=4)
    with pytest.raises(ValueError, match=r"too short.*item_1"):
        rolling_origin_windows(df, config)
    # 20 rows = 3 windows of 6 (18 rows) + 2 context rows; season_length=1 needs exactly 2.
    rolling_origin_windows(df, dataclasses.replace(config, season_length=1))


def test_series_starting_after_the_first_cutoff_raise() -> None:
    df = make_series((60, 10))
    config = BacktestConfig(prediction_length=6, n_windows=3, freq="h", season_length=2)
    with pytest.raises(ValueError, match=r"too short.*'item_1': '0'"):
        rolling_origin_windows(df, config)


def test_misaligned_series_raise() -> None:
    df = make_series((60, 60))
    df = df[~((df[ID_COL] == "item_1") & (df[TIMESTAMP_COL] == df[TIMESTAMP_COL].max()))]
    config = BacktestConfig(prediction_length=6, n_windows=2, freq="h", season_length=4)
    with pytest.raises(ValueError, match="same timestamp"):
        rolling_origin_windows(df, config)


@pytest.mark.parametrize(
    ("corrupt", "error", "message"),
    [
        (lambda df: df.drop(index=10), ValueError, "regular 'h' grid"),
        (
            lambda df: df.assign(
                **{
                    TIMESTAMP_COL: df[TIMESTAMP_COL].mask(
                        df.index == 10, df[TIMESTAMP_COL] + pd.Timedelta(minutes=30)
                    )
                }
            ),
            ValueError,
            "regular 'h' grid",
        ),
        (lambda df: pd.concat([df, df.iloc[[3]]]), ValueError, "duplicated"),
        (
            lambda df: df.assign(**{TIMESTAMP_COL: df[TIMESTAMP_COL].dt.tz_localize("UTC")}),
            ValueError,
            "timezone-naive",
        ),
        (
            lambda df: df.assign(**{TIMESTAMP_COL: df[TIMESTAMP_COL].astype(str)}),
            TypeError,
            "datetime64",
        ),
        (lambda df: df.drop(columns=TARGET_COL), ValueError, "missing columns"),
        (lambda df: df.iloc[:0], ValueError, "no rows"),
        (lambda df: df.assign(**{TARGET_COL: "x"}), TypeError, "numeric"),
        (lambda df: df.to_numpy(), TypeError, "DataFrame"),
    ],
)
def test_malformed_frames_raise(
    corrupt: Callable[[pd.DataFrame], pd.DataFrame], error: type[Exception], message: str
) -> None:
    config = BacktestConfig(prediction_length=6, n_windows=2, freq="h", season_length=4)
    with pytest.raises(error, match=message):
        rolling_origin_windows(corrupt(make_series()), config)


def test_missing_known_covariate_raises() -> None:
    config = BacktestConfig(
        prediction_length=6, n_windows=2, freq="h", season_length=4, known_covariates=("temp",)
    )
    with pytest.raises(ValueError, match=r"missing columns \['temp'\]"):
        rolling_origin_windows(make_series(), config)


# --------------------------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------------------------


def test_backtest_never_passes_future_values() -> None:
    df = make_series((60, 45))
    config = BacktestConfig(
        prediction_length=6,
        n_windows=4,
        freq="h",
        season_length=4,
        step=3,
        known_covariates=("known_cov",),
    )
    spy = LeakageSpy(known=config.known_covariates)
    forecasts = backtest(spy, df, config)  # the spy asserts inside predict()
    harness_cutoffs = forecasts.drop_duplicates(WINDOW_COL)[CUTOFF_COL].tolist()
    assert spy.cutoffs_seen == harness_cutoffs


def test_forecasts_do_not_depend_on_values_after_the_cutoff() -> None:
    """Replacing everything after a cutoff with noise must not change that window's forecast."""
    df = make_series((60, 45))
    config = BacktestConfig(
        prediction_length=6, n_windows=3, freq="h", season_length=4, known_covariates=("known_cov",)
    )
    model = UsesEverything()
    reference = backtest(model, df, config)
    rng = np.random.default_rng(0)
    for window in rolling_origin_windows(df, config):
        after_cutoff = df[TIMESTAMP_COL] > window.cutoff
        after_horizon = df[TIMESTAMP_COL] > window.actuals[TIMESTAMP_COL].max()
        poisoned = df.copy()
        poisoned.loc[after_cutoff, [TARGET_COL, "past_cov"]] = rng.normal(
            0, 1e6, (after_cutoff.sum(), 2)
        )
        poisoned.loc[after_horizon, "known_cov"] = rng.normal(0, 1e6, after_horizon.sum())
        rerun = backtest(model, poisoned, config)
        pick = [*FORECAST_COLUMNS, CUTOFF_COL]
        pd.testing.assert_frame_equal(
            rerun.loc[rerun[WINDOW_COL] == window.index, pick].reset_index(drop=True),
            reference.loc[reference[WINDOW_COL] == window.index, pick].reset_index(drop=True),
        )
    # Sanity check of the test itself: changing a value at the cutoff does change forecasts.
    changed = df.copy()
    last_cutoff = rolling_origin_windows(df, config)[-1].cutoff
    changed.loc[changed[TIMESTAMP_COL] == last_cutoff, TARGET_COL] += 1.0
    rerun = backtest(model, changed, config)
    assert not rerun[PREDICTION_COL].equals(reference[PREDICTION_COL])


def test_forecaster_mutating_its_inputs_cannot_corrupt_the_backtest() -> None:
    """Pins the contract: each window's frames are built fresh from ``df`` for one call.

    It would fail if a refactor reused a window's frames after ``predict`` (e.g. one context
    frame grown in place across windows), or wrote through to ``df``.
    """
    df = make_series()
    before = df.copy()
    config = BacktestConfig(
        prediction_length=6, n_windows=3, freq="h", season_length=4, known_covariates=("known_cov",)
    )
    vandal = backtest(Vandal(), df, config)
    honest = backtest(LastValue(name="vandal"), df, config)
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(
        vandal.drop(columns=SECONDS_COL), honest.drop(columns=SECONDS_COL)
    )


# --------------------------------------------------------------------------------------------
# backtest output
# --------------------------------------------------------------------------------------------


def test_backtest_output_schema_and_actuals(capsys: pytest.CaptureFixture[str]) -> None:
    df = make_series((60, 45))
    config = BacktestConfig(prediction_length=6, n_windows=3, freq="h", season_length=4)
    model = LastValue()
    forecasts = backtest(model, df, config, progress=True)
    assert "[last_value] window 3/3" in capsys.readouterr().out

    assert list(forecasts.columns) == [
        *FORECAST_COLUMNS,
        WINDOW_COL,
        CUTOFF_COL,
        MODEL_COL,
        ACTUAL_COL,
        SECONDS_COL,
    ]
    assert len(forecasts) == 3 * 2 * 6
    assert (forecasts[MODEL_COL] == "last_value").all()
    assert sorted(forecasts[WINDOW_COL].unique()) == [0, 1, 2]
    assert (forecasts[SECONDS_COL] >= 0).all()
    assert all(forecasts[column].dtype == "float64" for column in QUANTILE_COLUMNS)
    assert (forecasts[TIMESTAMP_COL] > forecasts[CUTOFF_COL]).all()
    item_index = forecasts[ID_COL].str.split("_").str[1].astype(int)
    expected_actual = [
        encoded_value(i, ts) for i, ts in zip(item_index, forecasts[TIMESTAMP_COL], strict=True)
    ]
    assert forecasts[ACTUAL_COL].tolist() == expected_actual
    assert model.calls[0]["target"] == TARGET_COL
    assert model.calls[0]["quantile_levels"] == list(EVAL_QUANTILES)
    assert all(call["future_covariates"] is None for call in model.calls)


def _shift_timestamps(forecast: pd.DataFrame) -> pd.DataFrame:
    return forecast.assign(**{TIMESTAMP_COL: forecast[TIMESTAMP_COL] + STEP})


def _set_first(column: str, value: float) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def mutate(forecast: pd.DataFrame) -> pd.DataFrame:
        forecast = forecast.copy()
        forecast.loc[0, column] = value
        return forecast

    return mutate


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (lambda f: f.drop(columns=quantile_col(0.99)), ValueError, r"missing columns \['0.99'\]"),
        (lambda f: f.drop(columns=PREDICTION_COL), ValueError, "missing columns"),
        (_set_first(quantile_col(0.1), np.nan), ValueError, "NaN or infinite.*'0.1': 1"),
        (_set_first(PREDICTION_COL, np.inf), ValueError, "NaN or infinite"),
        (_shift_timestamps, ValueError, "Missing 2 rows"),
        (lambda f: f.iloc[:-1], ValueError, "Missing 1 rows"),
        (lambda f: pd.concat([f, f.iloc[[0]]]), ValueError, "duplicated"),
        (lambda f: f.assign(**{TARGET_NAME_COL: "load"}), ValueError, "unexpected 12 rows"),
        (lambda f: f.assign(**{ID_COL: f[ID_COL] + "x"}), ValueError, "unexpected 12 rows"),
        (
            lambda f: f.assign(**{TIMESTAMP_COL: f[TIMESTAMP_COL].dt.tz_localize("UTC")}),
            ValueError,
            "timezone-naive",
        ),
        (lambda f: f.assign(**{quantile_col(0.5): "x"}), TypeError, "numeric"),
        (lambda f: f.to_numpy(), TypeError, "DataFrame"),
    ],
)
def test_backtest_rejects_invalid_forecaster_output(
    mutate: Callable[[pd.DataFrame], object], error: type[Exception], message: str
) -> None:
    config = BacktestConfig(prediction_length=6, n_windows=2, freq="h", season_length=4)
    with pytest.raises(error, match=message) as info:
        backtest(Broken(mutate), make_series(), config)
    if error is ValueError and "missing columns" not in message:
        assert "'broken' for window 0" in str(info.value)


def test_backtest_accepts_other_dtypes_and_drops_extra_columns() -> None:
    def loosen(forecast: pd.DataFrame) -> pd.DataFrame:
        return forecast.assign(
            **{
                ID_COL: forecast[ID_COL].astype(object),
                TIMESTAMP_COL: forecast[TIMESTAMP_COL].astype("datetime64[ns]"),
                "0.25": 0.0,
            }
        ).iloc[::-1]

    config = BacktestConfig(prediction_length=6, n_windows=2, freq="h", season_length=4)
    loose = backtest(Broken(loosen, name="last_value"), make_series(), config)
    strict = backtest(LastValue(), make_series(), config)
    columns = [c for c in strict.columns if c != SECONDS_COL]
    pd.testing.assert_frame_equal(loose[columns], strict[columns])


def test_backtest_rejects_non_forecasters() -> None:
    config = BacktestConfig(prediction_length=6, n_windows=2, freq="h", season_length=4)
    with pytest.raises(TypeError, match="Forecaster protocol"):
        backtest(object(), make_series(), config)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string"):
        backtest(LastValue(name=""), make_series(), config)
    assert isinstance(LastValue(), Forecaster)


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------

# Daily toy series: context [1, 3, 2, 4] (seasonal diffs at lag 2 are all 1), actuals [3, 5].
TOY_VALUES = [1.0, 3.0, 2.0, 4.0, 3.0, 5.0]
TOY_CONFIG = BacktestConfig(prediction_length=2, n_windows=1, freq="D", season_length=2)
# A hand-picked forecast distribution, constant over the horizon.
TOY_QUANTILES = {
    0.01: 0.0,
    0.1: 2.5,
    0.2: 3.0,
    0.3: 3.5,
    0.4: 3.8,
    0.5: 4.0,
    0.6: 4.2,
    0.7: 4.5,
    0.8: 4.6,
    0.9: 4.8,
    0.99: 6.0,
}


def toy_frame(values: Sequence[float] = TOY_VALUES) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame({ID_COL: "toy", TIMESTAMP_COL: timestamps, TARGET_COL: values})


def deciles(quantiles: dict[float, float]) -> dict[float, float]:
    """The levels fev-bench scores, out of a forecast distribution."""
    return {level: quantiles[level] for level in FEV_BENCH_DECILES}


def hand_wql(actuals: Sequence[float], quantiles: dict[float, float]) -> float:
    """WQL written out: mean over levels of mean quantile loss / max(mean |y|, 1)."""
    observed = [y for y in actuals if not np.isnan(y)]
    scale = max(sum(abs(y) for y in observed) / len(observed), 1.0)
    per_level = []
    for tau, q in quantiles.items():
        losses = [2 * abs((y - q) * ((1.0 if y <= q else 0.0) - tau)) for y in observed]
        per_level.append(sum(losses) / len(losses) / scale)
    return sum(per_level) / len(per_level)


def test_metrics_match_hand_computation_for_a_degenerate_forecast() -> None:
    # Every quantile equals 4: the loss is 2(1 - tau) for y=3 and 2 tau for y=5, mean 1 per level.
    model = Fixed(point=[4.0, 4.0], quantiles=dict.fromkeys(EVAL_QUANTILES, 4.0))
    df = toy_frame()
    metrics = evaluate(backtest(model, df, TOY_CONFIG), df, TOY_CONFIG)
    row = metrics.iloc[0]
    assert row[N_SERIES_COL] == 1
    assert row["WAPE"] == pytest.approx((1 + 1) / 2 / ((3 + 5) / 2))  # 0.25
    assert row["MASE"] == pytest.approx(1.0)  # mean |error| 1, seasonal scale 1
    assert row["WQL"] == pytest.approx(1.0 / 4.0)
    assert row["SQL"] == pytest.approx(1.0)
    assert row["coverage_80"] == 0.0
    assert row["coverage_98"] == 0.0


def test_metrics_match_hand_computation() -> None:
    model = Fixed(point=[4.0, 4.0], quantiles=TOY_QUANTILES)
    df = toy_frame()
    metrics = evaluate(backtest(model, df, TOY_CONFIG), df, TOY_CONFIG)
    row = metrics.iloc[0]
    wql = hand_wql([3.0, 5.0], deciles(TOY_QUANTILES))
    assert wql != pytest.approx(hand_wql([3.0, 5.0], TOY_QUANTILES))  # the tails would count
    assert row["WQL"] == pytest.approx(wql)
    assert row["SQL"] == pytest.approx(wql * 4.0)  # seasonal scale 1 instead of mean |y| = 4
    assert row["WAPE"] == pytest.approx(0.25)
    assert row["MASE"] == pytest.approx(1.0)
    assert row["coverage_80"] == 0.5  # [2.5, 4.8] holds 3 but not 5
    assert row["coverage_98"] == 1.0  # [0, 6] holds both


def test_nan_actuals_are_ignored() -> None:
    df = toy_frame([1.0, 3.0, 2.0, 4.0, 3.0, np.nan])
    model = Fixed(point=[4.0, 4.0], quantiles=TOY_QUANTILES)
    forecasts = backtest(model, df, TOY_CONFIG)
    assert forecasts[ACTUAL_COL].isna().sum() == 1
    row = evaluate(forecasts, df, TOY_CONFIG).iloc[0]
    assert row["WAPE"] == pytest.approx(1.0 / 3.0)
    assert row["MASE"] == pytest.approx(1.0)
    assert row["WQL"] == pytest.approx(hand_wql([3.0], deciles(TOY_QUANTILES)))
    assert row["coverage_80"] == 1.0
    assert row["coverage_98"] == 1.0


def test_a_window_without_observed_actuals_scores_nan() -> None:
    df = toy_frame([1.0, 3.0, 2.0, 4.0, np.nan, np.nan])
    model = Fixed(point=[4.0, 4.0], quantiles=TOY_QUANTILES)
    row = evaluate(backtest(model, df, TOY_CONFIG), df, TOY_CONFIG).iloc[0]
    assert row[[*METRICS, *COVERAGE_COLUMNS]].isna().all()


@pytest.mark.parametrize(
    ("values", "max_context", "scale"),
    [
        # Context [1, 3, 3, 7]: lag-2 differences 2 and 4, scale 3. Adding the horizon
        # [5, 13] to the history would give (2 + 4 + 2 + 6) / 4 = 3.5.
        ([1.0, 3.0, 3.0, 7.0, 5.0, 13.0], None, 3.0),
        # Full context [10, 0, 1, 3, 3, 7]: differences 9, 3, 2, 4, scale 4.5. Its last 4
        # rows are the context above, scale 3.
        ([10.0, 0.0, 1.0, 3.0, 3.0, 7.0, 5.0, 13.0], None, 4.5),
        ([10.0, 0.0, 1.0, 3.0, 3.0, 7.0, 5.0, 13.0], 4, 3.0),
    ],
)
def test_mase_and_sql_are_scaled_by_the_window_context_only(
    values: list[float], max_context: int | None, scale: float
) -> None:
    config = dataclasses.replace(TOY_CONFIG, max_context=max_context)
    df = toy_frame(values)
    model = Fixed(point=[4.0, 4.0], quantiles=dict.fromkeys(EVAL_QUANTILES, 4.0))
    row = evaluate(backtest(model, df, config), df, config).iloc[0]
    # Actuals [5, 13] vs a forecast of 4: mean absolute error 5. Every quantile at 4 gives
    # losses 2 tau and 18 tau, mean 10 tau, i.e. 5 averaged over the deciles.
    assert row["MASE"] == pytest.approx(5.0 / scale)
    assert row["SQL"] == pytest.approx(5.0 / scale)
    assert row["WAPE"] == pytest.approx(5.0 / 9.0)  # divided by mean |y| = 9
    assert row["WQL"] == pytest.approx(5.0 / 9.0)


def test_wql_and_sql_score_the_deciles_only() -> None:
    """The 0.01 and 0.99 tails feed coverage_98 but not WQL or SQL, as in fev-bench."""
    df = toy_frame()
    tight = dict.fromkeys(EVAL_QUANTILES, 4.0)
    wide_tails = tight | {0.01: -100.0, 0.99: 100.0}
    rows = [
        evaluate(backtest(Fixed(point=[4.0, 4.0], quantiles=q), df, TOY_CONFIG), df, TOY_CONFIG)
        for q in (tight, wide_tails)
    ]
    for metric in METRICS:
        assert rows[1][metric].iloc[0] == rows[0][metric].iloc[0]
    assert (rows[0]["coverage_98"].iloc[0], rows[1]["coverage_98"].iloc[0]) == (0.0, 1.0)


@pytest.mark.parametrize("max_context", [None, 12])
def test_evaluate_equals_calling_fev_directly(max_context: int | None) -> None:
    df = with_random_walks(make_series((70, 40, 55)), [TARGET_COL], seed=3)
    df.loc[df.index[::7], TARGET_COL] = np.nan  # NaN in contexts and actuals
    config = BacktestConfig(
        prediction_length=5,
        n_windows=4,
        freq="h",
        season_length=3,
        step=3,
        max_context=max_context,
    )
    forecasts = pd.concat(
        [backtest(Noisy(), df, config), backtest(LastValue(), df, config)], ignore_index=True
    )
    metrics = evaluate(forecasts, df, config)
    assert len(metrics) == 2 * 4

    for row in metrics.itertuples(index=False):
        rows = forecasts[
            (forecasts[MODEL_COL] == row.model) & (forecasts[WINDOW_COL] == row.window)
        ]
        y_true, y_pred, q_pred, past = [], [], [], []
        for item_id, item_rows in rows.groupby(ID_COL, sort=True):
            item_rows = item_rows.sort_values(TIMESTAMP_COL)
            series = df[df[ID_COL] == item_id]
            history = series.loc[series[TIMESTAMP_COL] <= row.cutoff, TARGET_COL].to_numpy()
            past.append(history if max_context is None else history[-max_context:])
            future = series[series[TIMESTAMP_COL] > row.cutoff].head(config.prediction_length)
            y_true.append(future[TARGET_COL].to_numpy())
            y_pred.append(item_rows[PREDICTION_COL].to_numpy())
            q_pred.append(item_rows[DECILE_COLUMNS].to_numpy())
        arrays = {
            "y_true": np.stack(y_true)[:, :, None],
            "y_pred": np.stack(y_pred)[:, :, None],
            "q_pred": np.stack(q_pred)[:, :, None, :],
            "y_past": np.concatenate(past)[:, None],
            "y_past_lengths": np.array([len(p) for p in past]),
            "seasonality": config.season_length,
        }
        for name, metric in FEV_BENCH_METRICS.items():
            assert getattr(row, name) == pytest.approx(fev_direct(metric, **arrays), rel=1e-12)
        assert row.n_series == 3


def test_multi_target_pairs_are_separate_fev_series() -> None:
    df = with_random_walks(
        make_series((50, 50), targets=("load", "price")), ["load", "price"], seed=4
    )
    df["price"] = df["price"] / 100.0 + 20.0  # a different scale from load
    config = BacktestConfig(
        prediction_length=4, n_windows=2, freq="h", season_length=3, target=("load", "price")
    )
    model = Noisy()
    forecasts = backtest(model, df, config)
    assert sorted(forecasts[TARGET_NAME_COL].unique()) == ["load", "price"]
    assert len(forecasts) == 2 * 2 * 2 * 4  # windows x items x targets x steps
    assert set(forecasts.groupby([ID_COL, TARGET_NAME_COL])[ACTUAL_COL].first().index) == {
        ("item_0", "load"),
        ("item_0", "price"),
        ("item_1", "load"),
        ("item_1", "price"),
    }
    metrics = evaluate(forecasts, df, config)
    assert (metrics[N_SERIES_COL] == 4).all()

    for row in metrics.itertuples(index=False):
        rows = forecasts[forecasts[WINDOW_COL] == row.window]
        y_true, y_pred, q_pred, past = [], [], [], []
        for (item_id, target), pair in rows.groupby([ID_COL, TARGET_NAME_COL], sort=True):
            series = df[df[ID_COL] == item_id]
            past.append(series.loc[series[TIMESTAMP_COL] <= row.cutoff, target].to_numpy())
            y_true.append(pair[ACTUAL_COL].to_numpy())
            y_pred.append(pair[PREDICTION_COL].to_numpy())
            q_pred.append(pair[DECILE_COLUMNS].to_numpy())
        arrays = {
            "y_true": np.stack(y_true)[:, :, None],
            "y_pred": np.stack(y_pred)[:, :, None],
            "q_pred": np.stack(q_pred)[:, :, None, :],
            "y_past": np.concatenate(past)[:, None],
            "y_past_lengths": np.array([len(p) for p in past]),
            "seasonality": 3,
        }
        for name, metric in FEV_BENCH_METRICS.items():
            assert getattr(row, name) == pytest.approx(fev_direct(metric, **arrays), rel=1e-12)


def zero_price(forecast: pd.DataFrame) -> pd.DataFrame:
    """Forecast 0 for every price step and quantile: a 100% error on price."""
    forecast = forecast.copy()
    forecast.loc[forecast[TARGET_NAME_COL] == "price", [PREDICTION_COL, *QUANTILE_COLUMNS]] = 0.0
    return forecast


def test_by_target_scores_each_target_separately() -> None:
    targets = ("load", "price")
    df = make_series((50, 50), targets=targets)
    df["load"] = df["load"] + 28_000.0  # large values, like electricity load in MW
    df["price"] = df["price"] / 100.0 + 20.0  # small values, like a price in EUR/MWh
    config = BacktestConfig(
        prediction_length=4, n_windows=2, freq="h", season_length=3, target=targets
    )
    forecasts = backtest(Broken(zero_price, name="m"), df, config)
    pooled = evaluate(forecasts, df, config)
    per_target = evaluate(forecasts, df, config, by_target=True)

    assert list(per_target.columns) == [
        MODEL_COL,
        WINDOW_COL,
        CUTOFF_COL,
        TARGET_NAME_COL,
        N_SERIES_COL,
        *METRICS,
        *COVERAGE_COLUMNS,
        SECONDS_COL,
    ]
    assert per_target[TARGET_NAME_COL].tolist() == ["load", "price"] * 2
    assert (per_target[N_SERIES_COL] == 2).all()
    # Every price quantile is 0 below positive actuals: the loss is 2 tau y, so WQL = 2 mean(tau).
    price = per_target[per_target[TARGET_NAME_COL] == "price"]
    assert price["WQL"].to_numpy() == pytest.approx(1.0)
    assert price["WAPE"].to_numpy() == pytest.approx(1.0)
    assert (price["coverage_80"] == 0.0).all()
    # Pooled, the large load values hide the price failure.
    assert (pooled["WQL"] < 0.01).all() and (pooled["WAPE"] < 0.01).all()

    # A target's rows equal scoring that target alone ...
    for target, rows in per_target.groupby(TARGET_NAME_COL):
        alone = evaluate(
            forecasts[forecasts[TARGET_NAME_COL] == target],
            df,
            dataclasses.replace(config, target=target),
        )
        scored = rows.drop(columns=TARGET_NAME_COL).reset_index(drop=True)
        pd.testing.assert_frame_equal(scored, alone)
    # ... and their mean over targets equals fev's multivariate scoring (D = 2).
    for row in pooled.itertuples(index=False):
        rows = forecasts[forecasts[WINDOW_COL] == row.window].sort_values([ID_COL, TIMESTAMP_COL])
        dims = [rows[rows[TARGET_NAME_COL] == target] for target in targets]
        past = df[df[TIMESTAMP_COL] <= row.cutoff].sort_values([ID_COL, TIMESTAMP_COL])
        arrays = {
            "y_true": np.stack([d[ACTUAL_COL].to_numpy().reshape(2, 4) for d in dims], axis=-1),
            "y_pred": np.stack([d[PREDICTION_COL].to_numpy().reshape(2, 4) for d in dims], axis=-1),
            "q_pred": np.stack(
                [d[DECILE_COLUMNS].to_numpy().reshape(2, 4, len(DECILE_COLUMNS)) for d in dims],
                axis=2,
            ),
            "y_past": past[list(targets)].to_numpy(),
            "y_past_lengths": past.groupby(ID_COL).size().to_numpy(),
            "seasonality": 3,
        }
        window_rows = per_target[per_target[WINDOW_COL] == row.window]
        for name, metric in FEV_BENCH_METRICS.items():
            expected = fev_direct(metric, **arrays)
            assert window_rows[name].mean() == pytest.approx(expected, rel=1e-12)

    summary = summarize(per_target)
    pairs = summary[[MODEL_COL, TARGET_NAME_COL]].to_numpy().tolist()
    assert pairs == [["m", "load"], ["m", "price"]]
    with pytest.raises(ValueError, match="several targets"):
        skill_score(per_target, model="m", baseline="m")
    assert skill_score(price, model="m", baseline="m").skill == pytest.approx(0.0)


def test_multi_target_forecaster_receives_a_list_of_targets() -> None:
    df = make_series(targets=("load", "price"))
    config = BacktestConfig(
        prediction_length=4, n_windows=1, freq="h", season_length=3, target=("load", "price")
    )
    model = LastValue()
    backtest(model, df, config)
    assert model.calls[0]["target"] == ["load", "price"]


def test_evaluate_keeps_seconds_and_rejects_mismatched_inputs() -> None:
    df = make_series()
    config = BacktestConfig(prediction_length=6, n_windows=2, freq="h", season_length=4)
    forecasts = backtest(LastValue(), df, config)
    metrics = evaluate(forecasts, df, config)
    assert list(metrics.columns) == [
        MODEL_COL,
        WINDOW_COL,
        CUTOFF_COL,
        N_SERIES_COL,
        "WQL",
        "MASE",
        "WAPE",
        "SQL",
        "coverage_80",
        "coverage_98",
        SECONDS_COL,
    ]
    assert (
        metrics[SECONDS_COL].tolist() == forecasts.groupby(WINDOW_COL)[SECONDS_COL].first().tolist()
    )

    edited = df.copy()
    edited.loc[edited.index[-1], TARGET_COL] += 1.0
    with pytest.raises(ValueError, match="differs from the values in df"):
        evaluate(forecasts, edited, config)
    # At realistic scale a small edit must still be caught (no relative tolerance).
    large = df.assign(**{TARGET_COL: df[TARGET_COL] + 28_000.0})
    nudged = large.copy()
    nudged.loc[nudged.index[-1], TARGET_COL] += 0.25
    with pytest.raises(ValueError, match="differs from the values in df"):
        evaluate(backtest(LastValue(), large, config), nudged, config)
    with pytest.raises(ValueError, match="cutoff"):
        evaluate(forecasts, df, dataclasses.replace(config, n_windows=3))
    with pytest.raises(ValueError, match="does not exist"):
        last_window = forecasts[forecasts[WINDOW_COL] == 1]
        evaluate(last_window, df, dataclasses.replace(config, n_windows=1))
    with pytest.raises(ValueError, match="missing columns"):
        evaluate(forecasts.drop(columns=quantile_col(0.5)), df, config)
    # Forecasts without the 'actual' column are scored from df alone.
    without_actual = evaluate(forecasts.drop(columns=ACTUAL_COL), df, config)
    pd.testing.assert_frame_equal(without_actual, metrics)


def test_summarize_means_and_stds_per_model() -> None:
    metrics = pd.DataFrame(
        {
            MODEL_COL: ["b", "b", "a", "a"],
            WINDOW_COL: [0, 1, 0, 1],
            "WQL": [1.0, 3.0, 0.5, 0.5],
            "MASE": [2.0, 2.0, np.nan, 1.0],
        }
    )
    summary = summarize(metrics)
    assert summary[MODEL_COL].tolist() == ["a", "b"]  # best WQL first
    assert list(summary.columns) == [MODEL_COL, "n_windows", "WQL", "WQL_std", "MASE", "MASE_std"]
    b = summary.set_index(MODEL_COL).loc["b"]
    assert b["WQL"] == 2.0
    assert b["WQL_std"] == pytest.approx(np.sqrt(2.0))
    assert b["n_windows"] == 2
    assert np.isnan(summary.set_index(MODEL_COL).loc["a", "MASE"])  # NaN is not skipped


# --------------------------------------------------------------------------------------------
# Skill scores
# --------------------------------------------------------------------------------------------


def metrics_frame(**errors: Sequence[float]) -> pd.DataFrame:
    frames = [
        pd.DataFrame({MODEL_COL: model, WINDOW_COL: range(len(values)), "WQL": values})
        for model, values in errors.items()
    ]
    return pd.concat(frames, ignore_index=True)


def test_identical_errors_give_zero_skill() -> None:
    errors = [0.2, 0.5, 0.1, 0.3]
    score = skill_score(metrics_frame(m=errors, seasonal_naive=errors), model="m")
    assert score.skill == pytest.approx(0.0)
    assert score.ci_low == pytest.approx(0.0) and score.ci_high == pytest.approx(0.0)
    assert score.win_rate == 0.5
    assert score.n_units == 4
    assert (score.model, score.baseline, score.metric) == ("m", "seasonal_naive", "WQL")


def test_halved_errors_give_half_skill() -> None:
    baseline = np.array([0.2, 0.5, 0.1, 0.3])
    score = skill_score(metrics_frame(m=baseline / 2, seasonal_naive=baseline), model="m")
    assert score.skill == pytest.approx(0.5)
    assert score.ci_low == pytest.approx(0.5) and score.ci_high == pytest.approx(0.5)
    assert score.win_rate == 1.0


def test_relative_errors_are_clipped() -> None:
    metrics = metrics_frame(perfect=[0.0, 0.0], awful=[1e6, 1e6], seasonal_naive=[1.0, 2.0])
    assert skill_score(metrics, model="perfect").skill == pytest.approx(1 - 0.01)
    assert skill_score(metrics, model="awful").skill == pytest.approx(1 - 100.0)
    assert skill_score(metrics, model="awful", clip=(0.1, 10.0)).skill == pytest.approx(-9.0)
    with pytest.raises(ValueError, match="clip"):
        skill_score(metrics, model="awful", clip=(0.0, 10.0))


def test_skill_matches_gmean_and_ci_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    baseline = rng.uniform(0.5, 1.5, 30)
    model = baseline * rng.uniform(0.4, 1.2, 30)
    metrics = metrics_frame(m=model, seasonal_naive=baseline)
    score = skill_score(metrics, model="m", seed=7)

    ratios = np.clip(model / baseline, 0.01, 100.0)
    assert score.skill == pytest.approx(1 - scipy.stats.gmean(ratios))
    assert score.ci_low < score.skill < score.ci_high
    assert score.win_rate == pytest.approx(np.mean(model < baseline))
    assert skill_score(metrics, model="m", seed=7) == score
    assert skill_score(metrics, model="m", seed=8) != score


def test_block_bootstrap_keeps_the_estimate_and_widens_correlated_intervals() -> None:
    rng = np.random.default_rng(3)
    # A slow wave under the ratios: neighbouring windows share most of their error.
    wave = np.sin(np.arange(280) / 9.0)
    baseline = np.ones(280)
    model = np.exp(-0.3 + 0.5 * wave + 0.05 * rng.normal(size=280))
    metrics = metrics_frame(m=model, seasonal_naive=baseline)
    independent = skill_score(metrics, model="m")
    blocks = skill_score(metrics, model="m", block_length=14)

    assert blocks.skill == pytest.approx(independent.skill)
    assert blocks.win_rate == independent.win_rate
    assert (independent.block_length, blocks.block_length) == (1, 14)
    assert blocks.ci_low < blocks.skill < blocks.ci_high
    assert blocks.ci_high - blocks.ci_low > 1.5 * (independent.ci_high - independent.ci_low)
    assert skill_score(metrics, model="m", block_length=14) == blocks


def test_block_bootstrap_agrees_with_the_independent_one_on_uncorrelated_windows() -> None:
    rng = np.random.default_rng(5)
    baseline = rng.uniform(0.5, 1.5, 400)
    model = baseline * rng.uniform(0.6, 1.2, 400)
    metrics = metrics_frame(m=model, seasonal_naive=baseline)
    independent = skill_score(metrics, model="m", n_resamples=4000)
    blocks = skill_score(metrics, model="m", n_resamples=4000, block_length=7)
    assert blocks.ci_high - blocks.ci_low == pytest.approx(
        independent.ci_high - independent.ci_low, rel=0.2
    )


def test_block_length_is_validated() -> None:
    metrics = metrics_frame(m=[1.0, 2.0, 3.0], seasonal_naive=[2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="block_length"):
        skill_score(metrics, model="m", block_length=0)
    with pytest.raises(ValueError, match="exceeds the 3 windows"):
        skill_score(metrics, model="m", block_length=4)
    table = skill_table(metrics, block_length=3)
    assert table["block_length"].tolist() == [3, 3]


def test_win_rate_counts_ties_as_half() -> None:
    metrics = metrics_frame(m=[1.0, 2.0, 3.0, 4.0], seasonal_naive=[1.0, 3.0, 2.0, 5.0])
    assert skill_score(metrics, model="m").win_rate == pytest.approx((2 + 0.5) / 4)


def test_skill_score_rejects_unpaired_or_undefined_errors() -> None:
    with pytest.raises(ValueError, match=r"no rows for \['seasonal_naive'\]"):
        skill_score(metrics_frame(m=[1.0]), model="m")
    with pytest.raises(ValueError, match="same windows"):
        skill_score(metrics_frame(m=[1.0, 2.0], seasonal_naive=[1.0]), model="m")
    with pytest.raises(ValueError, match="NaN"):
        skill_score(metrics_frame(m=[1.0, np.nan], seasonal_naive=[1.0, 1.0]), model="m")
    with pytest.raises(ValueError, match="0/0"):
        skill_score(metrics_frame(m=[0.0, 1.0], seasonal_naive=[0.0, 1.0]), model="m")


def test_skill_table_on_a_real_backtest() -> None:
    df = make_series((80, 80))
    config = BacktestConfig(prediction_length=6, n_windows=5, freq="h", season_length=4)
    forecasts = pd.concat(
        [
            backtest(LastValue(name="seasonal_naive", spread=50.0), df, config),
            backtest(LastValue(name="sharp", spread=2.0), df, config),
            backtest(Noisy(), df, config),
        ],
        ignore_index=True,
    )
    metrics = evaluate(forecasts, df, config)
    table = skill_table(metrics, baseline="seasonal_naive")
    assert list(table.columns) == [
        "model",
        "baseline",
        "metric",
        "skill",
        "ci_low",
        "ci_high",
        "win_rate",
        "n_units",
        "block_length",
    ]
    assert table["skill"].is_monotonic_decreasing
    assert set(table["model"]) == {"seasonal_naive", "sharp", "noisy"}
    baseline_row = table.set_index("model").loc["seasonal_naive"]
    assert baseline_row["skill"] == pytest.approx(0.0)
    assert (table["n_units"] == 5).all()
    assert (table["ci_low"] <= table["skill"] + 1e-12).all()
    assert (table["skill"] <= table["ci_high"] + 1e-12).all()
    pd.testing.assert_frame_equal(table, skill_table(metrics, baseline="seasonal_naive"))


# --------------------------------------------------------------------------------------------
# Diagnostics: reliability by level, error by step
# --------------------------------------------------------------------------------------------


def scored_frame(
    actual: np.ndarray, quantiles: dict[float, np.ndarray], model: str
) -> pd.DataFrame:
    """A backtest output of one window and one series with the given values."""
    steps = len(actual)
    frame = pd.DataFrame(
        {
            MODEL_COL: model,
            WINDOW_COL: 0,
            ID_COL: "a",
            TARGET_NAME_COL: "y",
            TIMESTAMP_COL: pd.date_range("2024-01-01", periods=steps, freq="h"),
            ACTUAL_COL: actual,
            PREDICTION_COL: quantiles[0.5],
        }
    )
    for level, values in quantiles.items():
        frame[quantile_col(level)] = values
    return frame


def test_reliability_table_counts_actuals_at_or_below_each_quantile() -> None:
    actual = np.arange(1.0, 11.0)  # 1..10
    quantiles = {level: np.full(10, 10 * level) for level in SCORED_QUANTILES}  # 1, 2, ..., 9
    table = reliability_table(scored_frame(actual, quantiles, "m"))
    assert table["level"].tolist() == list(SCORED_QUANTILES)
    assert table["observed"].tolist() == pytest.approx(list(SCORED_QUANTILES))
    assert (table["n"] == 10).all()


def test_reliability_table_shows_narrow_intervals_and_skips_missing_actuals() -> None:
    actual = np.array([-2.0, -1.0, 1.0, 2.0, np.nan])
    quantiles = {level: np.full(5, level - 0.5) for level in SCORED_QUANTILES}  # all within +-0.4
    table = reliability_table(scored_frame(actual, quantiles, "narrow")).set_index("level")
    assert (table["n"] == 4).all()
    assert table.loc[0.1, "observed"] == 0.5  # half the actuals below the 0.1 quantile
    assert table.loc[0.9, "observed"] == 0.5  # and only half below the 0.9 quantile


def test_error_by_step_matches_the_definitions() -> None:
    actual = np.array([10.0, 20.0, np.nan])
    quantiles = {level: np.array([12.0, 20.0, 5.0]) for level in SCORED_QUANTILES}
    frames = [scored_frame(actual, quantiles, "m"), scored_frame(actual, quantiles, "m")]
    frames[1][WINDOW_COL] = 1
    table = error_by_step(pd.concat(frames)).set_index("step")  # duplicated index on purpose
    assert table.index.tolist() == [1, 2]  # step 3 has no actual
    assert (table["n"] == 2).all()
    assert table.loc[1, "WAPE"] == pytest.approx(2.0 / 10.0)
    assert table.loc[2, "WAPE"] == 0.0
    # A point forecast repeated at every level: pinball loss averages to |error| / 2.
    assert table.loc[1, "WQL"] == pytest.approx(2.0 / 10.0)
    assert table.loc[2, "WQL"] == 0.0


def test_error_by_horizon_block_pools_steps_within_each_window() -> None:
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    quantiles = {level: np.array([12.0, 20.0, 30.0, 44.0]) for level in SCORED_QUANTILES}
    frames = [scored_frame(actual, quantiles, "m"), scored_frame(actual * 2, quantiles, "m")]
    frames[1][WINDOW_COL] = 1
    table = error_by_horizon_block(pd.concat(frames), edges=(2, 4))
    assert table.columns.tolist() == [MODEL_COL, WINDOW_COL, "block", "WAPE", "WQL", "n"]
    assert table["block"].astype(str).tolist() == ["1-2", "3-4", "1-2", "3-4"]
    assert table[WINDOW_COL].tolist() == [0, 0, 1, 1]
    first = table.set_index([WINDOW_COL, "block"])
    assert first.loc[(0, "1-2"), "WAPE"] == pytest.approx(2.0 / 30.0)
    assert first.loc[(0, "3-4"), "WAPE"] == pytest.approx(4.0 / 70.0)
    assert first.loc[(0, "3-4"), "WQL"] == pytest.approx(4.0 / 70.0)  # a point forecast
    assert (table["n"] == 2).all()
    # Steps past the last edge are left out, and the edges must make sense.
    assert error_by_horizon_block(pd.concat(frames), edges=(3,))["n"].tolist() == [3, 3]
    with pytest.raises(ValueError, match="edges must be increasing"):
        error_by_horizon_block(pd.concat(frames), edges=(2, 2))


def test_error_by_series_gives_one_row_per_model_and_series() -> None:
    actual = np.array([10.0, 20.0, np.nan])
    quantiles = {level: np.array([12.0, 20.0, 5.0]) for level in SCORED_QUANTILES}
    first = scored_frame(actual, quantiles, "m")
    second = scored_frame(actual * 2, quantiles, "m").assign(**{ID_COL: "b"})
    table = error_by_series(pd.concat([first, second])).set_index(ID_COL)
    assert table.index.tolist() == ["a", "b"]
    assert (table["n"] == 2).all()
    assert table.loc["a", "WAPE"] == pytest.approx(2.0 / 30.0)
    assert table.loc["b", "WAPE"] == pytest.approx((8.0 + 20.0) / 60.0)
    assert table.loc["a", "WQL"] == pytest.approx(2.0 / 30.0)  # a point forecast: WQL = WAPE


def test_skill_can_resample_series_instead_of_windows() -> None:
    rows = [
        {MODEL_COL: model, ID_COL: series, "WQL": error}
        for model, errors in (("base", (1.0, 2.0, 4.0)), ("half", (0.5, 1.0, 2.0)))
        for series, error in zip("abc", errors, strict=True)
    ]
    table = skill_table(pd.DataFrame(rows), baseline="base", unit=ID_COL).set_index(MODEL_COL)
    assert table.loc["half", "skill"] == pytest.approx(0.5)
    assert table.loc["half", "n_units"] == 3
    with pytest.raises(ValueError, match="window"):
        skill_table(pd.DataFrame(rows), baseline="base")  # no window column


def test_diagnostics_run_on_a_real_backtest() -> None:
    df = make_series((80, 80))
    config = BacktestConfig(prediction_length=6, n_windows=5, freq="h", season_length=4)
    forecasts = backtest(LastValue(name="wide", spread=50.0), df, config)
    reliability = reliability_table(forecasts)
    assert len(reliability) == len(SCORED_QUANTILES)
    assert reliability["observed"].is_monotonic_increasing
    steps = error_by_step(forecasts)
    assert steps["step"].tolist() == [1, 2, 3, 4, 5, 6]
    assert (steps["n"] == 10).all()  # 5 windows x 2 series


# --------------------------------------------------------------------------------------------
# Several runs of one configuration
# --------------------------------------------------------------------------------------------


def test_average_runs_averages_each_window_over_the_runs() -> None:
    runs = pd.concat(
        [
            pd.DataFrame(
                {MODEL_COL: "seed_1", WINDOW_COL: [0, 1], "WQL": [0.2, 0.4], "seconds": 1.0}
            ),
            pd.DataFrame(
                {MODEL_COL: "seed_2", WINDOW_COL: [1, 0], "WQL": [0.8, 0.4], "seconds": 3.0}
            ),
        ],
        ignore_index=True,
    )
    runs[CUTOFF_COL] = pd.Timestamp("2024-01-01") + pd.to_timedelta(runs[WINDOW_COL], unit="D")
    averaged = average_runs(runs, "lora")
    assert averaged[MODEL_COL].tolist() == ["lora", "lora"]
    assert averaged[WINDOW_COL].tolist() == [0, 1]
    assert averaged["WQL"].tolist() == pytest.approx([0.3, 0.6])
    assert averaged["seconds"].tolist() == [2.0, 2.0]
    assert averaged[CUTOFF_COL].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    # The averaged run goes through the skill table like any model.
    baseline = pd.DataFrame({MODEL_COL: "seasonal_naive", WINDOW_COL: [0, 1], "WQL": [0.6, 1.2]})
    table = skill_table(pd.concat([averaged, baseline], ignore_index=True)).set_index(MODEL_COL)
    assert table.loc["lora", "skill"] == pytest.approx(0.5)


def test_average_runs_rejects_runs_on_different_windows() -> None:
    runs = pd.concat(
        [
            pd.DataFrame({MODEL_COL: "seed_1", WINDOW_COL: [0, 1], "WQL": [0.2, 0.4]}),
            pd.DataFrame({MODEL_COL: "seed_2", WINDOW_COL: [0, 2], "WQL": [0.2, 0.4]}),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="same windows"):
        average_runs(runs, "lora")
    with pytest.raises(ValueError, match="is empty"):
        average_runs(runs.iloc[:0], "lora")


def test_summarize_runs_scores_the_average_and_reports_the_spread() -> None:
    baseline = pd.DataFrame({MODEL_COL: "zero-shot", WINDOW_COL: range(4), "WQL": 1.0})
    seed_1 = pd.DataFrame({MODEL_COL: "a_s1", WINDOW_COL: range(4), "WQL": [0.5, 0.5, 0.5, 0.5]})
    seed_2 = pd.DataFrame({MODEL_COL: "a_s2", WINDOW_COL: range(4), "WQL": [1.0, 1.0, 1.0, 1.0]})
    other = pd.DataFrame({MODEL_COL: "b_s1", WINDOW_COL: range(4), "WQL": [2.0, 2.0, 2.0, 2.0]})
    metrics = pd.concat([baseline, seed_1, seed_2, other], ignore_index=True)
    metrics["coverage_80"] = 0.8
    summary, averaged = summarize_runs(metrics, {"a": ["a_s1", "a_s2"], "b": ["b_s1"]})
    summary = summary.set_index(MODEL_COL)
    assert summary.loc["a", "n_runs"] == 2
    assert summary.loc["a", "WQL"] == pytest.approx(0.75)
    assert summary.loc["a", "mean_change"] == pytest.approx(-0.25)
    assert summary.loc["a", "skill"] == pytest.approx(0.25)
    assert summary.loc["a", "run_skill_min"] == pytest.approx(0.0)
    assert summary.loc["a", "run_skill_max"] == pytest.approx(0.5)
    assert summary.loc["b", "skill"] == pytest.approx(-1.0)
    assert summary.loc["b", "coverage_80"] == pytest.approx(0.8)
    assert sorted(averaged[MODEL_COL].unique()) == ["a", "b", "zero-shot"]
    assert skill_table(averaged, baseline="zero-shot")["model"].tolist() == ["a", "zero-shot", "b"]


def test_summarize_runs_validates_groups() -> None:
    metrics = pd.DataFrame({MODEL_COL: ["zero-shot", "a"], WINDOW_COL: [0, 0], "WQL": [1.0, 0.5]})
    with pytest.raises(ValueError, match="groups is empty"):
        summarize_runs(metrics, {})
    with pytest.raises(ValueError, match="no rows for"):
        summarize_runs(metrics, {"g": ["missing"]})
    with pytest.raises(ValueError, match="like the baseline"):
        summarize_runs(metrics, {"zero-shot": ["a"]})
