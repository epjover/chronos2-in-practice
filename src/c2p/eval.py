"""Rolling-origin backtests that cannot leak the future, scored with fev metrics.

A backtest replays history: pick a cutoff, give the model everything up to it, ask for the
next ``prediction_length`` steps, compare with what really happened, move the cutoff forward
and repeat. Three rules keep it honest, and this module enforces them by construction:

1. The forecaster only receives rows with ``timestamp <= cutoff`` (the *context*).
2. For the horizon it only receives ``item_id``, ``timestamp`` and the columns declared as
   ``known_covariates``; targets and past-only covariates never cross the cutoff.
3. Actuals are joined to the forecasts *after* the forecaster has answered.

Metrics come from ``fev`` (the library behind fev-bench, used in the Chronos-2 paper) with
fev-bench's settings, so every window is scored the way fev-bench scores a task. Skill scores
here are computed over the windows of one dataset, not over fev-bench's tasks, so they are not
the paper's leaderboard numbers. MASE and SQL are scaled with each window's own context, never
with data after the cutoff.

Example::

    config = BacktestConfig(prediction_length=24, n_windows=20, freq="h", season_length=24)
    forecasts = pd.concat([backtest(model, df, config) for model in (chronos, naive)])
    metrics = evaluate(forecasts, df, config)  # one row per (model, window)
    summarize(metrics)
    skill_table(metrics, baseline="seasonal_naive")
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats
from fev.analysis import bootstrap
from fev.metrics import MASE, SQL, WAPE, WQL, Metric
from pandas.tseries.frequencies import to_offset

from c2p.config import SEED
from c2p.schema import (
    EVAL_QUANTILES,
    ID_COL,
    INTERVALS,
    PREDICTION_COL,
    TARGET_COL,
    TARGET_NAME_COL,
    TIMESTAMP_COL,
    Forecaster,
    quantile_col,
)

__all__ = [
    "ACTUAL_COL",
    "COVERAGE_COLUMNS",
    "CUTOFF_COL",
    "METRICS",
    "MODEL_COL",
    "N_SERIES_COL",
    "SCORED_QUANTILES",
    "SECONDS_COL",
    "WINDOW_COL",
    "BacktestConfig",
    "SkillScore",
    "Window",
    "average_runs",
    "backtest",
    "coverage_column",
    "error_by_horizon_block",
    "error_by_series",
    "error_by_step",
    "evaluate",
    "iter_windows",
    "reliability_table",
    "rolling_origin_windows",
    "skill_score",
    "skill_table",
    "summarize",
    "summarize_runs",
]

logger = logging.getLogger(__name__)

WINDOW_COL = "window"
CUTOFF_COL = "cutoff"
MODEL_COL = "model"
ACTUAL_COL = "actual"
SECONDS_COL = "seconds"
N_SERIES_COL = "n_series"

# fev-bench settings: epsilon=1.0 keeps WQL and WAPE finite for series close to zero.
METRICS: Mapping[str, Metric] = {
    "WQL": WQL(epsilon=1.0),
    "MASE": MASE(),
    "WAPE": WAPE(epsilon=1.0),
    "SQL": SQL(),
}
# fev-bench scores WQL and SQL on the nine deciles. Other requested levels (the 0.01 and 0.99
# tails of EVAL_QUANTILES) are only used for coverage.
SCORED_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_SCORED_COLUMNS = [quantile_col(level) for level in SCORED_QUANTILES]

# Row order of a scored window: lets fev reshape the long frame into [series, horizon] arrays.
_SERIES_ORDER = [ID_COL, TARGET_NAME_COL, TIMESTAMP_COL]
_REPORTED_ACTUAL = "_reported_actual"


def coverage_column(level: float) -> str:
    """Name of the coverage column of a nominal interval level, e.g. ``0.8 -> "coverage_80"``."""
    return f"coverage_{round(level * 100, 6):g}"


COVERAGE_COLUMNS: tuple[str, ...] = tuple(coverage_column(level) for level in INTERVALS)


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    """Settings of a rolling-origin backtest.

    Parameters
    ----------
    prediction_length
        Forecast horizon in steps.
    n_windows
        Number of forecast origins (cutoffs).
    freq
        Pandas frequency of the series grid, e.g. ``"h"``.
    season_length
        Seasonal period that scales MASE and SQL (24 for hourly data with a daily cycle).
    step
        Steps between consecutive cutoffs. ``None`` (the default) means ``prediction_length``:
        horizons that touch without overlapping. Use :attr:`stride` for the resolved value; it
        keeps following ``prediction_length`` through ``dataclasses.replace``.
    quantile_levels
        Levels requested from every forecaster. Must contain the deciles of
        ``SCORED_QUANTILES`` (scored by WQL and SQL, as in fev-bench) and the bounds of every
        interval in ``INTERVALS`` (used for coverage).
    target
        Target column, or a tuple of target columns. ``item_id``, ``timestamp`` and
        ``actual`` are reserved names.
    known_covariates
        Columns whose future values are known in advance (calendar, published forecasts...).
        Only these reach the forecaster for the horizon; other columns are past-only.
    max_context
        If set, each window's context keeps only the last ``max_context`` rows per series.
    """

    prediction_length: int
    n_windows: int
    freq: str
    season_length: int
    step: int | None = None
    quantile_levels: tuple[float, ...] = EVAL_QUANTILES
    target: str | tuple[str, ...] = TARGET_COL
    known_covariates: tuple[str, ...] = ()
    max_context: int | None = None

    def __post_init__(self) -> None:
        for name in ("prediction_length", "n_windows", "season_length"):
            self._normalize(name, _positive_int(name, getattr(self, name)))
        if self.step is not None:  # None stays None: `stride` resolves it
            self._normalize("step", _positive_int("step", self.step))
        self._normalize("freq", _valid_freq(self.freq))
        self._normalize("quantile_levels", _valid_quantile_levels(self.quantile_levels))
        self._normalize("target", _valid_target(self.target))
        self._normalize(
            "known_covariates", _valid_known_covariates(self.known_covariates, self.targets)
        )
        if self.max_context is not None:
            self._normalize("max_context", _valid_max_context(self.max_context, self.season_length))

    def _normalize(self, name: str, value: object) -> None:
        # The dataclass is frozen; normalising validated inputs once at construction is the
        # documented way to set fields from __post_init__.
        object.__setattr__(self, name, value)

    @property
    def stride(self) -> int:
        """Steps between consecutive cutoffs: ``step``, or ``prediction_length`` if unset."""
        return self.prediction_length if self.step is None else self.step

    @property
    def targets(self) -> tuple[str, ...]:
        """Target columns as a tuple, also for a single target."""
        return (self.target,) if isinstance(self.target, str) else self.target

    @property
    def quantile_columns(self) -> tuple[str, ...]:
        """Forecast-frame column names of ``quantile_levels``."""
        return tuple(quantile_col(level) for level in self.quantile_levels)

    @property
    def min_series_length(self) -> int:
        """Rows a series needs: ``season_length + 1`` context points plus every horizon."""
        return self.season_length + 1 + self.prediction_length + (self.n_windows - 1) * self.stride


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _valid_freq(freq: object) -> str:
    if not isinstance(freq, str) or not freq:
        raise ValueError(f"freq must be a pandas frequency string such as 'h', got {freq!r}.")
    try:
        to_offset(freq)
    except ValueError as err:
        raise ValueError(f"freq={freq!r} is not a valid pandas frequency ('h', 'D', ...).") from err
    return freq


def _valid_quantile_levels(levels: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(level) for level in levels)
    if not normalized:
        raise ValueError("quantile_levels must not be empty.")
    if outside := [level for level in normalized if not 0.0 < level < 1.0]:
        raise ValueError(f"quantile_levels must be in (0, 1), got {outside}.")
    if any(later <= earlier for earlier, later in pairwise(normalized)):
        raise ValueError(f"quantile_levels must be sorted and unique, got {list(normalized)}.")
    interval_bounds = {bound for bounds in INTERVALS.values() for bound in bounds}
    if missing := sorted((set(SCORED_QUANTILES) | interval_bounds) - set(normalized)):
        raise ValueError(
            f"quantile_levels must include the deciles {list(SCORED_QUANTILES)} (scored by WQL "
            f"and SQL, as in fev-bench) and the interval bounds {sorted(interval_bounds)} "
            f"(used for coverage); missing {missing}."
        )
    return normalized


def _valid_target(target: str | Sequence[str]) -> str | tuple[str, ...]:
    names = (target,) if isinstance(target, str) else tuple(target)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"target must be a column name or a sequence of names, got {target!r}.")
    if len(set(names)) != len(names):
        raise ValueError(f"target contains duplicated names: {list(names)}.")
    if reserved := sorted(set(names) & {ID_COL, TIMESTAMP_COL, ACTUAL_COL}):
        raise ValueError(
            f"target cannot be named {reserved}: the backtest uses these names for the id, "
            "the timestamp and the joined true values. Rename the column before backtesting, "
            "e.g. df.rename(columns={'actual': 'load'})."
        )
    return target if isinstance(target, str) else names


def _valid_known_covariates(columns: Sequence[str], targets: tuple[str, ...]) -> tuple[str, ...]:
    names = (columns,) if isinstance(columns, str) else tuple(columns)
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"known_covariates must be column names, got {columns!r}.")
    if len(set(names)) != len(names):
        raise ValueError(f"known_covariates contains duplicated names: {list(names)}.")
    if clash := sorted(set(names) & {*targets, ID_COL, TIMESTAMP_COL}):
        raise ValueError(
            f"known_covariates cannot include the id, timestamp or target columns: {clash}. "
            "A target's future values are exactly what the forecaster must not see."
        )
    return names


def _valid_max_context(max_context: object, season_length: int) -> int:
    value = _positive_int("max_context", max_context)
    if value <= season_length:
        raise ValueError(
            f"max_context={value} must be larger than season_length={season_length}: "
            "MASE and SQL need at least one seasonal difference in every context."
        )
    return value


# --------------------------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Window:
    """One forecast origin of a backtest.

    - ``context``: every column of the series frame, only rows with ``timestamp <= cutoff``.
    - ``future_covariates``: ``item_id``, ``timestamp`` and the known covariates for the
      ``prediction_length`` steps after ``cutoff``; ``None`` without known covariates.
    - ``actuals``: ``item_id``, ``timestamp`` and the targets for the same steps. The
      forecaster never receives them.

    Every frame is new data built for this window (rows selected by a boolean mask, with a
    fresh index and no ``attrs``), so writing to it never changes ``df`` or another window.
    ``eq=False`` because DataFrames have no boolean equality: windows compare by identity.
    """

    index: int
    cutoff: pd.Timestamp
    context: pd.DataFrame
    future_covariates: pd.DataFrame | None
    actuals: pd.DataFrame

    def __repr__(self) -> str:
        future = "None" if self.future_covariates is None else f"{len(self.future_covariates)} rows"
        return (
            f"Window(index={self.index}, cutoff={self.cutoff}, context={len(self.context)} rows, "
            f"future_covariates={future}, actuals={len(self.actuals)} rows)"
        )


def rolling_origin_windows(df: pd.DataFrame, config: BacktestConfig) -> list[Window]:
    """Split a series frame into ``config.n_windows`` forecast origins, oldest first.

    The last window's horizon ends at the series' last timestamp and consecutive cutoffs are
    ``config.stride`` steps apart. All series must sit on a regular ``config.freq`` grid and end
    at the same timestamp, so they share the cutoffs (run ``validate_and_prepare`` first).

    Raises
    ------
    TypeError
        If ``df`` is not a DataFrame, its timestamps are not datetime64 or its targets are not
        numeric.
    ValueError
        If the frame is empty or malformed, the series are not aligned, or a series has fewer
        than ``season_length + 1`` context rows in the first window.
    """
    return list(iter_windows(df, config))


def iter_windows(df: pd.DataFrame, config: BacktestConfig) -> Iterator[Window]:
    """Lazy :func:`rolling_origin_windows`: validates now, builds one window at a time."""
    series = _prepare_series(df, config)
    origins = _origins(series[TIMESTAMP_COL].max(), config)
    _check_context_length(series, origins[0][0], config)
    return (
        _build_window(series, index, cutoff, horizon_end, config)
        for index, (cutoff, horizon_end) in enumerate(origins)
    )


def _prepare_series(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Validate the series frame and return a sorted copy (the input is never modified)."""
    _check_is_frame(df, "df")
    _check_columns(df, [ID_COL, TIMESTAMP_COL, *config.targets, *config.known_covariates], "df")
    if df.empty:
        raise ValueError("df has no rows.")
    _check_timestamps(df[TIMESTAMP_COL], "df")
    if df[ID_COL].isna().any():
        raise ValueError(f"df has missing values in {ID_COL!r}.")
    _check_numeric(df, config.targets, "df")
    series = df.sort_values([ID_COL, TIMESTAMP_COL], kind="stable", ignore_index=True)
    _check_unique_keys(series, [ID_COL, TIMESTAMP_COL], "df")
    _check_regular_grid(series, config.freq)
    _check_common_end(series)
    return series


def _check_regular_grid(series: pd.DataFrame, freq: str) -> None:
    off_grid = []
    for item_id, timestamps in series.groupby(ID_COL, sort=False, observed=True)[TIMESTAMP_COL]:
        expected = pd.date_range(timestamps.iloc[0], periods=len(timestamps), freq=freq)
        if not np.array_equal(timestamps.to_numpy(), expected.to_numpy()):
            off_grid.append(item_id)
    if off_grid:
        raise ValueError(
            f"Series {off_grid[:5]} are not on a regular {freq!r} grid (missing rows or "
            "off-grid timestamps). Reindex them first, e.g. with "
            f"c2p.data.validate_and_prepare(df, freq={freq!r})."
        )


def _check_common_end(series: pd.DataFrame) -> None:
    ends = series.groupby(ID_COL, sort=True, observed=True)[TIMESTAMP_COL].max()
    if (ends != ends.iloc[0]).any():
        raise ValueError(
            "All series must end at the same timestamp so that they share cutoffs; got "
            f"{_preview(ends)}. Trim them to a common end or backtest them separately."
        )


def _origins(
    last_timestamp: pd.Timestamp, config: BacktestConfig
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """``(cutoff, horizon_end)`` per window, oldest first; the last horizon ends at the end."""
    last_cutoff = (config.n_windows - 1) * config.stride
    grid = pd.date_range(
        end=last_timestamp, periods=last_cutoff + config.prediction_length + 1, freq=config.freq
    )
    return [
        (grid[start], grid[start + config.prediction_length])
        for start in range(0, last_cutoff + 1, config.stride)
    ]


def _check_context_length(
    series: pd.DataFrame, first_cutoff: pd.Timestamp, config: BacktestConfig
) -> None:
    needed = config.season_length + 1
    in_context = series.loc[series[TIMESTAMP_COL] <= first_cutoff]
    counts = (
        in_context.groupby(ID_COL, observed=True)
        .size()
        .reindex(series[ID_COL].unique(), fill_value=0)
    )
    short = counts[counts < needed]
    if not short.empty:
        raise ValueError(
            f"{len(short)} series are too short for {config.n_windows} windows: context rows "
            f"before the first cutoff {first_cutoff}: {_preview(short)}; need at least "
            f"season_length + 1 = {needed}. Each series needs >= {config.min_series_length} "
            "rows; reduce n_windows, step or prediction_length, or drop these series."
        )


def _build_window(
    series: pd.DataFrame,
    index: int,
    cutoff: pd.Timestamp,
    horizon_end: pd.Timestamp,
    config: BacktestConfig,
) -> Window:
    timestamps = series[TIMESTAMP_COL]
    context = series.loc[timestamps <= cutoff]
    if config.max_context is not None:
        context = context.groupby(ID_COL, sort=False, observed=True).tail(config.max_context)
    horizon = series.loc[(timestamps > cutoff) & (timestamps <= horizon_end)]
    future = None
    if config.known_covariates:  # an explicit allow-list: targets and past covariates never pass
        future = _detached(horizon[[ID_COL, TIMESTAMP_COL, *config.known_covariates]])
    return Window(
        index=index,
        cutoff=cutoff,
        context=_detached(context),
        future_covariates=future,
        actuals=_detached(horizon[[ID_COL, TIMESTAMP_COL, *config.targets]]),
    )


def _detached(frame: pd.DataFrame) -> pd.DataFrame:
    """Fresh index and no ``attrs`` (metadata of the full frame).

    No deep copy is needed: ``frame`` comes from a boolean row selection, which returns new
    data, and pandas copy-on-write keeps later writes local to each frame.
    """
    detached = frame.reset_index(drop=True)
    detached.attrs = {}
    return detached


# --------------------------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------------------------


def backtest(
    forecaster: Forecaster,
    df: pd.DataFrame,
    config: BacktestConfig,
    *,
    progress: bool = False,
) -> pd.DataFrame:
    """Run ``forecaster`` on every window and return its forecasts with the actuals joined.

    The result is the forecast frame (``item_id``, ``timestamp``, ``target_name``,
    ``predictions`` and the quantile columns of ``config.quantile_levels``, as float64) plus
    ``window``, ``cutoff``, ``model`` (``forecaster.name``), ``actual`` (joined *after* the
    forecaster has answered) and ``seconds`` (wall-clock time of that window's ``predict``
    call). Other columns returned by the forecaster are dropped.

    Each window's frames are built from ``df`` for that call only, so a forecaster that
    modifies its inputs cannot change ``df`` or later windows. Windows run oldest first: even a
    stateful forecaster has only seen earlier contexts when it forecasts a window.

    Raises
    ------
    TypeError
        If ``forecaster`` does not implement ``Forecaster``, ``df`` has wrong types (see
        :func:`rolling_origin_windows`), or an output is not a DataFrame or has non-numeric
        forecasts.
    ValueError
        If ``df`` is malformed (see :func:`rolling_origin_windows`), or an output misses or
        adds rows (ids, targets, timestamps), lacks a quantile column, or contains NaN or
        infinite forecasts. Output errors name the model and the window.
    RuntimeError
        If the leakage guard finds context after a cutoff or unexpected future columns right
        before ``predict`` (a bug in this module, never expected).
    """
    _check_forecaster(forecaster)
    frames = []
    for window in iter_windows(df, config):
        forecast, seconds = _timed_predict(forecaster, window, config)
        frames.append(_with_actuals(forecast, window, config, forecaster.name, seconds))
        logger.debug("%s: window %d done in %.3f s", forecaster.name, window.index, seconds)
        if progress:
            print(
                f"[{forecaster.name}] window {window.index + 1}/{config.n_windows} "
                f"(cutoff {window.cutoff}): {seconds:.2f} s",
                flush=True,
            )
    return pd.concat(frames, ignore_index=True)


def _check_forecaster(forecaster: object) -> None:
    if not isinstance(forecaster, Forecaster):
        raise TypeError(
            f"{type(forecaster).__name__} does not implement the Forecaster protocol "
            "(a `name` attribute and a `predict` method)."
        )
    if not isinstance(forecaster.name, str) or not forecaster.name:
        raise ValueError("forecaster.name must be a non-empty string: it labels the results.")


def _timed_predict(
    forecaster: Forecaster, window: Window, config: BacktestConfig
) -> tuple[object, float]:
    _check_no_leakage(window, config)
    start = time.perf_counter()
    forecast = forecaster.predict(
        window.context,
        prediction_length=config.prediction_length,
        quantile_levels=list(config.quantile_levels),
        freq=config.freq,
        target=config.target if isinstance(config.target, str) else list(config.target),
        future_covariates=window.future_covariates,
    )
    return forecast, time.perf_counter() - start


def _check_no_leakage(window: Window, config: BacktestConfig) -> None:
    """Re-check the window invariants right before the forecaster sees the data."""
    if (window.context[TIMESTAMP_COL] > window.cutoff).any():
        raise RuntimeError(f"Leakage guard: window {window.index} has context after its cutoff.")
    future = window.future_covariates
    if future is None:
        return
    allowed = [ID_COL, TIMESTAMP_COL, *config.known_covariates]
    if list(future.columns) != allowed:
        raise RuntimeError(
            f"Leakage guard: future covariates of window {window.index} have columns "
            f"{list(future.columns)}, expected exactly {allowed}."
        )
    if (future[TIMESTAMP_COL] <= window.cutoff).any():
        raise RuntimeError(f"Leakage guard: window {window.index} has future rows <= cutoff.")


def _with_actuals(
    forecast: object, window: Window, config: BacktestConfig, model: str, seconds: float
) -> pd.DataFrame:
    where = f"The forecast of {model!r} for window {window.index} (cutoff {window.cutoff})"
    if not isinstance(forecast, pd.DataFrame):
        raise TypeError(f"{where} is a {type(forecast).__name__}, expected a pandas DataFrame.")
    columns = _forecast_columns(config)
    _check_columns(forecast, columns, where)
    joined = _join_actuals(forecast[columns], window.actuals, config.targets, where)
    _check_finite(joined, [PREDICTION_COL, *config.quantile_columns], where)
    joined = joined.astype(dict.fromkeys([PREDICTION_COL, *config.quantile_columns], "float64"))
    labels = {
        WINDOW_COL: window.index,
        CUTOFF_COL: window.cutoff,
        MODEL_COL: model,
        SECONDS_COL: seconds,
    }
    return joined.assign(**labels)[
        [*columns, WINDOW_COL, CUTOFF_COL, MODEL_COL, ACTUAL_COL, SECONDS_COL]
    ]


def _forecast_columns(config: BacktestConfig) -> list[str]:
    return [ID_COL, TIMESTAMP_COL, TARGET_NAME_COL, PREDICTION_COL, *config.quantile_columns]


def _join_actuals(
    forecast: pd.DataFrame, actuals: pd.DataFrame, targets: Sequence[str], where: str
) -> pd.DataFrame:
    """Attach the true values, requiring exactly one forecast row per series, target and step.

    Returns the rows sorted by ``item_id``, ``target_name``, ``timestamp``.
    """
    expected = actuals.melt(
        id_vars=[ID_COL, TIMESTAMP_COL],
        value_vars=list(targets),
        var_name=TARGET_NAME_COL,
        value_name=ACTUAL_COL,
    )
    _check_timestamps(forecast[TIMESTAMP_COL], where)
    forecast = forecast.astype({key: expected[key].dtype for key in _SERIES_ORDER})
    _check_unique_keys(forecast, _SERIES_ORDER, where)
    merged = expected.merge(forecast, on=_SERIES_ORDER, how="outer", indicator=True)
    missing = merged.loc[merged["_merge"] == "left_only", _SERIES_ORDER]
    unexpected = merged.loc[merged["_merge"] == "right_only", _SERIES_ORDER]
    if len(missing) or len(unexpected):
        raise ValueError(
            f"{where} must have one row per series, target and horizon step ({len(expected)} "
            f"rows), continuing each series' grid after the cutoff. Missing {len(missing)} "
            f"rows, e.g. {_examples(missing)}; unexpected {len(unexpected)} rows, e.g. "
            f"{_examples(unexpected)}."
        )
    return merged.drop(columns="_merge").sort_values(_SERIES_ORDER, ignore_index=True)


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------


def evaluate(
    forecasts: pd.DataFrame,
    df: pd.DataFrame,
    config: BacktestConfig,
    *,
    by_target: bool = False,
) -> pd.DataFrame:
    """Score backtest forecasts with fev metrics: one row per (model, window).

    Columns: ``model``, ``window``, ``cutoff`` (then ``target_name`` with ``by_target=True``),
    ``n_series``, ``WQL``, ``MASE``, ``WAPE``, ``SQL``, ``coverage_80``, ``coverage_98`` (and
    ``seconds`` when ``forecasts`` has it).

    - Metrics use fev-bench settings: ``WQL(epsilon=1.0)``, ``WAPE(epsilon=1.0)``, ``MASE()``
      and ``SQL()`` with ``seasonality=config.season_length``, and WQL and SQL average the
      ``SCORED_QUANTILES`` deciles. MASE and SQL are scaled by the window's context only
      (``y_past``, cut to ``max_context``), never by data after the cutoff.
    - Each (item, target) pair is one fev series (``D=1``), like fev's
      ``generate_univariate_targets_from``. With several targets, WQL and WAPE are therefore
      dominated by the target with the largest values (MASE and SQL are scaled per pair).
      ``by_target=True`` gives one row per (model, window, target) instead; averaging those
      rows over targets reproduces fev's multivariate scoring (one dimension per target).
    - NaN actuals are ignored (fev uses ``nanmean``). Coverage is the share of non-NaN
      actuals inside each ``INTERVALS`` interval (NaN if no actual is observed).
    - True values come from ``df``. If ``forecasts`` has an ``actual`` column it must equal
      them exactly, and the cutoffs must match ``config``. This catches another ``df`` or other
      windows (``n_windows``, ``step``, ``prediction_length``), but not another
      ``season_length`` or ``max_context``, which change MASE and SQL: reuse the backtest's
      config.

    Raises
    ------
    TypeError
        If ``forecasts`` or ``df`` is not a DataFrame or has wrong column types.
    ValueError
        If ``forecasts`` is empty, misses columns, rows or windows, or does not match ``df``
        and ``config`` as described above.
    """
    _check_is_frame(forecasts, "forecasts")
    _check_columns(
        forecasts, [*_forecast_columns(config), WINDOW_COL, CUTOFF_COL, MODEL_COL], "forecasts"
    )
    if forecasts.empty:
        raise ValueError("forecasts is empty.")
    _check_is_frame(df, "df")
    _check_columns(df, [ID_COL, TIMESTAMP_COL, *config.targets], "df")
    targets_only = replace(config, known_covariates=())  # scoring needs the targets only
    windows = {
        window.index: window
        for window in iter_windows(df[[ID_COL, TIMESTAMP_COL, *config.targets]], targets_only)
    }
    rows = [
        row
        for (model, index), group in forecasts.groupby([MODEL_COL, WINDOW_COL], sort=True)
        for row in _score_window(group, windows, model, index, config, by_target=by_target)
    ]
    return pd.DataFrame(rows)


def _score_window(
    forecast: pd.DataFrame,
    windows: Mapping[int, Window],
    model: str,
    index: int,
    config: BacktestConfig,
    *,
    by_target: bool,
) -> list[dict[str, object]]:
    """Metric rows of one (model, window): one row, or one per target with ``by_target``."""
    where = f"The forecast of {model!r} for window {index}"
    if index not in windows:
        raise ValueError(
            f"{where} does not exist for this config (windows 0..{config.n_windows - 1}). "
            "Evaluate with the same df and config as the backtest."
        )
    window = windows[index]
    _check_cutoff(forecast, window, where)
    joined = _join_actuals(
        _with_reported_actual(forecast, config), window.actuals, config.targets, where
    )
    joined = _drop_checked_reported_actual(joined, where)
    _check_finite(joined, [PREDICTION_COL, *config.quantile_columns], where)
    labels: dict[str, object] = {MODEL_COL: model, WINDOW_COL: index, CUTOFF_COL: window.cutoff}
    timing = {SECONDS_COL: float(forecast[SECONDS_COL].iloc[0])} if SECONDS_COL in forecast else {}
    if not by_target:
        return [labels | _scores(joined, window.context, config) | timing]
    return [
        labels
        | {TARGET_NAME_COL: target}
        | _scores(joined.loc[joined[TARGET_NAME_COL] == target], window.context, config)
        | timing
        for target in config.targets
    ]


def _scores(
    joined: pd.DataFrame, context: pd.DataFrame, config: BacktestConfig
) -> dict[str, object]:
    """``n_series``, the fev metrics and coverage of a joined, sorted block of forecasts."""
    arrays = _fev_arrays(joined, context, config)
    return {
        N_SERIES_COL: len(arrays["y_true"]),
        **_fev_scores(arrays, config),
        **{
            coverage_column(level): _coverage(joined, *bounds)
            for level, bounds in INTERVALS.items()
        },
    }


def _check_cutoff(forecast: pd.DataFrame, window: Window, where: str) -> None:
    cutoffs = {pd.Timestamp(cutoff) for cutoff in forecast[CUTOFF_COL].unique()}
    if cutoffs != {window.cutoff}:
        raise ValueError(
            f"{where} has cutoff {sorted(cutoffs)}, but df and config give {window.cutoff}. "
            "Evaluate with the same df and config as the backtest."
        )


def _with_reported_actual(forecast: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    columns = _forecast_columns(config)
    if ACTUAL_COL not in forecast:
        return forecast[columns]
    return forecast[[*columns, ACTUAL_COL]].rename(columns={ACTUAL_COL: _REPORTED_ACTUAL})


def _drop_checked_reported_actual(joined: pd.DataFrame, where: str) -> pd.DataFrame:
    """Drop the forecasts' own ``actual`` column after checking it against ``df``.

    Both come from ``df`` unchanged, so they must be equal exactly: a tolerance would let a
    small edit pass at large scales (e.g. 0.25 MW on a load of 28,000 MW).
    """
    if _REPORTED_ACTUAL not in joined:
        return joined
    reported = joined[_REPORTED_ACTUAL].to_numpy("float64", na_value=np.nan)
    actual = joined[ACTUAL_COL].to_numpy("float64", na_value=np.nan)
    if not np.array_equal(reported, actual, equal_nan=True):
        raise ValueError(
            f"{where}: its 'actual' column differs from the values in df. "
            "Evaluate with the same df and config as the backtest."
        )
    return joined.drop(columns=_REPORTED_ACTUAL)


def _fev_arrays(
    joined: pd.DataFrame, context: pd.DataFrame, config: BacktestConfig
) -> dict[str, np.ndarray]:
    """Fev inputs: ``y_true``/``y_pred`` [N, H, 1], ``q_pred`` [N, H, 1, Q], ragged ``y_past``.

    ``joined`` is sorted by item, target and timestamp and complete, so a reshape gives one
    row per (item, target) pair; ``y_past`` holds the same pairs' context values. ``q_pred``
    holds the ``SCORED_QUANTILES`` columns only.
    """
    horizon = config.prediction_length
    pairs = joined[[ID_COL, TARGET_NAME_COL]].iloc[::horizon]
    n_series = len(pairs)
    past = _past_values(context, pairs)
    return {
        "y_true": joined[ACTUAL_COL].to_numpy("float64").reshape(n_series, horizon, 1),
        "y_pred": joined[PREDICTION_COL].to_numpy("float64").reshape(n_series, horizon, 1),
        "q_pred": joined[_SCORED_COLUMNS]
        .to_numpy("float64")
        .reshape(n_series, horizon, 1, len(_SCORED_COLUMNS)),
        "y_past": np.concatenate(past)[:, None],
        "y_past_lengths": np.array([len(values) for values in past]),
    }


def _past_values(context: pd.DataFrame, pairs: pd.DataFrame) -> list[np.ndarray]:
    by_item = dict(tuple(context.groupby(ID_COL, sort=False, observed=True)))
    return [
        by_item[item][target].to_numpy("float64") for item, target in pairs.itertuples(index=False)
    ]


def _fev_scores(arrays: Mapping[str, np.ndarray], config: BacktestConfig) -> dict[str, float]:
    with warnings.catch_warnings():
        # Same as fev: an all-NaN slice yields a NaN score instead of a RuntimeWarning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            name: metric.compute(
                **arrays,
                seasonality=config.season_length,
                quantile_levels=list(SCORED_QUANTILES),
            )
            for name, metric in METRICS.items()
        }


def _coverage(joined: pd.DataFrame, lower: float, upper: float) -> float:
    """Share of non-NaN actuals inside ``[q_lower, q_upper]``; NaN if none is observed.

    ``BacktestConfig`` requires the ``INTERVALS`` levels, so both columns always exist.
    """
    observed = joined.loc[joined[ACTUAL_COL].notna()]
    if observed.empty:
        return float("nan")
    inside = observed[ACTUAL_COL].between(
        observed[quantile_col(lower)], observed[quantile_col(upper)]
    )
    return float(inside.mean())


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over windows of each model's metrics, best WQL first.

    With per-target metrics (``evaluate(..., by_target=True)``) there is one row per model and
    target, sorted by target and then WQL. NaN scores are not skipped: fev also averages
    windows with a plain mean, and a NaN means a metric was undefined in some window, which a
    skipping mean would hide.
    """
    _check_is_frame(metrics, "metrics")
    _check_columns(metrics, [MODEL_COL, WINDOW_COL], "metrics")
    candidates = (*METRICS, *COVERAGE_COLUMNS, SECONDS_COL)
    columns = [column for column in candidates if column in metrics]
    if not columns:
        raise ValueError(f"metrics has none of the metric columns {list(candidates)}.")
    per_target = TARGET_NAME_COL in metrics
    keys = [MODEL_COL, TARGET_NAME_COL] if per_target else [MODEL_COL]
    grouped = metrics.groupby(keys, sort=True)[columns]
    summary = pd.concat(
        [
            grouped.size().rename("n_windows"),
            grouped.mean(skipna=False),
            grouped.std(skipna=False).add_suffix("_std"),
        ],
        axis=1,
    )
    ordered = ["n_windows", *[name for column in columns for name in (column, f"{column}_std")]]
    summary = summary[ordered].reset_index()
    if "WQL" in summary:
        order = [TARGET_NAME_COL, "WQL"] if per_target else ["WQL"]
        summary = summary.sort_values(order, kind="stable", ignore_index=True)
    return summary


# --------------------------------------------------------------------------------------------
# Diagnostics over the scored forecasts: calibration by level, error by horizon step
# --------------------------------------------------------------------------------------------

STEP_COL = "step"


def reliability_table(
    forecasts: pd.DataFrame, quantile_levels: Sequence[float] = SCORED_QUANTILES
) -> pd.DataFrame:
    """Share of actuals at or below each forecast quantile, per model.

    ``forecasts`` is the output of :func:`backtest` (forecasts with ``actual`` joined). A
    calibrated model has ``observed`` equal to ``level`` at every level: 10% of the actuals
    below its 0.1 quantile, 50% below its median. ``observed`` above ``level`` at the low
    levels and below it at the high ones means intervals that are too narrow. Rows with a NaN
    actual are left out; ``n`` counts the rows used.

    Returns one row per (model, level): ``model``, ``level``, ``observed``, ``n``.
    """
    _check_is_frame(forecasts, "forecasts")
    columns = [quantile_col(level) for level in quantile_levels]
    _check_columns(forecasts, [MODEL_COL, ACTUAL_COL, *columns], "forecasts")
    observed = forecasts.loc[forecasts[ACTUAL_COL].notna()]
    rows = [
        {
            MODEL_COL: model,
            "level": float(level),
            "observed": float((group[ACTUAL_COL] <= group[column]).mean()),
            "n": len(group),
        }
        for model, group in observed.groupby(MODEL_COL, sort=True)
        for level, column in zip(quantile_levels, columns, strict=True)
    ]
    return pd.DataFrame(rows)


def error_by_step(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Error of each model by horizon step, pooled over windows and series.

    ``step`` counts from 1 at the first forecast timestamp of each window. ``WAPE`` is the sum
    of the point forecast's absolute errors over the sum of the absolute actuals at that step,
    and ``WQL`` is twice the mean pinball loss over ``SCORED_QUANTILES`` over the mean absolute
    actual, so both read as a share of the level and are comparable across steps. Rows with a
    NaN actual are left out.

    Returns one row per (model, step): ``model``, ``step``, ``WAPE``, ``WQL``, ``n``.
    """
    _check_is_frame(forecasts, "forecasts")
    keys = [MODEL_COL, WINDOW_COL, ID_COL, TARGET_NAME_COL]
    _check_columns(forecasts, [*keys, TIMESTAMP_COL], "forecasts")
    forecasts = forecasts.reset_index(drop=True)
    # The step is counted before NaN actuals are dropped, so a missing hour keeps its number.
    steps = forecasts.groupby(keys, sort=False)[TIMESTAMP_COL].rank(method="first").astype(int)
    return _pooled_errors(forecasts.assign(**{STEP_COL: steps}), STEP_COL)


def error_by_horizon_block(forecasts: pd.DataFrame, edges: Sequence[int]) -> pd.DataFrame:
    """Error of each model on each window, separately for blocks of horizon steps.

    ``edges`` are the last steps of the blocks, in increasing order: ``(12, 24)`` scores steps
    1 to 12 and steps 13 to 24 apart. Within a (model, window, block) every series and step is
    pooled, with ``WAPE`` and ``WQL`` defined as in :func:`error_by_step`. The result has the
    columns :func:`skill_table` needs, so one block can be compared across models with the
    same intervals as a whole window. Steps beyond the last edge are left out.

    Returns one row per (model, window, block): ``model``, ``window``, ``block`` (``"1-12"``),
    ``WAPE``, ``WQL``, ``n``.
    """
    _check_is_frame(forecasts, "forecasts")
    keys = [MODEL_COL, WINDOW_COL, ID_COL, TARGET_NAME_COL]
    _check_columns(forecasts, [*keys, TIMESTAMP_COL], "forecasts")
    edges = [int(edge) for edge in edges]
    if not edges or edges[0] < 1 or any(b <= a for a, b in pairwise(edges)):
        raise ValueError(f"edges must be increasing step numbers >= 1, got {edges}.")
    forecasts = forecasts.reset_index(drop=True)
    steps = forecasts.groupby(keys, sort=False)[TIMESTAMP_COL].rank(method="first").astype(int)
    starts = [1, *(edge + 1 for edge in edges[:-1])]
    labels = [f"{start}-{edge}" for start, edge in zip(starts, edges, strict=True)]
    block = pd.cut(steps, bins=[0, *edges], labels=labels)
    kept = forecasts.loc[block.notna()].copy()
    # One pooled error per (window, block): the pair is packed into a single grouping key.
    kept["_cell"] = kept[WINDOW_COL].astype(str) + "|" + block[block.notna()].astype(str)
    table = _pooled_errors(kept, "_cell")
    cell = table.pop("_cell").str.split("|", n=1, expand=True)
    table.insert(1, WINDOW_COL, cell[0].astype(forecasts[WINDOW_COL].dtype))
    table.insert(2, "block", pd.Categorical(cell[1], categories=labels, ordered=True))
    return table.sort_values([MODEL_COL, WINDOW_COL, "block"], ignore_index=True)


def error_by_series(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Error of each model on each series, pooled over windows and horizon steps.

    The per-window metrics of :func:`evaluate` pool every series of a window into one number,
    which is how benchmarks score a panel. This is the other cut: one row per (model, series),
    so a panel of a thousand series gives a thousand paired errors to compare two models on
    (``skill_table(..., unit="item_id")``). ``WAPE`` and ``WQL`` are defined as in
    :func:`error_by_step`. A series whose actuals are all zero has undefined (NaN) errors.

    Returns one row per (model, series): ``model``, ``item_id``, ``WAPE``, ``WQL``, ``n``.
    """
    _check_is_frame(forecasts, "forecasts")
    _check_columns(forecasts, [MODEL_COL, ID_COL], "forecasts")
    return _pooled_errors(forecasts.reset_index(drop=True), ID_COL)


def _pooled_errors(forecasts: pd.DataFrame, by: str) -> pd.DataFrame:
    """WAPE and WQL of each model per value of ``by``, pooling every row that has an actual."""
    _check_columns(forecasts, [ACTUAL_COL, PREDICTION_COL, *_SCORED_COLUMNS], "forecasts")
    frame = forecasts.loc[forecasts[ACTUAL_COL].notna()].copy()
    actual = frame[ACTUAL_COL].to_numpy("float64")
    frame["_abs_error"] = np.abs(actual - frame[PREDICTION_COL].to_numpy("float64"))
    frame["_abs_actual"] = np.abs(actual)
    quantiles = frame[_SCORED_COLUMNS].to_numpy("float64")
    levels = np.asarray(SCORED_QUANTILES)
    diff = actual[:, None] - quantiles
    frame["_pinball"] = np.maximum(levels * diff, (levels - 1.0) * diff).mean(axis=1)
    table = (
        frame.groupby([MODEL_COL, by], sort=True, observed=True)
        .agg(
            _abs_error=("_abs_error", "sum"),
            _abs_actual=("_abs_actual", "sum"),
            _pinball=("_pinball", "mean"),
            n=("_pinball", "size"),
        )
        .reset_index()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        table["WAPE"] = table["_abs_error"] / table["_abs_actual"]
        table["WQL"] = 2.0 * table["_pinball"] / (table["_abs_actual"] / table["n"])
    return table[[MODEL_COL, by, "WAPE", "WQL", "n"]]


# --------------------------------------------------------------------------------------------
# Skill scores
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillScore:
    """Skill of ``model`` over ``baseline`` on ``metric``, as returned by :func:`skill_score`.

    ``skill`` is ``1 - gmean(model / baseline)`` over the windows (the typical window) or,
    when it was asked for with ``aggregate="total"``, ``1 - sum(model) / sum(baseline)`` (the
    error of all the windows together).
    ``ci_low``/``ci_high`` bound the 95% bootstrap interval over the ``n_units`` windows;
    ``win_rate`` is the share of windows where the model beats the baseline (ties count 0.5).
    ``block_length`` is the run of consecutive windows the bootstrap resampled together (1 =
    windows resampled independently).
    """

    model: str
    baseline: str
    metric: str
    skill: float
    ci_low: float
    ci_high: float
    win_rate: float
    n_units: int
    block_length: int = 1


def skill_score(
    metrics: pd.DataFrame,
    *,
    model: str,
    baseline: str = "seasonal_naive",
    metric: str = "WQL",
    n_resamples: int = 1000,
    seed: int = SEED,
    clip: tuple[float, float] = (0.01, 100.0),
    block_length: int = 1,
    unit: str = WINDOW_COL,
    aggregate: Literal["geometric", "total"] = "geometric",
) -> SkillScore:
    """Skill score of ``model`` relative to ``baseline``, with a bootstrap confidence interval.

    Follows the fev-bench leaderboard with windows as units: the relative error of each window
    (model / baseline) is clipped to ``clip`` so a single window cannot dominate, the skill is
    ``1 - gmean(relative errors)`` (0 = as good as the baseline, 0.5 = errors halved), and
    the 95% interval comes from a bootstrap over windows. Deterministic for a given ``seed``.
    Per-target metrics must be filtered to one target first.

    ``block_length=1`` resamples windows independently (``fev.analysis.bootstrap``). That is
    right for windows far apart in time and too narrow for consecutive ones, whose errors are
    correlated: a cold week is seven bad windows, not seven independent draws. A larger
    ``block_length`` resamples runs of that many consecutive windows instead (circular
    moving-block bootstrap), which keeps the correlation inside each run. The point estimate
    is the same either way.

    ``unit`` names the column whose values are the paired units, one row per model and unit:
    windows by default, or ``"item_id"`` for the per-series errors of :func:`error_by_series`,
    which resamples the series of a panel instead of its windows.

    ``aggregate`` chooses what the windows are summarised into, and the two answer different
    questions. ``"geometric"`` (the default, described above) gives every window the same
    weight whatever the size of its error: it is the change on a *typical* window.
    ``"total"`` is ``1 - sum(model errors) / sum(baseline errors)``, the change of the error
    of all the windows together, in which a bad day weighs as much as its error; ``clip`` does
    not apply to it. When a few windows hold much of the error the two can differ by many
    points, and even in sign, so a comparison between models should report both.
    """
    model_errors, baseline_errors = _paired_errors(metrics, model, baseline, metric, unit)
    relative = _relative_errors(model_errors, baseline_errors, clip)
    n_resamples = _positive_int("n_resamples", n_resamples)
    block_length = _positive_int("block_length", block_length)
    if block_length > len(relative):
        raise ValueError(
            f"block_length={block_length} exceeds the {len(relative)} windows being resampled."
        )
    if aggregate not in ("geometric", "total"):
        raise ValueError(f"aggregate must be 'geometric' or 'total', got {aggregate!r}.")
    if aggregate == "total":
        skill, ci_low, ci_high = _total_bootstrap(
            model_errors,
            baseline_errors,
            block_length=block_length,
            n_resamples=n_resamples,
            seed=seed,
        )
    elif block_length == 1:
        skill, ci_low, ci_high = bootstrap(
            relative[:, None], statistic=_skill, n_resamples=n_resamples, seed=seed
        )
    else:
        skill, ci_low, ci_high = _block_bootstrap(
            relative, block_length=block_length, n_resamples=n_resamples, seed=seed
        )
    return SkillScore(
        model=model,
        baseline=baseline,
        metric=metric,
        skill=float(skill[0]),
        ci_low=float(ci_low[0]),
        ci_high=float(ci_high[0]),
        win_rate=_win_rate(model_errors, baseline_errors),
        n_units=len(relative),
        block_length=block_length,
    )


def _block_bootstrap(
    relative: np.ndarray, *, block_length: int, n_resamples: int, seed: int, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point estimate and (1 - alpha) interval of the skill, resampling runs of windows.

    ``relative`` holds one relative error per window, ordered in time. Each resample joins
    ``ceil(n / block_length)`` runs that start at random windows and wrap around the end (the
    circular variant, so every window is equally likely to be drawn) and keeps the first ``n``
    values. Same return shape as ``fev.analysis.bootstrap`` for one model.
    """
    indices = _block_indices(len(relative), block_length, n_resamples, seed)
    resampled = _skill(relative[indices].T)
    point = _skill(relative[:, None])
    lower = np.quantile(resampled, 0.5 * alpha, keepdims=True)
    upper = np.quantile(resampled, 1 - 0.5 * alpha, keepdims=True)
    return point, lower, upper


def _block_indices(n_windows: int, block_length: int, n_resamples: int, seed: int) -> np.ndarray:
    """Window positions of every resample, one row each: runs of ``block_length`` windows."""
    rng = np.random.default_rng(seed=seed)
    n_blocks = -(-n_windows // block_length)
    starts = rng.integers(0, n_windows, size=(n_resamples, n_blocks))
    indices = (starts[:, :, None] + np.arange(block_length)) % n_windows
    return indices.reshape(n_resamples, -1)[:, :n_windows]


def _total_bootstrap(
    model_errors: np.ndarray,
    baseline_errors: np.ndarray,
    *,
    block_length: int,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point estimate and interval of ``1 - sum(model) / sum(baseline)``, resampling windows.

    The same resamples as :func:`_block_bootstrap` (independent windows when ``block_length``
    is 1), with both models read on the same resampled windows so the comparison stays paired.
    """
    if not baseline_errors.sum() > 0:
        raise ValueError("The baseline's total error is zero: the total skill is undefined.")
    indices = _block_indices(len(model_errors), block_length, n_resamples, seed)
    resampled = _total_skill(model_errors[indices].T, baseline_errors[indices].T)
    point = _total_skill(model_errors[:, None], baseline_errors[:, None])
    lower = np.quantile(resampled, 0.5 * alpha, keepdims=True)
    upper = np.quantile(resampled, 1 - 0.5 * alpha, keepdims=True)
    return point, lower, upper


def skill_table(
    metrics: pd.DataFrame,
    baseline: str = "seasonal_naive",
    metric: str = "WQL",
    *,
    n_resamples: int = 1000,
    seed: int = SEED,
    clip: tuple[float, float] = (0.01, 100.0),
    block_length: int = 1,
    unit: str = WINDOW_COL,
    aggregate: Literal["geometric", "total", "both"] = "geometric",
) -> pd.DataFrame:
    """:func:`skill_score` of every model in ``metrics`` (baseline included), best first.

    ``aggregate="both"`` returns the geometric table with three more columns, ``total_skill``,
    ``total_ci_low`` and ``total_ci_high``: the same comparison on the error of all the
    windows together. Rows are ordered by ``skill`` in every case.
    """
    _check_is_frame(metrics, "metrics")
    _check_columns(metrics, [MODEL_COL], "metrics")
    if aggregate not in ("geometric", "total", "both"):
        raise ValueError(f"aggregate must be 'geometric', 'total' or 'both', got {aggregate!r}.")

    def scores(kind: Literal["geometric", "total"]) -> pd.DataFrame:
        rows = [
            asdict(
                skill_score(
                    metrics,
                    model=model,
                    baseline=baseline,
                    metric=metric,
                    n_resamples=n_resamples,
                    seed=seed,
                    clip=clip,
                    block_length=block_length,
                    unit=unit,
                    aggregate=kind,
                )
            )
            for model in sorted(metrics[MODEL_COL].unique())
        ]
        return pd.DataFrame(rows)

    if aggregate == "both":
        table = scores("geometric")
        total = scores("total")
        table["total_skill"] = total["skill"]
        table["total_ci_low"] = total["ci_low"]
        table["total_ci_high"] = total["ci_high"]
    else:
        table = scores(aggregate)
    return table.sort_values("skill", ascending=False, kind="stable", ignore_index=True)


def _paired_errors(
    metrics: pd.DataFrame, model: str, baseline: str, metric: str, unit: str = WINDOW_COL
) -> tuple[np.ndarray, np.ndarray]:
    """Errors of both models on the same units (windows by default), ordered by unit."""
    _check_is_frame(metrics, "metrics")
    _check_columns(metrics, [MODEL_COL, unit, metric], "metrics")
    targets = sorted(metrics[TARGET_NAME_COL].unique()) if TARGET_NAME_COL in metrics else []
    if len(targets) > 1:
        raise ValueError(
            f"metrics has rows for several targets {targets}; score one target at a time, "
            f"e.g. metrics[metrics[{TARGET_NAME_COL!r}] == {targets[0]!r}]."
        )
    available = sorted(metrics[MODEL_COL].unique())
    if missing := [name for name in (model, baseline) if name not in available]:
        raise ValueError(f"metrics has no rows for {missing}; available models: {available}.")
    model_errors = _errors_by_window(metrics, model, metric, unit)
    baseline_errors = _errors_by_window(metrics, baseline, metric, unit)
    if not model_errors.index.equals(baseline_errors.index):
        raise ValueError(
            f"{model!r} and {baseline!r} must be scored on the same windows; got "
            f"{model_errors.index.tolist()} and {baseline_errors.index.tolist()}."
        )
    if undefined := model_errors.index[model_errors.isna() | baseline_errors.isna()].tolist():
        raise ValueError(f"{metric} is NaN for {model!r} or {baseline!r} in windows {undefined}.")
    return model_errors.to_numpy("float64"), baseline_errors.to_numpy("float64")


def _errors_by_window(
    metrics: pd.DataFrame, model: str, metric: str, unit: str = WINDOW_COL
) -> pd.Series:
    rows = metrics.loc[metrics[MODEL_COL] == model]
    if rows[unit].duplicated().any():
        raise ValueError(f"metrics has several rows for the same {unit} of {model!r}.")
    return rows.set_index(unit)[metric].sort_index().astype("float64")


def _relative_errors(
    model_errors: np.ndarray, baseline_errors: np.ndarray, clip: tuple[float, float]
) -> np.ndarray:
    low, high = clip
    if not 0.0 < low < high:
        raise ValueError(f"clip must satisfy 0 < low < high, got {clip}.")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = model_errors / baseline_errors
    if np.isnan(ratio).any():
        raise ValueError("Relative error is 0/0 in some windows: both models have zero error.")
    return np.clip(ratio, low, high)


def _skill(relative_errors: np.ndarray) -> np.ndarray:
    """Bootstrap statistic: ``1 - gmean`` over windows (axis 0), as in fev's leaderboard."""
    return 1.0 - scipy.stats.gmean(relative_errors, axis=0)


def _total_skill(model_errors: np.ndarray, baseline_errors: np.ndarray) -> np.ndarray:
    """Bootstrap statistic: ``1 - sum(model) / sum(baseline)`` over windows (axis 0)."""
    return 1.0 - model_errors.sum(axis=0) / baseline_errors.sum(axis=0)


def _win_rate(model_errors: np.ndarray, baseline_errors: np.ndarray) -> float:
    wins = np.mean(model_errors < baseline_errors)
    ties = np.mean(model_errors == baseline_errors)
    return float(wins + 0.5 * ties)


# --------------------------------------------------------------------------------------------
# Small checks shared by the public functions
# --------------------------------------------------------------------------------------------


def _check_is_frame(frame: object, where: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{where} must be a pandas DataFrame, got {type(frame).__name__}.")


def _check_columns(frame: pd.DataFrame, columns: Sequence[str], where: str) -> None:
    if missing := [column for column in columns if column not in frame.columns]:
        raise ValueError(f"{where} is missing columns {missing}; it has {list(frame.columns)}.")


def _check_timestamps(timestamps: pd.Series, where: str) -> None:
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            f"{where}: timestamps must be timezone-naive UTC, got {timestamps.dtype}. "
            "Convert with .dt.tz_convert('UTC').dt.tz_localize(None)."
        )
    if not pd.api.types.is_datetime64_dtype(timestamps.dtype):
        raise TypeError(
            f"{where}: {TIMESTAMP_COL!r} must be datetime64, got {timestamps.dtype}. Parse it "
            "with pd.to_datetime (or read_csv(..., parse_dates=[...]))."
        )
    if timestamps.isna().any():
        raise ValueError(f"{where}: {TIMESTAMP_COL!r} has missing values (NaT).")


def _check_numeric(frame: pd.DataFrame, columns: Sequence[str], where: str) -> None:
    if bad := [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
    ]:
        raise TypeError(f"{where}: columns {bad} must be numeric.")


def _check_unique_keys(frame: pd.DataFrame, keys: list[str], where: str) -> None:
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        raise ValueError(
            f"{where} has {int(duplicated.sum())} rows with duplicated {keys}, e.g. "
            f"{_examples(frame.loc[duplicated, keys])}."
        )


def _check_finite(frame: pd.DataFrame, columns: list[str], where: str) -> None:
    _check_numeric(frame, columns, where)
    finite = np.isfinite(frame[columns].to_numpy("float64", na_value=np.nan))
    if not finite.all():
        counts = {
            column: int(n) for column, n in zip(columns, (~finite).sum(axis=0), strict=True) if n
        }
        raise ValueError(f"{where} has NaN or infinite values (count per column): {counts}.")


def _examples(frame: pd.DataFrame, n: int = 3) -> list[tuple[str, ...]]:
    return [tuple(str(value) for value in row) for row in frame.head(n).itertuples(index=False)]


def _preview(values: pd.Series, n: int = 5) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.head(n).items()}


# --------------------------------------------------------------------------------------------
# Several runs of one configuration
# --------------------------------------------------------------------------------------------


def average_runs(metrics: pd.DataFrame, name: str) -> pd.DataFrame:
    """Average the per-window metrics of several runs into the metrics of one model.

    ``metrics`` holds the :func:`evaluate` rows of runs that differ only in something the
    experimenter does not choose, such as the random seed of a fine-tune. For every window the
    numeric columns are averaged over the runs and the result is labelled ``name``, so it can
    go through :func:`summarize` and :func:`skill_table` like any model. What it estimates is
    the error to expect from *a* run of that configuration, which is the quantity a single
    seed reports with luck added.

    Every run must cover the same windows.
    """
    _check_is_frame(metrics, "metrics")
    _check_columns(metrics, [MODEL_COL, WINDOW_COL], "metrics")
    if metrics.empty:
        raise ValueError("metrics is empty.")
    windows_by_run = metrics.groupby(MODEL_COL, sort=True)[WINDOW_COL].apply(
        lambda column: tuple(sorted(column))
    )
    if len(set(windows_by_run)) != 1:
        raise ValueError(
            f"The runs {windows_by_run.index.tolist()} do not cover the same windows; "
            "average only runs scored on the same backtest."
        )
    keys = [column for column in (WINDOW_COL, CUTOFF_COL, TARGET_NAME_COL) if column in metrics]
    numeric = [
        column
        for column in metrics.columns
        if column not in keys and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    averaged = metrics.groupby(keys, sort=True, as_index=False)[numeric].mean()
    averaged.insert(0, MODEL_COL, name)
    return averaged


def summarize_runs(
    metrics: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    *,
    baseline: str = "zero-shot",
    metric: str = "WQL",
    block_length: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score groups of runs against ``baseline``: each group averaged, plus its spread.

    ``groups`` maps a name (``"lora_r8"``) to the models of ``metrics`` that are runs of that
    configuration (its seeds). Each group is averaged window by window with
    :func:`average_runs` and scored like a model; the single runs are scored too, so the
    summary shows how far one run can sit from the average.

    Returns ``(summary, averaged)``. ``summary`` has one row per group: ``model``, ``n_runs``,
    the mean ``metric``, ``mean_change`` (mean metric over the baseline's, minus one: the
    arithmetic counterpart of the skill, dominated by the windows with large errors),
    ``skill`` with ``ci_low``, ``ci_high`` and ``win_rate`` (the geometric one, of the averaged
    runs), ``run_skill_min`` and ``run_skill_max`` (the skill of the worst and best single
    run) and the coverage columns present. ``averaged`` holds the baseline's rows and the
    averaged rows of every group, ready for :func:`skill_table`.
    """
    _check_is_frame(metrics, "metrics")
    _check_columns(metrics, [MODEL_COL, WINDOW_COL, metric], "metrics")
    if not groups:
        raise ValueError("groups is empty.")
    available = set(metrics[MODEL_COL].unique())
    missing = sorted({baseline, *(run for runs in groups.values() for run in runs)} - available)
    if missing:
        raise ValueError(f"metrics has no rows for {missing}; available: {sorted(available)}.")
    if baseline in groups:
        raise ValueError(f"A group cannot be named like the baseline {baseline!r}.")
    reference = metrics[metrics[MODEL_COL] == baseline]
    averaged = pd.concat(
        [reference]
        + [
            average_runs(metrics[metrics[MODEL_COL].isin(list(runs))], name)
            for name, runs in groups.items()
        ],
        ignore_index=True,
    )
    means = averaged.groupby(MODEL_COL)[
        [column for column in (metric, *COVERAGE_COLUMNS) if column in averaged]
    ].mean()
    rows = []
    for name, runs in groups.items():
        score = skill_score(
            averaged, model=name, baseline=baseline, metric=metric, block_length=block_length
        )
        run_skills = [
            skill_score(metrics, model=run, baseline=baseline, metric=metric).skill for run in runs
        ]
        row: dict[str, object] = {
            MODEL_COL: name,
            "n_runs": len(runs),
            metric: float(means.loc[name, metric]),
            "mean_change": float(means.loc[name, metric] / means.loc[baseline, metric] - 1.0),
            "skill": score.skill,
            "ci_low": score.ci_low,
            "ci_high": score.ci_high,
            "win_rate": score.win_rate,
            "run_skill_min": min(run_skills),
            "run_skill_max": max(run_skills),
        }
        row |= {
            column: float(means.loc[name, column]) for column in COVERAGE_COLUMNS if column in means
        }
        rows.append(row)
    return pd.DataFrame(rows), averaged
