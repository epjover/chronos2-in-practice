"""Baseline forecasters behind the shared ``Forecaster`` protocol.

- ``SeasonalNaive``: repeats the last season (the reference of every skill score).
- ``AutoETS``: automatic exponential smoothing from statsforecast.
- ``MSTL``: a decomposition with several seasonal periods (daily and weekly for hourly data),
  the classical model that fits a series with two cycles.
- ``ChronosBolt``: the previous Chronos generation, univariate only.
- ``PublishedForecast``: a forecast somebody else published before the cutoff (a grid
  operator's day-ahead forecast), replayed as a model so it can sit in the same table.

The first four are univariate: each (item, target) pair is forecast on its own, multiple
targets are handled by a loop, and covariate columns and ``future_covariates`` are ignored.
``PublishedForecast`` is the opposite: it reads only ``future_covariates``.
``ResidualForecaster`` combines the two: any forecaster, run on what a published forecast got
wrong.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast import models as sf_models

from c2p.models import (
    CHRONOS_BOLT_BASE,
    finalize_forecast,
    load_chronos_bolt,
    validate_forecast_inputs,
    warn_untrained_levels,
)
from c2p.schema import (
    ID_COL,
    PREDICTION_COL,
    TARGET_COL,
    TARGET_NAME_COL,
    TIMESTAMP_COL,
    Forecaster,
    quantile_col,
)

if TYPE_CHECKING:
    from chronos import ChronosBoltPipeline

__all__ = [
    "MSTL",
    "AutoETS",
    "ChronosBolt",
    "KnownUntil",
    "PublishedForecast",
    "ResidualForecaster",
    "SeasonalNaive",
]

# statsforecast's ETS only fits seasonal models up to this period; above it, AutoETS silently
# falls back to a non-seasonal model.
_MAX_ETS_SEASON_LENGTH = 24
# statsforecast's ETS refuses shorter series (NotImplementedError: "tiny datasets").
_MIN_ETS_POINTS = 7


# --------------------------------------------------------------------------------------------
# statsforecast output -> forecast frame
# --------------------------------------------------------------------------------------------


def _interval_level(quantile_level: float) -> int | float:
    """Statsforecast ``level`` (in %) of the central interval bounded by ``quantile_level``.

    Level L has bounds at quantiles (1 - L/100) / 2 and 1 - (1 - L/100) / 2, so the 0.1 and 0.9
    quantiles are the bounds of the 80% interval. Rounded so 0.15 maps to 70, not 70.00000000001.
    """
    level = round(abs(2 * quantile_level - 1) * 100, 8)
    return int(level) if level.is_integer() else level


def _statsforecast_levels(quantile_levels: Sequence[float]) -> list[int | float]:
    """Interval levels to request so that every quantile except the median is a bound."""
    return sorted({_interval_level(q) for q in quantile_levels if q != 0.5})


def _statsforecast_column(alias: str, quantile_level: float) -> str:
    """Column of a statsforecast output holding ``quantile_level`` (median = point forecast)."""
    if quantile_level == 0.5:
        return alias
    side = "lo" if quantile_level < 0.5 else "hi"
    return f"{alias}-{side}-{_interval_level(quantile_level)}"


def _from_statsforecast(
    raw: pd.DataFrame, *, alias: str, target: str, quantile_levels: Sequence[float]
) -> pd.DataFrame:
    """Convert a statsforecast frame (columns ``alias``, ``alias-lo-80``, ...) to a forecast frame.

    The point forecast becomes ``predictions`` and the 0.5 quantile. Quantiles are sorted across
    levels for every row, so they are monotone even if a model returned crossing intervals (a
    no-op for the symmetric normal intervals of SeasonalNaive and AutoETS).
    """
    values = np.column_stack(
        [raw[_statsforecast_column(alias, q)].to_numpy(dtype="float64") for q in quantile_levels]
    )
    order = np.argsort(quantile_levels)
    values[:, order] = np.sort(values[:, order], axis=1)
    columns = {
        ID_COL: raw[ID_COL].to_numpy(),
        TIMESTAMP_COL: raw[TIMESTAMP_COL].to_numpy(),
        TARGET_NAME_COL: target,
        PREDICTION_COL: raw[alias].to_numpy(dtype="float64"),
    }
    quantiles = {quantile_col(q): values[:, i] for i, q in enumerate(quantile_levels)}
    return pd.DataFrame(columns | quantiles)


class _StatsForecastBaseline(ABC):
    """Shared ``predict`` of the statsforecast baselines: one StatsForecast call per target."""

    name: str

    @abstractmethod
    def _model(self) -> Any:
        """The statsforecast model instance to fit (its ``alias`` names the output columns)."""

    @abstractmethod
    def _prepare_series(self, values: np.ndarray, label: str) -> np.ndarray:
        """Check one series and return the values given to statsforecast.

        The result may drop the oldest values: it is aligned with the most recent timestamps.
        ``label`` names the (item, target) pair in error messages.
        """

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
        """Forecast each (item, target) independently; ``future_covariates`` is ignored."""
        targets, levels = validate_forecast_inputs(
            context,
            target=target,
            quantile_levels=quantile_levels,
            prediction_length=prediction_length,
            freq=freq,
        )
        frames = [
            self._predict_target(context, name, prediction_length, levels, freq) for name in targets
        ]
        return finalize_forecast(
            pd.concat(frames, ignore_index=True),
            context=context,
            targets=targets,
            quantile_levels=levels,
            prediction_length=prediction_length,
            freq=freq,
            model_name=self.name,
        )

    def _predict_target(
        self,
        context: pd.DataFrame,
        target: str,
        prediction_length: int,
        quantile_levels: list[float],
        freq: str,
    ) -> pd.DataFrame:
        model = self._model()
        series = self._prepare(context[[ID_COL, TIMESTAMP_COL, target]], target)
        raw = StatsForecast(models=[model], freq=freq).forecast(
            h=prediction_length,
            df=series,
            level=_statsforecast_levels(quantile_levels) or None,
            id_col=ID_COL,
            time_col=TIMESTAMP_COL,
            target_col=target,
        )
        return _from_statsforecast(
            raw, alias=model.alias, target=target, quantile_levels=quantile_levels
        )

    def _prepare(self, series: pd.DataFrame, target: str) -> pd.DataFrame:
        """Return the (id, timestamp, target) frame given to statsforecast, series by series."""
        parts = []
        for item_id, group in series.groupby(ID_COL, sort=False):
            label = f"{item_id!r}/{target!r}"
            values = self._prepare_series(group[target].to_numpy(dtype="float64"), label)
            parts.append(group.tail(len(values)).assign(**{target: values}))
        return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------------------------
# Seasonal naive
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonalNaive(_StatsForecastBaseline):
    """Seasonal naive: every step repeats the value observed one season earlier.

    This is statsforecast's ``SeasonalNaive`` with its native normal intervals, the baseline
    fev-bench uses for its skill scores. We checked the alternative of empirical residual
    quantiles and kept statsforecast for comparability with the paper: its quantiles are
    ``point + z_q * sigma * sqrt(k)``, where ``sigma`` is the RMS of the in-sample seasonal
    differences ``y[t] - y[t - season_length]`` and ``k`` is the season the step falls in
    (1 for the first ``season_length`` steps). They are symmetric around the point forecast
    (so the median equals ``predictions``), monotone across levels and widen with the horizon.

    Missing values: statsforecast copies the last season verbatim, so a NaN there would become
    a NaN forecast. Only NaNs in the last season are filled, each with the latest observed value
    at the same position of the season (what seasonal naive would have predicted for it).
    statsforecast divides the squared differences by ``n - season_length``, counting missing
    differences (and the exact zeros the fill creates) as zeros, so ``sigma`` would shrink as
    the share of NaN grows; the intervals are rescaled so that ``sigma`` is the RMS of the
    observed differences only. On a complete series nothing changes (fev-bench parity). Each
    series needs at least one pair of observed values one season apart.

    Parameters
    ----------
    season_length
        Steps per season, e.g. 24 for a daily cycle in hourly data or 168 for a weekly one.
    name
        Model name used in results tables.
    """

    season_length: int
    name: str = "seasonal_naive"

    def __post_init__(self) -> None:
        if self.season_length < 1:
            raise ValueError(f"season_length must be >= 1, got {self.season_length}.")

    def _model(self) -> sf_models.SeasonalNaive:
        return sf_models.SeasonalNaive(season_length=self.season_length, alias="SeasonalNaive")

    def _prepare_series(self, values: np.ndarray, label: str) -> np.ndarray:
        if _count_seasonal_differences(values, self.season_length) == 0:
            raise ValueError(
                f"Series {label} has no two observed values one season ({self.season_length} "
                f"steps) apart among its {len(values)} points, so seasonal naive cannot estimate "
                "its intervals. Provide more than one season of context."
            )
        return _fill_last_season(values, self.season_length, label=label)

    def _predict_target(
        self,
        context: pd.DataFrame,
        target: str,
        prediction_length: int,
        quantile_levels: list[float],
        freq: str,
    ) -> pd.DataFrame:
        forecast = super()._predict_target(
            context, target, prediction_length, quantile_levels, freq
        )
        scales = {
            item_id: _observed_sigma_scale(values.to_numpy(dtype="float64"), self.season_length)
            for item_id, values in context.groupby(ID_COL, sort=False)[target]
        }
        point = forecast[PREDICTION_COL].to_numpy()
        scale = forecast[ID_COL].map(scales).to_numpy(dtype="float64")
        rescaled = {
            quantile_col(q): point + (forecast[quantile_col(q)].to_numpy() - point) * scale
            for q in quantile_levels
        }
        return forecast.assign(**rescaled)


def _count_seasonal_differences(values: np.ndarray, season_length: int) -> int:
    """Number of observed seasonal differences ``y[t] - y[t - season_length]``."""
    return int(np.isfinite(values[season_length:] - values[:-season_length]).sum())


def _observed_sigma_scale(values: np.ndarray, season_length: int) -> float:
    """Factor turning statsforecast's seasonal-naive ``sigma`` into the RMS of observed differences.

    statsforecast divides the sum of squared differences by ``n - season_length`` although only
    ``n_observed`` of them are observed; the rest are NaN or zeros created by filling the last
    season. Hence the factor ``sqrt((n - season_length) / n_observed)``, 1 for a complete series.
    """
    n_observed = _count_seasonal_differences(values, season_length)
    return float(np.sqrt((len(values) - season_length) / n_observed))


def _fill_last_season(values: np.ndarray, season_length: int, *, label: str) -> np.ndarray:
    """Fill NaNs of the last season with the latest observed value at the same seasonal position."""
    position = np.arange(len(values)) % season_length
    latest = pd.Series(values).groupby(position).ffill().to_numpy()
    filled = values.copy()
    filled[-season_length:] = latest[-season_length:]
    unobserved = int(np.isnan(filled[-season_length:]).sum())
    if unobserved:
        raise ValueError(
            f"Series {label} has never been observed at {unobserved} of the {season_length} "
            "positions of the season, so seasonal naive cannot forecast them."
        )
    return filled


# --------------------------------------------------------------------------------------------
# AutoETS
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoETS(_StatsForecastBaseline):
    """Automatic exponential smoothing (statsforecast ``AutoETS``) with normal intervals.

    The model (error, trend and season types) is chosen per series by AICc. Intervals from
    ``level=`` are converted to quantiles: level L gives the quantiles (1 - L/100) / 2 and
    1 - (1 - L/100) / 2, and the point forecast is used as the median.

    Missing values: statsforecast's ETS fails on NaN, so, *only inside this baseline*, NaNs are
    filled by linear interpolation (leading and trailing NaNs take the nearest observed value).

    Short contexts: each series needs at least ``min_context`` points (two seasons, and at least
    7). With fewer, the seasonal component has about as many parameters as there are points (or
    cannot be fitted at all) and statsforecast silently fits a non-seasonal model, so this is
    an error. With only a few seasons, AICc may still prefer a non-seasonal model; that is model
    selection, not a fallback.

    Parameters
    ----------
    season_length
        Steps per season, at most 24: statsforecast cannot fit seasonal ETS models with longer
        seasons and would silently fit a non-seasonal one (use 24 for hourly data).
    name
        Model name used in results tables.
    max_context
        Fit on the last ``max_context`` steps of each series only (fitting time grows with the
        length); the default 2,016 is 12 weeks of hourly data. ``None`` uses the whole context.
        Must be at least ``min_context``.
    """

    season_length: int
    name: str = "auto_ets"
    max_context: int | None = 2016

    def __post_init__(self) -> None:
        if not 1 <= self.season_length <= _MAX_ETS_SEASON_LENGTH:
            raise ValueError(
                f"season_length must be in [1, {_MAX_ETS_SEASON_LENGTH}], got "
                f"{self.season_length}: statsforecast drops the seasonal component of ETS models "
                "with longer seasons. Use season_length=24 for hourly data."
            )
        if self.max_context is not None and self.max_context < self.min_context:
            raise ValueError(
                f"max_context must be at least {self.min_context} (two seasons of "
                f"{self.season_length} steps, and at least {_MIN_ETS_POINTS}), "
                f"got {self.max_context}."
            )

    @property
    def min_context(self) -> int:
        """Fewest points per series: two seasons, and statsforecast's minimum of 7."""
        return max(2 * self.season_length, _MIN_ETS_POINTS)

    def _model(self) -> sf_models.AutoETS:
        return sf_models.AutoETS(season_length=self.season_length, alias="AutoETS")

    def _prepare_series(self, values: np.ndarray, label: str) -> np.ndarray:
        if self.max_context is not None:
            values = values[-self.max_context :]
        if len(values) < self.min_context:
            raise ValueError(
                f"Series {label} has {len(values)} points, but AutoETS(season_length="
                f"{self.season_length}) needs at least {self.min_context}: with fewer than two "
                "seasons the seasonal component cannot be estimated and statsforecast silently "
                "fits a non-seasonal model."
            )
        return _interpolate_linear(values, label=label)


def _interpolate_linear(values: np.ndarray, *, label: str) -> np.ndarray:
    """Linearly interpolate NaNs; leading/trailing NaNs take the nearest observed value."""
    if np.isnan(values).all():
        raise ValueError(f"Series {label} has no observed values in the context.")
    filled = pd.Series(values).interpolate(method="linear", limit_direction="both")
    return filled.to_numpy()


# --------------------------------------------------------------------------------------------
# MSTL
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MSTL(_StatsForecastBaseline):
    """Multiple seasonal-trend decomposition (statsforecast ``MSTL``) with normal intervals.

    The series is split by LOESS into one seasonal component per period, a trend and a
    remainder. Each seasonal component is carried forward by repeating its last cycle, and the
    trend plus remainder is forecast by a non-seasonal ``AutoETS``, whose intervals become the
    quantiles (level L gives the quantiles (1 - L/100) / 2 and 1 - (1 - L/100) / 2).

    This is the classical model that fits a series with a daily and a weekly cycle, which
    ``AutoETS`` cannot represent (statsforecast's ETS stops at a period of 24).

    Missing values are filled by linear interpolation, as in ``AutoETS`` and only inside this
    baseline. Each series needs two cycles of the longest period.

    Parameters
    ----------
    season_lengths
        Seasonal periods in steps, e.g. ``(24, 168)`` for hourly data.
    name
        Model name used in results tables.
    max_context
        Fit on the last ``max_context`` steps of each series only; the default 2,016 is 12
        weeks of hourly data. ``None`` uses the whole context. Must be at least
        ``min_context``.
    """

    season_lengths: tuple[int, ...]
    name: str = "mstl"
    max_context: int | None = 2016

    def __post_init__(self) -> None:
        lengths = tuple(self.season_lengths)
        if not lengths or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in lengths
        ):
            raise ValueError(
                f"season_lengths must be integers of at least 2, got {self.season_lengths!r}."
            )
        if len(set(lengths)) != len(lengths):
            raise ValueError(f"season_lengths has repeated periods: {list(lengths)}.")
        object.__setattr__(self, "season_lengths", lengths)
        if self.max_context is not None and self.max_context < self.min_context:
            raise ValueError(
                f"max_context must be at least {self.min_context} (two cycles of the longest "
                f"period, {max(lengths)} steps), got {self.max_context}."
            )

    @property
    def min_context(self) -> int:
        """Fewest points per series: two cycles of the longest seasonal period."""
        return 2 * max(self.season_lengths)

    def _model(self) -> sf_models.MSTL:
        return sf_models.MSTL(
            season_length=list(self.season_lengths),
            trend_forecaster=sf_models.AutoETS(model="ZZN"),
            alias="MSTL",
        )

    def _prepare_series(self, values: np.ndarray, label: str) -> np.ndarray:
        if self.max_context is not None:
            values = values[-self.max_context :]
        if len(values) < self.min_context:
            raise ValueError(
                f"Series {label} has {len(values)} points, but MSTL(season_lengths="
                f"{self.season_lengths}) needs at least {self.min_context}: two cycles of the "
                "longest period."
            )
        return _interpolate_linear(values, label=label)


# --------------------------------------------------------------------------------------------
# A published forecast, replayed
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishedForecast:
    """A forecast published before the cutoff, replayed as a model.

    Some datasets carry the forecast a professional made at the time: the Spanish table has
    the grid operator's day-ahead load forecast next to the load. Passing that column to the
    backtest as a known covariate and wrapping it in this class scores it on the same windows
    and with the same metrics as every other model, which is the strongest reference a table
    can have.

    The published value is a point forecast, so it is returned as ``predictions`` and as every
    quantile. Read the metrics accordingly: MASE and WAPE are the published forecast's own;
    WQL of a forecast with no spread equals its WAPE when the levels are symmetric around the
    median, so it is still comparable with the WQL of a probabilistic model; coverage is not
    meaningful (an interval of zero width covers nothing).

    Only use it with a column that was available at the cutoff. The class cannot check that.

    Parameters
    ----------
    column
        Name of the known-future covariate that holds the published forecast.
    name
        Model name used in results tables.
    """

    column: str
    name: str = "published_forecast"

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
        """Return the published column of ``future_covariates`` as the forecast of one target."""
        targets, levels = validate_forecast_inputs(
            context,
            target=target,
            quantile_levels=quantile_levels,
            prediction_length=prediction_length,
            freq=freq,
        )
        if len(targets) != 1:
            raise ValueError(
                f"{self.name} replays one published forecast for one target, got {targets}."
            )
        if future_covariates is None or self.column not in future_covariates.columns:
            raise ValueError(
                f"{self.name} needs {self.column!r} in future_covariates: declare it in "
                "BacktestConfig(known_covariates=...)."
            )
        published = future_covariates[self.column].to_numpy(dtype="float64")
        if np.isnan(published).any():
            raise ValueError(
                f"{self.column!r} has missing values in the horizon; a published forecast with "
                "holes cannot be scored as a model."
            )
        columns = {
            ID_COL: future_covariates[ID_COL].to_numpy(),
            TIMESTAMP_COL: future_covariates[TIMESTAMP_COL].to_numpy(),
            TARGET_NAME_COL: targets[0],
            PREDICTION_COL: published,
        }
        quantiles = {quantile_col(level): published for level in levels}
        return finalize_forecast(
            pd.DataFrame(columns | quantiles),
            context=context,
            targets=targets,
            quantile_levels=levels,
            prediction_length=prediction_length,
            freq=freq,
            model_name=self.name,
        )


class ResidualForecaster:
    """Forecast the error of a published forecast with ``base``, and add the forecast back.

    The context's target is replaced by ``target - column`` (what the published forecast got
    wrong in the past), ``base`` forecasts that residual on its own, and the published values
    for the horizon are added to its point forecast and to every quantile. The published
    forecast carries the level and the shape; the model only has to learn whether its errors
    persist. The quantiles are therefore the model's uncertainty about the residual, placed
    around the published value.

    Parameters
    ----------
    base
        Any ``Forecaster``; it receives the residual as a single univariate target.
    column
        Known-future covariate that holds the published forecast (present in the context too).
    name
        Model name used in results tables; defaults to ``"<base.name>_on_<column>_error"``.
    """

    def __init__(self, base: Forecaster, column: str, *, name: str | None = None) -> None:
        self.base = base
        self.column = column
        self.name = name or f"{base.name}_on_{column}_error"

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
        """Forecast one target as the published forecast plus ``base``'s forecast of its error."""
        targets, levels = validate_forecast_inputs(
            context,
            target=target,
            quantile_levels=quantile_levels,
            prediction_length=prediction_length,
            freq=freq,
        )
        if len(targets) != 1:
            raise ValueError(f"{self.name} corrects one published forecast, got {targets}.")
        if self.column not in context.columns:
            raise ValueError(f"{self.name} needs {self.column!r} in the context.")
        if future_covariates is None or self.column not in future_covariates.columns:
            raise ValueError(
                f"{self.name} needs {self.column!r} in future_covariates: declare it in "
                "BacktestConfig(known_covariates=...)."
            )
        name = targets[0]
        residual = context[[ID_COL, TIMESTAMP_COL]].assign(
            **{name: context[name].to_numpy("float64") - context[self.column].to_numpy("float64")}
        )
        forecast = self.base.predict(
            residual,
            prediction_length=prediction_length,
            quantile_levels=levels,
            freq=freq,
            target=name,
        )
        published = future_covariates[[ID_COL, TIMESTAMP_COL, self.column]]
        merged = forecast.merge(published, on=[ID_COL, TIMESTAMP_COL], how="left", validate="1:1")
        offset = merged[self.column].to_numpy("float64")
        if np.isnan(offset).any():
            raise ValueError(f"{self.column!r} has missing values in the horizon.")
        for column in (PREDICTION_COL, *(quantile_col(level) for level in levels)):
            merged[column] = merged[column].to_numpy("float64") + offset
        return finalize_forecast(
            merged.drop(columns=self.column),
            context=context,
            targets=targets,
            quantile_levels=levels,
            prediction_length=prediction_length,
            freq=freq,
            model_name=self.name,
        )


class KnownUntil:
    """Let ``base`` know its future covariates only for the first ``steps`` steps of the horizon.

    The known covariates of every later step are replaced by NaN before ``base`` sees them.
    ``Chronos2Pipeline.predict_df`` reads a NaN in ``future_df`` as "not known at this step"
    (a ``future_df`` that is NaN everywhere gives the same forecast as no ``future_df``), so
    this is how a covariate whose own forecast is shorter than the horizon reaches the model:
    a day-ahead weather forecast under a week-ahead horizon, a promotion calendar that stops at
    the end of the month.

    Parameters
    ----------
    base
        Any ``Forecaster`` that accepts NaN in its future covariates (``Chronos2Forecaster``).
    steps
        Number of leading horizon steps for which the covariates stay known; 0 hides them all.
    name
        Model name used in results tables; defaults to ``"<base.name>_known_until_<steps>"``.
    """

    def __init__(self, base: Forecaster, steps: int, *, name: str | None = None) -> None:
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}.")
        self.base = base
        self.steps = steps
        self.name = name or f"{base.name}_known_until_{steps}"

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
        """Forecast with the future covariates hidden after the first ``steps`` steps."""
        if future_covariates is None:
            raise ValueError(
                f"{self.name} needs future_covariates: declare them in "
                "BacktestConfig(known_covariates=...)."
            )
        future = future_covariates.copy()
        step = future.groupby(ID_COL, sort=False, observed=True)[TIMESTAMP_COL].rank(method="first")
        late = (step > self.steps).to_numpy()
        for column in future.columns.difference([ID_COL, TIMESTAMP_COL]):
            if pd.api.types.is_numeric_dtype(future[column]):
                future[column] = future[column].astype("float64")
            future.loc[late, column] = np.nan
        return self.base.predict(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            freq=freq,
            target=target,
            future_covariates=future,
        )


# --------------------------------------------------------------------------------------------
# Chronos-Bolt
# --------------------------------------------------------------------------------------------


class _ClampNotice(logging.Filter):
    """Drop the library's per-call notice about clamped quantile levels.

    ``chronos_bolt`` logs it at every ``predict`` call, 365 times in a year of daily windows.
    ``warn_untrained_levels`` says the same once per forecaster and adds the consequence.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "are not within the range of quantiles" not in record.getMessage()


def _silence_clamp_notice() -> None:
    """Attach :class:`_ClampNotice` to the library's logger, once."""
    from chronos import chronos_bolt

    if not any(isinstance(item, _ClampNotice) for item in chronos_bolt.logger.filters):
        chronos_bolt.logger.addFilter(_ClampNotice())


class ChronosBolt:
    """Chronos-Bolt (``amazon/chronos-bolt-*``) as a univariate zero-shot baseline.

    Uses the pipeline's own ``predict_df`` once per target, passing ``freq`` (after checking the
    regular grid, since ``freq`` disables the library's gap detection) and only the id, timestamp
    and target columns: Bolt has no covariate support.

    Bolt natively predicts the quantiles 0.1, 0.2, ..., 0.9. Levels in between are interpolated by
    the library; levels outside [0.1, 0.9] are clamped (``ChronosBoltPipeline.predict_quantiles``
    pads the trained quantiles with copies of the 0.1 and 0.9 ones before interpolating), so e.g.
    the 0.01 and 0.99 quantiles equal the 0.1 and 0.9 ones and a 98% interval is really the 80%
    one. They are returned as produced and a warning is logged.

    Parameters
    ----------
    pipeline
        A loaded ``ChronosBoltPipeline``; when ``None`` it is loaded from ``model_id``.
    model_id
        Hub id or local path, used only when ``pipeline`` is ``None``.
    device
        ``"cpu"`` or ``"cuda"`` for loading; ``None`` uses ``c2p.config.get_device()``.
    name
        Model name used in results tables.
    """

    def __init__(
        self,
        pipeline: ChronosBoltPipeline | None = None,
        *,
        model_id: str = CHRONOS_BOLT_BASE,
        device: str | None = None,
        name: str = "chronos-bolt",
    ) -> None:
        from chronos import ChronosBoltPipeline

        if pipeline is None:
            pipeline = load_chronos_bolt(model_id, device=device)
        elif device is not None or model_id != CHRONOS_BOLT_BASE:
            raise ValueError("Pass either a loaded pipeline or model_id/device to load, not both.")
        if not isinstance(pipeline, ChronosBoltPipeline):
            raise TypeError(
                f"Expected a ChronosBoltPipeline, got {type(pipeline).__name__}. "
                "Use c2p.models.Chronos2Forecaster for Chronos-2 models."
            )
        self.pipeline = pipeline
        self.name = name
        self._reported_levels: set[tuple[float, ...]] = set()
        _silence_clamp_notice()

    def __repr__(self) -> str:
        return f"ChronosBolt(name={self.name!r})"

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
        """Forecast each (item, target) independently; ``future_covariates`` is ignored."""
        targets, levels = validate_forecast_inputs(
            context,
            target=target,
            quantile_levels=quantile_levels,
            prediction_length=prediction_length,
            freq=freq,
        )
        warn_untrained_levels(
            levels, self.pipeline.quantiles, model_name=self.name, seen=self._reported_levels
        )
        frames = [
            self.pipeline.predict_df(
                context[[ID_COL, TIMESTAMP_COL, name]],
                id_column=ID_COL,
                timestamp_column=TIMESTAMP_COL,
                target=name,
                prediction_length=prediction_length,
                quantile_levels=levels,
                freq=freq,
            )
            for name in targets
        ]
        return finalize_forecast(
            pd.concat(frames, ignore_index=True),
            context=context,
            targets=targets,
            quantile_levels=levels,
            prediction_length=prediction_length,
            freq=freq,
            model_name=self.name,
        )
