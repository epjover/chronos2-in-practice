"""What drives a forecast: Shapley values of covariates and context segments, and cheap baselines.

Any :class:`~c2p.schema.Forecaster` can be explained. A *player* is a part of the model input
that can be switched off: a group of covariate columns (:class:`CovariateGroup`) or a block of
past target values (:class:`ContextSegment`). A *coalition* is a set of players left on, and its
*value* is computed from the forecast made with only those players: the mean forecast over the
horizon, the forecast at every step, or the reduction in WQL compared with the empty coalition.
Exact Shapley values average each player's marginal contribution over all ``2**n`` coalitions,
so they are only affordable for a few players (at most :data:`MAX_EXACT_PLAYERS`): grouping the
inputs into a handful of meaningful players is what makes them practical.

This follows Hertel et al. (2026), "Explainable Load Forecasting with Covariate-Informed Time
Series Foundation Models", arXiv:2604.28149, who explain Chronos-2 with exact Shapley values over
four context segments (last day, rest of the week, rest of four weeks, older) and three
covariates.

How players are switched off:

- Covariates: ``masking="drop"`` removes the columns, as Hertel et al. do, so the empty coalition
  is the forecast without covariates. ``masking="nan"`` keeps the columns with every value set
  to NaN. Chronos-2 treats NaN as missing, but an all-NaN covariate is still a series in the
  group, so the forecast is not exactly the one without that column (and a categorical NaN is a
  category of its own).
- Context segments: target values inside the segment are set to NaN. For Chronos-2, NaN before
  the remaining observations is equivalent to a shorter context once the covariates' history is
  blanked too (``mask_covariate_history=True``), which is how Hertel et al. mask the oldest
  segment. Without any target value the model has nothing to anchor the level on (Chronos-2
  then forecasts values near 0), so by default those coalitions forecast the mean of the full
  context instead (``empty_context="context_mean"``; Hertel et al. use a long-run mean too).

Forecasting coalitions one at a time is slow, so coalitions that share the same columns are sent
together in a single ``predict`` call, as separate series with temporary ids (``batch=True``).
That gives the same forecasts only when the forecaster treats series independently (Chronos-2
without cross-learning, univariate baselines): a forecaster whose ``cross_learning`` attribute is
true is rejected unless ``batch=False``.

Example::

    explanation = covariate_shapley(
        forecaster, window.context, window.future_covariates,
        covariates=["load_forecast_tso", "solar_forecast_tso", "hour"],
        prediction_length=24, freq="h", target="price_actual", value="prediction",
    )
    explanation.importance()  # one row per covariate, largest mean |Shapley value| first
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, get_args

import numpy as np
import pandas as pd

from c2p.config import SEED
from c2p.eval import METRICS, SCORED_QUANTILES, Window
from c2p.schema import ID_COL, PREDICTION_COL, TARGET_COL, TIMESTAMP_COL, Forecaster, quantile_col

__all__ = [
    "COALITION_COL",
    "IMPORTANCE_COL",
    "MAX_EXACT_PLAYERS",
    "METRIC_COL",
    "PLAYER_COL",
    "SHAPLEY_COL",
    "SIZE_COL",
    "VALUE_COL",
    "ContextSegment",
    "CovariateGroup",
    "ShapleyExplanation",
    "ablation_importance",
    "context_segments",
    "covariate_players",
    "covariate_shapley",
    "exact_shapley",
    "importance_summary",
    "occlusion",
    "permutation_importance",
    "shapley_values",
    "stack_windows",
    "temporal_shapley",
]

logger = logging.getLogger(__name__)

MAX_EXACT_PLAYERS = 12

PLAYER_COL = "player"
SHAPLEY_COL = "shapley"
COALITION_COL = "coalition"
SIZE_COL = "size"
VALUE_COL = "value"
IMPORTANCE_COL = "importance"
METRIC_COL = "metric"

Masking = Literal["nan", "drop"]
ValueName = Literal["prediction", "prediction_mean", "wql_reduction"]
EmptyContext = Literal["context_mean", "model"]

_REQUEST_COL = "_request"  # position of a masked input in the list sent to the forecaster
_TEMPORARY_ID = "_temporary_id"
_WQL = METRICS["WQL"]
_SCORED_COLUMNS = [quantile_col(level) for level in SCORED_QUANTILES]

# A masked input: context and future covariates (None when there is no known covariate).
_Input = tuple[pd.DataFrame, pd.DataFrame | None]


# --------------------------------------------------------------------------------------------
# Exact Shapley values
# --------------------------------------------------------------------------------------------


def exact_shapley(
    players: Sequence[str], value_fn: Callable[[frozenset[str]], float]
) -> dict[str, float]:
    """Exact Shapley values of a cooperative game by enumerating every coalition.

    ``value_fn`` is called exactly once per coalition (``2**len(players)`` calls), so it may be
    expensive. The values satisfy efficiency: they sum to ``v(all players) - v(no player)``.

    Raises ``ValueError`` for duplicated names, no player, more than :data:`MAX_EXACT_PLAYERS`
    players (group them instead) or a non-finite coalition value.
    """
    names = _check_player_names(players)
    table = np.empty((1 << len(names), 1))
    for mask in range(len(table)):
        coalition = frozenset(_members(mask, names))
        value = float(value_fn(coalition))
        if not math.isfinite(value):
            raise ValueError(f"value_fn returned {value} for the coalition {sorted(coalition)}.")
        table[mask, 0] = value
    return dict(zip(names, _shapley_matrix(table, len(names))[:, 0].tolist(), strict=True))


def _check_player_names(players: Sequence[str]) -> tuple[str, ...]:
    if isinstance(players, str):
        raise TypeError("players must be a sequence of names, not a single string.")
    names = tuple(players)
    if not names:
        raise ValueError("At least one player is required.")
    if not all(isinstance(name, str) and name for name in names):
        raise TypeError(f"Player names must be non-empty strings, got {names!r}.")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicated player names: {names}.")
    if len(names) > MAX_EXACT_PLAYERS:
        raise ValueError(
            f"{len(names)} players need {2 ** len(names):,} coalitions; exact Shapley values "
            f"are limited to {MAX_EXACT_PLAYERS} players. Group covariates or context segments "
            "into fewer players."
        )
    return names


def _members(mask: int, names: Sequence[str]) -> tuple[str, ...]:
    """Players in the coalition encoded by ``mask`` (bit ``i`` set = player ``i`` present)."""
    return tuple(name for i, name in enumerate(names) if mask >> i & 1)


def _shapley_matrix(table: np.ndarray, n_players: int) -> np.ndarray:
    """Shapley values ``[n_players, m]`` of ``m`` games given their values ``[2**n_players, m]``.

    Row ``mask`` of ``table`` is the value of the coalition encoded by ``mask``. Player ``i``
    receives the sum over coalitions ``S`` without ``i`` of
    ``|S|! (n - |S| - 1)! / n! * (v(S + i) - v(S))``.
    """
    n_coalitions = 1 << n_players
    if table.shape[0] != n_coalitions:
        raise ValueError(f"Expected {n_coalitions} coalition values, got {table.shape[0]}.")
    masks = np.arange(n_coalitions)
    sizes = np.array([mask.bit_count() for mask in range(n_coalitions)])
    weight_by_size = np.array(
        [
            math.factorial(size) * math.factorial(n_players - size - 1) / math.factorial(n_players)
            for size in range(n_players)
        ]
    )
    shapley = np.empty((n_players, table.shape[1]))
    for player in range(n_players):
        bit = 1 << player
        without = masks[(masks & bit) == 0]
        shapley[player] = weight_by_size[sizes[without]] @ (table[without | bit] - table[without])
    return shapley


# --------------------------------------------------------------------------------------------
# Players
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CovariateGroup:
    """A player made of one or more covariate columns, switched on or off together."""

    name: str
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.columns, str):
            raise TypeError(f"columns of {self.name!r} must be a sequence, not a string.")
        object.__setattr__(self, "columns", tuple(self.columns))
        if not self.name or not self.columns:
            raise ValueError("A covariate group needs a name and at least one column.")


@dataclass(frozen=True)
class ContextSegment:
    """A player made of the target values ``first_lag`` to ``last_lag`` steps before the cutoff.

    Lag 1 is the last observation of the context; ``last_lag=None`` reaches the oldest one.
    """

    name: str
    first_lag: int
    last_lag: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A context segment needs a name.")
        if self.first_lag < 1:
            raise ValueError(f"first_lag must be >= 1, got {self.first_lag} for {self.name!r}.")
        if self.last_lag is not None and self.last_lag < self.first_lag:
            raise ValueError(
                f"last_lag ({self.last_lag}) must be >= first_lag ({self.first_lag}) "
                f"for {self.name!r}."
            )

    def contains(self, lags: np.ndarray) -> np.ndarray:
        """Boolean mask of the rows whose lag falls inside the segment."""
        inside = lags >= self.first_lag
        return inside if self.last_lag is None else inside & (lags <= self.last_lag)


Player = CovariateGroup | ContextSegment


def covariate_players(
    covariates: Sequence[str] | Mapping[str, Sequence[str]],
) -> list[CovariateGroup]:
    """One player per covariate column, or per named group of columns when given a mapping.

    ``{"calendar": ["hour", "day_of_week"], "load": ["load_forecast_tso"]}`` makes two players.
    """
    if isinstance(covariates, str):
        raise TypeError("covariates must be a list of column names or a mapping, not a string.")
    if isinstance(covariates, Mapping):
        return [CovariateGroup(name, tuple(columns)) for name, columns in covariates.items()]
    return [CovariateGroup(column, (column,)) for column in covariates]


def context_segments(
    segments: int | Sequence[int], *, context_length: int | None = None
) -> list[ContextSegment]:
    """Contiguous blocks of the context, most recent first, named by their lags.

    - An int splits the last ``context_length`` steps into that many blocks of (nearly) equal
      length; the oldest block also takes anything older.
    - A sequence of increasing lags cuts the context there, with one last block for everything
      older: ``(24, 168, 672)`` gives "lags 1-24", "lags 25-168", "lags 169-672" and
      "lags 673+" on hourly data (last day, rest of the week, rest of four weeks, older): the
      segments of Hertel et al. (2026).
    """
    if isinstance(segments, bool):
        raise TypeError("segments must be an int or a sequence of lags.")
    if isinstance(segments, int | np.integer):
        boundaries = _equal_boundaries(int(segments), context_length)
    else:
        boundaries = [int(lag) for lag in segments]
        increasing = all(a < b for a, b in pairwise(boundaries))
        if not boundaries or boundaries[0] < 1 or not increasing:
            raise ValueError(f"Boundaries must be increasing positive lags, got {segments}.")
    firsts = [1, *(lag + 1 for lag in boundaries)]
    lasts: list[int | None] = [*boundaries, None]
    return [
        ContextSegment(_segment_name(first, last), first, last)
        for first, last in zip(firsts, lasts, strict=True)
    ]


def _equal_boundaries(n_segments: int, context_length: int | None) -> list[int]:
    if n_segments < 1:
        raise ValueError(f"segments must be >= 1, got {n_segments}.")
    if context_length is None:
        raise ValueError("context_length is required to split the context into equal segments.")
    if context_length < n_segments:
        raise ValueError(f"Cannot split {context_length} context steps into {n_segments} segments.")
    sizes = [len(block) for block in np.array_split(np.arange(context_length), n_segments)]
    return [int(lag) for lag in np.cumsum(sizes[:-1])]


def _segment_name(first: int, last: int | None) -> str:
    return f"lags {first}+" if last is None else f"lags {first}-{last}"


# --------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ShapleyExplanation:
    """Shapley values of one explanation run, with every coalition's value and forecast.

    - ``shapley``: ``item_id``, ``timestamp`` (only for ``value="prediction"``), ``player``,
      ``shapley``.
    - ``coalitions``: the same keys plus ``coalition`` (tuple of present players), ``size``
      and ``value``.
    - ``forecasts``: the forecast frame of every coalition, with a ``coalition`` column.
    - ``n_forecasts`` series forecasts were made in ``n_calls`` calls to ``predict``, which took
      ``seconds`` in total.
    """

    players: tuple[str, ...]
    value: str
    shapley: pd.DataFrame
    coalitions: pd.DataFrame
    forecasts: pd.DataFrame
    n_forecasts: int
    n_calls: int
    seconds: float

    def __repr__(self) -> str:
        return (
            f"ShapleyExplanation(players={self.players}, value={self.value!r}, "
            f"n_forecasts={self.n_forecasts}, n_calls={self.n_calls}, seconds={self.seconds:.2f})"
        )

    @property
    def keys(self) -> list[str]:
        """Columns identifying one explained quantity: item, plus timestamp for step values."""
        return [ID_COL, TIMESTAMP_COL] if TIMESTAMP_COL in self.shapley.columns else [ID_COL]

    def base_value(self) -> pd.Series:
        """Value of the empty coalition, indexed by :attr:`keys`."""
        return self._coalition_value(0)

    def full_value(self) -> pd.Series:
        """Value with every player present, indexed by :attr:`keys`."""
        return self._coalition_value(len(self.players))

    def _coalition_value(self, size: int) -> pd.Series:
        rows = self.coalitions.loc[self.coalitions[SIZE_COL] == size]
        return rows.set_index(self.keys)[VALUE_COL]

    def efficiency_gap(self) -> float:
        """Largest ``|sum of Shapley values - (full value - base value)|`` (0 up to rounding)."""
        total = self.shapley.groupby(self.keys, sort=False)[SHAPLEY_COL].sum()
        expected = (self.full_value() - self.base_value()).reindex(total.index)
        return float((total - expected).abs().max())

    def importance(self) -> pd.DataFrame:
        """One row per player, most important first.

        ``mean_abs`` is the mean absolute Shapley value over items (and steps), ``share_pct`` its
        share of the total in percent (the importance Hertel et al. report) and ``mean`` the
        signed mean, whose sign gives the direction of the effect.
        """
        grouped = self.shapley.groupby(PLAYER_COL, sort=False)[SHAPLEY_COL]
        table = pd.DataFrame(
            {"mean_abs": grouped.agg(lambda values: values.abs().mean()), "mean": grouped.mean()}
        ).reindex(list(self.players))
        total = table["mean_abs"].sum()
        table["share_pct"] = 100 * table["mean_abs"] / total if total > 0 else np.nan
        table = table.sort_values("mean_abs", ascending=False, kind="stable")
        table["rank"] = np.arange(1, len(table) + 1)
        return table.rename_axis(PLAYER_COL).reset_index()


# --------------------------------------------------------------------------------------------
# Shapley values of a forecaster
# --------------------------------------------------------------------------------------------


def covariate_shapley(
    forecaster: Forecaster,
    context: pd.DataFrame,
    future_covariates: pd.DataFrame | None,
    *,
    covariates: Sequence[str] | Mapping[str, Sequence[str]],
    prediction_length: int,
    freq: str,
    target: str = TARGET_COL,
    masking: Masking = "drop",
    value: ValueName = "prediction_mean",
    actuals: pd.DataFrame | None = None,
    batch: bool = True,
) -> ShapleyExplanation:
    """Shapley values of covariates (one player per column, or per group given a mapping).

    A coalition keeps its covariates and switches the others off with ``masking`` (see the
    module docstring) in both ``context`` and ``future_covariates``; covariates that are not
    players stay on. ``value="prediction_mean"`` explains the mean forecast over the horizon,
    ``"prediction"`` every step separately, and ``"wql_reduction"`` the reduction in WQL on
    ``actuals`` compared with no covariate (positive = the covariates help). See
    :func:`shapley_values` for the parameters.
    """
    return shapley_values(
        forecaster,
        context,
        future_covariates,
        players=covariate_players(covariates),
        prediction_length=prediction_length,
        freq=freq,
        target=target,
        masking=masking,
        value=value,
        actuals=actuals,
        batch=batch,
    )


def temporal_shapley(
    forecaster: Forecaster,
    context: pd.DataFrame,
    future_covariates: pd.DataFrame | None = None,
    *,
    segments: int | Sequence[int],
    prediction_length: int,
    freq: str,
    target: str = TARGET_COL,
    value: ValueName = "prediction_mean",
    actuals: pd.DataFrame | None = None,
    empty_context: EmptyContext = "context_mean",
    mask_covariate_history: bool = False,
    batch: bool = True,
) -> ShapleyExplanation:
    """Shapley values of contiguous blocks of the context (see :func:`context_segments`).

    A coalition keeps the target values of its segments and sets the others to NaN; covariates
    stay on unless ``mask_covariate_history`` also blanks their past values in the masked
    segments. An int ``segments`` splits the longest series of the context into equal blocks.
    See :func:`shapley_values` for the other parameters.
    """
    _check_frame(context, [ID_COL, TIMESTAMP_COL], "context")
    longest = int(context.groupby(ID_COL, sort=False).size().max())
    return shapley_values(
        forecaster,
        context,
        future_covariates,
        players=context_segments(segments, context_length=longest),
        prediction_length=prediction_length,
        freq=freq,
        target=target,
        value=value,
        actuals=actuals,
        empty_context=empty_context,
        mask_covariate_history=mask_covariate_history,
        batch=batch,
    )


def shapley_values(
    forecaster: Forecaster,
    context: pd.DataFrame,
    future_covariates: pd.DataFrame | None = None,
    *,
    players: Sequence[Player],
    prediction_length: int,
    freq: str,
    target: str = TARGET_COL,
    masking: Masking = "drop",
    value: ValueName = "prediction_mean",
    actuals: pd.DataFrame | None = None,
    empty_context: EmptyContext = "context_mean",
    mask_covariate_history: bool = False,
    batch: bool = True,
) -> ShapleyExplanation:
    """Exact Shapley values of any mix of covariate groups and context segments.

    Each of the ``2**len(players)`` coalitions is forecast once for every series in ``context``,
    and every series is explained separately. Mixing both kinds of players reproduces the setup
    of Hertel et al. (2026)::

        players = [*context_segments((24, 168, 672)), *covariate_players(["temp", "holiday"])]

    Parameters
    ----------
    forecaster, context, future_covariates, prediction_length, freq, target
        As in ``Forecaster.predict``, with a single target column.
    players
        :class:`CovariateGroup` and :class:`ContextSegment` instances with unique names.
    masking
        How absent covariates are switched off: ``"drop"`` the columns or set them to ``"nan"``.
    value
        ``"prediction_mean"`` (mean point forecast over the horizon), ``"prediction"`` (one
        explanation per forecast step) or ``"wql_reduction"`` (WQL of the empty coalition minus
        WQL of the coalition on ``actuals``, with the fev-bench settings of ``c2p.eval``).
    actuals
        ``item_id``, ``timestamp`` and ``target`` for the horizon; required for WQL.
    empty_context
        What a coalition without any target value forecasts: ``"context_mean"`` (the mean of
        the full context, for every step and quantile) or ``"model"`` (whatever the forecaster
        returns for an all-NaN target).
    mask_covariate_history
        Also set the covariates' past values to NaN inside absent context segments.
    batch
        Forecast coalitions that share the same columns in one call (see the module docstring).
    """
    names = _player_names(players)
    request = _ExplainRequest.build(
        context, future_covariates, players=players, target=target, value=value, actuals=actuals
    )
    _check_choice("masking", masking, Masking)
    _check_choice("empty_context", empty_context, EmptyContext)
    _check_batchable(forecaster, batch)

    n_coalitions = 1 << len(names)
    inputs = [
        _mask_players(
            request,
            [player for i, player in enumerate(players) if not mask >> i & 1],
            masking=masking,
            mask_covariate_history=mask_covariate_history,
        )
        for mask in range(n_coalitions)
    ]
    start = time.perf_counter()
    forecasts, n_calls = _forecast_inputs(
        forecaster,
        inputs,
        prediction_length=prediction_length,
        freq=freq,
        target=target,
        quantile_levels=_levels(needs_wql=value == "wql_reduction"),
        batch=batch,
    )
    seconds = time.perf_counter() - start
    if empty_context == "context_mean":
        forecasts = _replace_empty_context(forecasts, inputs, request)
    shapley, coalitions = _explain(_coalition_values(forecasts, value, request), names, value)
    logger.info(
        "Explained %d series with %d players: %d coalitions in %d predict calls (%.1f s).",
        request.n_items,
        len(names),
        n_coalitions,
        n_calls,
        seconds,
    )
    labels = forecasts[_REQUEST_COL].map(lambda mask: _members(mask, names))
    return ShapleyExplanation(
        players=names,
        value=value,
        shapley=shapley,
        coalitions=coalitions,
        forecasts=forecasts.assign(**{COALITION_COL: labels}).drop(columns=_REQUEST_COL),
        n_forecasts=n_coalitions * request.n_items,
        n_calls=n_calls,
        seconds=seconds,
    )


@dataclass(frozen=True, eq=False)
class _ExplainRequest:
    """Validated inputs shared by every explanation: the sorted context and each row's lag."""

    context: pd.DataFrame
    future: pd.DataFrame | None
    target: str
    lags: np.ndarray
    actuals: pd.DataFrame | None

    @property
    def n_items(self) -> int:
        return int(self.context[ID_COL].nunique())

    @property
    def covariate_columns(self) -> list[str]:
        excluded = {ID_COL, TIMESTAMP_COL, self.target}
        return [column for column in self.context.columns if column not in excluded]

    @classmethod
    def build(
        cls,
        context: pd.DataFrame,
        future: pd.DataFrame | None,
        *,
        players: Sequence[Player],
        target: str,
        value: str | None = None,
        actuals: pd.DataFrame | None = None,
    ) -> _ExplainRequest:
        if not isinstance(target, str):
            raise TypeError(f"Explanations need a single target column name, got {target!r}.")
        _check_frame(context, [ID_COL, TIMESTAMP_COL, target], "context")
        if not pd.api.types.is_numeric_dtype(context[target]):
            raise ValueError(f"Target column {target!r} must be numeric.")
        if future is not None:
            _check_frame(future, [ID_COL, TIMESTAMP_COL], "future_covariates")
        sorted_context = context.sort_values([ID_COL, TIMESTAMP_COL], kind="stable")
        sorted_context = sorted_context.reset_index(drop=True)
        observed = sorted_context.groupby(ID_COL, sort=False)[target].count()
        if (observed == 0).any():
            raise ValueError(
                f"Series without any observed target value: {list(observed[observed == 0].index)}."
            )
        _check_players(players, sorted_context, target)
        if value is not None:
            _check_choice("value", value, ValueName)
            if value == "wql_reduction" and actuals is None:
                raise ValueError("value='wql_reduction' needs the actuals of the horizon.")
        if actuals is not None:
            _check_frame(actuals, [ID_COL, TIMESTAMP_COL, target], "actuals")
        lags = sorted_context.groupby(ID_COL, sort=False).cumcount(ascending=False).to_numpy() + 1
        return cls(sorted_context, future, target, lags, actuals)


def _check_frame(frame: object, columns: Sequence[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame, got {type(frame).__name__}.")
    if frame.empty:
        raise ValueError(f"{name} is empty.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} lacks the columns {missing}.")


def _check_choice(name: str, choice: object, options: object) -> None:
    allowed = get_args(options)
    if choice not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {choice!r}.")


def _player_names(players: Sequence[Player]) -> tuple[str, ...]:
    if not all(isinstance(player, CovariateGroup | ContextSegment) for player in players):
        raise TypeError("players must be CovariateGroup or ContextSegment instances.")
    return _check_player_names([player.name for player in players])


def _check_players(players: Sequence[Player], context: pd.DataFrame, target: str) -> None:
    """Covariate players must name distinct covariate columns of the context."""
    covariates = set(context.columns) - {ID_COL, TIMESTAMP_COL, target}
    owner: dict[str, str] = {}
    for player in players:
        if not isinstance(player, CovariateGroup):
            continue
        for column in player.columns:
            if column not in covariates:
                raise ValueError(
                    f"Player {player.name!r}: {column!r} is not a covariate column of the "
                    f"context (covariates: {sorted(covariates)})."
                )
            if column in owner:
                raise ValueError(
                    f"Column {column!r} belongs to both {owner[column]!r} and {player.name!r}."
                )
            owner[column] = player.name


def _check_batchable(forecaster: Forecaster, batch: bool) -> None:
    if batch and getattr(forecaster, "cross_learning", False):
        raise ValueError(
            f"{getattr(forecaster, 'name', forecaster)!r} uses cross-learning, so coalitions "
            "batched as separate series would share information. Pass batch=False (one predict "
            "call per coalition) or explain a forecaster without cross-learning."
        )


def _levels(*, needs_wql: bool) -> tuple[float, ...]:
    """Quantile levels to request: the deciles WQL scores, or just the median."""
    return SCORED_QUANTILES if needs_wql else (0.5,)


# --------------------------------------------------------------------------------------------
# Masking and batched forecasting
# --------------------------------------------------------------------------------------------


def _mask_players(
    request: _ExplainRequest,
    absent: Sequence[Player],
    *,
    masking: Masking,
    mask_covariate_history: bool,
) -> _Input:
    """Context and future frame with the ``absent`` players switched off (new frames)."""
    context = request.context.copy()
    future = None if request.future is None else request.future.copy()
    for player in absent:
        if isinstance(player, CovariateGroup):
            context, future = _mask_covariates(context, future, player.columns, masking)
            continue
        keep = ~player.contains(request.lags)
        columns = [request.target]
        if mask_covariate_history:
            columns += [c for c in request.covariate_columns if c in context.columns]
        for column in columns:
            context[column] = context[column].where(keep)  # NaN, keeping categorical dtypes
    return context, future


def _mask_covariates(
    context: pd.DataFrame, future: pd.DataFrame | None, columns: Sequence[str], masking: Masking
) -> _Input:
    future_columns = [] if future is None else [c for c in columns if c in future.columns]
    if masking == "drop":
        context = context.drop(columns=list(columns))
        if future is not None:
            future = future.drop(columns=future_columns)
            if set(future.columns) <= {ID_COL, TIMESTAMP_COL}:
                future = None  # no known covariate left
        return context, future
    for column in columns:
        context[column] = _blank(context[column])
    if future is not None:
        for column in future_columns:
            future[column] = _blank(future[column])
    return context, future


def _blank(values: pd.Series) -> pd.Series:
    """All-missing copy of ``values``, keeping its dtype when it can hold NaN."""
    return values.where(np.zeros(len(values), dtype=bool))


def _forecast_inputs(
    forecaster: Forecaster,
    inputs: Sequence[_Input],
    *,
    prediction_length: int,
    freq: str,
    target: str,
    quantile_levels: Sequence[float],
    batch: bool,
) -> tuple[pd.DataFrame, int]:
    """Forecast every masked input; return the forecasts and the number of ``predict`` calls.

    The forecasts keep the original ids, with the input's position in ``_request``. With
    ``batch``, inputs with the same columns go into one call, each series under a temporary id.
    """
    groups: dict[object, list[int]] = {}
    for position, (context, future) in enumerate(inputs):
        signature = _signature(context, future) if batch else position
        groups.setdefault(signature, []).append(position)
    outputs = []
    for positions in groups.values():
        context, future, origin = _stack_inputs(inputs, positions)
        forecast = forecaster.predict(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            freq=freq,
            target=target,
            future_covariates=future,
        )
        outputs.append(_restore_ids(forecast, origin))
    forecasts = pd.concat(outputs, ignore_index=True)
    return forecasts.sort_values([_REQUEST_COL], kind="stable", ignore_index=True), len(groups)


def _signature(context: pd.DataFrame, future: pd.DataFrame | None) -> tuple:
    return tuple(context.columns), None if future is None else tuple(future.columns)


def _stack_inputs(
    inputs: Sequence[_Input], positions: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Concatenate inputs under ids ``"<position>|<item_id>"``; also return the id mapping.

    A single input keeps its own ids, so an unbatched call sees exactly the masked request.
    """
    contexts, futures, origins = [], [], []
    for position in positions:
        context, future = inputs[position]
        items = context[ID_COL].drop_duplicates()
        temporary = {item: item if len(positions) == 1 else f"{position}|{item}" for item in items}
        contexts.append(context.assign(**{ID_COL: context[ID_COL].map(temporary)}))
        if future is not None:
            futures.append(future.assign(**{ID_COL: future[ID_COL].map(temporary)}))
        origins.append(
            pd.DataFrame(
                {
                    _TEMPORARY_ID: list(temporary.values()),
                    ID_COL: items.to_numpy(),
                    _REQUEST_COL: position,
                }
            )
        )
    stacked_future = pd.concat(futures, ignore_index=True) if futures else None
    origin = pd.concat(origins, ignore_index=True)
    return pd.concat(contexts, ignore_index=True), stacked_future, origin


def _restore_ids(forecast: pd.DataFrame, origin: pd.DataFrame) -> pd.DataFrame:
    restored = forecast.rename(columns={ID_COL: _TEMPORARY_ID}).merge(
        origin, on=_TEMPORARY_ID, how="left", validate="many_to_one"
    )
    if restored[_REQUEST_COL].isna().any():
        raise ValueError("The forecaster returned series ids that were not in its input.")
    columns = [_REQUEST_COL, ID_COL, *(c for c in forecast.columns if c != ID_COL)]
    return restored[columns]


def _replace_empty_context(
    forecasts: pd.DataFrame, inputs: Sequence[_Input], request: _ExplainRequest
) -> pd.DataFrame:
    """Forecast the context mean for every (input, series) left without any target value."""
    empty_pairs = [
        (position, item)
        for position, (context, _) in enumerate(inputs)
        for item, count in context.groupby(ID_COL, sort=False)[request.target].count().items()
        if count == 0
    ]
    if not empty_pairs:
        return forecasts
    pairs = pd.MultiIndex.from_frame(forecasts[[_REQUEST_COL, ID_COL]])
    rows = pairs.isin(empty_pairs)
    means = request.context.groupby(ID_COL, sort=False)[request.target].mean()
    fill = forecasts.loc[rows, ID_COL].map(means).to_numpy("float64")
    replaced = forecasts.copy()
    for column in (PREDICTION_COL, *(c for c in forecasts.columns if _is_level(c))):
        replaced.loc[rows, column] = fill
    return replaced


def _is_level(column: object) -> bool:
    """Whether a forecast column holds a quantile (named like ``"0.1"``)."""
    try:
        return 0.0 < float(str(column)) < 1.0
    except ValueError:
        return False


# --------------------------------------------------------------------------------------------
# Coalition values and Shapley frames
# --------------------------------------------------------------------------------------------


def _coalition_values(
    forecasts: pd.DataFrame, value: ValueName, request: _ExplainRequest
) -> pd.DataFrame:
    """One value per (coalition, item[, timestamp]) in a ``value`` column."""
    if value == "prediction":
        columns = [_REQUEST_COL, ID_COL, TIMESTAMP_COL, PREDICTION_COL]
        return forecasts[columns].rename(columns={PREDICTION_COL: VALUE_COL})
    if value == "prediction_mean":
        means = forecasts.groupby([_REQUEST_COL, ID_COL], sort=False)[PREDICTION_COL].mean()
        return means.rename(VALUE_COL).reset_index()
    wql = _wql_by_series(forecasts, request.actuals, request.target)
    empty = wql.loc[wql[_REQUEST_COL] == 0].set_index(ID_COL)["wql"]
    return wql.assign(**{VALUE_COL: wql[ID_COL].map(empty) - wql["wql"]}).drop(columns="wql")


def _explain(
    coalition_values: pd.DataFrame, names: tuple[str, ...], value: ValueName
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shapley values and tidy coalition values from a long table of coalition values."""
    keys = [ID_COL, TIMESTAMP_COL] if value == "prediction" else [ID_COL]
    matrix, columns = _coalition_matrix(coalition_values, keys, 1 << len(names))
    shapley = _shapley_matrix(matrix, len(names))
    values = pd.concat(
        [
            columns.assign(**{PLAYER_COL: name, SHAPLEY_COL: shapley[i]})
            for i, name in enumerate(names)
        ],
        ignore_index=True,
    ).sort_values(keys, kind="stable", ignore_index=True)
    masks = coalition_values[_REQUEST_COL]
    coalitions = coalition_values.assign(
        **{
            COALITION_COL: masks.map(lambda mask: _members(mask, names)),
            SIZE_COL: masks.map(int.bit_count),
        }
    ).sort_values([*keys, _REQUEST_COL], kind="stable", ignore_index=True)
    return values, coalitions[[*keys, COALITION_COL, SIZE_COL, VALUE_COL]]


def _coalition_matrix(
    coalition_values: pd.DataFrame, keys: list[str], n_coalitions: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """Values as ``[n_coalitions, m]`` (row = coalition mask) and the ``keys`` of the m columns."""
    ordered = coalition_values.sort_values([*keys, _REQUEST_COL], kind="stable")
    columns = ordered[keys].drop_duplicates().reset_index(drop=True)
    expected = np.tile(np.arange(n_coalitions), len(columns))
    masks = ordered[_REQUEST_COL].to_numpy()
    matrix = ordered[VALUE_COL].to_numpy("float64")
    if len(masks) != len(expected) or (masks != expected).any() or not np.isfinite(matrix).all():
        raise ValueError(
            "Some coalition values are missing or not finite (e.g. WQL without any observed "
            "actual, or a forecaster that skipped series); cannot compute Shapley values."
        )
    return matrix.reshape(len(columns), n_coalitions).T, columns


def _wql_by_series(
    forecasts: pd.DataFrame, actuals: pd.DataFrame | None, target: str
) -> pd.DataFrame:
    """WQL of every (input, series) forecast on ``actuals``: columns ``_request``, id, ``wql``.

    Same metric as ``c2p.eval``: fev's WQL with ``epsilon=1`` on the nine deciles.
    """
    if actuals is None:
        raise ValueError("WQL needs the actuals of the horizon.")
    truth = actuals[[ID_COL, TIMESTAMP_COL, target]].rename(columns={target: "_actual"})
    joined = forecasts.merge(
        truth, on=[ID_COL, TIMESTAMP_COL], how="left", validate="many_to_one", indicator=True
    )
    if (joined["_merge"] != "both").any():
        raise ValueError("actuals do not cover every forecast (item_id, timestamp).")
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN actuals give NaN
        for (position, item), group in joined.groupby([_REQUEST_COL, ID_COL], sort=False):
            horizon = len(group)
            score = _WQL.compute(
                y_true=group["_actual"].to_numpy("float64").reshape(1, horizon, 1),
                y_pred=group[PREDICTION_COL].to_numpy("float64").reshape(1, horizon, 1),
                y_past=np.zeros((1, 1)),  # WQL does not use past values
                y_past_lengths=np.array([1]),
                q_pred=group[_SCORED_COLUMNS].to_numpy("float64").reshape(1, horizon, 1, -1),
                seasonality=1,
                quantile_levels=list(SCORED_QUANTILES),
            )
            rows.append((position, item, score))
    return pd.DataFrame(rows, columns=[_REQUEST_COL, ID_COL, "wql"])


# --------------------------------------------------------------------------------------------
# Cheap baselines: ablation, permutation and occlusion
# --------------------------------------------------------------------------------------------


def ablation_importance(
    forecaster: Forecaster,
    context: pd.DataFrame,
    future_covariates: pd.DataFrame | None,
    *,
    covariates: Sequence[str] | Mapping[str, Sequence[str]],
    prediction_length: int,
    freq: str,
    target: str = TARGET_COL,
    masking: Masking = "drop",
    actuals: pd.DataFrame | None = None,
    batch: bool = True,
) -> pd.DataFrame:
    """Leave one covariate (group) out at a time and measure how much the forecast changes.

    Returns ``item_id``, ``player``, ``importance`` and ``metric``: the WQL increase when
    ``actuals`` are given (``"wql_increase"``, positive = the covariate helps), otherwise the
    mean absolute change of the point forecast over the horizon (``"mean_abs_change"``). Needs
    ``len(covariates) + 1`` forecasts per series instead of ``2**len(covariates)``, but ignores
    interactions: two redundant covariates both look unimportant.
    """
    _check_choice("masking", masking, Masking)
    players = covariate_players(covariates)
    names = _check_player_names([player.name for player in players])
    request = _ExplainRequest.build(
        context, future_covariates, players=players, target=target, actuals=actuals
    )
    _check_batchable(forecaster, batch)
    inputs: list[_Input] = [(request.context, request.future)]
    inputs += [
        _mask_players(request, [player], masking=masking, mask_covariate_history=False)
        for player in players
    ]
    scores = _change_scores(
        forecaster,
        inputs,
        request,
        prediction_length=prediction_length,
        freq=freq,
        batch=batch,
    )
    scores[PLAYER_COL] = scores[_REQUEST_COL].map(lambda position: names[position - 1])
    return scores[[ID_COL, PLAYER_COL, IMPORTANCE_COL, METRIC_COL]]


def permutation_importance(
    forecaster: Forecaster,
    context: pd.DataFrame,
    future_covariates: pd.DataFrame | None,
    *,
    covariates: Sequence[str] | Mapping[str, Sequence[str]],
    prediction_length: int,
    freq: str,
    target: str = TARGET_COL,
    n_repeats: int = 5,
    seed: int = SEED,
    actuals: pd.DataFrame | None = None,
    batch: bool = True,
) -> pd.DataFrame:
    """Shuffle one covariate (group) at a time within each series and measure the change.

    Rows are shuffled separately in the context and in the future frame (the columns of a group
    move together), ``n_repeats`` times with a generator seeded by ``seed``. Returns
    ``item_id``, ``player``, ``importance`` (mean over repeats), ``importance_std`` and
    ``metric``, scored as in :func:`ablation_importance`.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}.")
    players = covariate_players(covariates)
    _check_player_names([player.name for player in players])
    request = _ExplainRequest.build(
        context, future_covariates, players=players, target=target, actuals=actuals
    )
    _check_batchable(forecaster, batch)
    rng = np.random.default_rng(seed)
    inputs: list[_Input] = [(request.context, request.future)]
    labels: list[str] = []
    for player in players:
        for _ in range(n_repeats):
            inputs.append(_shuffled(request, player.columns, rng))
            labels.append(player.name)
    scores = _change_scores(
        forecaster,
        inputs,
        request,
        prediction_length=prediction_length,
        freq=freq,
        batch=batch,
    )
    scores[PLAYER_COL] = scores[_REQUEST_COL].map(lambda position: labels[position - 1])
    summary = (
        scores.groupby([ID_COL, PLAYER_COL, METRIC_COL], sort=False)[IMPORTANCE_COL]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": IMPORTANCE_COL, "std": f"{IMPORTANCE_COL}_std"})
    )
    return summary[[ID_COL, PLAYER_COL, IMPORTANCE_COL, f"{IMPORTANCE_COL}_std", METRIC_COL]]


def _shuffled(request: _ExplainRequest, columns: Sequence[str], rng: np.random.Generator) -> _Input:
    """Context and future frame with ``columns`` shuffled within each series."""
    context = _shuffle_within_items(request.context, columns, rng)
    future = request.future
    if future is not None:
        future = _shuffle_within_items(future, [c for c in columns if c in future.columns], rng)
    return context, future


def _shuffle_within_items(
    frame: pd.DataFrame, columns: Sequence[str], rng: np.random.Generator
) -> pd.DataFrame:
    """Copy of ``frame`` whose ``columns`` rows are permuted inside each series."""
    shuffled = frame.copy()
    if not columns:
        return shuffled
    columns = list(columns)
    for rows in frame.groupby(ID_COL, sort=False).indices.values():
        source = frame[columns].iloc[rng.permutation(rows)]
        shuffled.loc[frame.index[rows], columns] = source.to_numpy()
    return shuffled


def occlusion(
    forecaster: Forecaster,
    context: pd.DataFrame,
    future_covariates: pd.DataFrame | None = None,
    *,
    window: int,
    prediction_length: int,
    freq: str,
    target: str = TARGET_COL,
    stride: int | None = None,
    max_lag: int | None = None,
    actuals: pd.DataFrame | None = None,
    batch: bool = True,
) -> pd.DataFrame:
    """Slide a NaN window over the last ``max_lag`` target values and measure the change.

    Each position hides lags ``first_lag`` to ``last_lag`` of the target (lag 1 = last
    observation); positions start every ``stride`` steps (default ``window``) and ``max_lag``
    defaults to the shortest context. Returns ``item_id``, ``first_lag``, ``last_lag``,
    ``start`` and ``end`` (timestamps of the hidden block), ``importance`` and ``metric``,
    scored as in :func:`ablation_importance`. Needs one forecast per position and series.
    """
    request = _ExplainRequest.build(
        context, future_covariates, players=(), target=target, actuals=actuals
    )
    _check_batchable(forecaster, batch)
    shortest = int(request.context.groupby(ID_COL, sort=False).size().min())
    max_lag = shortest if max_lag is None else max_lag
    stride = window if stride is None else stride
    if window < 1 or stride < 1:
        raise ValueError(f"window and stride must be >= 1, got {window} and {stride}.")
    if not window <= max_lag <= shortest:
        raise ValueError(
            f"max_lag must be between window ({window}) and the shortest context ({shortest}), "
            f"got {max_lag}."
        )
    first_lags = list(range(1, max_lag - window + 2, stride))
    inputs: list[_Input] = [(request.context, request.future)]
    for first in first_lags:
        segment = ContextSegment("occluded", first, first + window - 1)
        masked = _mask_players(request, [segment], masking="nan", mask_covariate_history=False)
        inputs.append(masked)
    scores = _change_scores(
        forecaster,
        inputs,
        request,
        prediction_length=prediction_length,
        freq=freq,
        batch=batch,
    )
    scores["first_lag"] = scores[_REQUEST_COL].map(lambda position: first_lags[position - 1])
    scores["last_lag"] = scores["first_lag"] + window - 1
    columns = [ID_COL, "first_lag", "last_lag", "start", "end", IMPORTANCE_COL, METRIC_COL]
    return _with_lag_timestamps(scores, request)[columns]


def _with_lag_timestamps(scores: pd.DataFrame, request: _ExplainRequest) -> pd.DataFrame:
    """Add the timestamps at ``last_lag`` (``start``) and ``first_lag`` (``end``) per series."""
    by_lag = request.context[[ID_COL, TIMESTAMP_COL]].assign(_lag=request.lags)
    start = by_lag.rename(columns={TIMESTAMP_COL: "start", "_lag": "last_lag"})
    end = by_lag.rename(columns={TIMESTAMP_COL: "end", "_lag": "first_lag"})
    merged = scores.merge(start, on=[ID_COL, "last_lag"], how="left")
    return merged.merge(end, on=[ID_COL, "first_lag"], how="left")


def _change_scores(
    forecaster: Forecaster,
    inputs: Sequence[_Input],
    request: _ExplainRequest,
    *,
    prediction_length: int,
    freq: str,
    batch: bool,
) -> pd.DataFrame:
    """Forecast ``inputs`` and score how much each one differs from ``inputs[0]``, per series.

    WQL increase when the request has actuals, mean absolute change of the point forecast
    otherwise. Returns ``_request`` (positions >= 1), ``item_id``, ``importance``, ``metric``.
    """
    forecasts, _ = _forecast_inputs(
        forecaster,
        inputs,
        prediction_length=prediction_length,
        freq=freq,
        target=request.target,
        quantile_levels=_levels(needs_wql=request.actuals is not None),
        batch=batch,
    )
    if request.actuals is not None:
        wql = _wql_by_series(forecasts, request.actuals, request.target)
        reference = wql.loc[wql[_REQUEST_COL] == 0].set_index(ID_COL)["wql"]
        scores = wql.loc[wql[_REQUEST_COL] != 0].reset_index(drop=True)
        scores[IMPORTANCE_COL] = scores["wql"] - scores[ID_COL].map(reference)
        return scores.drop(columns="wql").assign(**{METRIC_COL: "wql_increase"})
    is_reference = forecasts[_REQUEST_COL] == 0
    reference = forecasts.loc[is_reference, [ID_COL, TIMESTAMP_COL, PREDICTION_COL]]
    joined = forecasts.loc[~is_reference].merge(
        reference, on=[ID_COL, TIMESTAMP_COL], suffixes=("", "_reference"), validate="many_to_one"
    )
    change = (joined[PREDICTION_COL] - joined[f"{PREDICTION_COL}_reference"]).abs()
    scores = change.groupby([joined[_REQUEST_COL], joined[ID_COL]], sort=False).mean()
    scores = scores.rename(IMPORTANCE_COL).reset_index()
    return scores.assign(**{METRIC_COL: "mean_abs_change"})


# --------------------------------------------------------------------------------------------
# Several backtest windows in one pass
# --------------------------------------------------------------------------------------------


def stack_windows(
    windows: Sequence[Window],
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Merge backtest windows into one request: context, future covariates and actuals.

    Each series of each window becomes its own series with id ``"<item_id>@<cutoff>"``, so the
    stacked frames explain every window at once, in a few large ``predict`` calls instead of
    many small ones.
    """
    if not windows:
        raise ValueError("At least one window is required.")
    with_future = [window.future_covariates is not None for window in windows]
    if any(with_future) and not all(with_future):
        raise ValueError("Either every window or none must have future covariates.")
    contexts = [_relabel(window.context, window.cutoff) for window in windows]
    actuals = [_relabel(window.actuals, window.cutoff) for window in windows]
    future = None
    if all(with_future):
        futures = [
            _relabel(window.future_covariates, window.cutoff)
            for window in windows
            if window.future_covariates is not None
        ]
        future = pd.concat(futures, ignore_index=True)
    return pd.concat(contexts, ignore_index=True), future, pd.concat(actuals, ignore_index=True)


def _relabel(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    suffix = f"@{cutoff:%Y-%m-%dT%H:%M}"
    return frame.assign(**{ID_COL: frame[ID_COL].astype(str) + suffix})


# --------------------------------------------------------------------------------------------
# From per-window attributions to a table with intervals
# --------------------------------------------------------------------------------------------


def importance_summary(
    per_window: pd.DataFrame,
    *,
    value: str,
    by: Sequence[str] = (PLAYER_COL,),
    n_resamples: int = 2000,
    seed: int = SEED,
    alpha: float = 0.05,
    block_length: int = 1,
) -> pd.DataFrame:
    """Mean of a per-window attribution with a bootstrap interval over the windows.

    Every function of this module returns one row per stacked window (``item_id``) and player.
    The mean over windows is what gets reported, and a mean of a dozen noisy values needs its
    uncertainty next to it: a player whose interval contains zero has not been shown to
    matter, and two players whose intervals overlap have not been ranked.

    Returns one row per group of ``by`` with ``mean``, ``ci_low`` and ``ci_high`` (the
    ``1 - alpha`` percentile interval of the mean, resampling windows with replacement),
    ``share_positive`` (windows in which the value is above zero) and ``n_windows``.
    Deterministic for a given ``seed``.

    ``block_length`` above 1 resamples runs of that many consecutive windows instead of single
    windows (circular moving-block bootstrap), for windows that follow each other in time and
    whose values are therefore correlated. The rows of each group must then be in time order,
    which is the order :func:`stack_windows` produces.
    """
    columns = [*by, value]
    missing = [column for column in columns if column not in per_window.columns]
    if missing:
        raise ValueError(f"per_window lacks the columns {missing}.")
    if per_window.empty:
        raise ValueError("per_window is empty.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if block_length < 1:
        raise ValueError(f"block_length must be at least 1, got {block_length}.")
    rng = np.random.default_rng(seed)
    rows = []
    for key, group in per_window.groupby(list(by), sort=False):
        values = group[value].to_numpy("float64")
        if np.isnan(values).any():
            raise ValueError(f"{value} is NaN in some windows of {key}.")
        if block_length > len(values):
            raise ValueError(f"block_length={block_length} exceeds the {len(values)} windows.")
        n_blocks = -(-len(values) // block_length)
        starts = rng.integers(0, len(values), size=(n_resamples, n_blocks))
        draws = (starts[:, :, None] + np.arange(block_length)) % len(values)
        draws = draws.reshape(n_resamples, -1)[:, : len(values)]
        means = values[draws].mean(axis=1)
        labels = key if isinstance(key, tuple) else (key,)
        rows.append(
            dict(zip(by, labels, strict=True))
            | {
                "mean": float(values.mean()),
                "ci_low": float(np.quantile(means, 0.5 * alpha)),
                "ci_high": float(np.quantile(means, 1.0 - 0.5 * alpha)),
                "share_positive": float((values > 0).mean()),
                "n_windows": len(values),
            }
        )
    return pd.DataFrame(rows)
