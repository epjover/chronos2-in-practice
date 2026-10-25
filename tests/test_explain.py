"""Tests for ``c2p.explain``.

Fast tests use deterministic fake forecasters whose forecasts are known functions of their
inputs, so the exact Shapley values are known analytically. The slow test explains the real
Chronos-2 on CPU on a short Spanish electricity-price example with calendar covariates.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from c2p.data import make_future_frame
from c2p.eval import BacktestConfig, rolling_origin_windows
from c2p.explain import (
    COALITION_COL,
    IMPORTANCE_COL,
    MAX_EXACT_PLAYERS,
    METRIC_COL,
    PLAYER_COL,
    SHAPLEY_COL,
    SIZE_COL,
    VALUE_COL,
    ContextSegment,
    CovariateGroup,
    ShapleyExplanation,
    ablation_importance,
    context_segments,
    covariate_players,
    covariate_shapley,
    exact_shapley,
    occlusion,
    permutation_importance,
    shapley_values,
    stack_windows,
    temporal_shapley,
)
from c2p.schema import (
    ID_COL,
    PREDICTION_COL,
    TARGET_COL,
    TARGET_NAME_COL,
    TIMESTAMP_COL,
    quantile_col,
)

FREQ = "h"
HORIZON = 6
COVARIATES = ("a", "b", "c")
BETAS = {"a": 2.0, "b": -3.0, "c": 0.5}
INTERCEPT = 10.0


# --------------------------------------------------------------------------------------------
# Fake forecasters
# --------------------------------------------------------------------------------------------


def _forecast_frame(
    future: pd.DataFrame, predictions: np.ndarray, quantile_levels: Sequence[float], target: str
) -> pd.DataFrame:
    """Forecast frame with quantiles spread symmetrically around the point forecast."""
    frame = future[[ID_COL, TIMESTAMP_COL]].assign(
        **{TARGET_NAME_COL: target, PREDICTION_COL: predictions}
    )
    for level in quantile_levels:
        frame[quantile_col(level)] = predictions + 4.0 * (level - 0.5)
    return frame


class LinearForecaster:
    """Forecast at step t = intercept + sum_c beta_c * x_c(t) (+ gamma * x_a(t) * x_b(t)).

    ``x_c(t)`` is the future value of covariate ``c``; a missing or NaN covariate counts as 0,
    so dropping a column and setting it to NaN give the same forecast. Records every call.
    """

    def __init__(
        self,
        betas: Mapping[str, float],
        *,
        intercept: float = INTERCEPT,
        interaction: tuple[str, str, float] | None = None,
        cross_learning: bool = False,
    ) -> None:
        self.name = "linear"
        self.betas = dict(betas)
        self.intercept = intercept
        self.interaction = interaction
        self.cross_learning = cross_learning
        self.calls: list[tuple[pd.DataFrame, pd.DataFrame | None]] = []

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
        self.calls.append((context, future_covariates))
        future = future_covariates
        if future is None:
            future = make_future_frame(context, prediction_length=prediction_length, freq=freq)
        predictions = np.full(len(future), self.intercept)
        for column, beta in self.betas.items():
            predictions += beta * _covariate(future, column)
        if self.interaction is not None:
            first, second, gamma = self.interaction
            predictions += gamma * _covariate(future, first) * _covariate(future, second)
        assert isinstance(target, str)
        return _forecast_frame(future, predictions, quantile_levels, target)


def _covariate(future: pd.DataFrame, column: str) -> np.ndarray:
    if column not in future.columns:
        return np.zeros(len(future))
    return np.nan_to_num(future[column].to_numpy("float64"), nan=0.0)


class LagForecaster:
    """Constant forecast = intercept + sum over lags of weight[lag - 1] * y(lag), NaN = 0."""

    def __init__(self, weights: np.ndarray, *, intercept: float = 1.0) -> None:
        self.name = "lags"
        self.weights = np.asarray(weights, dtype="float64")
        self.intercept = intercept
        self.calls: list[pd.DataFrame] = []

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
        self.calls.append(context)
        assert isinstance(target, str)
        future = make_future_frame(context, prediction_length=prediction_length, freq=freq)
        level = {
            item: self.intercept + _lag_sum(block[target].to_numpy("float64"), self.weights)
            for item, block in context.groupby(ID_COL, sort=False)
        }
        predictions = future[ID_COL].map(level).to_numpy("float64")
        return _forecast_frame(future, predictions, quantile_levels, target)


def _lag_sum(values: np.ndarray, weights: np.ndarray) -> float:
    recent_first = np.nan_to_num(values[::-1][: len(weights)], nan=0.0)
    return float(recent_first @ weights[: len(recent_first)])


# --------------------------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------------------------


def make_request(
    *, items: Sequence[str] = ("A", "B"), n: int = 48, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Context, future covariates and actuals for hourly series with covariates a, b, c."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n + HORIZON, freq=FREQ)
    frames = []
    for item in items:
        frame = pd.DataFrame({ID_COL: item, TIMESTAMP_COL: timestamps})
        for column in COVARIATES:
            frame[column] = rng.normal(1.0, 1.0, len(timestamps))
        frame[TARGET_COL] = 50 + rng.normal(0, 1, len(timestamps))
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    in_context = frame.groupby(ID_COL).cumcount() < n
    context = frame.loc[in_context].reset_index(drop=True)
    horizon = frame.loc[~in_context].reset_index(drop=True)
    return (
        context,
        horizon[[ID_COL, TIMESTAMP_COL, *COVARIATES]],
        horizon[[ID_COL, TIMESTAMP_COL, TARGET_COL]],
    )


def expected_linear_shapley(future: pd.DataFrame, betas: Mapping[str, float]) -> pd.DataFrame:
    """Analytic Shapley values of the mean forecast of ``LinearForecaster`` (no interaction).

    One row per item (sorted), one column per covariate.
    """
    means = future.groupby(ID_COL)[list(betas)].mean()
    return (means * pd.Series(betas)).rename_axis(columns=PLAYER_COL)


def wide(explanation: ShapleyExplanation) -> pd.DataFrame:
    """Shapley values of the mean forecast as items (rows, sorted) x players (columns)."""
    table = explanation.shapley.pivot_table(index=ID_COL, columns=PLAYER_COL, values=SHAPLEY_COL)
    return table[list(explanation.players)]


KWARGS = {"prediction_length": HORIZON, "freq": FREQ}


# --------------------------------------------------------------------------------------------
# exact_shapley
# --------------------------------------------------------------------------------------------


def test_exact_shapley_additive_game_gives_each_player_its_weight():
    weights = {"x": 1.5, "y": -2.0, "z": 4.0}
    values = exact_shapley(list(weights), lambda s: 7.0 + sum(weights[p] for p in s))
    assert values == pytest.approx(weights)


def test_exact_shapley_efficiency_on_a_random_game():
    rng = np.random.default_rng(1)
    players = [f"p{i}" for i in range(6)]
    table = {
        frozenset(s): rng.normal() for k in range(7) for s in itertools.combinations(players, k)
    }
    values = exact_shapley(players, table.__getitem__)
    assert sum(values.values()) == pytest.approx(table[frozenset(players)] - table[frozenset()])


def test_exact_shapley_symmetry_and_dummy():
    # x and y are interchangeable (unanimity game on {x, y}); z changes nothing.
    values = exact_shapley(["x", "y", "z"], lambda s: float({"x", "y"} <= s))
    assert values["x"] == pytest.approx(values["y"]) == pytest.approx(0.5)
    assert values["z"] == pytest.approx(0.0)


def test_exact_shapley_matches_the_permutation_definition():
    players = ["x", "y", "z", "w"]
    rng = np.random.default_rng(2)
    table = {
        frozenset(s): rng.normal() for k in range(5) for s in itertools.combinations(players, k)
    }
    expected = dict.fromkeys(players, 0.0)
    orders = list(itertools.permutations(players))
    for order in orders:
        for i, player in enumerate(order):
            before = frozenset(order[:i])
            expected[player] += (table[before | {player}] - table[before]) / len(orders)
    assert exact_shapley(players, table.__getitem__) == pytest.approx(expected)


def test_exact_shapley_evaluates_each_coalition_once():
    seen: list[frozenset[str]] = []

    def value(coalition: frozenset[str]) -> float:
        seen.append(coalition)
        return float(len(coalition))

    exact_shapley(["x", "y", "z", "w"], value)
    assert len(seen) == 16 and len(set(seen)) == 16


def test_exact_shapley_guards():
    with pytest.raises(ValueError, match=f"limited to {MAX_EXACT_PLAYERS}"):
        exact_shapley([f"p{i}" for i in range(MAX_EXACT_PLAYERS + 1)], lambda s: 0.0)
    with pytest.raises(ValueError, match="Duplicated"):
        exact_shapley(["x", "x"], lambda s: 0.0)
    with pytest.raises(ValueError, match="At least one"):
        exact_shapley([], lambda s: 0.0)
    with pytest.raises(TypeError, match="single string"):
        exact_shapley("xy", lambda s: 0.0)
    with pytest.raises(ValueError, match="nan"):
        exact_shapley(["x"], lambda s: math.nan)


# --------------------------------------------------------------------------------------------
# Players
# --------------------------------------------------------------------------------------------


def test_covariate_players_single_columns_and_groups():
    assert covariate_players(["a", "b"]) == [
        CovariateGroup("a", ("a",)),
        CovariateGroup("b", ("b",)),
    ]
    groups = covariate_players({"ab": ["a", "b"], "c": ["c"]})
    assert groups == [CovariateGroup("ab", ("a", "b")), CovariateGroup("c", ("c",))]
    with pytest.raises(TypeError):
        covariate_players("a")


def test_context_segments_from_lag_boundaries_follow_hertel_et_al():
    segments = context_segments((24, 168, 672))
    assert [s.name for s in segments] == ["lags 1-24", "lags 25-168", "lags 169-672", "lags 673+"]
    lags = np.arange(1, 2000)
    owners = np.stack([s.contains(lags) for s in segments])
    assert (owners.sum(axis=0) == 1).all()  # every lag belongs to exactly one segment


def test_context_segments_equal_blocks_cover_the_context():
    segments = context_segments(4, context_length=10)
    assert [(s.first_lag, s.last_lag) for s in segments] == [(1, 3), (4, 6), (7, 8), (9, None)]
    with pytest.raises(ValueError, match="context_length"):
        context_segments(4)
    with pytest.raises(ValueError, match="increasing"):
        context_segments((24, 24))
    with pytest.raises(ValueError, match="Cannot split"):
        context_segments(5, context_length=3)


def test_player_validation():
    context, future, _ = make_request()
    forecaster = LinearForecaster(BETAS)
    with pytest.raises(ValueError, match="not a covariate column"):
        covariate_shapley(forecaster, context, future, covariates=["a", "zzz"], **KWARGS)
    with pytest.raises(ValueError, match="not a covariate column"):
        covariate_shapley(forecaster, context, future, covariates=[TARGET_COL], **KWARGS)
    with pytest.raises(ValueError, match="belongs to both"):
        covariate_shapley(
            forecaster, context, future, covariates={"x": ["a"], "y": ["a"]}, **KWARGS
        )
    with pytest.raises(ValueError, match="first_lag"):
        ContextSegment("bad", 0)
    with pytest.raises(TypeError, match="instances"):
        shapley_values(forecaster, context, future, players=["a"], **KWARGS)  # type: ignore[list-item]


# --------------------------------------------------------------------------------------------
# covariate_shapley
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("masking", ["nan", "drop"])
def test_covariate_shapley_matches_the_analytic_values(masking):
    context, future, _ = make_request()
    explanation = covariate_shapley(
        LinearForecaster(BETAS), context, future, covariates=COVARIATES, masking=masking, **KWARGS
    )
    pd.testing.assert_frame_equal(wide(explanation), expected_linear_shapley(future, BETAS))
    assert explanation.efficiency_gap() < 1e-9
    assert explanation.base_value().to_numpy() == pytest.approx([INTERCEPT, INTERCEPT])
    assert explanation.n_forecasts == 2 * 2**3


def test_nan_masking_batches_coalitions_in_one_call_and_keeps_columns():
    context, future, _ = make_request()
    forecaster = LinearForecaster(BETAS)
    explanation = covariate_shapley(
        forecaster, context, future, covariates=COVARIATES, masking="nan", **KWARGS
    )
    assert explanation.n_calls == len(forecaster.calls) == 1
    batched_context, batched_future = forecaster.calls[0]
    assert set(COVARIATES) <= set(batched_context.columns)
    assert batched_context[ID_COL].nunique() == 2 * 2**3
    assert batched_future is not None
    # Exactly the coalitions without "a" see an all-NaN "a" column.
    blank = batched_future.groupby(ID_COL)["a"].apply(lambda s: s.isna().all())
    assert blank.sum() == 2 * 2**2


def test_drop_masking_removes_columns_one_call_per_coalition():
    context, future, _ = make_request()
    forecaster = LinearForecaster(BETAS)
    explanation = covariate_shapley(
        forecaster, context, future, covariates=COVARIATES, masking="drop", **KWARGS
    )
    assert explanation.n_calls == len(forecaster.calls) == 2**3
    column_sets = {frozenset(set(c.columns) & set(COVARIATES)) for c, _ in forecaster.calls}
    assert len(column_sets) == 2**3
    no_covariates = [f for c, f in forecaster.calls if not set(c.columns) & set(COVARIATES)]
    assert no_covariates == [None]  # future frame left out when no known covariate remains
    for call_context, _ in forecaster.calls:
        kept = [column for column in COVARIATES if column in call_context.columns]
        assert not call_context[kept].isna().any().any()


def test_unbatched_coalitions_give_the_same_values():
    context, future, _ = make_request()
    batched = covariate_shapley(
        LinearForecaster(BETAS), context, future, covariates=COVARIATES, masking="nan", **KWARGS
    )
    forecaster = LinearForecaster(BETAS)
    single = covariate_shapley(
        forecaster, context, future, covariates=COVARIATES, masking="nan", batch=False, **KWARGS
    )
    assert single.n_calls == 2**3
    assert set(forecaster.calls[0][0][ID_COL]) == {"A", "B"}  # original ids without batching
    pd.testing.assert_frame_equal(batched.shapley, single.shapley)


def test_cross_learning_forecasters_cannot_be_batched():
    context, future, _ = make_request()
    forecaster = LinearForecaster(BETAS, cross_learning=True)
    with pytest.raises(ValueError, match="cross-learning"):
        covariate_shapley(forecaster, context, future, covariates=COVARIATES, **KWARGS)
    explanation = covariate_shapley(
        forecaster, context, future, covariates=COVARIATES, batch=False, **KWARGS
    )
    assert explanation.n_calls == 2**3


def test_interaction_is_split_equally_and_dummy_gets_zero():
    context, future, _ = make_request(items=("A",))
    betas = {"a": 1.0, "b": 2.0, "c": 0.0}
    forecaster = LinearForecaster(betas, interaction=("a", "b", 3.0))
    explanation = covariate_shapley(forecaster, context, future, covariates=COVARIATES, **KWARGS)
    values = wide(explanation).iloc[0]
    a, b = future["a"].to_numpy(), future["b"].to_numpy()
    half_interaction = 3.0 * np.mean(a * b) / 2
    assert values["a"] == pytest.approx(a.mean() + half_interaction)
    assert values["b"] == pytest.approx(2.0 * b.mean() + half_interaction)
    assert values["c"] == pytest.approx(0.0)


def test_covariate_groups_are_one_player():
    context, future, _ = make_request()
    explanation = covariate_shapley(
        LinearForecaster(BETAS),
        context,
        future,
        covariates={"ab": ["a", "b"], "c": ["c"]},
        **KWARGS,
    )
    expected = expected_linear_shapley(future, BETAS)
    values = wide(explanation)
    assert values["ab"].to_numpy() == pytest.approx((expected["a"] + expected["b"]).to_numpy())
    assert explanation.players == ("ab", "c")


def test_per_step_values_explain_every_forecast_step():
    context, future, _ = make_request()
    explanation = covariate_shapley(
        LinearForecaster(BETAS),
        context,
        future,
        covariates=COVARIATES,
        value="prediction",
        **KWARGS,
    )
    assert explanation.keys == [ID_COL, TIMESTAMP_COL]
    assert len(explanation.shapley) == 2 * HORIZON * len(COVARIATES)
    step = explanation.shapley.set_index([ID_COL, TIMESTAMP_COL, PLAYER_COL])[SHAPLEY_COL]
    for column, beta in BETAS.items():
        truth = future.set_index([ID_COL, TIMESTAMP_COL])[column] * beta
        assert step.xs(column, level=PLAYER_COL).to_numpy() == pytest.approx(truth.to_numpy())
    assert explanation.efficiency_gap() < 1e-9
    # Mean of the step values = values of the mean forecast (Shapley values are linear).
    of_mean = covariate_shapley(
        LinearForecaster(BETAS), context, future, covariates=COVARIATES, **KWARGS
    )
    averaged = explanation.shapley.pivot_table(index=ID_COL, columns=PLAYER_COL, values=SHAPLEY_COL)
    pd.testing.assert_frame_equal(averaged[list(COVARIATES)], wide(of_mean))


def test_importance_ranks_players_by_mean_absolute_value():
    context, future, _ = make_request()
    explanation = covariate_shapley(
        LinearForecaster(BETAS), context, future, covariates=COVARIATES, **KWARGS
    )
    table = explanation.importance()
    assert list(table.columns) == [PLAYER_COL, "mean_abs", "mean", "share_pct", "rank"]
    assert table["share_pct"].sum() == pytest.approx(100.0)
    assert table["mean_abs"].is_monotonic_decreasing
    assert list(table["rank"]) == [1, 2, 3]
    expected = expected_linear_shapley(future, BETAS).abs().mean()
    assert table.set_index(PLAYER_COL)["mean_abs"].to_dict() == pytest.approx(expected.to_dict())


def test_coalitions_frame_and_forecasts():
    context, future, _ = make_request()
    explanation = covariate_shapley(
        LinearForecaster(BETAS), context, future, covariates=COVARIATES, **KWARGS
    )
    coalitions = explanation.coalitions
    assert list(coalitions.columns) == [ID_COL, COALITION_COL, SIZE_COL, VALUE_COL]
    assert len(coalitions) == 2 * 2**3
    full = coalitions.loc[coalitions[SIZE_COL] == 3, COALITION_COL].iloc[0]
    assert full == COVARIATES
    assert len(explanation.forecasts) == 2 * 2**3 * HORIZON
    assert set(explanation.forecasts[ID_COL]) == {"A", "B"}


def test_wql_reduction_value_needs_actuals_and_credits_informative_covariates():
    context, future, actuals = make_request()
    betas = {"a": 2.0, "b": -3.0, "c": 0.0}
    truth = INTERCEPT + 2.0 * future["a"] - 3.0 * future["b"]
    actuals = actuals.assign(**{TARGET_COL: truth.to_numpy()})
    forecaster = LinearForecaster(betas)
    with pytest.raises(ValueError, match="actuals"):
        covariate_shapley(
            forecaster, context, future, covariates=COVARIATES, value="wql_reduction", **KWARGS
        )
    explanation = covariate_shapley(
        forecaster,
        context,
        future,
        covariates=COVARIATES,
        value="wql_reduction",
        actuals=actuals,
        **KWARGS,
    )
    values = wide(explanation)
    assert (values[["a", "b"]] > 0).all().all()
    assert values["c"].to_numpy() == pytest.approx([0.0, 0.0])
    assert explanation.base_value().to_numpy() == pytest.approx([0.0, 0.0])
    assert explanation.efficiency_gap() < 1e-9


def test_inputs_are_not_modified():
    context, future, actuals = make_request()
    shuffled = context.sample(frac=1.0, random_state=0)
    copies = [shuffled.copy(), future.copy(), actuals.copy()]
    covariate_shapley(
        LinearForecaster(BETAS), shuffled, future, covariates=COVARIATES, masking="nan", **KWARGS
    )
    temporal_shapley(
        LagForecaster(np.ones(4)),
        shuffled,
        future,
        segments=3,
        mask_covariate_history=True,
        **KWARGS,
    )
    for original, copy in zip([shuffled, future, actuals], copies, strict=True):
        pd.testing.assert_frame_equal(original, copy)


# --------------------------------------------------------------------------------------------
# temporal_shapley
# --------------------------------------------------------------------------------------------


def test_temporal_shapley_matches_the_lag_weights_with_model_base():
    context, future, _ = make_request(n=12)
    weights = np.linspace(1.0, 0.1, 12)
    explanation = temporal_shapley(
        LagForecaster(weights), context, future, segments=(3, 6), empty_context="model", **KWARGS
    )
    assert explanation.players == ("lags 1-3", "lags 4-6", "lags 7+")
    values = wide(explanation)
    for item, block in context.groupby(ID_COL):
        contributions = block[TARGET_COL].to_numpy()[::-1] * weights
        expected = [contributions[:3].sum(), contributions[3:6].sum(), contributions[6:].sum()]
        assert values.loc[item].to_numpy() == pytest.approx(expected)
    assert explanation.base_value().to_numpy() == pytest.approx([1.0, 1.0])  # the intercept


def test_temporal_shapley_uses_the_context_mean_without_any_target():
    context, future, _ = make_request(n=12)
    explanation = temporal_shapley(
        LagForecaster(np.ones(12)), context, future, segments=2, **KWARGS
    )
    means = context.groupby(ID_COL)[TARGET_COL].mean()
    assert explanation.base_value().to_numpy() == pytest.approx(means.to_numpy())
    empty = explanation.forecasts.loc[explanation.forecasts[COALITION_COL] == ()]
    assert (empty[quantile_col(0.5)] == empty[PREDICTION_COL]).all()
    assert explanation.efficiency_gap() < 1e-9


def test_temporal_masking_sets_only_target_values_to_nan_by_default():
    context, future, _ = make_request(n=12, items=("A",))
    forecaster = LagForecaster(np.ones(12))
    temporal_shapley(forecaster, context, future, segments=(4,), **KWARGS)
    stacked = forecaster.calls[0]
    without_recent = stacked.loc[stacked[ID_COL] == "2|A"]  # coalition {"lags 5+"}
    assert without_recent[TARGET_COL].isna().sum() == 4
    assert without_recent[TARGET_COL].iloc[-4:].isna().all()
    assert not without_recent[list(COVARIATES)].isna().any().any()

    forecaster = LagForecaster(np.ones(12))
    temporal_shapley(
        forecaster, context, future, segments=(4,), mask_covariate_history=True, **KWARGS
    )
    without_recent = forecaster.calls[0].loc[forecaster.calls[0][ID_COL] == "2|A"]
    assert without_recent[list(COVARIATES)].iloc[-4:].isna().all().all()
    assert not without_recent[list(COVARIATES)].iloc[:-4].isna().any().any()


def test_mixed_players_reproduce_both_kinds_of_values():
    context, future, _ = make_request(n=12)
    players = [*context_segments((6,)), *covariate_players(["a", "b"])]
    explanation = shapley_values(
        LinearForecaster({"a": 2.0, "b": -3.0}),
        context,
        future,
        players=players,
        empty_context="model",
        **KWARGS,
    )
    values = wide(explanation)
    expected = expected_linear_shapley(future, {"a": 2.0, "b": -3.0})
    assert values[["a", "b"]].to_numpy() == pytest.approx(expected[["a", "b"]].to_numpy())
    assert values[["lags 1-6", "lags 7+"]].abs().max().max() == pytest.approx(0.0)


# --------------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------------


def test_ablation_importance_is_the_mean_absolute_change():
    context, future, _ = make_request()
    table = ablation_importance(
        LinearForecaster(BETAS), context, future, covariates=COVARIATES, **KWARGS
    )
    assert list(table.columns) == [ID_COL, PLAYER_COL, IMPORTANCE_COL, METRIC_COL]
    assert set(table[METRIC_COL]) == {"mean_abs_change"}
    for (item, player), importance in table.set_index([ID_COL, PLAYER_COL])[IMPORTANCE_COL].items():
        block = future.loc[future[ID_COL] == item, player]
        assert importance == pytest.approx(np.abs(BETAS[player] * block).mean())


def test_ablation_importance_with_actuals_reports_the_wql_increase():
    context, future, actuals = make_request()
    truth = INTERCEPT + 2.0 * future["a"]
    actuals = actuals.assign(**{TARGET_COL: truth.to_numpy()})
    table = ablation_importance(
        LinearForecaster({"a": 2.0, "b": 0.0}),
        context,
        future,
        covariates=["a", "b"],
        actuals=actuals,
        **KWARGS,
    )
    importance = table.set_index([ID_COL, PLAYER_COL])[IMPORTANCE_COL]
    assert set(table[METRIC_COL]) == {"wql_increase"}
    assert (importance.xs("a", level=PLAYER_COL) > 0).all()
    assert importance.xs("b", level=PLAYER_COL).to_numpy() == pytest.approx([0.0, 0.0])


def test_permutation_importance_is_seeded_and_ignores_dummies():
    context, future, _ = make_request()
    betas = {"a": 2.0, "b": -3.0, "c": 0.0}
    run = {"covariates": COVARIATES, "n_repeats": 3, **KWARGS}
    first = permutation_importance(LinearForecaster(betas), context, future, seed=7, **run)
    again = permutation_importance(LinearForecaster(betas), context, future, seed=7, **run)
    other = permutation_importance(LinearForecaster(betas), context, future, seed=8, **run)
    pd.testing.assert_frame_equal(first, again)
    importance = first.set_index([ID_COL, PLAYER_COL])[IMPORTANCE_COL]
    assert not np.allclose(importance, other.set_index([ID_COL, PLAYER_COL])[IMPORTANCE_COL])
    assert importance.xs("c", level=PLAYER_COL).to_numpy() == pytest.approx([0.0, 0.0])
    assert (importance.xs("b", level=PLAYER_COL) > 0).all()
    assert list(first.columns) == [ID_COL, PLAYER_COL, IMPORTANCE_COL, "importance_std", METRIC_COL]


def test_permutation_shuffles_within_each_series_only():
    context, future, _ = make_request()
    forecaster = LinearForecaster(BETAS)
    permutation_importance(
        forecaster, context, future, covariates=["a"], n_repeats=1, seed=0, **KWARGS
    )
    stacked, stacked_future = forecaster.calls[0]
    assert stacked_future is not None
    for item in ("A", "B"):
        shuffled = stacked.loc[stacked[ID_COL] == f"1|{item}", "a"].to_numpy()
        original = context.loc[context[ID_COL] == item, "a"].to_numpy()
        assert sorted(shuffled) == sorted(original) and not np.array_equal(shuffled, original)
        other = stacked.loc[stacked[ID_COL] == f"1|{item}", "b"].to_numpy()
        assert np.array_equal(other, context.loc[context[ID_COL] == item, "b"].to_numpy())


def test_occlusion_only_detects_the_lags_the_model_uses():
    context, future, _ = make_request(n=24)
    weights = np.r_[np.ones(4), np.zeros(20)]
    table = occlusion(LagForecaster(weights), context, future, window=4, max_lag=12, **KWARGS)
    assert list(table.columns) == [
        ID_COL, "first_lag", "last_lag", "start", "end", IMPORTANCE_COL, METRIC_COL,
    ]  # fmt: skip
    assert sorted(set(table["first_lag"])) == [1, 5, 9]
    first = table.loc[table["first_lag"] == 1].set_index(ID_COL)
    recent = context.groupby(ID_COL)[TARGET_COL].apply(lambda s: s.iloc[-4:].sum())
    assert first[IMPORTANCE_COL].to_numpy() == pytest.approx(recent.abs().to_numpy())
    assert (table.loc[table["first_lag"] > 1, IMPORTANCE_COL] == 0).all()
    last = context.groupby(ID_COL)[TIMESTAMP_COL].max()
    assert (first["end"] == last).all()
    assert (first["start"] == last - pd.Timedelta(hours=3)).all()
    with pytest.raises(ValueError, match="max_lag"):
        occlusion(LagForecaster(weights), context, future, window=4, max_lag=100, **KWARGS)


# --------------------------------------------------------------------------------------------
# Backtest windows
# --------------------------------------------------------------------------------------------


def test_stack_windows_explains_every_window_at_once():
    context, future, actuals = make_request(n=60)
    series = pd.concat(
        [
            context,
            future.merge(actuals, on=[ID_COL, TIMESTAMP_COL]),
        ],
        ignore_index=True,
    ).sort_values([ID_COL, TIMESTAMP_COL], ignore_index=True)
    config = BacktestConfig(
        prediction_length=HORIZON,
        n_windows=3,
        freq=FREQ,
        season_length=12,
        known_covariates=COVARIATES,
        max_context=24,
    )
    windows = rolling_origin_windows(series, config)
    stacked_context, stacked_future, stacked_actuals = stack_windows(windows)
    assert stacked_context[ID_COL].nunique() == 2 * 3
    assert stacked_future is not None and len(stacked_future) == 2 * 3 * HORIZON
    assert len(stacked_actuals) == 2 * 3 * HORIZON
    explanation = covariate_shapley(
        LinearForecaster(BETAS), stacked_context, stacked_future, covariates=COVARIATES, **KWARGS
    )
    assert explanation.n_calls == 2**3  # drop masking: one call per coalition, all windows
    for window in windows:
        single = covariate_shapley(
            LinearForecaster(BETAS),
            window.context,
            window.future_covariates,
            covariates=COVARIATES,
            **KWARGS,
        )
        suffix = f"@{window.cutoff:%Y-%m-%dT%H:%M}"
        expected = wide(single)
        expected.index = expected.index + suffix
        assert wide(explanation).loc[expected.index].to_numpy() == pytest.approx(
            expected.to_numpy()
        )


# --------------------------------------------------------------------------------------------
# Real Chronos-2 (slow)
# --------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_chronos2_calendar_covariates_on_spanish_prices():
    from c2p.data import add_calendar_features, spain_energy_frame, validate_and_prepare
    from c2p.models import Chronos2Forecaster, load_chronos2

    target, calendar = "price_actual", ["hour", "day_of_week", "is_holiday"]
    frame, _ = validate_and_prepare(spain_energy_frame(), freq=FREQ, target=target)
    frame = add_calendar_features(frame)[[ID_COL, TIMESTAMP_COL, target, *calendar]]
    cutoff = pd.Timestamp("2018-04-30 22:00")  # local midnight before the May 1 holiday
    context = frame.loc[frame[TIMESTAMP_COL] <= cutoff].tail(24 * 14).reset_index(drop=True)
    horizon = frame.loc[frame[TIMESTAMP_COL] > cutoff].head(24).reset_index(drop=True)
    future = horizon[[ID_COL, TIMESTAMP_COL, *calendar]]
    actuals = horizon[[ID_COL, TIMESTAMP_COL, target]]
    forecaster = Chronos2Forecaster(load_chronos2(device="cpu"))
    run = {"covariates": calendar, "prediction_length": 24, "freq": FREQ, "target": target}

    by_masking = {}
    for masking in ("nan", "drop"):
        explanation = covariate_shapley(
            forecaster, context, future, masking=masking, value="prediction", **run
        )
        assert explanation.n_calls == (1 if masking == "nan" else 8)
        assert np.isfinite(explanation.shapley[SHAPLEY_COL]).all()
        assert explanation.efficiency_gap() < 1e-6
        by_masking[masking] = explanation
    # Batching coalitions as separate series reproduces one-call-per-coalition forecasts.
    single = covariate_shapley(
        forecaster, context, future, masking="nan", value="prediction", batch=False, **run
    )
    batched = by_masking["nan"].shapley[SHAPLEY_COL].to_numpy()
    assert np.abs(single.shapley[SHAPLEY_COL].to_numpy() - batched).max() < 1e-2
    # Without covariates (drop masking), the base value is the univariate forecast.
    univariate = forecaster.predict(
        context[[ID_COL, TIMESTAMP_COL, target]],
        prediction_length=24,
        quantile_levels=[0.5],
        freq=FREQ,
        target=target,
    )
    base = by_masking["drop"].base_value().to_numpy()
    assert np.abs(base - univariate[PREDICTION_COL].to_numpy()).max() < 1e-2
    wql = covariate_shapley(
        forecaster, context, future, masking="drop", value="wql_reduction", actuals=actuals, **run
    )
    assert wql.efficiency_gap() < 1e-9
