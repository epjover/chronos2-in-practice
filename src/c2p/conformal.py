"""Calibrate the prediction intervals: conformal calibration of Chronos-2's quantiles.

Every part of the series found the same defect (the 80% interval covers about 73% of the
hours, less in winter) and none of them fixed it, because the levers they pulled (context,
covariates, fine-tuning) all target the median. This module targets the bands directly, with
no retraining: **conformalized quantile regression** (CQR, Romano et al., 2019).

What CQR promises, and when. If the misses of the calibration period and of the period being
forecast are exchangeable (in practice: drawn from the same distribution), the calibrated
interval covers at least its nominal share of the actuals. A time series does not give you
that for free: a winter that behaves unlike the calibration months breaks the assumption, and
the coverage with it. That is why this module also has :class:`AdaptiveConformalForecaster`,
which keeps recalibrating on its own realized forecasts (a rolling pool of recent misses,
and optionally the adaptive conformal inference update of Gibbs and Candès, 2021), and why
``coverage_by_period`` exists: the yearly number can be on target while a season is not.

The idea fits in a sentence. On a calibration period where the actuals are known, measure by how
much the model's interval missed, ``max(lower - y, y - upper)``, positive when the actual fell
outside, and widen every future interval by the quantile of those misses that makes the
calibration coverage equal to the nominal level. The adjustment is learned once, per quantile
pair and per horizon step, and applied as a post-processing step to any later forecast. It can
also *narrow* an interval that was too wide.

Two scalings are offered. ``additive`` learns a shift in the target's units, which is the
textbook CQR and the right choice for one series. ``scaled`` divides the miss by the model's own
interval width first, so the correction is a multiplier that travels across series of different
levels: the choice for a frame of many series or for a model whose width already tracks
difficulty.

Nothing here knows about Chronos-2. :class:`ConformalForecaster` wraps any ``Forecaster`` and
runs through the same backtest harness, so the before/after comparison is one more row in the
same table.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from c2p.eval import ACTUAL_COL, MODEL_COL, WINDOW_COL
from c2p.schema import (
    ID_COL,
    PREDICTION_COL,
    TARGET_COL,
    TARGET_NAME_COL,
    TIMESTAMP_COL,
    Forecaster,
    quantile_col,
)

__all__ = [
    "STEP_COL",
    "AdaptiveConformalForecaster",
    "ConformalForecaster",
    "QuantileCalibration",
    "calibrate",
    "conformalize",
    "coverage_by_period",
    "horizon_step",
    "symmetric_pairs",
]

logger = logging.getLogger(__name__)

STEP_COL = "step"
Mode = Literal["additive", "scaled"]


def symmetric_pairs(quantile_levels: Sequence[float]) -> list[tuple[float, float]]:
    """``(lower, upper)`` pairs with ``lower + upper == 1``, widest first; the median is skipped."""
    levels = sorted({float(level) for level in quantile_levels})
    pairs = []
    for lower in levels:
        upper = round(1.0 - lower, 10)
        if lower < 0.5 and upper in levels:
            pairs.append((lower, upper))
    if not pairs:
        raise ValueError(f"No symmetric quantile pairs in {levels}; CQR needs lower + upper = 1.")
    return pairs


def horizon_step(forecast: pd.DataFrame) -> pd.Series:
    """1-based position of each row within its series' horizon, from the timestamps."""
    keys = [column for column in (WINDOW_COL, ID_COL, TARGET_NAME_COL) if column in forecast]
    return forecast.groupby(keys, sort=False)[TIMESTAMP_COL].rank(method="first").astype(int)


@dataclass(frozen=True)
class QuantileCalibration:
    """Adjustments learned by :func:`calibrate`: one row per quantile pair and horizon step.

    ``table`` has columns ``lower``, ``upper``, ``step`` and ``delta``. In ``additive`` mode
    ``delta`` is added to the upper quantile and subtracted from the lower one, in the target's
    units; in ``scaled`` mode the interval's half-width grows by ``delta`` times the width.
    """

    table: pd.DataFrame
    mode: Mode
    n_calibration_rows: int

    def delta(self, lower: float, upper: float) -> pd.Series:
        """Per-horizon-step adjustment for the ``(lower, upper)`` pair, indexed by step."""
        rows = self.table[(self.table["lower"] == lower) & (self.table["upper"] == upper)]
        if rows.empty:
            raise KeyError(f"No calibration for the ({lower}, {upper}) pair.")
        return rows.set_index(STEP_COL)["delta"]


def calibrate(
    backtest_frame: pd.DataFrame,
    *,
    quantile_levels: Sequence[float],
    mode: Mode = "additive",
    by_step: bool = True,
    miss_rates: Mapping[tuple[float, float], float] | None = None,
) -> QuantileCalibration:
    """Learn CQR adjustments from a backtest frame that carries ``actual``.

    For each symmetric pair the nonconformity score of a row is ``max(lower - y, y - upper)``
    (divided by the interval width in ``scaled`` mode), and ``delta`` is the empirical
    ``ceil((n + 1)(1 - alpha)) / n`` quantile of the scores, ``alpha`` being the pair's miss
    rate. With ``by_step`` the scores are grouped by horizon step, so a band that is on target
    one hour ahead and too narrow at hour 24 is corrected differently at each.

    ``miss_rates`` overrides ``alpha`` for the pairs it names (keys are ``(lower, upper)``);
    adaptive conformal inference uses it to aim above or below the nominal level.
    """
    required = [ID_COL, TIMESTAMP_COL, ACTUAL_COL]
    missing = [column for column in required if column not in backtest_frame.columns]
    if missing:
        raise ValueError(f"The backtest frame lacks the columns {missing}.")
    rows = backtest_frame[backtest_frame[ACTUAL_COL].notna()]
    if rows.empty:
        raise ValueError("No calibration row has an actual value.")
    steps = horizon_step(rows) if by_step else pd.Series(1, index=rows.index)
    actual = rows[ACTUAL_COL].to_numpy("float64")
    records = []
    for lower, upper in symmetric_pairs(quantile_levels):
        low = rows[quantile_col(lower)].to_numpy("float64")
        high = rows[quantile_col(upper)].to_numpy("float64")
        scores = np.maximum(low - actual, actual - high)
        if mode == "scaled":
            width = np.maximum(high - low, np.finfo("float64").tiny)
            scores = scores / width
        alpha = (miss_rates or {}).get((lower, upper), 1.0 - (upper - lower))
        frame = pd.DataFrame({STEP_COL: steps.to_numpy(), "score": scores})
        for step, group in frame.groupby(STEP_COL, sort=True):
            n = len(group)
            level = float(np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0))
            delta = float(np.quantile(group["score"].to_numpy(), level, method="higher"))
            records.append({"lower": lower, "upper": upper, STEP_COL: int(step), "delta": delta})
    table = pd.DataFrame(records)
    logger.info(
        "Calibrated %d quantile pairs on %d rows (%s): widest-pair delta range %.4g to %.4g.",
        table[["lower", "upper"]].drop_duplicates().shape[0],
        len(rows),
        mode,
        table["delta"].min(),
        table["delta"].max(),
    )
    return QuantileCalibration(table=table, mode=mode, n_calibration_rows=len(rows))


def conformalize(
    forecast: pd.DataFrame,
    calibration: QuantileCalibration,
    *,
    quantile_levels: Sequence[float],
) -> pd.DataFrame:
    """Apply the learned adjustments to a forecast frame and keep the quantiles ordered.

    Steps beyond the calibrated horizon reuse the last calibrated step. After adjusting every
    pair, the quantile columns of each row are sorted (the rearrangement of Chernozhukov et
    al., 2010), so no pair can cross another; the median column is untouched unless a
    crossing forces it.
    """
    out = forecast.copy()
    steps = horizon_step(out)
    columns = [quantile_col(level) for level in sorted(quantile_levels)]
    for lower, upper in symmetric_pairs(quantile_levels):
        delta = calibration.delta(lower, upper)
        aligned = delta.reindex(steps.clip(upper=delta.index.max())).to_numpy()
        low_col, high_col = quantile_col(lower), quantile_col(upper)
        if calibration.mode == "additive":
            out[low_col] = out[low_col] - aligned
            out[high_col] = out[high_col] + aligned
        else:
            width = (out[high_col] - out[low_col]).to_numpy()
            out[low_col] = out[low_col] - aligned * width
            out[high_col] = out[high_col] + aligned * width
    out[columns] = np.sort(out[columns].to_numpy("float64"), axis=1)
    if quantile_col(0.5) in out.columns:
        out[PREDICTION_COL] = out[quantile_col(0.5)]
    return out


class ConformalForecaster:
    """Any ``Forecaster`` plus a :class:`QuantileCalibration`, as one ``Forecaster``."""

    def __init__(
        self, base: Forecaster, calibration: QuantileCalibration, *, name: str | None = None
    ) -> None:
        self.base = base
        self.calibration = calibration
        self.name = name or f"{base.name}+conformal_{calibration.mode}"

    def predict(  # noqa: D102 - the ``Forecaster`` protocol documents this method
        self,
        context: pd.DataFrame,
        *,
        prediction_length: int,
        quantile_levels: Sequence[float],
        freq: str,
        target: str | Sequence[str] = TARGET_COL,
        future_covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        forecast = self.base.predict(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            freq=freq,
            target=target,
            future_covariates=future_covariates,
        )
        return conformalize(forecast, self.calibration, quantile_levels=quantile_levels)


class AdaptiveConformalForecaster:
    """Conformal calibration that keeps learning from its own realized forecasts.

    :class:`ConformalForecaster` learns its adjustment once. This one starts from the same
    calibration frame and then, at every call, looks in the ``context`` it is given for the
    actuals of the forecasts it issued before: each fully realized forecast joins the pool of
    scores, and the adjustment is recomputed before the new forecast leaves. Nothing from after
    the cutoff is ever used, so it runs through the backtest harness like any other model, as
    long as the windows are visited in time order (they are).

    Parameters
    ----------
    base
        The forecaster whose intervals are calibrated.
    calibration_frame
        A backtest frame with ``actual`` (and ``window``), the initial pool of scores.
    mode, by_step
        As in :func:`calibrate`.
    window
        Keep only the scores of the most recent ``window`` forecasts (a rolling pool). ``None``
        keeps every forecast seen so far (an expanding pool).
    gamma
        Step of adaptive conformal inference (Gibbs and Candès, 2021). After each realized
        forecast the miss rate each pair aims for moves by ``gamma * (nominal - observed)``:
        down (wider bands) after a forecast that missed more than its nominal rate, up after
        one that missed less. ``0`` turns it off.

    Only single-target forecasts are supported: the realized values are read from the
    ``target`` column of the context.
    """

    _MIN_RATE = 0.001

    def __init__(
        self,
        base: Forecaster,
        calibration_frame: pd.DataFrame,
        *,
        mode: Mode = "additive",
        by_step: bool = True,
        window: int | None = None,
        gamma: float = 0.0,
        name: str | None = None,
    ) -> None:
        if window is not None and window < 1:
            raise ValueError(f"window must be at least 1 forecast, got {window}.")
        if gamma < 0:
            raise ValueError(f"gamma must not be negative, got {gamma}.")
        if WINDOW_COL not in calibration_frame.columns:
            raise ValueError(f"The calibration frame lacks the {WINDOW_COL!r} column.")
        self.base = base
        self.mode: Mode = mode
        self.by_step = by_step
        self.window = window
        self.gamma = gamma
        self.name = name or f"{base.name}+adaptive_conformal_{mode}"
        rows = calibration_frame[calibration_frame[ACTUAL_COL].notna()]
        self._pool = [group for _, group in rows.groupby(WINDOW_COL, sort=True)]
        self._pending: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        self._miss_rates: dict[tuple[float, float], float] = {}
        self._next_window = -1
        self.history: list[dict[str, float]] = []

    @property
    def miss_rates(self) -> dict[tuple[float, float], float]:
        """The miss rate each quantile pair currently aims for (empty until ``gamma`` moves one)."""
        return dict(self._miss_rates)

    def _absorb(self, context: pd.DataFrame, target: str, levels: Sequence[float]) -> None:
        """Move every fully realized pending forecast into the pool and update the miss rates."""
        actuals = context[[ID_COL, TIMESTAMP_COL, target]].rename(columns={target: ACTUAL_COL})
        still_pending = []
        for raw, adjusted in self._pending:
            realized = raw.merge(actuals, on=[ID_COL, TIMESTAMP_COL], how="left")
            if realized[ACTUAL_COL].isna().any():
                still_pending.append((raw, adjusted))
                continue
            self._pool.append(realized.assign(**{WINDOW_COL: self._next_window}))
            self._next_window -= 1
            if self.gamma > 0:
                actual = realized[ACTUAL_COL].to_numpy("float64")
                for lower, upper in symmetric_pairs(levels):
                    nominal = 1.0 - (upper - lower)
                    low = adjusted[quantile_col(lower)].to_numpy("float64")
                    high = adjusted[quantile_col(upper)].to_numpy("float64")
                    observed = float(((actual < low) | (actual > high)).mean())
                    current = self._miss_rates.get((lower, upper), nominal)
                    moved = current + self.gamma * (nominal - observed)
                    self._miss_rates[(lower, upper)] = float(
                        np.clip(moved, self._MIN_RATE, 1.0 - self._MIN_RATE)
                    )
        self._pending = still_pending
        if self.window is not None:
            self._pool = self._pool[-self.window :]

    def predict(  # noqa: D102 - the ``Forecaster`` protocol documents this method
        self,
        context: pd.DataFrame,
        *,
        prediction_length: int,
        quantile_levels: Sequence[float],
        freq: str,
        target: str | Sequence[str] = TARGET_COL,
        future_covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not isinstance(target, str):
            raise ValueError("AdaptiveConformalForecaster supports a single target.")
        self._absorb(context, target, quantile_levels)
        raw = self.base.predict(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            freq=freq,
            target=target,
            future_covariates=future_covariates,
        )
        calibration = calibrate(
            pd.concat(self._pool, ignore_index=True),
            quantile_levels=quantile_levels,
            mode=self.mode,
            by_step=self.by_step,
            miss_rates=self._miss_rates,
        )
        adjusted = conformalize(raw, calibration, quantile_levels=quantile_levels)
        self._pending.append((raw, adjusted))
        widest = symmetric_pairs(quantile_levels)[0]
        self.history.append(
            {
                "n_pool_forecasts": float(len(self._pool)),
                "mean_delta_widest": float(calibration.delta(*widest).mean()),
                "miss_rate_widest": self._miss_rates.get(widest, 1.0 - (widest[1] - widest[0])),
            }
        )
        return adjusted


def coverage_by_period(
    backtest_frame: pd.DataFrame,
    *,
    lower: float,
    upper: float,
    period: str = "M",
    model_column: str = MODEL_COL,
) -> pd.DataFrame:
    """Share of actuals inside ``[lower, upper]`` per model and calendar period of the timestamp.

    This is the diagnostic the yearly coverage number hides: a band that is on target on
    average and too narrow every winter.
    """
    rows = backtest_frame[backtest_frame[ACTUAL_COL].notna()]
    frame = pd.DataFrame(
        {
            model_column: rows[model_column].to_numpy(),
            "period": rows[TIMESTAMP_COL].dt.to_period(period).astype(str).to_numpy(),
            "inside": rows[ACTUAL_COL]
            .between(rows[quantile_col(lower)], rows[quantile_col(upper)])
            .to_numpy(),
        }
    )
    return (
        frame.groupby([model_column, "period"], as_index=False)
        .agg(coverage=("inside", "mean"), n=("inside", "size"))
        .assign(nominal=upper - lower)
    )
