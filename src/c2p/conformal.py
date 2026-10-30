"""Make the prediction intervals honest: conformal calibration of Chronos-2's quantiles.

Every part of the series found the same defect (the 80% interval covers about 73% of the
hours, less in winter) and none of them fixed it, because the levers they pulled (context,
covariates, fine-tuning) all target the median. This module targets the bands directly, with the
one method that comes with a guarantee and no retraining: **conformalized quantile regression**
(CQR, Romano et al., 2019).

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
from collections.abc import Sequence
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
) -> QuantileCalibration:
    """Learn CQR adjustments from a backtest frame that carries ``actual``.

    For each symmetric pair the nonconformity score of a row is ``max(lower - y, y - upper)``
    (divided by the interval width in ``scaled`` mode), and ``delta`` is the empirical
    ``ceil((n + 1)(1 - alpha)) / n`` quantile of the scores, ``alpha`` being the pair's miss
    rate. With ``by_step`` the scores are grouped by horizon step, so a band that is honest one
    hour ahead and too narrow at hour 24 is corrected differently at each.
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
        alpha = 1.0 - (upper - lower)
        frame = pd.DataFrame({STEP_COL: steps.to_numpy(), "score": scores})
        for step, group in frame.groupby(STEP_COL, sort=True):
            n = len(group)
            level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
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


def coverage_by_period(
    backtest_frame: pd.DataFrame,
    *,
    lower: float,
    upper: float,
    period: str = "M",
    model_column: str = MODEL_COL,
) -> pd.DataFrame:
    """Share of actuals inside ``[lower, upper]`` per model and calendar period of the timestamp.

    This is the diagnostic the yearly coverage number hides: a band that is honest on average
    and dishonest every winter.
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
