"""Shared data contract: column names, quantile levels and the ``Forecaster`` protocol.

All modules exchange pandas DataFrames in the long format used by
``Chronos2Pipeline.predict_df``:

- **Series frame** (inputs): one row per (series, timestamp) with ``item_id`` (str),
  ``timestamp`` (timezone-naive UTC ``datetime64``), one or more target columns (float) and
  optional covariate columns. Rows are sorted by ``item_id`` then ``timestamp`` and every series
  sits on a regular grid; missing values are NaN, never sentinels.
- **Future frame** (known covariates): ``item_id``, ``timestamp`` and the known-future covariate
  columns only, exactly one row per series and forecast step.
- **Forecast frame** (outputs): ``item_id``, ``timestamp``, ``target_name``, ``predictions``
  (point forecast, the median for quantile models) and one column per quantile level named
  ``quantile_col(q)``, e.g. ``"0.1"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd

__all__ = [
    "EVAL_QUANTILES",
    "ID_COL",
    "INTERVALS",
    "PREDICTION_COL",
    "TARGET_COL",
    "TARGET_NAME_COL",
    "TIMESTAMP_COL",
    "Forecaster",
    "quantile_col",
]

ID_COL = "item_id"
TIMESTAMP_COL = "timestamp"
TARGET_COL = "target"
TARGET_NAME_COL = "target_name"
PREDICTION_COL = "predictions"

# Levels used for evaluation: the nine deciles (as in fev-bench) plus the tails of the
# 98% interval.
EVAL_QUANTILES: tuple[float, ...] = (0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99)
# Central prediction intervals reported as coverage: nominal level -> (lower, upper) quantile.
INTERVALS: dict[float, tuple[float, float]] = {0.8: (0.1, 0.9), 0.98: (0.01, 0.99)}


def quantile_col(level: float) -> str:
    """Column name of a quantile level, matching ``Chronos2Pipeline.predict_df`` (``str(q)``)."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"Quantile levels must be in (0, 1), got {level}.")
    return str(level)


@runtime_checkable
class Forecaster(Protocol):
    """Anything that turns a context frame into a forecast frame.

    Implementations must use only ``context`` (past observations) and ``future_covariates``
    (known-future covariates for the horizon); they never see the values they forecast.
    """

    name: str

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
        """Forecast ``prediction_length`` steps after the end of each series in ``context``."""
        ...
