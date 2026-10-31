"""Watch a deployed forecaster: log predictions, join actuals later, alert on drift.

A backtest tells you whether a model *was* good on history. Monitoring answers the only question
production cares about (is it still good?), and the two differ in one structural way: in
production the actuals arrive *after* the forecast, often much later. Everything here is built
around that delay.

The flow has three steps, and each is a function:

1. :func:`log_predictions` writes one row per forecast step, stamped with the model version and a
   hash of the exact input the model saw. The hash is what makes an old forecast reproducible:
   without it, "the model changed" and "the input changed" are indistinguishable six weeks later.
2. :func:`join_actuals` attaches the observed values when they arrive, leaving rows unresolved
   until then.
3. :func:`rolling_metrics` computes error and interval coverage over a trailing window, and
   :func:`retraining_alerts` turns a threshold on those into the one decision this is all for.

:func:`input_drift` is the complement: errors tell you something broke *after* the fact, while a
shift in the input distribution can be seen before the actuals land.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from c2p.schema import ID_COL, PREDICTION_COL, TARGET_NAME_COL, TIMESTAMP_COL, quantile_col

__all__ = [
    "ACTUAL_COL",
    "ALERT_COL",
    "CUTOFF_COL",
    "INPUT_HASH_COL",
    "MODEL_VERSION_COL",
    "append_log",
    "context_hash",
    "input_drift",
    "join_actuals",
    "log_predictions",
    "log_summary",
    "read_log",
    "retraining_alerts",
    "rolling_metrics",
]

logger = logging.getLogger(__name__)

MODEL_VERSION_COL = "model_version"
INPUT_HASH_COL = "input_hash"
CUTOFF_COL = "cutoff"
ACTUAL_COL = "actual"
ALERT_COL = "alert"

_HASH_LENGTH = 16


def context_hash(context: pd.DataFrame, *, columns: Sequence[str] | None = None) -> str:
    """Stable short hash of the exact input a forecast was made from.

    Hashes the values (not the row order, which the schema already fixes) of ``columns``, or of
    every column except the id and timestamp. NaN hashes as NaN, so a gap that was filled later
    changes the hash, which is the point: it is how you tell a model change from a data change
    when you come back to an old forecast.
    """
    if context.empty:
        raise ValueError("Cannot hash an empty context.")
    columns = list(
        columns if columns is not None else [c for c in context.columns if c not in (ID_COL,)]
    )
    missing = [column for column in columns if column not in context.columns]
    if missing:
        raise ValueError(f"The context lacks the columns {missing}.")
    frame = context[columns]
    digest = hashlib.sha256()
    digest.update(",".join(columns).encode("utf-8"))
    for column in columns:
        values = frame[column]
        if pd.api.types.is_datetime64_any_dtype(values):
            payload = values.astype("int64").to_numpy()
        elif pd.api.types.is_numeric_dtype(values):
            payload = values.to_numpy("float64")
        else:
            payload = values.astype(str).to_numpy().astype("S")
        digest.update(np.ascontiguousarray(payload).tobytes())
    return digest.hexdigest()[:_HASH_LENGTH]


def log_predictions(
    forecast: pd.DataFrame,
    context: pd.DataFrame,
    *,
    model_version: str,
    cutoff: pd.Timestamp | None = None,
    quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),
    extra: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Turn one forecast into log rows: one per series, target and horizon step.

    Columns: ``model_version``, ``input_hash``, ``cutoff``, the forecast-frame keys, the point
    forecast and the requested quantile columns, plus anything in ``extra``. The actual value is
    deliberately absent: it does not exist yet.
    """
    if not model_version:
        raise ValueError("model_version must be a non-empty string: it is what you compare across.")
    required = [ID_COL, TIMESTAMP_COL, TARGET_NAME_COL, PREDICTION_COL]
    levels = [quantile_col(level) for level in quantile_levels]
    missing = [column for column in (*required, *levels) if column not in forecast.columns]
    if missing:
        raise ValueError(f"The forecast lacks the columns {missing}.")
    cutoff = (
        pd.Timestamp(cutoff) if cutoff is not None else pd.Timestamp(context[TIMESTAMP_COL].max())
    )
    rows = forecast[[*required, *levels]].copy()
    rows.insert(0, CUTOFF_COL, cutoff)
    rows.insert(0, INPUT_HASH_COL, context_hash(context))
    rows.insert(0, MODEL_VERSION_COL, model_version)
    for key, value in (extra or {}).items():
        rows[key] = value
    return rows


def append_log(rows: pd.DataFrame, path: Path | str) -> Path:
    """Append log rows to a CSV, writing the header only when the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def read_log(path: Path | str) -> pd.DataFrame:
    """Read a prediction log written by :func:`append_log`, parsing its timestamps."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist yet; log a forecast first.")
    log = pd.read_csv(path)
    for column in (TIMESTAMP_COL, CUTOFF_COL):
        if column in log.columns:
            log[column] = pd.to_datetime(log[column])
    return log


def join_actuals(log: pd.DataFrame, actuals: pd.DataFrame, *, target: str) -> pd.DataFrame:
    """Attach observed values to logged predictions, keeping rows that have none yet.

    ``actuals`` is a series frame with ``target``. Rows whose actual has not arrived keep NaN in
    ``actual``, so the share of resolved rows is itself a monitoring signal.
    """
    for frame, name in ((log, "log"), (actuals, "actuals")):
        missing = [c for c in (ID_COL, TIMESTAMP_COL) if c not in frame.columns]
        if missing:
            raise ValueError(f"The {name} frame lacks the columns {missing}.")
    if target not in actuals.columns:
        raise ValueError(f"The actuals frame lacks the target column {target!r}.")
    observed = actuals[[ID_COL, TIMESTAMP_COL, target]].rename(columns={target: ACTUAL_COL})
    joined = log.merge(observed, on=[ID_COL, TIMESTAMP_COL], how="left", validate="m:1")
    resolved = int(joined[ACTUAL_COL].notna().sum())
    logger.info("Joined actuals: %d of %d logged rows resolved.", resolved, len(joined))
    return joined


def rolling_metrics(
    joined: pd.DataFrame,
    *,
    window: int,
    quantile_levels: tuple[float, float] = (0.1, 0.9),
    nominal: float = 0.8,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Rolling WAPE and interval coverage per cutoff, over a trailing ``window`` of cutoffs.

    One row per (model version, cutoff): the forecasts made at that cutoff are scored, then
    averaged over the last ``window`` cutoffs. Unresolved rows are ignored. ``nominal`` is
    carried through so the alert rule and the figure share one definition of the promise.
    """
    if window < 1:
        raise ValueError(f"window must be a positive number of cutoffs, got {window}.")
    low, high = (quantile_col(level) for level in quantile_levels)
    missing = [c for c in (ACTUAL_COL, PREDICTION_COL, low, high) if c not in joined.columns]
    if missing:
        raise ValueError(f"The joined log lacks the columns {missing}; run join_actuals first.")
    resolved = joined[joined[ACTUAL_COL].notna()]
    if resolved.empty:
        raise ValueError("No logged prediction has an actual yet: nothing to score.")
    per_cutoff = (
        resolved.assign(
            abs_error=(resolved[PREDICTION_COL] - resolved[ACTUAL_COL]).abs(),
            inside=resolved[ACTUAL_COL].between(resolved[low], resolved[high]),
        )
        .groupby([MODEL_VERSION_COL, CUTOFF_COL], as_index=False)
        .agg(
            abs_error=("abs_error", "sum"),
            total=(ACTUAL_COL, "sum"),
            inside=("inside", "mean"),
            n=(ACTUAL_COL, "size"),
        )
    )
    frames = []
    for _version, rows in per_cutoff.groupby(MODEL_VERSION_COL, sort=True):
        group = rows.sort_values(CUTOFF_COL)
        rolling = group.rolling(window, min_periods=min_periods or window)
        frames.append(
            group.assign(
                wape=rolling["abs_error"].sum() / rolling["total"].sum(),
                coverage=rolling["inside"].mean(),
                nominal=nominal,
                window=window,
            )
        )
    columns = [MODEL_VERSION_COL, CUTOFF_COL, "n", "wape", "coverage", "nominal", "window"]
    return pd.concat(frames, ignore_index=True)[columns]


def retraining_alerts(rolling: pd.DataFrame, *, wape_threshold: float) -> pd.DataFrame:
    """Flag the cutoffs where the rolling WAPE crossed ``wape_threshold``.

    Adds ``alert`` (bool) and ``first_breach`` (True only on the cutoff where a run of breaches
    starts), so an operator gets one notification per episode instead of one per day.
    """
    if not 0 < wape_threshold < 1:
        raise ValueError(f"wape_threshold must be a fraction in (0, 1), got {wape_threshold}.")
    if "wape" not in rolling.columns:
        raise ValueError("rolling lacks a 'wape' column; run rolling_metrics first.")
    frames = []
    for _, rows in rolling.groupby(MODEL_VERSION_COL, sort=True):
        group = rows.sort_values(CUTOFF_COL).copy()
        group[ALERT_COL] = group["wape"] > wape_threshold
        group["first_breach"] = group[ALERT_COL] & ~group[ALERT_COL].shift(fill_value=False)
        frames.append(group)
    out = pd.concat(frames, ignore_index=True)
    out["wape_threshold"] = wape_threshold
    logger.info(
        "Retraining rule at WAPE > %.3f: %d breaching cutoffs in %d episodes.",
        wape_threshold,
        int(out[ALERT_COL].sum()),
        int(out["first_breach"].sum()),
    )
    return out


def input_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: Sequence[str],
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Compare the distribution of each input column between a reference period and now.

    Returns one row per column with both means, both standard deviations, the shift in means
    expressed in reference standard deviations (``z_shift``) and the share of current values
    outside the reference's ``quantiles`` range. Deliberately simple: a drift number you can
    explain beats a test statistic nobody acts on.
    """
    missing = sorted({*columns} - set(reference.columns) - set(current.columns))
    if missing:
        raise ValueError(f"Columns {missing} are not in both frames.")
    low, high = min(quantiles), max(quantiles)
    rows = []
    for column in columns:
        past = reference[column].to_numpy("float64")
        now = current[column].to_numpy("float64")
        past_std = float(np.nanstd(past))
        bounds = np.nanquantile(past, [low, high])
        outside = float(np.nanmean((now < bounds[0]) | (now > bounds[1])))
        rows.append(
            {
                "column": column,
                "reference_mean": float(np.nanmean(past)),
                "current_mean": float(np.nanmean(now)),
                "reference_std": past_std,
                "current_std": float(np.nanstd(now)),
                "z_shift": float((np.nanmean(now) - np.nanmean(past)) / past_std)
                if past_std > 0
                else float("nan"),
                "share_outside_reference_range": outside,
            }
        )
    return pd.DataFrame(rows)


def log_summary(log: pd.DataFrame) -> str:
    """One-line description of a prediction log, for a notebook or a cron job's output."""
    versions = sorted(log[MODEL_VERSION_COL].unique())
    hashes = log[INPUT_HASH_COL].nunique()
    resolved = (
        f", {int(log[ACTUAL_COL].notna().sum()):,} resolved" if ACTUAL_COL in log.columns else ""
    )
    return (
        f"{len(log):,} logged forecasts{resolved}; versions {versions}; "
        f"{hashes:,} distinct inputs; cutoffs {log[CUTOFF_COL].min()} to {log[CUTOFF_COL].max()}"
    )
