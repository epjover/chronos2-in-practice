"""Chronos-2 behind the shared ``Forecaster`` protocol, plus input checks every forecaster reuses.

Typical use::

    from c2p.models import Chronos2Forecaster, load_chronos2
    from c2p.schema import EVAL_QUANTILES

    forecaster = Chronos2Forecaster(load_chronos2())
    forecast = forecaster.predict(
        context, prediction_length=24, quantile_levels=EVAL_QUANTILES, freq="h"
    )

The pipeline is loaded once and wrapped; the wrapper only translates between our frames and
``Chronos2Pipeline.predict_df`` and adds the checks the library skips when ``freq`` is given.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from c2p.config import get_device
from c2p.schema import (
    ID_COL,
    PREDICTION_COL,
    TARGET_COL,
    TARGET_NAME_COL,
    TIMESTAMP_COL,
    quantile_col,
)

if TYPE_CHECKING:
    from chronos import BaseChronosPipeline, Chronos2Pipeline, ChronosBoltPipeline

logger = logging.getLogger(__name__)

CHRONOS2 = "amazon/chronos-2"
CHRONOS2_SMALL = "autogluon/chronos-2-small"
# Cross-learning batch size suggested by the ``predict_df`` docstring ("around 100, as used in
# the Chronos-2 technical report"); larger batches "deviate from the maximum group size used
# during pretraining" and make the forecast depend on how the frame is cut into batches.
CROSS_LEARNING_BATCH_HINT = 100
CHRONOS2_SYNTH = "autogluon/chronos-2-synth"
CHRONOS_BOLT_BASE = "amazon/chronos-bolt-base"


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


def parameter_count(pipeline: BaseChronosPipeline) -> int:
    """Number of parameters of the network wrapped by a Chronos pipeline."""
    return sum(parameter.numel() for parameter in pipeline.inner_model.parameters())


DTYPES = ("float32", "bfloat16", "float16")


def load_chronos2(
    model_id: str = CHRONOS2, *, device: str | None = None, dtype: str = "float32"
) -> Chronos2Pipeline:
    """Load a Chronos-2 pipeline from the Hugging Face Hub or a local folder.

    Parameters
    ----------
    model_id
        Hub id (e.g. ``CHRONOS2``, ``CHRONOS2_SMALL``) or path to a fine-tuned checkpoint.
    device
        ``"cpu"`` or ``"cuda"``; ``None`` uses ``c2p.config.get_device()``.
    dtype
        Precision of the weights: ``"float32"`` (the checkpoint's own), or ``"bfloat16"`` /
        ``"float16"`` for half precision, which on a GPU roughly doubles the throughput and
        lowers the memory. Half precision is for inference on a GPU: on a CPU it is slower, and
        fine-tuning should start from ``float32``. ``bench/cheaper_variants_accuracy.py``
        measures what it does to the forecasts.

    Raises ``TypeError`` when ``model_id`` is a checkpoint of another Chronos family, such as
    Chronos-Bolt (use ``load_chronos_bolt`` for those).
    """
    import torch
    from chronos import Chronos2Pipeline

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}.")
    _check_chronos2_checkpoint(model_id)
    device = device or get_device()
    # float32 is the checkpoint's own precision: leave the library's default untouched.
    precision = {} if dtype == "float32" else {"dtype": getattr(torch, dtype)}
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device, **precision)
    _log_loaded(model_id, device, pipeline)
    return pipeline


def _check_chronos2_checkpoint(model_id: str) -> None:
    """Raise ``TypeError`` if the checkpoint's config names a pipeline other than Chronos-2.

    Only the small config file is read. LoRA adapters (saved by ``Chronos2Pipeline.fit``) have no
    such config and are left to the library, which loads them on top of their base model.
    """
    from transformers import AutoConfig
    from transformers.utils.peft_utils import find_adapter_config_file

    if find_adapter_config_file(model_id) is not None:
        return
    pipeline_class = getattr(AutoConfig.from_pretrained(model_id), "chronos_pipeline_class", None)
    if pipeline_class is not None and pipeline_class != "Chronos2Pipeline":
        raise TypeError(
            f"{model_id!r} is a {pipeline_class} checkpoint, not a Chronos-2 one. "
            "Use load_chronos_bolt() or c2p.baselines.ChronosBolt for Chronos-Bolt models."
        )


def load_chronos_bolt(
    model_id: str = CHRONOS_BOLT_BASE, *, device: str | None = None
) -> ChronosBoltPipeline:
    """Load a Chronos-Bolt pipeline; raise ``TypeError`` if ``model_id`` is another family."""
    from chronos import BaseChronosPipeline, ChronosBoltPipeline

    device = device or get_device()
    pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device)
    if not isinstance(pipeline, ChronosBoltPipeline):
        raise TypeError(
            f"{model_id!r} loads a {type(pipeline).__name__}, not a ChronosBoltPipeline. "
            "Use load_chronos2() for Chronos-2 models."
        )
    _log_loaded(model_id, device, pipeline)
    return pipeline


def _log_loaded(model_id: str, device: str, pipeline: BaseChronosPipeline) -> None:
    logger.info(
        "Loaded %s on %s (%s parameters).", model_id, device, f"{parameter_count(pipeline):,}"
    )


# --------------------------------------------------------------------------------------------
# Checks and output formatting shared by every forecaster (Chronos and baselines)
# --------------------------------------------------------------------------------------------


def assert_regular_grid(
    df: pd.DataFrame,
    freq: str,
    *,
    id_column: str = ID_COL,
    timestamp_column: str = TIMESTAMP_COL,
) -> None:
    """Raise ``ValueError`` unless every series sits on a regular ``freq`` grid without gaps.

    Each series must be sorted, free of duplicated timestamps and have one row per grid step
    (missing values are NaN rows, not missing rows). Chronos only checks this when it has to
    infer the frequency: with ``freq`` given, a series with missing rows is silently treated as
    contiguous and the forecast degrades. Statistical baselines make the same assumption.
    """
    offset = _to_offset(freq)
    problems = []
    for item_id, timestamps in df.groupby(id_column, sort=False)[timestamp_column]:
        problem = _grid_problem(pd.DatetimeIndex(timestamps), offset)
        if problem is not None:
            problems.append(f"{item_id!r}: {problem}")
    if problems:
        raise ValueError(
            f"{len(problems)} series are not on a regular {freq!r} grid "
            f"({'; '.join(problems[:3])}). Reindex every series to a regular grid with NaN "
            "for missing values (c2p.data.validate_and_prepare does this) before forecasting."
        )


def _grid_problem(timestamps: pd.DatetimeIndex, offset: pd.offsets.BaseOffset) -> str | None:
    """Describe why ``timestamps`` is not a regular grid, or return ``None`` if it is."""
    if timestamps.has_duplicates:
        return "duplicated timestamps"
    if not timestamps.is_monotonic_increasing:
        return "timestamps are not sorted in ascending order"
    expected = pd.date_range(timestamps[0], periods=len(timestamps), freq=offset)
    if timestamps.equals(expected):
        return None
    first = int(np.flatnonzero(timestamps != expected)[0])
    if first == 0:
        return f"first timestamp {timestamps[0]} is not on the grid"
    return f"expected {expected[first]} after {timestamps[first - 1]}, found {timestamps[first]}"


def validate_forecast_inputs(
    context: pd.DataFrame,
    *,
    target: str | Sequence[str],
    quantile_levels: Sequence[float],
    prediction_length: int,
    freq: str,
) -> tuple[list[str], list[float]]:
    """Check a forecast request and return the target names and quantile levels as lists.

    Raises ``ValueError`` (or ``TypeError`` for wrong types) with an actionable message when the
    context lacks columns, has missing or categorical series ids, non-numeric targets or
    timezone-aware timestamps, is not on a regular grid, or when the horizon or quantile levels
    are invalid.
    """
    targets = _as_target_list(target)
    levels = _as_quantile_list(quantile_levels)
    if isinstance(prediction_length, bool) or not isinstance(prediction_length, int | np.integer):
        raise TypeError(f"prediction_length must be an int, got {prediction_length!r}.")
    if prediction_length < 1:
        raise ValueError(f"prediction_length must be >= 1, got {prediction_length}.")
    _check_context_columns(context, targets)
    assert_regular_grid(context, freq)
    return targets, levels


def _to_offset(freq: str) -> pd.offsets.BaseOffset:
    try:
        return to_offset(freq)
    except ValueError as error:
        raise ValueError(
            f"freq={freq!r} is not a pandas frequency string (e.g. 'h', 'D', '15min')."
        ) from error


def _as_target_list(target: str | Sequence[str]) -> list[str]:
    targets = [target] if isinstance(target, str) else list(target)
    if not targets:
        raise ValueError("At least one target column is required.")
    if not all(isinstance(name, str) for name in targets):
        raise TypeError(f"Target column names must be strings, got {targets!r}.")
    if len(set(targets)) != len(targets):
        raise ValueError(f"Duplicated target columns: {targets}.")
    return targets


def _as_quantile_list(quantile_levels: Sequence[float]) -> list[float]:
    levels = [float(level) for level in quantile_levels]
    if not levels:
        raise ValueError("At least one quantile level is required.")
    if len(set(levels)) != len(levels):
        raise ValueError(f"Duplicated quantile levels: {levels}.")
    for level in levels:
        quantile_col(level)  # raises if the level is outside (0, 1)
    return levels


def _check_context_columns(context: pd.DataFrame, targets: list[str]) -> None:
    if context.empty:
        raise ValueError("The context frame is empty.")
    missing = [col for col in (ID_COL, TIMESTAMP_COL, *targets) if col not in context.columns]
    if missing:
        raise ValueError(f"The context frame lacks the columns {missing}.")
    _check_ids(context[ID_COL])
    timestamps = context[TIMESTAMP_COL]
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            f"{TIMESTAMP_COL!r} is timezone-aware ({timestamps.dtype}); Chronos needs naive "
            "timestamps. Convert with .dt.tz_convert('UTC').dt.tz_localize(None)."
        )
    if not pd.api.types.is_datetime64_dtype(timestamps):
        raise ValueError(f"{TIMESTAMP_COL!r} must be datetime64, got {timestamps.dtype}.")
    non_numeric = [col for col in targets if not pd.api.types.is_numeric_dtype(context[col])]
    if non_numeric:
        raise ValueError(f"Target columns must be numeric: {non_numeric}.")


def _check_ids(ids: pd.Series) -> None:
    """Reject series ids that Chronos would silently misalign.

    Chronos splits the frame with ``value_counts(sort=False)``, which drops missing ids and, for
    a categorical column, follows the category order and counts unused categories as empty
    series: the forecasts then belong to the wrong series.
    """
    if isinstance(ids.dtype, pd.CategoricalDtype):
        raise ValueError(
            f"{ID_COL!r} is categorical; Chronos misaligns series with categorical ids. "
            "Convert it with .astype(str)."
        )
    n_missing = int(ids.isna().sum())
    if n_missing:
        raise ValueError(f"{ID_COL!r} has {n_missing} missing values; every row needs a series id.")


def warn_untrained_levels(
    quantile_levels: Sequence[float],
    trained_levels: Sequence[float],
    *,
    model_name: str,
    seen: set[tuple[float, ...]] | None = None,
) -> None:
    """Log a warning for levels outside the range a quantile model was trained on.

    Chronos-2 and Chronos-Bolt interpolate levels between trained ones and clamp levels outside
    the trained range to the nearest trained level. The library already logs a generic warning;
    this one names the model and states the consequence: those intervals are too narrow.

    ``seen`` remembers the sets of levels already reported. A forecaster passes its own, so a
    backtest of 365 windows logs the warning once and not 365 times.
    """
    low, high = min(trained_levels), max(trained_levels)
    outside = tuple(level for level in quantile_levels if not low <= level <= high)
    if not outside or (seen is not None and outside in seen):
        return
    if seen is not None:
        seen.add(outside)
    logger.warning(
        "%s was trained on quantile levels in [%s, %s]; the levels %s are clamped to the "
        "nearest trained level, so their intervals are too narrow.",
        model_name,
        low,
        high,
        list(outside),
    )


_FORECAST_KEYS = [ID_COL, TARGET_NAME_COL, TIMESTAMP_COL]


def finalize_forecast(
    raw: pd.DataFrame,
    *,
    context: pd.DataFrame,
    targets: Sequence[str],
    quantile_levels: Sequence[float],
    prediction_length: int,
    freq: str,
    model_name: str,
) -> pd.DataFrame:
    """Turn a model output into a checked forecast frame (see ``c2p.schema``).

    Keeps exactly the schema columns, with the context's id and timestamp dtypes and float64
    forecasts. Raises ``ValueError`` unless there is exactly one row per (item, target, step) on
    the grid after each series' last context timestamp and no forecast is NaN, so a model that
    drops, duplicates or misaligns series fails loudly. Rows follow the context: items in order
    of first appearance, targets in the requested order, then timestamps.
    """
    value_columns = [PREDICTION_COL, *(quantile_col(level) for level in quantile_levels)]
    columns = [ID_COL, TIMESTAMP_COL, TARGET_NAME_COL, *value_columns]
    missing = [col for col in columns if col not in raw.columns]
    if missing:
        raise ValueError(f"{model_name} returned a frame without the columns {missing}.")
    key_dtypes = {ID_COL: context[ID_COL].dtype, TIMESTAMP_COL: context[TIMESTAMP_COL].dtype}
    forecast = raw[columns].astype(key_dtypes | dict.fromkeys(value_columns, "float64"))
    expected = _expected_forecast_keys(context, targets, prediction_length, freq)
    forecast = _align_to_keys(forecast, expected, model_name=model_name)
    has_nan = forecast[value_columns].isna().any(axis=1)
    if has_nan.any():
        pairs = forecast.loc[has_nan, [ID_COL, TARGET_NAME_COL]].drop_duplicates()
        raise ValueError(
            f"{model_name} produced NaN forecasts for (item, target) pairs "
            f"{list(pairs.itertuples(index=False, name=None))[:5]}."
        )
    return forecast[columns]


def _expected_forecast_keys(
    context: pd.DataFrame, targets: Sequence[str], prediction_length: int, freq: str
) -> pd.MultiIndex:
    """(item, target, timestamp) of every row the forecast must have, in output order."""
    offset = _to_offset(freq)
    last_timestamps = context.groupby(ID_COL, sort=False)[TIMESTAMP_COL].max()
    blocks = [
        pd.DataFrame(
            {
                ID_COL: item_id,
                TARGET_NAME_COL: target,
                TIMESTAMP_COL: pd.date_range(last + offset, periods=prediction_length, freq=offset),
            }
        )
        for item_id, last in last_timestamps.items()
        for target in targets
    ]
    keys = pd.concat(blocks, ignore_index=True)
    key_dtypes = {ID_COL: context[ID_COL].dtype, TIMESTAMP_COL: context[TIMESTAMP_COL].dtype}
    return pd.MultiIndex.from_frame(keys.astype(key_dtypes))


def _align_to_keys(
    forecast: pd.DataFrame, expected: pd.MultiIndex, *, model_name: str
) -> pd.DataFrame:
    """Reorder ``forecast`` rows to ``expected``; raise if rows are duplicated, missing or extra."""
    actual = pd.MultiIndex.from_frame(forecast[_FORECAST_KEYS])
    if actual.has_duplicates:
        examples = actual[actual.duplicated()].unique()[:3].tolist()
        raise ValueError(
            f"{model_name} returned duplicated (item, target, timestamp) rows, e.g. {examples}."
        )
    missing = expected.difference(actual, sort=False)
    unexpected = actual.difference(expected, sort=False)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"{model_name} returned a forecast that does not match the request: "
            f"{len(missing)} expected (item, target, timestamp) rows are missing "
            f"(e.g. {missing[:3].tolist()}) and {len(unexpected)} rows are unexpected "
            f"(e.g. {unexpected[:3].tolist()}). Every (item, target) needs prediction_length "
            "rows continuing its grid after the last context timestamp."
        )
    return forecast.set_index(_FORECAST_KEYS).reindex(expected).reset_index()


# --------------------------------------------------------------------------------------------
# Chronos-2 forecaster
# --------------------------------------------------------------------------------------------


class Chronos2Forecaster:
    """Chronos-2 zero-shot forecaster implementing ``c2p.schema.Forecaster``.

    Covariates follow Chronos' own rule: every column of ``context`` other than the id, the
    timestamp and the targets is a covariate; the ones also present in ``future_covariates`` are
    known-future covariates, the others past-only covariates. Several targets are forecast
    jointly (multivariate), not one by one.

    Parameters
    ----------
    pipeline
        A loaded ``Chronos2Pipeline`` (see ``load_chronos2``).
    name
        Model name used in results tables.
    context_length
        Maximum number of past steps fed to the model; ``None`` uses the model's default.
    cross_learning
        Forecast all series in a batch jointly, sharing information across them.
    batch_size
        Number of series (targets and covariates included) per forward pass.
    """

    def __init__(
        self,
        pipeline: Chronos2Pipeline,
        *,
        name: str = "chronos-2",
        context_length: int | None = None,
        cross_learning: bool = False,
        batch_size: int = 256,
    ) -> None:
        from chronos import Chronos2Pipeline

        if not isinstance(pipeline, Chronos2Pipeline):
            raise TypeError(
                f"Expected a Chronos2Pipeline, got {type(pipeline).__name__}. "
                "Use c2p.baselines.ChronosBolt for Chronos-Bolt models."
            )
        if context_length is not None and context_length < 1:
            raise ValueError(f"context_length must be >= 1 or None, got {context_length}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        self.pipeline = pipeline
        self.name = name
        self._reported_levels: set[tuple[float, ...]] = set()
        self._warned_batching = False
        self.context_length = context_length
        self.cross_learning = cross_learning
        self.batch_size = batch_size

    def __repr__(self) -> str:
        return (
            f"Chronos2Forecaster(name={self.name!r}, context_length={self.context_length}, "
            f"cross_learning={self.cross_learning}, batch_size={self.batch_size})"
        )

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
        """Forecast ``prediction_length`` steps after the end of each series in ``context``.

        ``freq`` is passed to Chronos so it does not have to infer it; because that disables
        Chronos' gap detection, the regular grid is checked here first.
        """
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
        n_series = int(context[ID_COL].nunique())
        oversized = self.cross_learning and n_series > CROSS_LEARNING_BATCH_HINT
        if oversized and not self._warned_batching:  # once per forecaster, not per window
            self._warned_batching = True
            warnings.warn(
                f"cross_learning=True over {n_series} series with batch_size={self.batch_size}: "
                "every series in a batch joins one group, so the forecast depends on how the "
                "frame is cut into batches. The library suggests about "
                f"{CROSS_LEARNING_BATCH_HINT} series per batch (the size used in the technical "
                "report); prefer that and frames of related series.",
                stacklevel=2,
            )
        raw = self.pipeline.predict_df(
            context,
            future_df=future_covariates,
            id_column=ID_COL,
            timestamp_column=TIMESTAMP_COL,
            target=targets,
            prediction_length=prediction_length,
            quantile_levels=levels,
            batch_size=self.batch_size,
            context_length=self.context_length,
            cross_learning=self.cross_learning,
            freq=freq,
        )
        return finalize_forecast(
            raw,
            context=context,
            targets=targets,
            quantile_levels=levels,
            prediction_length=prediction_length,
            freq=freq,
            model_name=self.name,
        )
