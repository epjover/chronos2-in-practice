"""Fine-tune Chronos-2 with LoRA or on all weights, on an honest temporal split.

``Chronos2Pipeline.fit`` does the training; this module prepares its inputs from our series
frames, fixes three things it leaves to the caller, and measures what a fine-tune costs:

1. **Data**: train, validation and test periods never overlap, and every validation (or test)
   window takes its context only from earlier data (:func:`temporal_split`,
   :func:`window_series`, :func:`to_fit_inputs`).
2. **The model you evaluate is the model you saved**: the pipeline ``fit`` returns keeps
   traces of training. Without validation it stays in training mode (dropout makes every
   forecast random), and after a full fine-tune with mixed precision its forward pass stays
   wrapped in bfloat16 autocast (forecasts off by ~0.3%). :func:`finetune` returns the saved
   checkpoint reloaded from disk instead (LoRA adapters merged into the weights).
3. **Precision**: on Ampere or newer GPUs, ``fit`` enables TF32 for the whole process and does
   not switch it off, which changes every later GPU forecast. :func:`finetune` restores it.

Typical use::

    split = temporal_split(df, val_start="2017-07-01", test_start="2018-01-01")
    train_inputs = to_fit_inputs(split.train, freq="h", prediction_length=24)
    val_frame = window_series(split.train, split.val, prediction_length=24, freq="h", step=168)
    result = finetune(
        pipeline, train_inputs, val_inputs=to_fit_inputs(val_frame, freq="h", prediction_length=24),
        mode="lora", prediction_length=24, num_steps=1000, learning_rate=1e-5,
        batch_size=32, context_length=2048, output_dir=paths.models / "lora",
        early_stopping_patience=3,
    )
    result.pipeline           # ready to forecast (e.g. wrapped in Chronos2Forecaster)
    load_finetuned(result.checkpoint_dir)  # the same model, reloaded from disk
"""

from __future__ import annotations

import gc
import logging
import math
import shutil
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple

import pandas as pd
import torch
from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_data_frame
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from c2p.config import SEED, set_seed
from c2p.models import assert_regular_grid, load_chronos2
from c2p.schema import ID_COL, TARGET_COL, TIMESTAMP_COL

__all__ = [
    "CHECKPOINT_NAME",
    "LORA_TARGET_MODULES",
    "FinetuneMode",
    "FinetuneResult",
    "TemporalSplit",
    "finetune",
    "load_finetuned",
    "lora_config",
    "temporal_split",
    "to_fit_inputs",
    "to_univariate_series",
    "window_series",
]

logger = logging.getLogger(__name__)

FinetuneMode = Literal["lora", "full"]
_MODES: tuple[str, ...] = ("lora", "full")

# Folder, inside ``output_dir``, where ``Chronos2Pipeline.fit`` saves the final model.
CHECKPOINT_NAME = "finetuned-ckpt"
# Chronos-2's default LoRA targets: the four projections of both attention layers of each block
# (attention over time and over the series of a group) plus the output layer of the head.
LORA_TARGET_MODULES: tuple[str, ...] = (
    "self_attention.q",
    "self_attention.k",
    "self_attention.v",
    "self_attention.o",
    "output_patch_embedding.output_layer",
)
_ADAPTER_CONFIG = "adapter_config.json"
_MODEL_CONFIG = "config.json"
_BYTES_PER_MB = 2**20


# --------------------------------------------------------------------------------------------
# Data: temporal split, validation windows and conversion to fit inputs
# --------------------------------------------------------------------------------------------


class TemporalSplit(NamedTuple):
    """Rows of each period; the periods never overlap and follow each other in time."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def temporal_split(
    df: pd.DataFrame,
    *,
    val_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None = None,
    timestamp_column: str = TIMESTAMP_COL,
) -> TemporalSplit:
    """Split a series frame by time into train, validation and test rows.

    ``train`` has the rows before ``val_start``, ``val`` those from ``val_start`` to before
    ``test_start`` and ``test`` those from ``test_start`` to before ``test_end`` (to the end
    when ``None``). Boundaries are timezone-naive UTC like the timestamps. The pieces do not
    overlap, so a forecast window's context must be rebuilt from earlier pieces: use
    :func:`window_series` for validation windows and ``c2p.eval.backtest`` on the rows before
    ``test_end`` for the test windows.
    """
    if timestamp_column not in df.columns:
        raise ValueError(f"df has no {timestamp_column!r} column.")
    timestamps = df[timestamp_column]
    if not pd.api.types.is_datetime64_dtype(timestamps):
        raise ValueError(
            f"{timestamp_column!r} must be timezone-naive datetime64, got {timestamps.dtype}."
        )
    val_begin = _naive_timestamp("val_start", val_start)
    test_begin = _naive_timestamp("test_start", test_start)
    end = None if test_end is None else _naive_timestamp("test_end", test_end)
    if not val_begin < test_begin or (end is not None and not test_begin < end):
        raise ValueError(
            "Expected val_start < test_start < test_end, got "
            f"{val_start}, {test_start}, {test_end}."
        )
    in_test = timestamps >= test_begin
    if end is not None:
        in_test &= timestamps < end
    split = TemporalSplit(
        train=df.loc[timestamps < val_begin].reset_index(drop=True),
        val=df.loc[(timestamps >= val_begin) & (timestamps < test_begin)].reset_index(drop=True),
        test=df.loc[in_test].reset_index(drop=True),
    )
    empty = [name for name, part in split._asdict().items() if part.empty]
    if empty:
        raise ValueError(
            f"The {', '.join(empty)} period(s) have no rows: the data covers "
            f"{timestamps.min()} to {timestamps.max()}. Move the split dates inside that range."
        )
    return split


def _naive_timestamp(name: str, value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        raise ValueError(
            f"{name}={value!r} is timezone-aware; pass naive UTC like the series timestamps."
        )
    return timestamp


def to_univariate_series(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    id_column: str = ID_COL,
    timestamp_column: str = TIMESTAMP_COL,
    sep: str = "/",
) -> pd.DataFrame:
    """Turn several columns into separate univariate series in the ``target`` column.

    Each (series, column) pair becomes one series with id ``f"{item_id}{sep}{column}"``, e.g.
    ``"ES/load_actual"``: a quick way to fine-tune one model on many related series.
    """
    columns = list(columns)
    missing = [c for c in (id_column, timestamp_column, *columns) if c not in df.columns]
    if missing:
        raise ValueError(f"df lacks the columns {missing}.")
    if not columns:
        raise ValueError("At least one column is required.")
    long = df[[id_column, timestamp_column, *columns]].melt(
        id_vars=[id_column, timestamp_column], var_name="_column", value_name="_value"
    )
    long[id_column] = long[id_column].astype(str) + sep + long["_column"].astype(str)
    long = long.drop(columns="_column").rename(columns={"_value": TARGET_COL})
    long = long.astype({TARGET_COL: "float64"})
    return long.sort_values([id_column, timestamp_column], kind="stable", ignore_index=True)


def window_series(
    earlier: pd.DataFrame,
    period: pd.DataFrame,
    *,
    prediction_length: int,
    freq: str,
    step: int | None = None,
    max_context: int | None = None,
    id_column: str = ID_COL,
    timestamp_column: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """Cut ``period`` into forecast windows, each returned as a separate series.

    Window ``k`` of a series forecasts the ``prediction_length`` steps of ``period`` that start
    ``k * step`` steps after the period's first timestamp (``step`` defaults to
    ``prediction_length``); windows that would run past the period are dropped. Its context is
    every earlier row of ``earlier`` and ``period`` (only the last ``max_context`` if set), so
    no window ever sees data after its own horizon.

    The result is a series frame with one series per (item, window), id
    ``f"{item_id}@{window_start:%Y-%m-%dT%H:%M}"``, whose last ``prediction_length`` rows are
    the window's horizon: the layout ``Chronos2Pipeline.fit`` expects for
    ``validation_inputs`` (it scores the last ``prediction_length`` steps of each series).
    """
    _check_positive("prediction_length", prediction_length)
    stride = prediction_length if step is None else _check_positive("step", step)
    if max_context is not None:
        _check_positive("max_context", max_context)
    _check_ordered_periods(earlier, period, id_column=id_column, timestamp_column=timestamp_column)
    full = pd.concat([earlier, period], ignore_index=True).sort_values(
        [id_column, timestamp_column], kind="stable", ignore_index=True
    )
    assert_regular_grid(full, freq, id_column=id_column, timestamp_column=timestamp_column)
    period_starts = period.groupby(id_column, sort=False)[timestamp_column].min()
    windows = [
        window
        for item_id, rows in full.groupby(id_column, sort=False)
        if item_id in period_starts.index
        for window in _item_windows(
            rows.reset_index(drop=True),
            first_start=period_starts[item_id],
            prediction_length=prediction_length,
            stride=stride,
            max_context=max_context,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )
    ]
    if not windows:
        raise ValueError(
            f"No window of prediction_length={prediction_length} fits in the period "
            f"({len(period)} rows); use a longer period or a shorter horizon."
        )
    return pd.concat(windows, ignore_index=True)


def _check_ordered_periods(
    earlier: pd.DataFrame, period: pd.DataFrame, *, id_column: str, timestamp_column: str
) -> None:
    """Raise if, for some series, ``period`` does not start after the end of ``earlier``."""
    if period.empty:
        raise ValueError("period has no rows.")
    earlier_end = earlier.groupby(id_column, sort=False)[timestamp_column].max()
    period_start = period.groupby(id_column, sort=False)[timestamp_column].min()
    common = earlier_end.index.intersection(period_start.index)
    overlapping = common[(earlier_end[common] >= period_start[common]).to_numpy()]
    if len(overlapping):
        raise ValueError(
            f"period must start after earlier ends, but they overlap for series "
            f"{list(overlapping[:5])}; build both with temporal_split()."
        )


def _item_windows(
    rows: pd.DataFrame,
    *,
    first_start: pd.Timestamp,
    prediction_length: int,
    stride: int,
    max_context: int | None,
    id_column: str,
    timestamp_column: str,
) -> Iterator[pd.DataFrame]:
    """Windows of one series; ``rows`` is its sorted, regular history (earlier + period)."""
    item_id = rows[id_column].iloc[0]
    first = int(rows[timestamp_column].searchsorted(first_start))
    if first == 0:
        raise ValueError(
            f"Series {item_id!r} has no rows before its period starts at {first_start}: "
            "its first window would have no context."
        )
    for start in range(first, len(rows) - prediction_length + 1, stride):
        context_start = 0 if max_context is None else max(0, start - max_context)
        window = rows.iloc[context_start : start + prediction_length].copy()
        window[id_column] = f"{item_id}@{rows[timestamp_column].iloc[start]:%Y-%m-%dT%H:%M}"
        yield window


def to_fit_inputs(
    df: pd.DataFrame,
    *,
    freq: str,
    prediction_length: int,
    target: str | Sequence[str] = TARGET_COL,
    past_covariates: Sequence[str] = (),
    known_covariates: Sequence[str] = (),
    id_column: str = ID_COL,
    timestamp_column: str = TIMESTAMP_COL,
) -> list[dict[str, Any]]:
    """Convert a series frame into inputs for ``Chronos2Pipeline.fit``.

    Wraps the library's ``chronos.chronos2.preprocess.from_data_frame`` (which encodes
    categorical covariates) after checking what it does not: every series must sit on a
    regular ``freq`` grid (gaps as NaN rows, see ``c2p.data.validate_and_prepare``), otherwise
    the model would silently train on series with missing steps. Only the listed columns are
    used. ``known_covariates`` are covariates whose future values are known at forecast time:
    during training, their values over each sampled horizon are shown to the model, like
    ``future_df`` at prediction time. ``prediction_length`` must match the one given to
    :func:`finetune`.
    """
    targets = [target] if isinstance(target, str) else list(target)
    columns = [*targets, *past_covariates, *known_covariates]
    if not targets:
        raise ValueError("At least one target column is required.")
    if len(set(columns)) != len(columns):
        raise ValueError(f"Targets and covariates must be distinct columns, got {columns}.")
    missing = [c for c in (id_column, timestamp_column, *columns) if c not in df.columns]
    if missing:
        raise ValueError(f"df lacks the columns {missing}.")
    if df.empty:
        raise ValueError("df has no rows.")
    non_numeric = [c for c in targets if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Target columns must be numeric: {non_numeric}.")
    _check_positive("prediction_length", prediction_length)
    assert_regular_grid(df, freq, id_column=id_column, timestamp_column=timestamp_column)
    frame = df[[id_column, timestamp_column, *columns]].astype({id_column: str})
    prepared = from_data_frame(
        frame,
        target_columns=targets,
        prediction_length=prediction_length,
        known_covariates_names=list(known_covariates) or None,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    return [dict(item) for item in prepared]


# --------------------------------------------------------------------------------------------
# Fine-tuning
# --------------------------------------------------------------------------------------------


def lora_config(
    r: int = 8,
    *,
    lora_alpha: float | None = None,
    target_modules: Sequence[str] = LORA_TARGET_MODULES,
    lora_dropout: float = 0.0,
) -> dict[str, Any]:
    """LoRA settings for :func:`finetune`; the defaults reproduce Chronos-2's own.

    ``lora_alpha`` defaults to ``2 * r`` (Chronos-2 uses ``r=8, lora_alpha=16``), which keeps
    the scale of the update, ``lora_alpha / r``, fixed when sweeping ``r``. Always pass the
    target modules: a config without them makes ``peft`` guess from the T5 architecture and
    adapt fewer layers than Chronos-2 does by default.
    """
    _check_positive("r", r)
    alpha = 2 * r if lora_alpha is None else lora_alpha
    if alpha <= 0:
        raise ValueError(f"lora_alpha must be > 0, got {alpha}.")
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError(f"lora_dropout must be in [0, 1), got {lora_dropout}.")
    if not target_modules:
        raise ValueError("target_modules must name at least one module.")
    return {
        "r": r,
        "lora_alpha": alpha,
        "target_modules": list(target_modules),
        "lora_dropout": lora_dropout,
    }


@dataclass(frozen=True)
class FinetuneResult:
    """A fine-tuned pipeline and what it cost.

    Attributes
    ----------
    pipeline
        The saved checkpoint reloaded from disk (``load_finetuned(checkpoint_dir)``): in
        evaluation mode, LoRA weights merged, full float32 precision.
    mode
        ``"lora"`` or ``"full"``.
    train_seconds
        Wall-clock time of ``Chronos2Pipeline.fit`` (training, validation and saving).
    peak_vram_mb
        Peak GPU memory allocated by PyTorch during ``fit``, including the base model that
        stays loaded (``None`` on CPU). ``nvidia-smi`` shows more: add the CUDA context.
    peak_vram_reserved_mb
        Peak memory held by PyTorch's caching allocator (``None`` on CPU).
    artifact_mb
        Size on disk of the saved checkpoint: the LoRA adapter, or the whole model.
    output_dir
        Folder given to ``fit``.
    checkpoint_dir
        ``output_dir / "finetuned-ckpt"``, the folder to pass to :func:`load_finetuned`.
    trainable_parameters, total_parameters
        Parameters updated by training, and all parameters (LoRA weights included).
    num_steps, steps
        Optimisation steps requested, and run (fewer if early stopping triggered).
    best_step, best_eval_loss
        Validation step and loss of the kept model (``None`` without validation).
    log_history
        Training and validation logs of the Hugging Face ``Trainer``, one dict per log event.
    """

    pipeline: Chronos2Pipeline = field(repr=False)
    mode: FinetuneMode
    train_seconds: float
    peak_vram_mb: float | None
    peak_vram_reserved_mb: float | None
    artifact_mb: float
    output_dir: Path
    checkpoint_dir: Path
    trainable_parameters: int
    total_parameters: int
    num_steps: int
    steps: int
    best_step: int | None
    best_eval_loss: float | None
    log_history: tuple[dict[str, float], ...] = field(repr=False)

    @property
    def stopped_early(self) -> bool:
        """Whether training ended before ``num_steps`` because validation stopped improving."""
        return self.steps < self.num_steps

    def losses(self) -> pd.DataFrame:
        """Training and validation loss per logged step, for learning-curve plots.

        Columns: ``step``, ``train_loss``, ``eval_loss``.
        """
        rows: dict[int, dict[str, float]] = {}
        for entry in self.log_history:
            step = int(entry["step"])
            if "loss" in entry:
                rows.setdefault(step, {})["train_loss"] = float(entry["loss"])
            if "eval_loss" in entry:
                rows.setdefault(step, {})["eval_loss"] = float(entry["eval_loss"])
        frame = pd.DataFrame.from_dict(rows, orient="index").rename_axis("step").sort_index()
        return frame.reindex(columns=["train_loss", "eval_loss"]).reset_index()

    def summary(self) -> dict[str, object]:
        """One row of costs for a results table (no pipeline, no logs)."""
        return {
            "mode": self.mode,
            "train_seconds": self.train_seconds,
            "peak_vram_mb": self.peak_vram_mb,
            "peak_vram_reserved_mb": self.peak_vram_reserved_mb,
            "artifact_mb": self.artifact_mb,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "num_steps": self.num_steps,
            "steps": self.steps,
            "best_step": self.best_step,
            "best_eval_loss": self.best_eval_loss,
            "stopped_early": self.stopped_early,
        }


class _TrainingRecorder(TrainerCallback):
    """Keeps what ``Chronos2Pipeline.fit`` does not return: parameter counts and logs."""

    def __init__(self) -> None:
        self.trainable_parameters = 0
        self.total_parameters = 0
        self.steps = 0
        self.log_history: tuple[dict[str, float], ...] = ()

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        if model is not None:
            parameters = list(model.parameters())
            self.total_parameters = sum(p.numel() for p in parameters)
            self.trainable_parameters = sum(p.numel() for p in parameters if p.requires_grad)

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self.steps = int(state.global_step)
        self.log_history = tuple(dict(entry) for entry in state.log_history)


def finetune(
    pipeline: Chronos2Pipeline,
    train_inputs: Sequence[Mapping[str, Any]],
    *,
    val_inputs: Sequence[Mapping[str, Any]] | None,
    mode: FinetuneMode,
    prediction_length: int,
    num_steps: int,
    learning_rate: float,
    batch_size: int,
    context_length: int,
    output_dir: Path,
    gradient_accumulation_steps: int = 1,
    lora_config: Mapping[str, Any] | None = None,
    early_stopping_patience: int | None = None,
    eval_steps: int = 100,
    min_past: int | None = None,
    seed: int = SEED,
    overwrite: bool = False,
    keep_checkpoints: bool = False,
    progress: bool = False,
) -> FinetuneResult:
    """Fine-tune a copy of ``pipeline`` with ``Chronos2Pipeline.fit`` and measure the cost.

    Parameters
    ----------
    pipeline
        The base model; it is not modified.
    train_inputs, val_inputs
        Outputs of :func:`to_fit_inputs`. Training samples random windows of the training
        series; validation scores the last ``prediction_length`` steps of each validation
        series (build them with :func:`window_series`). ``None`` disables validation.
    mode
        ``"lora"`` (low-rank adapters, a few MB on disk) or ``"full"`` (all weights).
    prediction_length, context_length
        Horizon trained for, and maximum context length of the training windows.
    num_steps, learning_rate, batch_size
        Optimiser steps, peak learning rate (linear decay to zero, no warm-up) and series per
        batch. Chronos recommends ~1e-5 for LoRA and lower (~1e-6) for full fine-tuning.
    output_dir
        Where checkpoints go; the final model is saved in ``output_dir / "finetuned-ckpt"``.
    gradient_accumulation_steps
        Batches per optimiser step. GPU memory grows with ``batch_size * context_length``; to
        fit a small GPU, divide ``batch_size`` and multiply this by the same factor.
    lora_config
        LoRA settings (see :func:`lora_config`); ``None`` uses Chronos-2's default
        (``r=8``, ``lora_alpha=16``). Only valid with ``mode="lora"``.
    early_stopping_patience
        Stop after this many validations without improvement (needs ``val_inputs``). The
        model with the best validation loss is kept either way.
    eval_steps
        Validate (and save a checkpoint) every ``eval_steps`` steps, and at the last step.
    min_past
        Shortest context of a training window; Chronos uses ``prediction_length`` by default.
    seed
        Seeds Python, NumPy and PyTorch before training (LoRA initialisation, sampled training
        windows, dropout). Runs on CPU repeat exactly; GPU kernels may still differ slightly.
    overwrite
        Delete ``output_dir`` first if it exists; otherwise a non-empty folder is an error.
    keep_checkpoints
        Keep the intermediate ``checkpoint-*`` folders (deleted by default to save disk).
    progress
        Show the ``Trainer`` progress bar.

    The returned pipeline is the saved checkpoint reloaded from disk (see the module
    docstring). TF32 is enabled for training on Ampere or newer GPUs, as Chronos does, and the
    previous float32 precision is restored afterwards.
    """
    _check_finetune_arguments(
        pipeline=pipeline,
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        mode=mode,
        lora_config=lora_config,
        early_stopping_patience=early_stopping_patience,
        positive_ints={
            "prediction_length": prediction_length,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "context_length": context_length,
            "eval_steps": eval_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
        learning_rate=learning_rate,
    )
    output_dir = _prepare_output_dir(Path(output_dir), overwrite=overwrite)
    recorder = _TrainingRecorder()
    callbacks: list[TrainerCallback] = [recorder]
    validation_kwargs: dict[str, Any] = {}
    if val_inputs is not None:
        validation_kwargs = {"eval_steps": eval_steps, "save_steps": eval_steps}
        if early_stopping_patience is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    device = pipeline.model.device
    logger.info(
        "Fine-tuning (%s) on %s: %d training series, %s validation series, %d steps.",
        mode,
        device,
        len(train_inputs),
        "no" if val_inputs is None else len(val_inputs),
        num_steps,
    )
    set_seed(seed)  # LoRA weights are initialised before the Trainer seeds anything
    _reset_peak_memory(device)
    start = time.perf_counter()
    with _float32_precision_restored(enable_tf32=_supports_tf32(device)):
        finetuned = pipeline.fit(
            inputs=train_inputs,
            prediction_length=prediction_length,
            validation_inputs=val_inputs,
            finetune_mode=mode,
            lora_config=None if lora_config is None else dict(lora_config),
            context_length=context_length,
            learning_rate=learning_rate,
            num_steps=num_steps,
            batch_size=batch_size,
            output_dir=output_dir,
            min_past=min_past,
            finetuned_ckpt_name=CHECKPOINT_NAME,
            callbacks=callbacks,
            remove_printer_callback=True,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=min(eval_steps, num_steps),
            disable_tqdm=not progress,
            **validation_kwargs,
        )
    train_seconds = time.perf_counter() - start
    peak_allocated, peak_reserved = _peak_memory_mb(device)
    del finetuned  # carries training state; the saved checkpoint is reloaded below
    _free_memory(device)
    if not keep_checkpoints:
        _remove_intermediate_checkpoints(output_dir)
    checkpoint_dir = output_dir / CHECKPOINT_NAME
    best_step, best_loss = _best_validation(recorder.log_history)
    result = FinetuneResult(
        pipeline=load_finetuned(checkpoint_dir, device=str(device)),
        mode=mode,
        train_seconds=train_seconds,
        peak_vram_mb=peak_allocated,
        peak_vram_reserved_mb=peak_reserved,
        artifact_mb=_folder_mb(checkpoint_dir),
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        trainable_parameters=recorder.trainable_parameters,
        total_parameters=recorder.total_parameters,
        num_steps=num_steps,
        steps=recorder.steps,
        best_step=best_step,
        best_eval_loss=best_loss,
        log_history=recorder.log_history,
    )
    logger.info("Fine-tuning done: %s", result.summary())
    return result


def _check_finetune_arguments(
    *,
    pipeline: object,
    train_inputs: Sequence[Mapping[str, Any]],
    val_inputs: Sequence[Mapping[str, Any]] | None,
    mode: str,
    lora_config: Mapping[str, Any] | None,
    early_stopping_patience: int | None,
    positive_ints: Mapping[str, int],
    learning_rate: float,
) -> None:
    if not isinstance(pipeline, Chronos2Pipeline):
        raise TypeError(f"Expected a Chronos2Pipeline, got {type(pipeline).__name__}.")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}.")
    if lora_config is not None:
        if mode != "lora":
            raise ValueError("lora_config is only used with mode='lora'.")
        if "target_modules" not in lora_config:
            raise ValueError(
                "lora_config has no 'target_modules'; peft would guess them from the T5 "
                "architecture and adapt fewer layers than Chronos-2's default. "
                "Build it with c2p.finetune.lora_config()."
            )
    if not train_inputs:
        raise ValueError("train_inputs is empty.")
    if val_inputs is not None and not val_inputs:
        raise ValueError("val_inputs is empty; pass None to train without validation.")
    if early_stopping_patience is not None:
        _check_positive("early_stopping_patience", early_stopping_patience)
        if val_inputs is None:
            raise ValueError("early_stopping_patience needs val_inputs to monitor.")
    for name, value in positive_ints.items():
        _check_positive(name, value)
    if not (math.isfinite(learning_rate) and learning_rate > 0):
        raise ValueError(f"learning_rate must be a positive number, got {learning_rate}.")


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} is not empty; pass overwrite=True to replace it or choose "
                "another folder (checkpoints of different runs must not mix)."
            )
        logger.info("Deleting the previous run in %s.", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _supports_tf32(device: torch.device) -> bool:
    """Mirror Chronos: TF32 is used to train on CUDA GPUs of compute capability 8 or more."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


@contextmanager
def _float32_precision_restored(*, enable_tf32: bool) -> Iterator[None]:
    """Train with TF32 if asked, then restore PyTorch's float32 precision settings.

    ``TrainingArguments(tf32=True)`` (set by ``fit`` on Ampere GPUs) switches the global
    ``torch.backends.fp32_precision`` to ``"tf32"``, but only the first time in a process (the
    transformers helper is cached), and Chronos restores only the CUDA matmul and cuDNN
    settings. Without this, every later GPU forecast would run in TF32 (measured: up to 0.3%
    different), and a second fine-tune in the same process would train without TF32.
    """
    # ``fp32_precision`` is newer than the type stubs PyTorch ships, hence getattr/setattr.
    backends = (torch.backends, torch.backends.cuda.matmul, torch.backends.cudnn)
    saved = [getattr(backend, "fp32_precision") for backend in backends]  # noqa: B009
    if enable_tf32:
        setattr(torch.backends, "fp32_precision", "tf32")  # noqa: B010
    try:
        yield
    finally:
        for backend, value in zip(backends, saved, strict=True):
            setattr(backend, "fp32_precision", value)  # noqa: B010


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    torch.cuda.synchronize(device)
    return (
        torch.cuda.max_memory_allocated(device) / _BYTES_PER_MB,
        torch.cuda.max_memory_reserved(device) / _BYTES_PER_MB,
    )


def _free_memory(device: torch.device) -> None:
    """Release the optimiser state and activations left in PyTorch's cache by training."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _remove_intermediate_checkpoints(output_dir: Path) -> None:
    for checkpoint in output_dir.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)


def _best_validation(log_history: Sequence[Mapping[str, float]]) -> tuple[int | None, float | None]:
    """Step and value of the lowest validation loss (the first one on ties, like ``Trainer``)."""
    evaluations = [(float(e["eval_loss"]), int(e["step"])) for e in log_history if "eval_loss" in e]
    if not evaluations:
        return None, None
    best = min(evaluations, key=lambda loss_step: loss_step[0])
    return best[1], best[0]


def _folder_mb(folder: Path) -> float:
    if not folder.is_dir():
        raise FileNotFoundError(f"Expected a saved checkpoint in {folder}, found nothing.")
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) / _BYTES_PER_MB


def load_finetuned(path: Path | str, *, device: str | None = None) -> Chronos2Pipeline:
    """Load a checkpoint saved by :func:`finetune` (``result.checkpoint_dir``).

    A full fine-tune folder holds the whole model. A LoRA folder holds only the adapter: the
    base model named in its ``adapter_config.json`` (``amazon/chronos-2``) is loaded from the
    Hugging Face cache or Hub and the adapter is merged into it. To ship a self-contained model
    instead (e.g. in a container), save the loaded pipeline with ``save_pretrained(folder)``.
    ``device`` defaults to ``c2p.config.get_device()``.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a folder; pass result.checkpoint_dir.")
    if not ((path / _ADAPTER_CONFIG).is_file() or (path / _MODEL_CONFIG).is_file()):
        raise FileNotFoundError(
            f"{path} has neither {_ADAPTER_CONFIG} (LoRA) nor {_MODEL_CONFIG} (full model); "
            f"pass the {CHECKPOINT_NAME!r} folder inside the fine-tuning output_dir."
        )
    pipeline = load_chronos2(str(path), device=device)
    pipeline.model.eval()
    return pipeline


def _check_positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value
