"""Load a published fev-bench task and restate it in this repository's own protocol.

Article 3 replicates the EPF-DE task of fev-bench (German day-ahead electricity price with the
grid operator's load and renewable forecasts as known covariates) before applying the same
design to Spanish prices. Replicating it means using the *published* task definition rather than
a similar-looking one of our own, so this module does two things:

- :func:`load_fev_task` fetches the task from the fev-bench task list.
- :func:`task_series_frame` and :func:`task_backtest_config` translate it into the series frame
  and :class:`~c2p.eval.BacktestConfig` that the rest of the repository already speaks, so the
  replication is scored by the same harness, with the same metrics, as every other experiment.

:func:`assert_cutoffs_match` is the check that makes the translation trustworthy: it compares
the cutoffs our harness derives with the ones fev computes, and fails if they differ by a single
step. Without it, "we replicated EPF-DE" would be an assertion rather than a test.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from c2p.eval import BacktestConfig, rolling_origin_windows
from c2p.schema import EVAL_QUANTILES, ID_COL, TIMESTAMP_COL

if TYPE_CHECKING:
    import fev

logger = logging.getLogger(__name__)

# The task list of fev-bench, the benchmark the Chronos-2 paper reports in its Table 3.
FEV_BENCH_TASKS = (
    "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml"
)
EPF_DE = "epf_de"


def load_fev_task(dataset_config: str = EPF_DE, *, tasks_url: str = FEV_BENCH_TASKS) -> fev.Task:
    """Return the single fev-bench task whose dataset config is ``dataset_config``.

    The dataset is loaded eagerly, because a task cannot report its frequency or cutoffs before
    its data is there.
    """
    from fev import Benchmark

    benchmark = Benchmark.from_yaml(tasks_url)
    matches = [task for task in benchmark.tasks if task.dataset_config == dataset_config]
    if len(matches) != 1:
        available = sorted({task.dataset_config for task in benchmark.tasks})
        raise ValueError(
            f"Expected exactly one task with dataset_config={dataset_config!r}, found "
            f"{len(matches)}. Available configs: {available}."
        )
    task = matches[0]
    task.load_full_dataset()
    logger.info(
        "Loaded fev task %s: horizon %s, %s windows, known covariates %s",
        task.task_name,
        task.horizon,
        task.num_windows,
        task.known_dynamic_columns,
    )
    return task


def task_series_frame(task: fev.Task, static_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Flatten a fev task's dataset into a series frame (see :mod:`c2p.schema`).

    One row per (series, timestamp), with the task's target and every dynamic column it
    declares. Numeric columns become ``float64``; a dynamic column of strings (a holiday type,
    a promotion code) becomes a ``category``, which is how Chronos-2 is told to treat it as
    categorical.

    Static attributes (one value per series: a store type, a region) are dropped unless they
    are named in ``static_columns``. Chronos-2 has no input for them: it takes covariates as
    series, so the only way in is a column that repeats the value at every timestamp, which is
    what ``static_columns`` builds. Whether such a column can tell the model anything is a
    question to measure, not to assume (notebook 10 does).
    """
    dataset = task.load_full_dataset()
    static_columns = list(static_columns)
    columns = [task.target, *task.known_dynamic_columns, *task.past_dynamic_columns]
    frames = []
    for record in dataset:
        timestamps = pd.DatetimeIndex(record[task.timestamp_column]).tz_localize(None)
        values: dict[str, object] = {column: _dynamic_values(record[column]) for column in columns}
        if missing := [column for column in static_columns if column not in record]:
            raise ValueError(f"The task's dataset has no static columns {missing}.")
        values.update({column: _static_value(record[column]) for column in static_columns})
        frames.append(
            pd.DataFrame({ID_COL: str(record[task.id_column]), TIMESTAMP_COL: timestamps, **values})
        )
    frame = pd.concat(frames, ignore_index=True)
    columns = [*columns, *static_columns]
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype("category")
    return frame.sort_values([ID_COL, TIMESTAMP_COL], kind="stable", ignore_index=True)


def _static_value(value: object) -> float | str:
    """One static attribute as a float, or as text when it is not a number (None included)."""
    if isinstance(value, bool | int | float | np.number):
        return float(value)
    return "missing" if value is None else str(value)


def _dynamic_values(values: object) -> np.ndarray:
    """A dynamic column of one record as float64, or as strings when it is not numeric."""
    array = np.asarray(values)
    if array.dtype.kind in "biuf":
        return array.astype("float64")
    return array.astype(object)


def task_backtest_config(
    task: fev.Task, *, with_covariates: bool = True, n_windows: int | None = None
) -> BacktestConfig:
    """Restate a fev task as a :class:`~c2p.eval.BacktestConfig`.

    The horizon, the number of windows, the step between cutoffs and the seasonality come from
    the published task; ``quantile_levels`` stays this repository's own set, which contains the
    nine deciles fev scores plus the tails used for coverage.

    Parameters
    ----------
    with_covariates
        Declare the task's known dynamic columns as known covariates. ``False`` gives the same
        task without them, which is the ablation article 3 is built on.
    n_windows
        Use fewer windows than the task defines (for a fast smoke run). The cutoffs then no
        longer match the published ones, so :func:`assert_cutoffs_match` will refuse them.
    """
    known = tuple(task.known_dynamic_columns) if with_covariates else ()
    return BacktestConfig(
        prediction_length=int(task.horizon),
        n_windows=int(n_windows if n_windows is not None else task.num_windows),
        freq=task.freq,
        season_length=int(task.seasonality),
        step=int(task.window_step_size),
        quantile_levels=EVAL_QUANTILES,
        target=str(task.target),
        known_covariates=known,
    )


def assert_cutoffs_match(task: fev.Task, frame: pd.DataFrame, config: BacktestConfig) -> None:
    """Raise ``AssertionError`` unless our windows sit on the task's published cutoffs.

    fev states a cutoff as the last timestamp the model may see; our harness derives the same
    thing from the end of the series, the horizon and the step. They must agree exactly, or the
    replication is scoring a different problem from the benchmark.
    """
    ours = [pd.Timestamp(window.cutoff) for window in rolling_origin_windows(frame, config)]
    theirs = _task_cutoffs(task, frame)
    if len(ours) != len(theirs) or any(a != b for a, b in zip(ours, theirs, strict=True)):
        raise AssertionError(
            f"Cutoffs do not match the published task {task.task_name!r}: ours run "
            f"{ours[0]} to {ours[-1]} ({len(ours)} windows), fev's {theirs[0]} to {theirs[-1]} "
            f"({len(theirs)} windows)."
        )
    logger.info("Cutoffs match fev task %s on all %d windows.", task.task_name, len(ours))


def _task_cutoffs(task: fev.Task, frame: pd.DataFrame) -> list[pd.Timestamp]:
    """The task's cutoffs as timestamps, whether fev states them as offsets or dates."""
    offset = pd.tseries.frequencies.to_offset(task.freq)
    end = pd.Timestamp(frame[TIMESTAMP_COL].max())
    cutoffs = []
    for cutoff in task.cutoffs:
        if isinstance(cutoff, int | np.integer):
            # fev slices the series as ``past = values[:cutoff]``, so a negative cutoff counts
            # steps back from the end and the last observed step is ``end + cutoff * offset``.
            cutoffs.append(end + int(cutoff) * offset)
        else:
            cutoffs.append(pd.Timestamp(cutoff).tz_localize(None))
    return cutoffs
