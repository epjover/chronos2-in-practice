"""Tests for ``c2p.finetune``.

Fast tests use synthetic hourly series and, for the fine-tuning machinery, a tiny randomly
initialised Chronos-2 saved to a temporary folder (no download, a few seconds on CPU). The slow
test fine-tunes the real ``amazon/chronos-2`` for a few steps on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from chronos import Chronos2Model, Chronos2Pipeline
from chronos.chronos2 import Chronos2CoreConfig
from safetensors.torch import load_file

from c2p.finetune import (
    CHECKPOINT_NAME,
    LORA_TARGET_MODULES,
    FinetuneResult,
    _best_validation,
    _float32_precision_restored,
    finetune,
    load_finetuned,
    lora_config,
    temporal_split,
    to_fit_inputs,
    to_univariate_series,
    window_series,
)
from c2p.models import CHRONOS2, Chronos2Forecaster, load_chronos2
from c2p.schema import CHRONOS2_QUANTILES, ID_COL, TARGET_COL, TIMESTAMP_COL

pytestmark = pytest.mark.filterwarnings(
    "ignore:The model is being fine-tuned on the CPU:UserWarning"
)

FREQ = "h"
HORIZON = 24
LEVELS = [0.1, 0.5, 0.9]


# --------------------------------------------------------------------------------------------
# Helpers and fixtures
# --------------------------------------------------------------------------------------------


def make_series(
    *,
    items: tuple[str, ...] = ("A", "B"),
    periods: int = 24 * 30,
    start: str = "2020-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    """Hourly daily-sine series with noise and a couple of NaNs, plus two covariates."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=periods, freq=FREQ)
    hours = np.arange(periods)
    frames = []
    for scale, item in enumerate(items, start=1):
        target = 10 + scale * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.2, periods)
        target[[5, periods // 2]] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    ID_COL: item,
                    TIMESTAMP_COL: timestamps,
                    TARGET_COL: target,
                    "past": rng.normal(size=periods),
                    "known": (hours % 24).astype(float),
                    "unused": 1.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def tiny_pipeline(tmp_path_factory: pytest.TempPathFactory) -> Chronos2Pipeline:
    """A tiny random Chronos-2 saved to disk, so LoRA adapters can name it as their base."""
    torch.manual_seed(0)
    config = Chronos2CoreConfig(
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=1,
        num_heads=2,
        chronos_config={
            "context_length": 512,
            "input_patch_size": 16,
            "input_patch_stride": 16,
            "output_patch_size": 16,
            "quantiles": list(CHRONOS2_QUANTILES),
            "use_reg_token": True,
            "use_arcsinh": True,
            "max_output_patches": 4,
        },
        architectures=["Chronos2Model"],
        chronos_pipeline_class="Chronos2Pipeline",
    )
    folder = tmp_path_factory.mktemp("tiny_chronos2")
    Chronos2Model(config).save_pretrained(folder)
    return Chronos2Pipeline.from_pretrained(str(folder), device_map="cpu")


@pytest.fixture(scope="module")
def tiny_data() -> dict[str, object]:
    """Train inputs, validation inputs and a forecast context from the synthetic series."""
    df = make_series(periods=24 * 40)
    split = temporal_split(df, val_start="2020-01-31", test_start="2020-02-05")
    windows = window_series(
        split.train, split.val, prediction_length=HORIZON, freq=FREQ, step=24, max_context=128
    )
    return {
        "train": to_fit_inputs(split.train, freq=FREQ, prediction_length=HORIZON),
        "val": to_fit_inputs(windows, freq=FREQ, prediction_length=HORIZON),
        "context": pd.concat([split.train, split.val], ignore_index=True)[
            [ID_COL, TIMESTAMP_COL, TARGET_COL]
        ],
    }


def run_tiny(
    pipeline: Chronos2Pipeline, data: dict[str, object], output_dir: Path, **overrides: object
) -> FinetuneResult:
    kwargs: dict[str, object] = {
        "val_inputs": data["val"],
        "mode": "lora",
        "prediction_length": HORIZON,
        "num_steps": 4,
        "learning_rate": 1e-3,
        "batch_size": 4,
        "context_length": 128,
        "output_dir": output_dir,
        "eval_steps": 2,
    }
    kwargs.update(overrides)
    return finetune(pipeline, data["train"], **kwargs)


def forecast(pipeline: Chronos2Pipeline, context: pd.DataFrame) -> np.ndarray:
    frame = Chronos2Forecaster(pipeline).predict(
        context, prediction_length=HORIZON, quantile_levels=LEVELS, freq=FREQ
    )
    return frame[[str(level) for level in LEVELS]].to_numpy()


def weights(pipeline: Chronos2Pipeline) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in pipeline.model.state_dict().items()}


# --------------------------------------------------------------------------------------------
# temporal_split
# --------------------------------------------------------------------------------------------


def test_temporal_split_periods_are_disjoint_and_ordered() -> None:
    df = make_series()
    original = df.copy()
    split = temporal_split(df, val_start="2020-01-20", test_start="2020-01-25")

    assert split.train[TIMESTAMP_COL].max() < pd.Timestamp("2020-01-20")
    assert split.val[TIMESTAMP_COL].min() == pd.Timestamp("2020-01-20")
    assert split.val[TIMESTAMP_COL].max() < pd.Timestamp("2020-01-25")
    assert split.test[TIMESTAMP_COL].min() == pd.Timestamp("2020-01-25")
    assert len(split.train) + len(split.val) + len(split.test) == len(df)
    assert split.train is split[0]  # also a plain (train, val, test) tuple
    pd.testing.assert_frame_equal(df, original)


def test_temporal_split_test_end_is_exclusive() -> None:
    split = temporal_split(
        make_series(), val_start="2020-01-20", test_start="2020-01-25", test_end="2020-01-26"
    )
    assert split.test[TIMESTAMP_COL].max() == pd.Timestamp("2020-01-25 23:00")
    assert len(split.test) == 2 * 24


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ({"val_start": "2020-01-25", "test_start": "2020-01-20"}, "val_start < test_start"),
        (
            {"val_start": "2020-01-20", "test_start": "2020-01-25", "test_end": "2020-01-25"},
            "test_start < test_end",
        ),
        ({"val_start": "2020-01-20", "test_start": "2021-01-01"}, "test period"),
        ({"val_start": "2020-01-20 00:00+01:00", "test_start": "2020-01-25"}, "timezone-aware"),
    ],
)
def test_temporal_split_rejects_bad_bounds(bounds: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        temporal_split(make_series(), **bounds)


def test_temporal_split_rejects_timezone_aware_timestamps() -> None:
    df = make_series()
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-naive"):
        temporal_split(df, val_start="2020-01-20", test_start="2020-01-25")


# --------------------------------------------------------------------------------------------
# to_univariate_series
# --------------------------------------------------------------------------------------------


def test_to_univariate_series_stacks_columns_as_series() -> None:
    df = make_series(items=("ES",), periods=48)
    long = to_univariate_series(df, ["past", "known"])

    assert list(long.columns) == [ID_COL, TIMESTAMP_COL, TARGET_COL]
    assert long[ID_COL].unique().tolist() == ["ES/known", "ES/past"]
    known = long.loc[long[ID_COL] == "ES/known", TARGET_COL].to_numpy()
    np.testing.assert_array_equal(known, df["known"].to_numpy())
    assert long[TARGET_COL].dtype == "float64"


def test_to_univariate_series_requires_columns() -> None:
    with pytest.raises(ValueError, match="lacks the columns"):
        to_univariate_series(make_series(periods=48), ["missing"])
    with pytest.raises(ValueError, match="At least one column"):
        to_univariate_series(make_series(periods=48), [])


# --------------------------------------------------------------------------------------------
# window_series
# --------------------------------------------------------------------------------------------


def split_for_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = make_series(periods=160)  # 100 earlier rows + 60 period rows per series
    cutoff = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=100)
    return (
        df.loc[df[TIMESTAMP_COL] < cutoff].reset_index(drop=True),
        df.loc[df[TIMESTAMP_COL] >= cutoff].reset_index(drop=True),
    )


def test_window_series_contexts_end_where_horizons_start() -> None:
    earlier, period = split_for_windows()
    windows = window_series(earlier, period, prediction_length=HORIZON, freq=FREQ, step=12)

    assert windows[ID_COL].nunique() == 2 * 4  # starts 0, 12, 24, 36 of the 60-row period
    period_a = period.loc[period[ID_COL] == "A"].reset_index(drop=True)
    for k, (window_id, window) in enumerate(
        (i, w) for i, w in windows.groupby(ID_COL, sort=False) if i.startswith("A@")
    ):
        horizon = window.tail(HORIZON)
        expected = period_a.iloc[k * 12 : k * 12 + HORIZON]
        np.testing.assert_array_equal(horizon[TIMESTAMP_COL], expected[TIMESTAMP_COL])
        np.testing.assert_array_equal(horizon[TARGET_COL], expected[TARGET_COL])
        assert window_id == f"A@{expected[TIMESTAMP_COL].iloc[0]:%Y-%m-%dT%H:%M}"
        assert len(window) == 100 + k * 12 + HORIZON  # every earlier row, nothing later
        assert window[TIMESTAMP_COL].is_monotonic_increasing


def test_window_series_max_context_and_default_step() -> None:
    earlier, period = split_for_windows()
    windows = window_series(earlier, period, prediction_length=HORIZON, freq=FREQ, max_context=50)

    sizes = windows.groupby(ID_COL).size()
    assert (sizes == 50 + HORIZON).all()
    assert len(sizes) == 2 * 2  # step defaults to the horizon: starts 0 and 24


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("overlap", "overlap"),
        ("gap", "regular"),
        ("no_context", "no rows before"),
        ("short_period", "No window"),
    ],
)
def test_window_series_rejects_bad_inputs(case: str, message: str) -> None:
    earlier, period = split_for_windows()
    if case == "overlap":
        earlier = pd.concat([earlier, period.head(1)], ignore_index=True)
    elif case == "gap":
        earlier = earlier.drop(index=10)
    elif case == "no_context":
        earlier = earlier.loc[earlier[ID_COL] == "A"]
    elif case == "short_period":
        period = period.groupby(ID_COL).head(HORIZON - 1)
    with pytest.raises(ValueError, match=message):
        window_series(earlier, period, prediction_length=HORIZON, freq=FREQ)


# --------------------------------------------------------------------------------------------
# to_fit_inputs
# --------------------------------------------------------------------------------------------


def test_to_fit_inputs_orders_target_past_then_known_covariates() -> None:
    df = make_series(periods=100)
    inputs = to_fit_inputs(
        df,
        freq=FREQ,
        prediction_length=HORIZON,
        past_covariates=["past"],
        known_covariates=["known"],
    )

    assert len(inputs) == 2
    first = inputs[0]
    assert first["context"].shape == (3, 100)  # target, past-only, known; "unused" is dropped
    assert (first["n_targets"], first["n_covariates"], first["n_future_covariates"]) == (1, 2, 1)
    rows_a = df.loc[df[ID_COL] == "A"]
    np.testing.assert_array_equal(first["context"][0].numpy(), rows_a[TARGET_COL].astype("float32"))
    np.testing.assert_array_equal(first["context"][1].numpy(), rows_a["past"].astype("float32"))
    np.testing.assert_array_equal(first["context"][2].numpy(), rows_a["known"].astype("float32"))
    assert torch.isnan(first["context"][0]).sum() == 2  # NaN kept: the model masks them


def test_to_fit_inputs_multiple_targets() -> None:
    inputs = to_fit_inputs(
        make_series(periods=100), freq=FREQ, prediction_length=HORIZON, target=[TARGET_COL, "past"]
    )
    assert inputs[0]["n_targets"] == 2
    assert inputs[0]["n_covariates"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": "missing"}, "lacks the columns"),
        ({"past_covariates": [TARGET_COL]}, "distinct"),
        ({"target": []}, "At least one target"),
        ({"prediction_length": 0}, "positive integer"),
    ],
)
def test_to_fit_inputs_rejects_bad_arguments(kwargs: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {"freq": FREQ, "prediction_length": HORIZON} | kwargs
    with pytest.raises(ValueError, match=message):
        to_fit_inputs(make_series(periods=100), **arguments)


def test_to_fit_inputs_rejects_gaps_and_text_targets() -> None:
    df = make_series(periods=100)
    with pytest.raises(ValueError, match="regular"):
        to_fit_inputs(df.drop(index=[20, 21]), freq=FREQ, prediction_length=HORIZON)
    with pytest.raises(ValueError, match="numeric"):
        to_fit_inputs(df.assign(target="x"), freq=FREQ, prediction_length=HORIZON)


# --------------------------------------------------------------------------------------------
# lora_config and argument checks
# --------------------------------------------------------------------------------------------


def test_lora_config_defaults_match_chronos() -> None:
    config = lora_config()
    assert config["r"] == 8
    assert config["lora_alpha"] == 16
    assert config["target_modules"] == list(LORA_TARGET_MODULES)
    assert lora_config(4)["lora_alpha"] == 8  # alpha / r stays 2 when sweeping r
    assert lora_config(4, lora_alpha=4)["lora_alpha"] == 4


@pytest.mark.parametrize(
    "kwargs", [{"r": 0}, {"lora_alpha": 0}, {"lora_dropout": 1.0}, {"target_modules": ()}]
)
def test_lora_config_rejects_bad_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        lora_config(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"mode": "qlora"}, ValueError, "mode must be"),
        ({"mode": "full", "lora_config": lora_config()}, ValueError, "only used with"),
        ({"lora_config": {"r": 4}}, ValueError, "target_modules"),
        ({"val_inputs": None, "early_stopping_patience": 2}, ValueError, "needs val_inputs"),
        ({"val_inputs": []}, ValueError, "val_inputs is empty"),
        ({"num_steps": 0}, ValueError, "num_steps"),
        ({"learning_rate": -1.0}, ValueError, "learning_rate"),
    ],
)
def test_finetune_rejects_bad_arguments(
    tiny_pipeline: Chronos2Pipeline,
    tiny_data: dict[str, object],
    tmp_path: Path,
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        run_tiny(tiny_pipeline, tiny_data, tmp_path / "run", **overrides)
    assert not (tmp_path / "run").exists()  # nothing is written before the checks pass


def test_finetune_rejects_other_pipelines(tiny_data: dict[str, object], tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Chronos2Pipeline"):
        finetune(
            object(),  # type: ignore[arg-type]
            tiny_data["train"],
            val_inputs=None,
            mode="lora",
            prediction_length=HORIZON,
            num_steps=1,
            learning_rate=1e-3,
            batch_size=4,
            context_length=64,
            output_dir=tmp_path,
        )


def test_finetune_refuses_to_mix_runs(
    tiny_pipeline: Chronos2Pipeline, tiny_data: dict[str, object], tmp_path: Path
) -> None:
    (tmp_path / "old.txt").write_text("previous run")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        run_tiny(tiny_pipeline, tiny_data, tmp_path)


# --------------------------------------------------------------------------------------------
# Fine-tuning machinery (tiny model on CPU)
# --------------------------------------------------------------------------------------------


def test_lora_finetune_with_early_stopping_round_trips(
    tiny_pipeline: Chronos2Pipeline, tiny_data: dict[str, object], tmp_path: Path
) -> None:
    before = weights(tiny_pipeline)
    result = run_tiny(
        tiny_pipeline,
        tiny_data,
        tmp_path / "lora",
        lora_config=lora_config(4),
        num_steps=6,
        eval_steps=1,
        early_stopping_patience=1,
        learning_rate=0.5,  # large enough to overshoot: validation loss rises at step 3
        keep_checkpoints=True,
    )

    assert result.stopped_early
    assert result.best_step is not None
    assert result.best_step < result.steps
    adapter = "adapter_model.safetensors"
    assert (result.checkpoint_dir / "adapter_config.json").is_file()
    saved = load_file(result.checkpoint_dir / adapter)
    best = load_file(result.output_dir / f"checkpoint-{result.best_step}" / adapter)
    assert saved.keys() == best.keys()
    for name, value in saved.items():  # the best checkpoint is kept, not the last one
        torch.testing.assert_close(value, best[name], rtol=0, atol=0)
    assert 0 < result.trainable_parameters < result.total_parameters
    assert result.peak_vram_mb is None  # CPU
    assert 0 < result.artifact_mb < 1
    losses = result.losses()
    assert losses["eval_loss"].notna().sum() == result.steps
    assert result.best_eval_loss == losses["eval_loss"].min()
    assert not result.pipeline.model.training
    expected = forecast(result.pipeline, tiny_data["context"])
    reloaded = load_finetuned(result.checkpoint_dir, device="cpu")
    np.testing.assert_array_equal(forecast(reloaded, tiny_data["context"]), expected)
    reloaded.save_pretrained(tmp_path / "merged")  # a standalone model, no base model needed
    assert (tmp_path / "merged" / "model.safetensors").is_file()
    merged = load_finetuned(tmp_path / "merged", device="cpu")
    # Same weights (checked bit for bit), but CPU kernels may round the last bit differently.
    np.testing.assert_allclose(forecast(merged, tiny_data["context"]), expected, rtol=1e-6)
    for name, value in weights(tiny_pipeline).items():  # the base model is left untouched
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_full_finetune_without_validation_forecasts_deterministically(
    tiny_pipeline: Chronos2Pipeline, tiny_data: dict[str, object], tmp_path: Path
) -> None:
    result = run_tiny(tiny_pipeline, tiny_data, tmp_path / "full", mode="full", val_inputs=None)

    assert (result.checkpoint_dir / "model.safetensors").is_file()
    assert result.trainable_parameters == result.total_parameters
    assert result.best_step is None
    assert result.best_eval_loss is None
    assert not result.stopped_early
    # Chronos' fit leaves the model in training mode without validation: dropout would make
    # these two forecasts differ.
    first = forecast(result.pipeline, tiny_data["context"])
    np.testing.assert_array_equal(first, forecast(result.pipeline, tiny_data["context"]))
    with open(result.checkpoint_dir / "config.json", encoding="utf-8") as file:
        assert json.load(file)["architectures"] == ["Chronos2Model"]


def test_finetune_is_reproducible_and_cleans_up(
    tiny_pipeline: Chronos2Pipeline, tiny_data: dict[str, object], tmp_path: Path
) -> None:
    first = run_tiny(tiny_pipeline, tiny_data, tmp_path / "a")
    second = run_tiny(tiny_pipeline, tiny_data, tmp_path / "a", overwrite=True)

    assert not list(second.output_dir.glob("checkpoint-*"))  # intermediate checkpoints removed
    pd.testing.assert_frame_equal(first.losses(), second.losses())  # same seed, same run
    np.testing.assert_array_equal(
        forecast(first.pipeline, tiny_data["context"]),
        forecast(second.pipeline, tiny_data["context"]),
    )


def test_float32_precision_is_restored_even_on_errors() -> None:
    saved = torch.backends.fp32_precision
    with pytest.raises(RuntimeError), _float32_precision_restored(enable_tf32=True):
        assert torch.backends.fp32_precision == "tf32"
        raise RuntimeError("training failed")
    assert torch.backends.fp32_precision == saved


def test_best_validation_takes_the_first_lowest_loss() -> None:
    history = [
        {"loss": 3.0, "step": 1},
        {"eval_loss": 2.0, "step": 1},
        {"eval_loss": 1.5, "step": 2},
        {"eval_loss": 1.5, "step": 3},
    ]
    assert _best_validation(history) == (2, 1.5)
    assert _best_validation([{"loss": 1.0, "step": 1}]) == (None, None)


def test_load_finetuned_rejects_folders_without_a_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a folder"):
        load_finetuned(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match=CHECKPOINT_NAME):
        load_finetuned(tmp_path)


# --------------------------------------------------------------------------------------------
# Slow: the real model
# --------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_lora_finetune_of_chronos2_on_cpu_round_trips(tmp_path: Path) -> None:
    pipeline = load_chronos2(CHRONOS2, device="cpu")
    df = make_series(periods=24 * 25)
    split = temporal_split(df, val_start="2020-01-21", test_start="2020-01-24")
    windows = window_series(split.train, split.val, prediction_length=HORIZON, freq=FREQ)
    context = pd.concat([split.train, split.val], ignore_index=True)[
        [ID_COL, TIMESTAMP_COL, TARGET_COL]
    ]

    result = finetune(
        pipeline,
        to_fit_inputs(split.train, freq=FREQ, prediction_length=HORIZON),
        val_inputs=to_fit_inputs(windows, freq=FREQ, prediction_length=HORIZON),
        mode="lora",
        prediction_length=HORIZON,
        num_steps=5,
        learning_rate=1e-4,
        batch_size=4,
        context_length=128,
        output_dir=tmp_path / "lora",
        eval_steps=5,
    )

    adapter = json.loads((result.checkpoint_dir / "adapter_config.json").read_text("utf-8"))
    assert adapter["base_model_name_or_path"] == CHRONOS2
    assert adapter["r"] == 8
    assert result.steps == 5
    assert result.artifact_mb < 10
    reloaded = load_finetuned(result.checkpoint_dir, device="cpu")
    np.testing.assert_array_equal(forecast(result.pipeline, context), forecast(reloaded, context))
    assert not np.array_equal(forecast(pipeline, context), forecast(reloaded, context))
