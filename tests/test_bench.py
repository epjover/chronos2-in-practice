"""Tests of ``bench/bench_cpu_gpu.py`` with a stand-in pipeline (no model is downloaded)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from c2p.schema import ID_COL, TIMESTAMP_COL

SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "bench_cpu_gpu.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_cpu_gpu", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_script()


def test_summarise_times_reports_location_spread_and_tails() -> None:
    times = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 1.00]  # one outlier
    stats = bench.summarise_times(times)
    assert stats["repeats"] == 10
    assert stats["seconds_min"] == 0.10 and stats["seconds_max"] == 1.00
    assert stats["seconds_median"] == pytest.approx(0.145)
    assert stats["seconds_mean"] == pytest.approx(np.mean(times))
    assert stats["seconds_std"] == pytest.approx(np.std(times, ddof=1))
    assert stats["seconds_p05"] < stats["seconds_median"] < stats["seconds_p95"]
    assert stats["seconds_p95"] < stats["seconds_max"]  # the outlier sits beyond p95
    assert stats["cv"] == pytest.approx(stats["seconds_std"] / stats["seconds_mean"])


def test_summarise_times_single_sample_has_zero_spread() -> None:
    stats = bench.summarise_times([0.5])
    assert stats["seconds_std"] == 0.0 and stats["cv"] == 0.0
    assert stats["seconds_median"] == stats["seconds_p05"] == stats["seconds_p95"] == 0.5


def test_summarise_times_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        bench.summarise_times([])


def test_time_predict_honours_repeats_floor_and_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowPipeline:
        calls = 0

        def predict_df(self, *args: object, **kwargs: object) -> None:
            self.calls += 1

    clock = iter(np.arange(0, 1000, 0.5))  # every timed call appears to take 0.5 s
    monkeypatch.setattr(bench.time, "perf_counter", lambda: float(next(clock)))
    frame = pd.DataFrame({ID_COL: ["a"], TIMESTAMP_COL: [pd.Timestamp("2024-01-01")], "y": [1.0]})

    pipeline = SlowPipeline()
    times = bench.time_predict(pipeline, frame, horizon=1, repeats=30, budget_s=2.0)
    # budget of 2 s would stop after 4 calls, but the floor is MIN_REPEATS
    assert len(times) == bench.MIN_REPEATS
    assert pipeline.calls == bench.MIN_REPEATS + 1  # plus the discarded warm-up
    assert all(t == pytest.approx(0.5) for t in times)

    pipeline = SlowPipeline()
    times = bench.time_predict(pipeline, frame, horizon=1, repeats=4, budget_s=1e9)
    assert len(times) == 4  # repeats smaller than the floor wins


def test_tile_series_copies_the_tail_with_distinct_ids() -> None:
    series = pd.DataFrame(
        {
            ID_COL: "ES",
            TIMESTAMP_COL: pd.date_range("2024-01-01", periods=10, freq="h"),
            bench.LOAD: np.arange(10.0),
        }
    )
    tiled = bench.tile_series(series, n_series=3, context_length=4)
    assert len(tiled) == 12
    assert tiled[ID_COL].nunique() == 3
    assert tiled[bench.LOAD].tolist()[:4] == [6.0, 7.0, 8.0, 9.0]
    with pytest.raises(ValueError, match="fewer than the requested context"):
        bench.tile_series(series, n_series=1, context_length=11)


class InstantPipeline:
    """Stands in for a Chronos-2 pipeline: accepts the call and returns at once."""

    def predict_df(self, *args: object, **kwargs: object) -> None:
        return None


def hourly_load(hours: int = 600) -> pd.DataFrame:
    stamps = pd.date_range("2018-01-01", periods=hours, freq="h")
    values = 100.0 + np.sin(np.arange(hours) / 24.0)
    return pd.DataFrame({ID_COL: "spain", TIMESTAMP_COL: stamps, bench.LOAD: values})


def test_main_writes_one_row_per_cell_and_one_per_timed_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("C2P_ROOT", str(tmp_path))
    monkeypatch.setattr(bench, "spain_energy_frame", hourly_load)
    monkeypatch.setattr(bench, "load_chronos2", lambda *args, **kwargs: InstantPipeline())
    monkeypatch.setattr(bench, "parameter_count", lambda pipeline: 119_500_000)
    assert bench.main(["--fast", "--models", "stand-in", "--threads", "1"]) == 0

    cells = pd.read_csv(tmp_path / "results" / "bench_latency.csv")
    samples = pd.read_csv(tmp_path / "results" / "bench_latency_samples.csv")
    assert len(cells) == len(bench.FAST_BATCHES) * len(bench.FAST_CONTEXTS)
    assert len(samples) == 3 * len(cells)  # --fast times three calls per cell
    assert cells["model"].eq("stand-in").all() and cells["device"].eq("cpu").all()
    assert cells["params_m"].eq(119.5).all()
    assert cells["peak_vram_mb"].isna().all()  # no GPU memory to report on a CPU
    assert np.allclose(cells["series_per_second"], cells["batch_size"] / cells["seconds_median"])
    # the slow tail of the calls is the low end of the throughput, and the other way round
    assert (cells["series_per_second_p05"] <= cells["series_per_second"]).all()
    assert (cells["series_per_second"] <= cells["series_per_second_p95"]).all()
    assert "Saved" in capsys.readouterr().out


def test_main_refuses_a_gpu_run_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        bench.main(["--device", "cuda"])
