"""Accuracy of the two cheaper ways to run Chronos-2, on the series' day-ahead protocol.

``bench_cpu_gpu.py`` measures how much faster the 28M-parameter checkpoint is, and how much
faster the full model runs in half precision. Speed is only half of a trade, so this script
measures the other half with the protocol of article 2 (365 day-ahead windows over 2018,
24-hour horizon, 2,048 hours of context, 11 quantiles) and reports the skill of each cheaper
variant against the full model in single precision, with a block-bootstrap interval over
windows:

- ``chronos-2-bf16``: the same weights loaded as ``bfloat16`` (GPU only);
- ``chronos-2-small``: the 28M-parameter checkpoint.

The numbers belong side by side: a variant that is 2-4x faster and 1% worse is a different
decision from one that is 2-4x faster and 15% worse.

Usage::

    uv run python bench/cheaper_variants_accuracy.py            # 365 windows, ~2 min on a GPU
    C2P_FAST_MODE=1 uv run python bench/cheaper_variants_accuracy.py

Writes ``results/bench_variants_metrics.csv`` (one row per model and window) and
``results/bench_variants_skill.csv``.
"""

from __future__ import annotations

import time

import pandas as pd

from c2p.config import get_device, get_paths, is_fast_mode, set_seed
from c2p.data import spain_energy_frame, validate_and_prepare
from c2p.eval import backtest, evaluate, skill_table
from c2p.experiments import DAY_AHEAD_BLOCK, FREQ, LOAD, day_ahead_config
from c2p.models import CHRONOS2, CHRONOS2_SMALL, Chronos2Forecaster, load_chronos2, parameter_count
from c2p.results import save_results
from c2p.schema import ID_COL, TIMESTAMP_COL

CONTEXT = 2048
BASELINE = "chronos-2"


def variants(device: str) -> list[tuple[str, str, str]]:
    """``(name, checkpoint, dtype)`` of the models to compare; half precision only on a GPU."""
    rows = [(BASELINE, CHRONOS2, "float32")]
    if device == "cuda":
        rows.append(("chronos-2-bf16", CHRONOS2, "bfloat16"))
    rows.append(("chronos-2-small", CHRONOS2_SMALL, "float32"))
    return rows


def main() -> int:
    """Run every variant through the day-ahead protocol and write the comparison."""
    set_seed()
    fast = is_fast_mode(default=False)
    paths = get_paths(fast_mode=fast)
    device = get_device()
    raw = spain_energy_frame()[[ID_COL, TIMESTAMP_COL, LOAD]]
    series, _ = validate_and_prepare(raw, freq=FREQ, target=LOAD)
    config = day_ahead_config(fast, max_context=CONTEXT)

    frames = []
    for name, model_id, dtype in variants(device):
        pipeline = load_chronos2(model_id, device=device, dtype=dtype)
        forecaster = Chronos2Forecaster(pipeline, name=name)
        started = time.perf_counter()
        metrics = evaluate(backtest(forecaster, series, config), series, config)
        metrics["params_m"] = parameter_count(pipeline) / 1e6
        metrics["dtype"] = dtype
        frames.append(metrics)
        print(
            f"{name:<16} {metrics['params_m'].iloc[0]:6.1f}M params {dtype:<9} "
            f"WQL {metrics['WQL'].mean():.4f}  cov80 {metrics['coverage_80'].mean():.3f}  "
            f"{time.perf_counter() - started:5.1f} s for {config.n_windows} windows"
        )
        del pipeline, forecaster
    metrics = pd.concat(frames, ignore_index=True)
    print("Saved", save_results(metrics, "bench_variants_metrics", paths).name)

    skill = skill_table(metrics, baseline=BASELINE, metric="WQL", block_length=DAY_AHEAD_BLOCK)
    print("Saved", save_results(skill, "bench_variants_skill", paths).name)
    for row in skill[skill["model"] != BASELINE].itertuples():
        print(
            f"{row.model} vs {BASELINE}: skill {row.skill:+.2%} "
            f"[{row.ci_low:+.2%}, {row.ci_high:+.2%}], wins {row.win_rate:.1%} "
            f"of {int(row.n_units)} windows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
