"""Accuracy of ``chronos-2-small`` against ``chronos-2`` on the series' day-ahead protocol.

``bench_cpu_gpu.py`` measures how much faster the 28M-parameter checkpoint is. Speed is only
half of a trade, so this script measures the other half with the exact protocol of article 2
(365 day-ahead windows over 2018, 24-hour horizon, 2,048 hours of context, 11 quantiles) and
reports the skill of the small model against the base one with a bootstrap interval over windows.

The two numbers belong side by side: a checkpoint that is 3-4x faster and 1% worse is a
different decision from one that is 3-4x faster and 15% worse.

Usage::

    uv run python bench/small_vs_base_accuracy.py            # 365 windows, ~1 min on a GPU
    C2P_FAST_MODE=1 uv run python bench/small_vs_base_accuracy.py

Writes ``results/bench_small_vs_base_metrics.csv`` (one row per model and window) and
``results/bench_small_vs_base_skill.csv``.
"""

from __future__ import annotations

import time

import pandas as pd

from c2p.config import get_device, get_paths, is_fast_mode, set_seed
from c2p.data import spain_energy_frame, validate_and_prepare
from c2p.eval import backtest, evaluate, skill_table
from c2p.experiments import FREQ, LOAD, day_ahead_config
from c2p.models import CHRONOS2, CHRONOS2_SMALL, Chronos2Forecaster, load_chronos2, parameter_count
from c2p.results import save_results
from c2p.schema import ID_COL, TIMESTAMP_COL

CONTEXT = 2048


def main() -> int:
    """Run both checkpoints through the day-ahead protocol and write the comparison."""
    set_seed()
    fast = is_fast_mode(default=False)
    paths = get_paths(fast_mode=fast)
    device = get_device()
    raw = spain_energy_frame()[[ID_COL, TIMESTAMP_COL, LOAD]]
    series, _ = validate_and_prepare(raw, freq=FREQ, target=LOAD)
    config = day_ahead_config(fast, max_context=CONTEXT)

    frames = []
    for model_id, name in ((CHRONOS2, "chronos-2"), (CHRONOS2_SMALL, "chronos-2-small")):
        pipeline = load_chronos2(model_id, device=device)
        forecaster = Chronos2Forecaster(pipeline, name=name)
        started = time.perf_counter()
        metrics = evaluate(backtest(forecaster, series, config), series, config)
        metrics["params_m"] = parameter_count(pipeline) / 1e6
        frames.append(metrics)
        print(
            f"{name:<16} {metrics['params_m'].iloc[0]:6.1f}M params  "
            f"WQL {metrics['WQL'].mean():.4f}  cov80 {metrics['coverage_80'].mean():.3f}  "
            f"{time.perf_counter() - started:5.1f} s for {config.n_windows} windows"
        )
    metrics = pd.concat(frames, ignore_index=True)
    print("Saved", save_results(metrics, "bench_small_vs_base_metrics", paths).name)

    skill = skill_table(metrics, baseline="chronos-2", metric="WQL")
    print("Saved", save_results(skill, "bench_small_vs_base_skill", paths).name)
    small = skill[skill["model"] == "chronos-2-small"].iloc[0]
    print(
        f"\nchronos-2-small vs chronos-2: skill {small['skill']:+.2%} "
        f"[{small['ci_low']:+.2%}, {small['ci_high']:+.2%}], wins {small['win_rate']:.1%} "
        f"of {int(small['n_units'])} windows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
