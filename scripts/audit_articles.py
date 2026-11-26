"""Check the article drafts against the results they quote.

The articles promise that every number is read from a CSV in ``results/``. This script holds
them to it without running a model:

1. **Tables.** Every numeric table row of the articles is rebuilt from the CSV files, with
   each value rounded once from the stored number, and must appear in the article word for
   word. A value typed from a rounded printout (0.0185 read as "1.9%") fails here.
2. **Code.** Every ``from c2p... import`` in a Python block must resolve, and every call to an
   imported ``c2p`` function or class must bind to its real signature, so a snippet that omits
   a required argument is caught before a reader copies it.
3. **Text hygiene.** No em dashes, and none of the prefixes an assisted rewrite can leave
   behind ("Here is the rewritten paragraph").

Usage::

    uv run python scripts/audit_articles.py        # exit code 1 if anything is off

The same checks run in the fast test suite (``tests/test_audit_articles.py``).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from c2p.config import find_repo_root

MINUS = "\u2212"  # the articles write negative numbers with a true minus sign
EM_DASH = "\u2014"
LEAK_PATTERNS = re.compile(
    r"(^here is the|^here's the|rewritten (paragraph|passage|text)|as requested|^sure[,!]|"
    r"certainly[,!]|i have (rewritten|removed)|without em dashes)",
    re.IGNORECASE | re.MULTILINE,
)

# --- Formatting: one rounding, from the stored value -----------------------------------------


def half_up(value: float, decimals: int) -> str:
    """``value`` rounded half up to ``decimals`` places, as an unsigned string.

    The value is first snapped to ten decimals so that binary noise (0.76249999999) does not
    turn a schoolbook tie into a round-down.
    """
    snapped = Decimal(repr(round(abs(float(value)), 10)))
    return str(snapped.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def signed(value: float, decimals: int = 1) -> str:
    """``+1.8`` / ``−0.2``: explicit sign, true minus."""
    return f"{MINUS if value < 0 else '+'}{half_up(value, decimals)}"


def negative(value: float, decimals: int) -> str:
    """Unsigned when positive, true minus when negative (``0.00053`` / ``−0.00007``)."""
    return f"{MINUS if value < 0 else ''}{half_up(value, decimals)}"


def pct(value: float, decimals: int = 1) -> str:
    """A fraction as an unsigned percentage without the sign: 0.6386 -> ``63.9``."""
    return half_up(100 * value, decimals)


def spct(value: float, decimals: int = 1) -> str:
    """A fraction as a signed percentage: 0.018492 -> ``+1.8``."""
    return signed(100 * value, decimals)


def thousands(value: float, decimals: int = 0) -> str:
    """Half-up rounding with a thousands separator: 4210.9 -> ``4,211``."""
    whole, _, fraction = half_up(value, decimals).partition(".")
    return f"{int(whole):,}" + (f".{fraction}" if fraction else "")


def table_row(*values: str) -> str:
    """A Markdown table row from its cells: ``| a | b |``."""
    return "| " + " | ".join(values) + " |"


def interval(row: pd.Series) -> str:
    """``+1.8% [+0.5, +3.1]`` from a skill-table row."""
    return f"{spct(row['skill'])}% [{spct(row['ci_low'])}, {spct(row['ci_high'])}]"


# --- Expected rows, article by article --------------------------------------------------------


class Results:
    """Lazy access to ``results/<name>.csv``."""

    def __init__(self, root: Path) -> None:
        self.folder = root / "results"
        self._cache: dict[str, pd.DataFrame] = {}

    def __call__(self, name: str) -> pd.DataFrame:
        """Return the CSV ``results/<name>.csv`` as a DataFrame (cached)."""
        if name not in self._cache:
            self._cache[name] = pd.read_csv(self.folder / f"{name}.csv")
        return self._cache[name]


def article_1(r: Results) -> Iterator[str]:
    """First forecast and the messy-data table."""
    first = r("01_first_forecast_metrics").iloc[0]
    mae = f"{half_up(first['MAE'], 0)} MW ({pct(first['WAPE'])}% of mean load)"
    yield table_row("MAE of the median", mae)
    yield f"| WQL | {half_up(first['WQL'], 4)} |"
    yield f"| 80% band held | {pct(first['coverage_80'])}% of 168 hours |"
    yield f"| 98% band held | {pct(first['coverage_98'])}% of 168 hours |"
    labels = {
        "spike_as_nan": "Spike set to NaN",
        "gaps_nan": "Gaps as NaN",
        "spike": "Spike (×3)",
        "gaps_zero": "Gaps as 0",
        "rows_dropped_freq": 'Rows dropped, freq="h"',
    }
    messy = r("00_messy_data_summary").set_index("scenario")
    for key, label in labels.items():
        row = messy.loc[key]
        low, high = row["WQL_change_ci_low_pct"], row["WQL_change_ci_high_pct"]
        yield table_row(
            label,
            f"{signed(row['WQL_change_pct'])}%",
            f"{signed(row['WQL_change_gmean_pct'])}%",
            f"[{signed(low)}%, {signed(high)}%]",
            f"{int(row['n_windows_worse'])} / {int(row['n_windows'])}",
        )


def article_2(r: Results) -> Iterator[str]:
    """Baselines, coverage, context sweep, cold start and the two prose intervals."""
    summary = r("02_model_summary").set_index("model")
    skill = r("02_skill_table").set_index("model")
    labels = {
        "operator_forecast": "operator's forecast (reference)",
        "chronos-bolt": "chronos-bolt",
        "chronos-2": "chronos-2",
        "mstl": "mstl",
        "seasonal_naive_weekly": "seasonal_naive (168 h)",
        "auto_ets": "auto_ets",
    }
    for key, label in labels.items():
        s, k = summary.loc[key], skill.loc[key]
        if k["skill"] > 0:
            cell = f"{pct(k['skill'])}% [{pct(k['ci_low'])}, {pct(k['ci_high'])}]"
        else:
            cell = interval(k)
        yield (
            f"| {label} | {half_up(s['WQL'], 4)} | {half_up(s['MASE'], 3)} | {pct(s['WAPE'])}% "
            f"| {cell} | {pct(k['win_rate'], 0)}% |"
        )
    naive = summary.loc["seasonal_naive"]
    yield (
        f"| seasonal_naive (24 h) | {half_up(naive['WQL'], 4)} | {half_up(naive['MASE'], 3)} "
        f"| {pct(naive['WAPE'])}% | 0.0% | n/a |"
    )
    coverage = r("02_coverage").pivot_table(index="model", columns="nominal", values="achieved")
    for model in ("chronos-2", "chronos-bolt", "mstl", "seasonal_naive", "auto_ets"):
        yield f"| {model} | {pct(coverage.loc[model, 0.8])}% | {pct(coverage.loc[model, 0.98])}% |"
    reliability = r("02_reliability").set_index(["model", "level"])["observed"]
    yield f"{pct(reliability[('chronos-2', 0.1)])}% of the actuals fall below its 0.1 quantile"
    yield f"{pct(reliability[('chronos-2', 0.9)])}% below its 0.9 quantile"
    by_step = r("02_error_by_step").set_index(["model", "step"])["WAPE"]
    peak = by_step["chronos-2"]
    yield f"peaks at {pct(peak.max())}% at step {int(peak.idxmax())}"
    backtest = r("02_backtest_metrics")
    versus_bolt = _skill_row(backtest, "chronos-2", "chronos-bolt")
    yield (
        f"Chronos-2's skill over Bolt is {spct(versus_bolt['skill'])}% with an interval of "
        f"[{spct(versus_bolt['ci_low'])}, {spct(versus_bolt['ci_high'])}]"
    )
    versus_mstl = _skill_row(backtest, "chronos-2", "mstl")
    yield (
        f"{pct(versus_mstl['skill'], 0)}% lower than MSTL's "
        f"[{pct(versus_mstl['ci_low'])}, {pct(versus_mstl['ci_high'])}]"
    )
    sweep = r("03_context_sweep").pivot_table(index="context", columns="horizon", values="skill")
    for context, row in sweep.iterrows():
        yield f"| {int(context):,} | {pct(row[24])}% | {pct(row[168])}% |"
    names = {
        72: "72 h (3 days)",
        168: "168 h (1 week)",
        336: "336 h (2 weeks)",
        504: "504 h (3 weeks)",
    }
    cold = r("04_coldstart_sweep").set_index(["history_hours", "measure"])
    for hours, name in names.items():
        cost = cold.loc[(hours, "Cost of a short history")]
        gain = cold.loc[(hours, "Cross-learning gain")]
        yield f"| {name} | {interval(cost)} | {interval(gain)} |"
    synth = r("02_contamination").set_index("model").loc["chronos-2-synth"]
    yield f"{pct(synth['skill'])}% worse"
    yield f"[{pct(synth['ci_high'])}%, {pct(synth['ci_low'])}%]"
    joint = r("04_multivariate_skill").set_index("target")
    yield f"load changes by {interval(joint.loc['load_actual'])}"
    yield f"price by {interval(joint.loc['price_actual'])}"


def _attribution(row: pd.Series) -> str:
    """``+0.0038 [+0.0015, +0.0065]`` from an importance-summary row."""
    return f"{signed(row['mean'], 4)} [{signed(row['ci_low'], 4)}, {signed(row['ci_high'], 4)}]"


def _skill_row(metrics: pd.DataFrame, model: str, baseline: str) -> pd.Series:
    """Skill of ``model`` against ``baseline`` on the 365 daily windows, weekly blocks."""
    from c2p.eval import skill_table
    from c2p.experiments import DAY_AHEAD_BLOCK

    pair = metrics[metrics["model"].isin([model, baseline])]
    table = skill_table(pair, baseline=baseline, metric="WQL", block_length=DAY_AHEAD_BLOCK)
    return table.set_index("model").loc[model]


def article_3(r: Results) -> Iterator[str]:
    """EPF-DE, the covariate ladder, holidays, the leak, attribution and the what-if."""
    epf_wql = r("05_epf_de_metrics").groupby("model")["WQL"].mean()
    epf = r("05_epf_de_skill").set_index("model")
    yield f"| {half_up(epf_wql['univariate'], 4)} | n/a | n/a |"
    for arm in ("past_only", "known_future"):
        wins = f"{pct(epf.loc[arm, 'win_rate'], 0)}%"
        yield table_row(half_up(epf_wql[arm], 4), interval(epf.loc[arm]), wins)
    coverage = r("05_epf_de_metrics").groupby("model")["coverage_80"].mean()
    yield (
        f"going {pct(coverage['univariate'])}%, then {pct(coverage['past_only'])}%, "
        f"then {pct(coverage['known_future'])}%"
    )
    wql = r("05_price_ladder_metrics").groupby("model")["WQL"].mean()
    ladder = r("05_price_ladder_skill").set_index("model")
    yield f"| {half_up(wql['none'], 4)} | n/a | n/a |"
    for rung in ("+calendar", "+TSO forecasts", "+past generation", "TSO forecasts, no calendar"):
        wins = f"{pct(ladder.loc[rung, 'win_rate'], 0)}%"
        yield table_row(half_up(wql[rung], 4), interval(ladder.loc[rung]), wins)
    ladder_metrics = r("05_price_ladder_metrics")
    no_calendar = _skill_row(ladder_metrics, "TSO forecasts, no calendar", "+TSO forecasts")
    yield (
        f"by {pct(no_calendar['skill'])}% "
        f"[{spct(no_calendar['ci_low'])}, {spct(no_calendar['ci_high'])}]"
    )
    generation = _skill_row(ladder_metrics, "+past generation", "TSO forecasts, no calendar")
    yield f"the difference is {interval(generation)}"
    columns = r("05_price_columns").set_index("measure")["value"]
    yield f"a steady {half_up(columns['actual_minus_omie_mean'], 1)} EUR/MWh above it"
    yield f"(standard deviation {half_up(columns['actual_minus_omie_std'], 1)})"
    yield f"a correlation of {half_up(columns['actual_vs_omie_correlation'], 3)}"
    yield f"on {int(columns['days_day_ahead_column_equal_all_hours'])} days of 2018"
    yield f"of another {int(columns['days_day_ahead_column_equal_no_hour'])}"
    holidays = r("05_holiday_windows").set_index(["model", "windows"])
    for arm in ("+holiday flag only", "+calendar"):
        special = holidays.loc[(arm, "holiday windows")]
        wins = round(special["win_rate"] * special["n_units"])
        yield (
            f"| {arm} | {interval(holidays.loc[(arm, 'all windows')])} "
            f"| {interval(special)}, wins {wins} of {int(special['n_units'])} "
            f"| {interval(holidays.loc[(arm, 'ordinary windows')])} |"
        )
    tso, observed = ladder.loc["+TSO forecasts"], ladder.loc["LEAK: observed as known"]
    target = ladder.loc["LEAK: the target itself"]
    yield f"| +TSO forecasts (no leak) | {spct(tso['skill'])}% | {pct(tso['win_rate'], 0)}% |"
    yield (
        f"| LEAK: observed values as known-future | {spct(observed['skill'])}% "
        f"| {pct(observed['win_rate'], 0)}% |"
    )
    yield (
        f"| LEAK: the target itself as known-future | {interval(target)} "
        f"| {pct(target['win_rate'], 0)}% |"
    )
    weekday_wql = r("05_categorical_metrics").groupby("model")["WQL"].mean()
    weekday = r("05_categorical_skill").set_index("model")
    for arm, label in (
        ("+weekday as a number", "a number, 0 to 6"),
        ("+weekday as a category", "a category, seven labels"),
    ):
        yield table_row(label, half_up(weekday_wql[arm], 4), interval(weekday.loc[arm]))
    load = r("05_operator_metrics").groupby("model")[["WQL", "WAPE", "coverage_80"]].mean()
    point = r("05_operator_point_skill").set_index("model")
    arms = {
        "history only": "Chronos-2, history only",
        "operator forecast as covariate": "Chronos-2, operator forecast as a covariate",
        "operator forecast, corrected": "Chronos-2 on the operator's errors",
    }
    for arm, label in arms.items():
        yield table_row(
            label,
            half_up(load.loc[arm, "WQL"], 4),
            f"{pct(load.loc[arm, 'WAPE'], 2)}%",
            f"{pct(load.loc[arm, 'coverage_80'])}%",
            interval(point.loc[arm]),
        )
    operator = load.loc["operator forecast"]
    yield table_row(
        "The operator's forecast",
        half_up(operator["WQL"], 4),
        f"{pct(operator['WAPE'], 2)}%",
        "n/a",
        "n/a",
    )
    corrected = r("05_operator_skill").set_index("model").loc["operator forecast, corrected"]
    yield (
        f"is {pct(corrected['skill'])}% "
        f"[{spct(corrected['ci_low'])}, {spct(corrected['ci_high'])}] below the operator's"
    )
    cheap = r("06_covariate_importance_summary").set_index(["player", "method"])
    shapley = r("06_covariate_shapley_summary").set_index("player")
    for player in ("wind forecast", "load forecast", "solar forecast", "calendar"):
        cells = [
            _attribution(cheap.loc[(player, "Ablation")]),
            _attribution(cheap.loc[(player, "Permutation")]),
            _attribution(shapley.loc[player]),
        ]
        yield table_row(player, *cells)
    yield table_row("all four together", "n/a", "n/a", _attribution(shapley.loc["all four"]))
    segments = {
        "lags 1-24": "lags 1–24 (the last day)",
        "lags 25-168": "lags 25–168 (rest of the week)",
        "lags 169-672": "lags 169–672 (rest of four weeks)",
        "lags 673+": "lags 673+ (older)",
    }
    context = r("06_context_shapley_summary").set_index("player")
    for key, label in segments.items():
        yield table_row(label, _attribution(context.loc[key]))
    yield table_row("the whole context", _attribution(context.loc["whole context"]))
    scales = r("06_context_vs_covariates").iloc[0]
    yield (
        f"context removes {half_up(scales['context_wql_removed'], 4)} of WQL; the four "
        f"covariate players of section 7 remove "
        f"{half_up(scales['covariates_wql_removed'], 4)} between them"
    )
    yield (
        f"The ratio is {half_up(scales['ratio'], 0)}, with an interval from "
        f"{half_up(scales['ratio_ci_low'], 0)} to {half_up(scales['ratio_ci_high'], 0)}"
    )
    whole = context.loc["whole context", "mean"]
    older = context.loc[["lags 25-168", "lags 169-672", "lags 673+"], "mean"].sum()
    yield f"holds {pct(context.loc['lags 1-24', 'mean'] / whole, 0)}% of the total"
    yield f"and hold {pct(context.loc['lags 673+', 'mean'] / whole, 0)}%"
    yield f"the three older segments hold {pct(older / whole, 0)}% of"
    hidden = r("06_occlusion_summary").set_index("first_lag")
    last, rest = hidden.loc[1], hidden.drop(index=1)["mean"].max()
    yield (
        f"{half_up(last['mean'], 4)} [{half_up(last['ci_low'], 4)}, "
        f"{half_up(last['ci_high'], 4)}] of WQL, and no other single day costs more than "
        f"{half_up(rest, 4)}"
    )
    price = r("06_what_if").groupby("wind_factor")["predictions"].mean()
    yield f"| Operator's forecast | {half_up(price[1.0], 2)} EUR/MWh |"
    yield f"| Wind × 1.5 | {half_up(price[1.5], 2)} ({signed(price[1.5] - price[1.0], 2)}) |"
    yield f"| Wind × 0.5 | {half_up(price[0.5], 2)} ({signed(price[0.5] - price[1.0], 2)}) |"


def _runs(row: pd.Series) -> str:
    """``+2.3% to +3.0%``: the skill of the worst and the best single run behind a row."""
    return f"{spct(row['run_skill_min'])}% to {spct(row['run_skill_max'])}%"


def article_4(r: Results) -> Iterator[str]:
    """Rank and seed sweep, the budget chosen on validation, full fine-tune, forgetting, price."""
    costs = r("07_lora_costs")
    for rank, group in costs.groupby("rank", sort=True):
        c = group.mean(numeric_only=True)
        yield table_row(
            f"`lora_r{int(rank)}`",
            f"{int(c['trainable_parameters']):,}",
            f"{pct(c['trainable_parameters'] / c['total_parameters'], 2)}%",
            f"{half_up(c['train_seconds'] / 60, 1)} min",
            f"{thousands(c['peak_vram_mb'])} MB",
            f"{half_up(c['artifact_mb'], 1)} MB",
            half_up(c["best_eval_loss"], 4),
        )
    zero = r("07_lora_metrics")
    zero = zero.loc[zero["model"] == "zero-shot", ["WQL", "coverage_80"]].mean()
    yield table_row(
        "zero-shot", half_up(zero["WQL"], 4), "n/a", "n/a", "n/a", f"{pct(zero['coverage_80'])}%"
    )
    ranks = r("07_lora_rank_summary").set_index("model")
    for run in ("lora_r4", "lora_r8", "lora_r16"):
        k = ranks.loc[run]
        yield table_row(
            f"`{run}`",
            half_up(k["WQL"], 4),
            interval(k),
            _runs(k),
            f"{pct(k['win_rate'], 0)}%",
            f"{pct(k['coverage_80'])}%",
        )
    single = r("07_lora_run_skill")
    single = single.loc[single["model"] != "zero-shot", "skill"]
    yield (
        f"differ by {half_up(100 * (ranks['skill'].max() - ranks['skill'].min()), 1)} points; two "
        f"single runs can differ by {half_up(100 * (single.max() - single.min()), 1)}"
    )
    validation = r("07_budget_validation").set_index("candidate")
    test = r("07_budget_test").set_index("candidate")
    for candidate, v in validation.iterrows():
        yield table_row(
            str(candidate),
            f"{int(v['best_step']):,} of {int(v['num_steps']):,}",
            half_up(v["best_eval_loss"], 4),
            interval(v),
            interval(test.loc[candidate]),
        )
    names = {
        "lora_r8": "LoRA rank 8, library defaults",
        "lora_tuned": "LoRA rank 8, budget chosen on validation",
        "full": "full fine-tune, library defaults",
    }
    compare = r("08_comparison_summary").set_index("model")
    for run, label in names.items():
        k = compare.loc[run]
        yield table_row(
            label,
            half_up(k["WQL"], 4),
            interval(k),
            _runs(k),
            f"{spct(k['mean_change'])}%",
            f"{pct(k['coverage_80'])}%",
        )
        yield table_row(
            label,
            f"{half_up(k['train_seconds'] / 60, 1)} min",
            f"{thousands(k['peak_vram_mb'])} MB",
            f"{half_up(k['artifact_mb'], 1)} MB",
            f"{half_up(k['latency_ms'], 0)} ms",
        )
    away = r("08_forgetting_summary").set_index("model")
    away_zero = r("08_forgetting_metrics")
    away_zero = away_zero.loc[away_zero["model"] == "zero-shot", "WQL"].mean()
    yield table_row("zero-shot", half_up(away_zero, 4), "n/a", "n/a", "n/a")
    for run, label in names.items():
        k = away.loc[run]
        yield table_row(
            label, half_up(k["WQL"], 4), interval(k), _runs(k), f"{pct(k['win_rate'], 0)}%"
        )
    price_metrics = r("08_price_metrics")
    price_zero = price_metrics.loc[price_metrics["model"] == "zero-shot"]
    price_zero = price_zero[["WQL", "coverage_80"]].mean()
    yield table_row(
        "zero-shot, with the covariates",
        half_up(price_zero["WQL"], 4),
        "n/a",
        "n/a",
        f"{pct(price_zero['coverage_80'])}%",
    )
    price = r("08_price_summary").set_index("model")
    price_validation = r("08_price_validation").set_index("candidate")
    labels = {
        "price LoRA, lr 1e-5": ("lr 1e-5", "LoRA, lr 1e-5 (one seed)"),
        "price LoRA, lr 1e-4": ("lr 1e-4", "LoRA, lr 1e-4 (chosen, three seeds)"),
    }
    for model, (candidate, label) in labels.items():
        k = price.loc[model]
        yield table_row(
            label,
            half_up(k["WQL"], 4),
            interval(price_validation.loc[candidate]),
            interval(k),
            f"{pct(k['coverage_80'])}%",
        )
    sweep = r("03_context_sweep")
    by_context = sweep[sweep["horizon"] == 24].set_index("context")["skill"]
    gain = 100 * (by_context[2048] - by_context[64])
    ladder = r("05_price_ladder_skill").set_index("model")
    yield f"| Context 64 → 2,048 hours (Part 2) | +{half_up(gain, 0)} points |"
    yield (
        f"| The operator's forecasts as covariates (Part 3) | "
        f"{spct(ladder.loc['TSO forecasts, no calendar', 'skill'])}% |"
    )
    yield f"| LoRA, budget chosen on validation | {spct(compare.loc['lora_tuned', 'skill'])}% |"
    yield f"| LoRA, library defaults | {spct(compare.loc['lora_r8', 'skill'])}% |"
    yield f"| Full fine-tune, library defaults | {spct(compare.loc['full', 'skill'])}% |"
    yield f"| LoRA on top of covariates | {spct(price.loc['price LoRA, lr 1e-4', 'skill'])}% |"


def _skill_against(metrics: pd.DataFrame, model: str, baseline: pd.Series) -> float:
    """fev-bench skill of ``model`` against a per-cutoff baseline WQL series."""
    own = metrics.loc[metrics["model"] == model].set_index("cutoff")["WQL"]
    ratio = own / baseline.reindex(own.index)
    return float(1 - np.exp(np.log(ratio).mean()))


def article_5(r: Results) -> Iterator[str]:
    """Throughput, the cheaper variants, service latency and costs."""
    devices = {"cuda": "RTX 3070", "cpu": "8 CPU cores"}
    names = {"amazon/chronos-2": "chronos-2", "autogluon/chronos-2-small": "chronos-2-small"}
    for model, name in names.items():
        for device, label in devices.items():
            bench = r(f"bench_latency_{device}")
            rows = bench[(bench["model"] == model) & (bench["context_length"] == 2048)]
            speed = rows.set_index("batch_size")["series_per_second"]
            cells = " | ".join(f"{half_up(speed[batch], 1)}/s" for batch in (1, 8, 32, 128))
            yield f"| {name}, {label} | {cells} | {half_up(speed.max(), 0)}/s |"
    small = r("bench_variants_metrics")
    means = small.groupby("model")[["WQL", "coverage_80"]].mean()
    k = r("bench_variants_skill").set_index("model").loc["chronos-2-small"]
    naive = r("02_backtest_metrics")
    naive = naive.loc[naive["model"] == "seasonal_naive"].set_index("cutoff")["WQL"]
    base, tiny = means.loc["chronos-2"], means.loc["chronos-2-small"]
    yield table_row(
        "chronos-2",
        "119.5M",
        half_up(base["WQL"], 4),
        f"{pct(base['coverage_80'])}%",
        "n/a",
        "n/a",
        f"{pct(_skill_against(small, 'chronos-2', naive))}%",
    )
    yield (
        f"| chronos-2-small | 27.9M | {half_up(tiny['WQL'], 4)} | {pct(tiny['coverage_80'])}% "
        f"| {interval(k)} | {pct(k['win_rate'], 0)}% "
        f"| {pct(_skill_against(small, 'chronos-2-small', naive))}% |"
    )
    half = means.loc["chronos-2-bf16"]
    k16 = r("bench_variants_skill").set_index("model").loc["chronos-2-bf16"]
    yield (
        f"| chronos-2, bfloat16 | 119.5M | {half_up(half['WQL'], 4)} "
        f"| {pct(half['coverage_80'])}% | {interval(k16)} | {pct(k16['win_rate'], 0)}% "
        f"| {pct(_skill_against(small, 'chronos-2-bf16', naive))}% |"
    )
    for precision, name in (
        ("float32", "bench_latency_cuda"),
        ("bfloat16", "bench_latency_cuda_bf16"),
    ):
        bench = r(name)
        rows = bench[(bench["model"] == "amazon/chronos-2") & (bench["context_length"] == 2048)]
        rows = rows.set_index("batch_size")
        cells = " | ".join(
            f"{half_up(rows.loc[b, 'series_per_second'], 1)}/s" for b in (1, 8, 32, 128)
        )
        yield f"| {precision} | {cells} | {thousands(rows.loc[128, 'peak_vram_mb'])} MB |"
        yield f"{half_up(rows.loc[1, 'peak_vram_mb'], 0)} MB"
    for _, row in r("09_service_latency").iterrows():
        yield (
            f"| {int(row['n_series'])} | {thousands(row['client_p50_ms'], 1)} ms "
            f"| {thousands(row['client_p95_ms'], 1)} ms | {thousands(row['server_p50_ms'], 1)} ms |"
        )
    cost = r("09_cost_per_million")
    cost = cost[(cost["model"] == "amazon/chronos-2") & (cost["context_length"] == 2048)]
    rates = r("09_hosting_rates").set_index("instance")
    for instance in (
        "ml.g4dn.xlarge",
        "ml.g6.xlarge",
        "ml.g5.xlarge",
        "ml.g5.2xlarge",
        "ml.c7i.2xlarge",
        "ml.m5.2xlarge",
    ):
        c = cost.set_index("instance").loc[instance]
        gpu = rates.loc[instance, "gpu"]
        yield (
            f"| {instance} | {int(c['vcpu'])} / {'none' if not gpu > 0 else int(gpu)} "
            f"| {half_up(c['usd_per_hour'], 3)} | {thousands(c['usd_per_month'])} "
            f"| {half_up(c['usd_per_million_forecasts'], 2)} |"
        )
    for _, row in r("09_serverless_cost").iterrows():
        yield table_row(
            names[row["model"]],
            f"{int(row['context_length']):,}",
            half_up(row["seconds_per_request"], 3),
            half_up(row["usd_per_million_requests"], 2),
        )
    offer = pd.read_csv(r.folder.parent / "data" / "reference" / "cloud_prices.csv").iloc[0]
    yield (
        f"offer version {offer['offer_version']} for us-east-1, "
        f"with rates effective {str(offer['effective_date'])[:10]}"
    )


def article_6(r: Results) -> Iterator[str]:
    """The monitoring replay and the static and adaptive calibration of the intervals."""
    rolling = r("09_monitoring_rolling")
    wape, cov = rolling["wape"], rolling["coverage"]
    yield table_row(
        "rolling WAPE (28 cutoffs)",
        *(f"{pct(value, 2)}%" for value in (wape.min(), wape.median(), wape.max())),
    )
    yield table_row(
        "80% interval coverage",
        *(f"{pct(value)}%" for value in (cov.min(), cov.median(), cov.max())),
    )
    breaches, episodes = int(rolling["alert"].sum()), int(rolling["first_breach"].sum())
    yield f"{breaches} of {len(rolling)} cutoffs breach, in {episodes} episodes"
    summary = r("09_conformal_summary").set_index("model")
    months = r("09_conformal_by_month")
    worst = months.loc[months.groupby("model")["coverage"].idxmin()].set_index("model")
    labels = {
        "zero-shot": "zero-shot",
        "conformal_additive": "conformal, additive",
        "conformal_scaled": "conformal, scaled",
        "conformal_rolling": "conformal, rolling pool",
        "conformal_aci": "conformal, ACI",
    }
    for key, label in labels.items():
        s, w = summary.loc[key], worst.loc[key]
        month = pd.Period(w["period"]).strftime("%b")
        skill = "n/a" if key == "zero-shot" else interval(s)
        yield table_row(
            label,
            half_up(s["WQL"], 4),
            f"{pct(s['coverage_80'])}%",
            f"{pct(s['coverage_98'])}%",
            skill,
            f"{pct(w['coverage'])}% ({month})",
        )
    tail = summary["WQL_worst_decile"]
    yield f"({half_up(tail['conformal_aci'], 3)} with ACI against {half_up(tail['zero-shot'], 3)})"
    monthly = months.pivot_table(index="period", columns="model", values="coverage")
    january, february = monthly.loc["2018-01"], monthly.loc["2018-02"]
    yield (
        f"go from {pct(january['conformal_additive'])}% and "
        f"{pct(february['conformal_additive'])}% to {pct(january['conformal_rolling'])}% and "
        f"{pct(february['conformal_rolling'])}% with the rolling pool, and to "
        f"{pct(january['conformal_aci'])}% and {pct(february['conformal_aci'])}% with ACI"
    )
    aci, static, raw = (
        monthly[name] for name in ("conformal_aci", "conformal_additive", "zero-shot")
    )
    yield (
        f"ACI stays between {pct(aci.min())}% and {pct(aci.max())}%, where the static correction "
        f"ran from {pct(static.min())}% to {pct(static.max())}% and the uncorrected model from "
        f"{pct(raw.min())}% to {pct(raw.max())}%"
    )
    december = monthly.loc["2018-12"]
    assert all(
        monthly[name].idxmin() == "2018-12" for name in ("conformal_rolling", "conformal_aci")
    )
    yield (
        f"at {pct(december['conformal_rolling'])}% for the rolling pool and "
        f"{pct(december['conformal_aci'])}% for ACI, as it was for the uncorrected model at "
        f"{pct(december['zero-shot'])}%"
    )
    assert static.idxmin() == "2018-02"
    yield (
        f"its worst month in February, at {pct(static.min())}%, with December a close second at "
        f"{pct(december['conformal_additive'])}%"
    )


def _plain_interval(row: pd.Series) -> str:
    """``34.3% [28.1, 39.9]``: a skill-table row without the signs."""
    return f"{pct(row['skill'])}% [{pct(row['ci_low'])}, {pct(row['ci_high'])}]"


def article_7(r: Results) -> Iterator[str]:
    """Totals, the daily frequency, the panel of stores, the embeddings and the one-page table."""
    total = r("10_daily_total_summary").set_index("model")
    labels = {
        "hourly quantiles, summed": "hourly forecast, quantiles summed",
        "hourly quantiles, independent hours": "hourly forecast, hours taken as unrelated",
        "daily series, forecast directly": "daily series, forecast directly",
        "seasonal_naive": "seasonal naive (the same weekday last week)",
    }
    for key, label in labels.items():
        row = total.loc[key]
        yield table_row(
            label,
            f"{pct(row['WAPE'], 2)}%",
            half_up(row["WQL"], 4),
            f"{pct(row['coverage_80'])}%",
            f"{pct(row['coverage_98'])}%",
        )
    operator = total.loc["operator forecast, summed"]
    yield table_row(
        "the operator's hourly forecast, summed",
        f"{pct(operator['WAPE'], 2)}%",
        half_up(operator["WQL"], 4),
        "n/a",
        "n/a",
    )
    paired = r("10_paired_skill").set_index(["comparison", "model"])
    summed = paired.loc[("tomorrow's total", "hourly quantiles, summed")]
    yield f"a WQL {_plain_interval(summed)} lower"
    calibrated = r("10_total_calibration_summary").set_index("model")
    labels = {
        "hourly quantiles, summed": "hourly quantiles summed, as they come",
        "calibrated hours, summed": "hours calibrated, then summed",
        "summed, then calibrated": "summed, then calibrated as a daily series",
    }
    for key, label in labels.items():
        row = calibrated.loc[key]
        yield table_row(
            label,
            half_up(row["WQL"], 4),
            f"{pct(row['coverage_80'])}%",
            f"{pct(row['coverage_98'])}%",
        )
    week = r("10_week_ahead_summary").set_index("model")
    labels = {
        "daily series, forecast directly": "daily series, forecast directly",
        "hourly quantiles, summed": "hourly forecast (168 steps), quantiles summed",
        "mstl": "MSTL on the daily series",
    }
    for key, label in labels.items():
        row = week.loc[key]
        yield table_row(
            label,
            half_up(row["WQL"], 4),
            f"{pct(row['WAPE'], 2)}%",
            f"{pct(row['coverage_80'])}%",
            f"{pct(row['skill'])}% [{pct(row['ci_low'])}, {pct(row['ci_high'])}]",
        )
    naive = week.loc["seasonal_naive"]
    yield table_row(
        "seasonal naive (the same weekday last week)",
        half_up(naive["WQL"], 4),
        f"{pct(naive['WAPE'], 2)}%",
        f"{pct(naive['coverage_80'])}%",
        "n/a",
    )
    direct = paired.loc[("a week of totals", "daily series, forecast directly")]
    yield f"ahead by {interval(direct)}"
    panel = r("10_panel_summary").set_index("model")
    labels = {
        "history only": "history only",
        "+known covariates": "+ known covariates",
        "+known and past covariates": "+ known and past covariates",
        "+known covariates, cross-learning": "+ known covariates, cross-learning",
    }
    for key, label in labels.items():
        row = panel.loc[key]
        yield table_row(
            label,
            half_up(row["WQL"], 4),
            f"{pct(row['coverage_80'])}%",
            _plain_interval(row.filter(like="window_").rename(lambda name: name[7:])),
            f"{pct(row['store_skill'])}% [{pct(row['store_ci_low'])}, {pct(row['store_ci_high'])}]",
        )
    naive = panel.loc["seasonal_naive"]
    yield table_row(
        "seasonal naive (the same weekday last week)",
        half_up(naive["WQL"], 4),
        f"{pct(naive['coverage_80'])}%",
        "n/a",
        "n/a",
    )
    stores = paired.loc["panel"]
    known, past, cross = (
        stores.loc[name]
        for name in (
            "+known covariates",
            "+known and past covariates",
            "+known covariates, cross-learning",
        )
    )
    yield f"lower by {_plain_interval(known)}, and it is lower on {pct(known['win_rate'], 0)}% of"
    speed = panel["stores_per_second"]
    yield (
        f"forecast {half_up(speed['history only'], 0)} stores per second and the arm with five "
        f"known covariates {half_up(speed['+known covariates'], 0)}"
    )
    yield f"({interval(past)}, better on {pct(past['win_rate'], 0)}% of the stores)"
    yield f"({interval(cross)}, better on {pct(cross['win_rate'], 0)}% of the stores)"
    probes = r("11_probe_accuracy").pivot_table(
        index="representation", columns="task", values="accuracy", sort=False
    )
    names = {
        "raw window": "raw window (control)",
        "embedding (mean)": "embedding, mean of the patches",
        "embedding (reg)": "embedding, [REG] token",
        "embedding (forecast)": "embedding, future-patch token",
    }
    for key, label in names.items():
        yield table_row(
            label,
            f"{pct(probes.loc[key, 'weekday of the next day'])}%",
            f"{pct(probes.loc[key, 'month of the next day'])}%",
        )
    floor = r("11_probe_accuracy").groupby("task", sort=False)["majority_class"].first()
    yield (
        f"{pct(floor['weekday of the next day'])}% for the weekday and "
        f"{pct(floor['month of the next day'])}% for the month"
    )
    analog = r("11_analog_forecast").set_index("forecast")
    yield table_row(
        "yesterday repeated", f"{pct(analog.loc['yesterday repeated', 'WAPE'], 2)}%", "n/a"
    )
    yield table_row(
        "analogs by raw window (control)",
        f"{pct(analog.loc['analogs: raw window', 'WAPE'], 2)}%",
        "n/a",
    )
    for key, label in list(names.items())[1:]:
        row = analog.loc[f"analogs: {key}"]
        gap = (
            f"{spct(row['vs_raw_window'], 2)} [{spct(row['vs_raw_ci_low'], 2)}, "
            f"{spct(row['vs_raw_ci_high'], 2)}]"
        )
        yield table_row(f"analogs by {label}", f"{pct(row['WAPE'], 2)}%", gap)
    auc = r("11_anomaly_auc").pivot_table(
        index="anomaly", columns="representation", values="auc", sort=False
    )
    counts = r("11_anomaly_auc").groupby("anomaly", sort=False)["n_anomalies"].first()
    for anomaly, row in auc.iterrows():
        yield table_row(str(anomaly), *(half_up(row[key], 2) for key in names))
    yield f"the {int(counts['holiday on a working day'])} windows that end on a public holiday"
    neighbors = r("11_series_neighbors").set_index("series")["cosine"]
    yield (
        f"(cosine {half_up(neighbors['wind_onshore_forecast_tso'], 3)}), "
        f"the load forecast the load "
        f"({half_up(neighbors['load_forecast_tso'], 3)}), the solar forecast the solar generation "
        f"({half_up(neighbors['solar_forecast_tso'], 3)})"
    )
    for _, row in r("12_series_summary").iterrows():
        yield table_row(row["lever"], row["effect"], row["measured_as"], row["part"])
    backtest = r("02_backtest_metrics")
    wape = backtest.loc[backtest["model"] == "chronos-2", "WAPE"].mean()
    best = analog.loc["analogs: embedding (forecast)", "WAPE"]
    yield f"WAPE of {pct(wape)}%, and five analog days reach {pct(best, 2)}%"
    joint = r("04_multivariate_skill").set_index("target")
    ladder = r("05_price_ladder_skill").set_index("model")
    weekday = r("05_categorical_skill").set_index("model")
    yield f"({interval(joint.loc['load_actual'])})"
    yield f"({interval(ladder.loc['+calendar'])})"
    yield f"({interval(weekday.loc['+weekday as a category'])})"
    yield f"({interval(_skill_row(backtest, 'chronos-2', 'chronos-bolt'))})"


EXPECTED: dict[str, Callable[[Results], Iterator[str]]] = {
    "01": article_1,
    "02": article_2,
    "03": article_3,
    "04": article_4,
    "05": article_5,
    "06": article_6,
    "07": article_7,
}

# --- Checks ---------------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Drop bold and code marks and collapse whitespace, so only words and numbers compare."""
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", text)


def check_rows(article: str, text: str, results: Results) -> list[str]:
    """Rows rebuilt from the CSV files that the article does not contain."""
    haystack = normalise(text)
    return [
        f"{article}: not found (rebuilt from results/): {row}"
        for row in EXPECTED[article](results)
        if normalise(row) not in haystack
    ]


def check_text(article: str, text: str) -> list[str]:
    """Em dashes and leftovers of an assisted rewrite."""
    problems = []
    if EM_DASH in text:
        problems.append(f"{article}: {text.count(EM_DASH)} em dash(es)")
    for match in LEAK_PATTERNS.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        problems.append(f"{article}: rewrite leftover at line {line}: {match.group(0)!r}")
    return problems


def check_code(article: str, text: str) -> list[str]:
    """Imports that do not resolve and ``c2p`` calls that do not bind to the real signature."""
    problems = []
    imported: dict[str, object] = {}  # an article imports a name once and reuses it later
    for block in re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL):
        try:
            tree = ast.parse(block)
        except SyntaxError as error:
            problems.append(f"{article}: Python block does not parse: {error}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("c2p"):
                module = importlib.import_module(str(node.module))
                for alias in node.names:
                    if hasattr(module, alias.name):
                        imported[alias.asname or alias.name] = getattr(module, alias.name)
                    else:
                        problems.append(f"{article}: {node.module} has no {alias.name!r}")
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            target = imported.get(node.func.id)
            if target is None or not callable(target):
                continue
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
                keyword.arg is None for keyword in node.keywords
            ):
                continue  # *args / **kwargs: nothing to check statically
            try:
                inspect.signature(target).bind(
                    *([None] * len(node.args)),
                    **{str(keyword.arg): None for keyword in node.keywords},
                )
            except TypeError as error:
                problems.append(
                    f"{article}: {node.func.id}(...) at line {node.lineno} of its block: {error}"
                )
    return problems


def audit(root: Path | None = None) -> list[str]:
    """Run every check on the articles and return the problems found."""
    root = root if root is not None else find_repo_root()
    results = Results(root)
    problems: list[str] = []
    for article in EXPECTED:
        (path,) = sorted((root / "articles").glob(f"{article}-*.md"))
        text = path.read_text(encoding="utf-8")
        problems += check_text(article, text)
        problems += check_rows(article, text, results)
        problems += check_code(article, text)
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Print the problems, if any, and return the exit code."""
    del argv
    problems = audit()
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s) in the article drafts.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
