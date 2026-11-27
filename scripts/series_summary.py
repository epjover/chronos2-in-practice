"""Collect what the series measured into one table and one figure.

Part 7 closes the series with a page that puts every lever next to what it bought. Typing that
page by hand would break the promise the articles make (every number is read from a CSV), so
this script builds it from the results the notebooks and benchmarks already wrote:

- ``results/12_series_summary.csv``: one row per lever, with the effect as it is printed in the
  article, what it was measured as, and the part that measured it.
- ``results/12_two_readings.csv``: every comparison between two models that the series quotes,
  read both ways from the per-window (or per-store) errors the notebooks saved: the skill on a
  typical window and the skill on the total error, each with its bootstrap interval.
- ``results/12_decision_path.csv`` and ``results/figures/12_decision_path.png``: the order in
  which to pull the levers on a new series, each step with the measurement that backs it.

It runs no model. Run it after the notebooks, the benchmarks and ``scripts/cost_table.py``.

Usage::

    uv run python scripts/series_summary.py
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import NamedTuple

import pandas as pd

from c2p.config import Paths, get_paths
from c2p.eval import MODEL_COL, WINDOW_COL, skill_score
from c2p.experiments import DAY_AHEAD_BLOCK
from c2p.plots.architecture import plot_decision_path
from c2p.plotting import save_figure
from c2p.results import load_results, save_results

MINUS = "\u2212"  # the articles write negative numbers with a true minus sign
TIMES = "\u00d7"
CONTEXT = 2048  # the context length of the day-ahead protocol
SUMMARY_COLUMNS = ["part", "lever", "effect", "total", "measured_as"]
NOT_A_COMPARISON = "n/a"  # rows of the summary that are not a skill of one model over another
STORE = "item_id"


class Comparison(NamedTuple):
    """One model against another on the errors saved in ``results/<metrics>.csv``."""

    part: int
    metrics: str
    model: str
    baseline: str
    block_length: int = DAY_AHEAD_BLOCK  # 365 consecutive days are resampled in weeks
    unit: str = WINDOW_COL
    metric: str = "WQL"
    more: str | None = None  # a second file whose rows join the first (another model)


COMPARISONS = (
    Comparison(2, "02_backtest_metrics", "chronos-2", "seasonal_naive"),
    Comparison(2, "02_backtest_metrics", "chronos-bolt", "seasonal_naive"),
    Comparison(2, "02_backtest_metrics", "mstl", "seasonal_naive"),
    Comparison(2, "02_backtest_metrics", "chronos-2", "chronos-bolt"),
    Comparison(2, "02_backtest_metrics", "chronos-2", "mstl"),
    Comparison(2, "02_backtest_metrics", "chronos-2-synth", "chronos-2", more="02_synth_metrics"),
    Comparison(3, "05_epf_de_metrics", "known_future", "univariate", block_length=1),
    Comparison(3, "05_epf_de_metrics", "past_only", "univariate", block_length=1),
    Comparison(3, "05_price_ladder_metrics", "TSO forecasts, no calendar", "none"),
    Comparison(3, "05_price_ladder_metrics", "+past generation", "TSO forecasts, no calendar"),
    Comparison(3, "05_operator_metrics", "operator forecast, corrected", "operator forecast"),
    Comparison(3, "05_operator_metrics", "history only", "operator forecast", metric="WAPE"),
    Comparison(4, "08_comparison_metrics", "lora_r8", "zero-shot"),
    Comparison(4, "08_comparison_metrics", "lora_tuned", "zero-shot"),
    Comparison(4, "08_comparison_metrics", "full", "zero-shot"),
    Comparison(4, "08_comparison_metrics", "lora_tuned", "lora_r8"),
    Comparison(4, "08_comparison_metrics", "full", "lora_r8"),
    Comparison(4, "07_budget_validation_metrics", "4,000 steps, lr 1e-4", "zero-shot"),
    Comparison(4, "07_budget_test_metrics", "4,000 steps, lr 1e-4", "zero-shot"),
    Comparison(4, "08_forgetting_metrics", "lora_tuned", "zero-shot", block_length=1),
    Comparison(5, "bench_variants_metrics", "chronos-2-small", "chronos-2"),
    Comparison(5, "bench_variants_metrics", "chronos-2-bf16", "chronos-2"),
    Comparison(6, "09_conformal_metrics", "conformal_additive", "zero-shot"),
    Comparison(6, "09_conformal_metrics", "conformal_rolling", "zero-shot"),
    Comparison(6, "09_conformal_metrics", "conformal_aci", "zero-shot"),
    Comparison(
        7, "10_daily_total_metrics", "hourly quantiles, summed", "daily series, forecast directly"
    ),
    Comparison(
        7,
        "10_week_ahead_metrics",
        "daily series, forecast directly",
        "hourly quantiles, summed",
        block_length=1,
    ),
    Comparison(
        7, "10_panel_store_errors", "+known covariates", "history only", block_length=1, unit=STORE
    ),
    Comparison(
        7,
        "10_panel_store_errors",
        "+known covariates, cross-learning",
        "+known covariates",
        block_length=1,
        unit=STORE,
    ),
)
# Result name -> (row of the table, words of the decision path).
FINE_TUNES = {
    "lora_r8": ("LoRA rank 8, library defaults", "LoRA at the library defaults"),
    "lora_tuned": ("LoRA rank 8, budget chosen on validation", "LoRA with a tuned budget"),
    "full": ("Full fine-tune, 119M weights", "a full fine-tune"),
}


def half_up(value: float, decimals: int = 1) -> str:
    """``value`` rounded half up, unsigned: the rounding the articles use everywhere."""
    snapped = Decimal(repr(round(abs(float(value)), 10)))
    return str(snapped.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def pct(fraction: float, decimals: int = 1) -> str:
    """A fraction as an unsigned percentage: 0.6386 -> ``63.9%``."""
    return f"{half_up(100 * fraction, decimals)}%"


def signed(fraction: float, decimals: int = 1) -> str:
    """A fraction as a signed percentage without the unit: 0.018 -> ``+1.8``."""
    return f"{MINUS if fraction < 0 else '+'}{half_up(100 * fraction, decimals)}"


def interval(row: pd.Series, prefix: str = "") -> str:
    """``+1.8% [+0.5, +3.1]`` from a row with ``skill``, ``ci_low`` and ``ci_high``.

    ``prefix="total_"`` reads the same three columns of the total-error skill instead.
    """
    low, high = signed(row[f"{prefix}ci_low"]), signed(row[f"{prefix}ci_high"])
    return f"{signed(row[f'{prefix}skill'])}% [{low}, {high}]"


def two_readings(paths: Paths) -> pd.DataFrame:
    """Every comparison of :data:`COMPARISONS`, on the typical window and on the total error.

    Both readings come from the same saved errors and the same bootstrap resamples. A unit
    (a window, or a store) is scored when every model of its file has a positive error on it,
    which is the rule notebook 10 applies to stores that were closed for the whole test.
    """
    rows = []
    for comparison in COMPARISONS:
        errors = load_results(comparison.metrics, paths)
        if comparison.more is not None:
            errors = pd.concat([errors, load_results(comparison.more, paths)], ignore_index=True)
        positive = errors[comparison.metric] > 0
        scored = positive.groupby(errors[comparison.unit]).transform("all")
        pair = errors[scored & errors[MODEL_COL].isin([comparison.model, comparison.baseline])]
        readings = {
            aggregate: skill_score(
                pair,
                model=comparison.model,
                baseline=comparison.baseline,
                metric=comparison.metric,
                block_length=comparison.block_length,
                unit=comparison.unit,
                aggregate=aggregate,
            )
            for aggregate in ("geometric", "total")
        }
        typical, total = readings["geometric"], readings["total"]
        rows.append(
            {
                "part": comparison.part,
                "metrics": comparison.metrics,
                "model": comparison.model,
                "baseline": comparison.baseline,
                "metric": comparison.metric,
                "unit": comparison.unit,
                "n_units": typical.n_units,
                "block_length": comparison.block_length,
                "win_rate": typical.win_rate,
                "skill": typical.skill,
                "ci_low": typical.ci_low,
                "ci_high": typical.ci_high,
                "total_skill": total.skill,
                "total_ci_low": total.ci_low,
                "total_ci_high": total.ci_high,
            }
        )
    return pd.DataFrame(rows)


class Measurements:
    """The handful of numbers the table and the figure share, each read from its results file."""

    def __init__(self, paths: Paths) -> None:
        self.paths = paths

    def table(self, name: str, index: str | None = None) -> pd.DataFrame:
        """``results/<name>.csv``, indexed by the column ``index`` when one is given."""
        frame = load_results(name, self.paths)
        return frame.set_index(index) if index else frame

    def total(self, metrics: str, model: str, baseline: str) -> str:
        """The total-error skill of one comparison of ``12_two_readings.csv``, as printed."""
        readings = self.table("12_two_readings")
        row = readings[
            (readings["metrics"] == metrics)
            & (readings["model"] == model)
            & (readings["baseline"] == baseline)
        ]
        if len(row) != 1:
            raise KeyError(f"12_two_readings has no single row for {model!r} vs {baseline!r}.")
        return interval(row.iloc[0], prefix="total_")

    def skill(self, name: str, model: str, comparison: str | None = None) -> pd.Series:
        """The row of ``model`` in a skill table (of one ``comparison`` when the file has many)."""
        frame = self.table(name)
        if comparison is not None:
            frame = frame[frame["comparison"] == comparison]
        return frame.set_index("model").loc[model]

    def messy(self) -> pd.Series:
        """Mean change in WQL, in percent, of each way of handling missing hours."""
        return self.table("00_messy_data_summary", "scenario")["WQL_change_pct"] / 100

    def context(self) -> pd.Series:
        """Day-ahead skill over the seasonal naive by context length."""
        sweep = self.table("03_context_sweep")
        return sweep[sweep["horizon"] == 24].set_index("context")["skill"]

    def speed(self, device: str, model: str = "amazon/chronos-2") -> float:
        """Best measured series per second at the protocol's context, over batch sizes."""
        bench = self.table(f"bench_latency_{device}")
        rows = bench[(bench["model"] == model) & (bench["context_length"] == CONTEXT)]
        return float(rows["series_per_second"].max())

    def cheapest_million(self) -> pd.Series:
        """The cheapest always-on instance per million forecasts at the protocol's context."""
        cost = self.table("09_cost_per_million")
        cost = cost[(cost["model"] == "amazon/chronos-2") & (cost["context_length"] == CONTEXT)]
        return cost.loc[cost["usd_per_million_forecasts"].idxmin()]

    def coverage(self) -> pd.Series:
        """Yearly coverage of the 80% interval, per calibration method."""
        return self.table("09_conformal_summary", "model")["coverage_80"]

    def worst_month(self) -> pd.Series:
        """Coverage of the 80% interval in each method's worst month."""
        return self.table("09_conformal_by_month").groupby("model")["coverage"].min()

    def anomaly(self, anomaly: str) -> pd.Series:
        """AUC of each representation at flagging one kind of anomaly."""
        scores = self.table("11_anomaly_auc")
        return scores[scores["anomaly"] == anomaly].set_index("representation")["auc"]


def summary_rows(m: Measurements) -> pd.DataFrame:
    """One row per lever: the part, the lever, its effect as printed, and how it was measured."""
    messy, context = m.messy(), m.context()
    zero_shot = m.skill("02_skill_table", "chronos-2")
    coverage, worst = m.coverage(), m.worst_month()
    frozen = m.anomaly("last 12 hours frozen")
    small = m.skill("bench_variants_skill", "chronos-2-small")
    small_speed = m.speed("cuda", "autogluon/chronos-2-small") / m.speed("cuda")
    rows = [
        (
            1,
            "Missing hours left as NaN on the regular grid, not dropped",
            f"WQL {signed(messy['rows_dropped_freq'], 0)}% if dropped, "
            f"{signed(messy['gaps_nan'], 0)}% as NaN",
            NOT_A_COMPARISON,
            "WQL against the clean series, Spanish load, 40 windows",
        ),
        (
            2,
            "Zero-shot, nothing tuned",
            interval(zero_shot),
            m.total("02_backtest_metrics", "chronos-2", "seasonal_naive"),
            "WQL against the seasonal naive, Spanish load",
        ),
        (
            2,
            "Context 64 → 2,048 hours",
            f"{pct(context[64])} → {pct(context[CONTEXT])}",
            NOT_A_COMPARISON,
            "skill over the seasonal naive, Spanish load",
        ),
        (
            2,
            "Chronos-2 instead of Chronos-Bolt",
            interval(m.skill("02_head_to_head", "chronos-2")),
            m.total("02_backtest_metrics", "chronos-2", "chronos-bolt"),
            "WQL against Chronos-Bolt, Spanish load",
        ),
        (
            3,
            "Covariates the market knows in advance",
            interval(m.skill("05_epf_de_skill", "known_future")),
            m.total("05_epf_de_metrics", "known_future", "univariate"),
            "WQL against no covariates, EPF-DE price",
        ),
        (
            3,
            "The operator's forecasts as covariates",
            interval(m.skill("05_price_ladder_skill", "TSO forecasts, no calendar")),
            m.total("05_price_ladder_metrics", "TSO forecasts, no calendar", "none"),
            "WQL against no covariates, Spanish price",
        ),
        (
            3,
            "Forecasting the operator's error instead of the load",
            interval(m.skill("05_operator_skill", "operator forecast, corrected")),
            m.total("05_operator_metrics", "operator forecast, corrected", "operator forecast"),
            "WQL against the operator's forecast, Spanish load",
        ),
    ]
    tuned = m.table("08_comparison_skill", "model")
    rows += [
        (
            4,
            label,
            interval(tuned.loc[model]),
            m.total("08_comparison_metrics", model, "zero-shot"),
            "WQL against zero-shot, Spanish load",
        )
        for model, (label, _) in FINE_TUNES.items()
        if model in tuned.index
    ]
    summed, direct = "hourly quantiles, summed", "daily series, forecast directly"
    rows += [
        (
            5,
            "A GPU instead of 8 CPU cores",
            f"{half_up(m.speed('cuda') / m.speed('cpu'))}{TIMES} the throughput",
            NOT_A_COMPARISON,
            "series per second, same model and context",
        ),
        (
            5,
            "`chronos-2-small` instead of `chronos-2`",
            f"{half_up(small_speed)}{TIMES} the throughput, {interval(small)}",
            m.total("bench_variants_metrics", "chronos-2-small", "chronos-2"),
            "WQL against the base model, Spanish load",
        ),
        (
            6,
            "Conformal correction, learned once",
            f"{pct(coverage['zero-shot'])} → {pct(coverage['conformal_additive'])}",
            NOT_A_COMPARISON,
            "coverage of the 80% interval over the year",
        ),
        (
            6,
            "Conformal correction, updated with every forecast",
            f"{pct(worst['zero-shot'])} → {pct(worst['conformal_aci'])}",
            NOT_A_COMPARISON,
            "coverage of the 80% interval in the worst month",
        ),
        (
            7,
            "Summing the hourly forecast instead of forecasting the daily total",
            interval(m.skill("10_paired_skill", summed, "tomorrow's total")),
            m.total("10_daily_total_metrics", summed, direct),
            "WQL of tomorrow's total against the daily series",
        ),
        (
            7,
            "Opening days, promotions and holidays known in advance, 1,115 stores",
            interval(m.skill("10_paired_skill", "+known covariates", "panel")),
            m.total("10_panel_store_errors", "+known covariates", "history only"),
            "WQL against history only, stores as units",
        ),
        (
            7,
            "Embeddings instead of raw values to flag a frozen feed",
            f"AUC {half_up(frozen['raw window'], 2)} → "
            f"{half_up(frozen['embedding (forecast)'], 2)}",
            NOT_A_COMPARISON,
            "last 12 hours repeated, 365 windows",
        ),
    ]
    table = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    table["part"] = "Part " + table["part"].astype(str)
    return table


def measured(m: Measurements) -> dict[str, str]:
    """The sentence under each step of the decision path, from the same files as the table."""
    messy, context = m.messy(), m.context()
    skill = m.table("02_skill_table", "model")
    ladder = m.table("05_price_ladder_skill", "model")["skill"]
    corrected = m.skill("05_operator_skill", "operator forecast, corrected")
    tuned = m.table("08_comparison_skill", "model")
    coverage, worst = m.coverage(), m.worst_month()
    cheapest = m.cheapest_million()
    rolling = m.table("09_monitoring_rolling")
    breaches, episodes = int(rolling["alert"].sum()), int(rolling["first_breach"].sum())
    speedup = half_up(m.speed("cuda") / m.speed("cpu"))
    return {
        "data": (
            f"on Spanish load, dropping the rows of the missing hours cost "
            f"{signed(messy['rows_dropped_freq'], 0)}% WQL; leaving them as NaN, "
            f"{signed(messy['gaps_nan'], 0)}%"
        ),
        "baseline": (
            f"zero-shot Chronos-2's error was {pct(skill.loc['chronos-2', 'skill'], 0)} below "
            f"the seasonal naive's on a typical day and "
            f"{pct(skill.loc['chronos-2', 'total_skill'], 0)} below over the year; the grid "
            f"operator's own forecast, {pct(skill.loc['operator_forecast', 'total_skill'], 0)} "
            f"below"
        ),
        "covariates": (
            f"on price, the operator's forecasts {signed(ladder['TSO forecasts, no calendar'])}% "
            f"and the calendar {signed(ladder['+calendar'])}%; on load, correcting the "
            f"operator's forecast {signed(corrected['skill'])}% over it"
        ),
        "context": (
            f"skill {pct(context[64], 0)} at 64 hours, {pct(context[1024], 0)} at 1,024, "
            f"{pct(context[CONTEXT], 0)} at 2,048 and {pct(context[8192], 0)} at 8,192"
        ),
        "finetune": "; ".join(
            f"{words} {interval(tuned.loc[model])}"
            for model, (_, words) in FINE_TUNES.items()
            if model in tuned.index
        ),
        "calibration": (
            f"the 80% interval covered {pct(coverage['zero-shot'])} zero-shot and "
            f"{pct(coverage['conformal_additive'])} corrected once; its worst month went from "
            f"{pct(worst['zero-shot'])} to {pct(worst['conformal_aci'])} when updated daily"
        ),
        "serving": (
            f"a GPU gave {speedup}{TIMES} the throughput of 8 CPU cores; from "
            f"${half_up(cheapest['usd_per_million_forecasts'], 2)} per million forecasts on an "
            f"always-on {cheapest['instance']}"
        ),
        "monitoring": (
            f"replaying 2018, {breaches} of {len(rolling)} cutoffs crossed a threshold, in "
            f"{episodes} episodes: {episodes} alerts to send, not {breaches}"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write the summary table, the decision path and its figure."""
    del argv  # no options: the inputs are the results folder
    paths = get_paths()
    print("Saved", save_results(two_readings(paths), "12_two_readings", paths).name)
    m = Measurements(paths)
    table = summary_rows(m)
    steps = measured(m)
    print("Saved", save_results(table, "12_series_summary", paths).name)
    path = pd.DataFrame({"step": list(steps), "measured": list(steps.values())})
    print("Saved", save_results(path, "12_decision_path", paths).name)
    (figure,) = save_figure(plot_decision_path(steps), "12_decision_path", paths.figures)
    print("Saved", figure.name)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
