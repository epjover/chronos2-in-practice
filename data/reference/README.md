# Reference data

Small, versioned tables used by the notebooks and figures. Unlike `data/raw/`, these files are
committed to the repository.

| File | Contents | Source |
| --- | --- | --- |
| `paper_benchmarks.csv` | Published skill scores of Chronos-2 and other pretrained forecasting models on three benchmarks. Long format: `model, benchmark, skill_score_pct, source`. | Ansari et al. (2025), *Chronos-2: From Univariate to Universal Forecasting*, technical report ([arXiv:2510.15821](https://arxiv.org/abs/2510.15821)), Tables 3-5. |

## `paper_benchmarks.csv`

- **Metric**: skill score in percent relative to the Seasonal Naive baseline,
  `100 * (1 - geometric mean of model error / Seasonal Naive error)` over the benchmark's
  tasks. Higher is better; 0 means "as good as Seasonal Naive".
- **Error used**: the scaled quantile loss (SQL) for `fev-bench` (Table 3); the weighted
  quantile loss (WQL) for `GIFT-Eval` and `Chronos Benchmark II` (panel (a) of Tables 4 and
  5). The benchmarks therefore do not all use the same error, and none of them is directly
  comparable with a skill score that `c2p.eval` computes on another dataset.
- Values are copied from the paper and not recomputed here. Plot them with
  `c2p.plotting.plot_benchmarks(pd.read_csv(PATHS.data_reference / "paper_benchmarks.csv"))`.
