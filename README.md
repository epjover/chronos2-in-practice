# Chronos-2 in Practice

[![CI](https://github.com/YOUR_GITHUB_USER/chronos2-in-practice/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USER/chronos2-in-practice/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](pyproject.toml)

Companion code for the Medium series *Chronos-2 in Practice*.
[Chronos-2](https://huggingface.co/amazon/chronos-2) is Amazon's open-weight (Apache-2.0) time
series foundation model: a single ~120M-parameter model that returns probabilistic forecasts for
univariate, multivariate and covariate-informed series without training on your data. The series
follows one real dataset (hourly Spanish electricity demand and prices, 2015-2018) from a first
forecast to production: data checks, evaluation against baselines, covariates, fine-tuning,
deployment and monitoring. Every experiment in the articles can be rerun from this repository.

## The series

| # | Article | Code |
| --- | --- | --- |
| 1 | How It Works, Messy Data and Your First Forecast (coming soon) | [`00_data_checks`](notebooks/00_data_checks.ipynb), [`01_quickstart`](notebooks/01_quickstart.ipynb) |
| 2 | Evaluation, Context Length, Multivariate and Cold Start (coming soon) | [`02_evaluation`](notebooks/02_evaluation.ipynb), [`03_context_length`](notebooks/03_context_length.ipynb), [`04_multivariate_coldstart`](notebooks/04_multivariate_coldstart.ipynb) |
| 3 | Covariates and What Drives the Forecast (coming soon) | [`05_covariates`](notebooks/05_covariates.ipynb), [`06_explainability`](notebooks/06_explainability.ipynb) |
| 4 | LoRA vs Full Fine-Tuning (coming soon) | [`07_finetune_lora`](notebooks/07_finetune_lora.ipynb), [`08_finetune_full`](notebooks/08_finetune_full.ipynb) (GPU) |
| 5 | Deployment, Performance and Monitoring (coming soon) | [`bench/`](bench/), [`deploy/`](deploy/), [`monitoring/`](monitoring/), [`scripts/cost_table.py`](scripts/cost_table.py) |

Everything runs on a CPU except the fine-tuning notebooks `07` and `08`, which need a GPU (local
or Colab).

## Quickstart (uv)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repository, create the environment (uv installs Python 3.11 if needed) and start
   Jupyter:

   ```bash
   git clone https://github.com/YOUR_GITHUB_USER/chronos2-in-practice.git chronos2-in-practice
   cd chronos2-in-practice
   uv sync
   uv run jupyter lab
   ```

3. Open `notebooks/00_data_checks.ipynb`. The dataset and the model are downloaded on first use;
   `uv run python data/download.py` fetches the data ahead of time.

`uv.lock` pins every package, so `uv sync` reproduces the exact environment used for the articles.

**Choosing the PyTorch build.** Plain `uv sync` installs PyPI's PyTorch build: CPU-only on Windows
and macOS, CUDA-enabled on Linux (several GB). Two extras change that:

| Your machine | Command |
| --- | --- |
| Windows or Linux with an NVIDIA GPU | `uv sync --extra cu130` |
| Linux without a GPU (smaller download) | `uv sync --extra cpu` |

The CUDA 13.0 wheels need an NVIDIA driver 580 or newer and a Turing or newer GPU (RTX 20xx, T4
or later; check with `nvidia-smi`). Because `uv run` re-syncs the environment before each
command, keep passing the extra you installed with (`uv run --extra cu130 jupyter lab`), or use
`uv run --no-sync ...`. `get_device()` picks the CPU whenever PyTorch cannot see a GPU; set
`C2P_DEVICE=cuda` to get an error instead.

## Your own data

The notebooks start from the Spanish electricity dataset. `examples/forecast_your_series.py` is
the same pipeline pointed at a CSV of yours: the data checklist of part 1, a zero-shot forecast at
the quantiles you ask for, a fan chart, and (optionally) the rolling-origin backtest of part 2
against a seasonal naive baseline, with the skill score and its bootstrap interval:

```bash
uv run python examples/forecast_your_series.py data.csv \
    --id-column store --timestamp-column date --target sales --freq D --horizon 14 --backtest 8
```

Omit `--id-column` for a single series. `--known col1 col2` passes known-future covariates
(part 3; append the horizon rows with the target left empty), `--source-tz Europe/Madrid`
converts naive local timestamps, `--sentinels 0 -999` turns placeholder values into gaps, and
`--model path/to/checkpoint` runs a fine-tuned model from part 4. Outputs land next to the CSV
(or in `--out`): `<name>_forecast.csv`, `<name>_fan.png` and `<name>_backtest.csv`.

## Without uv

The project needs Python 3.11, 3.12 or 3.13 (the articles use 3.11); pip refuses newer versions.

```bash
python -m venv .venv          # Windows, to pick the version: py -3.11 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .              # keeps an installed PyTorch, else installs it from PyPI
pip install jupyterlab        # notebooks; add pytest and nbmake to run the tests
```

pip resolves the version ranges in `pyproject.toml` (with `chronos-forecasting==2.3.2` pinned),
not the exact versions in `uv.lock`. There is no `requirements.txt` on purpose: the locked CPU
build of PyTorch (`torch==2.14.1+cpu`) is only published on the PyTorch package index, so a pinned
export breaks a plain `pip install -r` on Linux and Colab.

## Google Colab

| Notebook | |
| --- | --- |
| `00_data_checks` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USER/chronos2-in-practice/blob/main/notebooks/00_data_checks.ipynb) |
| `01_quickstart` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USER/chronos2-in-practice/blob/main/notebooks/01_quickstart.ipynb) |
| `02_evaluation` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USER/chronos2-in-practice/blob/main/notebooks/02_evaluation.ipynb) |
| `03_context_length` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USER/chronos2-in-practice/blob/main/notebooks/03_context_length.ipynb) |
| `04_multivariate_coldstart` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USER/chronos2-in-practice/blob/main/notebooks/04_multivariate_coldstart.ipynb) |
| `05_covariates` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USER/chronos2-in-practice/blob/main/notebooks/05_covariates.ipynb) |
| `06_explainability` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USER/chronos2-in-practice/blob/main/notebooks/06_explainability.ipynb) |
| `07_finetune_lora` (GPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USER/chronos2-in-practice/blob/main/notebooks/07_finetune_lora.ipynb) |
| `08_finetune_full` (GPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USER/chronos2-in-practice/blob/main/notebooks/08_finetune_full.ipynb) |

The notebooks import helpers from this repository, so on Colab run this first (it keeps Colab's
own PyTorch build):

```python
!git clone --depth 1 https://github.com/YOUR_GITHUB_USER/chronos2-in-practice.git chronos2-in-practice
%cd chronos2-in-practice
!pip install -q .
```

For `07` and `08`, pick a GPU runtime first (Runtime > Change runtime type > T4 GPU).

## Repository layout

```text
.
├── data/
│   ├── download.py         # downloads the dataset (data files are not committed)
│   └── reference/          # cited reference numbers, e.g. benchmark results from the paper
├── src/c2p/                # reusable code imported by the notebooks
│   ├── config.py           # paths, fast mode, seeds, device selection
│   ├── schema.py           # column names, quantile levels, Forecaster protocol
│   ├── data.py             # loading and the data checklist (validate_and_prepare)
│   ├── eval.py             # rolling-origin backtests, metrics, skill scores
│   ├── models.py           # Chronos-2 forecaster
│   ├── baselines.py        # Seasonal Naive, AutoETS, Chronos-Bolt
│   ├── finetune.py         # splits, LoRA and full fine-tuning
│   ├── fev_tasks.py        # published fev-bench tasks, translated to our protocol
│   ├── monitoring.py       # prediction log, delayed actuals, rolling metrics, alerts
│   ├── conformal.py        # conformalized quantile regression: honest intervals after the fact
│   ├── plotting.py         # article figures
│   ├── plots/              # figures specific to one article (architecture, evaluation)
│   └── results.py          # saving and loading experiment CSV files
├── notebooks/              # 00_data_checks ... 08_finetune_full
├── examples/
│   └── forecast_your_series.py  # checklist + zero-shot forecast + backtest on your CSV
├── bench/
│   ├── bench_cpu_gpu.py    # latency and throughput vs batch, context, checkpoint, device
│   └── small_vs_base_accuracy.py  # chronos-2-small vs chronos-2 on the day-ahead protocol
├── deploy/
│   ├── fastapi/            # the container you run: app.py, Dockerfile, smoke_test.py
│   ├── sagemaker/          # a managed endpoint you own: inference.py + lifecycle script
│   └── autogluon_cloud/    # batch, real-time or serverless, with no handler to write
├── monitoring/
│   ├── build_panel.py      # replays 2018 as production and builds the alert panel
│   └── calibrate_intervals.py  # calibrates the bands on 2017-H2, re-scores 2018
├── results/                # one CSV per experiment + figures/
├── scripts/
│   ├── set_repo_url.py     # sets the GitHub owner/name used in links
│   ├── fetch_cloud_prices.py  # AWS Price List API -> data/reference/cloud_prices.csv
│   └── cost_table.py       # joins measured throughput with published rates
├── tests/                  # pytest suite
├── pyproject.toml
└── uv.lock
```

## How results are traced

Each experiment writes `results/<notebook>_<experiment>.csv`, and every figure in
`results/figures/` is drawn from those CSV files or, for numbers cited from the Chronos-2 paper,
from `data/reference/` (copied from the paper, with the tables named in the `source` column). No
figure uses numbers typed into a notebook. To check a number in an article, open the CSV it comes
from; to regenerate it, rerun the notebook.

Cloud prices follow the same rule. `scripts/fetch_cloud_prices.py` reads the AWS Price List API
(public, no credentials, no resources created) and writes `data/reference/cloud_prices.csv`
with the SKU, the effective date and the offer-file version beside every rate, so a stale table
is visible instead of silently wrong. `scripts/cost_table.py` joins those rates with the
throughput measured by `bench/bench_cpu_gpu.py`; it never claims local throughput transfers to a
cloud instance, and reports the rate-only break-even ratio separately for that reason.

The one figure that shows no measurement is the architecture diagram of article 1
(`c2p.plots.architecture`): it describes the model. Even there the sizes it prints are read from
the loaded checkpoint with `chronos2_sizes(pipeline)`, so the drawing cannot disagree with the
weights, and a slow test asserts that `amazon/chronos-2` still matches the published defaults.

## Fast mode

Set `C2P_FAST_MODE=1` to run any notebook in a few minutes on a CPU (fewer windows, fewer series,
shorter contexts). Fast-mode results go to `results/fast/`, which is git-ignored, so they never
overwrite the published ones. `C2P_DEVICE=cpu` or `C2P_DEVICE=cuda` forces the device.

```bash
C2P_FAST_MODE=1 uv run jupyter lab              # bash
$env:C2P_FAST_MODE = "1"; uv run jupyter lab    # PowerShell
```

Inside a notebook, `FAST_MODE = is_fast_mode(default=False)` reads the variable; change the
default to `True` to make it stick.

## Tests

```bash
uv run pytest                      # fast tests, no network
uv run pytest -m slow              # downloads Chronos-2 and the dataset
uv run ruff check . && uv run ruff format --check .
# The notebooks in fast mode on CPU, without the GPU notebooks 07 and 08:
C2P_FAST_MODE=1 C2P_DEVICE=cpu uv run pytest --nbmake notebooks/ \
  --ignore-glob='notebooks/07_*' --ignore-glob='notebooks/08_*' \
  --nbmake-kernel=python3 --nbmake-timeout=900
```

CI (`.github/workflows/ci.yml`) runs all of the above except the slow tests on every push and
pull request to `main`. `--nbmake-kernel=python3` runs every notebook in the project environment,
whatever kernel it was saved with; `--nbmake-timeout` is per cell, in seconds.

## Deployment and cost

Three paths, in increasing order of how much AWS does for you, each with its own README:

| Path | What you manage | Teardown |
| --- | --- | --- |
| [`deploy/fastapi`](deploy/fastapi/) | A container you run anywhere | `docker rm -f` |
| [`deploy/sagemaker`](deploy/sagemaker/) | A handler and an endpoint in your account | `deploy_endpoint.py --delete --yes` |
| [`deploy/autogluon_cloud`](deploy/autogluon_cloud/) | Nothing; batch, real-time or serverless | `deploy_timeseries.py --delete --yes` |

The two cloud scripts create nothing unless you pass `--yes`: `--plan` is the default and prints
exactly what would be created and what would keep billing. `boto3`, `sagemaker` and
`autogluon.cloud` are deliberately not dependencies of this project; add them with
`uv add --group cloud ...` when you want them.

## Data

[Hourly energy demand generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)
(Kaggle, `nicholasjhana/energy-consumption-generation-prices-and-weather`, version 1): hourly
electricity load, generation, prices and the grid operator's day-ahead forecasts for Spain,
plus weather for five Spanish cities, 2015-2018. License: CC0 (public domain). The files are
downloaded with `kagglehub` (no Kaggle account needed), checked against SHA-256 hashes and stored
in `data/raw/`, which is git-ignored: the data is never committed to this repository.

## Versions

The code targets `chronos-forecasting==2.3.2` (pinned in `pyproject.toml`) on Python 3.11, with
every other package locked in `uv.lock`. The Chronos API has changed between minor releases, so
upgrade deliberately and rerun the tests.

## License

The code in this repository is released under the [Apache License 2.0](LICENSE). The Chronos-2
model weights ([amazon/chronos-2](https://huggingface.co/amazon/chronos-2)) and the
[chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) library are also
Apache-2.0. The dataset is CC0 (see [Data](#data)).

## Citation

If you use Chronos-2, cite the paper
([arXiv:2510.15821](https://arxiv.org/abs/2510.15821)):

```bibtex
@article{ansari2025chronos2,
  title   = {Chronos-2: From Univariate to Universal Forecasting},
  author  = {Abdul Fatir Ansari and Oleksandr Shchur and Jaris Küken and Andreas Auer and Boran Han and Pedro Mercado and Syama Sundar Rangapuram and Huibin Shen and Lorenzo Stella and Xiyuan Zhang and Mononito Goswami and Shubham Kapoor and Danielle C. Maddix and Pablo Guerron and Tony Hu and Junming Yin and Nick Erickson and Prateek Mutalik Desai and Hao Wang and Huzefa Rangwala and George Karypis and Yuyang Wang and Michael Bohlke-Schneider},
  journal = {arXiv preprint arXiv:2510.15821},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.15821}
}
```
