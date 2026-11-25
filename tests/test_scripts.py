"""Smoke tests of the command-line scripts that the other test modules do not import.

Each script's ``main`` runs end to end in a temporary project folder. The scripts that need
measurements read the CSV files committed under ``results/`` (they are the published numbers,
so the tests also check that those files still have the columns the scripts expect); the ones
that need a model get a seasonal-naive stand-in and a synthetic series, so nothing is
downloaded.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from c2p.baselines import SeasonalNaive
from c2p.config import Paths, get_paths
from c2p.experiments import LOAD
from c2p.schema import ID_COL, TIMESTAMP_COL

REPO = Path(__file__).resolve().parents[1]


def load_script(relative: str) -> ModuleType:
    """Import a script by path, as ``python <path>`` would, without running its ``main``."""
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Paths:
    """An empty project folder that ``get_paths`` resolves to, fast mode or not."""
    monkeypatch.setenv("C2P_ROOT", str(tmp_path))
    paths = get_paths(root=tmp_path).ensure()
    paths.data_reference.mkdir(parents=True, exist_ok=True)
    return paths


def copy_results(paths: Paths, *names: str) -> None:
    for name in names:
        shutil.copy(REPO / "results" / f"{name}.csv", paths.results / f"{name}.csv")


def synthetic_load(days: int = 420) -> pd.DataFrame:
    """An hourly series ending on 31 December 2018 with a daily and a weekly cycle."""
    stamps = pd.date_range(end="2018-12-31 23:00", periods=days * 24, freq="h")
    hours = np.arange(len(stamps))
    values = (
        28_000
        + 4_000 * np.sin(2 * np.pi * hours / 24)
        + 1_500 * np.sin(2 * np.pi * hours / 168)
        + np.random.default_rng(0).normal(0, 300, len(stamps))
    )
    return pd.DataFrame({ID_COL: "spain", TIMESTAMP_COL: stamps, LOAD: values})


def use_stand_in_model(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the data and the model of a script with a synthetic series and seasonal naive."""
    monkeypatch.setattr(module, "spain_energy_frame", synthetic_load)
    monkeypatch.setattr(module, "load_chronos2", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        module,
        "Chronos2Forecaster",
        lambda pipeline, name="chronos-2": SeasonalNaive(season_length=24, name=name),
    )


# --- scripts that only read results -----------------------------------------------------------


def test_cost_table_joins_the_committed_benchmarks_with_the_price_reference(project):
    copy_results(project, "bench_latency_cuda", "bench_latency_cpu")
    shutil.copy(REPO / "data" / "reference" / "cloud_prices.csv", project.data_reference)
    script = load_script("scripts/cost_table.py")
    assert script.main([]) == 0
    costs = pd.read_csv(project.results / "09_cost_per_million.csv")
    assert len(costs) > 0 and costs.select_dtypes("number").notna().any().all()
    for name in ("09_hosting_rates", "09_break_even"):
        assert (project.results / f"{name}.csv").is_file()
    assert (project.figures / "09_cost_per_million.png").is_file()
    assert (project.figures / "09_throughput.png").is_file()


def test_cost_table_points_at_the_fetch_script_when_prices_are_missing(project):
    script = load_script("scripts/cost_table.py")
    with pytest.raises(SystemExit, match="fetch_cloud_prices"):
        script.load_prices(project)


def test_summarise_stability_writes_one_row_per_benchmark_cell(project):
    names = [
        f"bench_latency_{device}{suffix}" for device in ("cuda", "cpu") for suffix in ("", "_rerun")
    ]
    copy_results(project, *names)
    script = load_script("bench/summarise_stability.py")
    assert script.main([]) == 0
    table = pd.read_csv(project.results / "bench_stability.csv")
    runs = sum(
        len(pd.read_csv(project.results / f"bench_latency_{d}.csv")) for d in ("cuda", "cpu")
    )
    assert len(table) == runs
    assert table["rerun_vs_median"].notna().all()
    assert (project.figures / "09_latency_spread.png").is_file()


def test_calibrate_intervals_rebuilds_its_summary_from_saved_results(project):
    copy_results(project, "09_conformal_metrics", "09_conformal_by_month")
    script = load_script("monitoring/calibrate_intervals.py")
    assert script.main(["--from-results"]) == 0
    summary = pd.read_csv(project.results / "09_conformal_summary.csv").set_index("model")
    assert summary.loc["zero-shot", "skill"] == 0.0
    assert {"conformal_additive", "conformal_rolling", "conformal_aci"} <= set(summary.index)
    assert (project.figures / "09_conformal_coverage.png").is_file()


# --- the price fetcher, offline ---------------------------------------------------------------


def toy_offer() -> dict:
    def product(sku, **attributes):
        return sku, {"attributes": attributes}

    products = dict(
        [
            product("HOST", component="Hosting", instanceName="ml.g5.xlarge", vCpu="4", gpu="1"),
            product("OTHER", component="Hosting", instanceName="ml.not.listed"),
            product("SRVL", usagetype="USE1-ServerlessInf:Mem-6GB"),
            product("NOTERM", component="Hosting", instanceName="ml.c7i.2xlarge"),
        ]
    )

    def term(price, unit):
        dimension = {"pricePerUnit": {"USD": str(price)}, "unit": unit}
        return {"t": {"priceDimensions": {"d": dimension}, "effectiveDate": "2026-11-01"}}

    return {
        "version": "v1",
        "publicationDate": "2026-11-11",
        "products": products,
        "terms": {"OnDemand": {"HOST": term(1.408, "Hrs"), "SRVL": term(0.00012, "Second")}},
    }


def test_fetch_cloud_prices_keeps_the_listed_instances_and_the_serverless_meters():
    script = load_script("scripts/fetch_cloud_prices.py")
    prices = script.extract(toy_offer(), region="us-east-1", url="https://example.invalid/offer")
    assert prices["sku"].tolist() == ["HOST", "SRVL"]
    assert prices.set_index("sku").loc["HOST", "price_usd"] == pytest.approx(1.408)
    assert prices.set_index("sku").loc["SRVL", "component"] == "ServerlessInference"
    assert script.on_demand_rate(toy_offer(), "NOTERM") is None
    with pytest.raises(SystemExit, match="No matching SKUs"):
        script.extract({**toy_offer(), "products": {}}, region="us-east-1", url="u")


def test_fetch_cloud_prices_main_writes_the_reference_file(project, monkeypatch, capsys):
    script = load_script("scripts/fetch_cloud_prices.py")
    index = {"regions": {"us-east-1": {"currentVersionUrl": "/offers/v1/index.json"}}}
    monkeypatch.setattr(
        script, "fetch_json", lambda url: index if url == script.REGION_INDEX else toy_offer()
    )
    assert script.main(["--out", "prices.csv"]) == 0
    assert len(pd.read_csv(project.data_reference / "prices.csv")) == 2
    with pytest.raises(SystemExit, match="not in the offer index"):
        script.main(["--region", "nowhere-1"])


# --- scripts that run a model, with a stand-in ------------------------------------------------


def test_build_panel_replays_the_year_and_writes_the_panel(project, monkeypatch):
    script = load_script("monitoring/build_panel.py")
    use_stand_in_model(script, monkeypatch)
    assert script.main(["--fast"]) == 0
    log = pd.read_csv(project.results / "09_monitoring_log.csv")
    rolling = pd.read_csv(project.results / "09_monitoring_rolling.csv")
    assert log["cutoff"].nunique() == 12
    assert {"wape", "coverage"} <= set(rolling.columns)
    assert (project.results / "09_monitoring_drift.csv").is_file()
    assert (project.figures / "09_monitoring_panel.png").is_file()


def test_cheaper_variants_compares_against_the_full_model(project, monkeypatch):
    script = load_script("bench/cheaper_variants_accuracy.py")
    assert [name for name, _, _ in script.variants("cpu")] == ["chronos-2", "chronos-2-small"]
    assert ("chronos-2-bf16", script.CHRONOS2, "bfloat16") in script.variants("cuda")
    use_stand_in_model(script, monkeypatch)
    monkeypatch.setattr(script, "parameter_count", lambda pipeline: 1_000_000)
    monkeypatch.setattr(script, "get_device", lambda: "cpu")
    monkeypatch.setenv("C2P_FAST_MODE", "1")
    assert script.main() == 0
    skill = pd.read_csv(project.results / "fast" / "bench_variants_skill.csv").set_index("model")
    assert skill.loc["chronos-2-small", "skill"] == pytest.approx(0.0)  # same stand-in model
    assert skill.loc["chronos-2-small", "block_length"] == 7
