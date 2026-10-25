"""Smoke test and latency benchmark for the Chronos-2 forecasting service.

With the service running (see ``deploy/fastapi/README.md``), from the repository root::

    uv run --group deploy python deploy/fastapi/smoke_test.py
    uv run --group deploy python deploy/fastapi/smoke_test.py --wait 300 --bench 1 100

The smoke test sends one week of hourly synthetic data with a gap (six missing hours and one
``null`` value), checks the response and prints the forecast quantiles and the latency.
``--wait`` first polls ``/health`` until the model is loaded and reports how long that took;
``--bench`` measures the median (p50) and p95 latency of requests with N series each.
Only ``httpx`` and the standard library are needed. The exit code is 1 when a check fails.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

START = datetime(2026, 1, 5, tzinfo=UTC)  # a Monday
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
GAP_HOURS = range(80, 86)  # rows left out of the request: six missing timestamps
NULL_HOUR = 120  # a timestamp sent with a null value


class SmokeTestError(RuntimeError):
    """The service answered, but not as expected."""


def synthetic_series(item_id: str, *, hours: int = 168, seed: int = 0) -> dict[str, Any]:
    """One week of hourly load-like data (daily and weekly cycles plus noise) with a gap."""
    rng = random.Random(seed)
    level = 1_000.0 * (1 + seed % 7)
    timestamps, values = [], []
    for hour in range(hours):
        if hour in GAP_HOURS:
            continue
        daily = 0.2 * math.sin(2 * math.pi * (hour - 8) / 24)
        weekend = -0.15 if (hour // 24) % 7 >= 5 else 0.0
        value = level * (1 + daily + weekend + rng.gauss(0, 0.02))
        timestamps.append((START + timedelta(hours=hour)).isoformat())
        values.append(None if hour == NULL_HOUR else round(value, 3))
    return {"item_id": item_id, "timestamps": timestamps, "values": values}


def forecast_payload(n_series: int, *, prediction_length: int) -> dict[str, Any]:
    return {
        "series": [synthetic_series(f"series-{i:04d}", seed=i) for i in range(n_series)],
        "prediction_length": prediction_length,
        "freq": "h",
        "quantile_levels": list(QUANTILE_LEVELS),
    }


def wait_until_healthy(client: httpx.Client, timeout: float) -> float:
    """Poll ``/health`` until it answers 200; return the seconds waited."""
    started = time.perf_counter()
    while (waited := time.perf_counter() - started) < timeout:
        try:
            if client.get("/health", timeout=2.0).status_code == 200:
                return waited
        except httpx.TransportError:
            pass  # the server is not listening yet
        time.sleep(0.25)
    raise SmokeTestError(f"/health did not answer 200 within {timeout:.0f} s.")


def post_forecast(client: httpx.Client, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """POST /forecast; return the response body and the client-side latency in ms."""
    started = time.perf_counter()
    response = client.post("/forecast", json=payload)
    latency_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        raise SmokeTestError(f"/forecast returned {response.status_code}: {response.text[:500]}")
    return response.json(), latency_ms


def check_forecast(body: dict[str, Any], payload: dict[str, Any]) -> None:
    """Raise ``SmokeTestError`` unless the response matches the request."""
    horizon = payload["prediction_length"]
    expected_ids = [series["item_id"] for series in payload["series"]]
    problems = []
    if [entry["item_id"] for entry in body["forecasts"]] != expected_ids:
        problems.append("series are missing or out of order")
    if body["input_summary"]["n_missing_timestamps"] != len(GAP_HOURS) * len(expected_ids):
        problems.append(f"expected {len(GAP_HOURS)} inserted timestamps per series")
    for entry in body["forecasts"]:
        quantiles = [entry["quantiles"][str(level)] for level in QUANTILE_LEVELS]
        if any(len(values) != horizon for values in [entry["predictions"], *quantiles]):
            problems.append(f"{entry['item_id']}: forecast length is not {horizon}")
        if not _increasing(quantiles):
            problems.append(f"{entry['item_id']}: quantiles are not increasing")
    if problems:
        raise SmokeTestError("; ".join(problems[:5]))


def _increasing(quantiles: Sequence[Sequence[float]]) -> bool:
    """True if every quantile path lies on or above the previous (lower) level's path."""
    return all(
        low <= high
        for lower, upper in itertools.pairwise(quantiles)
        for low, high in zip(lower, upper, strict=True)
    )


def print_forecast(body: dict[str, Any], client_ms: float, *, steps: int = 6) -> None:
    entry = body["forecasts"][0]
    print(f"model {body['model_id']}  input_hash {body['input_hash'][:16]}...")
    print(f"input summary: {body['input_summary']}")
    header = "timestamp (UTC)".ljust(22) + "".join(f"q{level:<10}" for level in QUANTILE_LEVELS)
    print(f"\nfirst {steps} steps of {entry['item_id']}:\n{header}")
    for i, timestamp in enumerate(entry["timestamps"][:steps]):
        row = "".join(f"{entry['quantiles'][str(level)][i]:<11.1f}" for level in QUANTILE_LEVELS)
        print(f"{timestamp:<22}{row}")
    print(f"\nlatency: server {body['latency_ms']:.1f} ms, client round trip {client_ms:.1f} ms")


@dataclass(frozen=True)
class BenchResult:
    """Latency of ``repeats`` identical requests with ``n_series`` series each."""

    n_series: int
    repeats: int
    client_ms: list[float]
    server_ms: list[float]

    def __str__(self) -> str:
        client, server = _p50_p95(self.client_ms), _p50_p95(self.server_ms)
        return (
            f"{self.n_series:>5} series x {self.repeats} requests: "
            f"client p50 {client[0]:.1f} ms (p95 {client[1]:.1f}), "
            f"server p50 {server[0]:.1f} ms (p95 {server[1]:.1f})"
        )


def benchmark(
    client: httpx.Client, n_series: int, *, repeats: int, prediction_length: int
) -> BenchResult:
    """Time ``repeats`` requests (client round trip and server-side) after one warm-up."""
    payload = forecast_payload(n_series, prediction_length=prediction_length)
    body, _ = post_forecast(client, payload)  # warm-up, not timed
    check_forecast(body, payload)
    client_ms, server_ms = [], []
    for _ in range(repeats):
        body, latency_ms = post_forecast(client, payload)
        client_ms.append(latency_ms)
        server_ms.append(body["latency_ms"])
    return BenchResult(n_series, repeats, client_ms, server_ms)


def _p50_p95(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], values[0]
    return statistics.median(values), statistics.quantiles(values, n=20, method="inclusive")[-1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000", help="service base URL")
    parser.add_argument(
        "--wait", type=float, default=0.0, help="seconds to wait for /health first (0: no wait)"
    )
    parser.add_argument(
        "--bench", type=int, nargs="*", default=[], metavar="N", help="series per request to time"
    )
    parser.add_argument("--repeats", type=int, default=20, help="timed requests per size")
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=300.0, help="request timeout in seconds")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        try:
            if args.wait > 0:
                waited = wait_until_healthy(client, args.wait)
                print(f"healthy after {waited:.1f} s of polling")
            print(f"health: {client.get('/health').raise_for_status().json()}")
            payload = forecast_payload(1, prediction_length=args.prediction_length)
            body, client_ms = post_forecast(client, payload)
            check_forecast(body, payload)
            print_forecast(body, client_ms)
            for n_series in args.bench:
                result = benchmark(
                    client,
                    n_series,
                    repeats=args.repeats,
                    prediction_length=args.prediction_length,
                )
                print(f"bench {result}")
        except (SmokeTestError, httpx.HTTPError) as error:
            print(f"SMOKE TEST FAILED: {error}", file=sys.stderr)
            return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
