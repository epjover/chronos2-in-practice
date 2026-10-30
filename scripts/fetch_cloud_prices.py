"""Fetch SageMaker on-demand rates from the AWS Price List API into ``data/reference``.

Article 5 puts a cost per million forecasts next to a measured throughput, and a cost table is
only worth printing if its rates are checkable. Secondary sources for SageMaker pricing disagree
by up to 40% on the same instance, so this script reads the rates from the one place AWS
publishes them: the Price List Bulk API, which needs no credentials and stamps every offer file
with a version and a publication date.

The output ``data/reference/cloud_prices.csv`` carries, for every row, the price, the SKU, the
rate's effective date, the offer-file version and the URL it came from: enough for a reader to
re-derive the number a year later, or to see at a glance that it has gone stale.

Usage::

    uv run python scripts/fetch_cloud_prices.py                    # us-east-1
    uv run python scripts/fetch_cloud_prices.py --region eu-west-1

This reads public pricing metadata only. It creates nothing in an AWS account and costs nothing.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime

import pandas as pd

from c2p.config import get_paths

BASE_URL = "https://pricing.us-east-1.amazonaws.com"
SERVICE_CODE = "AmazonSageMaker"
REGION_INDEX = f"{BASE_URL}/offers/v1.0/aws/{SERVICE_CODE}/current/region_index.json"

# The meters article 5 compares: a real-time endpoint (what the FastAPI service maps onto), the
# batch-transform meter for the nightly-job alternative, and the training meter for fine-tuning.
COMPONENTS = ("Hosting", "BatchTransform", "Training", "Spot Training")

# CPU and GPU instances in the size range this model actually needs. Chronos-2 is 119.5M
# parameters: it fits in a few GB, so the interesting comparison is small CPU against small GPU,
# not the accelerator sizes an LLM would force on you. Inferentia instances are left out: they
# need a Neuron-compiled model, which this repository does not build or measure.
INSTANCES = (
    "ml.m5.large",
    "ml.m5.xlarge",
    "ml.m5.2xlarge",
    "ml.c7i.xlarge",
    "ml.c7i.2xlarge",
    "ml.c7i.4xlarge",
    "ml.g4dn.xlarge",
    "ml.g5.xlarge",
    "ml.g5.2xlarge",
    "ml.g6.xlarge",
)

# Serverless inference bills per GB-second instead of per instance-hour, so it is the only option
# whose cost really does fall to zero between requests.
SERVERLESS_USAGE_TYPES = ("ServerlessInf:Mem-1GB", "ServerlessInf:Mem-6GB")

TIMEOUT_S = 600


def fetch_json(url: str) -> dict:
    """GET ``url`` and parse the JSON body (the offer files are tens of megabytes)."""
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return json.load(response)


def region_offer_url(region: str) -> str:
    """Resolve the current offer-file URL of ``region`` from the SageMaker region index."""
    index = fetch_json(REGION_INDEX)
    regions = index["regions"]
    if region not in regions:
        raise SystemExit(f"{region} is not in the offer index. Available: {sorted(regions)[:8]}...")
    return BASE_URL + regions[region]["currentVersionUrl"]


def on_demand_rate(offer: dict, sku: str) -> tuple[float, str, str] | None:
    """The single on-demand price dimension of a SKU: price, unit and effective date."""
    terms = offer.get("terms", {}).get("OnDemand", {}).get(sku)
    if not terms:
        return None
    for term in terms.values():
        for dimension in term["priceDimensions"].values():
            return (
                float(dimension["pricePerUnit"]["USD"]),
                dimension["unit"],
                term["effectiveDate"],
            )
    return None


def extract(offer: dict, *, region: str, url: str) -> pd.DataFrame:
    """One row per (component, instance) meter we care about, plus the serverless meters."""
    version = offer["version"]
    published = offer["publicationDate"]
    checked_on = datetime.now(UTC).date().isoformat()
    rows = []
    for sku, product in offer["products"].items():
        attributes = product["attributes"]
        component = attributes.get("component")
        instance = attributes.get("instanceName")
        usage_type = str(attributes.get("usagetype", ""))
        serverless = any(usage_type.endswith(suffix) for suffix in SERVERLESS_USAGE_TYPES)
        if not serverless and (component not in COMPONENTS or instance not in INSTANCES):
            continue
        rate = on_demand_rate(offer, sku)
        if rate is None:
            continue
        price, unit, effective = rate
        rows.append(
            {
                "service": "SageMaker",
                "component": "ServerlessInference" if serverless else component,
                "instance": instance or usage_type.rsplit(":", maxsplit=1)[-1],
                "vcpu": attributes.get("vCpu", ""),
                "gpu": attributes.get("gpu", ""),
                "memory": attributes.get("memory", ""),
                "region": region,
                "price_usd": price,
                "unit": unit,
                "sku": sku,
                "effective_date": effective,
                "offer_version": version,
                "offer_published": published,
                "source": url,
                "checked_on": checked_on,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No matching SKUs: the offer file's attribute names may have changed.")
    return frame.sort_values(["component", "price_usd", "instance"], ignore_index=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Download the SageMaker offer file for one region and write the price reference CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--out", default="cloud_prices.csv", help="file under data/reference/")
    args = parser.parse_args(argv)

    url = region_offer_url(args.region)
    print(f"Reading {url}")
    prices = extract(fetch_json(url), region=args.region, url=url)

    path = get_paths().data_reference / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index=False)
    print(f"Saved {path} ({len(prices)} rates, offer version {prices['offer_version'].iloc[0]})")
    hosting = prices[prices["component"] == "Hosting"]
    print(hosting[["instance", "vcpu", "gpu", "price_usd", "unit"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
