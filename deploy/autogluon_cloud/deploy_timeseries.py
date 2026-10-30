r"""Serve Chronos-2 through AutoGluon Cloud: batch, real-time, or serverless with scale to zero.

This is the path AWS recommends, and the only one of the three in this repository that can bill
you nothing while idle. ``TimeSeriesFoundationModel`` wraps a pretrained checkpoint (no
training job, no handler, no image) and exposes the three shapes a forecasting workload
actually takes:

- ``--mode batch``: one SageMaker job that reads a frame, writes forecasts to S3 and exits. The
  right shape for a nightly day-ahead run; you pay for the job's duration and nothing else.
- ``--mode realtime``: a dedicated endpoint, billed per hour until deleted. Lowest latency.
- ``--mode serverless``: a SageMaker Serverless Inference endpoint, billed per GB-second of
  request duration. It **scales to zero**, at the cost of a cold start and CPU-only execution.
  It requires a cached model artifact in S3, which ``cache_model_artifact`` writes, because the
  container cannot reach the Hugging Face Hub on a cold start.

Usage from the repository root::

    uv add --group cloud "autogluon.cloud>=0.7"
    uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --plan
    uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --yes --mode batch \\
        --s3-path s3://my-bucket/chronos2 --role arn:aws:iam::123456789012:role/AGCloudRole
    uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --yes \\
        --mode serverless --s3-path s3://my-bucket/chronos2 --role <role-arn>
    uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --delete \\
        --endpoint <name> --yes

**This creates paid AWS resources.** ``--plan`` is the default and creates nothing; every mode
that leaves something running prints the command that removes it. Nothing in this repository has
been run against a real account: article 5's costs come from locally measured throughput and the
published rates in ``data/reference/cloud_prices.csv``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import pandas as pd

from c2p.config import get_paths
from c2p.data import spain_energy_frame, validate_and_prepare
from c2p.experiments import FREQ, LOAD
from c2p.schema import ID_COL, TIMESTAMP_COL

MODEL_ID = "chronos-2"
PREDICTION_LENGTH = 24
QUANTILES = [0.1, 0.5, 0.9]
CONTEXT_HOURS = 2048
# AutoGluon Cloud's own registry defaults, for reference: batch ml.m5.2xlarge,
# real-time ml.g5.xlarge. Serverless takes no instance type, only memory and concurrency.
BATCH_INSTANCE = "ml.m5.2xlarge"
REALTIME_INSTANCE = "ml.c7i.xlarge"
SERVERLESS_MEMORY_MB = 6144
SERVERLESS_CONCURRENCY = 5


def require_cloud() -> type:
    """Import ``TimeSeriesFoundationModel`` lazily, with an actionable message when absent."""
    try:
        from autogluon.cloud import TimeSeriesFoundationModel
    except ModuleNotFoundError as error:  # pragma: no cover - depends on the local environment
        raise SystemExit(
            "autogluon.cloud is not installed. It is not a default dependency of this "
            'repository. Add it with:\n    uv add --group cloud "autogluon.cloud>=0.7"'
        ) from error
    return TimeSeriesFoundationModel


def history(*, hours: int = CONTEXT_HOURS) -> pd.DataFrame:
    """The last ``hours`` of Spanish load, in AutoGluon's long format."""
    raw = spain_energy_frame()[[ID_COL, TIMESTAMP_COL, LOAD]]
    series, _ = validate_and_prepare(raw, freq=FREQ, target=LOAD)
    return series.tail(hours).rename(columns={LOAD: "target"}).reset_index(drop=True)


def build_model(args: argparse.Namespace) -> Any:
    """Instantiate the foundation model wrapper from the parsed arguments."""
    cls = require_cloud()
    return cls(
        MODEL_ID,
        cloud_output_path=args.s3_path,
        role=args.role,
        model_artifact_uri=args.model_artifact_uri,
    )


def run_batch(args: argparse.Namespace) -> None:
    """One job: forecast and write to S3, then exit. Nothing is left running."""
    model = build_model(args)
    data = history()
    print(f"Forecasting {data[ID_COL].nunique()} series from {len(data):,} rows")
    forecasts = model.predict(
        data,
        target="target",
        id_column=ID_COL,
        timestamp_column=TIMESTAMP_COL,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILES,
        instance_type=args.batch_instance,
        wait=True,
    )
    print(forecasts.head(12).to_string())
    print("\nThe job has finished. Nothing is still billing except the S3 objects it wrote.")


def run_endpoint(args: argparse.Namespace) -> None:
    """Deploy a real-time or serverless endpoint, send one request, and say how to delete it."""
    model = build_model(args)
    serverless = args.mode == "serverless"
    if serverless and not args.model_artifact_uri:
        # Serverless containers start without Hub access, so the weights must already be in S3.
        print("Caching the model artifact to S3 (required for serverless)...")
        model = model.cache_model_artifact(f"{args.s3_path.rstrip('/')}/artifact")
    endpoint = model.deploy(
        instance_type=None if serverless else args.realtime_instance,
        endpoint_name=args.endpoint,
        inference_mode="serverless" if serverless else "realtime",
        inference_config=(
            {
                "memory_size_in_mb": args.serverless_memory_mb,
                "max_concurrency": args.serverless_concurrency,
            }
            if serverless
            else None
        ),
        wait=True,
    )
    print(f"Endpoint {endpoint.endpoint_name} is live.")
    forecasts = endpoint.predict(
        history(),
        target="target",
        id_column=ID_COL,
        timestamp_column=TIMESTAMP_COL,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILES,
    )
    print(forecasts.head(12).to_string())
    if serverless:
        print(
            "\nServerless: billed per GB-second of request duration, zero while idle. "
            "The first call after a quiet period pays a cold start."
        )
    else:
        print("\nReal-time: billed per hour from now until you delete it.")
    print(f"Delete it with: --delete --endpoint {endpoint.endpoint_name} --yes")


def delete(args: argparse.Namespace) -> None:
    """Delete an endpoint by name. S3 artifacts are left alone."""
    if not args.endpoint:
        raise SystemExit("--delete needs --endpoint <name>.")
    from autogluon.cloud import TimeSeriesEndpoint

    TimeSeriesEndpoint(args.endpoint).delete_endpoint()
    print(f"Deleted endpoint {args.endpoint}; billing stops.")
    print(f"Still in S3: {args.s3_path or '<s3 path>'}; remove it with `aws s3 rm --recursive`.")


def plan(args: argparse.Namespace) -> None:
    """Print what the chosen mode would create and what would keep billing; create nothing."""
    leaves_running = {
        "batch": "nothing (the job exits)",
        "realtime": f"an endpoint on {args.realtime_instance}, billed per hour",
        "serverless": (
            f"a serverless endpoint, {args.serverless_memory_mb} MB, "
            f"max concurrency {args.serverless_concurrency}, billed per GB-second"
        ),
    }[args.mode]
    print(
        "PLAN (nothing is created)\n"
        f"  model             {MODEL_ID} (weights: autogluon/chronos-2)\n"
        f"  mode              {args.mode}\n"
        f"  s3 path           {args.s3_path or '<s3://bucket/prefix>'}\n"
        f"  role              {args.role or '<AutoGluon Cloud role arn>'}\n"
        f"  horizon           {PREDICTION_LENGTH} steps, quantiles {QUANTILES}\n"
        f"  input             last {CONTEXT_HOURS} hours of Spanish load\n"
        f"  leaves running    {leaves_running}\n"
        "\nRates: data/reference/cloud_prices.csv (components Hosting, BatchTransform, "
        "ServerlessInference).\nRe-run with --yes to create it."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the command line: mode, instance type, S3 path and the ``--yes`` safety switch."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("batch", "realtime", "serverless"), default="batch")
    parser.add_argument("--s3-path", help="s3://bucket/prefix for artifacts and outputs")
    parser.add_argument("--role", help="IAM role ARN AutoGluon Cloud assumes")
    parser.add_argument("--endpoint", help="endpoint name (required for --delete)")
    parser.add_argument("--model-artifact-uri", help="cached model.tar.gz in S3, if you have one")
    parser.add_argument("--batch-instance", default=BATCH_INSTANCE)
    parser.add_argument("--realtime-instance", default=REALTIME_INSTANCE)
    parser.add_argument("--serverless-memory-mb", type=int, default=SERVERLESS_MEMORY_MB)
    parser.add_argument("--serverless-concurrency", type=int, default=SERVERLESS_CONCURRENCY)
    parser.add_argument("--plan", action="store_true", help="print the plan and exit (default)")
    parser.add_argument("--delete", action="store_true", help="delete an endpoint by name")
    parser.add_argument("--yes", action="store_true", help="actually touch AWS")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to plan, batch, endpoint or delete; ``--plan`` (the default) creates nothing."""
    args = parse_args(argv)
    get_paths()  # fail early if the repository layout is wrong
    if args.delete:
        if not args.yes:
            print(f"Would delete endpoint {args.endpoint or '<name>'}. Pass --yes.")
            return 0
        delete(args)
        return 0
    if not args.yes or args.plan:
        plan(args)
        return 0
    if not (args.s3_path and args.role):
        raise SystemExit("--s3-path and --role are required to run on AWS.")
    if args.mode == "batch":
        run_batch(args)
    else:
        run_endpoint(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
