"""Create, invoke and, above all, delete a SageMaker real-time endpoint for Chronos-2.

A real-time endpoint bills for every hour it exists, whether or not anything calls it. That is
the single most expensive mistake in this series, so this script is built around teardown: it
prints the plan and does nothing unless you pass ``--yes``, it tags everything it creates, and
``--delete`` removes the endpoint, the endpoint configuration and the model registration in the
order AWS requires.

What it builds: a ``model.tar.gz`` holding ``inference.py`` and the ``c2p`` package (optionally
with a fine-tuned checkpoint from article 4), uploaded to S3 and served by the managed PyTorch
inference container. The handler in ``inference.py`` speaks the same JSON as the FastAPI
service, so ``deploy/fastapi/smoke_test.py``'s payloads work against both.

Usage from the repository root::

    uv add --group cloud boto3 sagemaker            # once; not a default dependency
    uv run --group cloud python deploy/sagemaker/deploy_endpoint.py --plan
    uv run --group cloud python deploy/sagemaker/deploy_endpoint.py --yes --bucket my-bucket \\
        --role arn:aws:iam::123456789012:role/SageMakerExecutionRole
    uv run --group cloud python deploy/sagemaker/deploy_endpoint.py --invoke
    uv run --group cloud python deploy/sagemaker/deploy_endpoint.py --delete --yes

**This script creates paid AWS resources.** ``--plan`` (the default) creates nothing. Nothing in
this repository was ever run against a real account: the numbers in article 5 come from the
local service and from the published rates in ``data/reference/cloud_prices.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tarfile
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

ENDPOINT_NAME = "chronos2-forecast"
INSTANCE_TYPE = "ml.c7i.xlarge"  # CPU: see data/reference/cloud_prices.csv for the hourly rate
INSTANCE_COUNT = 1
FRAMEWORK_VERSION = "2.9"  # managed PyTorch inference container
PY_VERSION = "py312"
TAGS = [{"Key": "project", "Value": "chronos-2-practical"}, {"Key": "managed-by", "Value": "repo"}]
REPO_ROOT = Path(__file__).resolve().parents[2]


def require_boto3():
    """Import boto3 lazily, with an actionable message when it is not installed."""
    try:
        import boto3
    except ModuleNotFoundError as error:  # pragma: no cover - depends on the local environment
        raise SystemExit(
            "boto3 is not installed. It is not a default dependency of this repository because "
            "nothing here needs an AWS account. Add it with:\n"
            "    uv add --group cloud boto3 sagemaker"
        ) from error
    return boto3


def build_artifact(destination: Path, *, checkpoint: Path | None) -> Path:
    """Package ``inference.py``, the ``c2p`` package and an optional checkpoint into a tarball.

    The managed container puts ``code/`` on ``sys.path`` and imports ``code/inference.py``;
    ``requirements.txt`` beside it is installed at container start.
    """
    archive = destination / "model.tar.gz"
    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        code = staging / "code"
        code.mkdir()
        shutil.copy2(Path(__file__).with_name("inference.py"), code / "inference.py")
        shutil.copytree(REPO_ROOT / "src" / "c2p", code / "c2p")
        (code / "requirements.txt").write_text(
            "chronos-forecasting==2.3.2\npandas>=3.0\n", encoding="utf-8"
        )
        if checkpoint is not None:
            if not checkpoint.is_dir():
                raise SystemExit(f"{checkpoint} is not a folder with a Chronos-2 checkpoint.")
            shutil.copytree(checkpoint, staging / "chronos-2")
        with tarfile.open(archive, "w:gz") as tar:
            for entry in sorted(staging.iterdir()):
                tar.add(entry, arcname=entry.name)
    return archive


def upload(boto3, archive: Path, *, bucket: str, prefix: str) -> str:
    key = f"{prefix.rstrip('/')}/{archive.name}"
    boto3.client("s3").upload_file(str(archive), bucket, key)
    return f"s3://{bucket}/{key}"


def create(args, *, model_data: str) -> None:
    """Register the model, create the endpoint configuration and the endpoint, then wait."""
    boto3 = require_boto3()
    from sagemaker import image_uris

    sagemaker_client = boto3.client("sagemaker")
    region = boto3.session.Session().region_name
    image = image_uris.retrieve(
        framework="pytorch",
        region=region,
        version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        instance_type=args.instance_type,
        image_scope="inference",
    )
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    model_name = f"{args.name}-{stamp}"
    sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=args.role,
        PrimaryContainer={
            "Image": image,
            "ModelDataUrl": model_data,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "MODEL_ID": args.model_id,
                "DEVICE": "cuda" if args.instance_type.startswith("ml.g") else "cpu",
                "MAX_CONTEXT": str(args.max_context),
            },
        },
        Tags=TAGS,
    )
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=model_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": args.instance_count,
                "InstanceType": args.instance_type,
            }
        ],
        Tags=TAGS,
    )
    sagemaker_client.create_endpoint(
        EndpointName=args.name, EndpointConfigName=model_name, Tags=TAGS
    )
    print(f"Creating endpoint {args.name} ({args.instance_type}); billing starts when InService.")
    waiter = sagemaker_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=args.name)
    print(f"InService. Delete it with: python {Path(__file__).name} --delete --yes")


def invoke(args) -> None:
    """Send one small request and print the first few steps and the server-side latency."""
    boto3 = require_boto3()
    # Two weeks of a daily sine, so a correct response is visibly periodic at a glance.
    start = datetime(2018, 12, 1, tzinfo=UTC)
    hours = 336
    payload = {
        "series": [
            {
                "item_id": "ES",
                "timestamps": [
                    (start + timedelta(hours=step)).isoformat() for step in range(hours)
                ],
                "values": [
                    25000 + 5000 * math.sin(2 * math.pi * step / 24) for step in range(hours)
                ],
            }
        ],
        "prediction_length": 24,
        "freq": "h",
        "quantile_levels": [0.1, 0.5, 0.9],
    }
    began = time.perf_counter()
    response = boto3.client("sagemaker-runtime").invoke_endpoint(
        EndpointName=args.name, ContentType="application/json", Body=json.dumps(payload)
    )
    round_trip_ms = (time.perf_counter() - began) * 1000.0
    body = json.loads(response["Body"].read())
    forecast = body["forecasts"][0]
    print(f"input_hash {body['input_hash']}  server {body['latency_ms']:.1f} ms")
    for stamp, median in list(zip(forecast["timestamps"], forecast["predictions"], strict=True))[
        :6
    ]:
        print(f"  {stamp}  {median:10.1f}")
    print(f"round trip {round_trip_ms:.1f} ms")


def delete(args) -> None:
    """Remove the endpoint, its configuration and the model, ignoring what is already gone."""
    boto3 = require_boto3()
    client = boto3.client("sagemaker")
    try:
        described = client.describe_endpoint(EndpointName=args.name)
        config_name = described["EndpointConfigName"]
    except client.exceptions.ClientError:
        config_name = None
        print(f"Endpoint {args.name} does not exist.")
    if config_name:
        client.delete_endpoint(EndpointName=args.name)
        print(f"Deleted endpoint {args.name} (billing stops).")
        model_name = client.describe_endpoint_config(EndpointConfigName=config_name)[
            "ProductionVariants"
        ][0]["ModelName"]
        client.delete_endpoint_config(EndpointConfigName=config_name)
        print(f"Deleted endpoint config {config_name}.")
        client.delete_model(ModelName=model_name)
        print(f"Deleted model {model_name}.")
    print(
        "Still billing if you leave them: the model.tar.gz in S3, and any Studio or notebook "
        "instance you used to run this."
    )


def plan(args) -> None:
    """Print exactly what would be created, and what it would cost per hour, then stop."""
    print(
        f"PLAN (nothing is created)\n"
        f"  endpoint        {args.name}\n"
        f"  instance        {args.instance_count} x {args.instance_type}\n"
        f"  model           {args.model_id}"
        + (f" (checkpoint {args.checkpoint})" if args.checkpoint else "")
        + f"\n  artifact        s3://{args.bucket or '<bucket>'}/{args.prefix}/model.tar.gz\n"
        f"  role            {args.role or '<execution role arn>'}\n"
        f"  tags            {TAGS}\n"
        f"\nHourly rates are in data/reference/cloud_prices.csv (component Hosting), fetched from "
        f"the AWS Price List API.\nRe-run with --yes to create it, and with --delete --yes to "
        f"remove it."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=ENDPOINT_NAME)
    parser.add_argument("--bucket", help="S3 bucket for the model artifact")
    parser.add_argument("--prefix", default="chronos2")
    parser.add_argument("--role", help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default=INSTANCE_TYPE)
    parser.add_argument("--instance-count", type=int, default=INSTANCE_COUNT)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument(
        "--checkpoint", type=Path, help="folder with a fine-tuned checkpoint to package"
    )
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--plan", action="store_true", help="print the plan and exit (default)")
    parser.add_argument("--invoke", action="store_true", help="call an endpoint that already runs")
    parser.add_argument("--delete", action="store_true", help="tear the endpoint down")
    parser.add_argument("--yes", action="store_true", help="actually touch AWS")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.invoke:
        invoke(args)
        return 0
    if args.delete:
        if not args.yes:
            print(f"Would delete endpoint {args.name}, its config and its model. Pass --yes.")
            return 0
        delete(args)
        return 0
    if not args.yes or args.plan:
        plan(args)
        return 0
    if not (args.bucket and args.role):
        print("--bucket and --role are required to create an endpoint.", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as workspace:
        archive = build_artifact(Path(workspace), checkpoint=args.checkpoint)
        size_mb = archive.stat().st_size / 2**20
        print(f"Built {archive.name} ({size_mb:.1f} MB)")
        model_data = upload(require_boto3(), archive, bucket=args.bucket, prefix=args.prefix)
    print(f"Uploaded {model_data}")
    create(args, model_data=model_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
