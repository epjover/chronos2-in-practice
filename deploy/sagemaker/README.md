# Chronos-2 on a SageMaker real-time endpoint

The managed counterpart to `deploy/fastapi`, used in Part 5. The handler speaks the same JSON
as the FastAPI service, so the only thing that changes between the two is who runs the container.

| File | Purpose |
| --- | --- |
| `inference.py` | The handler the managed PyTorch container imports (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) |
| `deploy_endpoint.py` | Build the model artifact, create the endpoint, invoke it, **delete it** |
| `../../tests/test_sagemaker_inference.py` | Contract tests with a fake forecaster (fast, no AWS) |

> **This creates paid AWS resources.** A real-time endpoint bills for every hour it exists, even
> at zero traffic. `deploy_endpoint.py` creates nothing unless you pass `--yes`, and
> `--delete --yes` removes everything it made. Nothing in this repository has ever been run
> against a real account: Part 5's cloud costs are computed from locally measured throughput
> and the published rates in `data/reference/cloud_prices.csv`.

## Prerequisites

`boto3` and the SageMaker SDK are deliberately *not* dependencies of this project; nothing else
here needs an AWS account. Add them to a separate group when you want them:

```bash
uv add --group cloud boto3 sagemaker
```

You also need an S3 bucket in the same region and a SageMaker execution role whose policy allows
reading that bucket.

## Deploy, invoke, delete

```bash
# 1. See exactly what would be created, and nothing else happens
uv run --group cloud python deploy/sagemaker/deploy_endpoint.py --plan

# 2. Create it (packages inference.py + src/c2p, uploads model.tar.gz, waits for InService)
uv run --group cloud python deploy/sagemaker/deploy_endpoint.py --yes \
    --bucket my-bucket --role arn:aws:iam::123456789012:role/SageMakerExecutionRole

# 3. One request, with the server-side latency the endpoint reports
uv run --group cloud python deploy/sagemaker/deploy_endpoint.py --invoke

# 4. Delete it: endpoint, endpoint configuration and model registration
uv run --group cloud python deploy/sagemaker/deploy_endpoint.py --delete --yes
```

Step 4 does not remove the `model.tar.gz` in S3 (storage, cents per month) or any Studio or
notebook instance you used to run these commands; those are separate meters. `aws s3 rm` and
the Studio console handle them.

To serve a fine-tuned model from Part 4, package the checkpoint into the artifact:

```bash
uv run --group cloud python deploy/sagemaker/deploy_endpoint.py --yes \
    --bucket my-bucket --role <role-arn> --checkpoint models/chronos-2-lora
```

The handler then loads `/opt/ml/model/chronos-2` from disk instead of pulling from the Hub, so
the endpoint needs no internet access at inference time.

## Choosing the instance

Chronos-2 is 119.5M parameters: it fits in a few GB, so the choice is a small CPU instance
against a small GPU one, and it is decided by how many series you send per call. Rates for
`us-east-1` are in `data/reference/cloud_prices.csv`, refreshed with:

```bash
uv run python scripts/fetch_cloud_prices.py --region us-east-1
```

That script reads the public AWS Price List API (no credentials, no resources, no cost) and
records the SKU, the effective date and the offer-file version next to every price, so a stale
table is visible rather than silently wrong.

`--instance-type ml.g5.xlarge` switches the handler to `cuda` automatically (any `ml.g*`
instance does); everything else runs on CPU.

## What the handler does differently from `app.py`

Nothing, by design, except error handling: SageMaker turns an exception into a 400 with the
message, so `inference.py` raises `ValueError` where `app.py` builds an HTTP response. Both run
every request through `c2p.data.validate_and_prepare` and both return a `input_hash` of the
canonical model input, which is what ties a served forecast to a row in the monitoring log
(`src/c2p/monitoring.py`).
