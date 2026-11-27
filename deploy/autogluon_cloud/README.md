# Chronos-2 through AutoGluon Cloud

The third deployment path of Part 5, and the one with the least code: no Dockerfile, no
`model_fn`, no training job. `TimeSeriesFoundationModel` wraps the pretrained checkpoint and
`deploy_timeseries.py` exposes the three shapes a forecasting workload takes.

| `--mode` | What it creates | What keeps billing afterwards |
| --- | --- | --- |
| `batch` (default) | One SageMaker job that reads a frame, writes forecasts to S3 and exits | Nothing but the S3 objects |
| `realtime` | A dedicated endpoint on `--realtime-instance` | The endpoint, per hour, until deleted |
| `serverless` | A SageMaker Serverless Inference endpoint (CPU only) | Nothing while idle; GB-seconds per request |

> **This creates paid AWS resources.** `--plan` is the default and creates nothing; nothing
> touches AWS without `--yes`. Every mode that leaves something running prints the command
> that removes it.

> **Status: written against the documented API, never run against an AWS account.** The
> script has not created a job or an endpoint, so treat it as a starting point and expect to
> adjust it. The costs in Part 5 do not depend on it: they come from throughput measured
> locally and the published rates in `data/reference/cloud_prices.csv`.

## Prerequisites

```bash
uv add --group cloud "autogluon.cloud>=0.7"
```

Plus an S3 prefix and an IAM role AutoGluon Cloud can assume (its documentation lists the
required policies; they are broader than the SageMaker execution role in `../sagemaker`,
because the library also launches jobs on your behalf).

## Use it

```bash
# 1. Print what would be created and what would keep billing. Creates nothing.
uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --plan --mode serverless

# 2. A batch job: forecast the next 24 hours of Spanish load, write to S3, exit.
uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --yes --mode batch \
    --s3-path s3://my-bucket/chronos2 --role arn:aws:iam::123456789012:role/AGCloudRole

# 3. A serverless endpoint that scales to zero.
uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --yes --mode serverless \
    --s3-path s3://my-bucket/chronos2 --role <role-arn>

# 4. Delete an endpoint by name (real-time or serverless).
uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --delete \
    --endpoint <name> --yes
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mode` | `batch` | `batch`, `realtime` or `serverless` |
| `--s3-path`, `--role` | none | Required with `--yes`: where artifacts and outputs go, and the role assumed |
| `--batch-instance` | `ml.m5.2xlarge` | Instance of the batch job |
| `--realtime-instance` | `ml.c7i.xlarge` | Instance of the real-time endpoint |
| `--serverless-memory-mb` | `6144` | Memory of the serverless endpoint (the largest size) |
| `--serverless-concurrency` | `5` | Maximum concurrent invocations |
| `--model-artifact-uri` | none | A cached `model.tar.gz` in S3, if you already have one |
| `--endpoint` | none | Endpoint name; required with `--delete` |
| `--plan` | on unless `--yes` | Print the plan and exit |
| `--delete` | off | Delete the endpoint named by `--endpoint` |
| `--yes` | off | The safety switch: without it nothing is created or deleted |

`--delete` removes the endpoint only. What the run wrote to S3 (the cached model artifact, the
batch outputs) stays there; remove it with `aws s3 rm --recursive s3://my-bucket/chronos2`.

## What you give up, and what you get

- **Serverless needs the weights in S3.** A serverless container starts without access to the
  Hugging Face Hub, so the script calls `cache_model_artifact` first (or reuses
  `--model-artifact-uri`). It runs on CPU and the first call after a quiet period pays a cold
  start: Part 5 prices that trade.
- **The request format is AutoGluon's**, not the JSON that `deploy/fastapi` and
  `deploy/sagemaker` share, so a client written against one does not work against this one.
  That is the cost of the managed path; the saving is that there is no handler, no image and
  no endpoint wiring to maintain.
- **No fine-tuning here.** This script serves the pretrained checkpoint. Fine-tuning through
  AutoGluon goes through `TimeSeriesPredictor` with `hyperparameters={"Chronos2": {...}}`,
  which Part 4 describes next to the notebooks that do it with the library directly.
