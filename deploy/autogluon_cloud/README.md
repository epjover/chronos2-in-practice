# Chronos-2 through AutoGluon Cloud

The third deployment path in article 5, and the one with the least code: `fit` runs a SageMaker
training job, `deploy` creates an endpoint, and there is no Dockerfile and no `model_fn`.

| File | Purpose |
| --- | --- |
| `deploy_timeseries.py` | Fit a `TimeSeriesPredictor` restricted to Chronos-2 on SageMaker, deploy it, **delete it** |

> **This creates paid AWS resources**: a training job billed per second and, unless you pass
> `--no-deploy`, a real-time endpoint billed per hour until deleted. `--plan` is the default and
> creates nothing; `--cleanup --yes` deletes the endpoint. Nothing here has been run against a
> real account.

## Prerequisites

```bash
uv add --group cloud "autogluon.cloud>=0.7" "autogluon.timeseries>=1.6"
```

Plus an S3 prefix and an IAM role AutoGluon Cloud can assume (its documentation lists the
required policies — they are broader than the SageMaker execution role in `../sagemaker`,
because the library also launches jobs on your behalf).

## Use it

```bash
uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --plan

# zero-shot: no gradient steps, the job just packages and registers the predictor
uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --yes \
    --s3-path s3://my-bucket/chronos2 --role arn:aws:iam::123456789012:role/AGCloudRole

# LoRA fine-tuning on a GPU instance, no endpoint afterwards
uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --yes --fine-tune \
    --no-deploy --s3-path s3://my-bucket/chronos2 --role <role-arn>

uv run --group cloud python deploy/autogluon_cloud/deploy_timeseries.py --cleanup --yes
```

`--cleanup` deletes the endpoint only. The trained predictor stays in S3; remove it with
`aws s3 rm --recursive s3://my-bucket/chronos2`.

## What you give up, and what you get

The script pins `hyperparameters={"Chronos2": {...}}` so AutoGluon trains Chronos-2 and nothing
else. Without that it would fit an ensemble, and the resulting accuracy would no longer be
attributable to the model this series is about.

The request format is AutoGluon's, not the JSON `deploy/fastapi` and `deploy/sagemaker` share,
so a client written against one does not work against this one. That is the real cost of the
managed path; the saving is that there is no handler, no image and no endpoint wiring to
maintain.

One detail worth knowing before you choose between fine-tuning here and the notebooks in article
4: `Chronos2Model` in `autogluon.timeseries` 1.6.3 defaults to LoRA at `fine_tune_lr=1e-5`,
`fine_tune_steps=1000`, `fine_tune_batch_size=32` and `fine_tune_context_length=2048`, which are
the settings article 4 arrives at independently. The managed path is not a different experiment,
only less code around the same one.
