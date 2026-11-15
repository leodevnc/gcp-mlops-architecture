# Google Cloud MLOps reference implementation

## Run Level 0 locally

Install Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/). Run the following commands from the repository root. The bundled delivery-time dataset is synthetic.

```bash
uv sync --locked
uv run --locked python -m mlops_demo prepare data/deliveries.csv --output runs/demo/prepared.json
uv run --locked python -m mlops_demo train runs/demo/prepared.json --output runs/demo/model.json
uv run --locked python -m mlops_demo evaluate runs/demo/prepared.json runs/demo/model.json \
  --report runs/demo/evaluation.json --max-mae 5
uv run --locked python -m mlops_demo serve runs/demo/model.json --report runs/demo/evaluation.json
```

From another terminal, request a prediction.

```bash
curl -sS --fail-with-body http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"distance_km":12.5,"package_count":3}'
```

## Data and model artifacts

| Step | Output | Behavior |
| --- | --- | --- |
| Prepare | `prepared.json` | Validates the CSV, records its SHA-256 digest and split seed, and holds out about 25% of rows. |
| Train | `model.json` | Fits a linear regression and computes a mean-label baseline on the training rows. |
| Evaluate | `evaluation.json` | Computes both models' mean absolute error (MAE) on the same holdout. The candidate must beat the baseline and meet the configured MAE limit. A rejection writes the report and exits with code 1. |
| Serve | `GET /health` and `POST /predict` | Starts a local API after confirming that the passing report matches the model's SHA-256 digest. |

The CSV requires at least 12 rows. Each row contains a distance greater than zero and at most 100 km, an integer package count from 1 to 20, and a delivery time greater than zero and at most 1,440 minutes. Prediction requests use the same feature constraints.

Run the workflow tests with `uv run --locked python -m unittest discover -s tests -v`.

## Run Level 1 on Google Cloud

The continuous training pipeline reads a versioned snapshot of the input CSV, validates and splits the data, trains a candidate, and evaluates it on the holdout. A passing run publishes an approved model version and its evaluation report. The pipeline can run on demand or on a recurring Vertex AI schedule. [Runbook](docs/level1.md)

## Release Level 2 through CI/CD

GitHub Actions tests each PR and builds the worker container. With staging credentials and variables configured, a push to the main branch releases the verified image and KFP template to Artifact Registry and validates the release in Vertex AI. [CI/CD setup](docs/level2.md)

## Implementation stages

| Google Cloud maturity level | Repository scope |
| --- | --- |
| 0 · Manual process | Local commands for data preparation, training, holdout validation, and prediction. |
| 1 · ML pipeline automation | Vertex AI Pipelines, scheduled retraining, ML Metadata, and gated publication to Model Registry. |
| 2 · CI/CD pipeline automation | PR tests and container build, followed by a versioned staging release and a quality-gated Vertex AI run. |

The stages follow the [Google Cloud Architecture Center guide to MLOps CI, CD, and CT](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?hl=ko).
