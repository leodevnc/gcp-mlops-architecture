# Level 1 continuous training on Google Cloud

## 1. Configure the project

Use a billing-enabled Google Cloud project, the Google Cloud CLI, Python 3.12+, and uv. Set the following values for an existing regional Cloud Storage bucket, regional Artifact Registry Docker repository, and pipeline runtime service account. Use the same project and region for the pipeline and repository.

```bash
export GCP_PROJECT="your-project-id"
export GCP_REGION="us-central1"
export GCP_BUCKET="your-bucket-name"
export GCP_DOCKER_REPO="your-docker-repository"
export GCP_PIPELINE_SA="pipeline-runner@$GCP_PROJECT.iam.gserviceaccount.com"

gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com storage.googleapis.com --project "$GCP_PROJECT"
gcloud auth application-default login
gcloud auth application-default set-quota-project "$GCP_PROJECT"
```

Use a gcloud identity with access to submit builds, upload the sample CSV, and read the published image digest. The Cloud Build runtime identity needs permission to push to the Docker repository. The Python submitter needs Vertex AI pipeline and schedule permissions and permission to act as the pipeline service account. Grant that service account access to Vertex AI and the dataset and pipeline buckets. It needs bucket metadata access and permission to read the source and create and read run artifacts. The official setup guide covers project and service account provisioning. [Configure a pipeline project](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/pipelines/configure-project)

## 2. Build the worker and compile the pipeline

Run from the repository root. The sample dataset is synthetic.

```bash
IMAGE_NAME="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GCP_DOCKER_REPO/delivery-mlops"
IMAGE_TAG="$IMAGE_NAME:$(git rev-parse --short HEAD)"
GCP_SOURCE_URI="gs://$GCP_BUCKET/delivery-mlops/input/deliveries.csv"

gcloud storage cp data/deliveries.csv "$GCP_SOURCE_URI" --project "$GCP_PROJECT"
gcloud builds submit --project "$GCP_PROJECT" --ignore-file=.dockerignore --tag "$IMAGE_TAG" .
IMAGE_URI="$(gcloud artifacts docker images describe "$IMAGE_TAG" \
  --project "$GCP_PROJECT" --format='value(image_summary.fully_qualified_digest)')"

uv sync --locked --group gcp
uv run --locked --group gcp python -m mlops_demo.gcp compile \
  --image-uri "$IMAGE_URI" --output runs/level1/pipeline.yaml
```

The compiled definition pins the container digest. The image also supplies the Vertex prediction server for registered model versions. [Custom inference container contract](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/custom-container-requirements)

## 3. Submit a run and create a schedule

```bash
run_args=(
  --template runs/level1/pipeline.yaml
  --project "$GCP_PROJECT"
  --region "$GCP_REGION"
  --source-uri "$GCP_SOURCE_URI"
  --pipeline-root "gs://$GCP_BUCKET/delivery-mlops/runs"
  --service-account "$GCP_PIPELINE_SA"
  --model-id delivery-time
  --max-mae 5
)

uv run --locked --group gcp python -m mlops_demo.gcp submit "${run_args[@]}" --wait

uv run --locked --group gcp python -m mlops_demo.gcp schedule "${run_args[@]}" \
  --name delivery-time-daily --cron 'TZ=Asia/Seoul 0 3 * * *'
```

The example schedule runs at 03:00 Korea time with at most one active run for that schedule. It reads the current Cloud Storage object generation each time, so publishing a new CSV at the same URI provides the next run's data. The schedule command reports an existing schedule with the same name. Manage schedule updates and pauses through Vertex AI. [Pipeline scheduling](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/pipelines/schedule-pipeline-run)

## Inspect a run

| Stage | Evidence |
| --- | --- |
| Prepare | Input Cloud Storage URI and generation, CSV SHA-256, deterministic split seed, and training and holdout counts. |
| Train | Candidate model and its link to the prepared dataset. |
| Evaluate | Candidate MAE, training-mean baseline MAE, threshold, and a pass or reject decision. A bad CSV fails the job. A rejected candidate records its metrics and skips Model Registry publication. |
| Register | A Model Registry version and an approved bundle with the exact model and evaluation report. If the bundle and serving image match an existing version alias, the run reuses that version. |

Use the printed pipeline job resource to find the run in the Google Cloud console. Vertex ML Metadata stores the task executions and artifact lineage. Registry publication produces a deployable version. Sending online traffic to a version requires a separate Endpoint deployment. [Artifact lineage](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/pipelines/lineage)

## Verify locally

```bash
uv run --locked --group gcp python -m unittest discover -s tests/gcp -v
```

The tests compile the pipeline definition and run the Python training, validation, and registry paths with simulated Cloud Storage and Model Registry responses.
