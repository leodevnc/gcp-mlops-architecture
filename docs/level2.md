# Level 2 pipeline CI/CD

Each pull request and push to main runs the workflow and pipeline tests, builds the Docker image, and tests training and inference inside it. With staging enabled, main-branch builds push the verified image, publish a KFP template pinned to that image digest, and run the published template in Vertex AI. A rejected candidate fails the staging validation.

## Configure staging

1. Set up a Cloud Storage input object and pipeline root, a Vertex pipeline service account, and a regional Artifact Registry Docker repository using the [Level 1 runbook](level1.md).
2. Create a KFP repository in the same region. [Artifact Registry KFP setup](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/pipelines/create-pipeline-template)
3. Set up a Workload Identity Federation provider for this GitHub repository and bind it to a release service account. Scope the provider's attribute condition to the numeric repository ID and the main branch. The release account needs Artifact Registry Writer on the two repositories, Vertex AI pipeline job submission access, and permission to act as the pipeline runtime service account. The runtime account needs the Level 1 pipeline, storage, and model permissions. [Google Cloud federation setup](https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines)
4. Create the GitHub environment named `staging` and add its variables below. Set the repository variable `GCP_RELEASE_ENABLED` to `true` when staging is ready.

| Staging variable | Value |
| --- | --- |
| GCP_PROJECT | Staging Google Cloud project ID |
| GCP_REGION | Region shared by Vertex AI and Artifact Registry |
| GCP_DOCKER_REPO | Docker repository ID |
| GCP_KFP_REPO | KFP repository ID |
| GCP_SOURCE_URI | Cloud Storage URI of the labeled staging CSV |
| GCP_PIPELINE_ROOT | Cloud Storage prefix for staging outputs |
| GCP_PIPELINE_SA | Vertex pipeline runtime service account email |
| GCP_WIF_PROVIDER | Full resource name of the Workload Identity Federation provider |
| GCP_RELEASE_SA | GitHub release service account email |

## Promote a validated version

The staging job summary contains the Docker image and KFP template digests. Use that exact template URL for a one-off production run or a new recurring schedule with the Level 1 CLI. Point the run at the target project's source, output bucket, runtime service account, and model ID. When using a separate project, grant that environment access to the template and grant its Vertex service agent access to the same-region serving image. [Custom container permissions](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/custom-container-requirements)

```bash
uv run --locked --group gcp python -m mlops_demo.gcp submit \
  --template "$VALIDATED_TEMPLATE_URI" --project "$TARGET_PROJECT" --region "$TARGET_REGION" \
  --source-uri "$TARGET_SOURCE_URI" --pipeline-root "$TARGET_PIPELINE_ROOT" \
  --service-account "$TARGET_PIPELINE_SA" --model-id delivery-time --wait --require-approved
```

For a scheduled rollout, use the same digest URL with the Level 1 `schedule` command and pause the prior schedule in Vertex AI when switching versions. Select an earlier validated digest to roll back the pipeline code. Use an explicit Model Registry version when deploying an online Endpoint.
