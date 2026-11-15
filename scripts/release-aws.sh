#!/usr/bin/env bash
set -euo pipefail

revision="${CODEBUILD_RESOLVED_SOURCE_VERSION:?Missing Git revision}"
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Expected a Git commit SHA from CodePipeline." >&2
  exit 2
fi
release_id="$revision-${CODEBUILD_BUILD_NUMBER:?Missing build number}"
image="${WORKER_URI:?}:$release_id"
bash scripts/build-aws-worker.sh "$image"

registry="${WORKER_URI%%/*}"
aws ecr get-login-password --region "$AWS_REGION" |
  docker login --username AWS --password-stdin "$registry"
docker push "$image"
digest="$(aws ecr describe-images --repository-name "$WORKER_REPO" \
  --image-ids "imageTag=$release_id" --query 'imageDetails[0].imageDigest' --output text)"
staging_source="s3://$MODEL_BUCKET/staging/$release_id/deliveries.csv"
aws s3 cp data/deliveries.csv "$staging_source" --only-show-errors

uv run --locked --group aws python -m mlops_demo.aws release \
  --image-uri "$WORKER_URI@$digest" --revision "$revision" \
  --pipeline-name "$TRAINING_PIPELINE" --pipeline-role "$TRAINING_PIPELINE_ROLE" \
  --processing-role "$PROCESSING_ROLE" --artifact-root "s3://$MODEL_BUCKET/runs" \
  --source-uri "s3://$MODEL_BUCKET/$SOURCE_KEY" --model-group "$MODEL_GROUP" \
  --staging-source "$staging_source" --staging-group "$STAGING_MODEL_GROUP" \
  --manifest-uri "s3://$MODEL_BUCKET/releases/$release_id.json" \
  --schedule-name "$RETRAINING_NAME" --scheduler-role "$SCHEDULER_ROLE" \
  --schedule-expression "${RETRAINING_EXPRESSION:-}" --schedule-timezone "$RETRAINING_TIMEZONE"
