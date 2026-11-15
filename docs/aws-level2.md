# AWS Level 2 pipeline CI/CD

| Stage | Implementation |
| --- | --- |
| Source | CodePipeline V2 tracks the configured GitHub branch through CodeConnections and queues releases in order. |
| Verify and build | CodeBuild runs the local and AWS integration tests, builds the image, and checks training and prediction inside the container. |
| Release | The verified image is pushed to ECR. The SageMaker pipeline definition and model package use its SHA-256 digest. CodeBuild waits for a successful staging run of that exact pipeline version. |
| Continuous training | When enabled, EventBridge Scheduler starts the validated SageMaker pipeline version on the configured cadence. A successful later release updates the schedule to its version. |
| Model validation | SageMaker Processing reads one S3 object version, validates and splits it, trains the model, and evaluates MAE on the holdout. Passing candidates enter SageMaker Model Registry as pending approval. |

Staging uses the bundled synthetic CSV and a separate model group. Continuous training uses the configured dataset key and the training model group. A staging rejection fails delivery before the retraining schedule is updated. Training runs record the source version ID, CSV digest, split metadata, matching evaluation report, and model bundle under an execution-specific S3 prefix.

## Verify locally

```bash
uv sync --locked --group aws
uv run --locked --group aws python -m unittest discover -s tests -v
uv run --locked --group aws python -m unittest discover -s tests/aws -v
bash scripts/build-aws-worker.sh delivery-mlops-aws:local
```

The API tests use stubbed AWS responses. The container check needs Docker.

## Set up the AWS stack

Use an AWS CLI session in a region with CodeConnections, CodePipeline, CodeBuild, SageMaker AI, and EventBridge Scheduler. Create and authorize the GitHub connection in the AWS console before passing its ARN. Limit the GitHub App installation to this repository. [CodeConnections source setup](https://docs.aws.amazon.com/codepipeline/latest/userguide/action-reference-CodestarConnectionSource.html)

```bash
aws cloudformation deploy --stack-name delivery-mlops-aws \
  --template-file infra/aws/mlops.yaml --capabilities CAPABILITY_IAM \
  --parameter-overrides "ConnectionArn=$CONNECTION_ARN" SourceBranch=feat/aws-level-2

aws cloudformation describe-stacks --stack-name delivery-mlops-aws \
  --query 'Stacks[0].Outputs' --output table
```

Use a distinct `NamePrefix` for each stack in an account and region. The first build stages the example CSV, tests the release in SageMaker, and writes its execution ARN, Git revision, ECR image digest, and pipeline version under the bucket's `releases/` prefix. Follow the build and processing logs in CloudWatch.

## Enable scheduled retraining

Upload your labeled CSV to the stack's `DatasetUri` output. To run the example daily at 03:00 Korea time, update the stack with the expression below and start a new CodePipeline execution. Later CSV uploads at the same key supply the next run with the latest S3 version.

```bash
aws s3 cp data/deliveries.csv "$DATASET_URI"

aws cloudformation deploy --stack-name delivery-mlops-aws \
  --template-file infra/aws/mlops.yaml --capabilities CAPABILITY_IAM \
  --parameter-overrides "ConnectionArn=$CONNECTION_ARN" SourceBranch=feat/aws-level-2 \
    'RetrainingExpression=cron(0 3 * * ? *)' RetrainingTimezone=Asia/Seoul

aws codepipeline start-pipeline-execution --name "$DELIVERY_PIPELINE_NAME"
```

The release build creates or updates the schedule after the staging pipeline succeeds. Inspect its pinned pipeline version with `aws scheduler get-schedule --name "$RETRAINING_SCHEDULE_NAME"`. Manage a schedule's enabled state in EventBridge Scheduler.

## Use a registered model

The worker also serves `GET /ping` and `POST /invocations` from SageMaker's model directory. Approve and deploy a specific Model Registry version to use it at an online endpoint. A prediction request uses the same `instances` payload as the batch API. [SageMaker custom inference container](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html)
