import io
import json
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import boto3
from botocore.exceptions import WaiterError
from botocore.stub import ANY, Stubber
from fastapi.testclient import TestClient

from mlops_demo.aws.pipeline import build_pipeline_definition
from mlops_demo.aws.processing import process_s3
from mlops_demo.aws.release import deploy_and_validate, schedule_validated_version
from mlops_demo.serving import create_batch_app
from mlops_demo.workflow import load_approved_model

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "deliveries.csv"
IMAGE = f"111122223333.dkr.ecr.us-east-1.amazonaws.com/delivery-worker@sha256:{'a' * 64}"
PIPELINE_NAME = "delivery-training"
PIPELINE = f"arn:aws:sagemaker:us-east-1:111122223333:pipeline/{PIPELINE_NAME}"
EXECUTION = f"{PIPELINE}/execution/abc123"
ROLE = "arn:aws:iam::111122223333:role/training"
SOURCE = "s3://delivery-bucket/datasets/deliveries.csv"
STAGING = "s3://delivery-bucket/staging/deliveries.csv"


class PipelineTest(unittest.TestCase):
    def test_definition_gates_model_registration_and_pins_the_container(self) -> None:
        definition = build_pipeline_definition(
            image_uri=IMAGE,
            processing_role=ROLE,
            source_uri=SOURCE,
            artifact_root="s3://delivery-bucket/runs",
            model_group="delivery-models",
        )
        process, gate, staging = definition["Steps"]
        self.assertEqual(process["Arguments"]["AppSpecification"]["ImageUri"], IMAGE)
        model = gate["Arguments"]["IfSteps"][0]["Arguments"]
        self.assertEqual(model["InferenceSpecification"]["Containers"][0]["Image"], IMAGE)
        self.assertEqual(gate["Arguments"]["Conditions"][0]["RightValue"], True)
        self.assertEqual(gate["Arguments"]["ElseSteps"], [])
        self.assertEqual(staging["Arguments"]["IfSteps"][0]["Type"], "Fail")
        self.assertEqual(
            staging["Arguments"]["Conditions"][1]["LeftValue"],
            {"Get": "Parameters.RequireApproval"},
        )

    def test_s3_snapshot_is_recorded_and_bundle_serves_with_sagemaker_routes(self) -> None:
        source = SAMPLE.read_bytes()
        stream = io.BytesIO(source)
        with TemporaryDirectory() as directory, patch("mlops_demo.aws.processing.boto3.client") as factory:
            client = factory.return_value
            client.get_object.return_value = {"Body": stream, "VersionId": "snapshot-v3", "ETag": '"etag"'}
            output = Path(directory)
            evaluation = process_s3(SOURCE, output, seed=42, max_mae=5)

            client.get_object.assert_called_once_with(Bucket="delivery-bucket", Key="datasets/deliveries.csv")
            self.assertTrue(stream.closed)
            self.assertTrue(evaluation.passed)
            manifest = json.loads((output / "lineage" / "manifest.json").read_text())
            self.assertEqual(manifest["source_version_id"], "snapshot-v3")
            self.assertEqual(manifest["holdout_rows"], 18)
            bundle = output / "model"
            with tarfile.open(bundle / "model.tar.gz") as archive:
                self.assertEqual(set(archive.getnames()), {"model.json", "evaluation.json"})
                archive.extractall(bundle, filter="data")
            model = load_approved_model(bundle / "model.json", bundle / "evaluation.json")
            with TestClient(create_batch_app(model, health_path="/ping", prediction_path="/invocations")) as server:
                self.assertEqual(server.get("/ping").status_code, 200)
                response = server.post(
                    "/invocations", json={"instances": [{"distance_km": 12.5, "package_count": 3}]}
                )
                self.assertEqual(response.status_code, 200)
                self.assertAlmostEqual(response.json()["predictions"][0]["delivery_minutes"], 32.084, places=3)


class ReleaseTest(unittest.TestCase):
    def client(self, service: str):
        client = boto3.client(
            service, region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
        )
        self.addCleanup(client.close)
        return client

    def publish(self, result: str, *, first: bool = False) -> dict:
        client = self.client("sagemaker")
        definition = {"Version": "2020-12-01", "Steps": []}
        document = json.dumps(definition, separators=(",", ":"), sort_keys=True)
        version = 1 if first else 7
        with Stubber(client) as stub:
            update = {"PipelineName": PIPELINE_NAME, "RoleArn": ROLE, "PipelineDefinition": document}
            if first:
                stub.add_client_error("describe_pipeline", service_error_code="ResourceNotFound")
                stub.add_response("create_pipeline", {"PipelineArn": PIPELINE}, {**update, "ClientRequestToken": ANY})
                stub.add_response(
                    "list_pipeline_versions",
                    {"PipelineVersionSummaries": [{"PipelineVersionId": version, "PipelineArn": PIPELINE}]},
                    {"PipelineName": PIPELINE_NAME, "MaxResults": 1, "SortOrder": "Descending"},
                )
                stub.add_response(
                    "describe_pipeline", {"PipelineDefinition": document},
                    {"PipelineName": PIPELINE_NAME, "PipelineVersionId": version},
                )
            else:
                stub.add_response("describe_pipeline", {"PipelineArn": PIPELINE}, {"PipelineName": PIPELINE_NAME})
                stub.add_response("update_pipeline", {"PipelineArn": PIPELINE, "PipelineVersionId": 7}, update)
            stub.add_response(
                "start_pipeline_execution",
                {"PipelineExecutionArn": EXECUTION},
                {
                    "PipelineName": PIPELINE_NAME,
                    "PipelineVersionId": version,
                    "PipelineParameters": [
                        {"Name": "SourceUri", "Value": STAGING},
                        {"Name": "ModelGroup", "Value": "delivery-validation"},
                        {"Name": "RequireApproval", "Value": "true"},
                    ],
                    "ClientRequestToken": ANY,
                },
            )
            stub.add_response(
                "describe_pipeline_execution", {"PipelineExecutionStatus": result}, {"PipelineExecutionArn": EXECUTION}
            )
            try:
                return deploy_and_validate(
                    client,
                    pipeline_name=PIPELINE_NAME,
                    pipeline_role=ROLE,
                    definition=definition,
                    staging_source=STAGING,
                    staging_group="delivery-validation",
                )
            finally:
                stub.assert_no_pending_responses()

    def test_first_pipeline_is_validated_at_its_initial_version(self) -> None:
        self.assertEqual(self.publish("Succeeded", first=True)["pipeline_version"], 1)

    def test_staging_failure_cannot_return_a_release(self) -> None:
        with self.assertRaises(WaiterError):
            self.publish("Failed")

    def test_scheduler_targets_the_exact_validated_version(self) -> None:
        release = self.publish("Succeeded")
        client = self.client("scheduler")
        with Stubber(client) as stub:
            stub.add_client_error("get_schedule", service_error_code="ResourceNotFoundException")
            stub.add_response(
                "create_schedule",
                {"ScheduleArn": "arn:aws:scheduler:us-east-1:111122223333:schedule/default/delivery-retrain"},
                {
                    "Name": "delivery-retrain",
                    "ScheduleExpression": "rate(1 day)",
                    "ScheduleExpressionTimezone": "Asia/Seoul",
                    "State": "ENABLED",
                    "FlexibleTimeWindow": {"Mode": "OFF"},
                    "ClientToken": ANY,
                    "Target": {
                        "Arn": "arn:aws:scheduler:::aws-sdk:sagemaker:startPipelineExecution",
                        "RoleArn": ROLE,
                        "Input": json.dumps({
                            "PipelineName": PIPELINE_NAME,
                            "PipelineVersionId": 7,
                            "ClientRequestToken": "scheduled-retraining-<aws.scheduler.scheduled-time>",
                        }),
                        "RetryPolicy": {"MaximumRetryAttempts": 2, "MaximumEventAgeInSeconds": 300},
                    },
                },
            )
            schedule_validated_version(
                client, name="delivery-retrain", expression="rate(1 day)",
                timezone="Asia/Seoul", scheduler_role=ROLE, release=release,
            )
            stub.assert_no_pending_responses()

    def test_older_release_cannot_replace_the_schedule_of_a_newer_version(self) -> None:
        client = Mock()
        client.get_schedule.return_value = {
            "Target": {"Input": json.dumps({"PipelineName": PIPELINE_NAME, "PipelineVersionId": 8})}
        }
        with self.assertRaisesRegex(ValueError, "newer validated version"):
            schedule_validated_version(
                client, name="delivery-retrain", expression="rate(1 day)",
                timezone="Asia/Seoul", scheduler_role=ROLE,
                release={"pipeline_name": PIPELINE_NAME, "pipeline_arn": PIPELINE, "pipeline_version": 7},
            )
        client.update_schedule.assert_not_called()
