import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient
from google.api_core.exceptions import Conflict, NotFound, PreconditionFailed

from mlops_demo.gcp.__main__ import main
from mlops_demo.gcp.pipeline import compile_pipeline
from mlops_demo.gcp.steps import (
    evaluate_candidate,
    prepare_dataset,
    register_approved_model,
    train_candidate,
)
from mlops_demo.schemas import LinearModel
from mlops_demo.serving import create_app

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "deliveries.csv"
IMAGE = f"us-central1-docker.pkg.dev/example-project/mlops/runner@sha256:{'a' * 64}"
MODEL = "projects/example-project/locations/us-central1/models/delivery-time"


class PipelineTest(unittest.TestCase):
    def test_compiled_pipeline_connects_artifacts_to_the_quality_gate(self) -> None:
        with TemporaryDirectory() as directory:
            template = Path(directory) / "pipeline.yaml"
            compile_pipeline(IMAGE, template)
            spec = yaml.safe_load(template.read_text(encoding="utf-8"))

        tasks = spec["root"]["dag"]["tasks"]
        self.assertFalse(tasks["prepare-data"]["cachingOptions"].get("enableCache", False))
        self.assertEqual(set(tasks["evaluate-model"]["dependentTasks"]), {"prepare-data", "train-model"})
        gate = next(task for task in tasks.values() if "triggerPolicy" in task)
        self.assertIn("evaluate-model-Output", gate["triggerPolicy"]["condition"])
        self.assertTrue(gate["triggerPolicy"]["condition"].endswith("== true"))
        self.assertEqual(
            set(spec["components"][gate["componentRef"]["name"]]["dag"]["tasks"]),
            {"register-model"},
        )
        for executor in spec["deploymentSpec"]["executors"].values():
            self.assertEqual(executor["container"]["image"], IMAGE)

    def test_schedule_uses_the_same_job_parameters_and_runtime_identity(self) -> None:
        arguments = [
            "mlops_demo.gcp", "schedule", "--template", "pipeline.yaml",
            "--project", "example-project", "--region", "us-central1",
            "--source-uri", "gs://demo/input.csv", "--pipeline-root", "gs://demo/runs",
            "--service-account", "runner@example-project.iam.gserviceaccount.com",
            "--name", "delivery-time-daily", "--cron", "TZ=Asia/Seoul 0 3 * * *",
        ]
        with (
            patch("sys.argv", arguments),
            patch("mlops_demo.gcp.__main__.aiplatform.PipelineJob") as pipeline_job,
            patch("mlops_demo.gcp.__main__.aiplatform.PipelineJobSchedule") as schedules,
            redirect_stdout(StringIO()),
        ):
            schedules.list.return_value = []
            main()

        parameters = pipeline_job.call_args.kwargs
        self.assertEqual(parameters["parameter_values"]["source_uri"], "gs://demo/input.csv")
        self.assertFalse(parameters["enable_caching"])
        pipeline_job.return_value.submit.assert_not_called()
        pipeline_job.return_value.create_schedule.assert_called_once_with(
            display_name="delivery-time-daily",
            cron="TZ=Asia/Seoul 0 3 * * *",
            service_account="runner@example-project.iam.gserviceaccount.com",
            max_concurrent_run_count=1,
            max_concurrent_active_run_count=1,
            allow_queueing=False,
        )


class CloudStepsTest(unittest.TestCase):
    source = "gs://demo/data.csv"
    prepared = "gs://demo/run/prepared"
    candidate = "gs://demo/run/candidate"
    report = "gs://demo/run/report"
    bundle = "gs://demo/run/registered"

    def setUp(self) -> None:
        self.objects = {self.source: SAMPLE.read_bytes()}
        self.download_preconditions = []
        self.enterContext(patch("mlops_demo.gcp.steps.storage.Client"))
        self.enterContext(patch("mlops_demo.gcp.gcs.storage.Blob.from_uri", side_effect=self.blob))

    def blob(self, uri: str, *, client) -> SimpleNamespace:
        blob = SimpleNamespace(generation=None)

        def reload():
            blob.generation = 17

        def download(path, *, if_generation_match):
            self.download_preconditions.append((uri, if_generation_match))
            Path(path).write_bytes(self.objects[uri])

        def upload(path, *, if_generation_match):
            self.assertEqual(if_generation_match, 0)
            if uri in self.objects:
                raise PreconditionFailed("object exists")
            self.objects[uri] = Path(path).read_bytes()

        blob.reload = reload
        blob.download_to_filename = download
        blob.upload_from_filename = upload
        blob.download_as_bytes = lambda: self.objects[uri]
        return blob

    def register(self) -> dict[str, str]:
        return register_approved_model(
            self.candidate,
            self.report,
            self.bundle,
            IMAGE,
            project="example-project",
            region="us-central1",
            model_id="delivery-time",
        )

    def test_steps_reject_bad_model_and_register_valid_bundle_idempotently(self) -> None:
        metadata = prepare_dataset(self.source, self.prepared, seed=42)
        self.assertEqual(metadata["source_generation"], "17")
        self.assertIn((self.source, 17), self.download_preconditions)
        self.assertEqual(metadata["holdout_rows"], 18)
        train_candidate(self.prepared, self.candidate)

        rejected_report = self.report + "-rejected"
        evaluate_candidate(self.prepared, self.candidate, rejected_report, max_mae=0)
        with patch("mlops_demo.gcp.steps.aiplatform.Model") as registry:
            with self.assertRaisesRegex(ValueError, "failed its offline quality gate"):
                register_approved_model(
                    self.candidate,
                    rejected_report,
                    self.bundle,
                    IMAGE,
                    project="example-project",
                    region="us-central1",
                    model_id="delivery-time",
                )
            registry.upload.assert_not_called()
            self.assertNotIn(f"{self.bundle}/model.json", self.objects)

        assessment = evaluate_candidate(self.prepared, self.candidate, self.report, max_mae=5)
        self.assertLess(assessment.mae, assessment.baseline_mae)
        version = SimpleNamespace(resource_name=MODEL, versioned_resource_name=f"{MODEL}@1")
        with patch("mlops_demo.gcp.steps.aiplatform.Model") as registry:
            registry.side_effect = [NotFound("version"), NotFound("model"), version]
            registry.upload.return_value = version
            first = self.register()
            second = self.register()
            registry.upload.assert_called_once()
            self.assertEqual(registry.upload.call_args.kwargs["model_id"], "delivery-time")
            self.assertEqual(first, second)
            self.assertEqual(second["versioned_resource_name"], f"{MODEL}@1")
            self.assertEqual(registry.call_args.args[0], f"{MODEL}@{first['version_alias']}")
        self.assertEqual(self.objects[f"{self.bundle}/model.json"], self.objects[self.candidate])
        self.assertEqual(self.objects[f"{self.bundle}/evaluation.json"], self.objects[self.report])
        with TestClient(create_app(LinearModel.model_validate_json(self.objects[self.candidate]), vertex=True)) as client:
            response = client.post("/predict", json={"instances": [{"distance_km": 12.5, "package_count": 3}]})
            self.assertEqual(response.status_code, 200)
            self.assertAlmostEqual(response.json()["predictions"][0]["delivery_minutes"], 32.084, places=3)

    def test_registry_resolves_a_concurrent_first_upload(self) -> None:
        prepare_dataset(self.source, self.prepared, seed=42)
        train_candidate(self.prepared, self.candidate)
        evaluate_candidate(self.prepared, self.candidate, self.report, max_mae=5)
        version = SimpleNamespace(resource_name=MODEL, versioned_resource_name=f"{MODEL}@2")
        with patch("mlops_demo.gcp.steps.aiplatform.Model") as registry:
            registry.side_effect = [NotFound("version"), NotFound("model"), NotFound("version"), version]
            registry.upload.side_effect = [Conflict("model exists"), version]
            self.assertEqual(self.register()["versioned_resource_name"], f"{MODEL}@2")
            retry = registry.upload.call_args.kwargs
            self.assertEqual(retry["parent_model"], MODEL)
            self.assertFalse(retry["is_default_version"])
