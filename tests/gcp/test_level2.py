import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml
from kfp.dsl import Artifact, Dataset, Metrics, Model

from mlops_demo.gcp.pipeline import compile_pipeline
from mlops_demo.gcp.release import publish_pipeline, validate_template
from mlops_demo.schemas import Evaluation

IMAGE = f"us-central1-docker.pkg.dev/example-project/mlops/runner@sha256:{'a' * 64}"
REPOSITORY = "https://us-central1-kfp.pkg.dev/example-project/pipelines"
PACKAGE = "delivery-time-continuous-training"


class ReleaseTest(unittest.TestCase):
    def test_publish_returns_an_immutable_uri_for_the_compiled_template(self) -> None:
        with TemporaryDirectory() as directory, patch("mlops_demo.gcp.release.RegistryClient") as registry:
            output = Path(directory) / "pipeline.yaml"

            def upload(*, file_name: str, tags: list[str]) -> tuple[str, str]:
                self.assertEqual(tags, ["git-release"])
                source = Path(file_name).read_bytes()
                spec = yaml.safe_load(source)
                for executor in spec["deploymentSpec"]["executors"].values():
                    self.assertEqual(executor["container"]["image"], IMAGE)
                return PACKAGE, f"sha256:{sha256(source).hexdigest()}"

            registry.return_value.upload_pipeline.side_effect = upload
            uri = publish_pipeline(IMAGE, REPOSITORY, "git-release", output)
            expected = sha256(output.read_bytes()).hexdigest()
            self.assertEqual(uri, f"{REPOSITORY}/{PACKAGE}/sha256:{expected}")
            self.assertEqual(validate_template(uri), uri)
            with self.assertRaisesRegex(ValueError, "digest-pinned"):
                validate_template(f"{REPOSITORY}/{PACKAGE}/latest")

    def test_required_approval_fails_the_staging_run_on_rejection(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "pipeline.yaml"
            compile_pipeline(IMAGE, output)
            spec = yaml.safe_load(output.read_text(encoding="utf-8"))
        runtime_code = spec["deploymentSpec"]["executors"]["exec-evaluate-model"]["container"]["command"][-1]
        component = {}
        exec(runtime_code, component)
        evaluate = component["evaluate_model"]
        rejected = Evaluation(
            model_sha256="a" * 64, holdout_rows=18, mae=8, baseline_mae=20, max_mae=5, passed=False
        )
        arguments = {
            "prepared": Dataset(uri="gs://example/prepared"),
            "candidate": Model(uri="gs://example/candidate"),
            "max_mae": 5,
            "report": Artifact(uri="gs://example/report"),
            "metrics": Metrics(),
        }
        with patch("mlops_demo.gcp.steps.evaluate_candidate", return_value=rejected):
            self.assertFalse(evaluate(**arguments, require_approval=False))
            with self.assertRaisesRegex(ValueError, "quality gate"):
                evaluate(**arguments, require_approval=True)
