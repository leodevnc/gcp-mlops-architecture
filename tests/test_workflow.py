import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlops_demo.workflow import evaluate, load_approved_model, prepare, train

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "deliveries.csv"


class WorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.prepared = directory / "prepared.json"
        self.model = directory / "model.json"
        self.report = directory / "evaluation.json"

    def test_prepare_rejects_invalid_feature_values(self) -> None:
        source = self.prepared.with_suffix(".csv")
        source.write_text(
            "distance_km,package_count,delivery_minutes\n3.5,0,24\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "CSV line 2"):
            prepare(source, self.prepared, seed=42)
        self.assertFalse(self.prepared.exists())

    def test_evaluation_requires_the_original_split(self) -> None:
        prepare(SAMPLE, self.prepared, seed=42)
        train(self.prepared, self.model)
        prepare(SAMPLE, self.prepared, seed=43)
        with self.assertRaisesRegex(ValueError, "different prepared dataset"):
            evaluate(self.prepared, self.model, self.report, max_mae=5)
        self.assertFalse(self.report.exists())

    def test_serving_requires_a_passing_report_for_the_exact_model(self) -> None:
        data = prepare(SAMPLE, self.prepared, seed=42)
        model = train(self.prepared, self.model)

        rejected = evaluate(self.prepared, self.model, self.report, max_mae=0)
        self.assertFalse(rejected.passed)
        with self.assertRaisesRegex(ValueError, "failed its offline quality gate"):
            load_approved_model(self.model, self.report)

        approved = evaluate(self.prepared, self.model, self.report, max_mae=5)
        self.assertEqual(approved.holdout_rows, len(data.holdout))
        self.assertLess(approved.mae, approved.baseline_mae)
        self.assertEqual(load_approved_model(self.model, self.report), model)

        prepare(SAMPLE, self.prepared, seed=43)
        train(self.prepared, self.model)
        with self.assertRaisesRegex(ValueError, "different model artifact"):
            load_approved_model(self.model, self.report)
