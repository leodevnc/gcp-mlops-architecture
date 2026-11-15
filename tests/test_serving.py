import unittest

from fastapi.testclient import TestClient

from mlops_demo.schemas import LinearModel
from mlops_demo.serving import create_app


class ServingTest(unittest.TestCase):
    def setUp(self) -> None:
        model = LinearModel(
            prepared_sha256="a" * 64,
            intercept=10,
            distance_coefficient=1.5,
            package_coefficient=2,
            baseline_minutes=45,
        )
        self.client = self.enterContext(TestClient(create_app(model)))

    def test_predict_uses_the_serving_feature_schema(self) -> None:
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        response = self.client.post("/predict", json={"distance_km": 12.5, "package_count": 3})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"delivery_minutes": 34.75})

    def test_out_of_range_numbers_return_a_validation_error(self) -> None:
        response = self.client.post(
            "/predict",
            content='{"distance_km":1e400,"package_count":3}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["loc"], ["body", "distance_km"])
