import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.routes import get_prediction_service
from app.main import create_app
from app.services.prediction_service import PredictionService


ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_MODEL_PATH = ROOT_DIR / "models" / "artifacts" / "test_pipeline.joblib"
FIXTURE_PATH = ROOT_DIR / "data" / "fixtures" / "loan_application_valid.json"


class TestPredictionAPI(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.dependency_overrides[get_prediction_service] = lambda: PredictionService(
            model_path=TEST_MODEL_PATH
        )
        self.client = TestClient(app)
        with FIXTURE_PATH.open("r", encoding="utf-8") as f:
            self.payload = json.load(f)

    def test_health_returns_healthy(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
        self.assertEqual(response.json()["model_version"], "1.0.0")

    def test_version_returns_model_version(self):
        response = self.client.get("/version")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"model_version": "1.0.0"})

    def test_predict_returns_prediction_response(self):
        response = self.client.post("/predict", json=self.payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("request_id", body)
        self.assertEqual(body["prediction"], 1)
        self.assertAlmostEqual(body["default_probability"], 0.8992022532399504, places=6)
        self.assertEqual(body["model_version"], "1.0.0")

    def test_invalid_request_returns_422(self):
        payload = dict(self.payload)
        payload.pop("State")

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

