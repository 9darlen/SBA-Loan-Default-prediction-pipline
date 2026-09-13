import json
import unittest
from pathlib import Path

from app.services.prediction_service import PredictionService


ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_MODEL_PATH = ROOT_DIR / "models" / "artifacts" / "test_pipeline.joblib"
FIXTURE_PATH = ROOT_DIR / "data" / "fixtures" / "loan_application_valid.json"


class TestModelRegression(unittest.TestCase):
    def test_fixture_prediction_is_stable_for_test_artifact(self):
        with FIXTURE_PATH.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        result = PredictionService(model_path=TEST_MODEL_PATH).predict_one(payload)

        self.assertEqual(result.prediction, 1)
        self.assertAlmostEqual(result.default_probability, 0.8992022532399504, places=6)
        self.assertEqual(result.model_version, "1.0.0")


if __name__ == "__main__":
    unittest.main()
