import json
import unittest
from pathlib import Path

from app.services.prediction_service import (
    PredictionInputError,
    PredictionService,
    REQUIRED_INPUT_FIELDS,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_MODEL_PATH = ROOT_DIR / "models" / "artifacts" / "test_pipeline.joblib"
FIXTURE_PATH = ROOT_DIR / "data" / "fixtures" / "loan_application_valid.json"


class TestPredictionService(unittest.TestCase):
    def setUp(self):
        self.service = PredictionService(model_path=TEST_MODEL_PATH)
        with FIXTURE_PATH.open("r", encoding="utf-8") as f:
            self.payload = json.load(f)

    def test_model_loads(self):
        model = self.service.load()
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_predict_one_returns_expected_shape(self):
        result = self.service.predict_one(self.payload)
        output = result.to_dict()

        self.assertIn("request_id", output)
        self.assertIn("prediction", output)
        self.assertIn("default_probability", output)
        self.assertIn("model_version", output)
        self.assertIn(output["prediction"], [0, 1])
        self.assertGreaterEqual(output["default_probability"], 0.0)
        self.assertLessEqual(output["default_probability"], 1.0)

    def test_missing_required_field_raises_controlled_error(self):
        payload = dict(self.payload)
        payload.pop(REQUIRED_INPUT_FIELDS[0])

        with self.assertRaises(PredictionInputError):
            self.service.predict_one(payload)


if __name__ == "__main__":
    unittest.main()

