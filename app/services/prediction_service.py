"""Prediction service for the existing sklearn credit-risk pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd

from app.core.config import (
    DEFAULT_MODEL_VERSION,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
)


REQUIRED_INPUT_FIELDS = [
    "State",
    "BankState",
    "NAICS",
    "NewExist",
    "UrbanRural",
    "RevLineCr",
    "LowDoc",
    "FranchiseCode",
    "GrAppv",
    "SBA_Appv",
    "Term",
    "NoEmp",
    "ApprovalDate",
    "ApprovalFY",
]


class PredictionInputError(ValueError):
    """Raised when input cannot be converted into model-ready raw features."""


@dataclass(frozen=True)
class PredictionResult:
    request_id: str
    prediction: int
    default_probability: float
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prediction": self.prediction,
            "default_probability": self.default_probability,
            "model_version": self.model_version,
        }


def _read_model_version(metadata_path: Path = MODEL_METADATA_PATH) -> str:
    if not metadata_path.exists():
        return DEFAULT_MODEL_VERSION

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return str(metadata.get("model_version", DEFAULT_MODEL_VERSION))


@lru_cache(maxsize=1)
def load_model(model_path: str | Path = MODEL_ARTIFACT_PATH) -> Any:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def _to_dataframe(raw_input: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw_input, pd.DataFrame):
        df = raw_input.copy()
    elif isinstance(raw_input, dict):
        df = pd.DataFrame([raw_input])
    else:
        raise PredictionInputError("Prediction input must be a dict or pandas DataFrame.")

    missing = [field for field in REQUIRED_INPUT_FIELDS if field not in df.columns]
    if missing:
        raise PredictionInputError(f"Missing required fields: {missing}")

    return df


class PredictionService:
    def __init__(
        self,
        model_path: str | Path = MODEL_ARTIFACT_PATH,
        metadata_path: str | Path = MODEL_METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

    @property
    def model_version(self) -> str:
        return _read_model_version(self.metadata_path)

    def load(self) -> Any:
        return load_model(self.model_path)

    def predict_one(self, raw_input: dict[str, Any]) -> PredictionResult:
        df = _to_dataframe(raw_input)
        model = self.load()

        prediction = int(model.predict(df)[0])
        default_probability = float(model.predict_proba(df)[0][1])

        if not 0.0 <= default_probability <= 1.0:
            raise RuntimeError("Model returned an invalid probability.")

        return PredictionResult(
            request_id=str(uuid4()),
            prediction=prediction,
            default_probability=default_probability,
            model_version=self.model_version,
        )

    def health(self) -> dict[str, str]:
        self.load()
        return {
            "status": "healthy",
            "model_version": self.model_version,
        }


def predict_one(raw_input: dict[str, Any]) -> dict[str, Any]:
    return PredictionService().predict_one(raw_input).to_dict()

