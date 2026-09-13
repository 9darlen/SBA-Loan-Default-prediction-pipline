"""Application configuration for model serving."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_ARTIFACT_PATH = Path(
    os.environ.get(
        "CREDIT_RISK_MODEL_PATH",
        ROOT_DIR / "models" / "artifacts" / "best_pipeline.joblib",
    )
)
MODEL_METADATA_PATH = Path(
    os.environ.get(
        "CREDIT_RISK_MODEL_METADATA_PATH",
        ROOT_DIR / "models" / "metadata" / "model_metadata.json",
    )
)
DEFAULT_MODEL_VERSION = "1.0.0"

