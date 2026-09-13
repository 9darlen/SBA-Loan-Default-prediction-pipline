# Current Architecture

## Modules

- `training/features/feature_builder.py`: custom sklearn transformer for feature engineering.
- `training/pipelines/train_pipeline.py`: training entry point.
- `app/main.py`: FastAPI application entry point.
- `app/api/routes.py`: `/predict`, `/health`, and `/version` API routes.
- `app/schemas/prediction.py`: Pydantic request and response schemas.
- `app/page.py`: Streamlit runtime interface.
- `app/services/prediction_service.py`: reusable model loading and prediction service.
- `app/core/config.py`: application model-serving configuration.
- `feature_builder.py`: compatibility shim for existing serialized model artifacts.
- `tests/unit/test_pipeline.py`: unit tests for feature engineering behavior.

## Training Flow

1. Read `data/raw/SBAnational.csv`.
2. Drop rows without `MIS_Status`.
3. Convert `MIS_Status` to binary target where `CHGOFF` is `1`.
4. Split train/test data with stratification.
5. Apply `FeatureBuilder`.
6. Apply target encoding, one-hot encoding, scaling, and feature selection.
7. Train `RandomForestClassifier`.
8. Report AUC and feature importances.
9. Save pipeline artifact under `models/artifacts/`.

## Inference Flow

### Streamlit

1. Streamlit gathers single-loan form input or uploaded CSV rows.
2. The app loads `models/artifacts/best_pipeline.joblib`.
3. The serialized sklearn pipeline performs preprocessing and prediction.
4. The app displays default probability and simple risk level output.

### Prediction Service

1. Python caller passes a raw loan application dictionary.
2. `PredictionService` validates required raw feature fields.
3. The service loads the configured sklearn pipeline artifact.
4. The service returns `request_id`, `prediction`, `default_probability`, and `model_version`.

### FastAPI

1. Client sends raw loan application JSON to `POST /predict`.
2. Pydantic validates request fields.
3. API route calls `PredictionService`.
4. Response contains `request_id`, `prediction`, `default_probability`, and `model_version`.

Operational endpoints:

- `GET /health`
- `GET /version`

## External Dependencies

Dependencies are listed in `requirements.txt`, including pandas, scikit-learn, category_encoders, xgboost, matplotlib, and streamlit.

## Data Flow

Raw data is stored in `data/raw/`. Processed Power BI export data is stored in `data/processed/`. Analysis images are stored in `reports/analysis/`.

## Model Artifact Flow

Training produces `.joblib` artifacts in `models/artifacts/`. The Streamlit app consumes `best_pipeline.joblib` from that location.

## Deployment Approach

No production deployment configuration currently exists. The API can run locally with `uvicorn app.main:app`. CI runs Python checks through GitHub Actions.
