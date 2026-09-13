# SBA Loan Default Prediction

This repository contains a machine-learning project for predicting SBA loan charge-off risk.

The current implementation trains a scikit-learn pipeline from the SBA dataset, stores model artifacts, and serves predictions through a Streamlit application.

## Architecture Summary

- Training code lives in `training/`.
- Runtime Streamlit application code lives in `app/`.
- Serialized model artifacts live in `models/artifacts/`.
- Raw and processed data live in `data/`.
- Tests live in `tests/`.
- Project specifications and architecture records live in `specs/`.

## Repository Structure

```text
app/                       Streamlit inference UI
training/features/         Feature engineering transformers
training/pipelines/        Model training pipeline
models/artifacts/          Existing serialized model artifacts
data/raw/                  Raw SBA dataset and source documentation
data/processed/            Processed export files
notebooks/                 Exploratory notebooks
reports/analysis/          Existing model analysis images
tests/unit/                Unit tests
specs/                     Product, requirement, architecture, ADR, and Unit docs
```

## Local Setup

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

## Training

```powershell
python training\pipelines\train_pipeline.py
```

Smoke-test mode:

```powershell
python training\pipelines\train_pipeline.py --mode=test
```

## Inference

```powershell
streamlit run app\page.py
```

The app loads `models/artifacts/best_pipeline.joblib`.

## Testing

```powershell
python tests\unit\test_pipeline.py
python -m py_compile app\page.py feature_builder.py training\features\feature_builder.py training\pipelines\train_pipeline.py tests\unit\test_pipeline.py
```

## Current Development Unit

`UNIT-000 Repository Restructuring`

See `specs/units/UNIT-000-repository-restructuring/`.

## Roadmap

Future work should be described as Units under `specs/units/` before implementation.
