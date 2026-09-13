# Status

## State

Complete

## Notes

The primary artifact `models/artifacts/best_pipeline.joblib` is intentionally ignored by Git because of size. Tests use the tracked smoke-test artifact where necessary.

Implemented:

- model metadata;
- app configuration for model and metadata paths;
- prediction service with `predict_one()`;
- service tests;
- regression fixture and regression test.
