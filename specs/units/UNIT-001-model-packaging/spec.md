# UNIT-001 - Model Packaging

## Intent

Make the existing SBA credit-risk model reusable without retraining or changing model behavior.

## Scope

- Load the existing sklearn pipeline artifact.
- Add model metadata.
- Isolate inference logic in an application service.
- Accept one raw loan application dictionary.
- Return `request_id`, `prediction`, `default_probability`, and `model_version`.
- Add regression protection for a fixed raw input.

## Out of Scope

- Retraining.
- Reserializing the primary model.
- Changing `FeatureBuilder`.
- Adding an HTTP API.
- Docker or infrastructure.
