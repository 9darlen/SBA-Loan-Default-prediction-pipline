# Acceptance Criteria

- Prediction service loads a configured model artifact.
- Prediction service exposes a Python prediction function.
- Prediction output includes `request_id`, `prediction`, `default_probability`, and `model_version`.
- `default_probability` is between `0` and `1`.
- Missing required fields produce a controlled validation error.
- Regression fixture verifies stable output for a representative input.
- Existing feature-engineering tests still pass.
