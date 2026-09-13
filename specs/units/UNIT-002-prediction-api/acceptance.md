# Acceptance Criteria

- `POST /predict` accepts a valid raw loan application and returns a prediction response.
- Invalid request bodies return validation errors.
- `GET /health` returns healthy when the model service can load.
- `GET /version` returns the active model version.
- API tests pass.
- Existing service and feature-engineering tests still pass.
