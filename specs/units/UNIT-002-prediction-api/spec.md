# UNIT-002 - Prediction API

## Intent

Expose the packaged model through an HTTP API.

## Scope

- Create a FastAPI application.
- Define Pydantic schemas for raw loan input and prediction output.
- Implement `POST /predict`.
- Implement `GET /health`.
- Implement `GET /version`.
- Add API tests.

## Out of Scope

- Docker.
- Database persistence.
- Authentication.
- Cloud deployment.
- Changing model behavior.
