# ADR-004 - Use FastAPI for Prediction API

## Status

Accepted

## Context

The next product Unit requires an HTTP API for prediction. The external platform specification names FastAPI, Pydantic, and Uvicorn as the initial API-serving stack.

## Decision

Use FastAPI for `UNIT-002 Create Prediction API`, with Pydantic schemas for request and response validation.

## Consequences

- `fastapi` and API-test dependencies must be added to `requirements.txt`.
- API behavior must remain a thin serving layer over the existing prediction service.
- This does not approve Docker, databases, cloud deployment, microservices, or Kubernetes.
