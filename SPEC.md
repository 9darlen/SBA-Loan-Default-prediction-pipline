# SBA Credit Risk Prediction Platform

> Spec-Driven Development / AI-DLC Project Specification

---

## 0. Document Metadata

```yaml
project_name: sba-credit-risk-platform
project_type: machine-learning-serving-platform
development_method: spec-driven-development
delivery_method: AI-DLC
status: active
spec_version: 0.2.0
primary_language: Python
current_priority: UNIT-002-prediction-api
```

## Source of Truth

This specification describes the intended product and engineering roadmap for the current repository.

When this document conflicts with the existing working model behavior, preserve the existing working behavior and record the conflict in specs or ADRs before changing behavior.

AI coding agents must:

1. Read this specification before modifying product behavior.
2. Implement only functionality covered by an approved Unit.
3. Prefer the simplest architecture that satisfies the current Unit.
4. Never introduce infrastructure only because it may be useful later.
5. Add or update tests whenever behavior changes.
6. Update this specification when requirements or architecture decisions change.
7. Never silently change model behavior, preprocessing logic, API contracts, or data schemas.

---

## 1. Project Intent

## INTENT-001 - Productionize the Existing SBA Loan Default Model

The repository already contains:

- an sklearn pipeline artifact used for default-risk prediction;
- custom feature engineering in `training/features/feature_builder.py`;
- a training script in `training/pipelines/train_pipeline.py`;
- a Streamlit UI in `app/page.py`;
- basic feature-engineering unit tests.

The next objective is to make prediction available as a reusable service boundary:

```text
Raw loan application input
  -> validation
  -> existing sklearn pipeline
  -> predicted class
  -> default probability
  -> versioned response
```

The project is a portfolio and learning project. Predictions must not be represented as production lending decisions.

---

## 2. Product Vision

The system should become both:

1. a working SBA loan default prediction platform;
2. a learning project demonstrating practical ML engineering and incremental MLOps.

The architecture evolves incrementally:

```text
Existing Model Artifact
  -> Prediction Module
  -> FastAPI
  -> Docker
  -> Persistence
  -> CI/CD
  -> Cloud Deployment
  -> Logging / Monitoring
  -> Advanced Model Lifecycle
```

Do not begin Kubernetes, microservices, MLflow, or cloud infrastructure until earlier Units are complete and a specific ADR approves the change.

---

## 3. Primary Users

## USER-001 - API Consumer

A developer or system that wants to request a loan default prediction.

The initial API uses the raw feature names expected by the existing production preprocessing pipeline:

```json
{
  "State": "CA",
  "BankState": "CA",
  "NAICS": "236115",
  "NewExist": 1.0,
  "UrbanRural": 1,
  "RevLineCr": "N",
  "LowDoc": "N",
  "FranchiseCode": "0",
  "GrAppv": 50000,
  "SBA_Appv": 40000,
  "Term": 60,
  "NoEmp": 10,
  "ApprovalDate": "01-Jan-06",
  "ApprovalFY": 2006
}
```

Example response:

```json
{
  "request_id": "uuid",
  "prediction": 0,
  "default_probability": 0.15,
  "model_version": "1.0.0"
}
```

---

## 4. Current Repository Baseline

## Existing Runtime

`app/page.py` provides a Streamlit UI that loads `models/artifacts/best_pipeline.joblib` and calls the sklearn pipeline directly.

## Existing Model Artifacts

- `models/artifacts/best_pipeline.joblib`: existing primary pipeline artifact, intentionally ignored by Git because of size.
- `models/artifacts/test_pipeline.joblib`: small tracked smoke-test artifact.
- `models/artifacts/sba_xgboost_model.json`: existing XGBoost experiment artifact.

The primary artifact depends on the historical `feature_builder.FeatureBuilder` module path. Keep the root compatibility shim unless a future Unit replaces the artifact intentionally.

## Existing Training Flow

```text
data/raw/SBAnational.csv
  -> FeatureBuilder
  -> encoding / scaling / feature selection
  -> RandomForestClassifier
  -> models/artifacts/*.joblib
```

## Existing CI

GitHub Actions performs dependency installation, feature-engineering tests, training smoke test, syntax check, and artifact upload for the test pipeline.

---

## 5. Project Scope

## Phase 1 - Model Packaging

Goal:

> Make the existing model reusable outside the Streamlit UI without retraining.

Required:

- load persisted model without retraining;
- isolate prediction logic in an application service;
- expose a Python prediction function;
- return prediction, default probability, request ID, and model version;
- validate probability bounds;
- add model regression coverage for a fixed input.

## Phase 2 - API Serving

Goal:

> Expose model inference through an HTTP API.

Technology:

```text
FastAPI
Pydantic
Uvicorn
```

Required endpoints:

```text
POST /predict
GET /health
GET /version
```

## Phase 3 - Containerization

Goal:

> Make the application reproducible across machines.

Not part of UNIT-001 or UNIT-002.

## Phase 4 - Persistence

Goal:

> Store prediction metadata.

Not part of UNIT-001 or UNIT-002.

---

## 6. Functional Requirements

## FR-001 - Model Loading

The application must load a persisted ML model without retraining during application startup or prediction service initialization.

## FR-002 - Python Prediction Function

The system must expose a Python prediction function that accepts one raw loan application dictionary and returns prediction output.

## FR-003 - Prediction API

The system must expose:

```text
POST /predict
```

## FR-004 - Input Validation

API requests must be validated before inference. Invalid requests must return a client error and must not crash the service.

## FR-005 - Prediction Probability

Responses must include:

```text
prediction
default_probability
```

`default_probability` must satisfy:

```text
0 <= default_probability <= 1
```

## FR-006 - Health Endpoint

The service must expose:

```text
GET /health
```

The endpoint should verify that the configured model can be loaded.

## FR-007 - Version Endpoint

The service must expose:

```text
GET /version
```

Every prediction should include the active model version.

---

## 7. Model Requirements

## MODEL-001 - Reproducible Preprocessing

Production preprocessing must use the existing sklearn pipeline artifact so training and inference preprocessing do not diverge.

## MODEL-002 - Artifact Safety

Do not reserialize or retrain the primary model merely to support API work. Use the existing joblib artifact.

## MODEL-003 - Regression Protection

A fixed input fixture must produce stable prediction fields. Any future intentional model change must update regression expectations through a planned Unit.

---

## 8. API Contract

## Prediction Request

The initial request schema uses existing raw feature names:

```text
State
BankState
NAICS
NewExist
UrbanRural
RevLineCr
LowDoc
FranchiseCode
GrAppv
SBA_Appv
Term
NoEmp
ApprovalDate
ApprovalFY
```

The client should not one-hot encode, standardize, target encode, or generate engineered columns.

## Prediction Response

```json
{
  "request_id": "uuid",
  "prediction": 0,
  "default_probability": 0.1734,
  "model_version": "1.0.0"
}
```

## Error Response

Errors must not reveal stack traces, credentials, or internal filesystem details.

---

## 9. Non-Functional Requirements

## NFR-001 - Reproducibility

The application must run from source and declared dependencies.

## NFR-002 - Reliability

Malformed prediction requests must not terminate the API process.

## NFR-003 - Maintainability

Business logic, API routing, request schemas, and model inference should remain separated.

## NFR-004 - Observability

Initial implementation should include request IDs in prediction responses. Structured logging is future work.

## NFR-005 - Security

Do not commit secrets. Do not expose internal file paths in API error responses.

## NFR-006 - Performance

Initial API p95 latency target is TBD until benchmarking exists.

---

## 10. Architecture

## Version 1 - Current Target

```text
Python caller
  -> PredictionService
  -> sklearn Pipeline
  -> PredictionResult
```

## Version 2 - UNIT-002 Target

```text
Client
  -> FastAPI
  -> Pydantic schema
  -> PredictionService
  -> sklearn Pipeline
```

## Version 3 - Future

```text
Client
  -> Containerized API
  -> PredictionService
  -> Model Artifact
  -> Optional persistence
```

---

## 11. AI-DLC Units

## UNIT-001 - Package Existing Model

Intent:

> Make the trained credit-risk model reusable without retraining.

Tasks:

- create model metadata;
- isolate model loading and prediction service;
- create a Python `predict_one()` capability;
- create tests for model loading and probability bounds;
- create a regression fixture.

Exit criteria:

```text
Python dict -> predict_one() -> stable prediction response
```

## UNIT-002 - Create Prediction API

Depends on:

```text
UNIT-001
```

Tasks:

- create FastAPI application;
- define Pydantic request and response schemas;
- implement `/predict`;
- implement `/health`;
- implement `/version`;
- implement safe error handling;
- add API tests.

Exit criteria:

```text
POST /predict returns a valid prediction response.
GET /health returns healthy when model loads.
```

---

## 12. Out of Scope for UNIT-001 and UNIT-002

- Docker
- PostgreSQL
- cloud deployment
- Kubernetes
- microservices
- MLflow
- model retraining
- online learning
- changing feature engineering
- changing thresholds, class weights, or hyperparameters

---

## 13. Definition of Done

A Unit is done only when:

- implementation satisfies this specification;
- relevant automated tests exist;
- relevant tests pass;
- no credentials are committed;
- application remains runnable;
- README/specs are updated when usage changes;
- important decisions are documented in ADRs where appropriate;
- changes are committed and pushed.

---

## 14. Current Priority

```text
UNIT-002 Create Prediction API
```

Do not begin Docker, database, cloud, MLflow, microservices, or Kubernetes work yet.
