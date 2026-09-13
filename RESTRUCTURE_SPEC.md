# Repository Restructuring Specification

> Migration of an existing ML project into a Spec-Driven Development / AI-DLC repository structure.

---

# 0. Purpose

This repository already contains existing source code, machine-learning logic, scripts, models, configuration, tests, notebooks, or other project files.

The goal of this task is **NOT to rewrite the project**.

The goal is to:

1. inspect the existing repository;
2. understand the purpose of existing files;
3. preserve current behavior;
4. reorganize files into a maintainable enterprise-style repository;
5. introduce a Spec-Driven Development structure;
6. prepare the repository for future AI-DLC development;
7. avoid unnecessary architecture changes.

The existing implementation is the baseline.

---

# 1. Core Migration Rule

The restructuring MUST follow this principle:

```text
Understand
   ↓
Classify
   ↓
Plan
   ↓
Move
   ↓
Fix References
   ↓
Test
   ↓
Document
```

Do NOT perform:

```text
Scan files
   ↓
Guess architecture
   ↓
Rewrite everything
```

---

# 2. Source of Truth

During this migration:

```text
Existing working behavior
+
This restructuring specification
```

are the primary sources of truth.

If documentation conflicts with functioning implementation:

1. preserve functioning implementation;
2. record the conflict;
3. do not silently modify behavior;
4. propose a future Unit or ADR if the behavior should change.

---

# 3. AI Agent Instructions

Before modifying any file, the AI agent MUST inspect the repository.

The agent MUST first produce:

1. Current repository tree
2. Classification of existing files
3. Current application architecture
4. Current ML training flow
5. Current ML inference flow
6. Existing test structure
7. Existing infrastructure / CI configuration
8. Proposed destination of each important file
9. Files that should remain unchanged
10. Potential migration risks

Only after this analysis should restructuring begin.

---

# 4. Important Restrictions

The migration MUST NOT intentionally change:

```text
ML model behavior
feature engineering
preprocessing
model hyperparameters
decision thresholds
API behavior
business logic
training results
prediction results
database behavior
external interfaces
```

unless required to make file imports or paths continue working after relocation.

---

# 5. No Unnecessary Technologies

Do NOT introduce the following during repository restructuring unless they already exist:

```text
Kubernetes
Kafka
Redis
Terraform
MLflow
Airflow
Celery
Service Mesh
Microservices
GraphQL
Prometheus
Grafana
```

This task concerns:

> repository organization and documentation.

It is NOT an infrastructure modernization task.

---

# 6. Target Repository Architecture

The preferred target structure is:

```text
project-root/
│
├── README.md
├── AGENTS.md
├── RESTRUCTURE_SPEC.md
│
├── specs/
│   │
│   ├── product/
│   │   ├── vision.md
│   │   ├── scope.md
│   │   └── glossary.md
│   │
│   ├── requirements/
│   │   ├── functional.md
│   │   └── non-functional.md
│   │
│   ├── contracts/
│   │   ├── api/
│   │   │   └── openapi.yaml
│   │   └── data/
│   │       └── data-schema.md
│   │
│   ├── architecture/
│   │   ├── current.md
│   │   ├── target.md
│   │   ├── data-flow.md
│   │   └── deployment.md
│   │
│   ├── decisions/
│   │   ├── ADR-001-existing-model.md
│   │   ├── ADR-002-repository-structure.md
│   │   └── README.md
│   │
│   ├── units/
│   │   ├── UNIT-001-model-packaging/
│   │   │   ├── spec.md
│   │   │   ├── acceptance.md
│   │   │   ├── tasks.md
│   │   │   └── status.md
│   │   │
│   │   └── UNIT-002-api/
│   │       ├── spec.md
│   │       ├── acceptance.md
│   │       ├── tasks.md
│   │       └── status.md
│   │
│   └── policies/
│       ├── coding-standards.md
│       ├── testing-policy.md
│       ├── ai-agent-policy.md
│       └── security-policy.md
│
├── app/
│   ├── api/
│   ├── schemas/
│   ├── services/
│   ├── domain/
│   ├── infrastructure/
│   └── core/
│
├── models/
│   ├── artifacts/
│   ├── metadata/
│   └── schemas/
│
├── training/
│   ├── pipelines/
│   ├── features/
│   └── evaluation/
│
├── notebooks/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── e2e/
│
├── infra/
│   ├── docker/
│   └── environments/
│
├── scripts/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── fixtures/
│
└── .github/
    └── workflows/
```

---

# 7. Important Qualification

The target structure is a guideline, NOT a requirement to create empty folders.

Do NOT create folders that serve no current purpose.

For example:

If the project currently has no API:

```text
app/api/
```

does NOT need to be populated.

If there is no database:

```text
database-related structure
```

does NOT need to be invented.

If there is no Kubernetes configuration:

```text
infra/kubernetes/
```

MUST NOT be created.

Prefer:

> minimal structure based on actual existing project contents.

---

# 8. Existing File Classification

Each existing file should be classified into one of the following categories.

## A. Application Code

Runtime application code belongs under:

```text
app/
```

Examples:

```text
FastAPI routes
request schemas
prediction services
configuration
application utilities
```

---

## B. Training Code

ML training logic belongs under:

```text
training/
```

Possible structure:

```text
training/
├── pipelines/
├── features/
└── evaluation/
```

Examples:

```text
train_pipeline.py
feature_builder.py
hyperparameter tuning
model evaluation
feature selection
```

---

## C. Model Artifacts

Serialized model artifacts belong under:

```text
models/artifacts/
```

Examples:

```text
.pkl
.joblib
.onnx
```

Model metadata belongs under:

```text
models/metadata/
```

Examples:

```text
model version
training timestamp
evaluation metrics
feature metadata
```

---

## D. Tests

Tests should be classified by purpose.

```text
tests/
├── unit/
├── integration/
├── regression/
└── e2e/
```

### Unit

Test isolated functions.

### Integration

Test multiple components together.

Example:

```text
API → model
```

### Regression

Verify that existing model behavior does not unintentionally change.

### E2E

Test full workflows where applicable.

---

## E. Notebooks

Exploratory notebooks belong under:

```text
notebooks/
```

Notebooks MUST NOT become production runtime dependencies.

Reusable logic contained inside notebooks SHOULD eventually be moved into Python modules.

However:

> do not refactor notebook logic during this repository migration unless necessary.

---

## F. Infrastructure

Infrastructure-related configuration belongs under:

```text
infra/
```

Examples:

```text
Dockerfile
docker-compose.yml
deployment configuration
environment configuration
```

If Dockerfile currently exists at repository root, it MAY remain at root if that simplifies build tooling.

Do not move infrastructure files merely for aesthetic reasons if doing so breaks existing workflows.

---

## G. Scripts

One-off or operational scripts belong under:

```text
scripts/
```

Examples:

```text
data migration
model conversion
local setup
database initialization
utility scripts
```

---

# 9. ML-Specific Rules

This project contains machine-learning components.

Training and inference MUST remain conceptually separated.

Preferred direction:

```text
training/
   ↓
produces
   ↓
models/artifacts/
   ↓
consumed by
   ↓
app/services/
```

Runtime application code SHOULD NOT retrain the model.

---

# 10. Desired ML Boundary

Long-term architecture:

```text
Training
────────

Raw Data
   ↓
Feature Engineering
   ↓
Training Pipeline
   ↓
Evaluation
   ↓
Model Artifact


Serving
───────

Input
   ↓
Validation
   ↓
Production Preprocessing
   ↓
Model Artifact
   ↓
Prediction
```

During restructuring, identify where each existing component belongs in this flow.

---

# 11. Spec Structure

The `specs/` directory acts as the human + AI development knowledge base.

---

## specs/product/

Contains:

```text
WHY are we building this project?
```

### vision.md

Should describe:

- project purpose;
- long-term goal;
- expected users;
- learning / portfolio objective if applicable.

### scope.md

Should distinguish:

```text
In Scope
Out of Scope
Future Scope
```

---

# 12. Requirements

## specs/requirements/functional.md

Document behavior already supported by the repository.

Do NOT invent requirements.

Example:

```text
FR-001

The system shall load the existing trained ML model.

FR-002

The system shall accept valid model input and return a prediction.
```

---

## specs/requirements/non-functional.md

Document relevant existing or known targets.

Examples:

```text
reproducibility
maintainability
security
testability
performance
observability
```

Unknown values should be marked:

```text
TBD
```

instead of guessed.

---

# 13. Architecture Documentation

## specs/architecture/current.md

This file is critical.

It MUST describe what the repository currently does.

Include:

```text
Current modules
Training flow
Inference flow
External dependencies
Data flow
Model artifact flow
Existing deployment approach
```

The AI MUST derive this from actual code.

Do not describe an imaginary architecture.

---

## specs/architecture/target.md

Describe the desired evolution.

Example:

```text
Current
   ↓
Model packaging
   ↓
FastAPI
   ↓
Docker
   ↓
CI/CD
   ↓
Cloud
   ↓
Observability
   ↓
Future MLOps
```

Do not represent future architecture as if it already exists.

---

# 14. Architecture Decision Records

Use ADRs for significant technical decisions.

Format:

```text
# ADR-XXX — Decision Title

## Status

Accepted / Proposed / Deprecated / Superseded

## Context

Why does this decision exist?

## Decision

What was decided?

## Consequences

What are the trade-offs?
```

---

## Initial ADRs

Create ADRs only when supported by the current repository or this specification.

Suggested initial ADRs:

```text
ADR-001-preserve-existing-model-behavior.md

ADR-002-adopt-spec-driven-repository.md

ADR-003-incremental-modernization.md
```

---

# 15. AI-DLC Units

Units describe deliverable capabilities.

The migration itself is:

```text
UNIT-000
Repository Restructuring
```

Suggested future Units:

```text
UNIT-001
Model Packaging

UNIT-002
Prediction API

UNIT-003
Dockerization

UNIT-004
Persistence

UNIT-005
CI Pipeline

UNIT-006
Cloud Deployment

UNIT-007
Observability

UNIT-008
Model Lifecycle Management

UNIT-009
Microservices Experiment

UNIT-010
Kubernetes
```

Only Units relevant to the real project should be created.

---

# 16. UNIT-000 — Repository Restructuring

Create:

```text
specs/units/UNIT-000-repository-restructuring/
```

with:

```text
spec.md
acceptance.md
tasks.md
status.md
```

---

## UNIT-000 Intent

Reorganize the repository without changing application behavior.

---

## Acceptance Criteria

The migration is complete when:

```text
existing code remains available

existing model behavior is preserved

existing tests still work or are correctly relocated

imports are valid

paths are valid

existing CI remains operational where applicable

repository structure is understandable

architecture documentation reflects reality

future work is represented as Units

AI development rules exist
```

---

# 17. Migration Tasks

Suggested tasks:

```text
BOLT-000.1
Inventory repository

BOLT-000.2
Classify existing files

BOLT-000.3
Document current architecture

BOLT-000.4
Create minimal target directories

BOLT-000.5
Move source files

BOLT-000.6
Update imports and paths

BOLT-000.7
Reorganize tests

BOLT-000.8
Create architecture documentation

BOLT-000.9
Create initial ADRs

BOLT-000.10
Create future Unit skeletons

BOLT-000.11
Run full validation

BOLT-000.12
Update README
```

Each Bolt should be independently reviewable.

---

# 18. File Move Rules

Before moving a file, identify:

```text
current path
purpose
dependencies
import references
runtime references
CI references
configuration references
target path
```

Example migration table:

| Current | Purpose | Target | Action |
|---|---|---|---|
| `train_pipeline.py` | model training | `training/pipelines/train_pipeline.py` | Move |
| `feature_builder.py` | feature engineering | `training/features/feature_builder.py` | Move |
| `model.pkl` | model artifact | `models/artifacts/model.pkl` | Move |
| `test_pipeline.py` | regression test | `tests/regression/test_pipeline.py` | Move |
| `README.md` | repository documentation | `README.md` | Update |

The actual table MUST be generated from existing repository files.

---

# 19. Import Safety

After moving Python modules, verify all imports.

Example:

Before:

```python
from feature_builder import build_features
```

After:

```python
from training.features.feature_builder import build_features
```

Only modify imports when required by file movement.

Do NOT perform unrelated refactoring.

---

# 20. Path Safety

Pay special attention to code using relative file paths.

Examples:

```python
"./model.pkl"

"../data/train.csv"

"./config.json"
```

Moving files may silently break these references.

Identify all filesystem-dependent code before relocation.

Prefer robust path handling where required.

Example:

```python
from pathlib import Path
```

But avoid broad path refactoring unless necessary.

---

# 21. Data Safety

Data files require special handling.

Do NOT:

```text
delete data
rename datasets without reason
commit sensitive information
move large artifacts unnecessarily
```

Before reorganizing data, identify:

```text
raw data
processed data
training fixtures
temporary data
model outputs
```

If classification is uncertain:

> preserve current location and document the uncertainty.

---

# 22. Model Artifact Safety

Serialized models are potentially sensitive to:

```text
Python version
scikit-learn version
dependency version
file path
preprocessing implementation
```

Do not reserialize the model merely because it is being moved.

Move the existing artifact unless reserialization is explicitly required.

---

# 23. AGENTS.md

Create a root-level:

```text
AGENTS.md
```

It should instruct future AI agents to:

1. read relevant specifications before coding;
2. read relevant ADRs;
3. work on one Unit at a time;
4. make minimal changes;
5. preserve ML behavior;
6. add tests for behavioral changes;
7. not introduce unapproved infrastructure;
8. update specs when system behavior changes;
9. report assumptions;
10. stop and document architectural uncertainty rather than silently inventing architecture.

---

# 24. README.md

README should eventually contain:

```text
Project Overview

Architecture Summary

Repository Structure

Local Setup

Training

Inference

Testing

Current Development Unit

Roadmap
```

README should remain concise.

Detailed architecture belongs under:

```text
specs/
```

---

# 25. Preserve Git History

Whenever practical:

```text
move files
```

instead of:

```text
delete + recreate
```

This helps preserve Git history.

Avoid rewriting files solely to make formatting look cleaner.

---

# 26. Validation

After restructuring, perform all available validation.

At minimum inspect:

```text
Python imports
syntax
unit tests
existing pipeline tests
model loading
prediction behavior
Docker build if Docker exists
CI configuration if present
```

Compare relevant output before and after restructuring.

---

# 27. Regression Protection

If possible, select representative existing inputs.

Before migration:

```text
Input A
→ Prediction A
```

After migration:

```text
Input A
→ Prediction A
```

Results should remain equivalent.

If they differ:

> stop treating the migration as successful.

Document and investigate the difference.

---

# 28. Do Not Hide Problems

If existing repository problems are discovered, classify them.

Example:

```text
ISSUE-001

Model path is hardcoded.

Impact:
Moving model artifact may break inference.

Current action:
Preserve current behavior.

Future recommendation:
Create dedicated configuration layer.
```

Do not silently fix unrelated technical debt during repository restructuring.

---

# 29. Migration Report

When restructuring is complete, create:

```text
specs/architecture/migration-report.md
```

Include:

```text
Original structure

Final structure

Files moved

Files renamed

Imports changed

Configuration changed

Tests executed

Validation results

Known issues

Technical debt discovered

Recommended next Unit
```

---

# 30. Human Approval Gates

Stop and request human review before intentionally:

```text
changing model behavior

changing feature engineering

deleting files

changing API behavior

changing database schema

introducing new infrastructure

splitting application into microservices

changing model format

changing training data

changing prediction thresholds
```

Repository organization itself does not require approval unless behavior may change.

---

# 31. Desired Final Development Flow

After migration, future development should follow:

```text
Product Intent
      ↓
Requirement
      ↓
Architecture / ADR
      ↓
Unit
      ↓
Acceptance Criteria
      ↓
Bolt
      ↓
Implementation
      ↓
Tests
      ↓
Review
      ↓
Spec Update
```

---

# 32. First Task for the AI Agent

After reading this document:

## STEP 1

Inspect the entire current repository.

Do not modify files yet.

## STEP 2

Produce:

```text
A. Existing repository tree

B. File classification

C. Current architecture

D. Current training pipeline

E. Current inference pipeline

F. Existing testing approach

G. Existing deployment / CI approach

H. Proposed target repository tree

I. File-by-file migration map

J. Migration risks
```

## STEP 3

Create an implementation plan for:

```text
UNIT-000 Repository Restructuring
```

using small Bolts.

## STEP 4

Only after the plan is complete:

begin restructuring incrementally.

## STEP 5

After each meaningful migration:

run relevant validation.

## STEP 6

At completion:

produce:

```text
migration-report.md
```

---

# 33. First Instruction to Execute

Use the following as the immediate task:

```text
Read RESTRUCTURE_SPEC.md completely.

Inspect the existing repository before changing anything.

Your first task is UNIT-000: Repository Restructuring.

Do NOT implement new product features.
Do NOT modify ML behavior.
Do NOT introduce new infrastructure.

First produce:

1. the current repository tree;
2. classification of every important file;
3. current architecture;
4. current ML training and inference flow;
5. proposed target tree;
6. a file migration map;
7. migration risks;
8. a Bolt-by-Bolt restructuring plan.

Use the target architecture in RESTRUCTURE_SPEC.md as guidance,
but adapt it to the repository that actually exists.

Do not create unnecessary empty directories.

Preserve existing behavior, Git history, imports, model artifacts,
tests, configuration, and existing CI/CD behavior.

After presenting the analysis and plan, begin UNIT-000 incrementally.

Run relevant tests after file movements and fix only migration-related issues.

When complete, update README.md and create:

specs/architecture/migration-report.md
```

---

# 34. Success Criteria

The restructuring succeeds when:

```text
Existing project still works
          +
Repository is easier to understand
          +
Training / serving responsibilities are clear
          +
Specifications are discoverable
          +
Architecture decisions are recorded
          +
AI agents have explicit rules
          +
Future development can proceed Unit-by-Unit
```

The goal is NOT maximum folder complexity.

The goal is:

> minimum structure necessary for disciplined, scalable, AI-assisted development.