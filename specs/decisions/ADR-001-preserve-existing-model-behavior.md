# ADR-001 - Preserve Existing Model Behavior

## Status

Accepted

## Context

The repository already contains feature engineering code and serialized model artifacts.

## Decision

Repository restructuring must not intentionally change feature engineering, preprocessing, hyperparameters, thresholds, or prediction behavior.

## Consequences

Compatibility shims may remain in place when serialized artifacts depend on historical module paths.
