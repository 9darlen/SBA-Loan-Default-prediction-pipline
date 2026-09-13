# Functional Requirements

## FR-001 Load Existing Model Artifact

The application shall load the existing trained pipeline artifact from `models/artifacts/best_pipeline.joblib`.

## FR-002 Single Loan Prediction

The Streamlit application shall accept a single loan input through form controls and produce a default probability.

## FR-003 Batch CSV Prediction

The Streamlit application shall accept a CSV upload containing required model input columns and produce batch predictions.

## FR-004 Feature Engineering

The training and inference pipeline shall use the existing `FeatureBuilder` transformer behavior.

## FR-005 Training Pipeline

The training script shall read the existing SBA training data and produce a joblib pipeline artifact.

## FR-006 Unit Tests

The repository shall provide tests for core feature engineering behavior.
