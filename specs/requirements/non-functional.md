# Non-Functional Requirements

## Maintainability

Training, runtime application code, model artifacts, tests, and documentation should be separated by responsibility.

## Reproducibility

The project uses `requirements.txt` and Python 3.11 for local setup. Exact model reproducibility beyond the existing training script is TBD.

## Compatibility

Existing serialized joblib artifacts must remain loadable after repository restructuring.

## Testability

Feature engineering behavior should remain covered by unit tests. Broader regression tests are TBD.

## Security

No explicit production security controls exist. Any future public deployment requires a dedicated security review.

## Performance

Specific latency and throughput targets are TBD.
