# Target Architecture

The target architecture is incremental and preserves current behavior.

## Current Target

- Keep training code in `training/`.
- Keep Streamlit runtime code in `app/`.
- Keep serialized artifacts in `models/artifacts/`.
- Keep tests in `tests/`.
- Keep specifications and ADRs in `specs/`.

## Future Evolution

Potential future Units may add:

- Model packaging cleanup.
- Prediction API.
- Dockerization.
- Expanded CI.
- Regression tests for stable predictions.
- Model lifecycle documentation.

These future capabilities are not implemented by the repository restructuring Unit.
