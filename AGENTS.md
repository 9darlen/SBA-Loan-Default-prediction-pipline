# AI Agent Instructions

Future AI agents working in this repository must:

1. Read `RESTRUCTURE_SPEC.md` and relevant files under `specs/` before coding.
2. Read relevant ADRs before changing architecture.
3. Work on one Unit at a time.
4. Make minimal changes that preserve existing ML behavior.
5. Avoid changing feature engineering, preprocessing, thresholds, hyperparameters, or model artifacts unless explicitly requested.
6. Add or update tests for behavioral changes.
7. Do not introduce unapproved infrastructure or new architectural layers.
8. Update specs when system behavior changes.
9. Report assumptions and known uncertainty.
10. Stop and document architectural uncertainty instead of inventing architecture silently.

The existing trained artifacts may depend on historical Python module paths. Preserve compatibility shims unless a planned migration explicitly replaces the artifacts.
