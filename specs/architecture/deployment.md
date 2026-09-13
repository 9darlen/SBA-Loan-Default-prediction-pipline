# Deployment

## Current State

No production deployment configuration exists.

## Local Runtime

The FastAPI application can be run locally:

```powershell
uvicorn app.main:app --reload
```

The Streamlit application can also be run locally:

```powershell
streamlit run app\page.py
```

## CI

GitHub Actions runs basic Python validation in `.github/workflows/check.yml`.

## Future Work

Dockerization or cloud deployment should be proposed as future Units before implementation.
