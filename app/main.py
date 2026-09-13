"""FastAPI entry point for the credit risk prediction service."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="SBA Credit Risk Prediction API",
        version="1.0.0",
        description="API for serving the existing SBA loan default prediction pipeline.",
    )
    app.include_router(router)
    return app


app = create_app()

