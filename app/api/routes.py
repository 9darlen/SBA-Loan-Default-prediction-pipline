"""HTTP routes for model prediction."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.prediction import (
    HealthResponse,
    LoanApplication,
    PredictionResponse,
    VersionResponse,
)
from app.services.prediction_service import PredictionInputError, PredictionService


router = APIRouter()


def get_prediction_service() -> PredictionService:
    return PredictionService()


@router.get("/health", response_model=HealthResponse)
def health(service: PredictionService = Depends(get_prediction_service)) -> dict[str, str]:
    try:
        return service.health()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model artifact unavailable")


@router.get("/version", response_model=VersionResponse)
def version(service: PredictionService = Depends(get_prediction_service)) -> dict[str, str]:
    return {"model_version": service.model_version}


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: LoanApplication,
    service: PredictionService = Depends(get_prediction_service),
) -> dict[str, object]:
    try:
        return service.predict_one(payload.to_model_input()).to_dict()
    except PredictionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model artifact unavailable")

