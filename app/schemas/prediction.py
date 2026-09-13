"""Pydantic schemas for prediction API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LoanApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    State: str = Field(..., min_length=1)
    BankState: str = Field(..., min_length=1)
    NAICS: str = Field(..., min_length=1)
    NewExist: float
    UrbanRural: int
    RevLineCr: str = Field(..., min_length=1)
    LowDoc: str = Field(..., min_length=1)
    FranchiseCode: str = Field(..., min_length=1)
    GrAppv: float = Field(..., ge=0)
    SBA_Appv: float = Field(..., ge=0)
    Term: int = Field(..., ge=0)
    NoEmp: int = Field(..., ge=0)
    ApprovalDate: str = Field(..., min_length=1)
    ApprovalFY: int

    def to_model_input(self) -> dict[str, object]:
        return self.model_dump()


class PredictionResponse(BaseModel):
    request_id: str
    prediction: int
    default_probability: float = Field(..., ge=0, le=1)
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str


class VersionResponse(BaseModel):
    model_version: str

