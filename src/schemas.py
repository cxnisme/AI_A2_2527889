"""Typed request/response models used by the API and the prediction CLI."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import RANGES


class FarmRecord(BaseModel):
    """One farm. Strict types, all fields required, unknown fields rejected."""

    model_config = ConfigDict(extra="forbid", strict=True)

    rainfall_mm: float = Field(ge=RANGES["rainfall_mm"][0], le=RANGES["rainfall_mm"][1])
    avg_temp_c: float = Field(ge=RANGES["avg_temp_c"][0], le=RANGES["avg_temp_c"][1])
    soil_moisture_pct: float = Field(
        ge=RANGES["soil_moisture_pct"][0], le=RANGES["soil_moisture_pct"][1]
    )
    fertilizer_kg: float = Field(ge=RANGES["fertilizer_kg"][0], le=RANGES["fertilizer_kg"][1])
    field_size_ha: float = Field(ge=RANGES["field_size_ha"][0], le=RANGES["field_size_ha"][1])
    season: Literal["A", "B"]
    cooperative_support: Literal[0, 1]


class PredictionResponse(BaseModel):
    verification_code: str
    yield_estimate_tons: float
    high_risk_prediction: int
    high_risk_label: str
    high_risk_probability: float
    cluster: int
    cluster_profile: str
    notice: str
