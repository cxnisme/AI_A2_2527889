"""FastAPI service: GET /health, GET /model-info, POST /predict.

Run:  uvicorn app:app --host 127.0.0.1 --port 8000   (docs at /docs)
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config import PROJECT_NAME, VERIFICATION_CODE
from src.inference import ModelsNotFoundError, load_metadata, models_available, predict_record
from src.schemas import FarmRecord, PredictionResponse

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description=(
        f"Classroom prototype ({VERIFICATION_CODE}) trained on synthetic data. "
        "Not a replacement for an agronomist; human review required."
    ),
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a readable 422 listing every invalid field."""
    details = [
        {"field": ".".join(str(p) for p in e["loc"] if p != "body") or "body",
         "problem": e["msg"], "type": e["type"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"verification_code": VERIFICATION_CODE, "error": "invalid_input",
                 "message": "The farm record failed validation.", "details": details},
    )


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "project": PROJECT_NAME, "verification_code": VERIFICATION_CODE,
            "models_loaded": models_available()}


@app.get("/model-info")
def model_info() -> JSONResponse:
    """Selected models, test metrics, data fingerprint and verification code."""
    try:
        return JSONResponse(load_metadata())
    except ModelsNotFoundError as exc:
        return JSONResponse(status_code=503, content={
            "verification_code": VERIFICATION_CODE, "error": "models_not_found",
            "message": str(exc)})


@app.post("/predict", response_model=PredictionResponse)
def predict(record: FarmRecord) -> PredictionResponse | JSONResponse:
    """Estimate yield, risk class, risk probability and cluster for one farm."""
    try:
        return PredictionResponse(**predict_record(record.model_dump()))
    except ModelsNotFoundError as exc:
        return JSONResponse(status_code=503, content={
            "verification_code": VERIFICATION_CODE, "error": "models_not_found",
            "message": str(exc)})
