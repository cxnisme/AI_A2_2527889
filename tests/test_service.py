"""Tests: validation, inference, reproducibility, CLI guard and API."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import app
from run_all import main as run_all_main
from src.config import SEED, STUDENT_ID, VERIFICATION_CODE
from src.inference import predict_record
from src.pipeline import run_pipeline
from src.validation import DataValidationError, check_no_leakage

VALID = {"rainfall_mm": 1050.0, "avg_temp_c": 18.5, "soil_moisture_pct": 55.0,
         "fertilizer_kg": 140.0, "field_size_ha": 1.8, "season": "A", "cooperative_support": 1}
client = TestClient(app)


def test_valid_inference():
    out = predict_record(VALID)
    assert out["verification_code"] == VERIFICATION_CODE
    assert out["yield_estimate_tons"] >= 0.5
    assert out["high_risk_prediction"] in (0, 1)
    assert 0.0 <= out["high_risk_probability"] <= 1.0
    assert isinstance(out["cluster"], int)


def test_missing_field_rejected():
    rec = {k: v for k, v in VALID.items() if k != "field_size_ha"}
    with pytest.raises(ValidationError):
        predict_record(rec)


def test_wrong_type_rejected():
    with pytest.raises(ValidationError):
        predict_record({**VALID, "avg_temp_c": "warm"})


def test_range_error_rejected():
    with pytest.raises(ValidationError):
        predict_record({**VALID, "rainfall_mm": 2500.0})


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return a == pytest.approx(b, rel=1e-9, abs=1e-9)
    return a == b


def test_reproducibility(tmp_path):
    first = run_pipeline(tmp_path / "a", build_report=False, log=lambda *_: None)
    second = run_pipeline(tmp_path / "b", build_report=False, log=lambda *_: None)
    assert first["sha256"] == second["sha256"]
    assert _same(first["regression"], second["regression"])
    assert _same(first["classification"], second["classification"])
    assert _same(first["clustering"], second["clustering"])
    committed = Path(__file__).resolve().parents[1] / "artifacts" / "data_fingerprint.txt"
    assert f"sha256: {first['sha256']}" in committed.read_text()


def test_api_predict_response():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["verification_code"] == VERIFICATION_CODE
    assert {"yield_estimate_tons", "high_risk_probability", "cluster"} <= body.keys()


def test_api_invalid_returns_422():
    r = client.post("/predict", json={**VALID, "season": "C", "rainfall_mm": 5})
    assert r.status_code == 422
    fields = {d["field"] for d in r.json()["details"]}
    assert {"season", "rainfall_mm"} <= fields


def test_api_health_and_model_info():
    assert client.get("/health").json()["verification_code"] == VERIFICATION_CODE
    info = client.get("/model-info").json()
    assert info["verification_code"] == VERIFICATION_CODE and info["seed"] == SEED


def test_leakage_guard():
    with pytest.raises(DataValidationError):
        check_no_leakage(["rainfall_mm", "yield_tons"])


def test_run_all_rejects_wrong_seed(capsys):
    assert run_all_main(["--student-id", STUDENT_ID, "--seed", "1"]) == 2
    assert "7889" in capsys.readouterr().err
