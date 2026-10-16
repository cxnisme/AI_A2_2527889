"""Shared inference logic used by predict.py and app.py."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURES, MIN_YIELD_TONS, VERIFICATION_CODE, Paths
from .schemas import FarmRecord

NOTICE = (
    "Classroom prototype on synthetic data. Not a substitute for an agronomist and not "
    "for decisions on finance, insurance or public services. Human review required."
)


class ModelsNotFoundError(FileNotFoundError):
    """Raised when saved models are missing."""


@lru_cache(maxsize=4)
def _load(models_dir: str) -> dict:
    d = Path(models_dir)
    needed = ["regression_model.joblib", "classification_model.joblib",
              "clustering_model.joblib", "model_metadata.json"]
    missing = [n for n in needed if not (d / n).exists()]
    if missing:
        raise ModelsNotFoundError(
            f"Saved model files not found in {d}: {missing}. Run "
            "'python run_all.py --student-id ug25/27889@ines.ac.rw --seed 7889' first."
        )
    return {
        "regression": joblib.load(d / "regression_model.joblib"),
        "classification": joblib.load(d / "classification_model.joblib"),
        "clustering": joblib.load(d / "clustering_model.joblib"),
        "metadata": json.loads((d / "model_metadata.json").read_text(encoding="utf-8")),
    }


def clear_cache() -> None:
    _load.cache_clear()


def load_metadata(models_dir: Path | None = None) -> dict:
    """Return model_metadata.json (selected models, metrics summary, fingerprint)."""
    return _load(str((models_dir or Paths().models).resolve()))["metadata"]


def models_available(models_dir: Path | None = None) -> bool:
    try:
        _load(str((models_dir or Paths().models).resolve()))
        return True
    except ModelsNotFoundError:
        return False


def predict_record(record: dict, models_dir: Path | None = None) -> dict:
    """Validate one farm record and return yield, risk, probability and cluster.

    Raises pydantic.ValidationError for invalid input and ModelsNotFoundError if the
    models have not been trained yet.
    """
    farm = FarmRecord.model_validate(record)
    bundle = _load(str((models_dir or Paths().models).resolve()))
    row = pd.DataFrame([farm.model_dump()])[FEATURES]
    est = float(bundle["regression"].predict(row)[0])
    est = max(est, MIN_YIELD_TONS)
    proba = float(bundle["classification"].predict_proba(row)[0, 1])
    pred = int(bundle["classification"].predict(row)[0])
    cluster = int(bundle["clustering"].predict(row)[0])
    profiles = bundle["metadata"]["cluster_profiles"]
    return {
        "verification_code": VERIFICATION_CODE,
        "yield_estimate_tons": round(est, 4),
        "high_risk_prediction": pred,
        "high_risk_label": "high risk" if pred == 1 else "not high risk",
        "high_risk_probability": round(proba, 4),
        "cluster": cluster,
        "cluster_profile": profiles[cluster]["summary"],
        "notice": NOTICE,
    }
