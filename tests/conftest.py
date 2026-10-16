"""Shared pytest setup: make sure trained models exist before inference tests run."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def trained_models():
    """Train once if the saved models are missing (e.g. on a fresh clone)."""
    from src.config import Paths
    from src.pipeline import run_pipeline

    p = Paths()
    if not (p.models / "model_metadata.json").exists():
        run_pipeline(build_report=False, log=lambda *_: None)
    yield
