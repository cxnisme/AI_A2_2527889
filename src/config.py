"""Project-wide constants tied to the student's registration digits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROJECT_NAME = "AI_A2_2527889"
FULL_NAME = "NIWENIRINGIYE Christian"
REG_NUMBER = "25/27889"
STUDENT_ID = "ug25/27889@ines.ac.rw"
GITHUB_USER = "cxnisme"
VERIFICATION_CODE = "AI-A2-7889"
SEED = 7889

BASE_ROWS = 240 + (SEED % 41)  # 240 + 17 = 257
N_DUPLICATES = 3
MISSING_FRACTION = 0.05
TEST_SIZE = 0.25
CV_FOLDS = 5

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
]
CATEGORICAL_FEATURES = ["season"]
BINARY_FEATURES = ["cooperative_support"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

TARGET_REGRESSION = "yield_tons"
TARGET_CLASSIFICATION = "high_risk"
FORBIDDEN_FEATURES = {"yield_tons", "yield_per_ha", "high_risk"}
REQUIRED_COLUMNS = FEATURES + [TARGET_REGRESSION, TARGET_CLASSIFICATION]

# Valid ranges (inclusive) taken from the data-generation rules.
RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}
SEASONS = ("A", "B")
MIN_YIELD_TONS = 0.50
K_RANGE = range(2, 7)


@dataclass(frozen=True)
class Paths:
    """All project paths, derived from one root so tests can use a temp dir."""

    root: Path = ROOT

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def csv(self) -> Path:
        return self.data_dir / f"{PROJECT_NAME}.csv"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def plots(self) -> Path:
        return self.artifacts / "plots"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    def make_dirs(self) -> None:
        for p in (self.data_dir, self.models, self.artifacts, self.plots, self.docs):
            p.mkdir(parents=True, exist_ok=True)
