"""Synthetic potato-farm dataset generation (seeded by registration digits)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    MIN_YIELD_TONS,
    MISSING_FRACTION,
    N_DUPLICATES,
    PROJECT_NAME,
    SEED,
    VERIFICATION_CODE,
)


def base_row_count(seed: int = SEED) -> int:
    """Return 240 + (seed mod 41) base records."""
    return 240 + (seed % 41)


def generate_dataframe(seed: int = SEED) -> pd.DataFrame:
    """Generate the raw dataset: base rows, injected missing values, 3 duplicates.

    The random sequence is drawn from ``numpy.random.default_rng(seed)`` in a fixed
    order so the same seed always yields the same table.
    """
    rng = np.random.default_rng(seed)
    n = base_row_count(seed)

    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    soil = rng.uniform(20, 80, n)
    fert = rng.uniform(0, 250, n)
    field = rng.uniform(0.2, 4.0, n)
    season = rng.choice(["A", "B"], size=n)
    support = rng.integers(0, 2, size=n)
    noise = rng.normal(0.0, 0.80, n)

    season_effect = np.where(season == "A", 0.50, -0.20)
    yield_tons = (
        0.004 * rainfall
        + 0.06 * soil
        + 0.02 * fert
        + 1.30 * field
        - 0.25 * np.abs(temp - 18)
        + 0.70 * support
        + season_effect
        + noise
    )
    yield_tons = np.clip(yield_tons, MIN_YIELD_TONS, None)

    # high_risk: yield per hectare strictly below its median (computed on base rows).
    yield_per_ha = yield_tons / field
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)

    df = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "avg_temp_c": temp,
            "soil_moisture_pct": soil,
            "fertilizer_kg": fert,
            "field_size_ha": field,
            "season": season,
            "cooperative_support": support,
            "yield_tons": yield_tons,
            "high_risk": high_risk,
        }
    )

    # Inject missing values: m unique positions each, disjoint between the two columns.
    m = round(MISSING_FRACTION * n)
    positions = rng.choice(n, size=2 * m, replace=False)
    df.loc[positions[:m], "soil_moisture_pct"] = np.nan
    df.loc[positions[m:], "fertilizer_kg"] = np.nan

    # Append exact duplicate rows.
    dup_positions = rng.choice(n, size=N_DUPLICATES, replace=False)
    df = pd.concat([df, df.iloc[dup_positions]], ignore_index=True)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the CSV with four-decimal numerics and platform-independent newlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")


def sha256_of_file(path: Path) -> str:
    """SHA-256 hex digest of the file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fingerprint(csv_path: Path, out_path: Path, n_rows: int) -> str:
    """Hash the final CSV bytes and save the fingerprint file. Returns the digest."""
    digest = sha256_of_file(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"sha256: {digest}\n"
        f"file: data/{PROJECT_NAME}.csv\n"
        f"rows: {n_rows}\n"
        f"verification_code: {VERIFICATION_CODE}\n",
        encoding="utf-8",
    )
    return digest
