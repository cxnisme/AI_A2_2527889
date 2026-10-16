"""Data validation and cleaning: columns, ranges, missing values, duplicates, leakage."""
from __future__ import annotations

import pandas as pd

from .config import (
    FEATURES,
    FORBIDDEN_FEATURES,
    MIN_YIELD_TONS,
    RANGES,
    REQUIRED_COLUMNS,
    SEASONS,
)


class DataValidationError(ValueError):
    """Raised when the dataset breaks a structural rule (columns, ranges, labels)."""


def check_no_leakage(feature_columns: list[str]) -> None:
    """Raise if any target-derived column is used as a model input."""
    leaked = sorted(set(feature_columns) & FORBIDDEN_FEATURES)
    suspicious = [c for c in feature_columns if "yield" in c.lower() or "risk" in c.lower()]
    if leaked or suspicious:
        raise DataValidationError(
            f"Target leakage: {sorted(set(leaked + suspicious))} must not be model inputs."
        )


def validate_raw(df: pd.DataFrame) -> dict:
    """Validate the raw table and return an evidence report.

    Missing values and duplicates are *reported* here and handled later (duplicates
    are dropped in ``remove_duplicates``; missing values are imputed inside the
    training pipeline). Structural problems raise ``DataValidationError``.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    check_no_leakage(FEATURES)

    problems: list[str] = []
    for col, (lo, hi) in RANGES.items():
        vals = df[col].dropna()
        bad = int(((vals < lo) | (vals > hi)).sum())
        if bad:
            problems.append(f"{col}: {bad} value(s) outside [{lo}, {hi}]")
    if not df["season"].isin(SEASONS).all():
        problems.append("season contains values other than A/B")
    if not df["cooperative_support"].isin([0, 1]).all():
        problems.append("cooperative_support contains values other than 0/1")
    if not df["high_risk"].isin([0, 1]).all():
        problems.append("high_risk contains values other than 0/1")
    if df["yield_tons"].isna().any() or (df["yield_tons"] < MIN_YIELD_TONS).any():
        problems.append(f"yield_tons has missing values or values below {MIN_YIELD_TONS}")
    if problems:
        raise DataValidationError("; ".join(problems))

    n_dups = int(df.duplicated().sum())
    return {
        "rows_raw": int(len(df)),
        "columns": list(df.columns),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v},
        "duplicate_rows": n_dups,
        "ranges_ok": True,
        "leakage_check": "passed: no yield_tons, yield_per_ha or high_risk in model inputs",
        "model_inputs": FEATURES,
    }


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (before splitting, so copies cannot leak across sets)."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)
