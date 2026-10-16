"""K-means clustering of farm conditions with silhouette-based choice of K."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BINARY_FEATURES,
    FEATURES,
    K_RANGE,
    NUMERIC_FEATURES,
    SEED,
    VERIFICATION_CODE,
)
from .preprocessing import build_preprocessor

WORDS = {
    "rainfall_mm": "rainfall",
    "avg_temp_c": "temperature",
    "soil_moisture_pct": "soil moisture",
    "fertilizer_kg": "fertilizer use",
    "field_size_ha": "field size",
}


def _scaler_pipeline() -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("scale", StandardScaler())])


def build_cluster_pipeline(k: int, seed: int = SEED) -> Pipeline:
    """Impute/encode, scale every input, then K-means (all fitted on training rows)."""
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
        ]
    )


def _describe(rows: pd.DataFrame, overall: pd.DataFrame, std: pd.Series) -> str:
    """Plain-language summary of how a cluster differs from the training average."""
    parts = []
    for col in NUMERIC_FEATURES:
        z = (rows[col].mean() - overall[col]) / std[col]
        if z >= 0.4:
            parts.append(f"higher {WORDS[col]}")
        elif z <= -0.4:
            parts.append(f"lower {WORDS[col]}")
    share = rows["cooperative_support"].mean()
    if share >= 0.65:
        parts.append("mostly supported by the cooperative")
    elif share <= 0.35:
        parts.append("mostly without cooperative support")
    a = (rows["season"] == "A").mean()
    if a >= 0.65:
        parts.append("mostly season A")
    elif a <= 0.35:
        parts.append("mostly season B")
    if not parts:
        return "conditions close to the cooperative average"
    return ", ".join(parts)


def run_clustering(train: pd.DataFrame, seed: int = SEED):
    """Test K=2..6 on scaled training features; pick K by silhouette score.

    Returns (report_dict, fitted_pipeline, metadata_profiles).
    """
    X = train[FEATURES]
    Z = _scaler_pipeline().fit_transform(X)
    scores = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        scores[k] = {
            "silhouette": float(silhouette_score(Z, km.labels_)),
            "inertia": float(km.inertia_),
        }
    selected = max(scores, key=lambda k: (scores[k]["silhouette"], -k))
    pipe = build_cluster_pipeline(selected, seed).fit(X)
    labels = pipe.predict(X)

    overall = train[NUMERIC_FEATURES].mean()
    std = train[NUMERIC_FEATURES].std()
    profiles = []
    for c in range(selected):
        rows = train[labels == c]
        means = {col: float(rows[col].mean()) for col in NUMERIC_FEATURES}
        means["cooperative_support_share"] = float(rows["cooperative_support"].mean())
        means["season_A_share"] = float((rows["season"] == "A").mean())
        profiles.append(
            {
                "cluster": c,
                "n_train": int(len(rows)),
                "share_of_train": float(len(rows) / len(train)),
                "feature_means": means,
                "mean_yield_tons": float(rows["yield_tons"].mean()),
                "high_risk_rate": float(rows["high_risk"].mean()),
                "summary": _describe(rows, overall, std),
            }
        )
    best_sil = scores[selected]["silhouette"]
    if best_sil < 0.25:
        strength = "weak: the groups overlap and should be read as soft segments"
    elif best_sil < 0.50:
        strength = "moderate: the groups are partly separated"
    else:
        strength = "strong: the groups are clearly separated"
    report = {
        "verification_code": VERIFICATION_CODE,
        "seed": seed,
        "features_used": FEATURES,
        "scaling": "imputation, one-hot encoding and StandardScaler fitted on training rows",
        "scores": {str(k): v for k, v in scores.items()},
        "selected_k": int(selected),
        "selection_rule": "highest silhouette score among K = 2..6",
        "profiles": profiles,
        "interpretation": (
            f"K={selected} gave the highest silhouette score ({best_sil:.4f}). Cluster structure "
            f"is {strength}. Yield and risk rates are shown for description only; they were not "
            "used to form clusters. The data are synthetic and drawn from uniform distributions, "
            "so clusters are convenient groupings of similar conditions, not natural farm types."
        ),
    }
    return report, pipe, profiles
