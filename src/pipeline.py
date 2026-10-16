"""End-to-end pipeline: generate -> validate -> clean -> train -> evaluate -> save."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .clustering import run_clustering
from .config import (
    FEATURES,
    PROJECT_NAME,
    ROOT,
    SEED,
    STUDENT_ID,
    VERIFICATION_CODE,
    Paths,
)
from .data_generation import generate_dataframe, write_csv, write_fingerprint
from .evaluation import plot_clusters, plot_confusion_matrix, plot_residuals
from .modeling import run_classification, run_regression
from .preprocessing import split_data
from .validation import remove_duplicates, validate_raw


def _dump(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def run_pipeline(root: Path = ROOT, seed: int = SEED, student_id: str = STUDENT_ID,
                 build_report: bool = True, log=print) -> dict:
    """Run every stage and (re)write all data, models and artifacts under ``root``."""
    paths = Paths(Path(root))
    paths.make_dirs()
    tag = f"[{VERIFICATION_CODE}]"

    # 1. Data generation + fingerprint (hash of the final CSV bytes)
    raw_generated = generate_dataframe(seed)
    write_csv(raw_generated, paths.csv)
    digest = write_fingerprint(paths.csv, paths.artifacts / "data_fingerprint.txt", len(raw_generated))
    log(f"{tag} data written: {paths.csv.name} ({len(raw_generated)} rows)")
    log(f"{tag} SHA-256 fingerprint: {digest}")

    # 2. Validation and cleaning (read back from the CSV so training uses the saved file)
    raw = pd.read_csv(paths.csv)
    validation = validate_raw(raw)
    df = remove_duplicates(raw)
    validation["rows_after_cleaning"] = int(len(df))
    validation["handling"] = {
        "duplicates": "dropped exact duplicate rows before the split",
        "missing_values": "median imputation (numeric) fitted on training rows only, inside the pipeline",
    }
    _dump({"verification_code": VERIFICATION_CODE, **validation}, paths.artifacts / "validation_report.json")
    log(f"{tag} validation: missing={validation['missing_values']} duplicates={validation['duplicate_rows']}")

    # 3. Fixed split
    s = split_data(df, seed)
    log(f"{tag} split: train={len(s['train'])} test={len(s['test'])} (random_state={seed})")

    # 4. Regression
    reg_metrics, reg_model, reg_pred = run_regression(
        s["X_train"], s["y_reg_train"], s["X_test"], s["y_reg_test"], seed)
    _dump(reg_metrics, paths.artifacts / "regression_metrics.json")
    log(f"{tag} regression selected: {reg_metrics['selected_model']}")

    # 5. Classification
    clf_metrics, clf_model, clf_pred, clf_proba = run_classification(
        s["X_train"], s["y_clf_train"], s["X_test"], s["y_clf_test"], seed)
    _dump(clf_metrics, paths.artifacts / "classification_metrics.json")
    log(f"{tag} classification selected: {clf_metrics['selected_model']}")

    # 6. Clustering
    clu_report, clu_model, profiles = run_clustering(s["train"], seed)
    _dump(clu_report, paths.artifacts / "clustering_report.json")
    log(f"{tag} clustering selected K={clu_report['selected_k']}")

    # 7. Test predictions
    test = s["test"]
    tp = pd.DataFrame({
        "row_id": test.index.to_numpy(),
        "actual_yield_tons": test["yield_tons"].to_numpy(),
        "predicted_yield_tons": reg_pred,
        "yield_error_tons": reg_pred - test["yield_tons"].to_numpy(),
        "abs_yield_error_tons": np.abs(reg_pred - test["yield_tons"].to_numpy()),
        "actual_high_risk": test["high_risk"].to_numpy(),
        "predicted_high_risk": clf_pred,
        "high_risk_probability": clf_proba,
        "risk_correct": (clf_pred == test["high_risk"].to_numpy()).astype(int),
        "cluster": clu_model.predict(s["X_test"]),
    }).sort_values("row_id")
    tp.to_csv(paths.artifacts / "test_predictions.csv", index=False,
              float_format="%.4f", lineterminator="\n")

    # 8. Plots
    plot_residuals(test["yield_tons"], reg_pred, paths.plots / "residuals.png")
    sel_clf = clf_metrics["candidates"][clf_metrics["selected_model"]]
    plot_confusion_matrix(sel_clf["confusion_matrix"]["matrix"], paths.plots / "confusion_matrix.png")
    Z = clu_model[:-1].transform(df[FEATURES])
    plot_clusters(Z, clu_model.predict(df[FEATURES]), clu_report["scores"],
                  clu_report["selected_k"], paths.plots / "clusters.png")

    # 9. Save models + metadata (loaded by predict.py and app.py)
    joblib.dump(reg_model, paths.models / "regression_model.joblib")
    joblib.dump(clf_model, paths.models / "classification_model.joblib")
    joblib.dump(clu_model, paths.models / "clustering_model.joblib")
    sel_reg = reg_metrics["candidates"][reg_metrics["selected_model"]]
    metadata = {
        "project": PROJECT_NAME,
        "verification_code": VERIFICATION_CODE,
        "student_id": student_id,
        "seed": seed,
        "data_sha256": digest,
        "data_rows_raw": validation["rows_raw"],
        "data_rows_clean": validation["rows_after_cleaning"],
        "synthetic_data": True,
        "inputs": FEATURES,
        "selected_regression_model": reg_metrics["selected_model"],
        "selected_classification_model": clf_metrics["selected_model"],
        "selected_k": clu_report["selected_k"],
        "test_metrics": {
            "regression": {k: sel_reg[k] for k in ("mae", "rmse", "r2")},
            "classification": {k: sel_clf[k] for k in ("precision", "recall", "f1", "roc_auc")},
        },
        "cluster_profiles": [
            {"cluster": p["cluster"], "summary": p["summary"], "n_train": p["n_train"]}
            for p in profiles
        ],
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "intended_use": "Classroom decision-support prototype; not a replacement for an agronomist.",
    }
    _dump(metadata, paths.models / "model_metadata.json")

    from .inference import clear_cache
    clear_cache()

    if build_report:
        from .report import build_report as make_report
        make_report(paths, log=log)

    log(f"{tag} pipeline finished")
    return {"sha256": digest, "regression": reg_metrics, "classification": clf_metrics,
            "clustering": clu_report, "validation": validation}
