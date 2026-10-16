"""Regression and classification: baselines, candidates, selection, test metrics."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, SEED, VERIFICATION_CODE
from .preprocessing import build_preprocessor


def _pipe(estimator) -> Pipeline:
    """Preprocessor + estimator so training and inference share transformations."""
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def regression_candidates(seed: int = SEED) -> dict[str, Pipeline]:
    return {
        "linear_regression": _pipe(LinearRegression()),
        "random_forest": _pipe(RandomForestRegressor(n_estimators=200, random_state=seed)),
        "gradient_boosting": _pipe(GradientBoostingRegressor(random_state=seed)),
    }


def classification_candidates(seed: int = SEED) -> dict[str, Pipeline]:
    return {
        "logistic_regression": _pipe(LogisticRegression(max_iter=1000, random_state=seed)),
        "random_forest": _pipe(RandomForestClassifier(n_estimators=200, random_state=seed)),
        "gradient_boosting": _pipe(GradientBoostingClassifier(random_state=seed)),
    }


def _reg_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_regression(X_train, y_train, X_test, y_test, seed: int = SEED):
    """Compare a mean baseline with candidates; select by 5-fold CV RMSE on train only.

    Returns (metrics_dict, fitted_selected_pipeline, test_predictions).
    """
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    baseline = _pipe(DummyRegressor(strategy="mean")).fit(X_train, y_train)
    out = {
        "verification_code": VERIFICATION_CODE,
        "seed": seed,
        "target": "yield_tons",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "baseline": {"name": "mean_baseline", **_reg_metrics(y_test, baseline.predict(X_test))},
        "candidates": {},
    }
    fitted = {}
    for name, pipe in regression_candidates(seed).items():
        cv_rmse = -cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
        )
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        out["candidates"][name] = {
            **_reg_metrics(y_test, pipe.predict(X_test)),
            "cv_rmse_mean": float(cv_rmse.mean()),
            "cv_rmse_std": float(cv_rmse.std()),
        }
    best = min(out["candidates"], key=lambda k: out["candidates"][k]["cv_rmse_mean"])
    b, base = out["candidates"][best], out["baseline"]
    out["selected_model"] = best
    out["selection_rule"] = f"lowest {CV_FOLDS}-fold cross-validated RMSE on the training set"
    out["selection_reason"] = (
        f"{best} had the lowest cross-validated RMSE ({b['cv_rmse_mean']:.4f} t). On the "
        f"held-out test set it scored MAE {b['mae']:.4f}, RMSE {b['rmse']:.4f}, R2 {b['r2']:.4f} "
        f"versus the mean baseline MAE {base['mae']:.4f}, RMSE {base['rmse']:.4f}, "
        f"R2 {base['r2']:.4f}."
    )
    return out, fitted[best], fitted[best].predict(X_test)


def run_classification(X_train, y_train, X_test, y_test, seed: int = SEED):
    """Compare a majority baseline with candidates; select by 5-fold CV ROC-AUC on train.

    Returns (metrics_dict, fitted_selected_pipeline, predicted_labels, probabilities).
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

    def metrics(model) -> dict:
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        both_classes = len(np.unique(y_test)) == 2
        return {
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)) if both_classes else None,
            "confusion_matrix": {
                "labels": [0, 1],
                "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            },
        }

    baseline = _pipe(DummyClassifier(strategy="most_frequent")).fit(X_train, y_train)
    out = {
        "verification_code": VERIFICATION_CODE,
        "seed": seed,
        "target": "high_risk",
        "positive_class": "1 = high production risk (yield per hectare below median)",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "baseline": {"name": "majority_baseline", **metrics(baseline)},
        "candidates": {},
    }
    fitted = {}
    for name, pipe in classification_candidates(seed).items():
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        out["candidates"][name] = {
            **metrics(pipe),
            "cv_roc_auc_mean": float(cv_auc.mean()),
            "cv_roc_auc_std": float(cv_auc.std()),
        }
    best = max(out["candidates"], key=lambda k: out["candidates"][k]["cv_roc_auc_mean"])
    b, base = out["candidates"][best], out["baseline"]
    out["selected_model"] = best
    out["selection_rule"] = f"highest {CV_FOLDS}-fold cross-validated ROC-AUC on the training set"
    out["selection_reason"] = (
        f"{best} had the highest cross-validated ROC-AUC ({b['cv_roc_auc_mean']:.4f}). On the "
        f"held-out test set it scored precision {b['precision']:.4f}, recall {b['recall']:.4f}, "
        f"F1 {b['f1']:.4f}, ROC-AUC {b['roc_auc']:.4f}; the majority baseline scored F1 "
        f"{base['f1']:.4f} and ROC-AUC {base['roc_auc']:.4f}."
    )
    model = fitted[best]
    return out, model, model.predict(X_test), model.predict_proba(X_test)[:, 1]
