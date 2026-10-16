"""Plots written to artifacts/plots/ (Agg backend, no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from .config import SEED, VERIFICATION_CODE  # noqa: E402


def plot_residuals(y_true, y_pred, path: Path) -> None:
    """Residuals vs predicted values and residual histogram."""
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].scatter(y_pred, resid, s=18, alpha=0.7, color="#2b6cb0")
    ax[0].axhline(0, color="black", lw=1)
    ax[0].set_xlabel("Predicted yield (tons)")
    ax[0].set_ylabel("Residual = actual - predicted (tons)")
    ax[0].set_title("Residuals vs predictions")
    ax[1].hist(resid, bins=15, color="#2f855a", edgecolor="white")
    ax[1].set_xlabel("Residual (tons)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Residual distribution")
    fig.suptitle(f"Test-set residuals  [{VERIFICATION_CODE}]", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(cm, path: Path) -> None:
    """Annotated confusion matrix (rows = actual, columns = predicted)."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Not high risk", "High risk"])
    ax.set_yticks([0, 1], ["Not high risk", "High risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (test)  [{VERIFICATION_CODE}]", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_clusters(Z, labels, scores: dict, selected_k: int, path: Path) -> None:
    """PCA view of clusters plus silhouette score for each K."""
    xy = PCA(n_components=2, random_state=SEED).fit_transform(Z)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    sc = ax[0].scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=18, alpha=0.8)
    ax[0].set_xlabel("PCA component 1")
    ax[0].set_ylabel("PCA component 2")
    ax[0].set_title(f"Farm clusters (K={selected_k})")
    ax[0].legend(*sc.legend_elements(), title="Cluster", fontsize=7, loc="best")
    ks = sorted(int(k) for k in scores)
    vals = [scores[str(k)]["silhouette"] for k in ks]
    ax[1].bar(ks, vals, color=["#c05621" if k == selected_k else "#a0aec0" for k in ks])
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("Silhouette score")
    ax[1].set_title("Choosing K")
    fig.suptitle(f"Clustering  [{VERIFICATION_CODE}]", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
