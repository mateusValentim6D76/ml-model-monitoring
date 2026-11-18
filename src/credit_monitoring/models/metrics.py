"""Métricas de performance usadas em risco de crédito."""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ks_statistic(y_true: np.ndarray, proba: np.ndarray) -> float:
    """KS de separação entre bons e maus (métrica clássica de scoring de crédito)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    return float(ks_2samp(proba[y_true == 1], proba[y_true == 0]).statistic)


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, proba),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "default_rate": float(y_true.mean()),
        "predicted_default_rate": float(pred.mean()),
        "mean_score": float(proba.mean()),
    }


def best_f1_threshold(y_true, proba) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])
