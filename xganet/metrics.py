"""Classification metrics from Equations (37)–(40)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, object]:
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(
            y_true, y_pred, target_names=list(class_names), zero_division=0
        ),
    }
    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        metrics["auc"] = float(auc)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics
