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
    all_labels = list(range(len(class_names)))
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average="macro", zero_division=0
    )
    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average="weighted", zero_division=0
    )
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "weighted_precision": float(w_precision),
        "weighted_recall": float(w_recall),
        "weighted_f1": float(w_f1),
        "w_precision": float(w_precision),
        "w_recall": float(w_recall),
        "w_f1": float(w_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_labels),
        "report": classification_report(
            y_true,
            y_pred,
            labels=all_labels,
            target_names=list(class_names),
            zero_division=0,
        ),
    }
    try:
        present_classes = np.unique(y_true)
        if len(present_classes) >= 2:
            if y_prob.shape[1] == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            elif len(present_classes) == y_prob.shape[1]:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            else:
                auc = roc_auc_score(
                    y_true,
                    y_prob[:, present_classes],
                    labels=present_classes,
                    multi_class="ovr",
                    average="macro",
                )
            metrics["auc"] = float(auc)
        else:
            metrics["auc"] = float("nan")
    except (ValueError, IndexError):
        metrics["auc"] = float("nan")
    return metrics
