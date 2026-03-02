"""
metrics.py

Purpose:
- Compute per-epoch classification metrics from collected predictions.
- Kept stateless: functions receive lists/tensors, return plain dicts.

Metrics computed:
    accuracy    — fraction of correct predictions
    f1_macro    — macro-averaged F1-score (unweighted mean over classes)
    per_class_f1 — F1 per class (for classification report)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    all_labels: List[int],
    all_preds: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute accuracy and macro F1 from flat label / prediction lists.

    Parameters
    ----------
    all_labels : list[int]
        Ground-truth class indices.
    all_preds : list[int]
        Predicted class indices.
    class_names : list[str] | None
        Optional class names for per-class F1 keys.

    Returns
    -------
    dict with keys:
        ``accuracy``        — float in [0, 1]
        ``f1_macro``        — float in [0, 1]
        ``f1_<class>``      — per-class F1 (one key per class if class_names provided)
    """
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    
    labels_arg = list(range(len(class_names))) if class_names is not None else None
    f1_macro = float(f1_score(y_true, y_pred, labels=labels_arg, average="macro", zero_division=0))

    result: Dict[str, float] = {
        "accuracy": acc,
        "f1_macro": f1_macro,
    }

    if class_names:
        per_class = f1_score(y_true, y_pred, labels=labels_arg, average=None, zero_division=0)
        for name, score in zip(class_names, per_class):
            result[f"f1_{name}"] = float(score)

    return result


def get_classification_report(
    all_labels: List[int],
    all_preds: List[int],
    class_names: Optional[List[str]] = None,
) -> str:
    """Return a formatted sklearn classification report string."""
    labels_arg = list(range(len(class_names))) if class_names is not None else None
    return classification_report(
        all_labels,
        all_preds,
        labels=labels_arg,
        target_names=class_names,
        zero_division=0,
    )


def get_confusion_matrix(
    all_labels: List[int],
    all_preds: List[int],
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Return the confusion matrix as a 2-D numpy array."""
    labels_arg = list(range(len(class_names))) if class_names is not None else None
    return confusion_matrix(all_labels, all_preds, labels=labels_arg)
