"""
TrueLens AI — Evaluation Metrics Calculator

Computes: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, 
per-class metrics, and forensic-specific FPR/FNR.

Author: TrueLens AI Team
License: MIT
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Classification metrics calculator for forensic evaluation."""

    CLASS_NAMES_BINARY = ['Real', 'AI-Generated']

    def __init__(self, num_classes: int = 2) -> None:
        self.num_classes = num_classes
        self.class_names = self.CLASS_NAMES_BINARY if num_classes == 2 else [f'Class_{i}' for i in range(num_classes)]

    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict:
        """Compute all classification metrics."""
        metrics = {}
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        avg = 'binary' if self.num_classes == 2 else 'weighted'
        pos = 1 if self.num_classes == 2 else None
        metrics['precision'] = float(precision_score(y_true, y_pred, average=avg, pos_label=pos, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, average=avg, pos_label=pos, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, average=avg, pos_label=pos, zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

        if y_prob is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1] if self.num_classes == 2 else y_prob, multi_class='ovr' if self.num_classes > 2 else 'raise', average='weighted' if self.num_classes > 2 else None))
            except ValueError:
                metrics['roc_auc'] = 0.0

        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        metrics['confusion_matrix'] = cm.tolist()

        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)
        metrics['per_class'] = {name: {'precision': report[name]['precision'], 'recall': report[name]['recall'], 'f1': report[name]['f1-score'], 'support': report[name]['support']} for name in self.class_names if name in report}

        if self.num_classes == 2 and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['false_positive_rate'] = float(fp / max(fp + tn, 1))
            metrics['false_negative_rate'] = float(fn / max(fn + tp, 1))

        return metrics
