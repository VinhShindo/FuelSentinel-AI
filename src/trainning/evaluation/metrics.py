#!/usr/bin/env python3
"""
Module: metrics.py
Mục tiêu: Tính toán tất cả metrics cho bài toán classification.
Hỗ trợ: accuracy, precision, recall, f1, balanced accuracy, per-class metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
)

logger = logging.getLogger(__name__)

# Label mapping
LABEL_NAMES = ['Driving', 'Idle', 'Refuel', 'Theft']
NUM_CLASSES = 4


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    average: str = 'weighted',
) -> Dict[str, float]:
    """
    Tính toán các metrics cơ bản.
    
    Args:
        y_true: (N,) - ground truth labels.
        y_pred: (N,) - predicted labels.
        y_probs: (N, C) - predicted probabilities (optional, cho ROC AUC).
        average: 'micro', 'macro', 'weighted'.
        
    Returns:
        Dict các metrics.
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Balanced accuracy
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Precision
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Recall
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    # F1
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Cohen's Kappa
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # ROC AUC (nếu có probabilities)
    if y_probs is not None:
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(
                y_true, y_probs, multi_class='ovr', average='weighted'
            )
        except Exception:
            metrics['roc_auc_ovr'] = np.nan
        
        try:
            metrics['avg_precision'] = average_precision_score(
                y_true, y_probs, average='weighted'
            )
        except Exception:
            metrics['avg_precision'] = np.nan
    
    return metrics


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    """
    Tính toán tất cả metrics bao gồm per-class và classification report.
    
    Args:
        y_true: (N,) - ground truth.
        y_pred: (N,) - predictions.
        y_probs: (N, C) - probabilities.
        
    Returns:
        Dict với 'overall' metrics, 'per_class' metrics, 'classification_report'.
    """
    # Overall metrics
    overall = compute_metrics(y_true, y_pred, y_probs)
    
    # Per-class metrics
    per_class = {}
    for i, label_name in enumerate(LABEL_NAMES):
        # Chuyển về binary cho class i
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        
        per_class[label_name] = {
            'precision': precision_score(y_true_bin, y_pred_bin, zero_division=0),
            'recall': recall_score(y_true_bin, y_pred_bin, zero_division=0),
            'f1': f1_score(y_true_bin, y_pred_bin, zero_division=0),
            'support': int(np.sum(y_true == i)),
        }
    
    # Classification report (string)
    class_report = classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        zero_division=0,
        output_dict=True,
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'overall': overall,
        'per_class': per_class,
        'classification_report': class_report,
        'confusion_matrix': cm,
    }


class ClassificationReport:
    """
    Tạo classification report dạng bảng đẹp.
    """
    
    def __init__(self, y_true, y_pred, y_probs=None):
        self.metrics = compute_all_metrics(y_true, y_pred, y_probs)
    
    def to_dict(self) -> Dict:
        """Trả về dict."""
        return self.metrics
    
    def to_string(self) -> str:
        """Trả về string đẹp."""
        lines = []
        lines.append("=" * 70)
        lines.append("CLASSIFICATION REPORT")
        lines.append("=" * 70)
        
        # Overall metrics
        overall = self.metrics['overall']
        lines.append("\nOverall Metrics:")
        lines.append("-" * 40)
        lines.append(f"  Accuracy:           {overall['accuracy']:.4f}")
        lines.append(f"  Balanced Accuracy:  {overall['balanced_accuracy']:.4f}")
        lines.append(f"  F1 (Macro):         {overall['f1_macro']:.4f}")
        lines.append(f"  F1 (Weighted):      {overall['f1_weighted']:.4f}")
        lines.append(f"  Precision (Macro):  {overall['precision_macro']:.4f}")
        lines.append(f"  Recall (Macro):     {overall['recall_macro']:.4f}")
        lines.append(f"  Cohen's Kappa:      {overall['cohen_kappa']:.4f}")
        lines.append(f"  MCC:                {overall['mcc']:.4f}")
        
        if 'roc_auc_ovr' in overall:
            lines.append(f"  ROC AUC (OvR):      {overall['roc_auc_ovr']:.4f}")
        
        # Per-class metrics
        lines.append("\nPer-Class Metrics:")
        lines.append("-" * 70)
        lines.append(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        lines.append("  " + "-" * 52)
        
        for label_name, metrics in self.metrics['per_class'].items():
            lines.append(
                f"  {label_name:<12} "
                f"{metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} "
                f"{metrics['f1']:>10.4f} "
                f"{metrics['support']:>10d}"
            )
        
        # Confusion matrix
        lines.append("\nConfusion Matrix:")
        lines.append("-" * 40)
        cm = self.metrics['confusion_matrix']
        header = f"  {'':>12} " + " ".join(f"{name:>8}" for name in LABEL_NAMES)
        lines.append(header)
        for i, row in enumerate(cm):
            line = f"  {LABEL_NAMES[i]:>12} " + " ".join(f"{val:>8d}" for val in row)
            lines.append(line)
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def __str__(self):
        return self.to_string()


class MetricsTracker:
    """
    Theo dõi metrics trong quá trình training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset tất cả metrics."""
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        self.epoch_times = []
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float = None,
        epoch_time: float = None,
    ) -> None:
        """Cập nhật metrics cho một epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        if lr is not None:
            self.learning_rates.append(lr)
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """Lấy epoch tốt nhất dựa trên metric."""
        if metric == 'val_loss':
            values = self.val_losses
        elif metric == 'val_acc':
            values = self.val_accs
        elif metric == 'train_loss':
            values = self.train_losses
        else:
            values = self.train_accs
        
        if mode == 'min':
            return int(np.argmin(values)) + 1
        else:
            return int(np.argmax(values)) + 1
    
    def to_dict(self) -> Dict:
        """Chuyển về dict."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Tạo dữ liệu giả
    np.random.seed(42)
    n = 200
    
    y_true = np.random.randint(0, 4, n)
    y_pred = y_true.copy()
    # Thêm noise
    noise_idx = np.random.choice(n, size=int(n * 0.2), replace=False)
    y_pred[noise_idx] = np.random.randint(0, 4, len(noise_idx))
    
    y_probs = np.random.rand(n, 4)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    
    # Test compute_metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)
    print("Overall Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test compute_all_metrics
    all_metrics = compute_all_metrics(y_true, y_pred, y_probs)
    print(f"\nPer-class metrics keys: {list(all_metrics['per_class'].keys())}")
    
    # Test ClassificationReport
    report = ClassificationReport(y_true, y_pred, y_probs)
    print("\n" + str(report))
    
    # Test MetricsTracker
    tracker = MetricsTracker()
    for epoch in range(5):
        tracker.update(
            epoch,
            train_loss=0.5 - epoch * 0.05,
            val_loss=0.4 - epoch * 0.03,
            train_acc=0.7 + epoch * 0.05,
            val_acc=0.75 + epoch * 0.04,
            lr=0.001 * (0.9 ** epoch),
            epoch_time=10.0 - epoch * 0.5,
        )
    
    print(f"\nBest epoch (val_loss): {tracker.get_best_epoch('val_loss', 'min')}")
    print(f"Best epoch (val_acc): {tracker.get_best_epoch('val_acc', 'max')}")
    
    print("\nAll metrics tests passed!")