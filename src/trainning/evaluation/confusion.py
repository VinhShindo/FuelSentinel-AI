#!/usr/bin/env python3
"""
Module: confusion.py
Mục tiêu: Phân tích chi tiết confusion matrix.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_NAMES = ['Driving', 'Idle', 'Refuel', 'Theft']


class ConfusionMatrixAnalyzer:
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
    ):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.labels = labels or LABEL_NAMES
        self.n_classes = len(self.labels)

        self.cm = confusion_matrix(y_true, y_pred)
        self.cm_normalized = self.cm.astype('float') / self.cm.sum(axis=1, keepdims=True).clip(min=1e-9)
        self.per_class_metrics = self._compute_per_class_metrics()
        self.top_confusions = self._find_top_confusions()

    def _compute_per_class_metrics(self) -> Dict:
        metrics = {}
        for i, label in enumerate(self.labels):
            tp = self.cm[i, i]
            fp = self.cm[:, i].sum() - tp
            fn = self.cm[i, :].sum() - tp
            tn = self.cm.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

            metrics[label] = {
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
                'precision': precision, 'recall': recall, 'f1': f1,
                'accuracy': accuracy, 'support': int(self.cm[i, :].sum()),
            }
        return metrics

    def _find_top_confusions(self, top_k: int = 10) -> List[Dict]:
        confusions = []
        total = self.cm.sum()
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i != j and self.cm[i, j] > 0:
                    confusions.append({
                        'true_class': self.labels[i],
                        'pred_class': self.labels[j],
                        'count': int(self.cm[i, j]),
                        'percentage': float(self.cm[i, j] / total * 100),
                    })
        confusions.sort(key=lambda x: x['count'], reverse=True)
        return confusions[:top_k]

    def get_most_confused_classes(self) -> List[Tuple[str, str, int]]:
        return [(c['true_class'], c['pred_class'], c['count']) for c in self.top_confusions[:5]]

    def get_per_class_accuracy(self) -> Dict[str, float]:
        return {label: m['recall'] for label, m in self.per_class_metrics.items()}

    def print_summary(self) -> None:
        print("=" * 70)
        print("CONFUSION MATRIX ANALYSIS")
        print("=" * 70)
        print("\nPer-Class Metrics:")
        print("-" * 80)
        header = f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
        print(header)
        print("  " + "-" * 52)
        for label, m in self.per_class_metrics.items():
            print(f"  {label:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10d}")

        print("\nConfusion Matrix (Normalized):")
        print("-" * 60)
        header = f"  {'':>12} " + " ".join(f"{name:>8}" for name in self.labels)
        print(header)
        for i, row in enumerate(self.cm_normalized):
            line = f"  {self.labels[i]:>12} " + " ".join(f"{val:>8.1%}" for val in row)
            print(line)

        print("\nTop Misclassifications:")
        print("-" * 50)
        for i, conf in enumerate(self.top_confusions[:5], 1):
            print(f"  {i}. {conf['true_class']} → {conf['pred_class']}: {conf['count']} samples ({conf['percentage']:.2f}%)")
        print("=" * 70)

    # ĐÂY LÀ METHOD PLOT ĐÃ SỬA THỤT LỀ
    def plot(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
        title: str = "Confusion Matrix",
    ) -> plt.Figure:
        """
        Vẽ confusion matrix dạng số đếm (counts).
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            self.cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.labels,
            yticklabels=self.labels,
            ax=ax,
            cbar_kws={'label': 'Count'},
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def to_dict(self) -> Dict:
        return {
            'confusion_matrix': self.cm.tolist(),
            'confusion_matrix_normalized': self.cm_normalized.tolist(),
            'per_class_metrics': self.per_class_metrics,
            'top_confusions': self.top_confusions,
        }