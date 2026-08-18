#!/usr/bin/env python3
"""
Module: roc_pr.py
Mục tiêu: Phân tích ROC và Precision-Recall curves chi tiết.
Tính AUC, AP, optimal threshold cho từng class.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

LABEL_NAMES = ['Driving', 'Idle', 'Refuel', 'Theft']
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']


class ROCPRCurves:
    """
    Phân tích ROC và PR curves cho multi-class classification.
    
    Usage:
        analyzer = ROCPRCurves(y_true, y_probs)
        analyzer.print_summary()
        analyzer.plot_roc(save_path='roc.png')
        analyzer.plot_pr(save_path='pr.png')
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        labels: Optional[List[str]] = None,
    ):
        """
        Args:
            y_true: (N,) ground truth labels.
            y_probs: (N, C) predicted probabilities.
            labels: Tên các class.
        """
        self.y_true = np.array(y_true)
        self.y_probs = np.array(y_probs)
        self.labels = labels or LABEL_NAMES
        self.n_classes = len(self.labels)
        
        # Binarize labels
        self.y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        # Compute ROC
        self.roc_data = self._compute_roc()
        
        # Compute PR
        self.pr_data = self._compute_pr()
        
        # Find optimal thresholds (Youden's J statistic)
        self.optimal_thresholds = self._find_optimal_thresholds()
    
    def _compute_roc(self) -> Dict:
        """Tính ROC curves cho từng class."""
        roc_data = {}
        
        for i, label in enumerate(self.labels):
            fpr, tpr, thresholds = roc_curve(
                self.y_true_bin[:, i], self.y_probs[:, i]
            )
            roc_auc = auc(fpr, tpr)
            
            roc_data[label] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc,
            }
        
        # Macro-average
        all_fpr = np.unique(np.concatenate([roc_data[l]['fpr'] for l in self.labels]))
        mean_tpr = np.zeros_like(all_fpr)
        
        for label in self.labels:
            mean_tpr += np.interp(all_fpr, roc_data[label]['fpr'], roc_data[label]['tpr'])
        
        mean_tpr /= self.n_classes
        
        roc_data['macro'] = {
            'fpr': all_fpr,
            'tpr': mean_tpr,
            'auc': auc(all_fpr, mean_tpr),
        }
        
        # Weighted average
        try:
            roc_data['weighted'] = {
                'auc': roc_auc_score(self.y_true, self.y_probs, multi_class='ovr', average='weighted'),
            }
        except Exception:
            roc_data['weighted'] = {'auc': np.nan}
        
        return roc_data
    
    def _compute_pr(self) -> Dict:
        """Tính PR curves cho từng class."""
        pr_data = {}
        
        for i, label in enumerate(self.labels):
            precision, recall, thresholds = precision_recall_curve(
                self.y_true_bin[:, i], self.y_probs[:, i]
            )
            ap = average_precision_score(self.y_true_bin[:, i], self.y_probs[:, i])
            
            pr_data[label] = {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds,
                'ap': ap,
            }
        
        # Weighted average
        try:
            pr_data['weighted'] = {
                'ap': average_precision_score(self.y_true, self.y_probs, average='weighted'),
            }
        except Exception:
            pr_data['weighted'] = {'ap': np.nan}
        
        return pr_data
    
    def _find_optimal_thresholds(self) -> Dict[str, float]:
        """
        Tìm optimal threshold dùng Youden's J statistic.
        J = TPR - FPR.
        
        Returns:
            Dict {class_name: optimal_threshold}.
        """
        thresholds = {}
        
        for label in self.labels:
            data = self.roc_data[label]
            j_scores = data['tpr'] - data['fpr']
            optimal_idx = np.argmax(j_scores)
            thresholds[label] = float(data['thresholds'][optimal_idx])
        
        return thresholds
    
    def get_auc_scores(self) -> Dict[str, float]:
        """Trả về AUC scores cho từng class."""
        return {label: self.roc_data[label]['auc'] for label in self.labels}
    
    def get_ap_scores(self) -> Dict[str, float]:
        """Trả về Average Precision scores cho từng class."""
        return {label: self.pr_data[label]['ap'] for label in self.labels}
    
    def print_summary(self) -> None:
        """In tóm tắt."""
        print("=" * 60)
        print("ROC & PR CURVES ANALYSIS")
        print("=" * 60)
        
        print("\nROC AUC Scores:")
        print("-" * 40)
        for label in self.labels:
            print(f"  {label:<12}: {self.roc_data[label]['auc']:.4f}")
        print(f"  {'Macro':<12}: {self.roc_data['macro']['auc']:.4f}")
        if 'weighted' in self.roc_data:
            print(f"  {'Weighted':<12}: {self.roc_data['weighted']['auc']:.4f}")
        
        print("\nAverage Precision Scores:")
        print("-" * 40)
        for label in self.labels:
            print(f"  {label:<12}: {self.pr_data[label]['ap']:.4f}")
        if 'weighted' in self.pr_data:
            print(f"  {'Weighted':<12}: {self.pr_data['weighted']['ap']:.4f}")
        
        print("\nOptimal Thresholds (Youden's J):")
        print("-" * 40)
        for label, thresh in self.optimal_thresholds.items():
            print(f"  {label:<12}: {thresh:.4f}")
        
        print("=" * 60)
    
    def plot_roc(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "ROC Curves (One-vs-Rest)",
    ) -> plt.Figure:
        """
        Vẽ ROC curves.
        
        Args:
            save_path: Đường dẫn lưu ảnh.
            figsize: Kích thước figure.
            title: Tiêu đề.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot per-class
        for i, label in enumerate(self.labels):
            data = self.roc_data[label]
            ax.plot(
                data['fpr'], data['tpr'],
                color=COLORS[i],
                linewidth=2,
                label=f"{label} (AUC={data['auc']:.3f})"
            )
        
        # Plot macro
        data = self.roc_data['macro']
        ax.plot(
            data['fpr'], data['tpr'],
            'k--', linewidth=2,
            label=f"Macro (AUC={data['auc']:.3f})"
        )
        
        # Baseline
        ax.plot([0, 1], [0, 1], 'gray', linestyle=':', linewidth=1, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        return fig
    
    def plot_pr(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Precision-Recall Curves",
    ) -> plt.Figure:
        """
        Vẽ PR curves.
        
        Args:
            save_path: Đường dẫn lưu ảnh.
            figsize: Kích thước figure.
            title: Tiêu đề.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, label in enumerate(self.labels):
            data = self.pr_data[label]
            ax.plot(
                data['recall'], data['precision'],
                color=COLORS[i],
                linewidth=2,
                label=f"{label} (AP={data['ap']:.3f})"
            )
        
        # Baseline (random classifier)
        baseline = np.mean(self.y_true_bin, axis=0)
        ax.axhline(y=np.mean(baseline), color='gray', linestyle=':', linewidth=1,
                    label=f'Random (AP={np.mean(baseline):.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"PR curves saved to {save_path}")
        
        return fig
    
    def plot_combined(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Vẽ ROC và PR side-by-side.
        
        Args:
            save_path: Đường dẫn lưu ảnh.
            figsize: Kích thước figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # ROC
        ax = axes[0]
        for i, label in enumerate(self.labels):
            data = self.roc_data[label]
            ax.plot(data['fpr'], data['tpr'], color=COLORS[i], linewidth=2,
                    label=f"{label} (AUC={data['auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'gray', linestyle=':', linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curves', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # PR
        ax = axes[1]
        for i, label in enumerate(self.labels):
            data = self.pr_data[label]
            ax.plot(data['recall'], data['precision'], color=COLORS[i], linewidth=2,
                    label=f"{label} (AP={data['ap']:.3f})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Combined ROC/PR curves saved to {save_path}")
        
        return fig
    
    def to_dict(self) -> Dict:
        """Chuyển về dict."""
        return {
            'auc_scores': self.get_auc_scores(),
            'ap_scores': self.get_ap_scores(),
            'optimal_thresholds': self.optimal_thresholds,
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    
    y_true = np.random.randint(0, 4, n)
    y_probs = np.random.rand(n, 4)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    
    analyzer = ROCPRCurves(y_true, y_probs)
    analyzer.print_summary()
    
    analyzer.plot_combined(save_path='outputs/test/roc_pr_combined.png')
    print("\nROC/PR combined plot saved.")