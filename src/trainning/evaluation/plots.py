#!/usr/bin/env python3
"""
Module: plots.py
Mục tiêu: Các hàm vẽ biểu đồ cho evaluation và benchmark.
Bao gồm: confusion matrix, ROC, PR curves, learning curves, comparison charts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
import seaborn as sns

logger = logging.getLogger(__name__)

# Cấu hình style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

LABEL_NAMES = ['Driving', 'Idle', 'Refuel', 'Theft']
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 150) -> None:
    """Lưu figure ra file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figure saved: {filepath}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Vẽ confusion matrix.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        save_path: Đường dẫn lưu ảnh.
        normalize: Chuẩn hóa về tỉ lệ.
        title: Tiêu đề.
        figsize: Kích thước figure.
        
    Returns:
        Matplotlib figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
        fmt = '.2%'
        vmax = 1.0
    else:
        fmt = 'd'
        vmax = cm.max()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        vmin=0,
        vmax=vmax,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC Curves (One-vs-Rest)",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Vẽ ROC curves cho từng class.
    
    Args:
        y_true: Ground truth labels (N,).
        y_probs: Predicted probabilities (N, C).
        save_path: Đường dẫn lưu ảnh.
        title: Tiêu đề.
        figsize: Kích thước figure.
        
    Returns:
        Matplotlib figure.
    """
    n_classes = y_probs.shape[1]
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        ax.plot(
            fpr[i], tpr[i],
            color=COLORS[i],
            linewidth=2,
            label=f'{LABEL_NAMES[i]} (AUC = {roc_auc[i]:.3f})'
        )
    
    # Đường baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curves",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Vẽ Precision-Recall curves cho từng class.
    
    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities.
        save_path: Đường dẫn lưu ảnh.
        title: Tiêu đề.
        figsize: Kích thước figure.
        
    Returns:
        Matplotlib figure.
    """
    n_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i], y_probs[:, i]
        )
        ap = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        
        ax.plot(
            recall, precision,
            color=COLORS[i],
            linewidth=2,
            label=f'{LABEL_NAMES[i]} (AP = {ap:.3f})'
        )
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_learning_curve(
    history: Dict,
    save_path: Optional[str] = None,
    title: str = "Learning Curve",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Vẽ learning curve (loss và accuracy).
    
    Args:
        history: Dict chứa train_loss, val_loss, train_acc, val_acc.
        save_path: Đường dẫn lưu ảnh.
        title: Tiêu đề.
        figsize: Kích thước figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Đánh dấu epoch tốt nhất
    best_epoch = np.argmin(history['val_loss']) + 1
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
               label=f'Best Epoch ({best_epoch})')
    
    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
    ax.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Đánh dấu epoch tốt nhất
    best_acc_epoch = np.argmax(history['val_acc']) + 1
    ax.axvline(x=best_acc_epoch, color='green', linestyle='--', alpha=0.5,
               label=f'Best Acc Epoch ({best_acc_epoch})')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_metrics_comparison(
    models_metrics: Dict[str, Dict],
    metric_name: str = 'accuracy',
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    horizontal: bool = True,
) -> plt.Figure:
    """
    Vẽ biểu đồ so sánh metrics giữa các model.
    
    Args:
        models_metrics: Dict {model_name: {metric_name: value, ...}}
        metric_name: Tên metric cần so sánh.
        save_path: Đường dẫn lưu ảnh.
        title: Tiêu đề.
        figsize: Kích thước figure.
        horizontal: Vẽ ngang.
        
    Returns:
        Matplotlib figure.
    """
    models = list(models_metrics.keys())
    values = [models_metrics[m].get(metric_name, 0) for m in models]
    
    # Sắp xếp theo giá trị
    sorted_idx = np.argsort(values)
    models = [models[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if horizontal:
        bars = ax.barh(models, values, color=COLORS[:len(models)])
        ax.set_xlabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        
        # Thêm giá trị
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=10)
    else:
        bars = ax.bar(models, values, color=COLORS[:len(models)])
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Thêm giá trị
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', fontsize=10)
    
    if title is None:
        title = f"Model Comparison: {metric_name.replace('_', ' ').title()}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x' if horizontal else 'y')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Tạo dữ liệu giả
    np.random.seed(42)
    n = 200
    
    y_true = np.random.randint(0, 4, n)
    y_pred = y_true.copy()
    noise_idx = np.random.choice(n, size=int(n * 0.2), replace=False)
    y_pred[noise_idx] = np.random.randint(0, 4, len(noise_idx))
    
    y_probs = np.random.rand(n, 4)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    
    # Test confusion matrix
    fig = plot_confusion_matrix(y_true, y_pred, save_path='outputs/test/confusion_matrix.png')
    print("Confusion matrix plot created.")
    
    # Test ROC curves
    fig = plot_roc_curves(y_true, y_probs, save_path='outputs/test/roc_curves.png')
    print("ROC curves plot created.")
    
    # Test PR curves
    fig = plot_pr_curves(y_true, y_probs, save_path='outputs/test/pr_curves.png')
    print("PR curves plot created.")
    
    # Test learning curve
    history = {
        'train_loss': [0.8, 0.6, 0.5, 0.4, 0.35],
        'val_loss': [0.9, 0.7, 0.55, 0.5, 0.52],
        'train_acc': [0.5, 0.6, 0.7, 0.8, 0.85],
        'val_acc': [0.45, 0.55, 0.68, 0.72, 0.70],
    }
    fig = plot_learning_curve(history, save_path='outputs/test/learning_curve.png')
    print("Learning curve plot created.")
    
    # Test metrics comparison
    models_metrics = {
        'BiLSTM': {'accuracy': 0.85, 'f1_weighted': 0.84},
        'GRU': {'accuracy': 0.83, 'f1_weighted': 0.82},
        'Transformer': {'accuracy': 0.87, 'f1_weighted': 0.86},
        'TCN': {'accuracy': 0.82, 'f1_weighted': 0.81},
    }
    fig = plot_metrics_comparison(
        models_metrics, 'accuracy',
        save_path='outputs/test/metrics_comparison.png'
    )
    print("Metrics comparison plot created.")
    
    print("\nAll plot tests passed!")