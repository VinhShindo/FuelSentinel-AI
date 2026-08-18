#!/usr/bin/env python3
"""
Evaluation package for FuelSentinel-AI training pipeline.
"""

from .metrics import (
    compute_metrics,
    compute_all_metrics,
    ClassificationReport,
    MetricsTracker,
)
from .plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    plot_learning_curve,
    plot_metrics_comparison,
)
from .benchmark_report import BenchmarkReport, ModelRanking
from .confusion import ConfusionMatrixAnalyzer
from .roc_pr import ROCPRCurves

__all__ = [
    'compute_metrics',
    'compute_all_metrics',
    'ClassificationReport',
    'MetricsTracker',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_pr_curves',
    'plot_learning_curve',
    'plot_metrics_comparison',
    'BenchmarkReport',
    'ModelRanking',
    'ConfusionMatrixAnalyzer',
    'ROCPRCurves',
]