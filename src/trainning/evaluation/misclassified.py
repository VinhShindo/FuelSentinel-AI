#!/usr/bin/env python3
"""
Module: misclassified.py
Mục tiêu: Trực quan hóa các mẫu bị dự đoán sai.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

LABEL_NAMES = ['Driving', 'Idle', 'Refuel', 'Theft']


def plot_misclassified_examples(
    sequences: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray = None,
    max_examples: int = 9,
    save_path: str = None,
):
    """
    Vẽ lưới các mẫu bị dự đoán sai, hiển thị chuỗi fuel.
    sequences: np.array of shape (N, T, 4) hoặc object array (sẽ lấy element đầu).
    Nếu là object array (các sequence có độ dài khác nhau), tự động pad về max length.
    """
    # Tìm chỉ số bị sai
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        logger.info("Không có mẫu nào bị dự đoán sai.")
        return None

    if len(mis_idx) > max_examples:
        mis_idx = np.random.choice(mis_idx, max_examples, replace=False)

    n = len(mis_idx)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    # Nếu sequences là object array, pad về max length
    if sequences.dtype == object:
        lengths = [seq.shape[0] for seq in sequences[mis_idx]]
        max_len = max(lengths)
        padded = np.zeros((len(mis_idx), max_len, 4))
        for j, idx in enumerate(mis_idx):
            seq = sequences[idx]
            padded[j, :seq.shape[0], :] = seq
        seqs_to_plot = padded
    else:
        seqs_to_plot = sequences[mis_idx]

    for i, idx in enumerate(mis_idx):
        ax = axes[i]
        fuel_vals = seqs_to_plot[i, :, 0]  # cột fuel
        ax.plot(fuel_vals, linewidth=1.5)
        true_label = LABEL_NAMES[y_true[idx]]
        pred_label = LABEL_NAMES[y_pred[idx]]
        if y_probs is not None:
            prob = y_probs[idx, y_pred[idx]]
            title = f"True: {true_label} → Pred: {pred_label} ({prob:.2f})"
        else:
            title = f"True: {true_label} → Pred: {pred_label}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Fuel')
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Misclassified Examples (Fuel Signal)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Misclassified examples plot saved to {save_path}")

    return fig