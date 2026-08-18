#!/usr/bin/env python3
"""
generate_confusion_matrices.py
- Duyệt qua tất cả các model đã benchmark (có checkpoint)
- Load model, đánh giá trên tập test
- Vẽ và lưu confusion matrix cho từng model (chỉ 1 bảng counts)
- In bảng confusion matrix ra console
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# Thêm thư mục src vào sys.path
# ------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[2]   # src/
sys.path.insert(0, str(SRC_DIR))

from trainning.utils.config import Config
from trainning.utils.device import get_device
from trainning.datasets.dataset_builder import build_datasets
from trainning.datasets.collate import DynamicSequenceCollator
from trainning.models.factory import ModelFactory
from trainning.models import all_models
# Không cần import ConfusionMatrixAnalyzer nữa vì tự vẽ

LABEL_NAMES = ['Driving', 'Idle', 'Refuel', 'Theft']

def plot_confusion_matrix(y_true, y_pred, labels, save_path, title):
    """Vẽ confusion matrix chỉ với raw counts (1 bảng)."""
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=0.5)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    # Config cho việc đánh giá
    config = Config.from_yaml('src/configs/benchmark.yaml')
    device = get_device('auto')

    # Load test dataset (chỉ cần test)
    print("Loading test dataset...")
    _, _, test_ds = build_datasets(
        train_path=config.data.train_path,
        val_path=config.data.val_path,
        test_path=config.data.test_path,
        config={'min_sequence_length': 2, 'fillna_strategy': 'median'}
    )

    # Tạo DataLoader cho test set
    collate_fn = DynamicSequenceCollator()
    test_loader = DataLoader(
        test_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    print(f"Test loader: {len(test_loader)} batches")

    # Thư mục chứa checkpoint của từng model
    base_dir = Path('outputs/benchmark')
    out_dir = base_dir / 'confusion_matrices'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Các tham số chung cho model (phải khớp với benchmark)
    model_kwargs = dict(
        input_dim=4, hidden_dim=128, num_layers=2, dropout=0.3,
        bidirectional=True, feature_dim=21, num_classes=4,
        fusion_type='concat', fusion_dim=256,   # concat fusion như benchmark
        nhead=8, dim_feedforward=512, num_transformer_layers=3,
        tcn_channels=[64,128,256], kernel_size=3,
        cnn_channels=[64,128], cnn_kernel_size=3
    )

    # Duyệt qua các thư mục model
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        checkpoint_path = model_dir / 'checkpoints' / 'best_model.pt'
        if not checkpoint_path.exists():
            continue

        model_name = model_dir.name
        print(f"\nProcessing {model_name}...")

        try:
            model = ModelFactory.create_from_name(model_name, **model_kwargs)
            model.to(device)

            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.eval()

            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in test_loader:
                    seq = batch['sequence'].to(device)
                    mask = batch['mask'].to(device)
                    feat = batch['feature'].to(device)
                    labels = batch['label'].to(device)
                    logits = model(seq, mask, feat)
                    preds = logits.argmax(dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            y_pred = torch.cat(all_preds).numpy()
            y_true = torch.cat(all_labels).numpy()

            # ---- In bảng confusion matrix (số lượng) ----
            cm = confusion_matrix(y_true, y_pred)
            df_cm = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
            print(f"\nConfusion Matrix for {model_name} (counts):")
            print(df_cm)

            # ---- Vẽ và lưu confusion matrix (chỉ 1 bảng counts) ----
            save_path = out_dir / f'{model_name}_confusion.png'
            plot_confusion_matrix(
                y_true, y_pred,
                labels=LABEL_NAMES,
                save_path=str(save_path),
                title=f'Confusion Matrix - {model_name}'
            )
            print(f"  Saved confusion matrix for {model_name}")

        except Exception as e:
            print(f"  Failed to process {model_name}: {e}")

if __name__ == '__main__':
    main()