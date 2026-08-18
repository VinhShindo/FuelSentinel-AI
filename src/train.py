#!/usr/bin/env python3
"""
train.py - Entry point cho Phase 2: Final Model Training (hỗ trợ resume)

Output được tổ chức tập trung trong một thư mục duy nhất:
    outputs/final_model/<model_name>_<timestamp>/
        ├── checkpoints/
        ├── logs/
        │   ├── training_output.log   (toàn bộ log terminal)
        │   └── training_log.csv
        ├── tensorboard/
        ├── reports/
        ├── figures/
        ├── config.yaml
        ├── config.json
        ├── final_model.pt
        ├── model.onnx
        └── summary.json

Usage:
    python train.py --config src/configs/final_train.yaml
    python train.py --config src/configs/final_train.yaml --resume outputs/benchmark/cnn_gru/checkpoints/best_model.pt
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import argparse
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from trainning.utils.config import Config
from trainning.utils.seed import set_seed
from trainning.utils.device import get_device, print_device_info
from trainning.datasets.dataset_builder import build_datasets
from trainning.datasets.dataloader import DataLoaderFactory
from trainning.models.factory import ModelFactory
from trainning.models import all_models
from trainning.trainer.trainer import Trainer
from trainning.trainer.optimizer import OptimizerFactory
from trainning.evaluation.metrics import compute_all_metrics, ClassificationReport
from trainning.evaluation.confusion import ConfusionMatrixAnalyzer
from trainning.evaluation.roc_pr import ROCPRCurves
from trainning.evaluation.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    plot_learning_curve,
    save_figure,
)
from trainning.evaluation.misclassified import plot_misclassified_examples


def setup_logging(log_dir: str) -> logging.Logger:
    """
    Cấu hình root logger để ghi ra cả console và file training_output.log
    trong thư mục log_dir.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / 'training_output.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def train_final_model(config: Config, resume_path: str = None) -> dict:
    """
    Train final model với config đầy đủ. Toàn bộ output được lưu trong:
    outputs/final_model/<model_name>_<timestamp>/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model.name
    output_dir = Path(config.output.base_dir) / f"{model_name}_{timestamp}"

    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    tensorboard_dir = output_dir / 'tensorboard'
    reports_dir = output_dir / 'reports'
    figures_dir = output_dir / 'figures'

    for d in [output_dir, checkpoint_dir, log_dir, tensorboard_dir, reports_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config.to_yaml(output_dir / 'config.yaml')
    config.to_json(output_dir / 'config.json')

    logger = setup_logging(str(log_dir))
    logger.info(f"Output directory: {output_dir}")

    set_seed(config.seed, config.deterministic)
    device = get_device(config.get('device', 'auto'))
    print_device_info(device)

    logger.info("Building datasets...")
    train_ds, val_ds, test_ds = build_datasets(
        train_path=config.data.train_path,
        val_path=config.data.val_path,
        test_path=config.data.test_path,
        config={
            'min_sequence_length': config.get('data.min_sequence_length', 2),
            'fillna_strategy': config.get('data.fillna_strategy', 'median'),
        }
    )
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    dataloader_factory = DataLoaderFactory(config)
    train_loader, val_loader, test_loader = dataloader_factory.create_all(train_ds, val_ds, test_ds)

    logger.info(f"Creating model: {model_name}")
    model = ModelFactory.create(config).to(device)
    model_info = model.get_model_info()
    logger.info(f"Parameters: {model_info['total_params']:,} | Size: {model_info['model_size_mb']:.2f} MB")

    optimizer = OptimizerFactory.create_optimizer(model, config)
    scheduler = OptimizerFactory.create_scheduler(optimizer, config)
    criterion = OptimizerFactory.create_loss(config, train_dataset=train_ds)
    criterion.to(device)

    class SimpleDict:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def get(self, key, default=None):
            return getattr(self, key, default) if hasattr(self, key) else default

    trainer_config = SimpleDict()
    trainer_config.training = config.training
    trainer_config.checkpoint = SimpleDict(checkpoint_dir=str(checkpoint_dir), save_best=True, save_last=True)
    trainer_config.logging = SimpleDict(log_dir=str(log_dir), tensorboard_dir=str(tensorboard_dir), csv_logger=True, tensorboard=True)
    trainer_config.output = SimpleDict(base_dir=str(output_dir))

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, device=device,
        config=trainer_config, scheduler=scheduler, test_loader=test_loader,
        use_amp=config.training.get('use_amp', False),
        gradient_clip_val=config.training.get('gradient_clip_val', 1.0),
    )

    start_epoch = 0
    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)
        start_epoch = trainer.current_epoch
        logger.info(f"Resumed at epoch {start_epoch}")

    epochs = config.training.epochs
    logger.info(f"Starting training from epoch {start_epoch} to {epochs}...")
    history = trainer.train(epochs=epochs)

    best_checkpoint = checkpoint_dir / 'best_model.pt'
    if best_checkpoint.exists():
        logger.info("Loading best model checkpoint...")
        trainer.load_checkpoint(str(best_checkpoint))

    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
    logger.info("Predicting on test set...")
    predictions = trainer.predict(test_loader)

    y_true = predictions['labels'].numpy()
    y_pred = predictions['preds'].numpy()
    y_probs = predictions['probs'].numpy()

    metrics = compute_all_metrics(y_true, y_pred, y_probs)
    report = ClassificationReport(y_true, y_pred, y_probs)
    logger.info("\n" + str(report))

    with open(reports_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(str(report))
    with open(reports_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            'overall': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics['overall'].items()},
            'per_class': metrics['per_class'],
        }, f, indent=2, ensure_ascii=False)

    logger.info("Generating plots...")

    plot_learning_curve(history, save_path=str(figures_dir / 'learning_curve.png'), title=f"Learning Curve - {model_name}")

    cm_analyzer = ConfusionMatrixAnalyzer(y_true, y_pred)
    cm_analyzer.print_summary()
    cm_analyzer.plot(save_path=str(figures_dir / 'confusion_matrix.png'), title=f"Confusion Matrix - {model_name}")
    with open(reports_dir / 'confusion_matrix.json', 'w') as f:
        json.dump(cm_analyzer.to_dict(), f, indent=2)

    if not np.isnan(y_probs).any():
        roc_pr = ROCPRCurves(y_true, y_probs)
        roc_pr.print_summary()
        roc_pr.plot_roc(save_path=str(figures_dir / 'roc_curves.png'), title=f"ROC Curves - {model_name}")
        roc_pr.plot_pr(save_path=str(figures_dir / 'pr_curves.png'), title=f"Precision-Recall Curves - {model_name}")
        roc_pr.plot_combined(save_path=str(figures_dir / 'roc_pr_combined.png'))
        with open(reports_dir / 'roc_pr_data.json', 'w') as f:
            json.dump(roc_pr.to_dict(), f, indent=2)
    else:
        logger.warning("y_probs contains NaN, skipping ROC/PR curves.")

    test_sequences = np.array([s['sequence'] for s in test_ds.samples], dtype=object)
    plot_misclassified_examples(
        test_sequences, y_true, y_pred, y_probs,
        max_examples=9,
        save_path=str(figures_dir / 'misclassified_examples.png')
    )

    final_model_path = output_dir / 'final_model.pt'
    trainer.save_model(str(final_model_path))

    if config.get('export.onnx', False):
        logger.info("Exporting to ONNX...")
        try:
            export_onnx(model, str(output_dir / 'model.onnx'), config)
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

    summary = {
        'model_name': model_name,
        'timestamp': timestamp,
        'resumed_from': resume_path,
        'best_val_loss': trainer.best_val_loss,
        'best_val_acc': trainer.best_val_acc,
        'best_epoch': trainer.best_epoch,
        'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics['overall'].items()},
        'model_info': model_info,
        'training_time_per_epoch': history.get('train_time', [None])[-1] if history.get('train_time') else None,
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Resumed from: {resume_path}")
    logger.info(f"Best Epoch: {trainer.best_epoch}")
    logger.info(f"Best Val Loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Best Val Acc: {trainer.best_val_acc:.4f}")
    logger.info(f"Test Accuracy: {metrics['overall']['accuracy']:.4f}")
    logger.info(f"Test F1 (Weighted): {metrics['overall']['f1_weighted']:.4f}")
    logger.info(f"Test Balanced Acc: {metrics['overall']['balanced_accuracy']:.4f}")
    logger.info(f"\nAll outputs saved to: {output_dir}")
    logger.info("=" * 60)

    return summary


def export_onnx(model, path, config):
    """Export model sang ONNX format."""
    model.eval()
    batch_size, seq_len = 1, 100
    device = next(model.parameters()).device
    dummy_seq = torch.randn(batch_size, seq_len, 4).to(device)
    dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    dummy_feat = torch.randn(batch_size, 21).to(device)

    torch.onnx.export(
        model, (dummy_seq, dummy_mask, dummy_feat), path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['sequence', 'mask', 'feature'], output_names=['logits'],
        dynamic_axes={
            'sequence': {0: 'batch_size', 1: 'seq_len'},
            'mask': {0: 'batch_size', 1: 'seq_len'},
            'feature': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        }
    )
    print(f"Model exported to ONNX: {path}")


def main():
    parser = argparse.ArgumentParser(description='FuelSentinel-AI Final Model Training')
    parser.add_argument('--config', type=str, default='src/configs/final_train_cnn_gru.yaml')
    parser.add_argument('--model', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu', 'mps'])
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')

    args = parser.parse_args()
    config = Config.from_yaml(args.config)

    if args.model:
        config.set('model.name', args.model)
    if args.epochs:
        config.set('training.epochs', args.epochs)
    if args.lr:
        config.set('training.learning_rate', args.lr)
    if args.batch_size:
        config.set('data.batch_size', args.batch_size)
    if args.device:
        config.set('device', args.device)

    resume_path = args.resume

    summary = train_final_model(config, resume_path)
    print(f"\n✅ Training completed!")
    print(f"Test Accuracy: {summary['test_metrics']['accuracy']:.4f}")
    print(f"Test F1 (Weighted): {summary['test_metrics']['f1_weighted']:.4f}")


if __name__ == '__main__':
    main()