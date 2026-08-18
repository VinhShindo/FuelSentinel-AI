#!/usr/bin/env python3
"""
Module: callbacks.py
Mục tiêu: Callbacks cho training loop:
    - EarlyStopping
    - ModelCheckpoint
    - CSVLogger
    - TensorBoardLogger
    - LearningRateMonitor
"""

import logging
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# =============================================================================
# Base Callback
# =============================================================================

class Callback:
    """Base class cho tất cả callbacks."""
    
    def on_train_start(self, trainer) -> None:
        """Gọi khi bắt đầu training."""
        pass
    
    def on_train_end(self, trainer) -> None:
        """Gọi khi kết thúc training."""
        pass
    
    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Gọi khi bắt đầu mỗi epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict) -> None:
        """Gọi khi kết thúc mỗi epoch."""
        pass
    
    def on_batch_start(self, trainer, batch_idx: int) -> None:
        """Gọi khi bắt đầu mỗi batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, logs: Dict) -> None:
        """Gọi khi kết thúc mỗi batch."""
        pass
    
    def on_validation_start(self, trainer) -> None:
        """Gọi khi bắt đầu validation."""
        pass
    
    def on_validation_end(self, trainer, logs: Dict) -> None:
        """Gọi khi kết thúc validation."""
        pass


class CallbackList:
    """Quản lý danh sách callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)
    
    def __getattr__(self, name: str):
        """Gọi method trên tất cả callbacks."""
        def method(*args, **kwargs):
            for callback in self.callbacks:
                fn = getattr(callback, name, None)
                if fn is not None:
                    fn(*args, **kwargs)
        return method


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping(Callback):
    """
    Dừng training sớm nếu metric không cải thiện.
    
    Args:
        monitor: Metric để theo dõi (mặc định: 'val_loss').
        mode: 'min' hoặc 'max'.
        patience: Số epoch chờ đợi trước khi dừng.
        min_delta: Ngưỡng thay đổi tối thiểu để coi là cải thiện.
        verbose: In log.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'min',
        patience: int = 15,
        min_delta: float = 0.001,
        verbose: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.stopped_epoch = 0
        self.should_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y - min_delta
            self.best_score = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda x, y: x > y + min_delta
            self.best_score = float('-inf')
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict) -> None:
        current = logs.get(self.monitor)
        
        if current is None:
            logger.warning(f"EarlyStopping: {self.monitor} not found in logs. Available: {list(logs.keys())}")
            return
        
        if self.monitor_op(current, self.best_score):
            self.best_score = current
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(f"EarlyStopping: {self.monitor} improved to {current:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} did not improve. "
                    f"Counter: {self.counter}/{self.patience}"
                )
            
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                logger.info(f"EarlyStopping triggered at epoch {epoch}")
    
    def on_train_end(self, trainer) -> None:
        if self.should_stop:
            logger.info(
                f"Training stopped early at epoch {self.stopped_epoch}. "
                f"Best {self.monitor}: {self.best_score:.6f} at epoch {self.best_epoch}"
            )


# =============================================================================
# Model Checkpoint
# =============================================================================

class ModelCheckpoint(Callback):
    """
    Lưu checkpoint trong quá trình training.
    
    Args:
        checkpoint_dir: Thư mục lưu checkpoint.
        monitor: Metric để theo dõi.
        mode: 'min' hoặc 'max'.
        save_best: Lưu model tốt nhất.
        save_last: Lưu model cuối cùng.
        save_interval: Lưu mỗi N epoch (0 = không lưu định kỳ).
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'outputs/checkpoints',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best: bool = True,
        save_last: bool = True,
        save_interval: int = 0,
    ):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.save_interval = save_interval
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
    
    def _save_checkpoint(self, trainer, filename: str) -> None:
        """Lưu checkpoint."""
        checkpoint = {
            'epoch': trainer.current_epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_score': self.best_score,
            'config': trainer.config.to_dict() if hasattr(trainer.config, 'to_dict') else {},
            'metrics': trainer.history,
        }
        
        if trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict) -> None:
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        # Save best
        if self.save_best:
            is_better = (
                (self.mode == 'min' and current < self.best_score) or
                (self.mode == 'max' and current > self.best_score)
            )
            
            if is_better:
                self.best_score = current
                self.best_epoch = epoch
                self._save_checkpoint(trainer, 'best_model.pt')
        
        # Save last
        if self.save_last:
            self._save_checkpoint(trainer, 'last_model.pt')
        
        # Save interval
        if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
            self._save_checkpoint(trainer, f'checkpoint_epoch_{epoch+1}.pt')


# =============================================================================
# CSV Logger
# =============================================================================

class CSVLogger(Callback):
    """
    Log metrics ra file CSV.
    """
    
    def __init__(self, log_dir: str = 'outputs/logs', filename: str = 'training_log.csv'):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self.csv_writer = None
        self.csv_file = None
        self.header_written = False
    
    def on_train_start(self, trainer) -> None:
        self.csv_file = open(self.filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict) -> None:
        if not self.header_written:
            header = ['epoch'] + list(logs.keys())
            self.csv_writer.writerow(header)
            self.header_written = True
        
        row = [epoch + 1] + [logs.get(k, '') for k in logs.keys()]
        self.csv_writer.writerow(row)
        self.csv_file.flush()
    
    def on_train_end(self, trainer) -> None:
        if self.csv_file:
            self.csv_file.close()
            logger.info(f"Training log saved to {self.filepath}")


# =============================================================================
# TensorBoard Logger
# =============================================================================

class TensorBoardLogger(Callback):
    """
    Log metrics lên TensorBoard.
    """
    
    def __init__(
        self,
        log_dir: str = 'outputs/tensorboard',
        log_graph: bool = True,
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_graph = log_graph
        self.writer = None
        self.global_step = 0
    
    def on_train_start(self, trainer) -> None:
        self.writer = SummaryWriter(self.log_dir)
        
        if self.log_graph and hasattr(trainer, 'model'):
            try:
                # Dùng torch.jit.trace thay vì add_graph
                device = next(trainer.model.parameters()).device
                model_eval = trainer.model.eval()
                
                dummy_seq = torch.randn(1, 50, trainer.model.input_dim).to(device)
                dummy_feat = torch.randn(1, trainer.model.feature_dim).to(device)
                dummy_mask = torch.ones(1, 50, dtype=torch.bool).to(device)
                
                # Chỉ trace 1 lần với torch.jit
                traced = torch.jit.trace(model_eval, (dummy_seq, dummy_mask, dummy_feat))
                self.writer.add_graph(traced, (dummy_seq, dummy_mask, dummy_feat))
            except Exception as e:
                logger.debug(f"Could not log model graph: {e}")
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict) -> None:
        for key, value in logs.items():
            if isinstance(value, (int, float, np.floating)):
                self.writer.add_scalar(key, value, epoch + 1)
        
        # Log learning rate
        if trainer.optimizer:
            lr = trainer.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('learning_rate', lr, epoch + 1)
    
    def on_train_end(self, trainer) -> None:
        if self.writer:
            self.writer.close()
            logger.info(f"TensorBoard logs saved to {self.log_dir}")


# =============================================================================
# Learning Rate Monitor
# =============================================================================

class LearningRateMonitor(Callback):
    """
    Log learning rate trong quá trình training.
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        self.lr_history = []
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict) -> None:
        current_lr = trainer.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        if self.verbose:
            logger.info(f"Learning Rate: {current_lr:.2e}")
        
        # Thêm vào logs
        logs['lr'] = current_lr


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test callbacks
    import sys
    
    # Mock trainer
    class MockTrainer:
        def __init__(self):
            self.current_epoch = 0
            self.model = nn.Linear(10, 4)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.scheduler = None
            self.config = {'model': {'name': 'test'}}
            self.history = {}
    
    trainer = MockTrainer()
    
    # Test EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=3)
    
    metrics_history = [
        {'val_loss': 0.5},
        {'val_loss': 0.4},
        {'val_loss': 0.45},
        {'val_loss': 0.46},
        {'val_loss': 0.47},
        {'val_loss': 0.48},
    ]
    
    for epoch, metrics in enumerate(metrics_history):
        es.on_epoch_end(trainer, epoch, metrics)
        if es.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Test CSVLogger
    csv_logger = CSVLogger('outputs/test_logs', 'test.csv')
    csv_logger.on_train_start(trainer)
    csv_logger.on_epoch_end(trainer, 0, {'train_loss': 0.5, 'val_loss': 0.4, 'accuracy': 0.8})
    csv_logger.on_epoch_end(trainer, 1, {'train_loss': 0.4, 'val_loss': 0.35, 'accuracy': 0.85})
    csv_logger.on_train_end(trainer)
    
    print("\nAll callback tests passed!")