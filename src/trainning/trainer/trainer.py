#!/usr/bin/env python3
"""
Module: trainer.py
Mục tiêu: Trainer đầy đủ với AMP, gradient clipping, logging, callbacks.
Hỗ trợ training, validation, testing, resume từ checkpoint.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from .base_trainer import BaseTrainer
from .callbacks import CallbackList, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoardLogger, LearningRateMonitor

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: Any,
        scheduler: Optional[Any] = None,
        test_loader: Optional[DataLoader] = None,
        callbacks: Optional[list] = None,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0,
        log_interval: int = 10,
    ):
        super().__init__(model, train_loader, val_loader, optimizer, criterion, device, config, scheduler)

        self.test_loader = test_loader
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        self.log_interval = log_interval

        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        self.callbacks = CallbackList(callbacks or [])
        self._setup_default_callbacks()

        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'train_time': [],
            'val_time': [],
        }

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        self.patience_counter = 0
        self.training_start_time = None

    def _setup_default_callbacks(self) -> None:
        # Early Stopping
        if hasattr(self.config, 'training'):
            patience = self.config.training.get('early_stopping_patience', 15)
            min_delta = self.config.training.get('early_stopping_min_delta', 0.001)
        else:
            patience = 15
            min_delta = 0.001

        has_early_stop = any(isinstance(cb, EarlyStopping) for cb in self.callbacks.callbacks)
        if not has_early_stop and patience > 0:
            self.callbacks.add(EarlyStopping(
                monitor='val_loss', mode='min',
                patience=patience, min_delta=min_delta,
            ))

        # Model Checkpoint
        if hasattr(self.config, 'checkpoint'):
            checkpoint_dir = self.config.checkpoint.get('checkpoint_dir', 'outputs/checkpoints')
        else:
            checkpoint_dir = 'outputs/checkpoints'

        has_checkpoint = any(isinstance(cb, ModelCheckpoint) for cb in self.callbacks.callbacks)
        if not has_checkpoint:
            self.callbacks.add(ModelCheckpoint(
                checkpoint_dir=checkpoint_dir,
                monitor='val_loss', mode='min',
                save_best=True, save_last=True,
            ))

        # CSV Logger
        if hasattr(self.config, 'logging'):
            log_dir = self.config.logging.get('log_dir', 'outputs/logs')
            use_csv = self.config.logging.get('csv_logger', True)
        else:
            log_dir = 'outputs/logs'
            use_csv = True

        if use_csv:
            has_csv = any(isinstance(cb, CSVLogger) for cb in self.callbacks.callbacks)
            if not has_csv:
                self.callbacks.add(CSVLogger(log_dir=log_dir))

        # TensorBoard
        if hasattr(self.config, 'logging'):
            use_tb = self.config.logging.get('tensorboard', True)
            tb_dir = self.config.logging.get('tensorboard_dir', 'outputs/tensorboard')
        else:
            use_tb = True
            tb_dir = 'outputs/tensorboard'

        if use_tb:
            has_tb = any(isinstance(cb, TensorBoardLogger) for cb in self.callbacks.callbacks)
            if not has_tb:
                self.callbacks.add(TensorBoardLogger(log_dir=tb_dir))

        # LR Monitor
        has_lr = any(isinstance(cb, LearningRateMonitor) for cb in self.callbacks.callbacks)
        if not has_lr:
            self.callbacks.add(LearningRateMonitor(verbose=False))

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total_samples = 0.0, 0, 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            sequence = batch['sequence'].to(self.device)
            mask = batch['mask'].to(self.device)
            feature = batch['feature'].to(self.device)
            labels = batch['label'].to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(sequence, mask, feature)
                loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.gradient_clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * sequence.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += sequence.size(0)

            if batch_idx % self.log_interval == 0 or batch_idx == len(self.train_loader) - 1:
                logger.debug(f"  Batch {batch_idx}/{len(self.train_loader)} | Loss: {total_loss/total_samples:.4f} | Acc: {correct/total_samples:.4f}")

            self.global_step += 1

        epoch_time = time.time() - epoch_start
        return {'loss': total_loss / total_samples, 'accuracy': correct / total_samples, 'time': epoch_time}

    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        total_loss, correct, total_samples = 0.0, 0, 0
        val_start = time.time()

        for batch in loader:
            sequence = batch['sequence'].to(self.device)
            mask = batch['mask'].to(self.device)
            feature = batch['feature'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.model(sequence, mask, feature)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * sequence.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += sequence.size(0)

        val_time = time.time() - val_start
        return {'loss': total_loss / total_samples, 'accuracy': correct / total_samples, 'time': val_time}

    @torch.no_grad()
    def predict(self, loader: Optional[DataLoader] = None) -> Dict[str, torch.Tensor]:
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        all_logits, all_preds, all_labels, all_probs = [], [], [], []

        for batch in loader:
            sequence = batch['sequence'].to(self.device)
            mask = batch['mask'].to(self.device)
            feature = batch['feature'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.model(sequence, mask, feature)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_logits.append(logits.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

        return {
            'logits': torch.cat(all_logits),
            'preds': torch.cat(all_preds),
            'labels': torch.cat(all_labels),
            'probs': torch.cat(all_probs),
        }

    def train(self, epochs: int) -> Dict:
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Device: {self.device}, AMP: {self.use_amp}, Gradient clipping: {self.gradient_clip_val}")
        self.training_start_time = time.time()
        self.callbacks.on_train_start(self)

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            self.callbacks.on_epoch_start(self, epoch)

            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(current_lr)
            self.history['train_time'].append(train_metrics['time'])
            self.history['val_time'].append(val_metrics['time'])

            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch + 1

            logs = {
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'learning_rate': current_lr,
            }
            self.callbacks.on_epoch_end(self, epoch, logs)

            for callback in self.callbacks.callbacks:
                if isinstance(callback, EarlyStopping) and callback.should_stop:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                continue
            break

        total_time = time.time() - self.training_start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        logger.info(f"Best val accuracy: {self.best_val_acc:.4f}")
        self.callbacks.on_train_end(self)
        return self.history

    def test(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        if loader is None:
            loader = self.test_loader
        if loader is None:
            raise ValueError("No test loader provided.")
        logger.info("Evaluating on test set...")
        test_metrics = self.validate(loader)
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}")
        return test_metrics

    def save_checkpoint(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)

        # Merge history để đảm bảo đủ key
        loaded_history = checkpoint.get('history', {})
        default_history = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': [], 'train_time': [], 'val_time': [],
        }
        if loaded_history:
            for key in default_history:
                if key not in loaded_history:
                    loaded_history[key] = []
            self.history = loaded_history
        else:
            self.history = default_history

        logger.info(f"Checkpoint loaded. Resuming from epoch {self.current_epoch}")

    def save_model(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")