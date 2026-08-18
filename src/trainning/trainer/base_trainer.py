#!/usr/bin/env python3
"""
Module: base_trainer.py
Mục tiêu: Abstract Base Trainer định nghĩa interface chuẩn.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """
    Abstract Base Class cho Trainer.
    
    Subclass phải implement:
        - train_one_epoch()
        - validate()
        - predict()
    """
    
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
    ):
        """
        Args:
            model: PyTorch model.
            train_loader: DataLoader cho training.
            val_loader: DataLoader cho validation.
            optimizer: Optimizer.
            criterion: Loss function.
            device: Device (CPU/CUDA).
            config: Config object.
            scheduler: Learning rate scheduler (optional).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        
        self.current_epoch = 0
        self.best_metrics = {}
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        self.callbacks = None
    
    @abstractmethod
    def train_one_epoch(self) -> Dict[str, float]:
        """Train một epoch, trả về dict metrics."""
        pass
    
    @abstractmethod
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate trên loader, trả về dict metrics."""
        pass
    
    @abstractmethod
    def predict(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Dự đoán trên loader, trả về dict predictions."""
        pass
    
    @abstractmethod
    def train(self, epochs: int) -> Dict:
        """Training loop chính."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Lưu checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        pass