#!/usr/bin/env python3
"""
Trainer package for FuelSentinel-AI training pipeline.
"""

from .optimizer import OptimizerFactory, create_optimizer, create_scheduler, create_loss
from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
    TensorBoardLogger,
    LearningRateMonitor,
    CallbackList,
)
from .trainer import Trainer
from .base_trainer import BaseTrainer

__all__ = [
    'OptimizerFactory',
    'create_optimizer',
    'create_scheduler',
    'create_loss',
    'EarlyStopping',
    'ModelCheckpoint',
    'CSVLogger',
    'TensorBoardLogger',
    'LearningRateMonitor',
    'CallbackList',
    'Trainer',
    'BaseTrainer',
]