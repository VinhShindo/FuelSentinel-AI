#!/usr/bin/env python3
"""
Module: optimizer.py
Mục tiêu: Factory tạo optimizer, scheduler và loss function từ config.
Hỗ trợ class weights cho CrossEntropyLoss khi use_class_weights=True.
"""

import logging
from typing import Dict, Optional, Any
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    OneCycleLR,
    LambdaLR,
)

logger = logging.getLogger(__name__)


class OptimizerFactory:
    OPTIMIZERS = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
    }

    SCHEDULERS = {
        'cosine': CosineAnnealingLR,
        'cosine_warm': CosineAnnealingWarmRestarts,
        'reduce_on_plateau': ReduceLROnPlateau,
        'step': StepLR,
        'one_cycle': OneCycleLR,
    }

    @classmethod
    def create_optimizer(cls, model: nn.Module, config) -> optim.Optimizer:
        if hasattr(config, 'training'):
            opt_name = config.training.optimizer
            lr = config.training.learning_rate
            weight_decay = config.training.weight_decay
        else:
            opt_name = config.get('optimizer', 'adamw')
            lr = config.get('learning_rate', 0.001)
            weight_decay = config.get('weight_decay', 1e-5)

        opt_name = opt_name.lower()
        if opt_name not in cls.OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {opt_name}. Available: {list(cls.OPTIMIZERS.keys())}")

        optimizer_class = cls.OPTIMIZERS[opt_name]
        opt_kwargs = {'lr': lr, 'weight_decay': weight_decay}
        if opt_name == 'sgd':
            opt_kwargs['momentum'] = config.get('momentum', 0.9)
            opt_kwargs['nesterov'] = config.get('nesterov', True)

        optimizer = optimizer_class(model.parameters(), **opt_kwargs)
        logger.info(f"Optimizer: {opt_name} (lr={lr}, weight_decay={weight_decay})")
        return optimizer

    @classmethod
    def create_scheduler(cls, optimizer, config):
        if hasattr(config, 'training'):
            sched_name = config.training.scheduler
            sched_params = config.training.get('scheduler_params', {})
        else:
            sched_name = config.get('scheduler', None)
            sched_params = config.get('scheduler_params', {})

        if sched_name is None or sched_name.lower() == 'none':
            logger.info("No scheduler used.")
            return None

        sched_name = sched_name.lower()
        if sched_name not in cls.SCHEDULERS:
            raise ValueError(f"Unknown scheduler: {sched_name}. Available: {list(cls.SCHEDULERS.keys())}")

        scheduler_class = cls.SCHEDULERS[sched_name]

        if sched_name == 'cosine':
            scheduler = scheduler_class(optimizer, T_max=sched_params.get('T_max', 100), eta_min=sched_params.get('eta_min', 1e-6))
        elif sched_name == 'cosine_warm':
            scheduler = scheduler_class(optimizer, T_0=sched_params.get('T_0', 10), T_mult=sched_params.get('T_mult', 2), eta_min=sched_params.get('eta_min', 1e-6))
        elif sched_name == 'reduce_on_plateau':
            scheduler = scheduler_class(optimizer, mode=sched_params.get('mode', 'min'), factor=sched_params.get('factor', 0.5), patience=sched_params.get('patience', 5), min_lr=sched_params.get('min_lr', 1e-7))
        elif sched_name == 'step':
            scheduler = scheduler_class(optimizer, step_size=sched_params.get('step_size', 30), gamma=sched_params.get('gamma', 0.1))
        elif sched_name == 'one_cycle':
            scheduler = scheduler_class(optimizer, max_lr=sched_params.get('max_lr', 0.01), total_steps=sched_params.get('total_steps', 100))
        else:
            scheduler = scheduler_class(optimizer, **sched_params)

        logger.info(f"Scheduler: {sched_name}")
        return scheduler

    @classmethod
    def create_loss(cls, config, train_dataset=None) -> nn.Module:
        """
        Tạo loss function. Nếu use_class_weights=True và train_dataset được cung cấp,
        tự động tính class weights dựa trên phân phối nhãn trong tập train.
        """
        if hasattr(config, 'training'):
            loss_name = config.training.get('loss', 'cross_entropy')
            use_class_weights = config.training.get('use_class_weights', False)
        else:
            loss_name = config.get('loss', 'cross_entropy')
            use_class_weights = config.get('use_class_weights', False)

        loss_name = loss_name.lower()

        if loss_name == 'cross_entropy':
            if use_class_weights and train_dataset is not None:
                # Tính class weights từ tập train
                labels = [sample['label_id'] for sample in train_dataset.samples]
                class_counts = Counter(labels)
                total = sum(class_counts.values())
                # weight = total / count (class hiếm được phạt nặng hơn)
                weights = [total / max(class_counts.get(i, 1), 1) for i in range(4)]
                weights = torch.tensor(weights, dtype=torch.float)
                logger.info(f"Class weights (use_class_weights=True): {weights.tolist()}")
                criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                criterion = nn.CrossEntropyLoss()
        elif loss_name == 'nll':
            criterion = nn.NLLLoss()
        elif loss_name == 'mse':
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

        logger.info(f"Loss function: {loss_name}")
        return criterion


# Convenience functions
def create_optimizer(model, config):
    return OptimizerFactory.create_optimizer(model, config)

def create_scheduler(optimizer, config):
    return OptimizerFactory.create_scheduler(optimizer, config)

def create_loss(config, train_dataset=None):
    return OptimizerFactory.create_loss(config, train_dataset)