#!/usr/bin/env python3
"""
Module: runner.py
Mục tiêu: Chạy benchmark cho tất cả model trong config.
Lưu log, checkpoint, tensorboard RIÊNG cho từng model.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import numpy as np

from ..models.factory import ModelFactory, list_available_models
from ..datasets.dataloader import DataLoaderFactory
from ..trainer.trainer import Trainer
from ..trainer.optimizer import OptimizerFactory
from ..evaluation.metrics import compute_all_metrics
from ..utils.config import Config
from ..utils.logger import setup_logger
from ..models import all_models

logger = logging.getLogger(__name__)


class SimpleDict:
    """Dict-like object hỗ trợ .get() và truy cập thuộc tính."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def get(self, key, default=None):
        return getattr(self, key, default) if hasattr(self, key) else default


class BenchmarkRunner:
    def __init__(self, config, train_dataset, val_dataset, test_dataset):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        from ..utils.device import get_device
        self.device = get_device(config.get('device', 'auto'))
        
        self.dataloader_factory = DataLoaderFactory(config)
        self.models_to_benchmark = config.get('models_to_benchmark', list_available_models())
        
        self.results = {}
        self.best_model_name = None
        self.best_model_path = None
        self.base_output_dir = Path(self.config.output.base_dir)
        
        logger.info(f"BenchmarkRunner initialized for {len(self.models_to_benchmark)} models")
        logger.info(f"Models: {self.models_to_benchmark}")
        logger.info(f"Device: {self.device}")
    
    def run(self) -> Dict[str, Dict]:
        logger.info("=" * 70)
        logger.info("BENCHMARK STARTED")
        logger.info("=" * 70)
        
        for i, model_name in enumerate(self.models_to_benchmark, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Model {i}/{len(self.models_to_benchmark)}: {model_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = self._benchmark_single_model(model_name)
                self.results[model_name] = result
                logger.info(f"✓ {model_name} benchmark completed")
                logger.info(f"  Accuracy: {result['metrics']['overall']['accuracy']:.4f}")
                logger.info(f"  F1 (Weighted): {result['metrics']['overall']['f1_weighted']:.4f}")
                logger.info(f"  Training Time: {result['training_time_seconds']:.1f}s")
            except Exception as e:
                logger.error(f"✗ {model_name} benchmark failed: {e}", exc_info=True)
                self.results[model_name] = {'error': str(e)}
        
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK COMPLETED")
        logger.info("=" * 70)
        self._determine_best_model()
        return self.results
    
    def _benchmark_single_model(self, model_name: str) -> Dict:
        # Thư mục riêng
        model_output_dir = self.base_output_dir / model_name
        model_log_dir = model_output_dir / 'logs'
        model_checkpoint_dir = model_output_dir / 'checkpoints'
        model_tb_dir = model_output_dir / 'tensorboard'
        
        for d in [model_output_dir, model_log_dir, model_checkpoint_dir, model_tb_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Logger riêng
        model_logger = setup_logger(
            name=f'Benchmark_{model_name}',
            log_dir=str(model_log_dir),
            level='INFO',
        )
        model_logger.info(f"Benchmark: {model_name}")
        
        # DataLoader
        train_loader, val_loader, test_loader = self.dataloader_factory.create_all(
            self.train_dataset, self.val_dataset, self.test_dataset
        )
        
        # Model
        model_config = self.config.model.to_dict() if hasattr(self.config.model, 'to_dict') else dict(self.config.model)
        model_config['name'] = model_name
        
        excluded_keys = ['name', 'nhead', 'num_transformer_layers', 'dim_feedforward',
                         'tcn_channels', 'cnn_channels', 'cnn_kernel_size',
                         'fusion_type', 'fusion_dim']
        
        model = ModelFactory.create_from_name(model_name, **{
            k: v for k, v in model_config.items() if k not in excluded_keys
        })
        model = model.to(self.device)
        
        model_info = model.get_model_info()
        model_logger.info(f"Parameters: {model_info['total_params']:,}")
        
        # Optimizer & Loss
        optimizer = OptimizerFactory.create_optimizer(model, self.config)
        scheduler = OptimizerFactory.create_scheduler(optimizer, self.config)
        criterion = OptimizerFactory.create_loss(self.config)
        
        # Config tạm với SimpleDict (hỗ trợ .get())
        temp_config = SimpleDict()
        temp_config.training = self.config.training
        temp_config.checkpoint = SimpleDict(
            checkpoint_dir=str(model_checkpoint_dir),
            save_best=True,
            save_last=True,
        )
        temp_config.logging = SimpleDict(
            log_dir=str(model_log_dir),
            tensorboard_dir=str(model_tb_dir),
            csv_logger=True,
            tensorboard=True,
        )
        temp_config.output = SimpleDict(base_dir=str(model_output_dir))
        
        # Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            config=temp_config,
            scheduler=scheduler,
            test_loader=test_loader,
            use_amp=self.config.training.get('use_amp', True),
            gradient_clip_val=self.config.training.get('gradient_clip_val', 1.0),
        )
        
        # Train
        epochs = self.config.training.get('epochs', 100)
        training_start = time.time()
        history = trainer.train(epochs=epochs)
        training_time = time.time() - training_start
        
        # Evaluate
        predictions = trainer.predict(test_loader)
        inference_time = (time.time() - training_start - training_time) / len(self.test_dataset) if len(self.test_dataset) > 0 else 0
        
        y_true = predictions['labels'].numpy()
        y_pred = predictions['preds'].numpy()
        y_probs = predictions['probs'].numpy()
        
        metrics = compute_all_metrics(y_true, y_pred, y_probs)
        
        # Kết quả
        result = {
            'model_name': model_name,
            'output_dir': str(model_output_dir),
            'model_path': str(model_checkpoint_dir / 'best_model.pt'),
            'metrics': metrics,
            'training_time_seconds': training_time,
            'inference_time_per_sample': inference_time,
            'best_val_loss': trainer.best_val_loss,
            'best_val_acc': trainer.best_val_acc,
            'best_epoch': trainer.best_epoch,
            'total_params': model_info['total_params'],
            'trainable_params': model_info['trainable_params'],
            'model_size_mb': model_info['model_size_mb'],
            'flops': None,
            'history': history,
            'model_info': model_info,
        }
        
        # Lưu results.json
        results_file = model_output_dir / 'results.json'
        try:
            serializable = {
                'model_name': result['model_name'],
                'output_dir': result['output_dir'],
                'model_path': result['model_path'],
                'training_time_seconds': result['training_time_seconds'],
                'inference_time_per_sample': result['inference_time_per_sample'],
                'best_val_loss': result['best_val_loss'],
                'best_val_acc': result['best_val_acc'],
                'best_epoch': result['best_epoch'],
                'total_params': result['total_params'],
                'model_size_mb': result['model_size_mb'],
                'metrics': {
                    'overall': {
                        k: float(v) if isinstance(v, (np.floating, float, np.integer)) else str(v)
                        for k, v in metrics['overall'].items()
                    },
                },
            }
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        except Exception as e:
            model_logger.warning(f"Failed to save results.json: {e}")
        
        model_logger.info(f"Benchmark {model_name} COMPLETED")
        return result
    
    def _determine_best_model(self) -> None:
        best_acc = 0.0
        for model_name, result in self.results.items():
            if 'error' in result:
                continue
            acc = result['metrics']['overall']['accuracy']
            if acc > best_acc:
                best_acc = acc
                self.best_model_name = model_name
                self.best_model_path = result.get('model_path')
        
        if self.best_model_name:
            logger.info(f"\nBEST MODEL: {self.best_model_name} (Accuracy: {best_acc:.4f})")
        else:
            logger.warning("No successful model found!")
    
    def save_results(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable_results = {}
        for model_name, result in self.results.items():
            if 'error' in result:
                serializable_results[model_name] = {'error': result['error']}
            else:
                serializable_results[model_name] = {
                    'model_name': result['model_name'],
                    'output_dir': result['output_dir'],
                    'model_path': result['model_path'],
                    'training_time_seconds': result['training_time_seconds'],
                    'inference_time_per_sample': result['inference_time_per_sample'],
                    'best_val_loss': result['best_val_loss'],
                    'best_val_acc': result['best_val_acc'],
                    'best_epoch': result['best_epoch'],
                    'total_params': result['total_params'],
                    'model_size_mb': result['model_size_mb'],
                    'metrics': {
                        'overall': {
                            k: float(v) if isinstance(v, (np.floating, float, np.integer)) else str(v)
                            for k, v in result['metrics']['overall'].items()
                        },
                    },
                }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark results saved to {path}")
    
    def get_best_model_config(self) -> Dict:
        if self.best_model_name is None:
            self._determine_best_model()
        if self.best_model_name is None:
            return {}
        return {
            'model_name': self.best_model_name,
            'model_path': self.best_model_path,
            'output_dir': str(self.base_output_dir / self.best_model_name),
        }