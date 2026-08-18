#!/usr/bin/env python3
"""
Module: config.py
Mục tiêu: Load, validate và quản lý cấu hình YAML cho training pipeline.
Hỗ trợ dot notation access, merge với default config.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """
    Dictionary hỗ trợ dot notation access.
    Ví dụ: config.data.batch_size thay vì config['data']['batch_size']
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
    
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value
    
    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def to_dict(self) -> Dict:
        """Convert về dict thông thường."""
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


class Config:
    """
    Quản lý cấu hình cho training pipeline.
    
    Usage:
        config = Config.from_yaml('configs/benchmark.yaml')
        batch_size = config.data.batch_size
        config.model.hidden_dim = 256  # override
    """
    
    # Default config áp dụng cho mọi phase
    DEFAULT_CONFIG = {
        'project': 'FuelSentinel-AI',
        'seed': 42,
        'deterministic': True,
        'device': 'auto',  # auto, cuda, cpu
        
        'data': {
            'train_path': 'data/splits/train.csv',
            'val_path': 'data/splits/val.csv',
            'test_path': 'data/splits/test.csv',
            'batch_size': 64,
            'num_workers': 4,
            'pin_memory': True,
            'prefetch_factor': 2,
        },
        
        'model': {
            'name': 'bilstm',
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'feature_dim': 21,
            'num_classes': 4,
            # Transformer specific
            'nhead': 8,
            'dim_feedforward': 512,
            'num_transformer_layers': 3,
            # TCN specific
            'tcn_channels': [64, 128, 256],
            'kernel_size': 3,
            # CNN specific
            'cnn_channels': [64, 128],
            'cnn_kernel_size': 3,
            # Fusion
            'fusion_type': 'gate',  # concat, add, gate
            'fusion_dim': 256,
        },
        
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'scheduler_params': {
                'T_max': 100,
                'eta_min': 1e-6,
            },
            'loss': 'cross_entropy',
            'early_stopping_patience': 15,
            'early_stopping_min_delta': 0.001,
            'gradient_clip_val': 1.0,
            'use_amp': True,
            'val_check_interval': 1.0,  # validate every N epochs
        },
        
        'evaluation': {
            'metrics': [
                'accuracy',
                'precision',
                'recall',
                'f1',
                'balanced_accuracy',
                'macro_f1',
                'weighted_f1',
            ],
            'save_confusion_matrix': True,
            'save_roc_curve': True,
            'save_pr_curve': True,
        },
        
        'logging': {
            'level': 'INFO',
            'save_logs': True,
            'log_dir': 'outputs/logs',
            'tensorboard': True,
            'csv_logger': True,
        },
        
        'checkpoint': {
            'save_best': True,
            'save_last': True,
            'save_interval': 10,  # epochs
            'metric_to_monitor': 'val_loss',
            'mode': 'min',  # min for loss, max for accuracy
        },
        
        'output': {
            'base_dir': 'outputs',
            'save_plots': True,
            'export_formats': ['csv', 'json', 'markdown'],
        }
    }
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """
        Khởi tạo config từ dict.
        
        Args:
            config_dict: Dictionary cấu hình, nếu None dùng default.
        """
        merged = self._deep_merge(self.DEFAULT_CONFIG, config_dict or {})
        self._config = ConfigDict(merged)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """
        Load config từ file YAML.
        
        Args:
            yaml_path: Đường dẫn đến file YAML.
            
        Returns:
            Config instance.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            logger.warning(f"Config file {yaml_path} not found, using default config.")
            return cls()
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {yaml_path}")
        return cls(config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'Config':
        """Load config từ file JSON."""
        json_path = Path(json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Merge 2 dicts lồng nhau, override ghi đè base.
        
        Args:
            base: Dict cơ sở.
            override: Dict ghi đè.
            
        Returns:
            Dict đã merge.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def __getattr__(self, key: str) -> Any:
        """Truy cập config qua dot notation."""
        return getattr(self._config, key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Lấy giá trị config với key dạng 'data.batch_size'."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, (dict, ConfigDict)):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set giá trị config với key dạng 'model.hidden_dim'."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = ConfigDict()
            config = config[k]
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """Convert toàn bộ config về dict."""
        return self._config.to_dict()
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Lưu config ra file YAML."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Config saved to {yaml_path}")
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Lưu config ra file JSON."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Config saved to {json_path}")
    
    def __repr__(self) -> str:
        return f"Config({json.dumps(self.to_dict(), indent=2, ensure_ascii=False)})"
    
    def __str__(self) -> str:
        return self.__repr__()


# Singleton config instance (optional)
_global_config = None

def get_global_config() -> Config:
    """Lấy global config instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config

def set_global_config(config: Config) -> None:
    """Set global config instance."""
    global _global_config
    _global_config = config


# Test
if __name__ == "__main__":
    # Test load config
    config = Config.from_yaml('configs/benchmark.yaml')
    print(f"Project: {config.project}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Model: {config.model.name}")
    
    # Test dot notation
    config.model.hidden_dim = 256
    print(f"Updated hidden_dim: {config.model.hidden_dim}")
    
    # Test get/set
    print(f"Learning rate: {config.get('training.learning_rate')}")
    config.set('training.epochs', 200)
    print(f"Updated epochs: {config.training.epochs}")