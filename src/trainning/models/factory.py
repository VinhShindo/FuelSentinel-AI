#!/usr/bin/env python3
"""
Module: factory.py
Mục tiêu: Model Registry và Factory để tạo model từ config.
Hỗ trợ đăng ký model tùy chỉnh và tự động build từ config name.
"""

import logging
from typing import Dict, Type, Optional, Any

import torch.nn as nn
import torch

from .base_model import BaseStateClassifier

logger = logging.getLogger(__name__)

# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY: Dict[str, Type[BaseStateClassifier]] = {}


def register_model(name: str):
    """
    Decorator để đăng ký model vào registry.
    
    Usage:
        @register_model('bilstm')
        class BiLSTMClassifier(BaseStateClassifier):
            ...
    
    Args:
        name: Tên model (dùng trong config).
    
    Returns:
        Decorator function.
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            logger.warning(f"Model '{name}' already registered. Overwriting.")
        MODEL_REGISTRY[name] = cls
        logger.debug(f"Registered model: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_model_class(name: str) -> Type[BaseStateClassifier]:
    """
    Lấy class model từ registry.
    
    Args:
        name: Tên model.
        
    Returns:
        Model class.
        
    Raises:
        ValueError: Nếu model không tồn tại.
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available}"
        )
    return MODEL_REGISTRY[name]


def list_available_models() -> list:
    """Liệt kê tất cả model đã đăng ký."""
    return list(MODEL_REGISTRY.keys())


# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    """
    Factory class để tạo model từ config.
    
    Usage:
        config = Config.from_yaml('config.yaml')
        model = ModelFactory.create(config)
    """
    
    @staticmethod
    def create(config) -> BaseStateClassifier:
        """
        Tạo model từ config.
        
        Args:
            config: Config object với model.name và model params.
            
        Returns:
            Model instance đã build xong.
        """
        model_name = config.model.name
        model_class = get_model_class(model_name)
        
        # Trích xuất model params từ config
        model_params = config.model.to_dict() if hasattr(config.model, 'to_dict') else dict(config.model)
        
        # Loại bỏ key 'name' vì không phải tham số của model
        model_params.pop('name', None)
        
        logger.info(f"Creating model: {model_name}")
        logger.info(f"Model params: {model_params}")
        
        # Tạo model
        model = model_class(**model_params)
        
        # Build model (khởi tạo các layer chưa được tạo trong __init__)
        if hasattr(model, 'build'):
            model.build()
        
        logger.info(f"Model created: {model.__class__.__name__}")
        logger.info(f"Total parameters: {model.count_parameters():,}")
        
        return model
    
    @staticmethod
    def create_from_name(name: str, **kwargs) -> BaseStateClassifier:
        """
        Tạo model từ tên và tham số.
        
        Args:
            name: Tên model.
            **kwargs: Tham số cho model.
            
        Returns:
            Model instance.
        """
        model_class = get_model_class(name)
        model = model_class(**kwargs)
        if hasattr(model, 'build'):
            model.build()
        return model
    
    @staticmethod
    def register_and_create(
        config,
        custom_models: Optional[Dict[str, Type[BaseStateClassifier]]] = None,
    ) -> BaseStateClassifier:
        """
        Đăng ký model tùy chỉnh (nếu có) rồi tạo model từ config.
        
        Args:
            config: Config object.
            custom_models: Dict các model tùy chỉnh {name: class}.
            
        Returns:
            Model instance.
        """
        if custom_models:
            for name, cls in custom_models.items():
                register_model(name)(cls)
        
        return ModelFactory.create(config)


# =============================================================================
# Đăng ký các model có sẵn (sẽ import sau khi định nghĩa)
# =============================================================================

def _register_builtin_models():
    """
    Đăng ký tất cả model built-in.
    Import ở đây để tránh circular import.
    """
    # Import sẽ được thực hiện khi cần
    pass


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test registry
    @register_model('test_model')
    class TestModel(BaseStateClassifier):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.linear = nn.Linear(self.input_dim, self.hidden_dim)
            self.build()
        
        def encode_sequence(self, x, mask=None):
            return x.mean(dim=1)
        
        def encode_features(self, x_feat):
            return self.feature_encoder(x_feat)
    
    # Test list models
    print(f"Registered models: {list_available_models()}")
    
    # Test get model class
    cls = get_model_class('test_model')
    print(f"Model class: {cls.__name__}")
    
    # Test create
    model = ModelFactory.create_from_name(
        'test_model',
        input_dim=4,
        hidden_dim=128,
        feature_dim=21,
        num_classes=4,
        dropout=0.3,
    )
    print(f"Created model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Test forward
    batch_size = 8
    seq_len = 50
    x_seq = torch.randn(batch_size, seq_len, 4)
    x_feat = torch.randn(batch_size, 21)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    logits = model(x_seq, mask, x_feat)
    print(f"Output shape: {logits.shape}")
    
    print("\nAll factory tests passed!")