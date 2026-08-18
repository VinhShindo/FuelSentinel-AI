#!/usr/bin/env python3
"""
Models package for FuelSentinel-AI training pipeline.
"""

from .base_model import BaseStateClassifier
from .fusion import FusionLayer, GatedFusion, ConcatFusion, AddFusion
from .classifier import ClassifierHead
from .factory import ModelFactory, MODEL_REGISTRY, register_model

# QUAN TRỌNG: Import all_models để đăng ký tất cả model
from . import all_models

__all__ = [
    'BaseStateClassifier',
    'FusionLayer',
    'GatedFusion',
    'ConcatFusion',
    'AddFusion',
    'ClassifierHead',
    'ModelFactory',
    'MODEL_REGISTRY',
    'register_model',
]