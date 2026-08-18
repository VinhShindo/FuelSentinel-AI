#!/usr/bin/env python3
"""
Module: seed.py
Mục tiêu: Đảm bảo reproducibility cho training.
Set seed cho tất cả thư viện: random, numpy, torch.
"""

import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set seed cho tất cả thư viện để đảm bảo reproducibility.
    
    Args:
        seed: Giá trị seed.
        deterministic: Nếu True, set thêm các flag deterministic cho PyTorch.
        
    Note:
        Deterministic mode có thể làm chậm training một chút.
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Set environment variable
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        logger.info("Deterministic mode enabled. Training may be slower.")
    
    logger.info(f"Seed set to {seed}")


def get_seed() -> int:
    """Lấy seed hiện tại từ PyTorch."""
    return torch.initial_seed()


if __name__ == "__main__":
    set_seed(42, deterministic=True)
    
    # Test reproducibility
    a = torch.randn(3, 3)
    set_seed(42)
    b = torch.randn(3, 3)
    print(f"Reproducible: {torch.allclose(a, b)}")