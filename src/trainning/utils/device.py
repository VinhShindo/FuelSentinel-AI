#!/usr/bin/env python3
"""
Module: device.py
Mục tiêu: Tự động phát hiện và cấu hình device (CPU/CUDA/MPS).
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(preferred: str = "auto") -> torch.device:
    """
    Tự động chọn device phù hợp.
    
    Args:
        preferred: 'auto', 'cuda', 'cpu', 'mps'
        
    Returns:
        torch.device instance.
    """
    if preferred == "cpu":
        device = torch.device("cpu")
    elif preferred == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
    elif preferred == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            logger.warning("MPS requested but not available, falling back to CPU")
            device = torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    return device


def get_device_info(device: torch.device) -> dict:
    """
    Lấy thông tin chi tiết về device.
    
    Args:
        device: torch.device instance.
        
    Returns:
        Dict chứa thông tin device.
    """
    info = {
        "device_type": str(device),
        "device_name": "CPU",
        "device_count": 1,
        "memory_total_gb": None,
        "memory_free_gb": None,
    }
    
    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
        
        # Memory info
        mem = torch.cuda.get_device_properties(0).total_memory
        info["memory_total_gb"] = mem / (1024**3)
        
        mem_free = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        info["memory_free_gb"] = mem_free / (1024**3)
    
    elif device.type == "mps":
        info["device_name"] = "Apple MPS"
    
    return info


def print_device_info(device: torch.device) -> None:
    """In thông tin device ra log."""
    info = get_device_info(device)
    logger.info(f"Device: {info['device_name']}")
    if info['device_count'] > 1:
        logger.info(f"GPU count: {info['device_count']}")
    if info['memory_total_gb']:
        logger.info(f"Total GPU memory: {info['memory_total_gb']:.2f} GB")


if __name__ == "__main__":
    device = get_device("auto")
    print_device_info(device)
    
    # Test
    x = torch.randn(1000, 1000).to(device)
    y = x @ x.T
    print(f"Test computation on {device}: success")