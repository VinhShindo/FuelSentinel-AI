#!/usr/bin/env python3
"""
Module: logger.py
Mục tiêu: Cấu hình logging cho toàn bộ project.
Hỗ trợ log ra console, file, và TensorBoard.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "FuelSentinel-AI",
    log_dir: Optional[str] = None,
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Cấu hình logger cho project.
    
    Args:
        name: Tên logger.
        log_dir: Thư mục lưu log file.
        level: Mức log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Ghi log ra file.
        log_to_console: In log ra console.
        
    Returns:
        Logger đã cấu hình.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger(name: str = "FuelSentinel-AI") -> logging.Logger:
    """
    Lấy logger đã được cấu hình.
    
    Args:
        name: Tên logger.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# Default logger
logger = get_logger()


if __name__ == "__main__":
    # Test
    log = setup_logger(
        name="Test",
        log_dir="outputs/logs",
        level="DEBUG"
    )
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")