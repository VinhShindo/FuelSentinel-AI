#!/usr/bin/env python3
"""
Module: classifier.py
Mục tiêu: Classification head cho bài toán 4 lớp.
"""

import torch
import torch.nn as nn
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ClassifierHead(nn.Module):
    """
    Classification head đơn giản với 2 lớp Linear.
    
    Input: (batch, input_dim)
    Output: (batch, num_classes)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
        activation: str = 'relu',
    ):
        """
        Args:
            input_dim: Kích thước đầu vào (từ fusion layer).
            num_classes: Số lớp đầu ra.
            hidden_dim: Kích thước hidden (mặc định = input_dim // 2).
            dropout: Dropout rate.
            activation: Hàm activation ('relu', 'gelu', 'silu').
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, num_classes * 8)
        
        # Chọn activation
        act_fn = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
        }.get(activation.lower(), nn.ReLU())
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self._initialize_weights()
        
        logger.debug(
            f"ClassifierHead: {input_dim} → {hidden_dim} → {hidden_dim // 2} → {num_classes}"
        )
    
    def _initialize_weights(self) -> None:
        """Khởi tạo trọng số với Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, input_dim)
            
        Returns:
            logits: (batch, num_classes)
        """
        return self.classifier(x)


class MultiHeadClassifier(nn.Module):
    """
    Classifier với nhiều head (cho bài toán multi-task nếu cần mở rộng).
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        num_heads: int = 1,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Kích thước đầu vào.
            num_classes: Số lớp cho mỗi head.
            num_heads: Số lượng head.
            hidden_dim: Kích thước hidden.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.heads = nn.ModuleList([
            ClassifierHead(input_dim, num_classes, hidden_dim, dropout)
            for _ in range(num_heads)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            List các logits từ mỗi head.
        """
        return [head(x) for head in self.heads]


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    batch_size = 16
    input_dim = 256
    num_classes = 4
    
    x = torch.randn(batch_size, input_dim)
    
    # Test ClassifierHead
    classifier = ClassifierHead(input_dim, num_classes, dropout=0.3)
    logits = classifier(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Logits (first sample): {logits[0]}")
    print(f"Predicted class: {logits.argmax(dim=1)}")
    
    # Test with probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"Probabilities sum: {probs.sum(dim=1)}")
    
    # Test MultiHeadClassifier
    multi_head = MultiHeadClassifier(input_dim, num_classes, num_heads=2)
    outputs = multi_head(x)
    print(f"\nMultiHead outputs: {len(outputs)} heads")
    for i, out in enumerate(outputs):
        print(f"  Head {i}: shape={out.shape}")
    
    print("\nAll classifier tests passed!")