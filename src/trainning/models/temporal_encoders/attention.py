#!/usr/bin/env python3
"""
Module: attention.py
Mục tiêu: Self-Attention Encoder cho chuỗi thời gian.
"""

import torch
import torch.nn as nn
from typing import Optional


class SelfAttentionEncoder(nn.Module):
    """
    Self-Attention Encoder.
    Dùng multi-head self-attention để học dependencies trong chuỗi.
    
    Input: (batch, seq_len, input_dim)
    Output: (batch, d_model)
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        dropout: float = 0.3,
        pooling: str = 'mean',
    ):
        """
        Args:
            input_dim: Số kênh đầu vào.
            d_model: Kích thước embedding.
            nhead: Số head.
            dropout: Dropout.
            pooling: Cách pooling ('mean', 'max', 'attn').
        """
        super().__init__()
        
        self.d_model = d_model
        self.pooling = pooling
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Attention pooling (nếu dùng)
        if pooling == 'attn':
            self.attn_pool = nn.Linear(d_model, 1)
        
        self.output_dim = d_model
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len)
            
        Returns:
            output: (batch, d_model)
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Self-attention
        if mask is not None:
            attn_mask = ~mask  # False cho padding
        else:
            attn_mask = None
        
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + attn_out)  # Residual connection
        
        # Feedforward
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)  # Residual connection
        
        # Pooling
        if self.pooling == 'mean':
            if mask is not None:
                x = x * mask.unsqueeze(-1).float()
                x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif self.pooling == 'max':
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            x = x.max(dim=1).values
        elif self.pooling == 'attn':
            # Attention pooling
            attn_weights = torch.softmax(self.attn_pool(x).squeeze(-1), dim=1)  # (batch, seq_len)
            if mask is not None:
                attn_weights = attn_weights * mask.float()
                attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
            x = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        return x
    
    def get_output_dim(self) -> int:
        return self.output_dim