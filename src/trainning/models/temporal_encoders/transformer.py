#!/usr/bin/env python3
"""
Module: transformer.py
Mục tiêu: Transformer Encoder cho chuỗi thời gian.
Sử dụng multi-head self-attention.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding cho Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder cho chuỗi thời gian.
    
    Input: (batch, seq_len, input_dim)
    Output: (batch, d_model)
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        max_seq_len: int = 1024,
        pooling: str = 'mean',  # 'mean', 'max', 'last'
    ):
        """
        Args:
            input_dim: Số kênh đầu vào.
            d_model: Kích thước embedding.
            nhead: Số head trong multi-head attention.
            num_layers: Số lớp TransformerEncoderLayer.
            dim_feedforward: Kích thước feedforward.
            dropout: Dropout.
            max_seq_len: Độ dài tối đa cho positional encoding.
            pooling: Cách pooling output ('mean', 'max', 'last').
        """
        super().__init__()
        
        self.d_model = d_model
        self.pooling = pooling
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
            mask: (batch, seq_len) - True cho vị trí thực
            
        Returns:
            output: (batch, d_model)
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        if mask is not None:
            # Transformer cần mask dạng (batch, seq_len) với False cho padding
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None
        
        x = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch, seq_len, d_model)
        
        # Pooling
        if self.pooling == 'mean':
            if mask is not None:
                # Masked mean pooling
                x = x * mask.unsqueeze(-1).float()
                x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif self.pooling == 'max':
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            x = x.max(dim=1).values
        elif self.pooling == 'last':
            if mask is not None:
                # Lấy vị trí cuối cùng không phải padding
                lengths = mask.sum(dim=1) - 1  # (batch,)
                lengths = lengths.clamp(min=0).long()
                x = x[torch.arange(x.size(0)), lengths]
            else:
                x = x[:, -1, :]
        
        return x
    
    def get_output_dim(self) -> int:
        return self.output_dim