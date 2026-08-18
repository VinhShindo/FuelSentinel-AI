#!/usr/bin/env python3
"""
Module: gru.py
Mục tiêu: GRU Encoder cho chuỗi thời gian.
"""

import torch
import torch.nn as nn
from typing import Optional


class GRUEncoder(nn.Module):
    """
    GRU Encoder (unidirectional).
    
    Input: (batch, seq_len, input_dim)
    Output: (batch, hidden_dim)
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Số kênh đầu vào.
            hidden_dim: Kích thước hidden.
            num_layers: Số lớp GRU.
            dropout: Dropout giữa các lớp.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        
        self.output_dim = hidden_dim
    
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
            output: (batch, output_dim)
        """
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            lengths = torch.clamp(lengths, min=1)
            
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, hn = self.gru(packed)
        else:
            gru_out, hn = self.gru(x)
        
        # Lấy hidden state cuối cùng
        output = hn[-1, :, :]
        
        return output
    
    def get_output_dim(self) -> int:
        return self.output_dim