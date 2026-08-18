#!/usr/bin/env python3
"""
Module: bigru.py
Mục tiêu: Bidirectional GRU Encoder cho chuỗi thời gian.
"""

import torch
import torch.nn as nn
from typing import Optional


class BiGRUEncoder(nn.Module):
    """
    Bidirectional GRU Encoder.
    
    Input: (batch, seq_len, input_dim)
    Output: (batch, hidden_dim * 2)
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
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
            bidirectional=True,
        )
        
        self.output_dim = hidden_dim * 2
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            lengths = torch.clamp(lengths, min=1)
            
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, hn = self.gru(packed)
        else:
            gru_out, hn = self.gru(x)
        
        # Lấy forward và backward của lớp cuối
        forward_hidden = hn[-2, :, :]
        backward_hidden = hn[-1, :, :]
        output = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return output
    
    def get_output_dim(self) -> int:
        return self.output_dim