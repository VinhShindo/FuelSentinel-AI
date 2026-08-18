#!/usr/bin/env python3
"""
Module: cnn_encoder.py
Mục tiêu: 1D CNN Encoder để trích xuất local patterns từ chuỗi thời gian.
Dùng làm tiền xử lý trước LSTM/GRU.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class CNNEncoder(nn.Module):
    """
    1D CNN Encoder cho chuỗi thời gian.
    
    Input: (batch, seq_len, input_dim)
    Output: (batch, seq_len_reduced, last_channel)
    
    Dùng trước LSTM/GRU để trích xuất local patterns.
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        channels: List[int] = [64, 128],
        kernel_size: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_dim: Số kênh đầu vào.
            channels: List số filters cho mỗi lớp CNN.
            kernel_size: Kernel size.
            dropout: Dropout.
            use_batch_norm: Sử dụng BatchNorm sau mỗi lớp CNN.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = channels[-1] if channels else input_dim
        
        layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(channels):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # same padding
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) - optional
            
        Returns:
            output: (batch, seq_len, output_dim)
        """
        # Chuyển về (batch, input_dim, seq_len) cho Conv1d
        x = x.transpose(1, 2)
        
        # Áp dụng mask nếu có
        if mask is not None:
            # Zero out các vị trí padding trước khi convolution
            x = x * mask.unsqueeze(1).float()
        
        # CNN
        x = self.conv_layers(x)
        
        # Chuyển về (batch, seq_len, output_dim)
        x = x.transpose(1, 2)
        
        return x
    
    def get_output_dim(self) -> int:
        return self.output_dim