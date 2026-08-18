#!/usr/bin/env python3
"""
Module: tcn.py
Mục tiêu: Temporal Convolutional Network (TCN) Encoder.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class Chomp1d(nn.Module):
    """Cắt bỏ padding để đảm bảo causality."""
    
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Một block TCN với dilated convolution và residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network Encoder.
    
    Input: (batch, seq_len, input_dim)
    Output: (batch, last_channel)
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        channels: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Số kênh đầu vào.
            channels: Số channels cho mỗi lớp.
            kernel_size: Kernel size.
            dropout: Dropout.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = channels[-1]
        
        layers = []
        num_levels = len(channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else channels[i-1]
            out_ch = channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        
        self.network = nn.Sequential(*layers)
    
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
        # Chuyển về (batch, input_dim, seq_len) cho Conv1d
        x = x.transpose(1, 2)
        
        # TCN
        x = self.network(x)  # (batch, last_channel, seq_len)
        
        # Global max pooling
        if mask is not None:
            # Mask pooling
            mask_conv = mask.unsqueeze(1).float()  # (batch, 1, seq_len)
            x = x * mask_conv
            x = x.max(dim=2).values
        else:
            x = x.max(dim=2).values
        
        return x
    
    def get_output_dim(self) -> int:
        return self.output_dim