#!/usr/bin/env python3
"""
Module: fusion.py
Mục tiêu: Fusion layer kết hợp sequence encoding và feature encoding.
Hỗ trợ 3 kiểu fusion: concat, add, gate.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class FusionLayer(nn.Module):
    """
    Factory class tạo fusion layer dựa trên config.
    
    Các kiểu fusion:
        - 'concat': Concatenate + Linear projection
        - 'add': Cộng trực tiếp (yêu cầu 2 input cùng dim)
        - 'gate': Gated fusion (học trọng số cho từng branch)
    """
    
    def __new__(
        cls,
        seq_dim: int,
        feat_dim: int,
        fusion_type: str = 'concat',
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Tạo fusion layer phù hợp.
        
        Args:
            seq_dim: Kích thước sequence encoding.
            feat_dim: Kích thước feature encoding.
            fusion_type: 'concat', 'add', 'gate'.
            output_dim: Kích thước đầu ra mong muốn.
            dropout: Dropout rate.
            
        Returns:
            Fusion layer instance.
        """
        fusion_type = fusion_type.lower()
        
        if fusion_type == 'concat':
            return ConcatFusion(seq_dim, feat_dim, output_dim, dropout)
        elif fusion_type == 'add':
            return AddFusion(seq_dim, feat_dim, output_dim, dropout)
        elif fusion_type == 'gate':
            return GatedFusion(seq_dim, feat_dim, output_dim, dropout)
        else:
            raise ValueError(
                f"Unknown fusion type: {fusion_type}. "
                f"Choose from: 'concat', 'add', 'gate'"
            )


class ConcatFusion(nn.Module):
    """
    Concatenation Fusion.
    Ghép nối 2 vector và chiếu xuống output_dim.
    """
    
    def __init__(
        self,
        seq_dim: int,
        feat_dim: int,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            seq_dim: Kích thước sequence encoding.
            feat_dim: Kích thước feature encoding.
            output_dim: Kích thước đầu ra.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.seq_dim = seq_dim
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        
        total_dim = seq_dim + feat_dim
        
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        
        logger.debug(
            f"ConcatFusion: {seq_dim} + {feat_dim} → {total_dim} → {output_dim}"
        )
    
    def forward(
        self,
        seq_encoding: torch.Tensor,
        feat_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            seq_encoding: (batch, seq_dim)
            feat_encoding: (batch, feat_dim)
            
        Returns:
            fused: (batch, output_dim)
        """
        fused = torch.cat([seq_encoding, feat_encoding], dim=1)
        return self.projection(fused)


class AddFusion(nn.Module):
    """
    Additive Fusion.
    Chiếu 2 branch về cùng kích thước rồi cộng.
    """
    
    def __init__(
        self,
        seq_dim: int,
        feat_dim: int,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            seq_dim: Kích thước sequence encoding.
            feat_dim: Kích thước feature encoding.
            output_dim: Kích thước đầu ra.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.output_dim = output_dim
        
        # Chiếu cả 2 branch về cùng kích thước
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        logger.debug(
            f"AddFusion: seq({seq_dim}) + feat({feat_dim}) → {output_dim}"
        )
    
    def forward(
        self,
        seq_encoding: torch.Tensor,
        feat_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        seq_proj = self.seq_proj(seq_encoding)
        feat_proj = self.feat_proj(feat_encoding)
        return seq_proj + feat_proj


class GatedFusion(nn.Module):
    """
    Gated Fusion.
    Học trọng số gate để kết hợp 2 branch một cách thông minh.
    
    Công thức:
        gate = sigmoid(W_g * [seq_encoding, feat_encoding])
        fused = gate * seq_encoding + (1 - gate) * feat_encoding
    """
    
    def __init__(
        self,
        seq_dim: int,
        feat_dim: int,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            seq_dim: Kích thước sequence encoding.
            feat_dim: Kích thước feature encoding.
            output_dim: Kích thước đầu ra.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.output_dim = output_dim
        
        # Chiếu cả 2 branch về cùng kích thước
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Gate network
        total_dim = seq_dim + feat_dim
        self.gate = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        logger.debug(
            f"GatedFusion: seq({seq_dim}) + feat({feat_dim}) → {output_dim}"
        )
    
    def forward(
        self,
        seq_encoding: torch.Tensor,
        feat_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass với gate mechanism.
        
        Args:
            seq_encoding: (batch, seq_dim)
            feat_encoding: (batch, feat_dim)
            
        Returns:
            fused: (batch, output_dim)
        """
        # Chiếu về cùng không gian
        seq_proj = self.seq_proj(seq_encoding)
        feat_proj = self.feat_proj(feat_encoding)
        
        # Tính gate
        gate_input = torch.cat([seq_encoding, feat_encoding], dim=1)
        gate_values = self.gate(gate_input)
        
        # Gated fusion
        fused = gate_values * seq_proj + (1 - gate_values) * feat_proj
        
        # Final projection
        fused = self.output_proj(fused)
        
        return fused


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    batch_size = 16
    seq_dim = 256
    feat_dim = 64
    output_dim = 256
    
    seq_enc = torch.randn(batch_size, seq_dim)
    feat_enc = torch.randn(batch_size, feat_dim)
    
    print("=== Testing Fusion Layers ===\n")
    
    # Test Concat
    concat_fusion = ConcatFusion(seq_dim, feat_dim, output_dim)
    out_concat = concat_fusion(seq_enc, feat_enc)
    print(f"ConcatFusion output shape: {out_concat.shape}")
    assert out_concat.shape == (batch_size, output_dim)
    
    # Test Add
    add_fusion = AddFusion(seq_dim, feat_dim, output_dim)
    out_add = add_fusion(seq_enc, feat_enc)
    print(f"AddFusion output shape: {out_add.shape}")
    assert out_add.shape == (batch_size, output_dim)
    
    # Test Gate
    gate_fusion = GatedFusion(seq_dim, feat_dim, output_dim)
    out_gate = gate_fusion(seq_enc, feat_enc)
    print(f"GatedFusion output shape: {out_gate.shape}")
    assert out_gate.shape == (batch_size, output_dim)
    
    # Test Factory
    factory = FusionLayer(seq_dim, feat_dim, 'gate', output_dim)
    out_factory = factory(seq_enc, feat_enc)
    print(f"FusionLayer (factory) output shape: {out_factory.shape}")
    
    print("\nAll fusion tests passed!")