#!/usr/bin/env python3
"""
Module: base_model.py
Mục tiêu: Định nghĩa abstract base class cho tất cả model.
Mọi model đều phải kế thừa class này và implement các method abstract.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseStateClassifier(nn.Module, ABC):
    """
    Abstract Base Class cho mọi model phân loại trạng thái.
    
    Kiến trúc chung:
        1. Sequence Encoder: xử lý chuỗi (T, 4) → seq_encoding (batch, hidden_dim)
        2. Feature Encoder: xử lý feature vector (21,) → feat_encoding (batch, feat_dim)
        3. Fusion: kết hợp seq_encoding và feat_encoding → fused (batch, fusion_dim)
        4. Classifier: dự đoán 4 class → logits (batch, 4)
    
    Subclass phải implement:
        - encode_sequence()
        - encode_features()
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        feature_dim: int = 21,
        num_classes: int = 4,
        dropout: float = 0.3,
        fusion_type: str = 'concat',
        fusion_dim: int = 256,
        **kwargs
    ):
        """
        Args:
            input_dim: Số kênh đầu vào của chuỗi thời gian (mặc định 4: fuel, speed, distance_step, bearing).
            hidden_dim: Kích thước hidden của sequence encoder.
            feature_dim: Số feature thống kê (mặc định 21).
            num_classes: Số lớp đầu ra (4).
            dropout: Tỷ lệ dropout.
            fusion_type: Loại fusion ('concat', 'add', 'gate').
            fusion_dim: Kích thước đầu ra sau fusion.
            **kwargs: Tham số bổ sung cho subclass.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim
        
        # Sẽ được khởi tạo trong subclass hoặc build()
        self.sequence_encoder = None
        self.feature_encoder = None
        self.fusion_layer = None
        self.classifier = None
        
        # Lưu config để get_model_info()
        self._model_config = kwargs
        
    @abstractmethod
    def encode_sequence(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode chuỗi thời gian thành vector đặc trưng.
        
        Args:
            x: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) - True cho vị trí thực, False cho padding
            
        Returns:
            seq_encoding: (batch_size, hidden_dim)
        """
        pass
    
    @abstractmethod
    def encode_features(self, x_feat: torch.Tensor) -> torch.Tensor:
        """
        Encode vector feature thống kê.
        
        Args:
            x_feat: (batch_size, feature_dim)
            
        Returns:
            feat_encoding: (batch_size, feat_hidden_dim)
        """
        pass
    
    def build(self) -> None:
        """
        Xây dựng các layer chưa được khởi tạo.
        Gọi method này sau khi __init__ để hoàn thiện model.
        Subclass có thể override để thêm logic khởi tạo.
        """
        from .fusion import FusionLayer
        from .classifier import ClassifierHead
        
        # Feature encoder mặc định (MLP đơn giản)
        if self.feature_encoder is None:
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
        
        # Fusion layer
        if self.fusion_layer is None:
            seq_output_dim = self._get_seq_output_dim()
            feat_output_dim = 64  # Mặc định từ feature encoder
            
            self.fusion_layer = FusionLayer(
                seq_dim=seq_output_dim,
                feat_dim=feat_output_dim,
                fusion_type=self.fusion_type,
                output_dim=self.fusion_dim,
            )
        
        # Classifier
        if self.classifier is None:
            self.classifier = ClassifierHead(
                input_dim=self.fusion_dim,
                num_classes=self.num_classes,
                dropout=self.dropout_rate,
            )
        
        logger.info(f"Model built: {self.__class__.__name__}")
        logger.info(f"  Sequence encoder output dim: {self._get_seq_output_dim()}")
        logger.info(f"  Fusion type: {self.fusion_type}")
        logger.info(f"  Fusion output dim: {self.fusion_dim}")
        logger.info(f"  Number of classes: {self.num_classes}")
    
    def _get_seq_output_dim(self) -> int:
        """
        Tính kích thước đầu ra của sequence encoder.
        Subclass override nếu output khác hidden_dim (ví dụ bidirectional LSTM).
        
        Returns:
            Output dimension.
        """
        return self.hidden_dim
    
    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sequence: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) - True cho vị trí thực
            feature: (batch_size, feature_dim)
            
        Returns:
            logits: (batch_size, num_classes)
        """
        # Encode sequence
        seq_encoding = self.encode_sequence(sequence, mask)
        
        # Encode feature
        if feature is not None:
            feat_encoding = self.encode_features(feature)
        else:
            # Nếu không có feature, tạo zero vector
            feat_encoding = torch.zeros(
                sequence.shape[0], 64,
                device=sequence.device
            )
        
        # Fusion
        fused = self.fusion_layer(seq_encoding, feat_encoding)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
    
    def get_model_info(self) -> Dict:
        """
        Lấy thông tin model: số tham số, kích thước.
        
        Returns:
            Dict chứa thông tin model.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Tính model size (MB)
        param_size = total_params * 4 / (1024 * 1024)  # 4 bytes/param (float32)
        
        info = {
            'model_name': self.__class__.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': round(param_size, 2),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'fusion_type': self.fusion_type,
            'fusion_dim': self.fusion_dim,
        }
        
        return info
    
    def count_parameters(self) -> int:
        """Đếm tổng số tham số có thể huấn luyện."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_flops(
        self,
        seq_len: int = 100,
        batch_size: int = 1
    ) -> Optional[float]:
        """
        Ước lượng FLOPs của model.
        
        Args:
            seq_len: Độ dài chuỗi mẫu.
            batch_size: Batch size.
            
        Returns:
            Số FLOPs hoặc None nếu không tính được.
        """
        try:
            from thop import profile
            
            device = next(self.parameters()).device
            x_seq = torch.randn(batch_size, seq_len, self.input_dim).to(device)
            x_feat = torch.randn(batch_size, self.feature_dim).to(device)
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
            
            flops, params = profile(
                self,
                inputs=(x_seq, mask, x_feat),
                verbose=False
            )
            return flops
        except ImportError:
            logger.warning("thop not installed. Cannot estimate FLOPs.")
            return None
        except Exception as e:
            logger.warning(f"FLOPs estimation failed: {e}")
            return None


# =============================================================================
# Test
# =============================================================================

class DummyModel(BaseStateClassifier):
    """Model mẫu để test base class."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.build()
    
    def encode_sequence(self, x, mask=None):
        lstm_out, (hn, cn) = self.lstm(x)
        # Lấy hidden state cuối cùng từ cả 2 hướng
        seq_enc = torch.cat((hn[-2], hn[-1]), dim=1)
        return seq_enc
    
    def _get_seq_output_dim(self):
        return self.hidden_dim * 2  # bidirectional
    
    def encode_features(self, x_feat):
        return self.feature_encoder(x_feat)


if __name__ == "__main__":
    # Test base model
    model = DummyModel(
        input_dim=4,
        hidden_dim=128,
        feature_dim=21,
        num_classes=4,
        dropout=0.3,
        fusion_type='gate',
    )
    
    print("\n=== Model Info ===")
    info = model.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    # Test forward
    batch_size = 16
    seq_len = 50
    
    x_seq = torch.randn(batch_size, seq_len, 4)
    x_feat = torch.randn(batch_size, 21)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -10:] = False  # 10 vị trí cuối là padding
    
    logits = model(x_seq, mask, x_feat)
    print(f"\nInput sequence shape: {x_seq.shape}")
    print(f"Input feature shape: {x_feat.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits: {logits[0]}")
    
    # Test với sequence có độ dài khác nhau
    x_seq_short = torch.randn(batch_size, 20, 4)
    mask_short = torch.ones(batch_size, 20, dtype=torch.bool)
    
    logits_short = model(x_seq_short, mask_short, x_feat)
    print(f"\nShort sequence (len=20) output shape: {logits_short.shape}")