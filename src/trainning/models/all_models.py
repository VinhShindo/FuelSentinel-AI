#!/usr/bin/env python3
"""
Module: all_models.py
Mục tiêu: Import và đăng ký TẤT CẢ model vào registry.
File này phải được import trước khi sử dụng ModelFactory.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base_model import BaseStateClassifier
from .factory import register_model, MODEL_REGISTRY
from .temporal_encoders.bilstm import BiLSTMEncoder
from .temporal_encoders.gru import GRUEncoder
from .temporal_encoders.bigru import BiGRUEncoder
from .temporal_encoders.cnn_encoder import CNNEncoder
from .temporal_encoders.transformer import TransformerEncoder
from .temporal_encoders.tcn import TCNEncoder
from .temporal_encoders.attention import SelfAttentionEncoder


# =============================================================================
# 1. BiLSTM Classifier
# =============================================================================
@register_model('bilstm')
class BiLSTMClassifier(BaseStateClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence_encoder = BiLSTMEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=kwargs.get('num_layers', 2),
            dropout=self.dropout_rate,
            bidirectional=True,
        )
        self.build()
    
    def encode_sequence(self, x, mask=None):
        return self.sequence_encoder(x, mask)
    
    def _get_seq_output_dim(self):
        return self.hidden_dim * 2
    
    def encode_features(self, x_feat):
        return self.feature_encoder(x_feat)


# =============================================================================
# 2. GRU Classifier
# =============================================================================
@register_model('gru')
class GRUClassifier(BaseStateClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence_encoder = GRUEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=kwargs.get('num_layers', 2),
            dropout=self.dropout_rate,
        )
        self.build()
    
    def encode_sequence(self, x, mask=None):
        return self.sequence_encoder(x, mask)
    
    def _get_seq_output_dim(self):
        return self.hidden_dim
    
    def encode_features(self, x_feat):
        return self.feature_encoder(x_feat)


# =============================================================================
# 3. BiGRU Classifier
# =============================================================================
@register_model('bigru')
class BiGRUClassifier(BaseStateClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence_encoder = BiGRUEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=kwargs.get('num_layers', 2),
            dropout=self.dropout_rate,
        )
        self.build()
    
    def encode_sequence(self, x, mask=None):
        return self.sequence_encoder(x, mask)
    
    def _get_seq_output_dim(self):
        return self.hidden_dim * 2
    
    def encode_features(self, x_feat):
        return self.feature_encoder(x_feat)


# =============================================================================
# 4. CNN + BiLSTM Classifier
# =============================================================================
@register_model('cnn_bilstm')
class CNNBiLSTMClassifier(BaseStateClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cnn_channels = kwargs.get('cnn_channels', [64, 128])
        self.cnn_encoder = CNNEncoder(
            input_dim=self.input_dim,
            channels=cnn_channels,
            kernel_size=kwargs.get('cnn_kernel_size', 3),
            dropout=self.dropout_rate,
        )
        self.lstm_encoder = BiLSTMEncoder(
            input_dim=cnn_channels[-1],
            hidden_dim=self.hidden_dim,
            num_layers=kwargs.get('num_layers', 2),
            dropout=self.dropout_rate,
            bidirectional=True,
        )
        self.build()
    
    def encode_sequence(self, x, mask=None):
        x = self.cnn_encoder(x, mask)
        return self.lstm_encoder(x, mask)
    
    def _get_seq_output_dim(self):
        return self.hidden_dim * 2
    
    def encode_features(self, x_feat):
        return self.feature_encoder(x_feat)


# =============================================================================
# 5. CNN + GRU Classifier
# =============================================================================
@register_model('cnn_gru')
class CNNGRUClassifier(BaseStateClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cnn_channels = kwargs.get('cnn_channels', [64, 128])
        self.cnn_encoder = CNNEncoder(
            input_dim=self.input_dim,
            channels=cnn_channels,
            kernel_size=kwargs.get('cnn_kernel_size', 3),
            dropout=self.dropout_rate,
        )
        self.gru_encoder = GRUEncoder(
            input_dim=cnn_channels[-1],
            hidden_dim=self.hidden_dim,
            num_layers=kwargs.get('num_layers', 2),
            dropout=self.dropout_rate,
        )
        self.build()
    
    def encode_sequence(self, x, mask=None):
        x = self.cnn_encoder(x, mask)
        return self.gru_encoder(x, mask)
    
    def _get_seq_output_dim(self):
        return self.hidden_dim
    
    def encode_features(self, x_feat):
        return self.feature_encoder(x_feat)


# =============================================================================
# 6. Transformer Classifier
# =============================================================================
@register_model('transformer')
class TransformerClassifier(BaseStateClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence_encoder = TransformerEncoder(
            input_dim=self.input_dim,
            d_model=self.hidden_dim,
            nhead=kwargs.get('nhead', 8),
            num_layers=kwargs.get('num_transformer_layers', 3),
            dim_feedforward=kwargs.get('dim_feedforward', 512),
            dropout=self.dropout_rate,
            pooling='mean',
        )
        self.build()
    
    def encode_sequence(self, x, mask=None):
        return self.sequence_encoder(x, mask)
    
    def _get_seq_output_dim(self):
        return self.hidden_dim
    
    def encode_features(self, x_feat):
        return self.feature_encoder(x_feat)


# =============================================================================
# 7. TCN Classifier
# =============================================================================
@register_model('tcn')
class TCNClassifier(BaseStateClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tcn_channels = kwargs.get('tcn_channels', [64, 128, 256])
        self.sequence_encoder = TCNEncoder(
            input_dim=self.input_dim,
            channels=tcn_channels,
            kernel_size=kwargs.get('kernel_size', 3),
            dropout=self.dropout_rate,
        )
        self.build()
    
    def encode_sequence(self, x, mask=None):
        return self.sequence_encoder(x, mask)
    
    def _get_seq_output_dim(self):
        return self.sequence_encoder.output_dim
    
    def encode_features(self, x_feat):
        return self.feature_encoder(x_feat)


# =============================================================================
# 8. LSTM + Attention Classifier
# =============================================================================
@register_model('lstm_attention')
class LSTMAttentionClassifier(BaseStateClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstm_encoder = BiLSTMEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=kwargs.get('num_layers', 2),
            dropout=self.dropout_rate,
            bidirectional=True,
        )
        self.attention = SelfAttentionEncoder(
            input_dim=self.hidden_dim * 2,
            d_model=self.hidden_dim,
            nhead=kwargs.get('nhead', 8),
            dropout=self.dropout_rate,
            pooling='mean',
        )
        self.build()
    
    def encode_sequence(self, x, mask=None):
        # LSTM lấy toàn bộ output
        if mask is not None:
            lengths = mask.sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm_encoder.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm_encoder.lstm(x)
        
        # Attention trên toàn bộ sequence
        return self.attention(lstm_out, mask)
    
    def _get_seq_output_dim(self):
        return self.hidden_dim
    
    def encode_features(self, x_feat):
        return self.feature_encoder(x_feat)


# =============================================================================
# 9. GRU + Attention Classifier
# =============================================================================
@register_model('gru_attention')
class GRUAttentionClassifier(BaseStateClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gru_encoder = BiGRUEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=kwargs.get('num_layers', 2),
            dropout=self.dropout_rate,
        )
        self.attention = SelfAttentionEncoder(
            input_dim=self.hidden_dim * 2,
            d_model=self.hidden_dim,
            nhead=kwargs.get('nhead', 8),
            dropout=self.dropout_rate,
            pooling='mean',
        )
        self.build()
    
    def encode_sequence(self, x, mask=None):
        # GRU lấy toàn bộ output
        if mask is not None:
            lengths = mask.sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            gru_out, _ = self.gru_encoder.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        else:
            gru_out, _ = self.gru_encoder.gru(x)
        
        # Attention trên toàn bộ sequence
        return self.attention(gru_out, mask)
    
    def _get_seq_output_dim(self):
        return self.hidden_dim
    
    def encode_features(self, x_feat):
        return self.feature_encoder(x_feat)