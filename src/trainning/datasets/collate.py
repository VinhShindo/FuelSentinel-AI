#!/usr/bin/env python3
"""
Module: collate.py
Mục tiêu: Custom collate function cho DataLoader.
Pad sequence về cùng độ dài trong batch, tạo attention mask.
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class SequenceCollator:
    """
    Collate function cho batch sequence có độ dài thay đổi.
    
    Padding strategy:
    - Sequence: pad về max length trong batch
    - Feature: stack bình thường (kích thước cố định)
    - Label: stack thành tensor
    - Tạo attention mask cho các vị trí được pad
    
    Output format:
    {
        'sequence': (batch_size, max_seq_len, 4),
        'mask': (batch_size, max_seq_len) - True cho vị trí thực, False cho padding
        'feature': (batch_size, 21),
        'label': (batch_size,),
        'length': (batch_size,) - độ dài gốc
        'car_id': list of str,
        'segment_id': list,
        'metadata': dict chứa thông tin bổ sung
    }
    """
    
    def __init__(
        self,
        max_seq_len: Optional[int] = None,
        pad_value: float = 0.0,
        return_metadata: bool = True,
    ):
        """
        Args:
            max_seq_len: Nếu set, pad hoặc truncate về độ dài cố định.
                         Nếu None, pad theo max trong batch.
            pad_value: Giá trị dùng để pad.
            return_metadata: Trả về thêm metadata (car_id, segment_id...)
        """
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.return_metadata = return_metadata
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate một list các sample thành batch.
        
        Args:
            batch: List các dict sample từ Dataset.__getitem__
            
        Returns:
            Dict batch đã được xử lý.
        """
        if not batch:
            return {}
        
        # 1. Trích xuất các thành phần
        sequences = [torch.from_numpy(sample['sequence']).float() for sample in batch]
        features = [torch.from_numpy(sample['segment_feature']).float() for sample in batch]
        labels = [sample['label_id'] for sample in batch]
        lengths = [sample['length'] for sample in batch]
        
        # 2. Lấy max length trong batch
        batch_max_len = max(lengths)
        
        # Nếu có max_seq_len, giới hạn
        if self.max_seq_len is not None:
            target_len = min(batch_max_len, self.max_seq_len)
        else:
            target_len = batch_max_len
        
        # 3. Pad sequence
        padded_seqs, masks = self._pad_sequences(sequences, target_len)
        
        # 4. Stack features
        stacked_features = torch.stack(features, dim=0)  # (batch, 21)
        
        # 5. Stack labels
        stacked_labels = torch.tensor(labels, dtype=torch.long)  # (batch,)
        
        # 6. Truncate lengths
        truncated_lengths = torch.tensor(
            [min(l, target_len) for l in lengths],
            dtype=torch.long
        )
        
        # 7. Tạo output dict
        output = {
            'sequence': padded_seqs,        # (batch, target_len, 4)
            'mask': masks,                  # (batch, target_len)
            'feature': stacked_features,    # (batch, 21)
            'label': stacked_labels,        # (batch,)
            'length': truncated_lengths,    # (batch,)
        }
        
        # 8. Metadata (optional)
        if self.return_metadata:
            output['car_id'] = [sample.get('car_id', '') for sample in batch]
            output['segment_id'] = [sample.get('segment_id', '') for sample in batch]
            output['label_name'] = [sample.get('label_name', '') for sample in batch]
            
            # Thêm start_time, end_time nếu có
            if 'start_time' in batch[0]:
                output['start_time'] = [sample.get('start_time', '') for sample in batch]
            if 'end_time' in batch[0]:
                output['end_time'] = [sample.get('end_time', '') for sample in batch]
        
        return output
    
    def _pad_sequences(
        self, 
        sequences: List[torch.Tensor], 
        target_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad list các sequence về cùng độ dài.
        
        Args:
            sequences: List các tensor (T_i, 4)
            target_len: Độ dài mục tiêu
            
        Returns:
            padded: (batch, target_len, 4)
            mask: (batch, target_len) - True cho vị trí thực, False cho padding
        """
        batch_size = len(sequences)
        feat_dim = sequences[0].shape[1]  # 4
        
        # Tạo tensor padding
        padded = torch.full(
            (batch_size, target_len, feat_dim),
            self.pad_value,
            dtype=torch.float32
        )
        
        # Tạo mask (True = real, False = pad)
        mask = torch.zeros(batch_size, target_len, dtype=torch.bool)
        
        for i, seq in enumerate(sequences):
            seq_len = min(seq.shape[0], target_len)
            padded[i, :seq_len, :] = seq[:seq_len, :]
            mask[i, :seq_len] = True
        
        return padded, mask


class DynamicSequenceCollator(SequenceCollator):
    """
    Collator variant: pad theo max trong batch (dynamic), không truncate.
    """
    
    def __init__(self, pad_value: float = 0.0, return_metadata: bool = True):
        super().__init__(
            max_seq_len=None,  # Không giới hạn
            pad_value=pad_value,
            return_metadata=return_metadata,
        )


class FixedLengthCollator(SequenceCollator):
    """
    Collator variant: pad hoặc truncate về độ dài cố định.
    """
    
    def __init__(
        self, 
        max_seq_len: int, 
        pad_value: float = 0.0, 
        return_metadata: bool = True
    ):
        super().__init__(
            max_seq_len=max_seq_len,
            pad_value=pad_value,
            return_metadata=return_metadata,
        )


def test_collator():
    """Test function cho collator."""
    # Tạo sample giả
    samples = [
        {
            'sequence': np.random.randn(10, 4).astype(np.float32),
            'segment_feature': np.random.randn(21).astype(np.float32),
            'label_id': 0,
            'length': 10,
            'car_id': 'Car 1',
            'segment_id': '1',
            'label_name': 'Driving',
        },
        {
            'sequence': np.random.randn(5, 4).astype(np.float32),
            'segment_feature': np.random.randn(21).astype(np.float32),
            'label_id': 1,
            'length': 5,
            'car_id': 'Car 2',
            'segment_id': '2',
            'label_name': 'Idle',
        },
        {
            'sequence': np.random.randn(15, 4).astype(np.float32),
            'segment_feature': np.random.randn(21).astype(np.float32),
            'label_id': 2,
            'length': 15,
            'car_id': 'Car 3',
            'segment_id': '3',
            'label_name': 'Refuel',
        },
    ]
    
    # Test dynamic collator
    collator = DynamicSequenceCollator()
    batch = collator(samples)
    
    print("Dynamic Collator Output:")
    print(f"  sequence shape: {batch['sequence'].shape}")  # (3, 15, 4)
    print(f"  mask shape: {batch['mask'].shape}")          # (3, 15)
    print(f"  feature shape: {batch['feature'].shape}")    # (3, 21)
    print(f"  label shape: {batch['label'].shape}")        # (3,)
    print(f"  length: {batch['length']}")                  # [10, 5, 15]
    print(f"  mask[0]: {batch['mask'][0]}")                # True x10, False x5
    print(f"  mask[1]: {batch['mask'][1]}")                # True x5, False x10
    
    # Test fixed length collator
    collator_fixed = FixedLengthCollator(max_seq_len=8)
    batch_fixed = collator_fixed(samples)
    
    print("\nFixed Length Collator Output (max_len=8):")
    print(f"  sequence shape: {batch_fixed['sequence'].shape}")  # (3, 8, 4)
    print(f"  mask shape: {batch_fixed['mask'].shape}")          # (3, 8)
    print(f"  length: {batch_fixed['length']}")                  # [8, 5, 8]


if __name__ == "__main__":
    test_collator()