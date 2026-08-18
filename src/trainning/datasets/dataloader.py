#!/usr/bin/env python3
"""
Module: dataloader.py
Mục tiêu: Factory function tạo DataLoader cho train/val/test.
Hỗ trợ dynamic batching, multi-worker, prefetch.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .collate import SequenceCollator, DynamicSequenceCollator

logger = logging.getLogger(__name__)


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    max_seq_len: Optional[int] = None,
    drop_last_train: bool = True,
    shuffle_train: bool = True,
    collator_kwargs: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoader cho train, validation và test.
    
    Args:
        train_dataset: Dataset huấn luyện.
        val_dataset: Dataset validation.
        test_dataset: Dataset test.
        batch_size: Kích thước batch.
        num_workers: Số worker cho data loading.
        pin_memory: Pin memory để tăng tốc transfer lên GPU.
        prefetch_factor: Số batch prefetch mỗi worker.
        max_seq_len: Nếu set, pad/truncate về độ dài cố định.
                     Nếu None, dynamic padding theo batch.
        drop_last_train: Drop batch cuối nếu không đủ batch_size (chỉ train).
        shuffle_train: Shuffle dữ liệu train.
        collator_kwargs: Tham số thêm cho collator.
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    collator_kwargs = collator_kwargs or {}
    
    # Tạo collator
    if max_seq_len is not None:
        from .collate import FixedLengthCollator
        collate_fn = FixedLengthCollator(
            max_seq_len=max_seq_len,
            **collator_kwargs
        )
    else:
        collate_fn = DynamicSequenceCollator(**collator_kwargs)
    
    logger.info(f"Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}")
    if max_seq_len:
        logger.info(f"Fixed max_seq_len: {max_seq_len}")
    else:
        logger.info("Dynamic sequence length (per batch)")
    
    # Train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=drop_last_train,
        collate_fn=collate_fn,
    )
    
    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    # Test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")
    logger.info(f"Test loader: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    max_seq_len: Optional[int] = None,
    drop_last: bool = False,
    collator_kwargs: Optional[Dict] = None,
) -> DataLoader:
    """
    Tạo một DataLoader đơn.
    
    Args:
        dataset: Dataset.
        batch_size: Kích thước batch.
        shuffle: Shuffle dữ liệu.
        num_workers: Số worker.
        pin_memory: Pin memory.
        prefetch_factor: Prefetch factor.
        max_seq_len: Độ dài tối đa (None = dynamic).
        drop_last: Drop batch cuối.
        collator_kwargs: Tham số collator.
        
    Returns:
        DataLoader instance.
    """
    collator_kwargs = collator_kwargs or {}
    
    if max_seq_len is not None:
        from .collate import FixedLengthCollator
        collate_fn = FixedLengthCollator(max_seq_len=max_seq_len, **collator_kwargs)
    else:
        collate_fn = DynamicSequenceCollator(**collator_kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


class DataLoaderFactory:
    """
    Factory class quản lý việc tạo DataLoader từ config.
    
    Usage:
        factory = DataLoaderFactory(config)
        train_loader, val_loader, test_loader = factory.create_all(train_ds, val_ds, test_ds)
    """
    
    def __init__(self, config):
        """
        Args:
            config: Config object từ utils.config
        """
        self.config = config
        self.collator_kwargs = {
            'pad_value': config.get('data.pad_value', 0.0),
            'return_metadata': config.get('data.return_metadata', True),
        }
    
    def create_all(self, train_ds, val_ds, test_ds):
        """Tạo cả 3 DataLoader."""
        return create_dataloaders(
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.get('prefetch_factor', 2),
            max_seq_len=self.config.data.get('max_seq_len'),
            drop_last_train=True,
            shuffle_train=True,
            collator_kwargs=self.collator_kwargs,
        )
    
    def create_single(self, dataset, shuffle=False, drop_last=False):
        """Tạo một DataLoader."""
        return create_dataloader(
            dataset=dataset,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            max_seq_len=self.config.data.get('max_seq_len'),
            drop_last=drop_last,
            collator_kwargs=self.collator_kwargs,
        )


def test_dataloader():
    """Test DataLoader với dataset giả."""
    import sys
    sys.path.insert(0, '..')
    from dataset_builder import FuelSequenceDataset
    
    # Tạo dataset giả
    samples = [
        {
            'sequence': torch.randn(10, 4).numpy(),
            'segment_feature': torch.randn(21).numpy(),
            'label_id': 0,
            'label_name': 'Driving',
            'length': 10,
            'car_id': f'Car_{i}',
            'segment_id': f'{i}',
            'start_time': '2025-01-01',
            'end_time': '2025-01-01',
            'original_segment': None,
            'virtual_segment': False,
        }
        for i in range(200)
    ]
    # Tạo lengths khác nhau
    for i, s in enumerate(samples):
        s['length'] = max(3, (i % 20) + 3)
        s['sequence'] = torch.randn(s['length'], 4).numpy()
    
    dataset = FuelSequenceDataset(samples)
    
    # Test DataLoader
    loader = create_dataloader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # 0 để test dễ
    )
    
    print(f"Number of batches: {len(loader)}")
    
    for i, batch in enumerate(loader):
        if i == 0:
            print(f"\nFirst batch:")
            print(f"  sequence shape: {batch['sequence'].shape}")
            print(f"  mask shape: {batch['mask'].shape}")
            print(f"  feature shape: {batch['feature'].shape}")
            print(f"  label shape: {batch['label'].shape}")
            print(f"  length: {batch['length']}")
            print(f"  car_id: {batch['car_id'][:3]}")
            
            # Kiểm tra mask
            for j in range(min(3, len(batch['length']))):
                real_len = batch['length'][j].item()
                mask_sum = batch['mask'][j].sum().item()
                print(f"  Sample {j}: real_len={real_len}, mask_sum={mask_sum}, match={real_len == mask_sum}")
            break
    
    print("\nDataLoader test passed!")


if __name__ == "__main__":
    test_dataloader()