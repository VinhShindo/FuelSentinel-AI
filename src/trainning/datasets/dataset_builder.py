#!/usr/bin/env python3
"""
Module: dataset_builder.py (Final Version – Multi‑source data loader)
Mục tiêu: Đọc CSV train/val/test đã chia, xây dựng PyTorch Dataset cho mô hình
          nhận diện trạng thái (Driving, Idle, Refuel, Theft) từ dữ liệu cảm biến.

Hỗ trợ nạp đồng thời dữ liệu từ nhiều nguồn:
  1. File split gốc (data/splits/*.csv)
  2. Dữ liệu synthetic cũ (data/sample/car1_synthetic_*.csv)
  3. Dữ liệu V2 (data/splits_v2/*.csv)
Mỗi split (train/val/test) có thể nhận một danh sách các file bổ sung.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Cấu hình cố định
# =============================================================================

SEQUENCE_COLS_RAW = ["fuel", "speed", "latitude", "longitude"]
SEQUENCE_COLS_PROCESSED = ["fuel", "speed", "distance_step", "bearing"]

FEATURE_COLS = [
    "duration_min", "sample_count", "fuel_start", "fuel_end",
    "fuel_change", "fuel_mean", "fuel_std", "fuel_slope",
    "trend_r2", "trend_rmse", "max_drop", "max_rise",
    "drop_count", "rise_count", "fuel_range", "fuel_mad",
    "oscillation_count", "total_distance", "speed_mean",
    "speed_max", "speed_zero_ratio"
]

LABEL_MAP = {
    "Driving": 0,
    "Idle": 1,
    "Refuel": 2,
    "Theft": 3
}

IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
MIN_SEQUENCE_LENGTH = 2

# =============================================================================
# GPS helpers
# =============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

def convert_gps_to_relative(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    lats = group["latitude"].values
    lons = group["longitude"].values
    distances = [0.0]
    bearings = [0.0]
    for i in range(1, len(lats)):
        d = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
        b = calculate_bearing(lats[i-1], lons[i-1], lats[i], lons[i])
        distances.append(d)
        bearings.append(b)
    group["distance_step"] = distances
    group["bearing"] = bearings
    return group

# =============================================================================
# Dataset
# =============================================================================
class FuelSequenceDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

# =============================================================================
# Builder
# =============================================================================
class FuelDatasetBuilder:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.min_sequence_length = self.config.get("min_sequence_length", MIN_SEQUENCE_LENGTH)
        self.strict_feature_check = self.config.get("strict_feature_check", False)
        self.fillna_strategy = self.config.get("fillna_strategy", "median")
        self.verbose_nan_log = self.config.get("verbose_nan_log", True)
        self.feature_fill_values = None
        self.removed_segments = []
        self.total_nan_filled = 0
        self.nan_log = defaultdict(list)
        self.nan_summary = defaultdict(int)

    def build(self, csv_path: str, is_train: bool = False,
              extra_csv: Optional[Union[str, List[str]]] = None) -> FuelSequenceDataset:
        """
        Args:
            csv_path: File CSV chính (phải có).
            is_train: True nếu là tập train.
            extra_csv: Một file bổ sung hoặc danh sách các file bổ sung (có thể None).
        """
        logger.info(f"{'='*60}")
        logger.info(f"Bắt đầu xây dựng dataset từ: {csv_path}")
        logger.info(f"  + Chế độ: {'TRAIN (tính thống kê NaN)' if is_train else 'VAL/TEST (dùng thống kê từ train)'}")
        logger.info(f"  + Chiến lược điền NaN: {self.fillna_strategy}")
        logger.info(f"  + Độ dài tối thiểu segment: {self.min_sequence_length} điểm")
        if extra_csv:
            if isinstance(extra_csv, list):
                logger.info(f"  + File bổ sung: {extra_csv}")
            else:
                logger.info(f"  + File bổ sung: {extra_csv}")
        logger.info(f"{'='*60}")

        # Đọc file chính
        df = self._load_csv(csv_path)

        # Nạp các file bổ sung (nếu có) và ghép vào df
        if extra_csv:
            # Đảm bảo extra_csv là list để xử lý chung
            if isinstance(extra_csv, str):
                extra_files = [extra_csv]
            else:
                extra_files = extra_csv
            for file in extra_files:
                if file is None:
                    continue
                extra_file = Path(file)
                if extra_file.is_file():
                    logger.info(f"Tải dữ liệu bổ sung từ: {extra_file}")
                    df_extra = pd.read_csv(extra_file, dtype={"segment_id": str}, low_memory=False)
                    source_prefix = extra_file.stem  # Lấy tên file không đuôi
                    df_extra["segment_id"] = source_prefix + "_" + df_extra["segment_id"].astype(str)
                    # Giữ các cột chung
                    common = [c for c in df.columns if c in df_extra.columns]
                    df_extra = df_extra[common]
                    if "timestamp" in df_extra.columns:
                        df_extra["timestamp"] = pd.to_datetime(df_extra["timestamp"])
                    df = pd.concat([df, df_extra], ignore_index=True)
                    logger.info(f"Đã thêm {len(df_extra)} mẫu từ file bổ sung.")
                else:
                    logger.warning(f"Không tìm thấy file bổ sung: {extra_file}")

        df = self._preprocess(df)
        df = self._process_gps(df)
        df = self._remove_invalid_segments(df)
        samples = self._extract_samples(df, is_train=is_train)
        self._quality_checks(samples)
        self._generate_report(samples)
        return FuelSequenceDataset(samples)

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        logger.info(f"Đọc file: {csv_path}")
        df = pd.read_csv(csv_path, dtype={"segment_id": str}, low_memory=False)
        required_cols = ["car_id", "segment_id", "final_label"] + SEQUENCE_COLS_RAW + FEATURE_COLS
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc trong CSV: {missing}")
        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        nan_mask = df["final_label"].isna()
        if nan_mask.any():
            logger.warning(f"Phát hiện {nan_mask.sum()} dòng NaN trong final_label, loại bỏ.")
            df = df[~nan_mask].copy()
        return df

    def _process_gps(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Chuyển đổi GPS: lat/lon → distance_step + bearing...")
        processed_groups = []
        for (car_id, seg_id), group in df.groupby(["car_id", "segment_id"], sort=False):
            if "timestamp" in group.columns:
                group = group.sort_values("timestamp")
            group = convert_gps_to_relative(group)
            processed_groups.append(group)
        df = pd.concat(processed_groups, ignore_index=True)
        logger.info("Hoàn tất chuyển đổi GPS.")
        return df

    def _remove_invalid_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        segment_lengths = df.groupby(["car_id", "segment_id"], sort=False).size()
        self.removed_segments = []
        valid_segments = []
        removed_count = 0
        logger.info("Phân tích chất lượng segment...")
        for (car, seg), length in segment_lengths.items():
            if length < self.min_sequence_length:
                label = df[(df["car_id"] == car) & (df["segment_id"] == seg)]["final_label"].iloc[0]
                self.removed_segments.append({
                    "car_id": car, "segment_id": seg, "length": length,
                    "label": label, "reason": f"Quá ngắn ({length} điểm < {self.min_sequence_length})"
                })
                removed_count += 1
            else:
                valid_segments.append((car, seg))
        if removed_count > 0:
            logger.warning(f"Phát hiện {removed_count} segment không hợp lệ (1 điểm):")
            removed_by_label = defaultdict(list)
            for seg_info in self.removed_segments:
                removed_by_label[seg_info["label"]].append(seg_info)
            for label, segs in removed_by_label.items():
                logger.warning(f"  → {label}: {len(segs)} segment bị loại")
                if self.verbose_nan_log:
                    for seg in segs[:5]:
                        logger.warning(f"      • {seg['car_id']}-{seg['segment_id']}: {seg['length']} điểm")
                    if len(segs) > 5:
                        logger.warning(f"      ... và {len(segs)-5} segment khác")
            df = df.set_index(["car_id", "segment_id"]).loc[valid_segments].reset_index()
            logger.info(f"✓ Còn lại {len(valid_segments)} segment hợp lệ (≥ {self.min_sequence_length} điểm).")
        else:
            logger.info("✓ Tất cả segment đều hợp lệ (≥ 2 điểm).")
        return df

    def _extract_samples(self, df: pd.DataFrame, is_train: bool = False) -> List[Dict]:
        samples = []
        all_features = [] if is_train else None
        self.nan_log.clear()
        self.nan_summary.clear()
        self.total_nan_filled = 0

        for (car_id, seg_id), group in df.groupby(["car_id", "segment_id"], sort=False):
            seg_key = f"{car_id}-{seg_id}"
            unique_labels = group["final_label"].unique()
            if len(unique_labels) != 1:
                raise ValueError(f"Segment {seg_key} chứa nhiều label: {unique_labels}")
            label_str = unique_labels[0]
            if label_str not in LABEL_MAP:
                raise ValueError(f"Label không hợp lệ: {label_str}")

            seq = group[SEQUENCE_COLS_PROCESSED].values.astype(np.float32)
            T = seq.shape[0]

            seq_nan_fuel_speed = np.isnan(seq[:, :2])
            if seq_nan_fuel_speed.any():
                nan_positions = np.where(seq_nan_fuel_speed)
                for row, col in zip(nan_positions[0], nan_positions[1]):
                    col_name = SEQUENCE_COLS_PROCESSED[col]
                    logger.error(f"  ✗ NaN trong sequence: {seg_key} | dòng={row} | cột={col_name}")
                raise ValueError(f"Segment {seg_key}: NaN trong fuel/speed.")

            seq_nan_gps = np.isnan(seq[:, 2:])
            if seq_nan_gps.any():
                nan_count = seq_nan_gps.sum()
                logger.warning(f"  ⚠ Segment {seg_key}: {nan_count} NaN trong distance_step/bearing → điền 0")
                seq = np.nan_to_num(seq, nan=0.0)

            feat_row = group[FEATURE_COLS].iloc[0].values.astype(np.float32)
            feat_nan_mask = np.isnan(feat_row)
            has_nan = feat_nan_mask.any()

            if is_train:
                all_features.append(feat_row)
                if has_nan:
                    for col_idx in np.where(feat_nan_mask)[0]:
                        col_name = FEATURE_COLS[col_idx]
                        self.nan_summary[col_name] += 1
                        self.total_nan_filled += 1
            else:
                if has_nan and self.verbose_nan_log:
                    for col_idx in np.where(feat_nan_mask)[0]:
                        col_name = FEATURE_COLS[col_idx]
                        fill_val = self.feature_fill_values[col_idx] if self.feature_fill_values is not None else "??"
                        self.nan_log[seg_key].append((col_name, fill_val))
                        self.nan_summary[col_name] += 1
                        self.total_nan_filled += 1
                feat_row = self._fill_feature_nan(feat_row)

            meta = {
                "car_id": car_id,
                "segment_id": seg_id,
                "original_segment": None,
                "virtual_segment": False,
            }
            if "_part" in str(seg_id):
                meta["virtual_segment"] = True
                meta["original_segment"] = str(seg_id).split("_part")[0]
            if "timestamp" in group.columns:
                meta["start_time"] = str(group["timestamp"].min())
                meta["end_time"] = str(group["timestamp"].max())
            else:
                meta["start_time"] = None
                meta["end_time"] = None

            sample = {
                "sequence": seq,
                "segment_feature": feat_row,
                "label_id": LABEL_MAP[label_str],
                "label_name": label_str,
                "length": T,
                **meta
            }
            samples.append(sample)

        if is_train and all_features:
            all_features = np.array(all_features)
            nan_mask_train = np.isnan(all_features)
            if nan_mask_train.any():
                nan_counts_per_col = np.sum(nan_mask_train, axis=0)
                logger.info(f"Phát hiện NaN trong tập train ({nan_mask_train.sum()} giá trị):")
                for col_idx, count in enumerate(nan_counts_per_col):
                    if count > 0:
                        col_name = FEATURE_COLS[col_idx]
                        affected_segments = []
                        for i, sample_feat in enumerate(all_features):
                            if np.isnan(sample_feat[col_idx]):
                                if i < len(samples):
                                    affected_segments.append(f"{samples[i]['car_id']}-{samples[i]['segment_id']}")
                        logger.warning(f"  → {col_name}: {count}/{len(all_features)} segment bị NaN")
                        if self.verbose_nan_log and len(affected_segments) <= 10:
                            logger.warning(f"     Các segment: {', '.join(affected_segments)}")
                        elif self.verbose_nan_log:
                            logger.warning(f"     Các segment: {', '.join(affected_segments[:10])} ... và {len(affected_segments)-10} segment khác")
                if self.fillna_strategy == "median":
                    self.feature_fill_values = np.nanmedian(all_features, axis=0)
                else:
                    self.feature_fill_values = np.nanmean(all_features, axis=0)
                logger.info(f"Đã tính feature fill values ({self.fillna_strategy}) từ tập train:")
                for col_idx, col_name in enumerate(FEATURE_COLS):
                    if nan_counts_per_col[col_idx] > 0:
                        logger.info(f"  + {col_name}: fill_value = {self.feature_fill_values[col_idx]:.6f}")
                train_nan_filled = 0
                for sample in samples:
                    old_feat = sample["segment_feature"].copy()
                    sample["segment_feature"] = self._fill_feature_nan(sample["segment_feature"])
                    if not np.array_equal(old_feat, sample["segment_feature"]):
                        train_nan_filled += np.sum(np.isnan(old_feat))
                logger.info(f"✓ Đã điền {train_nan_filled} giá trị NaN trong tập train.")
            else:
                if self.fillna_strategy == "median":
                    self.feature_fill_values = np.nanmedian(all_features, axis=0)
                else:
                    self.feature_fill_values = np.nanmean(all_features, axis=0)
                logger.info("✓ Không phát hiện NaN trong tập train.")

        if not is_train and self.total_nan_filled > 0:
            self._print_nan_summary()

        return samples

    def _fill_feature_nan(self, feat: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(feat)
        if nan_mask.any():
            if self.feature_fill_values is None:
                raise RuntimeError("feature_fill_values chưa được khởi tạo. Hãy build tập train trước với is_train=True.")
            feat = feat.copy()
            feat[nan_mask] = self.feature_fill_values[nan_mask]
        return feat

    def _print_nan_summary(self) -> None:
        if self.total_nan_filled == 0:
            return
        logger.warning(f"Tổng số giá trị NaN đã điền: {self.total_nan_filled}")
        logger.info("Phân bố NaN theo cột feature:")
        for col_name in FEATURE_COLS:
            count = self.nan_summary.get(col_name, 0)
            if count > 0:
                fill_val = self.feature_fill_values[FEATURE_COLS.index(col_name)]
                logger.info(f"  + {col_name}: {count} NaN → điền bằng {fill_val:.6f}")
        if self.verbose_nan_log:
            logger.info("Chi tiết các segment có NaN được điền:")
            for seg_key, nan_list in sorted(self.nan_log.items()):
                col_names = [item[0] for item in nan_list]
                logger.info(f"  → {seg_key}: {len(nan_list)} NaN ở các cột {col_names}")

    def _quality_checks(self, samples: List[Dict]) -> None:
        logger.info("Thực hiện kiểm tra chất lượng tổng thể...")
        nan_in_feature = 0
        nan_in_sequence = 0
        for s in samples:
            assert s["length"] >= self.min_sequence_length, \
                f"Segment {s['car_id']}-{s['segment_id']} có {s['length']} điểm < {self.min_sequence_length}"
            if np.isnan(s["segment_feature"]).any():
                nan_in_feature += 1
                logger.error(f"  ✗ Sample {s['car_id']}-{s['segment_id']} vẫn chứa NaN trong feature!")
            if np.isnan(s["sequence"][:, :2]).any():
                nan_in_sequence += 1
                logger.error(f"  ✗ Sample {s['car_id']}-{s['segment_id']} có NaN trong fuel/speed!")
        if nan_in_feature > 0:
            raise RuntimeError(f"Còn {nan_in_feature} sample chứa NaN trong feature!")
        if nan_in_sequence > 0:
            raise RuntimeError(f"Còn {nan_in_sequence} sample chứa NaN trong fuel/speed!")
        logger.info("✓ Tất cả kiểm tra đạt yêu cầu.")

    def _generate_report(self, samples: List[Dict]) -> None:
        logger.info("=" * 60)
        logger.info("DATASET REPORT")
        logger.info("=" * 60)
        num_samples = len(samples)
        cars = set(s["car_id"] for s in samples)
        label_counts = {lbl: 0 for lbl in LABEL_MAP.values()}
        for s in samples:
            label_counts[s["label_id"]] += 1
        lengths = [s["length"] for s in samples]
        min_len = np.min(lengths) if lengths else 0
        max_len = np.max(lengths) if lengths else 0
        mean_len = np.mean(lengths) if lengths else 0
        median_len = np.median(lengths) if lengths else 0
        p95_len = np.percentile(lengths, 95) if lengths else 0
        logger.info(f"Samples              : {num_samples}")
        logger.info(f"Cars                 : {len(cars)} ({sorted(cars)})")
        removed_count = len(self.removed_segments)
        if removed_count > 0:
            logger.info(f"Loại bỏ (1 điểm)     : {removed_count}")
            removed_labels = defaultdict(int)
            for seg in self.removed_segments:
                removed_labels[seg["label"]] += 1
            for label, count in removed_labels.items():
                logger.info(f"  - {label}: {count}")
        logger.info("Labels:")
        for lbl_str, idx in LABEL_MAP.items():
            logger.info(f"  {lbl_str:10s} (id={idx}): {label_counts[idx]}")
        logger.info("Sequence length:")
        logger.info(f"  min        : {min_len}")
        logger.info(f"  max        : {max_len}")
        logger.info(f"  mean       : {mean_len:.1f}")
        logger.info(f"  median     : {median_len}")
        logger.info(f"  95th perc  : {p95_len}")
        virtual_count = sum(1 for s in samples if s.get("virtual_segment"))
        logger.info(f"Virtual segments     : {virtual_count}")
        if self.total_nan_filled > 0:
            logger.info(f"NaN đã điền (tổng)   : {self.total_nan_filled}")
        car_sample_counts = {car: 0 for car in cars}
        for s in samples:
            car_sample_counts[s["car_id"]] += 1
        logger.info("Samples per car:")
        for car, cnt in sorted(car_sample_counts.items()):
            logger.info(f"  {car}: {cnt}")
        longest = max(samples, key=lambda x: x["length"]) if samples else None
        shortest = min(samples, key=lambda x: x["length"]) if samples else None
        if longest:
            logger.info(f"Longest sequence: car={longest['car_id']}, seg={longest['segment_id']}, len={longest['length']}, label={longest['label_name']}")
        if shortest:
            logger.info(f"Shortest sequence: car={shortest['car_id']}, seg={shortest['segment_id']}, len={shortest['length']}, label={shortest['label_name']}")
        logger.info("=" * 60)


# =============================================================================
# Hàm tiện ích (3 nguồn dữ liệu: splits gốc, synthetic cũ, splits V2)
# =============================================================================

# Đường dẫn đến các file bổ sung cho từng tập
_SYNTHETIC_TRAIN = "data/sample/car1_synthetic_train.csv"
_SYNTHETIC_VAL   = "data/sample/car1_synthetic_val.csv"
_SYNTHETIC_TEST  = "data/sample/car1_synthetic_test.csv"
_V2_TRAIN = "data/splits_v2/train.csv"
_V2_VAL   = "data/splits_v2/val.csv"
_V2_TEST  = "data/splits_v2/test.csv"
_IDLE_AUG_TRAIN = "data/sample/idle_augmented/augmented_idle_train.csv"
_IDLE_AUG_VAL   = "data/sample/idle_augmented/augmented_idle_val.csv"
_IDLE_AUG_TEST  = "data/sample/idle_augmented/augmented_idle_test.csv"

def build_datasets(
    train_path: str = "data/splits/train.csv",
    val_path: str = "data/splits/val.csv",
    test_path: str = "data/splits/test.csv",
    config: Optional[Dict] = None,
    extra_train_csv: Optional[Union[str, List[str]]] = None,
    extra_val_csv: Optional[Union[str, List[str]]] = None,
    extra_test_csv: Optional[Union[str, List[str]]] = None,
    include_v2: bool = True,
    include_old_synthetic: bool = True,
    include_idle_augmented: bool = True,
) -> Tuple[FuelSequenceDataset, FuelSequenceDataset, FuelSequenceDataset]:
    """
    Build cả 3 dataset.
    Bạn có thể bật/tắt từng nguồn dữ liệu bằng các cờ include_*.
    """

    def _build_extra_list(
        custom: Optional[Union[str, List[str]]],
        synthetic_file: str,
        v2_file: str,
        idle_file: str,
    ) -> Optional[List[str]]:
        extra = []
        if include_old_synthetic:
            extra.append(synthetic_file)
        if include_v2:
            extra.append(v2_file)
        if include_idle_augmented:
            extra.append(idle_file)
        if custom:
            if isinstance(custom, str):
                extra.append(custom)
            else:
                extra.extend(custom)
        return extra if extra else None

    train_extra = _build_extra_list(extra_train_csv, _SYNTHETIC_TRAIN, _V2_TRAIN, _IDLE_AUG_TRAIN)
    val_extra   = _build_extra_list(extra_val_csv, _SYNTHETIC_VAL, _V2_VAL, _IDLE_AUG_VAL)
    test_extra  = _build_extra_list(extra_test_csv, _SYNTHETIC_TEST, _V2_TEST, _IDLE_AUG_TEST)

    builder = FuelDatasetBuilder(config)

    logger.info("===== BUILD TRAIN DATASET =====")
    train_dataset = builder.build(train_path, is_train=True, extra_csv=train_extra)

    logger.info("===== BUILD VALIDATION DATASET =====")
    val_dataset = builder.build(val_path, is_train=False, extra_csv=val_extra)

    logger.info("===== BUILD TEST DATASET =====")
    test_dataset = builder.build(test_path, is_train=False, extra_csv=test_extra)

    return train_dataset, val_dataset, test_dataset

# =============================================================================
# Chạy thử
# =============================================================================
if __name__ == "__main__":
    config = {
        "min_sequence_length": 2,
        "fillna_strategy": "median",
        "strict_feature_check": False,
        "verbose_nan_log": True,
    }

    train_ds, val_ds, test_ds = build_datasets(config=config)

    print("\n=== Sample đầu tiên từ Train ===")
    sample = train_ds[0]
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape}, min={v.min():.2f}, max={v.max():.2f}")
        else:
            print(f"{k}: {v}")