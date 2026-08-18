#!/usr/bin/env python3
"""
Module: dataset_split.py
Mục tiêu: Chia tập dữ liệu fusion thành train/val/test theo thời gian (time‑based split),
          mỗi xe riêng biệt, hỗ trợ Virtual Segment cho segment quá dài.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    def __init__(
        self,
        # input_path: str = "data/processed/fusion/fusion_dataset.csv",
        # output_dir: str = "data/splits",
        input_path="data/processed/fusion_v2/fusion_dataset.csv",
        output_dir="data/splits_v2",  
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        max_segment_points: int = 1024,
        enable_virtual_segment: bool = True,
    ):
        """
        Khởi tạo bộ chia dữ liệu.

        Args:
            input_path: Đường dẫn file fusion CSV.
            output_dir: Thư mục lưu kết quả.
            train_ratio, val_ratio, test_ratio: tỉ lệ chia (tổng = 1).
            random_state: seed cho ngẫu nhiên (dành cho các xử lý phụ, không ảnh hưởng chia chính).
            max_segment_points: độ dài tối đa của một segment (điểm).
            enable_virtual_segment: nếu True, tự động chia segment dài thành các virtual segment.
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.max_segment_points = max_segment_points
        self.enable_virtual_segment = enable_virtual_segment

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
            raise ValueError("Tổng train_ratio + val_ratio + test_ratio phải bằng 1.0")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.segment_table_raw = None
        self.segment_table = None
        self.virtual_seg_info = []
        self.train_ids = set()
        self.val_ids = set()
        self.test_ids = set()
        self.removed_indices = []

    def load_and_validate(self) -> pd.DataFrame:
        """Đọc CSV, làm sạch NaN final_label, kiểm tra segment nhất quán."""
        logger.info(f"Đọc dữ liệu từ {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        logger.info(f"Đã đọc {len(self.df)} dòng dữ liệu")

        required_cols = ["car_id", "segment_id", "final_label"]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc: {missing}")

        # Phát hiện NaN trong final_label
        nan_mask = self.df["final_label"].isna()
        nan_count = nan_mask.sum()
        if nan_count > 0:
            logger.warning(f"Phát hiện {nan_count} dòng có final_label là NaN.")
            self.removed_indices = self.df[nan_mask].index.tolist()
            logger.info("Danh sách các dòng bị loại bỏ (index trong file gốc):")
            for idx in self.removed_indices:
                row = self.df.loc[idx]
                logger.info(
                    f"  Index {idx}: car_id={row['car_id']}, "
                    f"segment_id={row['segment_id']}, "
                    f"timestamp={row.get('timestamp', '?')}"
                )
            self.df = self.df[~nan_mask].copy()
            logger.info(f"Còn lại {len(self.df)} dòng sau khi loại bỏ NaN.")

        # Kiểm tra segment chỉ có 1 label duy nhất
        logger.info("Kiểm tra tính duy nhất của label trong mỗi segment...")
        segment_labels = self.df.groupby(["car_id", "segment_id"])["final_label"].nunique()
        inconsistent = segment_labels[segment_labels > 1]
        if len(inconsistent) > 0:
            logger.error("Phát hiện segment có nhiều label khác nhau:")
            for idx, count in inconsistent.items():
                logger.error(f"  - car_id={idx[0]}, segment_id={idx[1]}: {count} labels")
            raise ValueError("Dữ liệu không nhất quán: một segment chứa nhiều final_label.")
        logger.info("Tất cả segment đều hợp lệ (duy nhất một label).")
        return self.df

    def build_segment_table(self) -> pd.DataFrame:
        """Tạo bảng segment gốc (mỗi segment 1 dòng)."""
        logger.info("Xây dựng bảng segment gốc...")
        df = self.df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        seg = df.groupby(["car_id", "segment_id"])
        self.segment_table_raw = seg.agg(
            label=("final_label", "first"),
            num_points=("final_label", "count"),
            start_time=("timestamp", "min") if "timestamp" in df.columns else None,
            end_time=("timestamp", "max") if "timestamp" in df.columns else None,
        ).reset_index()
        self.segment_table_raw.rename(columns={"label": "final_label"}, inplace=True)

        logger.info(f"Tổng số segment gốc: {len(self.segment_table_raw)}")
        self.segment_table = self.segment_table_raw.copy()  # mặc định dùng bảng gốc
        return self.segment_table_raw

    def _apply_virtual_segmentation(self) -> None:
        """
        Chia các segment quá dài thành các virtual segment nhỏ hơn.
        Cập nhật self.df (thay đổi segment_id) và self.segment_table.
        """
        if not self.enable_virtual_segment:
            logger.info("Virtual Segment bị tắt (ENABLE_VIRTUAL_SEGMENT=False). Bỏ qua bước chia.")
            return

        logger.info(
            f"Áp dụng Virtual Segment với độ dài tối đa {self.max_segment_points} điểm..."
        )

        df = self.df.copy()
        if "timestamp" in df.columns:
            df = df.sort_values(["car_id", "segment_id", "timestamp"])
        else:
            # Giả định dữ liệu đã theo thứ tự thời gian
            pass

        updated_rows = []
        new_segments = []
        virtual_info_list = []

        for (car, seg_id), group in df.groupby(["car_id", "segment_id"]):
            num_points = len(group)
            if num_points <= self.max_segment_points:
                updated_rows.append(group)
                new_segments.append({
                    "car_id": car,
                    "segment_id": seg_id,
                    "final_label": group["final_label"].iloc[0],
                    "num_points": num_points,
                    "start_time": group["timestamp"].min() if "timestamp" in group.columns else None,
                    "end_time": group["timestamp"].max() if "timestamp" in group.columns else None,
                })
            else:
                num_parts = int(np.ceil(num_points / self.max_segment_points))
                logger.info(
                    f"Chia segment {car}-{seg_id} ({num_points} điểm) thành {num_parts} phần."
                )
                part_lengths = []
                for i in range(num_parts):
                    start_idx = i * self.max_segment_points
                    end_idx = min((i + 1) * self.max_segment_points, num_points)
                    part = group.iloc[start_idx:end_idx].copy()
                    new_seg_id = f"{seg_id}_part{i+1}"
                    part["segment_id"] = new_seg_id
                    updated_rows.append(part)

                    part_len = len(part)
                    part_lengths.append(part_len)

                    new_segments.append({
                        "car_id": car,
                        "segment_id": new_seg_id,
                        "final_label": part["final_label"].iloc[0],
                        "num_points": part_len,
                        "start_time": part["timestamp"].min() if "timestamp" in part.columns else None,
                        "end_time": part["timestamp"].max() if "timestamp" in part.columns else None,
                    })

                virtual_info_list.append({
                    "car_id": car,
                    "segment_id_original": seg_id,
                    "num_points_original": num_points,
                    "num_parts": num_parts,
                    "part_lengths": part_lengths,
                })

        self.df = pd.concat(updated_rows, ignore_index=True)
        self.segment_table = pd.DataFrame(new_segments)
        if "start_time" in self.segment_table.columns:
            self.segment_table["start_time"] = pd.to_datetime(self.segment_table["start_time"])
            self.segment_table["end_time"] = pd.to_datetime(self.segment_table["end_time"])

        self.virtual_seg_info = virtual_info_list

        logger.info(
            f"Virtual Segment: {len(self.segment_table_raw)} segment gốc, "
            f"{len(virtual_info_list)} segment bị chia, "
            f"tổng số segment sau chia: {len(self.segment_table)}"
        )

        self._validate_virtual_segments()

    def _validate_virtual_segments(self) -> None:
        """Kiểm tra sau khi chia virtual segment."""
        logger.info("Kiểm tra tính toàn vẹn sau khi chia Virtual Segment...")
        if len(self.df) != len(self.df_original_before_virtual):
            logger.error(
                f"Số dòng thay đổi: trước={len(self.df_original_before_virtual)}, "
                f"sau={len(self.df)}"
            )
            raise RuntimeError("Virtual Segment làm thay đổi số dòng dữ liệu!")
        for info in self.virtual_seg_info:
            car = info["car_id"]
            seg_orig = info["segment_id_original"]
            expected_total = info["num_points_original"]
            actual_total = sum(info["part_lengths"])
            if actual_total != expected_total:
                raise RuntimeError(
                    f"Segment {car}-{seg_orig}: tổng điểm không khớp "
                    f"({actual_total} != {expected_total})"
                )
        label_counts = self.df.groupby(["car_id", "segment_id"])["final_label"].nunique()
        if (label_counts > 1).any():
            raise RuntimeError("Phát hiện Virtual Segment chứa nhiều label!")
        if "timestamp" in self.df.columns:
            for (car, seg_id), group in self.df.groupby(["car_id", "segment_id"]):
                if not group["timestamp"].is_monotonic_increasing:
                    raise RuntimeError(
                        f"Segment {car}-{seg_id} không đảm bảo thứ tự thời gian!"
                    )
        logger.info("Virtual Segment hợp lệ.")

    def time_based_split_by_car(self) -> Dict[str, List]:
        """
        Chia dữ liệu theo thời gian: với mỗi xe, các segment được sắp xếp
        theo start_time, sau đó cắt tuần tự thành train, val, test dựa trên tỉ lệ.
        """
        logger.info("Bắt đầu chia segment theo thời gian (time‑based split)...")
        segments = self.segment_table

        if "start_time" not in segments.columns:
            raise ValueError("Không có cột 'start_time' trong segment_table, không thể chia theo thời gian.")

        car_ids = segments["car_id"].unique()
        logger.info(f"Số xe: {len(car_ids)}: {list(car_ids)}")

        train_list, val_list, test_list = [], [], []

        for car in car_ids:
            car_segments = segments[segments["car_id"] == car].copy()
            car_segments.sort_values("start_time", inplace=True)
            seg_ids = car_segments[["car_id", "segment_id"]].values.tolist()
            n = len(seg_ids)

            if n == 0:
                continue

            if n == 1:
                logger.warning(f"Xe {car} chỉ có 1 segment, gán vào train.")
                train_list.extend(seg_ids)
                continue

            # Tính số lượng segment cho từng tập
            train_end = max(1, int(n * self.train_ratio))
            val_end = train_end + max(1, int(n * self.val_ratio))

            # Đảm bảo test có ít nhất 1 segment nếu có thể
            if n >= 3:
                # Điều chỉnh để val và test không bị rỗng
                if val_end >= n:
                    val_end = n - 1
                if train_end >= n:
                    train_end = n - 2
                    val_end = n - 1
            elif n == 2:
                # Với 2 segment, chỉ chia train/test, bỏ qua val nếu val_ratio > 0
                if self.val_ratio > 0:
                    logger.warning(f"Xe {car} chỉ có 2 segment, không đủ cho val, sẽ chỉ chia train/test.")
                train_end = 1
                val_end = 1  # val sẽ rỗng

            train_ids = seg_ids[:train_end]
            if n == 2 and self.val_ratio > 0:
                val_ids = []
                test_ids = seg_ids[train_end:]
            else:
                val_ids = seg_ids[train_end:val_end]
                test_ids = seg_ids[val_end:]

            logger.info(
                f"Xe {car}: {n} segments, "
                f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
            )
            train_list.extend(train_ids)
            val_list.extend(val_ids)
            test_list.extend(test_ids)

        self.train_ids = set(map(tuple, train_list))
        self.val_ids = set(map(tuple, val_list))
        self.test_ids = set(map(tuple, test_list))

        logger.info(
            f"Kết quả chia: Train={len(self.train_ids)}, "
            f"Val={len(self.val_ids)}, Test={len(self.test_ids)} segments"
        )
        return {"train": train_list, "val": val_list, "test": test_list}

    def save_splits(self) -> None:
        """Lưu các file train.csv, val.csv, test.csv."""
        logger.info("Lưu các file dữ liệu đã chia...")
        df = self.df

        def filter_by_ids(ids_set):
            mask = df[["car_id", "segment_id"]].apply(tuple, axis=1).isin(ids_set)
            return df[mask]

        train_df = filter_by_ids(self.train_ids)
        val_df = filter_by_ids(self.val_ids)
        test_df = filter_by_ids(self.test_ids)

        train_df.to_csv(self.output_dir / "train.csv", index=False)
        val_df.to_csv(self.output_dir / "val.csv", index=False)
        test_df.to_csv(self.output_dir / "test.csv", index=False)

        logger.info(f"Train: {len(train_df)} dòng -> train.csv")
        logger.info(f"Validation: {len(val_df)} dòng -> val.csv")
        logger.info(f"Test: {len(test_df)} dòng -> test.csv")

        all_saved = (
            set(map(tuple, train_df[["car_id", "segment_id"]].values))
            | set(map(tuple, val_df[["car_id", "segment_id"]].values))
            | set(map(tuple, test_df[["car_id", "segment_id"]].values))
        )
        all_current = set(map(tuple, df[["car_id", "segment_id"]].values))
        missing = all_current - all_saved
        if missing:
            logger.error(f"Có segment bị bỏ sót: {missing}")
        else:
            logger.info("Tất cả segment đã được phân vào train/val/test.")

    def generate_report(self) -> str:
        """Tạo báo cáo markdown chi tiết."""
        logger.info("Tạo báo cáo split_report.md")
        df = self.df
        segments = self.segment_table
        total_rows = len(df)
        total_segments = len(segments)
        total_cars = df["car_id"].nunique()

        def get_split_df(ids_set):
            mask = df[["car_id", "segment_id"]].apply(tuple, axis=1).isin(ids_set)
            return df[mask]

        train_df = get_split_df(self.train_ids)
        val_df = get_split_df(self.val_ids)
        test_df = get_split_df(self.test_ids)

        label_all = df["final_label"].value_counts()
        label_train = train_df["final_label"].value_counts()
        label_val = val_df["final_label"].value_counts()
        label_test = test_df["final_label"].value_counts()

        def car_seg_counts(split_df):
            return split_df.groupby("car_id")["segment_id"].nunique()

        seg_train_car = car_seg_counts(train_df)
        seg_val_car = car_seg_counts(val_df)
        seg_test_car = car_seg_counts(test_df)

        seg_lengths = segments["num_points"]
        min_len = seg_lengths.min()
        max_len = seg_lengths.max()
        mean_len = seg_lengths.mean()
        median_len = seg_lengths.median()

        all_labels = set(df["final_label"].unique())
        missing_labels_train = all_labels - set(train_df["final_label"].unique())
        missing_labels_val = all_labels - set(val_df["final_label"].unique())
        missing_labels_test = all_labels - set(test_df["final_label"].unique())

        cars_all = set(df["car_id"].unique())
        cars_train = set(train_df["car_id"].unique())
        cars_val = set(val_df["car_id"].unique())
        cars_test = set(test_df["car_id"].unique())
        missing_cars_train = cars_all - cars_train
        missing_cars_val = cars_all - cars_val
        missing_cars_test = cars_all - cars_test

        # Virtual segment report
        virtual_report = ""
        if self.enable_virtual_segment:
            orig_count = len(self.segment_table_raw)
            split_count = len(self.virtual_seg_info)
            virtual_created = len(self.segment_table) - orig_count
            virtual_report = f"""
## Virtual Segment Report

- Original Segments: {orig_count}
- Segments Split: {split_count}
- Virtual Segments Created: {virtual_created}
- Final Segments: {len(self.segment_table)}

### Chi tiết các segment bị chia
| Segment (Car-ID gốc) | Original Length | Parts | New Lengths |
|-----------------------|-----------------|-------|-------------|
"""
            for info in self.virtual_seg_info:
                car = info["car_id"]
                seg_orig = info["segment_id_original"]
                orig_len = info["num_points_original"]
                parts = info["num_parts"]
                lengths_str = " / ".join(map(str, info["part_lengths"]))
                virtual_report += f"| {car}-{seg_orig} | {orig_len} | {parts} | {lengths_str} |\n"
        else:
            virtual_report = "\n## Virtual Segment Report\nVirtual Segment is disabled.\n"

        report = f"""# Data Split Report

## Tổng quan
- Tổng số dòng (sau khi loại bỏ NaN): {total_rows}
- Tổng số segment (sau khi áp dụng Virtual Segment): {total_segments}
- Tổng số xe: {total_cars}
- Số dòng bị loại bỏ do `final_label` NaN: {len(self.removed_indices)}
  (Xem log để biết chi tiết index từng dòng)
- **Phương pháp chia: Time‑based (theo thứ tự thời gian từng xe)**

## Phân bố tập dữ liệu
| Tập         | Số dòng | Số segment | Tỉ lệ dòng |
|-------------|---------|------------|------------|
| Train       | {len(train_df)} | {len(self.train_ids)} | {len(train_df)/total_rows:.2%} |
| Validation  | {len(val_df)} | {len(self.val_ids)} | {len(val_df)/total_rows:.2%} |
| Test        | {len(test_df)} | {len(self.test_ids)} | {len(test_df)/total_rows:.2%} |

## Phân bố label

### Toàn bộ dữ liệu
{label_all.to_markdown()}

### Train
{label_train.to_markdown()}

### Validation
{label_val.to_markdown()}

### Test
{label_test.to_markdown()}

## Phân bố segment theo xe

### Train
{seg_train_car.to_markdown()}

### Validation
{seg_val_car.to_markdown()}

### Test
{seg_test_car.to_markdown()}

## Thông tin độ dài segment
- Min: {min_len}
- Max: {max_len}
- Mean: {mean_len:.2f}
- Median: {median_len}

## Kiểm tra ràng buộc
- Segment bị chia cắt: Không (chia nguyên segment)
- Label bị mất trong Train: {missing_labels_train if missing_labels_train else 'Không'}
- Label bị mất trong Validation: {missing_labels_val if missing_labels_val else 'Không'}
- Label bị mất trong Test: {missing_labels_test if missing_labels_test else 'Không'}
- Xe không có trong Train: {missing_cars_train if missing_cars_train else 'Không'}
- Xe không có trong Validation: {missing_cars_val if missing_cars_val else 'Không'}
- Xe không có trong Test: {missing_cars_test if missing_cars_test else 'Không'}

{virtual_report}
"""
        return report

    def save_report(self, report_content: str) -> None:
        """Lưu báo cáo ra file markdown."""
        report_path = self.output_dir / "split_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info(f"Báo cáo đã được lưu tại {report_path}")

    def run(self) -> None:
        """Thực hiện toàn bộ quy trình."""
        logger.info("===== BẮT ĐẦU CHIA DỮ LIỆU =====")
        self.load_and_validate()
        self.build_segment_table()
        self.df_original_before_virtual = self.df.copy()
        self._apply_virtual_segmentation()
        self.time_based_split_by_car()
        self.save_splits()
        report = self.generate_report()
        self.save_report(report)
        logger.info("===== HOÀN THÀNH =====")


def main():
    splitter = DatasetSplitter(
        # input_path="data/processed/fusion/fusion_dataset.csv",
        # output_dir="data/splits",
        input_path="data/processed/fusion_v2/fusion_dataset.csv",   # Đường dẫn mới
        output_dir="data/splits_v2", 
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        max_segment_points=1024,
        enable_virtual_segment=True,
    )
    splitter.run()


if __name__ == "__main__":
    main()