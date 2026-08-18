#!/usr/bin/env python3
"""
state_recognition.py — FuelSentinel-AI Pipeline, Step 4: State Recognition (v4.4 - Dynamic Col Detection)
Tự động xuất 5 biểu đồ trạng thái. 
Sửa lỗi KeyError bằng cách tự động phát hiện cột 'FuelTime' và 'FuelLevel'.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cấu hình thư viện vẽ biểu đồ
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration — đường dẫn I/O
# ---------------------------------------------------------------------------
SEGMENT_FEATURES_CANDIDATES = [
    Path("data/processed/features/segment_feature_timeline.csv"),
    Path("data/processed/features/segment_features.csv"),
]
TIMELINE_CSV: Path = Path("data/processed/segment/segment_data.csv")

OUTPUT_DIR: Path = Path("data/processed/labels")    
OUTPUT_LABELS_CSV: Path = OUTPUT_DIR / "segment_labels.csv"
OUTPUT_TIMELINE_LABELED_CSV: Path = OUTPUT_DIR / "timeline_labeled.csv"
REPORT_MD: Path = OUTPUT_DIR / "state_report.md"

CHART_OUTPUT_DIR: Path = Path("data/processed/labels/charts") 

CAR_ID_COL = "car_id"
SEGMENT_ID_COL = "segment_id"

# ---------------------------------------------------------------------------
# Cấu hình logic gán nhãn
# ---------------------------------------------------------------------------
# [IDLE]
IDLE_MIN_DURATION = 30.0          
IDLE_SPEED_ZERO_RATIO = 0.98      
IDLE_ALLOWED_FUEL_CHANGE = 1.5    
IDLE_BASE_CONF = 0.90

# [REFUEL]
REFUEL_FUEL_CHANGE_MIN = 5.0          
REFUEL_BASE_CONF = 0.95

# [THEFT]
THEFT_MAX_DURATION = 120.0           
THEFT_FUEL_CHANGE_MIN = -8.0        
THEFT_RATE_PER_HOUR_THRESHOLD = -15.0 
THEFT_MAX_DROP_THRESHOLD = 8.0       
THEFT_BASE_CONF = 0.85

# [DRIVING]
DRIVING_BASE_CONF = 0.80
DRIVING_PAUSE_CONF = 0.70      
DRIVING_MIN_DISTANCE = 100.0   

# [Quality / Fallback]
SINGLE_POINT_MAX_SAMPLES = 1
SHORT_SEGMENT_MAX_SAMPLES = 5
SINGLE_POINT_CONF_CAP = 0.40
SHORT_SEGMENT_CONF_CAP = 0.75
MIN_CONFIDENCE_KEEP = 0.45

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("state_recognition")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm(value: float, low: float, high: float) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if high == low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))

def _safe(row: pd.Series, col: str) -> float:
    v = row.get(col, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan

def _mean_ignore_nan(*vals: float, default: float = 0.5) -> float:
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return default
    return float(np.mean(clean))

def _rate_per_hour(row: pd.Series) -> float:
    fuel_change = _safe(row, "fuel_change")
    duration_min = _safe(row, "duration_min")
    if math.isnan(fuel_change):
        return np.nan
    if math.isnan(duration_min) or duration_min <= 0:
        return np.nan
    return fuel_change / (duration_min / 60.0)

# ---------------------------------------------------------------------------
# Quality Check
# ---------------------------------------------------------------------------
def _determine_quality(row: pd.Series) -> Tuple[str, bool]:
    sample_count = _safe(row, "sample_count")
    fuel_start = _safe(row, "fuel_start")
    fuel_end = _safe(row, "fuel_end")

    if math.isnan(sample_count) or sample_count <= 0 or math.isnan(fuel_start) or math.isnan(fuel_end):
        return "INVALID_DATA", True

    if sample_count <= SINGLE_POINT_MAX_SAMPLES:
        return "SINGLE_POINT", False
    if sample_count <= SHORT_SEGMENT_MAX_SAMPLES:
        return "SHORT_SEGMENT", False
    return "GOOD", False

# ---------------------------------------------------------------------------
# Logic gán nhãn chính
# ---------------------------------------------------------------------------
def classify_segment(row: pd.Series) -> Tuple[str, float, str]:
    quality, is_invalid = _determine_quality(row)
    if is_invalid:
        return "NULL", 0.0, quality

    fuel_change = _safe(row, "fuel_change")
    max_rise = _safe(row, "max_rise")
    max_drop = _safe(row, "max_drop")
    duration_min = _safe(row, "duration_min")
    total_distance = _safe(row, "total_distance")
    speed_mean = _safe(row, "speed_mean")
    speed_zero_ratio = _safe(row, "speed_zero_ratio")
    rate_per_hour = _rate_per_hour(row)

    is_stationary = (
        (not math.isnan(speed_zero_ratio) and speed_zero_ratio >= 0.98) or 
        (not math.isnan(speed_mean) and speed_mean < 0.5)
    )
    has_movement = (not math.isnan(total_distance) and total_distance > DRIVING_MIN_DISTANCE) or (not math.isnan(speed_mean) and speed_mean > 5.0)

    # [PRIORITY 1] REFUEL
    if fuel_change >= REFUEL_FUEL_CHANGE_MIN or max_rise >= 10.0:
        abs_confidence = 1.0 if max_rise > 50.0 else REFUEL_BASE_CONF
        return "Refuel", abs_confidence, quality

    # [PRIORITY 2] THEFT
    if is_stationary and not math.isnan(duration_min) and duration_min <= THEFT_MAX_DURATION:
        is_fast_drop = (fuel_change <= THEFT_FUEL_CHANGE_MIN)
        is_high_rate = (fuel_change < 0 and rate_per_hour <= THEFT_RATE_PER_HOUR_THRESHOLD)
        is_single_shock = (max_drop >= THEFT_MAX_DROP_THRESHOLD)

        if is_fast_drop or is_high_rate or is_single_shock:
            abs_confidence = 1.0 if max_drop > 50.0 else THEFT_BASE_CONF
            return "Theft", abs_confidence, quality

    # [PRIORITY 3] IDLE
    if is_stationary and duration_min > IDLE_MIN_DURATION and abs(fuel_change) <= IDLE_ALLOWED_FUEL_CHANGE:
        return "Idle", IDLE_BASE_CONF, quality

    # [PRIORITY 4] DRIVING
    if has_movement or fuel_change < -2.5:
        distance_score = _norm(total_distance, DRIVING_MIN_DISTANCE, DRIVING_MIN_DISTANCE * 10)
        speed_score = _norm(speed_mean, 5.0, 30.0)
        conf = float(np.clip(DRIVING_BASE_CONF + 0.15 * _mean_ignore_nan(distance_score, speed_score), 0.0, 1.0))
        return "Driving", conf, quality

    if is_stationary:
        return "Driving", DRIVING_PAUSE_CONF, quality
    return "NULL", 0.40, quality

# ---------------------------------------------------------------------------
# [HELPERS] Tìm cột động (Fix KeyError triệt để)
# ---------------------------------------------------------------------------
def _get_datetime_col(df: pd.DataFrame) -> str:
    """Tìm cột thời gian."""
    for col in ['FuelTime', 'timestamp', 'Timestamp', 'time', 'Time', 'datetime', 'DateTime', 't']:
        if col in df.columns:
            return col
    raise KeyError(f"Không tìm thấy cột thời gian. Các cột hiện tại: {list(df.columns)}")

def _get_fuel_col(df: pd.DataFrame) -> str:
    """Tìm cột nhiên liệu."""
    for col in ['FuelLevel', 'fuel', 'Fuel', 'fuel_level', 'Fuel_Level']:
        if col in df.columns:
            return col
    raise KeyError(f"Không tìm thấy cột nhiên liệu. Các cột hiện tại: {list(df.columns)}")

# ---------------------------------------------------------------------------
# [CHART EXPORTER] Tự động vẽ biểu đồ
# ---------------------------------------------------------------------------
def export_state_charts(timeline_labeled_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Không tìm thấy thư viện matplotlib. Bỏ qua bước xuất ảnh báo cáo.")
        return

    if not timeline_labeled_path.exists():
        logger.warning("File timeline_labeled.csv chưa tồn tại. Bỏ qua bước xuất ảnh báo cáo.")
        return

    try:
        df = pd.read_csv(timeline_labeled_path)
        
        # [FIXED] Tìm cột thời gian và cột nhiên liệu tự động
        datetime_col = _get_datetime_col(df)
        fuel_col = _get_fuel_col(df)
        
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # 1. Chọn chiếc xe có dữ liệu tốt nhất
        car_candidates = df[CAR_ID_COL].unique()
        selected_car = None
        max_score = 0
        
        for car in car_candidates:
            car_df = df[df[CAR_ID_COL] == car]
            unique_dates = car_df[datetime_col].dt.date.nunique()
            unique_labels = car_df['final_label'].nunique()
            score = unique_dates * 10 + unique_labels * 50
            if score > max_score:
                max_score = score
                selected_car = car
        
        if selected_car is None:
            logger.warning("Không tìm thấy xe nào có dữ liệu hợp lệ để xuất biểu đồ.")
            return

        logger.info(f"Đang chuẩn bị xuất ảnh báo cáo cho xe: {selected_car} (Đã tìm thấy cột '{fuel_col}')")
        df_car = df[df[CAR_ID_COL] == selected_car]

        # 2. Tìm các ngày có đủ ít nhất 3 trạng thái
        date_ranking = defaultdict(lambda: {'labels': set(), 'counts': defaultdict(int)})
        for _, row in df_car.iterrows():
            date = row[datetime_col].date()
            label = row['final_label']
            if pd.notna(label):
                date_ranking[date]['labels'].add(label)
                date_ranking[date]['counts'][label] += 1
        
        sorted_dates = sorted(
            date_ranking.keys(),
            key=lambda d: (
                len(date_ranking[d]['labels']) * 100 + 
                (50 if 'Theft' in date_ranking[d]['labels'] else 0) +
                (30 if 'Refuel' in date_ranking[d]['labels'] else 0)
            ),
            reverse=True
        )

        selected_dates = []
        for d in sorted_dates:
            if len(selected_dates) >= 5:
                break
            if len(date_ranking[d]['labels']) >= 3:
                selected_dates.append(d)
        
        if len(selected_dates) < 3:
            for d in sorted_dates:
                if d not in selected_dates and len(selected_dates) < 5:
                    selected_dates.append(d)
        
        # 3. Vẽ biểu đồ cho từng ngày
        CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        STATE_COLORS = {
            'Driving': '#7d8aff',
            'Idle': '#8FBC8F',
            'Refuel': '#FFD700',
            'Theft': '#FF7F7F',
            'NULL': '#FFFFFF'
        }

        for date in selected_dates:
            day_df = df_car[df_car[datetime_col].dt.date == date].sort_values(datetime_col)
            if len(day_df) < 10:
                continue

            fig, ax = plt.subplots(figsize=(16, 6))
            
            # Tô màu vùng trạng thái
            current_label = None
            start_time = None
            
            for i, row in day_df.iterrows():
                label = row['final_label']
                ts = row[datetime_col]
                
                if pd.isna(label):
                    label = 'NULL'
                
                if label != current_label:
                    if current_label is not None and start_time is not None:
                        ax.axvspan(start_time, ts, color=STATE_COLORS.get(current_label, '#FFFFFF'), alpha=0.6)
                    current_label = label
                    start_time = ts
            
            # Tô màu đoạn cuối
            if current_label is not None and start_time is not None:
                ax.axvspan(start_time, day_df[datetime_col].iloc[-1], color=STATE_COLORS.get(current_label, '#FFFFFF'), alpha=0.6)
            
            # [FIXED] Vẽ đường nhiên liệu dùng tên cột linh hoạt
            ax.plot(day_df[datetime_col], day_df[fuel_col], color='black', linewidth=1.5, label='Fuel Level')
            
            # Format
            ax.set_title(f"Fuel Timeline - {selected_car} | {date.strftime('%Y-%m-%d')}", fontsize=14, fontweight='bold')
            ax.set_ylabel("Nhiên liệu (L)", fontsize=11)
            ax.set_xlabel("Thời gian", fontsize=11)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Tạo Legend (Chú thích)
            patches = []
            for label, color in STATE_COLORS.items():
                if label != 'NULL':
                    patches.append(plt.Rectangle((0,0), 1, 1, color=color, alpha=0.6, label=label))
            patches.append(plt.Line2D([0], [0], color='black', lw=1.5, label='Fuel'))
            ax.legend(handles=patches, loc='upper right')
            
            # Lưu ảnh
            chart_path = CHART_OUTPUT_DIR / f"{selected_car}_{date.strftime('%Y-%m-%d')}_timeline.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=150)
            plt.close()
            logger.info(f"  - Đã xuất ảnh báo cáo: {chart_path}")

    except Exception as e:
        logger.error(f"Lỗi khi xuất biểu đồ báo cáo: {e}", exc_info=True)

# ---------------------------------------------------------------------------
# Report & Main pipeline
# ---------------------------------------------------------------------------
def generate_report(labels_df: pd.DataFrame, report_path: Path) -> None:
    n_total = len(labels_df)
    md = [
        "# State Recognition Report (v4.4 - Auto Chart Export & Dynamic Col)",
        "",
        "## Tổng quan",
        f"- Tổng số segment: {n_total}",
    ]

    if n_total:
        label_counts = labels_df["final_label"].value_counts(dropna=False)
        null_ratio = label_counts.get("NULL", 0) / n_total * 100
        md += [
            f"- Tỉ lệ NULL (không đủ tin cậy): {null_ratio:.1f}%",
            "",
            "## Phân phối final_label",
            "",
            "| final_label | Số segment | Tỉ lệ |",
            "|---|---|---|",
        ]
        for label, cnt in label_counts.items():
            md.append(f"| {label} | {cnt} | {cnt / n_total * 100:.1f}% |")

        md += ["", "## Phân phối segment_quality", "", "| segment_quality | Số segment |", "|---|---|"]
        for q, cnt in labels_df["segment_quality"].value_counts().items():
            md.append(f"| {q} | {cnt} |")

        md += ["", "## Thống kê confidence theo final_label", "",
               "| final_label | mean | median | min | max |", "|---|---|---|---|---|"]
        for label, grp in labels_df.groupby("final_label")["confidence"]:
            md.append(f"| {label} | {grp.mean():.2f} | {grp.median():.2f} | {grp.min():.2f} | {grp.max():.2f} |")

        low_conf_good = labels_df[
            (labels_df["segment_quality"] == "GOOD")
            & (labels_df["final_label"] != "NULL")
            & (labels_df["confidence"] < 0.55)
        ]
        md += [
            "",
            "## Cảnh báo cần xem lại",
            f"- Segment chất lượng GOOD nhưng confidence thấp (<0.55): {len(low_conf_good)} "
            f"({len(low_conf_good) / n_total * 100:.1f}% tổng số) — nên review thủ công trước khi đưa vào training.",
        ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("Báo cáo đã được lưu tại %s", report_path)

def _load_segment_features() -> pd.DataFrame:
    for path in SEGMENT_FEATURES_CANDIDATES:
        if path.exists():
            logger.info("Đọc đặc trưng segment từ %s", path)
            df = pd.read_csv(path)
            required_keys = {CAR_ID_COL, SEGMENT_ID_COL}
            missing_keys = required_keys - set(df.columns)
            if missing_keys:
                logger.error("%s thiếu cột khoá bắt buộc %s để join với timeline.", path, missing_keys)
                sys.exit(1)
            before = len(df)
            df = df.drop_duplicates(subset=[CAR_ID_COL, SEGMENT_ID_COL], keep="first").reset_index(drop=True)
            return df
    logger.error("Không tìm thấy file đặc trưng segment. Đã thử: %s", [str(p) for p in SEGMENT_FEATURES_CANDIDATES])
    sys.exit(1)

def main() -> None:
    features_df = _load_segment_features()
    logger.info("Gán nhãn cho %d segment...", len(features_df))
    results = features_df.apply(classify_segment, axis=1, result_type="expand")
    results.columns = ["final_label", "confidence", "segment_quality"]

    labels_df = pd.concat([
        features_df[[CAR_ID_COL, SEGMENT_ID_COL]].reset_index(drop=True), 
        results
    ], axis=1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(OUTPUT_LABELS_CSV, index=False)
    logger.info("Đã lưu %s (%d dòng)", OUTPUT_LABELS_CSV, len(labels_df))

    if TIMELINE_CSV.exists():
        logger.info("Đọc timeline thô từ %s", TIMELINE_CSV)
        try:
            timeline_df = pd.read_csv(TIMELINE_CSV)
            merge_cols = [CAR_ID_COL, SEGMENT_ID_COL, "final_label", "confidence", "segment_quality"]
            timeline_labeled = timeline_df.merge(labels_df[merge_cols], on=[CAR_ID_COL, SEGMENT_ID_COL], how="left")
            timeline_labeled.to_csv(OUTPUT_TIMELINE_LABELED_CSV, index=False)
            logger.info("Đã lưu %s (%d dòng)", OUTPUT_TIMELINE_LABELED_CSV, len(timeline_labeled))
            
            # Xuất ảnh báo cáo
            export_state_charts(OUTPUT_TIMELINE_LABELED_CSV)

        except Exception as e:
            logger.error("Không thể tạo timeline_labeled.csv: %s", e)
    else:
        logger.warning("Không tìm thấy %s — bỏ qua bước tạo timeline_labeled.csv", TIMELINE_CSV)

    generate_report(labels_df, REPORT_MD)

if __name__ == "__main__":
    main()