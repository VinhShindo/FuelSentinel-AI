"""
utils.py – Các tiện ích riêng cho module EDA (Step2).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import MAX_FUEL_LITERS, MAX_SPEED_KMH


def get_logger(name: str) -> logging.Logger:
    """Tạo logger với format thống nhất."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_dir(path: Path) -> None:
    """Tạo thư mục nếu chưa tồn tại."""
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig, path: Path, logger=None) -> None:
    """Lưu figure và đóng lại."""
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info("Đã lưu biểu đồ: %s", path.name)


def describe_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Thống kê missing cho mỗi cột."""
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"missing_count": missing, "percent": percent})
    return missing_df[missing_df["missing_count"] > 0]


def compute_sampling_stats(df: pd.DataFrame, timestamp_col: str = "timestamp") -> dict:
    """Tính thống kê khoảng cách thời gian giữa các mẫu."""
    if timestamp_col not in df.columns:
        return {}
    timestamps = pd.to_datetime(df[timestamp_col])
    diffs = timestamps.diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        return {}
    return {
        "min_sec": diffs.min(),
        "max_sec": diffs.max(),
        "median_sec": diffs.median(),
        "mean_sec": diffs.mean(),
        "mode_sec": diffs.mode().iloc[0] if not diffs.mode().empty else np.nan,
    }


def describe_extended(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Bảng thống kê mở rộng: mean, std, min, 25%, 50%, 75%, max, CV, skewness, kurtosis."""
    desc = df[columns].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["CV"] = desc["std"] / desc["mean"].abs().replace(0, np.nan)
    desc["skewness"] = df[columns].skew()
    desc["kurtosis"] = df[columns].kurtosis()
    return desc.round(4)


def describe_by_car(df: pd.DataFrame, car_col: str = "car_id") -> pd.DataFrame:
    """Thống kê cơ bản theo từng xe: số dòng, thời gian, missing, outlier (IQR)."""
    stats = []
    for car, grp in df.groupby(car_col):
        n = len(grp)
        time_min = grp["timestamp"].min()
        time_max = grp["timestamp"].max()
        missing_fuel = grp["fuel"].isnull().sum()
        missing_speed = grp["speed"].isnull().sum()
        missing_lat = grp["latitude"].isnull().sum()
        missing_lon = grp["longitude"].isnull().sum()

        # Outlier IQR fuel
        Q1_fuel = grp["fuel"].quantile(0.25)
        Q3_fuel = grp["fuel"].quantile(0.75)
        IQR_fuel = Q3_fuel - Q1_fuel
        lower_fuel = Q1_fuel - 1.5 * IQR_fuel
        upper_fuel = Q3_fuel + 1.5 * IQR_fuel
        outlier_fuel = ((grp["fuel"] < lower_fuel) | (grp["fuel"] > upper_fuel)).sum()

        # Outlier IQR speed
        Q1_speed = grp["speed"].quantile(0.25)
        Q3_speed = grp["speed"].quantile(0.75)
        IQR_speed = Q3_speed - Q1_speed
        lower_speed = Q1_speed - 1.5 * IQR_speed
        upper_speed = Q3_speed + 1.5 * IQR_speed
        outlier_speed = ((grp["speed"] < lower_speed) | (grp["speed"] > upper_speed)).sum()

        stats.append({
            "car_id": car,
            "rows": n,
            "time_start": time_min,
            "time_end": time_max,
            "missing_fuel": missing_fuel,
            "missing_speed": missing_speed,
            "missing_lat": missing_lat,
            "missing_lon": missing_lon,
            "outlier_fuel": outlier_fuel,
            "outlier_speed": outlier_speed,
        })
    return pd.DataFrame(stats)


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Kiểm tra chất lượng dữ liệu."""
    issues = {}
    total = len(df)

    if "fuel" in df.columns:
        fuel_negative = df["fuel"] < 0
        fuel_zero = df["fuel"] == 0
        fuel_over_max = df["fuel"] > MAX_FUEL_LITERS
        issues["fuel_negative"] = {"count": int(fuel_negative.sum()), "percent": round(fuel_negative.sum() / total * 100, 2), "description": "Fuel < 0"}
        issues["fuel_zero"] = {"count": int(fuel_zero.sum()), "percent": round(fuel_zero.sum() / total * 100, 2), "description": "Fuel = 0"}
        issues["fuel_over_max"] = {"count": int(fuel_over_max.sum()), "percent": round(fuel_over_max.sum() / total * 100, 2), "description": f"Fuel > {MAX_FUEL_LITERS}"}

    if "speed" in df.columns:
        speed_negative = df["speed"] < 0
        speed_over_max = df["speed"] > MAX_SPEED_KMH
        issues["speed_negative"] = {"count": int(speed_negative.sum()), "percent": round(speed_negative.sum() / total * 100, 2), "description": "Speed < 0"}
        issues["speed_over_max"] = {"count": int(speed_over_max.sum()), "percent": round(speed_over_max.sum() / total * 100, 2), "description": f"Speed > {MAX_SPEED_KMH} km/h"}

    if "latitude" in df.columns and "longitude" in df.columns:
        lat_zero = df["latitude"] == 0
        lon_zero = df["longitude"] == 0
        both_zero = lat_zero & lon_zero
        issues["gps_both_zero"] = {"count": int(both_zero.sum()), "percent": round(both_zero.sum() / total * 100, 2), "description": "Lat & Lng = 0 (mất GPS)"}
        issues["latitude_zero"] = {"count": int(lat_zero.sum()), "percent": round(lat_zero.sum() / total * 100, 2), "description": "Latitude = 0"}
        issues["longitude_zero"] = {"count": int(lon_zero.sum()), "percent": round(lon_zero.sum() / total * 100, 2), "description": "Longitude = 0"}

    if "fuel" in df.columns and "speed" in df.columns:
        fuel_zero_speed_pos = (df["fuel"] == 0) & (df["speed"] > 0)
        issues["fuel_zero_speed_pos"] = {"count": int(fuel_zero_speed_pos.sum()), "percent": round(fuel_zero_speed_pos.sum() / total * 100, 2), "description": "Fuel = 0 khi Speed > 0"}

    return issues


def generate_auto_comments(df: pd.DataFrame, sampling_stats: dict, quality_issues: dict = None, car_stats: pd.DataFrame = None) -> str:
    """Sinh nhận xét tự động (cập nhật cho nhiều xe)."""
    lines = []

    # Nhận xét chung
    if "fuel" in df.columns:
        fuel = df["fuel"].dropna()
        cv = fuel.std() / (fuel.mean() + 1e-9) if fuel.mean() != 0 else 0
        lines.append(f"- **Fuel**: CV = {cv:.3f}, biến động {'lớn' if cv > 0.3 else 'trung bình' if cv > 0.1 else 'thấp'}.")
    if "speed" in df.columns:
        speed = df["speed"].dropna()
        cv_speed = speed.std() / (speed.mean() + 1e-9) if speed.mean() != 0 else 0
        zero_pct = (speed == 0).mean() * 100
        lines.append(f"- **Speed**: CV = {cv_speed:.3f}, tỉ lệ dừng = {zero_pct:.1f}%.")

    if sampling_stats:
        mode = sampling_stats.get("mode_sec")
        mean = sampling_stats.get("mean_sec")
        if mode is not None and not np.isnan(mode):
            uniformity = "đều đặn" if abs(mode - mean) < 5 else "không đều"
            lines.append(f"- **Khoảng lấy mẫu chung**: trung bình {mean:.1f}s, mode {mode:.1f}s → {uniformity}.")

    if quality_issues:
        lines.append("### Cảnh báo chất lượng dữ liệu")
        for key, info in quality_issues.items():
            if info["count"] > 0:
                lines.append(f"- ⚠️ {info['description']}: {info['count']} dòng ({info['percent']}%)")

    if car_stats is not None:
        lines.append("### Nhận xét theo xe")
        for _, row in car_stats.iterrows():
            lines.append(f"- **{row['car_id']}**: {row['rows']} dòng, từ {row['time_start']} đến {row['time_end']}.")
            if row['outlier_fuel'] > 0:
                lines.append(f"  - Fuel outlier: {row['outlier_fuel']} dòng")
            if row['outlier_speed'] > 0:
                lines.append(f"  - Speed outlier: {row['outlier_speed']} dòng")

    return "\n".join(lines)