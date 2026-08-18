"""
visualization.py – Các hàm vẽ biểu đồ cho EDA (Step2) – hỗ trợ nhiều xe.
Bổ sung đầy đủ các biểu đồ theo yêu cầu phân tích.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EDAConfig
from .utils import get_logger, ensure_dir, save_fig

logger = get_logger(__name__)


# ---------- Hàm tiện ích ----------
def get_car_colors(car_ids: list) -> dict:
    """Gán màu cho từng car_id."""
    cmap = plt.cm.tab10
    return {car: cmap(i % 10) for i, car in enumerate(car_ids)}


def haversine_np(lon1: np.ndarray, lat1: np.ndarray,
                 lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    Tính khoảng cách Haversine vector hoá (trả về mét).
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371_000 * c  # mét


# ---------- 1. Histogram số lượng record của từng xe ----------
def plot_record_count_by_car(df: pd.DataFrame, out_path: Path,
                             car_col: str = "car_id") -> None:
    car_counts = df[car_col].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(car_counts.index.astype(str), car_counts.values,
                  color=plt.cm.tab10(np.arange(len(car_counts)) % 10))
    ax.set_title("Số lượng record của từng xe")
    ax.set_xlabel("Xe")
    ax.set_ylabel("Số dòng")
    # Thêm giá trị trên đỉnh cột
    for bar, val in zip(bars, car_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(car_counts.values),
                str(val), ha='center', va='bottom')
    save_fig(fig, out_path, logger)


# ---------- 2. Missing Value Analysis (Bar chart) ----------
def plot_missing_bar(df: pd.DataFrame, out_path: Path,
                     cols: Optional[List[str]] = None) -> None:
    if cols is None:
        cols = ["fuel", "speed", "latitude", "longitude", "timestamp"]
    missing_counts = df[cols].isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]  # chỉ vẽ cột có missing
    if missing_counts.empty:
        logger.info("Không có missing value để vẽ bar chart.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(missing_counts.index, missing_counts.values, color="salmon")
    ax.set_title("Số lượng giá trị thiếu (Missing Value)")
    ax.set_ylabel("Số dòng thiếu")
    for bar, val in zip(bars, missing_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(missing_counts.values),
                str(val), ha='center', va='bottom')
    save_fig(fig, out_path, logger)


# ---------- 3. Duplicate Analysis ----------
def plot_duplicate_ratio(df: pd.DataFrame, out_path: Path) -> None:
    dup_count = df.duplicated().sum()
    unique_count = len(df) - dup_count
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Unique", "Duplicate"], [unique_count, dup_count],
           color=["#4CAF50", "#F44336"])
    ax.set_title(f"Duplicate Rows (Tổng: {len(df)})")
    ax.set_ylabel("Số dòng")
    # Ghi phần trăm
    for i, (label, val) in enumerate(zip(["Unique", "Duplicate"], [unique_count, dup_count])):
        pct = val / len(df) * 100
        ax.text(i, val + 0.02*len(df), f"{pct:.1f}%", ha='center', va='bottom')
    save_fig(fig, out_path, logger)


# ---------- 4. Fuel Distribution (Histogram toàn bộ) ----------
def plot_fuel_distribution(df: pd.DataFrame, out_path: Path) -> None:
    if "fuel" not in df.columns:
        return
    fuel = df["fuel"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(fuel, bins=50, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.set_title("Fuel Distribution (toàn bộ)")
    ax.set_xlabel("Fuel (lít)")
    ax.set_ylabel("Tần suất")
    # Thêm đường trung bình
    mean_fuel = fuel.mean()
    ax.axvline(mean_fuel, color="red", linestyle="--", label=f"Mean = {mean_fuel:.1f}")
    ax.legend()
    save_fig(fig, out_path, logger)


# ---------- 5. Fuel Time Series (không smoothing) ----------
# Sử dụng lại plot_timeseries_by_car đã có, không cần thay đổi.


# ---------- 6. Speed Distribution ----------
def plot_speed_distribution(df: pd.DataFrame, out_path: Path) -> None:
    if "speed" not in df.columns:
        return
    speed = df["speed"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(speed, bins=50, color="#FF9800", alpha=0.8, edgecolor="white")
    ax.set_title("Speed Distribution (toàn bộ)")
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Tần suất")
    zero_pct = (speed == 0).mean() * 100
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.text(0.6, 0.9, f"Speed = 0: {zero_pct:.1f}%", transform=ax.transAxes, fontsize=10, color="red")
    save_fig(fig, out_path, logger)


# ---------- 7. Speed Time Series (overlay Fuel) ----------
def plot_speed_timeseries_overlay_fuel(df: pd.DataFrame, out_path: Path,
                                       car_col: str = "car_id",
                                       timestamp_col: str = "timestamp") -> None:
    if not {"speed", "fuel"}.issubset(df.columns):
        return
    car_ids = df[car_col].unique()
    n_cars = len(car_ids)
    colors = get_car_colors(car_ids)
    fig, axes = plt.subplots(nrows=n_cars, figsize=(12, 2.5 * n_cars), sharex=True)
    if n_cars == 1:
        axes = [axes]
    for ax, car in zip(axes, car_ids):
        car_df = df[df[car_col] == car]
        ax2 = ax.twinx()
        ax.plot(car_df[timestamp_col], car_df["speed"], color=colors[car], linewidth=0.8, label="Speed")
        ax2.plot(car_df[timestamp_col], car_df["fuel"], color="red", alpha=0.6, linewidth=0.8, label="Fuel")
        ax.set_title(f"{car}")
        ax.set_ylabel("Speed (km/h)", color=colors[car])
        ax2.set_ylabel("Fuel (lít)", color="red")
        ax.tick_params(axis='y', labelcolor=colors[car])
        ax2.tick_params(axis='y', labelcolor="red")
    axes[-1].set_xlabel("Thời gian")
    fig.suptitle("Speed & Fuel theo thời gian (từng xe)", y=1.02)
    # Tạo legend chung đơn giản
    lines1, labels1 = axes[-1].get_legend_handles_labels()
    lines2, labels2 = axes[-1].twinx().get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    save_fig(fig, out_path, logger)


# ---------- 8. GPS Trajectory (cho từng xe riêng) ----------
def plot_gps_trajectory_per_car(df: pd.DataFrame, out_path: Path,
                                car_col: str = "car_id") -> None:
    if not {"latitude", "longitude"}.issubset(df.columns):
        return
    car_ids = df[car_col].unique()
    n_cars = len(car_ids)
    cols = min(3, n_cars)
    rows = int(np.ceil(n_cars / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
    for idx, car in enumerate(car_ids):
        ax = axes[idx // cols][idx % cols]
        car_df = df[df[car_col] == car]
        ax.scatter(car_df["longitude"], car_df["latitude"], s=1, alpha=0.5, color=plt.cm.tab10(idx % 10))
        ax.set_title(f"GPS Trajectory - {car}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="datalim")
    # Ẩn các axes thừa
    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes[j // cols][j % cols])
    fig.suptitle("GPS Trajectory - từng xe", y=1.02)
    save_fig(fig, out_path, logger)


# ---------- 9. Fuel vs Speed Scatter (dùng dữ liệu gốc) ----------
# Đã có plot_scatter_by_car, có thể giữ hoặc làm riêng. Giữ nguyên plot_scatter_by_car.


# ---------- 10. Fuel Difference Distribution ----------
def plot_fuel_diff_distribution(df: pd.DataFrame, out_path: Path,
                                car_col: str = "car_id") -> None:
    if "fuel" not in df.columns:
        return
    # Tính diff nội tại trong mỗi xe
    diffs = df.groupby(car_col)["fuel"].diff().dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diffs, bins=100, color="#9C27B0", alpha=0.8, edgecolor="white")
    ax.set_title("Fuel Difference Distribution (Fuel(t) - Fuel(t-1))")
    ax.set_xlabel("Fuel Change (lít)")
    ax.set_ylabel("Tần suất")
    # Thêm đường zero
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    save_fig(fig, out_path, logger)


# ---------- 11. Time Interval Distribution (Histogram) ----------
def plot_time_interval_histogram(df: pd.DataFrame, out_path: Path,
                                 car_col: str = "car_id",
                                 timestamp_col: str = "timestamp") -> None:
    if timestamp_col not in df.columns:
        return
    intervals = df.groupby(car_col)[timestamp_col].diff().dt.total_seconds().dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(intervals.clip(upper=intervals.quantile(0.99)), bins=60,
            color="#009688", alpha=0.8, edgecolor="white")
    ax.set_title("Time Interval Distribution (khoảng cách giữa các mẫu)")
    ax.set_xlabel("Giây")
    ax.set_ylabel("Tần suất")
    save_fig(fig, out_path, logger)


# ---------- 12. GPS Drift Analysis ----------
def plot_gps_drift_histogram(df: pd.DataFrame, out_path: Path,
                             car_col: str = "car_id",
                             speed_threshold: float = 0.1) -> None:
    """
    Tính GPS displacement giữa các điểm liên tiếp khi xe gần như đứng yên (speed < threshold).
    """
    if not {"latitude", "longitude", "speed"}.issubset(df.columns):
        return
    drift_list = []
    for car, grp in df.groupby(car_col):
        grp = grp.sort_values("timestamp")  # đảm bảo thứ tự thời gian
        mask_stand = grp["speed"] < speed_threshold
        if mask_stand.sum() < 2:
            continue
        stand_idx = grp.index[mask_stand]
        # Tính displacement giữa các điểm đứng liên tiếp
        for i in range(len(stand_idx) - 1):
            idx1, idx2 = stand_idx[i], stand_idx[i+1]
            dist = haversine_np(grp.loc[idx1, "longitude"], grp.loc[idx1, "latitude"],
                                grp.loc[idx2, "longitude"], grp.loc[idx2, "latitude"])
            drift_list.append(dist)
    if not drift_list:
        logger.info("Không đủ dữ liệu đứng yên để phân tích GPS drift.")
        return
    drift = np.array(drift_list)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(drift, bins=50, color="#607D8B", alpha=0.8, edgecolor="white")
    ax.set_title(f"GPS Drift khi xe đứng (Speed < {speed_threshold} km/h)")
    ax.set_xlabel("Displacement (m)")
    ax.set_ylabel("Tần suất")
    save_fig(fig, out_path, logger)


# ---------- 13. Fuel Change vs GPS Distance ----------
def plot_fuel_change_vs_distance(df: pd.DataFrame, out_path: Path,
                                 car_col: str = "car_id") -> None:
    if not {"latitude", "longitude", "fuel"}.issubset(df.columns):
        return
    diffs_fuel = []
    distances = []
    for car, grp in df.groupby(car_col):
        grp = grp.sort_values("timestamp")
        fuel_diff = grp["fuel"].diff().dropna()
        # Tính khoảng cách giữa các dòng liên tiếp
        lon1 = grp["longitude"].shift(1).values[1:]
        lat1 = grp["latitude"].shift(1).values[1:]
        lon2 = grp["longitude"].values[1:]
        lat2 = grp["latitude"].values[1:]
        dist = haversine_np(lon1, lat1, lon2, lat2)
        diffs_fuel.extend(fuel_diff.values)
        distances.extend(dist)
    if len(diffs_fuel) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(distances, diffs_fuel, s=2, alpha=0.4, color="purple")
    ax.set_xlabel("GPS Distance (m)")
    ax.set_ylabel("Fuel Change (lít)")
    ax.set_title("Fuel Change vs GPS Distance")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    save_fig(fig, out_path, logger)


# ---------- Các hàm cũ giữ lại (đã có) ----------
def plot_timeseries_by_car(df, column, out_path, car_col="car_id", timestamp_col="timestamp"):
    if column not in df.columns:
        return
    car_ids = df[car_col].unique()
    n_cars = len(car_ids)
    colors = get_car_colors(car_ids)
    fig, axes = plt.subplots(nrows=n_cars, figsize=(12, 2 * n_cars), sharex=True)
    if n_cars == 1:
        axes = [axes]
    for ax, car in zip(axes, car_ids):
        car_df = df[df[car_col] == car]
        ax.plot(car_df[timestamp_col], car_df[column], linewidth=0.8, color=colors[car])
        ax.set_title(f"{car}")
        ax.set_ylabel(column)
    axes[-1].set_xlabel("Thời gian")
    fig.suptitle(f"{column} theo thời gian - từng xe", y=1.02)
    save_fig(fig, out_path, logger)


def plot_histograms_by_car(df, column, out_path, car_col="car_id"):
    if column not in df.columns:
        return
    car_ids = df[car_col].unique()
    colors = get_car_colors(car_ids)
    fig, ax = plt.subplots(figsize=(8, 5))
    for car in car_ids:
        car_data = df[df[car_col] == car][column].dropna()
        ax.hist(car_data, bins=40, alpha=0.5, label=str(car), color=colors[car])
    ax.set_title(f"Histogram {column} - so sánh các xe")
    ax.set_xlabel(column)
    ax.legend()
    save_fig(fig, out_path, logger)


def plot_boxplots_by_car(df, column, out_path, car_col="car_id"):
    if column not in df.columns:
        return
    car_ids = df[car_col].unique()
    data = [df[df[car_col] == car][column].dropna() for car in car_ids]
    fig, ax = plt.subplots(figsize=(max(6, len(car_ids)*1.2), 4))
    bp = ax.boxplot(data, tick_labels=list(car_ids), patch_artist=True)
    for patch, car in zip(bp['boxes'], car_ids):
        patch.set_facecolor(plt.cm.tab10(list(car_ids).index(car) % 10))
    ax.set_title(f"Boxplot {column} theo xe")
    ax.set_ylabel(column)
    save_fig(fig, out_path, logger)


def plot_correlation_heatmap(df, columns, out_path):
    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(6, len(cols)), max(5, len(cols)*0.8)))
    im = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="left")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="black" if abs(corr.iloc[i, j]) < 0.7 else "white", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Ma trận tương quan (toàn bộ dữ liệu)")
    save_fig(fig, out_path, logger)


def plot_scatter_by_car(df, x_col, y_col, out_path, car_col="car_id"):
    if x_col not in df.columns or y_col not in df.columns:
        return
    car_ids = df[car_col].unique()
    colors = get_car_colors(car_ids)
    fig, ax = plt.subplots(figsize=(8, 6))
    for car in car_ids:
        car_df = df[df[car_col] == car]
        ax.scatter(car_df[x_col], car_df[y_col], s=3, alpha=0.5, color=colors[car], label=str(car))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col} - màu theo xe")
    ax.legend(markerscale=2)
    save_fig(fig, out_path, logger)


def plot_sampling_interval_boxplots(df, out_path, car_col="car_id", timestamp_col="timestamp"):
    if timestamp_col not in df.columns:
        return
    car_ids = df[car_col].unique()
    diffs_list, labels = [], []
    for car in car_ids:
        ts = df[df[car_col] == car][timestamp_col].dropna()
        ts = pd.to_datetime(ts)
        diffs = ts.diff().dt.total_seconds().dropna()
        if len(diffs) > 0:
            diffs_list.append(diffs)
            labels.append(str(car))
    if not diffs_list:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(labels)*1.2), 4))
    ax.boxplot(diffs_list, tick_labels=labels)
    ax.set_title("Khoảng cách lấy mẫu (giây) theo xe")
    ax.set_ylabel("Giây")
    save_fig(fig, out_path, logger)


def plot_missing_heatmap(df, out_path, cols=None, car_col="car_id"):
    if cols is None:
        cols = ["fuel", "speed", "latitude", "longitude"]
    missing = df[cols].isnull()
    if missing.empty:
        return
    car_ids = df[car_col].unique()
    fig, axes = plt.subplots(len(car_ids), 1, figsize=(12, 0.5 * len(car_ids) + 1), sharex=True)
    if len(car_ids) == 1:
        axes = [axes]
    for ax, car in zip(axes, car_ids):
        mask = df[car_col] == car
        missing_car = missing.loc[mask].T
        ax.pcolormesh(missing_car, cmap="binary", edgecolors="none")
        ax.set_yticks(np.arange(len(cols)) + 0.5)
        ax.set_yticklabels(cols)
        ax.set_ylabel(car)
        if ax == axes[-1]:
            ax.set_xlabel("Chỉ số dòng")
    plt.suptitle("Missing Value Heatmap theo xe (đen = missing)")
    save_fig(fig, out_path, logger)


# ---------- Tạo tất cả biểu đồ (cập nhật) ----------
def generate_all_plots(df: pd.DataFrame, figures_dir: Path) -> Dict[str, Path]:
    ensure_dir(figures_dir)
    cfg = EDAConfig()

    # Định nghĩa thứ tự biểu đồ: từ tổng quan đến chi tiết
    plot_map = {
        # 1. Tổng quan dữ liệu
        "01_record_count_by_car.png": lambda: plot_record_count_by_car(df, figures_dir / "01_record_count_by_car.png"),
        "02_missing_bar.png": lambda: plot_missing_bar(df, figures_dir / "02_missing_bar.png"),
        "03_missing_heatmap.png": lambda: plot_missing_heatmap(df, figures_dir / "03_missing_heatmap.png"),
        "04_duplicate_ratio.png": lambda: plot_duplicate_ratio(df, figures_dir / "04_duplicate_ratio.png"),

        # 2. Phân phối đơn biến
        "05_fuel_distribution.png": lambda: plot_fuel_distribution(df, figures_dir / "05_fuel_distribution.png"),
        "06_speed_distribution.png": lambda: plot_speed_distribution(df, figures_dir / "06_speed_distribution.png"),
        "07_fuel_histograms_by_car.png": lambda: plot_histograms_by_car(df, "fuel", figures_dir / "07_fuel_histograms_by_car.png"),
        "08_speed_histograms_by_car.png": lambda: plot_histograms_by_car(df, "speed", figures_dir / "08_speed_histograms_by_car.png"),

        # 3. Timeseries
        "09_fuel_timeseries.png": lambda: plot_timeseries_by_car(df, "fuel", figures_dir / "09_fuel_timeseries.png"),
        "10_speed_timeseries.png": lambda: plot_timeseries_by_car(df, "speed", figures_dir / "10_speed_timeseries.png"),
        "11_speed_timeseries_overlay_fuel.png": lambda: plot_speed_timeseries_overlay_fuel(df, figures_dir / "11_speed_timeseries_overlay_fuel.png"),

        # 4. Quan hệ hai biến & hành vi
        "12_scatter_fuel_vs_speed.png": lambda: plot_scatter_by_car(df, "fuel", "speed", figures_dir / "12_scatter_fuel_vs_speed.png"),
        "13_fuel_diff_distribution.png": lambda: plot_fuel_diff_distribution(df, figures_dir / "13_fuel_diff_distribution.png"),
        "14_fuel_change_vs_distance.png": lambda: plot_fuel_change_vs_distance(df, figures_dir / "14_fuel_change_vs_distance.png"),

        # 5. GPS & chuyển động
        "15_gps_trajectory_per_car.png": lambda: plot_gps_trajectory_per_car(df, figures_dir / "15_gps_trajectory_per_car.png"),
        "16_gps_speed_map.png": lambda: plot_gps_speed_map(df, figures_dir / "16_gps_speed_map.png"),  # giữ lại
        "17_gps_drift_histogram.png": lambda: plot_gps_drift_histogram(df, figures_dir / "17_gps_drift_histogram.png"),

        # 6. Thời gian lấy mẫu
        "18_sampling_interval_boxplots.png": lambda: plot_sampling_interval_boxplots(df, figures_dir / "18_sampling_interval_boxplots.png"),
        "19_time_interval_histogram.png": lambda: plot_time_interval_histogram(df, figures_dir / "19_time_interval_histogram.png"),

        # 7. Tổng hợp & ngoại lai
        "20_correlation_heatmap.png": lambda: plot_correlation_heatmap(df, cfg.HEATMAP_COLS, figures_dir / "20_correlation_heatmap.png"),
        "21_boxplots_fuel.png": lambda: plot_boxplots_by_car(df, "fuel", figures_dir / "21_boxplots_fuel.png"),
        "22_boxplots_speed.png": lambda: plot_boxplots_by_car(df, "speed", figures_dir / "22_boxplots_speed.png"),
        "23_outlier_summary.png": lambda: plot_outlier_summary_by_car(df, cfg.SIGNAL_COLUMNS[:2], figures_dir / "23_outlier_summary.png"),
        "24_two_day_detail.png": lambda: plot_two_day_detail(df, figures_dir / "24_two_day_detail.png"),
    }

    generated = {}
    for fname, func in plot_map.items():
        logger.info("Đang tạo %s ...", fname)
        try:
            func()
            generated[fname] = figures_dir / fname
        except Exception as e:
            logger.error("Lỗi khi tạo %s: %s", fname, e)

    return generated


# Hàm phụ trợ giữ lại
def plot_gps_speed_map(df, out_path):
    if not {"latitude", "longitude", "speed"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["longitude"], df["latitude"], c=df["speed"], cmap="RdYlGn", s=2, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Speed (km/h)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GPS Path – màu theo tốc độ (tất cả xe)")
    ax.set_aspect("equal", adjustable="datalim")
    save_fig(fig, out_path, logger)


def plot_outlier_summary_by_car(df, columns, out_path, car_col="car_id"):
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return
    car_ids = df[car_col].unique()
    results = {col: [] for col in cols}
    for car in car_ids:
        grp = df[df[car_col] == car]
        for col in cols:
            s = grp[col].dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            pct = ((s < lower) | (s > upper)).mean() * 100 if IQR != 0 else 0.0
            results[col].append(pct)
    n_cols = len(cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.bar(car_ids, results[col], color=plt.cm.tab10(np.arange(len(car_ids)) % 10))
        ax.set_title(f"{col}")
        ax.set_ylabel("% outlier")
        ax.set_ylim(0, max(results[col]) * 1.2 if max(results[col]) > 0 else 1)
    plt.suptitle("Tỉ lệ outlier (IQR) theo xe")
    save_fig(fig, out_path, logger)

# ---------- NEW: 2‑day detail for one car ----------
def plot_two_day_detail(df: pd.DataFrame, out_path: Path,
                        car_col: str = "car_id",
                        timestamp_col: str = "timestamp") -> None:
    """
    Vẽ chi tiết fuel & speed trong 2 ngày liên tiếp của 1 xe.
    Tự động chọn xe có nhiều dữ liệu nhất và cặp ngày liên tiếp có nhiều mẫu nhất.
    """
    if not {"fuel", "speed", timestamp_col}.issubset(df.columns):
        return

    # Chọn xe có nhiều record nhất
    car_counts = df[car_col].value_counts()
    if car_counts.empty:
        return
    selected_car = car_counts.idxmax()
    car_df = df[df[car_col] == selected_car].copy()

    # Tạo cột ngày
    car_df["date"] = car_df[timestamp_col].dt.date
    day_counts = car_df.groupby("date").size()

    # Tìm cặp ngày liên tiếp có tổng record lớn nhất
    sorted_days = sorted(day_counts.index)
    best_pair = None
    best_total = 0
    for i in range(len(sorted_days) - 1):
        d1, d2 = sorted_days[i], sorted_days[i + 1]
        if (d2 - d1).days == 1:
            total = day_counts[d1] + day_counts[d2]
            if total > best_total:
                best_total = total
                best_pair = (d1, d2)

    if best_pair is None:
        # fallback: lấy 2 ngày đầu tiên có dữ liệu (có thể không liên tiếp)
        if len(sorted_days) >= 2:
            best_pair = (sorted_days[0], sorted_days[-1])
        else:
            return

    day1, day2 = best_pair
    mask = (car_df["date"] >= day1) & (car_df["date"] <= day2)
    plot_df = car_df[mask]
    if plot_df.empty:
        return

    # Vẽ dual axis
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(plot_df[timestamp_col], plot_df["fuel"], color="#1f77b4", linewidth=0.8, label="Fuel")
    ax1.set_ylabel("Fuel (lít)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(plot_df[timestamp_col], plot_df["speed"], color="#d62728", linewidth=0.8, alpha=0.8, label="Speed")
    ax2.set_ylabel("Speed (km/h)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title(f"Chi tiết 2 ngày liên tiếp – Xe {selected_car} ({day1} & {day2})")
    # Legend kết hợp
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    save_fig(fig, out_path, logger)