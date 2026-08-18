#!/usr/bin/env python3
"""
feature_extraction.py — FuelSentinel-AI Pipeline, Step 3: Feature Extraction

--------------------------------------------------------------------------
SCOPE OF THIS MODULE (read before editing)
--------------------------------------------------------------------------
This module has exactly ONE job: read `segment_data.csv` (raw, cleaned
telemetry rows already carrying a `segment_id` from the upstream
Behavior-based Segmentation step) and compute a fixed, small, explainable
set of 22 descriptive features PER SEGMENT.

This module explicitly does NOT:
    - Re-segment the timeline or touch `segment_id` boundaries in any way.
    - Modify Fuel / Speed / GPS values (no smoothing, no cleaning, no
      interpolation, no outlier correction).
    - Use a sliding window — every feature is computed over the WHOLE
      segment, once.
    - Detect or label Theft / Refuel / Fuel Loss / Idle or any other
      operational state. `movement_state` here is a purely descriptive
      Moving/Stationary summary of the segment's own speed samples (the
      same physical notion used upstream), not a new classification.
    - Decide what any feature *means* for anomaly detection — that is the
      job of the downstream State Recognition / ML Classification steps.

--------------------------------------------------------------------------
THE 22 FEATURES (grouped exactly as specified, no more, no fewer)
--------------------------------------------------------------------------
 1. Segment Information : duration_min, sample_count, movement_state
 2. Fuel Level           : fuel_start, fuel_end, fuel_change, fuel_mean, fuel_std
 3. Fuel Trend            : fuel_slope, trend_r2, trend_rmse   (Linear Regression, whole segment)
 4. Fuel Dynamics         : max_drop, max_rise, drop_count, rise_count (from fuel diff series)
 5. Fuel Stability        : fuel_range, fuel_mad, oscillation_count
 6. GPS                   : total_distance (Haversine, whole segment)
 7. Speed                 : speed_mean, speed_max, speed_zero_ratio

Every feature is computed once per segment, over the full set of points
belonging to that segment — never per-point, never with a rolling window.

Outputs
-------
data/processed/features/segment_features.csv          1 row per segment (22 features)
data/processed/features/segment_feature_timeline.csv   segment_data.csv + segment features broadcast per row (for viz/debug only — NOT a training file)
data/processed/features/feature_report.md              dataset-wide feature QA report
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_SEGMENT_CSV: Path = Path("data/processed/segment/segment_data.csv")

OUTPUT_DIR: Path = Path("data/processed/features")
OUTPUT_FEATURES_CSV: Path = OUTPUT_DIR / "segment_features.csv"
OUTPUT_TIMELINE_CSV: Path = OUTPUT_DIR / "segment_feature_timeline.csv"
REPORT_MD: Path = OUTPUT_DIR / "feature_report.md"

# Column names as produced by the upstream cleaning/segmentation steps.
RENAME_MAP: Dict[str, str] = {
    "FuelTime": "timestamp", "FuelLevel": "fuel",
    "Lat": "latitude", "Lng": "longitude", "Speed": "speed",
}
REQUIRED_COLUMNS = {"timestamp", "fuel", "latitude", "longitude", "speed", "car_id", "segment_id"}

# Descriptive-only threshold reused from segmentation purely to summarise ad
# segment's own speed samples as Moving/Stationary. This does NOT feed back 
# into segmentation and does NOT create/alter any boundary.
SPEED_STATIONARY_KMH = 3.0

# QA thresholds used only for reporting (feature_report.md), never for
# filtering/dropping data.
CONSTANT_STD_THRESHOLD = 1e-9
HIGH_CORRELATION_THRESHOLD = 0.95
SHORT_SEGMENT_SAMPLE_THRESHOLD = 5

# The exact, fixed feature set — order defines the column order of
# segment_features.csv. Nothing is added beyond this list.
FEATURE_COLUMNS: List[str] = [
    # 1. Segment Information
    "duration_min", "sample_count", "movement_state",
    # 2. Fuel Level
    "fuel_start", "fuel_end", "fuel_change", "fuel_mean", "fuel_std",
    # 3. Fuel Trend
    "fuel_slope", "trend_r2", "trend_rmse",
    # 4. Fuel Dynamics
    "max_drop", "max_rise", "drop_count", "rise_count",
    # 5. Fuel Stability
    "fuel_range", "fuel_mad", "oscillation_count",
    # 6. GPS
    "total_distance",
    # 7. Speed
    "speed_mean", "speed_max", "speed_zero_ratio",
]
NUMERIC_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c != "movement_state"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("feature_extraction")


# ---------------------------------------------------------------------------
# Geometry helper (read-only: computes a feature, never mutates GPS data)
# ---------------------------------------------------------------------------
def _haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371000.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Per-segment feature blocks
# Each function takes ONE segment's rows (already sorted by timestamp) and
# returns a small dict — computed exactly once over the whole segment.
# ---------------------------------------------------------------------------
def _segment_info_features(g: pd.DataFrame) -> Dict[str, object]:
    duration_min = (g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]).total_seconds() / 60.0
    sample_count = len(g)

    # Purely descriptive summary of this segment's own speed samples —
    # majority rule, no window, no re-segmentation.
    stationary_ratio = float((g["speed"] < SPEED_STATIONARY_KMH).mean())
    movement_state = "Stationary" if stationary_ratio >= 0.5 else "Moving"

    return {
        "duration_min": round(float(duration_min), 2),
        "sample_count": int(sample_count),
        "movement_state": movement_state,
    }


def _fuel_level_features(g: pd.DataFrame) -> Dict[str, float]:
    fuel = g["fuel"]
    return {
        "fuel_start": round(float(fuel.iloc[0]), 3),
        "fuel_end": round(float(fuel.iloc[-1]), 3),
        "fuel_change": round(float(fuel.iloc[-1] - fuel.iloc[0]), 3),
        "fuel_mean": round(float(fuel.mean()), 3),
        # ddof=0 so a 1-point segment yields 0.0 instead of NaN; genuinely
        # under-sampled segments are still flagged separately in the QA report.
        "fuel_std": round(float(fuel.std(ddof=0)), 3) if len(fuel) > 0 else np.nan,
    }


def _fuel_trend_features(g: pd.DataFrame) -> Dict[str, float]:
    """Linear Regression of Fuel vs elapsed time, fit ONCE over the whole
    segment (no window). Requires >=2 points with nonzero time span;
    otherwise the trend is genuinely undefined and reported as NaN rather
    than guessed."""
    t_sec = (g["timestamp"] - g["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    y = g["fuel"].to_numpy(dtype=float)

    if len(g) < 2 or np.ptp(t_sec) <= 0:
        return {"fuel_slope": np.nan, "trend_r2": np.nan, "trend_rmse": np.nan}

    X = t_sec.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    slope_per_sec = float(model.coef_[0])
    slope_per_hour = slope_per_sec * 3600.0  # expressed as Δfuel/hour for interpretability

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 1e-12:
        # Fuel is (numerically) constant across the segment — R^2 is
        # mathematically undefined, not "perfect" or "zero".
        r2 = np.nan
    else:
        ss_res = float(np.sum((y - y_pred) ** 2))
        r2 = 1.0 - ss_res / ss_tot

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    return {
        "fuel_slope": round(slope_per_hour, 5),
        "trend_r2": round(r2, 5) if not np.isnan(r2) else np.nan,
        "trend_rmse": round(rmse, 5),
    }


def _fuel_dynamics_features(g: pd.DataFrame) -> Dict[str, float]:
    diff = g["fuel"].diff().dropna()
    if diff.empty:
        return {"max_drop": np.nan, "max_rise": np.nan, "drop_count": 0, "rise_count": 0}

    min_diff = float(diff.min())
    max_diff = float(diff.max())
    return {
        # Reported as positive magnitudes for readability: "the largest
        # single-step drop/rise seen in this segment".
        "max_drop": round(-min_diff, 3) if min_diff < 0 else 0.0,
        "max_rise": round(max_diff, 3) if max_diff > 0 else 0.0,
        "drop_count": int((diff < 0).sum()),
        "rise_count": int((diff > 0).sum()),
    }


def _fuel_stability_features(g: pd.DataFrame) -> Dict[str, float]:
    fuel = g["fuel"]
    fuel_range = float(fuel.max() - fuel.min())
    fuel_mad = float(scipy_stats.median_abs_deviation(fuel.to_numpy(), scale=1.0)) if len(fuel) > 0 else np.nan

    diff = g["fuel"].diff().dropna().to_numpy()
    signs = np.sign(diff)
    nonzero_signs = signs[signs != 0]
    oscillation_count = int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1])) if len(nonzero_signs) > 1 else 0

    return {
        "fuel_range": round(fuel_range, 3),
        "fuel_mad": round(fuel_mad, 3) if not np.isnan(fuel_mad) else np.nan,
        "oscillation_count": oscillation_count,
    }


def _gps_features(g: pd.DataFrame) -> Dict[str, float]:
    lat, lon = g["latitude"].to_numpy(), g["longitude"].to_numpy()
    if len(g) < 2:
        return {"total_distance": 0.0}
    step_dist = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return {"total_distance": round(float(np.nansum(step_dist)), 2)}


def _speed_features(g: pd.DataFrame) -> Dict[str, float]:
    speed = g["speed"]
    return {
        "speed_mean": round(float(speed.mean()), 3),
        "speed_max": round(float(speed.max()), 3),
        "speed_zero_ratio": round(float((speed == 0).mean()), 4),
    }


def extract_segment_features(g: pd.DataFrame) -> Dict[str, object]:
    """Compute all 22 features for a single (car_id, segment_id) group.
    Each block is independent and side-effect free — `g` is only read."""
    g = g.sort_values("timestamp")
    features: Dict[str, object] = {}
    features.update(_segment_info_features(g))
    features.update(_fuel_level_features(g))
    features.update(_fuel_trend_features(g))
    features.update(_fuel_dynamics_features(g))
    features.update(_fuel_stability_features(g))
    features.update(_gps_features(g))
    features.update(_speed_features(g))
    return features


# ---------------------------------------------------------------------------
# Segment-level table construction
# ---------------------------------------------------------------------------
def build_segment_features_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (car_id, segment_id), g in df.groupby(["car_id", "segment_id"], sort=False):
        row = {"car_id": car_id, "segment_id": segment_id}
        row.update(extract_segment_features(g))
        rows.append(row)

    features_df = pd.DataFrame(rows, columns=["car_id", "segment_id"] + FEATURE_COLUMNS)
    features_df = features_df.sort_values(["car_id", "segment_id"]).reset_index(drop=True)
    return features_df


def build_feature_timeline(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """segment_data.csv rows + the segment's own features broadcast onto
    every row of that segment. For visualisation/debugging only — NOT the
    training file (that is `segment_features.csv`)."""
    timeline = df.merge(features_df, on=["car_id", "segment_id"], how="left", suffixes=("", "_feat"))
    return timeline.sort_values(["car_id", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _feature_summary_table(features_df: pd.DataFrame) -> pd.DataFrame:
    return features_df[NUMERIC_FEATURE_COLUMNS].agg(["mean", "std", "min", "max", "median"]).T


def _missing_value_report(features_df: pd.DataFrame) -> pd.DataFrame:
    na_counts = features_df[FEATURE_COLUMNS].isna().sum()
    na_counts = na_counts[na_counts > 0]
    return na_counts.rename("missing_count").to_frame()


def _constant_feature_report(features_df: pd.DataFrame) -> List[str]:
    constant_features = []
    for col in NUMERIC_FEATURE_COLUMNS:
        col_std = features_df[col].std(ddof=0)
        if pd.isna(col_std) or col_std < CONSTANT_STD_THRESHOLD:
            constant_features.append(col)
    return constant_features


def _correlation_report(features_df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    corr = features_df[NUMERIC_FEATURE_COLUMNS].corr()
    high_pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if pd.notna(c) and abs(c) > HIGH_CORRELATION_THRESHOLD:
                high_pairs.append(f"{cols[i]} ~ {cols[j]} (r = {c:.3f})")
    return corr, high_pairs


def _quality_checks(features_df: pd.DataFrame) -> Dict[str, object]:
    short_segments = features_df[features_df["sample_count"] < SHORT_SEGMENT_SAMPLE_THRESHOLD]
    single_point_segments = features_df[features_df["sample_count"] == 1]

    numeric = features_df[NUMERIC_FEATURE_COLUMNS]
    inf_mask = numeric.apply(lambda s: np.isinf(s.to_numpy(dtype=float))).any(axis=1)
    inf_segments = features_df[inf_mask]

    nan_per_segment = numeric.isna().sum(axis=1)
    segments_with_nan = features_df[nan_per_segment > 0]

    return {
        "short_segments": short_segments,
        "single_point_segments": single_point_segments,
        "inf_segments": inf_segments,
        "segments_with_nan": segments_with_nan,
    }


def generate_report(df: pd.DataFrame, features_df: pd.DataFrame, report_path: Path) -> None:
    n_cars = features_df["car_id"].nunique()
    n_segments = len(features_df)
    n_features = len(FEATURE_COLUMNS)

    md_lines = [
        "# Feature Extraction Report",
        "",
        "## Dataset",
        f"- Tổng số xe: {n_cars}",
        f"- Tổng số segment: {n_segments}",
        f"- Tổng số feature: {n_features}",
        "",
        "## Feature Summary",
        "",
        "| Feature | mean | std | min | max | median |",
        "|---------|------|-----|-----|-----|--------|",
    ]
    summary = _feature_summary_table(features_df)
    for feat, row in summary.iterrows():
        md_lines.append(
            f"| {feat} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | "
            f"{row['max']:.4f} | {row['median']:.4f} |"
        )

    md_lines += ["", "## Missing Value Check", ""]
    na_report = _missing_value_report(features_df)
    if na_report.empty:
        md_lines.append("- Không có feature nào chứa giá trị NA.")
    else:
        md_lines += ["| Feature | Số lượng NA |", "|---------|-------------|"]
        for feat, row in na_report.iterrows():
            md_lines.append(f"| {feat} | {int(row['missing_count'])} |")

    md_lines += ["", "## Constant Feature Check", ""]
    constant_features = _constant_feature_report(features_df)
    if constant_features:
        md_lines.append(
            f"- Feature gần như không đổi (std < {CONSTANT_STD_THRESHOLD}): "
            + ", ".join(constant_features)
        )
    else:
        md_lines.append("- Không có feature nào gần như hằng số.")

    md_lines += ["", "## Correlation Matrix", ""]
    corr, high_pairs = _correlation_report(features_df)
    md_lines.append("Ma trận tương quan đầy đủ được lưu trong `segment_features.csv` "
                     "(có thể tính lại bằng `DataFrame.corr()`). Các cặp tương quan cao:")
    md_lines.append("")
    if high_pairs:
        for pair in high_pairs:
            md_lines.append(f"- ⚠️ {pair} — cân nhắc loại bỏ một trong hai ở bước sau.")
    else:
        md_lines.append(f"- Không có cặp feature nào có |correlation| > {HIGH_CORRELATION_THRESHOLD}.")

    md_lines += ["", "## Quality Check", ""]
    qc = _quality_checks(features_df)
    md_lines.append(f"- Segment quá ngắn (< {SHORT_SEGMENT_SAMPLE_THRESHOLD} điểm): {len(qc['short_segments'])}")
    md_lines.append(f"- Segment chỉ có 1 điểm: {len(qc['single_point_segments'])}")
    md_lines.append(f"- Segment có giá trị vô hạn (inf) trong feature: {len(qc['inf_segments'])}")
    md_lines.append(f"- Segment có ít nhất 1 feature NaN: {len(qc['segments_with_nan'])}")

    if len(qc["short_segments"]):
        md_lines += ["", "### Danh sách segment quá ngắn", "",
                     "| car_id | segment_id | sample_count |", "|--------|------------|--------------|"]
        for _, row in qc["short_segments"].iterrows():
            md_lines.append(f"| {row['car_id']} | {row['segment_id']} | {row['sample_count']} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Báo cáo đã được lưu tại %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_segment_data(path: Path) -> pd.DataFrame:
    logger.info("Đọc segment_data từ %s", path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error("Không thể đọc file đầu vào: %s", e)
        sys.exit(1)

    df = df.rename(columns=RENAME_MAP)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.error("Thiếu cột bắt buộc: %s", missing)
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def main() -> None:
    df = load_segment_data(INPUT_SEGMENT_CSV)
    logger.info("Đã đọc %d dòng, %d xe, %d segment (car_id, segment_id)",
                len(df), df["car_id"].nunique(), df.groupby(["car_id", "segment_id"]).ngroups)

    features_df = build_segment_features_table(df)
    timeline_df = build_feature_timeline(df, features_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(OUTPUT_FEATURES_CSV, index=False)
    logger.info("Đã lưu %s (%d segment, %d feature)", OUTPUT_FEATURES_CSV, len(features_df), len(FEATURE_COLUMNS))

    timeline_df.to_csv(OUTPUT_TIMELINE_CSV, index=False)
    logger.info("Đã lưu %s (%d dòng) — chỉ phục vụ trực quan hóa/debug", OUTPUT_TIMELINE_CSV, len(timeline_df))

    generate_report(df, features_df, REPORT_MD)


if __name__ == "__main__":
    main()