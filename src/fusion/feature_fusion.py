#!/usr/bin/env python3
"""
feature_fusion.py — FuelSentinel-AI Pipeline, Step 5: Feature Fusion (v4.6 – Auto-detect FuelTime/Lat/Lng)

--------------------------------------------------------------------------
SCOPE OF THIS MODULE (read before editing)
--------------------------------------------------------------------------
This module has exactly ONE job: fuse the per-point + per-segment feature
timeline with the rule-based labels into a single, flat dataset that is
the direct input to model training.

Output columns are strictly ordered as:
    car_id, segment_id, timestamp, Address, latitude, longitude,
    fuel, speed, final_label, <22 segment features>

This module explicitly does NOT:
    - Compute any new feature (all 22 segment features are carried through
      unchanged from `segment_feature_timeline.csv`).
    - Assign, re-derive, or alter any label. `final_label` is copied
      verbatim (after resolving column aliases) from the upstream labeling step.
    - Smooth, interpolate, normalize, standardize, scale, encode, run PCA,
      do feature selection, apply any rule, or run any Machine Learning.
    - Perform an approximate/nearest-time merge — the join is an EXACT
      match on (car_id, segment_id, timestamp).

If an "Address" column does not exist in the input data, it is added as an
empty column (NaN) to preserve the required schema. Columns not listed in
the final schema (e.g., confidence, segment_quality, note, behavior_state)
are dropped from the output dataset but may appear in the report.

Outputs
-------
data/processed/fusion/fusion_dataset.csv   final point+segment+label dataset for training
data/processed/fusion/fusion_report.md     QA report (distributions, missing values, duplicates, consistency, feature stats)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — paths
# ---------------------------------------------------------------------------
INPUT_TIMELINE_FEATURES_CSV: Path = Path("data/processed/features/segment_feature_timeline.csv")

# Primary input (actual upstream output from state_recognition.py)
INPUT_LABELED_DATA_CSV: Path = Path("data/processed/labels/timeline_labeled.csv")
# Fallback (older naming convention)
INPUT_LABELED_DATA_FALLBACK_CSV: Path = Path("data/processed/labels/labeled_segment_data.csv")

OUTPUT_DIR: Path = Path("data/processed/fusion")
OUTPUT_DATASET_CSV: Path = OUTPUT_DIR / "fusion_dataset.csv"
REPORT_MD: Path = OUTPUT_DIR / "fusion_report.md"

# ---------------------------------------------------------------------------
# Configuration — schema
# ---------------------------------------------------------------------------
MERGE_KEYS: List[str] = ["car_id", "segment_id", "timestamp"]

# The 22 segment features, carried through unchanged.
FEATURE_COLUMNS: List[str] = [
    "duration_min", "sample_count", "movement_state",
    "fuel_start", "fuel_end", "fuel_change", "fuel_mean", "fuel_std",
    "fuel_slope", "trend_r2", "trend_rmse",
    "max_drop", "max_rise", "drop_count", "rise_count",
    "fuel_range", "fuel_mad", "oscillation_count",
    "total_distance",
    "speed_mean", "speed_max", "speed_zero_ratio",
]

# Column-name aliases for the label field we need (final_label).
LABELED_DATA_COLUMN_ALIASES: Dict[str, List[str]] = {
    "final_label": ["final_label", "final_state"],
}

# Additional columns we extract from labeled data for reporting (but drop from final output)
REPORT_ONLY_COLS: List[str] = ["confidence", "segment_quality", "rule_state", "behavior_state"]

# The exact ordered column list for the final dataset (training-ready)
FINAL_DATASET_COLUMNS: List[str] = [
    "car_id", "segment_id", "timestamp", "Address",
    "latitude", "longitude", "fuel", "speed", "final_label"
] + FEATURE_COLUMNS

EXPECTED_FINAL_LABELS: List[str] = ["Idle", "Driving", "Refuel", "Theft"]

# Small, extensible default rule set for the Consistency Check (report only)
RULE_FINAL_CONTRADICTIONS: List[Tuple[str, str]] = [
    ("increas", "Theft"),
    ("rise", "Theft"),
    ("decreas", "Refuel"),
    ("drop", "Refuel"),
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("feature_fusion")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def _read_csv_preserving_literal_null(path: Path) -> pd.DataFrame:
    """Read a CSV without pandas' default NA-string inference, so a literal
    "NULL" value is kept as the text "NULL" rather than silently becoming NaN."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _resolve_column_alias(df: pd.DataFrame, canonical: str, aliases: List[str], source_path: Path) -> pd.DataFrame:
    for alias in aliases:
        if alias in df.columns:
            if alias != canonical:
                logger.warning("'%s' không có trong %s — dùng cột '%s' làm '%s'.",
                                canonical, source_path, alias, canonical)
                df = df.rename(columns={alias: canonical})
            return df
    logger.error("Không tìm thấy cột nào trong %s cho '%s' (đã thử: %s)", source_path, canonical, aliases)
    sys.exit(1)


def _standardize_timestamp_column(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Ensure a 'timestamp' column exists by detecting common datetime column names."""
    if "timestamp" in df.columns:
        return df
    datetime_candidates = ["FuelTime", "timestamp", "Timestamp", "time", "DateTime", "datetime", "t"]
    for col in datetime_candidates:
        if col in df.columns:
            logger.info("Phát hiện cột thời gian '%s' trong %s -> đổi thành 'timestamp'", col, source_path)
            return df.rename(columns={col: "timestamp"})
    logger.error("Không tìm thấy cột thời gian nào trong %s (đã thử: %s)", source_path, datetime_candidates)
    sys.exit(1)


def load_timeline_features(path: Path) -> pd.DataFrame:
    logger.info("Đọc segment feature timeline từ %s", path)
    if not path.exists():
        logger.error("Không tìm thấy file: %s", path)
        sys.exit(1)
    df = _read_csv_preserving_literal_null(path)

    # Ensure timestamp is parsed
    df = _standardize_timestamp_column(df, path)

    missing = set(MERGE_KEYS + FEATURE_COLUMNS) - set(df.columns)
    if missing:
        logger.error("Thiếu cột bắt buộc trong %s: %s", path, missing)
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_labeled_data(primary_path: Path, fallback_path: Path) -> pd.DataFrame:
    if primary_path.exists():
        path = primary_path
    elif fallback_path.exists():
        logger.warning("Không tìm thấy %s — dùng file: %s", primary_path, fallback_path)
        path = fallback_path
    else:
        logger.error("Không tìm thấy file nhãn ở %s hoặc %s", primary_path, fallback_path)
        sys.exit(1)

    logger.info("Đọc labeled data từ %s", path)
    df = _read_csv_preserving_literal_null(path)

    # Standardize timestamp column name (FuelTime -> timestamp)
    df = _standardize_timestamp_column(df, path)

    missing_keys = set(MERGE_KEYS) - set(df.columns)
    if missing_keys:
        logger.error("Thiếu merge key trong %s: %s", path, missing_keys)
        sys.exit(1)

    # Resolve final_label column (may be named final_state)
    for canonical, aliases in LABELED_DATA_COLUMN_ALIASES.items():
        df = _resolve_column_alias(df, canonical, aliases, path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Keep only necessary columns: keys + final_label + optional report columns
    keep_cols = MERGE_KEYS + ["final_label"] + [c for c in REPORT_ONLY_COLS if c in df.columns]
    # Ensure uniqueness of columns (if some report cols overlap with keys)
    keep_cols = list(dict.fromkeys(keep_cols))
    return df[keep_cols].copy()


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    timeline_df = load_timeline_features(INPUT_TIMELINE_FEATURES_CSV)
    labeled_df = load_labeled_data(INPUT_LABELED_DATA_CSV, INPUT_LABELED_DATA_FALLBACK_CSV)
    return timeline_df, labeled_df


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_tables(timeline_df: pd.DataFrame, labeled_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Exact left join on (car_id, segment_id, timestamp)."""
    duplicate_key_count = int(labeled_df.duplicated(subset=MERGE_KEYS, keep=False).sum())
    if duplicate_key_count > 0:
        logger.warning("%d dòng trong labeled data có (car_id, segment_id, timestamp) trùng lặp — "
                        "có thể làm nhân bản dòng sau merge.", duplicate_key_count)

    rows_before = len(timeline_df)
    fused = timeline_df.merge(labeled_df, on=MERGE_KEYS, how="left", indicator=True)
    rows_after = len(fused)

    unmatched_count = int((fused["_merge"] == "left_only").sum())
    fused = fused.drop(columns=["_merge"])

    integrity = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "row_count_preserved": rows_before == rows_after,
        "unmatched_rows": unmatched_count,
        "duplicate_label_keys": duplicate_key_count,
    }
    if not integrity["row_count_preserved"]:
        logger.warning("Số dòng trước merge (%d) khác số dòng sau merge (%d) — kiểm tra duplicate key.",
                        rows_before, rows_after)
    if unmatched_count > 0:
        logger.warning("%d dòng không khớp được nhãn (final_label sẽ là NaN).", unmatched_count)

    return fused, integrity


# ---------------------------------------------------------------------------
# Final column ordering and cleaning
# ---------------------------------------------------------------------------
def prepare_final_dataset(fused: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns, add missing Address column if needed, drop extra columns."""
    # Add Address column if not present (should be present in timeline from upstream)
    if "Address" not in fused.columns:
        fused["Address"] = np.nan
        logger.info("Cột 'Address' không có trong dữ liệu, tạo cột rỗng (NaN).")

    # Ensure all required columns exist
    missing_cols = set(FINAL_DATASET_COLUMNS) - set(fused.columns)
    if missing_cols:
        logger.error("Thiếu cột sau merge: %s", missing_cols)
        sys.exit(1)

    # Select and order exactly as required
    final_df = fused[FINAL_DATASET_COLUMNS].copy()
    return final_df


# ---------------------------------------------------------------------------
# Quality Check
# ---------------------------------------------------------------------------
def missing_value_report(df: pd.DataFrame) -> pd.Series:
    check_cols = [c for c in FINAL_DATASET_COLUMNS if c in df.columns]
    return df[check_cols].isna().sum()


def duplicate_check(df: pd.DataFrame, keys: List[str]) -> Tuple[int, pd.DataFrame]:
    dup_mask = df.duplicated(subset=keys, keep=False)
    return int(dup_mask.sum()), df.loc[dup_mask, keys]


def _movement_final_contradictions(df: pd.DataFrame) -> pd.DataFrame:
    if "movement_state" not in df.columns or "final_label" not in df.columns:
        return pd.DataFrame()
    moving_but_idle = (df["movement_state"] == "Moving") & (df["final_label"] == "Idle")
    stationary_but_driving = (df["movement_state"] == "Stationary") & (df["final_label"] == "Driving")
    mask = moving_but_idle | stationary_but_driving
    cols = MERGE_KEYS + ["movement_state", "final_label"]
    if "rule_state" in df.columns:
        cols.append("rule_state")
    return df.loc[mask, cols]


def _rule_final_contradictions(df: pd.DataFrame) -> pd.DataFrame:
    if "rule_state" not in df.columns or "final_label" not in df.columns:
        return pd.DataFrame()
    rule_lower = df["rule_state"].astype(str).str.lower()
    mask = pd.Series(False, index=df.index)
    for keyword, contradictory_final in RULE_FINAL_CONTRADICTIONS:
        mask |= rule_lower.str.contains(keyword, na=False) & (df["final_label"] == contradictory_final)
    return df.loc[mask, MERGE_KEYS + ["rule_state", "final_label"]]


def consistency_check(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "movement_vs_final": _movement_final_contradictions(df),
        "rule_vs_final": _rule_final_contradictions(df),
    }


def feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features = [c for c in FEATURE_COLUMNS if c != "movement_state"]
    return df[numeric_features].agg(["mean", "std", "min", "max"]).T


def unexpected_final_labels(df: pd.DataFrame) -> pd.Series:
    if "final_label" not in df.columns:
        return pd.Series(dtype=int)
    counts = df["final_label"].value_counts()
    return counts[~counts.index.isin(EXPECTED_FINAL_LABELS)]


def quality_check(df: pd.DataFrame) -> Dict[str, object]:
    dup_count, dup_rows = duplicate_check(df, MERGE_KEYS)
    return {
        "missing": missing_value_report(df),
        "duplicate_count": dup_count,
        "duplicate_rows": dup_rows,
        "consistency": consistency_check(df),
        "feature_stats": feature_statistics(df),
        "unexpected_final_labels": unexpected_final_labels(df),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(df_full: pd.DataFrame, integrity: Dict[str, object],
                    qc: Dict[str, object], report_path: Path) -> None:
    n_rows = len(df_full)
    n_cars = df_full["car_id"].nunique()
    n_segments = df_full.groupby(["car_id", "segment_id"]).ngroups

    md_lines = [
        "# Fusion Report",
        "",
        "## Dataset",
        f"- Rows: {n_rows}",
        f"- Cars: {n_cars}",
        f"- Segments: {n_segments}",
        "",
        "## Merge Integrity",
        f"- Rows before merge (segment_feature_timeline.csv): {integrity['rows_before']}",
        f"- Rows after merge (fusion_dataset.csv): {integrity['rows_after']}",
        f"- Row count preserved: {'✅ Yes' if integrity['row_count_preserved'] else '⚠️ NO — investigate duplicate keys'}",
        f"- Unmatched rows (no label found): {integrity['unmatched_rows']}",
        f"- Duplicate (car_id, segment_id, timestamp) keys in labeled data: {integrity['duplicate_label_keys']}",
        "",
        "## Label Distribution (final_label)",
        "",
        "| Final Label | Count |",
        "|-------------|------:|",
    ]
    final_counts = df_full["final_label"].value_counts()
    for label in EXPECTED_FINAL_LABELS:
        md_lines.append(f"| {label} | {int(final_counts.get(label, 0))} |")

    unexpected = qc["unexpected_final_labels"]
    if not unexpected.empty:
        md_lines += ["", "⚠️ Các giá trị final_label ngoài 4 lớp kỳ vọng:", ""]
        md_lines += ["| Final Label | Count |", "|-------------|------:|"]
        for label, count in unexpected.items():
            md_lines.append(f"| {label} | {int(count)} |")

    if "rule_state" in df_full.columns:
        md_lines += ["", "## Rule Distribution (rule_state)", "",
                     "| Rule State | Count |", "|------------|------:|"]
        for rule, count in df_full["rule_state"].value_counts().items():
            md_lines.append(f"| {rule} | {int(count)} |")

    md_lines += ["", "## Missing Value (in final dataset columns)", "",
                 "| Column | Missing Count |", "|--------|---------------:|"]
    missing = qc["missing"]
    missing_nonzero = missing[missing > 0]
    if missing_nonzero.empty:
        md_lines.append("| (không có cột nào thiếu giá trị) | 0 |")
    else:
        for col, count in missing_nonzero.items():
            md_lines.append(f"| {col} | {int(count)} |")
    if "Address" in missing_nonzero and missing_nonzero["Address"] == n_rows:
        md_lines.append("")
        md_lines.append("Lưu ý: Cột 'Address' toàn bộ là NaN do không có dữ liệu đầu vào, "
                         "đây là thiết kế chủ động để đảm bảo schema đầu ra.")

    md_lines += ["", "## Duplicate Check", ""]
    md_lines.append(f"- Số dòng trùng lặp theo (car_id, segment_id, timestamp): {qc['duplicate_count']}")

    md_lines += ["", "## Consistency Check", ""]
    mv = qc["consistency"]["movement_vs_final"]
    rv = qc["consistency"]["rule_vs_final"]
    md_lines.append(f"- Mâu thuẫn movement_state vs final_label: {len(mv)}")
    md_lines.append(f"- Mâu thuẫn rule_state vs final_label: {len(rv)}")
    if len(mv) > 0:
        md_lines += ["", "### Mẫu movement_state vs final_label (tối đa 20)", "",
                     "| car_id | segment_id | timestamp | movement_state | final_label |",
                     "|--------|------------|-----------|-----------------|-------------|"]
        for _, row in mv.head(20).iterrows():
            md_lines.append(f"| {row['car_id']} | {row['segment_id']} | {row['timestamp']} | "
                            f"{row['movement_state']} | {row['final_label']} |")
    if len(rv) > 0:
        md_lines += ["", "### Mẫu rule_state vs final_label (tối đa 20)", "",
                     "| car_id | segment_id | timestamp | rule_state | final_label |",
                     "|--------|------------|-----------|------------|--------------|"]
        for _, row in rv.head(20).iterrows():
            md_lines.append(f"| {row['car_id']} | {row['segment_id']} | {row['timestamp']} | "
                            f"{row['rule_state']} | {row['final_label']} |")

    md_lines += ["", "## Feature Statistics", "",
                 "| Feature | mean | std | min | max |",
                 "|---------|-----:|----:|----:|----:|"]
    stats = qc["feature_stats"]
    for feat, row in stats.iterrows():
        md_lines.append(f"| {feat} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |")

    md_lines += [
        "",
        "## Ghi chú",
        "- File đầu ra `fusion_dataset.csv` chỉ chứa các cột theo đúng thứ tự yêu cầu. "
        "Các cột phụ trợ (confidence, segment_quality, note, behavior_state, ...) đã bị loại bỏ khỏi file training.",
        "- `final_label` có thể chứa chuỗi 'NULL' nếu upstream không xác định được trạng thái.",
        "- Cột 'Address' được giữ nguyên từ dữ liệu; nếu không có sẽ là NaN.",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Báo cáo đã được lưu tại %s", report_path)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, na_rep='')
    logger.info("Đã lưu %s (%d dòng, %d cột)", path, len(df), len(df.columns))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    timeline_df, labeled_df = load_data()
    logger.info("segment_feature_timeline: %d dòng | labeled data: %d dòng", len(timeline_df), len(labeled_df))

    fused_full, integrity = merge_tables(timeline_df, labeled_df)

    qc = quality_check(fused_full)

    fusion_dataset = prepare_final_dataset(fused_full)

    save_dataset(fusion_dataset, OUTPUT_DATASET_CSV)
    generate_report(fused_full, integrity, qc, REPORT_MD)


if __name__ == "__main__":
    main()