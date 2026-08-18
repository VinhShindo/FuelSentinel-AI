#!/usr/bin/env python3
"""
segment_detection.py — FuelSentinel-AI Pipeline, Step 2: Segment Detection
V5: Thêm rule tách "đuôi" biến động/giảm sau đỉnh của segment FuelUp.
- V4: Khắc phục lỗi "nuốt refuel" và "chia vụn do window nhỏ".
- V5 (mới):
    1. split_fuelup_peak_tail(): với mỗi segment Stationary-FuelUp, tìm điểm đỉnh
       (điểm cuối cùng còn "gần đỉnh"). Nếu sau đó có đuôi dữ liệu giảm/biến động
       đủ dài & đủ lớn -> TÁCH thành 1 segment riêng, điểm tách là điểm dữ liệu
       bắt đầu giảm thật sự. Nếu đỉnh nằm ở cuối segment (không có đuôi) hoặc chỉ
       là nhiễu nhỏ quanh đỉnh -> giữ nguyên, không tách.
    2. Thêm 1 pass dọn dẹp phân mảnh sau khi tách (ngưỡng nhỏ hơn), có bảo vệ
       riêng cho segment vừa được tách ra để không bị merge ngược lại.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV: Path = Path("data/processed/clean_data.csv")
OUTPUT_SEGMENT_CSV: Path = Path("data/processed/segment/segment_data.csv")
OUTPUT_SUMMARY_CSV: Path = Path("data/processed/segment/segment_summary.csv")
REPORT_MD: Path = Path("data/processed/segment/segment_report.md")
FIGURES_DIR: Path = Path("data/processed/segment/segments")
SKIP_CAR_NUMBERS = {1}

# --- Movement detection ---
MOVEMENT_GPS_WINDOW = 5
SPEED_STATIONARY_KMH = 3.0
GPS_DRIFT_M = 20.0
BEHAVIOR_SPEED_WEIGHT = 0.6
BEHAVIOR_GPS_WEIGHT = 0.4
BEHAVIOR_SCORE_THRESHOLD = 1.0

# --- Fuel trend (tăng window lên 9 và delta lên 2.0 để chống nhấp nháy) ---
FUEL_ANALYSIS_WINDOW = 9          # số điểm trong cửa sổ trượt (từ 5 lên 9)
FUEL_SLOPE_THRESHOLD = 0.015      # L/phút
FUEL_DELTA_THRESHOLD = 2.0        # L (tăng lên 2.0 để nhiễu nhẹ không kích hoạt State đổi)

# --- Debounce / persistence ---
MIN_STATE_PERSISTENCE = 3

# --- Segment boundaries & merge ---
TIME_GAP_MINUTES = 30
MIN_SEGMENT_POINTS = 15           # Tăng lên 15, refuel ngắn vẫn được giữ nhờ luật merge mới

# --- V5: Tách đuôi biến động/giảm sau đỉnh của segment FuelUp ---
FUELUP_PEAK_EPSILON = 5.0         # L, dung sai để coi 1 điểm là "vẫn còn gần đỉnh"
FUELUP_TAIL_MIN_DROP = 4.0        # L, mức sụt giảm tối thiểu từ đỉnh -> cuối đuôi để coi là giảm thật
FUELUP_TAIL_MIN_LEN = 5           # số điểm tối thiểu của đuôi để được tách thành segment riêng

# --- V5: Dọn dẹp phân mảnh còn sót lại sau khi tách đuôi ---
POST_SPLIT_MIN_POINTS = 5         # ngưỡng nhỏ hơn MIN_SEGMENT_POINTS, chỉ để dọn rác thật vụn
PROTECTED_SPLIT_REASON = "FuelUp Peak Tail Split"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("segment_detection")

# ---------------------------------------------------------------------------
# Vectorised geometry helpers
# ---------------------------------------------------------------------------
def _haversine_km_vec(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _car_number(car_id) -> Optional[int]:
    m = re.search(r"(\d+)\s*$", str(car_id))
    return int(m.group(1)) if m else None


def _gps_radius_m(lat: np.ndarray, lon: np.ndarray) -> float:
    if len(lat) == 0:
        return 0.0
    centroid_lat = float(np.mean(lat))
    centroid_lon = float(np.mean(lon))
    dists = _haversine_km_vec(lat, lon, np.full_like(lat, centroid_lat), np.full_like(lon, centroid_lon)) * 1000.0
    return float(np.max(dists)) if len(dists) else 0.0


# ---------------------------------------------------------------------------
# Point-level derived signals
# ---------------------------------------------------------------------------
def _movement_candidate(speed: pd.Series, gps_window_disp: pd.Series) -> pd.Series:
    """Moving/Stationary logic cũ đã hoạt động tốt."""
    speed_component = (speed / SPEED_STATIONARY_KMH).clip(lower=0)
    gps_component = (gps_window_disp / GPS_DRIFT_M).clip(lower=0).fillna(0.0)
    score = BEHAVIOR_SPEED_WEIGHT * speed_component + BEHAVIOR_GPS_WEIGHT * gps_component
    return pd.Series(np.where(score > BEHAVIOR_SCORE_THRESHOLD, "Moving", "Stationary"), index=speed.index)


def _fuel_trend_candidate(fuel: pd.Series, timestamps: pd.Series) -> pd.Series:
    """
    Chỉ 3 nhãn thô: Stable / Increasing / Decreasing.
    Cửa sổ FUEL_ANALYSIS_WINDOW được tăng lên để làm mịn dữ liệu, ngăn phân mảnh state.
    """
    ts_minutes = (timestamps - timestamps.iloc[0]).dt.total_seconds().div(60)
    n = len(fuel)
    trend = np.full(n, "Stable", dtype=object)

    if n < 2:
        return pd.Series(trend, index=fuel.index)

    fuel_arr = fuel.to_numpy()
    time_arr = ts_minutes.to_numpy()
    half_w = FUEL_ANALYSIS_WINDOW // 2

    for i in range(n):
        start_idx = max(0, i - half_w)
        end_idx = min(n, i + half_w + (FUEL_ANALYSIS_WINDOW % 2))
        if end_idx - start_idx < 2:
            continue

        t_window = time_arr[start_idx:end_idx]
        f_window = fuel_arr[start_idx:end_idx]

        if np.std(t_window) != 0:
            slope = np.polyfit(t_window, f_window, 1)[0]  # L/phút
        else:
            slope = 0.0
        delta = f_window[-1] - f_window[0]  # L

        if slope > FUEL_SLOPE_THRESHOLD and delta > FUEL_DELTA_THRESHOLD:
            trend[i] = "Increasing"
        elif slope < -FUEL_SLOPE_THRESHOLD and delta < -FUEL_DELTA_THRESHOLD:
            trend[i] = "Decreasing"
        else:
            trend[i] = "Stable"

    return pd.Series(trend, index=fuel.index)


def compute_behaviour_signals(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.reset_index(drop=True).copy()
    lat, lon = sub["latitude"].to_numpy(), sub["longitude"].to_numpy()
    step_disp = np.full(len(sub), np.nan)
    if len(sub) > 1:
        step_disp[1:] = _haversine_km_vec(lat[:-1], lon[:-1], lat[1:], lon[1:]) * 1000.0

    sub["gps_step_disp_m"] = step_disp
    sub["gps_window_disp_m"] = (
        pd.Series(step_disp, index=sub.index).rolling(MOVEMENT_GPS_WINDOW, center=True, min_periods=1).sum()
    )

    sub["movement_candidate"] = _movement_candidate(sub["speed"], sub["gps_window_disp_m"])
    sub["fuel_trend_candidate"] = _fuel_trend_candidate(sub["fuel"], sub["timestamp"])
    return sub


# ---------------------------------------------------------------------------
# Gán 1 trong 4 behavioral state
# ---------------------------------------------------------------------------
def _assign_state(movement: str, fuel_trend: str) -> str:
    if movement == "Moving":
        return "Moving"
    if fuel_trend == "Increasing":
        return "Stationary-FuelUp"
    if fuel_trend == "Decreasing":
        return "Stationary-FuelDown"
    return "Stationary-Stable"


def assign_behavior_state(sub: pd.DataFrame) -> pd.DataFrame:
    sub["raw_state"] = [
        _assign_state(m, f) for m, f in zip(sub["movement_candidate"], sub["fuel_trend_candidate"])
    ]
    return sub


# ---------------------------------------------------------------------------
# Debounce / hysteresis
# ---------------------------------------------------------------------------
def _debounce_states(states: pd.Series, min_persistence: int) -> pd.Series:
    n = len(states)
    if n == 0:
        return states.copy()
    vals = states.to_numpy()
    confirmed = np.empty(n, dtype=vals.dtype)
    current = vals[0]
    candidate = None
    count = 0
    confirmed[0] = current
    for i in range(1, n):
        v = vals[i]
        if v == current:
            candidate = None
            count = 0
        else:
            if v == candidate:
                count += 1
            else:
                candidate = v
                count = 1
            if count >= min_persistence:
                current = v
                candidate = None
                count = 0
        confirmed[i] = current
    return pd.Series(confirmed, index=states.index)


def debounce_behaviour(sub: pd.DataFrame) -> pd.DataFrame:
    sub["behavior_state"] = _debounce_states(sub["raw_state"], MIN_STATE_PERSISTENCE)
    return sub


# ---------------------------------------------------------------------------
# Segment boundary detection
# ---------------------------------------------------------------------------
def detect_segments(sub: pd.DataFrame) -> Tuple[List[Tuple[int, int]], List[str]]:
    n = len(sub)
    if n == 0:
        return [], []
    starts = [0]
    reasons = ["Start of car log"]
    ts = sub["timestamp"]
    state = sub["behavior_state"]
    for i in range(1, n):
        dt = (ts.iloc[i] - ts.iloc[i - 1]).total_seconds()
        if dt > TIME_GAP_MINUTES * 60:
            starts.append(i)
            reasons.append("Time Gap")
        elif state.iloc[i] != state.iloc[i - 1]:
            starts.append(i)
            reasons.append("Behavior Changed")
    starts.append(n)
    segments = [(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]
    return segments, reasons


def _segment_dominant_state(sub: pd.DataFrame, s: int, e: int) -> str:
    return sub["behavior_state"].iloc[s:e].mode().iat[0]


# ---------------------------------------------------------------------------
# Merge short segments: Đã tối ưu hóa logic để bảo vệ Segment đặc biệt khỏi bị nuốt
# ---------------------------------------------------------------------------
def merge_short_segments(
    sub: pd.DataFrame,
    segments: List[Tuple[int, int]],
    reasons: List[str],
    min_points: int = MIN_SEGMENT_POINTS,
    protect_reasons: Optional[set] = None,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    protect_reasons: tập hợp các "segment_reason" cần được bảo vệ tuyệt đối,
    không bao giờ bị gộp dù ngắn hơn min_points (dùng cho các segment vừa
    được tách chủ đích ở bước split_fuelup_peak_tail, tránh bị nuốt ngược lại).
    """
    protect_reasons = protect_reasons or set()
    segments = list(segments)
    reasons = list(reasons)
    if len(segments) <= 1:
        return segments, reasons

    dom_state = [_segment_dominant_state(sub, s, e) for s, e in segments]

    changed = True
    while changed and len(segments) > 1:
        changed = False
        for idx, (s, e) in enumerate(segments):
            cur_len = e - s
            if cur_len >= min_points:
                continue
            if reasons[idx] in protect_reasons:
                continue

            can_left = idx > 0 and reasons[idx] != "Time Gap"
            can_right = idx < len(segments) - 1 and reasons[idx + 1] != "Time Gap"
            if not can_left and not can_right:
                continue

            cur_state = dom_state[idx]

            # Ưu tiên 1: Gộp vào bên có cùng behavior_state
            same_left = can_left and dom_state[idx - 1] == cur_state
            same_right = can_right and dom_state[idx + 1] == cur_state

            if same_left and same_right:
                left_len = segments[idx - 1][1] - segments[idx - 1][0]
                right_len = segments[idx + 1][1] - segments[idx + 1][0]
                direction = "left" if left_len >= right_len else "right"
            elif same_left:
                direction = "left"
            elif same_right:
                direction = "right"
            else:
                # Không có bên nào cùng state.
                # Nếu đây là trạng thái đặc biệt (Refuel/FuelTheft), giữ lại dù ngắn để bảo vệ sự kiện
                if cur_state in ("Stationary-FuelUp", "Stationary-FuelDown"):
                    continue  # Giữ nguyên, không gộp

                # Các segment ngắn do nhiễu (thường là Stationary-Stable hoặc Moving)
                # Ép gộp vào bên lớn hơn (dù khác state) để dọn rác.
                if can_left and can_right:
                    left_len = segments[idx - 1][1] - segments[idx - 1][0]
                    right_len = segments[idx + 1][1] - segments[idx + 1][0]
                    direction = "left" if left_len >= right_len else "right"
                elif can_left:
                    direction = "left"
                elif can_right:
                    direction = "right"
                else:
                    continue

            # Thực hiện gộp
            if direction == "left":
                # Nếu segment bên trái được bảo vệ và có reason đặc biệt, không được
                # phép "nuốt" nó vào current — nhưng ở đây current mới là bên bị gộp đi
                # nên segment bên trái (idx-1) chỉ mở rộng thêm, reason của nó giữ nguyên.
                ps, _pe = segments[idx - 1]
                segments[idx - 1] = (ps, e)
                dom_state[idx - 1] = _segment_dominant_state(sub, ps, e)
                del segments[idx]
                del reasons[idx]
                del dom_state[idx]
            else:
                _ns, ne = segments[idx + 1]
                # Nếu segment bên phải (idx+1) là segment được bảo vệ (vừa tách ra chủ đích),
                # không mở rộng đè lên nó — bỏ qua để tránh phá hỏng ranh giới đã tách.
                if reasons[idx + 1] in protect_reasons:
                    continue
                segments[idx] = (s, ne)
                dom_state[idx] = _segment_dominant_state(sub, s, ne)
                del segments[idx + 1]
                del reasons[idx + 1]
                del dom_state[idx + 1]
            changed = True
            break
    return segments, reasons


# ---------------------------------------------------------------------------
# Merge adjacent segments that ended up with the same dominant state (dọn dẹp tàn dư)
# ---------------------------------------------------------------------------
def merge_identical_adjacent_segments(
    sub: pd.DataFrame, segments: List[Tuple[int, int]], reasons: List[str]
) -> Tuple[List[Tuple[int, int]], List[str]]:
    segments = list(segments)
    reasons = list(reasons)
    if len(segments) <= 1:
        return segments, reasons
    changed = True
    while changed and len(segments) > 1:
        changed = False
        idx = 0
        while idx < len(segments) - 1:
            if reasons[idx + 1] == "Time Gap":
                idx += 1
                continue
            s1, e1 = segments[idx]
            _s2, e2 = segments[idx + 1]
            if _segment_dominant_state(sub, s1, e1) == _segment_dominant_state(sub, segments[idx + 1][0], segments[idx + 1][1]):
                segments[idx] = (s1, e2)
                reasons[idx] = "Merged Identical Adjacent State"
                del segments[idx + 1]
                del reasons[idx + 1]
                changed = True
                continue
            idx += 1
    return segments, reasons


# ---------------------------------------------------------------------------
# V5: Tách đuôi biến động/giảm sau đỉnh của segment FuelUp
# ---------------------------------------------------------------------------
def _find_split_point_after_peak(fuel_arr: np.ndarray, s: int, e: int) -> Optional[int]:
    """
    Với đoạn fuel[s:e] được coi là 1 segment Stationary-FuelUp (tăng mạnh), tìm
    điểm mà sau đó dữ liệu bắt đầu biến động/giảm thật sự (không phải nhiễu nhỏ
    dao động quanh đỉnh).

    Cách làm:
    - Xác định giá trị đỉnh (max) trong toàn segment.
    - Tìm điểm cuối cùng (local) mà dữ liệu vẫn còn nằm trong ngưỡng
      FUELUP_PEAK_EPSILON quanh đỉnh (tức vẫn được coi là "còn ở đỉnh/plateau").
    - Phần còn lại sau điểm đó là "đuôi" ứng viên.
    - Đuôi chỉ được coi là hợp lệ để tách nếu đủ dài (FUELUP_TAIL_MIN_LEN) VÀ
      có mức sụt giảm thật sự từ điểm cuối-gần-đỉnh xuống điểm cuối segment
      (FUELUP_TAIL_MIN_DROP). Nếu không, trả về None -> không tách.

    Trả về global index (tuyệt đối trong toàn bộ `sub`) là điểm bắt đầu của
    phần đuôi cần tách, hoặc None nếu không có đuôi đáng kể.
    """
    seg = fuel_arr[s:e]
    n = len(seg)
    if n == 0:
        return None
    peak_val = float(np.max(seg))

    near_peak_mask = seg >= (peak_val - FUELUP_PEAK_EPSILON)
    near_peak_idx = np.where(near_peak_mask)[0]
    if len(near_peak_idx) == 0:
        return None
    last_near_peak_local = int(near_peak_idx[-1])

    tail_local_start = last_near_peak_local + 1
    tail_len = n - tail_local_start
    if tail_len < FUELUP_TAIL_MIN_LEN:
        # Đỉnh nằm gần cuối segment (hoặc ngay tại cuối) -> không có đuôi đáng kể
        return None

    drop = float(seg[last_near_peak_local] - seg[-1])
    if drop < FUELUP_TAIL_MIN_DROP:
        # Chỉ là nhiễu nhỏ dao động quanh đỉnh, không phải giảm thật -> bỏ qua
        return None

    return s + tail_local_start


def split_fuelup_peak_tail(
    sub: pd.DataFrame,
    segments: List[Tuple[int, int]],
    reasons: List[str],
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    Rule bổ sung (V5): với mỗi segment có behavior_state chủ đạo là
    Stationary-FuelUp, kiểm tra xem sau khi lên đỉnh, có đoạn dữ liệu
    biến động/giảm thật ở cuối segment hay không:
        - Nếu KHÔNG (đỉnh nằm ở cuối, hoặc chỉ là nhiễu nhỏ) -> bỏ qua, giữ nguyên.
        - Nếu CÓ -> tách thành 2 segment, điểm tách là điểm mà dữ liệu bắt đầu
          giảm/biến động thật sự. Segment mới được đánh dấu reason
          "FuelUp Peak Tail Split" để các bước merge phía sau không nuốt lại.
    """
    fuel_arr = sub["fuel"].to_numpy()
    new_segments: List[Tuple[int, int]] = []
    new_reasons: List[str] = []

    for (s, e), reason in zip(segments, reasons):
        state = _segment_dominant_state(sub, s, e)
        if state != "Stationary-FuelUp":
            new_segments.append((s, e))
            new_reasons.append(reason)
            continue

        split_at = _find_split_point_after_peak(fuel_arr, s, e)
        if split_at is None or split_at <= s or split_at >= e:
            new_segments.append((s, e))
            new_reasons.append(reason)
            continue

        # Tách: [s, split_at) giữ nguyên là phần tăng mạnh + đỉnh
        #       [split_at, e) là đuôi giảm/biến động, tách riêng
        new_segments.append((s, split_at))
        new_reasons.append(reason)
        new_segments.append((split_at, e))
        new_reasons.append(PROTECTED_SPLIT_REASON)

    return new_segments, new_reasons


# ---------------------------------------------------------------------------
# Segment summary construction
# ---------------------------------------------------------------------------
SUMMARY_COLUMNS = [
    "segment_id", "car_id", "start_time", "end_time", "duration_min", "sample_count",
    "movement_behavior", "mean_speed", "max_speed", "total_distance_m", "gps_radius_m",
    "fuel_start", "fuel_end", "fuel_change", "behavior_state", "segment_reason",
]


def _movement_behavior_from_state(state: str) -> str:
    return "Moving" if state == "Moving" else "Stationary"


def build_segment_tables(
    sub: pd.DataFrame, car_id: str, segments: List[Tuple[int, int]], reasons: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub = sub.copy()
    sub["segment_id"] = 0
    summary_rows = []
    for seg_num, ((s, e), reason) in enumerate(zip(segments, reasons), start=1):
        sub.iloc[s:e, sub.columns.get_loc("segment_id")] = seg_num
        seg = sub.iloc[s:e]
        state = _segment_dominant_state(sub, s, e)
        summary_rows.append({
            "segment_id": seg_num,
            "car_id": car_id,
            "start_time": seg["timestamp"].iloc[0],
            "end_time": seg["timestamp"].iloc[-1],
            "duration_min": round((seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds() / 60.0, 1),
            "sample_count": e - s,
            "movement_behavior": _movement_behavior_from_state(state),
            "mean_speed": round(float(seg["speed"].mean()), 2),
            "max_speed": round(float(seg["speed"].max()), 2),
            "total_distance_m": round(float(np.nansum(seg["gps_step_disp_m"].to_numpy())), 1),
            "gps_radius_m": round(_gps_radius_m(seg["latitude"].to_numpy(), seg["longitude"].to_numpy()), 1),
            "fuel_start": round(float(seg["fuel"].iloc[0]), 2),
            "fuel_end": round(float(seg["fuel"].iloc[-1]), 2),
            "fuel_change": round(float(seg["fuel"].iloc[-1] - seg["fuel"].iloc[0]), 2),
            "behavior_state": state,
            "segment_reason": reason,
        })
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    return sub, summary


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_segments(sub: pd.DataFrame, summary: pd.DataFrame, car_id: str, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(sub["timestamp"], sub["fuel"], color="blue", linewidth=1.1, label="Fuel (clean)")
    for _, row in summary.iterrows():
        ax.axvline(row["start_time"], color="red", linestyle="--", linewidth=1.0, alpha=0.8)
        mid_time = row["start_time"] + (row["end_time"] - row["start_time"]) / 2
        y_top = sub["fuel"].max()
        ax.text(mid_time, y_top, str(int(row["segment_id"])), color="red", fontsize=9, ha="center", va="bottom")
    if len(summary):
        ax.axvline(summary["end_time"].iloc[-1], color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title(f"Segment Detection – {car_id}", fontsize=14)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Fuel Level")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = figures_dir / f"{str(car_id).lower().replace(' ', '_')}_segments.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(all_summaries: pd.DataFrame, processed_cars: List[str], report_path: Path) -> None:
    n_cars = len(processed_cars)
    n_segments = len(all_summaries)
    md_lines = [
        "# Segment Detection Report (4-State Model: Stationary-Stable/FuelUp/FuelDown, Moving)",
        "",
        "## Dataset",
        f"- Tổng số xe: {n_cars} ({', '.join(sorted(map(str, processed_cars)))})",
        f"- Tổng số segment: {n_segments}",
    ]
    if n_segments:
        per_car_counts = all_summaries.groupby("car_id")["segment_id"].count()
        md_lines += ["", "## Segment Count per Car", "", "| car_id | Số segment |", "|--------|------------|"]
        for car, cnt in per_car_counts.items():
            md_lines.append(f"| {car} | {cnt} |")
        samples = all_summaries["sample_count"]
        duration = all_summaries["duration_min"]
        md_lines += ["", f"- Median segment length: {samples.median():.1f} điểm ({duration.median():.1f} phút)"]

        state_counts = all_summaries["behavior_state"].value_counts()
        md_lines += ["", "## Phân bố behavior_state", "", "| behavior_state | Số segment |", "|---|---|"]
        for state, cnt in state_counts.items():
            md_lines.append(f"| {state} | {cnt} |")

        reason_counts = all_summaries["segment_reason"].value_counts()
        md_lines += ["", "## Phân bố segment_reason", "", "| segment_reason | Số segment |", "|---|---|"]
        for reason, cnt in reason_counts.items():
            md_lines.append(f"| {reason} | {cnt} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Báo cáo đã được lưu tại %s", report_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Đọc dữ liệu đã cleaning từ %s", INPUT_CSV)
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        logger.error("Không thể đọc file đầu vào: %s", e)
        sys.exit(1)

    rename_map = {"FuelTime": "timestamp", "FuelLevel": "fuel", "Lat": "latitude", "Lng": "longitude", "Speed": "speed"}
    df = df.rename(columns=rename_map)
    required = {"timestamp", "fuel", "latitude", "longitude", "speed", "car_id"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Thiếu cột bắt buộc: %s", missing)
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["car_id", "timestamp"]).reset_index(drop=True)

    all_car_ids = sorted(df["car_id"].unique(), key=lambda c: (_car_number(c) is None, _car_number(c) or 0, str(c)))
    processed_cars = [c for c in all_car_ids if _car_number(c) not in SKIP_CAR_NUMBERS]
    segment_frames = []
    summary_frames = []
    original_cols = list(df.columns)

    for car_id in processed_cars:
        sub = df[df["car_id"] == car_id].reset_index(drop=True)
        if len(sub) == 0:
            continue
        logger.info("Xử lý %s (%d điểm) …", car_id, len(sub))

        sub = compute_behaviour_signals(sub)
        sub = assign_behavior_state(sub)
        sub = debounce_behaviour(sub)

        segments, reasons = detect_segments(sub)
        segments, reasons = merge_short_segments(sub, segments, reasons)
        segments, reasons = merge_identical_adjacent_segments(sub, segments, reasons)

        # V5 — Rule mới: tách đuôi biến động/giảm sau đỉnh của segment FuelUp
        segments, reasons = split_fuelup_peak_tail(sub, segments, reasons)

        # V5 — Dọn dẹp phân mảnh còn sót lại sau khi tách (ngưỡng nhỏ hơn),
        # nhưng bảo vệ tuyệt đối segment vừa được tách ra chủ đích ở trên.
        segments, reasons = merge_short_segments(
            sub, segments, reasons,
            min_points=POST_SPLIT_MIN_POINTS,
            protect_reasons={PROTECTED_SPLIT_REASON},
        )
        segments, reasons = merge_identical_adjacent_segments(sub, segments, reasons)

        sub_out, summary = build_segment_tables(sub, car_id, segments, reasons)
        keep_cols = original_cols + ["segment_id", "behavior_state"]
        segment_frames.append(sub_out[keep_cols])
        summary_frames.append(summary)
        plot_segments(sub_out, summary, car_id, FIGURES_DIR)

    if not segment_frames:
        logger.error("Không có xe nào được xử lý — kiểm tra lại INPUT_CSV / SKIP_CAR_NUMBERS.")
        sys.exit(1)

    segment_data = pd.concat(segment_frames, ignore_index=True)
    all_summaries = pd.concat(summary_frames, ignore_index=True)
    reverse_rename = {v: k for k, v in rename_map.items()}
    segment_out = segment_data.rename(columns=reverse_rename)

    OUTPUT_SEGMENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    segment_out.to_csv(OUTPUT_SEGMENT_CSV, index=False)
    all_summaries.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    generate_report(all_summaries, processed_cars, REPORT_MD)


if __name__ == "__main__":
    main()