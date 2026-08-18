#!/usr/bin/env python3
"""
post_split_peak_tail.py – Phát hiện & tách đuôi sau đỉnh và phần đầu biến động trước refuel.
- Tự động tìm điểm bắt đầu tăng mạnh (onset) và điểm kết thúc vùng đỉnh (tail_start).
- Tách segment gốc thành tối đa 3 đoạn: pre‑rise, rise, post‑peak tail.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# ---------------------------- Cấu hình ----------------------------
INPUT_SEGMENT_DATA = Path("data/processed/segment_v2/segment_data.csv")
INPUT_SEGMENT_SUMMARY = Path("data/processed/segment_v2/segment_summary.csv")
OUTPUT_SEGMENT_DATA = INPUT_SEGMENT_DATA          # ghi đè
OUTPUT_SEGMENT_SUMMARY = INPUT_SEGMENT_SUMMARY    # ghi đè
SPLIT_REPORT_MD = Path("data/processed/segment_v2/split_summary.md")

# --- Ngưỡng tách đuôi (sau đỉnh) ---
PEAK_EPSILON = 2.0              # L, dung sai "vẫn gần đỉnh"
MIN_TAIL_DROP = 2.0             # L, sụt giảm tối thiểu từ đỉnh -> cuối đuôi
MIN_TAIL_LEN = 3                # số điểm tối thiểu của đuôi

# --- Ngưỡng phát hiện refuel ---
MIN_FUEL_INCREASE_L = 7.0       # L, mức tăng tối thiểu để coi là refuel
MIN_PRE_RISE_LEN = 3            # số điểm tối thiểu của phần trước khi tăng để tách riêng

# --- Gán trạng thái ---
MOVING_SPEED_THRESHOLD = 3.0    # km/h
FUEL_INCREASE_THRESHOLD = 10.0  # L, để gán FuelUp
FUEL_DECREASE_THRESHOLD = 5.0   # L, để gán FuelDown

# ---------------------------- Logging ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("post_split")

# ---------------------------- Hàm phụ trợ ----------------------------
def haversine_km_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def gps_radius_m(lat, lon):
    if len(lat) == 0:
        return 0.0
    clat, clon = np.mean(lat), np.mean(lon)
    dists = haversine_km_vec(lat, lon, np.full_like(lat, clat), np.full_like(lon, clon)) * 1000.0
    return float(np.max(dists)) if len(dists) else 0.0

def compute_step_disp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["car_id", "timestamp"]).reset_index(drop=True)
    lat = df["latitude"].values
    lon = df["longitude"].values
    step = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        if df.loc[i, "car_id"] == df.loc[i-1, "car_id"]:
            step[i] = haversine_km_vec(
                np.array([lat[i-1]]), np.array([lon[i-1]]),
                np.array([lat[i]]),   np.array([lon[i]])
            )[0] * 1000.0
    df["step_disp_m"] = step
    return df

def assign_behavior_state(speed: pd.Series, fuel_change: float) -> str:
    """Gán behavior_state dựa trên tốc độ trung bình và thay đổi nhiên liệu."""
    if speed.mean() > MOVING_SPEED_THRESHOLD:
        return "Moving"
    if fuel_change > FUEL_INCREASE_THRESHOLD:
        return "Stationary-FuelUp"
    elif fuel_change < -FUEL_DECREASE_THRESHOLD:
        return "Stationary-FuelDown"
    return "Stationary-Stable"

def find_split_point(fuel_arr: np.ndarray) -> Optional[int]:
    """Trả về vị trí bắt đầu đuôi (sau đỉnh) nếu có, None nếu không."""
    n = len(fuel_arr)
    if n < MIN_TAIL_LEN + 1:
        return None
    peak = np.max(fuel_arr)
    # Yêu cầu mức tăng từ đầu mảng (hoặc từ onset) đã được kiểm tra bên ngoài
    near_peak_mask = fuel_arr >= (peak - PEAK_EPSILON)
    near_idx = np.where(near_peak_mask)[0]
    if len(near_idx) == 0:
        return None
    last_near = int(near_idx[-1])
    tail_start = last_near + 1
    if tail_start >= n or (n - tail_start) < MIN_TAIL_LEN:
        return None
    drop = float(fuel_arr[last_near] - fuel_arr[-1])
    if drop < MIN_TAIL_DROP:
        return None
    return tail_start

# ---------------------------- Logic chính ----------------------------
def process_splits():
    # 1. Đọc dữ liệu
    logger.info("Đọc %s", INPUT_SEGMENT_DATA)
    data = pd.read_csv(INPUT_SEGMENT_DATA)

    # Đổi tên cột về dạng chuẩn
    rename_map = {
        "FuelTime": "timestamp",
        "Lat": "latitude",
        "Lng": "longitude",
        "Speed": "speed",
        "FuelLevel": "fuel"
    }
    existing_cols = {k: v for k, v in rename_map.items() if k in data.columns}
    if existing_cols:
        data = data.rename(columns=existing_cols)
        logger.info("Đã đổi tên cột: %s", existing_cols)

    required = {"timestamp", "latitude", "longitude", "speed", "fuel", "car_id", "segment_id"}
    missing = required - set(data.columns)
    if missing:
        logger.error("Thiếu cột bắt buộc: %s", missing)
        sys.exit(1)

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    # Đọc summary cũ
    old_summary = pd.read_csv(INPUT_SEGMENT_SUMMARY) if INPUT_SEGMENT_SUMMARY.exists() else pd.DataFrame()

    # 2. Tính step_disp_m
    if "step_disp_m" not in data.columns:
        data = compute_step_disp(data)

    # 3. Gom nhóm theo (car_id, segment_id)
    data = data.sort_values(["car_id", "timestamp"]).reset_index(drop=True)
    groups = data.groupby(["car_id", "segment_id"])

    split_log = []                 # Lưu thông tin tách
    segment_mapping = {}           # (car, orig_seg) -> list of new_ids
    original_reason = {}

    if not old_summary.empty:
        for _, row in old_summary.iterrows():
            original_reason[(row["car_id"], row["segment_id"])] = row.get("segment_reason", "Unknown")

    seg_counter = {}

    for (car, orig_seg), grp in groups:
        if car not in seg_counter:
            seg_counter[car] = 1

        fuel_arr = grp["fuel"].to_numpy()
        n = len(fuel_arr)
        peak_idx = np.argmax(fuel_arr)
        peak_val = fuel_arr[peak_idx]

        # Kiểm tra toàn bộ segment có mức tăng đủ lớn không
        if peak_val - fuel_arr[0] < MIN_FUEL_INCREASE_L:
            # Không đủ điều kiện refuel -> giữ nguyên segment
            new_id = seg_counter[car]
            data.loc[grp.index, "new_segment_id"] = new_id
            seg_counter[car] += 1
            segment_mapping.setdefault((car, orig_seg), []).append(new_id)
            continue

        # Tìm điểm bắt đầu tăng mạnh (onset) = đáy thấp nhất trước đỉnh
        # Lưu ý: đáy có thể nằm ở đầu mảng -> không tách phần đầu
        sub_arr_to_peak = fuel_arr[:peak_idx+1]
        valley_idx = int(np.argmin(sub_arr_to_peak))  # vị trí đáy trong toàn bộ mảng
        # Điều kiện để tách phần đầu:
        # - valley_idx > 0 (không phải điểm đầu)
        # - valley_idx >= MIN_PRE_RISE_LEN (đủ dài để coi là một đoạn riêng)
        # - Mức tăng từ valley lên đỉnh >= MIN_FUEL_INCREASE_L
        onset = 0
        if (valley_idx > 0 and
            valley_idx >= MIN_PRE_RISE_LEN and
            peak_val - fuel_arr[valley_idx] >= MIN_FUEL_INCREASE_L):
            onset = valley_idx
        # Nếu không thoả, onset = 0 (không tách phần đầu)

        # Danh sách các điểm chia tuyệt đối trong grp
        split_points = []
        if onset > 0:
            split_points.append(onset)

        # Xét phần từ onset đến cuối để tách đuôi sau đỉnh
        tail_segment = fuel_arr[onset:]
        tail_split = find_split_point(tail_segment)
        if tail_split is not None:
            abs_tail = onset + tail_split
            # Kiểm tra abs_tail hợp lệ (không trùng onset, không vượt quá n)
            if onset < abs_tail < n:
                split_points.append(abs_tail)

        # Nếu không có điểm tách nào -> giữ nguyên segment
        if not split_points:
            new_id = seg_counter[car]
            data.loc[grp.index, "new_segment_id"] = new_id
            seg_counter[car] += 1
            segment_mapping.setdefault((car, orig_seg), []).append(new_id)
            continue

        # Sắp xếp và loại bỏ trùng lặp
        split_points = sorted(set(split_points))
        # Thêm điểm cuối
        split_points.append(n)

        # Tiến hành tách
        prev = 0
        new_ids = []
        for sp in split_points:
            if sp == prev:
                continue
            new_id = seg_counter[car]
            data.loc[grp.index[prev:sp], "new_segment_id"] = new_id
            seg_counter[car] += 1
            new_ids.append(new_id)
            prev = sp

        segment_mapping[(car, orig_seg)] = new_ids

        # Ghi log
        desc = f"{car}-{orig_seg}"
        if onset > 0:
            desc += f" (tách đầu từ điểm {onset})"
        if tail_split is not None:
            desc += f" (tách đuôi từ {onset + tail_split})"
        logger.info("Tách %s → %s", desc, new_ids)

        split_log.append({
            "car_id": car,
            "original_segment_id": orig_seg,
            "new_segment_ids": ", ".join(map(str, new_ids)),
            "peak_fuel": round(float(peak_val), 1),
            "onset_index": onset,
            "tail_abs_index": onset + tail_split if tail_split is not None else None,
            "parts": len(new_ids)
        })

    # 4. Cập nhật segment_id chính thức
    data["segment_id"] = data["new_segment_id"].astype(int)
    data.drop(columns=["new_segment_id"], inplace=True, errors="ignore")

    # 5. Tính lại summary và gán behavior_state mới
    summary_rows = []
    for (car, seg_id), grp in data.groupby(["car_id", "segment_id"]):
        grp = grp.sort_values("timestamp")
        seg_duration_min = (grp["timestamp"].iloc[-1] - grp["timestamp"].iloc[0]).total_seconds() / 60.0
        fuel_start = grp["fuel"].iloc[0]
        fuel_end = grp["fuel"].iloc[-1]
        fuel_change = fuel_end - fuel_start

        # Tìm reason và phần của segment
        reason = "Original"
        for (ocar, oid), new_ids in segment_mapping.items():
            if car == ocar and seg_id in new_ids:
                if len(new_ids) == 1:
                    reason = original_reason.get((ocar, oid), "Unknown")
                else:
                    idx = new_ids.index(seg_id)
                    if idx == 0 and len(new_ids) >= 2:
                        # Nếu có onset > 0 thì phần đầu là pre-rise
                        # Kiểm tra xem có onset không bằng cách tìm trong split_log
                        reason = "Split: Pre-rise phase"
                    elif idx == len(new_ids) - 1 and len(new_ids) >= 2:
                        reason = "Split: Post-peak tail"
                    else:
                        reason = "Split: FuelUp phase"
                break

        summary_rows.append({
            "segment_id": seg_id,
            "car_id": car,
            "start_time": grp["timestamp"].iloc[0],
            "end_time": grp["timestamp"].iloc[-1],
            "duration_min": round(seg_duration_min, 1),
            "sample_count": len(grp),
            "movement_behavior": "Moving" if grp["speed"].mean() > MOVING_SPEED_THRESHOLD else "Stationary",
            "mean_speed": round(grp["speed"].mean(), 2),
            "max_speed": round(grp["speed"].max(), 2),
            "total_distance_m": round(np.nansum(grp["step_disp_m"]), 1),
            "gps_radius_m": round(gps_radius_m(grp["latitude"].values, grp["longitude"].values), 1),
            "fuel_start": round(fuel_start, 2),
            "fuel_end": round(fuel_end, 2),
            "fuel_change": round(fuel_change, 2),
            "behavior_state": assign_behavior_state(grp["speed"], fuel_change),
            "segment_reason": reason
        })
    new_summary = pd.DataFrame(summary_rows)

    # 6. Ghi đè file, giữ tên cột gốc
    reverse_rename = {v: k for k, v in rename_map.items()}
    data_out = data.rename(columns=reverse_rename)
    if "step_disp_m" in data_out.columns:
        data_out.drop(columns=["step_disp_m"], inplace=True)

    behavior_map = dict(zip(zip(new_summary["car_id"], new_summary["segment_id"]), new_summary["behavior_state"]))
    data_out["behavior_state"] = data_out.apply(
        lambda row: behavior_map.get((row["car_id"], row["segment_id"]), "Unknown"), axis=1
    )

    OUTPUT_SEGMENT_DATA.parent.mkdir(parents=True, exist_ok=True)
    data_out.to_csv(OUTPUT_SEGMENT_DATA, index=False)
    new_summary.to_csv(OUTPUT_SEGMENT_SUMMARY, index=False)
    logger.info("Đã ghi đè %s và %s", OUTPUT_SEGMENT_DATA, OUTPUT_SEGMENT_SUMMARY)

    # 7. Báo cáo tách
    lines = [
        "# Báo cáo tách segment (pre‑rise & post‑peak tail)",
        "",
        f"Tổng số segment gốc bị tách: {len(split_log)}",
        "",
        "| Car ID | Seg gốc | Các seg mới | Đỉnh (L) | Ghi chú |",
        "|--------|---------|-------------|----------|---------|"
    ]
    for entry in split_log:
        note = []
        if entry["onset_index"] > 0:
            note.append(f"pre‑rise tách tại {entry['onset_index']}")
        if entry["tail_abs_index"] is not None:
            note.append(f"đuôi tách tại {entry['tail_abs_index']}")
        lines.append(
            f"| {entry['car_id']} | {entry['original_segment_id']} | {entry['new_segment_ids']} "
            f"| {entry['peak_fuel']} | {', '.join(note) if note else 'chỉ tách đuôi'} |"
        )
    if not split_log:
        lines.append("| *(không có segment nào bị tách)* | | | | |")
    SPLIT_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Báo cáo lưu tại %s", SPLIT_REPORT_MD)

if __name__ == "__main__":
    process_splits()