"""
generate_dataset.py
====================================================================
Sinh bộ dữ liệu MÔ PHỎNG phục vụ bài toán phát hiện sự kiện nhiên liệu
(Fuel Event Detection) trên ô tô theo thời gian thực.

Tác giả: Senior Data Engineer / AI Engineer
Yêu cầu: Python 3.11, pandas, numpy, matplotlib

Output: fuel_sensor_dataset.csv (~1000 dòng, mẫu 5 giây/lần)
Cấu trúc cột (KHÔNG đổi qua các phiên bản):
    timestamp, device_id, vehicle_id, adc_raw, voltage, fuel_raw,
    speed, latitude, longitude, heading, event_label

--------------------------------------------------------------------
PHIÊN BẢN v7 – TÍN HIỆU CẢM BIẾN THÔ (RAW SENSOR) ĐÚNG VẬT LÝ

Thay đổi chính:
  - Loại bỏ mọi bộ lọc làm mượt (ECU filter, EMA cuối pipeline).
  - Chỉ giữ 1 tầng sensor lag vật lý (EMA với alpha cố định).
  - Sloshing mô phỏng bằng hệ dao động tắt dần (second-order damped
    oscillator) khi có kích thích từ gia tốc / góc lái / địa hình.
  - Transient fluctuation nhỏ, hiếm (0.2%, 0.2-0.5L, 2-4 mẫu).
  - Outlier mô phỏng lỗi ADC (0.1% đơn, 0.05% burst, 0.8-1.5L).
  - Giữ hysteresis, calibration bias, sensor drift.
  - Đầu ra là tín hiệu RAW – đủ “bẩn” để pipeline xử lý có ý nghĩa.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
RANDOM_SEED = 42
N_SAMPLES = 1000
SAMPLE_INTERVAL_SEC = 5
TANK_CAPACITY_L = 60.0

START_TIME = datetime(2026, 1, 1, 8, 0, 0)
DEVICE_ID = "DEV001"
VEHICLE_ID = "29A-12345"

START_LAT = 21.0285
START_LON = 105.8542

ADC_MIN, ADC_MAX = 500, 3500
VOLTAGE_MIN, VOLTAGE_MAX = 2.1, 4.9
NOMINAL_VOLTAGE = 3.5

EVENT_STATES = ["Driving", "Idle", "Refuel", "Fuel Theft"]

# Tỉ lệ mục tiêu
EVENT_TARGET_RATIO = {"Driving": 0.545, "Idle": 0.255, "Refuel": 0.10, "Fuel Theft": 0.10}

# Chu kỳ vận hành Driving
DRIVING_SPEED_TYPES = [
    "accelerating", "decelerating", "steady_cruise", "urban_road",
    "stop_and_go", "traffic_jam", "highway", "mountain_road",
]
DRIVING_SPEED_WEIGHTS = [0.13, 0.13, 0.18, 0.14, 0.12, 0.10, 0.10, 0.10]

DRIVING_ROAD_TYPES = ["straight", "turn_left", "turn_right", "u_turn"]
DRIVING_ROAD_WEIGHTS = [0.5, 0.2, 0.2, 0.1]

# Bảng calibration ADC phi tuyến
_CALIB_FUEL_RATIO = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
_CALIB_ADC = np.array([ADC_MIN, 640, 1080, 1900, 2760, 3180, ADC_MAX])

# Cấu hình Idle đa dạng
IDLE_TYPES = ["stable", "engine_on", "temp_drift", "vehicle_shake"]
IDLE_TYPE_WEIGHTS = [0.3, 0.3, 0.2, 0.2]

# Cấu hình Fuel Theft
THEFT_TYPES = ["fast", "slow", "very_slow"]
THEFT_TYPE_WEIGHTS = [0.5, 0.3, 0.2]

# Lượng tử hoá mức tiêu hao
FUEL_QUANTIZATION_RESOLUTION = 0.05

# --- Sensor lag (vật lý) ---
SENSOR_LAG_ALPHA = 0.15           # EMA cố định, mô phỏng độ trễ của phao

# --- Sloshing: hệ dao động tắt dần bậc 2 ---
SLOSH_NATURAL_FREQ = 0.25          # tần số góc (rad/mẫu)
SLOSH_DECAY = 0.85                 # hệ số suy giảm (0<R<1)
SLOSH_EXCITATION_GAIN = 0.02       # nhân cho gia tốc dọc
SLOSH_TURN_GAIN = 0.015            # nhân cho thay đổi heading
SLOSH_ROAD_FACTOR = {
    "mountain_road": 2.5,
    "turn_left": 1.4,
    "turn_right": 1.4,
    "u_turn": 2.0,
    "straight": 1.0,
}

# --- Transient fluctuation ---
TRANSIENT_RATE = 0.002             # 0.2%
TRANSIENT_AMP = (0.2, 0.5)         # Lít
TRANSIENT_LEN = (2, 4)             # mẫu

# --- Outlier (lỗi ADC) ---
OUTLIER_POINT_RATE = 0.001         # 0.1%
OUTLIER_POINT_AMP = (0.8, 1.5)     # Lít
OUTLIER_BURST_RATE = 0.0005        # 0.05%
OUTLIER_BURST_AMP = (0.8, 1.5)     # Lít
OUTLIER_BURST_LEN = (2, 3)         # mẫu

# --- Sensor drift ---
DRIFT_SIGMA = 0.0006               # độ lệch chuẩn mỗi mẫu

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ====================================================================
# TẦNG 1: VEHICLE PHYSICS
# ====================================================================

def _sample_event_length_and_amount(state):
    """Trả về (length, amount) cho một đoạn sự kiện."""
    if state == "Refuel":
        r = random.random()
        if r < 0.25:
            length = random.randint(3, 6)
            amount = random.uniform(2, 3)
        elif r < 0.80:
            length = random.randint(18, 35)
            amount = random.uniform(10, 25)
        else:
            length = random.randint(30, 50)
            amount = random.uniform(30, 38)
        return length, amount

    if state == "Fuel Theft":
        r = random.random()
        if r < 0.25:
            length = random.randint(2, 4)
            amount = random.uniform(1.5, 3)
        elif r < 0.80:
            length = random.randint(8, 25)
            amount = random.uniform(5, 20)
        else:
            length = random.randint(20, 40)
            amount = random.uniform(15, 35)
        return length, amount

    if state == "Idle":
        r = random.random()
        if r < 0.55:
            length = random.randint(2, 5)
        elif r < 0.85:
            length = random.randint(6, 15)
        else:
            length = random.randint(16, 30)
        return length, None

    # Driving
    r = random.random()
    if r < 0.25:
        length = random.randint(3, 9)
    elif r < 0.75:
        length = random.randint(10, 40)
    else:
        length = random.randint(41, 80)
    return length, None


def generate_event_segments(total_samples):
    remaining = {s: total_samples * r for s, r in EVENT_TARGET_RATIO.items()}
    segments = []
    last_state = None
    total = 0
    guard = 0

    def _make_idle_transition(length):
        idle_type = random.choices(IDLE_TYPES, weights=IDLE_TYPE_WEIGHTS)[0]
        return {
            "state": "Idle", "length": length, "amount": None,
            "speed_type": None, "road_type": "stop", "idle_type": idle_type,
            "theft_type": None
        }

    while total < total_samples and guard < 20000:
        guard += 1
        candidates = [s for s in EVENT_STATES if s != last_state and remaining[s] > 0]
        if not candidates:
            candidates = [s for s in EVENT_STATES if s != last_state] or EVENT_STATES

        w = np.array([max(remaining[s], 1e-6) for s in candidates], dtype=float)
        w = w / w.sum()
        state = np.random.choice(candidates, p=w)

        if state in ("Refuel", "Fuel Theft") and last_state == "Driving":
            t_len = random.randint(2, 5)
            segments.append(_make_idle_transition(t_len))
            remaining["Idle"] -= t_len
            total += t_len
            last_state = "Idle"

        length, amount = _sample_event_length_and_amount(state)
        seg = {
            "state": state, "length": length, "amount": amount,
            "speed_type": None, "road_type": "stop",
            "idle_type": None, "theft_type": None
        }

        if state == "Driving":
            seg["speed_type"] = random.choices(DRIVING_SPEED_TYPES, weights=DRIVING_SPEED_WEIGHTS)[0]
            seg["road_type"] = random.choices(DRIVING_ROAD_TYPES, weights=DRIVING_ROAD_WEIGHTS)[0]
        elif state == "Idle":
            seg["idle_type"] = random.choices(IDLE_TYPES, weights=IDLE_TYPE_WEIGHTS)[0]
        elif state == "Fuel Theft":
            seg["theft_type"] = random.choices(THEFT_TYPES, weights=THEFT_TYPE_WEIGHTS)[0]

        segments.append(seg)
        remaining[state] -= length
        total += length
        last_state = state

        if state in ("Refuel", "Fuel Theft") and total < total_samples:
            t_len = random.randint(2, 5)
            segments.append(_make_idle_transition(t_len))
            remaining["Idle"] -= t_len
            total += t_len
            last_state = "Idle"

    return segments


def _generate_driving_speed_profile(speed_type, length):
    if speed_type == "accelerating":
        target = random.uniform(40, 85)
        v = np.linspace(random.uniform(5, 20), target, length) + np.random.normal(0, 2, length)
    elif speed_type == "decelerating":
        start = random.uniform(40, 85)
        target = random.uniform(0, 15)
        v = np.linspace(start, target, length) + np.random.normal(0, 2, length)
    elif speed_type == "steady_cruise":
        base = random.uniform(30, 65)
        v = base + np.random.normal(0, 3, length)
    elif speed_type == "urban_road":
        base = random.uniform(15, 40)
        v = base + np.random.normal(0, 5, length)
        stop_mask = np.random.random(length) < 0.08
        v[stop_mask] = 0.0
    elif speed_type == "stop_and_go":
        cycle = max(4, length // random.randint(2, 4))
        v = np.array([
            random.uniform(15, 30) if (i % cycle) < cycle * 0.5 else 0.0
            for i in range(length)
        ]) + np.random.normal(0, 1.5, length)
    elif speed_type == "traffic_jam":
        v = np.array([random.uniform(0, 10) if random.random() < 0.5 else 0.0 for _ in range(length)])
        v = v + np.random.normal(0, 1, length)
    elif speed_type == "highway":
        base = random.uniform(70, 100)
        v = base + np.random.normal(0, 4, length)
    elif speed_type == "mountain_road":
        base = random.uniform(20, 50)
        v = base + np.random.normal(0, 6, length)
    else:
        v = np.random.uniform(20, 60, length)
    return np.clip(v, 0, 110)


def _simulate_road_slope(n):
    elevation = np.cumsum(np.random.normal(0, 1.2, size=n))
    return np.diff(elevation, prepend=elevation[0])


def _physical_fuel_consumption(speed, accel, slope):
    idle_burn = 0.004
    speed_burn = 0.0006 * speed
    accel_burn = 0.02 * max(accel, 0.0)
    decel_bonus = 0.15 * min(accel, 0.0)
    slope_effect = -0.01 * slope
    noise = np.random.normal(0, 0.004)
    return idle_burn + speed_burn + accel_burn + slope_effect - decel_bonus + noise


def _apply_fuel_quantization(fuel_array, resolution=FUEL_QUANTIZATION_RESOLUTION):
    quantized = np.zeros_like(fuel_array)
    base = fuel_array[0]
    for i in range(len(fuel_array)):
        delta = fuel_array[i] - base
        quantized_delta = round(delta / resolution) * resolution
        quantized[i] = base + quantized_delta
    return np.clip(quantized, 0, TANK_CAPACITY_L)


def _generate_idle_fuel_curve(idle_type, length):
    deltas = np.zeros(length)
    if idle_type == "stable":
        deltas = np.random.normal(0, 0.005, size=length)
    elif idle_type == "engine_on":
        base_rate = random.uniform(0.001, 0.005)
        for i in range(length):
            if random.random() < 0.7:
                deltas[i] = -base_rate
            else:
                deltas[i] = 0.0
    elif idle_type == "temp_drift":
        drift = np.cumsum(np.random.normal(0, 0.008, size=length))
        drift = np.clip(drift, -0.2, 0.2)
        deltas = np.diff(drift, prepend=0.0) + np.random.normal(0, 0.002, size=length)
    elif idle_type == "vehicle_shake":
        i = 0
        while i < length:
            r = random.random()
            if r < 0.5:
                seg_len = min(random.randint(3, 6), length - i)
                deltas[i:i+seg_len] = np.random.normal(0, 0.005, size=seg_len)
                i += seg_len
            else:
                seg_len = min(random.randint(2, 3), length - i)
                amp = random.uniform(0.05, 0.1) * random.choice([1, -1])
                bump = amp * np.sin(np.linspace(0, np.pi, seg_len))
                deltas[i:i+seg_len] = bump + np.random.normal(0, 0.01, size=seg_len)
                i += seg_len
    return deltas


def _simulate_refuel_curve(length, total_amount):
    settle_len = max(1, int(length * 0.15))
    refuel_len = length - settle_len

    increments = np.zeros(length)
    remaining = total_amount
    i = 0
    while i < refuel_len and remaining > 0.01:
        seg_len = min(random.randint(2, 4), refuel_len - i)
        if seg_len <= 0:
            break
        r = random.random()
        if r < 0.3:
            portion = remaining * random.uniform(0.12, 0.25)
        elif r < 0.7:
            portion = remaining * random.uniform(0.01, 0.06)
        else:
            portion = remaining * random.uniform(0.05, 0.15)

        per_sample = portion / seg_len
        noise = np.random.normal(0, per_sample * 0.15 + 0.01, size=seg_len)
        increments[i:i + seg_len] = np.maximum(per_sample + noise, 0)
        remaining -= portion
        i += seg_len

    if settle_len > 0:
        # dao động nhẹ ±0.05L rồi ổn định
        increments[-settle_len:] = np.random.normal(0, 0.05, size=settle_len)

    leftover = total_amount - increments.sum()
    if leftover > 0:
        last_idx = min(max(i, 1), length) - 1
        increments[last_idx] += leftover
    return increments


def _simulate_theft_curve(length, total_amount, theft_type="fast"):
    decrements = np.zeros(length)
    if theft_type == "very_slow":
        rate = random.uniform(0.005, 0.03)
        decrements = rate + np.random.normal(0, rate * 0.05, size=length)
        total = decrements.sum()
        if total > total_amount:
            decrements = decrements * (total_amount / total)
        return np.clip(decrements, 0, None)

    if theft_type == "slow":
        rate = random.uniform(0.02, 0.08)
        for i in range(length):
            if random.random() < 0.15:
                decrements[i] = 0.0
            else:
                decrements[i] = rate + np.random.normal(0, rate * 0.1)
        total = decrements.sum()
        if total > total_amount:
            decrements = decrements * (total_amount / total)
        return np.clip(decrements, 0, None)

    remaining = total_amount
    i = 0
    while i < length and remaining > 0.01:
        r = random.random()
        seg_len = min(random.randint(1, 3), length - i)
        if seg_len <= 0:
            break
        if r < 0.2:
            decrements[i:i + seg_len] = np.random.normal(0, 0.02, size=seg_len).clip(min=0)
            i += seg_len
            continue
        if r < 0.55:
            portion = remaining * random.uniform(0.20, 0.35)
        else:
            portion = remaining * random.uniform(0.05, 0.15)

        per_sample = portion / seg_len
        noise = np.random.normal(0, per_sample * 0.10 + 0.01, size=seg_len)
        decrements[i:i + seg_len] = np.maximum(per_sample + noise, 0)
        remaining -= portion
        i += seg_len

    leftover = total_amount - decrements.sum()
    if leftover > 0:
        last_idx = min(max(i, 1), length) - 1
        decrements[last_idx] += leftover
    return decrements


def simulate_fuel_and_speed(segments):
    total_len = sum(seg["length"] for seg in segments)
    slope = _simulate_road_slope(max(total_len, 1))

    fuel_true_raw = []
    speed = []
    event_label = []
    road_type_list = []
    current_fuel = random.uniform(25, 50)
    idx = 0

    for seg in segments:
        state = seg["state"]
        length = seg["length"]

        if state == "Driving":
            v_profile = _generate_driving_speed_profile(seg["speed_type"], length)
            accel_profile = np.diff(v_profile, prepend=v_profile[0])
            seg_slope = slope[idx: idx + length]
            for k in range(length):
                v = v_profile[k]
                a = accel_profile[k]
                s = seg_slope[k] if k < len(seg_slope) else 0.0
                consumption = _physical_fuel_consumption(v, a, s)
                current_fuel = float(np.clip(current_fuel - consumption, 0, TANK_CAPACITY_L))
                fuel_true_raw.append(current_fuel)
                speed.append(float(v))
                event_label.append("Driving")
                road_type_list.append(seg["road_type"])

        elif state == "Idle":
            deltas = _generate_idle_fuel_curve(seg["idle_type"], length)
            for d in deltas:
                current_fuel = float(np.clip(current_fuel + d, 0, TANK_CAPACITY_L))
                fuel_true_raw.append(current_fuel)
                speed.append(0.0)
                event_label.append("Idle")
                road_type_list.append("stop")

        elif state == "Refuel":
            inc = _simulate_refuel_curve(length, seg["amount"])
            for d in inc:
                current_fuel = float(np.clip(current_fuel + d, 0, TANK_CAPACITY_L))
                fuel_true_raw.append(current_fuel)
                speed.append(0.0)
                event_label.append("Refuel")
                road_type_list.append("stop")

        else:  # Fuel Theft
            theft_type = seg.get("theft_type", "fast")
            dec = _simulate_theft_curve(length, seg["amount"], theft_type)
            for d in dec:
                current_fuel = float(np.clip(current_fuel - d, 0, TANK_CAPACITY_L))
                fuel_true_raw.append(current_fuel)
                speed.append(0.0)
                event_label.append("Fuel Theft")
                road_type_list.append("stop")

        idx += length

    fuel_true_raw = np.array(fuel_true_raw)
    fuel_true_quant = _apply_fuel_quantization(fuel_true_raw)
    return fuel_true_quant, np.array(speed), event_label, road_type_list


def _smooth_transitions(speed, window=3):
    speed = speed.copy()
    n = len(speed)
    if n < 3:
        return speed
    diffs = np.abs(np.diff(speed))
    boundary_idx = np.where(diffs > 15)[0]
    for b in boundary_idx:
        lo = max(0, b - window + 1)
        hi = min(n, b + window + 1)
        # chỉ làm mượt cục bộ bằng nội suy tuyến tính thay vì moving average
        if hi - lo > 1:
            speed[lo:hi] = np.linspace(speed[lo], speed[hi-1], hi - lo)
    return np.clip(speed, 0, 110)


def simulate_gps(segments, speed_array):
    n = len(speed_array)
    lat = np.zeros(n)
    lon = np.zeros(n)
    heading = np.zeros(n)

    cur_lat, cur_lon = START_LAT, START_LON
    cur_heading = random.uniform(0, 359)
    idx = 0

    gps_drift_lat = np.cumsum(np.random.normal(0, 2e-7, size=n))
    gps_drift_lon = np.cumsum(np.random.normal(0, 2e-7, size=n))

    loss_mask = np.zeros(n, dtype=bool)
    if n > 10:
        n_loss_windows = max(1, int(n * 0.004))
        loss_starts = np.random.choice(np.arange(2, n - 4), size=min(n_loss_windows, n - 6), replace=False)
        for s in loss_starts:
            w = random.randint(2, 4)
            loss_mask[s:s + w] = True

    n_jumps = max(1, int(n * 0.002))
    jump_idx = set(np.random.choice(n, size=min(n_jumps, n), replace=False))

    frozen_lat, frozen_lon = None, None
    in_loss = False

    for seg in segments:
        if idx >= n:
            break
        length = seg["length"]
        state = seg["state"]
        road_type = seg.get("road_type", "stop")

        if state != "Driving" or road_type == "stop":
            heading_deltas = np.zeros(length)
        elif road_type == "straight":
            heading_deltas = np.random.normal(0, 1.5, length)
        elif road_type == "turn_left":
            total_turn = -random.uniform(30, 90)
            heading_deltas = np.full(length, total_turn / length) + np.random.normal(0, 1, length)
        elif road_type == "turn_right":
            total_turn = random.uniform(30, 90)
            heading_deltas = np.full(length, total_turn / length) + np.random.normal(0, 1, length)
        elif road_type == "u_turn":
            total_turn = random.choice([-1, 1]) * random.uniform(160, 180)
            heading_deltas = np.full(length, total_turn / length) + np.random.normal(0, 2, length)
        else:
            heading_deltas = np.random.normal(0, 1.5, length)

        for i in range(length):
            if idx >= n:
                break
            v = speed_array[idx]
            cur_heading = (cur_heading + heading_deltas[i]) % 360
            heading[idx] = cur_heading

            if v > 0:
                distance_km = v * (SAMPLE_INTERVAL_SEC / 3600.0)
                rad = np.radians(cur_heading)
                d_lat = (distance_km / 111.0) * np.cos(rad)
                d_lon = (distance_km / (111.0 * np.cos(np.radians(cur_lat)))) * np.sin(rad)
                d_lat += np.random.normal(0, 0.000005)
                d_lon += np.random.normal(0, 0.000005)
                cur_lat += d_lat
                cur_lon += d_lon
            else:
                cur_lat += np.random.normal(0, 0.0000008)
                cur_lon += np.random.normal(0, 0.0000008)

            out_lat, out_lon = cur_lat, cur_lon

            if loss_mask[idx]:
                if not in_loss:
                    frozen_lat = lat[idx - 1] if idx > 0 else cur_lat
                    frozen_lon = lon[idx - 1] if idx > 0 else cur_lon
                    in_loss = True
                out_lat, out_lon = frozen_lat, frozen_lon
            else:
                in_loss = False

            if idx in jump_idx:
                out_lat += random.choice([-1, 1]) * random.uniform(0.0008, 0.003)
                out_lon += random.choice([-1, 1]) * random.uniform(0.0008, 0.003)

            out_lat += gps_drift_lat[idx]
            out_lon += gps_drift_lon[idx]

            lat[idx] = out_lat
            lon[idx] = out_lon
            idx += 1

    return lat, lon, heading


def simulate_uphill_flag(n):
    elevation = np.cumsum(np.random.normal(0, 1, size=n))
    slope = np.diff(elevation, prepend=elevation[0])
    return slope > 0.3


# ====================================================================
# TẦNG 2: SENSOR MODEL (Tín hiệu RAW, không filter)
# ====================================================================

def _apply_sensor_lag(fuel_clean, alpha=SENSOR_LAG_ALPHA):
    """Độ trễ vật lý của cảm biến phao (EMA bậc 1)."""
    lagged = np.zeros_like(fuel_clean)
    lagged[0] = fuel_clean[0]
    for i in range(1, len(fuel_clean)):
        lagged[i] = alpha * fuel_clean[i] + (1 - alpha) * lagged[i-1]
    return lagged


def _apply_fuel_hysteresis(fuel_signal, band=0.15):
    diff = np.diff(fuel_signal, prepend=fuel_signal[0])
    offset = np.where(diff > 0, band / 2, np.where(diff < 0, -band / 2, 0.0))
    return fuel_signal + offset


def _apply_zone_calibration_bias(fuel_signal, capacity=TANK_CAPACITY_L):
    ratio = np.clip(fuel_signal, 0, capacity) / capacity
    bias = np.where(ratio < 0.15, -0.3, np.where(ratio > 0.85, 0.3, 0.0))
    return fuel_signal + bias


def simulate_fuel_sloshing(speed, heading, road_type_array):
    """
    Sloshing thực tế: hệ dao động tắt dần bậc 2, kích thích bởi gia tốc / đánh lái.
    """
    n = len(speed)
    slosh = np.zeros(n)
    accel = np.diff(speed, prepend=speed[0])          # km/h / mẫu
    heading_change = np.abs(np.diff(heading, prepend=heading[0]))
    heading_change = np.minimum(heading_change, 360 - heading_change)

    # Hệ số dao động: y[n] = 2*R*cos(w)*y[n-1] - R^2*y[n-2] + excitation
    w = SLOSH_NATURAL_FREQ
    R = SLOSH_DECAY
    b1 = 2 * R * np.cos(w)
    b2 = - R**2

    # Trạng thái bộ lọc
    y_prev1 = 0.0
    y_prev2 = 0.0

    for i in range(n):
        # Excitation tổng hợp từ gia tốc dọc và thay đổi heading
        road = road_type_array[i] if road_type_array and i < len(road_type_array) else "straight"
        factor = SLOSH_ROAD_FACTOR.get(road, 1.0)

        excite = (
            SLOSH_EXCITATION_GAIN * abs(accel[i]) * factor
            + SLOSH_TURN_GAIN * (heading_change[i] / 45.0) * factor
        )

        # Damped oscillator
        y_new = b1 * y_prev1 + b2 * y_prev2 + excite
        # Thêm một ít nhiễu nền
        slosh[i] = y_new + np.random.normal(0, 0.003)

        y_prev2 = y_prev1
        y_prev1 = y_new

    return slosh


def simulate_sensor_drift(n):
    drift = np.cumsum(np.random.normal(0, DRIFT_SIGMA, size=n))
    return np.clip(drift, -1.5, 1.5)


def inject_transient_fluctuations(fuel_raw, event_label):
    fuel_raw = fuel_raw.copy()
    n = len(fuel_raw)
    labels = np.array(event_label)
    eligible_idx = np.where((labels == "Driving") | (labels == "Idle"))[0]
    if len(eligible_idx) == 0:
        return fuel_raw

    n_bursts = max(1, int(len(eligible_idx) * TRANSIENT_RATE))
    starts = np.random.choice(eligible_idx, size=min(n_bursts, len(eligible_idx)), replace=False)

    for start in starts:
        length = random.randint(*TRANSIENT_LEN)
        end = min(start + length, n)
        if end <= start:
            continue
        amp = random.uniform(*TRANSIENT_AMP) * random.choice([1, -1])
        # dạng bump tăng rồi giảm
        bump = amp * np.sin(np.linspace(0, np.pi, end - start))
        fuel_raw[start:end] += bump

    return fuel_raw


def add_fuel_sensor_noise(fuel_clean, speed, heading, uphill_flag, road_type_array):
    """
    Từ fuel_true đã lượng tử -> thêm đặc tính cảm biến (lag, hysteresis, bias)
    và nhiễu động (sloshing, phanh, rẽ, dốc).
    KHÔNG có bộ lọc làm mượt.
    """
    n = len(fuel_clean)

    # Trễ cảm biến (vật lý)
    fuel_sensor = _apply_sensor_lag(fuel_clean)
    # Hysteresis
    fuel_sensor = _apply_fuel_hysteresis(fuel_sensor)
    # Sai số calibration
    fuel_sensor = _apply_zone_calibration_bias(fuel_sensor)

    fuel_raw = fuel_sensor.copy()

    # Sloshing thực tế
    fuel_raw += simulate_fuel_sloshing(speed, heading, road_type_array)

    # Phanh / tăng tốc mạnh (nhiễu nhẹ, nằm trong biên độ cho phép)
    speed_diff = np.diff(speed, prepend=speed[0])
    brake_mask = speed_diff < -12
    accel_mask = speed_diff > 12
    fuel_raw[brake_mask] += np.random.normal(0, 0.06, size=int(brake_mask.sum()))
    fuel_raw[accel_mask] += np.random.normal(0, 0.04, size=int(accel_mask.sum()))

    # Rẽ gấp
    heading_diff = np.abs(np.diff(heading, prepend=heading[0]))
    heading_diff = np.minimum(heading_diff, 360 - heading_diff)
    turning_mask = heading_diff > 30
    fuel_raw[turning_mask] += np.random.normal(0, 0.04, size=int(turning_mask.sum()))

    # Lên dốc
    fuel_raw += np.where(uphill_flag, np.random.uniform(0.01, 0.05, size=n), 0.0)

    fuel_raw = np.clip(fuel_raw, -2.0, TANK_CAPACITY_L + 2.0)
    return fuel_raw


def inject_outliers(fuel_raw):
    """Outlier mô phỏng lỗi ADC."""
    fuel_raw = fuel_raw.copy()
    n = len(fuel_raw)

    # Điểm đơn lẻ
    n_point = max(1, int(round(n * OUTLIER_POINT_RATE)))
    idx = np.random.choice(n, size=n_point, replace=False)
    signs = np.random.choice([1, -1], size=n_point)
    magnitudes = np.random.uniform(*OUTLIER_POINT_AMP, size=n_point)
    fuel_raw[idx] += signs * magnitudes

    # Cụm ngắn 2-3 mẫu
    n_burst = max(1, int(round(n * OUTLIER_BURST_RATE)))
    starts = np.random.choice(n, size=n_burst, replace=False)
    for s in starts:
        length = random.randint(*OUTLIER_BURST_LEN)
        e = min(s + length, n)
        sign = random.choice([1, -1])
        mag = random.uniform(*OUTLIER_BURST_AMP)
        fuel_raw[s:e] += sign * mag

    fuel_raw = np.clip(fuel_raw, -5.0, TANK_CAPACITY_L + 5.0)
    return fuel_raw


def _fuel_ratio_to_adc_nonlinear(fuel_ratio):
    base = np.interp(fuel_ratio, _CALIB_FUEL_RATIO, _CALIB_ADC)
    curvature = 40.0 * np.sin(fuel_ratio * np.pi)
    return base + curvature


def simulate_adc_and_voltage(fuel_raw, n):
    voltage_drift = np.cumsum(np.random.normal(0, 0.0004, size=n))
    voltage_drift = np.clip(voltage_drift, -0.4, 0.4)
    long_term_wave = 0.05 * np.sin(np.linspace(0, 3 * np.pi, n))
    voltage_base = NOMINAL_VOLTAGE + voltage_drift + long_term_wave
    voltage_noise = np.random.normal(0, 0.05, size=n)
    voltage = voltage_base + voltage_noise

    n_spikes = max(1, n // 200)
    spike_idx = np.random.choice(n, size=n_spikes, replace=False)
    voltage[spike_idx] += np.random.uniform(-0.3, 0.3, size=n_spikes)
    voltage = np.clip(voltage, VOLTAGE_MIN, VOLTAGE_MAX)

    fuel_ratio = np.clip(fuel_raw, 0, TANK_CAPACITY_L) / TANK_CAPACITY_L
    adc_base = _fuel_ratio_to_adc_nonlinear(fuel_ratio)
    sensor_noise = np.random.normal(0, 15, size=n)

    adc_voltage_coupling = (voltage - NOMINAL_VOLTAGE) * 120.0
    adc_raw = np.clip(adc_base + sensor_noise + adc_voltage_coupling, ADC_MIN, ADC_MAX)

    return adc_raw, voltage


# ====================================================================
# TẦNG 3: OUTPUT & THỐNG KÊ
# ====================================================================

def inject_missing_and_duplicate_samples(df):
    df = df.copy()
    n = len(df)

    n_dup = max(1, int(round(n * 0.005)))
    dup_positions = sorted(np.random.choice(df.index, size=n_dup, replace=False), reverse=True)
    rows = df.to_dict("records")
    for pos in dup_positions:
        rows.insert(pos + 1, dict(rows[pos]))
    df = pd.DataFrame(rows)

    n2 = len(df)
    n_missing = max(1, int(round(n2 * 0.005)))
    missing_idx = np.random.choice(df.index, size=n_missing, replace=False)
    df = df.drop(index=missing_idx).reset_index(drop=True)

    return df


def validate_dataset(df):
    errors = []
    if df["adc_raw"].isnull().any() or df["voltage"].isnull().any() or df["fuel_raw"].isnull().any():
        errors.append("Tồn tại giá trị NaN trong các cột số")
    if df["adc_raw"].min() < ADC_MIN - 50 or df["adc_raw"].max() > ADC_MAX + 50:
        errors.append("adc_raw vượt khoảng cho phép")
    if df["voltage"].min() < VOLTAGE_MIN - 0.2 or df["voltage"].max() > VOLTAGE_MAX + 0.2:
        errors.append("voltage vượt khoảng cho phép")
    if df["fuel_raw"].min() < -6 or df["fuel_raw"].max() > TANK_CAPACITY_L + 6:
        errors.append("fuel_raw vượt khoảng bất thường quá mức")
    if df["speed"].min() < 0 or df["speed"].max() > 120:
        errors.append("speed vượt khoảng cho phép")
    invalid_labels = set(df["event_label"].unique()) - set(EVENT_STATES)
    if invalid_labels:
        errors.append(f"event_label không hợp lệ: {invalid_labels}")
    if df["device_id"].isnull().any() or df["vehicle_id"].isnull().any():
        errors.append("device_id/vehicle_id bị thiếu")

    if errors:
        print("⚠ Cảnh báo kiểm tra dữ liệu:")
        for e in errors:
            print("   -", e)
    else:
        print("✔ Dữ liệu đã qua kiểm tra hợp lệ (trong ngưỡng vật lý cho phép).")
    return len(errors) == 0


def generate_report(df):
    print("\n========== BÁO CÁO CHẤT LƯỢNG DỮ LIỆU ==========")
    total = len(df)
    event_counts = df["event_label"].value_counts()
    print(f"Tổng số mẫu: {total}")
    for state in EVENT_STATES:
        cnt = event_counts.get(state, 0)
        print(f"  {state:12s}: {cnt:5d} ({cnt/total:.1%})")

    print("\n--- Độ dài các đoạn sự kiện (mẫu) ---")
    seg_lengths = []
    current_label = None
    current_len = 0
    for label in df["event_label"]:
        if label == current_label:
            current_len += 1
        else:
            if current_label is not None:
                seg_lengths.append((current_label, current_len))
            current_label = label
            current_len = 1
    if current_label is not None:
        seg_lengths.append((current_label, current_len))
    seg_df = pd.DataFrame(seg_lengths, columns=["event", "length"])
    for state in EVENT_STATES:
        subset = seg_df[seg_df["event"] == state]["length"]
        if len(subset) > 0:
            print(f"  {state:12s}: avg={subset.mean():.1f}, min={subset.min()}, max={subset.max()}, std={subset.std():.1f}")
        else:
            print(f"  {state:12s}: không có")

    cols = ["fuel_raw", "speed", "adc_raw", "voltage"]
    print("\n--- Thống kê biến số ---")
    for col in cols:
        s = df[col]
        print(f"  {col:10s}: mean={s.mean():.3f}, std={s.std():.3f}, min={s.min():.3f}, max={s.max():.3f}")

    df["fuel_delta"] = df["fuel_raw"].diff().abs()
    print("\n--- Dao động nhiên liệu trung bình (|Δfuel|) theo trạng thái ---")
    for state in EVENT_STATES:
        mask = df["event_label"] == state
        if mask.sum() > 1:
            avg_delta = df.loc[mask, "fuel_delta"].mean()
            print(f"  {state:12s}: {avg_delta:.4f} L/mẫu")
        else:
            print(f"  {state:12s}: không đủ dữ liệu")

    print("\n--- Tốc độ thay đổi fuel (L/mẫu) ---")
    for state in ["Driving", "Idle", "Refuel", "Fuel Theft"]:
        mask = df["event_label"] == state
        if mask.sum() > 1:
            diffs = df.loc[mask, "fuel_raw"].diff()
            if state in ["Refuel"]:
                avg_change = diffs[diffs > 0].mean()
                print(f"  {state:12s}: tăng TB = {avg_change:.4f} L/mẫu (khi nạp)")
            elif state == "Fuel Theft":
                avg_change = diffs[diffs < 0].mean()
                print(f"  {state:12s}: giảm TB = {avg_change:.4f} L/mẫu (khi rút)")
            else:
                avg_change = diffs.mean()
                print(f"  {state:12s}: thay đổi TB = {avg_change:.4f} L/mẫu")
        else:
            print(f"  {state:12s}: không đủ dữ liệu")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        ax = axes[0, 0]
        ax2 = ax.twinx()
        ax.plot(df.index, df["fuel_raw"], 'b-', alpha=0.7, label='fuel_raw')
        ax2.plot(df.index, df["speed"], 'r-', alpha=0.5, label='speed')
        ax.set_title("Fuel_raw & Speed theo thời gian")
        ax.set_xlabel("Mẫu")
        ax.set_ylabel("Fuel (L)", color='b')
        ax2.set_ylabel("Speed (km/h)", color='r')

        axes[0, 1].hist(df["fuel_raw"], bins=30, color='steelblue', edgecolor='black')
        axes[0, 1].set_title("Phân bố fuel_raw")

        bp_data = [df.loc[df["event_label"] == s, "fuel_delta"].dropna() for s in EVENT_STATES]
        bp = axes[0, 2].boxplot(bp_data)
        axes[0, 2].set_xticklabels(EVENT_STATES)
        axes[0, 2].set_title("Phân bố |Δfuel| theo trạng thái")
        axes[0, 2].set_ylabel("|Δfuel| (L)")

        axes[1, 0].hist(df["fuel_delta"].dropna(), bins=50, color='darkorange', edgecolor='black')
        axes[1, 0].set_title("Histogram của Δfuel (mẫu)")

        axes[1, 1].scatter(df["speed"], df["fuel_delta"], c=df["speed"], cmap='viridis', alpha=0.5, s=10)
        axes[1, 1].set_title("Tốc độ vs Biên độ dao động fuel")
        axes[1, 1].set_xlabel("Speed (km/h)")
        axes[1, 1].set_ylabel("|Δfuel| (L)")

        state_duration = seg_df.groupby("event")["length"].mean()
        axes[1, 2].bar(state_duration.index, state_duration.values,
                       color=['blue','green','orange','red'])
        axes[1, 2].set_title("Độ dài trung bình mỗi sự kiện (mẫu)")
        axes[1, 2].set_ylabel("Số mẫu")

        plt.tight_layout()
        fig.savefig("fuel_data_quality_report.png")
        print("\n✔ Đã lưu biểu đồ chất lượng dữ liệu: fuel_data_quality_report.png")
    except Exception as e:
        print(f"\n⚠ Không thể tạo biểu đồ (thiếu matplotlib hoặc lỗi): {e}")


# ====================================================================
# HÀM MAIN
# ====================================================================
def main():
    n = N_SAMPLES

    # ===== TẦNG 1: VẬT LÝ XE =====
    segments = generate_event_segments(n)

    fuel_true, speed_true, event_label, road_type_array = simulate_fuel_and_speed(segments)
    fuel_true = fuel_true[:n]
    speed_true = speed_true[:n]
    event_label = event_label[:n]
    road_type_array = road_type_array[:n]

    speed = _smooth_transitions(speed_true)

    lat, lon, heading = simulate_gps(segments, speed)
    uphill_flag = simulate_uphill_flag(n)

    # ===== TẦNG 2: CẢM BIẾN RAW =====
    fuel_sensor = add_fuel_sensor_noise(fuel_true, speed, heading, uphill_flag, road_type_array)
    fuel_sensor = fuel_sensor + simulate_sensor_drift(n)
    fuel_sensor = inject_transient_fluctuations(fuel_sensor, event_label)
    fuel_raw = inject_outliers(fuel_sensor)        # raw, không filter

    adc_raw, voltage = simulate_adc_and_voltage(fuel_raw, n)

    # ===== TẦNG 3: OUTPUT =====
    timestamps = [START_TIME + timedelta(seconds=SAMPLE_INTERVAL_SEC * i) for i in range(n)]

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "device_id": DEVICE_ID,
            "vehicle_id": VEHICLE_ID,
            "adc_raw": np.round(adc_raw, 1),
            "voltage": np.round(voltage, 3),
            "fuel_raw": np.round(fuel_raw, 3),
            "speed": np.round(speed, 1),
            "latitude": np.round(lat, 6),
            "longitude": np.round(lon, 6),
            "heading": np.round(heading, 1),
            "event_label": event_label,
        }
    )

    df = inject_missing_and_duplicate_samples(df)

    validate_dataset(df)

    output_path = "fuel_sensor_dataset.csv"
    df.to_csv(output_path, index=False)

    generate_report(df)

    total = len(df)
    n_driving = int((df["event_label"] == "Driving").sum())
    n_idle = int((df["event_label"] == "Idle").sum())
    n_refuel = int((df["event_label"] == "Refuel").sum())
    n_theft = int((df["event_label"] == "Fuel Theft").sum())

    print("\n===== THỐNG KÊ DỮ LIỆU =====")
    print(f"Tổng số mẫu (dòng)   : {total}")
    print(f"Số mẫu Driving       : {n_driving} ({n_driving / total:.1%})")
    print(f"Số mẫu Idle          : {n_idle} ({n_idle / total:.1%})")
    print(f"Số mẫu Refuel        : {n_refuel} ({n_refuel / total:.1%})")
    print(f"Số mẫu Fuel Theft    : {n_theft} ({n_theft / total:.1%})")
    print(f"Đã lưu file          : {output_path}")


if __name__ == "__main__":
    main()