#!/usr/bin/env python3
"""
simulate_car2_demo_v2.py – Mô phỏng dữ liệu realtime cho Car 2 với các tham số thực tế hơn.
Kịch bản giữ nguyên cấu trúc: Driving → Idle → Refuel → Driving → Idle → Refuel → Idle → Theft → Driving.
"""

import time
import sys
import numpy as np
import requests
from datetime import datetime, timedelta

# ============================================================
# CẤU HÌNH
# ============================================================
API_URL = "http://127.0.0.1:5000/api/predict"
CAR_ID = "Car 2"
POINT_INTERVAL_S = 300          # 5 phút / điểm (thực tế)
SEND_INTERVAL_S = 2             # 2 giây gửi 1 điểm (giả lập nhanh)

# ------------------------------
# Tham số vật lý thực tế
# ------------------------------
TANK_CAPACITY = 450.0           # dung tích bình xăng (L)
IDLE_FUEL_RATE = 1.5 / 3600     # 1.5 L/h -> L/s
DRIVING_FUEL_RATE = 12.0 / 3600 # 12 L/h -> L/s (tiêu hao trung bình khi chạy)
AVG_SPEED_KMH = 45.0            # km/h
AVG_SPEED_MS = AVG_SPEED_KMH / 3.6
GPS_STEP_DEG = (AVG_SPEED_MS * POINT_INTERVAL_S) / 111000.0  # ~0.0056 độ

# ------------------------------
# Điểm xuất phát mặc định nếu không có file fusion
# ------------------------------
DEFAULT_LAT = 21.0285
DEFAULT_LON = 105.8541
DEFAULT_FUEL = 200.0

# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def get_start_point():
    """Lấy điểm cuối cùng của Car 2 từ file fusion, nếu không có thì dùng mặc định."""
    try:
        import pandas as pd
        df = pd.read_csv("data/processed/fusion/fusion_dataset.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        car_df = df[df["car_id"] == CAR_ID].sort_values("timestamp")
        if not car_df.empty:
            last_row = car_df.iloc[-1]
            return {
                "timestamp": last_row["timestamp"],
                "fuel": float(last_row["fuel"]),
                "latitude": float(last_row.get("latitude", DEFAULT_LAT)),
                "longitude": float(last_row.get("longitude", DEFAULT_LON)),
            }
    except Exception as e:
        print(f"⚠️  Không đọc được file fusion, dùng mặc định: {e}")
    return {
        "timestamp": datetime.now(),
        "fuel": DEFAULT_FUEL,
        "latitude": DEFAULT_LAT,
        "longitude": DEFAULT_LON,
    }

def make_point(ts, fuel, speed, lat, lon):
    return {
        "car_id": CAR_ID,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "fuel": round(float(fuel), 2),
        "speed": round(float(speed), 2),
        "latitude": round(float(lat), 6),
        "longitude": round(float(lon), 6),
    }

# ============================================================
# CÁC PHÂN ĐOẠN MÔ PHỎNG
# ============================================================
def segment_driving(start_ts, start_fuel, lat, lon, n=40):
    pts = []
    fuel = start_fuel
    for i in range(n):
        ts = start_ts + timedelta(seconds=POINT_INTERVAL_S * i)
        # Tiêu hao nhiên liệu theo thời gian thực
        fuel -= DRIVING_FUEL_RATE * POINT_INTERVAL_S + np.random.normal(0, 0.1)
        fuel = max(fuel, 0)
        speed = max(0, np.random.normal(AVG_SPEED_KMH, 10))
        # Di chuyển GPS thực tế
        lat_i = lat + i * GPS_STEP_DEG + np.random.uniform(-0.00005, 0.00005)
        lon_i = lon + i * GPS_STEP_DEG + np.random.uniform(-0.00005, 0.00005)
        pts.append(make_point(ts, fuel, speed, lat_i, lon_i))
    return pts, ts + timedelta(seconds=POINT_INTERVAL_S * n), fuel, lat_i, lon_i

def segment_idle(start_ts, start_fuel, lat, lon, n=15):
    pts = []
    fuel = start_fuel
    for i in range(n):
        ts = start_ts + timedelta(seconds=POINT_INTERVAL_S * i)
        fuel -= IDLE_FUEL_RATE * POINT_INTERVAL_S + np.random.uniform(-0.005, 0.005)
        fuel = max(fuel, 0)
        pts.append(make_point(ts, fuel, 0, lat, lon))
    return pts, ts + timedelta(seconds=POINT_INTERVAL_S * n), fuel, lat, lon

def segment_refuel(start_ts, start_fuel, lat, lon, n=3, total_added=200.0):
    pts = []
    fuel = start_fuel
    # Giới hạn không vượt quá dung tích bình
    actual_added = min(total_added, TANK_CAPACITY - start_fuel)
    if actual_added <= 0:
        print(f"⚠️  Bình đã đầy ({start_fuel:.1f}L), không thể đổ thêm.")
        return pts, start_ts, start_fuel, lat, lon
    step = actual_added / n
    for i in range(n):
        ts = start_ts + timedelta(seconds=POINT_INTERVAL_S * i)
        fuel += step + np.random.uniform(-0.5, 0.5)
        pts.append(make_point(ts, fuel, 0, lat, lon))
    return pts, ts + timedelta(seconds=POINT_INTERVAL_S * n), fuel, lat, lon

def segment_theft(start_ts, start_fuel, lat, lon, n=4, total_lost=25.0):
    pts = []
    fuel = start_fuel
    step = total_lost / n
    for i in range(n):
        ts = start_ts + timedelta(seconds=POINT_INTERVAL_S * i)
        fuel -= step + np.random.uniform(-0.3, 0.3)
        fuel = max(fuel, 0)
        speed = 0.0  # <--- SỬA Ở ĐÂY: Vận tốc = 0 để xe đứng yên khi bị trộm
        pts.append(make_point(ts, fuel, speed, lat, lon))
    return pts, ts + timedelta(seconds=POINT_INTERVAL_S * n), fuel, lat, lon

def send_points(points):
    total = len(points)
    for i, p in enumerate(points, 1):
        try:
            resp = requests.post(API_URL, json=p, timeout=10)
            data = resp.json()
            status = data.get("status") or data.get("tracker_status", "?")
            record = data.get("record", {})
            confidence = record.get("confidence", 0.0)
            if i % 5 == 0 or status in ("thinking", "confirmed"):
                print(f"  [{i}/{total}] fuel={p['fuel']:.1f}L | status={status} | conf={confidence:.1f}%")
        except requests.RequestException as e:
            print(f"  ❌ Lỗi điểm {i}: {e}", file=sys.stderr)
        time.sleep(SEND_INTERVAL_S)

# ============================================================
# CHẠY KỊCH BẢN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"🚛 MÔ PHỎNG REALTIME: {CAR_ID} (v2)")
    print("=" * 60)

    start = get_start_point()
    ts = start["timestamp"] + timedelta(minutes=1)
    fuel = start["fuel"]
    lat, lon = start["latitude"], start["longitude"]

    print(f"📍 Xuất phát: fuel={fuel:.1f}L, time={start['timestamp']}")
    print(f"⏱  Gửi mỗi {SEND_INTERVAL_S}s, mỗi điểm cách {POINT_INTERVAL_S}s (5 phút)")
    print(f"⛽ Dung tích bình: {TANK_CAPACITY}L, tiêu hao: ~{DRIVING_FUEL_RATE*3600:.1f} L/h (chạy), ~{IDLE_FUEL_RATE*3600:.1f} L/h (dừng)")
    print()

    # --- Kịch bản ---
    print("\n▶ DRIVING (5 điểm) - Chuẩn bị vào trạm...")
    pts, ts, fuel, lat, lon = segment_driving(ts, fuel, lat, lon, n=5)
    send_points(pts)

    print("\n▶ IDLE (2 điểm) - Dừng xe trước khi đổ xăng...")
    pts, ts, fuel, lat, lon = segment_idle(ts, fuel, lat, lon, n=2)
    send_points(pts)

    print(f"\n▶ REFUEL (3 điểm) - Đổ xăng nhanh (mục tiêu 200L)...")
    pts, ts, fuel, lat, lon = segment_refuel(ts, fuel, lat, lon, n=3, total_added=200.0)
    if pts:
        send_points(pts)

    print("\n▶ DRIVING (35 điểm) - Chạy sau đổ xăng (175 phút)...")
    pts, ts, fuel, lat, lon = segment_driving(ts, fuel, lat, lon, n=35)
    send_points(pts)

    print("\n▶ IDLE (10 điểm) - Dừng nghỉ (50 phút)...")
    pts, ts, fuel, lat, lon = segment_idle(ts, fuel, lat, lon, n=10)
    send_points(pts)

    print(f"\n▶ REFUEL (3 điểm) - Đổ xăng lần 2 (mục tiêu 200L)...")
    pts, ts, fuel, lat, lon = segment_refuel(ts, fuel, lat, lon, n=3, total_added=200.0)
    if pts:
        send_points(pts)

    print("\n▶ IDLE (8 điểm) - Dừng sau đổ xăng (40 phút)...")
    pts, ts, fuel, lat, lon = segment_idle(ts, fuel, lat, lon, n=8)
    send_points(pts)

    print("\n▶ THEFT (4 điểm) - Hút trộm nhiên liệu (20 phút)...")
    pts, ts, fuel, lat, lon = segment_theft(ts, fuel, lat, lon, n=4)
    send_points(pts)

    print("\n▶ DRIVING (20 điểm) - Chạy tiếp (100 phút)...")
    pts, ts, fuel, lat, lon = segment_driving(ts, fuel, lat, lon, n=20)
    send_points(pts)

    print("\n" + "=" * 60)
    print(f"✅ HOÀN THÀNH mô phỏng {CAR_ID}")
    print("=" * 60)