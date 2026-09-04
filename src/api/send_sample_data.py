#!/usr/bin/env python3
"""
send_sample_data.py – Mô phỏng dữ liệu realtime cho Car 2 với NHIỄU MẠNH
Mục đích: Kiểm tra khả năng lọc nhiễu gai (spike) của AdaptiveNoiseFilter
"""

import time
import sys
import numpy as np
import requests
from datetime import datetime, timedelta
import random

# ============================================================
# CẤU HÌNH
# ============================================================
API_URL = "http://127.0.0.1:8000/api/predict"
CAR_ID = "Car 2"
POINT_INTERVAL_S = 300          # 5 phút / điểm
SEND_INTERVAL_S = 2             # 2 giây gửi 1 điểm

# ------------------------------
# Tham số vật lý
# ------------------------------
TANK_CAPACITY = 450.0
IDLE_FUEL_RATE = 1.5 / 3600      # tiêu hao khi nổ máy không tải (L/s)
DRIVING_FUEL_RATE = 12.0 / 3600  # tiêu hao khi chạy (L/s)
AVG_SPEED_KMH = 45.0
AVG_SPEED_MS = AVG_SPEED_KMH / 3.6
GPS_STEP_DEG = (AVG_SPEED_MS * POINT_INTERVAL_S) / 111000.0

# ------------------------------
# CẤU HÌNH NHIỄU (TĂNG CƯỜNG)
# ------------------------------
NOISE_CONFIG = {
    'gaussian': {
        'enabled': True,
        'fuel_std': 0.8,          # giữ nguyên
        'speed_std': 5.0,
    },
    'outlier': {
        'enabled': True,
        'probability': 0.05,      # tăng lên 5%
        'fuel_magnitude': (15.0, 30.0),  # biên độ lớn hơn
        'speed_magnitude': (15.0, 35.0),
    },
    'dropout': {
        'enabled': False,
        'probability': 0.02,
    },
    'drift': {
        'enabled': True,
        'fuel_rate': 0.02,
        'speed_rate': 0.5,
        'max_drift': 3.0,
    },
    'quantization': {
        'enabled': True,
        'fuel_step': 1.0,
        'speed_step': 2.0,
    }
}

# ============================================================
# HÀM TẠO NHIỄU
# ============================================================

class NoiseGenerator:
    def __init__(self, config):
        self.config = config
        self.drift_fuel = 0.0
        self.drift_speed = 0.0
        self.point_count = 0
        # Đếm số điểm để chèn spike cố định
        self.global_point_count = 0
        
    def add_noise(self, fuel, speed):
        """Thêm nhiễu vào dữ liệu, bao gồm cả spike cố định.
        Lưu ý: nếu speed = 0 (xe đứng yên) thì không thêm nhiễu tốc độ."""
        self.point_count += 1
        self.global_point_count += 1
        fuel_noisy = fuel
        speed_noisy = speed
        
        # Lưu giá trị speed gốc để kiểm tra xe có đứng yên không
        speed_original = speed
        
        # 1. Gaussian noise
        if self.config['gaussian']['enabled']:
            fuel_noisy += np.random.normal(0, self.config['gaussian']['fuel_std'])
            # Chỉ thêm nhiễu tốc độ nếu xe đang di chuyển
            if speed_original > 0:
                speed_noisy += np.random.normal(0, self.config['gaussian']['speed_std'])
        
        # 2. Outlier ngẫu nhiên
        if self.config['outlier']['enabled']:
            if random.random() < self.config['outlier']['probability']:
                fuel_mag = random.uniform(*self.config['outlier']['fuel_magnitude'])
                fuel_noisy += fuel_mag if random.random() > 0.5 else -fuel_mag
                
                # Chỉ thêm outlier tốc độ nếu xe đang di chuyển
                if speed_original > 0:
                    speed_mag = random.uniform(*self.config['outlier']['speed_magnitude'])
                    speed_noisy += speed_mag if random.random() > 0.5 else -speed_mag
                print(f"  ⚡ OUTLIER: fuel={fuel_noisy:.1f}L, speed={speed_noisy:.1f}km/h")
        
        # 3. SPIKE CỐ ĐỊNH (thêm vào để kiểm tra filter)
        # Cứ 5 điểm lại xuất hiện một spike lớn ±20L
        if self.global_point_count % 5 == 0:
            spike_val = random.choice([-25.0, 25.0, -20.0, 20.0])
            fuel_noisy += spike_val
            print(f"  💥 SPIKE CỐ ĐỊNH: fuel={fuel_noisy:.1f}L (Δ={spike_val:+.1f}L)")
        
        # 4. Drift
        if self.config['drift']['enabled']:
            self.drift_fuel += random.uniform(-self.config['drift']['fuel_rate'], 
                                              self.config['drift']['fuel_rate'])
            self.drift_fuel = np.clip(self.drift_fuel, 
                                     -self.config['drift']['max_drift'],
                                     self.config['drift']['max_drift'])
            fuel_noisy += self.drift_fuel
            
            # Drift tốc độ chỉ khi xe đang di chuyển
            if speed_original > 0:
                self.drift_speed += random.uniform(-self.config['drift']['speed_rate'],
                                                   self.config['drift']['speed_rate'])
                self.drift_speed = np.clip(self.drift_speed,
                                          -self.config['drift']['max_drift'] * 5,
                                          self.config['drift']['max_drift'] * 5)
                speed_noisy += self.drift_speed
        
        # 5. Quantization
        if self.config['quantization']['enabled']:
            fuel_noisy = round(fuel_noisy / self.config['quantization']['fuel_step']) * self.config['quantization']['fuel_step']
            speed_noisy = round(speed_noisy / self.config['quantization']['speed_step']) * self.config['quantization']['speed_step']
        
        # Nếu xe đứng yên (speed_original = 0) thì bất kể nhiễu thế nào cũng giữ speed = 0
        if speed_original == 0:
            speed_noisy = 0.0
        
        # Đảm bảo giá trị hợp lý
        fuel_noisy = max(0, min(TANK_CAPACITY, fuel_noisy))
        speed_noisy = max(0, speed_noisy)
        
        return fuel_noisy, speed_noisy
    
    def reset(self):
        self.drift_fuel = 0.0
        self.drift_speed = 0.0
        self.point_count = 0
        # Không reset global_point_count để spike duy trì xuyên suốt

# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

def get_start_point():
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
                "latitude": float(last_row.get("latitude", 21.0285)),
                "longitude": float(last_row.get("longitude", 105.8541)),
            }
    except:
        pass
    return {
        "timestamp": datetime.now(),
        "fuel": 200.0,
        "latitude": 21.0285,
        "longitude": 105.8541,
    }

def make_point(ts, fuel, speed, lat, lon):
    if isinstance(ts, datetime):
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    else:
        ts_str = str(ts)
    return {
        "car_id": CAR_ID,
        "timestamp": ts_str,
        "fuel": round(float(fuel), 2),
        "speed": round(float(speed), 2),
        "latitude": round(float(lat), 6),
        "longitude": round(float(lon), 6),
    }

def send_point_with_log(p, noise_info=""):
    try:
        resp = requests.post(API_URL, json=p, timeout=10)
        data = resp.json()
        status = data.get("status") or data.get("tracker_status", "?")
        record = data.get("record", {})
        confidence = record.get("confidence", 0.0)
        pred_label = record.get("prediction") or record.get("predicted_label") or "?"
        print(f"  fuel={p['fuel']:6.1f}L | speed={p['speed']:5.1f} | "
              f"pred={str(pred_label):8s} | status={str(status):10s} | conf={float(confidence):5.1f}% | "
              f"{noise_info}")
        return data
    except requests.RequestException as e:
        print(f"  ❌ Lỗi: {e}")
        return None

def send_points_with_noise(points, noise_gen, prefix=""):
    total = len(points)
    noised_count = 0
    print(f"\n{prefix} (Tổng {total} điểm):")
    print("-" * 80)
    for i, p in enumerate(points, 1):
        if isinstance(p['timestamp'], datetime):
            ts = p['timestamp']
        else:
            try:
                ts = datetime.strptime(p['timestamp'], "%Y-%m-%d %H:%M:%S")
            except:
                ts = datetime.now()
        
        fuel_noisy, speed_noisy = noise_gen.add_noise(p['fuel'], p['speed'])
        
        is_noised = (abs(fuel_noisy - p['fuel']) > 0.1 or abs(speed_noisy - p['speed']) > 0.1)
        if is_noised:
            noised_count += 1
            noise_info = f"[NOISE Δfuel={fuel_noisy-p['fuel']:+.1f}L]"
        else:
            noise_info = "[CLEAN]"
        
        p_noisy = make_point(ts, fuel_noisy, speed_noisy, p['latitude'], p['longitude'])
        send_point_with_log(p_noisy, noise_info)
        time.sleep(SEND_INTERVAL_S)
    
    noise_gen.reset()
    print(f"\n📊 Thống kê segment: {noised_count}/{total} điểm bị nhiễu")
    print("-" * 80)

# ============================================================
# CÁC PHÂN ĐOẠN MÔ PHỎNG
# ============================================================

def segment_driving(start_ts, start_fuel, lat, lon, n=40):
    pts = []
    fuel = start_fuel
    for i in range(n):
        ts = start_ts + timedelta(seconds=POINT_INTERVAL_S * i)
        fuel -= DRIVING_FUEL_RATE * POINT_INTERVAL_S
        fuel = max(fuel, 0)
        speed = max(0, np.random.normal(AVG_SPEED_KMH, 5))
        lat_i = lat + i * GPS_STEP_DEG + np.random.uniform(-0.00003, 0.00003)
        lon_i = lon + i * GPS_STEP_DEG + np.random.uniform(-0.00003, 0.00003)
        pts.append({
            'timestamp': ts,
            'fuel': fuel,
            'speed': speed,
            'latitude': lat_i,
            'longitude': lon_i
        })
    return pts, ts + timedelta(seconds=POINT_INTERVAL_S * n), fuel, lat_i, lon_i

def segment_idle(start_ts, start_fuel, lat, lon, n=15):
    pts = []
    fuel = start_fuel
    for i in range(n):
        ts = start_ts + timedelta(seconds=POINT_INTERVAL_S * i)
        fuel -= IDLE_FUEL_RATE * POINT_INTERVAL_S
        fuel = max(fuel, 0)
        pts.append({
            'timestamp': ts,
            'fuel': fuel,
            'speed': 0,
            'latitude': lat,
            'longitude': lon
        })
    return pts, ts + timedelta(seconds=POINT_INTERVAL_S * n), fuel, lat, lon

def segment_refuel(start_ts, start_fuel, lat, lon, n=3, total_added=200.0):
    pts = []
    fuel = start_fuel
    actual_added = min(total_added, TANK_CAPACITY - start_fuel)
    if actual_added <= 0:
        return pts, start_ts, start_fuel, lat, lon
    step = actual_added / n
    for i in range(n):
        ts = start_ts + timedelta(seconds=POINT_INTERVAL_S * i)
        fuel += step
        pts.append({
            'timestamp': ts,
            'fuel': fuel,
            'speed': 0,          # xe dừng khi đổ xăng
            'latitude': lat,
            'longitude': lon
        })
    return pts, ts + timedelta(seconds=POINT_INTERVAL_S * n), fuel, lat, lon

def segment_theft(start_ts, start_fuel, lat, lon, n=4, total_lost=25.0):
    pts = []
    fuel = start_fuel
    step = total_lost / n
    for i in range(n):
        ts = start_ts + timedelta(seconds=POINT_INTERVAL_S * i)
        fuel -= step
        fuel = max(fuel, 0)
        pts.append({
            'timestamp': ts,
            'fuel': fuel,
            'speed': 0,          # xe dừng khi bị hút trộm
            'latitude': lat,
            'longitude': lon
        })
    return pts, ts + timedelta(seconds=POINT_INTERVAL_S * n), fuel, lat, lon

# ============================================================
# CHẠY MÔ PHỎNG
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print(f"🚛 MÔ PHỎNG REALTIME VỚI NHIỄU MẠNH: {CAR_ID}")
    print("=" * 80)
    
    print("\n📡 CẤU HÌNH NHIỄU (TĂNG CƯỜNG):")
    print(f"  - Gaussian: fuel_std={NOISE_CONFIG['gaussian']['fuel_std']}L, speed_std={NOISE_CONFIG['gaussian']['speed_std']}km/h")
    print(f"  - Outlier: prob={NOISE_CONFIG['outlier']['probability']*100:.0f}%, mag={NOISE_CONFIG['outlier']['fuel_magnitude']}L")
    print(f"  - Drift: rate={NOISE_CONFIG['drift']['fuel_rate']}L/point, max={NOISE_CONFIG['drift']['max_drift']}L")
    print(f"  - Quantization: fuel={NOISE_CONFIG['quantization']['fuel_step']}L, speed={NOISE_CONFIG['quantization']['speed_step']}km/h")
    print(f"  - SPIKE CỐ ĐỊNH: cứ 5 điểm xuất hiện ±20L (hoặc ±25L)")
    print(f"  - Khi xe dừng (speed=0): KHÔNG thêm nhiễu tốc độ, chỉ nhiễu fuel.")
    print()
    
    start = get_start_point()
    ts = start["timestamp"] + timedelta(minutes=1)
    fuel = start["fuel"]
    lat, lon = start["latitude"], start["longitude"]
    noise_gen = NoiseGenerator(NOISE_CONFIG)
    
    print(f"📍 Xuất phát: fuel={fuel:.1f}L")
    print(f"⏱  Mỗi điểm cách {POINT_INTERVAL_S}s (5 phút), gửi mỗi {SEND_INTERVAL_S}s")
    print("=" * 80)
    
    # --- Kịch bản ---
    pts, ts, fuel, lat, lon = segment_driving(ts, fuel, lat, lon, n=5)
    send_points_with_noise(pts, noise_gen, "▶ DRIVING (5 điểm) - Chuẩn bị vào trạm")
    
    pts, ts, fuel, lat, lon = segment_idle(ts, fuel, lat, lon, n=2)
    send_points_with_noise(pts, noise_gen, "▶ IDLE (2 điểm) - Dừng xe trước khi đổ xăng")
    
    pts, ts, fuel, lat, lon = segment_refuel(ts, fuel, lat, lon, n=3, total_added=200.0)
    if pts:
        send_points_with_noise(pts, noise_gen, "▶ REFUEL (3 điểm) - Đổ xăng 200L (xe đứng yên)")
    
    pts, ts, fuel, lat, lon = segment_driving(ts, fuel, lat, lon, n=35)
    send_points_with_noise(pts, noise_gen, "▶ DRIVING (35 điểm) - Chạy sau đổ xăng")
    
    pts, ts, fuel, lat, lon = segment_idle(ts, fuel, lat, lon, n=10)
    send_points_with_noise(pts, noise_gen, "▶ IDLE (10 điểm) - Dừng nghỉ (nổ máy chờ)")
    
    pts, ts, fuel, lat, lon = segment_refuel(ts, fuel, lat, lon, n=3, total_added=200.0)
    if pts:
        send_points_with_noise(pts, noise_gen, "▶ REFUEL (3 điểm) - Đổ xăng lần 2 (xe đứng yên)")
    
    pts, ts, fuel, lat, lon = segment_idle(ts, fuel, lat, lon, n=8)
    send_points_with_noise(pts, noise_gen, "▶ IDLE (8 điểm) - Dừng sau đổ xăng (có thể xe tắt máy hoặc nổ máy)")
    
    pts, ts, fuel, lat, lon = segment_theft(ts, fuel, lat, lon, n=4, total_lost=78.0)
    send_points_with_noise(pts, noise_gen, "▶ THEFT (4 điểm) - Hút trộm 78L, giảm đều 19.5L/điểm (xe đứng yên, ⚠️ QUAN SÁT KỸ)")
    
    pts, ts, fuel, lat, lon = segment_driving(ts, fuel, lat, lon, n=20)
    send_points_with_noise(pts, noise_gen, "▶ DRIVING (20 điểm) - Chạy tiếp")
    
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH MÔ PHỎNG VỚI NHIỄU MẠNH")
    print("=" * 80)
    print("\n📝 QUAN SÁT ĐẶC BIỆT:")
    print("  1. Filter có phát hiện và xử lý spike ±20L không?")
    print("  2. Có còn OUTLIER sau lọc không? (xem log FILTERED)")
    print("  3. Refuel/Theft có bị làm mờ quá mức không?")
    print("  4. Status có ổn định không?")
    print("  5. Khi xe dừng, tốc độ luôn = 0 dù có nhiễu.")
    print("=" * 80)