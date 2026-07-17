import threading
import time
import random
import numpy as np
from datetime import datetime

class SensorSimulator:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            raise RuntimeError("SensorSimulator is a singleton. Only one instance allowed.")
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SensorSimulator, cls).__new__(cls)
        return cls._instance

    def __init__(self, vehicle_id="VN001", sample_interval=1.0, log_file="log.txt"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        self.vehicle_id = vehicle_id
        self.sample_interval = sample_interval
        self.lock = threading.Lock()
        self.data_history = []
        self.events = []

        # Mở file log mới, xóa cũ
        self.log_file = log_file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp\tVehicleID\tState\tFuel\tSpeed\tLat\tLng\n")

        # Chuỗi trạng thái lặp vĩnh viễn
        self.state_sequence = [
            "Idle",         # 0
            "Driving",      # 1
            "Idle",         # 2 (trước Refuel)
            "Refuel",       # 3
            "Driving",      # 4
            "Fuel Theft",   # 5
            "Idle",         # 6 (sau Theft)
            "Driving"       # 7
        ]
        self.state_index = 0
        self.current_state = self.state_sequence[self.state_index]
        self.state_start_time = datetime.now()
        self.state_duration_target = self._get_duration_for_state("Idle")

        # Biến vật lý
        self.current_fuel = 50.0          # lít
        self.current_speed = 0.0          # km/h
        self.target_speed = 0.0
        self.current_lat = 21.0285
        self.current_lng = 105.8541

        # Tham số từng trạng thái (Đã chỉnh sửa: noise_fuel và noise_speed của Idle = 0)
        self.state_params = {
            "Idle": {
                "speed_target": 0,
                "fuel_rate": 0.0,
                "noise_fuel": 0.0,         # Đã tắt noise fuel để Idle nằm ngang tuyệt đối
                "noise_speed": 0.0         # Đã tắt noise speed để xe đứng yên tuyệt đối
            },
            "Driving": {
                "speed_target": (45, 55),
                "fuel_rate": -0.06,
                "noise_fuel": 0.002,
                "noise_speed": 1.0
            },
            "Refuel": {
                "speed_target": 0,
                "fuel_rate": 0.45,
                "noise_fuel": 0.002,
                "noise_speed": 0.1
            },
            "Fuel Theft": {
                "speed_target": 0,
                "fuel_rate": -1.2,
                "noise_fuel": 0.002,
                "noise_speed": 0.1
            }
        }

        # Ghi log khởi động
        self._write_log(f"Simulator started with VehicleID={vehicle_id}")
        self._write_log(f"Initial state: {self.current_state}, duration={self.state_duration_target:.1f}s")

        self.thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.thread.start()

    def _get_duration_for_state(self, state):
        if state == "Idle":
            return random.uniform(8, 15)
        elif state == "Driving":
            return random.uniform(35, 50)
        elif state == "Refuel":
            return random.uniform(15, 25)
        elif state == "Fuel Theft":
            return random.uniform(12, 18)
        return 20

    def _choose_next_state(self):
        self.state_index = (self.state_index + 1) % len(self.state_sequence)
        return self.state_sequence[self.state_index]

    def _enter_state(self, new_state):
        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = datetime.now()
        self.state_duration_target = self._get_duration_for_state(new_state)

        params = self.state_params[new_state]
        if new_state == "Driving":
            low, high = params["speed_target"]
            self.target_speed = random.uniform(low, high)
            if self.current_speed < 5:
                self.current_speed = random.uniform(5, 15)
        else:
            self.target_speed = 0.0
            if self.current_speed > 1:
                self.current_speed *= 0.3

        self._write_log(f"STATE CHANGE: {old_state} -> {new_state} (duration: {self.state_duration_target:.1f}s)")

    def _write_log(self, message):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")

    def _log_data(self, row):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\t"
                    f"{row['VehicleID']}\t{row['State']}\t{row['Fuel']}\t"
                    f"{row['Speed']:.1f}\t{row['Latitude']:.6f}\t{row['Longitude']:.6f}\n")

    def _run_simulation(self):
        while True:
            loop_start = time.perf_counter()

            # 1. Chuyển trạng thái nếu hết thời gian
            elapsed = (datetime.now() - self.state_start_time).total_seconds()
            if elapsed >= self.state_duration_target:
                next_state = self._choose_next_state()
                self._enter_state(next_state)
                elapsed = 0.0

            # 2. Sinh dữ liệu
            params = self.state_params[self.current_state]
            fuel_rate = params["fuel_rate"]
            fuel_noise_std = params["noise_fuel"]

            # SỬA: Chỉ Refuel mới có Plateau ở 30% cuối (Fuel Theft phải giảm liên tục đến hết)
            progress = elapsed / self.state_duration_target if self.state_duration_target > 0 else 0
            if self.current_state == "Refuel" and progress > 0.7:
                fuel_rate = 0.0
                fuel_noise_std = 0.0

            fuel_noise = np.random.normal(0, fuel_noise_std)
            self.current_fuel += fuel_rate * self.sample_interval + fuel_noise
            self.current_fuel = max(5.0, min(80.0, self.current_fuel))

            # Tốc độ
            speed_noise = np.random.normal(0, params.get("noise_speed", 0.5))
            self.current_speed += (self.target_speed - self.current_speed) * 0.2 + speed_noise
            if self.current_state == "Driving":
                self.current_speed = max(5.0, min(70.0, self.current_speed))
            else:
                self.current_speed = max(0.0, min(5.0, self.current_speed))
                if elapsed > 2.0:
                    self.current_speed = 0.0

            # GPS
            if self.current_state == "Driving" and self.current_speed > 2:
                heading = random.uniform(0, 360)
                rad = np.radians(heading)
                v_ms = self.current_speed / 3.6
                dist = v_ms * self.sample_interval
                dlat = dist * np.cos(rad) / 111000.0
                dlng = dist * np.sin(rad) / (111000.0 * np.cos(np.radians(self.current_lat)))
                self.current_lat += dlat
                self.current_lng += dlng
            else:
                self.current_lat += np.random.normal(0, 0.00001)
                self.current_lng += np.random.normal(0, 0.00001)

            adc = int(1000 + self.current_fuel * 15 + np.random.randint(-5, 5))

            timestamp = datetime.now()
            raw_row = {
                "Timestamp": timestamp,
                "VehicleID": self.vehicle_id,
                "State": self.current_state,
                "Fuel": round(self.current_fuel, 2),
                "Speed": round(self.current_speed, 1),
                "Latitude": round(self.current_lat, 6),
                "Longitude": round(self.current_lng, 6),
                "ADC": adc
            }

            # Ghi log dữ liệu
            self._log_data(raw_row)

            with self.lock:
                self.data_history.append(raw_row)
                if len(self.data_history) > 200:
                    self.data_history.pop(0)

            # Duy trì đúng sample interval
            elapsed_time = time.perf_counter() - loop_start
            time.sleep(max(0, self.sample_interval - elapsed_time))

    # Các getter giữ nguyên
    def get_latest_raw(self):
        with self.lock:
            return self.data_history[-1] if self.data_history else None

    def get_history_raw(self, limit=100):
        with self.lock:
            return self.data_history[-limit:]

    def get_events(self):
        with self.lock:
            return self.events