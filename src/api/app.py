#!/usr/bin/env python3
"""
FuelSentinel-AI Real-time Dashboard (AI backend)
--- V2: Adaptive Sliding Window Architecture ---

=====================================================================
[FIX] api_alerts(): get_active_summary() được gọi mà KHÔNG truyền
full_df -> tham số mặc định None -> toàn bộ khối tính
fuel_start/fuel_current/delta_fuel/duration_s bên trong hàm bị bỏ qua
(chỉ chạy khi full_df is not None) -> active_summary luôn rỗng dù
event đang có đầy đủ dữ liệu trong processed_buffer. Đây là lý do
popup toast (dùng /api/alerts) hiển thị "--"/0 trong khi dropdown
(dùng /api/toasts, được ghi log tại update() với buffer_df truyền
đúng) vẫn hiển thị số liệu chính xác. Fix: build full_df từ
processed_buffer giống hệt cách api_tracker_status() đã làm, rồi
truyền vào get_active_summary(full_df).
=====================================================================
"""
import os
import sys
import io
import json
import sqlite3
import threading
import logging
import traceback
from datetime import datetime
from collections import deque

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, jsonify, send_file, request

log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

logger = logging.getLogger('FuelSentinel')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler('data/logs/realtime.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(BASE_DIR, '..'))

from trainning.models.all_models import CNNGRUClassifier
from trainning.datasets.dataset_builder import convert_gps_to_relative
from feature_extraction.feature_extraction import build_segment_features_table

WINDOW_SIZE = 10
BUFFER_SIZE = 120
MAX_POINTS_HISTORY = 100
DEFAULT_VISIBLE_POINTS = 20
CANDIDATE_THRESH = 0.55
CONFIRM_THRESH = 0.75
CONFIRM_COUNT = 2
FINISH_THRESH = 0.5
MODEL_PATH = "outputs/final_model/cnn_gru_20260815_075909/final_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ['Driving', 'Idle', 'Refuel', 'Theft']
FUSION_CSV = "data/processed/fusion/fusion_dataset.csv"

CANDIDATE_MISMATCH_TOLERANCE = 1

# --- THÊM MỚI ---
FUEL_EVENT_MISMATCH_TOLERANCE = 0     # Refuel/Theft: sai 1 điểm là huỷ ngay, không "giữ niềm tin"
REFUEL_MIN_DELTA = 3.0                # L, ngưỡng tối thiểu để tin là refuel thật (chống nhiễu model)
THEFT_MIN_DELTA = 1.0                 # L, ngưỡng tối thiểu để tin là theft thật

IDLE_FUEL_SLOPE_RESET = 0.35

BACKTRACK_MAX_LOOKBACK = 20
BACKTRACK_FUEL_THRESHOLD = 0.2

EVENT_LOG_DIR = "data/logs"
EVENT_LOG_PATH = os.path.join(EVENT_LOG_DIR, "event_log.jsonl")
TOAST_LOG_PATH = os.path.join(EVENT_LOG_DIR, "toast_log.jsonl")
REALTIME_DB_PATH = os.path.join(EVENT_LOG_DIR, "realtime.db")
os.makedirs(EVENT_LOG_DIR, exist_ok=True)

FEATURE_COLS_21 = [
    "duration_min", "sample_count", "fuel_start", "fuel_end",
    "fuel_change", "fuel_mean", "fuel_std", "fuel_slope",
    "trend_r2", "trend_rmse", "max_drop", "max_rise",
    "drop_count", "rise_count", "fuel_range", "fuel_mad",
    "oscillation_count", "total_distance", "speed_mean",
    "speed_max", "speed_zero_ratio"
]

WINDOW_RANGES = {
    'normal':    (6, 8),
    'candidate': (4, 5),
    'Refuel':    (3, 4),
    'Theft':     (2, 4),
    'Idle':      (2, 7),      # sửa 4 → 2
    'Driving':   (5, 8),
}
IDLE_STEPS = [2, 4, 6, 8]
STABILITY_STD_THRESHOLD = 0.10

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

def make_json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    return obj

def _ts_to_iso(ts):
    return ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)

# ------------------------------------------------------------------
# Ghi log cho lịch sử thông báo (Toast Start / End)
# ------------------------------------------------------------------
def _append_toast_log(log_type, car_id, data):
    try:
        entry = {
            'type': log_type,  # START hoặc END
            'car_id': car_id,
            'event': data.get('event'),
            'start_time': _ts_to_iso(data.get('start_time')),
            'end_time': _ts_to_iso(data.get('end_time')) if data.get('end_time') else None,
            'fuel_start': round(data.get('fuel_before', 0), 2) if data.get('fuel_before') is not None else None,
            'fuel_current': round(data.get('fuel_after', 0), 2) if data.get('fuel_after') is not None else None,
            'delta': round(data.get('fuel_change', 0), 2) if data.get('fuel_change') is not None else None,
            'duration_s': data.get('duration_s'),
            'logged_at': datetime.now().isoformat()
        }
        # Làm sạch dict để không bị lỗi JSON nếu có giá trị NaN
        cleaned_entry = {k: v for k, v in entry.items() if v is not None}

        with open(TOAST_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(make_json_safe(cleaned_entry), ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"⚠️ Không ghi được toast log: {e}")

def _read_toast_log(limit=4, car_id=None):
    if not os.path.exists(TOAST_LOG_PATH):
        return []
    lines = []
    try:
        with open(TOAST_LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if car_id and obj.get('car_id') != car_id:
                    continue
                lines.append(obj)
        return lines[-limit:]
    except Exception as e:
        logger.error(f"⚠️ Không đọc được toast log: {e}")
        return []

# ------------------------------------------------------------------

def _append_event_log(record, car_id):
    try:
        entry = dict(record)
        entry['car_id'] = car_id
        entry['logged_at'] = datetime.now().isoformat()
        with open(EVENT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(make_json_safe(entry), ensure_ascii=False) + "\n")
        logger.info(f"📝 Event log saved: {car_id} | {record.get('event')} | {record.get('start_time')} → {record.get('end_time')}")
    except Exception as e:
        logger.error(f"⚠️ Không ghi được event log: {e}")

def _read_event_log(limit=200, car_id=None):
    if not os.path.exists(EVENT_LOG_PATH):
        return []
    lines = []
    try:
        with open(EVENT_LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if car_id and obj.get('car_id') != car_id:
                    continue
                lines.append(obj)
    except Exception as e:
        logger.error(f"⚠️ Không đọc được event log: {e}")
        return []
    return lines[-limit:]

_db_lock = threading.Lock()

def _get_db_conn():
    conn = sqlite3.connect(REALTIME_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def _init_db():
    with _db_lock:
        conn = _get_db_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS realtime_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                car_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                fuel REAL,
                speed REAL,
                latitude REAL,
                longitude REAL,
                prediction TEXT,
                confidence REAL,
                point_status TEXT,
                window_size INTEGER,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_realtime_points_car_ts
            ON realtime_points (car_id, timestamp)
        """)
        conn.commit()
        conn.close()

def _save_point_to_db(car_id, record):
    try:
        with _db_lock:
            conn = _get_db_conn()
            conn.execute(
                """INSERT INTO realtime_points
                   (car_id, timestamp, fuel, speed, latitude, longitude,
                    prediction, confidence, point_status, window_size, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    car_id,
                    _ts_to_iso(record['Timestamp']),
                    float(record.get('Fuel', 0)),
                    float(record.get('Speed', 0)),
                    float(record.get('Latitude', 0)),
                    float(record.get('Longitude', 0)),
                    record.get('Prediction'),
                    float(record.get('Confidence', 0)),
                    record.get('PointStatus', 'normal'),
                    int(record.get('WindowSize', 0)),
                    datetime.now().isoformat(),
                )
            )
            conn.commit()
            conn.close()
    except Exception as e:
        logger.error(f"⚠️ Không ghi được realtime point vào DB: {e}")

_init_db()

class ModelWrapper:
    def __init__(self, model_path):
        self.model = CNNGRUClassifier(
            input_dim=4, hidden_dim=128, num_layers=2, dropout=0.3,
            bidirectional=True, feature_dim=21, num_classes=4,
            fusion_type='concat', fusion_dim=256,
            cnn_channels=[64, 128], cnn_kernel_size=3
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        logger.info("✅ AI Model loaded successfully.")

    def predict(self, sequence, mask, feature):
        with torch.no_grad():
            seq_t = torch.from_numpy(sequence).float().unsqueeze(0).to(DEVICE)
            mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(DEVICE)
            feat_t = torch.from_numpy(feature).float().unsqueeze(0).to(DEVICE)
            logits = self.model(seq_t, mask_t, feat_t)
            return torch.softmax(logits, dim=1).cpu().numpy()[0]

class EventTracker:
    def __init__(self):
        self.last_prediction = None
        self.last_confidence = 0.0
        self.confidence_history = []
        self._processed_buffer_ref = None   # <-- THÊM DÒNG NÀY
        self.reset()
        self.last_finished_event = None
        self._post_refuel_flag = False

    def reset(self):
        self.status = 'normal'
        self.candidate_label = None
        self.candidate_start = None
        self.counter = 0
        self.best_conf = 0.0
        self.event_record = None

        if len(self.confidence_history) > 200:
            self.confidence_history = self.confidence_history[-200:]

        self.stage_history = []
        self._just_confirmed = False
        self.confirmed_ticks = 0
        self.mismatch_streak = 0
        self._post_refuel_flag = False

    @property
    def point_status(self):
        if self.status == 'candidate':
            return 'thinking'
        if self.status in ('confirmed', 'finished'):
            return 'confirmed'
        return 'normal'

    @property
    def display_stage(self):
        if self.status == 'confirmed':
            return 'confirmed' if self._just_confirmed else 'monitoring'
        return self.status

    @property
    def active_event_label(self):
        if self.status == 'candidate':
            return self.candidate_label
        if self.status in ('confirmed', 'finished') and self.event_record:
            return self.event_record.get('event')
        return None

    def _push_stage(self, stage, timestamp):
        self.stage_history.append({'stage': stage, 'timestamp': _ts_to_iso(timestamp)})

    def get_active_summary(self, full_df=None):
        if self.status != 'normal':
            event_type = self.candidate_label
            if self.event_record:
                event_type = self.event_record.get('event', event_type)

            fuel_start_val = None
            fuel_current_val = None
            delta_fuel = 0.0
            duration_s = 0
            fuel_status = 'Stable'

            if full_df is not None and len(full_df) > 0:
                fuel_current_val = float(full_df.iloc[-1]['fuel'])

                if self.event_record is not None:
                    if self.event_record.get('fuel_before'):
                        fuel_start_val = self.event_record['fuel_before']
                    else:
                        df = full_df.sort_values('timestamp')
                        start_ts = self.candidate_start
                        if self.event_record and self.event_record.get('start_time'):
                            start_ts = self.event_record['start_time']
                        try:
                            start_idx = df['timestamp'].searchsorted(start_ts)
                            if start_idx < len(df):
                                fuel_start_val = float(df.iloc[start_idx]['fuel'])
                        except Exception:
                            fuel_start_val = None

                if fuel_start_val is not None:
                    delta_fuel = fuel_current_val - fuel_start_val
                    if delta_fuel > 0.5:
                        fuel_status = 'Refueling'
                    elif delta_fuel < 0:
                        if event_type == 'Theft':
                            fuel_status = 'Theft'
                        else:
                            fuel_status = 'Consuming'
                    else:
                        fuel_status = 'Stable'

                if self.status == 'finished':
                    if self.event_record and self.event_record.get('duration_s'):
                        duration_s = self.event_record['duration_s']
                else:
                    start_ts = None
                    if self.event_record and self.event_record.get('start_time'):
                        start_ts = self.event_record['start_time']
                    elif self.candidate_start:
                        start_ts = self.candidate_start

                    if start_ts and full_df is not None:
                        latest_timestamp = full_df.iloc[-1]['timestamp']
                        duration_s = int((latest_timestamp - start_ts).total_seconds())

            return {
                'event_type': event_type,
                'stage': self.display_stage,
                'start_time': _ts_to_iso(self.candidate_start) if self.candidate_start else None,
                'confidence': self.best_conf,
                'stage_history': list(self.stage_history) if self.stage_history else [{'stage': self.display_stage, 'timestamp': _ts_to_iso(self.candidate_start)}],
                'fuel_start': fuel_start_val,
                'fuel_current': fuel_current_val,
                'delta_fuel': round(delta_fuel, 2),
                'fuel_status': fuel_status,
                'duration_s': duration_s
            }

        if full_df is not None and len(full_df) > 0:
            current_ts = full_df.iloc[-1]['timestamp']
            fuel_current_val = float(full_df.iloc[-1]['fuel'])
            self.stage_history = [{'stage': 'normal', 'timestamp': _ts_to_iso(current_ts)}]
            return {
                'event_type': self.last_prediction or 'Driving',
                'stage': 'normal',
                'start_time': _ts_to_iso(current_ts),
                'confidence': self.last_confidence if self.last_prediction else 1.0,
                'stage_history': self.stage_history,
                'fuel_start': None,
                'fuel_current': fuel_current_val,
                'delta_fuel': 0.0,
                'fuel_status': 'Stable',
                'duration_s': 0
            }
        return None

    def pop_finished_event(self):
        ev = self.last_finished_event
        self.last_finished_event = None
        return ev

    def retroactively_fix_point_status(self, buffer_df, processed_buffer):
        if self.status != 'confirmed' or not self.event_record:
            return
        start_ts = self.event_record['start_time']
        event_label = self.event_record['event']

        for record in processed_buffer:
            if (record['Timestamp'] >= start_ts
                    and record.get('PointStatus') == 'thinking'
                    and record.get('Prediction') == event_label):
                record['PointStatus'] = 'confirmed'

    def _retroactively_clear_false_candidate(self, buffer_df):
        """Khi 1 candidate Refuel/Theft/Driving/Idle bị huỷ do mismatch vượt tolerance,
        các điểm đã lỡ gắn PointStatus='thinking' cho candidate đó (nhưng chưa bao giờ
        được confirmed) cần được trả lại đúng trạng thái, tránh 'lấn' UI/alert sang
        nhãn sai. processed_buffer được inject từ CarState khi gọi (xem bước 5)."""
        if not self._processed_buffer_ref or not self.candidate_start:
            return
        start_ts = self.candidate_start
        bad_label = self.candidate_label
        for record in reversed(self._processed_buffer_ref):
            if record['Timestamp'] < start_ts:
                break
            if record.get('PointStatus') == 'thinking' and record.get('Prediction') == bad_label:
                # Trả về "normal" — điểm này thực chất không thuộc event nào cả,
                # nó chỉ là nhiễu một lần
                record['PointStatus'] = 'normal'

    def _maybe_reopen_candidate(self, label_name, conf, timestamp, buffer_df=None):
        if self.status != 'normal':
            return
        if conf < CANDIDATE_THRESH:
            return

        # --- GUARD MỚI: Refuel/Theft cần delta fuel thật sự đáng kể ---
        if label_name in ('Refuel', 'Theft') and buffer_df is not None and len(buffer_df) >= 2:
            df = buffer_df.sort_values('timestamp')
            delta = float(df.iloc[-1]['fuel']) - float(df.iloc[-2]['fuel'])
            if label_name == 'Refuel' and delta < REFUEL_MIN_DELTA:
                return   # tăng không đủ mạnh -> khả năng cao là model bắn nhầm, bỏ qua
            if label_name == 'Theft' and delta > -THEFT_MIN_DELTA:
                return   # giảm không đủ mạnh -> bỏ qua

        self.status = 'candidate'
        self.candidate_label = label_name
        self.candidate_start = timestamp
        self.counter = 1
        self.best_conf = conf
        self.mismatch_streak = 0
        self.stage_history = [{'stage': 'normal', 'timestamp': _ts_to_iso(timestamp)}]
        self._push_stage('candidate', timestamp)

    def update(self, prob, buffer_df, timestamp):
        label_id = int(np.argmax(prob))
        conf = float(prob[label_id])
        label_name = LABELS[label_id]
        self.last_prediction = label_name
        self.last_confidence = conf

        self.confidence_history.append((timestamp, label_name, conf))
        if len(self.confidence_history) > 200:
            self.confidence_history.pop(0)

        effective_candidate_thresh = CANDIDATE_THRESH
        if label_name == 'Idle':
            # Nếu xe đứng yên và cửa sổ có tốc độ ≈ 0, hạ thấp ngưỡng
            current_speed = float(buffer_df.iloc[-1]['speed'])
            if current_speed < 1.0:
                effective_candidate_thresh = 0.4

        if self.status == 'normal':
            self._maybe_reopen_candidate(label_name, conf, timestamp, buffer_df)

        elif self.status == 'candidate':
            if label_name != self.candidate_label:
                if (self.candidate_label == 'Idle'
                        and label_name in ('Refuel', 'Theft')
                        and buffer_df is not None and len(buffer_df) >= 2):
                    df_chk = buffer_df.sort_values('timestamp')
                    delta_chk = float(df_chk.iloc[-1]['fuel']) - float(df_chk.iloc[-2]['fuel'])
                    is_real_refuel = label_name == 'Refuel' and delta_chk >= REFUEL_MIN_DELTA
                    is_real_theft = label_name == 'Theft' and delta_chk <= -THEFT_MIN_DELTA
                    if is_real_refuel or is_real_theft:
                        self.reset()
                        self._maybe_reopen_candidate(label_name, conf, timestamp, buffer_df)
                        return
                    
                self.mismatch_streak += 1
            
                if self.candidate_label == 'Idle':
                    tolerance = 2
                else:
                    tolerance = (FUEL_EVENT_MISMATCH_TOLERANCE
                                if self.candidate_label in ('Refuel', 'Theft')
                                else CANDIDATE_MISMATCH_TOLERANCE)

                if self.candidate_label == 'Idle':
                    current_speed = float(buffer_df.iloc[-1]['speed'])
                    # Tính độ dốc nhiên liệu gần đây (3 điểm cuối)
                    recent_fuel = [buffer_df.iloc[-i-1]['fuel'] for i in range(min(3, len(buffer_df)))] if len(buffer_df) >= 3 else [0, 0, 0]
                    fuel_slope = abs(recent_fuel[0] - recent_fuel[-1]) / max(1, len(recent_fuel) - 1)
                    # Nếu vẫn dừng và nhiên liệu ổn định, reset mismatch
                    if current_speed < 1.0 and fuel_slope < IDLE_FUEL_SLOPE_RESET:
                        self.mismatch_streak = 0

                if self.mismatch_streak > tolerance:
                    self._retroactively_clear_false_candidate(buffer_df)
                    self.reset()
                    self._maybe_reopen_candidate(label_name, conf, timestamp, buffer_df)
                return
            else:
                self.mismatch_streak = 0

            if conf >= CANDIDATE_THRESH:
                self.counter += 1
                self.best_conf = max(self.best_conf, conf)
                if self.candidate_label == 'Idle':
                    # Hạ ngưỡng cho Idle xuống 0.65 để bắt kịp dừng ngắn 10p
                    effective_confirm_thresh = 0.5
                else:
                    effective_confirm_thresh = CONFIRM_THRESH
                if self.counter >= CONFIRM_COUNT and conf >= effective_confirm_thresh:
                    self.status = 'confirmed'
                    self._just_confirmed = True
                    self.confirmed_ticks = 0
                    self.mismatch_streak = 0
                    self.event_record = {
                        'event': self.candidate_label,
                        'start_time': self.candidate_start,
                        'confidence': self.best_conf,
                    }
                    self._backtrack_start(buffer_df)
                    self._push_stage('confirmed', timestamp)

                    summary = self.get_active_summary(buffer_df)
                    _append_toast_log('START', self.car_id, {
                        'event': self.candidate_label,
                        'start_time': self.event_record['start_time'],
                        'fuel_before': summary.get('fuel_start'),
                        'fuel_after': summary.get('fuel_current'),
                        'fuel_change': summary.get('delta_fuel'),
                        'duration_s': summary.get('duration_s')
                    })
            else:
                self.mismatch_streak += 1
                tolerance = (FUEL_EVENT_MISMATCH_TOLERANCE
                            if self.candidate_label in ('Refuel', 'Theft')
                            else CANDIDATE_MISMATCH_TOLERANCE)
                if self.mismatch_streak > tolerance:
                    self._retroactively_clear_false_candidate(buffer_df)
                    self.reset()

        elif self.status == 'confirmed':
            self.confirmed_ticks += 1
            if self._just_confirmed:
                self._just_confirmed = False
                self._push_stage('monitoring', timestamp)

            self.event_record['fuel_current'] = float(buffer_df.iloc[-1]['fuel'])

            try:
                df = buffer_df.sort_values('timestamp')
                start_idx = df['timestamp'].searchsorted(self.event_record['start_time'])
                if start_idx < len(df):
                    window_fuel = df.iloc[start_idx:]['fuel']
                    if len(window_fuel) > 0:
                        current_min_fuel = round(float(window_fuel.min()), 2)
                        self.event_record['min_fuel'] = current_min_fuel
            except Exception as e:
                logger.error(f"Active Min Fuel calc error: {e}")

            current_speed = float(buffer_df.iloc[-1]['speed'])
            event_type = self.event_record.get('event')
            if event_type in ['Refuel', 'Theft'] and current_speed > 10.0:
                self._finish_event(buffer_df, timestamp)
                self._maybe_reopen_candidate(label_name, conf, timestamp, buffer_df)
                return

            if conf < FINISH_THRESH or label_name != self.event_record['event']:
                self._finish_event(buffer_df, timestamp)
                self._maybe_reopen_candidate(label_name, conf, timestamp, buffer_df)   # <-- thêm buffer_df
            else:
                self.best_conf = max(self.best_conf, conf)
                self.event_record['confidence'] = self.best_conf

    def _finish_event(self, buffer_df, end_time):
        self.status = 'finished'
        self._backtrack_end(buffer_df, end_time)
        # self.event_record['end_time'] = end_time
        self._push_stage('finished', end_time)

        df = buffer_df.sort_values('timestamp')
        start_ts = self.event_record['start_time']
        end_ts = self.event_record['end_time']
        min_fuel = None
        try:
            start_idx = df['timestamp'].searchsorted(start_ts)
            end_idx = df['timestamp'].searchsorted(end_ts)
            if start_idx < len(df) and end_idx < len(df):
                self.event_record['fuel_before'] = round(float(df.iloc[start_idx]['fuel']), 2)
                self.event_record['fuel_after'] = round(float(df.iloc[end_idx]['fuel']), 2)
                self.event_record['fuel_change'] = round(
                    self.event_record['fuel_after'] - self.event_record['fuel_before'], 2)
                lo, hi = sorted([start_idx, end_idx])
                window_fuel = df.iloc[lo:hi + 1]['fuel']
                if len(window_fuel):
                    min_fuel = round(float(window_fuel.min()), 2)
        except Exception as e:
            logger.error(f"Fuel change error: {e}")

        self.event_record['min_fuel'] = min_fuel
        self.event_record['fuel_added'] = self.event_record.get('fuel_change', 0)
        self.event_record['max_confidence'] = self.best_conf
        self.event_record['duration_s'] = int(
            (self.event_record['end_time'] - self.event_record['start_time']).total_seconds())
        self.event_record['stage_history'] = list(self.stage_history)

        was_refuel = (self.event_record['event'] == 'Refuel')   # <-- lưu lại trước khi reset

        self.last_finished_event = dict(self.event_record)
        _append_toast_log('END', self.car_id, self.event_record)
        self.reset()

        if was_refuel:
            self._post_refuel_flag = True

    def _backtrack_start(self, buffer_df):
        threshold = BACKTRACK_FUEL_THRESHOLD
        event_label = self.event_record['event']
        df = buffer_df.sort_values('timestamp').reset_index(drop=True)
        idx = df['timestamp'].searchsorted(self.event_record['start_time'])
        if idx >= len(df):
            idx = len(df) - 1

        # --- MỚI: Refuel/Theft ưu tiên bằng chứng vật lý (delta fuel thô),
        # không phụ thuộc nhãn dự đoán cũ của điểm trước (vốn được gán lúc
        # model CHƯA thấy đủ bằng chứng nên luôn trễ 1 nhịp). ---
        if event_label in ('Refuel', 'Theft') and idx > 0:
            min_delta = REFUEL_MIN_DELTA if event_label == 'Refuel' else THEFT_MIN_DELTA
            prev_fuel = float(df.iloc[idx - 1]['fuel'])
            cur_fuel = float(df.iloc[idx]['fuel'])
            delta = cur_fuel - prev_fuel
            cond = delta >= min_delta if event_label == 'Refuel' else delta <= -min_delta
            if cond:
                self.event_record['start_time'] = df.iloc[idx - 1]['timestamp']
                self.event_record['boundary_point'] = True   # đánh dấu: điểm giao 2 trạng thái
                return
            # nếu bước nhảy không nằm ngay tại idx-1->idx, fallback về logic cũ bên dưới

        label_by_ts = {}
        for ts, lbl, _ in self.confidence_history:
            label_by_ts[pd.Timestamp(ts)] = lbl

        lower_bound = max(0, idx - BACKTRACK_MAX_LOOKBACK)
        new_start_idx = idx

        for i in range(idx, lower_bound, -1):
            cur_ts = pd.Timestamp(df.iloc[i]['timestamp'])
            cur_label = label_by_ts.get(cur_ts)

            if cur_label is not None and cur_label != event_label:
                break

            new_start_idx = i
            if i > 0:
                prev_ts = pd.Timestamp(df.iloc[i - 1]['timestamp'])
                prev_label = label_by_ts.get(prev_ts)
                if prev_label is not None and prev_label != event_label:
                    break
                if abs(df.iloc[i]['fuel'] - df.iloc[i - 1]['fuel']) >= threshold:
                    break

        self.event_record['start_time'] = df.iloc[new_start_idx]['timestamp']

    def _backtrack_end(self, buffer_df, end_time):
        """
        [FIX] Quay ngược thời gian để tìm điểm kết thúc chính xác của sự kiện cũ,
        dựa trên sự thay đổi đột ngột của tín hiệu vật lý (tốc độ dừng / nhiên liệu nhảy).
        """
        df = buffer_df.sort_values('timestamp').reset_index(drop=True)
        idx = df['timestamp'].searchsorted(end_time)
        if idx >= len(df):
            idx = len(df) - 1
            
        event_label = self.event_record.get('event')
        
        # Tìm điểm quay ngược tối đa (ví dụ 3-5 điểm trước đó, đủ để bắt chuyển pha)
        # Trong trường hợp Driving -> Idle, sự thay đổi tốc độ là tức thời.
        backtrack_window = 5
        lower_bound = max(0, idx - backtrack_window)
        new_end_idx = idx
        
        for i in range(idx, lower_bound, -1):
            cur_speed = df.iloc[i]['speed']
            prev_speed = df.iloc[i-1]['speed'] if i > 0 else cur_speed
            
            # Quy tắc chuyển pha:
            # 1. Nếu sự kiện cũ là Driving, và tốc độ vừa mới tụt từ mức đang chạy (>8km/h) xuống dừng hẳn (<2km/h)
            if event_label == 'Driving' and prev_speed > 8.0 and cur_speed < 2.0:
                new_end_idx = i  # Điểm kết thúc chính xác là ngay tại điểm vừa dừng
                break
                
            # 2. Nếu sự kiện cũ là Idle, và tốc độ vừa mới tăng vọt lên (>8km/h)
            if event_label == 'Idle' and prev_speed < 2.0 and cur_speed > 8.0:
                new_end_idx = i
                break
                
            # 3. Nếu sự kiện cũ là Refuel/Theft, có thể dựa trên delta fuel (nhưng bỏ qua nếu chỉ muốn xử lý Idle/Driving hiện tại)
        
        self.event_record['end_time'] = df.iloc[new_end_idx]['timestamp']

class AdaptiveWindowManager:
    def __init__(self, ranges=None, idle_steps=None, stability_std=STABILITY_STD_THRESHOLD):
        self.ranges = ranges or WINDOW_RANGES
        self.idle_steps = idle_steps or IDLE_STEPS
        self.stability_std = stability_std

    def get_window_size(self, tracker: EventTracker, data_buffer=None) -> int:
        # ---------- 1. Tính tốc độ thay đổi nhiên liệu gần đây ----------
        fuel_change_rate = 0.0
        if data_buffer and len(data_buffer) >= 2:
            last2 = list(data_buffer)[-2:]
            fuel_change_rate = abs(last2[-1]['fuel'] - last2[0]['fuel'])   # delta tuyệt đối 2 điểm liền kề, không chia trung bình 3 điểm

        status = tracker.status

        # ---------- 2. Phản ứng với thay đổi đột ngột ----------
        if fuel_change_rate > 25.0:
            return 2
        if fuel_change_rate > 10.0:
            return 3
        if fuel_change_rate > 5.0:        # <-- MỚI: thêm bậc trung gian
            return 3
        if fuel_change_rate > 3.0:
            return 4
        if fuel_change_rate > 1.5:        # <-- MỚI
            return 4

        # ---------- 3. Sau Refuel kết thúc → cửa sổ nhỏ để bắt Idle ----------
        if status == 'normal' and getattr(tracker, '_post_refuel_flag', False):
            return 3

        # ---------- 4. Theo trạng thái tracker ----------
        if status == 'normal':
            return self.ranges['normal'][1]
        if status == 'candidate':
            if tracker.candidate_label == 'Idle':
                return self.ranges['Idle'][0]      # 4
            if tracker.candidate_label in ('Refuel', 'Theft'):
                return self.ranges['Refuel'][0]    # 3
            # Driving hoặc chưa rõ -> dùng cửa sổ lớn hơn để tránh nhiễu
            return self.ranges['candidate'][0]     # 4 (đã tăng)

        if status in ('confirmed', 'finished'):
            event = tracker.active_event_label
            lo, hi = self.ranges.get(event, self.ranges['normal'])
            if event in ('Refuel', 'Theft'):
                return lo                     # 2
            if event == 'Idle':
                return self._idle_step(tracker.confirmed_ticks)
            return self._stability_adjusted(lo, hi, tracker.confidence_history)

        return self.ranges['normal'][1]

    def _stability_adjusted(self, lo, hi, confidence_history):
        if hi <= lo or len(confidence_history) < 2:
            return lo
        recent = [c for _, _, c in confidence_history[-lo:]]
        if len(recent) < 2:
            return lo
        std = float(np.std(recent))
        if std <= self.stability_std:
            return lo
        ratio = min(std / 0.3, 1.0)
        window = lo + int(round((hi - lo) * ratio))
        return max(lo, min(hi, window))

    def _idle_step(self, confirmed_ticks):
        idx = min(max(confirmed_ticks, 0), len(self.idle_steps) - 1)
        return max(3, self.idle_steps[idx])

class SegmentBuilder:
    @staticmethod
    def build(data_buffer, window_size):
        available = len(data_buffer)
        actual_window = max(1, min(window_size, available))
        window_list = list(data_buffer)[-actual_window:]
        segment_df = pd.DataFrame(window_list).sort_values('timestamp').reset_index(drop=True)
        return segment_df, actual_window

window_manager = AdaptiveWindowManager()

class CarState:
    def __init__(self, car_id, prefill_points=None):
        self.car_id = car_id
        self.data_buffer = deque(maxlen=BUFFER_SIZE)
        if prefill_points:
            for p in prefill_points:
                self.data_buffer.append(p)
        self.processed_buffer = []
        self.event_history = []
        self.tracker = EventTracker()
        self.tracker.car_id = car_id
        self.tracker._processed_buffer_ref = self.processed_buffer   # <-- THÊM DÒNG NÀY
        self.lock = threading.Lock()

car_data_cache = {}
car_states = {}
_car_states_meta_lock = threading.Lock()
current_car = "Car 2"

def _baseline_points_for_car(car_id, n=WINDOW_SIZE):
    df = car_data_cache.get(car_id)
    if df is None or len(df) == 0:
        return []
    tail = df.tail(n)
    points = []
    for _, row in tail.iterrows():
        points.append({
            'timestamp': row['timestamp'],
            'fuel': float(row['fuel']),
            'speed': float(row['speed']),
            'latitude': float(row.get('latitude', 0) or 0),
            'longitude': float(row.get('longitude', 0) or 0),
            'adc': 0
        })
    return points

def get_car_state(car_id):
    with _car_states_meta_lock:
        state = car_states.get(car_id)
        if state is None:
            prefill = _baseline_points_for_car(car_id, WINDOW_SIZE)
            state = CarState(car_id, prefill_points=prefill)
            car_states[car_id] = state
            if prefill:
                logger.info(f"🧩 [{car_id}] Buffer khởi tạo sẵn {len(prefill)} điểm nền (baseline)")
            else:
                logger.info(f"🧩 [{car_id}] Không có dữ liệu nền (baseline) cho xe này")
        return state

def load_all_cars():
    if not os.path.exists(FUSION_CSV):
        logger.warning(f"⚠️ Không tìm thấy {FUSION_CSV}")
        return
    df = pd.read_csv(FUSION_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for car in df['car_id'].unique():
        car_df = df[df['car_id'] == car].sort_values('timestamp').reset_index(drop=True)
        car_data_cache[car] = car_df
    logger.info(f"✅ Đã nạp dữ liệu nền cho {len(car_data_cache)} xe: {list(car_data_cache.keys())}")

load_all_cars()

def process_window(window_df, car_id):
    df = window_df.copy()
    df['car_id'] = car_id
    df['segment_id'] = 1
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = convert_gps_to_relative(df)
    df['final_label'] = 'Driving'
    feature_table = build_segment_features_table(df)
    feat_df = feature_table[FEATURE_COLS_21]
    feat_vec = feat_df.iloc[0].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    seq_cols = ['fuel', 'speed', 'distance_step', 'bearing']
    seq = df[seq_cols].values.astype(np.float32)
    mask = np.ones(len(seq), dtype=bool)
    probs = model_wrapper.predict(seq, mask, feat_vec)
    fuel_vals = df['fuel'].values
    speed_vals = df['speed'].values
    duration_min = len(fuel_vals) * 5
    features = {
        'MovingAvg': float(np.mean(fuel_vals)),
        'RollingStd': float(np.std(fuel_vals)),
        'RegressionSlope': float(feat_vec[7]),
        'FuelRate': float((fuel_vals[-1] - fuel_vals[0]) / max(duration_min * 60, 1)),
        'WindowFuelDiff': float(feat_vec[4]),
        'MaxJump': float(max(0, feat_vec[10], feat_vec[11])),
        'StopDuration': int(duration_min * feat_vec[-1] * 60),
        'AvgSpeed': float(np.mean(speed_vals)),
        'FuelRange': float(feat_vec[14]),
    }
    return probs, features

model_wrapper = ModelWrapper(MODEL_PATH)

def _normalize_incoming_point(p):
    try:
        ts = pd.to_datetime(p['timestamp'])
    except Exception:
        ts = pd.Timestamp(datetime.now())
    return {
        'timestamp': ts,
        'fuel': float(p['fuel']),
        'speed': float(p.get('speed', 0)),
        'latitude': float(p.get('latitude', 0)),
        'longitude': float(p.get('longitude', 0)),
        'adc': p.get('ADC', p.get('adc', 0))
    }

def _ingest_and_predict_one(point, car_id):
    state = get_car_state(car_id)
    with state.lock:
        state.data_buffer.append(point)
        buffer_size = len(state.data_buffer)

        desired_window = window_manager.get_window_size(state.tracker, state.data_buffer)
        if state.tracker.status == 'normal':
            recent = list(state.data_buffer)[-3:]
            if len(recent) >= 2:
                fuel_delta = abs(recent[-1]['fuel'] - recent[0]['fuel'])
                if fuel_delta > 25:
                    desired_window = 2
                elif fuel_delta > 10:
                    desired_window = 3

        ready = buffer_size >= min(desired_window, WINDOW_SIZE) if state.tracker.status == 'normal' else buffer_size >= 1

        logger.info(
            f"📥 [{car_id}] Nhận dữ liệu | time={_ts_to_iso(point['timestamp'])} | "
            f"fuel={point['fuel']:.2f} | speed={point['speed']:.1f} | "
            f"buffer={buffer_size} | desired_window={desired_window} | tracker={state.tracker.status} | "
            f"{'sẵn sàng predict' if ready else 'đang chờ đủ dữ liệu'}"
        )
        if not ready:
            return {'car_id': car_id, 'status': 'buffering', 'buffered': buffer_size, 'need': desired_window}

        window_df, window_size = SegmentBuilder.build(state.data_buffer, desired_window)

        probs, feats = process_window(window_df, car_id)
                
        # --- Override 3 [MỚI]: phá thế hòa bằng bằng chứng vật lý ---
        # Khi top-1 và top-2 quá sát nhau (model "tung đồng xu"), ưu tiên nhãn
        # khớp với hướng delta fuel thật, bất kể tracker đang ở status nào.
        TIE_MARGIN = 0.15
        if len(state.data_buffer) >= 2:
            recent2 = list(state.data_buffer)[-2:]
            delta_fuel = recent2[-1]['fuel'] - recent2[0]['fuel']
            sorted_idx = np.argsort(probs)[::-1]
            margin = float(probs[sorted_idx[0]] - probs[sorted_idx[1]])
            if margin < TIE_MARGIN:
                if delta_fuel <= -THEFT_MIN_DELTA and 'Theft' in (LABELS[sorted_idx[0]], LABELS[sorted_idx[1]]):
                    probs = np.full_like(probs, 0.02)
                    probs[LABELS.index('Theft')] = 1.0 - 0.02 * (len(LABELS) - 1)
                elif delta_fuel >= REFUEL_MIN_DELTA and 'Refuel' in (LABELS[sorted_idx[0]], LABELS[sorted_idx[1]]):
                    probs = np.full_like(probs, 0.02)
                    probs[LABELS.index('Refuel')] = 1.0 - 0.02 * (len(LABELS) - 1)

        # --- Override 1: Idle vật lý (đã có) ---
        if len(state.data_buffer) >= 2:
            recent = list(state.data_buffer)[-2:]
            speeds = [p['speed'] for p in recent]
            fuels = [p['fuel'] for p in recent]
            if max(speeds) < 1.0 and (max(fuels) - min(fuels)) < 0.3:
                probs = np.zeros_like(probs)
                probs[LABELS.index('Idle')] = 1.0

        # --- Override 2 [MỚI]: giữ nhãn event đang confirmed khi tín hiệu mập mờ ---
        # Khi đang confirmed 1 event KHÔNG PHẢI Refuel/Theft, nếu điểm mới có
        # top-confidence yếu hoặc sát nút với hạng nhì, đây nhiều khả năng là
        # nhiễu biên giới (model "lưỡng lự") chứ không phải chuyển trạng thái
        # thật. Ép nhãn về đúng event đang chạy để tránh 1 điểm "mồ côi" hiện
        # lạc giữa 2 đoạn màu trên chart.
        TRANSITION_HOLD_CONF = 0.65
        TRANSITION_HOLD_MARGIN = 0.15
        if (state.tracker.status == 'confirmed' and state.tracker.event_record
                and state.tracker.event_record.get('event') not in ('Refuel', 'Theft')):
            sorted_idx = np.argsort(probs)[::-1]
            top_conf = float(probs[sorted_idx[0]])
            margin = float(probs[sorted_idx[0]] - probs[sorted_idx[1]])
            runner_up_label = LABELS[sorted_idx[1]]

            # --- GUARD MỚI: không giữ nhãn cũ nếu có bằng chứng vật lý đủ mạnh
            # cho thấy runner-up (Theft/Refuel) là thật, dù model chưa đủ tự tin ---
            skip_hold = False
            if len(state.data_buffer) >= 2:
                recent2 = list(state.data_buffer)[-2:]
                delta_fuel = recent2[-1]['fuel'] - recent2[0]['fuel']
                if runner_up_label == 'Theft' and delta_fuel <= -THEFT_MIN_DELTA:
                    skip_hold = True
                elif runner_up_label == 'Refuel' and delta_fuel >= REFUEL_MIN_DELTA:
                    skip_hold = True

            if not skip_hold and (top_conf < TRANSITION_HOLD_CONF or margin < TRANSITION_HOLD_MARGIN):
                held_label = state.tracker.event_record['event']
                probs = np.full_like(probs, 0.02)
                probs[LABELS.index(held_label)] = 1.0 - 0.02 * (len(LABELS) - 1)

        # --- pred_label/conf tính lại SAU khi mọi override đã áp dụng ---
        pred_label = LABELS[int(np.argmax(probs))]
        conf = float(probs[int(np.argmax(probs))])

        probs_str = ", ".join([f"{LABELS[i]}={probs[i]:.3f}" for i in range(4)])
        logger.info(f"🤖 [{car_id}] Dự đoán: {pred_label} (conf={conf:.3f}) | window={window_size} | Probs: [{probs_str}]")

        full_df = pd.DataFrame(list(state.data_buffer))
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
        state.tracker.update(probs, full_df, point['timestamp'])
        tracker_status = state.tracker.status
        display_stage = state.tracker.display_stage
        point_status = state.tracker.point_status
        logger.info(f"📊 [{car_id}] Tracker: status={tracker_status} | display_stage={display_stage} | point_status={point_status}")

        finished_event = state.tracker.pop_finished_event()
        if finished_event:
            state.event_history.append(finished_event)
            _append_event_log(finished_event, car_id)
            logger.info(f"🎯 [{car_id}] EVENT FINISHED: {finished_event.get('event')} | "
                       f"start={finished_event.get('start_time')} | end={finished_event.get('end_time')} | "
                       f"fuel_change={finished_event.get('fuel_change')}L | duration={finished_event.get('duration_s')}s")

        if tracker_status == 'candidate':
            logger.info(f"  ⏳ [{car_id}] Candidate: {state.tracker.candidate_label} (counter={state.tracker.counter}/{CONFIRM_COUNT}, mismatch={state.tracker.mismatch_streak})")

        if tracker_status == 'confirmed' and state.tracker._just_confirmed:
            logger.info(f"  ✅ [{car_id}] CONFIRMED: {state.tracker.event_record.get('event')} | confidence={state.tracker.best_conf:.3f}")
            state.tracker.retroactively_fix_point_status(full_df, state.processed_buffer)

        last = window_df.iloc[-1]
        record = {
            'Timestamp': point['timestamp'],    
            'Fuel': last['fuel'],
            'Speed': last['speed'],
            'ADC': last.get('adc', 0),
            'Latitude': last['latitude'],
            'Longitude': last['longitude'],
            'Prediction': pred_label,
            'PointStatus': point_status,
            'Confidence': conf * 100,
            'WindowSize': window_size,
            'BoundaryPoint': bool(state.tracker.event_record.get('boundary_point')) if state.tracker.event_record else False,  # <-- THÊM
            **feats
        }
        state.processed_buffer.append(record)
        if len(state.processed_buffer) > MAX_POINTS_HISTORY:
            del state.processed_buffer[0: len(state.processed_buffer) - MAX_POINTS_HISTORY]
        _save_point_to_db(car_id, record)
        return make_json_safe({
            'car_id': car_id,
            'status': point_status,
            'record': {
                'timestamp': record['Timestamp'].isoformat(),
                'fuel': record['Fuel'],
                'speed': record['Speed'],
                'prediction': record['Prediction'],
                'point_status': point_status,
                'confidence': record['Confidence'],
                'window_size': window_size
            },
            'tracker_status': tracker_status,
            'active_event': state.tracker.event_record
        })

@app.route('/')
def index():
    cars = sorted(set(list(car_data_cache.keys()) + list(car_states.keys())))
    return render_template('dashboard.html', cars=cars, current_car=current_car)

@app.route('/dataset')
def dataset():
    cars = sorted(set(list(car_data_cache.keys()) + list(car_states.keys())))
    return render_template('dataset.html', cars=cars, current_car=current_car)

@app.route('/feature')
def feature():
    cars = sorted(set(list(car_data_cache.keys()) + list(car_states.keys())))
    return render_template('feature.html', cars=cars, current_car=current_car)

@app.route('/realtime')
def realtime():
    cars = sorted(set(list(car_data_cache.keys()) + list(car_states.keys())))
    return render_template('realtime.html', cars=cars, current_car=current_car)

@app.route('/events')
def events():
    cars = sorted(set(list(car_data_cache.keys()) + list(car_states.keys())))
    return render_template('events.html', cars=cars, current_car=current_car)

@app.route('/report')
def report():
    cars = sorted(set(list(car_data_cache.keys()) + list(car_states.keys())))
    return render_template('report.html', cars=cars, current_car=current_car)

@app.route('/settings')
def settings():
    cars = sorted(set(list(car_data_cache.keys()) + list(car_states.keys())))
    return render_template('settings.html', cars=cars, current_car=current_car)

@app.route('/api/set_vehicle', methods=['POST'])
def set_vehicle():
    global current_car
    data = request.json or {}
    car_id = data.get('car_id')
    if not car_id: return jsonify({'error': 'Missing car_id'}), 400
    current_car = car_id
    get_car_state(car_id)
    logger.info(f"🚛 UI chuyển sang xem xe {car_id}")
    return jsonify({'status': 'ok', 'car_id': current_car})

@app.route('/api/current_vehicle')
def current_vehicle():
    return jsonify({'car_id': current_car})

@app.route('/api/car_data')
def api_car_data():
    car_id = request.args.get('car_id', current_car)
    limit = int(request.args.get('limit', 500))
    if car_id not in car_data_cache: return jsonify({'error': 'Car not found'}), 404
    df = car_data_cache[car_id].tail(limit).copy()
    data = []
    for _, row in df.iterrows():
        data.append(make_json_safe({
            'timestamp': row['timestamp'].isoformat(),
            'fuel': float(row['fuel']),
            'speed': float(row['speed']),
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'label': row.get('final_label', 'Driving')
        }))
    return jsonify(data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(silent=True)
    if payload is None: return jsonify({'error': 'Missing/invalid JSON body'}), 400
    raw_points = payload if isinstance(payload, list) else [payload]
    if len(raw_points) == 0: return jsonify({'error': 'Empty payload'}), 400
    results = []
    for p in raw_points:
        car_id = p.get('car_id') or p.get('carId')
        if not car_id: return jsonify({'error': "Missing 'car_id' in point payload"}), 400
        try:
            point = _normalize_incoming_point(p)
        except (KeyError, TypeError, ValueError) as e:
            return jsonify({'error': f'Invalid point payload: {e}'}), 400
        try:
            res = _ingest_and_predict_one(point, car_id)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Predict failed: {e}'}), 500
        results.append(res)
    return jsonify(results if isinstance(payload, list) else results[0])

@app.route('/api/points')
def api_points():
    car_id = request.args.get('car_id', current_car)
    limit = int(request.args.get('limit', DEFAULT_VISIBLE_POINTS))
    offset = int(request.args.get('offset', 0))
    limit = max(1, min(limit, MAX_POINTS_HISTORY))
    offset = max(0, offset)
    state = get_car_state(car_id)
    with state.lock:
        buf = list(state.processed_buffer)
    total = len(buf)
    end = max(0, total - offset)
    start = max(0, end - limit)
    page = buf[start:end]

    points = []
    for item in page:
        points.append(make_json_safe({
            'timestamp': item['Timestamp'].isoformat() if hasattr(item['Timestamp'], 'isoformat') else str(item['Timestamp']),
            'fuel': item['Fuel'],
            'speed': item['Speed'],
            'prediction': item['Prediction'],
            'point_status': item.get('PointStatus', 'normal'),
            'confidence': item['Confidence'],
            'window_size': item.get('WindowSize'),
            'boundary_point': item.get('BoundaryPoint', False)   # <-- THÊM
        }))
    return jsonify({'car_id': car_id, 'total': total, 'limit': limit, 'offset': offset, 'points': points})

@app.route('/api/history')
def api_history():
    car_id = request.args.get('car_id', current_car)
    state = get_car_state(car_id)
    with state.lock:
        history = state.processed_buffer[-100:] if state.processed_buffer else []
    result = []
    for item in history:
        result.append(make_json_safe({
            "raw": {
                "Timestamp": item['Timestamp'].isoformat() if hasattr(item['Timestamp'], 'isoformat') else str(item['Timestamp']),
                "Fuel": item['Fuel'],
                "Speed": item['Speed'],
                "ADC": item.get('ADC', 0),
                "Latitude": item['Latitude'],
                "Longitude": item['Longitude']
            },
            "processed": {
                "Prediction": item['Prediction'],
                "PointStatus": item.get('PointStatus', 'normal'),
                "RegressionSlope": item['RegressionSlope'],
                "RegressionR2": 0.0,
                "FuelRate": item['FuelRate'],
                "InstantFuelDiff": item.get('WindowFuelDiff', 0),
                "WindowFuelDiff": item.get('WindowFuelDiff', 0),
                "MaxJump": item.get('MaxJump', 0),
                "MovingAvg": item['MovingAvg'],
                "RollingStd": item['RollingStd'],
                "StopDuration": item['StopDuration'],
                "AvgSpeed": item['AvgSpeed'],
                "FuelRange": item['FuelRange'],
                "Confidence": item['Confidence'],
                "WindowSize": item.get('WindowSize')
            }
        }))
    return jsonify(result)

@app.route('/api/realtime')
def api_realtime():
    car_id = request.args.get('car_id', current_car)
    state = get_car_state(car_id)
    with state.lock:
        if not state.processed_buffer:
            return jsonify(make_json_safe({
                "raw": {"Timestamp": datetime.now().isoformat(), "Fuel": 0, "Speed": 0, "ADC": 0, "Latitude": 0, "Longitude": 0},
                "processed": {"Prediction": "Idle", "PointStatus": "normal", "RegressionSlope": 0, "FuelRate": 0, "MovingAvg": 0, "RollingStd": 0, "StopDuration": 0, "AvgSpeed": 0, "FuelRange": 0, "Confidence": 0, "WindowSize": 0}
            }))
        last = state.processed_buffer[-1]
    return jsonify(make_json_safe({
        "raw": {
            "Timestamp": last['Timestamp'].isoformat() if hasattr(last['Timestamp'], 'isoformat') else str(last['Timestamp']),
            "Fuel": last['Fuel'],
            "Speed": last['Speed'],
            "ADC": last.get('ADC', 0),
            "Latitude": last['Latitude'],
            "Longitude": last['Longitude']
        },
        "processed": {
            "Prediction": last['Prediction'],
            "PointStatus": last.get('PointStatus', 'normal'),
            "RegressionSlope": last['RegressionSlope'],
            "FuelRate": last['FuelRate'],
            "MovingAvg": last['MovingAvg'],
            "RollingStd": last['RollingStd'],
            "StopDuration": last['StopDuration'],
            "AvgSpeed": last['AvgSpeed'],
            "FuelRange": last['FuelRange'],
            "Confidence": last['Confidence'],
            "WindowSize": last.get('WindowSize')
        }
    }))

@app.route('/api/events')
def api_events():
    car_id = request.args.get('car_id', current_car)
    state = get_car_state(car_id)
    with state.lock:
        evts = list(state.event_history)
    result = []
    for evt in evts:
        result.append(make_json_safe({
            'state': evt.get('event', 'Unknown'),
            'start_time': evt['start_time'].isoformat() if hasattr(evt['start_time'], 'isoformat') else str(evt['start_time']),
            'end_time': evt['end_time'].isoformat() if hasattr(evt['end_time'], 'isoformat') else str(evt['end_time']),
            'start_fuel': evt.get('fuel_before', 0),
            'end_fuel': evt.get('fuel_after', 0),
            'delta': evt.get('fuel_change', 0),
            'min_fuel': evt.get('min_fuel'),
            'fuel_added': evt.get('fuel_added', evt.get('fuel_change', 0)),
            'duration_s': evt.get('duration_s', 0),
            'avg_rate': 0.0,
            'confidence': evt.get('confidence', 0) * 100,
            'max_confidence': evt.get('max_confidence', evt.get('confidence', 0)) * 100,
            'stage_history': evt.get('stage_history', [])
        }))
    return jsonify(result)

@app.route('/api/event_log')
def api_event_log():
    limit = int(request.args.get('limit', 1000))
    car_id = request.args.get('car_id')
    raw_logs = _read_event_log(limit=limit, car_id=car_id)

    result = []
    for evt in raw_logs:
        if not evt:
            continue
        duration_s = evt.get('duration_s', 0) or 0
        fuel_change = evt.get('fuel_change', evt.get('fuel_added', 0)) or 0
        # avg_rate (L/min) tính từ delta nhiên liệu / thời lượng sự kiện.
        avg_rate = round((fuel_change / duration_s) * 60, 4) if duration_s else 0.0
        raw_confidence = evt.get('confidence', 0) or 0
        raw_max_confidence = evt.get('max_confidence', raw_confidence) or 0

        result.append({
            'state': evt.get('event', 'Unknown'),
            'start_time': evt.get('start_time'),
            'end_time': evt.get('end_time'),
            'start_fuel': evt.get('fuel_before', 0),
            'end_fuel': evt.get('fuel_after', 0),
            'delta': fuel_change,
            'min_fuel': evt.get('min_fuel'),
            'fuel_added': evt.get('fuel_added', fuel_change),
            'duration_s': duration_s,
            'avg_rate': avg_rate,
            'confidence': raw_confidence * 100,
            'max_confidence': raw_max_confidence * 100,
            'stage_history': evt.get('stage_history', [])
        })

    return jsonify({'total': len(result), 'events': result})

# API lấy 4 thông báo toast gần nhất cho icon chuông
@app.route('/api/toasts')
def api_toasts():
    car_id = request.args.get('car_id', current_car)
    toasts = _read_toast_log(limit=4, car_id=car_id)
    return jsonify(toasts)

@app.route('/api/alerts')
def api_alerts():
    car_id = request.args.get('car_id', current_car)
    state = get_car_state(car_id)
    logs = _read_event_log(limit=50, car_id=car_id)
    active_event = None
    with state.lock:
        full_df = None
        if state.processed_buffer:
            full_df = pd.DataFrame(state.processed_buffer)
            full_df.rename(columns={
                'Timestamp': 'timestamp',
                'Fuel': 'fuel',
                'Speed': 'speed',
                'Prediction': 'prediction',
                'PointStatus': 'point_status',
                'Confidence': 'confidence'
            }, inplace=True)
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])

        if state.tracker.event_record:
            active_event = dict(state.tracker.event_record)
            if active_event.get('start_time'):
                active_event['start_time'] = _ts_to_iso(active_event['start_time'])
            if active_event.get('end_time'):
                active_event['end_time'] = _ts_to_iso(active_event['end_time'])

            # [FIX] Truyền full_df vào get_active_summary()
            active_summary = state.tracker.get_active_summary(full_df)
            if active_summary:
                active_event['fuel_current'] = active_summary.get('fuel_current')
                active_event['fuel_start'] = active_summary.get('fuel_start')
                active_event['delta_fuel'] = active_summary.get('delta_fuel')
                active_event['fuel_status'] = active_summary.get('fuel_status')
                active_event['duration_s'] = active_summary.get('duration_s')

    return jsonify(make_json_safe({'history': logs, 'active': active_event}))

@app.route('/api/download_report', methods=['POST'])
def api_download_report():
    data = request.get_json()
    car_id = data.get('car_id', 'Car 2')
    start_date = data.get('start_date', None)
    end_date = data.get('end_date', None)
    event_type = data.get('event_type', 'All')
    
    # 1. Lấy dữ liệu sự kiện từ event_log.jsonl
    raw_logs = _read_event_log(limit=50000, car_id=car_id)
    
    # 2. Lọc theo ngày tháng và trạng thái
    filtered_events = []
    start_dt = None
    end_dt = None
    if start_date:
        start_dt = pd.to_datetime(start_date + "T00:00:00")
    if end_date:
        end_dt = pd.to_datetime(end_date + "T23:59:59")

    for ev in raw_logs:
        if not ev:
            continue
        ev_start = ev.get('start_time')
        if ev_start:
            try:
                ev_dt = pd.to_datetime(ev_start)
                if start_dt and ev_dt < start_dt:
                    continue
                if end_dt and ev_dt > end_dt:
                    continue
            except:
                pass
        
        if event_type != 'All' and ev.get('event') != event_type:
            continue
        
        duration_s = ev.get('duration_s', 0) or 0
        fuel_change = ev.get('fuel_change', ev.get('fuel_added', 0)) or 0
        avg_rate = round((fuel_change / duration_s) * 60, 4) if duration_s else 0.0
        raw_confidence = ev.get('confidence', 0) or 0
        raw_max_confidence = ev.get('max_confidence', raw_confidence) or 0
        
        filtered_events.append({
            'state': ev.get('event', 'Unknown'),
            'start_time': ev.get('start_time'),
            'end_time': ev.get('end_time'),
            'start_fuel': ev.get('fuel_before', 0),
            'end_fuel': ev.get('fuel_after', 0),
            'delta': fuel_change,
            'duration_s': duration_s,
            'avg_rate': avg_rate,
            'max_confidence': raw_max_confidence * 100
        })

    # 3. Tạo file Excel
    output = io.BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "Báo cáo sự kiện"
    
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
    
    # Dòng 1 & 2: Tiêu đề chính
    ws.merge_cells('A1:I1')
    ws['A1'] = "BÁO CÁO HÀNH TRÌNH XE CHẠY"
    ws['A1'].font = Font(bold=True, size=18, name='Times New Roman')
    ws['A1'].alignment = Alignment(horizontal='center', vertical='center')
    
    ws.merge_cells('A2:I2')
    ws['A2'] = "(Theo QCVN 06:2024/BCA)"
    ws['A2'].font = Font(bold=True, size=12, name='Times New Roman')
    ws['A2'].alignment = Alignment(horizontal='center', vertical='center')
    
    # Dòng 3: Thời gian báo cáo
    start_str = start_date if start_date else "---"
    end_str = end_date if end_date else "---"
    ws.merge_cells('A3:I3')
    ws['A3'] = f"Từ 0 giờ 0 phút ngày {start_str} đến 23 giờ 59 phút ngày {end_str}"
    ws['A3'].font = Font(name='Times New Roman', size=11)
    ws['A3'].alignment = Alignment(horizontal='center', vertical='center')
    
    # Dòng 4 & 5: Đơn vị & Biển số
    ws.merge_cells('A4:I4')
    ws['A4'] = f"Đơn vị kinh doanh vận tải: Demo sensor nhiên liệu"
    ws['A4'].font = Font(name='Times New Roman', size=11)
    ws['A4'].alignment = Alignment(horizontal='center', vertical='center')
    
    ws.merge_cells('A5:I5')
    ws['A5'] = f"Biển số: {car_id}"
    ws['A5'].font = Font(name='Times New Roman', size=11)
    ws['A5'].alignment = Alignment(horizontal='center', vertical='center')
    
    # Dòng 7: Header bảng (đã chuẩn hóa tiếng Việt)
    row_offset = 7
    headers = [
        "Thời gian bắt đầu", 
        "Thời gian kết thúc", 
        "Nhiên liệu bắt đầu (L)", 
        "Nhiên liệu kết thúc (L)", 
        "Thay đổi nhiên liệu (L)", 
        "Thời lượng", 
        "Tốc độ thay đổi (L/h)",  # Thay thế Avg Rate
        "Loại sự kiện", 
        "Độ tin cậy (%)"
    ]
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=row_offset, column=col_idx)
        cell.value = header
        cell.font = Font(bold=True, name='Times New Roman')
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = thin_border
    
    # 4. Ghi dữ liệu sự kiện vào bảng
    for idx, ev in enumerate(filtered_events, start=1):
        row = row_offset + idx
        
        # Định dạng thời gian giờ:phút:giây - ngày/tháng/năm
        start_str = ev.get('start_time', '')
        end_str = ev.get('end_time', '')
        try:
            if start_str:
                start_dt = pd.to_datetime(start_str)
                start_str = start_dt.strftime('%H:%M:%S - %d/%m/%Y')
            if end_str:
                end_dt = pd.to_datetime(end_str)
                end_str = end_dt.strftime('%H:%M:%S - %d/%m/%Y')
        except:
            pass

        # Định dạng thời lượng dạng 1h 20m 30s
        dur_s = ev.get('duration_s', 0)
        h = int(dur_s // 3600)
        m = int((dur_s % 3600) // 60)
        s = int(dur_s % 60)
        dur_str = f"{h}h {m}m {s}s" if dur_s else "--"

        # Tốc độ thay đổi L/h
        rate_L_h = ev.get('avg_rate', 0) * 60

        ws.cell(row=row, column=1).value = start_str
        ws.cell(row=row, column=2).value = end_str
        ws.cell(row=row, column=3).value = round(ev.get('start_fuel', 0), 2)
        ws.cell(row=row, column=4).value = round(ev.get('end_fuel', 0), 2)
        ws.cell(row=row, column=5).value = round(ev.get('delta', 0), 2)
        ws.cell(row=row, column=6).value = dur_str
        ws.cell(row=row, column=7).value = round(rate_L_h, 2)
        ws.cell(row=row, column=8).value = ev.get('state', '')
        ws.cell(row=row, column=9).value = round(ev.get('max_confidence', 0), 2)
        
        for col_idx in range(1, 10):
            ws.cell(row=row, column=col_idx).border = thin_border
            
    # Điều chỉnh độ rộng cột
    for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        ws.column_dimensions[col].width = 18

    wb.save(output)
    output.seek(0)
    return send_file(output, download_name=f'FuelSentinel_Report_{car_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx', as_attachment=True)

@app.route('/api/dashboard')
def api_dashboard():
    car_id = request.args.get('car_id', current_car)
    state = get_car_state(car_id)
    with state.lock:
        if not state.processed_buffer:
            return jsonify(make_json_safe({"vehicle": car_id, "fuel": 0, "speed": 0, "status": "Idle", "point_status": "normal", "slope": 0, "r2": 0, "confidence": 0, "stop_duration": 0, "alerts": 0, "last_update": datetime.now().isoformat()}))
        last = state.processed_buffer[-1]
        alerts_count = len(state.event_history)
    return jsonify(make_json_safe({
        "vehicle": car_id,
        "fuel": last['Fuel'],
        "speed": last['Speed'],
        "status": last['Prediction'],
        "point_status": last.get('PointStatus', 'normal'),
        "slope": last['RegressionSlope'],
        "r2": 0,
        "confidence": last['Confidence'],
        "stop_duration": last['StopDuration'],
        "alerts": alerts_count,
        "last_update": last['Timestamp'].isoformat() if hasattr(last['Timestamp'], 'isoformat') else str(last['Timestamp'])
    }))

@app.route('/api/tracker_status')
def api_tracker_status():
    car_id = request.args.get('car_id', current_car)
    state = get_car_state(car_id)
    tracker = state.tracker
    full_df = None
    with state.lock:
        if state.processed_buffer:
            full_df = pd.DataFrame(state.processed_buffer)
            full_df.rename(columns={
                'Timestamp': 'timestamp',
                'Fuel': 'fuel',
                'Speed': 'speed',
                'Prediction': 'prediction',
                'PointStatus': 'point_status',
                'Confidence': 'confidence'
            }, inplace=True)
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
        current_window = window_manager.get_window_size(tracker)
    active_summary = tracker.get_active_summary(full_df)
    status_info = {
        'car_id': car_id,
        'status': tracker.status,
        'point_status': tracker.point_status,
        'display_stage': tracker.display_stage,
        'current_label': tracker.candidate_label if tracker.status == 'candidate' else (tracker.event_record['event'] if tracker.event_record else (tracker.last_prediction or 'Normal')),
        'last_prediction': tracker.last_prediction,
        'confidence': tracker.best_conf,
        'current_window_size': current_window,
        'confidence_history': [
            {'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
             'label': lbl, 'confidence': conf}
            for ts, lbl, conf in tracker.confidence_history[-50:]
        ],
        'active_event': tracker.event_record,
        'active_summary': active_summary,
        'has_history': len(state.event_history) > 0,
    }
    return jsonify(make_json_safe(status_info))

@app.route('/api/report')
def api_report():
    car_id = request.args.get('car_id', current_car)
    state = get_car_state(car_id)
    with state.lock:
        if not state.processed_buffer:
            return "No data", 404
        df = pd.DataFrame(state.processed_buffer)
        df['Timestamp'] = df['Timestamp'].apply(lambda t: t.isoformat() if hasattr(t, 'isoformat') else str(t))
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='FuelSentinel_AI_Data', index=False)
    output.seek(0)
    return send_file(output, download_name=f'FuelSentinel_AI_Report_{car_id}.xlsx', as_attachment=True)

if __name__ == '__main__':
    logger.info("🚀 FuelSentinel-AI Server starting...")
    logger.info(f"   Model: {MODEL_PATH}")
    logger.info(f"   Device: {DEVICE}")
    logger.info(f"   Adaptive Window ranges: {WINDOW_RANGES}")
    logger.info(f"   Thresholds: candidate={CANDIDATE_THRESH}, confirm={CONFIRM_THRESH}, finish={FINISH_THRESH}, mismatch_tolerance={CANDIDATE_MISMATCH_TOLERANCE}")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)