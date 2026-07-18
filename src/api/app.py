import os
import sys
import io
import threading
import time
import traceback
from flask import Flask, render_template, jsonify, send_file, request
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.getcwd())

from src.data.simulate import SensorSimulator
from src.utils.feature_engineering import calculate_features
from src.detection.rule_based import batch_predict

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# --- THAM SỐ ---
MAX_RAW_HISTORY = 1000
BUFFER_MAX_LEN = 500
SIMULATOR_POLL_INTERVAL = 1.0

# --- KHỞI TẠO SIMULATOR (chạy nền) ---
simulator = SensorSimulator(vehicle_id="VN001", sample_interval=1.0)

# --- BIẾN TOÀN CỤC ---
master_raw = pd.DataFrame(columns=['Timestamp', 'VehicleID', 'Fuel', 'Speed',
                                   'Latitude', 'Longitude', 'ADC'])
processed_buffer = []   # list of dict với Timestamp là datetime
alerts_buffer = []
seen_alert_ids = set()
ALERTS_LOG_FILE = os.path.join(BASE_DIR, '../../data/alerts.txt')
buffer_lock = threading.Lock()

# --- Alert persistence ---
def _ensure_alerts_file():
    if not os.path.exists(ALERTS_LOG_FILE):
        with open(ALERTS_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write('alert_id\tcreated_at\tstate\tstart_time\tend_time\tstart_fuel\tend_fuel\tdelta\tduration_s\tavg_rate\tconfidence\n')


def _parse_alert_line(line):
    parts = line.strip().split('\t')
    if len(parts) < 11 or parts[0] == 'alert_id':
        return None
    try:
        return {
            'id': parts[0],
            'created_at': parts[1],
            'state': parts[2],
            'start_time': parts[3],
            'end_time': parts[4],
            'start_fuel': float(parts[5]),
            'end_fuel': float(parts[6]),
            'delta': float(parts[7]),
            'duration_s': int(parts[8]),
            'avg_rate': float(parts[9]),
            'confidence': float(parts[10])
        }
    except ValueError:
        return None


def load_alerts_log():
    global alerts_buffer, seen_alert_ids
    _ensure_alerts_file()
    alerts_buffer = []
    seen_alert_ids = set()
    with open(ALERTS_LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            alert = _parse_alert_line(line)
            if alert:
                alerts_buffer.append(alert)
                seen_alert_ids.add(alert['id'])


def append_alert(alert):
    if alert['id'] in seen_alert_ids:
        return
    seen_alert_ids.add(alert['id'])
    alerts_buffer.append(alert)
    with open(ALERTS_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('\t'.join([
            alert['id'],
            alert['created_at'],
            alert['state'],
            alert['start_time'],
            alert['end_time'],
            f"{alert['start_fuel']:.2f}",
            f"{alert['end_fuel']:.2f}",
            f"{alert['delta']:.2f}",
            str(alert['duration_s']),
            f"{alert['avg_rate']:.4f}",
            f"{alert['confidence']:.4f}"
        ]) + '\n')


def build_events_from_processed(rows):
    events = []
    prev_state = None
    current = None

    def make_idle_event(event_state, item):
        return {
            'id': f"{item['Timestamp'].isoformat()}_{event_state}_{float(item['Fuel']):.2f}",
            'state': event_state,
            'start_time': item['Timestamp'].isoformat(),
            'end_time': item['Timestamp'].isoformat(),
            'start_fuel': float(item['Fuel']),
            'end_fuel': float(item['Fuel']),
            'delta': 0.0,
            'duration_s': 0,
            'avg_rate': 0.0,
            'confidence': float(item.get('Confidence', 0) or 0),
            'created_at': item['Timestamp'].isoformat()
        }

    def close_segment(segment):
        start = segment['start_time']
        end = segment['end_time']
        duration_s = int((end - start).total_seconds())
        delta = segment['end_fuel'] - segment['start_fuel']
        avg_conf = sum(segment['confidences']) / len(segment['confidences']) if segment['confidences'] else 0.0
        return {
            'id': f"{start.isoformat()}_{segment['state']}_{segment['start_fuel']:.2f}_{segment['end_fuel']:.2f}",
            'state': segment['state'],
            'start_time': start.isoformat(),
            'end_time': end.isoformat(),
            'start_fuel': segment['start_fuel'],
            'end_fuel': segment['end_fuel'],
            'delta': round(delta, 2),
            'duration_s': duration_s,
            'avg_rate': round((delta / duration_s * 60) if duration_s else 0.0, 4),
            'confidence': round(avg_conf, 4),
            'created_at': end.isoformat()
        }

    for item in rows:
        state = item.get('Prediction')
        fuel = float(item['Fuel'])
        confidence = float(item.get('Confidence', 0) or 0)

        if prev_state is None:
            if state == 'Idle':
                events.append(make_idle_event('Idle Start', item))
            if state in ['Refuel', 'Fuel Theft']:
                current = {
                    'state': state,
                    'start_time': item['Timestamp'],
                    'end_time': item['Timestamp'],
                    'start_fuel': fuel,
                    'end_fuel': fuel,
                    'confidences': [confidence]
                }
        else:
            if state != prev_state:
                if prev_state == 'Idle':
                    events.append(make_idle_event('Idle End', item))
                if state == 'Idle':
                    events.append(make_idle_event('Idle Start', item))
                if prev_state in ['Refuel', 'Fuel Theft'] and current is not None:
                    events.append(close_segment(current))
                    current = None
                if state in ['Refuel', 'Fuel Theft']:
                    current = {
                        'state': state,
                        'start_time': item['Timestamp'],
                        'end_time': item['Timestamp'],
                        'start_fuel': fuel,
                        'end_fuel': fuel,
                        'confidences': [confidence]
                    }
            elif state in ['Refuel', 'Fuel Theft'] and current is not None:
                current['end_time'] = item['Timestamp']
                current['end_fuel'] = fuel
                current['confidences'].append(confidence)

        prev_state = state

    if current is not None:
        events.append(close_segment(current))

    return list(reversed(events))


def update_alerts_from_processed_buffer():
    events = build_events_from_processed(processed_buffer)
    for event in events:
        append_alert(event)


# --- HÀM TIỆN ÍCH ---
def rebuild_processed_buffer():
    global processed_buffer
    if master_raw.empty:
        processed_buffer = []
        return

    try:
        df = master_raw.sort_values('Timestamp').reset_index(drop=True)
        print(f"🔧 rebuild_processed_buffer: {len(df)} mẫu")
        df = calculate_features(df)
        df = batch_predict(df)

        # Giới hạn số dòng trong buffer
        df = df.tail(BUFFER_MAX_LEN)

        new_buffer = []
        for _, row in df.iterrows():
            new_buffer.append({
                "Timestamp": row['Timestamp'],
                "Fuel": row['Fuel'],
                "Speed": row['Speed'],
                "ADC": row['ADC'],
                "Latitude": row['Latitude'],
                "Longitude": row['Longitude'],
                "InstantFuelDiff": row['InstantFuelDiff'],
                "FuelRate": row['FuelRate'],
                "MovingAvg": row['MovingAvg'],
                "RollingStd": row['RollingStd'],
                "MaxJump": row['MaxJump'],
                "WindowFuelDiff": row['WindowFuelDiff'],
                "StopDuration": row['StopDuration'],
                "AvgSpeed": row['AvgSpeed'],
                "FuelRange": row['FuelRange'],
                "RegressionSlope": row['RegressionSlope'],
                "RegressionR2": row['RegressionR2'],
                "Prediction": row['Prediction'],
                "Confidence": row['Confidence']
            })
        processed_buffer = new_buffer
        update_alerts_from_processed_buffer()
    except Exception as e:
        print(f"❌ Lỗi trong rebuild_processed_buffer: {e}")
        traceback.print_exc()

def append_raw_rows(new_rows: pd.DataFrame):
    global master_raw
    if new_rows.empty:
        return

    existing_timestamps = set(master_raw['Timestamp'])
    new_rows = new_rows[~new_rows['Timestamp'].isin(existing_timestamps)]
    if new_rows.empty:
        return

    master_raw = pd.concat([master_raw, new_rows], ignore_index=True)
    if len(master_raw) > MAX_RAW_HISTORY:
        master_raw = master_raw.iloc[-MAX_RAW_HISTORY:]
    master_raw.reset_index(drop=True, inplace=True)

def load_initial_data_from_csv():
    csv_path = os.path.join(BASE_DIR, '../../data/sample/simulated_data.csv')
    if not os.path.exists(csv_path):
        print("⚠️ Không tìm thấy file simulated_data.csv.")
        return

    try:
        df = pd.read_csv(csv_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
        print(f"📂 Đọc CSV: {len(df)} dòng")
        append_raw_rows(df)
        rebuild_processed_buffer()
        print(f"✅ Đã nạp {len(df)} mẫu từ CSV vào lịch sử.")
    except Exception as e:
        print(f"❌ Lỗi khi đọc CSV: {e}")
        traceback.print_exc()

# Nạp alerts từ file nếu tồn tại
load_alerts_log()

# Nạp dữ liệu ban đầu
load_initial_data_from_csv()

# --- LUỒNG XỬ LÝ DỮ LIỆU MỚI TỪ SIMULATOR ---
def process_new_samples():
    while True:
        try:
            latest = simulator.get_latest_raw()
            if latest is not None:
                new_row = pd.DataFrame([latest])
                new_row['Timestamp'] = pd.to_datetime(new_row['Timestamp'])
                with buffer_lock:
                    append_raw_rows(new_row)
                    rebuild_processed_buffer()
        except Exception as e:
            print(f"Lỗi trong process_new_samples: {e}")
            traceback.print_exc()
        time.sleep(SIMULATOR_POLL_INTERVAL)

threading.Thread(target=process_new_samples, daemon=True).start()

# --- ROUTES ---
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/feature')
def feature():
    return render_template('feature.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/events')
def events():
    return render_template('events.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/api/history')
def api_history():
    with buffer_lock:
        history = processed_buffer[-100:] if processed_buffer else []

    result = []
    for item in history:
        result.append({
            "raw": {
                "Timestamp": item['Timestamp'].isoformat(),
                "Fuel": item['Fuel'],
                "Speed": item['Speed'],
                "ADC": item['ADC'],
                "Latitude": item['Latitude'],
                "Longitude": item['Longitude']
            },
            "processed": {
                "Prediction": item['Prediction'],
                "RegressionSlope": item['RegressionSlope'],
                "RegressionR2": item['RegressionR2'],
                "FuelRate": item['FuelRate'],
                "InstantFuelDiff": item['InstantFuelDiff'],
                "WindowFuelDiff": item['WindowFuelDiff'],
                "MaxJump": item['MaxJump'],
                "MovingAvg": item['MovingAvg'],
                "RollingStd": item['RollingStd'],
                "StopDuration": item['StopDuration'],
                "AvgSpeed": item['AvgSpeed'],
                "FuelRange": item['FuelRange'],
                "Confidence": item.get('Confidence', 0.0)
            }
        })
    return jsonify(result)

# ==========================================
# BỔ SUNG API /api/data ĐỂ CÁC TRANG KHÁC CÓ THỂ GỌI
# ==========================================
@app.route('/api/data')
def api_data():
    with buffer_lock:
        history = processed_buffer[-100:] if processed_buffer else []

    result = []
    for item in history:
        result.append({
            "Timestamp": item['Timestamp'].isoformat(),
            "VehicleID": "VN001",
            "Fuel": item['Fuel'],
            "Speed": item['Speed'],
            "Prediction": item['Prediction'],
            "FuelRate": item['FuelRate'],
            "MovingAvg": item['MovingAvg'],
            "RollingStd": item['RollingStd'],
            "WindowFuelDiff": item['WindowFuelDiff'],
            "MaxJump": item['MaxJump'],
            "RegressionSlope": item['RegressionSlope'],
            "StopDuration": item['StopDuration'],
            "Confidence": item['Confidence']
        })
    return jsonify(result)
# ==========================================

@app.route('/api/realtime')
def api_realtime():
    with buffer_lock:
        if not processed_buffer:
            return jsonify({
                "raw": {
                    "Timestamp": pd.Timestamp.now().isoformat(),
                    "Fuel": 50.0,
                    "Speed": 0,
                    "ADC": 1900,
                    "Latitude": 21.0285,
                    "Longitude": 105.8541
                },
                "processed": {
                    "Prediction": "Idle",
                    "RegressionSlope": 0,
                    "RegressionR2": 0,
                    "FuelRate": 0,
                    "InstantFuelDiff": 0,
                    "WindowFuelDiff": 0,
                    "MaxJump": 0,
                    "MovingAvg": 50,
                    "RollingStd": 0,
                    "StopDuration": 0,
                    "AvgSpeed": 0,
                    "FuelRange": 0,
                    "Confidence": 0.0
                }
            })
        last = processed_buffer[-1]

    return jsonify({
        "raw": {
            "Timestamp": last['Timestamp'].isoformat(),
            "Fuel": last['Fuel'],
            "Speed": last['Speed'],
            "ADC": last['ADC'],
            "Latitude": last['Latitude'],
            "Longitude": last['Longitude']
        },
        "processed": {
            "Prediction": last['Prediction'],
            "RegressionSlope": last['RegressionSlope'],
            "RegressionR2": last['RegressionR2'],
            "FuelRate": last['FuelRate'],
            "InstantFuelDiff": last['InstantFuelDiff'],
            "WindowFuelDiff": last['WindowFuelDiff'],
            "MaxJump": last['MaxJump'],
            "MovingAvg": last['MovingAvg'],
            "RollingStd": last['RollingStd'],
            "StopDuration": last['StopDuration'],
            "AvgSpeed": last['AvgSpeed'],
            "FuelRange": last['FuelRange'],
            "Confidence": last.get('Confidence', 0.0)
        }
    })

@app.route('/api/events')
def api_events():
    with buffer_lock:
        if not processed_buffer:
            return jsonify([])
        events = build_events_from_processed(processed_buffer)
    return jsonify(events)

@app.route('/api/alerts')
def api_alerts():
    with buffer_lock:
        return jsonify(alerts_buffer[-50:])

@app.route('/api/dashboard')
def api_dashboard():
    with buffer_lock:
        if not processed_buffer:
            return jsonify({
                "vehicle": "VN001",
                "fuel": 50.0,
                "speed": 0,
                "status": "Idle",
                "slope": 0,
                "r2": 0,
                "confidence": 0.0,
                "stop_duration": 0,
                "alerts": 0,
                "last_update": pd.Timestamp.now().isoformat()
            })
        last = processed_buffer[-1]
        alert_count = len(alerts_buffer)
    return jsonify({
        "vehicle": "VN001",
        "fuel": last['Fuel'],
        "speed": last['Speed'],
        "status": last['Prediction'],
        "slope": last['RegressionSlope'],
        "r2": last['RegressionR2'],
        "confidence": last.get('Confidence', 0.0),
        "stop_duration": last['StopDuration'],
        "alerts": alert_count,
        "last_update": last['Timestamp'].isoformat()
    })

@app.route('/api/report')
def api_report():
    with buffer_lock:
        if not processed_buffer:
            return "No data", 404
        df = pd.DataFrame(processed_buffer)
        df['Timestamp'] = df['Timestamp'].apply(lambda t: t.isoformat())
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='FuelSentinel_Data', index=False)
    output.seek(0)
    return send_file(output, download_name='FuelSentinel_Report.xlsx', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False, port=5000)