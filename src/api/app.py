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
buffer_lock = threading.Lock()

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
    with simulator.lock:
        events = simulator.events.copy()
    return jsonify(events)

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
        with simulator.lock:
            event_count = len([e for e in simulator.events if e['state'] in ['Refuel', 'Fuel Theft']])
    return jsonify({
        "vehicle": "VN001",
        "fuel": last['Fuel'],
        "speed": last['Speed'],
        "status": last['Prediction'],
        "slope": last['RegressionSlope'],
        "r2": last['RegressionR2'],
        "confidence": last.get('Confidence', 0.0),
        "stop_duration": last['StopDuration'],
        "alerts": event_count,
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