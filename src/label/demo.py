#!/usr/bin/env python3
"""
Flask demo — Fuel timeline with state shading per car and per day.
- Dùng POST để tạo ảnh khi bấm nút.
- Xử lý đúng tên cột gốc FuelTime, FuelLevel.
"""

import io
import sys
import base64
from pathlib import Path
from typing import List

# Đặt backend Matplotlib TRƯỚC KHI import pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
from flask import Flask, request, render_template_string, Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH: str = "data/processed/labels/timeline_labeled.csv"
LABEL_COLORS = {
    "Idle": "green",
    "Driving": "blue",
    "Refuel": "orange",
    "Theft": "red",
    "NULL": "gray",
}

app = Flask(__name__)

# Load data once at startup
DATA: pd.DataFrame = pd.DataFrame()
CARS: List[str] = []


def load_data() -> pd.DataFrame:
    """Load timeline CSV, map column names, and precompute date."""
    path = Path(CSV_PATH)
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    # Ánh xạ tên cột gốc sang tên chuẩn nội bộ
    rename_map = {
        "FuelTime": "timestamp",
        "Lat": "latitude",
        "Lng": "longitude",
        "Speed": "speed",
        "FuelLevel": "fuel",
    }
    # Chỉ đổi những cột tồn tại
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df = df.rename(columns=existing)

    # Kiểm tra cột bắt buộc
    required = {"timestamp", "fuel", "car_id", "segment_id", "final_label"}
    missing = required - set(df.columns)
    if missing:
        print(f"❌ Thiếu cột: {missing}")
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date.astype(str)  # yyyy-mm-dd
    return df


def get_available_dates(car_id: str) -> List[str]:
    """Trả về danh sách ngày (string) cho xe được chọn."""
    if DATA.empty:
        return []
    mask = DATA["car_id"] == car_id
    return sorted(DATA.loc[mask, "date"].unique())


def generate_plot(car_id: str, date_str: str) -> io.BytesIO:
    """Tạo biểu đồ fuel theo thời gian, tô màu theo final_label."""
    df = DATA[(DATA["car_id"] == car_id) & (DATA["date"] == date_str)].copy()
    if df.empty:
        raise ValueError("Không có dữ liệu cho xe/ngày này.")

    df = df.sort_values("timestamp")

    # Lấy biên segment và nhãn
    seg_info = (
        df.groupby("segment_id")
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            final_label=("final_label", "first"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["timestamp"], df["fuel"], color="black", linewidth=1, label="Fuel")

    used_labels = set()
    for _, seg in seg_info.iterrows():
        label = seg["final_label"]
        color = LABEL_COLORS.get(label, "lightgray")
        legend_label = label if label not in used_labels else None
        ax.axvspan(seg["start"], seg["end"], alpha=0.3, color=color, label=legend_label)
        used_labels.add(label)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Nhiên liệu (L)")
    ax.set_title(f"Car {car_id} — {date_str}")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Fuel Timeline Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; }
        form { margin-bottom: 1rem; }
        select, button { padding: 0.4rem 0.8rem; font-size: 1rem; }
        img { max-width: 100%; border: 1px solid #ccc; margin-top: 1rem; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>📊 Fuel Timeline with State Shading</h1>
    <form id="car-form" method="get" action="/">
        <label for="car">Chọn xe:</label>
        <select name="car_id" id="car" onchange="document.getElementById('car-form').submit()">
            <option value="" {% if not selected_car %}selected{% endif %}>-- Chọn xe --</option>
            {% for c in cars %}
            <option value="{{ c }}" {% if selected_car == c %}selected{% endif %}>{{ c }}</option>
            {% endfor %}
        </select>
    </form>

    {% if selected_car and dates %}
    <form method="post" action="/">
        <input type="hidden" name="car_id" value="{{ selected_car }}">
        <label for="date">Chọn ngày:</label>
        <select name="date" id="date">
            {% for d in dates %}
            <option value="{{ d }}" {% if selected_date == d %}selected{% endif %}>{{ d }}</option>
            {% endfor %}
        </select>
        <button type="submit">Vẽ biểu đồ</button>
    </form>
    {% endif %}

    {% if error %}
    <p class="error">{{ error }}</p>
    {% endif %}

    {% if plot_data %}
    <img src="data:image/png;base64,{{ plot_data }}" alt="fuel plot">
    {% endif %}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    selected_car = None
    selected_date = None
    dates: List[str] = []
    error = None
    plot_data = None

    if request.method == "POST":
        # Lấy tham số từ form POST
        selected_car = request.form.get("car_id", "")
        selected_date = request.form.get("date", "")
        if selected_car and selected_date:
            try:
                buf = generate_plot(selected_car, selected_date)
                plot_data = base64.b64encode(buf.read()).decode("utf-8")
            except ValueError as e:
                error = str(e)
            except Exception as e:
                error = f"Lỗi khi vẽ biểu đồ: {e}"
        # Sau khi POST, vẫn cần trả về danh sách ngày cho xe đó (để form hiển thị đúng)
        if selected_car:
            dates = get_available_dates(selected_car)
    else:
        # GET request: hiển thị form, có thể có car_id từ query string
        selected_car = request.args.get("car_id", "")
        if selected_car:
            dates = get_available_dates(selected_car)
        # Không vẽ plot khi GET

    return render_template_string(
        PAGE_TEMPLATE,
        cars=CARS,
        selected_car=selected_car,
        dates=dates,
        selected_date=selected_date,
        plot_data=plot_data,
        error=error,
    )


@app.route("/plot")
def plot_direct():
    """Đường dẫn trực tiếp tới ảnh PNG (dùng GET)."""
    car_id = request.args.get("car_id")
    date_str = request.args.get("date")
    if not car_id or not date_str:
        return "Thiếu car_id hoặc date", 400
    try:
        buf = generate_plot(car_id, date_str)
        return Response(buf.getvalue(), mimetype="image/png")
    except ValueError:
        return "Không có dữ liệu", 404


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA = load_data()
    CARS = sorted(DATA["car_id"].unique().tolist())
    print(f"✅ Đã tải {len(DATA)} dòng từ {CSV_PATH}")
    print(f"🚗 Xe tìm thấy: {', '.join(CARS)}")
    app.run(host="0.0.0.0", port=5000, debug=True)