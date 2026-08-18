#!/usr/bin/env python3
"""
flask_segment_viewer.py — FuelSentinel-AI Segment Inspector (by Car & Date)

A simple Flask web app to inspect segment boundaries on a daily basis.
Select a Car and a Date; the fuel trace is drawn with alternating coloured
background bands for each segment. Boundaries are clearly marked.

Usage:
    python flask_segment_viewer.py
    then open http://127.0.0.1:5000 in your browser.

Dependencies: flask, pandas, matplotlib
"""

import base64
import io
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from flask import Flask, render_template_string, request

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CSV_PATH = "data/processed/segment/segment_data.csv"

# Column names inside the CSV (as exported by segment_detection.py)
TIME_COL = "FuelTime"
FUEL_COL = "FuelLevel"
SPEED_COL = "Speed"
CAR_COL = "car_id"
SEG_COL = "segment_id"
BEHAVIOR_COL = "behavior_state"   # optional, for display

# Plot appearance
SEGMENT_ALPHA = 0.15              # transparency of segment background bands
FUEL_LINE_COLOR = "blue"
FUEL_LINE_WIDTH = 1.2

app = Flask(__name__)

# --------------------------------------------------------------------------
# Load data once at startup
# --------------------------------------------------------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Segment data not found at {CSV_PATH}. Adjust CSV_PATH in the script."
    )

df = pd.read_csv(CSV_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL])
df = df.sort_values([CAR_COL, TIME_COL])

# Add date column for easy filtering
df["date"] = df[TIME_COL].dt.date

# Build lookup: {car: [list of dates], ...}
car_dates = defaultdict(list)
for car in df[CAR_COL].unique():
    dates = sorted(df[df[CAR_COL] == car]["date"].unique())
    car_dates[car] = [str(d) for d in dates]

# --------------------------------------------------------------------------
# HTML template
# --------------------------------------------------------------------------
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Segment Viewer (Daily) – FuelSentinel-AI</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 30px;
            background-color: #f9f9f9;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h2 { color: #333; }
        form {
            background: #fff; padding: 15px; border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
        }
        select, input { padding: 8px; margin-right: 10px; font-size: 14px; }
        button {
            padding: 8px 20px; background: #007bff; color: white;
            border: none; border-radius: 4px; cursor: pointer; font-size: 14px;
        }
        button:hover { background: #0056b3; }
        .info {
            margin: 15px 0; padding: 10px; background: #eef;
            border-left: 4px solid #007bff; border-radius: 4px;
        }
        .plot-container img {
            border: 1px solid #ddd;
            border-radius: 4px;
            max-width: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .note { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🧪 Segment Viewer — FuelSentinel-AI</h2>
        <form method="GET" action="/plot">
            <label><strong>Car ID:</strong>
                <select name="car_id">
                    {% for car in cars %}
                        <option value="{{ car }}" {% if car == selected_car %}selected{% endif %}>{{ car }}</option>
                    {% endfor %}
                </select>
            </label>
            <label><strong>Date:</strong>
                <select name="date">
                    {% for d in dates %}
                        <option value="{{ d }}" {% if d == selected_date %}selected{% endif %}>{{ d }}</option>
                    {% endfor %}
                </select>
            </label>
            <button type="submit">🔍 View Daily Segments</button>
        </form>

        {% if plot_url %}
            <div class="info">
                <strong>Car:</strong> {{ selected_car }} |
                <strong>Date:</strong> {{ selected_date }} |
                <strong>Number of segments:</strong> {{ num_segments }} |
                <strong>Total points:</strong> {{ total_points }}
            </div>
            <div class="plot-container">
                <img src="data:image/png;base64,{{ plot_url }}" alt="Segment chart">
            </div>
        {% else %}
            <p class="note">Select a Car and a Date to visualise segment boundaries for that day.</p>
        {% endif %}
    </div>
</body>
</html>
"""

# --------------------------------------------------------------------------
# Helper: build daily segment plot with coloured background per segment
# --------------------------------------------------------------------------
def make_daily_plot(day_df: pd.DataFrame, car: str, date_str: str) -> str:
    """
    Plot fuel level vs time for one day, with segment boundaries highlighted
    by alternating background colours (axvspan). Each segment also shows its
    ID at the top.
    """
    # Sort by time just to be safe
    day_df = day_df.sort_values(TIME_COL)

    # Get unique segments in chronological order
    segments = day_df[SEG_COL].unique()
    # Sort segments by their first timestamp
    seg_start = day_df.groupby(SEG_COL)[TIME_COL].min()
    segments = sorted(segments, key=lambda s: seg_start[s])

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot fuel line across the entire day (single line)
    ax.plot(day_df[TIME_COL], day_df[FUEL_COL],
            color=FUEL_LINE_COLOR, linewidth=FUEL_LINE_WIDTH, zorder=5)

    # Colour palette for segment backgrounds (soft, alternating)
    colors = ["#e6f2ff", "#fff0e6", "#e6ffe6", "#ffe6f0", "#f0e6ff", "#ffffe6"]
    y_min = day_df[FUEL_COL].min()
    y_max = day_df[FUEL_COL].max()
    y_range = y_max - y_min
    # Extend a little for text placement
    y_low = y_min - 0.05 * y_range if y_range > 0 else y_min - 0.5
    y_high = y_max + 0.15 * y_range if y_range > 0 else y_max + 0.5

    for i, seg_id in enumerate(segments):
        seg_mask = day_df[SEG_COL] == seg_id
        seg_times = day_df.loc[seg_mask, TIME_COL]
        start = seg_times.min()
        end = seg_times.max()
        color = colors[i % len(colors)]

        # Background band
        ax.axvspan(start, end, facecolor=color, alpha=SEGMENT_ALPHA, zorder=0)

        # Draw a vertical dashed line at the start of the segment (except first)
        if i > 0:
            ax.axvline(start, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

        # Segment ID label at the top
        mid = start + (end - start) / 2
        ax.text(mid, y_high, str(seg_id),
                ha="center", va="bottom", fontsize=8,
                color="black", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # Add a final boundary at the end of the last segment
    if segments:
        last_end = day_df[day_df[SEG_COL] == segments[-1]][TIME_COL].max()
        ax.axvline(last_end, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    # Formatting
    ax.set_xlabel("Time")
    ax.set_ylabel("Fuel Level")
    ax.set_title(f"Car: {car} — Date: {date_str} — Fuel trace with segment boundaries")
    ax.set_ylim(y_low, y_high)

    # Time axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    fig.autofmt_xdate()

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    """Home page with selection form. Default to first car and its first date."""
    cars_list = sorted(car_dates.keys())
    if not cars_list:
        return "No car data found.", 500

    default_car = cars_list[0]
    default_dates = car_dates[default_car]
    default_date = default_dates[0] if default_dates else ""

    return render_template_string(
        TEMPLATE,
        cars=cars_list,
        dates=default_dates,
        selected_car=default_car,
        selected_date=default_date,
        plot_url=None,
        num_segments=0,
        total_points=0
    )

@app.route("/plot")
def plot_daily():
    """Display daily segment plot for chosen car and date."""
    car = request.args.get("car_id", "")
    date_str = request.args.get("date", "")

    # Retrieve all dates for this car (to keep the dropdown populated)
    dates_for_car = car_dates.get(car, [])

    # Filter data
    mask = (df[CAR_COL] == car) & (df["date"] == pd.to_datetime(date_str).date())
    day_df = df[mask].copy()

    if day_df.empty:
        # No data for this combination – show the form with a note
        return render_template_string(
            TEMPLATE,
            cars=sorted(car_dates.keys()),
            dates=dates_for_car,
            selected_car=car,
            selected_date=date_str,
            plot_url=None,
            num_segments=0,
            total_points=0
        )

    plot_b64 = make_daily_plot(day_df, car, date_str)

    num_segs = day_df[SEG_COL].nunique()
    total_pts = len(day_df)

    return render_template_string(
        TEMPLATE,
        cars=sorted(car_dates.keys()),
        dates=dates_for_car,
        selected_car=car,
        selected_date=date_str,
        plot_url=plot_b64,
        num_segments=num_segs,
        total_points=total_pts
    )

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Segment Viewer (daily) on http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)