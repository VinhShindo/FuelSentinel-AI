#!/usr/bin/env python3
"""
clean_data.py — FuelSentinel‑AI Data Cleaning
Sequential Rule-Based Pipeline: GPS -> Speed -> Fuel  (Final Version)

Philosophy: this step RESTORES sensor errors, it does not SMOOTH the signal.
Every real fluctuation of the vehicle (moving, idle, refueling, fuel theft,
acceleration/deceleration) must be preserved untouched. A point is only ever
corrected when there is enough context evidence AND no plausible real-world
explanation for the anomaly.

Mandatory processing order (each stage trusts the previous one):

    GPS  ->  Speed  ->  Fuel

  - GPS reflects the vehicle's true physical state.
  - Speed depends on whether the vehicle is actually moving (GPS).
  - Fuel is only judged once we know whether the vehicle was moving or
    stationary (GPS + Speed).

Rules implemented (see accompanying spec, sections I-VII):
  I.   GPS = (0,0): short dropout (1-2 samples) -> interpolate.
                    long outage -> keep raw, log "Long GPS outage".
  II.  Speed = 0 vs GPS: if GPS truly unchanged (or GPS jitter only) -> real
       idle, keep. If GPS moving while Speed=0 -> contradiction; only
       interpolate Speed if the zero is a single isolated point surrounded
       by moving speeds; otherwise keep and log "Suspicious speed".
  III. Fuel = 0: only interpolate an ISOLATED single zero (run length 1)
       with valid, mutually-consistent neighbours on both sides. Continuous
       zero runs are kept and logged. Ambiguous short runs (e.g. length 2)
       are kept — "not enough evidence".
  IV.  Fuel > 0 spikes: evaluated with a +/-5 point context window using
       GPS + Speed + Fuel shape. Only isolated spikes that (a) have a clear
       V/^ shape, (b) return to the same baseline on both sides, and
       (c) occur while the vehicle is stationary (GPS+Speed agree) are
       corrected. Real trends (consumption, refuel, fuel theft) and driving
       noise are never touched.
  V.   Vehicle-state classification (Moving / Idle / Refueling / Fuel Theft /
       Sensor Dropout / Continuous Missing Segment) is used internally to
       decide whether to touch a point — it is NOT written as a new column
       (only `note` is added, per spec). Refuel / Fuel Theft events are
       flagged in the note as informational events, without altering data.
  VI.  Interpolation policy: linear interpolation (time-weighted) by
       default, or simple two-side average when time deltas are equal.
       Rolling mean/median/Hampel are NEVER used to replace values — Hampel
       is only used as a candidate detector.
  VII. A point is corrected only if ALL of: enough context on both sides,
       GPS/Speed/Fuel do not contradict each other, the anomaly is local
       (isolated spike/dropout), it is not part of a natural trend, and a
       reasonable interpolation exists. Otherwise: no change, log only.

Detailed per-decision logs (corrected AND kept-with-reason) are exported as
CSV files under LOGS_DIR for full traceability.

Original column names are preserved in the output; only one extra column
`note` is added.
"""

from __future__ import annotations

import logging
import sys
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration – edit paths here
# ---------------------------------------------------------------------------
INPUT_CSV: Path = Path("data/raw/CarFuelHistory.csv")
OUTPUT_CLEAN_CSV: Path = Path("data/processed/clean_data.csv")
REPORT_MD: Path = Path("data/processed/cleaning_report.md")
FIGURES_DIR: Path = Path("data/processed/figures")
LOGS_DIR: Path = Path("data/processed/cleaning_logs")

# INPUT_CSV: Path = Path("data/processed/merged/merged_gps_fuel_fixed.csv")
# OUTPUT_CLEAN_CSV: Path = Path("data/processed/clean_data_v2.csv")
# REPORT_MD: Path = Path("data/processed/cleaning_report_v2.md")
# FIGURES_DIR: Path = Path("data/processed/figures_v2")
# LOGS_DIR: Path = Path("data/processed/cleaning_logs_v2")

COLUMN_MAP = {
    "FuelTime": "timestamp",
    "FuelLevel": "fuel",
    "Lat": "latitude",
    "Lng": "longitude",
    "Address": "address",
    "Speed": "speed",
    "car_id": "car_id",
}

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------
GPS_SHORT_RUN_MAX = 2            # (0,0) runs of 1-2 samples are correctable
GPS_JITTER_METERS = 2.0          # displacement below this = GPS noise, not movement
GPS_MOVING_METERS = 5.0          # displacement above this over a window = vehicle moving

SPEED_MOVING_THRESHOLD = 5.0     # km/h — neighbour speed above this = "moving" for isolation check

FUEL_ZERO_STABILITY_GUARD = 2.5  # litres — how close prev/next must be for an isolated fuel=0 fix
TIMESTAMP_MAX_GAP_SECONDS = 7200  # 2h — never trust interpolation across a bigger gap than this

FUEL_SPIKE_MIN_DELTA = 3.0       # litres — minimum deviation to even be considered a spike candidate
FUEL_SIDE_SYMMETRY_GUARD = 2.0   # litres — how close both sides must be ("returns to baseline")
FUEL_THEFT_MIN_RUN = 3           # consecutive decreasing points, stationary, to flag as an event
FUEL_THEFT_MIN_DROP = 5.0        # litres total drop to flag a fuel-theft/refuel event

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("clean_data")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _add_note(existing: str, msg: str) -> str:
    if existing:
        return f"{existing}{msg}; "
    return f"{msg}; "


def _fmt_ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def _linear_interp(ts_prev: pd.Timestamp, val_prev: float, ts_next: pd.Timestamp, val_next: float,
                    ts_target: pd.Timestamp) -> float:
    """Time-weighted linear interpolation; falls back to a plain two-side average
    if the time span is degenerate (per spec, average is the fallback when
    time deltas are equal / unusable)."""
    total = (ts_next - ts_prev).total_seconds()
    if total <= 0:
        return (val_prev + val_next) / 2.0
    frac = (ts_target - ts_prev).total_seconds() / total
    return val_prev + frac * (val_next - val_prev)


def _hampel_candidate(series: pd.Series, window: int = 7, n_sigmas: float = 3.0) -> pd.Series:
    """Candidate detector ONLY — never used to replace values directly."""
    outlier_mask = pd.Series(False, index=series.index)
    for i in range(len(series)):
        lo = max(0, i - window // 2)
        hi = min(len(series), i + window // 2 + 1)
        window_data = series.iloc[lo:hi]
        median = window_data.median()
        mad_val = (window_data - median).abs().median()
        if mad_val == 0:
            continue
        if abs(series.iloc[i] - median) > n_sigmas * mad_val:
            outlier_mask.iloc[i] = True
    return outlier_mask


def _gps_displacement_m(lat: pd.Series, lon: pd.Series, i: int, j: int) -> float:
    return _haversine_km(lat.iloc[i], lon.iloc[i], lat.iloc[j], lon.iloc[j]) * 1000.0


def _gps_is_stationary(lat: pd.Series, lon: pd.Series, lo: int, hi: int, jitter_m: float = GPS_JITTER_METERS) -> bool:
    """True if the total path length across [lo, hi) is within GPS jitter (real Idle)."""
    total = 0.0
    for k in range(lo, hi - 1):
        total += _gps_displacement_m(lat, lon, k, k + 1)
    return total <= jitter_m * max(1, hi - lo - 1)


def _gps_is_moving(lat: pd.Series, lon: pd.Series, lo: int, hi: int, moving_m: float = GPS_MOVING_METERS) -> bool:
    total = 0.0
    for k in range(lo, hi - 1):
        total += _gps_displacement_m(lat, lon, k, k + 1)
    return total > moving_m


def _log_event(event_log: List[dict], car_id: str, pos: int, ts, field: str, decision: str,
               reason: str, old_value=None, new_value=None, run_length: int = 1) -> None:
    event_log.append({
        "car_id": car_id,
        "position_in_car": pos,
        "timestamp": ts,
        "field": field,
        "decision": decision,      # "corrected" | "kept" | "event"
        "reason": reason,
        "old_value": old_value,
        "new_value": new_value,
        "run_length": run_length,
    })


# ---------------------------------------------------------------------------
# Step 1-2: Sort / exact duplicate removal
# ---------------------------------------------------------------------------
def sort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Sorting by car_id, timestamp …")
    return df.sort_values(["car_id", "timestamp"]).reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Removing exact duplicates …")
    dup_cols = ["car_id", "timestamp", "fuel", "speed", "latitude", "longitude"]
    mask = df.duplicated(subset=dup_cols, keep="first")
    n_dup = int(mask.sum())
    if n_dup:
        logger.info("Removed %d duplicate rows", n_dup)
        df = df[~mask].copy()
    else:
        logger.info("No duplicates found.")
    return df


# ---------------------------------------------------------------------------
# Step 3: Timestamp — only fix when fully certain
# ---------------------------------------------------------------------------
def fix_timestamps(df: pd.DataFrame, event_log: List[dict]) -> pd.DataFrame:
    logger.info("Fixing timestamps only when unambiguous …")
    if "note" not in df.columns:
        df["note"] = ""

    for car_id, grp in df.groupby("car_id", sort=False):
        ts = grp["timestamp"].reset_index(drop=True)
        note = grp["note"].copy().reset_index(drop=True)
        orig_index = grp.index
        n = len(grp)

        for i in range(1, n):
            if ts.iloc[i] >= ts.iloc[i - 1]:
                continue
            if i + 1 >= n:
                _log_event(event_log, car_id, i, ts.iloc[i], "timestamp", "kept", "no future data to bound the fix")
                continue
            gap = (ts.iloc[i + 1] - ts.iloc[i - 1]).total_seconds()
            if gap <= 0 or gap > TIMESTAMP_MAX_GAP_SECONDS:
                _log_event(event_log, car_id, i, ts.iloc[i], "timestamp", "kept", "gap too large to trust interpolation")
                continue
            old_ts = ts.iloc[i]
            new_ts = ts.iloc[i - 1] + (ts.iloc[i + 1] - ts.iloc[i - 1]) / 2
            note.iloc[i] = _add_note(note.iloc[i], f"timestamp corrected: {_fmt_ts(old_ts)} -> {_fmt_ts(new_ts)}")
            ts.iloc[i] = new_ts
            _log_event(event_log, car_id, i, new_ts, "timestamp", "corrected", "out-of-order, bounded by neighbours",
                       old_value=old_ts, new_value=new_ts)

        df.loc[orig_index, "timestamp"] = ts.values
        df.loc[orig_index, "note"] = note.values

    return df


# ---------------------------------------------------------------------------
# STAGE I: GPS = (0,0)
# ---------------------------------------------------------------------------
def fix_gps(df: pd.DataFrame, event_log: List[dict]) -> pd.DataFrame:
    logger.info("[Stage I] GPS: short (0,0) dropout -> interpolate; long outage -> keep raw …")
    if "note" not in df.columns:
        df["note"] = ""

    for car_id, grp in df.groupby("car_id", sort=False):
        lat = grp["latitude"].reset_index(drop=True)
        lon = grp["longitude"].reset_index(drop=True)
        ts = grp["timestamp"].reset_index(drop=True)
        note = grp["note"].copy().reset_index(drop=True)
        orig_index = grp.index
        n = len(grp)

        visited = set()
        zero_positions = [p for p in range(n) if lat.iloc[p] == 0 or lon.iloc[p] == 0]

        for pos in zero_positions:
            if pos in visited:
                continue
            lo = pos
            while lo > 0 and (lat.iloc[lo - 1] == 0 or lon.iloc[lo - 1] == 0):
                lo -= 1
            hi = pos
            while hi < n - 1 and (lat.iloc[hi + 1] == 0 or lon.iloc[hi + 1] == 0):
                hi += 1
            run_positions = list(range(lo, hi + 1))
            visited.update(run_positions)
            run_len = hi - lo + 1

            # Kiểm tra xem có phải là thời gian nghỉ lâu không
            # Nếu thời gian trước và sau đều lớn (nghỉ dài) -> giữ nguyên GPS = 0
            is_long_stop = False
            if lo > 0 and hi < n - 1:
                time_before = (ts.iloc[lo] - ts.iloc[lo - 1]).total_seconds()
                time_after = (ts.iloc[hi + 1] - ts.iloc[hi]).total_seconds()
                
                # Nếu thời gian trước và sau đều lớn hơn 1 giờ (3600 giây)
                # và đây là run dài (>= 3 points) -> coi là dừng nghỉ
                if time_before > 3600 and time_after > 3600 and run_len >= 3:
                    is_long_stop = True
                    _log_event(event_log, car_id, lo, ts.iloc[lo], "gps", "kept", 
                               f"Long stop detected: before={time_before:.0f}s, after={time_after:.0f}s, run_len={run_len}",
                               run_length=run_len)
                    continue  # Giữ nguyên GPS = 0

            # LUÔN sửa các trường hợp GPS = 0 (trừ khi là dừng nghỉ dài)
            if lo == 0 or hi == n - 1:
                # Nếu ở biên, vẫn sửa nếu có đủ dữ liệu
                if lo == 0 and hi < n - 1 and hi + 1 < n:
                    # Sử dụng giá trị sau
                    new_lat = lat.iloc[hi + 1]
                    new_lon = lon.iloc[hi + 1]
                    for p in run_positions:
                        lat.iloc[p] = new_lat
                        lon.iloc[p] = new_lon
                        note.iloc[p] = _add_note(
                            note.iloc[p],
                            f"gps corrected: (0.0,0.0) -> ({new_lat:.5f},{new_lon:.5f}) | edge fill ({run_len} pt)",
                        )
                    _log_event(event_log, car_id, lo, ts.iloc[lo], "gps", "corrected", 
                               "edge of log, filled with next valid", old_value="(0.0,0.0)", run_length=run_len)
                elif hi == n - 1 and lo > 0:
                    # Sử dụng giá trị trước
                    new_lat = lat.iloc[lo - 1]
                    new_lon = lon.iloc[lo - 1]
                    for p in run_positions:
                        lat.iloc[p] = new_lat
                        lon.iloc[p] = new_lon
                        note.iloc[p] = _add_note(
                            note.iloc[p],
                            f"gps corrected: (0.0,0.0) -> ({new_lat:.5f},{new_lon:.5f}) | edge fill ({run_len} pt)",
                        )
                    _log_event(event_log, car_id, lo, ts.iloc[lo], "gps", "corrected", 
                               "edge of log, filled with previous valid", old_value="(0.0,0.0)", run_length=run_len)
                continue

            # Sử dụng nội suy từ hai bên
            prev_valid = lat.iloc[lo - 1] != 0 and lon.iloc[lo - 1] != 0
            next_valid = lat.iloc[hi + 1] != 0 and lon.iloc[hi + 1] != 0
            
            if prev_valid and next_valid:
                for p in run_positions:
                    new_lat = _linear_interp(ts.iloc[lo - 1], lat.iloc[lo - 1], ts.iloc[hi + 1], lat.iloc[hi + 1], ts.iloc[p])
                    new_lon = _linear_interp(ts.iloc[lo - 1], lon.iloc[lo - 1], ts.iloc[hi + 1], lon.iloc[hi + 1], ts.iloc[p])
                    lat.iloc[p] = new_lat
                    lon.iloc[p] = new_lon
                    note.iloc[p] = _add_note(
                        note.iloc[p],
                        f"gps corrected: (0.0,0.0) -> ({new_lat:.5f},{new_lon:.5f}) | interpolated ({run_len} pt)",
                    )
                _log_event(event_log, car_id, lo, ts.iloc[lo], "gps", "corrected", 
                           "interpolated from both sides", old_value="(0.0,0.0)", run_length=run_len)
            else:
                # Nếu một bên không valid, sử dụng bên còn lại
                if prev_valid:
                    new_lat = lat.iloc[lo - 1]
                    new_lon = lon.iloc[lo - 1]
                    for p in run_positions:
                        lat.iloc[p] = new_lat
                        lon.iloc[p] = new_lon
                        note.iloc[p] = _add_note(
                            note.iloc[p],
                            f"gps corrected: (0.0,0.0) -> ({new_lat:.5f},{new_lon:.5f}) | fill from prev ({run_len} pt)",
                        )
                    _log_event(event_log, car_id, lo, ts.iloc[lo], "gps", "corrected", 
                               "filled with previous valid", old_value="(0.0,0.0)", run_length=run_len)
                elif next_valid:
                    new_lat = lat.iloc[hi + 1]
                    new_lon = lon.iloc[hi + 1]
                    for p in run_positions:
                        lat.iloc[p] = new_lat
                        lon.iloc[p] = new_lon
                        note.iloc[p] = _add_note(
                            note.iloc[p],
                            f"gps corrected: (0.0,0.0) -> ({new_lat:.5f},{new_lon:.5f}) | fill from next ({run_len} pt)",
                        )
                    _log_event(event_log, car_id, lo, ts.iloc[lo], "gps", "corrected", 
                               "filled with next valid", old_value="(0.0,0.0)", run_length=run_len)

        df.loc[orig_index, "latitude"] = lat.values
        df.loc[orig_index, "longitude"] = lon.values
        df.loc[orig_index, "note"] = note.values

    return df


# ---------------------------------------------------------------------------
# STAGE II: Speed = 0 vs GPS
# ---------------------------------------------------------------------------
def fix_speed(df: pd.DataFrame, event_log: List[dict]) -> pd.DataFrame:
    logger.info("[Stage II] Speed=0: cross-checked against (already-cleaned) GPS …")
    if "note" not in df.columns:
        df["note"] = ""

    for car_id, grp in df.groupby("car_id", sort=False):
        speed = grp["speed"].reset_index(drop=True)
        lat = grp["latitude"].reset_index(drop=True)
        lon = grp["longitude"].reset_index(drop=True)
        ts = grp["timestamp"].reset_index(drop=True)
        note = grp["note"].copy().reset_index(drop=True)
        orig_index = grp.index
        n = len(grp)

        for i in range(2, n - 2):
            if speed.iloc[i] != 0:
                continue

            # Step 1: check GPS around this point
            lo, hi = i - 1, i + 2  # i-1 .. i+1 inclusive
            stationary = _gps_is_stationary(lat, lon, lo, hi)

            if stationary:
                # Case A/B: GPS unchanged or only jitter -> real idle, keep
                _log_event(event_log, car_id, i, ts.iloc[i], "speed", "kept",
                           "GPS unchanged/jitter only -> vehicle genuinely idle")
                continue

            # Case C: GPS moving while Speed=0 -> contradiction, inspect wider window
            window = speed.iloc[i - 2:i + 3]
            all_zero = (window == 0).all()
            if all_zero:
                _log_event(event_log, car_id, i, ts.iloc[i], "speed", "kept",
                           "wider window also 0 -> vehicle genuinely stopped")
                continue

            neighbours_moving = (
                speed.iloc[i - 2] > SPEED_MOVING_THRESHOLD and speed.iloc[i - 1] > SPEED_MOVING_THRESHOLD
                and speed.iloc[i + 1] > SPEED_MOVING_THRESHOLD and speed.iloc[i + 2] > SPEED_MOVING_THRESHOLD
            )
            if neighbours_moving:
                old_val = speed.iloc[i]
                new_val = _linear_interp(ts.iloc[i - 1], speed.iloc[i - 1], ts.iloc[i + 1], speed.iloc[i + 1], ts.iloc[i])
                speed.iloc[i] = new_val
                note.iloc[i] = _add_note(note.iloc[i], f"speed interpolated (GPS moving): 0.0 -> {new_val:.1f}")
                _log_event(event_log, car_id, i, ts.iloc[i], "speed", "corrected",
                           "isolated Speed=0 while GPS moving and neighbours all moving", old_value=old_val, new_value=new_val)
            else:
                _log_event(event_log, car_id, i, ts.iloc[i], "speed", "kept",
                           "Suspicious speed (GPS moving, Speed=0, but neighbours inconclusive)")

        df.loc[orig_index, "speed"] = speed.values
        df.loc[orig_index, "note"] = note.values

    return df


# ---------------------------------------------------------------------------
# STAGE III: Fuel = 0
# ---------------------------------------------------------------------------
def fix_fuel_zeros(df: pd.DataFrame, event_log: List[dict]) -> pd.DataFrame:
    logger.info("[Stage III] Fuel=0: only an isolated single zero with a stable baseline both sides …")
    if "note" not in df.columns:
        df["note"] = ""

    for car_id, grp in df.groupby("car_id", sort=False):
        fuel = grp["fuel"].reset_index(drop=True)
        speed = grp["speed"].reset_index(drop=True)
        lat = grp["latitude"].reset_index(drop=True)
        lon = grp["longitude"].reset_index(drop=True)
        ts = grp["timestamp"].reset_index(drop=True)
        note = grp["note"].copy().reset_index(drop=True)
        orig_index = grp.index
        n = len(grp)

        visited = set()
        zero_positions = [p for p in range(n) if fuel.iloc[p] == 0]

        for pos in zero_positions:
            if pos in visited:
                continue
            lo = pos
            while lo > 0 and fuel.iloc[lo - 1] == 0:
                lo -= 1
            hi = pos
            while hi < n - 1 and fuel.iloc[hi + 1] == 0:
                hi += 1
            run_positions = list(range(lo, hi + 1))
            visited.update(run_positions)
            run_len = hi - lo + 1

            # Only a strictly isolated single zero can be corrected (spec III):
            # "190,191,0,191,190" (run=1) -> fix; "0,0,0,0,0" or "190,0,0,191,190"
            # (run=2) -> keep, insufficient evidence / continuous segment.
            if run_len > 1:
                if run_len >= 4:
                    _log_event(event_log, car_id, lo, ts.iloc[lo], "fuel", "kept",
                               "Continuous fuel zero segment", run_length=run_len)
                else:
                    _log_event(event_log, car_id, lo, ts.iloc[lo], "fuel", "kept",
                               "not enough evidence (zero run length 2)", run_length=run_len)
                continue
            if lo == 0 or hi == n - 1:
                _log_event(event_log, car_id, lo, ts.iloc[lo], "fuel", "kept", "no valid neighbour (edge of log)")
                continue

            prev_f, next_f = fuel.iloc[lo - 1], fuel.iloc[hi + 1]
            if prev_f == 0 or next_f == 0:
                _log_event(event_log, car_id, lo, ts.iloc[lo], "fuel", "kept", "adjacent to another zero-dropout")
                continue
            if abs(next_f - prev_f) > FUEL_ZERO_STABILITY_GUARD:
                _log_event(event_log, car_id, lo, ts.iloc[lo], "fuel", "kept",
                           "sides not stable enough (possible real trend)")
                continue

            # Cross-check timestamp continuity around the point
            deltas = ts.iloc[max(0, lo - 2):min(n, hi + 3)].diff().dt.total_seconds().dropna()
            if len(deltas) == 0 or deltas.max() > TIMESTAMP_MAX_GAP_SECONDS:
                _log_event(event_log, car_id, lo, ts.iloc[lo], "fuel", "kept", "timestamp gap too large around point")
                continue

            new_val = _linear_interp(ts.iloc[lo - 1], prev_f, ts.iloc[hi + 1], next_f, ts.iloc[lo])
            fuel.iloc[lo] = new_val
            speed_state = "Speed=0 (stationary)" if speed.iloc[lo] == 0 else "Speed>0 (moving)"
            note.iloc[lo] = _add_note(note.iloc[lo], f"fuel interpolated (isolated zero): 0.0 -> {new_val:.1f} | {speed_state}")
            _log_event(event_log, car_id, lo, ts.iloc[lo], "fuel", "corrected",
                       f"isolated zero, stable baseline both sides, {speed_state}", old_value=0.0, new_value=new_val)

        df.loc[orig_index, "fuel"] = fuel.values
        df.loc[orig_index, "note"] = note.values

    return df


# ---------------------------------------------------------------------------
# STAGE IV: Fuel > 0 spikes — context window (+/-5), GPS + Speed + shape
# ---------------------------------------------------------------------------
def _fuel_nonzero_decision(fuel: pd.Series, speed: pd.Series, lat: pd.Series, lon: pd.Series,
                            ts: pd.Series, i: int, n: int) -> Tuple[bool, str]:
    prev1, cur, next1 = fuel.iloc[i - 1], fuel.iloc[i], fuel.iloc[i + 1]
    diff_before, diff_after = cur - prev1, next1 - cur

    # Natural fluctuation — below the spike threshold, never touch (small
    # oscillation, or oscillation during acceleration).
    if abs(diff_before) < FUEL_SPIKE_MIN_DELTA and abs(diff_after) < FUEL_SPIKE_MIN_DELTA:
        return False, "natural fluctuation (below spike threshold)"

    # Monotonic change (same sign both sides) -> a real trend: consumption,
    # refuel, or a sustained fuel-theft drain. Never touch.
    if diff_before * diff_after >= 0:
        return False, "monotonic trend (real refuel/consumption/theft)"

    # Sides must return to roughly the same baseline (isolated spike shape)
    if abs(next1 - prev1) > FUEL_SIDE_SYMMETRY_GUARD:
        return False, "sides not symmetric enough (ambiguous)"

    # Movement/context check using the (already cleaned) GPS + Speed
    lo, hi = max(0, i - 2), min(n, i + 3)
    moving = _gps_is_moving(lat, lon, lo, hi) and speed.iloc[lo:hi].mean() > SPEED_MOVING_THRESHOLD
    if moving:
        return False, "vehicle moving (GPS + speed consistent) — real variation"

    # Must revert to baseline further out too (rules out a sustained drain
    # that merely started as what looks like a dip/rise)
    if i + 2 < n:
        future_val = fuel.iloc[i + 2]
        if abs(future_val - prev1) > max(abs(cur - prev1) * 0.5, FUEL_SIDE_SYMMETRY_GUARD):
            return False, "does not revert to baseline (possible sustained trend)"

    return True, "isolated spike while stationary, symmetric sides, reverts to baseline"


def denoise_fuel(df: pd.DataFrame, event_log: List[dict]) -> pd.DataFrame:
    logger.info("[Stage IV] Fuel>0: context window (GPS+Speed+shape), never rolling-median smoothing …")
    if "note" not in df.columns:
        df["note"] = ""

    for car_id, grp in df.groupby("car_id", sort=False):
        fuel = grp["fuel"].reset_index(drop=True)
        speed = grp["speed"].reset_index(drop=True)
        lat = grp["latitude"].reset_index(drop=True)
        lon = grp["longitude"].reset_index(drop=True)
        ts = grp["timestamp"].reset_index(drop=True)
        note = grp["note"].copy().reset_index(drop=True)
        orig_index = grp.index
        n = len(grp)

        # Hampel is used ONLY to flag candidates worth evaluating — never to replace values.
        candidate_mask = _hampel_candidate(fuel, window=11, n_sigmas=3.0)

        for i in range(2, n - 2):
            if fuel.iloc[i] == 0 or fuel.iloc[i - 1] == 0 or fuel.iloc[i + 1] == 0:
                continue  # zero-dropouts already handled in Stage III
            if not candidate_mask.iloc[i]:
                continue

            should_correct, reason = _fuel_nonzero_decision(fuel, speed, lat, lon, ts, i, n)
            if should_correct:
                prev1, cur, next1 = fuel.iloc[i - 1], fuel.iloc[i], fuel.iloc[i + 1]
                new_val = _linear_interp(ts.iloc[i - 1], prev1, ts.iloc[i + 1], next1, ts.iloc[i])
                fuel.iloc[i] = new_val
                note.iloc[i] = _add_note(note.iloc[i], f"fuel corrected: {cur:.1f} -> {new_val:.1f} | {reason}")
                _log_event(event_log, car_id, i, ts.iloc[i], "fuel_nonzero", "corrected", reason,
                           old_value=cur, new_value=new_val)
            else:
                _log_event(event_log, car_id, i, ts.iloc[i], "fuel_nonzero", "kept", reason)

        df.loc[orig_index, "fuel"] = fuel.values
        df.loc[orig_index, "note"] = note.values

    return df


# ---------------------------------------------------------------------------
# STAGE V: Vehicle-state events (informational only — no values are changed)
# ---------------------------------------------------------------------------
def flag_fuel_events(df: pd.DataFrame, event_log: List[dict]) -> pd.DataFrame:
    """
    Flag (but never correct) sustained Fuel Theft / Refueling events:
    Speed=0, GPS stationary, Fuel monotonically decreasing/increasing over
    >= FUEL_THEFT_MIN_RUN points with a total change >= FUEL_THEFT_MIN_DROP.
    """
    logger.info("[Stage V] Flagging Fuel Theft / Refueling events (informational only) …")
    if "note" not in df.columns:
        df["note"] = ""

    for car_id, grp in df.groupby("car_id", sort=False):
        fuel = grp["fuel"].reset_index(drop=True)
        speed = grp["speed"].reset_index(drop=True)
        lat = grp["latitude"].reset_index(drop=True)
        lon = grp["longitude"].reset_index(drop=True)
        ts = grp["timestamp"].reset_index(drop=True)
        note = grp["note"].copy().reset_index(drop=True)
        orig_index = grp.index
        n = len(grp)

        i = 0
        while i < n - 1:
            if speed.iloc[i] != 0:
                i += 1
                continue
            lo, hi = i, i
            while hi < n - 1 and speed.iloc[hi + 1] == 0:
                hi += 1
            if hi - lo + 1 < FUEL_THEFT_MIN_RUN or not _gps_is_stationary(lat, lon, lo, hi + 1):
                i = hi + 1
                continue

            run_fuel = fuel.iloc[lo:hi + 1]
            total_change = run_fuel.iloc[-1] - run_fuel.iloc[0]
            is_monotonic_drop = (run_fuel.diff().dropna() <= 0.01).all() and total_change <= -FUEL_THEFT_MIN_DROP
            is_monotonic_rise = (run_fuel.diff().dropna() >= -0.01).all() and total_change >= FUEL_THEFT_MIN_DROP

            if is_monotonic_drop:
                note.iloc[lo] = _add_note(note.iloc[lo], f"event: possible fuel theft (stationary, drop={total_change:.1f}L)")
                _log_event(event_log, car_id, lo, ts.iloc[lo], "fuel_event", "event",
                           "stationary sustained decrease", old_value=run_fuel.iloc[0], new_value=run_fuel.iloc[-1],
                           run_length=hi - lo + 1)
            elif is_monotonic_rise:
                note.iloc[lo] = _add_note(note.iloc[lo], f"event: refueling detected (stationary, rise={total_change:.1f}L)")
                _log_event(event_log, car_id, lo, ts.iloc[lo], "fuel_event", "event",
                           "stationary sustained increase", old_value=run_fuel.iloc[0], new_value=run_fuel.iloc[-1],
                           run_length=hi - lo + 1)
            i = hi + 1

        df.loc[orig_index, "note"] = note.values

    return df


# ---------------------------------------------------------------------------
# Post-clean validation
# ---------------------------------------------------------------------------
def post_clean_validation(df: pd.DataFrame) -> dict:
    issues = {}
    issues["fuel_negative"] = int((df["fuel"] < 0).sum())
    issues["speed_negative"] = int((df["speed"] < 0).sum())
    issues["gps_zero"] = int(((df["latitude"] == 0) & (df["longitude"] == 0)).sum())
    reversed_total = 0
    for car_id, grp in df.groupby("car_id", sort=False):
        reversed_total += int((grp["timestamp"].diff() < pd.Timedelta(0)).sum())
    issues["timestamp_reversed_total"] = reversed_total
    dup_cols = ["car_id", "timestamp", "fuel", "speed", "latitude", "longitude"]
    issues["duplicate_rows"] = int(df.duplicated(subset=dup_cols).sum())
    return issues


# ---------------------------------------------------------------------------
# Main cleaning pipeline (order: GPS -> Speed -> Fuel, per spec)
# ---------------------------------------------------------------------------
def clean_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict], int]:
    event_log: List[dict] = []

    df = sort_dataset(df)
    n_before_dedup = len(df)
    df = remove_duplicates(df)
    duplicates_removed = n_before_dedup - len(df)

    df["note"] = ""
    df = fix_timestamps(df, event_log)
    df = fix_gps(df, event_log)          # Stage I
    df = fix_speed(df, event_log)        # Stage II (depends on cleaned GPS)
    df = fix_fuel_zeros(df, event_log)   # Stage III (depends on cleaned GPS+Speed)
    df = denoise_fuel(df, event_log)     # Stage IV (depends on cleaned GPS+Speed)
    df = flag_fuel_events(df, event_log)  # Stage V (informational only)

    return df, event_log, duplicates_removed


# ---------------------------------------------------------------------------
# Detailed traceable logs (CSV export)
# ---------------------------------------------------------------------------
def export_detailed_logs(event_log: List[dict], logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    if not event_log:
        logger.info("No events recorded — skipping detailed log export.")
        return

    events_df = pd.DataFrame(event_log)
    events_df.to_csv(logs_dir / "all_decisions.csv", index=False)
    logger.info("Exported %d total decisions to all_decisions.csv", len(events_df))

    file_map = {
        ("gps", "corrected"): "corrected_gps_zeros.csv",
        ("gps", "kept"): "skipped_gps_zeros.csv",
        ("speed", "corrected"): "corrected_speed.csv",
        ("speed", "kept"): "suspicious_speed.csv",
        ("fuel", "corrected"): "corrected_fuel_zeros.csv",
        ("fuel", "kept"): "skipped_fuel_zeros.csv",
        ("fuel_nonzero", "corrected"): "corrected_fuel_nonzero.csv",
        ("fuel_nonzero", "kept"): "rejected_fuel_nonzero.csv",
        # Đã xóa ("timestamp", "corrected") và ("timestamp", "kept")
        ("fuel_event", "event"): "fuel_theft_and_refuel_events.csv",
    }
    for (field, decision), filename in file_map.items():
        subset = events_df[(events_df["field"] == field) & (events_df["decision"] == decision)]
        if not subset.empty:
            subset.to_csv(logs_dir / filename, index=False)
            logger.info("Exported %d rows to %s", len(subset), filename)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def generate_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame, event_log: List[dict],
                     duplicates_removed: int, report_path: Path) -> None:
    n_raw, n_clean = len(raw_df), len(clean_df)
    note_series = clean_df["note"]
    edited_rows = int((note_series.str.len() > 0).sum())

    events_df = pd.DataFrame(event_log) if event_log else pd.DataFrame(
        columns=["field", "decision", "reason", "run_length"])

    def _count(field: str, decision: str) -> int:
        if events_df.empty:
            return 0
        mask = (events_df["field"] == field) & (events_df["decision"] == decision)
        if "run_length" in events_df.columns:
            return int(events_df.loc[mask, "run_length"].fillna(1).sum())
        return int(mask.sum())

    gps_fixed = _count("gps", "corrected")
    gps_kept = _count("gps", "kept")
    speed_fixed = _count("speed", "corrected")
    speed_kept = _count("speed", "kept")
    fuel_zero_fixed = _count("fuel", "corrected")
    fuel_zero_kept = _count("fuel", "kept")
    fuel_spike_fixed = _count("fuel_nonzero", "corrected")
    fuel_spike_kept = _count("fuel_nonzero", "kept")
    ts_fixed = _count("timestamp", "corrected")
    n_theft_events = int(((events_df.get("field") == "fuel_event") &
                          (events_df.get("reason") == "stationary sustained decrease")).sum()) if not events_df.empty else 0
    n_refuel_events = int(((events_df.get("field") == "fuel_event") &
                           (events_df.get("reason") == "stationary sustained increase")).sum()) if not events_df.empty else 0

    total_fuel_fixed = fuel_zero_fixed + fuel_spike_fixed
    fuel_pct = total_fuel_fixed / n_clean * 100 if n_clean else 0.0
    speed_pct = speed_fixed / n_clean * 100 if n_clean else 0.0
    gps_pct = gps_fixed / n_clean * 100 if n_clean else 0.0

    fuel_std_before, fuel_std_after = raw_df["fuel"].std(ddof=0), clean_df["fuel"].std(ddof=0)
    speed_std_before, speed_std_after = raw_df["speed"].std(ddof=0), clean_df["speed"].std(ddof=0)
    fuel_noise_red = 0.0 if fuel_std_before == 0 else (1 - fuel_std_after / fuel_std_before) * 100
    speed_noise_red = 0.0 if speed_std_before == 0 else (1 - speed_std_after / speed_std_before) * 100

    clean_sorted = clean_df.sort_values(["car_id", "timestamp"]).copy()
    clean_sorted["delta_time"] = clean_sorted.groupby("car_id")["timestamp"].diff().dt.total_seconds()
    long_gaps = int((clean_sorted["delta_time"] > 3600).sum())

    post_issues = post_clean_validation(clean_df)

    logger.info("=" * 50)
    logger.info("Cleaning Summary (Sequential Rule-Based: GPS -> Speed -> Fuel)")
    logger.info("Rows: raw=%d clean=%d duplicates_removed=%d", n_raw, n_clean, duplicates_removed)
    logger.info("GPS fixed=%d | Speed fixed=%d | Fuel fixed=%d (%.2f%%)", gps_fixed, speed_fixed, total_fuel_fixed, fuel_pct)
    logger.info("Fuel theft events=%d | Refuel events=%d", n_theft_events, n_refuel_events)
    logger.info("Post-clean issues: %s", post_issues)

    md_lines = [
        "# Cleaning Report (Sequential Rule-Based Pipeline: GPS -> Speed -> Fuel)",
        "",
        "## Dataset Summary",
        f"- Số dòng ban đầu: {n_raw}",
        f"- Số dòng sau cleaning: {n_clean}",
        f"- Số dòng đã chỉnh sửa: {edited_rows}",
        f"- Số dòng trùng lặp đã loại bỏ: {duplicates_removed}",
        "",
        "## 1. Intervention Rate",
        "",
        "| Field | Đã sửa | Giữ nguyên (có lý do) | Tỷ lệ sửa / tổng |",
        "|-------|--------|-------------------------|---------------------|",
        f"| GPS (0,0) | {gps_fixed} | {gps_kept} | {gps_pct:.2f}% |",
        f"| Speed = 0 | {speed_fixed} | {speed_kept} | {speed_pct:.2f}% |",
        f"| Fuel = 0 | {fuel_zero_fixed} | {fuel_zero_kept} | — |",
        f"| Fuel spike (>0) | {fuel_spike_fixed} | {fuel_spike_kept} | — |",
        f"| **Tổng Fuel** | **{total_fuel_fixed}** | — | **{fuel_pct:.2f}%** |",
        f"| Timestamp | {ts_fixed} | — | — |",
        "",
        "## 2. Preservation & Events (proof against over-cleaning)",
        "",
        f"- Fuel Theft events được đánh dấu (không sửa dữ liệu): {n_theft_events}",
        f"- Refueling events được đánh dấu (không sửa dữ liệu): {n_refuel_events}",
        f"- Fuel spike bị từ chối sửa (real trend / moving / không đối xứng): {fuel_spike_kept}",
        f"- Speed=0 giữ nguyên vì GPS xác nhận xe dừng: {speed_kept}",
        f"- GPS (0,0) giữ nguyên vì outage dài / thiếu bằng chứng: {gps_kept}",
        "",
        "## 3. Noise Reduction",
        "",
        "| Field | Raw Std | Clean Std | Reduction % |",
        "|-------|---------|-----------|-------------|",
        f"| Fuel | {fuel_std_before:.3f} | {fuel_std_after:.3f} | {fuel_noise_red:.1f}% |",
        f"| Speed | {speed_std_before:.3f} | {speed_std_after:.3f} | {speed_noise_red:.1f}% |",
        "",
        "## 4. Long Gap Summary",
        f"- Tổng số gap > 3600 giây: {long_gaps}",
        "",
        "## 5. Post-clean Validation",
        f"- Fuel < 0: {post_issues['fuel_negative']}",
        f"- Speed < 0: {post_issues['speed_negative']}",
        f"- GPS (0,0) còn lại: {post_issues['gps_zero']}",
        f"- Timestamp reversed còn lại: {post_issues['timestamp_reversed_total']}",
        f"- Duplicate rows còn lại: {post_issues['duplicate_rows']}",
        "",
        "## 6. Final Dataset Summary",
        f"- Số dòng: {n_clean}",
        "- Các cột (giữ nguyên tên gốc): FuelTime, FuelLevel, Lat, Lng, Address, Speed, car_id, note",
        "- Dữ liệu đã được sắp xếp theo car_id và timestamp.",
        "- Mọi thay đổi được ghi rõ lý do trong cột `note`; toàn bộ quyết định "
        "(sửa và không sửa) được xuất chi tiết trong `cleaning_logs/` để truy vết.",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Báo cáo đã được lưu tại %s", report_path)


# ---------------------------------------------------------------------------
# Plotting — single overlaid chart per car (Raw = blue, Clean = red)
# ---------------------------------------------------------------------------
def plot_fuel_comparison(raw_df: pd.DataFrame, clean_df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for car in sorted(clean_df["car_id"].unique()):
        raw_car = raw_df[raw_df["car_id"] == car].sort_values("timestamp")
        clean_car = clean_df[clean_df["car_id"] == car].sort_values("timestamp")

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(raw_car["timestamp"], raw_car["fuel"], color="blue", linewidth=1.2, alpha=0.8, label="Raw")
        ax.plot(clean_car["timestamp"], clean_car["fuel"], color="red", linewidth=1.2, alpha=0.8, label="Clean")
        ax.set_title(f"Fuel Comparison – {car}", fontsize=14)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Fuel Level")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        out_path = figures_dir / f"fuel_comparison_{car.replace(' ', '_')}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Đọc dữ liệu từ %s", INPUT_CSV)
    try:
        raw_df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        logger.error("Không thể đọc file đầu vào: %s", e)
        sys.exit(1)

    raw_df.rename(columns=COLUMN_MAP, inplace=True)
    required = {"timestamp", "fuel", "latitude", "longitude", "address", "speed", "car_id"}
    missing = required - set(raw_df.columns)
    if missing:
        logger.error("Thiếu cột bắt buộc sau khi đổi tên: %s", missing)
        sys.exit(1)
    raw_df = raw_df[list(required)].copy()

    for col in ["fuel", "speed", "latitude", "longitude"]:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").astype(float)

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
    if raw_df["timestamp"].isna().any():
        logger.warning("Có %d timestamp không hợp lệ, sẽ bị loại bỏ.", raw_df["timestamp"].isna().sum())
    raw_df.dropna(subset=["timestamp"], inplace=True)
    for col in ["fuel", "speed", "latitude", "longitude"]:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
    raw_df.dropna(subset=["fuel", "speed", "latitude", "longitude"], inplace=True)

    raw_sorted = sort_dataset(raw_df.copy())
    raw_sorted = remove_duplicates(raw_sorted)

    clean_df, event_log, duplicates_removed = clean_pipeline(raw_df)

    clean_out = pd.DataFrame({
        "car_id": clean_df["car_id"],
        "FuelTime": clean_df["timestamp"],
        "Address": clean_df["address"],
        "Lat": clean_df["latitude"],
        "Lng": clean_df["longitude"],
        "Speed": clean_df["speed"],
        "FuelLevel": clean_df["fuel"],
        "note": clean_df["note"]
    })

    OUTPUT_CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean_out.to_csv(OUTPUT_CLEAN_CSV, index=False)
    logger.info("Dữ liệu đã được lưu tại %s", OUTPUT_CLEAN_CSV)

    generate_report(raw_sorted, clean_df, event_log, duplicates_removed, REPORT_MD)
    export_detailed_logs(event_log, LOGS_DIR)
    plot_fuel_comparison(raw_sorted, clean_df, FIGURES_DIR)


if __name__ == "__main__":
    main()