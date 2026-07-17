import pandas as pd
import numpy as np

def rule_based_classify(row):
    speed = row['Speed']
    fuel_rate = row['FuelRate']
    slope = row['RegressionSlope']
    window_diff = row['WindowFuelDiff']
    stop_duration = row['StopDuration']
    stopped = row['VehicleStopped']

    # Rule 1: Driving (ưu tiên cao nhất nếu có tốc độ)
    if speed > 5:
        return 'Driving', 0.9

    # Các luật khi xe dừng
    if stopped:
        # Rule 2: Refuel (Tăng nhanh)
        # SỬA: stop_duration > 5 (thay vì > 20) để nhận diện sớm hơn
        if window_diff > 2 and slope > 0.1 and stop_duration > 5:
            conf = 0.6 + (min(window_diff, 10) - 2) * (0.35 / 8)
            return 'Refuel', min(conf, 0.95)

        # Rule 3: Fuel Theft (Giảm nhanh)
        # SỬA: stop_duration > 5 (thay vì > 20) để nhận diện sớm hơn
        if window_diff < -2 and slope < -0.1 and stop_duration > 5:
            abs_diff = abs(window_diff)
            conf = 0.6 + (min(abs_diff, 10) - 2) * (0.35 / 8)
            return 'Fuel Theft', min(conf, 0.95)

        # Rule 4: Idle (Mặc định cho các trường hợp đứng yên còn lại)
        return 'Idle', 0.7

    # Rule 5: Unknown (chỉ dành cho các trường hợp hiếm hoi chưa thể phân loại)
    return 'Unknown', 0.0

def batch_predict(df):
    df = df.copy()
    predictions = []
    confidences = []
    for _, row in df.iterrows():
        label, conf = rule_based_classify(row)
        predictions.append(label)
        confidences.append(conf)
    df['Prediction'] = predictions
    df['Confidence'] = confidences
    return df