import pandas as pd
import numpy as np

def rule_based_classify(row):
    speed = row['Speed']
    fuel_rate = row['FuelRate']
    slope = row['RegressionSlope']
    window_diff = row['WindowFuelDiff']
    stop_duration = row['StopDuration']
    stopped = row['VehicleStopped']

    # Rule 1: If vehicle is stopped, prioritize stop-state rules first.
    if stopped:
        # Rule 2: Refuel (fast increase while stopped)
        if window_diff > 2 and slope > 0.1 and stop_duration > 5:
            conf = 0.6 + (min(window_diff, 10) - 2) * (0.35 / 8)
            return 'Refuel', min(conf, 0.95)

        # Rule 3: Fuel Theft (fast decrease while stopped)
        if window_diff < -2 and slope < -0.1 and stop_duration > 5:
            abs_diff = abs(window_diff)
            conf = 0.6 + (min(abs_diff, 10) - 2) * (0.35 / 8)
            return 'Fuel Theft', min(conf, 0.95)

        # Rule 4: Idle (default for remaining stopped states)
        return 'Idle', 0.7

    # Rule 5: Driving when movement is clearly present.
    if speed > 5:
        return 'Driving', 0.9

    # Rule 6: Fallback to Idle if not enough evidence for another class.
    return 'Idle', 0.6

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