import pandas as pd
import numpy as np
import traceback

def calculate_features(df):
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 1. Delta Time (giây)
    df['DeltaTime'] = df['Timestamp'].diff().dt.total_seconds().fillna(0)
    
    # 2. InstantFuelDiff (chênh lệch tức thời)
    df['InstantFuelDiff'] = df['Fuel'].diff().fillna(0)
    
    # 3. FuelRate (L/s) – an toàn tuyệt đối: thay 0 bằng NaN trước khi chia
    df['FuelRate'] = (df['InstantFuelDiff'] / df['DeltaTime'].replace(0, np.nan)).fillna(0)
    
    # 4. Moving Average (cửa sổ 5)
    df['MovingAvg'] = df['Fuel'].rolling(window=5, min_periods=1).mean()
    
    # 5. Rolling Std (cửa sổ 5)
    df['RollingStd'] = df['Fuel'].rolling(window=5, min_periods=1).std().fillna(0)
    
    # 6. WindowFuelDiff (10 mẫu)
    df['WindowFuelDiff'] = (df['Fuel'] - df['Fuel'].shift(10)).fillna(0)
    
    # 7. MaxJump (thay đổi lớn nhất trong 10 mẫu)
    abs_diff = df['InstantFuelDiff'].abs()
    df['MaxJump'] = abs_diff.rolling(window=10, min_periods=1).max().fillna(0)
    
    # 8. VehicleStopped (Speed < 2)
    df['VehicleStopped'] = (df['Speed'] < 2).astype(int)
    
    # 9. StopDuration (giây, reset khi chạy)
    run_id = (df['VehicleStopped'] != df['VehicleStopped'].shift()).cumsum()
    df['StopDuration'] = df.groupby(run_id)['DeltaTime'].cumsum() * df['VehicleStopped']
    
    # 10. AvgSpeed (10 mẫu)
    df['AvgSpeed'] = df['Speed'].rolling(window=10, min_periods=1).mean()
    
    # 11. MaxFuel, MinFuel, FuelRange (10 mẫu)
    df['MaxFuel'] = df['Fuel'].rolling(window=10, min_periods=1).max()
    df['MinFuel'] = df['Fuel'].rolling(window=10, min_periods=1).min()
    df['FuelRange'] = df['MaxFuel'] - df['MinFuel']
    
    # 12. PositiveRate / NegativeRate (10 mẫu)
    pos = (df['InstantFuelDiff'] > 0).astype(int)
    neg = (df['InstantFuelDiff'] < 0).astype(int)
    df['PositiveRate'] = pos.rolling(window=10, min_periods=1).sum()
    df['NegativeRate'] = neg.rolling(window=10, min_periods=1).sum()
    
    # 13. StableCounter (10 mẫu, |diff| < 0.1)
    stable = (df['InstantFuelDiff'].abs() < 0.1).astype(int)
    df['StableCounter'] = stable.rolling(window=10, min_periods=1).sum()
    
    # 14. Regression Slope & R² (cửa sổ 10, dùng timestamp thật)
    t0 = df['Timestamp'].iloc[0]
    df['TimeSec'] = (df['Timestamp'] - t0).dt.total_seconds()
    
    slopes = []
    r2s = []
    for i in range(len(df)):
        if i < 9:
            slopes.append(0.0)
            r2s.append(0.0)
            continue
        
        try:
            window = df.iloc[i-9:i+1]
            t_start = window['TimeSec'].iloc[0]
            X = (window['TimeSec'] - t_start).values
            y = window['Fuel'].values
            
            x_mean = X.mean()
            y_mean = y.mean()
            cov_xy = np.sum((X - x_mean) * (y - y_mean))
            var_x = np.sum((X - x_mean) ** 2)
            
            if var_x < 1e-12:
                a = 0.0
            else:
                a = cov_xy / var_x
            
            ss_tot = np.sum((y - y_mean) ** 2)
            if ss_tot < 1e-12:
                r2 = 1.0
            else:
                y_pred = a * X + (y_mean - a * x_mean)
                ss_res = np.sum((y - y_pred) ** 2)
                r2 = 1.0 - ss_res / ss_tot
            
            slopes.append(a)
            r2s.append(r2)
        except Exception as e:
            print(f"❌ Lỗi tại dòng {i} trong calculate_features: {e}")
            traceback.print_exc()
            slopes.append(0.0)
            r2s.append(0.0)
    
    df['RegressionSlope'] = slopes
    df['RegressionR2'] = r2s
    
    return df