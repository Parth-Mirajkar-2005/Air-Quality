import pandas as pd
import numpy as np

def create_time_features(df):
    """Creates time-based and cyclical features."""
    df['datetime'] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day, hour=df.hour))
    df = df.sort_values('datetime').set_index('datetime')
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
    return df

def create_satellite_features(df):
    """Creates missingness flags for satellite columns and then fills them."""
    sat_cols = [c for c in df.columns if 'satellite' in c]
    for c in sat_cols:
        df[f'{c}_missing'] = df[c].isna().astype(int)
    if sat_cols:
        df[sat_cols] = df[sat_cols].ffill().bfill()
    return df

def create_wind_features(df):
    """Creates wind speed and direction features from u/v components."""
    if {'u_forecast', 'v_forecast'}.issubset(df.columns):
        df['wind_speed'] = np.sqrt(df['u_forecast']**2 + df['v_forecast']**2)
        df['wind_dir'] = np.arctan2(df['v_forecast'], df['u_forecast'])
    return df

def create_proxy_features(df):
    """Creates physics-informed pollutant ratios and interactions."""
    if 'HCHO_satellite' in df.columns and 'NO2_satellite' in df.columns:
        df['HCHO_NO2_sat_ratio'] = df['HCHO_satellite'] / (df['NO2_satellite'].replace(0, np.nan))
    if 'NO2_forecast' in df.columns and 'O3_forecast' in df.columns:
        df['NO2_O3_forecast_ratio'] = df['NO2_forecast'] / (df['O3_forecast'].replace(0, np.nan))
    if 'T_forecast' in df.columns and 'hour_sin' in df.columns:
        df['T_x_hour'] = df['T_forecast'] * df['hour_sin']
    if 'NO2_forecast' in df.columns and 'wind_speed' in df.columns:
        df['NO2_x_wind'] = df['NO2_forecast'] * df['wind_speed']
    return df

def create_timeseries_features(df, is_train=True):
    """Creates time-lagged and rolling window features."""
    # During training, we can create lags from target variables.
    # During inference (prediction), we can only use forecast variables.
    lag_cols = ['O3_forecast', 'NO2_forecast', 'T_forecast']
    if is_train:
        lag_cols.extend(['O3_target', 'NO2_target'])

    for base in lag_cols:
        if base in df.columns:
            for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
                df[f'{base}_lag_{lag}h'] = df[base].shift(lag)
            for w in [3, 6, 12, 24]:
                shifted = df[base].shift(1)
                df[f'{base}_rollmean_{w}h'] = shifted.rolling(w, min_periods=1).mean()
                df[f'{base}_rollstd_{w}h'] = shifted.rolling(w, min_periods=1).std().fillna(0)
    return df

def run_feature_engineering(df, is_train=True):
    """Runs all feature engineering steps in the correct sequence."""
    df_with_time = create_time_features(df)
    df_with_sat = create_satellite_features(df_with_time)
    df_with_wind = create_wind_features(df_with_sat)
    df_with_proxy = create_proxy_features(df_with_wind)
    df_with_ts = create_timeseries_features(df_with_proxy, is_train=is_train)

    numeric_cols = df_with_ts.select_dtypes(include=np.number).columns
    df_with_ts[numeric_cols] = df_with_ts[numeric_cols].interpolate(method='time', limit=6).ffill().bfill()
    
    return df_with_ts.reset_index()

def ria(y_true, y_pred):
    """Calculates the Refined Index of Agreement (RIA)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    num = np.sum((y_pred - y_true)**2)
    denom = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true)))**2)
    if denom == 0:
        return 1.0 # Perfect agreement
    return 1 - (num / denom)