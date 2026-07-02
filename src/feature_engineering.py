import sys
import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

def feature_engineering(hourly_energy_profile: pd.DataFrame) -> pd.DataFrame:
    """
    Generates temporal, statistical, and cyclical features for forecasting.
    """
    logger.info("Starting Feature Engineering...")
    engineered = hourly_energy_profile.copy()

    # Lags
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for lag in lag_hours:
        engineered[f"lag_{lag}"] = engineered["hourly_energy_demand"].shift(lag)

    # Rolling windows
    rolling_windows = [6, 12, 24, 168]
    for w in rolling_windows:
        engineered[f"rolling_mean_{w}"] = engineered["hourly_energy_demand"].rolling(w).mean()

    engineered["rolling_std_24"] = engineered["hourly_energy_demand"].rolling(24).std()
    engineered["rolling_max_24"] = engineered["hourly_energy_demand"].rolling(24).max()
    engineered["rolling_min_24"] = engineered["hourly_energy_demand"].rolling(24).min()
    engineered["rolling_median_24"] = engineered["hourly_energy_demand"].rolling(24).median()

    engineered["ema_24"] = engineered["hourly_energy_demand"].ewm(span=24).mean()

    # Temporal features
    engineered["hour"] = engineered.index.hour
    engineered["weekday"] = engineered.index.dayofweek
    engineered["month"] = engineered.index.month
    engineered["quarter"] = engineered.index.quarter
    engineered["is_weekend"] = (engineered["weekday"] >= 5).astype(int)

    # Cyclical features
    engineered["hour_sin"] = np.sin(2 * np.pi * engineered["hour"] / 24)
    engineered["hour_cos"] = np.cos(2 * np.pi * engineered["hour"] / 24)
    engineered["weekday_sin"] = np.sin(2 * np.pi * engineered["weekday"] / 7)
    engineered["weekday_cos"] = np.cos(2 * np.pi * engineered["weekday"] / 7)

    # Differences
    engineered["hourly_change"] = engineered["hourly_energy_demand"].diff()
    engineered["daily_change"] = engineered["hourly_energy_demand"] - engineered["lag_24"]
    engineered["rolling_range_24"] = engineered["rolling_max_24"] - engineered["rolling_min_24"]
    engineered["ema_168"] = engineered["hourly_energy_demand"].ewm(span=168).mean()
    engineered["rolling_skew_24"] = engineered["hourly_energy_demand"].rolling(24).skew()
    engineered["rolling_kurtosis_24"] = engineered["hourly_energy_demand"].rolling(24).kurt()
    engineered["is_month_start"] = engineered.index.is_month_start.astype(int)
    engineered["is_month_end"] = engineered.index.is_month_end.astype(int)

    engineered.dropna(inplace=True)

    forecast_horizon_hours = 24
    for h in range(1, forecast_horizon_hours + 1):
        engineered[f"target_h{h}"] = engineered["hourly_energy_demand"].shift(-h)

    engineered.dropna(inplace=True)
    
    if engineered.empty:
        logger.error("Feature engineering resulted in an empty dataframe.")
        sys.exit(1)
        
    return engineered
