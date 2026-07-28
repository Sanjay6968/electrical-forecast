import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("GridCastAI")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

def load_data():
    path = DATA_DIR / "household_power_consumption.txt"
    if not path.exists():
        logger.error(f"Dataset not found at {path}")
        sys.exit(1)
    df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)
    if df.empty:
        sys.exit(1)
    return df

def clean_data(df):
    df["measurement_timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df.set_index("measurement_timestamp", inplace=True)
    df.drop(columns=["Date", "Time"], inplace=True)
    df = df.interpolate(method="time", limit_direction="both")
    df = df.ffill().bfill()
    num_cols = df.select_dtypes(include=["float64"]).columns
    df[num_cols] = df[num_cols].astype("float32")
    hourly = df[["Global_active_power"]].resample("1h").mean()
    hourly.rename(columns={"Global_active_power": "hourly_energy_demand"}, inplace=True)
    return hourly

def feature_engineering(hourly):
    engineered = hourly.copy()
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        engineered[f"lag_{lag}"] = engineered["hourly_energy_demand"].shift(lag)
    for w in [6, 12, 24, 168]:
        engineered[f"rolling_mean_{w}"] = engineered["hourly_energy_demand"].rolling(w).mean()
    engineered["rolling_std_24"] = engineered["hourly_energy_demand"].rolling(24).std()
    engineered["rolling_max_24"] = engineered["hourly_energy_demand"].rolling(24).max()
    engineered["rolling_min_24"] = engineered["hourly_energy_demand"].rolling(24).min()
    engineered["rolling_median_24"] = engineered["hourly_energy_demand"].rolling(24).median()
    engineered["ema_24"] = engineered["hourly_energy_demand"].ewm(span=24).mean()
    engineered["hour"] = engineered.index.hour
    engineered["weekday"] = engineered.index.dayofweek
    engineered["month"] = engineered.index.month
    engineered["quarter"] = engineered.index.quarter
    engineered["is_weekend"] = (engineered["weekday"] >= 5).astype(int)
    engineered["hour_sin"] = np.sin(2 * np.pi * engineered["hour"] / 24)
    engineered["hour_cos"] = np.cos(2 * np.pi * engineered["hour"] / 24)
    engineered["weekday_sin"] = np.sin(2 * np.pi * engineered["weekday"] / 7)
    engineered["weekday_cos"] = np.cos(2 * np.pi * engineered["weekday"] / 7)
    engineered["hourly_change"] = engineered["hourly_energy_demand"].diff()
    engineered["daily_change"] = engineered["hourly_energy_demand"] - engineered["lag_24"]
    engineered["rolling_range_24"] = engineered["rolling_max_24"] - engineered["rolling_min_24"]
    engineered["ema_168"] = engineered["hourly_energy_demand"].ewm(span=168).mean()
    engineered["rolling_skew_24"] = engineered["hourly_energy_demand"].rolling(24).skew()
    engineered["rolling_kurtosis_24"] = engineered["hourly_energy_demand"].rolling(24).kurt()
    engineered["is_month_start"] = engineered.index.is_month_start.astype(int)
    engineered["is_month_end"] = engineered.index.is_month_end.astype(int)
    engineered.dropna(inplace=True)
    for h in range(1, 25):
        engineered[f"target_h{h}"] = engineered["hourly_energy_demand"].shift(-h)
    engineered.dropna(inplace=True)
    return engineered

def train_and_evaluate(forecast_feature_bank):
    predictor_columns = forecast_feature_bank.drop(columns=["hourly_energy_demand"] + [f"target_h{h}" for h in range(1, 25)])
    prediction_targets = forecast_feature_bank[[f"target_h{h}" for h in range(1, 25)]]
    split_index = int(len(forecast_feature_bank) * 0.80)
    training_features = predictor_columns.iloc[:split_index]
    training_targets = prediction_targets.iloc[:split_index]
    testing_features = predictor_columns.iloc[split_index:]
    testing_targets = prediction_targets.iloc[split_index:]
    
    multi_horizon_models = {}
    evaluation_results = []
    
    lgb_params = {
        "objective": "regression",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1
    }
    
    for h in range(1, 25):
        logger.info(f"Training Model for Horizon {h}...")
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(training_features, training_targets[f"target_h{h}"])
        multi_horizon_models[f"model_h{h}"] = model
        
        preds = model.predict(testing_features)
        mae = mean_absolute_error(testing_targets[f"target_h{h}"], preds)
        rmse = np.sqrt(mean_squared_error(testing_targets[f"target_h{h}"], preds))
        
        evaluation_results.append({'Horizon': h, 'Model_MAE': mae, 'Model_RMSE': rmse})
        joblib.dump(model, MODEL_DIR / f"model_h{h}.joblib")
        
    eval_df = pd.DataFrame(evaluation_results)
    return multi_horizon_models, eval_df, predictor_columns

def forecast_next_24_hours(multi_horizon_models, predictor_columns):
    last_row = predictor_columns.iloc[[-1]]
    last_timestamp = last_row.index[0]
    forecast_results = []
    
    for h in range(1, 25):
        prediction = multi_horizon_models[f"model_h{h}"].predict(last_row)[0]
        forecast_timestamp = last_timestamp + pd.Timedelta(hours=h)
        forecast_results.append({
            'Forecast_Timestamp': forecast_timestamp,
            'Horizon': h,
            'Predicted_Demand': prediction
        })
        
    return pd.DataFrame(forecast_results)

if __name__ == "__main__":
    logger.info("Starting GridCastAI Pipeline...")
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    features_df = feature_engineering(clean_df)
    models, eval_df, predictor_cols = train_and_evaluate(features_df)
    forecast_df = forecast_next_24_hours(models, predictor_cols)
    
    forecast_df["Predicted_Demand"] = forecast_df["Predicted_Demand"].round(2)
    forecast_df["Forecast_Timestamp"] = forecast_df["Forecast_Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        forecast_df.to_csv(OUTPUT_DIR / "forecast_results.csv", index=False)
        eval_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    except Exception as e:
        logger.error(f"Failed to save outputs: {e}")
        
    logger.info("Pipeline Completed Successfully.")