import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError:
    print("Error: scikit-learn is not installed. Please install it using 'pip install scikit-learn'.")
    sys.exit(1)

try:
    import lightgbm as lgb
except ImportError:
    print("Error: LightGBM is not installed. Please install it using 'pip install lightgbm'.")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("Error: joblib is not installed. Please install it using 'pip install joblib'.")
    sys.exit(1)
    
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed. Please install it using 'pip install matplotlib'.")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "household_power_consumption.txt"
MODEL_DIRECTORY = PROJECT_ROOT / "models"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"


def load_data() -> pd.DataFrame:
    """
    Loads the household power consumption dataset.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    print("====================================================")
    print("Loading Dataset...")
    
    if not DATA_PATH.exists():
        print(f"Error: Dataset not found at {DATA_PATH}.")
        print("Please ensure the data file exists.")
        sys.exit(1)
        
    try:
        household_energy_archive = pd.read_csv(
            DATA_PATH,
            sep=";",
            na_values="?",
            low_memory=False
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
        
    if household_energy_archive.empty:
        print("Error: Loaded dataset is empty.")
        sys.exit(1)
        
    expected_columns = ["Date", "Time", "Global_active_power"]
    missing_columns = [col for col in expected_columns if col not in household_energy_archive.columns]
    if missing_columns:
        print(f"Error: Missing expected columns in dataset: {missing_columns}")
        sys.exit(1)
        
    return household_energy_archive


def clean_data(household_energy_archive: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataset, handles missing values, and resamples to hourly frequency.
    
    Args:
        household_energy_archive (pd.DataFrame): Raw dataset.
        
    Returns:
        pd.DataFrame: Cleaned and hourly resampled dataset.
    """
    print("Cleaning Data...")
    
    try:
        household_energy_archive["measurement_timestamp"] = pd.to_datetime(
            household_energy_archive["Date"] + " " + household_energy_archive["Time"],
            dayfirst=True
        )
    except Exception as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)
        
    household_energy_archive.set_index("measurement_timestamp", inplace=True)
    household_energy_archive.drop(columns=["Date", "Time"], inplace=True)
    
    household_energy_archive = household_energy_archive.interpolate(
        method="time",
        limit_direction="both"
    )
    household_energy_archive = household_energy_archive.ffill().bfill()
    
    numerical_feature_columns = household_energy_archive.select_dtypes(include=["float64"]).columns
    household_energy_archive[numerical_feature_columns] = (
        household_energy_archive[numerical_feature_columns].astype("float32")
    )
    
    if "Global_active_power" not in household_energy_archive.columns:
        print("Error: 'Global_active_power' column is missing after cleaning.")
        sys.exit(1)
        
    hourly_energy_profile = household_energy_archive[["Global_active_power"]].resample("1h").mean()
    hourly_energy_profile.rename(
        columns={"Global_active_power": "hourly_energy_demand"},
        inplace=True
    )
    
    if hourly_energy_profile.empty:
        print("Error: Hourly energy profile is empty after resampling.")
        sys.exit(1)
        
    return hourly_energy_profile


def feature_engineering(hourly_energy_profile: pd.DataFrame) -> pd.DataFrame:
    """
    Generates temporal, statistical, and cyclical features for forecasting.
    
    Args:
        hourly_energy_profile (pd.DataFrame): Hourly energy demand data.
        
    Returns:
        pd.DataFrame: Engineered feature bank with multi-horizon targets.
    """
    print("Feature Engineering...")
    engineered_feature_frame = hourly_energy_profile.copy()

    lag_hours = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for lag_hour in lag_hours:
        engineered_feature_frame[f"lag_{lag_hour}"] = (
            engineered_feature_frame["hourly_energy_demand"].shift(lag_hour)
        )

    rolling_windows = [6, 12, 24, 168]
    for rolling_window in rolling_windows:
        engineered_feature_frame[f"rolling_mean_{rolling_window}"] = (
            engineered_feature_frame["hourly_energy_demand"].rolling(rolling_window).mean()
        )

    engineered_feature_frame["rolling_std_24"] = engineered_feature_frame["hourly_energy_demand"].rolling(24).std()
    engineered_feature_frame["rolling_max_24"] = engineered_feature_frame["hourly_energy_demand"].rolling(24).max()
    engineered_feature_frame["rolling_min_24"] = engineered_feature_frame["hourly_energy_demand"].rolling(24).min()
    engineered_feature_frame["rolling_median_24"] = engineered_feature_frame["hourly_energy_demand"].rolling(24).median()

    engineered_feature_frame["ema_24"] = engineered_feature_frame["hourly_energy_demand"].ewm(span=24).mean()

    engineered_feature_frame["hour"] = engineered_feature_frame.index.hour
    engineered_feature_frame["weekday"] = engineered_feature_frame.index.dayofweek
    engineered_feature_frame["month"] = engineered_feature_frame.index.month
    engineered_feature_frame["quarter"] = engineered_feature_frame.index.quarter
    engineered_feature_frame["is_weekend"] = (engineered_feature_frame["weekday"] >= 5).astype(int)

    engineered_feature_frame["hour_sin"] = np.sin(2 * np.pi * engineered_feature_frame["hour"] / 24)
    engineered_feature_frame["hour_cos"] = np.cos(2 * np.pi * engineered_feature_frame["hour"] / 24)
    engineered_feature_frame["weekday_sin"] = np.sin(2 * np.pi * engineered_feature_frame["weekday"] / 7)
    engineered_feature_frame["weekday_cos"] = np.cos(2 * np.pi * engineered_feature_frame["weekday"] / 7)

    engineered_feature_frame["hourly_change"] = engineered_feature_frame["hourly_energy_demand"].diff()
    engineered_feature_frame["daily_change"] = (
        engineered_feature_frame["hourly_energy_demand"] - engineered_feature_frame["lag_24"]
    )

    engineered_feature_frame["rolling_range_24"] = (
        engineered_feature_frame["rolling_max_24"] - engineered_feature_frame["rolling_min_24"]
    )
    engineered_feature_frame["ema_168"] = engineered_feature_frame["hourly_energy_demand"].ewm(span=168).mean()
    engineered_feature_frame["rolling_skew_24"] = engineered_feature_frame["hourly_energy_demand"].rolling(24).skew()
    engineered_feature_frame["rolling_kurtosis_24"] = engineered_feature_frame["hourly_energy_demand"].rolling(24).kurt()
    engineered_feature_frame["is_month_start"] = engineered_feature_frame.index.is_month_start.astype(int)
    engineered_feature_frame["is_month_end"] = engineered_feature_frame.index.is_month_end.astype(int)

    engineered_feature_frame.dropna(inplace=True)

    forecast_feature_bank = engineered_feature_frame
    forecast_horizon_hours = 24

    for h in range(1, forecast_horizon_hours + 1):
        forecast_feature_bank[f"target_h{h}"] = (
            forecast_feature_bank["hourly_energy_demand"].shift(-h)
        )

    forecast_feature_bank.dropna(inplace=True)
    
    if forecast_feature_bank.empty:
        print("Error: Feature engineering resulted in an empty dataframe.")
        sys.exit(1)
        
    return forecast_feature_bank


def train_models(forecast_feature_bank: pd.DataFrame) -> Tuple[Dict[str, lgb.LGBMRegressor], pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    """
    Trains Direct Multi-Horizon LightGBM models.
    
    Args:
        forecast_feature_bank (pd.DataFrame): Dataset containing features and targets.
        
    Returns:
        Tuple containing:
            - Dict of trained models
            - Predictor columns DataFrame
            - Prediction targets DataFrame
            - Split index
            - Feature importances DataFrame
    """
    print("Training 24 LightGBM Models...")
    forecast_horizon_hours = 24
    
    predictor_columns = forecast_feature_bank.drop(
        columns=["hourly_energy_demand"] + [f"target_h{h}" for h in range(1, forecast_horizon_hours + 1)]
    )
    
    prediction_targets = forecast_feature_bank[[f"target_h{h}" for h in range(1, forecast_horizon_hours + 1)]]
    
    split_index = int(len(forecast_feature_bank) * 0.80)
    
    training_features = predictor_columns.iloc[:split_index]
    training_targets = prediction_targets.iloc[:split_index]
    
    testing_features = predictor_columns.iloc[split_index:]
    
    assert training_features.index.max() < testing_features.index.min(), "Data leakage detected: Overlapping train/test indices."
    
    multi_horizon_models = {}
    feature_importances_list = []
    
    for h in range(1, forecast_horizon_hours + 1):
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        model.fit(training_features, training_targets[f"target_h{h}"])
        multi_horizon_models[f"model_h{h}"] = model
        feature_importances_list.append(model.feature_importances_)
        
    avg_importances = np.mean(feature_importances_list, axis=0)
    feature_importance_df = pd.DataFrame({
        "Feature": predictor_columns.columns,
        "Average_Importance": avg_importances
    }).sort_values(by="Average_Importance", ascending=False)
    
    return multi_horizon_models, predictor_columns, prediction_targets, split_index, feature_importance_df


def evaluate_models(
    multi_horizon_models: Dict[str, lgb.LGBMRegressor], 
    predictor_columns: pd.DataFrame, 
    prediction_targets: pd.DataFrame, 
    split_index: int
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Evaluates multi-horizon models on the test set and captures samples for plotting.
    
    Args:
        multi_horizon_models (Dict): Trained LightGBM models.
        predictor_columns (pd.DataFrame): Features.
        prediction_targets (pd.DataFrame): Targets.
        split_index (int): Train/test split index.
        
    Returns:
        Tuple containing:
            - Evaluation metrics DataFrame.
            - Actual 24-hour targets array for plotting.
            - Predicted 24-hour targets array for plotting.
    """
    print("Evaluating...")
    forecast_horizon_hours = 24
    testing_features = predictor_columns.iloc[split_index:]
    testing_targets = prediction_targets.iloc[split_index:]
    
    evaluation_results = []
    
    actuals_h24 = None
    predictions_h24 = None
    
    for h in range(1, forecast_horizon_hours + 1):
        model_key = f"model_h{h}"
        target_column = f"target_h{h}"
        
        current_predictions = multi_horizon_models[model_key].predict(testing_features)
        
        if h == 24:
            actuals_h24 = testing_targets[target_column].values
            predictions_h24 = current_predictions
        
        horizon_mae = mean_absolute_error(
            testing_targets[target_column],
            current_predictions
        )
        
        naive_baseline_mae = mean_absolute_error(
            testing_targets[target_column],
            testing_features["lag_24"].values
        )
        
        improvement_percentage = (
            (naive_baseline_mae - horizon_mae) / naive_baseline_mae
        ) * 100
        
        evaluation_results.append({
            'Horizon': h,
            'Model_MAE': horizon_mae,
            'Naive_Baseline_MAE': naive_baseline_mae,
            'Improvement_vs_Baseline (%)': improvement_percentage
        })
        
    return pd.DataFrame(evaluation_results), actuals_h24, predictions_h24


def generate_plot(actuals: np.ndarray, predictions: np.ndarray) -> None:
    """
    Generates and saves a line plot comparing actual vs predicted demand,
    matching the original Colab notebook's aesthetics perfectly.
    
    Args:
        actuals (np.ndarray): Actual demand values.
        predictions (np.ndarray): Predicted demand values.
    """
    plt.figure(figsize=(18, 6))
    
    plt.plot(
        actuals,
        label="Actual",
        linewidth=2
    )
    
    plt.plot(
        predictions,
        label="Predicted",
        linewidth=2
    )
    
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_path = PROJECT_ROOT / "forecast_results.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def forecast_next_24_hours(
    multi_horizon_models: Dict[str, lgb.LGBMRegressor], 
    predictor_columns: pd.DataFrame
) -> pd.DataFrame:
    """
    Forecasts the next 24 hours based on the latest available data.
    
    Args:
        multi_horizon_models (Dict): Trained models.
        predictor_columns (pd.DataFrame): Available features.
        
    Returns:
        pd.DataFrame: Dataframe containing predictions for the next 24 hours.
    """
    last_row = predictor_columns.iloc[[-1]]
    last_timestamp = last_row.index[0]
    
    forecast_results = []
    
    for h in range(1, 25):
        model_key = f"model_h{h}"
        prediction = multi_horizon_models[model_key].predict(last_row)[0]
        forecast_timestamp = last_timestamp + pd.Timedelta(hours=h)
        forecast_results.append({
            'Forecast_Timestamp': forecast_timestamp,
            'Horizon': h,
            'Predicted_Demand': prediction
        })
        
    return pd.DataFrame(forecast_results)


def save_outputs(
    evaluation_df: pd.DataFrame, 
    forecast_df: pd.DataFrame, 
    multi_horizon_models: Dict[str, lgb.LGBMRegressor],
    feature_importance_df: pd.DataFrame
) -> None:
    """
    Saves metrics, forecasts, feature importances, and trained models cleanly formatting the Excel data.
    
    Args:
        evaluation_df (pd.DataFrame): Evaluation metrics.
        forecast_df (pd.DataFrame): Forecast predictions.
        multi_horizon_models (Dict): Trained models to save.
        feature_importance_df (pd.DataFrame): Feature importances.
    """
    print("Saving Outputs...")
    
    try:
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directories: {e}")
        sys.exit(1)
        
    # Format and round data for clean Excel/CSV viewing
    evaluation_df = evaluation_df.round(4)
    feature_importance_df["Average_Importance"] = feature_importance_df["Average_Importance"].round(4)
    
    forecast_df["Predicted_Demand"] = forecast_df["Predicted_Demand"].round(2)
    # Convert datetime to standardized string to prevent Excel '########' column overflow
    forecast_df["Forecast_Timestamp"] = forecast_df["Forecast_Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    metrics_path = OUTPUT_DIRECTORY / "metrics.csv"
    try:
        evaluation_df.to_csv(metrics_path, index=False)
    except PermissionError:
        print(f"Warning: Could not save metrics.csv. Please close it if open in Excel.")
        
    forecast_path = OUTPUT_DIRECTORY / "forecast_results.csv"
    try:
        forecast_df.to_csv(forecast_path, index=False)
    except PermissionError:
        print(f"Warning: Could not save forecast_results.csv. Please close it if open in Excel.")
    
    importance_path = OUTPUT_DIRECTORY / "feature_importance.csv"
    try:
        feature_importance_df.to_csv(importance_path, index=False)
    except PermissionError:
        print(f"Warning: Could not save feature_importance.csv. Please close it if open in Excel.")
    
    for model_name, model in multi_horizon_models.items():
        model_path = MODEL_DIRECTORY / f"{model_name}.joblib"
        try:
            joblib.dump(model, model_path)
        except Exception as e:
            print(f"Error saving model {model_name}: {e}")
            
            
def print_forecast_summary(evaluation_df: pd.DataFrame) -> None:
    """
    Prints a concise summary of the forecast horizons.
    
    Args:
        evaluation_df (pd.DataFrame): DataFrame containing the evaluation metrics.
    """
    best_row = evaluation_df.loc[evaluation_df['Model_MAE'].idxmin()]
    worst_row = evaluation_df.loc[evaluation_df['Model_MAE'].idxmax()]
    
    avg_mae = evaluation_df['Model_MAE'].mean()
    avg_improvement = evaluation_df['Improvement_vs_Baseline (%)'].mean()
    
    print("\n==================================================")
    print("Forecast Horizon Summary")
    print("==================================================")
    print(f"Best Horizon      : H{int(best_row['Horizon'])}")
    print(f"Best MAE          : {best_row['Model_MAE']:.4f}")
    print()
    print(f"Worst Horizon     : H{int(worst_row['Horizon'])}")
    print(f"Worst MAE         : {worst_row['Model_MAE']:.4f}")
    print()
    print(f"Average MAE       : {avg_mae:.4f}")
    print()
    print(f"Average Improvement over Baseline : {avg_improvement:.1f}%")
    print("==================================================\n")


def main() -> None:
    """Main pipeline execution point."""
    try:
        household_energy_archive = load_data()
        hourly_energy_profile = clean_data(household_energy_archive)
        forecast_feature_bank = feature_engineering(hourly_energy_profile)
        
        models_data = train_models(forecast_feature_bank)
        multi_horizon_models, predictor_columns, prediction_targets, split_index, feature_importance_df = models_data
        
        evaluation_df, actuals_h24, predictions_h24 = evaluate_models(
            multi_horizon_models, predictor_columns, prediction_targets, split_index
        )
        
        # Sample 300 points for a clean visualization as done in original notebooks
        generate_plot(actuals_h24[:300], predictions_h24[:300])
        
        forecast_df = forecast_next_24_hours(multi_horizon_models, predictor_columns)
        
        save_outputs(evaluation_df, forecast_df, multi_horizon_models, feature_importance_df)
        
        print_forecast_summary(evaluation_df)
        
        print("Pipeline Completed Successfully")
        print("====================================================")
        
    except Exception as e:
        print(f"Pipeline Failed due to an unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()