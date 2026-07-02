import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Tuple

from src.utils import get_logger, MODEL_DIR, PROJECT_ROOT

logger = get_logger(__name__)

# Ensure MLflow operates out of the correct directory locally
MLFLOW_DB = f"sqlite:///{PROJECT_ROOT}/mlruns/mlflow.db"
mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment("GridCastAI_Direct_MultiHorizon")

def train_and_evaluate(forecast_feature_bank: pd.DataFrame) -> Tuple[Dict[str, lgb.LGBMRegressor], pd.DataFrame]:
    """
    Trains Direct Multi-Horizon LightGBM models and evaluates them, logging everything to MLflow.
    """
    logger.info("Training and Evaluating 24 LightGBM Models...")
    forecast_horizon_hours = 24
    
    predictor_columns = forecast_feature_bank.drop(
        columns=["hourly_energy_demand"] + [f"target_h{h}" for h in range(1, forecast_horizon_hours + 1)]
    )
    prediction_targets = forecast_feature_bank[[f"target_h{h}" for h in range(1, forecast_horizon_hours + 1)]]
    
    split_index = int(len(forecast_feature_bank) * 0.80)
    training_features = predictor_columns.iloc[:split_index]
    training_targets = prediction_targets.iloc[:split_index]
    testing_features = predictor_columns.iloc[split_index:]
    testing_targets = prediction_targets.iloc[split_index:]
    
    assert training_features.index.max() < testing_features.index.min(), "Data leakage detected."
    
    multi_horizon_models = {}
    feature_importances_list = []
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
    
    with mlflow.start_run(run_name="Production_Training"):
        mlflow.log_params(lgb_params)
        mlflow.log_param("Train_Size", len(training_features))
        mlflow.log_param("Test_Size", len(testing_features))
        
        for h in range(1, forecast_horizon_hours + 1):
            logger.info(f"Training Model for Horizon {h}...")
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(training_features, training_targets[f"target_h{h}"])
            
            multi_horizon_models[f"model_h{h}"] = model
            feature_importances_list.append(model.feature_importances_)
            
            # Evaluate
            preds = model.predict(testing_features)
            mae = mean_absolute_error(testing_targets[f"target_h{h}"], preds)
            rmse = np.sqrt(mean_squared_error(testing_targets[f"target_h{h}"], preds))
            
            evaluation_results.append({
                'Horizon': h,
                'Model_MAE': mae,
                'Model_RMSE': rmse
            })
            
            # Log Model natively to MLFlow Registry
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path=f"model_h{h}",
                registered_model_name=f"GridCastAI_Model_H{h}"
            )
            
            # Also save locally for fast API fallback
            joblib.dump(model, MODEL_DIR / f"model_h{h}.joblib")
            
        eval_df = pd.DataFrame(evaluation_results)
        
        avg_mae = eval_df['Model_MAE'].mean()
        avg_rmse = eval_df['Model_RMSE'].mean()
        mlflow.log_metric("Average_MAE", avg_mae)
        mlflow.log_metric("Average_RMSE", avg_rmse)
        logger.info(f"Overall Average MAE: {avg_mae:.4f}")
        
    return multi_horizon_models, eval_df
