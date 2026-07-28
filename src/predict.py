import pandas as pd
import joblib
from typing import Dict
import lightgbm as lgb
from src.utils import get_logger, MODEL_DIR
logger = get_logger(__name__)

def load_models() -> Dict[str, lgb.LGBMRegressor]:
    models = {}
    for h in range(1, 25):
        model_path = MODEL_DIR / f'model_h{h}.joblib'
        if model_path.exists():
            models[f'model_h{h}'] = joblib.load(model_path)
        else:
            logger.error(f'Model for horizon {h} not found at {model_path}')
    return models

def forecast_next_24_hours(multi_horizon_models: Dict[str, lgb.LGBMRegressor], predictor_columns: pd.DataFrame) -> pd.DataFrame:
    logger.info('Generating Forecasts for the next 24 hours...')
    last_row = predictor_columns.iloc[[-1]]
    last_timestamp = last_row.index[0]
    forecast_results = []
    for h in range(1, 25):
        model_key = f'model_h{h}'
        prediction = multi_horizon_models[model_key].predict(last_row)[0]
        forecast_timestamp = last_timestamp + pd.Timedelta(hours=h)
        forecast_results.append({'Forecast_Timestamp': forecast_timestamp, 'Horizon': h, 'Predicted_Demand': prediction})
    return pd.DataFrame(forecast_results)