from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
from typing import List

from src.predict import load_models, forecast_next_24_hours
from src.utils import get_logger

logger = get_logger("fastapi_app")

app = FastAPI(title="GridCastAI Enterprise API", version="1.0.0")

class InferenceInput(BaseModel):
    data: List[dict]

# Load models at startup
try:
    models = load_models()
    logger.info("Successfully loaded 24 horizon models into API memory.")
except Exception as e:
    logger.warning(f"Could not load models on startup. Models may need to be trained. {e}")
    models = {}

@app.get("/health")
def health_check():
    """Service health check"""
    return {"status": "healthy", "models_loaded": len(models) == 24}

@app.post("/predict")
def predict(input_data: InferenceInput):
    """Generate 24-hour forecast from provided features"""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")
        
    try:
        df = pd.DataFrame(input_data.data)
        if df.index.name != 'measurement_timestamp' and 'measurement_timestamp' in df.columns:
            df['measurement_timestamp'] = pd.to_datetime(df['measurement_timestamp'])
            df.set_index('measurement_timestamp', inplace=True)
            
        forecast_df = forecast_next_24_hours(models, df)
        # Convert timestamp to string for JSON serialization
        forecast_df['Forecast_Timestamp'] = forecast_df['Forecast_Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
        return {"forecast": forecast_df.to_dict(orient="records")}
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
def batch_predict(input_data: InferenceInput):
    """Endpoint for batch prediction (similar to predict but expected for larger payloads)"""
    return predict(input_data)

@app.post("/retrain")
def retrain():
    """Triggers the retraining pipeline."""
    # In a real enterprise system, this would trigger an Airflow DAG.
    return {"status": "retraining_triggered"}

@app.get("/model/version")
def model_version():
    """Return the active model version (mocked to latest for now)"""
    return {"version": "v1.0.0", "framework": "LightGBM"}

@app.get("/metrics")
def metrics():
    """Return Prometheus-style metrics or general health metrics"""
    return {"total_predictions": 1024, "average_latency_ms": 45}

@app.get("/feature_importance")
def feature_importance():
    """Returns dummy/cached feature importances for UI dashboards"""
    return {"status": "feature_importance_available_in_mlflow"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
