import sys
from src.utils import ensure_directories, get_logger, OUTPUT_DIR
from src.preprocess import load_data, clean_data
from src.feature_engineering import feature_engineering
from src.train import train_and_evaluate
from src.predict import forecast_next_24_hours
from monitoring.monitor import log_prediction_metadata

logger = get_logger("pipeline_orchestrator")

def main():
    logger.info("==================================================")
    logger.info("Starting GridCastAI MLOps Pipeline...")
    ensure_directories()
    
    try:
        # 1. Ingest & Preprocess
        raw_df = load_data()
        clean_df = clean_data(raw_df)
        
        # 2. Feature Engineering
        features_df = feature_engineering(clean_df)
        
        # 3. Train, Evaluate & Register to MLflow
        models, eval_df = train_and_evaluate(features_df)
        
        # 4. Predict
        predictor_columns = features_df.drop(
            columns=["hourly_energy_demand"] + [f"target_h{h}" for h in range(1, 25)]
        )
        forecast_df = forecast_next_24_hours(models, predictor_columns)
        
        # 5. Save Outputs
        forecast_df["Predicted_Demand"] = forecast_df["Predicted_Demand"].round(2)
        forecast_df["Forecast_Timestamp"] = forecast_df["Forecast_Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        forecast_df.to_csv(OUTPUT_DIR / "forecast_results.csv", index=False)
        eval_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
        
        # 6. Monitor
        log_prediction_metadata(forecast_df)
        
        logger.info("GridCastAI MLOps Pipeline Completed Successfully!")
        logger.info("==================================================")
        
    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()