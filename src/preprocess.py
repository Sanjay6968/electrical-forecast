import sys
import pandas as pd
from pathlib import Path
from typing import Optional
from src.utils import get_logger, DATA_DIR

logger = get_logger(__name__)

def load_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the household power consumption dataset.
    """
    path = data_path or (DATA_DIR / "household_power_consumption.txt")
    logger.info(f"Loading Dataset from {path}...")
    
    if not path.exists():
        logger.error(f"Dataset not found at {path}.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)
        
    if df.empty:
        logger.error("Loaded dataset is empty.")
        sys.exit(1)
        
    expected_columns = ["Date", "Time", "Global_active_power"]
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        logger.error(f"Missing expected columns: {missing}")
        sys.exit(1)
        
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data and resamples to hourly frequency.
    """
    logger.info("Cleaning Data and resampling to hourly frequency...")
    
    try:
        df["measurement_timestamp"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], dayfirst=True
        )
    except Exception as e:
        logger.error(f"Error parsing dates: {e}")
        sys.exit(1)
        
    df.set_index("measurement_timestamp", inplace=True)
    df.drop(columns=["Date", "Time"], inplace=True)
    
    df = df.interpolate(method="time", limit_direction="both")
    df = df.ffill().bfill()
    
    num_cols = df.select_dtypes(include=["float64"]).columns
    df[num_cols] = df[num_cols].astype("float32")
    
    hourly = df[["Global_active_power"]].resample("1h").mean()
    hourly.rename(columns={"Global_active_power": "hourly_energy_demand"}, inplace=True)
    
    if hourly.empty:
        logger.error("Hourly energy profile is empty after resampling.")
        sys.exit(1)
        
    return hourly
