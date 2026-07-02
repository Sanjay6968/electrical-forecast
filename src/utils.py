import logging
from pathlib import Path

# Setup Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a standard Python logger for the module.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def ensure_directories():
    """
    Ensures all necessary directories exist for the MLOps pipeline.
    """
    for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, MLRUNS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
