from pathlib import Path
import joblib
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def get_raw_path(filename: str) -> Path:
    """Full path to a raw data file."""
    return RAW_DIR / filename


def get_processed_path(filename: str) -> Path:
    """Full path to a processed CSV."""
    return PROCESSED_DIR / filename


def get_model_path(filename: str) -> Path:
    """Full path to a model file."""
    return MODELS_DIR / filename


def save_model(model: Any, filename: str) -> None:
    """
    Save a model via joblib to MODELS_DIR/filename.
    """
    path = get_model_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(filename: str) -> Any:
    """
    Load and return a model from MODELS_DIR/filename.
    """
    return joblib.load(get_model_path(filename))
