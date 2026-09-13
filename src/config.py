import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for path in [PROCESSED_DATA_DIR, MODEL_DIR, RAW_DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

def _resolve_file(filename: str) -> Path:
    """Dynamically resolves file paths to prevent FileNotFoundError."""
    raw_path = RAW_DATA_DIR / filename
    root_path = BASE_DIR / filename
    
    if raw_path.exists():
        return raw_path
    elif root_path.exists():
        return root_path
    return raw_path

# File Paths
FEATURES_FILE = _resolve_file("all_uti_features.csv")
LABELS_FILE = _resolve_file("all_uti_resist_labels.csv")
PRESCRIPTIONS_FILE = _resolve_file("all_prescriptions.csv")