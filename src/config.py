import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for path in [PROCESSED_DATA_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# File Paths
FEATURES_FILE = RAW_DATA_DIR / "all_uti_features.csv"
LABELS_FILE = RAW_DATA_DIR / "all_uti_resist_labels.csv"
PRESCRIPTIONS_FILE = RAW_DATA_DIR / "all_prescriptions.csv"