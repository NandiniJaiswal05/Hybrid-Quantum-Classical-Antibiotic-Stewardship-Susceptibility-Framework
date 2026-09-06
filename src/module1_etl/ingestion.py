import pandas as pd
import logging
from src.config import FEATURES_FILE, LABELS_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_clinical_data() -> pd.DataFrame:
    """Ingests raw EHR feature and label datasets."""
    try:
        logger.info("Loading raw EHR data...")
        features_df = pd.read_csv(FEATURES_FILE)
        labels_df = pd.read_csv(LABELS_FILE)

        # FIX: both raw PhysioNet AMR-UTI files carry their own is_train /
        # uncomplicated columns. Merging without dropping one side's copies
        # first produces is_train_x/is_train_y and
        # uncomplicated_x/uncomplicated_y -- main.py's cols_to_drop check
        # looks for the exact names "is_train" and "uncomplicated", which no
        # longer exist post-merge, so all four suffixed columns silently
        # leak into the feature set used for training, including a
        # near-perfect proxy for the train/test split itself.
        overlap = [c for c in ("is_train", "uncomplicated") if c in features_df.columns and c in labels_df.columns]
        if overlap:
            logger.info(f"Dropping duplicate cohort/split columns from features table before merge: {overlap}")
            features_df = features_df.drop(columns=overlap)

        # Merge on example_id to create a unified dataset
        merged_df = pd.merge(features_df, labels_df, on="example_id", how="inner")
        logger.info(f"Successfully loaded {merged_df.shape[0]} records.")
        
        return merged_df
        
    except FileNotFoundError as e:
        logger.error(f"Data ingestion failed: {e}")
        raise