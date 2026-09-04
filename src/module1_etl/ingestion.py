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
        
        # Merge on example_id to create a unified dataset
        merged_df = pd.merge(features_df, labels_df, on="example_id", how="inner")
        logger.info(f"Successfully loaded {merged_df.shape[0]} records.")
        
        return merged_df
        
    except FileNotFoundError as e:
        logger.error(f"Data ingestion failed: {e}")
        raise