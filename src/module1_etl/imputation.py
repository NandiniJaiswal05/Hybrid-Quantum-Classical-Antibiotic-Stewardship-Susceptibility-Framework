import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import MODEL_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

def clean_and_scale_features(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    """
    Implements Sequential Bridging to handle missing temporal gaps 
    without flattening the time-series context.
    """
    meta_cols = [col for col in ['example_id', 'is_train', 'uncomplicated'] + target_cols if col in df.columns]
    feature_cols = [col for col in df.columns if col not in meta_cols]
    
    X = df[feature_cols].copy()
    missing_count = X.isnull().sum().sum()
    
    if missing_count > 0:
        logger.info(f"Missing temporal data detected ({missing_count} nulls). Applying Sequential Interpolation...")
        # Interpolate along axis=1 (columns) bridges gaps between sequential temporal states
        # Remaining NAs are filled with 0 (indicating baseline/no event)
        X_imputed = X.interpolate(method='linear', axis=1).fillna(0)
    else:
        logger.info("Complete history detected. Bypassing imputation.")
        X_imputed = X

    logger.info("Applying Standard ETL (Scaling)...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_cols, index=X_imputed.index)
    
    joblib.dump(scaler, MODEL_DIR / "standard_scaler.pkl")
    return pd.concat([df[meta_cols], X_scaled], axis=1)