import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from src.config import MODEL_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

def clean_and_scale_features(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    """
    Evaluates sparsity, applies KNN imputation if missing history is detected,
    and scales features via Standard ETL.
    """
    # Safely identify metadata columns that actually exist in the dataframe
    potential_meta = ['example_id', 'is_train', 'uncomplicated'] + target_cols
    meta_cols = [col for col in potential_meta if col in df.columns]
    
    # Separate features from IDs and targets
    feature_cols = [col for col in df.columns if col not in meta_cols]
    
    X = df[feature_cols].copy()
    
    # 1. Check Feature Sparsity/Nulls
    missing_count = X.isnull().sum().sum()
    
    if missing_count > 0:
        logger.info(f"Missing history detected ({missing_count} nulls). Routing to Predictive/KNN Imputation...")
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)
    else:
        logger.info("Complete history detected. Bypassing imputation.")
        X_imputed = X

    # 2. Standard ETL (Standardization)
    logger.info("Applying Standard ETL (Scaling)...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_cols, index=X_imputed.index)
    
    # Save the scaler for Module 4 (UI Inference)
    scaler_path = MODEL_DIR / "standard_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Recombine with metadata and targets
    processed_df = pd.concat([df[meta_cols], X_scaled], axis=1)
    
    # Save processed data
    output_path = PROCESSED_DATA_DIR / "processed_clinical_data.csv"
    processed_df.to_csv(output_path, index=False)
    
    return processed_df