import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import logging
from src.config import MODEL_DIR

logger = logging.getLogger(__name__)

def execute_classical_fallback(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """
    Executes a heavily penalized XGBoost baseline engine to compute resistance probabilities
    when quantum hardware is unreachable or times out.
    """
    logger.warning("Quantum hardware timeout/error simulated. Initializing Penalized XGBoost Engine...")
    
    # L1 (alpha) and L2 (lambda) regularization applied to prevent overfitting 
    # on the high-dimensional clinical dataset.
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        reg_alpha=0.5,      # L1 Penalty
        reg_lambda=1.5,     # L2 Penalty
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False
    )
    
    logger.info("Training Penalized XGBoost on complete history...")
    xgb_model.fit(X_train, y_train)
    
    # Save classical baseline artifact
    model_path = MODEL_DIR / "xgb_baseline.json"
    xgb_model.save_model(model_path)
    logger.info(f"Fallback XGBoost model saved to {model_path}")
    
    # Compute Probability Prediction
    predictions = xgb_model.predict_proba(X_test)[:, 1]
    
    return predictions