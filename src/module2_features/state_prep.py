import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from src.config import MODEL_DIR

logger = logging.getLogger(__name__)

def prepare_quantum_state(X_compressed: pd.DataFrame) -> pd.DataFrame:
    """
    Maps the compressed classical features into a Quantum Feature State Vector |x⟩.
    Scales data strictly between 0 and π for quantum angle embedding (RX, RY, RZ gates).
    """
    logger.info("Mapping compressed features to Quantum State Vector (Scaling to [0, π])...")
    
    # Bound the data for quantum rotations
    state_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_state = state_scaler.fit_transform(X_compressed)
    
    # Save the quantum state scaler
    scaler_path = MODEL_DIR / "quantum_state_scaler.pkl"
    joblib.dump(state_scaler, scaler_path)
    
    return pd.DataFrame(X_state, columns=X_compressed.columns, index=X_compressed.index)