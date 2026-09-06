import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from src.config import MODEL_DIR

logger = logging.getLogger(__name__)

def prepare_quantum_state(
    X_compressed: pd.DataFrame, fit: bool = True, scaler: MinMaxScaler = None
) -> pd.DataFrame:
    """
    Maps the compressed classical features into a Quantum Feature State Vector |x⟩.
    Scales data strictly between 0 and π for quantum angle embedding.

    fit=True (default -- used during training, e.g. main.py): fits a NEW
    scaler on X_compressed and saves it to MODEL_DIR/quantum_state_scaler.pkl.

    fit=False (MUST be used at inference time, e.g. app.py): reuses an
    already-fitted `scaler` instead of fitting a new one on whatever was
    just passed in.

    THE BUG THIS FIXES: this function previously always called
    fit_transform(), even when called on a single row at inference time.
    MinMaxScaler fit on ONE row has zero data range in every column, so
    every feature silently collapsed to the same constant angle regardless
    of the patient's actual values -- the QNN/QSVM effectively received an
    identical, content-free input for every patient. The scaler was
    already being saved during training; nothing ever loaded it back for
    reuse at serving time.
    """
    if fit:
        logger.info("Fitting quantum state scaler on this batch (training mode)...")
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_state = scaler.fit_transform(X_compressed)

        scaler_path = MODEL_DIR / "quantum_state_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved quantum state scaler to {scaler_path}")
    else:
        if scaler is None:
            raise ValueError(
                "prepare_quantum_state(fit=False) requires a pre-fitted `scaler` "
                "(load quantum_state_scaler.pkl and pass it in) -- refusing to "
                "silently fit a fresh scaler on what may be a single inference row."
            )
        logger.info("Applying saved quantum state scaler (inference mode)...")
        X_state = scaler.transform(X_compressed)

    return pd.DataFrame(X_state, columns=X_compressed.columns, index=X_compressed.index)