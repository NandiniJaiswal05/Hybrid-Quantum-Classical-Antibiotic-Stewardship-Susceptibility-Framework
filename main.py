import logging
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.config import MODEL_DIR
from src.module1_etl.ingestion import load_clinical_data
from src.module1_etl.imputation import clean_and_scale_features
from src.module2_features.reduction import parse_temporal_features
from src.module3_execution.quantum_circuit import HybridQuantumClassicalModel, execute_quantum_check, execute_quantum_optimization
from src.module3_execution.qsvm_fallback import train_and_predict_qsvm
from src.module3_execution.xgboost_engine import execute_classical_fallback

ENGINE_SPECIFIC_ARTIFACTS = [
    "hybrid_mps_qnn.pt", "hybrid_mps_qnn_meta.json",
    "qsvm_weights.pkl", "qsvm_train_embeddings.npy",
    "qsvm_mps_embedder.pt", "qsvm_mps_embedder_meta.json",
]

def _clear_stale_engine_artifacts():
    removed = []
    for fname in ENGINE_SPECIFIC_ARTIFACTS:
        p = MODEL_DIR / fname
        if p.exists():
            p.unlink()
            removed.append(fname)
    if removed:
        logger.info(f"Cleared stale engine artifact(s) from previous run: {removed}")

def run_pipeline():
    logger.info("=== Starting AMR-UTI Hybrid Quantum Pipeline ===")
    _clear_stale_engine_artifacts()

    logger.info("\n--- MODULE 1: ETL & INGESTION ---")
    raw_data = load_clinical_data()
    target_cols = ['NIT', 'SXT', 'CIP', 'LVX']
    processed_df = clean_and_scale_features(raw_data, target_cols).dropna(subset=['CIP'])

    cols_to_drop = [col for col in ['example_id', 'is_train', 'uncomplicated'] + target_cols if col in processed_df.columns]
    features = processed_df.drop(columns=cols_to_drop)
    targets = processed_df['CIP']

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    logger.info(f"Data split completed: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

    logger.info("Applying SMOTE (Synthetic Minority Over-sampling)...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    if not isinstance(X_train_resampled, pd.DataFrame):
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    if not isinstance(y_train_resampled, pd.Series):
        y_train_resampled = pd.Series(y_train_resampled, name=y_train.name)

    logger.info("\n--- MODULE 2 & 3: HYBRID MPS & EXECUTION ENGINE ---")
    feature_groups, static_features = parse_temporal_features(X_train_resampled)
    time_order = ['ALL', '180', '90', '30', '14', '7']
    active_steps = [t for t in time_order if len(feature_groups[t]) > 0]
    mps_input_dims = [len(feature_groups[t]) for t in active_steps]

    hybrid_model = HybridQuantumClassicalModel(
        mps_input_dims=mps_input_dims,
        static_dim=len(static_features),
        n_layers=3
    )
    hybrid_model.active_steps = active_steps

    execution_path = execute_quantum_check(hybrid_model, X_train_resampled)
    predictions = None

    if execution_path == "quantum_optimization":
        predictions = execute_quantum_optimization(hybrid_model, X_train_resampled, y_train_resampled)
    elif execution_path == "qsvm_fallback":
        logger.info("Path Selected: Barren Plateau Detected -> Routing to QSVM (Fallback 1).")
        predictions = train_and_predict_qsvm(hybrid_model, X_train_resampled.head(50), y_train_resampled.head(50), X_test.head(20))
    elif execution_path == "classical_fallback":
        logger.info("Path Selected: Hardware Timeout/Error -> Routing to Penalized XGBoost (Fallback 2).")
        predictions = execute_classical_fallback(X_train_resampled, y_train_resampled, X_test.head(20))

    if execution_path != "classical_fallback":
        logger.info("Training baseline XGBoost artifact for feature-set consistency checks...")
        execute_classical_fallback(X_train_resampled, y_train_resampled, X_test.head(20))

    logger.info(f"Engine used for this run: {execution_path}")
    logger.info("\n=== Pipeline Execution Complete. Artifacts generated in models/ ===")
    return predictions, execution_path

if __name__ == "__main__":
    run_pipeline()