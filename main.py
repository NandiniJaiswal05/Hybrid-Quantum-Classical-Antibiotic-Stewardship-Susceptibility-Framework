import logging
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure absolute path resolution for imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Import pipeline modules
from src.module1_etl.ingestion import load_clinical_data
from src.module1_etl.imputation import clean_and_scale_features
from src.module2_features.distribution import evaluate_distribution
from src.module2_features.reduction import compress_features
from src.module2_features.state_prep import prepare_quantum_state
from src.module3_execution.quantum_circuit import execute_quantum_check, execute_quantum_optimization
from src.module3_execution.qsvm_fallback import train_and_predict_qsvm
from src.module3_execution.xgboost_engine import execute_classical_fallback

def run_pipeline():
    """Sequentially executes Modules 1 through 3 with SMOTE balancing to generate model artifacts."""
    logger.info("=== Starting AMR-UTI Hybrid Quantum Pipeline ===")

    # ---------------------------------------------------------
    # MODULE 1: ETL & Ingestion
    # ---------------------------------------------------------
    logger.info("\n--- MODULE 1: ETL & INGESTION ---")
    
    raw_data = load_clinical_data()
    target_cols = ['NIT', 'SXT', 'CIP', 'LVX'] 
    processed_df = clean_and_scale_features(raw_data, target_cols)

    logger.info("Dropping records with missing target labels to prepare for model training...")
    processed_df = processed_df.dropna(subset=['CIP'])

    cols_to_drop = [col for col in ['example_id', 'is_train', 'uncomplicated'] + target_cols if col in processed_df.columns]
    features = processed_df.drop(columns=cols_to_drop)
    targets = processed_df['CIP']
    
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    logger.info(f"Data split completed: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

    # ---------------------------------------------------------
    # MODULE 2: Feature Engineering & Compression
    # ---------------------------------------------------------
    logger.info("\n--- MODULE 2: FEATURE ENGINEERING & COMPRESSION ---")
    
    distribution = evaluate_distribution(X_train)
    X_train_compressed = compress_features(X_train, distribution_type=distribution)
    X_train_state = prepare_quantum_state(X_train_compressed)

    # --- ADD-ON: SMOTE Data Balancing (Training Set Only) ---
    logger.info("Applying SMOTE (Synthetic Minority Over-sampling) to address class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_compressed, y_train)
    logger.info(f"Class distribution after SMOTE: {y_train_resampled.value_counts().to_dict()}")

    # ---------------------------------------------------------
    # MODULE 3: Execution Engine & Quantum Fallback
    # ---------------------------------------------------------
    logger.info("\n--- MODULE 3: EXECUTION ENGINE & QUANTUM FALLBACK ---")
    
    execution_path = execute_quantum_check(X_train_state)

    # THE FIX: `predictions` used to be set unconditionally by
    # execute_classical_fallback() *before* this branch even ran, so no
    # matter what execution_path came back as, the XGBoost result is what
    # got returned -- the "quantum_optimization" branch below had no code
    # in it at all. Now each branch sets `predictions` exactly once, and
    # the quantum path actually calls a QNN training routine instead of
    # just logging a message.
    predictions = None

    if execution_path == "quantum_optimization":
        # execute_quantum_optimization() logs its own "Path Selected" line
        # and reuses the exact circuit execute_quantum_check() evaluated,
        # so the barren-plateau decision and the trained model agree.
        predictions = execute_quantum_optimization(X_train_state, y_train)

    elif execution_path == "qsvm_fallback":
        logger.info("Path Selected: Barren Plateau Detected -> Routing to QSVM (Fallback 1).")
        subset_X_train = X_train_state.values[:50]
        subset_y_train = y_train.values[:50]
        subset_X_test = X_train_state.values[50:70]
        predictions = train_and_predict_qsvm(subset_X_train, subset_y_train, subset_X_test)

    elif execution_path == "classical_fallback":
        logger.info("Path Selected: Hardware Timeout/Error -> Routing to Penalized XGBoost (Fallback 2).")
        predictions = execute_classical_fallback(X_train_resampled, y_train_resampled, X_train_compressed.head(20))

    else:
        # Previously execute_quantum_check() could only return one of the
        # three known strings, but if that ever changes, fail loudly
        # instead of silently leaving `predictions` as None.
        raise ValueError(f"execute_quantum_check() returned an unrecognized execution_path: {execution_path!r}")

    logger.info(f"Engine used for this run: {execution_path}")
    logger.info("\n=== Pipeline Execution Complete. Artifacts generated in models/ ===")
    return predictions, execution_path

if __name__ == "__main__":
    run_pipeline()