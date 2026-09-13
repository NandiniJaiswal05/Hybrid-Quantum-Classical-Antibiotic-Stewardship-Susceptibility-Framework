from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.config import MODEL_DIR, PROCESSED_DATA_DIR
from src.module1_etl.imputation import clean_and_scale_features
from src.module1_etl.ingestion import load_clinical_data
from src.module2_features.reduction import parse_temporal_features
from src.module3_execution.quantum_circuit import HybridQuantumClassicalModel
from src.module3_execution.qsvm_fallback import extract_mps_embeddings, q_kernel_matrix

TARGET_COLS = ['NIT', 'SXT', 'CIP', 'LVX']
TIME_ORDER = ['ALL', '180', '90', '30', '14', '7']

def _load_processed_data() -> pd.DataFrame:
    data_path = PROCESSED_DATA_DIR / "processed_clinical_data.csv"
    if data_path.exists():
        return pd.read_csv(data_path).dropna(subset=['CIP'])

    raw_data = load_clinical_data()
    df = clean_and_scale_features(raw_data, TARGET_COLS).dropna(subset=['CIP'])
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    return df

def _predict_hybrid_qnn(X_test: pd.DataFrame) -> np.ndarray:
    hybrid_path = MODEL_DIR / "hybrid_mps_qnn.pt"
    meta_path = MODEL_DIR / "hybrid_mps_qnn_meta.json"
    if not (hybrid_path.exists() and meta_path.exists()):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    model = HybridQuantumClassicalModel(
        mps_input_dims=meta["input_dims"],
        static_dim=meta.get("static_dim", 0),
        n_layers=meta.get("n_layers", 3),
    )
    model.load_state_dict(torch.load(hybrid_path, map_location=torch.device('cpu')))
    model.eval()

    feature_groups, _ = parse_temporal_features(X_test)
    active_steps = [t for t in TIME_ORDER if len(feature_groups[t]) > 0]
    X_seq = [torch.tensor(X_test[feature_groups[t]].values, dtype=torch.float32) for t in active_steps]

    X_static = None
    static_names = meta.get("static_features", [])
    if static_names:
        X_static = torch.tensor(X_test[static_names].values, dtype=torch.float32)

    with torch.no_grad():
        probs = model(X_seq, X_static).numpy().flatten()
    return probs

def _predict_qsvm(X_test: pd.DataFrame) -> np.ndarray:
    qsvm_path = MODEL_DIR / "qsvm_weights.pkl"
    train_embed_path = MODEL_DIR / "qsvm_train_embeddings.npy"
    embedder_path = MODEL_DIR / "qsvm_mps_embedder.pt"
    embedder_meta_path = MODEL_DIR / "qsvm_mps_embedder_meta.json"

    if not all(p.exists() for p in [qsvm_path, train_embed_path, embedder_path, embedder_meta_path]):
        return None

    with open(embedder_meta_path) as f:
        meta = json.load(f)

    embedder = HybridQuantumClassicalModel(
        mps_input_dims=meta["input_dims"],
        static_dim=meta.get("static_dim", 0),
        n_layers=meta.get("n_layers", 3),
    )
    embedder.load_state_dict(torch.load(embedder_path, map_location=torch.device('cpu')))
    embedder.eval()

    qsvm = joblib.load(qsvm_path)
    X_train_embedded = np.load(train_embed_path)
    X_test_embedded = extract_mps_embeddings(embedder, X_test)

    kernel_test = q_kernel_matrix(X_test_embedded, X_train_embedded)
    return qsvm.predict_proba(kernel_test)[:, 1]

def _predict_xgb(X_test: pd.DataFrame) -> np.ndarray:
    xgb_path = MODEL_DIR / "xgb_baseline.json"
    if not xgb_path.exists():
        return None
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(xgb_path)
    return model.predict_proba(X_test)[:, 1]

def generate_seminar_metrics():
    print("Loading test data and evaluating the AMR-UTI pipeline...")
    df = _load_processed_data()
    cols_to_drop = [c for c in ['example_id', 'is_train', 'uncomplicated'] + TARGET_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df['CIP']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    active_engine, y_pred_proba = None, None
    for engine_name, predict_fn in [
        ("Hybrid MPS-QNN (Quantum Optimization, static-fused)", _predict_hybrid_qnn),
        ("Quantum Kernel SVM (Fallback 1)", _predict_qsvm),
        ("Penalized XGBoost (Fallback 2)", _predict_xgb),
    ]:
        try:
            result = predict_fn(X_test)
        except Exception:
            continue
        if result is not None:
            active_engine = engine_name
            y_pred_proba = result
            print(f"Evaluating using: {active_engine}")
            break

    if y_pred_proba is None:
        raise FileNotFoundError("No usable trained artifact found. Run main.py first.")

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5

    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    y_pred_tuned = (y_pred_proba >= optimal_threshold).astype(int)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("=" * 60)
    print(f" AMR-UTI PIPELINE METRICS (N={len(y_test):,})")
    print("=" * 60)
    print(f"Active Engine:              {active_engine}")
    print(f"ROC-AUC Score:              {roc_auc:.4f}")
    print(f"Optimal Decision Threshold: {optimal_threshold:.4f} (Default: 0.50)")
    print("=" * 60)
    print("\n--- DEFAULT THRESHOLD (0.50) CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred_default, target_names=["Susceptible (0)", "Resistant (1)"]))
    print("\n--- TUNED THRESHOLD CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred_tuned, target_names=["Susceptible (0)", "Resistant (1)"]))

if __name__ == "__main__":
    generate_seminar_metrics()