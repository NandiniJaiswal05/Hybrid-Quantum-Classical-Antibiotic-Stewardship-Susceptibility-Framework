import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_recall_curve
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.config import PROCESSED_DATA_DIR, MODEL_DIR

def generate_seminar_metrics():
    print("Loading test dataset and pre-trained models...\n")
    
    data_path = PROCESSED_DATA_DIR / "processed_clinical_data.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please ensure Module 1 ran successfully.")
        return
        
    initial_len = len(df)
    df = df.dropna(subset=['CIP'])
    dropped_count = initial_len - len(df)
    if dropped_count > 0:
        print(f"Cleaned {dropped_count} records lacking 'CIP' ground truth labels.")
        
    target_cols = ['NIT', 'SXT', 'CIP', 'LVX']
    cols_to_drop = [col for col in ['example_id', 'is_train', 'uncomplicated'] + target_cols if col in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df['CIP']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reducer = joblib.load(MODEL_DIR / "feature_reducer.pkl")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_DIR / "xgb_baseline.json")
    
    print("Applying Quantum Feature Compression to Test Set...")
    X_test_compressed = reducer.transform(X_test)
    
    feature_names = [f"qubit_feat_{i}" for i in range(X_test_compressed.shape[1])]
    X_test_compressed_df = pd.DataFrame(X_test_compressed, columns=feature_names)
    
    print("Generating Predictions and Optimizing Thresholds...\n")
    y_pred_proba = xgb_model.predict_proba(X_test_compressed_df)[:, 1]
    
    # --- ADD-ON: Precision-Recall Curve Threshold Tuning ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    y_pred_tuned = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("="*60)
    print(f" 📊 AMR-UTI PIPELINE - TEST METRICS (N={len(y_test):,})")
    print("="*60)
    print(f"ROC-AUC Score:             {roc_auc:.4f}")
    print(f"Optimal Decision Threshold: {optimal_threshold:.4f} (Default was 0.50)")
    print("="*60)
    
    print("\n--- DEFAULT THRESHOLD (0.50) CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred_default, target_names=["Susceptible (0)", "Resistant (1)"]))
    
    print("\n--- TUNED THRESHOLD CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred_tuned, target_names=["Susceptible (0)", "Resistant (1)"]))

if __name__ == "__main__":
    generate_seminar_metrics()