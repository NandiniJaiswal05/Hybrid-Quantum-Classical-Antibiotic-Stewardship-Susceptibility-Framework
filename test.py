"""
Smoke-test case: replays one documented patient scenario directly through
the saved model artifacts (scaler + hybrid MPS-QNN / QSVM / XGBoost,
whichever is present), bypassing the Streamlit form entirely.

Run from the project root (same level as app/ and models/):
    python test_case_cip_resistance.py

This does NOT invent column names -- it only sets values for columns that
actually exist in the saved scaler. Anything not listed in CASE_FEATURES
stays at the pipeline's default of 0 ("baseline / no event"), same as an
untouched field in the UI form.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))
MODEL_DIR = BASE_DIR / "models"

from module2_features.reduction import parse_temporal_features
from module3_execution.quantum_circuit import HybridQuantumClassicalModel

TIME_ORDER = ['ALL', '180', '90', '30', '14', '7']

# ---------------------------------------------------------------------------
# THE CASE: 74yo white female veteran, inpatient ward, recent CIP-resistant
# culture (90d), recent SXT exposure (30d), diabetes (90d & 180d), SNF stay
# (30d), elevated local CIP colonization pressure (0.42), no procedures, no
# other infection sites. Any column below that doesn't exist in your
# trained scaler is skipped with a printed warning rather than failing --
# your live feature set may not include every category (e.g. if procedures
# or colonization pressure weren't part of your training data).
# ---------------------------------------------------------------------------
CASE_FEATURES = {
    "demographics - age": 74,
    "demographics - is_white": 1,
    "demographics - is_veteran": 1,

    "hosp ward - IP": 1,

    "micro - prev resistance CIP 90": 1,

    "medication 30 - SXT": 1,

    "comorbidity 90 - diabetes uncomplicated": 1,
    "comorbidity 180 - diabetes uncomplicated": 1,

    "selected micro - colonization pressure CIP 90 - granular": 0.42,

    "custom 30 - nursing home": 1,
}

EXPECTED_DIRECTION = "resistant (low susceptibility score, likely above the 0.30 risk threshold)"


def build_padded_row(feature_names):
    padded = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    applied, skipped = [], []
    for col, val in CASE_FEATURES.items():
        if col in feature_names:
            padded.at[0, col] = val
            applied.append(col)
        else:
            skipped.append(col)
    return padded, applied, skipped


def main():
    scaler_path = MODEL_DIR / "standard_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"{scaler_path} not found -- run main.py first.")
    scaler = joblib.load(scaler_path)
    feature_names = list(scaler.feature_names_in_)

    padded_df, applied, skipped = build_padded_row(feature_names)

    print(f"Case features matched against trained scaler: {len(applied)}/{len(CASE_FEATURES)}")
    for c in applied:
        print(f"  [applied] {c} = {CASE_FEATURES[c]}")
    for c in skipped:
        print(f"  [skipped -- not in trained feature set] {c}")

    X_scaled = pd.DataFrame(scaler.transform(padded_df), columns=feature_names)

    hybrid_path = MODEL_DIR / "hybrid_mps_qnn.pt"
    meta_path = MODEL_DIR / "hybrid_mps_qnn_meta.json"
    xgb_path = MODEL_DIR / "xgb_baseline.json"

    if hybrid_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        model = HybridQuantumClassicalModel(
            mps_input_dims=meta["input_dims"],
            static_dim=meta.get("static_dim", 0),
            n_layers=meta.get("n_layers", 3),
        )
        model.load_state_dict(torch.load(hybrid_path, map_location=torch.device('cpu')))
        model.eval()

        feature_groups, _ = parse_temporal_features(X_scaled)
        active_steps = [t for t in TIME_ORDER if len(feature_groups[t]) > 0]
        X_seq = [torch.tensor(X_scaled[feature_groups[t]].values, dtype=torch.float32) for t in active_steps]

        static_names = meta.get("static_features", [])
        X_static = torch.tensor(X_scaled[static_names].values, dtype=torch.float32) if static_names else None

        with torch.no_grad():
            resistance_prob = float(model(X_seq, X_static).item())
        engine = "Hybrid MPS-QNN"

    elif xgb_path.exists():
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(xgb_path)
        resistance_prob = float(model.predict_proba(X_scaled)[0, 1])
        engine = "XGBoost baseline"

    else:
        raise FileNotFoundError("No trained artifact found in models/ -- run main.py first.")

    print("\n" + "=" * 60)
    print(f"Engine used:              {engine}")
    print(f"Predicted resistance prob: {resistance_prob:.4f}")
    print(f"Predicted susceptibility:  {1 - resistance_prob:.4f}")
    print(f"Expected direction:        {EXPECTED_DIRECTION}")
    print("=" * 60)


if __name__ == "__main__":
    main()