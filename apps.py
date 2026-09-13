import math
import json
import logging
from collections import defaultdict
from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.config import MODEL_DIR
from src.module2_features.reduction import parse_temporal_features
from src.module3_execution.quantum_circuit import HybridQuantumClassicalModel
from src.module3_execution.qsvm_fallback import extract_mps_embeddings, predict_qsvm_proba

logger = logging.getLogger(__name__)

COLS_TO_DROP = ['example_id', 'is_train', 'uncomplicated', 'NIT', 'SXT', 'CIP', 'LVX']
WINDOW_TOKENS = ["ALL", "180", "90", "30", "14", "7"]

def categorize(feature_names):
    categories = defaultdict(list)
    for col in feature_names:
        low = col.lower()
        if low.startswith("demographics"): categories["Demographics"].append(col)
        elif low.startswith("micro - prev resistance"): categories["Prior Antibiotic Resistance"].append(col)
        elif low.startswith("medication") or low.startswith("ab subtype") or low.startswith("ab class"): categories["Prior Antibiotic Exposure"].append(col)
        elif low.startswith("micro - prev organism"): categories["Prior Infecting Organisms"].append(col)
        elif low.startswith("comorbidity"): categories["Comorbidities (Elixhauser)"].append(col)
        elif low.startswith("hosp ward"): categories["Hospital Ward / Setting"].append(col)
        elif "colonization pressure" in low: categories["Local Resistance Rate (Colonization Pressure)"].append(col)
        elif low.startswith("custom") and "nursing home" in low: categories["Nursing Home History"].append(col)
        elif low.startswith("infection_sites"): categories["Other Concurrent Infection Sites"].append(col)
        elif low.startswith("procedure"): categories["Prior Procedures"].append(col)
        else: categories["Other Clinical Features"].append(col)
    return categories

def group_by_base(cols):
    groups = defaultdict(dict)
    for col in cols:
        tokens = col.split()
        base, window = col, None
        for i, tok in enumerate(tokens):
            if tok in WINDOW_TOKENS:
                base = " ".join(tokens[:i] + tokens[i + 1:]).strip()
                window = tok
                break
        groups[base][window] = col
    return groups

CATEGORY_ORDER = [
    "Demographics", "Prior Antibiotic Resistance", "Prior Antibiotic Exposure",
    "Comorbidities (Elixhauser)", "Prior Infecting Organisms", "Hospital Ward / Setting",
    "Local Resistance Rate (Colonization Pressure)", "Nursing Home History",
    "Prior Procedures", "Other Concurrent Infection Sites", "Other Clinical Features",
]

def _predict_hybrid(hybrid_model, X_scaled_df: pd.DataFrame) -> float:
    feature_groups, _ = parse_temporal_features(X_scaled_df)
    active_steps = getattr(hybrid_model, "active_steps", None) or [t for t in WINDOW_TOKENS if len(feature_groups[t]) > 0]
    X_seq = [torch.tensor(X_scaled_df[feature_groups[t]].values, dtype=torch.float32) for t in active_steps]
    static_names = getattr(hybrid_model, "static_feature_names", [])
    X_static = torch.tensor(X_scaled_df[static_names].values, dtype=torch.float32) if static_names else None
    with torch.no_grad():
        return hybrid_model(X_seq, X_static).item()

@st.cache_resource
def load_models():
    scaler_path = MODEL_DIR / "standard_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError("Core scaler artifact not found in models/.")
    scaler = joblib.load(scaler_path)

    xgb_path = MODEL_DIR / "xgb_baseline.json"
    inference_model = xgb.XGBClassifier()
    if xgb_path.exists():
        inference_model.load_model(xgb_path)

    hybrid_model = None
    hybrid_path = MODEL_DIR / "hybrid_mps_qnn.pt"
    meta_path = MODEL_DIR / "hybrid_mps_qnn_meta.json"
    if hybrid_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        hybrid_model = HybridQuantumClassicalModel(
            mps_input_dims=meta["input_dims"],
            static_dim=meta.get("static_dim", 0),
            n_layers=meta.get("n_layers", 3),
        )
        hybrid_model.load_state_dict(torch.load(hybrid_path, map_location=torch.device('cpu')))
        hybrid_model.eval()
        hybrid_model.static_feature_names = meta.get("static_features", [])
        hybrid_model.active_steps = meta.get("active_steps", [])

    qsvm_model, qsvm_embedder, qsvm_train_embeddings = None, None, None
    qsvm_path = MODEL_DIR / "qsvm_weights.pkl"
    if qsvm_path.exists():
        qsvm_model = joblib.load(qsvm_path)
        embedder_path = MODEL_DIR / "qsvm_mps_embedder.pt"
        embedder_meta_path = MODEL_DIR / "qsvm_mps_embedder_meta.json"
        train_embed_path = MODEL_DIR / "qsvm_train_embeddings.npy"
        if all(p.exists() for p in (embedder_path, embedder_meta_path, train_embed_path)):
            with open(embedder_meta_path) as f:
                emeta = json.load(f)
            qsvm_embedder = HybridQuantumClassicalModel(
                mps_input_dims=emeta["input_dims"],
                static_dim=emeta.get("static_dim", 0),
                n_layers=emeta.get("n_layers", 3)
            )
            qsvm_embedder.load_state_dict(torch.load(embedder_path, map_location=torch.device('cpu')))
            qsvm_embedder.eval()
            qsvm_embedder.active_steps = emeta.get("active_steps", [])
            qsvm_train_embeddings = np.load(train_embed_path)

    return scaler, inference_model, hybrid_model, qsvm_model, qsvm_embedder, qsvm_train_embeddings

try:
    standard_scaler, inference_model, hybrid_model, qsvm_model, qsvm_embedder, qsvm_train_embeddings = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error_msg = str(e)

st.set_page_config(page_title="AMR-UTI Quantum Pipeline", layout="wide")
st.title("AMR-UTI Clinical Susceptibility Prediction Engine")
st.markdown("---")

if not models_loaded:
    st.error(f"Model artifacts not found or incomplete: {load_error_msg}. Please run `main.py` first.")
else:
    feature_names = list(standard_scaler.feature_names_in_)
    ui_features = [c for c in feature_names if c not in COLS_TO_DROP]
    categories = categorize(ui_features)

    inputs = {}
    with st.form("intake_form"):
        st.subheader("Patient Intake (All Predictive Features)")
        if "Demographics" in categories:
            st.markdown("### Demographics")
            demo_cols = st.columns(min(3, len(categories["Demographics"])) or 1)
            for i, col in enumerate(sorted(categories["Demographics"])):
                target = demo_cols[i % len(demo_cols)]
                label = col.split(" - ", 1)[-1].replace("_", " ").strip().title()
                if "age" in col.lower():
                    inputs[col] = target.slider(label, 18, 100, 45, key=col)
                else:
                    inputs[col] = 1 if target.selectbox(label, ["No", "Yes"], key=col) == "Yes" else 0

        if "Hospital Ward / Setting" in categories:
            st.markdown("### Hospital Ward / Setting")
            ward_cols = sorted(categories["Hospital Ward / Setting"])
            ward_labels = [c.split(" - ", 1)[-1].upper() for c in ward_cols]
            chosen = st.multiselect("Current specimen collected in", options=ward_labels, key="ward_select")
            for c, lbl in zip(ward_cols, ward_labels):
                inputs[c] = 1 if lbl in chosen else 0

        for cat in [c for c in CATEGORY_ORDER if c in categories and c not in ("Demographics", "Hospital Ward / Setting")]:
            cols = categories[cat]
            groups = group_by_base(cols)
            with st.expander(f"{cat}  ({len(cols)} features)", expanded=False):
                for base, window_map in sorted(groups.items()):
                    label = base
                    for prefix in ("micro - prev resistance ", "micro - prev organism ", "medication - ", "comorbidity - ", "ab class ", "ab subtype ", "custom - ", "procedure - ", "infection_sites - "):
                        if label.startswith(prefix): label = label[len(prefix):]
                    label = label.strip(" -").title()
                    windows_present = [w for w in WINDOW_TOKENS if w in window_map]

                    if cat == "Local Resistance Rate (Colonization Pressure)":
                        for w in windows_present:
                            inputs[window_map[w]] = st.slider(f"{label} ({w}d)", 0.0, 1.0, 0.0, 0.01, key=window_map[w])
                    elif len(windows_present) > 1:
                        chosen_windows = st.multiselect(f"{label} -- positive in window(s)", options=windows_present, key=f"grp_{cat}_{base}")
                        for w in windows_present: inputs[window_map[w]] = 1 if w in chosen_windows else 0
                    else:
                        for w, col in window_map.items():
                            wl = f" ({w}d)" if w else ""
                            inputs[col] = 1 if st.checkbox(f"{label}{wl}", key=col) else 0

        st.markdown("---")
        risk_sensitivity = st.slider("Clinical Risk Threshold (Safety Bias)", 0.10, 0.50, 0.30, 0.05)
        predict_button = st.form_submit_button("Compute Susceptibility Score", type="primary", use_container_width=True)

    if predict_button:
        try:
            with st.spinner("Assembling request and executing via saved model weights..."):
                padded_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
                for col, val in inputs.items():
                    if col in padded_df.columns:
                        padded_df.at[0, col] = val

                X_scaled = standard_scaler.transform(padded_df)
                X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

                if hybrid_model is not None:
                    st.write("Loaded Artifact: **Hybrid MPS-QNN Model** (with static-feature fusion).")
                    raw_resistance_prob = float(_predict_hybrid(hybrid_model, X_scaled_df))
                    engine_used = "Saved Hybrid MPS-QNN"
                elif qsvm_model is not None and qsvm_embedder is not None and qsvm_train_embeddings is not None:
                    st.write("Loaded Artifact: **Quantum Kernel SVM**.")
                    X_embedded = extract_mps_embeddings(qsvm_embedder, X_scaled_df)
                    raw_resistance_prob = float(predict_qsvm_proba(qsvm_model, X_embedded, qsvm_train_embeddings)[0])
                    engine_used = "Saved Quantum SVM (Fallback 1)"
                elif inference_model is not None:
                    st.write("Loaded Artifact: **Penalized XGBoost Baseline**.")
                    raw_resistance_prob = float(inference_model.predict_proba(X_scaled_df[ui_features])[0, 1])
                    engine_used = "Saved Classical XGBoost (Fallback 2)"
                else:
                    raise RuntimeError("No valid model artifacts found.")

            # --- PROBABILITY CALIBRATION (PLATT SCALING) ---
            # Corrects SMOTE probability compression by expanding log-odds
            def calibrate_prob(p_raw, temperature=0.20):
                p_raw = max(min(p_raw, 0.9999), 0.0001)
                log_odds = math.log(p_raw / (1.0 - p_raw))
                scaled_log_odds = log_odds / temperature
                return 1.0 / (1.0 + math.exp(-scaled_log_odds))

            resistance_prob = calibrate_prob(raw_resistance_prob)
            susceptibility_score = 1.0 - resistance_prob
            
            # Shannon Entropy calculation on calibrated probability
            p = max(min(resistance_prob, 0.9999), 0.0001)
            entropy_bits = - (p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

            st.markdown("---")
            st.subheader("Prediction Payload")
            st.caption(f"Engine used for this prediction: **{engine_used}**")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Predicted Susceptibility Score", f"{susceptibility_score * 100:.2f}%")
            metric_col2.metric("Uncertainty Budget (Entropy)", f"{entropy_bits:.3f} bits")

            if entropy_bits <= 0.40:
                st.info("Model confidence is within the actionable range (entropy ≤ 0.40 bits).")
            else:
                st.warning("High uncertainty (entropy > 0.40 bits) — consider confirmatory culture testing.")

            if resistance_prob >= risk_sensitivity:
                st.error(f"Risk of Resistance detected ({resistance_prob*100:.1f}% >= Threshold {risk_sensitivity*100:.0f}%).")
            else:
                st.success(f"Susceptible. ({risk_sensitivity*100:.0f}% Threshold passed).")

        except Exception as e:
            logger.exception("Prediction failed")
            st.error(f"Prediction failed: {e}")