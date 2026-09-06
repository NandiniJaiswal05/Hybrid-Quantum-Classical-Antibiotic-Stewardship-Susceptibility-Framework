import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import MODEL_DIR
from src.module2_features.state_prep import prepare_quantum_state
from src.module3_execution.quantum_circuit import execute_quantum_check, variational_circuit
from src.module3_execution.qsvm_fallback import q_kernel_matrix

@st.cache_resource
def load_models():
    scaler = joblib.load(MODEL_DIR / "standard_scaler.pkl")
    reducer = joblib.load(MODEL_DIR / "feature_reducer.pkl")

    xgb_baseline = xgb.XGBClassifier()
    xgb_baseline.load_model(MODEL_DIR / "xgb_baseline.json")

    # QNN and QSVM artifacts are OPTIONAL: whichever one exists depends on
    # which branch main.py's execute_quantum_check() took the last time the
    # training pipeline ran. Missing files here are not an error -- the app
    # falls back to XGBoost below if the artifact the router wants isn't
    # on disk yet.
    qnn_weights = None
    qnn_weights_path = MODEL_DIR / "qnn_weights.npy"
    if qnn_weights_path.exists():
        qnn_weights = np.load(qnn_weights_path)

    qsvm_bundle = None
    qsvm_path = MODEL_DIR / "qsvm_weights.pkl"
    if qsvm_path.exists():
        qsvm_bundle = joblib.load(qsvm_path)

    # REQUIRED: prepare_quantum_state() must reuse the scaler fitted during
    # training (fit=False) rather than fitting a new one on a single
    # inference row -- see state_prep.py for why. This file is saved by
    # every training run, so treat a missing one the same as a missing
    # xgb_baseline: run the training pipeline first.
    quantum_state_scaler = joblib.load(MODEL_DIR / "quantum_state_scaler.pkl")

    return scaler, reducer, xgb_baseline, qnn_weights, qsvm_bundle, quantum_state_scaler

try:
    standard_scaler, pca_reducer, inference_model, qnn_weights, qsvm_bundle, quantum_state_scaler = load_models()
    models_loaded = True
except FileNotFoundError:
    st.error("Model artifacts not found. Please run the training pipeline (Modules 1-3) first.")
    models_loaded = False

st.set_page_config(page_title="AMR-UTI Quantum Pipeline", layout="wide")
st.title("AMR-UTI Clinical Susceptibility Prediction Engine")
st.markdown("---")

if models_loaded:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Patient Intake")
        age = st.slider("Patient Age", 18, 100, 45)
        
        col_demo1, col_demo2 = st.columns(2)
        with col_demo1:
            is_white = st.selectbox("Race (White)", [0, 1])
        with col_demo2:
            is_veteran = st.selectbox("Veteran Status", [0, 1])
        
        st.markdown("**Prior 30-Day Resistance History**")
        col_30_1, col_30_2 = st.columns(2)
        with col_30_1:
            prev_res_NIT_30 = st.selectbox("NIT 30-Day", [0, 1])
            prev_res_SXT_30 = st.selectbox("SXT 30-Day", [0, 1])
        with col_30_2:
            prev_res_CIP_30 = st.selectbox("CIP 30-Day", [0, 1])
            prev_res_LVX_30 = st.selectbox("LVX 30-Day", [0, 1])

        st.markdown("**Prior 90-Day Resistance History**")
        col_90_1, col_90_2 = st.columns(2)
        with col_90_1:
            prev_res_SXT_90 = st.selectbox("SXT 90-Day", [0, 1])
        with col_90_2:
            prev_res_CIP_90 = st.selectbox("CIP 90-Day", [0, 1])
        
        # Clinician Risk Sensitivity Slider for Threshold Tuning
        st.markdown("---")
        risk_sensitivity = st.slider("Clinical Risk Threshold (Safety Bias)", 0.10, 0.50, 0.30, 0.05, 
                                     help="Lower thresholds increase sensitivity for detecting resistant infections.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("Compute Susceptibility Score", type="primary", use_container_width=True)

    with col2:
        st.subheader("Pipeline Execution Status")
        
        if predict_button:
            with st.spinner("Assembling request and routing through pipeline..."):

                feature_names = list(standard_scaler.feature_names_in_)

                # FIX: values are now assigned by COLUMN NAME, not by
                # position. The previous version wrote the 9 manually
                # collected values into columns [0:9] of whatever order
                # the scaler happened to be fitted with -- almost
                # certainly NOT age/race/veteran/resistance-history in
                # that order, silently corrupting every prediction.
                manual_inputs = {
                    'demographics - age': age,
                    'demographics - is_white': is_white,
                    'demographics - is_veteran': is_veteran,
                    'micro - prev resistance NIT 30': prev_res_NIT_30,
                    'micro - prev resistance SXT 30': prev_res_SXT_30,
                    'micro - prev resistance CIP 30': prev_res_CIP_30,
                    'micro - prev resistance LVX 30': prev_res_LVX_30,
                    'micro - prev resistance SXT 90': prev_res_SXT_90,
                    'micro - prev resistance CIP 90': prev_res_CIP_90,
                }

                missing_cols = [c for c in manual_inputs if c not in feature_names]
                if missing_cols:
                    st.error(
                        "These expected column names were not found in the fitted "
                        f"scaler: {missing_cols}. Check the column naming used in "
                        "clean_and_scale_features() -- it must match exactly."
                    )
                    st.stop()

                # All other ~780 columns the scaler expects but the UI doesn't
                # collect are left at 0 post-scaling. That's a simplification,
                # not a clinically validated imputation -- worth revisiting if
                # this UI is used for anything beyond a demo.
                padded_df = pd.DataFrame(
                    np.zeros((1, len(feature_names))), columns=feature_names
                )
                for col, val in manual_inputs.items():
                    padded_df.at[0, col] = val

                st.write("✔️ **Module 1:** Standardizing raw clinical data...")
                X_scaled = standard_scaler.transform(padded_df)

                st.write("✔️ **Module 2:** Compressing features and mapping to Quantum State |x⟩...")
                X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
                X_compressed = pca_reducer.transform(X_scaled_df)
                # FIX: fit=False reuses the scaler saved during training
                # instead of fitting a brand-new MinMaxScaler on this single
                # row, which would collapse every feature to the same
                # constant angle regardless of the patient's actual values.
                X_state = prepare_quantum_state(
                    pd.DataFrame(X_compressed), fit=False, scaler=quantum_state_scaler
                )

                # FIX: Module 3 previously never ran at all -- this line
                # was missing entirely, and the app unconditionally showed
                # "Hardware timeout simulated. Engaging Fallback 2 (XGBoost)"
                # no matter what. The router now actually decides the path,
                # exactly as it does in the training pipeline (main.py).
                st.write("✔️ **Module 3:** Checking gradient landscape and routing to execution engine...")
                execution_path = execute_quantum_check(X_state)

                if execution_path == "quantum_optimization" and qnn_weights is not None:
                    st.write("➡️ Routed to: **Variational QNN** (healthy gradient landscape).")
                    angles = X_state.values[0]
                    expectation = variational_circuit(angles, qnn_weights)
                    resistance_prob = float((1 - expectation) / 2)
                    engine_used = "Quantum QNN"

                elif execution_path == "qsvm_fallback" and qsvm_bundle is not None:
                    st.write("➡️ Routed to: **Quantum Kernel SVM** (barren plateau detected).")
                    kernel_row = q_kernel_matrix(X_state.values[:1], qsvm_bundle["anchors_X"])
                    resistance_prob = float(qsvm_bundle["model"].predict_proba(kernel_row)[0, 1])
                    engine_used = "Quantum SVM (Fallback 1)"

                else:
                    if execution_path != "classical_fallback":
                        st.warning(
                            f"Router selected '{execution_path}' but that engine's "
                            "artifact isn't on disk yet (re-run the training "
                            "pipeline down that path first) -- falling back to "
                            "the classical baseline for this prediction."
                        )
                    st.write("➡️ Routed to: **Penalized XGBoost** (classical baseline).")
                    resistance_prob = float(inference_model.predict_proba(X_compressed)[0, 1])
                    engine_used = "Classical XGBoost (Fallback 2)"

                susceptibility_score = 1.0 - resistance_prob
                
                st.markdown("---")
                st.subheader("Prediction Payload")
                st.caption(f"Engine used for this prediction: **{engine_used}**")
                
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Predicted Susceptibility Score", f"{susceptibility_score * 100:.2f}%")
                
                uncertainty = 1.0 - abs(resistance_prob - 0.5) * 2
                metric_col2.metric("Uncertainty Budget (Entropy)", f"{uncertainty:.3f}")
                
                # Apply dynamic threshold mapping based on clinician setting
                is_resistant = resistance_prob >= risk_sensitivity
                
                if is_resistant:
                    st.error(f"⚠️ Risk of Resistance detected (Resistance Prob: {resistance_prob*100:.1f}% >= Threshold {risk_sensitivity*100:.0f}%). Avoid standard empirical script.")
                else:
                    st.success(f"✅ Susceptible. Resistance probability is below safety threshold ({risk_sensitivity*100:.0f}%). Empirical prescription may proceed.")