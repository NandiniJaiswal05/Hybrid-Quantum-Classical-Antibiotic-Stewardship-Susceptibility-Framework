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

@st.cache_resource
def load_models():
    scaler = joblib.load(MODEL_DIR / "standard_scaler.pkl")
    reducer = joblib.load(MODEL_DIR / "feature_reducer.pkl")
    
    xgb_baseline = xgb.XGBClassifier()
    xgb_baseline.load_model(MODEL_DIR / "xgb_baseline.json")
    
    return scaler, reducer, xgb_baseline

try:
    standard_scaler, pca_reducer, inference_model = load_models()
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
                
                input_data = pd.DataFrame({
                    'demographics - age': [age],
                    'demographics - is_white': [is_white],
                    'demographics - is_veteran': [is_veteran],
                    'micro - prev resistance NIT 30': [prev_res_NIT_30],
                    'micro - prev resistance SXT 30': [prev_res_SXT_30],
                    'micro - prev resistance CIP 30': [prev_res_CIP_30],
                    'micro - prev resistance LVX 30': [prev_res_LVX_30],
                    'micro - prev resistance SXT 90': [prev_res_SXT_90],
                    'micro - prev resistance CIP 90': [prev_res_CIP_90],
                })
                
                expected_features = standard_scaler.n_features_in_
                padded_data = np.zeros((1, expected_features))
                padded_data[0, :9] = input_data.values 
                
                feature_names = standard_scaler.feature_names_in_
                padded_df = pd.DataFrame(padded_data, columns=feature_names)
                
                st.write("✔️ **Module 1:** Standardizing raw clinical data...")
                X_scaled = standard_scaler.transform(padded_df)
                
                st.write("✔️ **Module 2:** Compressing features and mapping to Quantum State |x⟩...")
                X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
                X_compressed = pca_reducer.transform(X_scaled_df)
                
                X_state = prepare_quantum_state(pd.DataFrame(X_compressed))
                
                st.write("✔️ **Module 3:** Hardware timeout simulated. Engaging Fallback 2 (XGBoost)...")
                
                resistance_prob = inference_model.predict_proba(X_compressed)[0, 1]
                susceptibility_score = 1.0 - resistance_prob
                
                st.markdown("---")
                st.subheader("Prediction Payload")
                
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