# 🧬 Hybrid Quantum-Classical Antibiotic Stewardship Susceptibility Framework

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/Quantum-PennyLane-purple?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/Classical_ML-XGBoost-orange?style=for-the-badge)
![AWS Braket](https://img.shields.io/badge/Cloud-AWS_Braket-232F3E?style=for-the-badge&logo=amazon-aws)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)

## 📌 Executive Summary
The **Antimicrobial Resistance Urinary Tract Infection (AMR-UTI) Prediction Engine** is a fault-tolerant hybrid quantum-classical framework designed to bridge the 48-hour laboratory culture lag in emergency medicine. 

In modern clinical environments, "empirical prescribing" for UTIs often risks treatment failure and exacerbates global antimicrobial resistance. This project introduces an **Algorithmic Readiness Framework** utilizing a tri-stage cascading architecture to process tabular electronic health records (EHR). By optimizing minority-class recall for drug-resistant phenotypes through SMOTE, Platt Scaling calibration, and threshold tuning, the system provides immediate, high-fidelity decision support while preparing healthcare infrastructure for the NISQ (Noisy Intermediate-Scale Quantum) era.

---

## ⚙️ System Architecture

The pipeline operates on a robust fallback routing mechanism designed for high-stakes clinical environments. It dynamically routes execution between a Variational Quantum Neural Network (QNN), a precomputed Quantum SVM, or a penalized classical XGBoost fallback, ensuring zero downtime.

The architecture is divided into five distinct modular layers:

### 1. User & Frontend Layer
*   **Clinician / Physician Client:** The primary end-user interface.
*   **Streamlit UI Dashboard:** Handles real-time patient intake, capturing demographic variables, local colonization pressures, and distinct chronological windows for prior antibiotic exposure and resistance history.

### 2. Backend Layer (ETL & Ingestion)
*   **Data Ingestion & Leakage Prevention:** Merges raw EHR feature and label datasets via `example_id`, dynamically stripping duplicate cohort split flags (`is_train`, `uncomplicated`) to prevent data leakage[cite: 30].
*   **Sequential Imputation:** Resolves temporal gaps via linear interpolation across sequential time steps, assigning residual zeros for missing events to preserve chronological history without flattening the temporal context[cite: 29].
*   **StandardScaler (Z-Score):** Normalizes the feature set to ensure uniform variance ($z_i = \frac{x_i - \mu}{\sigma}$) prior to feature reduction[cite: 29].

### 3. AI/ML Processing Layer
*   **Matrix Product State (MPS) Tensor Network:** Tokenizes and slices flat clinical data into chronological lookback windows (`ALL`, `180`, `90`, `30`, `14`, `7`), iteratively contracting temporal features to maintain a hidden memory state passed forward through time[cite: 27].
*   **Quantum State Preparation:** Maps the compressed classical temporal features into a Quantum Feature State Vector, scaling data strictly between $0$ and $\pi$ for physical angle embedding[cite: 26].
*   **Hybrid QNN & Gradient Variance Check:** Combines the MPS layer with a PennyLane variational quantum circuit, utilizing PyTorch Autograd to measure gradient variance and automatically detect Barren Plateaus ($\text{Var}(\nabla_\theta) < 10^{-4}$)[cite: 25].
    *   **Path A (Healthy Gradient):** Routes to the **Variational QNN** for joint PyTorch-PennyLane parameter optimization, optionally fusing static features directly into the embedding[cite: 25].
    *   **Path B (Barren Plateau Detected):** Routes to the **Quantum SVM** fallback, utilizing a precomputed quantum kernel Gram matrix to evaluate similarity between quantum states[cite: 24].
    *   **Path C (Hardware Timeout / Error):** Routes to the **Penalized XGBoost** classical baseline engine, heavily regularized (L1/L2) to prevent overfitting on the high-dimensional clinical dataset[cite: 23].

### 4. Clinical Output Layer
*   **Susceptibility Score Calculation:** Inverts the target probability ($S(X) = 1.0 - P(Y=1|X)$) to provide a clinical baseline metric.
*   **Uncertainty Budget (True Shannon Entropy):** Applies temperature-based Platt Scaling to calibrate SMOTE-compressed probability distributions, calculating the True Shannon Entropy in bits ($H = - (p \log_2 p + (1 - p) \log_2 (1 - p))$) to quantify diagnostic uncertainty strictly. Values above 0.40 bits prompt the physician to order definitive lab cultures.
*   **Clinical Alert Decision:** Compares the calibrated resistance probability against a dynamically tuned clinical safety threshold to trigger final UI warnings.

---
## 📁 Project Directory Structure

Hybrid-Quantum-Classical-Antibiotic-Stewardship-Susceptibility-Framework/
│
├── data/
│   ├── raw/
│   │   ├── all_uti_features.csv
│   │   ├── all_uti_resist_labels.csv
│   │   ├── all_prescriptions.csv
│   │   └── data_dictionary.csv
│   └── processed/
│       └── processed_clinical_data.csv
│
├── models/
│   ├── standard_scaler.pkl
│   ├── quantum_state_scaler.pkl
│   ├── xgb_baseline.json
│   ├── hybrid_mps_qnn.pt
│   ├── hybrid_mps_qnn_meta.json
│   ├── qsvm_weights.pkl
│   ├── qsvm_train_embeddings.npy
│   ├── qsvm_mps_embedder.pt
│   └── qsvm_mps_embedder_meta.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── module1_etl/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── imputation.py
│   │   └── distribution.py
│   ├── module2_features/
│   │   ├── __init__.py
│   │   ├── reduction.py
│   │   └── state_prep.py
│   └── module3_execution/
│       ├── __init__.py
│       ├── quantum_circuit.py
│       ├── qsvm_fallback.py
│       └── xgboost_engine.py
│
├── apps.py
├── main.py
├── evaluate_metrics.py
├── requirements.txt
└── README.md


## 📊 Clinical Performance & Threshold Tuning

Raw accuracy is deeply deceptive in medical datasets due to severe class imbalance (the majority of patients are susceptible). This framework prioritizes **Patient Safety (Recall)** over pure accuracy by implementing Precision-Recall Curve threshold tuning and SMOTE (Synthetic Minority Over-sampling Technique) during training.

*   **Test Cohort (N):** 22,153 patient records
*   **Optimal Decision Threshold:** Tuned from the default 0.50 baseline to maximize F1 and minority recall.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Susceptible (0)** | 0.89 | 0.85 | 0.87 | 17,476 |
| **Resistant (1)** | 0.53 | 0.61 | 0.57 | 4,677 |
| **Overall Accuracy** | - | - | **80.0%** | 22,153 |

*Clinical Note: By shifting the decision threshold based on the precision-recall curve, the system effectively elevates minority-class (Resistant) recall to 61% while maintaining an 80% overall accuracy. This calibrated trade-off prevents life-threatening false negatives in emergency settings while keeping the false-positive rate manageable.*

---

## 🚀 Local Server Installation & Quick Start

Follow these steps to run the complete hybrid quantum-classical framework on your local server.

### Prerequisites
*   **Python 3.11+** (Strictly recommended for optimal PennyLane, PyTorch, and AWS Braket SDK compatibility).
*   Git installed on your machine.

### Step 1: Clone the Repository
Open your terminal or command prompt and clone the repository:
```bash
git clone [https://github.com/NandiniJaiswal05/Hybrid-Quantum-Classical-Antibiotic-Stewardship-Susceptibility-Framework.git](https://github.com/NandiniJaiswal05/Hybrid-Quantum-Classical-Antibiotic-Stewardship-Susceptibility-Framework.git)
cd Hybrid-Quantum-Classical-Antibiotic-Stewardship-Susceptibility-Framework
