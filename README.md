# 🧬 Hybrid Quantum-Classical Antibiotic Stewardship Susceptibility Framework

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/Quantum-PennyLane-purple?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/Classical_ML-XGBoost-orange?style=for-the-badge)
![AWS Braket](https://img.shields.io/badge/Cloud-AWS_Braket-232F3E?style=for-the-badge&logo=amazon-aws)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)

---

## 📌 Executive Summary

The **Antimicrobial Resistance Urinary Tract Infection (AMR-UTI) Prediction Engine** is a fault-tolerant hybrid quantum-classical framework designed to bridge the **48-hour laboratory culture lag** in emergency medicine.

In modern clinical environments, empirical prescribing for UTIs can increase the risk of treatment failure and contribute to the growing problem of antimicrobial resistance.

This project introduces an **Algorithmic Readiness Framework** built around a tri-stage cascading architecture for processing tabular Electronic Health Record (EHR) data.

The framework optimizes minority-class recall for drug-resistant phenotypes using:

- **SMOTE** for class-imbalance mitigation
- **Platt Scaling** for probability calibration
- **Precision-Recall threshold tuning**
- **Matrix Product State (MPS)** temporal feature reduction
- **Variational Quantum Neural Networks (QNNs)**
- **Quantum Support Vector Machines (QSVMs)**
- **Penalized XGBoost** as a classical fallback

The resulting system provides immediate, calibrated decision support while establishing an architecture that can transition toward **NISQ-era quantum computing infrastructure**.

---

# ⚙️ System Architecture

The pipeline uses a **fault-tolerant fallback routing mechanism** designed for high-stakes clinical environments.

Depending on model health and execution conditions, inference dynamically routes between:

1. **Variational Quantum Neural Network (QNN)**
2. **Precomputed Quantum SVM (QSVM)**
3. **Penalized Classical XGBoost**

This architecture ensures that model execution can continue even when quantum optimization becomes unstable or quantum execution encounters hardware/runtime failures.

The complete system is divided into **four modular layers**.

---

## 1. 👨‍⚕️ User & Frontend Layer

### Clinician / Physician Client

The primary end-user interacts with the system through a clinical decision-support interface.

### Streamlit UI Dashboard

The dashboard provides real-time patient intake and captures:

- Demographic variables
- Local colonization pressures
- Prior antibiotic exposure
- Historical resistance information
- Chronological windows of clinical history

The collected information is passed to the backend inference pipeline for preprocessing and prediction.

---

## 2. 🔄 Backend Layer — ETL & Data Ingestion

### Data Ingestion & Leakage Prevention

Raw EHR feature and resistance-label datasets are merged using `example_id`.

Duplicate cohort-split and task-specific flags such as:

- `is_train`
- `uncomplicated`

are dynamically removed to reduce the risk of **data leakage** between training and inference pipelines.

### Sequential Imputation

Missing temporal values are resolved using **linear interpolation across sequential time steps**.

Residual missing events are assigned zero values to preserve the chronological structure of the patient's history without flattening temporal information.

### StandardScaler — Z-Score Normalization

The feature space is normalized before dimensionality reduction using:

$$
z_i = \frac{x_i - \mu}{\sigma}
$$

where:

- $x_i$ = original feature value
- $\mu$ = feature mean
- $\sigma$ = feature standard deviation
- $z_i$ = normalized feature value

---

# 🧠 3. AI/ML Processing Layer

This layer performs temporal representation learning, quantum state preparation, model execution, and automated fallback routing.

---

### 🧬 Matrix Product State (MPS) Tensor Network

Flat clinical data is converted into chronological lookback windows:

```text
ALL
180 days
90 days
30 days
14 days
7 days
````

The MPS representation iteratively contracts temporal features while maintaining a hidden memory state that is propagated through successive time windows.

This allows the system to preserve temporal information while reducing the dimensionality of the original clinical feature space.

---

### ⚛️ Quantum State Preparation

The compressed classical temporal representation is transformed into a quantum feature state vector.

Features are scaled to the physical angular range:

$$
[0,\pi]
$$

This representation enables **angle-based quantum feature embedding** within the variational quantum circuit.

---

### 🔬 Hybrid QNN & Gradient Variance Monitoring

The MPS representation is combined with a **PennyLane Variational Quantum Circuit**.

PyTorch Autograd is used to monitor gradient variance and detect potential **Barren Plateaus**.

The framework considers the quantum optimization landscape unstable when:

$$
\operatorname{Var}(\nabla_\theta) < 10^{-4}
$$

Based on the detected condition, inference follows one of three execution paths.

---

### Path A — Healthy Gradient

When the quantum circuit demonstrates stable gradients, the system routes inference through the:

**Variational Quantum Neural Network (QNN)**

The model performs joint PyTorch-PennyLane parameter optimization and can optionally fuse static clinical features directly into the quantum embedding.

---

### Path B — Barren Plateau Detected

If the gradient variance indicates a barren plateau, the framework switches to the:

**Quantum Support Vector Machine (QSVM)**

The QSVM uses a precomputed quantum kernel Gram matrix to evaluate similarity between quantum feature states without relying on unstable variational optimization.

---

### Path C — Hardware Timeout / Quantum Runtime Error

If quantum execution encounters a timeout, runtime error, or unavailable quantum backend, the system falls back to:

**Penalized XGBoost**

The classical model uses regularization through L1/L2 penalties to reduce overfitting on the high-dimensional clinical feature space.

---

# 🏥 4. Clinical Output Layer

### Susceptibility Score

The framework converts the predicted resistance probability into a susceptibility-oriented score:

$$
S(X) = 1 - P(Y=1\mid X)
$$

where:

* $P(Y=1\mid X)$ = predicted probability of antimicrobial resistance
* $S(X)$ = estimated susceptibility score

---

### 📐 Uncertainty Budget — Shannon Entropy

The predicted probabilities are calibrated using **temperature-based Platt Scaling**.

The framework then calculates **Shannon entropy in bits**:

$$
H = -\left[p\log_2(p) + (1-p)\log_2(1-p)\right]
$$

The resulting entropy represents the model's diagnostic uncertainty.

When:

$$
H > 0.40\text{ bits}
$$

the system recommends that the physician obtain a **definitive laboratory culture**.

---

### 🚨 Clinical Alert Decision

The calibrated resistance probability is compared against a dynamically tuned clinical safety threshold.

Depending on the resulting probability and uncertainty, the Streamlit interface provides the appropriate clinical warning or decision-support output.

---

# 📁 Project Directory Structure
Hybrid-Quantum-Classical-Antibiotic-Stewardship-Susceptibility-Framework/
│
├── data/
│   ├── raw/
│   │   ├── all_uti_features.csv
│   │   ├── all_uti_resist_labels.csv
│   │   ├── all_prescriptions.csv
│   │   └── data_dictionary.csv
│   │
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
│   │
│   ├── module1_etl/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── imputation.py
│   │   └── distribution.py
│   │
│   ├── module2_features/
│   │   ├── __init__.py
│   │   ├── reduction.py
│   │   └── state_prep.py
│   │
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

---

# 📊 Clinical Performance & Threshold Tuning

Raw accuracy can be misleading in medical datasets with significant class imbalance, where the majority of patients are susceptible.

Therefore, this framework prioritizes **patient safety and minority-class recall** rather than optimizing accuracy alone.

The training pipeline incorporates:

* **SMOTE** for synthetic minority oversampling
* **Precision-Recall Curve analysis**
* **Decision threshold optimization**
* **Probability calibration**
* **Minority-class recall monitoring**

## Test Cohort

| Metric                     |                                 Value |
| -------------------------- | ------------------------------------: |
| **Test Cohort**            |                22,153 patient records |
| **Default Threshold**      |                                  0.50 |
| **Optimized Threshold**    | Tuned using Precision-Recall analysis |
| **Optimization Objective** |            F1 + minority-class recall |

## Classification Performance

| Class                | Precision | Recall |  F1-Score |    Support |
| -------------------- | --------: | -----: | --------: | ---------: |
| **Susceptible (0)**  |      0.89 |   0.85 |      0.87 |     17,476 |
| **Resistant (1)**    |      0.53 |   0.61 |      0.57 |      4,677 |
| **Overall Accuracy** |         — |      — | **80.0%** | **22,153** |

## Clinical Interpretation

By shifting the decision threshold using the precision-recall curve, the framework increases minority-class (**Resistant**) recall to **61%** while maintaining an overall accuracy of approximately **80%**.

This represents a deliberate trade-off between false negatives and false positives, emphasizing the identification of potentially resistant infections where missed resistance may lead to inappropriate empirical treatment.

> **Clinical Note:** This system is intended as a research and decision-support framework and does not replace microbiological culture, antimicrobial susceptibility testing, or qualified clinical judgment.

---

# 🚀 Local Installation & Quick Start

Follow the steps below to run the hybrid quantum-classical framework locally.

## Prerequisites

Make sure the following are installed:

* **Python 3.11+**
* **Git**
* Compatible **PyTorch** installation
* **PennyLane**
* **AWS Braket SDK** if cloud quantum execution is enabled

---

## Step 1 — Clone the Repository

Open a terminal or command prompt:

```bash
git clone https://github.com/NandiniJaiswal05/Hybrid-Quantum-Classical-Antibiotic-Stewardship-Susceptibility-Framework.git
```

Navigate into the project directory:

```bash
cd Hybrid-Quantum-Classical-Antibiotic-Stewardship-Susceptibility-Framework
```

---

## Step 2 — Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Run the Application

Launch the Streamlit interface:

```bash
streamlit run apps.py
```

The application should then be available at:

```text
http://localhost:8501
```

---

## Step 5 — Run the Pipeline Directly

For command-line execution of the main pipeline:

```bash
python main.py
```

To evaluate the trained models:

```bash
python evaluate_metrics.py
```

---

# ⚛️ Quantum Execution Strategy

The framework is designed around a **fault-tolerant execution hierarchy**:

```text
                    ┌──────────────────────┐
                    │   Clinical Patient   │
                    │        Input         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   ETL & Imputation   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     MPS Reduction    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Quantum State Prep   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Gradient Variance    │
                    │       Check          │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
          Healthy Gradient  Barren       Hardware
                 │          Plateau        Error
                 ▼             │             │
          ┌────────────┐       ▼             ▼
          │ Variational│  ┌──────────┐  ┌──────────┐
          │    QNN     │  │   QSVM   │  │ XGBoost  │
          └─────┬──────┘  └────┬─────┘  └────┬─────┘
                │              │             │
                └──────────────┼─────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Probability          │
                    │ Calibration           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Entropy & Threshold  │
                    │      Analysis        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Clinical Decision    │
                    │      Support         │
                    └──────────────────────┘
```

---

# 🧪 Research & Clinical Motivation

Antimicrobial resistance presents a significant challenge for empirical treatment of urinary tract infections.

Traditional antimicrobial susceptibility testing often requires substantial laboratory processing time. During this period, clinicians may need to initiate empirical therapy based on incomplete information.

This framework investigates whether a combination of:

* Temporal EHR representations
* Tensor-network feature compression
* Quantum machine learning
* Classical machine learning
* Probability calibration
* Uncertainty quantification
* Safety-oriented threshold optimization

can provide useful **early resistance-risk estimates** while maintaining an explicit fallback mechanism.

The core research direction is therefore not simply *"quantum machine learning for healthcare"*, but rather:

> **Can a fault-tolerant hybrid quantum-classical architecture provide calibrated, uncertainty-aware antimicrobial resistance predictions from longitudinal clinical data?**

---

# 🛡️ Safety & Intended Use

This repository represents a **research prototype** for antimicrobial resistance prediction and clinical decision-support research.

It should **not** be used as an autonomous diagnostic or prescribing system.

Predictions should not replace:

* Microbiological culture
* Antimicrobial susceptibility testing
* Infectious disease expertise
* Physician judgment
* Institutional antimicrobial stewardship protocols

Any real-world clinical deployment would require extensive external validation, prospective evaluation, regulatory review, and integration with appropriate clinical governance.

---

# 📜 License

Add the project's applicable license here.

For example:

```text
MIT License
```

---

# 👥 Contributors

**Nandini Jaiswal**

**Kanak Dharamthok**

---

# ⭐ Acknowledgements

This project integrates concepts and technologies from:

* PyTorch
* PennyLane
* XGBoost
* AWS Braket
* Streamlit
* Tensor-network / Matrix Product State methods
* Antimicrobial resistance research
* Clinical machine learning and uncertainty quantification

---

<p align="center">
  <b>🧬 Hybrid Quantum-Classical AI for Antimicrobial Stewardship</b>
  <br>
  Researching the intersection of Quantum ML, Clinical AI, and Antimicrobial Resistance
</p>
```
