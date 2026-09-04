# 🧬 Hybrid Quantum-Classical Antibiotic Stewardship Susceptibility Framework

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PennyLane](https://img.shields.io/badge/Quantum-PennyLane-purple?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/Classical_ML-XGBoost-orange?style=for-the-badge)
![AWS Braket](https://img.shields.io/badge/Cloud-AWS_Braket-232F3E?style=for-the-badge&logo=amazon-aws)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn)

## 📌 Executive Summary
The **Antimicrobial Resistance Urinary Tract Infection (AMR-UTI) Prediction Engine** is a fault-tolerant hybrid quantum-classical framework designed to bridge the 48-hour laboratory culture lag in emergency medicine. 

In modern clinical environments, "empirical prescribing" for UTIs often risks treatment failure and exacerbates global antimicrobial resistance. This project introduces an **Algorithmic Readiness Framework** utilizing a tri-stage cascading architecture to process tabular electronic health records (EHR). By optimizing minority-class recall for drug-resistant phenotypes through SMOTE and threshold tuning, the system provides immediate, high-fidelity decision support while preparing healthcare infrastructure for the NISQ (Noisy Intermediate-Scale Quantum) era.

---

## ⚙️ System Architecture

The pipeline operates on a robust fallback routing mechanism designed for high-stakes clinical environments. It dynamically routes execution between a Variational Quantum Neural Network (QNN), a pre-computed Quantum SVM, or a heavily penalized classical XGBoost fallback, ensuring zero downtime.

As detailed in the system schematic, the architecture is divided into five distinct operational layers:

### 1. User & Frontend Layer
*   **Clinician / Physician Client:** The primary end-user interface.
*   **Streamlit UI Dashboard:** Handles real-time patient intake, capturing demographic variables (Age, Race, Veteran Status) and critical 30/90-day prior antibiotic exposure history. It retrieves necessary scalers and base models from the **Model Storage Layer**.

### 2. Backend Layer (ETL & Ingestion)
*   **Request Assembler & Validator:** Parses raw UI inputs into a structured clinical payload.
*   **KNN Imputation:** Automatically estimates and resolves missing clinical values using nearest-neighbor statistical proximity.
*   **StandardScaler (Z-Score):** Normalizes the dataset to ensure uniform variance ($z_i = \frac{x_i - \mu}{\sigma}$) prior to dimensionality reduction.

### 3. AI/ML Processing Layer
*   **Kernel PCA (8 Dimensions):** Compresses high-dimensional clinical permutations down to an 8-dimensional latent space using a Gaussian RBF kernel ($k(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$).
*   **PennyLane Angle Embedding:** Maps the classical 8-dimensional vector into physical quantum states via Y-axis rotations ($|x\rangle = \bigotimes_{i=1}^{8} R_y(x_i) |0\rangle$).
*   **Gradient Variance Check:** Acts as the primary router by measuring the gradient tensor for Barren Plateaus ($\text{Var}(\nabla_\theta C) \le 10^{-4}$).
    *   **Path A (Healthy Gradient):** Routes to the **Variational QNN**, communicating with **External Cloud Services** (AWS Braket SV1 Simulator / real QPUs).
    *   **Path B (Flat Landscape):** Routes to the **Quantum SVM** fallback.
    *   **Path C (Timeout / Error):** Routes to the **Penalized XGBoost** classical baseline engine.

### 4. Clinical Output Layer
*   **Susceptibility Score Calculation:** Inverts the target probability ($S(X) = 1 - P(Y=1|X)$) to provide a clinical baseline.
*   **Uncertainty Budget (Entropy):** Calculates the model's confidence ($U(X) = 1 - 2|P - 0.5|$). Values near 1.0 indicate high entropy, prompting the physician to order definitive lab cultures.
*   **Clinical Alert Decision:** Applies a dynamically tuned threshold (e.g., 0.5948) to trigger the final safety warnings, feeding the result directly back to the Frontend UI.

---

## 📊 Clinical Performance & Threshold Tuning

Raw accuracy is deeply deceptive in medical datasets due to severe class imbalance (the majority of patients are susceptible). This framework prioritizes **Patient Safety (Recall)** over pure accuracy by implementing Precision-Recall Curve threshold tuning and SMOTE (Synthetic Minority Over-sampling Technique) during training.

*   **Test Cohort (N):** 22,153 patient records
*   **ROC-AUC Score:** 0.7865
*   **Optimal Decision Threshold:** 0.5948 (Tuned from the default 0.50 baseline)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Susceptible (0)** | 0.89 | 0.82 | 0.85 | 17,476 |
| **Resistant (1)** | 0.47 | 0.60 | 0.53 | 4,677 |
| **Overall Accuracy** | - | - | **78.0%** | 22,153 |

*Clinical Note: By shifting the decision threshold based on the precision-recall curve, the system effectively doubled the minority-class (Resistant) recall. We intentionally trade a marginal percentage of overall accuracy to prevent life-threatening false negatives in emergency settings.*

---

## 💻 Installation & Quick Start

**Prerequisites:** Python 3.11+ is strictly recommended to ensure PennyLane and AWS Braket SDK compatibility.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/NandiniJaiswal05/Hybrid-Quantum-Classical-Antibiotic-Stewardship-Susceptibility-Framework.git](https://github.com/NandiniJaiswal05/Hybrid-Quantum-Classical-Antibiotic-Stewardship-Susceptibility-Framework.git)
   cd Hybrid-Quantum-Classical-Antibiotic-Stewardship-Susceptibility-Framework
