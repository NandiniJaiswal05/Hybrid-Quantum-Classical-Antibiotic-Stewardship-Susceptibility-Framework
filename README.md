# 🧬 AMR-UTI: Hybrid Quantum-Classical Clinical Pipeline
**An Algorithmic Readiness Framework for Antimicrobial Stewardship**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum_ML-purple.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Classical_Fallback-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![AWS Braket](https://img.shields.io/badge/AWS-Braket_Ready-232F3E.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Executive Summary
The Antimicrobial Resistance Urinary Tract Infection (AMR-UTI) prediction engine is a fault-tolerant hybrid quantum-classical framework designed to bridge the 48-hour laboratory culture lag in emergency medicine. By utilizing a tri-stage cascading architecture, it processes tabular electronic health records (EHR) to provide real-time, high-confidence susceptibility predictions. The system maximizes clinical safety by optimizing minority-class recall for drug-resistant phenotypes and features a dynamic uncertainty entropy budget.

---

## 🏗️ System Architecture

The pipeline is built for **Algorithmic Readiness** in the NISQ (Noisy Intermediate-Scale Quantum) era. It routes predictions dynamically based on hardware health, falling back to penalized classical models (XGBoost) if quantum gradients collapse (Barren Plateaus) or hardware times out.

```mermaid
flowchart TD
    subgraph UI ["Clinician Frontend (Streamlit Dashboard)"]
        A[Patient Intake Inputs] --> B[Request Assembler & Payload Validator]
    end

    subgraph M1 ["Module 1: ETL & Ingestion Engine"]
        B --> C[KNN Imputation & Cleaning]
        C --> D[Z-Score Standardization]
    end

    subgraph M2 ["Module 2: Feature Compression"]
        D --> E[Non-Linear Kernel PCA Mapping]
        E --> F[8-Qubit Angle Embedding |x⟩]
    end

    subgraph M3 ["Module 3: Execution Router"]
        F --> G{Gradient Variance Check}
    end

    subgraph Paths ["Execution Routing Layer"]
        G -- "Healthy" --> H[Path A: Variational QNN]
        G -- "Flat" --> I[Path B: Quantum SVM]
        G -- "Timeout" --> J[Path C: Penalized XGBoost]
    end

    subgraph Out ["Clinical Decision Support"]
        H --> K[Output Assembler]
        I --> K
        J --> K
        K --> L[Calculate Susceptibility & Entropy Budget]
    end

    style UI fill:#2b323c,stroke:#333,stroke-width:2px,color:#fff
    style M1 fill:#1e3a5f,stroke:#333,stroke-width:2px,color:#fff
    style M2 fill:#1e5f3a,stroke:#333,stroke-width:2px,color:#fff
    style M3 fill:#5f4b1e,stroke:#333,stroke-width:2px,color:#fff
    style Paths fill:#5f1e1e,stroke:#333,stroke-width:2px,color:#fff
    style Out fill:#4b1e5f,stroke:#333,stroke-width:2px,color:#fff
