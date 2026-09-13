import pennylane as qml
from sklearn.svm import SVC
import numpy as np
import torch
import joblib
import json
import pandas as pd
import logging
from src.config import MODEL_DIR
from src.module2_features.reduction import parse_temporal_features

logger = logging.getLogger(__name__)

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

TIME_ORDER = ['ALL', '180', '90', '30', '14', '7']


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))


def q_kernel_matrix(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            matrix[i, j] = kernel_circuit(x1, x2)[0]
    return matrix


def extract_mps_embeddings(model, X_state: pd.DataFrame, active_steps=None) -> np.ndarray:
    """
    Builds the temporal sequence for the MPS layer.

    BUGFIX: previously this always recomputed `active_steps` from whatever
    DataFrame was passed in. At inference time that DataFrame may not
    produce the exact same active-step ordering/shape that the model was
    trained with (e.g. a single-row inference frame), which would silently
    feed a differently-shaped sequence into `model.mps` and blow up.
    Now it prefers the ordering frozen at training time -- passed in
    explicitly, or stored on the model as `model.active_steps` -- and only
    falls back to recomputing it if neither is available.
    """
    feature_groups, static_features = parse_temporal_features(X_state)

    steps = active_steps if active_steps is not None else getattr(model, "active_steps", None)
    if not steps:
        steps = [t for t in TIME_ORDER if len(feature_groups[t]) > 0]

    missing_groups = [t for t in steps if t not in feature_groups or len(feature_groups[t]) == 0]
    if missing_groups:
        raise ValueError(
            f"Feature grouping mismatch: expected temporal group(s) {missing_groups} "
            "based on the saved model, but they are not present for these inputs. "
            "The current feature set may not match what the model was trained on."
        )

    X_seq = [torch.tensor(X_state[feature_groups[t]].values, dtype=torch.float32) for t in steps]

    static_dim = getattr(model, "static_dim", 0)
    X_static = None
    if static_dim > 0 and static_features:
        X_static = torch.tensor(X_state[static_features].values, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        compressed = model.mps(X_seq)
        if static_dim > 0:
            if X_static is None:
                X_static = torch.zeros(compressed.shape[0], static_dim, dtype=compressed.dtype)
            if X_static.shape[1] != static_dim:
                raise ValueError("Static feature count mismatch.")
            compressed = model.static_fusion(torch.cat([compressed, X_static], dim=-1))

        angles = model.scaler(compressed) * torch.pi

    return angles.numpy()


def predict_qsvm_proba(qsvm_model, X_test_embedded: np.ndarray, X_train_embedded: np.ndarray) -> np.ndarray:
    """
    BUGFIX (root cause of the Streamlit crash): `qsvm_model` was fit with
    kernel="precomputed", which means it can *only* be scored by handing it
    a kernel (Gram) matrix computed against the training embeddings -- never
    raw embeddings directly (that raises a shape-mismatch error inside
    sklearn). Centralizing the correct call here so training, offline
    evaluation, and the Streamlit app all score it identically.
    """
    kernel_test = q_kernel_matrix(X_test_embedded, X_train_embedded)
    return qsvm_model.predict_proba(kernel_test)[:, 1]


def train_and_predict_qsvm(hybrid_model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
    logger.info("Extracting chronological + static embeddings via MPS for QSVM Fallback...")

    feature_groups, static_features = parse_temporal_features(X_train)
    active_steps = [t for t in TIME_ORDER if len(feature_groups[t]) > 0]
    hybrid_model.active_steps = active_steps  # freeze ordering for reuse at inference time

    X_train_embedded = extract_mps_embeddings(hybrid_model, X_train, active_steps=active_steps)
    X_test_embedded = extract_mps_embeddings(hybrid_model, X_test, active_steps=active_steps)

    logger.info("Initializing Support Vector Machine with Quantum Kernel...")
    qsvm = SVC(kernel="precomputed", probability=True)

    kernel_train = q_kernel_matrix(X_train_embedded, X_train_embedded)
    qsvm.fit(kernel_train, y_train.values)

    joblib.dump(qsvm, MODEL_DIR / "qsvm_weights.pkl")
    np.save(MODEL_DIR / "qsvm_train_embeddings.npy", X_train_embedded)

    # Persist specific embedder weights
    torch.save(hybrid_model.state_dict(), MODEL_DIR / "qsvm_mps_embedder.pt")

    # Persist embedder shape metadata -- now including active_steps (BUGFIX: this
    # was previously omitted here, even though quantum_circuit.py saves it for the
    # hybrid model. Without it, inference had no reliable way to reconstruct the
    # exact sequence ordering used at training time.)
    meta = {
        "input_dims": getattr(hybrid_model, "mps_input_dims", []),
        "static_dim": getattr(hybrid_model, "static_dim", 0),
        "n_layers": getattr(hybrid_model, "n_layers", 3),
        "static_features": static_features,
        "active_steps": active_steps,
    }
    with open(MODEL_DIR / "qsvm_mps_embedder_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("QSVM successfully fitted and saved, with embedder persisted for exact reconstruction.")

    return predict_qsvm_proba(qsvm, X_test_embedded, X_train_embedded)