import json
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
from src.config import MODEL_DIR
from src.module2_features.reduction import MatrixProductStateLayer, parse_temporal_features

logger = logging.getLogger(__name__)

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

TIME_ORDER = ['ALL', '180', '90', '30', '14', '7']


@qml.qnode(dev, interface="torch")
def variational_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))


class HybridQuantumClassicalModel(nn.Module):
    def __init__(self, mps_input_dims, static_dim: int = 0, n_layers: int = 3):
        super().__init__()
        self.mps_input_dims = mps_input_dims
        self.static_dim = static_dim
        self.n_layers = n_layers
        
        self.mps = MatrixProductStateLayer(input_dims=mps_input_dims, output_dim=n_qubits)

        if static_dim > 0:
            self.static_fusion = nn.Sequential(
                nn.Linear(n_qubits + static_dim, 16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, n_qubits),
            )

        self.scaler = nn.Sigmoid()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnn = qml.qnn.TorchLayer(variational_circuit, weight_shapes)

    def forward(self, x_seq, x_static: torch.Tensor = None):
        compressed_state = self.mps(x_seq)

        if self.static_dim > 0:
            if x_static is None:
                x_static = torch.zeros(
                    compressed_state.shape[0], self.static_dim, dtype=compressed_state.dtype
                )
            fused = torch.cat([compressed_state, x_static], dim=-1)
            compressed_state = self.static_fusion(fused)

        scaled_angles = self.scaler(compressed_state) * torch.pi
        expectation = self.qnn(scaled_angles)
        return (1 - expectation) / 2


def _build_sequences(X_state: pd.DataFrame):
    feature_groups, static_features = parse_temporal_features(X_state)
    active_steps = [t for t in TIME_ORDER if len(feature_groups[t]) > 0]
    input_dims = [len(feature_groups[t]) for t in active_steps]

    X_seq = [
        torch.tensor(X_state[feature_groups[t]].values, dtype=torch.float32)
        for t in active_steps
    ]

    static_dim = len(static_features)
    X_static = None
    if static_dim > 0:
        X_static = torch.tensor(X_state[static_features].values, dtype=torch.float32)

    return X_seq, X_static, static_features, active_steps, input_dims


def compute_gradient_variance_pytorch(
    model: nn.Module, X_batch_seq: list, X_batch_static: torch.Tensor = None
) -> float:
    model.zero_grad()
    predictions = model(X_batch_seq, X_batch_static)
    loss = predictions.mean()
    loss.backward()

    qnn_grads = model.qnn.weights.grad
    if qnn_grads is None:
        return 0.0

    return torch.var(qnn_grads).item()


def execute_quantum_check(model: HybridQuantumClassicalModel, X_state: pd.DataFrame, bp_threshold: float = 1e-4) -> str:
    """
    Executes the hardware check using the passed unified model.
    Returns the execution path string.
    """
    logger.info("Initializing Quantum Circuit Executor (PyTorch Autograd)...")

    try:
        features_batch = X_state.head(10)
        X_seq_batch, X_static_batch, static_features, _, _ = _build_sequences(features_batch)

        if static_features:
            logger.info(f"Fusing {len(static_features)} static feature(s) into the quantum embedding...")

        logger.info("Calculating gradient variance Var[∇θ]...")
        grad_variance = compute_gradient_variance_pytorch(model, X_seq_batch, X_static_batch)
        logger.info(f"Gradient variance computed: {grad_variance:.6e}")

        if np.isnan(grad_variance) or grad_variance < bp_threshold:
            logger.warning("Barren Plateau Detected. Routing to Fallback 1 (QSVM).")
            return "qsvm_fallback"
        else:
            logger.info("Normal Gradient detected. Proceeding with Joint MPS-QNN Parameter Optimization (θ).")
            return "quantum_optimization"

    except Exception as e:
        logger.error(f"Simulator error or timeout encountered: {e}")
        logger.info("Routing to Fallback 2 (Classical XGBoost).")
        return "classical_fallback"


def execute_quantum_optimization(
    model: HybridQuantumClassicalModel,
    X_state: pd.DataFrame,
    y_train: pd.Series,
    epochs: int = 20,
    lr: float = 0.01
) -> np.ndarray:
    """
    Runs training using the exact model instance previously validated.
    """
    logger.info("Initializing Joint PyTorch-PennyLane End-to-End Training...")

    X_seq_tensors, X_static_tensor, static_features, active_steps, input_dims = _build_sequences(X_state)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    static_dim = len(static_features)
    if static_dim > 0:
        logger.info(f"Training with {static_dim} static feature(s) fused.")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_seq_tensors, X_static_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        logger.info(f"  Hybrid Joint Epoch {epoch + 1}/{epochs} - loss={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        final_probs = model(X_seq_tensors, X_static_tensor).numpy().flatten()

    model_path = MODEL_DIR / "hybrid_mps_qnn.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Joint Hybrid model architecture saved to {model_path}")

    meta = {
        "active_steps": active_steps,
        "input_dims": input_dims,
        "static_features": static_features,
        "static_dim": static_dim,
        "n_layers": model.n_layers,
    }
    meta_path = MODEL_DIR / "hybrid_mps_qnn_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved hybrid model architecture metadata to {meta_path}")

    return final_probs