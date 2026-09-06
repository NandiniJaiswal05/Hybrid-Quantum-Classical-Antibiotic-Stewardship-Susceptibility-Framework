import pennylane as qml
from pennylane import numpy as np
import numpy as onp  # plain numpy, used only for artifact saving below
import pandas as pd
import logging
from src.config import MODEL_DIR

logger = logging.getLogger(__name__)

# Initialize a PennyLane state simulator with 8 qubits
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def variational_circuit(features, weights):
    """
    Quantum circuit applying AngleEmbedding for features |x⟩
    and strongly entangling layers for parameter optimization θ.
    """
    qml.AngleEmbedding(features, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

def compute_gradient_variance(features_batch: np.ndarray, weights: np.ndarray) -> float:
    """
    Computes the gradient of the variational circuit with explicit trainable weights
    to detect barren plateaus accurately.
    """
    # Ensure weights are tracked as differentiable parameters
    weights_trainable = np.array(weights, requires_grad=True)

    def cost_function(w):
        # FIX: stacking into a single array before np.mean() is required --
        # calling np.mean() directly on a Python list of autograd-traced
        # QNode outputs raises inside numpy's internal dtype-casting step
        # on current PennyLane/autograd versions. When that happens here,
        # the exception propagates up and execute_quantum_check()'s
        # try/except silently converts it into "classical_fallback",
        # regardless of whether the gradient landscape is actually healthy.
        expectations = np.stack([variational_circuit(f, w) for f in features_batch])
        return np.mean(expectations)

    gradient_fn = qml.grad(cost_function)
    gradients = gradient_fn(weights_trainable)
    
    # Flatten gradient structure and compute variance safely
    grad_array = np.hstack([np.ravel(g) for g in gradients]) if isinstance(gradients, tuple) else np.ravel(gradients)
    grad_variance = np.var(grad_array)
    
    return float(grad_variance)

def execute_quantum_check(X_state: pd.DataFrame, n_layers: int = 3, bp_threshold: float = 1e-4) -> str:
    """
    Executes the hardware check and routes the pipeline based on gradient variance Var[∇C] < 10^-4.
    """
    logger.info("Initializing Quantum Circuit Executor (PennyLane)...")
    
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    initial_weights = np.random.random(shape, requires_grad=True) * 2 * np.pi
    
    features_batch = X_state.values[:10]
    
    try:
        logger.info("Calculating gradient variance Var[∇C]...")
        grad_variance = compute_gradient_variance(features_batch, initial_weights)
        
        logger.info(f"Gradient variance computed: {grad_variance:.6e}")
        
        if np.isnan(grad_variance) or grad_variance < bp_threshold:
            logger.warning(f"Barren Plateau Detected or Invalid Variance. Routing to Fallback 1 (QSVM).")
            return "qsvm_fallback"
        else:
            logger.info("Normal Gradient detected. Proceeding with Parameter Optimization (θ).")
            return "quantum_optimization"
            
    except Exception as e:
        logger.error(f"Simulator error or timeout encountered: {e}")
        logger.info("Routing to Fallback 2 (Classical XGBoost).")
        return "classical_fallback"


def execute_quantum_optimization(
    X_state: pd.DataFrame,
    y_train: pd.Series,
    n_layers: int = 3,
    epochs: int = 20,
    lr: float = 0.1,
    max_rows: int = 200,  # <-- ADDED CAP
) -> np.ndarray:
    """
    Runs the QNN training + inference for the "Normal Gradient" path.
    Uses `max_rows` to subsample the training dataset, preventing 
    out-of-memory (OOM) errors caused by Autograd computation graphs.
    """
    logger.info("Initializing QNN parameter optimization (θ)...")

    # Subsample to cap memory usage during the Autograd backward pass
    if len(X_state) > max_rows:
        logger.info(f"Capping training data from {len(X_state)} to {max_rows} rows to avoid OOM.")
        # Data is already shuffled by train_test_split in main.py, so a simple slice is safe
        X_train_sub = X_state.values[:max_rows] if hasattr(X_state, "values") else np.array(X_state)[:max_rows]
        y_train_sub = y_train.values[:max_rows] if hasattr(y_train, "values") else np.array(y_train)[:max_rows]
    else:
        X_train_sub = X_state.values if hasattr(X_state, "values") else np.array(X_state)
        y_train_sub = y_train.values if hasattr(y_train, "values") else np.array(y_train)

    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    weights = np.random.random(shape, requires_grad=True) * 2 * np.pi

    opt = qml.AdamOptimizer(stepsize=lr)
    eps = 1e-7

    def cost(w):
        # The backward pass runs ONLY on the capped subset
        expectations = np.stack([variational_circuit(f, w) for f in X_train_sub])
        probs = np.clip((1 - expectations) / 2, eps, 1 - eps)
        return -np.mean(y_train_sub * np.log(probs) + (1 - y_train_sub) * np.log(1 - probs))

    for epoch in range(epochs):
        weights, loss = opt.step_and_cost(cost, weights)
        logger.info(f"  QNN epoch {epoch + 1}/{epochs} - loss={float(loss):.5f}")

    # Inference step runs on the FULL dataset (No gradients tracked = no memory blowup)
    X_full = X_state.values if hasattr(X_state, "values") else np.array(X_state)
    final_expectations = np.array([variational_circuit(f, weights) for f in X_full])
    final_probs = (1 - final_expectations) / 2

    # Save the trained weights artifact
    model_path = MODEL_DIR / "qnn_weights.npy"
    onp.save(model_path, onp.array(weights))
    logger.info(f"QNN weights saved to {model_path}")

    return final_probs