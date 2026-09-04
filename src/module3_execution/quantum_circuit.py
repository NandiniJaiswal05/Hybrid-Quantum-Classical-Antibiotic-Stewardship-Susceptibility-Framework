import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import logging

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
        return np.mean([variational_circuit(f, w) for f in features_batch])

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