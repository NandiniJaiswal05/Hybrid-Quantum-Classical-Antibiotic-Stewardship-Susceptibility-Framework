import pennylane as qml
from sklearn.svm import SVC
import numpy as np
import joblib
import logging
from src.config import MODEL_DIR

logger = logging.getLogger(__name__)

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    """
    Computes the quantum overlap between two data points.
    Applies the adjoint of the embedding to measure the distance in Hilbert space.
    """
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    
    # Return the probability of measuring the |0...0⟩ state
    return qml.probs(wires=range(n_qubits))

def q_kernel_matrix(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Constructs the kernel matrix required for the classical SVM using the quantum circuit.
    """
    logger.info(f"Computing Quantum Kernel Matrix of shape ({len(X1)}, {len(X2)})...")
    matrix = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            # The overlap is the probability of the all-zero state
            matrix[i, j] = kernel_circuit(x1, x2)[0]
    return matrix

def train_and_predict_qsvm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    """
    Trains a classical SVM using the custom Quantum Kernel to avoid barren plateaus.
    """
    logger.info("Initializing Support Vector Machine with Quantum Kernel...")
    
    # SVC explicitly configured to accept a precomputed matrix
    qsvm = SVC(kernel="precomputed", probability=True)
    
    # Compute the training matrix and fit
    kernel_train = q_kernel_matrix(X_train, X_train)
    qsvm.fit(kernel_train, y_train)
    
    # Save the hybrid model
    model_path = MODEL_DIR / "qsvm_weights.pkl"
    joblib.dump(qsvm, model_path)
    logger.info(f"QSVM successfully fitted and saved to {model_path}")
    
    # Compute test matrix and predict
    kernel_test = q_kernel_matrix(X_test, X_train)
    predictions = qsvm.predict_proba(kernel_test)[:, 1]
    
    return predictions