import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
import joblib
import logging
from src.config import MODEL_DIR

logger = logging.getLogger(__name__)

def compress_features(X: pd.DataFrame, distribution_type: str, n_components: int = 8) -> pd.DataFrame:
    """
    Applies PCA or Kernel PCA based on the distribution evaluation.
    Includes a safety subsample for Kernel PCA to prevent RAM exhaustion on large datasets.
    """
    if distribution_type == 'normal':
        reducer = PCA(n_components=n_components)
        logger.info(f"Fitting Standard Linear PCA to reduce features to {n_components} dimensions...")
        X_compressed = reducer.fit_transform(X)
    else:
        # KernelPCA(kernel='rbf') builds an (N x N) matrix. We configure a randomized solver
        # and subsample a representative batch if the dataset is large.
        logger.info("High variance/skew detected. Configuring Kernel PCA with memory optimization...")
        reducer = KernelPCA(
            n_components=n_components, 
            kernel='rbf', 
            fit_inverse_transform=True, 
            eigen_solver='randomized'
        )
        
        if len(X) > 10000:
            logger.info(f"Dataset size ({len(X)}) is large. Subsampling 10,000 rows to fit KernelPCA safely...")
            X_sample = X.sample(n=10000, random_state=42)
            reducer.fit(X_sample)
            X_compressed = reducer.transform(X)
        else:
            X_compressed = reducer.fit_transform(X)

    # Save the reducer so the exact same transformation can be applied in the UI/Production
    reducer_path = MODEL_DIR / "feature_reducer.pkl"
    joblib.dump(reducer, reducer_path)
    logger.info(f"Saved feature reducer to {reducer_path}")

    # Reconstruct into a DataFrame
    feature_names = [f"qubit_feat_{i}" for i in range(n_components)]
    return pd.DataFrame(X_compressed, columns=feature_names, index=X.index)