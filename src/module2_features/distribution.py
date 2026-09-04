import pandas as pd
import logging

logger = logging.getLogger(__name__)

def evaluate_distribution(X: pd.DataFrame, skew_threshold: float = 1.0) -> str:
    """
    Evaluates the skewness of the feature set to determine the compression routing.
    """
    # Calculate absolute skewness across all features
    skewness = X.skew().abs()
    mean_skew = skewness.mean()
    
    logger.info(f"Mean absolute feature skewness calculated: {mean_skew:.4f}")

    # Route based on the threshold
    if mean_skew <= skew_threshold:
        logger.info("Distribution is normal. Routing to Standard Linear PCA.")
        return 'normal'
    else:
        logger.info("High variance/skew detected. Routing to Kernel PCA.")
        return 'skewed'