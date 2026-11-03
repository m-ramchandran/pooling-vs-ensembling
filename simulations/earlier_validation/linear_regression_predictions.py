"""
Implementation of Ensemble and Merged predictions for linear regression on clustered data.
Based on equations from page 8 of the paper.
"""

import numpy as np
from typing import List, Tuple


def predict_ensemble(x_star: np.ndarray,
                     X_clusters: List[np.ndarray],
                     Y_clusters: List[np.ndarray]) -> float:
    """
    Compute Ensemble prediction using Inverse Variance Weighting.

    Formula:
    Y_E(x*) = [sum_b [x*^T(X_b^T X_b)^{-1} x*]^{-1}]^{-1} *
              sum_b [x*^T(X_b^T X_b)^{-1} x*]^{-1} * x*^T(X_b^T X_b)^{-1} X_b^T Y_b

    Parameters:
    -----------
    x_star : np.ndarray
        Test point of shape (S,) where S is the number of features
    X_clusters : List[np.ndarray]
        List of training covariate matrices, one per cluster
        Each X_b has shape (n_b, S) where n_b is the cluster size
    Y_clusters : List[np.ndarray]
        List of training outcome vectors, one per cluster
        Each Y_b has shape (n_b,)

    Returns:
    --------
    float
        The ensemble prediction at x_star
    """
    k = len(X_clusters)  # Number of clusters

    # Storage for inverse variances and cluster predictions
    inv_variances = []
    cluster_predictions = []

    for b in range(k):
        X_b = X_clusters[b]  # Shape: (n_b, S)
        Y_b = Y_clusters[b]  # Shape: (n_b,)

        # Compute X_b^T X_b
        XTX_b = X_b.T @ X_b  # Shape: (S, S)

        # Compute (X_b^T X_b)^{-1}
        XTX_b_inv = np.linalg.inv(XTX_b)  # Shape: (S, S)

        # Compute variance: x*^T (X_b^T X_b)^{-1} x*
        variance_b = x_star.T @ XTX_b_inv @ x_star  # Scalar

        # Compute inverse variance (weight)
        inv_var_b = 1.0 / variance_b
        inv_variances.append(inv_var_b)

        # Compute cluster prediction: x*^T (X_b^T X_b)^{-1} X_b^T Y_b
        pred_b = x_star.T @ XTX_b_inv @ X_b.T @ Y_b  # Scalar
        cluster_predictions.append(pred_b)

    # Sum of inverse variances (normalization constant)
    sum_inv_var = sum(inv_variances)

    # Weighted average of predictions
    weighted_sum = sum(w * pred for w, pred in zip(inv_variances, cluster_predictions))

    # Final ensemble prediction
    y_ensemble = weighted_sum / sum_inv_var

    return y_ensemble


def predict_merged(x_star: np.ndarray,
                   X: np.ndarray,
                   Y: np.ndarray) -> float:
    """
    Compute Merged prediction using standard linear regression on all data.

    Formula:
    Y_M(x*) = x*^T (X^T X)^{-1} X^T Y

    Parameters:
    -----------
    x_star : np.ndarray
        Test point of shape (S,) where S is the number of features
    X : np.ndarray
        Full training covariate matrix of shape (n, S)
        This is the concatenation of all clusters
    Y : np.ndarray
        Full training outcome vector of shape (n,)
        This is the concatenation of all cluster outcomes

    Returns:
    --------
    float
        The merged prediction at x_star
    """
    # Compute X^T X
    XTX = X.T @ X  # Shape: (S, S)

    # Compute (X^T X)^{-1}
    XTX_inv = np.linalg.inv(XTX)  # Shape: (S, S)

    # Compute prediction: x*^T (X^T X)^{-1} X^T Y
    y_merged = x_star.T @ XTX_inv @ X.T @ Y  # Scalar

    return y_merged


def generate_clustered_data(n_clusters: int,
                           samples_per_cluster: int,
                           n_features: int,
                           cluster_means: List[np.ndarray],
                           beta: np.ndarray,
                           noise_std: float = 0.1,
                           seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Generate synthetic clustered data for linear regression.

    Parameters:
    -----------
    n_clusters : int
        Number of clusters (k)
    samples_per_cluster : int
        Number of samples per cluster (n_b)
    n_features : int
        Number of features (S)
    cluster_means : List[np.ndarray]
        List of cluster-specific mean vectors, each of shape (S,)
    beta : np.ndarray
        True coefficient vector of shape (S,)
    noise_std : float
        Standard deviation of Gaussian noise
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    X_clusters : List[np.ndarray]
        List of covariate matrices, one per cluster
    Y_clusters : List[np.ndarray]
        List of outcome vectors, one per cluster
    X_merged : np.ndarray
        Concatenated covariate matrix
    Y_merged : np.ndarray
        Concatenated outcome vector
    """
    np.random.seed(seed)

    X_clusters = []
    Y_clusters = []

    for b in range(n_clusters):
        # Generate covariates from MVN(mu_b, I)
        X_b = np.random.multivariate_normal(
            mean=cluster_means[b],
            cov=np.eye(n_features),
            size=samples_per_cluster
        )  # Shape: (n_b, S)

        # Generate outcomes: Y_b = X_b * beta + noise
        Y_b = X_b @ beta + np.random.normal(0, noise_std, samples_per_cluster)

        X_clusters.append(X_b)
        Y_clusters.append(Y_b)

    # Merge all clusters
    X_merged = np.vstack(X_clusters)
    Y_merged = np.concatenate(Y_clusters)

    return X_clusters, Y_clusters, X_merged, Y_merged


def example_usage():
    """
    Example demonstrating the Ensemble and Merged predictions.
    """
    print("=" * 70)
    print("Linear Regression: Ensemble vs Merged Predictions")
    print("=" * 70)

    # Set parameters
    n_clusters = 2
    samples_per_cluster = 200
    n_features = 10
    noise_std = 0.5

    # True coefficient vector
    beta = np.random.randn(n_features)
    beta = beta / np.linalg.norm(beta)  # Normalize

    # Cluster means (mu_1 = -mu_2 as in Theorem 4)
    mu = np.random.randn(n_features)
    mu = mu / np.linalg.norm(mu)  # Unit norm
    cluster_means = [mu, -mu]

    print(f"\nParameters:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Samples per cluster: {samples_per_cluster}")
    print(f"  Number of features: {n_features}")
    print(f"  Noise std: {noise_std}")
    print(f"  Cluster means: mu_1 = -mu_2 (unit norm)")

    # Generate training data
    X_clusters, Y_clusters, X_merged, Y_merged = generate_clustered_data(
        n_clusters=n_clusters,
        samples_per_cluster=samples_per_cluster,
        n_features=n_features,
        cluster_means=cluster_means,
        beta=beta,
        noise_std=noise_std
    )

    print(f"\nTraining data generated:")
    for b in range(n_clusters):
        print(f"  Cluster {b+1}: X shape = {X_clusters[b].shape}, Y shape = {Y_clusters[b].shape}")
    print(f"  Merged: X shape = {X_merged.shape}, Y shape = {Y_merged.shape}")

    # Generate test point from MVN(0, I) as specified in the paper
    x_star = np.random.randn(n_features)
    true_y_star = x_star @ beta  # True outcome (no noise)

    print(f"\nTest point: x_star shape = {x_star.shape}")
    print(f"True outcome (no noise): {true_y_star:.4f}")

    # Make predictions
    y_ensemble = predict_ensemble(x_star, X_clusters, Y_clusters)
    y_merged = predict_merged(x_star, X_merged, Y_merged)

    print(f"\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)
    print(f"Ensemble prediction: {y_ensemble:.4f}")
    print(f"Merged prediction:   {y_merged:.4f}")
    print(f"True value:          {true_y_star:.4f}")

    print(f"\nPrediction errors:")
    print(f"  Ensemble error: {abs(y_ensemble - true_y_star):.4f}")
    print(f"  Merged error:   {abs(y_merged - true_y_star):.4f}")

    # Compute variances at x_star for each approach
    print(f"\n" + "=" * 70)
    print("VARIANCE ANALYSIS")
    print("=" * 70)

    # Ensemble variance
    inv_variances = []
    for b in range(n_clusters):
        X_b = X_clusters[b]
        XTX_b = X_b.T @ X_b
        XTX_b_inv = np.linalg.inv(XTX_b)
        variance_b = x_star.T @ XTX_b_inv @ x_star
        inv_variances.append(1.0 / variance_b)
        print(f"  Cluster {b+1} variance: {variance_b:.4f}")

    ensemble_variance = 1.0 / sum(inv_variances)
    print(f"\nEnsemble variance (IVW): {ensemble_variance:.4f}")

    # Merged variance
    XTX = X_merged.T @ X_merged
    XTX_inv = np.linalg.inv(XTX)
    merged_variance = x_star.T @ XTX_inv @ x_star
    print(f"Merged variance:         {merged_variance:.4f}")

    print(f"\nVariance ratio (Ensemble/Merged): {ensemble_variance/merged_variance:.4f}")
    print(f"Theory predicts ratio > 1 (Merged is better)")


if __name__ == "__main__":
    example_usage()
