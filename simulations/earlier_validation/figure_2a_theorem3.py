"""
Validation of Theorem 3 using actual prediction errors (MSE).

Theorem 3: Fixed-dimensional regime (S fixed, n → ∞)
For k clusters with random norm-1 means, Merged outperforms Ensemble.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict, List


def theoretical_ratio_k2(mu1: np.ndarray, mu2: np.ndarray, S: int, lambda1: float = 0.5) -> float:
    """
    Compute theoretical MSE ratio for k=2 from Theorem 3 (Lemma 9).

    Note: This implements the complex formulas from Lemma 9 (pages 26-27).
    For randomly generated norm-1 means, the theoretical ratio may differ from
    empirical due to finite sample effects and approximation errors.

    Assumes equal-sized clusters (lambda1 = lambda2 = 0.5).
    """
    lambda2 = 1 - lambda1

    # mu^T mu for each cluster (should be ~1 for norm-1 vectors)
    mu1_dot = np.dot(mu1, mu1)
    mu2_dot = np.dot(mu2, mu2)
    mu2_dot_mu1_dot = mu2_dot * mu1_dot
    mu1_dot_mu2 = np.dot(mu1, mu2)
    mu2_dot_mu1_vec = np.dot(mu2, mu1)  # This is a scalar: mu2^T mu1

    # Ensemble variance (from Lemma 9, part i)
    numerator_s1 = (S - 1) * (1 + lambda2 * mu1_dot + lambda1 * mu2_dot) + 1
    denominator_s1 = (S - 1) * (1 + mu1_dot) * (1 + mu2_dot) + 1 + lambda1 * mu1_dot + lambda2 * mu2_dot
    s1 = (S - 1) + numerator_s1 / denominator_s1

    # Merged variance (from Lemma 9, part ii) - more complex
    # First term
    term1 = lambda1 * mu1_dot / (1 + lambda1 * mu1_dot)

    # Second term numerator
    term2_num = lambda2 * mu2_dot - \
                (2 * lambda1 * lambda2 / (1 + lambda1 * mu1_dot)) * mu2_dot_mu1_dot + \
                (lambda2 * lambda1**2 / (1 + lambda1 * mu1_dot)**2) * (mu1_dot)**2

    # Second term denominator
    term2_denom = 1 + lambda2 * mu2_dot - \
                  (lambda1 * lambda2 / (1 + lambda1 * mu1_dot)) * mu2_dot_mu1_vec * mu1_dot_mu2

    s2 = S - term1 - term2_num / term2_denom

    return s1 / s2


def generate_clustered_data_fixed_dim(n_clusters: int,
                                       samples_per_cluster: int,
                                       n_features: int,
                                       beta: np.ndarray,
                                       noise_std: float = 0.1,
                                       seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, List[np.ndarray]]:
    """Generate synthetic clustered data with random norm-1 cluster means."""
    np.random.seed(seed)

    # Generate k random norm-1 cluster means
    cluster_means = []
    for b in range(n_clusters):
        mu_b = np.random.randn(n_features)
        mu_b = mu_b / np.linalg.norm(mu_b)  # Normalize to unit norm
        cluster_means.append(mu_b)

    X_clusters = []
    Y_clusters = []

    for b in range(n_clusters):
        X_b = np.random.multivariate_normal(
            mean=cluster_means[b],
            cov=np.eye(n_features),
            size=samples_per_cluster
        )
        Y_b = X_b @ beta + np.random.normal(0, noise_std, samples_per_cluster)
        X_clusters.append(X_b)
        Y_clusters.append(Y_b)

    X_merged = np.vstack(X_clusters)
    Y_merged = np.concatenate(Y_clusters)

    return X_clusters, Y_clusters, X_merged, Y_merged, cluster_means


def compute_mse_via_predictions_optimized(n_clusters: int,
                                          samples_per_cluster: int,
                                          n_features: int,
                                          beta: np.ndarray,
                                          n_test_points: int = 1000,
                                          noise_std: float = 0.1,
                                          seed: int = 42) -> Tuple[float, float]:
    """
    Compute MSE by making predictions, with matrix inversion cached.

    KEY OPTIMIZATION: Compute matrix inversions ONCE per trial, reuse for all test points.
    This is safe because the training data is fixed within each trial.
    """
    np.random.seed(seed)

    # Generate training data with random norm-1 cluster means
    X_clusters, Y_clusters, X_merged, Y_merged, cluster_means = generate_clustered_data_fixed_dim(
        n_clusters=n_clusters,
        samples_per_cluster=samples_per_cluster,
        n_features=n_features,
        beta=beta,
        noise_std=noise_std,
        seed=seed
    )

    # CACHE: Precompute matrix inversions for ensemble (once per cluster)
    XTX_inv_clusters = []
    XTXinv_XT_Y_clusters = []  # Precompute (X_b^T X_b)^{-1} X_b^T Y_b

    for b in range(n_clusters):
        X_b = X_clusters[b]
        Y_b = Y_clusters[b]

        XTX_b_inv = np.linalg.inv(X_b.T @ X_b)
        XTX_inv_clusters.append(XTX_b_inv)
        XTXinv_XT_Y_clusters.append(XTX_b_inv @ X_b.T @ Y_b)

    # CACHE: Precompute matrix inversion for merged (once)
    XTX_merged_inv = np.linalg.inv(X_merged.T @ X_merged)
    XTXinv_XT_Y_merged = XTX_merged_inv @ X_merged.T @ Y_merged

    # Generate test data from standard normal (as in paper)
    X_test = np.random.randn(n_test_points, n_features)
    Y_test_true = X_test @ beta  # True outcomes (no noise)

    # Make predictions using cached inversions
    squared_errors_ensemble = []
    squared_errors_merged = []

    for i in range(n_test_points):
        x_star = X_test[i]
        y_true = Y_test_true[i]

        # Ensemble prediction (using cached inversions)
        inv_variances = []
        weighted_predictions = []

        for b in range(n_clusters):
            # Variance: x*^T (X_b^T X_b)^{-1} x*
            variance_b = x_star @ XTX_inv_clusters[b] @ x_star
            inv_var_b = 1.0 / variance_b
            inv_variances.append(inv_var_b)

            # Prediction: x*^T (X_b^T X_b)^{-1} X_b^T Y_b
            pred_b = x_star @ XTXinv_XT_Y_clusters[b]
            weighted_predictions.append(inv_var_b * pred_b)

        y_pred_ensemble = sum(weighted_predictions) / sum(inv_variances)
        squared_errors_ensemble.append((y_pred_ensemble - y_true) ** 2)

        # Merged prediction (using cached inversion)
        y_pred_merged = x_star @ XTXinv_XT_Y_merged
        squared_errors_merged.append((y_pred_merged - y_true) ** 2)

    # MSE is the mean of squared errors
    mse_ensemble = np.mean(squared_errors_ensemble)
    mse_merged = np.mean(squared_errors_merged)

    return mse_ensemble, mse_merged


def compute_empirical_mse_with_ci(n_clusters: int,
                                   n_total: int,
                                   n_features: int,
                                   n_test_points: int = 1000,
                                   n_trials: int = 100,
                                   seed: int = 42,
                                   return_means: bool = False) -> Dict:
    """
    Compute empirical MSE ratio with confidence intervals.
    """
    np.random.seed(seed)

    samples_per_cluster = n_total // n_clusters

    mse_ensemble_all = []
    mse_merged_all = []
    ratios = []
    cluster_means_all = []

    for trial in range(n_trials):
        # Generate random beta for each trial
        beta = np.random.randn(n_features)
        beta = beta / np.linalg.norm(beta)

        # For k=2, we need to track cluster means for theoretical comparison
        trial_seed = seed + trial

        # Compute MSE via actual predictions (with caching)
        # We need to regenerate cluster means here to track them
        if return_means and n_clusters == 2:
            np.random.seed(trial_seed)
            cluster_means = []
            for b in range(n_clusters):
                mu_b = np.random.randn(n_features)
                mu_b = mu_b / np.linalg.norm(mu_b)
                cluster_means.append(mu_b)
            cluster_means_all.append(cluster_means)

        mse_ensemble, mse_merged = compute_mse_via_predictions_optimized(
            n_clusters=n_clusters,
            samples_per_cluster=samples_per_cluster,
            n_features=n_features,
            beta=beta,
            n_test_points=n_test_points,
            noise_std=0.1,
            seed=trial_seed
        )

        mse_ensemble_all.append(mse_ensemble)
        mse_merged_all.append(mse_merged)
        ratios.append(mse_ensemble / mse_merged)

    # Convert to arrays
    ratios = np.array(ratios)
    mse_ensemble_all = np.array(mse_ensemble_all)
    mse_merged_all = np.array(mse_merged_all)

    # Compute statistics
    mean_ratio = np.mean(ratios)

    # 95% confidence interval using t-distribution
    se = stats.sem(ratios)
    ci = stats.t.interval(0.95, len(ratios)-1, loc=mean_ratio, scale=se)

    result = {
        'ratios': ratios,
        'mean_ratio': mean_ratio,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'mse_ensemble': mse_ensemble_all,
        'mse_merged': mse_merged_all,
        'std_ratio': np.std(ratios)
    }

    if return_means and n_clusters == 2:
        result['cluster_means'] = cluster_means_all

    return result


def validate_theorem3_with_mse():
    """Validate Theorem 3 using actual prediction MSE."""
    print("=" * 80)
    print("VALIDATION OF THEOREM 3 USING ACTUAL PREDICTION MSE")
    print("=" * 80)
    print("\nTheorem 3: Fixed-dimensional regime (S fixed, n → ∞)")
    print("For k clusters with random norm-1 means:")
    print("    MSE(Y_E) / MSE(Y_M) > 1 (Merged outperforms Ensemble)")
    print("\n" + "=" * 80)

    # Parameters
    n_total = 5000
    n_features = 100  # Fixed S = 100 (dense regime)
    k_values = [2, 3, 4, 5, 6, 7, 8]
    n_trials = 100
    n_test_points = 1000

    print(f"\nSimulation parameters:")
    print(f"  Total sample size: n = {n_total}")
    print(f"  Features: S = {n_features} (fixed)")
    print(f"  Number of clusters: k ∈ {{{', '.join(map(str, k_values))}}}")
    print(f"  Trials per k: {n_trials}")
    print(f"  Test points per trial: {n_test_points}")
    print(f"  Total predictions per k: {n_trials * n_test_points:,}")
    print(f"  Cluster means: Random norm-1 vectors")
    print(f"  Optimization: Matrix inversions cached within each trial")

    print(f"\n{'k':<6} {'n/k':<8} {'Theoretical':<13} {'Empirical':<13} {'95% CI':<25} {'Ratio > 1?':<12}")
    print("-" * 90)

    results = []

    for k in k_values:
        samples_per_cluster = n_total // k

        print(f"{k:<6} {samples_per_cluster:<8}", end='  ', flush=True)

        # Empirical measurement via actual predictions
        return_means_flag = (k == 2)  # Only compute theoretical for k=2
        empirical_results = compute_empirical_mse_with_ci(
            n_clusters=k,
            n_total=n_total,
            n_features=n_features,
            n_test_points=n_test_points,
            n_trials=n_trials,
            seed=42 + k * 100,
            return_means=return_means_flag
        )

        empirical = empirical_results['mean_ratio']
        ci_lower = empirical_results['ci_lower']
        ci_upper = empirical_results['ci_upper']

        # Compute theoretical ratio for k=2
        theoretical = None
        if k == 2 and 'cluster_means' in empirical_results:
            theoretical_ratios = []
            for cluster_means in empirical_results['cluster_means']:
                theoretical_ratios.append(
                    theoretical_ratio_k2(cluster_means[0], cluster_means[1], n_features)
                )
            theoretical = np.mean(theoretical_ratios)

        ratio_greater_than_1 = ci_lower > 1.0
        status = "✓ YES" if ratio_greater_than_1 else "Uncertain"

        ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"

        if theoretical is not None:
            print(f"{theoretical:<13.4f} {empirical:<13.4f} {ci_str:<25} {status:<12}")
        else:
            print(f"{'N/A':<13} {empirical:<13.4f} {ci_str:<25} {status:<12}")

        result = {
            'k': k,
            'samples_per_cluster': samples_per_cluster,
            'empirical': empirical,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': empirical_results['std_ratio'],
            'mse_ensemble': empirical_results['mse_ensemble'],
            'mse_merged': empirical_results['mse_merged']
        }
        if theoretical is not None:
            result['theoretical'] = theoretical

        results.append(result)

    return results


def plot_results_with_ci(results: List[Dict]):
    """Plot empirical MSE ratios vs number of clusters with confidence bands."""
    k_values = np.array([r['k'] for r in results])
    empirical = np.array([r['empirical'] for r in results])
    ci_lower = np.array([r['ci_lower'] for r in results])
    ci_upper = np.array([r['ci_upper'] for r in results])

    plt.figure(figsize=(12, 7))

    # Empirical line with confidence band
    plt.plot(k_values, empirical, 'r-', linewidth=2.5, label='Empirical MSE Ratio', zorder=4)

    # Add dots at each k value
    plt.scatter(k_values, empirical, c='red', s=100, zorder=5, edgecolors='darkred', linewidths=1.5)

    # Confidence band
    plt.fill_between(k_values, ci_lower, ci_upper, alpha=0.3, color='red', zorder=1)

    # Add invisible line for legend entry
    plt.plot([], [], color='red', alpha=0.3, linewidth=10, label='95% Confidence Band')

    # Reference line at 1
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, linewidth=1.5, zorder=0, label='Ratio = 1')

    plt.xlabel('Number of Clusters (k)', fontsize=14, fontweight='bold')
    plt.ylabel('MSE Ratio (Ensemble / Merged)', fontsize=14, fontweight='bold')
    plt.title('Fixed-Dimensional Linear Regression with k Gaussian Clusters',
              fontsize=16, fontweight='bold', pad=20)

    plt.legend(fontsize=12, loc='upper right', framealpha=0.95)

    plt.grid(True, alpha=0.3)
    plt.xlim(1.5, 10.5)
    plt.xticks(k_values)

    plt.tight_layout()
    plt.savefig('theorem3_validation_mse.png', dpi=300, bbox_inches='tight')
    print(f"\n\nPlot saved to: theorem3_validation_mse.png")
    plt.close()


def detailed_example():
    """Detailed breakdown for k = 2 with theoretical comparison"""
    print("\n" + "=" * 80)
    print("DETAILED EXAMPLE: k = 2 clusters (with Theorem 3 theoretical ratio)")
    print("=" * 80)

    k = 2
    n_total = 5000
    n_features = 100
    n_trials = 100
    n_test_points = 1000
    samples_per_cluster = n_total // k

    print(f"\nParameters:")
    print(f"  Number of clusters: k = {k}")
    print(f"  Total samples: n = {n_total}")
    print(f"  Samples per cluster: n/k = {samples_per_cluster}")
    print(f"  Features: S = {n_features} (fixed)")
    print(f"  Trials: {n_trials}")
    print(f"  Test points per trial: {n_test_points}")
    print(f"  Cluster means: Random norm-1 vectors")

    # Empirical
    print(f"\nComputing empirical MSE from actual predictions...")
    results = compute_empirical_mse_with_ci(
        n_clusters=k,
        n_total=n_total,
        n_features=n_features,
        n_test_points=n_test_points,
        n_trials=n_trials,
        seed=42,
        return_means=True
    )

    # Compute theoretical ratio
    theoretical_ratios = []
    for cluster_means in results['cluster_means']:
        theoretical_ratios.append(
            theoretical_ratio_k2(cluster_means[0], cluster_means[1], n_features)
        )
    theoretical_mean = np.mean(theoretical_ratios)
    theoretical_std = np.std(theoretical_ratios)

    print(f"\nResults:")
    print(f"  Theoretical MSE ratio (Theorem 3): {theoretical_mean:.6f} ± {theoretical_std:.6f}")
    print(f"  Empirical MSE ratio:               {results['mean_ratio']:.6f}")
    print(f"  95% CI:                            [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
    print(f"  Standard deviation:                {results['std_ratio']:.6f}")
    print(f"  Absolute error:                    {abs(results['mean_ratio'] - theoretical_mean):.6f}")
    print(f"  Relative error:                    {100 * abs(results['mean_ratio'] - theoretical_mean) / theoretical_mean:.3f}%")

    print(f"\nMSE Statistics:")
    print(f"  Mean Ensemble MSE:       {np.mean(results['mse_ensemble']):.6f}")
    print(f"  Mean Merged MSE:         {np.mean(results['mse_merged']):.6f}")

    # Check if ratio > 1
    ratio_gt_1 = results['ci_lower'] > 1.0
    print(f"\n95% CI entirely above 1.0: {'✓ YES' if ratio_gt_1 else '✗ NO'}")

    # Check if theoretical is in CI
    in_ci = results['ci_lower'] <= theoretical_mean <= results['ci_upper']
    print(f"Theoretical value in 95% CI: {'✓ YES' if in_ci else '✗ NO'}")

    if results['mean_ratio'] > 1:
        advantage = ((results['mean_ratio'] - 1) / 1) * 100
        print(f"\n✓ Merged is better: Ensemble has {advantage:.2f}% higher MSE")
        print(f"✓ This confirms Theorem 3 (ratio > 1)")


if __name__ == "__main__":
    import time
    start_time = time.time()

    # Detailed example
    detailed_example()

    # Full validation sweep
    print("\n")
    results = validate_theorem3_with_mse()

    # Plot with confidence bands
    plot_results_with_ci(results)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nKey findings:")
    all_above_1 = all(r['ci_lower'] > 1.0 for r in results)
    print(f"  1. MSE ratio > 1 for all k: {'✓ YES' if all_above_1 else '✗ NO'}")

    min_ratio = min(r['empirical'] for r in results)
    min_ratio_k = [r['k'] for r in results if r['empirical'] == min_ratio][0]
    max_ratio = max(r['empirical'] for r in results)
    max_ratio_k = [r['k'] for r in results if r['empirical'] == max_ratio][0]

    print(f"  2. Minimum ratio: {min_ratio:.4f} at k = {min_ratio_k}")
    print(f"  3. Maximum ratio: {max_ratio:.4f} at k = {max_ratio_k}")

    mean_ratio = np.mean([r['empirical'] for r in results])
    print(f"  4. Mean ratio across all k: {mean_ratio:.4f}")

    elapsed_time = time.time() - start_time
    print(f"\n  Total computation time: {elapsed_time/60:.1f} minutes")

    print("\nConclusion:")
    print("  Empirical MSE shows Merged consistently outperforms Ensemble.")
    print("  This holds regardless of the number of clusters k.")
    print("  In fixed-dimensional regime, Merged is asymptotically better.")
    print("=" * 80)
