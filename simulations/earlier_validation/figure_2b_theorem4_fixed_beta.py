"""
Validation of Theorem 4 using actual prediction errors (MSE), with beta FIXED.

Difference from figure_2b_theorem4.py:
    The true beta vector is simulated ONCE (per gamma, since its dimension is S)
    from a standard normal distribution, and reused across all trials, instead of
    being resampled every trial.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict, List


def generate_clustered_data(n_clusters: int,
                           samples_per_cluster: int,
                           n_features: int,
                           cluster_means: List[np.ndarray],
                           beta: np.ndarray,
                           noise_std: float = 0.1,
                           seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """Generate synthetic clustered data."""
    np.random.seed(seed)

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

    return X_clusters, Y_clusters, X_merged, Y_merged


def compute_mse_via_predictions_optimized(n_clusters: int,
                                          samples_per_cluster: int,
                                          n_features: int,
                                          beta: np.ndarray,
                                          cluster_means: List[np.ndarray],
                                          n_test_points: int = 500,
                                          noise_std: float = 0.1,
                                          seed: int = 42) -> Tuple[float, float]:
    """
    Compute MSE by making predictions, with matrix inversion cached.
    """
    np.random.seed(seed)

    X_clusters, Y_clusters, X_merged, Y_merged = generate_clustered_data(
        n_clusters=n_clusters,
        samples_per_cluster=samples_per_cluster,
        n_features=n_features,
        cluster_means=cluster_means,
        beta=beta,
        noise_std=noise_std,
        seed=seed
    )

    XTX_inv_clusters = []
    XTXinv_XT_Y_clusters = []

    for b in range(n_clusters):
        X_b = X_clusters[b]
        Y_b = Y_clusters[b]

        XTX_b_inv = np.linalg.inv(X_b.T @ X_b)
        XTX_inv_clusters.append(XTX_b_inv)
        XTXinv_XT_Y_clusters.append(XTX_b_inv @ X_b.T @ Y_b)

    XTX_merged_inv = np.linalg.inv(X_merged.T @ X_merged)
    XTXinv_XT_Y_merged = XTX_merged_inv @ X_merged.T @ Y_merged

    X_test = np.random.randn(n_test_points, n_features)
    Y_test_true = X_test @ beta

    squared_errors_ensemble = []
    squared_errors_merged = []

    for i in range(n_test_points):
        x_star = X_test[i]
        y_true = Y_test_true[i]

        inv_variances = []
        weighted_predictions = []

        for b in range(n_clusters):
            variance_b = x_star @ XTX_inv_clusters[b] @ x_star
            inv_var_b = 1.0 / variance_b
            inv_variances.append(inv_var_b)

            pred_b = x_star @ XTXinv_XT_Y_clusters[b]
            weighted_predictions.append(inv_var_b * pred_b)

        y_pred_ensemble = sum(weighted_predictions) / sum(inv_variances)
        squared_errors_ensemble.append((y_pred_ensemble - y_true) ** 2)

        y_pred_merged = x_star @ XTXinv_XT_Y_merged
        squared_errors_merged.append((y_pred_merged - y_true) ** 2)

    mse_ensemble = np.mean(squared_errors_ensemble)
    mse_merged = np.mean(squared_errors_merged)

    return mse_ensemble, mse_merged


def compute_empirical_mse_with_ci(n_clusters: int,
                                   samples_per_cluster: int,
                                   n_features: int,
                                   n_test_points: int = 500,
                                   n_trials: int = 50,
                                   seed: int = 42) -> Dict:
    """
    Compute empirical MSE ratio with confidence intervals.

    Beta is drawn ONCE from a standard normal distribution and reused across
    all trials. Cluster means are still resampled per trial.
    """
    np.random.seed(seed)

    # Draw beta ONCE from standard normal, fixed across all trials
    beta = np.random.randn(n_features)

    mse_ensemble_all = []
    mse_merged_all = []
    ratios = []

    for trial in range(n_trials):
        # Cluster means still resampled per trial
        mu = np.random.randn(n_features)
        mu = mu / np.linalg.norm(mu)
        cluster_means = [mu, -mu]

        mse_ensemble, mse_merged = compute_mse_via_predictions_optimized(
            n_clusters=n_clusters,
            samples_per_cluster=samples_per_cluster,
            n_features=n_features,
            beta=beta,
            cluster_means=cluster_means,
            n_test_points=n_test_points,
            noise_std=0.1,
            seed=seed + trial
        )

        mse_ensemble_all.append(mse_ensemble)
        mse_merged_all.append(mse_merged)
        ratios.append(mse_ensemble / mse_merged)

    ratios = np.array(ratios)
    mse_ensemble_all = np.array(mse_ensemble_all)
    mse_merged_all = np.array(mse_merged_all)

    mean_ratio = np.mean(ratios)

    se = stats.sem(ratios)
    ci = stats.t.interval(0.95, len(ratios)-1, loc=mean_ratio, scale=se)

    return {
        'ratios': ratios,
        'mean_ratio': mean_ratio,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'mse_ensemble': mse_ensemble_all,
        'mse_merged': mse_merged_all,
        'std_ratio': np.std(ratios)
    }


def theoretical_variance_ratio(gamma: float) -> float:
    """Compute theoretical variance ratio from Theorem 4."""
    if gamma < 0 or gamma >= 0.5:
        raise ValueError("gamma must be in [0, 1/2)")
    return (1 - gamma) / (1 - 2 * gamma)


def validate_theorem4_with_mse():
    """Validate Theorem 4 using actual prediction MSE with fixed beta."""
    print("=" * 80)
    print("VALIDATION OF THEOREM 4 USING ACTUAL PREDICTION MSE (FIXED BETA)")
    print("=" * 80)
    print("\nBeta is drawn ONCE per gamma from N(0, I) and reused across all trials.")
    print("\n" + "=" * 80)

    n_total = 1000
    gamma_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    n_trials = 100
    n_test_points = 1000

    print(f"\nSimulation parameters:")
    print(f"  Total sample size: n = {n_total}")
    print(f"  Number of clusters: k = 2")
    print(f"  Trials per γ: {n_trials}")
    print(f"  Test points per trial: {n_test_points}")
    print(f"  Total predictions per γ: {n_trials * n_test_points:,}")

    print(f"\n{'γ (S/n)':<10} {'S':<6} {'Theoretical':<13} {'Empirical':<13} {'95% CI':<25} {'Error':<10}")
    print("-" * 90)

    results = []

    for idx, gamma in enumerate(gamma_values):
        S = int(gamma * n_total)

        if S >= n_total // 2:
            continue

        print(f"{gamma:<10.2f} {S:<6}", end='  ', flush=True)

        theoretical = theoretical_variance_ratio(gamma)

        empirical_results = compute_empirical_mse_with_ci(
            n_clusters=2,
            samples_per_cluster=n_total // 2,
            n_features=S,
            n_test_points=n_test_points,
            n_trials=n_trials,
            seed=42 + int(gamma * 100)
        )

        empirical = empirical_results['mean_ratio']
        ci_lower = empirical_results['ci_lower']
        ci_upper = empirical_results['ci_upper']
        error = abs(empirical - theoretical)

        ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        print(f"{theoretical:<13.4f} {empirical:<13.4f} {ci_str:<25} {error:<10.4f}")

        results.append({
            'gamma': gamma,
            'S': S,
            'theoretical': theoretical,
            'empirical': empirical,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'error': error,
            'std': empirical_results['std_ratio'],
            'mse_ensemble': empirical_results['mse_ensemble'],
            'mse_merged': empirical_results['mse_merged']
        })

    return results


def plot_results_with_ci(results: List[Dict]):
    """Plot theoretical vs empirical MSE ratios (fixed-beta variant)."""
    gammas = np.array([r['gamma'] for r in results])
    empirical = np.array([r['empirical'] for r in results])
    ci_lower = np.array([r['ci_lower'] for r in results])
    ci_upper = np.array([r['ci_upper'] for r in results])

    COLOR_THEORY = "#0072B2"
    COLOR_EMPIRICAL = "#D55E00"

    fig, ax = plt.subplots(figsize=(10, 6))

    gamma_smooth = np.linspace(0.05, 0.45, 200)
    theoretical_smooth = (1 - gamma_smooth) / (1 - 2 * gamma_smooth)
    ax.plot(gamma_smooth, theoretical_smooth, color=COLOR_THEORY,
            linewidth=2.2,
            label=r'Theoretical: $(1-\gamma)/(1-2\gamma)$',
            zorder=3)

    ax.plot(gammas, empirical, marker='o', linewidth=2.2, markersize=7,
            color=COLOR_EMPIRICAL, label='Empirical MSE Ratio (fixed β)', zorder=4)
    ax.fill_between(gammas, ci_lower, ci_upper, color=COLOR_EMPIRICAL,
                    alpha=0.20, zorder=1)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6,
               zorder=0)

    ax.set_xlabel(r'$\gamma = p/n$', fontsize=13)
    ax.set_ylabel('MSE Ratio (Ensemble / Merged)', fontsize=13)
    ax.set_title('Ridgeless Regression: Proportional Asymptotics Regime (Fixed β)',
                 fontsize=19, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    ax.set_ylim(0.0, 6.0)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xlim(0.04, 0.46)

    plt.tight_layout()
    plt.savefig('theorem4_validation_mse_fixed_beta.png', dpi=300, bbox_inches='tight')
    plt.savefig('theorem4_validation_mse_fixed_beta.pdf', bbox_inches='tight')
    print(f"\n\nPlot saved to: theorem4_validation_mse_fixed_beta.png and .pdf")
    plt.close()


def detailed_example():
    """Detailed breakdown for γ = 0.2 with fixed beta."""
    print("\n" + "=" * 80)
    print("DETAILED EXAMPLE (FIXED BETA): γ = 0.2")
    print("=" * 80)

    gamma = 0.2
    n_total = 1000
    S = int(gamma * n_total)
    n_trials = 100
    n_test_points = 1000

    print(f"\nParameters:")
    print(f"  γ = S/n = {gamma}")
    print(f"  Total samples: n = {n_total}")
    print(f"  Features: S = {S}")
    print(f"  Samples per cluster: n/2 = {n_total//2}")
    print(f"  Trials: {n_trials}")
    print(f"  Test points per trial: {n_test_points}")

    theoretical = theoretical_variance_ratio(gamma)
    print(f"\nTheoretical MSE ratio: (1-{gamma})/(1-2*{gamma}) = {theoretical:.6f}")

    print(f"\nComputing empirical MSE from actual predictions (fixed beta)...")
    results = compute_empirical_mse_with_ci(
        n_clusters=2,
        samples_per_cluster=n_total // 2,
        n_features=S,
        n_test_points=n_test_points,
        n_trials=n_trials,
        seed=42
    )

    print(f"\nResults:")
    print(f"  Empirical MSE ratio:     {results['mean_ratio']:.6f}")
    print(f"  95% CI:                  [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
    print(f"  Standard deviation:      {results['std_ratio']:.6f}")
    print(f"  Absolute error:          {abs(results['mean_ratio'] - theoretical):.6f}")
    print(f"  Relative error:          {100 * abs(results['mean_ratio'] - theoretical) / theoretical:.3f}%")

    print(f"\nMSE Statistics:")
    print(f"  Mean Ensemble MSE:       {np.mean(results['mse_ensemble']):.6f}")
    print(f"  Mean Merged MSE:         {np.mean(results['mse_merged']):.6f}")

    in_ci = results['ci_lower'] <= theoretical <= results['ci_upper']
    print(f"\nTheoretical value in 95% CI: {'YES' if in_ci else 'NO'}")


if __name__ == "__main__":
    import time
    start_time = time.time()

    detailed_example()

    print("\n")
    results = validate_theorem4_with_mse()

    plot_results_with_ci(results)

    print("\n" + "=" * 80)
    print("SUMMARY (FIXED BETA)")
    print("=" * 80)

    all_in_ci = all(r['ci_lower'] <= r['theoretical'] <= r['ci_upper'] for r in results)
    print(f"  1. Theoretical value in 95% CI for all γ: {'YES' if all_in_ci else 'NO'}")

    max_error = max(r['error'] for r in results)
    max_error_gamma = [r['gamma'] for r in results if r['error'] == max_error][0]
    print(f"  2. Maximum absolute error: {max_error:.4f} at γ = {max_error_gamma}")

    mean_error = np.mean([r['error'] for r in results])
    print(f"  3. Mean absolute error: {mean_error:.4f}")

    elapsed_time = time.time() - start_time
    print(f"\n  Total computation time: {elapsed_time/60:.1f} minutes")
    print("=" * 80)
