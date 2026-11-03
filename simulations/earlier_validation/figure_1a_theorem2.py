"""
Random Forest Simulation with Diagnostic Information
Shows raw MSE values and detailed progress
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# General parameters
ntrain = 5000  # 5000 samples
ntest = 2000
S = 20  # Number of covariates
k_values = [2, 4, 8, 12, 16]  # Number of clusters
nrep = 10  # 10 replicates for quick testing
n_trees = 100  # 100 trees per forest

# COEFFICIENT PERTURBATION FLAG - Set to True to perturb coefficients in merged method
# (for clusters other than the test cluster [0, 1/k])
PERTURB_COEFFICIENTS = False  # Set to False to disable perturbations

# Generate coefficients for outcome simulation (fixed across all simulations)
linear_coefs = np.random.randn(S)
# NO nonlinear coefficients - testing pure linear function

# Perturbation magnitude (uniform perturbation in [-PERTURB_SCALE, PERTURB_SCALE])
PERTURB_SCALE = 0.1  # Small perturbation

print("="*60)
print("Random Forest Simulation with Diagnostics")
print("="*60)
print(f"\nParameters:")
print(f"  Training samples: {ntrain}")
print(f"  Test samples: {ntest}")
print(f"  Covariates: {S}")
print(f"  K values: {k_values}")
print(f"  Replicates per k: {nrep}")
print(f"  Trees per RF: {n_trees}")
print(f"\nOutcome function:")
print(f"  Y = X @ linear_coefs (PURELY LINEAR, NO NOISE)")
print(f"  Coefficient perturbation: {'ENABLED' if PERTURB_COEFFICIENTS else 'DISABLED'}")
if PERTURB_COEFFICIENTS:
    print(f"    - Clusters 2-k have perturbed coefficients (±{PERTURB_SCALE})")
    print(f"    - Cluster 1 [0, 1/k] uses original coefficients (same as test)")
print(f"\nRF Settings:")
print(f"  max_features=S//3 (use 6 features per split)")
print(f"\nSAMPLE COMPARISON:")
print(f"  Ensemble: {ntrain} samples from [0, 1/k]")
print(f"  Merged: {ntrain} samples total from k clusters ({ntrain}//k per cluster)")


def simulate_outcome(X, coefs=None):
    """Simulate outcome values - PURELY LINEAR (no interaction)

    Args:
        X: Input features
        coefs: Coefficients to use (default: linear_coefs)
    """
    if coefs is None:
        coefs = linear_coefs

    n_samples = X.shape[0]

    # Linear part ONLY: X @ coefs
    linear_part = X @ coefs

    # NO nonlinear part
    # NO noise - deterministic outcome

    Y = linear_part

    return Y


def ensemble_method(k):
    """Ensemble method: Train RF on data from Uniform[0, 1/k]"""
    # Simulate training data from Uniform[0, 1/k]^S
    X_train = np.random.uniform(0, 1/k, size=(ntrain, S))
    Y_train = simulate_outcome(X_train)

    # Simulate test data from Uniform[0, 1/k]^S
    X_test = np.random.uniform(0, 1/k, size=(ntest, S))
    Y_test = simulate_outcome(X_test)

    # Train Random Forest with explicit hyperparameters
    # max_depth=None: unlimited depth (trees grow until leaves are pure)
    # max_features=S//3: use 20//3=6 features per split
    # min_samples_split=2: minimum samples to split a node
    # min_samples_leaf=1: minimum samples in a leaf
    rf = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=None,
        max_features=S//3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None
    )
    rf.fit(X_train, Y_train)

    # Predict on training data
    Y_train_pred = rf.predict(X_train)
    train_mse = np.mean((Y_train - Y_train_pred)**2)
    train_baseline_mse = np.mean((Y_train - np.mean(Y_train))**2)

    # Predict on test data
    Y_test_pred = rf.predict(X_test)
    test_mse = np.mean((Y_test - Y_test_pred)**2)
    test_baseline_mse = np.mean((Y_test - np.mean(Y_train))**2)

    return train_mse, train_baseline_mse, test_mse, test_baseline_mse


def merged_method(k):
    """Merged method: Train RF on data from k clusters spanning [0,1]"""
    # Use ntrain//k samples per cluster (ntrain total across all k clusters)
    samples_per_cluster = ntrain // k
    X_train_list = []
    Y_train_list = []

    # Generate training data from k non-overlapping clusters
    for b in range(1, k+1):
        lower = (b - 1) / k
        upper = b / k
        X_cluster = np.random.uniform(lower, upper, size=(samples_per_cluster, S))
        X_train_list.append(X_cluster)

        # Generate Y values for this cluster
        if PERTURB_COEFFICIENTS and b > 1:
            # Perturb coefficients for clusters 2, 3, ..., k (NOT cluster 1 which is [0, 1/k])
            # Add uniform perturbation to each coefficient
            perturbation = np.random.uniform(-PERTURB_SCALE, PERTURB_SCALE, size=S)
            perturbed_coefs = linear_coefs + perturbation
            Y_cluster = simulate_outcome(X_cluster, coefs=perturbed_coefs)
        else:
            # Use original coefficients for cluster 1 ([0, 1/k])
            Y_cluster = simulate_outcome(X_cluster)

        Y_train_list.append(Y_cluster)

    # Concatenate all clusters
    X_train = np.vstack(X_train_list)
    Y_train = np.hstack(Y_train_list)

    # Simulate test data from Uniform[0, 1/k]^S (always use original coefficients)
    X_test = np.random.uniform(0, 1/k, size=(ntest, S))
    Y_test = simulate_outcome(X_test)

    # Train Random Forest with explicit hyperparameters
    # max_depth=None: unlimited depth (trees grow until leaves are pure)
    # max_features=S//3: use 20//3=6 features per split
    # min_samples_split=2: minimum samples to split a node
    # min_samples_leaf=1: minimum samples in a leaf
    rf = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=None,
        max_features=S//3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None
    )
    rf.fit(X_train, Y_train)

    # Predict on training data
    Y_train_pred = rf.predict(X_train)
    train_mse = np.mean((Y_train - Y_train_pred)**2)
    train_baseline_mse = np.mean((Y_train - np.mean(Y_train))**2)

    # Predict on test data
    Y_test_pred = rf.predict(X_test)
    test_mse = np.mean((Y_test - Y_test_pred)**2)
    test_baseline_mse = np.mean((Y_test - np.mean(Y_train))**2)

    return train_mse, train_baseline_mse, test_mse, test_baseline_mse


def run_simulation():
    """Run the complete simulation with diagnostics"""
    results = {
        f'k={k}': [] for k in k_values
    }

    # Store raw MSE values for diagnostics (training and test)
    ensemble_train_mses = {f'k={k}': [] for k in k_values}
    ensemble_test_mses = {f'k={k}': [] for k in k_values}
    merged_train_mses = {f'k={k}': [] for k in k_values}
    merged_test_mses = {f'k={k}': [] for k in k_values}
    train_baseline_mses = {f'k={k}': [] for k in k_values}
    test_baseline_mses = {f'k={k}': [] for k in k_values}

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Running simulations for k={k}")
        print(f"{'='*60}")

        for rep in range(nrep):
            print(f"  [k={k:2d}, Rep {rep+1:2d}/{nrep}] Running...", end='', flush=True)

            # Run both methods
            ens_train_mse, ens_train_base, ens_test_mse, ens_test_base = ensemble_method(k)
            mer_train_mse, mer_train_base, mer_test_mse, mer_test_base = merged_method(k)

            # Store raw values
            ensemble_train_mses[f'k={k}'].append(ens_train_mse)
            ensemble_test_mses[f'k={k}'].append(ens_test_mse)
            merged_train_mses[f'k={k}'].append(mer_train_mse)
            merged_test_mses[f'k={k}'].append(mer_test_mse)
            train_baseline_mses[f'k={k}'].append(ens_train_base)
            test_baseline_mses[f'k={k}'].append(ens_test_base)

            # Compute test R² values
            ens_test_r2 = 1 - (ens_test_mse / ens_test_base)
            mer_test_r2 = 1 - (mer_test_mse / mer_test_base)

            # Compute training R² values
            ens_train_r2 = 1 - (ens_train_mse / ens_train_base)
            mer_train_r2 = 1 - (mer_train_mse / mer_train_base)

            # Compute ratio
            ratio = ens_test_mse / mer_test_mse
            results[f'k={k}'].append(ratio)

            print(f" Test MSE: E={ens_test_mse:.3f} M={mer_test_mse:.3f} | Train R²: E={ens_train_r2:.3f} M={mer_train_r2:.3f} | Test R²: E={ens_test_r2:.3f} M={mer_test_r2:.3f}")

    # Convert to DataFrames
    results_df = pd.DataFrame(results)
    ensemble_train_df = pd.DataFrame(ensemble_train_mses)
    ensemble_test_df = pd.DataFrame(ensemble_test_mses)
    merged_train_df = pd.DataFrame(merged_train_mses)
    merged_test_df = pd.DataFrame(merged_test_mses)
    train_baseline_df = pd.DataFrame(train_baseline_mses)
    test_baseline_df = pd.DataFrame(test_baseline_mses)

    return (results_df, ensemble_train_df, ensemble_test_df,
            merged_train_df, merged_test_df, train_baseline_df, test_baseline_df)


def print_diagnostics(ens_train_df, ens_test_df, mer_train_df, mer_test_df,
                      train_base_df, test_base_df):
    """Print diagnostic information about MSE values and R² (training vs test)"""
    print("\n" + "="*80)
    print("DIAGNOSTIC INFORMATION - TRAINING vs TEST PERFORMANCE")
    print("="*80)

    print("\n1. Mean Test MSE Values (lower is better):")
    print("-" * 80)
    print(f"{'k':<5} {'Ensemble MSE':<15} {'Merged MSE':<15} {'Test Baseline':<15}")
    print("-" * 80)

    for k in k_values:
        col = f'k={k}'
        ens_mean = ens_test_df[col].mean()
        mer_mean = mer_test_df[col].mean()
        base_mean = test_base_df[col].mean()
        print(f"{k:<5} {ens_mean:<15.4f} {mer_mean:<15.4f} {base_mean:<15.4f}")

    print("\n2. TRAINING R² (Proportion of variance explained on TRAINING data):")
    print("-" * 80)
    print(f"{'k':<5} {'Ensemble Train R²':<20} {'Merged Train R²':<20}")
    print("-" * 80)

    for k in k_values:
        col = f'k={k}'
        ens_train_r2 = 1 - (ens_train_df[col].mean() / train_base_df[col].mean())
        mer_train_r2 = 1 - (mer_train_df[col].mean() / train_base_df[col].mean())
        print(f"{k:<5} {ens_train_r2:<20.4f} {mer_train_r2:<20.4f}")

    print("\n3. TEST R² (Proportion of variance explained on TEST data):")
    print("-" * 80)
    print(f"{'k':<5} {'Ensemble Test R²':<20} {'Merged Test R²':<20}")
    print("-" * 80)

    for k in k_values:
        col = f'k={k}'
        ens_test_r2 = 1 - (ens_test_df[col].mean() / test_base_df[col].mean())
        mer_test_r2 = 1 - (mer_test_df[col].mean() / test_base_df[col].mean())
        print(f"{k:<5} {ens_test_r2:<20.4f} {mer_test_r2:<20.4f}")

    print("\n4. OVERFITTING CHECK (Train R² - Test R², should be close to 0):")
    print("-" * 80)
    print(f"{'k':<5} {'Ensemble Gap':<20} {'Merged Gap':<20} {'Interpretation'}")
    print("-" * 80)

    for k in k_values:
        col = f'k={k}'
        ens_train_r2 = 1 - (ens_train_df[col].mean() / train_base_df[col].mean())
        ens_test_r2 = 1 - (ens_test_df[col].mean() / test_base_df[col].mean())
        mer_train_r2 = 1 - (mer_train_df[col].mean() / train_base_df[col].mean())
        mer_test_r2 = 1 - (mer_test_df[col].mean() / test_base_df[col].mean())

        ens_gap = ens_train_r2 - ens_test_r2
        mer_gap = mer_train_r2 - mer_test_r2

        # Interpretation
        if abs(ens_gap) < 0.05 and abs(mer_gap) < 0.05:
            interp = "✓ No overfitting"
        elif ens_gap > 0.1 or mer_gap > 0.1:
            interp = "⚠ Overfitting"
        else:
            interp = "~ Mild overfitting"

        print(f"{k:<5} {ens_gap:<20.4f} {mer_gap:<20.4f} {interp}")


def create_plot(results_df, output_file='rf_simulation_linear_equal_samples_plot.png'):
    """Create plot with empirical MSE ratios, confidence bands, and theoretical line"""
    k_vals = k_values
    means = []
    lower_ci = []
    upper_ci = []

    for k in k_values:
        col_name = f'k={k}'
        data = results_df[col_name].values

        # Mean
        mean = np.mean(data)
        means.append(mean)

        # 95% confidence interval
        ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
        lower_ci.append(ci[0])
        upper_ci.append(ci[1])

    # Create plot in style of Figure 2a from COLT paper
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot 95% confidence band first (so it's behind the line)
    ax.fill_between(k_vals, lower_ci, upper_ci, alpha=0.2, color='#2E75B6',
                     label='95% CI')

    # Plot empirical MSE ratio with points connected by lines
    ax.plot(k_vals, means, 'o-', color='#2E75B6', linewidth=2, markersize=7,
            markerfacecolor='#2E75B6', markeredgecolor='white', markeredgewidth=1.5,
            label='Empirical', zorder=3)

    # Plot theoretical ratio = 1/2 with dotted line
    ax.axhline(y=0.5, color='#C55A11', linestyle=':', linewidth=2,
               label='Theoretical (1/2)', zorder=2)

    # Labels and title
    ax.set_xlabel('Number of Clusters', fontsize=13)
    ax.set_ylabel('Average Ratio of Ensemble MSE to Merged MSE', fontsize=13)
    perturb_str = " (Perturbed)" if PERTURB_COEFFICIENTS else ""
    ax.set_title(f'Uniform Clusters{perturb_str}', fontsize=14, fontweight='bold')
    ax.set_xticks(k_vals)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, frameon=True, loc='best')

    # Set y-axis limits for better visualization
    ax.set_ylim([0.3, 0.7])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")

    return fig, ax


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running simulations...")
    print("="*60)

    (results_df, ens_train_df, ens_test_df, mer_train_df, mer_test_df,
     train_base_df, test_base_df) = run_simulation()

    # Save results
    results_df.to_csv('rf_simulation_linear_equal_samples_ratios.csv', index=False)
    ens_train_df.to_csv('rf_simulation_linear_equal_samples_ens_train_mse.csv', index=False)
    ens_test_df.to_csv('rf_simulation_linear_equal_samples_ens_test_mse.csv', index=False)
    mer_train_df.to_csv('rf_simulation_linear_equal_samples_mer_train_mse.csv', index=False)
    mer_test_df.to_csv('rf_simulation_linear_equal_samples_mer_test_mse.csv', index=False)

    print("\n" + "="*60)
    print("FILES SAVED")
    print("="*60)
    print("  - rf_simulation_linear_equal_samples_ratios.csv")
    print("  - rf_simulation_linear_equal_samples_ens_train_mse.csv")
    print("  - rf_simulation_linear_equal_samples_ens_test_mse.csv")
    print("  - rf_simulation_linear_equal_samples_mer_train_mse.csv")
    print("  - rf_simulation_linear_equal_samples_mer_test_mse.csv")

    # Print diagnostics
    print_diagnostics(ens_train_df, ens_test_df, mer_train_df, mer_test_df,
                     train_base_df, test_base_df)

    # Summary statistics
    print("\n" + "="*60)
    print("MSE RATIO SUMMARY (Ensemble / Merged)")
    print("="*60)
    print("\nRatio < 1.0 means Ensemble is BETTER (lower MSE)")
    print("Ratio > 1.0 means Merged is BETTER (lower MSE)")
    print("Theoretical prediction: 0.5 (Ensemble 2x better)\n")

    for k in k_values:
        col_name = f'k={k}'
        mean_ratio = results_df[col_name].mean()
        std_ratio = results_df[col_name].std()
        print(f"  k={k:2d}: {mean_ratio:.4f} ± {std_ratio:.4f}")

    # Create plot
    print("\n" + "="*60)
    print("Creating plot...")
    print("="*60)
    create_plot(results_df)

    print("\n" + "="*60)
    print("Simulation completed successfully!")
    print("="*60)
