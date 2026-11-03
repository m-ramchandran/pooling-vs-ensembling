"""
Create a composite figure with Theorem 3 (left) and Theorem 4 (right) validation plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from figure_2a_theorem3 import validate_theorem3_with_mse
from figure_2b_theorem4 import validate_theorem4_with_mse


def create_composite_plot():
    """Generate composite plot with both Theorem 3 and Theorem 4 results."""

    print("Generating results for both theorems...")
    print("\n" + "="*80)
    print("RUNNING THEOREM 3 VALIDATION (Fixed-Dimensional)")
    print("="*80)
    results_t3 = validate_theorem3_with_mse()

    print("\n" + "="*80)
    print("RUNNING THEOREM 4 VALIDATION (High-Dimensional)")
    print("="*80)
    results_t4 = validate_theorem4_with_mse()

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # ============ LEFT PLOT: THEOREM 3 ============
    k_values = np.array([r['k'] for r in results_t3])
    empirical_t3 = np.array([r['empirical'] for r in results_t3])
    ci_lower_t3 = np.array([r['ci_lower'] for r in results_t3])
    ci_upper_t3 = np.array([r['ci_upper'] for r in results_t3])

    # Empirical line with confidence band
    ax1.plot(k_values, empirical_t3, 'r-', linewidth=2.5, label='Empirical MSE Ratio', zorder=4)

    # Add dots at each k value
    ax1.scatter(k_values, empirical_t3, c='red', s=100, zorder=5, edgecolors='darkred', linewidths=1.5)

    # Confidence band
    ax1.fill_between(k_values, ci_lower_t3, ci_upper_t3, alpha=0.3, color='red', zorder=1)

    # Add invisible line for legend entry
    ax1.plot([], [], color='red', alpha=0.3, linewidth=10, label='95% Confidence Band')

    # Reference line at 1
    ax1.axhline(y=1, color='k', linestyle='--', alpha=0.5, linewidth=1.5, zorder=0, label='MSE Ratio = 1')

    ax1.set_xlabel('Number of Clusters (k)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MSE Ratio (Ensemble / Merged)', fontsize=14, fontweight='bold')
    ax1.set_title('Fixed-Dimensional Regime',
                  fontsize=16, fontweight='bold', pad=20)

    # Add subplot label A
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes,
             fontsize=20, fontweight='bold', va='top')

    ax1.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1.5, 8.5)
    ax1.set_xticks(k_values)

    # ============ RIGHT PLOT: THEOREM 4 ============
    gammas = np.array([r['gamma'] for r in results_t4])
    theoretical_t4 = np.array([r['theoretical'] for r in results_t4])
    empirical_t4 = np.array([r['empirical'] for r in results_t4])
    ci_lower_t4 = np.array([r['ci_lower'] for r in results_t4])
    ci_upper_t4 = np.array([r['ci_upper'] for r in results_t4])

    # Empirical line with confidence band (plot first for legend order)
    ax2.plot(gammas, empirical_t4, 'r-', linewidth=2.5, label='Empirical MSE Ratio', zorder=4)

    # Add dots at each gamma value
    ax2.scatter(gammas, empirical_t4, c='red', s=80, zorder=5, edgecolors='darkred', linewidths=1.5)

    # Confidence band
    ax2.fill_between(gammas, ci_lower_t4, ci_upper_t4, alpha=0.3, color='red', zorder=1)

    # Add invisible line for legend entry
    ax2.plot([], [], color='red', alpha=0.3, linewidth=10, label='95% Confidence Band')

    # Theoretical line (smooth) - plot last for legend order
    gamma_smooth = np.linspace(0.05, 0.45, 100)
    theoretical_smooth = [(1-g)/(1-2*g) for g in gamma_smooth]
    ax2.plot(gamma_smooth, theoretical_smooth, 'b-', linewidth=2.5,
             label='Theoretical Ratio: (1-γ)/(1-2γ)', zorder=3)

    # Reference line at 1
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, linewidth=1, zorder=0)

    ax2.set_xlabel('γ = S/n', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MSE Ratio (Ensemble / Merged)', fontsize=14, fontweight='bold')
    ax2.set_title('High-Dimensional Regime',
                  fontsize=16, fontweight='bold', pad=20)

    # Add subplot label B
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes,
             fontsize=20, fontweight='bold', va='top')

    ax2.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.04, 0.46)

    # Add overall title
    fig.suptitle('Average MSE Ratio for Linear Regression Learners with Gaussian Clusters',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=4.0)
    plt.savefig('composite_theorem3_theorem4.pdf', bbox_inches='tight')
    print(f"\n\nComposite plot saved to: composite_theorem3_theorem4.pdf")
    plt.close()


if __name__ == "__main__":
    import time
    start_time = time.time()

    create_composite_plot()

    elapsed_time = time.time() - start_time
    print(f"\nTotal computation time: {elapsed_time/60:.1f} minutes")
