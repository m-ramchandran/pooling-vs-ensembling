# Ensemble vs Merged Learners: Validating Theorems 3 & 4

Implementation and empirical validation of theoretical results from the paper "The Bias-Variance Tradeoff Determines Benefit of Ensemble Learners versus Single-Model Learners on Clustered Data."

## Overview

This repository validates the bias-variance tradeoff principle for **Linear Regression** on clustered data:

**Linear Regression (Theorems 3 & 4):**
- Bias = 0 (correctly specified model) → Variance dominates MSE
- **Merged outperforms Ensemble**
- Fixed-dim: MSE ratio > 1 for all k
- High-dim: MSE ratio = (1-γ)/(1-2γ)

## Repository Structure

```
├── COLT_2026_Template_2 (1).pdf          # Original paper
│
├── # Linear Regression Validation (Theorems 3 & 4)
├── linear_regression_predictions.py       # Core prediction implementations
├── figure_2a_theorem3.py                  # Theorem 3 validation (fixed-dim)
├── figure_2b_theorem4.py                  # Theorem 4 validation (high-dim)
├── create_figure_2.py                     # Creates composite figure
├── composite_theorem3_theorem4.pdf        # Visualization
│
├── README.md                              # This file
└── .gitignore                             # Git ignore rules
```

## Quick Start

**Run full validation (composite plot):**

```bash
python create_figure_2.py
```

**Output:**
- Console output with detailed results for both theorems
- `composite_theorem3_theorem4.pdf` - publication-ready composite figure

**Computation time:** ~2 minutes

**Run individual validations:**

```bash
# Theorem 3 (Fixed-Dimensional)
python figure_2a_theorem3.py

# Theorem 4 (High-Dimensional)
python figure_2b_theorem4.py
```

## Simulation Parameters

### Theorem 3 (Fixed-Dimensional Regime)
```
Sample size:          n = 5000
Features:             S = 100 (fixed)
Clusters:             k = 2, 3, 4, 5, 6, 7, 8
Cluster means:        Random norm-1 vectors
Trials per k:         100 independent datasets
Test points/trial:    1000
Total predictions/k:  100,000
```

### Theorem 4 (High-Dimensional Regime)
```
Sample size:          n = 1000
Clusters:             k = 2 (Gaussian with μ₁ = -μ₂)
Gamma values:         γ ∈ [0.05, 0.45] (9 values)
Trials per γ:         100 independent datasets
Test points/trial:    1000
Total predictions/γ:  100,000
```

## Implementation Details

### Core Functions

**`linear_regression_predictions.py`**

```python
def predict_ensemble(x_star, X_clusters, Y_clusters):
    """
    Ensemble prediction with Inverse Variance Weighting (IVW).

    Formula:
    Ŷ_E = [Σ_b w_b]⁻¹ Σ_b w_b · Ŷ_b
    where w_b = [x*ᵀ(X_bᵀX_b)⁻¹x*]⁻¹
    """

def predict_merged(x_star, X_merged, Y_merged):
    """
    Merged prediction using all data.

    Formula:
    Ŷ_M = x*ᵀ(XᵀX)⁻¹XᵀY
    """
```

### Validation Approach

**Computing MSE from Actual Predictions:**

1. Generate training data with known β and cluster means
2. Fit Ensemble and Merged models
3. Generate test data with true outcomes Y* = X*β
4. Make predictions Ŷ_E and Ŷ_M
5. Compute MSE = E[(Ŷ - Y*)²] from squared errors

This is **not circular** - we compute actual prediction errors, not just closed-form variance expressions.

### Optimization

**Safe Caching (60x speedup):**

Within each trial:
- Generate ONE training dataset
- Compute matrix inversions ONCE
- Reuse for all test predictions (1000 per trial)

Reduced time from ~30 minutes to ~30 seconds per theorem with zero impact on accuracy.

## Results

### Theorem 3: Fixed-Dimensional Regime

| k | n/k  | Empirical | 95% CI              | Ratio > 1? |
|---|------|-----------|---------------------|------------|
| 2 | 2500 | 1.0173    | [1.0108, 1.0239]    | ✓ YES      |
| 3 | 1666 | 1.0452    | [1.0366, 1.0538]    | ✓ YES      |
| 4 | 1250 | 1.0686    | [1.0576, 1.0797]    | ✓ YES      |
| 5 | 1000 | 1.0943    | [1.0823, 1.1063]    | ✓ YES      |
| 6 | 833  | 1.1212    | [1.1059, 1.1366]    | ✓ YES      |
| 7 | 714  | 1.1508    | [1.1358, 1.1659]    | ✓ YES      |
| 8 | 625  | 1.1751    | [1.1571, 1.1930]    | ✓ YES      |

**Key Findings:**
- ✅ MSE ratio > 1 for all k (Merged always outperforms)
- ✅ Ratio increases with k (from 1.02 to 1.18)
- ✅ All 95% CIs entirely above 1.0

### Theorem 4: High-Dimensional Regime

| γ (S/n) | S   | Theoretical | Empirical | 95% CI              | Error  |
|---------|-----|-------------|-----------|---------------------|--------|
| 0.05    | 50  | 1.0556      | 1.0502    | [1.0375, 1.0629]    | 0.0053 |
| 0.10    | 100 | 1.1250      | 1.1362    | [1.1172, 1.1552]    | 0.0112 |
| 0.15    | 150 | 1.2143      | 1.2204    | [1.2012, 1.2395]    | 0.0061 |
| 0.20    | 200 | 1.3333      | 1.3364    | [1.3061, 1.3667]    | 0.0031 |
| 0.25    | 250 | 1.5000      | 1.4982    | [1.4687, 1.5276]    | 0.0018 |
| 0.30    | 300 | 1.7500      | 1.8135    | [1.7701, 1.8568]    | 0.0635 |
| 0.35    | 350 | 2.1667      | 2.1400    | [2.0907, 2.1893]    | 0.0267 |
| 0.40    | 400 | 3.0000      | 2.9679    | [2.8902, 3.0456]    | 0.0321 |
| 0.45    | 450 | 5.5000      | 5.6174    | [5.4075, 5.8274]    | 0.1174 |

**Key Findings:**
- ✅ Empirical ratios closely match theoretical predictions
- ✅ Mean absolute error: 0.0297
- ✅ Ratio increases dramatically as γ approaches 0.5
- ✅ Merged advantage grows with dimensionality

## Theoretical Background

### Why Merged Outperforms Ensemble

When the model is **correctly specified** (linear outcome, linear model):
- **Bias = 0** for both Ensemble and Merged
- **MSE = Variance** only

**The Merged learner is better because:**
1. ✓ Pools data from all clusters → larger effective sample size
2. ✓ Exploits cross-cluster covariance structure
3. ✓ More stable matrix inversion

**The Ensemble:**
1. ✗ Trains on smaller subsets (n/k per cluster)
2. ✗ Ignores cross-cluster information
3. ✗ Higher variance from reduced sample size

### Theorem 3 (Fixed-Dimensional)

For k clusters with norm-1 means, as n → ∞ with S fixed:
```
MSE(Ŷ_E) / MSE(Ŷ_M) > 1
```
Advantage increases with k (more clusters = smaller subsets per cluster).

### Theorem 4 (High-Dimensional)

For k=2 clusters with μ₁ = -μ₂, as n,S → ∞ with S/n = γ ∈ [0, 1/2):
```
Var(Ŷ_E) / Var(Ŷ_M) → (1-γ) / (1-2γ) > 1
```
Advantage increases dramatically as γ approaches 0.5.

### Inverse Variance Weighting

For each cluster b:
1. **Variance**: var_b = x*ᵀ(X_bᵀX_b)⁻¹x*
2. **Weight**: w_b = 1/var_b
3. **Prediction**: Ŷ_E = Σ(w_b · Ŷ_b) / Σ(w_b)

This is optimal weighting for combining unbiased estimators with different variances.

## Visualization

![Composite Validation](composite_theorem3_theorem4.pdf)

**Panel A (Fixed-Dimensional):** Shows ratio vs. k with confidence bands
**Panel B (High-Dimensional):** Shows empirical and theoretical curves vs. γ

Both panels confirm that Merged consistently outperforms Ensemble.

## Requirements

```
numpy
matplotlib
scipy
```

Install with:
```bash
pip install numpy matplotlib scipy
```

## References

- **Paper**: "The Bias-Variance Tradeoff Determines Benefit of Ensemble Learners versus Single-Model Learners on Clustered Data" (COLT 2026, under review)
- **Theorem 3**: Fixed-dimensional linear regression (page 8, Lemma 9 pages 26-27)
- **Theorem 4**: High-dimensional linear regression (page 9, Appendix B.4 pages 31-33)
- **Original R code**: https://github.com/m-ramchandran/ensembling-vs-merging

## Citation

If you use this code, please cite the original paper (COLT 2026, under review).

## License

MIT License

## Contact

For questions about the implementation, please open an issue or refer to the paper.
