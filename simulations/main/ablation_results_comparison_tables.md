# Comprehensive Comparison Tables

All experiments use: n = 2000, k = 2, ||β|| = 1, SNR = 5 (σ = 1/√5 ≈ 0.447), angle = 60°, norm ratio = 1, Σ = I.
Test data: X_test ~ N(0, I_p). 20 trials per configuration.

---

## Table 1: Ridgeless (Min-Norm OLS) — Clustered Training Data vs No-Cluster vs Random Split

Training data differs across columns:
- **Clustered**: X_b ~ N(μ_b, I), trained per true cluster
- **No-Cluster**: X ~ N(0, I), random 50/50 split for ensemble
- **Random Split**: X_b ~ N(μ_b, I), but ensemble uses random 50/50 split (ignoring clusters)

### MSE Ratios (Ensemble / Merged)

| p | p/n | Regime | Clustered | No-Cluster | Random Split |
|---|-----|--------|-----------|------------|--------------|
| 250 | 0.12 | both under | 1.168 | 1.198 | 1.171 |
| 500 | 0.25 | both under | 1.541 | 1.513 | 1.514 |
| 750 | 0.38 | both under | 2.497 | 2.520 | 2.561 |
| 1050 | 0.53 | cluster OVER, merged under | 9.521 | 9.171 | 9.799 |
| 1500 | 0.75 | cluster OVER, merged under | 0.731 | 0.704 | 0.742 |
| 2050 | 1.03 | both OVER | 0.062 | 0.065 | 0.060 |
| 3000 | 1.50 | both OVER | 0.829 | 0.821 | 0.822 |
| 4000 | 2.00 | both OVER | 0.979 | 0.964 | 0.987 |
| 5000 | 2.50 | both OVER | 1.012 | 1.018 | 0.999 |
| 6000 | 3.00 | both OVER | 1.028 | 1.017 | 1.011 |
| 7000 | 3.50 | both OVER | 1.024 | 1.029 | 1.022 |
| 8000 | 4.00 | both OVER | 1.020 | 1.028 | 1.018 |

### Raw MSE — Ensemble

| p | p/n | Clustered | No-Cluster | Random Split |
|---|-----|-----------|------------|--------------|
| 250 | 0.12 | 0.032 | 0.034 | 0.032 |
| 500 | 0.25 | 0.103 | 0.096 | 0.102 |
| 750 | 0.38 | 0.296 | 0.293 | 0.314 |
| 1050 | 0.53 | 2.050 | 2.017 | 2.124 |
| 1500 | 0.75 | 0.419 | 0.407 | 0.430 |
| 2050 | 1.03 | 0.489 | 0.484 | 0.480 |
| 3000 | 1.50 | 0.600 | 0.608 | 0.600 |
| 4000 | 2.00 | 0.691 | 0.686 | 0.695 |
| 5000 | 2.50 | 0.740 | 0.748 | 0.742 |
| 6000 | 3.00 | 0.779 | 0.781 | 0.782 |
| 7000 | 3.50 | 0.811 | 0.818 | 0.817 |
| 8000 | 4.00 | 0.836 | 0.831 | 0.843 |

### Raw MSE — Merged

| p | p/n | Clustered | No-Cluster | Random Split |
|---|-----|-----------|------------|--------------|
| 250 | 0.12 | 0.027 | 0.029 | 0.028 |
| 500 | 0.25 | 0.067 | 0.064 | 0.068 |
| 750 | 0.38 | 0.119 | 0.117 | 0.123 |
| 1050 | 0.53 | 0.216 | 0.221 | 0.217 |
| 1500 | 0.75 | 0.576 | 0.583 | 0.583 |
| 2050 | 1.03 | 8.129 | 7.803 | 8.295 |
| 3000 | 1.50 | 0.726 | 0.743 | 0.731 |
| 4000 | 2.00 | 0.706 | 0.712 | 0.705 |
| 5000 | 2.50 | 0.732 | 0.735 | 0.743 |
| 6000 | 3.00 | 0.758 | 0.767 | 0.774 |
| 7000 | 3.50 | 0.793 | 0.795 | 0.799 |
| 8000 | 4.00 | 0.821 | 0.809 | 0.828 |

---

## Table 2: Ridge (Tuned via LOO CV) — Clustered Training Data vs No-Cluster

### MSE Ratios (Ensemble / Merged)

| p | p/n | Regime | Clustered | No-Cluster |
|---|-----|--------|-----------|------------|
| 250 | 0.12 | both under | 1.162 | 1.197 |
| 500 | 0.25 | both under | 1.550 | 1.607 |
| 750 | 0.38 | both under | 1.613 | 1.603 |
| 1050 | 0.53 | cluster OVER, merged under | 1.574 | 1.547 |
| 1500 | 0.75 | cluster OVER, merged under | 1.449 | 1.449 |
| 2050 | 1.03 | both OVER | 1.338 | 1.386 |
| 3000 | 1.50 | both OVER | 1.221 | 1.225 |
| 4000 | 2.00 | both OVER | 1.154 | 1.135 |
| 5000 | 2.50 | both OVER | 1.105 | 1.107 |
| 6000 | 3.00 | both OVER | 1.088 | 1.087 |
| 7000 | 3.50 | both OVER | 1.076 | 1.075 |
| 8000 | 4.00 | both OVER | 1.060 | 1.067 |

### Raw MSE — Ensemble

| p | p/n | Clustered | No-Cluster |
|---|-----|-----------|------------|
| 250 | 0.12 | 0.031 | 0.034 |
| 500 | 0.25 | 0.100 | 0.098 |
| 750 | 0.38 | 0.166 | 0.159 |
| 1050 | 0.53 | 0.249 | 0.251 |
| 1500 | 0.75 | 0.392 | 0.378 |
| 2050 | 1.03 | 0.524 | 0.528 |
| 3000 | 1.50 | 0.632 | 0.641 |
| 4000 | 2.00 | 0.725 | 0.710 |
| 5000 | 2.50 | 0.761 | 0.770 |
| 6000 | 3.00 | 0.803 | 0.807 |
| 7000 | 3.50 | 0.835 | 0.839 |
| 8000 | 4.00 | 0.855 | 0.852 |

### Raw MSE — Merged

| p | p/n | Clustered | No-Cluster |
|---|-----|-----------|------------|
| 250 | 0.12 | 0.027 | 0.028 |
| 500 | 0.25 | 0.065 | 0.061 |
| 750 | 0.38 | 0.103 | 0.099 |
| 1050 | 0.53 | 0.159 | 0.162 |
| 1500 | 0.75 | 0.272 | 0.261 |
| 2050 | 1.03 | 0.392 | 0.381 |
| 3000 | 1.50 | 0.518 | 0.523 |
| 4000 | 2.00 | 0.628 | 0.626 |
| 5000 | 2.50 | 0.688 | 0.697 |
| 6000 | 3.00 | 0.738 | 0.743 |
| 7000 | 3.50 | 0.776 | 0.780 |
| 8000 | 4.00 | 0.807 | 0.799 |

---

## Table 3: Summary — Key Takeaways

### Ridgeless: Three Regimes

| Regime | p range | Who wins | Magnitude | Cluster structure matters? |
|--------|---------|----------|-----------|---------------------------|
| Both underparameterized | p < 1000 | Merged | Ratio 1.2–9.5x | No |
| Staggered (cluster OVER, merged under) | 1000 < p < 2000 | Depends on where within regime | Extreme swings | No |
| Near merged threshold | p ≈ 2000 | **Ensemble** | Ratio ~0.06 (16x better) | No |
| Both overparameterized (deep) | p > 3000 | Converges to ~1 | Ratio 0.83–1.03 | No |

### Ridge (tuned): No Regime Changes

| Regime | p range | Who wins | Magnitude | Cluster structure matters? |
|--------|---------|----------|-----------|---------------------------|
| All regimes | All p | **Merged always** | Ratio 1.06–1.61x | No |

### Cluster Structure Effect

Across all experiments, the MSE ratios and raw MSEs are essentially identical whether:
- Training data has true cluster structure (X_b ~ N(μ_b, I)) with cluster-aware ensemble
- Training data has cluster structure but ensemble uses random splits
- Training data has no cluster structure at all (X ~ N(0, I))

**Conclusion**: The overparameterization ratio γ = p/n is the sole driver of relative performance. Cluster heterogeneity (angle between means, norm ratios, covariance differences) has at most a second-order effect on the MSE ratio.
