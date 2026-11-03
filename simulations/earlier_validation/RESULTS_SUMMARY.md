# Ensemble vs. Merged: Complete Experiment Results

All experiments compare **MSE(Ensemble) / MSE(Merged)**. Ratio > 1 means merged outperforms ensemble.

---

## 1. OLS Linear Regression — Theorem 3: Fixed-Dimensional Regime

Measures how the MSE ratio changes as the number of clusters k increases, while total sample size n is held fixed (so each cluster gets fewer samples). With S=100 features and n=5,000 total samples, the regime is low-dimensional (p/n=0.02) and OLS is well-specified.

| Parameter | Value |
|-----------|-------|
| Estimator | OLS (ordinary least squares) |
| Total samples (n) | 5,000 |
| Features (S) | 100 |
| p/n ratio | 0.02 |
| Clusters (k) | varies: 2, 3, 4, 5, 6, 7, 8 |
| Samples per cluster | n/k (equal split) |
| Cluster means | Random unit-norm vectors |
| True beta | Random, unit-norm |
| Covariance | Identity (I) per cluster |
| Noise std | 0.1 |
| Ensemble weighting | Inverse-variance weighting (IVW) |
| Trials per k | 100 |
| Test points per trial | 1,000 |

| k | n/k  | MSE Ratio | 95% CI           | Ratio > 1? |
|---|------|-----------|------------------|------------|
| 2 | 2500 | 1.0173    | [1.0108, 1.0239] | YES |
| 3 | 1666 | 1.0452    | [1.0366, 1.0538] | YES |
| 4 | 1250 | 1.0686    | [1.0576, 1.0797] | YES |
| 5 | 1000 | 1.0943    | [1.0823, 1.1063] | YES |
| 6 | 833  | 1.1212    | [1.1059, 1.1366] | YES |
| 7 | 714  | 1.1508    | [1.1358, 1.1659] | YES |
| 8 | 625  | 1.1751    | [1.1571, 1.1930] | YES |

---

## 2. OLS Linear Regression — Theorem 4: High-Dimensional Regime

Measures how the MSE ratio changes as the dimensionality ratio gamma = S/n grows from 0.05 toward 0.5, with k=2 clusters and antipodal means (mu_1 = -mu_2). Empirical results are compared against the closed-form theoretical prediction (1-gamma)/(1-2*gamma) from Theorem 4.

| Parameter | Value |
|-----------|-------|
| Estimator | OLS (ordinary least squares) |
| Total samples (n) | 1,000 |
| Clusters (k) | 2 |
| Cluster means | mu_1 = -mu_2 (random unit-norm) |
| gamma (S/n) | varies: 0.05 to 0.45 |
| True beta | Random, unit-norm |
| Covariance | Identity (I) per cluster |
| Noise std | 0.1 |
| Ensemble weighting | Inverse-variance weighting (IVW) |
| Theoretical ratio | (1-gamma)/(1-2*gamma) |
| Trials per gamma | 100 |
| Test points per trial | 1,000 |

| gamma | S   | Theoretical | Empirical | 95% CI           | Abs Error |
|-------|-----|-------------|-----------|------------------|-----------|
| 0.05  | 50  | 1.0556      | 1.0502    | [1.0375, 1.0629] | 0.0053 |
| 0.10  | 100 | 1.1250      | 1.1362    | [1.1172, 1.1552] | 0.0112 |
| 0.15  | 150 | 1.2143      | 1.2204    | [1.2012, 1.2395] | 0.0061 |
| 0.20  | 200 | 1.3333      | 1.3364    | [1.3061, 1.3667] | 0.0031 |
| 0.25  | 250 | 1.5000      | 1.4982    | [1.4687, 1.5276] | 0.0018 |
| 0.30  | 300 | 1.7500      | 1.8135    | [1.7701, 1.8568] | 0.0635 |
| 0.35  | 350 | 2.1667      | 2.1400    | [2.0907, 2.1893] | 0.0267 |
| 0.40  | 400 | 3.0000      | 2.9679    | [2.8902, 3.0456] | 0.0321 |
| 0.45  | 450 | 5.5000      | 5.6174    | [5.4075, 5.8274] | 0.1174 |

---

## 3. Ridge Regression — Angle Study, Fixed-Dimensional

Measures how the MSE ratio changes as the angle between the two cluster mean vectors varies from 0 deg (identical means) to 90 deg (orthogonal means), with both norms fixed at 1. Uses ridge regression with per-learner CV tuning in the low-dimensional regime (p/n=0.02).

| Parameter | Value |
|-----------|-------|
| Estimator | Ridge regression (sklearn RidgeCV, LOO-based) |
| Lambda grid | np.logspace(-4, 4, 15) |
| Lambda tuning | Per learner (each cluster + merged tuned separately) |
| Total samples (n) | 5,000 |
| Features (S) | 100 |
| p/n ratio | 0.02 |
| Clusters (k) | 2 |
| Cluster means | mu_1: random unit-norm; mu_2: controlled angle, unit-norm |
| mu_2 construction | Gram-Schmidt (exact angle relative to mu_1) |
| True beta | Random, unit-norm |
| Covariance | Identity (I) per cluster |
| Noise std | 0.1 |
| Ensemble weighting | Simple average |
| Trials per angle | 100 |
| Test points per trial | 1,000 |

| Angle (deg) | MSE Ratio | 95% CI           | Ratio > 1? |
|-------------|-----------|------------------|------------|
| 0           | 1.0180    | [1.0109, 1.0251] | YES |
| 10          | 1.0241    | [1.0172, 1.0310] | YES |
| 20          | 1.0209    | [1.0147, 1.0271] | YES |
| 30          | 1.0225    | [1.0154, 1.0296] | YES |
| 40          | 1.0316    | [1.0251, 1.0382] | YES |
| 50          | 1.0209    | [1.0140, 1.0279] | YES |
| 60          | 1.0270    | [1.0204, 1.0335] | YES |
| 70          | 1.0284    | [1.0219, 1.0349] | YES |
| 80          | 1.0339    | [1.0260, 1.0418] | YES |
| 90          | 1.0362    | [1.0278, 1.0446] | YES |

---

## 4. Ridge Regression — Angle Study, High-Dimensional

Same angle sweep as Experiment 3 (0 deg to 90 deg, both norms fixed at 1), but in the high-dimensional regime (p/n=0.3) where the number of features is a substantial fraction of the sample size. Tests whether the angle between cluster means matters more when dimensionality is high.

| Parameter | Value |
|-----------|-------|
| Estimator | Ridge regression (sklearn RidgeCV, LOO-based) |
| Lambda grid | np.logspace(-4, 4, 15) |
| Lambda tuning | Per learner (each cluster + merged tuned separately) |
| Total samples (n) | 2,000 |
| Features (S) | 600 |
| p/n ratio | 0.3 |
| Clusters (k) | 2 |
| Cluster means | mu_1: random unit-norm; mu_2: controlled angle, unit-norm |
| mu_2 construction | Gram-Schmidt (exact angle relative to mu_1) |
| True beta | Random, unit-norm |
| Covariance | Identity (I) per cluster |
| Noise std | 0.1 |
| Ensemble weighting | Simple average |
| Trials per angle | 100 |
| Test points per trial | 1,000 |

| Angle (deg) | MSE Ratio | 95% CI           | Ratio > 1? |
|-------------|-----------|------------------|------------|
| 0           | 1.7380    | [1.7091, 1.7668] | YES |
| 10          | 1.7369    | [1.7109, 1.7628] | YES |
| 20          | 1.7272    | [1.6991, 1.7554] | YES |
| 30          | 1.7295    | [1.6967, 1.7623] | YES |
| 40          | 1.7583    | [1.7306, 1.7860] | YES |
| 50          | 1.7092    | [1.6813, 1.7371] | YES |
| 60          | 1.7215    | [1.6909, 1.7521] | YES |
| 70          | 1.7414    | [1.7099, 1.7729] | YES |
| 80          | 1.7363    | [1.7038, 1.7689] | YES |
| 90          | 1.7185    | [1.6916, 1.7454] | YES |

---

## 5. Ridge Regression — Norm Ratio Study, High-Dimensional

Measures how the MSE ratio changes as the norm of mu_2 grows from 1 to 10 while mu_1 is held at unit norm, with the angle fixed at 30 deg. Tests whether asymmetric cluster separation (one cluster mean much farther from the origin) affects the merging advantage in the high-dimensional regime.

| Parameter | Value |
|-----------|-------|
| Estimator | Ridge regression (sklearn RidgeCV, LOO-based) |
| Lambda grid | np.logspace(-4, 4, 15) |
| Lambda tuning | Per learner (each cluster + merged tuned separately) |
| Total samples (n) | 2,000 |
| Features (S) | 600 |
| p/n ratio | 0.3 |
| Clusters (k) | 2 |
| Angle between mu_1, mu_2 | 30 deg (fixed) |
| mu_1 norm | 1 (fixed) |
| mu_2 norm | c (varies: 1 to 10) |
| mu_2 construction | Gram-Schmidt (exact angle and norm relative to mu_1) |
| True beta | Random, unit-norm |
| Covariance | Identity (I) per cluster |
| Noise std | 0.1 |
| Ensemble weighting | Simple average |
| Trials per norm ratio | 100 |
| Test points per trial | 1,000 |

| Norm Ratio (c) | MSE Ratio | 95% CI           | Ratio > 1? |
|----------------|-----------|------------------|------------|
| 1              | 1.7331    | [1.7077, 1.7585] | YES |
| 2              | 1.7481    | [1.7203, 1.7759] | YES |
| 3              | 1.7655    | [1.7336, 1.7974] | YES |
| 4              | 1.7386    | [1.7077, 1.7695] | YES |
| 5              | 1.7838    | [1.7533, 1.8144] | YES |
| 6              | 1.7705    | [1.7379, 1.8032] | YES |
| 7              | 1.8284    | [1.7867, 1.8702] | YES |
| 8              | 1.7957    | [1.7562, 1.8352] | YES |
| 9              | 1.8965    | [1.8347, 1.9584] | YES |
| 10             | 1.8843    | [1.8326, 1.9360] | YES |

---

## 6. Ridge Regression — Dependent Covariates + Sparse Beta, High-Dimensional

Tests the merging advantage under two realistic complications: (1) the true coefficient vector is sparse (20% of entries are zero), and (2) covariates within each cluster are correlated via a compound-symmetry structure with cluster-specific correlation strengths. The multiplier d controls how much stronger the within-cluster correlation is in cluster 2 relative to cluster 1.

| Parameter | Value |
|-----------|-------|
| Estimator | Ridge regression (sklearn RidgeCV, LOO-based) |
| Lambda grid | np.logspace(-4, 4, 15) |
| Lambda tuning | Per learner (each cluster + merged tuned separately) |
| Total samples (n) | 2,000 |
| Features (S) | 600 |
| p/n ratio | 0.3 |
| Clusters (k) | 2 |
| Angle between mu_1, mu_2 | 30 deg (fixed) |
| mu_1, mu_2 norms | Both 1 |
| True beta | Random, unit-norm, then 20% of entries zeroed (sparse) |
| Covariance structure | Compound symmetry (cluster-specific) |
| Cluster 1 covariance | Sigma_1 = (1-rho)*I + rho*11^T (off-diag = rho) |
| Cluster 2 covariance | Sigma_2 = (1-rho*d)*I + rho*d*11^T (off-diag = rho*d) |
| Test data covariance | Identity (I) |
| Noise std | 0.1 |
| Ensemble weighting | Simple average |
| Trials per config | 100 |
| Test points per trial | 1,000 |

| rho  | d | Cluster 1 off-diag | Cluster 2 off-diag | MSE Ratio | 95% CI           | Ratio > 1? |
|------|---|--------------------|--------------------|-----------|------------------|------------|
| 0.15 | 1 | 0.15               | 0.15               | 1.6914    | [1.6643, 1.7185] | YES |
| 0.15 | 2 | 0.15               | 0.30               | 1.7233    | [1.6936, 1.7529] | YES |
| 0.40 | 1 | 0.40               | 0.40               | 1.6648    | [1.6382, 1.6915] | YES |
| 0.40 | 2 | 0.40               | 0.80               | 1.9952    | [1.9547, 2.0358] | YES |

---

## 7. Ridge Regression — Block-Correlation Covariance + Sparse Beta, High-Dimensional

Tests the merging advantage when each cluster has a fundamentally different correlation structure: covariates are randomly partitioned into 3 blocks (different partition per cluster), and each block is assigned its own correlation strength drawn from Uniform(0, 1). This means the two clusters differ in both which covariates are correlated with each other and how strongly, creating a realistic heterogeneous dependence structure.

| Parameter | Value |
|-----------|-------|
| Estimator | Ridge regression (sklearn RidgeCV, LOO-based) |
| Lambda grid | np.logspace(-4, 4, 15) |
| Lambda tuning | Per learner (each cluster + merged tuned separately) |
| Total samples (n) | 2,000 |
| Features (S) | 600 |
| p/n ratio | 0.3 |
| Clusters (k) | 2 |
| Angle between mu_1, mu_2 | 30 deg (fixed) |
| mu_1, mu_2 norms | Both 1 |
| True beta | Random, unit-norm, then 20% of entries zeroed (sparse) |
| Covariance structure | Block-diagonal, 3 blocks per cluster |
| Block assignments | Random partition of S features, DIFFERENT per cluster |
| Per-block rho | Drawn from Uniform(0, 0.99), DIFFERENT per cluster and block |
| Between-block correlation | 0 (block-diagonal) |
| Test data covariance | Identity (I) |
| Noise std | 0.1 |
| Ensemble weighting | Simple average |
| Trials | 100 |
| Test points per trial | 1,000 |

| MSE Ratio | 95% CI           | Std    | Ratio > 1? |
|-----------|------------------|--------|------------|
| 2.3579    | [2.2321, 2.4836] | 0.6306 | YES |

---

## 8. Ridge Regression — Shared Lambda Ablation (Experiments 6 & 7 re-run)

Tests whether the merging advantage in Experiments 6 and 7 is driven by better regularization tuning (merged has more data for CV) or by data pooling itself. Lambda is tuned on the merged data via RidgeCV, then that same lambda is used for both ensemble cluster models. All other parameters are identical to Experiments 6 and 7.

| Parameter | Value |
|-----------|-------|
| Estimator | Ridge regression |
| Lambda tuning | Tuned on MERGED data only; same lambda used for all learners |
| All other parameters | Identical to Experiments 6 and 7 |

**Compound symmetry (Experiment 6 re-run):**

| rho  | d | Per-learner lambda | Shared lambda | Diff    |
|------|---|--------------------|---------------|---------|
| 0.15 | 1 | 1.6914             | 1.7279        | +0.0365 |
| 0.15 | 2 | 1.7233             | 1.7345        | +0.0112 |
| 0.40 | 1 | 1.6648             | 1.7115        | +0.0467 |
| 0.40 | 2 | 1.9952             | 1.9874        | -0.0078 |

**Block-correlation (Experiment 7 re-run):**

| Per-learner lambda | Shared lambda | Diff    |
|--------------------|---------------|---------|
| 2.3579             | 2.3579        | 0.0000  |

**Observation:** The lambda tuning strategy has negligible impact on the MSE ratio (differences < 0.05 in all cases, with inconsistent sign). The merging advantage is almost entirely driven by data pooling — having 2x more samples to estimate coefficients — rather than any advantage in regularization parameter selection.
