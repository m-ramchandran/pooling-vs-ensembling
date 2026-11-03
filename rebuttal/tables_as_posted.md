# Tables from the rebuttal

These are the tables to use in the camera-ready appendix, copied from the rebuttal text.
Each cell is the ensemble test MSE divided by the pooled test MSE, with a 95% t interval over 100 replicates. A value above 1 means pooling did better.

## Table 1. Covariance structure by K: ridge, equal noise variance, shared beta

Posted on OpenReview (Reviewers NR51, 4ovb, 2nhs) and in the Area Chair draft.

| | k=2 | k=3 | k=4 | k=5 |
| --- | --- | --- | --- | --- |
| identity | 1.12 ± 0.02 | 1.13 ± 0.02 | 1.12 ± 0.02 | 1.10 ± 0.01 |
| diagonal, per-cluster | 1.11 ± 0.02 | 1.11 ± 0.01 | 1.09 ± 0.01 | 1.06 ± 0.005 |
| equicorr rho=0.2 | 1.15 ± 0.03 | 1.16 ± 0.02 | 1.15 ± 0.02 | 1.14 ± 0.01 |
| equicorr rho=0.5 | 1.15 ± 0.03 | 1.19 ± 0.02 | 1.18 ± 0.02 | 1.18 ± 0.02 |
| equicorr rho=0.8 | 1.25 ± 0.07 | 1.25 ± 0.04 | 1.25 ± 0.04 | 1.28 ± 0.05 |
| equicorr, per-cluster | 1.12 ± 0.02 | 1.16 ± 0.02 | 1.18 ± 0.02 | 1.19 ± 0.02 |
| factor (shared W) | 1.01 ± 0.002 | 1.00 ± 0.001 | 1.00 ± 0.001 | 1.00 ± 0.001 |
| factor (per-cluster W) | 1.01 ± 0.002 | 1.01 ± 0.001 | 1.00 ± 0.001 | 1.00 ± 0.001 |
| nonlinear | 1.19 ± 0.03 | 1.28 ± 0.04 | 1.29 ± 0.03 | 1.30 ± 0.03 |

## Table 2. Covariance structure by K: ridge, 16x spread in noise variance, shared beta

Posted on OpenReview (Reviewers NR51, 4ovb, 2nhs) and in the Area Chair draft.

| | k=2 | k=3 | k=4 | k=5 |
| --- | --- | --- | --- | --- |
| identity | 1.00 ± 0.01 | 1.02 ± 0.01 | 1.04 ± 0.01 | 1.04 ± 0.01 |
| diagonal, per-cluster | 1.02 ± 0.01 | 1.05 ± 0.01 | 1.05 ± 0.01 | 1.04 ± 0.005 |
| equicorr rho=0.2 | 1.01 ± 0.01 | 1.04 ± 0.01 | 1.04 ± 0.01 | 1.04 ± 0.01 |
| equicorr rho=0.5 | 1.01 ± 0.01 | 1.05 ± 0.01 | 1.07 ± 0.01 | 1.06 ± 0.01 |
| equicorr rho=0.8 | 1.07 ± 0.03 | 1.14 ± 0.03 | 1.11 ± 0.02 | 1.12 ± 0.02 |
| equicorr, per-cluster | 1.00 ± 0.01 | 1.03 ± 0.01 | 1.04 ± 0.01 | 1.05 ± 0.01 |
| factor (shared W) | 1.01 ± 0.002 | 1.00 ± 0.001 | 1.00 ± 0.001 | 1.00 ± 0.001 |
| factor (per-cluster W) | 1.01 ± 0.002 | 1.00 ± 0.001 | 1.00 ± 0.001 | 1.00 ± 0.001 |
| nonlinear | 1.04 ± 0.01 | 1.09 ± 0.02 | 1.13 ± 0.02 | 1.13 ± 0.01 |

## Table 3. Shared vs opposite-sign beta: ridgeless, identity covariance, equal noise

Posted on OpenReview (Reviewer 4ovb).

| | k=2 | k=3 | k=4 | k=5 |
| --- | --- | --- | --- | --- |
| beta shared | 1.13 ± 0.03 | 1.14 ± 0.02 | 1.14 ± 0.02 | 1.11 ± 0.01 |
| beta opposite-sign | 1.30 ± 0.07 | 1.43 ± 0.07 | 1.60 ± 0.06 | 1.72 ± 0.07 |

## Table 4. Ensembling scheme by noise: ridge, identity covariance, shared beta

In the HackMD Area Chair draft only (T3 there).

| | k=2 | k=3 | k=4 | k=5 |
| --- | --- | --- | --- | --- |
| equal sigma^2, simple avg | 1.95 ± 0.10 | 2.02 ± 0.08 | 1.98 ± 0.07 | 1.99 ± 0.07 |
| equal sigma^2, inverse-var | 1.95 ± 0.10 | 1.94 ± 0.09 | 1.86 ± 0.07 | 1.91 ± 0.08 |
| 16x spread, simple avg | 1.71 ± 0.08 | 1.73 ± 0.07 | 1.85 ± 0.07 | 1.75 ± 0.04 |
| 16x spread, inverse-var | 2.70 ± 0.19 | 1.70 ± 0.08 | 1.63 ± 0.07 | 1.52 ± 0.03 |

## Table 5. Shared vs opposite-sign beta: ridge, identity covariance, equal noise

In the HackMD Area Chair draft only (T4 there).

| | k=2 | k=3 | k=4 | k=5 |
| --- | --- | --- | --- | --- |
| beta shared | 1.12 ± 0.02 | 1.13 ± 0.02 | 1.12 ± 0.02 | 1.10 ± 0.01 |
| beta opposite-sign | 1.51 ± 0.08 | 1.53 ± 0.08 | 1.73 ± 0.07 | 1.80 ± 0.07 |

## What the numbers are

Taken from `simulations/rebuttal/` (`generators.py`, `methods.py`, `make_tables.py`). Captions should reflect these points.

- **Averaging window:** all five tables average over gamma = p/n_l <= 0.5, which is p in {20, 60, 100, 140, 180} with n_l = 400.
- **"Ensemble" in Tables 1, 2, 3 and 5:** the best of simple average, inverse-variance weighting and ridge stacking, chosen by mean test MSE. Ridge stacking was the winner in every cell of Tables 1 and 2. Stacking weights are not constrained to be convex. Table 4 reports simple average and inverse-variance weighting separately.
- **Test points:** 200 per cluster, drawn from the same mixture as the training data (same means and covariances), including outcome noise.
- **Cluster means:** mu_l ~ N(0, 9 I_p), so their norms grow with p.
- **Ridge tuning:** `RidgeCV` over 25 log-spaced values from 1e-4 to 1e4, with an intercept. Ridgeless also centers the data, which adds an intercept.
- **Row names:** "factor" in the tables is "spiked" in the rebuttal's design text, "factor (per-cluster W)" is "low-rank perturbation", and "nonlinear" is "non-gaussian".
