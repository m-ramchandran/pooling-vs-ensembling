# Pooling vs Ensembling for Ridge Regression Under Covariate Shift

Simulation code for Ramchandran & Mukherjee, *Pooling Versus Ensembling for Ridge Regression Under Covariate Shift*.

Clusters are modeled as a Gaussian mixture in the covariates. Each experiment compares the Pooled (merged) learner, fit on all training data, against ensemble combiners (Simple Average, Inverse Variance, Stack Ridge) built from per-cluster learners. Learners are ridge (with GCV-tuned penalty) or min-norm least squares ("ridgeless").

## Files

- `run_simulations.py` — runs the four main-text high-dimensional experiments (Ridge / Ridgeless crossed with Fixed-Angle / Fixed-Norm cluster-mean setups) and writes per-repetition raw MSEs to CSV.
- `main_paper_figures.py` — reads the CSVs produced by `run_simulations.py` and generates the main paper figures (MSE-ratio panel, raw-MSE panel, and 2x2 combined figure).
- `appendix_tables.py` — self-contained end-to-end pipeline for the supplemental appendix tables: clustered Gaussian-mixture data generation, estimators and ensemble combiners, the experiment sweep across covariance structures / cluster counts / noise and beta specs, and the formatter that prints Tables T1-T5 (Ensemble/Pooled test-MSE ratios with 95% t CIs).

## Usage

```bash
# Main paper figures
python run_simulations.py          # writes raw_{ridge,ridgeless}_fixed_{angle,norm}.csv
python main_paper_figures.py       # reads those CSVs, writes fig_*.{png,pdf}

# Appendix tables (end to end, resumable)
python appendix_tables.py                  # run experiments + print/write tables.txt
python appendix_tables.py --skip-run       # just format tables from existing results.csv
python appendix_tables.py --skip-tables    # only run experiments
```

`appendix_tables.py` also accepts focused sweep flags (`--structures`, `--ks`, `--noise-specs`, `--beta-specs`, `--p-grid`, `--reps`, `--n-jobs`); see `python appendix_tables.py --help`.

## Main-text simulation setup

- `N_train = 2000`, `K = 2` clusters of 1000 each; `N_test = 1000`
- `p` in `[250, 500, 750, 1050, 1500, 2050, 3000, 4000, 5000, 6000]`
- Training: `x_i ~ N(mu_b, I_p)` per cluster `b`
- Test: `X_test ~ N(0, I_p)` (no cluster structure)
- `beta` drawn fresh per repetition from `N(0, I_p)` and normalized to unit norm
- `y_train = X_train @ beta + epsilon`, `y_test = X_test @ beta` (noiseless)
- Noise `sigma = 1/sqrt(5)` (SNR = 5), 50 repetitions per configuration
- Setup A (Fixed Angle, Changing Norm): angle 60 deg, `||mu_2|| in {1, 5, 10}`
- Setup B (Fixed Norm, Changing Angle): `||mu_2|| = 1`, angle in `{30, 60, 90}` deg
- Ridge penalty selected by leave-one-out CV over `alphas = logspace(-6, 6, 20)`

## Appendix simulation setup

- `K in {2, 3, 4, 5}` clusters, `n_t = 400` train and `200` test points per cluster, `100` replicates
- Cluster means `mu_l ~ N(0, 9 I_p)` (well separated)
- Covariance structures: identity, equicorrelated (shared rho in {0.2, 0.5, 0.8} and per-cluster), factor / "spiked" (shared and per-cluster `W`, rank `round(p/10)`), diagonal per-cluster scale, and non-Gaussian `tanh(Wu)` features
- Noise: equal `sigma_l^2 = 1` or geometric ladder spanning 16x with mean 1
- `beta`: shared across clusters or opposite-sign, scaled so `beta' Sigma_l beta` averages 1 across clusters
- `p` grid: `[20, 60, 100, 140, 180, 220, 260, 300, 340, 380, 392, 408, 440, 500, 560, 640, 720, 800]`, plus 10 bracket points near each k's pooled-learner interpolation peak `p = 400k`
- Ridge penalty: GCV over `alphas = logspace(-4, 4, 25)`

## Requirements

```
numpy
pandas
scipy
scikit-learn
matplotlib
joblib
```
