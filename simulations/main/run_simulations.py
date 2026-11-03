"""
Final simulation script for paper figures.

Runs 4 experiments:
1. Ridgeless (min-norm OLS): Fixed angle (60 deg), varying norm of mu_2 (1, 5, 10)
2. Ridge (LOO CV): Fixed angle (60 deg), varying norm of mu_2 (1, 5, 10)
3. Ridgeless (min-norm OLS): Fixed norm (1), varying angle (30, 60, 90 deg)
4. Ridge (LOO CV): Fixed norm (1), varying angle (30, 60, 90 deg)

Setup:
- N_train = 2000, K = 2 clusters of 1000 each
- p in [250, 500, 750, 1050, 1500, 2050, 3000, 4000, 5000, 6000]
- Training data: x_i ~ N(mu_b, I_p) per cluster
- Test data: X_test ~ N(0, I_p) (NO cluster structure)
- beta: drawn fresh per rep from N(0, I_p), normalized to unit norm
- Outcome: y_train = X_train @ beta + epsilon (noisy), y_test = X_test @ beta (noiseless)
- Noise: sigma = 1/sqrt(5), SNR = 5
- 50 repetitions per config

Saves per-repetition raw MSEs to CSV so figures can be regenerated without rerunning.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import RidgeCV
import time
import os
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TRAIN = 2000
K = 2
N_PER_CLUSTER = N_TRAIN // K  # 1000
N_TEST = 1000
P_VALUES = [250, 500, 750, 1050, 1500, 2050, 3000, 4000, 5000, 6000]
N_REPS = 50
PROGRESS_EVERY = 25  # print after every PROGRESS_EVERY reps within a config
NOISE_STD = 1.0 / np.sqrt(5)  # SNR = 5
ALPHAS_GRID = np.logspace(-6, 6, 20)
N_JOBS = max(1, (os.cpu_count() * 5) // 8)  # 5/8 of available cores

# Setup A: Fixed angle, changing norm
SETUP_A_ANGLE = 60.0
SETUP_A_NORMS = [1.0, 5.0, 10.0]

# Setup B: Fixed norm, changing angle
SETUP_B_NORM = 1.0
SETUP_B_ANGLES = [30.0, 60.0, 90.0]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Mean construction
# ---------------------------------------------------------------------------

def make_mu1(p, rng):
    """Generate mu_1 ~ N(0, I_p), then rescale to unit norm."""
    mu1 = rng.randn(p)
    mu1 /= np.linalg.norm(mu1)
    return mu1


def make_mu2(mu1, angle_deg, target_norm, rng):
    """
    Construct mu_2 such that:
    - The angle between mu_1 and mu_2 is `angle_deg`
    - ||mu_2|| = target_norm

    Method: build a unit-norm vector at the given angle from mu_1,
    then rescale to target_norm.
    """
    p = len(mu1)
    angle_rad = np.deg2rad(angle_deg)

    # Random vector orthogonal to mu_1
    v = rng.randn(p)
    v_perp = v - np.dot(v, mu1) * mu1
    v_perp /= np.linalg.norm(v_perp)

    # Unit-norm vector at the given angle
    mu2_unit = np.cos(angle_rad) * mu1 + np.sin(angle_rad) * v_perp

    # Rescale to target norm
    mu2 = target_norm * mu2_unit
    return mu2


# ---------------------------------------------------------------------------
# Single repetition
# ---------------------------------------------------------------------------

def run_single_rep(p, angle_deg, mu2_norm, estimator, seed):
    """
    Run one repetition for a given (p, angle, ||mu_2||, estimator).
    Returns dict with raw MSEs and ratio.
    """
    rng = np.random.RandomState(seed)

    # beta: fresh per rep, unit norm, shared between train and test
    beta = rng.randn(p)
    beta /= np.linalg.norm(beta)

    # Cluster means
    mu1 = make_mu1(p, rng)
    mu2 = make_mu2(mu1, angle_deg, mu2_norm, rng)

    # Training data per cluster: x_i ~ N(mu_b, I_p)
    X1 = rng.randn(N_PER_CLUSTER, p) + mu1
    X2 = rng.randn(N_PER_CLUSTER, p) + mu2

    # Outcomes: y_train = X @ beta + epsilon
    y1 = X1 @ beta + rng.normal(0, NOISE_STD, N_PER_CLUSTER)
    y2 = X2 @ beta + rng.normal(0, NOISE_STD, N_PER_CLUSTER)

    X_merged = np.vstack([X1, X2])
    y_merged = np.concatenate([y1, y2])

    # Fit base learners
    if estimator == "ridgeless":
        beta_hat_1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
        beta_hat_2, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
        beta_hat_merged, _, _, _ = np.linalg.lstsq(X_merged, y_merged,
                                                   rcond=None)

        def predict(b, X):
            return X @ b

        m1, m2, m_merged = beta_hat_1, beta_hat_2, beta_hat_merged
        lam_c, lam_m = np.nan, np.nan

    elif estimator == "ridge":
        m1 = RidgeCV(alphas=ALPHAS_GRID, cv=None).fit(X1, y1)
        m2 = RidgeCV(alphas=ALPHAS_GRID, cv=None).fit(X2, y2)
        m_merged = RidgeCV(alphas=ALPHAS_GRID, cv=None).fit(X_merged, y_merged)
        lam_c = 0.5 * (m1.alpha_ + m2.alpha_)
        lam_m = m_merged.alpha_

        def predict(model, X):
            return model.predict(X)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # Test data: X_test ~ N(0, I_p) (standard MVN, NO cluster structure)
    X_test = rng.randn(N_TEST, p)
    # Noiseless test outcome
    y_test_true = X_test @ beta

    # Predictions
    pred_ensemble = 0.5 * (predict(m1, X_test) + predict(m2, X_test))
    pred_merged = predict(m_merged, X_test)

    mse_ensemble = float(np.mean((pred_ensemble - y_test_true) ** 2))
    mse_merged = float(np.mean((pred_merged - y_test_true) ** 2))

    return {
        "mse_ensemble": mse_ensemble,
        "mse_merged": mse_merged,
        "ratio": mse_ensemble / mse_merged,
        "lambda_cluster": lam_c,
        "lambda_merged": lam_m,
        "var_y_test": float(np.var(y_test_true)),
    }


# ---------------------------------------------------------------------------
# Run all reps for a single config
# ---------------------------------------------------------------------------

def run_config(p, angle_deg, mu2_norm, estimator, seed_base, log_prefix=""):
    """Run N_REPS reps in parallel for one config, with progress logging."""
    seeds = [seed_base + r for r in range(N_REPS)]

    # Use generator backend so we can report progress as reps finish.
    # Backend is loky by default, which prevents BLAS over-subscription
    # by setting thread limits in worker processes.
    parallel = Parallel(n_jobs=N_JOBS, return_as="generator")
    gen = parallel(
        delayed(run_single_rep)(p, angle_deg, mu2_norm, estimator, s)
        for s in seeds
    )

    results = []
    t_start = time.time()
    for r_idx, res in enumerate(gen):
        results.append(res)
        n_done = r_idx + 1
        if n_done % PROGRESS_EVERY == 0 or n_done == N_REPS:
            elapsed = time.time() - t_start
            print(f"{log_prefix}      ... {n_done}/{N_REPS} reps done "
                  f"({elapsed:.0f}s elapsed)", flush=True)

    rows = []
    for r_idx, res in enumerate(results):
        rows.append({
            "estimator": estimator,
            "angle_deg": angle_deg,
            "mu2_norm": mu2_norm,
            "p": p,
            "rep": r_idx,
            **res,
        })
    return rows


# ---------------------------------------------------------------------------
# Run a full experiment (one estimator + one setup type)
# ---------------------------------------------------------------------------

def run_experiment(experiment_name, estimator, configs, seed_base):
    """
    configs: list of (angle_deg, mu2_norm) tuples
    Returns DataFrame with all per-rep results.
    """
    print(f"\n{'=' * 100}", flush=True)
    print(f"EXPERIMENT: {experiment_name}", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"  Estimator: {estimator}", flush=True)
    print(f"  Configs: {configs}", flush=True)
    print(f"  p values: {P_VALUES}", flush=True)
    print(f"  Reps per config: {N_REPS}", flush=True)
    print(f"  Total trials: {len(configs) * len(P_VALUES) * N_REPS}", flush=True)

    all_rows = []
    for cfg_idx, (angle, mu2_norm) in enumerate(configs):
        print(f"\n  Config {cfg_idx+1}/{len(configs)}: angle={angle}, "
              f"||mu2||={mu2_norm}", flush=True)
        for p in P_VALUES:
            t0 = time.time()
            seed = seed_base + cfg_idx * 10000 + p
            print(f"    p={p}: starting {N_REPS} reps "
                  f"(parallel jobs={N_JOBS})", flush=True)
            rows = run_config(p, angle, mu2_norm, estimator, seed,
                              log_prefix="  ")
            all_rows.extend(rows)

            ratios = np.array([r["ratio"] for r in rows])
            mean_r = np.mean(ratios)
            sem = stats.sem(ratios)
            ci = stats.t.interval(0.95, len(ratios) - 1, loc=mean_r, scale=sem)
            elapsed = time.time() - t0
            print(f"    p={p:<5} ratio={mean_r:.4f} "
                  f"95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]  "
                  f"({elapsed:.0f}s total)", flush=True)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    start = time.time()

    print("=" * 100, flush=True)
    print("FINAL SIMULATIONS FOR PAPER FIGURES", flush=True)
    print("=" * 100, flush=True)
    print(f"  Output dir: {OUTPUT_DIR}", flush=True)
    print(f"  N_train = {N_TRAIN}, N_test = {N_TEST}", flush=True)
    print(f"  Noise std = {NOISE_STD:.4f} (SNR = 5)", flush=True)
    print(f"  Parallel jobs: {N_JOBS}", flush=True)

    # Build configs
    configs_A = [(SETUP_A_ANGLE, n) for n in SETUP_A_NORMS]    # fixed angle, vary norm
    configs_B = [(a, SETUP_B_NORM) for a in SETUP_B_ANGLES]    # fixed norm, vary angle

    # Run 4 experiments
    df1 = run_experiment(
        "Ridgeless: Fixed Angle (60), Changing Norm (1, 5, 10)",
        "ridgeless", configs_A, seed_base=1000)
    df1.to_csv(os.path.join(OUTPUT_DIR, "raw_ridgeless_fixed_angle.csv"),
               index=False)

    df2 = run_experiment(
        "Ridge: Fixed Angle (60), Changing Norm (1, 5, 10)",
        "ridge", configs_A, seed_base=2000)
    df2.to_csv(os.path.join(OUTPUT_DIR, "raw_ridge_fixed_angle.csv"),
               index=False)

    df3 = run_experiment(
        "Ridgeless: Fixed Norm (1), Changing Angle (30, 60, 90)",
        "ridgeless", configs_B, seed_base=3000)
    df3.to_csv(os.path.join(OUTPUT_DIR, "raw_ridgeless_fixed_norm.csv"),
               index=False)

    df4 = run_experiment(
        "Ridge: Fixed Norm (1), Changing Angle (30, 60, 90)",
        "ridge", configs_B, seed_base=4000)
    df4.to_csv(os.path.join(OUTPUT_DIR, "raw_ridge_fixed_norm.csv"),
               index=False)

    elapsed = time.time() - start
    print(f"\n{'=' * 100}", flush=True)
    print(f"All simulations complete. Total time: {elapsed/60:.1f} min",
          flush=True)
    print(f"Raw data saved to {OUTPUT_DIR}/", flush=True)
    print("=" * 100, flush=True)
