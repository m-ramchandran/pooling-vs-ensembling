"""Sweep harness for the rebuttal simulations.

Writes one long-format CSV row per (config, rep, estimator, method).
Safe to re-run / resume: completed (config, rep) cells found in the
output CSV are skipped, so short Slurm time limits are fine.

Examples:
    python run_experiments.py                          # full default grid
    python run_experiments.py --structures identity latent --ks 2 5 --reps 10
    python run_experiments.py --beta-specs opposite --structures identity
"""
import argparse
import csv
import os
import numpy as np
from joblib import Parallel, delayed

from generators import generate_dataset
from methods import evaluate_methods

STRUCTURES = ["identity", "diag_hetero",
              "equicorr_shared_0.2", "equicorr_shared_0.5", "equicorr_shared_0.8",
              "equicorr_hetero", "latent", "latent_hetero", "nonlinear"]
KS = [2, 3, 4, 5]
NOISE_SPECS = ["homo", "hetero"]
BETA_SPECS = ["shared", "opposite"]
ESTIMATORS = ["ridgeless", "ridge"]

# gamma = p / n_train_per_cluster runs over (0, 2]; extra points near the
# ridgeless interpolation peak at p = n = 400
P_GRID = [20, 60, 100, 140, 180, 220, 260, 300, 340, 380, 392, 408,
          440, 500, 560, 640, 720, 800]
N_TRAIN_PER_CLUSTER = 400

# Two interpolation thresholds matter, not one. Each base learner trains on
# N_TRAIN_PER_CLUSTER samples, so it peaks at p = 400 (gamma = 1) -- covered
# by P_GRID above. The MERGED learner trains on k * 400 samples, so it peaks
# at p = 400k (gamma = k), which falls outside P_GRID for k >= 3. These
# multipliers of 400k bracket the merged peak so both are visible per k.
MERGED_PSI = [0.6, 0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.2, 1.4]


def p_grid_for_k(k, base=None):
    """Full p grid for a given k: the shared base grid plus points
    bracketing that k's merged-learner interpolation peak."""
    base = P_GRID if base is None else base
    peak = {int(round(N_TRAIN_PER_CLUSTER * k * x)) for x in MERGED_PSI}
    return sorted(set(base) | peak)
N_TEST_PER_CLUSTER = 200
N_REPS = 100
SEPARATION = 3.0

FIELDS = ["structure", "k", "noise", "beta_spec", "p", "rep",
          "estimator", "method", "test_mse", "weights"]


def default_cells(structures, ks, noise_specs, beta_specs):
    """Grid cells to run. The full noise/k grid runs with shared beta;
    the opposite-sign add-on runs on identity with homoscedastic noise."""
    cells = []
    if "shared" in beta_specs:
        cells += [(s, k, nz, "shared")
                  for s in structures for k in ks for nz in noise_specs]
    if "opposite" in beta_specs:
        cells += [(s, k, "homo", "opposite")
                  for s in structures if s == "identity" for k in ks]
    return cells


def run_rep(structure, k, noise, beta_spec, p, rep):
    # deterministic seed per cell so resume and parallelism are reproducible
    seed = [STRUCTURES.index(structure), k, NOISE_SPECS.index(noise),
            BETA_SPECS.index(beta_spec), p, rep]
    rng = np.random.default_rng(seed)
    X_tr, y_tr, lab_tr, X_te, y_te, _ = generate_dataset(
        structure, k, p, N_TRAIN_PER_CLUSTER, N_TEST_PER_CLUSTER,
        noise, beta_spec, SEPARATION, rng)

    rows = []
    for estimator in ESTIMATORS:
        mses, ivw, stack_coefs = evaluate_methods(
            X_tr, y_tr, lab_tr, X_te, y_te, estimator)
        weights = {"inverse_var": ivw, "stack_ridge": stack_coefs}
        for method, mse in mses.items():
            w = ";".join(f"{v:.4f}" for v in weights.get(method, []))
            rows.append([structure, k, noise, beta_spec, p, rep,
                         estimator, method, mse, w])
    return rows


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add((row["structure"], int(row["k"]), row["noise"],
                          row["beta_spec"], int(row["p"]), int(row["rep"])))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", nargs="+", default=STRUCTURES,
                    choices=STRUCTURES)
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--noise-specs", nargs="+", default=NOISE_SPECS)
    ap.add_argument("--beta-specs", nargs="+", default=["shared"])
    ap.add_argument("--p-grid", nargs="+", type=int, default=None,
                    help="override the p grid; default is per-k "
                         "(base grid + that k's merged-peak bracket)")
    ap.add_argument("--reps", type=int, default=N_REPS)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results.csv"))
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    cells = default_cells(args.structures, args.ks,
                          args.noise_specs, args.beta_specs)
    done = load_done(args.out)
    write_header = not os.path.exists(args.out)

    # one worker pool for the whole run: spawning a pool per (cell, p) would
    # pay the numpy/sklearn import cost in every worker over a thousand times
    with open(args.out, "a", newline="") as f, \
            Parallel(n_jobs=args.n_jobs, return_as="generator_unordered") as parallel:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(FIELDS)
        for structure, k, noise, beta_spec in cells:
            p_grid = args.p_grid if args.p_grid else p_grid_for_k(k)
            # dispatch the whole (p x rep) block at once so workers stay busy
            todo = [(p, r) for p in p_grid for r in range(args.reps)
                    if (structure, k, noise, beta_spec, p, r) not in done]
            if not todo:
                continue
            n_written = 0
            for rows in parallel(
                    delayed(run_rep)(structure, k, noise, beta_spec, p, r)
                    for p, r in todo):
                writer.writerows(rows)
                n_written += 1
                if n_written % 200 == 0:  # bound the loss if the job is killed
                    f.flush()
            f.flush()
            print(f"{structure} k={k} noise={noise} beta={beta_spec}: "
                  f"{len(todo)} reps over {len(p_grid)} p values "
                  f"(max p={max(p_grid)})", flush=True)
    print("done")


if __name__ == "__main__":
    main()
