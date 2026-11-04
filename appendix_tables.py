"""Supplemental appendix tables: Ensemble vs Pooled, 95% t confidence intervals.

End-to-end pipeline, formerly split across `generators.py`, `methods.py`,
`run_experiments.py`, and `make_tables.py`:

  1. Data generation. Clustered Gaussian-mixture covariates with
     configurable per-cluster covariance structures, per-cluster noise
     variances, and shared or opposite-sign coefficients.
  2. Estimators. Min-norm least squares ("ridgeless") and ridge with
     GCV-tuned penalty ("ridge"). Combiners match Table 1 of the paper:
     Merged (the "Pooled" learner), Simple Average, Inverse Variance,
     Stack Ridge. The column name in results.csv stays "merged".
  3. Experiment sweep. Writes one long-format CSV row per (config, rep,
     estimator, method). Safe to re-run / resume: completed (config, rep)
     cells found in the output CSV are skipped.
  4. Table formatter. Reads results.csv and prints Tables T1-T5 (raw
     Ensemble/Pooled test-MSE ratios with 95% t CIs); the number is the
     scenario, the letter is the estimator (a = ridgeless, b = ridge).

Usage:
    python appendix_tables.py                  # run experiments + make tables
    python appendix_tables.py --skip-run       # use existing results.csv
    python appendix_tables.py --skip-tables    # only run experiments

    # focused sweeps (any experiment flag may be passed)
    python appendix_tables.py --structures identity latent --ks 2 5 --reps 10
    python appendix_tables.py --beta-specs opposite --structures identity
"""
import argparse
import csv
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import t
from sklearn.linear_model import RidgeCV


# ====================================================================
# Data generation (clustered Gaussian mixtures)
# ====================================================================

# std of tanh(g) for g ~ N(0,1); standardizes nonlinear features to unit variance
TANH_STD = 0.6278

# per-cluster parameter menus, truncated to the first k entries
DIAG_SCALES = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
HETERO_RHOS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

# heteroscedastic noise: geometric ladder in sigma^2 spanning a fixed
# max/min ratio, rescaled so mean(sigma^2) = 1 for every k. Holds both the
# average noise level (matching the homoscedastic baseline) and the degree
# of cluster-to-cluster inequality constant as k varies.
NOISE_SPREAD = 16.0

D_RATIO = 0.1     # latent dimension d = round(D_RATIO * p)
W_PERTURB = 0.5   # cluster-specific distortion of the shared latent W


def build_structure(name, k, p, rng):
    """Draw the per-replicate pieces of a covariance structure.

    `name` is one of: identity, diag_hetero, equicorr_shared_<rho>,
    equicorr_hetero, latent, latent_hetero, nonlinear.
    Returns a dict consumed by sample_X and make_betas.
    """
    s = {"name": name, "k": k, "p": p}
    if name == "identity" or name == "diag_hetero":
        pass
    elif name.startswith("equicorr_shared"):
        s["rhos"] = np.full(k, float(name.rsplit("_", 1)[1]))
    elif name == "equicorr_hetero":
        s["rhos"] = HETERO_RHOS[:k]
    elif name in ("latent", "latent_hetero"):
        d = max(1, round(D_RATIO * p))
        s["W"] = rng.normal(size=(p, d))
        if name == "latent_hetero":
            s["W_per_cluster"] = [s["W"] + W_PERTURB * rng.normal(size=(p, d))
                                  for _ in range(k)]
    elif name == "nonlinear":
        # d = p, NOT D_RATIO * p: unlike the latent model there is no
        # additive ambient noise here, so x = tanh(Wz) lies on a
        # d-dimensional manifold. With d << p the within-cluster design is
        # near rank-deficient, min-norm coefficients blow up, and
        # predictions on distant clusters diverge (observed: RMSE ~2e3 at
        # p=20, d=2). Full-rank random features keep the model well posed.
        d = p
        s["W"] = rng.normal(0.0, 1.0 / np.sqrt(d), size=(p, d))
    else:
        raise ValueError(f"unknown structure: {name}")
    return s


def sample_X(s, means, n_per_cluster, rng):
    """Sample covariates: n_per_cluster points from each of the k clusters."""
    k, p, name = s["k"], s["p"], s["name"]
    Xs = []
    for c in range(k):
        if name == "identity":
            X = rng.normal(size=(n_per_cluster, p))
        elif name == "diag_hetero":
            X = np.sqrt(DIAG_SCALES[c]) * rng.normal(size=(n_per_cluster, p))
        elif name.startswith("equicorr"):
            # (1-rho)I + rho*11' via a shared scalar factor per sample: O(np)
            rho = s["rhos"][c]
            Z = rng.normal(size=(n_per_cluster, p))
            g = rng.normal(size=(n_per_cluster, 1))
            X = np.sqrt(1.0 - rho) * Z + np.sqrt(rho) * g
        elif name in ("latent", "latent_hetero"):
            # Sigma = WW' + I
            W = s["W_per_cluster"][c] if name == "latent_hetero" else s["W"]
            eta = rng.normal(size=(n_per_cluster, W.shape[1]))
            X = eta @ W.T + rng.normal(size=(n_per_cluster, p))
        elif name == "nonlinear":
            # x = mu + tanh(Wz), standardized to unit feature variance
            Z = rng.normal(size=(n_per_cluster, s["W"].shape[1]))
            X = np.tanh(Z @ s["W"].T) / TANH_STD
        X += means[c]
        Xs.append(X)
    labels = np.repeat(np.arange(k), n_per_cluster)
    perm = rng.permutation(len(labels))
    return np.vstack(Xs)[perm], labels[perm]


def signal_variances(s, beta):
    """Per-cluster signal variance beta' Sigma_l beta.

    Closed forms; never builds the p x p covariance. For `nonlinear` there
    is no closed form, but the features are standardized to unit variance
    by construction and the quadratic form is within ~2% of ||beta||^2
    across the grid (verified empirically), so that is used.
    """
    k, name = s["k"], s["name"]
    nb2 = beta @ beta
    if name in ("identity", "nonlinear"):
        return np.full(k, nb2)
    if name == "diag_hetero":
        return DIAG_SCALES[:k] * nb2
    if name.startswith("equicorr"):
        # (1-rho)||b||^2 + rho (1'b)^2
        sb = beta.sum() ** 2
        return np.array([(1.0 - r) * nb2 + r * sb for r in s["rhos"]])
    if name in ("latent", "latent_hetero"):
        # Sigma = WW' + I  =>  ||W'b||^2 + ||b||^2
        Ws = s["W_per_cluster"] if name == "latent_hetero" else [s["W"]] * k
        return np.array([(W.T @ beta) @ (W.T @ beta) + nb2 for W in Ws])
    raise ValueError(f"unknown structure: {name}")


def make_betas(s, beta_spec, rng):
    """Per-cluster coefficient vectors, shape (k, p).

    beta_spec: "shared" (same beta for all clusters) or "opposite"
    (alternating sign across clusters). For latent structures beta lies
    in the span of the shared W, per Hastie et al. (2022).

    beta is scaled so the signal variance beta' Sigma_l beta AVERAGES to 1
    across clusters. Normalizing ||beta|| instead would leave the latent
    structures at SNR ~226 (Sigma = WW' + I amplifies exactly the
    directions beta lies in) while every other structure sits at SNR ~1,
    confounding the covariance comparison with signal strength. Per-cluster
    spread is deliberately preserved -- for diag_hetero the SNR really does
    differ 16x across clusters, and that is the structure, not an artifact.
    """
    if s["name"] in ("latent", "latent_hetero"):
        base = s["W"] @ rng.normal(size=s["W"].shape[1])
    else:
        base = rng.normal(size=s["p"])
    base /= np.linalg.norm(base)
    base /= np.sqrt(signal_variances(s, base).mean())
    betas = np.tile(base, (s["k"], 1))
    if beta_spec == "opposite":
        betas *= np.array([(-1.0) ** c for c in range(s["k"])])[:, None]
    elif beta_spec != "shared":
        raise ValueError(f"unknown beta_spec: {beta_spec}")
    return betas


def noise_variances(noise_spec, k, spread=NOISE_SPREAD):
    """Per-cluster noise variances; cluster 0 is always the quietest."""
    if noise_spec == "homo":
        return np.ones(k)
    if noise_spec == "hetero":
        v = spread ** np.linspace(0.0, 1.0, k)
        return v / v.mean()
    raise ValueError(f"unknown noise_spec: {noise_spec}")


def make_y(X, labels, betas, noise_vars, rng):
    y = np.einsum("ij,ij->i", X, betas[labels])
    return y + rng.normal(size=len(y)) * np.sqrt(noise_vars[labels])


def generate_dataset(name, k, p, n_train_per_cluster, n_test_per_cluster,
                     noise_spec, beta_spec, separation, rng):
    """One full replicate: train and test drawn from the same mixture."""
    s = build_structure(name, k, p, rng)
    means = rng.normal(0.0, separation, size=(k, p))
    betas = make_betas(s, beta_spec, rng)
    nv = noise_variances(noise_spec, k)

    X_tr, lab_tr = sample_X(s, means, n_train_per_cluster, rng)
    y_tr = make_y(X_tr, lab_tr, betas, nv, rng)
    X_te, lab_te = sample_X(s, means, n_test_per_cluster, rng)
    y_te = make_y(X_te, lab_te, betas, nv, rng)
    return X_tr, y_tr, lab_tr, X_te, y_te, lab_te


# ====================================================================
# Estimators and combining schemes
# ====================================================================

ALPHAS = np.logspace(-4, 4, 25)


def fit(X, y, estimator):
    """Fit one learner; returns a predict function."""
    if estimator == "ridge":
        model = RidgeCV(alphas=ALPHAS).fit(X, y)
        return model.predict
    if estimator == "ridgeless":
        # min-norm least squares with an intercept via centering
        x_mean, y_mean = X.mean(axis=0), y.mean()
        coef, *_ = np.linalg.lstsq(X - x_mean, y - y_mean, rcond=None)
        return lambda Xnew: (Xnew - x_mean) @ coef + y_mean
    raise ValueError(f"unknown estimator: {estimator}")


def evaluate_methods(X_tr, y_tr, lab_tr, X_te, y_te, estimator):
    """Fit the Merged learner and one learner per cluster, then score all
    four combining schemes on the test set.

    Returns (mse_by_method, ivw_weights, stack_coefs).
    """
    clusters = np.unique(lab_tr)
    merged = fit(X_tr, y_tr, estimator)
    scls = [fit(X_tr[lab_tr == c], y_tr[lab_tr == c], estimator)
            for c in clusters]

    P_tr = np.column_stack([f(X_tr) for f in scls])
    P_te = np.column_stack([f(X_te) for f in scls])

    # inverse-variance: weight each SCL by 1/var of its residuals on the
    # training clusters it was NOT trained on
    ivw = np.array([1.0 / np.var(y_tr[lab_tr != c] - P_tr[lab_tr != c, j])
                    for j, c in enumerate(clusters)])
    ivw /= ivw.sum()

    # stacking: ridge regression of y on the SCL training predictions
    stack = RidgeCV(alphas=ALPHAS).fit(P_tr, y_tr)

    mse = lambda pred: float(np.mean((y_te - pred) ** 2))
    results = {
        "merged": mse(merged(X_te)),
        "simple_avg": mse(P_te.mean(axis=1)),
        "inverse_var": mse(P_te @ ivw),
        "stack_ridge": mse(stack.predict(P_te)),
    }
    return results, ivw, stack.coef_


# ====================================================================
# Experiment sweep
# ====================================================================

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

N_TEST_PER_CLUSTER = 200
N_REPS = 100
SEPARATION = 3.0

FIELDS = ["structure", "k", "noise", "beta_spec", "p", "rep",
          "estimator", "method", "test_mse", "weights"]


def p_grid_for_k(k, base=None):
    """Full p grid for a given k: the shared base grid plus points
    bracketing that k's merged-learner interpolation peak."""
    base = P_GRID if base is None else base
    peak = {int(round(N_TRAIN_PER_CLUSTER * k * x)) for x in MERGED_PSI}
    return sorted(set(base) | peak)


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


def run_experiments(args):
    cells = default_cells(args.structures, args.ks,
                          args.noise_specs, args.beta_specs)
    done = load_done(args.results)
    write_header = not os.path.exists(args.results)

    # one worker pool for the whole run: spawning a pool per (cell, p) would
    # pay the numpy/sklearn import cost in every worker over a thousand times
    with open(args.results, "a", newline="") as f, \
            Parallel(n_jobs=args.n_jobs, return_as="generator_unordered") as parallel:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(FIELDS)
        for structure, k, noise, beta_spec in cells:
            p_grid = args.p_grid if args.p_grid else p_grid_for_k(k)
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
                if n_written % 200 == 0:
                    f.flush()
            f.flush()
            print(f"{structure} k={k} noise={noise} beta={beta_spec}: "
                  f"{len(todo)} reps over {len(p_grid)} p values "
                  f"(max p={max(p_grid)})", flush=True)
    print("experiments done")


# ====================================================================
# Table formatter
# ====================================================================

ENS = ["simple_avg", "inverse_var", "stack_ridge"]
POOLED = "merged"        # column name in results.csv; printed as "Pooled"
GMAX = 0.5               # well-conditioned cutoff on gamma = p/n_t

# Row labels. "factor" rather than "spiked": with d = round(p/10) and
# eigenvalues of order p, Sigma = WW' + I is a proportional-rank factor
# model, not a classical (fixed-rank, bounded-spike) spiked covariance.
SHORT = {"identity": "identity",
         "diag_hetero": "diagonal, per-cluster",
         "equicorr_shared_0.2": "equicorr rho=0.2",
         "equicorr_shared_0.5": "equicorr rho=0.5",
         "equicorr_shared_0.8": "equicorr rho=0.8",
         "equicorr_hetero": "equicorr, per-cluster",
         "latent": "factor (shared W)",
         "latent_hetero": "factor (per-cluster W)",
         "nonlinear": "nonlinear"}
ORDER = list(SHORT)

IDX = ["structure", "k", "noise", "beta_spec", "estimator", "p", "rep"]


def load(path):
    """One row per (config, p, rep) with a column per method."""
    df = pd.read_csv(path)
    df = df.drop_duplicates(
        subset=["structure", "k", "noise", "beta_spec", "p", "rep",
                "estimator", "method"], keep="last")
    w = df.pivot_table(index=IDX, columns="method",
                       values="test_mse").reset_index()
    w["gamma"] = w.p / N_TRAIN_PER_CLUSTER          # base-learner aspect ratio
    w["psi"] = w.p / (N_TRAIN_PER_CLUSTER * w.k)    # pooled aspect ratio
    return w


def _matrix(cell, col):
    """reps x p matrix of `col` for one table cell."""
    return cell.pivot_table(index="rep", columns="p", values=col).to_numpy()


def stat(cell, num_col, den_col=POOLED):
    """Cell ratio and its 95% t half-width. Returns (point, half).

    The ratio is formed WITHIN each replicate and averaged over p, giving
    one number per replicate; the pairing between the two learners is
    therefore preserved and the interval is not inflated the way an
    independent delta-method calculation would be.

    `num_col` may be "best_ens", in which case the scheme is chosen by its
    mean first. (nan, nan) if the cell is empty.
    """
    if not len(cell):
        return np.nan, np.nan
    if num_col == "best_ens":
        num_col = min(ENS, key=lambda c: cell[c].mean())
    num, den = _matrix(cell, num_col), _matrix(cell, den_col)
    if num.size == 0 or den.size == 0:
        return np.nan, np.nan

    per_rep = np.nanmean(num / den, axis=-1)      # one ratio per replicate
    n = np.count_nonzero(~np.isnan(per_rep))
    half = t.ppf(0.975, n - 1) * np.nanstd(per_rep, ddof=1) / np.sqrt(n)
    return float(np.nanmean(per_rep)), float(half)


COLW = 14


def fmt(v, half):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'--':>{COLW}}"
    prec = 0 if v >= 100 else (1 if v >= 10 else 2)
    # a half-width under half a unit in the last place would print as 0.00,
    # which reads as "exactly zero" rather than "too small to show"
    hprec = max(prec, 2)
    while hprec < 4 and half < 0.5 * 10 ** -hprec:
        hprec += 1
    if half < 0.5 * 10 ** -hprec:       # still rounds to zero at 4 dp
        return f"{v:.{prec}f}+/-<{10.0 ** -hprec:.{hprec}f}".rjust(COLW)
    return f"{v:.{prec}f}+/-{half:.{hprec}f}".rjust(COLW)


def table(rows, col_labels, row_labels, title, subtitle, note, w=17):
    """rows: list of (values, halves) pairs, one per row label."""
    width = w + COLW * len(col_labels)
    out = [title, "  " + subtitle, "-" * width,
           " " * w + "".join(f"{c:>{COLW}}" for c in col_labels)]
    for lab, r in zip(row_labels, rows):
        out.append(f"{lab[:w-1]:<{w}}" + "".join(fmt(v, s) for v, s in zip(*r)))
    out += ["-" * width] + ([note] if note else [])
    return "\n".join(out) + "\n"


RATIO_SUB = "Ensemble/Pooled test-MSE ratio +/- 95% t CI;  >1 = Pooled better"

NOISE_LABEL = {"homo": "equal noise variance",
               "hetero": "16x spread in sigma^2"}
# shorter form, for row labels where the column budget is tight
NOISE_SHORT = {"homo": "equal sigma^2", "hetero": "16x spread"}


def _row(sub, ks, col, **sel):
    """One table row: the cell statistic at each k, with half-widths."""
    out = []
    for k in ks:
        mask = (sub.k == k)
        for c, v in sel.items():
            mask &= (sub[c] == v)
        out.append(stat(sub[mask], col))
    return [v for v, _ in out], [s for _, s in out]


def t_structures(m, estimator, noise, tag):
    """T1/T2: covariance structure x k, at one estimator and noise setting."""
    sub = m[(m.estimator == estimator) & (m.beta_spec == "shared")
            & (m.noise == noise) & (m.gamma <= GMAX)]
    ks = sorted(sub.k.unique())
    present = [s for s in ORDER if s in set(sub.structure)]
    rows = [_row(sub, ks, "best_ens", structure=s) for s in present]
    return table(rows, [f"k={k}" for k in ks], [SHORT[s] for s in present],
                 f"{tag}. Covariance structure x k -- {estimator}, "
                 f"{NOISE_LABEL[noise]}, shared beta",
                 RATIO_SUB,
                 f"    (mean over gamma = p/n_t <= {GMAX})", w=24)


def t_peaks(m, estimator, tag):
    """T3: behaviour at each learner's own interpolation threshold."""
    sub = m[(m.beta_spec == "shared") & (m.noise == "homo")
            & (m.structure == "identity") & (m.estimator == estimator)]
    ks = sorted(sub.k.unique())
    rows, labs = [], []
    for col, lab in [("gamma", "gamma = 1  (base peak)"),
                     ("psi", "gamma = k  (pooled peak)")]:
        vals, halves = [], []
        for k in ks:
            c = sub[sub.k == k]
            uniq = c.drop_duplicates("p")
            keep = c.p.isin(uniq.iloc[(uniq[col] - 1.0).abs().argsort()[:2]].p)
            v, s = stat(c[keep], "best_ens")
            vals.append(v)
            halves.append(s)
        rows.append((vals, halves))
        labs.append(lab)
    return table(rows, [f"k={k}" for k in ks], labs,
                 f"{tag}. Interpolation peaks -- {estimator}, identity, "
                 f"equal noise, shared beta",
                 RATIO_SUB,
                 "    (each learner is worst near its OWN peak; ridge "
                 "regularization removes both)", w=26)


def t_noise(m, estimator, tag):
    """T4: weighting scheme x noise setting, schemes NOT collapsed."""
    sub = m[(m.beta_spec == "shared") & (m.gamma <= GMAX)
            & (m.structure == "identity") & (m.estimator == estimator)]
    ks = sorted(sub.k.unique())
    rows, labs = [], []
    for noise in ["homo", "hetero"]:
        for lab, col in [("simple avg", "simple_avg"),
                         ("inverse-var", "inverse_var")]:
            rows.append(_row(sub, ks, col, noise=noise))
            labs.append(f"{NOISE_SHORT[noise]}, {lab}")
    return table(rows, [f"k={k}" for k in ks], labs,
                 f"{tag}. Ensembling scheme x noise -- {estimator}, "
                 f"identity, shared beta",
                 RATIO_SUB,
                 f"    (mean over gamma <= {GMAX}; 16x spread holds "
                 "mean sigma^2 = 1)", w=28)


def t_beta(m, estimator, tag):
    """T5: shared vs opposite-sign coefficients across clusters."""
    sub = m[(m.noise == "homo") & (m.gamma <= GMAX)
            & (m.structure == "identity") & (m.estimator == estimator)]
    ks = sorted(sub.k.unique())
    rows = [_row(sub, ks, "best_ens", beta_spec=bs)
            for bs in ["shared", "opposite"]]
    return table(rows, [f"k={k}" for k in ks],
                 ["beta shared", "beta opposite-sign"],
                 f"{tag}. Shared vs opposite-sign beta -- {estimator}, "
                 f"identity, equal noise",
                 RATIO_SUB,
                 f"    (mean over gamma <= {GMAX})", w=20)


def header(n_reps):
    return f"""SUPPLEMENTAL SIMULATIONS -- Ensemble vs Pooled, 95% t confidence intervals

Design: n_t = {N_TRAIN_PER_CLUSTER} train / {N_TEST_PER_CLUSTER} test points per cluster, {n_reps} replicates.
Cells are RAW Ensemble/Pooled test-MSE ratios, so >1 means Pooled wins.
Nothing is log-scaled.

Interval: the ratio is formed WITHIN each replicate (so the two learners'
errors stay paired) and averaged over p, giving one value per replicate.
The half-width is t(.975,{n_reps-1})*sd/sqrt({n_reps}), matching the submission
(Sec 3.1: 95% CIs from the t-distribution).

Numbering: the number is the scenario, the letter is the estimator
(a = ridgeless, b = ridge).
  T1  covariance structure x k, equal noise variance
  T2  covariance structure x k, 16x spread in sigma^2
  T3  the two interpolation peaks (identity)
  T4  ensembling scheme under noise heterogeneity (identity)
  T5  shared vs opposite-sign beta (identity)
T1, T2, T3 and T5 report whichever of the three ensembling schemes does
best (simple average, inverse-variance, ridge stacking). T4 instead breaks
simple-average and inverse-variance out separately, which is what shows
whether inverse-variance weighting rescues the ensemble under unequal
noise; it is not a subset of T2.

Conventions:
  gamma = p/n_t is each BASE learner's aspect ratio. The Pooled learner
    trains on k*n_t points, so its aspect ratio is gamma/k and it peaks at
    gamma = k. Both thresholds are on the p grid.
  beta is scaled so the signal variance beta'Sigma_l beta averages 1 across
    clusters in EVERY structure, so all rows sit at the same overall SNR.
  "equal noise variance" refers to the ERROR term (sigma^2_l equal across
    clusters), not to the covariates.
  "factor" rows are Sigma = WW' + I with W of rank round(p/10).

Two things the interval does NOT cover:
  * the winning ensembling scheme is picked by its mean and then reported;
    selection uncertainty across the three schemes is not propagated.
  * averaging over p is an average across design conditions, not repeated
    measures, so the interval reflects replicate noise at fixed p, not
    sensitivity to the choice of p window.
"""


def make_tables(args):
    m = load(args.results)
    n_reps = m.rep.nunique()

    text = "\n".join([
        header(n_reps),
        t_structures(m, "ridgeless", "homo", "T1a"),
        t_structures(m, "ridge", "homo", "T1b"),
        t_structures(m, "ridgeless", "hetero", "T2a"),
        t_structures(m, "ridge", "hetero", "T2b"),
        t_peaks(m, "ridgeless", "T3a"),
        t_peaks(m, "ridge", "T3b"),
        t_noise(m, "ridgeless", "T4a"),
        t_noise(m, "ridge", "T4b"),
        t_beta(m, "ridgeless", "T5a"),
        t_beta(m, "ridge", "T5b"),
    ])
    with open(args.out, "w") as f:
        f.write(text)
    print(text)
    print(f"[written to {args.out}]")


# ====================================================================
# Main entry point
# ====================================================================

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(here, "results.csv"),
                    help="path to the long-format results CSV")
    ap.add_argument("--out", default=os.path.join(here, "tables.txt"),
                    help="path to write the formatted tables")
    ap.add_argument("--skip-run", action="store_true",
                    help="skip experiments; just make tables from --results")
    ap.add_argument("--skip-tables", action="store_true",
                    help="run experiments only; do not format tables")
    # experiment flags
    ap.add_argument("--structures", nargs="+", default=STRUCTURES,
                    choices=STRUCTURES)
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--noise-specs", nargs="+", default=NOISE_SPECS)
    ap.add_argument("--beta-specs", nargs="+", default=["shared"])
    ap.add_argument("--p-grid", nargs="+", type=int, default=None,
                    help="override the p grid; default is per-k "
                         "(base grid + that k's merged-peak bracket)")
    ap.add_argument("--reps", type=int, default=N_REPS)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    if not args.skip_run:
        run_experiments(args)
    if not args.skip_tables:
        make_tables(args)


if __name__ == "__main__":
    main()
