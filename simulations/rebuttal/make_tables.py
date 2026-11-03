"""Supplemental simulation tables: Ensemble vs Pooled, with 95% t CIs.

One table per (scenario, estimator). The number is the scenario, the letter
is the estimator -- a = ridgeless, b = ridge -- so T3a and T3b are the same
scenario for the two learners. Every cell is a RAW Ensemble/Pooled test-MSE
ratio; nothing is log-scaled.

Run on PSC, where results.csv lives:

    python make_tables.py [--results results.csv] [--out tables.txt]

How the interval is computed
----------------------------
The numerator and denominator of every cell come from the SAME replicates
and are strongly positively correlated, so any calculation treating them as
independent (the usual delta-method shortcut) overstates the width badly.
Instead the ratio is formed WITHIN each replicate and averaged over p,
giving one value per replicate. The point estimate is the mean of those and
the half-width is t_{.975,n-1}*sd/sqrt(n_reps), matching the submission's
"95% confidence intervals computed from the t-distribution".

Two things the interval deliberately does NOT cover, noted here so the
numbers are not over-read:
  * "best_ens" picks the winning ensembling scheme by its mean and reports
    that scheme. Selection uncertainty across the three schemes is not
    propagated.
  * Averaging over p is an average across design conditions, not repeated
    measures, so the interval reflects replicate noise at fixed p -- not
    sensitivity to the choice of p window.
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import t

from run_experiments import N_TRAIN_PER_CLUSTER, N_TEST_PER_CLUSTER

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
            # the two p values closest to that learner's own threshold
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
                 "mean sigma^2 = 1)", w=28)  # w fits "equal sigma^2, inverse-var"


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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "results.csv"))
    ap.add_argument("--out", default=os.path.join(here, "tables.txt"))
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
