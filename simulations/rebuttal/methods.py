"""Estimators and combining schemes.

Estimators: min-norm least squares ("ridgeless") and ridge with GCV-tuned
penalty ("ridge"). Combiners match Table 1 of the paper: Merged, Simple
(the "merged" learner is what the paper and tables.txt call "Pooled"; the
column name in results.csv stays "merged")
Average, Inverse Variance, Stack Ridge.
"""
import numpy as np
from sklearn.linear_model import RidgeCV

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
