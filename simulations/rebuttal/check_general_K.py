"""Numerical check of Corollary 2.6(1) for general K and unequal cluster sizes.

Condition for pooled dominance (Sigma = I, random effects):
    kappa_0  <=  min_{w in simplex} w' B w,    B = diag(k_l(1-k_l)) + kappa kappa'
Reports the worst-case margin over a wide grid of (alpha, gamma_0, cluster shares).
"""
import numpy as np
from itertools import product
from scipy.optimize import minimize

def kappa(g, a):
    t = a - 1.0 + 1.0 / g
    return (-t + np.sqrt(t * t + 4.0 * a)) / 2.0

def simplex_min(B):
    K = B.shape[0]
    # closed form for the equality-constrained min, projected check
    Bi = np.linalg.inv(B)
    w = Bi @ np.ones(K); w /= w.sum()
    if (w >= -1e-12).all():
        return float(w @ B @ w), w
    # fall back to a constrained solve if the unconstrained optimum leaves the simplex
    r = minimize(lambda v: v @ B @ v, np.ones(K) / K, method="SLSQP",
                 bounds=[(0, 1)] * K,
                 constraints={"type": "eq", "fun": lambda v: v.sum() - 1})
    return float(r.fun), r.x

ALPHAS = np.concatenate([np.logspace(-3, 3, 40)])
GAMMA0 = np.logspace(-3, 1.5, 40)          # p/n for the pooled learner
worst = {}
for K in [2, 3, 4, 5, 8]:
    # cluster share vectors: equal, mildly and severely unbalanced
    shares = [np.ones(K) / K,
              np.linspace(1, 2, K), np.linspace(1, 5, K), np.linspace(1, 20, K)]
    shares = [s / s.sum() for s in shares]
    m_min, arg = np.inf, None
    neg_w = 0
    for a, g0, pi in product(ALPHAS, GAMMA0, shares):
        gl = g0 / pi                        # gamma_l = p/n_l = (p/n)/(n_l/n)
        k0 = kappa(g0, a)
        kv = kappa(gl, a)
        B = np.diag(kv * (1 - kv)) + np.outer(kv, kv)
        val, w = simplex_min(B)
        if (w < -1e-9).any():
            neg_w += 1
        margin = val - k0                   # >= 0  <=>  pooled dominates
        if margin < m_min:
            m_min, arg = margin, (a, g0, tuple(np.round(pi, 3)))
    worst[K] = (m_min, arg, neg_w)
    print(f"K={K}: worst margin (min_w w'Bw - kappa_0) = {m_min:.3e}   "
          f"at alpha={arg[0]:.3g}, gamma_0={arg[1]:.3g}, shares={arg[2]}   "
          f"[optimal w left the simplex in {neg_w} cells]")
print("\nAll margins >= 0 means pooled dominance holds throughout the grid.")
