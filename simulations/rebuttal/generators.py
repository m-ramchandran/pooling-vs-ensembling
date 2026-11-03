"""Data generation for the rebuttal simulations.

Clustered Gaussian-mixture covariates with configurable per-cluster
covariance structures, per-cluster noise variances, and shared or
opposite-sign coefficients.
"""
import numpy as np

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
