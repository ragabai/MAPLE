"""Feature-matrix generation: correlated copula features, mixed marginals, irrelevant/redundant/categorical
features and MCAR missingness."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

MARGINALS = ("normal", "uniform", "lognormal", "bimodal", "student_t")


def random_correlation(rng: np.random.Generator, d: int, strength: float) -> np.ndarray:
    """Correlation matrix from a random low-rank factor model; ``strength`` in [0,1)."""
    k = max(1, d // 3)
    F = rng.normal(size=(d, k))
    S = F @ F.T
    S = strength * S / np.max(np.abs(np.diag(S))) + np.eye(d)
    dd = np.sqrt(np.diag(S))
    return S / dd[:, None] / dd[None, :]


def apply_marginal(z: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "normal":
        return z
    if kind == "uniform":
        return ndtr(z)
    if kind == "lognormal":
        return np.exp(rng.uniform(0.5, 1.2) * z)
    if kind == "bimodal":
        return z + np.where(ndtr(z * 1.7) > 0.5, 1.5, -1.5)
    if kind == "student_t":
        # monotone heavy-tailed transform of a Gaussian (sinh-arcsinh style)
        return np.sinh(0.6 * z) / 0.6
    raise ValueError(kind)


def generate_features(rng: np.random.Generator, n: int, d: int, n_active: int, cfg: dict) -> tuple[np.ndarray, dict]:
    """Return observed X (n, d) without missingness, and meta including active indices.

    Columns ``active`` drive the target; redundant columns are noisy linear combinations of active ones;
    the rest are irrelevant. A subset of columns may be discretised into ordinal categories.
    """
    if cfg.get("p_correlated", 0.5) > rng.random() and d > 1:
        C = random_correlation(rng, d, rng.uniform(0.2, 0.9))
        Z = rng.normal(size=(n, d)) @ np.linalg.cholesky(C).T
    else:
        Z = rng.normal(size=(n, d))
    perm = rng.permutation(d)
    active = np.sort(perm[:n_active])
    rest = perm[n_active:]
    n_red = 0
    if len(rest) and rng.random() < cfg.get("p_redundant", 0.4):
        n_red = int(rng.integers(1, max(2, len(rest) // 2 + 1)))
        for j in rest[:n_red]:
            src = rng.choice(active, size=min(len(active), int(rng.integers(1, 3))), replace=False)
            Z[:, j] = Z[:, src] @ rng.normal(size=len(src)) + rng.normal(0, rng.uniform(0.05, 0.5), size=n)
            Z[:, j] /= Z[:, j].std() + 1e-12
    probs = np.asarray(cfg.get("marginal_probs", [0.4, 0.2, 0.15, 0.1, 0.15]))
    kinds = rng.choice(MARGINALS, size=d, p=probs / probs.sum())
    X = np.empty_like(Z)
    for j in range(d):
        X[:, j] = apply_marginal(Z[:, j], kinds[j], rng)
    cat_cols: list[int] = []
    if rng.random() < cfg.get("p_categorical", 0.3):
        n_cat = int(rng.integers(1, min(3, d) + 1))
        cat_cols = rng.choice(d, size=n_cat, replace=False).tolist()
        for j in cat_cols:
            k = int(rng.integers(2, 8))
            qs = np.quantile(X[:, j], np.sort(rng.uniform(0.05, 0.95, size=k - 1)))
            codes = np.searchsorted(qs, X[:, j])
            X[:, j] = rng.permutation(k)[codes] if rng.random() < 0.5 else codes  # nominal or ordinal codes
    meta = {"active": active.tolist(), "n_irrelevant": int(len(rest) - n_red), "n_redundant": n_red,
            "n_categorical": len(cat_cols), "marginals": kinds.tolist()}
    return X, meta


def apply_missing(rng: np.random.Generator, X: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    if rng.random() >= cfg.get("p_missing", 0.2):
        return X, 0.0
    rate = float(rng.uniform(0.02, 0.2))
    cols = rng.random(X.shape[1]) < 0.5
    if not cols.any():
        cols[rng.integers(X.shape[1])] = True
    mask = (rng.random(X.shape) < rate) & cols[None, :]
    Xm = X.copy()
    Xm[mask] = np.nan
    return Xm, rate
