"""Disagreement measures over a committee prediction matrix ``P`` of shape (K experts, n points)."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def variance(P: np.ndarray) -> np.ndarray:
    """Committee ambiguity A(x) = (1/K) sum_k (f_k(x) - fbar(x))^2."""
    return P.var(axis=0)


def pairwise_mad(P: np.ndarray) -> np.ndarray:
    """2/(K(K-1)) sum_{i<j} |f_i(x) - f_j(x)|."""
    K = P.shape[0]
    if K < 2:
        return np.zeros(P.shape[1])
    diff = np.abs(P[:, None, :] - P[None, :, :]).sum(axis=(0, 1)) / 2.0
    return 2.0 * diff / (K * (K - 1))


def normalized(D: np.ndarray, sigma_y: float, kind: str = "var", eps: float = 1e-8) -> np.ndarray:
    """Scale-free disagreement: variance-type measures divide by sigma_y^2, absolute-type by sigma_y."""
    if kind == "var":
        return D / (sigma_y ** 2 + eps)
    if kind == "abs":
        return D / (sigma_y + eps)
    raise ValueError(kind)


def pfn_gap(p_model: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Delta(x) = (theta(x) - fbar(x))^2, the PFN-committee gap."""
    return (p_model - P.mean(axis=0)) ** 2


def residual_disagreement(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Dispersion of absolute residuals across experts, Var_k |y - f_k(x)|.

    (The variance of *signed* residuals equals the prediction variance, so it adds nothing.)
    """
    return np.abs(y[None, :] - P).var(axis=0)


def local(D: np.ndarray, X: np.ndarray, k: int = 10) -> np.ndarray:
    """kNN-smoothed disagreement: mean of D over each point's k nearest neighbours (itself included)."""
    k = int(min(k, X.shape[0]))
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    return D[idx].mean(axis=1)


def error_correlation(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    R = y[None, :] - P
    R = R - R.mean(axis=1, keepdims=True)
    sd = np.sqrt((R ** 2).mean(axis=1))
    sd = np.where(sd > 1e-12, sd, 1.0)
    C = (R @ R.T) / R.shape[1] / np.outer(sd, sd)
    np.fill_diagonal(C, 1.0)
    return C


def ambiguity_decomposition(P: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Krogh-Vedelsby: ens_err = avg_err - ambiguity (means over points)."""
    fbar = P.mean(axis=0)
    return {"avg_err": float(((P - y) ** 2).mean()), "ens_err": float(((fbar - y) ** 2).mean()),
            "ambiguity": float(variance(P).mean())}


SCORE_TYPES = ("loss", "disagreement", "ambiguity", "pairwise", "combined")


def task_score(kind: str, p_model: np.ndarray, P: np.ndarray, y: np.ndarray, clip: float) -> float:
    """Per-task score Y in [0, clip] used by the adaptive sampler (targets are context-standardised).

    loss         : PFN query MSE (label-based; contains noise variance, Prop. 4a)
    disagreement : mean PFN-committee gap Delta (label-free; default)
    ambiguity    : mean committee variance (label-free; ignores the PFN)
    pairwise     : mean pairwise absolute committee disagreement (label-free)
    combined     : Delta + ambiguity
    """
    if kind == "loss":
        v = np.mean((p_model - y) ** 2)
    elif kind == "disagreement":
        v = pfn_gap(p_model, P).mean()
    elif kind == "ambiguity":
        v = variance(P).mean()
    elif kind == "pairwise":
        v = pairwise_mad(P).mean()
    elif kind == "combined":
        v = pfn_gap(p_model, P).mean() + variance(P).mean()
    else:
        raise ValueError(kind)
    return float(min(clip, max(0.0, v)))
