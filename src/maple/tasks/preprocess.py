"""Feature preprocessing shared by synthetic training tasks and real-data inference.

All statistics are estimated on the context (training) rows only and then applied to query rows.
"""
from __future__ import annotations

import numpy as np

CLIP = 6.0


def fit_preprocessor(X_ctx: np.ndarray, skew_threshold: float = 2.0) -> dict:
    X = np.asarray(X_ctx, dtype=float)
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med[None, :])
    log_cols = np.zeros(X.shape[1], dtype=bool)
    mins = Xi.min(0) if Xi.shape[0] else np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = Xi[:, j]
        if mins[j] > 0 and len(np.unique(col)) > 10:
            sd = col.std()
            if sd > 0 and (((col - col.mean()) / sd) ** 3).mean() > skew_threshold:
                log_cols[j] = True
    Xt = Xi.copy()
    Xt[:, log_cols] = np.log(Xt[:, log_cols])
    mu = Xt.mean(0)
    sd = Xt.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return {"median": med, "log_cols": log_cols, "log_floor": np.where(log_cols, mins, 1.0), "mean": mu, "std": sd}


def transform(X: np.ndarray, p: dict) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xi = np.where(np.isfinite(X), X, p["median"][None, :])
    lc = p["log_cols"]
    if lc.any():
        Xi[:, lc] = np.log(np.maximum(Xi[:, lc], p["log_floor"][lc][None, :] * 0.5))
    return np.clip((Xi - p["mean"]) / p["std"], -CLIP, CLIP)


def preprocess_arrays(X_ctx: np.ndarray, X_qry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = fit_preprocessor(X_ctx)
    return transform(X_ctx, p), transform(X_qry, p)
