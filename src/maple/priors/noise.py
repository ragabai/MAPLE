"""Observation-noise models. Noise is scaled relative to the (unit-variance) noise-free target."""
from __future__ import annotations

import numpy as np

TRAIN_NOISE = ("gaussian", "heteroscedastic")
OOD_NOISE = ("student_t", "laplace")


def sample_noise_spec(rng: np.random.Generator, cfg: dict, types: tuple[str, ...] | None = None) -> dict:
    types = types or tuple(cfg.get("noise_types", TRAIN_NOISE))
    lo, hi = cfg.get("noise_rel_range", (0.01, 1.0))
    return {"type": str(rng.choice(types)), "rel": float(np.exp(rng.uniform(np.log(lo), np.log(hi))))}


def add_noise(rng: np.random.Generator, f: np.ndarray, Xs: np.ndarray, spec: dict) -> np.ndarray:
    """``f`` is standardised; ``Xs`` standardised features (used for heteroscedastic scale)."""
    n, rel = f.shape[0], spec["rel"]
    t = spec["type"]
    if t == "gaussian":
        eps = rng.normal(size=n)
    elif t == "heteroscedastic":
        w = rng.normal(size=Xs.shape[1])
        z = np.nan_to_num(Xs) @ (w / (np.linalg.norm(w) + 1e-12))
        scale = 0.3 + np.abs(z)
        eps = rng.normal(size=n) * scale / np.sqrt(np.mean(scale ** 2))
    elif t == "student_t":
        eps = rng.standard_t(2.5, size=n) / np.sqrt(2.5 / 0.5)  # unit variance for df=2.5
    elif t == "laplace":
        eps = rng.laplace(size=n) / np.sqrt(2.0)
    else:
        raise ValueError(t)
    return f + rel * eps
