"""Synthetic regression task generator built on the mathematical prior library."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from maple.priors.families import FAMILIES, train_families
from maple.priors.features import apply_missing, generate_features
from maple.priors.noise import add_noise, sample_noise_spec
from maple.tasks.preprocess import preprocess_arrays

Cell = tuple[str, int]


@dataclass
class Task:
    X_ctx: np.ndarray
    y_ctx: np.ndarray
    X_qry: np.ndarray
    y_qry: np.ndarray
    f_qry: np.ndarray  # noise-free target on query points (same standardisation as y)
    meta: dict = field(default_factory=dict)


def default_generator_config() -> dict:
    return {
        "families": train_families(),
        "n_active": [[1, 2], [2, 4], [3, 6]],
        "n_components": [[1, 1], [1, 2], [2, 3]],
        "p_correlated": 0.5,
        "p_redundant": 0.4,
        "p_categorical": 0.3,
        "p_missing": 0.2,
        "marginal_probs": [0.4, 0.2, 0.15, 0.1, 0.15],
        "noise_types": ["gaussian", "heteroscedastic"],
        "noise_rel_range": [0.01, 1.0],
        "param_scale": 1.0,
        "composition": "additive",
        "max_range_over_std": 25.0,
        "max_abs_ctx_z": 12.0,  # context-standardised targets must lie within +-12 context SDs
        "min_unique_frac": 0.05,
        "max_attempts": 200,  # per-attempt rejection can reach ~0.7 for hard cells; 200 makes failure ~1e-33
    }


def cells(cfg: dict) -> list[Cell]:
    return [(f, c) for f in cfg["families"] for c in (0, 1, 2)]


def _std(v: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(v).all():
        return None
    sd = v.std()
    if sd < 1e-8 or not np.isfinite(sd):
        return None
    return (v - v.mean()) / sd


class TaskGenerator:
    def __init__(self, cfg: dict | None = None):
        self.cfg = copy.deepcopy(cfg or default_generator_config())

    # -- target function -------------------------------------------------------------------------------------
    def _target(self, rng: np.random.Generator, cell: Cell, Xs: np.ndarray, active: list[int]) -> tuple[np.ndarray, dict] | None:
        fam, c = cell
        s = self.cfg["param_scale"]
        XA = Xs[:, active]
        lo, hi = self.cfg["n_components"][c]
        n_comp = int(rng.integers(lo, hi + 1))
        with np.errstate(all="ignore"):
            primary = FAMILIES[fam].sample(rng, XA.shape[1], c, s)
            base = _std(primary.f(XA))
            if base is None:
                return None
            metas = [primary.meta]
            if self.cfg["composition"] == "nested":
                pool = [f for f in train_families() if f not in ("conditional",)]
                outer_name = str(rng.choice(pool))
                outer = FAMILIES[outer_name].sample(rng, 1, max(0, c - 1), s)
                out = _std(outer.f(base[:, None]))
                if out is None:
                    return None
                metas.append({**outer.meta, "role": "outer"})
                return out, {"components": metas, "composition": "nested"}
            total = base
            pool = list(self.cfg["families"])
            for _ in range(n_comp - 1):
                name = str(rng.choice(pool))
                sub = rng.choice(XA.shape[1], size=int(rng.integers(1, XA.shape[1] + 1)), replace=False)
                comp = FAMILIES[name].sample(rng, len(sub), max(0, c - 1), s)
                v = _std(comp.f(XA[:, sub]))
                if v is None:
                    return None
                total = total + rng.uniform(0.2, 0.6) * v
                metas.append(comp.meta)
            out = _std(total)
        if out is None:
            return None
        return out, {"components": metas, "composition": "additive"}

    def _valid(self, f: np.ndarray) -> bool:
        if not np.isfinite(f).all():
            return False
        if (f.max() - f.min()) > self.cfg["max_range_over_std"]:
            return False
        return len(np.unique(np.round(f, 6))) >= self.cfg["min_unique_frac"] * len(f)

    # -- public API ------------------------------------------------------------------------------------------
    def sample(self, rng: np.random.Generator, cell: Cell, n_ctx: int, n_qry: int, d: int,
               noise_types: tuple[str, ...] | None = None) -> Task:
        fam, c = cell
        if fam not in FAMILIES:
            raise KeyError(fam)
        n = n_ctx + n_qry
        lo, hi = self.cfg["n_active"][c]
        for attempt in range(self.cfg["max_attempts"]):
            n_active = int(min(d, rng.integers(lo, hi + 1)))
            X, fmeta = generate_features(rng, n, d, n_active, self.cfg)
            mu, sd = X.mean(0), X.std(0)
            Xs = (X - mu) / np.where(sd > 1e-12, sd, 1.0)
            res = self._target(rng, cell, Xs, fmeta["active"])
            if res is None or not self._valid(res[0]):
                continue
            f, tmeta = res
            spec = sample_noise_spec(rng, self.cfg, noise_types)
            y = add_noise(rng, f, Xs, spec)
            Xm, miss = apply_missing(rng, X, self.cfg)
            Xc, Xq = preprocess_arrays(Xm[:n_ctx], Xm[n_ctx:])
            ym, ys = y[:n_ctx].mean(), y[:n_ctx].std()
            if ys < 1e-8:
                continue
            zmax = self.cfg["max_abs_ctx_z"]
            if np.abs((y - ym) / ys).max() > zmax or np.abs((f[n_ctx:] - ym) / ys).max() > zmax:
                continue
            meta = {"cell": cell, "family": fam, "complexity": c, "d": d, "n_ctx": n_ctx, "n_qry": n_qry,
                    "noise": spec, "missing_rate": miss, "attempts": attempt + 1,
                    "param_scale": self.cfg["param_scale"], **fmeta, **tmeta,
                    "y_scale": float(ys)}
            return Task(Xc, (y[:n_ctx] - ym) / ys, Xq, (y[n_ctx:] - ym) / ys, (f[n_ctx:] - ym) / ys, meta)
        raise RuntimeError(f"could not generate a valid task for cell {cell} after {self.cfg['max_attempts']} attempts")
