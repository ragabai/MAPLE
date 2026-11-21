"""Routing between the PFN and classical experts (variants E: global, F: local competence).

Weights use out-of-fold (OOF) squared errors on the training data, normalised by the best predictor's error so
that ``beta`` is scale-free:  w_k ∝ exp(-beta * e_k / min_j e_j).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from maple.data.tabular import TabularEncoder
from maple.experts.registry import COMMITTEE_DEFAULT, make_expert
from maple.model.predictor import MapleRegressor, pfn_predict_arrays, resolve_device, _load
from maple.tasks.preprocess import fit_preprocessor, transform

ROUTING_EXPERTS = list(COMMITTEE_DEFAULT)
DEFAULT_GRID = {"beta": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0], "k": [5, 10, 20, 40]}
EPS = 1e-12


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def global_weights(O: np.ndarray, y: np.ndarray, beta: float) -> np.ndarray:
    L = ((O - y[None]) ** 2).mean(axis=1)
    return _softmax_rows(-beta * L / max(L.min(), EPS))


def local_weights(O: np.ndarray, y: np.ndarray, Xtr: np.ndarray, Xte: np.ndarray, beta: float, k: int) -> np.ndarray:
    k = int(min(k, Xtr.shape[0]))
    idx = NearestNeighbors(n_neighbors=k).fit(Xtr).kneighbors(Xte, return_distance=False)
    E = (O - y[None]) ** 2                      # (K, n_tr)
    e = E[:, idx].mean(axis=-1).T               # (n_te, K)
    return _softmax_rows(-beta * e / np.maximum(e.min(axis=1, keepdims=True), EPS))


def combine(O, T, y, Xtr, Xte, mode: str, beta: float, k: int) -> np.ndarray:
    if mode == "global":
        return global_weights(O, y, beta) @ T
    if mode == "local":
        return (local_weights(O, y, Xtr, Xte, beta, k) * T.T).sum(axis=1)
    raise ValueError(mode)


def stacking_weights(O: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Breiman (1996) stacked regression: non-negative least squares on OOF predictions, normalised."""
    w, _ = nnls(O.T, y)
    return w / w.sum() if w.sum() > 0 else np.full(O.shape[0], 1.0 / O.shape[0])


# ----------------------------------------------------------------------------- base predictions

def _expert_matrix_fit(X: pd.DataFrame):
    enc = TabularEncoder().fit(X)
    Z = enc.transform(X)
    if Z.shape[1] == 0:
        Z = np.zeros((len(X), 1))
    p = fit_preprocessor(Z)
    return enc, p, transform(Z, p)


def _expert_matrix(enc, p, X: pd.DataFrame) -> np.ndarray:
    Z = enc.transform(X)
    if Z.shape[1] == 0:
        Z = np.zeros((len(X), 1))
    return transform(Z, p)


class BaseEnsemble:
    """Fits the PFN and experts once; exposes OOF matrix O (K+1, n_tr), test matrix T (K+1, n_te) and kNN space.

    Row 0 is the PFN; rows 1.. are ``experts`` in order. Fold models never see their held-out rows.
    """

    def __init__(self, checkpoint, experts=None, n_folds: int = 5, seed: int = 0, device: str = "auto",
                 n_ensemble: int = 4, max_ctx: int = 1024):
        self.checkpoint, self.experts = checkpoint, list(experts or ROUTING_EXPERTS)
        self.n_folds, self.seed, self.device, self.n_ensemble, self.max_ctx = n_folds, seed, device, n_ensemble, max_ctx
        self.names = ["pfn"] + self.experts

    def _pfn(self):
        return MapleRegressor(self.checkpoint, self.n_ensemble, self.max_ctx, self.device, self.seed)

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        self.y_ = y
        n = len(y)
        O = np.zeros((len(self.names), n))
        kf = KFold(n_splits=min(self.n_folds, n), shuffle=True, random_state=self.seed)
        for tr, va in kf.split(X):
            enc, p, Ztr = _expert_matrix_fit(X.iloc[tr])
            Zva = _expert_matrix(enc, p, X.iloc[va])
            O[0, va] = self._pfn().fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
            for j, e in enumerate(self.experts, start=1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    O[j, va] = make_expert(e, self.seed).fit(Ztr, y[tr]).predict(Zva)
        self.O_ = O
        self.enc_, self.prep_, self.Ztr_ = _expert_matrix_fit(X)
        self.pfn_ = self._pfn().fit(X, y)
        self.models_ = []
        for e in self.experts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models_.append(make_expert(e, self.seed).fit(self.Ztr_, y))
        return self

    def predict_base(self, X) -> tuple[np.ndarray, np.ndarray]:
        X = pd.DataFrame(X).reset_index(drop=True)
        Zte = _expert_matrix(self.enc_, self.prep_, X)
        T = np.vstack([self.pfn_.predict(X)] + [m.predict(Zte) for m in self.models_])
        return T, Zte


class RoutedRegressor:
    def __init__(self, checkpoint, mode: str = "local", beta: float = 4.0, k: int = 20, experts=None,
                 n_folds: int = 5, seed: int = 0, device: str = "auto"):
        self.mode, self.beta, self.k = mode, beta, k
        self.base = BaseEnsemble(checkpoint, experts, n_folds, seed, device)

    def fit(self, X, y):
        self.base.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        T, Zte = self.base.predict_base(X)
        return combine(self.base.O_, T, self.base.y_, self.base.Ztr_, Zte, self.mode, self.beta, self.k)


# ----------------------------------------------------------------------------- meta-selection on synthetic tasks

def synthetic_base_predictions(checkpoint, task, experts=None, n_folds: int = 5, seed: int = 0, device="auto"):
    """OOF (on the context) and query predictions for a preprocessed synthetic Task."""
    experts = list(experts or ROUTING_EXPERTS)
    dev = resolve_device(device)
    model = _load(str(checkpoint), str(dev))
    Xc, yc, Xq = task.X_ctx, task.y_ctx, task.X_qry
    n = len(yc)
    O = np.zeros((len(experts) + 1, n))
    for tr, va in KFold(n_splits=min(n_folds, n), shuffle=True, random_state=seed).split(Xc):
        ym, ys = yc[tr].mean(), yc[tr].std() + 1e-12
        O[0, va] = pfn_predict_arrays(model, Xc[tr], (yc[tr] - ym) / ys, Xc[va], dev) * ys + ym
        for j, e in enumerate(experts, start=1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                O[j, va] = make_expert(e, seed).fit(Xc[tr], yc[tr]).predict(Xc[va])
    T = np.zeros((len(experts) + 1, Xq.shape[0]))
    T[0] = pfn_predict_arrays(model, Xc, yc, Xq, dev)
    for j, e in enumerate(experts, start=1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T[j] = make_expert(e, seed).fit(Xc, yc).predict(Xq)
    return O, T


def meta_select(mode: str, bases: list[tuple], tasks: list, grid: dict | None = None) -> dict:
    """Choose (beta, k) minimising mean query NRMSE over synthetic tasks. ``bases`` = [(O, T)] per task."""
    grid = grid or DEFAULT_GRID
    ks = grid["k"] if mode == "local" else [grid["k"][0]]
    best, table = None, []
    for beta, k in itertools.product(grid["beta"], ks):
        errs = []
        for (O, T), t in zip(bases, tasks):
            p = combine(O, T, t.y_ctx, t.X_ctx, t.X_qry, mode, beta, k)
            errs.append(np.sqrt(np.mean((p - t.y_qry) ** 2)) / (t.y_qry.std() + 1e-12))
        score = float(np.mean(errs))
        table.append({"beta": beta, "k": k, "nrmse": score})
        if best is None or score < best["nrmse"]:
            best = {"beta": beta, "k": k, "nrmse": score}
    return {**best, "grid": table}
