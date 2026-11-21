"""Registry of lightweight expert regressors. Add an expert by adding one entry to ``_BUILDERS``."""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


class AdaptiveKNN(BaseEstimator, RegressorMixin):
    """kNN with k = clip(sqrt(n), 3, 15), distance weighting; safe for tiny contexts."""

    def fit(self, X, y):
        n = len(y)
        k = int(np.clip(round(np.sqrt(n)), 1, 15))
        self.model_ = KNeighborsRegressor(n_neighbors=min(k, n), weights="distance").fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class ScaledRBF(BaseEstimator, RegressorMixin):
    """RBF kernel ridge with median-heuristic bandwidth (inputs are pre-standardised)."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, X, y):
        n = X.shape[0]
        sub = X[: min(n, 256)]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        self.ym_ = float(np.mean(y))
        self.model_ = KernelRidge(kernel="rbf", alpha=self.alpha, gamma=1.0 / max(med, 1e-6)).fit(X, y - self.ym_)
        return self

    def predict(self, X):
        return self.model_.predict(X) + self.ym_


class QuietMLP(BaseEstimator, RegressorMixin):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        self.ym_, self.ys_ = float(np.mean(y)), float(np.std(y) + 1e-12)
        self.model_ = MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, learning_rate_init=1e-3,
                                   max_iter=300, early_stopping=len(y) >= 50, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(X, (y - self.ym_) / self.ys_)
        return self

    def predict(self, X):
        return self.model_.predict(X) * self.ys_ + self.ym_


_ALPHAS = np.logspace(-3, 3, 13)

_BUILDERS: dict[str, Callable[[int], BaseEstimator]] = {
    "ridge": lambda s: RidgeCV(alphas=_ALPHAS),
    "elasticnet": lambda s: ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=20, cv=3, max_iter=2000, random_state=s),
    "knn": lambda s: AdaptiveKNN(),
    "rf": lambda s: RandomForestRegressor(n_estimators=100, min_samples_leaf=1, n_jobs=1, random_state=s),
    "extratrees": lambda s: ExtraTreesRegressor(n_estimators=100, min_samples_leaf=1, n_jobs=1, random_state=s),
    "gbr": lambda s: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                               random_state=s),
    "hgb": lambda s: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, min_samples_leaf=10,
                                                   random_state=s),
    "rbf": lambda s: ScaledRBF(alpha=0.1),
    "mlp": lambda s: QuietMLP(seed=s),
}

EXPERT_NAMES = list(_BUILDERS)
COMMITTEE_DEFAULT = ["ridge", "knn", "extratrees", "hgb", "rbf"]
# Smaller committees for the ablation, nested so that size is the only change.
COMMITTEES = {2: ["ridge", "extratrees"], 3: ["ridge", "knn", "extratrees"], 5: COMMITTEE_DEFAULT}
# Faster settings used only for probe tasks during pretraining (documented in the methodology).
_PROBE_OVERRIDES = {"extratrees": {"n_estimators": 50}, "hgb": {"max_iter": 100}}


def make_expert(name: str, seed: int = 0, probe: bool = False) -> BaseEstimator:
    if name not in _BUILDERS:
        raise KeyError(f"unknown expert {name!r}; available: {EXPERT_NAMES}")
    est = _BUILDERS[name](seed)
    if probe and name in _PROBE_OVERRIDES:
        est.set_params(**_PROBE_OVERRIDES[name])
    return est


def fit_predict_committee(names: list[str], X_ctx: np.ndarray, y_ctx: np.ndarray, X_qry: np.ndarray, seed: int,
                          probe: bool = False) -> np.ndarray:
    out = np.empty((len(names), X_qry.shape[0]))
    for k, name in enumerate(names):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[k] = make_expert(name, seed, probe).fit(X_ctx, y_ctx).predict(X_qry)
    return out
