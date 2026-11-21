"""Standalone baseline regressors on raw DataFrames, with train-only preprocessing inside each pipeline.

Defaults are library defaults unless stated. ``*_tuned`` variants run a fixed-budget random search
(``n_trials`` configurations, 3-fold CV on the training split only) and refit the best configuration.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from maple.data.tabular import TabularEncoder
from maple.utils import rng as mk_rng


def _cat_cols(X: pd.DataFrame) -> list:
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)]


class OneHotPipe(BaseEstimator, RegressorMixin):
    """Numeric: median impute + standardise; categorical: most-frequent impute + one-hot (<= 20 levels)."""

    def __init__(self, est, scale_y: bool = False):
        self.est, self.scale_y = est, scale_y

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        cats = _cat_cols(X)
        for c in cats:
            X[c] = X[c].astype(str).where(X[c].notna(), np.nan)
        nums = [c for c in X.columns if c not in cats]
        ct = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), nums),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                                  OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20,
                                                sparse_output=False)), cats)])
        est = TransformedTargetRegressor(clone(self.est), transformer=StandardScaler()) if self.scale_y else clone(self.est)
        self.cats_ = cats
        self.pipe_ = Pipeline([("ct", ct), ("est", est)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipe_.fit(X, y)
        return self

    def predict(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cats_:
            X[c] = X[c].astype(str).where(X[c].notna(), np.nan)
        return self.pipe_.predict(X)


class OrdinalPipe(BaseEstimator, RegressorMixin):
    """Ordinal codes for categoricals; optional median imputation (for models without native NaN support)."""

    def __init__(self, est, impute: bool = True):
        self.est, self.impute = est, impute

    def fit(self, X, y):
        self.enc_ = TabularEncoder().fit(pd.DataFrame(X))
        Z = self._z(X, fit=True)
        self.model_ = clone(self.est)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(Z, y)
        return self

    def _z(self, X, fit=False):
        Z = self.enc_.transform(pd.DataFrame(X))
        if Z.shape[1] == 0:
            Z = np.zeros((len(Z), 1))
        if self.impute:
            if fit:
                self.med_ = np.nan_to_num(np.nanmedian(np.where(np.isfinite(Z), Z, np.nan), axis=0))
            Z = np.where(np.isfinite(Z), Z, self.med_)
        return Z

    def predict(self, X):
        return self.model_.predict(self._z(X))


class CatBoostPipe(BaseEstimator, RegressorMixin):
    def __init__(self, seed: int = 0, **params):
        self.seed, self.params = seed, params

    def get_params(self, deep=True):
        return {"seed": self.seed, **self.params}

    def set_params(self, **p):
        self.seed = p.pop("seed", self.seed)
        self.params.update(p)
        return self

    def fit(self, X, y):
        from catboost import CatBoostRegressor

        X = pd.DataFrame(X).copy()
        self.cats_ = _cat_cols(X)
        for c in self.cats_:
            X[c] = X[c].astype(str).fillna("NA")
        # one_hot_max_size=255: categoricals with <= 255 levels are one-hot encoded; CatBoost's CTR path is
        # ~20x slower single-threaded (measured ~16 s per 100 trees at n=512 on the benchmark host)
        params = {"one_hot_max_size": 255, **self.params}
        self.model_ = CatBoostRegressor(verbose=0, thread_count=1, random_seed=self.seed, allow_writing_files=False,
                                        **params)
        self.model_.fit(X, y, cat_features=self.cats_)
        return self

    def predict(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cats_:
            X[c] = X[c].astype(str).fillna("NA")
        return self.model_.predict(X)


def _xgb(seed, **p):
    from xgboost import XGBRegressor
    return OrdinalPipe(XGBRegressor(n_jobs=1, random_state=seed, verbosity=0, **p), impute=False)


def _lgbm(seed, **p):
    from lightgbm import LGBMRegressor
    return OrdinalPipe(LGBMRegressor(n_jobs=1, random_state=seed, verbose=-1, **p), impute=False)


def _space_xgb(r):
    return {"n_estimators": int(r.integers(100, 1001)), "learning_rate": float(np.exp(r.uniform(np.log(0.01), np.log(0.3)))),
            "max_depth": int(r.integers(2, 11)), "subsample": float(r.uniform(0.5, 1.0)),
            "colsample_bytree": float(r.uniform(0.5, 1.0)), "min_child_weight": float(np.exp(r.uniform(0, np.log(20)))),
            "reg_lambda": float(np.exp(r.uniform(np.log(1e-3), np.log(10))))}


def _space_lgbm(r):
    return {"n_estimators": int(r.integers(100, 1001)), "learning_rate": float(np.exp(r.uniform(np.log(0.01), np.log(0.3)))),
            "num_leaves": int(r.integers(8, 129)), "min_child_samples": int(r.integers(5, 51)),
            "subsample": float(r.uniform(0.5, 1.0)), "subsample_freq": 1, "colsample_bytree": float(r.uniform(0.5, 1.0)),
            "reg_lambda": float(np.exp(r.uniform(np.log(1e-3), np.log(10))))}


def _space_cat(r):
    return {"iterations": int(r.integers(200, 1001)), "learning_rate": float(np.exp(r.uniform(np.log(0.01), np.log(0.3)))),
            "depth": int(r.integers(3, 9)), "l2_leaf_reg": float(np.exp(r.uniform(0, np.log(10))))}


class TunedRegressor(BaseEstimator, RegressorMixin):
    """Random search with a fixed trial budget; 3-fold CV MSE on the training split only."""

    def __init__(self, kind: str, seed: int = 0, n_trials: int = 10, folds: int = 3):
        self.kind, self.seed, self.n_trials, self.folds = kind, seed, n_trials, folds

    def _build(self, params):
        return {"xgboost": lambda: _xgb(self.seed, **params), "lightgbm": lambda: _lgbm(self.seed, **params),
                "catboost": lambda: CatBoostPipe(self.seed, **params)}[self.kind]()

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, float)
        space = {"xgboost": _space_xgb, "lightgbm": _space_lgbm, "catboost": _space_cat}[self.kind]
        r = mk_rng(self.seed, "hpo", self.kind, len(y))
        self.trials_ = []
        kf = KFold(self.folds, shuffle=True, random_state=self.seed)
        for _ in range(self.n_trials):
            params = space(r)
            errs = []
            for tr, va in kf.split(X):
                m = self._build(params).fit(X.iloc[tr], y[tr])
                errs.append(np.mean((m.predict(X.iloc[va]) - y[va]) ** 2))
            self.trials_.append({"params": params, "cv_mse": float(np.mean(errs))})
        self.best_params_ = min(self.trials_, key=lambda t: t["cv_mse"])["params"]
        self.model_ = self._build(self.best_params_).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(pd.DataFrame(X))


class FoundationModel(BaseEstimator, RegressorMixin):
    """External pretrained tabular foundation models with library defaults (not equal-compute comparisons):
    'tabpfn_v2' (TabPFN v2 regression weights, Prior Labs License) and 'tabicl' (TabICL v2 regressor, BSD-3).
    Device from MAPLE_FM_DEVICE (default: cuda if available). Usage telemetry is disabled."""

    def __init__(self, kind: str, seed: int = 0):
        self.kind, self.seed = kind, seed

    def fit(self, X, y):
        import os
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import torch
        dev = os.environ.get("MAPLE_FM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.kind == "tabpfn_v2":
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
            self.model_ = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=dev,
                                                                      random_state=self.seed)
        elif self.kind == "tabicl":
            from tabicl import TabICLRegressor
            self.model_ = TabICLRegressor(device=dev, random_state=self.seed)
        else:
            raise KeyError(self.kind)
        self.model_.fit(np.asarray(X, np.float32), np.asarray(y, np.float32))
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X, np.float32)), float)


def make(name: str, seed: int = 0, n_trials: int = 10) -> BaseEstimator:
    alphas = np.logspace(-3, 3, 13)
    builders = {
        "mean": lambda: OrdinalPipe(DummyRegressor()),
        "ridge": lambda: OneHotPipe(RidgeCV(alphas=alphas)),
        "elasticnet": lambda: OneHotPipe(ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=20, cv=3, max_iter=5000,
                                                      random_state=seed), scale_y=True),
        "knn": lambda: OneHotPipe(KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "rf": lambda: OrdinalPipe(RandomForestRegressor(n_estimators=300, n_jobs=1, random_state=seed)),
        "extratrees": lambda: OrdinalPipe(ExtraTreesRegressor(n_estimators=300, n_jobs=1, random_state=seed)),
        "xgboost": lambda: _xgb(seed),
        "lightgbm": lambda: _lgbm(seed),
        "catboost": lambda: CatBoostPipe(seed),
        "mlp": lambda: OneHotPipe(MLPRegressor(hidden_layer_sizes=(256, 256), alpha=1e-4, early_stopping=True,
                                               max_iter=500, random_state=seed), scale_y=True),
        "xgboost_tuned": lambda: TunedRegressor("xgboost", seed, n_trials),
        "lightgbm_tuned": lambda: TunedRegressor("lightgbm", seed, n_trials),
        "catboost_tuned": lambda: TunedRegressor("catboost", seed, n_trials),
        # ordinal-coded categoricals for both; TabPFN handles NaN natively, TabICL gets median imputation
        "tabpfn_v2": lambda: OrdinalPipe(FoundationModel("tabpfn_v2", seed), impute=False),
        "tabicl": lambda: OrdinalPipe(FoundationModel("tabicl", seed), impute=True),
    }
    if name not in builders:
        raise KeyError(f"unknown baseline {name!r}")
    return builders[name]()


STANDALONE = ["mean", "ridge", "elasticnet", "knn", "rf", "extratrees", "xgboost", "lightgbm", "catboost", "mlp",
              "xgboost_tuned", "lightgbm_tuned", "catboost_tuned"]
EXTERNAL = ["tabpfn_v2", "tabicl"]  # external reference models (GPU; run by run_benchmarks --stage external)
