"""Train-only tabular encoding: drop constant/all-missing columns, ordinal-encode categoricals/strings.

Categories unseen at fit time become NaN (later imputed). Codes are assigned by descending training frequency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TabularEncoder:
    def fit(self, X: pd.DataFrame) -> "TabularEncoder":
        X = pd.DataFrame(X)
        self.columns_, self.categorical_, self.maps_ = [], [], {}
        for c in X.columns:
            col = X[c]
            is_cat = not pd.api.types.is_numeric_dtype(col) or isinstance(col.dtype, pd.CategoricalDtype)
            nonnull = col.dropna()
            if len(nonnull) == 0 or nonnull.nunique() <= 1:
                continue  # constant or all-missing: carries no information in training data
            self.columns_.append(c)
            if is_cat:
                vc = nonnull.astype(str).value_counts()
                self.maps_[c] = {k: float(i) for i, k in enumerate(vc.index)}
                self.categorical_.append(c)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        out = np.empty((len(X), len(self.columns_)), dtype=float)
        for j, c in enumerate(self.columns_):
            if c in self.maps_:
                s = X[c].astype(object).where(X[c].notna(), None)
                out[:, j] = [self.maps_[c].get(str(v), np.nan) if v is not None else np.nan for v in s]
            else:
                out[:, j] = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
        return out

    @property
    def categorical_mask_(self) -> np.ndarray:
        return np.array([c in self.maps_ for c in self.columns_], dtype=bool)
