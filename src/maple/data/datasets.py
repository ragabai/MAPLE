"""Benchmark datasets: OpenML-CTR23 (suite 353) plus scikit-learn's Diabetes. Cached under experiments/data/."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

from maple.utils import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "experiments" / "data"
CTR23_SUITE = 353
REGIMES = (128, 512, 2048)
REPEATS = {128: 10, 512: 5, 2048: 3}
EXCLUDED = {
    "boston_housing": "excluded: ethically problematic historical feature (racial-composition variable); "
                      "California Housing is used instead",
    "yacht_hydrodynamics": "not in OpenML-CTR23; with n=308 no regime leaves >=200 test rows at n_train>=128",
}


def _path(name: str) -> Path:
    return DATA_DIR / f"{name}.pkl"


def benchmark_names() -> list[str]:
    meta = DATA_DIR / "metadata.json"
    if not meta.exists():
        raise FileNotFoundError("run `python scripts/download_datasets.py` first")
    return [m["name"] for m in json.loads(meta.read_text())]


def load(name: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    p = _path(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `python scripts/download_datasets.py`")
    with p.open("rb") as fh:
        d = pickle.load(fh)
    return d["X"], d["y"], d["meta"]


def describe(name: str, X: pd.DataFrame, y: np.ndarray, source: dict) -> dict:
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)]
    return {"name": name, **source, "n": int(len(y)), "n_features": int(X.shape[1]),
            "n_numerical": int(X.shape[1] - len(cat)), "n_categorical": len(cat),
            "n_missing_cells": int(X.isna().sum().sum()), "missing_frac": float(X.isna().to_numpy().mean()),
            "target_mean": float(np.mean(y)), "target_std": float(np.std(y)), "target_min": float(np.min(y)),
            "target_max": float(np.max(y)), "target_skew": float(skew(y)), "n_unique_target": int(len(np.unique(y))),
            "size_class": "small" if len(y) < 2000 else ("medium" if len(y) <= 20000 else "large"),
            "regimes": [r for r in REGIMES if len(y) - r >= 200],
            "preprocessing": "train-only: drop constant/all-missing columns; ordinal-encode categoricals; median "
                             "imputation; log of strictly positive skewed columns; standardise; clip +-6; >32 "
                             "features -> mutual-information selection (PFN only)"}


def fetch_openml(did: int) -> tuple[pd.DataFrame, np.ndarray, dict]:
    import shutil
    import time

    import openml

    for attempt in range(4):
        try:
            ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                             download_features_meta_data=False)
            X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
            break
        except Exception:  # network errors leave partial files in the OpenML cache: remove and retry
            if attempt == 3:
                raise
            for d in Path(openml.config.get_cache_directory()).rglob(f"*{did}*"):
                shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink(missing_ok=True)
            time.sleep(5 * (attempt + 1))
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y)
    X = X.loc[keep].reset_index(drop=True)
    src = {"source": f"OpenML dataset {did}", "openml_id": did, "url": f"https://www.openml.org/d/{did}",
           "licence": ds.licence or "not stated", "citation": (ds.citation or "")[:300],
           "original_name": ds.name, "target": ds.default_target_attribute, "rows_dropped_nonfinite_target": int((~keep).sum())}
    return X, y[keep], src


def fetch_diabetes() -> tuple[pd.DataFrame, np.ndarray, dict]:
    from sklearn.datasets import load_diabetes

    d = load_diabetes(as_frame=True, scaled=False)
    src = {"source": "scikit-learn load_diabetes (Efron et al., 2004)", "openml_id": None,
           "url": "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html",
           "licence": "distributed with scikit-learn (BSD-3-Clause)", "citation": "Efron et al. (2004) Least angle regression",
           "original_name": "diabetes", "target": "disease progression", "rows_dropped_nonfinite_target": 0}
    return d.data, d.target.to_numpy(dtype=float), src


def save(name: str, X: pd.DataFrame, y: np.ndarray, meta: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _path(name).open("wb") as fh:
        pickle.dump({"X": X, "y": y, "meta": meta}, fh)
