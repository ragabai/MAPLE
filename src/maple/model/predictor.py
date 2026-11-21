"""scikit-learn style regressor wrapping a pretrained PFN checkpoint (Algorithm 2, steps 1-4)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import mutual_info_regression

from maple.data.tabular import TabularEncoder
from maple.model.pfn import PFN
from maple.tasks.preprocess import fit_preprocessor, transform


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@lru_cache(maxsize=16)
def _load(path: str, device: str) -> PFN:
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = st.get("model_config") or st["cfg_identity"]["model"]
    m = PFN(**cfg)
    m.load_state_dict(st["model"])
    return m.to(torch.device(device)).eval()


@torch.no_grad()
def pfn_predict_arrays(model: PFN, X_ctx: np.ndarray, y_ctx: np.ndarray, X_qry: np.ndarray,
                       device: torch.device, chunk: int = 1024) -> np.ndarray:
    """Predict standardised targets for preprocessed arrays (context targets must be standardised)."""
    Xc = torch.tensor(X_ctx, dtype=torch.float32, device=device)[None]
    yc = torch.tensor(y_ctx, dtype=torch.float32, device=device)[None]
    out = []
    for i in range(0, X_qry.shape[0], chunk):
        Xq = torch.tensor(X_qry[i:i + chunk], dtype=torch.float32, device=device)[None]
        use_amp = device.type == "cuda"
        dt = torch.bfloat16 if use_amp and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        with torch.autocast(device.type, dtype=dt, enabled=use_amp):
            out.append(model(Xc, yc, Xq).float().cpu().numpy()[0])
    return np.concatenate(out) if out else np.zeros(0)


class MapleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, checkpoint: str | Path, n_ensemble: int = 4, max_ctx: int = 1024, device: str = "auto",
                 seed: int = 0):
        self.checkpoint = checkpoint
        self.n_ensemble = n_ensemble
        self.max_ctx = max_ctx
        self.device = device
        self.seed = seed

    def _model(self) -> tuple[PFN, torch.device]:
        dev = resolve_device(self.device)
        return _load(str(Path(self.checkpoint).resolve()), str(dev)), dev

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        model, _ = self._model()
        rng = np.random.default_rng(self.seed)
        self.encoder_ = TabularEncoder().fit(X)
        Z = self.encoder_.transform(X)
        if Z.shape[1] == 0:
            Z = np.zeros((len(X), 1))
            self.encoder_ = None
        d_max = model.d_max
        if Z.shape[1] > d_max:
            Zi = np.where(np.isfinite(Z), Z, np.nanmedian(Z, axis=0))
            Zi = np.where(np.isfinite(Zi), Zi, 0.0)
            sub = rng.choice(len(Zi), size=min(len(Zi), 2000), replace=False)
            mi = mutual_info_regression(Zi[sub], y[sub], random_state=self.seed)
            self.selected_ = np.sort(np.argsort(-mi, kind="stable")[:d_max])
        else:
            self.selected_ = np.arange(Z.shape[1])
        Z = Z[:, self.selected_]
        n, d = Z.shape
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std())
        self.ctx_idx_, self.perms_, self.members_ = [], [], []
        for m in range(self.n_ensemble):
            idx = np.sort(rng.choice(n, size=self.max_ctx, replace=False)) if n > self.max_ctx else np.arange(n)
            perm = np.arange(d) if m == 0 else rng.permutation(d)
            p = fit_preprocessor(Z[idx])
            yc = y[idx]
            ym, ys = float(yc.mean()), float(yc.std())
            ys = ys if ys > 1e-12 else 1.0
            self.ctx_idx_.append(idx)
            self.perms_.append(perm)
            self.members_.append({"prep": p, "Xc": transform(Z[idx], p)[:, perm], "yc": (yc - ym) / ys,
                                  "ym": ym, "ys": ys})
        return self

    def _encode(self, X) -> np.ndarray:
        X = pd.DataFrame(X).reset_index(drop=True)
        if self.encoder_ is None:
            return np.zeros((len(X), 1))
        return self.encoder_.transform(X)[:, self.selected_]

    def predict(self, X) -> np.ndarray:
        model, dev = self._model()
        Z = self._encode(X)
        preds = []
        for mem, perm in zip(self.members_, self.perms_):
            Xq = transform(Z, mem["prep"])[:, perm]
            p = pfn_predict_arrays(model, mem["Xc"], mem["yc"], Xq, dev)
            preds.append(p * mem["ys"] + mem["ym"])
        return np.mean(preds, axis=0)
