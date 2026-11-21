"""Regression metrics. NRMSE = RMSE / std(y_test) (scale-free; 1.0 = mean predictor on the test set)."""
from __future__ import annotations

import numpy as np


def compute(y: np.ndarray, yhat: np.ndarray, sigma_ref: float | None = None) -> dict[str, float]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = yhat - y
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    sd = float(sigma_ref if sigma_ref is not None else y.std())
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"rmse": rmse, "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
            "nrmse": rmse / sd if sd > 0 else float("nan")}
