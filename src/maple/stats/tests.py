"""Multi-dataset statistical comparison (Demšar 2006; Benavoli et al. 2016).

Input convention: a DataFrame with one row per independent unit (dataset, or synthetic task) and one column per
method; values are a loss (lower is better), already averaged over repeats within each unit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def avg_ranks(df: pd.DataFrame) -> pd.Series:
    return df.rank(axis=1, method="average").mean(axis=0).sort_values()


def friedman(df: pd.DataFrame) -> dict:
    df = df.dropna()
    N, k = df.shape
    res = stats.friedmanchisquare(*[df[c].to_numpy() for c in df.columns])
    chi2 = float(res.statistic)
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) - chi2 > 0 else float("inf")
    p_ff = float(stats.f.sf(ff, k - 1, (k - 1) * (N - 1))) if np.isfinite(ff) else 0.0
    return {"chi2": chi2, "p": float(res.pvalue), "df": k - 1, "N": N, "k": k, "iman_davenport_F": float(ff),
            "iman_davenport_p": p_ff, "avg_ranks": avg_ranks(df).to_dict()}


def nemenyi_q(k: int, alpha: float = 0.05) -> float:
    """Critical value of the studentised range statistic divided by sqrt(2) (Demšar 2006, Table 5)."""
    return float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))


def nemenyi_cd(k: int, N: int, alpha: float = 0.05) -> float:
    return nemenyi_q(k, alpha) * np.sqrt(k * (k + 1) / (6.0 * N))


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014): (T+ - T-) / (T+ + T-), zero differences dropped."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    tp, tm = r[d > 0].sum(), r[d < 0].sum()
    return float((tp - tm) / (tp + tm))


def bootstrap_ci(v: np.ndarray, stat=np.mean, B: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    bs = np.apply_along_axis(stat, 1, v[idx]) if stat is not np.mean else v[idx].mean(axis=1)
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def wilcoxon_holm(df: pd.DataFrame, ref: str, alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Two-sided Wilcoxon signed-rank test of ``ref`` vs every other column, Holm-corrected.

    d = other - ref (positive => ref has the lower loss, i.e. ref is better). Effect size: rank-biserial r on d.
    ``median_diff`` and its bootstrap CI are for the per-unit relative difference (other - ref) / ref.
    """
    df = df.dropna()
    rows = []
    for c in df.columns:
        if c == ref:
            continue
        d = (df[c] - df[ref]).to_numpy()
        rel = d / np.maximum(np.abs(df[ref].to_numpy()), 1e-12)
        if np.allclose(d, 0):
            W, p = float("nan"), 1.0
        else:
            res = stats.wilcoxon(df[c], df[ref], zero_method="wilcox", alternative="two-sided")
            W, p = float(res.statistic), float(res.pvalue)
        lo, hi = bootstrap_ci(rel, np.median, B=4000, seed=seed)
        rows.append({"method": c, "W": W, "p": p, "r_rb": rank_biserial(d), "median_diff": float(np.median(rel)),
                     "mean_diff": float(np.mean(rel)), "ci_low": lo, "ci_high": hi,
                     "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()), "N": len(d)})
    out = pd.DataFrame(rows).set_index("method")
    out["p_holm"] = multipletests(out["p"].to_numpy(), method="holm")[1]
    out["significant"] = out["p_holm"] < alpha
    return out
