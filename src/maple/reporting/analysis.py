"""Loading and aggregating raw records. Every reported number is computed here from experiments/raw/*.jsonl."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from maple.utils import PROJECT_ROOT

RAW = PROJECT_ROOT / "experiments" / "raw"
MAIN_VARIANTS = ("P0", "A", "B", "C", "D")

CLASSICAL_LABELS = {
    "mean": "Mean", "ridge": "Ridge", "elasticnet": "ElasticNet", "knn": "kNN", "rf": "RF", "extratrees": "ExtraTrees",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "mlp": "MLP",
    "xgboost_tuned": "XGBoost-T", "lightgbm_tuned": "LightGBM-T", "catboost_tuned": "CatBoost-T",
    "avg_experts": "Avg(experts)", "stack_experts": "Stack(experts)", "expert_hgb": "HistGB", "expert_rbf": "RBF-KRR",
    "expert_ridge": "Ridge(expert)", "expert_knn": "kNN(expert)", "expert_extratrees": "ExtraTrees(expert)",
    "hgb": "HistGB", "rbf": "RBF-KRR",
    "tabpfn_v2": "TabPFN-v2", "tabicl": "TabICL",  # external pretrained foundation models (not equal-compute)
}
EXTERNAL_LABELS = ("TabPFN-v2", "TabICL")
VARIANT_LABEL = {"P0": "MAPLE-Net/SCM", "A": "MAPLE-Net/Uniform", "B": "MAPLE-Net/Loss", "C": "MAPLE-Net/Gap", "D": "MAPLE-Net"}
VARIANT_DESC = {"P0": "conventional SCM/MLP prior", "A": "uniform mathematical prior",
                "B": "adaptive, loss score", "C": "adaptive, disagreement score, no floor",
                "D": "adaptive, disagreement score + floor", "MAPLE-Global": "D + global routing",
                "MAPLE": "D + local competence routing (full framework)"}
# Routed variants (E: global, F: local competence routing) and helpers. Names that share the "MAPLE" prefix
# are told apart by these sets, never by prefix: "MAPLE-Net..." is a standalone network, not a routed method.
ROUTED_LABEL = {"E": "MAPLE-Global", "F": "MAPLE"}
NET_LABELS = tuple(VARIANT_LABEL.values())
ROUTED_LABELS = tuple(ROUTED_LABEL.values())


def is_net(label: str) -> bool:
    """Standalone pretrained network (any prior), including unnamed ablation networks MAPLE-Net[...]."""
    return label in NET_LABELS or label.startswith("MAPLE-Net[")


def is_routed(label: str) -> bool:
    return label in ROUTED_LABELS or label.startswith("Route-")


def is_ours(label: str) -> bool:
    """Any method built on the pretrained network: standalone, routed or stacked with the experts."""
    return is_net(label) or is_routed(label) or label.startswith("Stack(MAPLE-Net")

_RX = re.compile(r"^(pfn|E|F|stackpfn)_(.+)_s(\d+)$")


def parse_method(m: str) -> dict:
    mm = _RX.match(m)
    if not mm:
        return {"family": "classical", "variant": None, "seed": None, "label": CLASSICAL_LABELS.get(m, m)}
    fam, v, s = mm.group(1), mm.group(2), int(mm.group(3))
    vl = VARIANT_LABEL.get(v, f"MAPLE-Net[{v}]")
    if fam == "pfn":
        label = vl
    elif fam in ("E", "F"):
        label = ROUTED_LABEL[fam] if v == "D" else f"Route-{fam}({v})"
    else:
        label = f"Stack({vl})"
    return {"family": "pfn" if fam == "pfn" else "routed", "variant": v, "seed": s, "label": label}


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def real_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        info = parse_method(r["method"])
        rows.append({"dataset": r["dataset"], "regime": r["regime"], "repeat": r["repeat"], "method": r["method"],
                     **info, **r["metrics"], "fit_s": r.get("fit_s"), "predict_s": r.get("predict_s"),
                     "peak_mem_mb": r.get("peak_mem_mb"), "cuda_peak_mb": r.get("cuda_peak_mb"),
                     "n_train": r.get("n_train"), "n_test": r.get("n_test")})
    return pd.DataFrame(rows)


def per_dataset(df: pd.DataFrame, metric: str, regime: int | None = None, complete: bool = True,
                labels: list[str] | None = None, agg: str = "mean") -> pd.DataFrame:
    """Dataset x label table: mean over repeats (and pretraining seeds) within each dataset."""
    d = df if regime is None else df[df.regime == regime]
    if labels is not None:
        d = d[d.label.isin(labels)]
    t = d.groupby(["dataset", "label"])[metric].agg(agg).unstack("label")
    if labels is not None:
        t = t[[l for l in labels if l in t.columns]]
    return t.dropna() if complete else t


def seed_sd(df: pd.DataFrame, metric: str, regime: int) -> pd.DataFrame:
    """Dataset x PFN-label: standard deviation across pretraining seeds of the repeat-mean metric."""
    d = df[(df.regime == regime) & df.seed.notna()]
    m = d.groupby(["dataset", "label", "seed"])[metric].mean()
    return m.groupby(["dataset", "label"]).std().unstack("label")


def repeat_sd(df: pd.DataFrame, metric: str, regime: int, labels=None) -> pd.DataFrame:
    d = df[df.regime == regime]
    if labels is not None:
        d = d[d.label.isin(labels)]
    m = d.groupby(["dataset", "label", "repeat"])[metric].mean()
    return m.groupby(["dataset", "label"]).std().unstack("label")


def synthetic_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        info = parse_method(r["method"]) if r["method"].startswith("pfn_") else \
            {"family": "classical", "variant": None, "seed": None, "label": CLASSICAL_LABELS.get(r["method"], r["method"])}
        rows.append({"suite": r["suite"], "task": r["task"], "method": r["method"], **info, **r["metrics"],
                     "gap": r.get("gap"), "excess": r.get("excess"), "committee_excess": r.get("committee_excess"),
                     "ambiguity": r.get("ambiguity"), "noise_rel": r.get("noise_rel"), "noise_type": r.get("noise_type"),
                     "family_task": r.get("family"), "complexity": r.get("complexity"), "n_ctx": r.get("n_ctx"),
                     "d": r.get("d"), "fit_predict_s": r.get("fit_predict_s")})
    return pd.DataFrame(rows)


def per_task(sdf: pd.DataFrame, metric: str, suites=None, labels=None) -> pd.DataFrame:
    """(suite, task) x label, averaging PFN seeds."""
    d = sdf if suites is None else sdf[sdf.suite.isin(suites)]
    if labels is not None:
        d = d[d.label.isin(labels)]
    return d.groupby(["suite", "task", "label"])[metric].mean().unstack("label").dropna()


def variant_of(label: str) -> str:
    inv = {v: k for k, v in VARIANT_LABEL.items()}
    return inv.get(label, label)


def gmean(x) -> float:
    x = np.asarray(x, float)
    return float(np.exp(np.mean(np.log(np.maximum(x, 1e-12)))))


def adaptation_summary(sampler_logs: dict[str, list[dict]]) -> dict:
    """Per variant (seeds pooled): KL(q || uniform) at the last round (mean over seeds) and its maximum over all
    rounds and seeds, plus the extreme final family probabilities (cells summed by family) across seeds."""
    by_v: dict[str, list[list[dict]]] = {}
    for k, lg in sampler_logs.items():
        if lg and "_s" in k:
            by_v.setdefault(k.rsplit("_s", 1)[0], []).append(lg)
    out = {}
    for v, runs in by_v.items():
        finals, maxes, fam_max, fam_min, sds = [], [], [], [], []
        for lg in runs:
            if lg[-1].get("scores") is not None:  # spread of the cell scores that drive the Gibbs weights
                sds.append(float(np.std(np.asarray(lg[-1]["scores"], float))))
            kls = []
            for r in lg:
                q = np.asarray(r["probs"], float)
                kls.append(float(np.sum(q * np.log(q * len(q)))))
            finals.append(kls[-1])
            maxes.append(max(kls))
            fam = pd.Series(lg[-1]["probs"], index=[c.split("|")[0] for c in lg[-1]["cells"]]).groupby(level=0).sum()
            fam_max.append(float(fam.max()))
            fam_min.append(float(fam.min()))
        out[v] = {"n_seeds": len(runs), "kl_final_mean": float(np.mean(finals)), "kl_max": float(max(maxes)),
                  "family_prob_final_max": max(fam_max), "family_prob_final_min": min(fam_min),
                  "n_families": int(len(fam)), "score_sd_final_mean": float(np.mean(sds)) if sds else None}
    return out


def scaling_summary(s: pd.DataFrame) -> pd.DataFrame:
    """Mean fit+predict time and NRMSE per (n_train, label), over datasets x repeats, restricted to the training sizes
    that exist for EVERY dataset (a size missing for some datasets would change the dataset mix, not the method)."""
    n_ds = s.dataset.nunique()
    full = s.groupby("regime").dataset.nunique()
    keep = full.index[full == n_ds]
    d = s[s.regime.isin(keep)].assign(total_s=lambda x: x.fit_s + x.predict_s)
    agg = d.groupby(["regime", "label"]).agg(total_s=("total_s", "mean"), nrmse=("nrmse", "mean")).reset_index()
    return agg.rename(columns={"regime": "n_train"})
