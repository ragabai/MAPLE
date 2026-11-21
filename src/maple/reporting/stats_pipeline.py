"""All statistical analyses (research_questions.md), computed from raw records. Writes results/statistics/*.json."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from maple.reporting import analysis as A
from maple.stats import tests as T
from maple.utils import PROJECT_ROOT, write_json

STATS = PROJECT_ROOT / "results" / "statistics"
REF = "MAPLE-F"
MAIN_SET = ["Ridge", "kNN", "RF", "ExtraTrees", "HistGB", "MLP", "XGBoost-T", "LightGBM-T", "CatBoost-T",
            "Stack(experts)", "PFN-SCM", "PFN-A", "PFN-D", "MAPLE-E", "MAPLE-F"]
FULL_SET = ["Mean", "Ridge", "ElasticNet", "kNN", "RF", "ExtraTrees", "HistGB", "RBF-KRR", "MLP", "XGBoost",
            "LightGBM", "CatBoost", "XGBoost-T", "LightGBM-T", "CatBoost-T", "Avg(experts)", "Stack(experts)",
            "PFN-SCM", "PFN-A", "PFN-B", "PFN-C", "PFN-D", "Stack(PFN-D)", "MAPLE-E", "MAPLE-F",
            "TabPFN-v2", "TabICL"]  # last two: external references, never in MAIN_SET (pre-registered Friedman set)
OOD_SUITES = ["ood_families", "ood_nested", "ood_params", "ood_interact3", "ood_voronoi", "ood_noise", "ood_mixed"]
PFN_LABELS = ["PFN-SCM", "PFN-A", "PFN-B", "PFN-C", "PFN-D"]


def _wtab(df: pd.DataFrame, ref: str) -> dict:
    if ref not in df.columns or df.shape[0] < 5:
        return {}
    return T.wilcoxon_holm(df, ref).reset_index().to_dict(orient="records")


def _pair(df: pd.DataFrame, a: str, b: str) -> dict:
    """Two-sided Wilcoxon for a vs b (loss); positive median_diff => a better."""
    if a not in df or b not in df:
        return {}
    sub = df[[a, b]].dropna()
    if len(sub) < 5:
        return {}
    r = T.wilcoxon_holm(sub, ref=a).iloc[0].to_dict()
    r.update({"a": a, "b": b, "mean_a": float(sub[a].mean()), "mean_b": float(sub[b].mean()),
              "median_a": float(sub[a].median()), "median_b": float(sub[b].median())})
    return r


def _holm_family(rq: dict, keys: list[str]) -> None:
    """Holm correction within one pre-registered family of tests (in place: p_holm_family, family_size)."""
    from statsmodels.stats.multitest import multipletests
    present = [k for k in keys if rq.get(k) and not np.isnan(rq[k].get("p", np.nan))]
    if not present:
        return
    adj = multipletests([rq[k]["p"] for k in present], method="holm")[1]
    for k, a in zip(present, adj):
        rq[k]["p_holm_family"] = float(a)
        rq[k]["family_size"] = len(present)


def real_stats(df: pd.DataFrame, regimes) -> dict:
    out = {}
    for reg in regimes:
        res = {}
        for metric in ("nrmse", "mae_rel", "r2"):
            if metric not in df:
                continue
            labels = [l for l in FULL_SET if l in set(df.label)]
            full = A.per_dataset(df, metric, reg, complete=False, labels=labels)
            res[f"{metric}_per_dataset"] = full.round(6).to_dict()
        tab = A.per_dataset(df, "nrmse", reg, complete=True, labels=[l for l in MAIN_SET if l in set(df.label)])
        if tab.shape[0] >= 5 and tab.shape[1] >= 3:
            fr = T.friedman(tab)
            res["friedman"] = fr
            res["nemenyi_cd"] = T.nemenyi_cd(tab.shape[1], tab.shape[0], 0.05)
            res["wilcoxon_vs_ref"] = _wtab(tab, REF)
            res["n_datasets"] = int(tab.shape[0])
            res["methods"] = list(tab.columns)
        allm = A.per_dataset(df, "nrmse", reg, complete=False, labels=[l for l in FULL_SET if l in set(df.label)])
        rq = {}
        rq["RQ4_A_vs_P0"] = _pair(allm, "PFN-A", "PFN-SCM")
        rq["RQ4_D_vs_P0"] = _pair(allm, "PFN-D", "PFN-SCM")
        rq["RQ1_real_D_vs_A"] = _pair(allm, "PFN-D", "PFN-A")
        rq["RQ2_real_C_vs_B"] = _pair(allm, "PFN-C", "PFN-B")
        for b in ("PFN-D", "Avg(experts)", "Stack(experts)", "Stack(PFN-D)", "MAPLE-E"):
            rq[f"RQ5_F_vs_{b}"] = _pair(allm, "MAPLE-F", b)
        for b in ("CatBoost-T", "XGBoost-T", "LightGBM-T", "CatBoost", "RF", "ExtraTrees"):
            rq[f"RQ6_F_vs_{b}"] = _pair(allm, "MAPLE-F", b)
        _holm_family(rq, ["RQ4_A_vs_P0", "RQ4_D_vs_P0"])
        _holm_family(rq, ["RQ1_real_D_vs_A", "RQ2_real_C_vs_B"])
        _holm_family(rq, [k for k in rq if k.startswith("RQ5_")])
        _holm_family(rq, [k for k in rq if k.startswith("RQ6_")])
        res["rq"] = rq
        # exploratory (not pre-registered; raw p, no family correction): does PFN-D add to the stacked committee?
        res["exploratory"] = {"StackPFN_vs_StackExperts": _pair(allm, "Stack(PFN-D)", "Stack(experts)"),
                              "E_vs_F": _pair(allm, "MAPLE-E", "MAPLE-F"),
                              "F_vs_TabPFN-v2": _pair(allm, "MAPLE-F", "TabPFN-v2"),
                              "F_vs_TabICL": _pair(allm, "MAPLE-F", "TabICL"),
                              "D_vs_TabPFN-v2": _pair(allm, "PFN-D", "TabPFN-v2"),
                              "D_vs_TabICL": _pair(allm, "PFN-D", "TabICL")}
        res["avg_rank_full"] = T.avg_ranks(allm.dropna()).to_dict() if allm.dropna().shape[0] else {}
        res["n_datasets_full"] = int(allm.dropna().shape[0])
        out[str(reg)] = res
    return out


def synthetic_stats(sdf: pd.DataFrame) -> dict:
    out = {}
    labels = sorted(set(sdf.label))
    for metric in ("nrmse", "nrmse_f"):
        per = sdf.groupby(["suite", "task", "label"])[metric].mean().unstack("label")  # seeds averaged, no dropna
        out[f"{metric}_suite_mean"] = per.groupby(level="suite").mean().round(6).to_dict()
        out[f"{metric}_suite_median"] = per.groupby(level="suite").median().round(6).to_dict()
        pt = A.per_task(sdf, metric, labels=["PFN-A", "PFN-D"])
        id_ = pt.xs("id", level="suite") if "id" in pt.index.get_level_values(0) else None
        ood = pt[pt.index.get_level_values(0).isin(OOD_SUITES)]
        pid = per.xs("id", level="suite") if "id" in per.index.get_level_values(0) else None
        pood = per[per.index.get_level_values(0).isin(OOD_SUITES)]
        gap = {}
        for l in per.columns:
            if pid is not None and l in pid:
                gap[l] = {"id": float(pid[l].mean()), "ood": float(pood[l].mean()),
                          "gap": float(pood[l].mean() - pid[l].mean())}
        out[f"{metric}_generalization_gap"] = gap
        rq1 = {"id": _pair(id_, "PFN-D", "PFN-A") if id_ is not None else {}, "pooled_ood": _pair(ood, "PFN-D", "PFN-A")}
        rq1_suites = {}
        ps = []
        for s in OOD_SUITES:
            if s in pt.index.get_level_values(0):
                r = _pair(pt.xs(s, level="suite"), "PFN-D", "PFN-A")
                rq1_suites[s] = r
                ps.append(r.get("p", np.nan))
        if ps and not np.isnan(ps).all():
            from statsmodels.stats.multitest import multipletests
            adj = multipletests(np.nan_to_num(ps, nan=1.0), method="holm")[1]
            for s, p in zip([s for s in OOD_SUITES if s in rq1_suites], adj):
                rq1_suites[s]["p_holm_across_suites"] = float(p)
        rq1["per_suite"] = rq1_suites
        _holm_family(rq1, ["id", "pooled_ood"])
        out[f"RQ1_{metric}"] = rq1
        p4 = A.per_task(sdf, metric, labels=["PFN-A", "PFN-SCM"])
        p4id = p4[p4.index.get_level_values(0) == "id"]
        p4ood = p4[p4.index.get_level_values(0).isin(OOD_SUITES)]
        rq4 = {"id_A_vs_P0": _pair(p4id, "PFN-A", "PFN-SCM"), "ood_A_vs_P0": _pair(p4ood, "PFN-A", "PFN-SCM")}
        _holm_family(rq4, ["id_A_vs_P0", "ood_A_vs_P0"])
        out[f"RQ4_synthetic_{metric}"] = rq4
        # RQ2: C vs B overall and by noise level (pre-registered metric: nrmse; nrmse_f reported alongside)
        p2 = A.per_task(sdf, metric, labels=["PFN-B", "PFN-C"])
        noise = sdf[sdf.label == "PFN-B"].groupby(["suite", "task"])["noise_rel"].first().reindex(p2.index)
        rq2 = {"all": _pair(p2, "PFN-C", "PFN-B"), "high_noise": _pair(p2[noise >= noise.median()], "PFN-C", "PFN-B"),
               "low_noise": _pair(p2[noise < noise.median()], "PFN-C", "PFN-B"), "noise_median": float(noise.median())}
        _holm_family(rq2, ["all", "high_noise", "low_noise"])
        out[f"RQ2_{metric}"] = rq2
    out["RQ2"] = out["RQ2_nrmse"]
    # RQ3: disagreement vs excess error; unit = task (pretraining seeds averaged first)
    rq3 = {}
    for l in PFN_LABELS:
        d0 = sdf[(sdf.label == l) & sdf.gap.notna()].copy()
        if len(d0) < 20:
            continue
        d0["loss"] = d0["rmse"] ** 2 if "rmse" in d0 else d0["nrmse"] ** 2
        d = d0.groupby(["suite", "task"])[["gap", "excess", "committee_excess", "loss"]].mean().reset_index()
        rho = spearmanr(d.gap, d.excess)[0]
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(d), len(d))
            bs.append(spearmanr(d.gap.values[i], d.excess.values[i])[0])
        rq3[l] = {"spearman": float(rho), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                  "n": int(len(d)), "frac_committee_better": float(np.mean(d.committee_excess < d.excess)),
                  "spearman_loss_vs_excess": float(spearmanr(d.loss, d.excess)[0]),
                  "by_suite": {s: float(spearmanr(g.gap, g.excess)[0]) for s, g in d.groupby("suite") if len(g) > 10}}
    out["RQ3"] = rq3
    # external reference models vs PFN-D on synthetic suites (exploratory; unit = task)
    ext = {}
    for l in A.EXTERNAL_LABELS:
        if l in set(sdf.label):
            pt = A.per_task(sdf, "nrmse", labels=["PFN-D", l])
            ext[l] = {"id": _pair(pt[pt.index.get_level_values(0) == "id"], "PFN-D", l),
                      "pooled_ood": _pair(pt[pt.index.get_level_values(0).isin(OOD_SUITES)], "PFN-D", l)}
    out["external"] = ext
    # pretraining-seed spread of the main variants (scale against which seed-0 ablation effects are read)
    spread = {}
    if "seed" in sdf:
        for l in PFN_LABELS:
            d = sdf[(sdf.label == l) & sdf.seed.notna()]
            if d.seed.nunique() < 2:
                continue
            m = d.assign(ood=d.suite != "id").groupby(["seed", "ood"])["nrmse"].mean().unstack("ood")
            ids, oods = [float(x) for x in m[False]], [float(x) for x in m[True]]
            spread[l] = {"seeds": [int(x) for x in m.index], "id": ids, "ood": oods,
                         "id_range": max(ids) - min(ids), "ood_range": max(oods) - min(oods)}
    out["seed_spread"] = spread
    return out


def rq3_real(real_dis: pd.DataFrame) -> dict:
    """RQ3 on real data: unit = dataset (gap and PFN error averaged over regimes/repeats), bootstrap over datasets."""
    if real_dis is None or len(real_dis) < 5:
        return {}
    d = real_dis.groupby("dataset")[["gap", "pfn_mse", "committee_mse"]].mean()
    rho = spearmanr(d.gap, d.pfn_mse)[0]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(d), len(d))
        bs.append(spearmanr(d.gap.values[i], d.pfn_mse.values[i])[0])
    within = [spearmanr(g.gap, g.pfn_mse)[0] for _, g in real_dis.groupby("dataset") if len(g) >= 5]
    # excess-error proxy (Prop. 4(d)): regret against the committee = excess(PFN) - excess(committee); unlike the
    # raw MSE it does not contain the dataset's irreducible noise
    d["regret"] = d.pfn_mse - d.committee_mse
    rho_r = spearmanr(d.gap, d.regret)[0]
    bs_r = []
    for _ in range(2000):
        i = rng.integers(0, len(d), len(d))
        bs_r.append(spearmanr(d.gap.values[i], d.regret.values[i])[0])
    within_r = [spearmanr(g.gap, g.pfn_mse - g.committee_mse)[0] for _, g in real_dis.groupby("dataset") if len(g) >= 5]
    return {"spearman_datasets": float(rho), "ci": [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))],
            "n_datasets": int(len(d)), "frac_committee_better": float(np.mean(d.committee_mse < d.pfn_mse)),
            "median_within_dataset_spearman": float(np.nanmedian(within)) if within else float("nan"),
            "spearman_regret": float(rho_r),
            "ci_regret": [float(np.nanquantile(bs_r, 0.025)), float(np.nanquantile(bs_r, 0.975))],
            "median_within_dataset_spearman_regret": float(np.nanmedian(within_r)) if within_r else float("nan")}


def ablation_stats(sdf: pd.DataFrame, df: pd.DataFrame, names: list[str], ref: str = "D") -> dict:
    """Each ablation/sensitivity checkpoint (seed 0) against D_s0 on pooled synthetic tasks and real n=512."""
    out = {}
    s0 = sdf[(sdf.seed == 0) | sdf.seed.isna()]
    syn = s0[s0.family == "pfn"].groupby(["suite", "task", "variant"])["nrmse"].mean().unstack("variant")
    real = df[(df.regime == 512) & (df.family == "pfn") & (df.seed == 0)]
    real_t = real.groupby(["dataset", "variant"])["nrmse"].mean().unstack("variant") if len(real) else pd.DataFrame()
    for n in names:
        rec = {}
        def rel(df_):
            r = _pair(df_, ref, n)  # median_diff = median((n - ref)/ref): positive = ablation worse than ref
            if r:
                r["delta_vs_ref"] = r["median_diff"]
                r["delta_ci"] = [r["ci_low"], r["ci_high"]]
            return r
        if n in syn and ref in syn:
            sub = syn[[ref, n]].dropna()
            rec["synthetic"] = rel(sub)
            rec["synthetic_id"] = rel(sub[sub.index.get_level_values(0) == "id"])
            rec["synthetic_ood"] = rel(sub[sub.index.get_level_values(0) != "id"])
        if len(real_t) and n in real_t and ref in real_t:
            rec["real512"] = rel(real_t[[ref, n]].dropna())
        out[n] = rec
    return out


def disagreement_real(cache_dir: Path, splits_meta: list[dict], variant_key: str = "D_s0") -> pd.DataFrame:
    """Per split: mean PFN-committee gap on test points vs PFN test MSE (both on standardised scale)."""
    rows = []
    for sp in splits_meta:
        base = f"{sp['dataset']}__n{sp['regime']}__r{sp['repeat']}"
        e, p = cache_dir / f"{base}__experts.npz", cache_dir / f"{base}__pfn_{variant_key}.npz"
        if not (e.exists() and p.exists()):
            continue
        E, P = np.load(e), np.load(p)
        s = E["ytr"].std() + 1e-12
        fbar = E["T"].mean(0)
        rows.append({"dataset": sp["dataset"], "regime": sp["regime"], "repeat": sp["repeat"],
                     "gap": float(np.mean(((P["T"] - fbar) / s) ** 2)),
                     "pfn_mse": float(np.mean(((P["T"] - E["yte"]) / s) ** 2)),
                     "committee_mse": float(np.mean(((fbar - E["yte"]) / s) ** 2))})
    return pd.DataFrame(rows)


def run(real_records, syn_records, ablation_names, regimes=(128, 512, 2048), real_dis=None) -> dict:
    STATS.mkdir(parents=True, exist_ok=True)
    df = A.real_frame(real_records)
    mean_mae = df[df.method == "mean"].groupby(["dataset", "regime", "repeat"])["mae"].first()
    df["mae_rel"] = df["mae"] / df.set_index(["dataset", "regime", "repeat"]).index.map(mean_mae).to_numpy()
    sdf = A.synthetic_frame(syn_records) if syn_records else pd.DataFrame()
    res = {"real": real_stats(df, regimes)}
    if len(sdf):
        res["synthetic"] = synthetic_stats(sdf)
        names = ["P0", "A", "B", "C"] + [n for n in ablation_names if n not in ("P0", "A", "B", "C", "D")]
        res["ablation"] = ablation_stats(sdf, df, names)
        res["synthetic"]["RQ3_real"] = rq3_real(real_dis)
    write_json(STATS / "statistics.json", res)
    return res
