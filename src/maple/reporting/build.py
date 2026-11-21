"""Build all publication tables (1-12) and data figures (3-15, S1-S3) from raw records. No hand-entered numbers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from maple.reporting import analysis as A
from maple.reporting import figures as F
from maple.reporting import stats_pipeline as SP
from maple.reporting import style as S
from maple.reporting.tables import write_table
from maple.utils import PROJECT_ROOT, read_json, write_json

EXP = PROJECT_ROOT / "experiments"
TAB = PROJECT_ROOT / "results" / "tables"
FIG = PROJECT_ROOT / "results" / "figures"
TABLE_METHODS = ["Ridge", "RF", "ExtraTrees", "XGBoost-T", "LightGBM-T", "CatBoost-T", "Stack(experts)", "MAPLE-Net/SCM",
                 "MAPLE-Net/Uniform", "MAPLE-Net", "MAPLE"]
ABBREV = ("Abbreviations: RF, random forest; -T, tuned with a 10-trial random search (3-fold CV on the training "
          "split); Stack(experts), non-negative stacking of Ridge, kNN, ExtraTrees, HistGB, RBF kernel ridge; "
          "MAPLE-Net/SCM, PFN pretrained on the conventional SCM/MLP prior (P0); MAPLE-Net/Uniform, uniform mathematical prior; "
          "MAPLE-Net, disagreement-adaptive prior with floor; MAPLE, MAPLE-Net with local competence routing.")


def load_all():
    real = A.read_jsonl(EXP / "raw" / "real.jsonl")
    syn = A.read_jsonl(EXP / "raw" / "synthetic.jsonl")
    df = A.real_frame(real) if real else pd.DataFrame()
    if len(df):
        mean_mae = df[df.method == "mean"].groupby(["dataset", "regime", "repeat"])["mae"].first()
        df["mae_rel"] = df["mae"] / df.set_index(["dataset", "regime", "repeat"]).index.map(mean_mae).to_numpy()
    sdf = A.synthetic_frame(syn) if syn else pd.DataFrame()
    return real, syn, df, sdf


def _logs(names):
    tl, sl = {}, {}
    for n in names:
        d = EXP / "checkpoints" / n
        tl[n] = A.read_jsonl(d / "train_log.jsonl")
        sl[n] = A.read_jsonl(d / "sampler_log.jsonl")
    return tl, sl


def main_ckpt_names(seeds=(0, 1, 2)):
    return [f"{v}_s{s}" for v in A.MAIN_VARIANTS for s in seeds if (EXP / "checkpoints" / f"{v}_s{s}" / "final.pt").exists()]


# ============================================================================================ tables

def table1_datasets():
    meta = pd.DataFrame(json.loads((EXP / "data" / "metadata.json").read_text()))
    t = meta.set_index("name")[["n", "n_numerical", "n_categorical", "missing_frac", "target_mean", "target_std",
                                "target_skew", "size_class", "licence"]].copy()
    t["missing_frac"] = (t["missing_frac"] * 100).round(2)
    t["regimes"] = meta.set_index("name")["regimes"].apply(lambda r: ", ".join(map(str, r)))
    t.columns = ["n", "numerical", "categorical", "missing (%)", "target mean", "target SD", "target skew",
                 "size class", "licence", "regimes (n_train)"]
    write_table(t, TAB / "table01_datasets", "Benchmark dataset characteristics (OpenML-CTR23 suite and Diabetes).",
                "Regimes list the training-set "
                "sizes evaluated (a regime is used only if at least 200 test rows remain). Excluded: Boston Housing "
                "(ethical concerns), Yacht Hydrodynamics (not in suite; too small).", lower_is_better=None,
                fmt="{:.3g}", index_name="dataset")
    return t


def _external_versions() -> str:
    from importlib.metadata import PackageNotFoundError, version
    out = []
    for pkg, desc in (("tabpfn", "TabPFN v2 weights"), ("tabicl", "TabICL v2 regressor")):
        try:
            out.append(f"{desc} ({pkg} {version(pkg)})")
        except PackageNotFoundError:
            out.append(f"{desc} (not installed)")
    return "; ".join(out) + "; library defaults, GPU; not in the pre-registered comparison set"


def table2_config(*, include_external=True, out_dir=None):
    from maple.config import load_config
    from maple.model.train import default_train_config
    exp = load_config(PROJECT_ROOT / "configs" / "experiments.yaml")
    bm = load_config(PROJECT_ROOT / "configs" / "benchmarks.yaml")
    tc = default_train_config()
    ck = EXP / "checkpoints" / "D_s0" / "final.pt"
    n_params = None
    if ck.exists():
        import torch
        n_params = torch.load(ck, map_location="cpu", weights_only=False)["n_params"]
    rows = [
        ("PFN architecture", f"{tc['model']['n_layers']} pre-LN layers, width {tc['model']['d_model']}, {tc['model']['n_heads']} heads, FF {tc['model']['d_ff']}, d_max {tc['model']['d_max']}"),
        ("PFN parameters", f"{n_params:,}" if n_params else "see checkpoint"),
        ("Pretraining steps", f"{exp['train_defaults']['steps']:,} (AdamW, lr {tc['lr']}, wd {tc['weight_decay']}, warm-up {tc['warmup']}, cosine)"),
        ("Batch", f"token budget {tc['token_budget']:,}; context n ~ log-U{tc['n_ctx_range']}; d ~ log-U{tc['d_range']}; {tc['n_qry']} queries"),
        ("Prior cells", "18 training families × 3 complexity levels = 54"),
        ("Sampler", f"τ = {tc['sampler']['tau']}, γ = {tc['sampler']['gamma']} (MAPLE-Net), EMA λ = {tc['sampler']['ema']}, clip B = {tc['sampler']['clip_B']}"),
        ("Adaptation", f"every U = {exp['train_defaults']['sampler']['update_every']} steps, m = {exp['train_defaults']['sampler']['probes_per_cell']} probe tasks/cell, probe n = {tc['sampler']['probe_n_ctx']}"),
        ("Committee", ", ".join(tc["sampler"]["committee"])),
        ("Pretraining seeds", ", ".join(map(str, exp["seeds"])) + " (ablations: seed 0)"),
        ("Real-data regimes", f"n_train ∈ {bm['regimes']}; repeats {bm['repeats']}; test ≤ 2000 rows"),
        ("Routing", "5-fold OOF on training split; β, κ selected on 150 synthetic tasks from the learned prior"),
        ("HPO budget (-T)", f"{bm['n_trials']} random configurations, 3-fold CV on the training split"),
        ("CatBoost categoricals", "one-hot for <= 255 levels (one_hot_max_size = 255), default and -T"),
        ("Synthetic evaluation", f"{bm['synthetic']['n_tasks']} tasks per suite × {len(bm['synthetic']['suites'])} suites"),
        ("Statistics", "unit = dataset (real) / task (synthetic); Friedman + Nemenyi; Wilcoxon signed-rank with Holm; "
                       "rank-biserial r; bootstrap 95% CI"),
    ]
    if include_external:
        rows.insert(-2, ("External references", _external_versions()))
    t = pd.DataFrame(rows, columns=["setting", "value"]).set_index("setting")
    write_table(t, (out_dir or TAB) / "table02_configuration", "Main experimental configuration.",
                lower_is_better=None, index_name="setting")
    return t


def tables_3_4_5(df, regime=512, *, compact=False, out_dir=None):
    candidates = ["Ridge", "RF", "ExtraTrees", "CatBoost", "MAPLE"] if compact else TABLE_METHODS
    labels = [l for l in candidates if l in set(df.label)]
    out = {}
    for num, metric, name, lib in ((3, "rmse", "RMSE", True), (4, "mae", "MAE", True), (5, "r2", "R²", False)):
        t = A.per_dataset(df, metric, regime, complete=False, labels=labels)
        sd = A.repeat_sd(df, metric, regime, labels)
        n_rep = int(df[df.regime == regime].groupby(["dataset", "method"]).size().median()) if len(df) else 0
        pfns = " (MAPLE also averaged over 3 pretraining seeds)" if compact else (
            " (PFN rows: also over 3 pretraining seeds)" if 'MAPLE-Net' in labels else "")
        units = f"{name} is in the original target units. " if lib else ""
        abbrev = "RF = random forest. MAPLE uses local competence routing." if compact else ABBREV
        write_table(t, (out_dir or TAB) / f"table{num:02d}_{metric}_n{regime}",
                    f"{name} on the real-data benchmark (n_train = {regime}); mean ± SD over {n_rep} split repeats"
                    f"{pfns}.",
                    f"{'Lower' if lib else 'Higher'} is better; best in bold, second best underlined. {units}{abbrev}",
                    sd=sd, lower_is_better=lib, fmt="{:.3g}" if metric != "r2" else "{:.3f}",
                    index_name="dataset")
        out[metric] = t
    return out


def table6_efficiency(df, regime=512, *, compact=False, out_dir=None):
    labels = [l for l in SP.FULL_SET if l in set(df.label)]
    d = df[(df.regime == regime) & df.label.isin(labels)]
    g = d.groupby("label")
    t = pd.DataFrame({"fit time (s)": g.fit_s.mean(), "predict time (s)": g.predict_s.mean(),
                      "peak RAM increase (MB)": g.peak_mem_mb.mean(), "GPU peak (MB)": g.cuda_peak_mb.mean()})
    tl, _ = _logs(main_ckpt_names())
    pre = {}
    for v in A.MAIN_VARIANTS:
        secs = [sum(r["sec"] for r in pd.DataFrame(lg).drop_duplicates("step", keep="last").to_dict("records"))
                for k, lg in tl.items() if k.startswith(f"{v}_s") and lg]
        if secs:
            pre[A.VARIANT_LABEL[v]] = np.mean(secs) / 3600
    if compact and 'MAPLE-Net' in pre:
        pre['MAPLE'] = pre['MAPLE-Net']
    t["pretraining (GPU-h, one-off)"] = pd.Series(pre)
    ck = EXP / "checkpoints" / "D_s0" / "final.pt"
    if ck.exists():
        import torch
        npar = torch.load(ck, map_location="cpu", weights_only=False)["n_params"]
        for l in t.index:
            if A.is_ours(l):
                t.loc[l, "parameters"] = npar
    if "Stack(MAPLE-Net)" in t.index and "MAPLE" in t.index:
        # Stack(MAPLE-Net) reuses the cached MAPLE-Net inference of MAPLE-Global and MAPLE (its derived records carry no GPU field)
        t.loc["Stack(MAPLE-Net)", "GPU peak (MB)"] = t.loc["MAPLE", "GPU peak (MB)"]
    t = t.loc[[l for l in labels if l in t.index]]
    fit_note = ("MAPLE's fit time includes out-of-fold expert fits. PFN inference ran on a shared GPU. " if compact else
                "Fit time includes HPO for -T methods and out-of-fold fits for routed/stacked methods. "
                "PFN, TabPFN v2 and TabICL inference ran on a shared GPU. ")
    abbrev = "RF = random forest. MAPLE uses local competence routing." if compact else ABBREV
    write_table(t, (out_dir or TAB) / f"table06_efficiency_n{regime}",
                f"Computational cost per split (n_train = {regime}; mean over datasets and repeats).",
                "Lower is better. " + fit_note + "Peak RAM increase is sampled every 5 ms during fit + predict. "
                "Pretraining time is wall-clock on a shared GPU with 3 data workers and is a one-off cost. " + abbrev,
                lower_is_better=True, axis=0, fmt="{:.3g}", index_name="method",
                mark_cols=["fit time (s)", "predict time (s)", "peak RAM increase (MB)"])
    return t


def _relative_diff_summary(rec: dict) -> tuple:
    """(Δ, CI low, CI high, raw p) with Δ = median((row - D)/D); positive = row worse than MAPLE-Net."""
    if not rec:
        return (np.nan,) * 4
    lo, hi = rec.get("delta_ci", [np.nan, np.nan])
    return (rec.get("delta_vs_ref", np.nan), lo, hi, rec.get("p", np.nan))


def table7_ablation(stats, sdf, df, names):
    rows = {}
    ab = stats.get("ablation", {})
    pfn = sdf[sdf.family == "pfn"]
    s0 = pfn[pfn.seed == 0]
    real0 = df[(df.regime == 512) & (df.family == "pfn") & (df.seed == 0)]
    for v in ["D"] + list(A.MAIN_VARIANTS[:-1]) + names:
        s = s0[s0.variant == v]
        if not len(s):
            continue
        r = real0[real0.variant == v]
        rec = ab.get(v, {})
        sm, slo, shi, sp = _relative_diff_summary(rec.get("synthetic", {}))
        rm, _, _, rp = _relative_diff_summary(rec.get("real512", {}))
        rows[v] = {"ID NRMSE": s[s.suite == "id"].nrmse.mean(), "OOD NRMSE": s[s.suite != "id"].nrmse.mean(),
                   "real NRMSE (n=512)": r.groupby("dataset").nrmse.mean().mean() if len(r) else np.nan,
                   "Δ synthetic vs MAPLE-Net (%)": sm * 100 if v != "D" else 0.0, "p (synthetic)": sp,
                   "Δ real vs MAPLE-Net (%)": rm * 100 if v != "D" else 0.0, "p (real)": rp}
    t = pd.DataFrame(rows).T
    t.index = [A.VARIANT_LABEL.get(i, i) for i in t.index]  # main variants by name, ablations by identifier
    from statsmodels.stats.multitest import multipletests
    for col in ("p (synthetic)", "p (real)"):
        m = t[col].notna()
        if m.any():
            t.loc[m, col] = multipletests(t.loc[m, col].astype(float), method="holm")[1]
    write_table(t, TAB / "table07_ablation",
                "Ablation study (pretraining seed 0; each row changes one component of MAPLE-Net).",
                "NRMSE: lower is better. Δ = median over units of (row − MAPLE-Net)/MAPLE-Net NRMSE, over synthetic "
                "tasks (all 8 suites) or real datasets (n_train = 512); negative = better than MAPLE-Net. p: two-sided "
                "Wilcoxon signed-rank, Holm-corrected across rows. Rows: MAPLE-Net/SCM (no mathematical prior), "
                "MAPLE-Net/Uniform (no adaptive sampling), MAPLE-Net/Loss (loss instead of disagreement), "
                "MAPLE-Net/Gap (no diversity floor), committee_k (k experts), "
                "score_* (disagreement definition), update_U (adaptation period), complexity_c (max complexity), "
                "budget_x (fraction of training steps).", lower_is_better=True, axis=0, fmt="{:.4g}", index_name="model")
    return t


SENS_GROUPS = {"τ (temperature)": {"0.05": "tau_0.05", "0.2": "D", "1.0": "tau_1.0"},
               "γ (floor)": {"0": "C", "0.1": "D", "0.3": "gamma_0.3"},
               "m (probes/cell)": {"2": "probes_2", "4": "D", "8": "probes_8"},
               "λ (EMA)": {"0": "ema_0.0", "0.5": "D", "0.9": "ema_0.9"},
               "U (update period)": {"250": "update_250", "1000": "D", "4000": "update_4000"},
               "committee size": {"2": "committee_2", "3": "committee_3", "5": "D"},
               "training budget": {"0.25": "budget_0.25_D", "0.5": "budget_0.5_D", "1.0": "D"},
               "max complexity": {"0": "complexity_0", "1": "complexity_1", "2": "D"}}


def table8_sensitivity(sdf):
    s0 = sdf[(sdf.family == "pfn") & (sdf.seed == 0)]
    rows, sens = [], {}
    for g, mp in SENS_GROUPS.items():
        dd = {}
        for val, v in mp.items():
            s = s0[s0.variant == v]
            if len(s):
                dd[val] = {"id": s[s.suite == "id"].nrmse.mean(), "ood": s[s.suite != "id"].nrmse.mean()}
                rows.append({"parameter": g, "value": val, "run": v, "ID NRMSE": dd[val]["id"], "OOD NRMSE": dd[val]["ood"]})
        if len(dd) >= 2:
            sens[g] = pd.DataFrame(dd).T
    t = pd.DataFrame(rows)
    if len(t):
        t.index = t["parameter"] + " = " + t["value"]
        t = t[["run", "ID NRMSE", "OOD NRMSE"]]
        write_table(t, TAB / "table08_sensitivity", "Sensitivity of MAPLE-Net to sampler and training hyperparameters "
                    "(pretraining seed 0; synthetic suites).", "Lower is better. Default values: τ = 0.2, γ = 0.1, m = 4, "
                    "λ = 0.5, U = 1000, 5 experts, full budget, max complexity 2 (row 'D'). OOD = mean over 7 OOD suites.",
                    lower_is_better=True, axis=0, fmt="{:.4f}", index_name="setting")
    return t, sens


def table9_generalization(sdf, labels, *, out_dir=None):
    pt = A.per_task(sdf, "nrmse", labels=labels)
    means = pt.groupby(level="suite").mean()
    order = ["id"] + [s for s in SP.OOD_SUITES if s in means.index]
    means = means.loc[[o for o in order if o in means.index]]
    ood = means.drop(index="id", errors="ignore").mean()
    means.loc["OOD mean"] = ood
    if "id" in means.index:
        means.loc["gap (OOD − ID)"] = ood - means.loc["id"]
    write_table(means[[l for l in labels if l in means.columns]], (out_dir or TAB) / "table09_generalization",
                "In-distribution versus out-of-distribution generalisation on synthetic task suites (mean NRMSE).",
                f"Lower is better; {int(pt.groupby(level='suite').size().median())} tasks per suite; PFN rows averaged "
                "over pretraining seeds. Suites: ID = training families, fresh draws; held-out families (Gompertz, SIR, "
                "hyperbolic); nested compositions; parameter ranges ×2.5; 3-way interactions; Voronoi regions; "
                "Student-t/Laplace noise; soft regime mixtures. All suites apply the same validity filter as training "
                "(context-standardised targets within ±12 SD), which removes the most extreme extrapolation and "
                "heavy-tail tasks.", lower_is_better=True, axis=1, fmt="{:.3f}",
                index_name="suite")
    return means


def table10_statistics(stats, *, labels=None, out_dir=None):
    rows = []
    for reg, r in stats["real"].items():
        for w in r.get("wilcoxon_vs_ref", []):
            if labels is not None and w['method'] not in labels:
                continue
            rows.append({"regime": f"n={reg}", "comparison": f"MAPLE vs {w['method']}", "N": w["N"], "W": w["W"],
                         "p": w["p"], "p (Holm, all comparisons in regime)": w["p_holm"], "r_rb": w["r_rb"],
                         "median rel. diff (%)": 100 * w["median_diff"],
                         "95% CI (%)": f"[{100 * w['ci_low']:.1f}, {100 * w['ci_high']:.1f}]",
                         "W/L/T": f"{w['wins']}/{w['losses']}/{w['ties']}"})
    t = pd.DataFrame(rows)
    if len(t):
        t.index = t["regime"] + " " + t["comparison"]
        t = t.drop(columns=["regime", "comparison"])
        notes = []
        for reg, r in stats["real"].items():
            if "friedman" in r:
                f = r["friedman"]
                notes.append(f"n={reg}: Friedman χ²({f['df']}) = {f['chi2']:.2f}, p = {f['p']:.2g}, N = {f['N']}, "
                             f"Nemenyi CD = {r['nemenyi_cd']:.2f}")
        write_table(t, (out_dir or TAB) / "table10_statistics", "Pairwise statistical comparisons against MAPLE on real data "
                    "(dataset = unit; NRMSE averaged over repeats and seeds).",
                    "Two-sided Wilcoxon signed-rank tests, Holm-corrected over the full main comparison set for a regime (a "
                    "different family from the pre-registered families used in the text, so the adjusted p-values differ). r_rb: matched-pairs "
                    "rank-biserial correlation (positive = MAPLE better). Median relative difference = median over "
                    "datasets of (other − MAPLE)/MAPLE with a 95% bootstrap CI; positive = MAPLE better. "
                    "W/L/T = datasets where MAPLE wins/loses/ties. " + "; ".join(notes) + ".",
                    lower_is_better=None, fmt="{:.3g}", index_name="comparison")
    return t


def table11_summary(df, regimes=(128, 512, 2048), *, compact=False, out_dir=None):
    rows = []
    labels = [l for l in SP.FULL_SET if l in set(df.label)]
    routes = {}
    route_row = "MAPLE (best route)"
    if compact:
        labels = [l for l in labels if not A.is_ours(l) or l in A.ROUTED_LABELS]
    for reg in regimes:
        tab = A.per_dataset(df, "nrmse", reg, complete=True, labels=labels)
        if not len(tab):
            continue
        if compact:
            candidates = [l for l in A.ROUTED_LABELS if l in tab]
            if candidates:
                winner = tab[candidates].mean().idxmin()
                routes[reg] = winner
                tab = tab.drop(columns=[l for l in candidates if l != winner]).rename(columns={winner: route_row})
        ranks = tab.rank(axis=1).mean()
        best = tab.eq(tab.min(axis=1), axis=0).sum() if compact else tab.idxmin(axis=1).value_counts()
        for l in tab.columns:
            rows.append({"regime": reg, "method": l, "mean NRMSE": tab[l].mean(), "median NRMSE": tab[l].median(),
                         "mean rank": ranks[l], "wins": int(best.get(l, 0)), "N datasets": tab.shape[0]})
    t = pd.DataFrame(rows)
    if len(t):
        if routes:
            labels = [l for l in labels if l not in A.ROUTED_LABELS] + [route_row]
        piv = t.pivot(index="method", columns="regime", values="mean rank").loc[[l for l in labels if l in set(t.method)]]
        piv.columns = [f"mean rank (n={c})" for c in piv.columns]
        piv2 = t.pivot(index="method", columns="regime", values="mean NRMSE").loc[piv.index]
        piv2.columns = [f"mean NRMSE (n={c})" for c in piv2.columns]
        wins = t.pivot(index="method", columns="regime", values="wins").loc[piv.index]
        wins.columns = [f"wins (n={c})" for c in wins.columns]
        out = pd.concat([piv2, piv, wins], axis=1)
        Ns = t.groupby("regime")["N datasets"].first().to_dict()
        note = ABBREV
        display = out
        if routes:
            selected = ", ".join(f"{n}: {model}" for n, model in routes.items())
            note = (f"The MAPLE row uses the route with the lowest mean NRMSE at each size ({selected}). "
                    "Ranks and wins use the displayed methods. Tied methods each receive a win. RF = random forest.")
            display = out.copy()
            display[wins.columns] = wins.map(lambda n: f"{n:.0f}" if pd.notna(n) else "–")
        write_table(display, (out_dir or TAB) / "table11_summary", "Summary of performance across datasets for each training-set size.",
                    "Lower is better for NRMSE and rank. Wins count datasets on which the method has the lowest NRMSE. "
                    f"The dataset counts are N = {Ns}. " + note, lower_is_better=True, axis=0,
                    fmt="{:.3f}", index_name="method", mark_cols=list(piv2.columns) + list(piv.columns) if compact else None)
        return out
    return t


def table12_generator(sdf):
    from maple.priors.families import FAMILIES
    from maple.tasks.generator import TaskGenerator, cells, default_generator_config
    from maple.utils import rng as mk_rng
    _, sl = _logs(main_ckpt_names())
    finals = [F._family_probs(lg[-1]) for k, lg in sl.items() if k.startswith("D_s") and lg]
    qD = pd.concat(finals, axis=1).mean(1) if finals else pd.Series(dtype=float)
    gen = TaskGenerator(default_generator_config())
    att = {}
    for f, fam in FAMILIES.items():
        a = []
        for i in range(30):
            r = mk_rng(0, "table12", f, i)
            a.append(gen.sample(r, (f, i % 3), 128, 64, 8).meta["attempts"])
        att[f] = np.mean(a)
    rows = []
    for f, fam in FAMILIES.items():
        rows.append({"family": f, "split": fam.split, "description": fam.description, "cells": 3,
                     "mean attempts": att[f], "final q (MAPLE-Net)": qD.get(f, np.nan),
                     "uniform q": (1 / 18 if fam.split == "train" else np.nan)})
    t = pd.DataFrame(rows).set_index("family")
    write_table(t, TAB / "table12_generator", "Mathematical task generator: function families, splits and sampling.",
                "Split: train = available to pretraining; ood = held-out family; ood_structure = structure used only by "
                "OOD suites. Mean attempts = generator draws per valid task (validity checks). Final q = probability "
                "mass of the family in MAPLE-Net's final sampling distribution (mean over 3 seeds), versus 1/18 uniform.",
                lower_is_better=None, fmt="{:.3g}", index_name="family")
    return t


# ============================================================================================ figures

def real_disagreement() -> pd.DataFrame:
    """Per-split MAPLE-Net (seed 0) / committee gap and errors on real test data, from the caches."""
    from maple.config import load_config
    from maple.data import datasets as D
    from maple.evaluation.runner import build_splits
    p = PROJECT_ROOT / "results" / "statistics" / "disagreement_real.csv"
    bm = load_config(PROJECT_ROOT / "configs" / "benchmarks.yaml")
    try:
        sps = build_splits(D.benchmark_names(), bm["regimes"], {int(k): v for k, v in bm["repeats"].items()}, D.load,
                           EXP / "splits", bm["seed"])
    except FileNotFoundError:
        return pd.DataFrame()
    rd = SP.disagreement_real(EXP / "processed" / "cache", sps, "D_s0")
    if len(rd):
        p.parent.mkdir(parents=True, exist_ok=True)
        rd.to_csv(p, index=False)
    return rd


def cost_figure(df, out):
    d = df[(df.regime == 512) & df.label.isin(SP.FULL_SET)]
    g = d.groupby("label")
    per = A.per_dataset(df, "nrmse", 512, complete=True, labels=[l for l in SP.FULL_SET if l in set(d.label)])
    eff = pd.DataFrame({"time_s": g.fit_s.mean() + g.predict_s.mean(), "nrmse": per.mean()}).dropna()
    eff["group"] = ["MAPLE (routed)" if A.is_routed(l) else "MAPLE-Net (standalone)" if A.is_net(l) else
                    "combination" if l.startswith(("Stack(", "Avg(")) else
                    "pretrained FM (external)" if l in A.EXTERNAL_LABELS else
                    "tuned GBDT" if l.endswith("-T") else "classical"
                    for l in eff.index]
    F.fig08_cost(eff.drop(index="Mean", errors="ignore"), out)


def build_figures(stats, df, sdf):
    S.apply()
    from maple.reporting import diagrams as G
    G.figure1_architecture(FIG / "fig01_framework")
    G.figure2_generator(FIG / "fig02_task_generator")
    names = main_ckpt_names()
    tl, sl = _logs(names)
    _fig03(sl)
    F.fig04_adaptation(sl, FIG / "fig04_adaptation")
    F.fig05_family_distribution(sl, FIG / "fig05_family_distribution")
    if len(df):
        F.fig_dataset_dots(A.per_dataset(df, "nrmse", 512, complete=False), FIG / "fig06_nrmse_datasets",
                           "NRMSE = RMSE / SD(y_test), n_train = 512 (lower is better)")
        F.fig_dataset_dots(A.per_dataset(df, "mae_rel", 512, complete=False), FIG / "fig07_mae_datasets",
                           "relative MAE = MAE / MAE(mean predictor), n_train = 512 (lower is better)")
        cost_figure(df, FIG / "fig08_cost_accuracy")
        tabs = {}
        for reg in (128, 512, 2048):
            t = A.per_dataset(df, "nrmse", reg, complete=True, labels=[l for l in SP.MAIN_SET if l in set(df.label)])
            if t.shape[0] >= 5:
                tabs[f"n_train = {reg}"] = t
        if tabs:
            F.fig12_cd(tabs, FIG / "fig12_critical_difference")
    if len(sdf):
        abl_names = [n for n in sdf.variant.dropna().unique() if n not in A.MAIN_VARIANTS]
        ab = stats.get("ablation", {})
        rows = {}
        for n in ["P0", "A", "B", "C"] + sorted(abl_names):
            rec = ab.get(n, {})
            if not rec:
                continue
            sm, slo, shi, _ = _relative_diff_summary(rec.get("synthetic", {}))
            rm, rlo, rhi, _ = _relative_diff_summary(rec.get("real512", {}))
            rows[n] = {"syn_med": sm, "syn_lo": slo, "syn_hi": shi, "real_med": rm, "real_lo": rlo, "real_hi": rhi}
        if rows:
            abl = pd.DataFrame(rows).T
            abl.index = [A.VARIANT_LABEL.get(i, i) for i in abl.index]
            F.fig09_ablation(abl, FIG / "fig09_ablation")
        _, sens = table8_sensitivity(sdf)
        if sens:
            F.fig10_sensitivity(sens, FIG / "fig10_sensitivity")
        labels = [l for l in ["MAPLE-Net/SCM", "MAPLE-Net/Uniform", "MAPLE-Net", "ExtraTrees", "CatBoost", "RF"] if l in set(sdf.label)]
        pt = A.per_task(sdf, "nrmse", labels=labels)
        means = pt.groupby(level="suite").mean()
        order = ["id"] + [s for s in SP.OOD_SUITES if s in means.index]
        from maple.stats.tests import bootstrap_ci
        cis = {l: [bootstrap_ci(pt.xs(s, level="suite")[l].to_numpy(), B=2000) for s in order] for l in labels}
        F.fig11_generalization(means.loc[order][labels], pd.DataFrame(cis, index=order), FIG / "fig11_generalization")
        F.fig13_disagreement(sdf, real_disagreement(), FIG / "fig13_disagreement_vs_error")
    F.fig14_learning(tl, sl, FIG / "fig14_learning_curves")
    sc = A.read_jsonl(EXP / "raw" / "scaling.jsonl")
    if sc:
        agg = A.scaling_summary(A.real_frame(sc))
        F.fig15_scaling(agg, FIG / "fig15_scaling")
    if len(df):
        _supp_figs(df)


def _fig03(sl):
    """Real probe task (trigonometric, 1 active feature), real committee, real MAPLE-Net checkpoint and logged sampler state."""
    ck = EXP / "checkpoints" / "D_s0" / "final.pt"
    if not ck.exists() or not sl.get("D_s0"):
        return
    from maple.experts.registry import COMMITTEE_DEFAULT, fit_predict_committee
    from maple.model.predictor import _load, pfn_predict_arrays, resolve_device
    from maple.tasks.generator import TaskGenerator, default_generator_config
    from maple.utils import rng as mk_rng
    cfg = default_generator_config()
    cfg.update({"p_missing": 0.0, "p_categorical": 0.0, "p_correlated": 0.0, "p_redundant": 0.0,
                "marginal_probs": [1, 0, 0, 0, 0]})
    gen = TaskGenerator(cfg)
    task = gen.sample(mk_rng(0, "figure3"), ("trigonometric", 1), 48, 400, 1)
    P = fit_predict_committee(COMMITTEE_DEFAULT, task.X_ctx, task.y_ctx, task.X_qry, seed=0)
    dev = resolve_device("auto")
    pred = pfn_predict_arrays(_load(str(ck.resolve()), str(dev)), task.X_ctx, task.y_ctx, task.X_qry, dev)
    feat = int(task.meta["active"][0])
    F.fig03_mechanism(task, P, pred, COMMITTEE_DEFAULT, sl["D_s0"][-1], FIG / "fig03_disagreement_sampling", feat)


def _supp_figs(df, *, compact=False, out_dir=None):
    preds_dir = EXP / "processed" / "preds"
    items = []
    from maple.data import datasets as D
    from maple.data.splits import make_splits
    from maple.config import load_config
    bmc = load_config(PROJECT_ROOT / "configs" / "benchmarks.yaml")
    bm_seed, rep512 = bmc["seed"], int(bmc["repeats"][512])
    for ds in ("concrete_compressive_strength", "airfoil_self_noise", "kin8nm", "california_housing"):
        a = preds_dir / f"{ds}__n512__F_D_s0.npy"
        comparison = "catboost" if compact else "catboost_tuned"
        label = "CatBoost" if compact else "CatBoost-T"
        b = preds_dir / f"{ds}__n512__{comparison}.npy"
        if a.exists() and b.exists():
            X, y, _ = D.load(ds)
            sp = make_splits(ds, len(y), 512, rep512, bm_seed, EXP / "splits")
            yte = y[sp[0][1]]
            items.append((ds, yte, {"MAPLE": np.load(a), label: np.load(b)}))
    if items:
        F.figS_pred_vs_truth(items, (out_dir or FIG) / "figS1_pred_vs_truth_residuals")
    ss = A.seed_sd(df, "nrmse", 512)
    rs = A.repeat_sd(df, "nrmse", 512, ["MAPLE-Net"])
    if "MAPLE-Net" in ss and "MAPLE-Net" in rs:
        F.figS_seed_stability(ss["MAPLE-Net"], rs["MAPLE-Net"], (out_dir or FIG) / "figS2_seed_stability")


def build_all(regimes=(128, 512, 2048)):
    real, syn, df, sdf = load_all()
    abl = [n for n in sdf.variant.dropna().unique() if n not in A.MAIN_VARIANTS] if len(sdf) else []
    stats = SP.run(real, syn, abl, regimes, real_dis=real_disagreement())
    table1_datasets()
    table2_config()
    if len(df):
        for reg in regimes:
            tables_3_4_5(df, reg)
        table6_efficiency(df, 512)
        table10_statistics(stats)
        table11_summary(df, regimes)
    if len(sdf):
        table7_ablation(stats, sdf, df, sorted(abl))
        table8_sensitivity(sdf)
        table9_generalization(sdf, [l for l in ["Ridge", "kNN", "RF", "ExtraTrees", "HistGB", "CatBoost", "XGBoost",
                                                "MAPLE-Net/SCM", "MAPLE-Net/Uniform", "MAPLE-Net/Loss", "MAPLE-Net/Gap", "MAPLE-Net", "TabPFN-v2", "TabICL"] if l in set(sdf.label)])
        table12_generator(sdf)
    if len(df) and len(sdf):
        table13_rq_verdicts(stats)
    table14_scaling()
    build_figures(stats, df, sdf)
    return stats


def table14_scaling():
    """Supplementary: mean NRMSE and fit+predict time by training-set size (sizes present for every dataset)."""
    sc = A.read_jsonl(EXP / "raw" / "scaling.jsonl")
    if not sc:
        return pd.DataFrame()
    agg = A.scaling_summary(A.real_frame(sc))
    t = agg.pivot(index="n_train", columns="label", values="nrmse")
    t.columns = [f"{c} NRMSE" for c in t.columns]
    t2 = agg.pivot(index="n_train", columns="label", values="total_s")
    t2.columns = [f"{c} time (s)" for c in t2.columns]
    t = pd.concat([t, t2], axis=1)
    ds = sorted(set(A.real_frame(sc).dataset))
    write_table(t, TAB / "table14_scaling", "Scaling with training-set size (supplementary).",
                f"Datasets: {', '.join(ds)}; mean over datasets and three repeats, restricted to sizes available for "
                "every dataset. MAPLE-Net: GPU inference with 1024-row context bagging above 1024 rows; other methods: "
                "one CPU thread, default hyperparameters.", lower_is_better=None, fmt="{:.3f}",
                index_name="n_train")
    return t


# ============================================================================================ RQ verdicts

def _sig(rec: dict) -> bool:
    p = rec.get("p_holm_family", rec.get("p", np.nan)) if rec else np.nan
    return bool(rec) and p < 0.05


def _better(rec: dict) -> bool:  # first-named method has significantly lower error
    return _sig(rec) and rec.get("median_diff", 0) > 0


def _worse(rec: dict) -> bool:
    return _sig(rec) and rec.get("median_diff", 0) < 0


def _ev(rec: dict, a: str, b: str) -> str:
    """'a vs b: -2.4% (p_Holm = 0.002)'; negative = a has lower error (median relative difference)."""
    if not rec:
        return f"{a} vs {b}: n/a"
    p = rec.get("p_holm_family", rec.get("p"))
    ptxt = "p_Holm < 0.001" if p < 1e-3 else f"p_Holm = {p:.3f}"
    return f"{a} vs {b}: {-100 * rec['median_diff']:+.1f}% ({ptxt})"


def table13_rq_verdicts(stats: dict, *, compact=False, out_dir=None) -> pd.DataFrame:
    """Verdicts on the pre-registered hypotheses (research_questions.md), derived from statistics.json by fixed rules:
    significant = Holm-adjusted p < 0.05 within the pre-registered family; direction = sign of the median difference."""
    real, syn = stats.get("real", {}), stats.get("synthetic", {})
    regs = sorted(real, key=int)
    rows = {}
    r1 = syn.get("RQ1_nrmse", {})
    i, o = r1.get("id", {}), r1.get("pooled_ood", {})
    v1 = "Supported" if _better(i) and _better(o) else ("Partially supported" if _better(i) or _better(o) else "Not supported")
    rows["RQ1"] = ("MAPLE-Net < MAPLE-Net/Uniform on ID and pooled-OOD synthetic suites",
                   f"ID {_ev(i, 'D', 'A')}; OOD {_ev(o, 'D', 'A')}", v1)
    r2 = syn.get("RQ2", {})
    worse_real = [g for g in regs if _worse(real[g].get("rq", {}).get("RQ2_real_C_vs_B", {}))]
    if not _worse(r2.get("all", {})) and _better(r2.get("high_noise", {})):
        v2 = "Supported on synthetic tasks"
        if abs(r2["all"].get("median_diff", 0)) < 0.01:
            v2 += " (negligible effect)"
    else:
        v2 = "Not supported on synthetic tasks"
    if worse_real:
        v2 += "; reversed on real data (n = " + ", ".join(worse_real) + ")"
    rows["RQ2"] = ("MAPLE-Net/Gap ≤ MAPLE-Net/Loss overall; MAPLE-Net/Gap < MAPLE-Net/Loss on high-noise tasks",
                   f"all {_ev(r2.get('all', {}), 'C', 'B')}; high noise {_ev(r2.get('high_noise', {}), 'C', 'B')}", v2)
    r3, r3r = syn.get("RQ3", {}).get("MAPLE-Net", {}), syn.get("RQ3_real", {})
    # H3 concerns EXCESS error. On real data f* is unknown; the regret against the committee (Prop. 4(d)) is the
    # label-computable excess proxy and decides; raw test MSE (noise-confounded across datasets) is reported alongside
    rk, ck = ("spearman_regret", "ci_regret") if "ci_regret" in r3r else ("spearman_datasets", "ci")
    syn_ok = bool(r3) and r3["ci"][0] > 0
    real_ok = bool(r3r) and r3r[ck][0] > 0
    weak = (not syn_ok) or r3["spearman"] < 0.5 or (real_ok and r3r[rk] < 0.5)
    tag = " (weak to moderate)" if weak else ""
    v3 = ("Supported" + tag if syn_ok and real_ok else
          "Supported on synthetic data only" + tag if syn_ok else
          "Supported on real data only" + tag if real_ok else "Not supported")
    ev3 = "n/a"
    if r3 and r3r:
        ev3 = (f"synthetic ρ = {r3['spearman']:.2f} [{r3['ci'][0]:.2f}, {r3['ci'][1]:.2f}] (loss: "
               f"{r3['spearman_loss_vs_excess']:.2f}); real, regret proxy ρ = {r3r[rk]:.2f} "
               f"[{r3r[ck][0]:.2f}, {r3r[ck][1]:.2f}]")
        if rk != "spearman_datasets":
            ev3 += f"; real, raw MSE ρ = {r3r['spearman_datasets']:.2f} [{r3r['ci'][0]:.2f}, {r3r['ci'][1]:.2f}]"
    rows["RQ3"] = ("Spearman ρ(gap, excess error) > 0 on synthetic and real data", ev3, v3)
    better4 = [g for g in regs if _better(real[g]["rq"].get("RQ4_A_vs_P0", {})) or _better(real[g]["rq"].get("RQ4_D_vs_P0", {}))]
    s4 = syn.get("RQ4_synthetic_nrmse", {})
    rows["RQ4"] = ("MAPLE-Net/Uniform or MAPLE-Net < MAPLE-Net/SCM on real data",
                   "; ".join(f"n = {g}: {_ev(real[g]['rq'].get('RQ4_D_vs_P0', {}), 'D', 'SCM')}" for g in regs)
                   + (f"; synthetic OOD {_ev(s4.get('ood_A_vs_P0', {}), 'A', 'SCM')}" if s4 else ""),
                   "Supported on real data" if better4 else "Not supported on real data")
    f_d = all(_better(real[g]["rq"].get("RQ5_F_vs_MAPLE-Net", {})) for g in regs)
    f_stack = all(not _worse(real[g]["rq"].get(k, {})) for g in regs
                  for k in ("RQ5_F_vs_Stack(experts)", "RQ5_F_vs_Stack(MAPLE-Net)"))
    rows["RQ5"] = ("MAPLE < MAPLE-Net and MAPLE ≤ stacking",
                   "; ".join(f"n = {g}: {_ev(real[g]['rq'].get('RQ5_F_vs_MAPLE-Net', {}), 'F', 'D')}, "
                             f"{_ev(real[g]['rq'].get('RQ5_F_vs_Stack(experts)', {}), 'F', 'Stack(experts)')}, "
                             f"{_ev(real[g]['rq'].get('RQ5_F_vs_Stack(MAPLE-Net)', {}), 'F', 'Stack(MAPLE-Net)')}"
                             for g in regs),
                   "Supported" if f_d and f_stack else ("Partially supported" if f_d else "Not supported"))
    rows["RQ6"] = ("Exploratory: MAPLE vs tuned GBDTs",
                   "; ".join(f"n = {g}: {_ev(real[g]['rq'].get('RQ6_F_vs_CatBoost-T', {}), 'F', 'CatBoost-T')}"
                             for g in regs), "Exploratory")
    rows["RQ7"] = ("Exploratory: which components matter", "one-factor ablations and sensitivity runs (Section 6.4)", "Exploratory")
    if compact:
        rows["RQ5"] = (rows["RQ5"][0], "routing and stacking comparisons (Section 6.1)", rows["RQ5"][2])
        rows["RQ6"] = (rows["RQ6"][0], "gradient-boosting comparisons (Section 6.1)", rows["RQ6"][2])
    t = pd.DataFrame(rows, index=["hypothesis (pre-registered)", "key evidence", "verdict"]).T
    write_table(t, (out_dir or TAB) / "table13_rq_verdicts", "Pre-registered research questions: hypotheses, evidence and verdicts.",
                "Percentages are median relative NRMSE differences (negative = first method has lower error). "
                "p_Holm: two-sided Wilcoxon signed-rank, Holm-corrected within the pre-registered family. Unit: task "
                "(synthetic) or dataset (real). Significant = p_Holm < 0.05; |difference| < 1% is practically "
                "negligible. Verdicts follow fixed rules applied to these statistics.",
                lower_is_better=None, index_name="RQ",
                wrap={"hypothesis (pre-registered)": r">{\raggedright\arraybackslash}X",
                      "key evidence": r">{\raggedright\arraybackslash}X",
                      "verdict": r">{\raggedright\arraybackslash}p{3.3cm}"})
    return t
