"""Data-driven publication figures (Figures 3-15 and supplementary). Every figure reads raw records/caches/logs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from maple.reporting import analysis as A
from maple.reporting import style as S
from maple.stats import tests as T

PFN_COLORS = {"PFN-SCM": S.SERIES[4], "PFN-A": S.SERIES[1], "PFN-B": S.SERIES[3], "PFN-C": S.SERIES[2],
              "PFN-D": S.SERIES[0], "MAPLE-E": S.SERIES[6], "MAPLE-F": S.SERIES[0]}
FOCUS = ["MAPLE-F", "PFN-D", "PFN-SCM", "CatBoost-T", "Stack(experts)"]
FOCUS_COLORS = dict(zip(FOCUS, [S.SERIES[0], S.SERIES[6], S.SERIES[4], S.SERIES[5], S.SERIES[1]]))
FOCUS_MARKERS = dict(zip(FOCUS, ["o", "s", "v", "D", "^"]))


_ABBREV = {"concrete_compressive_strength": "concrete compr. strength",
           "geographical_origin_of_music": "geogr. origin of music"}


def _short(ds: str) -> str:
    return _ABBREV.get(ds, ds.replace("_", " "))


def place_labels(ax, xs, ys, labels, fontsize=5.5, color=None, pad_px=2.0):
    """Annotate points without label-label overlaps: greedy search over candidate offsets (display space)."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    cands = [(4, 2, "left", "bottom"), (4, -2, "left", "top"), (-4, 2, "right", "bottom"), (-4, -2, "right", "top")]
    for d in (8, 14, 20, 26, 32, 38, 44):
        for sx in (1, -1):
            for dx in (6, 18):
                cands += [(sx * dx, sgn * d, "left" if sx > 0 else "right", "center") for sgn in (1, -1)]
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    from matplotlib.transforms import Bbox
    rad = 4.5 * fig.dpi / 72.0  # marker radius (s~22 pt^2) plus a small halo, in display pixels
    placed = [Bbox.from_extents(x - rad, y - rad, x + rad, y + rad) for x, y in pts]  # points are obstacles

    def overlap(bb):
        tot = 0.0
        for o in placed:
            w = min(bb.x1, o.x1) - max(bb.x0, o.x0)
            h = min(bb.y1, o.y1) - max(bb.y0, o.y0)
            if w > 0 and h > 0:
                tot += w * h
        return tot

    texts = []
    order = np.argsort(np.asarray(ys))[::-1]
    for i in order:
        best = None  # (overlap, text, bbox); first zero-overlap candidate wins, else the least-overlapping one
        for dx, dy, ha, va in cands:
            t = ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                            color=color or "#52514d", ha=ha, va=va,
                            arrowprops=None if abs(dy) <= 2 else dict(arrowstyle="-", lw=0.4, color="#9c9a92",
                                                                      shrinkA=0, shrinkB=2))
            e = t.get_window_extent(r)
            bb = Bbox.from_extents(e.x0 - pad_px, e.y0 - pad_px, e.x1 + pad_px, e.y1 + pad_px)
            ov = overlap(bb)
            if best is None or ov < best[0]:
                if best is not None:
                    best[1].remove()
                best = (ov, t, bb)
            else:
                t.remove()
            if ov == 0:
                break
        placed.append(best[2])
        texts.append((i, best[1]))
    return [t for _, t in sorted(texts, key=lambda z: z[0])]


# ------------------------------------------------------------------------------------------- Fig 3

def fig03_mechanism(task, P: np.ndarray, pfn_pred: np.ndarray, committee: list[str], sampler_row: dict, out: Path,
                    feature: int) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(S.COL2, 2.3), gridspec_kw={"width_ratios": [1.25, 1, 1.1]})
    x = task.X_qry[:, feature]
    o = np.argsort(x)
    ax = axs[0]
    ax.scatter(task.X_ctx[:, feature], task.y_ctx, s=5, color=S.MUTED, zorder=1, label="context points", lw=0)
    for k, c in zip(range(P.shape[0]), S.SERIES[1:]):
        ax.plot(x[o], P[k][o], lw=0.9, color=c, label=committee[k])
    ax.plot(x[o], pfn_pred[o], lw=1.7, color=S.SERIES[0], label="PFN θ")
    ax.set_title("(a) probe task: experts and PFN")
    ax.set_xlabel(f"feature x{feature + 1} (standardised)")
    ax.set_ylabel("y (standardised)")
    ax.legend(fontsize=5.2, ncol=2, loc="lower right", frameon=True, framealpha=0.92, edgecolor=S.GRID)
    ax = axs[1]
    gap = (pfn_pred - P.mean(0)) ** 2
    amb = P.var(0)
    ax.plot(x[o], gap[o], color=S.SERIES[0], lw=1.2, label="gap Δ(x) = (θ − f̄)²")
    ax.plot(x[o], amb[o], color=S.SERIES[1], lw=1.0, ls="--", label="ambiguity A(x)")
    ax.set_title("(b) pointwise disagreement")
    ax.set_xlabel(f"feature x{feature + 1}")
    ax.legend(fontsize=5.5)
    ax = axs[2]
    cells = sampler_row["cells"]
    p = np.array(sampler_row["probs"])
    order = np.argsort(-p)
    show = list(order[:6]) + list(order[-4:])
    ax.barh(range(len(show)), p[show], color=[S.SERIES[0]] * 6 + [S.NEUTRAL] * 4, height=0.7)
    ax.axvline(1 / len(p), color=S.MUTED, ls="--", lw=0.8)
    ax.text(1 / len(p), len(show) - 0.4, " uniform", fontsize=5.5, color=S.MUTED, va="bottom")
    ax.set_yticks(range(len(show)), [cells[i].replace("_", " ").replace("|", " | c=") for i in show], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("sampling probability q_c")
    ax.set_title(f"(c) sampler after round {sampler_row['round']}")
    fig.tight_layout()
    S.save(fig, out)


# ------------------------------------------------------------------------------------------- Fig 4, 5, 14

def load_sampler_logs(ckpt_dir: Path) -> list[dict]:
    return A.read_jsonl(ckpt_dir / "sampler_log.jsonl")


def _family_probs(row: dict) -> pd.Series:
    s = pd.Series(row["probs"], index=[c.split("|")[0] for c in row["cells"]])
    return s.groupby(level=0).sum()


def fig04_adaptation(logs: dict[str, list[dict]], out: Path) -> pd.DataFrame:
    """(a) heatmap: family probability over rounds (D, seed 0); (b) KL(q||u) over rounds for B, C, D (seed means)."""
    fig, axs = plt.subplots(1, 2, figsize=(S.COL2, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    d0 = logs.get("D_s0", [])
    rows = []
    if d0:
        mat = pd.DataFrame({r["round"]: _family_probs(r) for r in d0})
        rows.append(mat)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("seq", S.SEQ_BLUE)
        im = axs[0].imshow(mat.values, aspect="auto", cmap=cmap)
        axs[0].set_yticks(range(len(mat.index)), [i.replace("_", " ") for i in mat.index], fontsize=5.5)
        axs[0].set_xticks(range(mat.shape[1]), mat.columns, fontsize=6)
        axs[0].set_xlabel("adaptation round (every 1000 steps)")
        axs[0].set_title("(a) PFN-D, seed 0: family probability")
        axs[0].grid(False)
        cb = fig.colorbar(im, ax=axs[0], fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("probability (summed over the family's cells)", fontsize=5.5)
    ax = axs[1]
    for v, c in (("B", PFN_COLORS["PFN-B"]), ("C", PFN_COLORS["PFN-C"]), ("D", PFN_COLORS["PFN-D"])):
        curves = []
        for k, lg in logs.items():
            if k.startswith(f"{v}_s") and lg:
                q = np.array([r["probs"] for r in lg])
                M = q.shape[1]
                curves.append([float(np.sum(qq * np.log(qq * M))) for qq in q])
        if curves:
            L = min(map(len, curves))
            arr = np.array([c_[:L] for c_ in curves])
            rounds = np.arange(1, L + 1)
            ax.plot(rounds, arr.mean(0), color=c, marker="o", label=f"PFN-{v}")
            if arr.shape[0] > 1:
                ax.fill_between(rounds, arr.min(0), arr.max(0), color=c, alpha=0.15, lw=0)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("adaptation round")
    ax.set_ylabel("KL(q ‖ uniform)")
    ax.set_title("(b) departure from the base prior")
    ax.legend()
    fig.tight_layout()
    S.save(fig, out)
    return rows[0] if rows else pd.DataFrame()


def fig05_family_distribution(logs: dict[str, list[dict]], out: Path) -> pd.DataFrame:
    fams = None
    res = {}
    for v in ("B", "C", "D"):
        ps = [_family_probs(lg[-1]) for k, lg in logs.items() if k.startswith(f"{v}_s") and lg]
        if ps:
            res[f"PFN-{v}"] = pd.concat(ps, axis=1).mean(1)
            fams = res[f"PFN-{v}"].index
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)
    uni = 1.0 / len(df)
    df = df.loc[df.mean(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(S.COL1 + 0.6, 3.6))
    h = 0.26
    for i, (col, c) in enumerate(zip(df.columns, [PFN_COLORS[x] for x in df.columns])):
        ax.barh(np.arange(len(df)) + (i - 1) * h, df[col], height=h, color=c, label=col, edgecolor="white", lw=0.3)
    ax.axvline(uni, color=S.MUTED, ls="--", lw=0.8)
    ax.text(uni, len(df) - 0.3, " uniform", color=S.MUTED, fontsize=6, va="bottom")
    ax.set_yticks(range(len(df)), [f.replace("_", " ") for f in df.index], fontsize=6)
    ax.set_xlabel("final sampling probability (summed over complexity levels)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    S.save(fig, out)
    return df


def fig14_learning(train_logs: dict[str, list[dict]], sampler_logs: dict[str, list[dict]], out: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(S.COL2, 2.4))
    for v in ("P0", "A", "B", "C", "D"):
        series = []
        for k, lg in train_logs.items():
            if k.startswith(f"{v}_s") and lg:
                d = pd.DataFrame(lg).drop_duplicates("step", keep="last").set_index("step")["loss_avg"]
                series.append(d)
        if series:
            m = pd.concat(series, axis=1).mean(1).rolling(10, min_periods=1).mean()
            axs[0].plot(m.index, m.values, color=PFN_COLORS[A.VARIANT_LABEL[v]], label=A.VARIANT_LABEL[v], lw=1.2)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("training step")
    axs[0].set_ylabel("training MSE (own prior, smoothed)")
    axs[0].set_title("(a) pretraining loss")
    axs[0].legend(fontsize=6)
    for v in ("A", "B", "C", "D"):
        curves = []
        for k, lg in sampler_logs.items():
            if k.startswith(f"{v}_s") and lg:
                curves.append([np.mean([dd["excess"] for dd in r["diag"].values()]) for r in lg])
        if curves:
            L = min(map(len, curves))
            arr = np.array([c[:L] for c in curves])
            axs[1].plot(np.arange(1, L + 1) * 1000, arr.mean(0), marker="o", color=PFN_COLORS[f"PFN-{v}"],
                        label=f"PFN-{v}")
    axs[1].set_xlabel("training step")
    axs[1].set_ylabel("excess MSE on uniform probe tasks")
    axs[1].set_title("(b) held-out probe error per round")
    axs[1].legend(fontsize=6)
    fig.tight_layout()
    S.save(fig, out)


# ------------------------------------------------------------------------------------------- Fig 6, 7

def fig_dataset_dots(tab: pd.DataFrame, out: Path, xlabel: str, log=False) -> None:
    cols = [c for c in FOCUS if c in tab.columns]
    t = tab[cols].dropna()
    if "MAPLE-F" in t:
        t = t.sort_values("MAPLE-F")
    fig, ax = plt.subplots(figsize=(S.COL2, 0.16 * len(t) + 0.9))
    y = np.arange(len(t))
    for c in cols:
        ax.scatter(t[c], y, s=16, marker=FOCUS_MARKERS[c], color=FOCUS_COLORS[c], label=c, zorder=3,
                   edgecolor="white", lw=0.4)
    for i in range(len(t)):
        ax.plot([t.iloc[i].min(), t.iloc[i].max()], [i, i], color=S.GRID, lw=1, zorder=1)
    ax.set_yticks(y, [_short(d) for d in t.index], fontsize=6)
    ax.set_xlabel(xlabel)
    if log:
        ax.set_xscale("log")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    S.save(fig, out)


# ------------------------------------------------------------------------------------------- Fig 8

def fig08_cost(eff: pd.DataFrame, out: Path) -> None:
    """eff: index=label, columns: time_s (fit+predict mean), nrmse (mean over datasets), group."""
    fig, ax = plt.subplots(figsize=(S.COL1 + 0.9, 2.8))
    gcol = {"classical": S.NEUTRAL, "tuned GBDT": S.SERIES[5], "PFN": S.SERIES[1], "MAPLE": S.SERIES[0],
            "combination": S.SERIES[6], "pretrained FM (external)": S.SERIES[7]}
    for g, sub in eff.groupby("group"):
        ax.scatter(sub.time_s, sub.nrmse, s=22, color=gcol.get(g, S.NEUTRAL), label=g, zorder=3, edgecolor="white", lw=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("mean fit + predict time per split (s, log scale)")
    ax.set_ylabel("mean NRMSE (lower is better)")
    ax.margins(x=0.22, y=0.08)
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()  # final geometry first, then labels (offsets are in display space)
    xs, ys, labs = merge_close_labels(ax, eff.time_s.to_numpy(), eff.nrmse.to_numpy(), list(eff.index))
    place_labels(ax, xs, ys, labs, fontsize=5.5, color=S.INK2)
    S.save(fig, out)


def merge_close_labels(ax, xs, ys, labels, tol_px: float = 6.0):
    """Points closer than tol_px (display space) share one label 'a / b' anchored at their centroid."""
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    groups: list[list[int]] = []
    for i in range(len(pts)):
        for g in groups:
            if any(np.hypot(*(pts[i] - pts[j])) < tol_px for j in g):
                g.append(i)
                break
        else:
            groups.append([i])
    gx, gy, gl = [], [], []
    for g in groups:
        c = pts[g].mean(0)
        x, y = ax.transData.inverted().transform(c)
        gx.append(x)
        gy.append(y)
        gl.append(" / ".join(sorted(labels[i] for i in g)))
    return np.array(gx), np.array(gy), gl


# ------------------------------------------------------------------------------------------- Fig 9, 10

def fig09_ablation(abl: pd.DataFrame, out: Path) -> None:
    """abl: index=ablation label; columns: syn_med, syn_lo, syn_hi, real_med, real_lo, real_hi (relative diff vs D)."""
    fig, axs = plt.subplots(1, 2, figsize=(S.COL2, 0.22 * len(abl) + 1.0), sharey=True)
    y = np.arange(len(abl))
    for ax, pre, title in ((axs[0], "syn", "synthetic tasks (ID + OOD)"), (axs[1], "real", "real data, n = 512")):
        med = abl[f"{pre}_med"] * 100
        lo, hi = abl[f"{pre}_lo"] * 100, abl[f"{pre}_hi"] * 100
        col = [S.SERIES[7] if m > 0 else S.SERIES[0] for m in med.fillna(0)]
        ax.barh(y, med, color=col, height=0.6)
        ax.errorbar(med, y, xerr=[med - lo, hi - med], fmt="none", ecolor=S.INK2, lw=0.7, capsize=1.5)
        ax.axvline(0, color=S.INK2, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("median relative NRMSE change vs PFN-D (%)")
    axs[0].set_yticks(y, list(abl.index), fontsize=6)
    axs[0].invert_yaxis()
    fig.tight_layout()
    S.save(fig, out)


def fig10_sensitivity(sens: dict[str, pd.DataFrame], out: Path) -> None:
    """sens: param -> DataFrame(index=value, columns=['id','ood'] mean NRMSE)."""
    n = len(sens)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(S.COL2, 1.8 * rows), squeeze=False)
    for ax, (p, df) in zip(axs.flat, sens.items()):
        xs = np.arange(len(df))
        ax.plot(xs, df["id"], marker="o", color=S.SERIES[0], label="ID")
        ax.plot(xs, df["ood"], marker="s", color=S.SERIES[1], label="OOD")
        ax.set_xticks(xs, [str(v) for v in df.index], fontsize=6)
        ax.set_title(p, fontsize=7)
        ax.set_ylabel("mean NRMSE", fontsize=6)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=6)
    fig.tight_layout()
    S.save(fig, out)


# ------------------------------------------------------------------------------------------- Fig 11

def fig11_generalization(means: pd.DataFrame, cis: pd.DataFrame | None, out: Path) -> None:
    """means: index=suite, columns=labels (<=6)."""
    cols = list(means.columns)
    colors = [PFN_COLORS.get(c, FOCUS_COLORS.get(c, S.NEUTRAL)) for c in cols]
    fallback = iter([S.SERIES[5], S.SERIES[7], S.SERIES[3]])
    colors = [c if c != S.NEUTRAL else next(fallback, S.NEUTRAL) for c in colors]
    fig, ax = plt.subplots(figsize=(S.COL2, 2.5))
    w = 0.8 / len(cols)
    x = np.arange(len(means))
    for i, (c, col) in enumerate(zip(cols, colors)):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, means[c], width=w * 0.92, color=col, label=c)
        if cis is not None and c in cis:
            lo, hi = np.array([v[0] for v in cis[c]]), np.array([v[1] for v in cis[c]])
            ax.errorbar(x + (i - (len(cols) - 1) / 2) * w, means[c], yerr=[means[c] - lo, hi - means[c]], fmt="none",
                        ecolor=S.INK2, lw=0.6, capsize=1)
    ax.set_xticks(x, [s.replace("ood_", "OOD: ").replace("id", "ID") for s in means.index], fontsize=6.5, rotation=15)
    ax.set_ylabel("mean NRMSE (lower is better)")
    ax.legend(ncol=len(cols), fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    S.save(fig, out)


# ------------------------------------------------------------------------------------------- Fig 12

def fig12_cd(tabs: dict[str, pd.DataFrame], out: Path, return_fig: bool = False):
    """Critical-difference diagrams (Demšar 2006 layout): best methods on the right, labels outside the connectors."""
    n = len(tabs)
    fig, axs = plt.subplots(n, 1, figsize=(S.COL2, 1.55 * n), squeeze=False)
    for ax, (title, tab) in zip(axs[:, 0], tabs.items()):
        ranks = T.avg_ranks(tab).sort_values()
        k, N = tab.shape[1], tab.shape[0]
        cd = T.nemenyi_cd(k, N)
        ax.set_xlim(k + 0.5, 0.5)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.plot([1, k], [0.85, 0.85], color=S.INK2, lw=0.8)
        for i in range(1, k + 1):
            ax.plot([i, i], [0.83, 0.87], color=S.INK2, lw=0.8)
            ax.text(i, 0.92, str(i), ha="center", fontsize=6, color=S.INK2)
        ax.plot([1, 1 + cd], [0.98, 0.98], color=S.SERIES[7], lw=1.5)
        ax.text(1 + cd / 2, 1.0, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=6, color=S.SERIES[7])
        items = list(ranks.items())
        half = (len(items) + 1) // 2
        for j, (lab, r) in enumerate(items):
            best = j < half  # best half on the right-hand side (low ranks)
            yy = 0.72 - (j if best else len(items) - 1 - j) * (0.62 / max(half, 1))
            x_end = 0.5 if best else k + 0.5  # connectors stop at the axis edge; labels sit in the margin
            col = S.SERIES[0] if lab.startswith("MAPLE") else (S.SERIES[1] if lab.startswith("PFN") else S.INK)
            ax.plot([r, r, x_end], [0.85, yy, yy], color=S.MUTED, lw=0.6, clip_on=False)
            ax.annotate(f"{lab} ({r:.2f})", (x_end, yy), xytext=(3 if best else -3, 0), textcoords="offset points",
                        ha="left" if best else "right", va="center", fontsize=5.8, color=col, annotation_clip=False)
        rs = ranks.values
        cliques = []
        for i in range(len(rs)):
            j = i
            while j + 1 < len(rs) and rs[j + 1] - rs[i] < cd:
                j += 1
            if j > i and not any(a <= i and j <= b for a, b in cliques):
                cliques.append((i, j))
        for c_i, (a, b) in enumerate(cliques):
            yy = 0.80 - 0.035 * c_i
            ax.plot([rs[a] - 0.03, rs[b] + 0.03], [yy, yy], color=S.INK, lw=2)
        ax.text(k + 0.5, 0.02, f"{title}: N = {N} datasets, k = {k}", fontsize=6.5, color=S.INK2)
    fig.subplots_adjust(left=0.17, right=0.83, top=0.97, bottom=0.03, hspace=0.25)
    S.save(fig, out)
    if return_fig:
        fig.canvas.draw()
        return fig
    plt.close(fig)
    return None


# ------------------------------------------------------------------------------------------- Fig 13

def fig13_disagreement(sdf: pd.DataFrame, real_dis: pd.DataFrame, out: Path, label: str = "PFN-D") -> None:
    fig, axs = plt.subplots(1, 2, figsize=(S.COL2, 2.6))
    d = sdf[(sdf.label == label) & sdf.gap.notna()]
    for (name, sub), c in zip(((n, g) for n, g in d.groupby(d.suite == "id")), (S.SERIES[1], S.SERIES[0])):
        lab = "ID tasks" if name else "OOD tasks"
        axs[0].scatter(sub.gap.clip(1e-5), sub.excess.clip(1e-5), s=3, alpha=0.45, color=c, label=lab, lw=0)
    lim = [1e-4, 10]
    axs[0].plot(lim, lim, color=S.MUTED, ls="--", lw=0.8)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("PFN–committee gap Δ̄ (label-free)")
    axs[0].set_ylabel("PFN excess error ‖θ − f*‖²")
    axs[0].set_title(f"(a) synthetic tasks, {label}")
    axs[0].legend(fontsize=6, markerscale=3)
    if len(real_dis):
        axs[1].scatter(real_dis.gap.clip(1e-5), real_dis.pfn_mse.clip(1e-5), s=6, color=S.SERIES[0], alpha=0.6, lw=0)
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
    axs[1].set_xlabel("PFN–committee gap on test points")
    axs[1].set_ylabel("PFN test MSE (standardised)")
    axs[1].set_title("(b) real-data splits")
    fig.tight_layout()
    S.save(fig, out)


# ------------------------------------------------------------------------------------------- Fig 15

def fig15_scaling(sc: pd.DataFrame, out: Path) -> None:
    """sc: rows with n_train, label, total_s, nrmse (means over datasets x repeats)."""
    fig, axs = plt.subplots(1, 2, figsize=(S.COL2, 2.4))
    fixed = {"PFN-D": S.SERIES[0], "ExtraTrees": S.SERIES[5], "CatBoost": S.SERIES[7], "RF": S.SERIES[3],
             "XGBoost": S.SERIES[6], "LightGBM": S.SERIES[2], "Ridge": S.INK2}
    spare = iter([c for c in S.SERIES if c not in fixed.values()])
    cols = {lab: fixed.get(lab) or next(spare) for lab in sc.label.unique()}
    for lab, g in sc.groupby("label"):
        g = g.sort_values("n_train")
        axs[0].plot(g.n_train, g.total_s, marker="o", color=cols[lab], label=lab, lw=1.1)
        axs[1].plot(g.n_train, g.nrmse, marker="o", color=cols[lab], label=lab, lw=1.1)
    for ax in axs:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("training-set size n")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("fit + predict time (s)")
    axs[1].set_ylabel("mean NRMSE")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, ncol=len(l_), fontsize=6, loc="lower center", bbox_to_anchor=(0.5, -0.01), frameon=False)
    axs[0].set_title("(a) computational cost")
    axs[1].set_title("(b) accuracy")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    S.save(fig, out)


# ------------------------------------------------------------------------------------------- supplementary

def figS_pred_vs_truth(items: list[tuple[str, np.ndarray, dict]], out: Path) -> None:
    """items: (dataset, y_test, {label: pred})"""
    n = len(items)
    fig, axs = plt.subplots(2, n, figsize=(S.COL2, 3.6), squeeze=False)
    for j, (ds, y, preds) in enumerate(items):
        for lab, c in zip(preds, (S.SERIES[0], S.SERIES[5])):
            p = preds[lab]
            axs[0, j].scatter(y, p, s=2, alpha=0.4, color=c, label=lab, lw=0)
            axs[1, j].hist(p - y, bins=40, histtype="step", color=c, label=lab, lw=1)
        lo, hi = np.percentile(y, [0.5, 99.5])
        axs[0, j].plot([lo, hi], [lo, hi], color=S.MUTED, ls="--", lw=0.7)
        axs[0, j].set_title(_short(ds), fontsize=7)
        axs[0, j].set_xlabel("true y", fontsize=6)
        axs[1, j].set_xlabel("residual (pred − true)", fontsize=6)
    axs[0, 0].set_ylabel("predicted y")
    axs[1, 0].set_ylabel("count")
    axs[0, 0].legend(fontsize=5.5, markerscale=4)
    fig.tight_layout()
    S.save(fig, out)


def figS_seed_stability(seed_sd: pd.Series, rep_sd: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(S.COL1 + 0.5, 2.6))
    common = seed_sd.dropna().index.intersection(rep_sd.dropna().index)
    ax.scatter(rep_sd[common], seed_sd[common], s=10, color=S.SERIES[0], lw=0)
    m = max(rep_sd[common].max(), seed_sd[common].max()) * 1.05
    ax.plot([0, m], [0, m], color=S.MUTED, ls="--", lw=0.8)
    ax.set_xlabel("SD of NRMSE across split repeats")
    ax.set_ylabel("SD of NRMSE across pretraining seeds")
    ax.set_title("PFN-D stability (n = 512, one point per dataset)", fontsize=7)
    fig.tight_layout()
    S.save(fig, out)
