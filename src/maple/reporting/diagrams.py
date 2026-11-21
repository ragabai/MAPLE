"""Schematic figures (Figures 1-2) drawn with matplotlib patches. Content mirrors research/methodology/formulation.md.

Figure 3 (mechanism) is data-driven and lives in maple.reporting.figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from maple.reporting import style as S

TINT = {"blue": "#e8f1fc", "orange": "#fdeee7", "aqua": "#e5f6f0", "violet": "#eeecfb", "gray": "#f3f3f1"}
EDGE = {"blue": S.SERIES[0], "orange": S.SERIES[1], "aqua": S.SERIES[2], "violet": S.SERIES[6], "gray": S.AXIS}
NL = "\n"


def _box(ax, x, y, w, h, title, body="", tone="blue", title_size=7.5, body_size=6.3, body_dy=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", lw=0.9,
                                ec=EDGE[tone], fc=TINT[tone]))
    ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top", fontsize=title_size, weight="bold", color=S.INK)
    if body:
        ax.text(x + w / 2, y + h - body_dy, body, ha="center", va="top", fontsize=body_size, color=S.INK2,
                linespacing=1.25)


def _arrow(ax, p, q, text="", color=S.INK2, rad=0.0, ls="-", tx=0.0, ty=0.012):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, ha="center", va="bottom", fontsize=6,
                color=S.INK2, style="italic")


def _canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _l(*lines):
    return NL.join(lines)


def figure1_architecture(out: Path) -> None:
    fig, ax = _canvas(S.COL2, 3.4)
    ax.text(0.01, 0.985, "Offline pretraining (synthetic tasks only)", fontsize=7.5, color=S.MUTED, va="top")
    ax.text(0.01, 0.405, "Inference on a real dataset", fontsize=7.5, color=S.MUTED, va="top")
    _box(ax, 0.01, 0.50, 0.19, 0.34, "Mathematical prior",
         _l("18 training families ×", "3 complexity levels", "= 54 cells c", "(polynomial, exp/log, trig,",
            "RBF, saturation, ODE,", "piecewise, interactions …)"))
    _box(ax, 0.25, 0.50, 0.17, 0.34, "Task generator G_c",
         _l("features: copula, mixed", "marginals, irrelevant,", "redundant, categorical,", "missing; noise:",
            "homo/heteroscedastic"), "aqua")
    _box(ax, 0.47, 0.70, 0.20, 0.14, "Probes + committee", "Ridge, kNN, ET, HGB, RBF", "orange")
    _box(ax, 0.47, 0.50, 0.20, 0.17, "PFN θ (transformer)", _l("context → query", "MSE, standardised y",
                                                                "≈ posterior mean"), "violet")
    _box(ax, 0.73, 0.50, 0.26, 0.34, "Adaptive sampler q",
         _l("cell score s_c = mean (θ(x) − f̄(x))²", "(label-free gap)", "", "q ∝ p⁰·exp(s/τ), floor γ/M",
            "(Props. 1–3, Theorem 1)"), "blue")
    _arrow(ax, (0.20, 0.67), (0.25, 0.67), "cells")
    _arrow(ax, (0.42, 0.77), (0.47, 0.77), "probes")
    _arrow(ax, (0.42, 0.585), (0.47, 0.585), "batches")
    _arrow(ax, (0.67, 0.77), (0.73, 0.77), "f̄(x)")
    _arrow(ax, (0.67, 0.585), (0.73, 0.585), "θ(x)")
    _arrow(ax, (0.86, 0.84), (0.105, 0.84), "every U steps: re-weight the cells", rad=0.09, color=S.SERIES[0], ty=0.04)
    _box(ax, 0.01, 0.04, 0.19, 0.30, "Training split",
         _l("train-only preprocessing:", "encode, impute, log-skew,", "standardise, MI-select ≤ 32"), "gray")
    _box(ax, 0.25, 0.04, 0.21, 0.30, "Pretrained PFN",
         _l("in-context prediction,", "4 feature permutations", "(context bagging if n > 1024)"), "violet")
    _box(ax, 0.51, 0.04, 0.20, 0.30, "Classical experts",
         _l("Ridge, kNN, ExtraTrees,", "HistGB, RBF kernel ridge;", "5-fold out-of-fold errors"), "orange")
    _box(ax, 0.76, 0.04, 0.23, 0.30, "Competence routing",
         _l("E: global w_k ∝ exp(−β e_k / e_min)", "F: local, κ nearest", "training points",
            "β, κ chosen on synthetic", "tasks only"), "blue")
    _arrow(ax, (0.20, 0.19), (0.25, 0.19))
    _arrow(ax, (0.46, 0.19), (0.51, 0.19))
    _arrow(ax, (0.71, 0.19), (0.76, 0.19))
    _arrow(ax, (0.57, 0.50), (0.37, 0.34), "checkpoint θ", ls="--", rad=-0.1, tx=0.06)
    S.save(fig, out)


def figure2_generator(out: Path) -> None:
    fig, ax = _canvas(S.COL2, 2.9)
    groups = {
        _l("Linear &", "polynomial"): ["linear", "polynomial", "interaction"],
        "Nonlinear": ["exponential", "logarithmic", "power_law", "rational", "trigonometric", "rbf"],
        _l("Saturation &", "growth"): ["logistic", "michaelis_menten", "piecewise"],
        _l("Physical &", "dynamical"): ["physics", "damped_oscillator", "logistic_ode", "lotka_volterra"],
        _l("Complex", "structure"): ["conditional", "discontinuous"],
        _l("Held out", "(OOD only)"): ["gompertz", "sir", "hyperbolic", "interaction3", "voronoi", "mixed_regime"],
    }
    tones = ["blue", "aqua", "orange", "violet", "gray", "orange"]
    x = 0.01
    for (g, fams), tone in zip(groups.items(), tones):
        _box(ax, x, 0.50, 0.155, 0.48, g, NL.join(f.replace("_", " ") for f in fams), tone, body_size=6.2,
             body_dy=0.13)
        x += 0.165
    ax.text(0.01, 0.455, "Task assembly for cell (family, complexity)", fontsize=7.5, color=S.MUTED, va="top")
    steps = [("1. Features", _l("copula correlation,", "mixed marginals,", "irrelevant/redundant,", "categorical codes")),
             ("2. Target", _l("primary component +", "0–2 extra components", "(additive; nested only", "in OOD suite)")),
             ("3. Validity", _l("finite, non-constant,", "range ≤ 25 sd,", "|z| ≤ 12 context sd;", "else resample")),
             ("4. Noise", _l("Gaussian or", "heteroscedastic", "(OOD: Student-t,", "Laplace)")),
             ("5. Missing + split", _l("MCAR 2–20 %;", "context / query;", "statistics from", "context only"))]
    x = 0.01
    for i, (t, b) in enumerate(steps):
        _box(ax, x, 0.02, 0.17, 0.38, t, b, "gray", body_size=6.0, body_dy=0.08)
        if i < len(steps) - 1:
            _arrow(ax, (x + 0.17, 0.21), (x + 0.2, 0.21))
        x += 0.2
    S.save(fig, out)
