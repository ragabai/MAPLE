"""Matplotlib house style for publication figures.

Palette: the validated reference categorical palette (dataviz skill, references/palette.md), checked on a white
surface with scripts/validate_palette.js (all hard gates pass; three slots are below 3:1 contrast, so every figure
has a legend and a companion table). Categorical slots are assigned in fixed order and follow the entity, never
its rank; scatter-type figures use at most three hues.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
NEUTRAL = "#b5b3ab"  # de-emphasised baselines

# Fixed entity -> colour map (stable across all figures).
METHOD_COLORS = {"MAPLE-F": SERIES[0], "MAPLE-E": SERIES[6], "D": SERIES[0], "C": SERIES[2], "B": SERIES[3],
                 "A": SERIES[1], "P0": SERIES[4], "catboost": SERIES[5], "catboost_tuned": SERIES[5],
                 "xgboost": SERIES[7], "xgboost_tuned": SERIES[7]}

COL1, COL2 = 3.4, 7.0  # inches: single column / full width (MDPI text width ~ 17 cm)


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4,
        "legend.frameon": False, "axes.prop_cycle": matplotlib.cycler(color=SERIES), "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
        "svg.fonttype": "none"})


def save(fig, stem: str | Path) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("pdf", "svg", "png"):
        p = stem.with_suffix("." + ext)
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out
