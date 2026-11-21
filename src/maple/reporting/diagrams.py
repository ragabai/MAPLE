"""Schematic figures (Figures 1-2) drawn with matplotlib patches at the manuscript's print width.

Content mirrors research/methodology/formulation.md and is read from the generator code where possible.

Figure 3 (mechanism) is data-driven and lives in maple.reporting.figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

from maple.reporting import style as S

# ------------------------------------------------------------------ shared card style (Figures 1-2)
# Soft role palette: one hue per meaning, pale fills, muted outlines, one grey for every connector.
ROLE = {
    "green": dict(fill="#E8F3EB", head="#D3E9D9", edge="#84B491", ink="#2F6F42"),   # synthetic tasks
    "blue": dict(fill="#E7F0FA", head="#D4E4F5", edge="#7FA5CE", ink="#2D5E91"),    # the PFN
    "amber": dict(fill="#FCF0E1", head="#F6E0C3", edge="#DDAA6F", ink="#9A5F1E"),   # classical experts
    "purple": dict(fill="#EEEAF7", head="#DFD8F0", edge="#A092CA", ink="#584791"),  # weighting decisions
    "gray": dict(fill="#EDEFF2", head="#DFE3E8", edge="#9BA2AD", ink="#4A515B"),    # real data
    "rose": dict(fill="#FBEFF1", head="#F2CFD6", edge="#D48C9B", ink="#A23F56"),    # held out (OOD)
    "slate": dict(fill="#F1F4F8", head="#DCE4EE", edge="#8FA0B8", ink="#3C4A5F"),   # panels
}
D_INK, D_MUTED, D_ARROW = "#1F2329", "#4F5660", "#8B929D"
D_RC = {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none"}
T_TITLE, T_DESC, T_EDGE, T_PANEL = 7.2, 6.3, 6.0, 7.6  # printed sizes (pt): drawn at print width
LINE = 0.112  # inches between description lines
TEXT_WIDTH_IN = 138.6 / 25.4  # MDPI text width (measured from the compiled manuscript)


def _card(ax, b, role, title, lines=(), tone="head", small=False, valign="center"):
    """Rounded card: bold title (one or more lines), centred description lines below (b = x0, y0, x1, y1, in)."""
    x0, y0, x1, y1 = b
    r = ROLE[role]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=r[tone], ec=r["edge"], lw=0.8, zorder=2, gid="card"))
    t_size, d_size, step = (T_TITLE - 0.4, T_DESC - 0.3, LINE - 0.006) if small else (T_TITLE, T_DESC, LINE)
    t_h = 0.118 + 0.118 * title.count(chr(10))  # height of the title block (one step per extra line)
    block = t_h + len(lines) * step
    top = y1 - 0.075 if valign == "top" else (y0 + y1) / 2 + block / 2
    cx = (x0 + x1) / 2
    ax.text(cx, top, title, ha="center", va="top", fontsize=t_size, weight="bold", color=D_INK, zorder=6,
            linespacing=1.15, gid="card-text")
    for i, line in enumerate(lines):
        ax.text(cx, top - t_h - 0.010 - i * step, line, ha="center", va="top", fontsize=d_size, color=D_INK,
                zorder=6, gid="card-text")


def _panel(ax, b, title, role="slate"):
    x0, y0, x1, y1 = b
    r = ROLE[role]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.07",
                                fc=r["fill"], ec=r["edge"], lw=0.8, ls=(0, (4, 2.5)), zorder=0))
    ax.text(x0 + 0.12, y1 - 0.07, title, ha="left", va="top", fontsize=T_PANEL, weight="bold",
            color=r["ink"], zorder=6)


def _rounded_path(pts, r):
    pts = [np.asarray(p, float) for p in pts]
    verts, codes = [pts[0]], [MPath.MOVETO]
    for i in range(1, len(pts) - 1):
        p, a, b = pts[i], pts[i - 1], pts[i + 1]
        d1, d2 = (p - a) / np.linalg.norm(p - a), (b - p) / np.linalg.norm(b - p)
        rr = min(r, np.linalg.norm(p - a) / 2, np.linalg.norm(b - p) / 2)
        verts += [p - d1 * rr, p, p + d2 * rr]
        codes += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
    verts.append(pts[-1])
    codes.append(MPath.LINETO)
    return MPath(verts, codes)


def _route(ax, pts, dashed=False):
    """Orthogonal connector with rounded elbows and one solid head; all connectors share colour and weight."""
    pts = [np.asarray(p, float) for p in pts]
    d = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
    shaft = pts[:-1] + [pts[-1] - d * 0.05]
    ax.add_patch(PathPatch(_rounded_path(shaft, 0.07), fc="none", ec=D_ARROW, lw=1.1, zorder=4,
                           ls=(0, (3.5, 2.2)) if dashed else "-", capstyle="butt", joinstyle="round"))
    ax.add_patch(FancyArrowPatch(tuple(pts[-1] - d * 0.06), tuple(pts[-1]), arrowstyle="-|>", mutation_scale=8,
                                 color=D_ARROW, lw=1.1, shrinkA=0, shrinkB=0, zorder=5))


def _label(ax, x, y, s, ha="center", va="bottom", rotation=0):
    ax.text(x, y, s, ha=ha, va=va, fontsize=T_EDGE, color=D_MUTED, rotation=rotation, zorder=6, gid="edge-label")


def figure1_architecture(out: Path, return_fig: bool = False):
    """MAPLE overview: the disagreement-steered pretraining loop (top) and routed inference (bottom).

    Drawn at the manuscript's print width (MDPI text width 138.6 mm), so font sizes are the printed sizes.
    Family and cell counts come from the generator and the default training config, as in the manuscript.
    """
    from maple.model.train import default_train_config
    from maple.priors.families import train_families
    n_families = len(train_families())
    n_levels = default_train_config()["sampler"]["max_complexity"] + 1
    n_cells = n_families * n_levels
    W, H = TEXT_WIDTH_IN, 3.99
    with plt.rc_context(D_RC):
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")

        # one 3-column grid shared by both panels; the checkpoint lane runs outside the panels on the right
        px0, px1, lane = 0.03, W - 0.24, W - 0.12
        g, x_lo, x_hi = 0.36, px0 + 0.12, px1 - 0.12
        w = (x_hi - x_lo - 2 * g) / 3
        c = [(x_lo + i * (w + g), x_lo + i * (w + g) + w) for i in range(3)]
        mid = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)  # noqa: E731

        # ---- top panel: pretraining loop
        top0 = 1.80
        _panel(ax, (px0, top0, px1, H - 0.03), "Pretraining on synthetic tasks")
        r2 = (top0 + 0.12, top0 + 0.74)
        r1 = (r2[1] + 0.50, r2[1] + 1.12)
        prior = (c[0][0], r1[0], c[0][1], r1[1])
        gen = (c[1][0], r1[0], c[1][1], r1[1])
        pfn = (c[2][0], r1[0], c[2][1], r1[1])
        samp = (c[0][0], r2[0], c[0][1], r2[1])
        probe = (c[1][0], r2[0], c[2][1], r2[1])
        _card(ax, prior, "green", "Mathematical prior",
              [f"{n_families} function families", f"× {n_levels} complexity levels",
               f"= {n_cells} cells, weights $q$"])
        _card(ax, gen, "green", r"Task generator $G_c$",
              ["correlated, irrelevant,", "categorical and missing", "features; varied noise"])
        _card(ax, pfn, "blue", r"MAPLE-Net $\theta$", ["PFN: pre-LN transformer,", "query MSE on",
                                                        "standardised targets"])
        _card(ax, samp, "purple", "Adaptive sampler",
              [r"$\hat s_c$: mean gap, smoothed", r"$q_c \propto p^0_c\, e^{\hat s_c/\tau}$",
               r"+ uniform floor $\gamma$"])
        _card(ax, probe, "amber", r"Probe round, every $U$ steps",
              [r"$m$ probe tasks per cell, never used for gradients",
               r"committee $\bar f$: ridge, kNN, ExtraTrees, HistGB, RBF kernel ridge",
               r"cell score: gap $\bar\Delta = \|\theta - \bar f\,\|^2$ on the queries"])
        ym = mid(prior)[1]
        _route(ax, [(prior[2], ym), (gen[0], ym)])
        _label(ax, (prior[2] + gen[0]) / 2, ym + 0.03, r"$c \sim q$")
        _route(ax, [(gen[2], ym), (pfn[0], ym)])
        _label(ax, (gen[2] + pfn[0]) / 2, ym + 0.03, "tasks")
        xg, xp, xs = mid(gen)[0], mid(pfn)[0], mid(samp)[0]
        ymid = (r1[0] + r2[1]) / 2
        _route(ax, [(xg, r1[0]), (xg, r2[1])])
        _label(ax, xg + 0.05, ymid, "probe tasks", ha="left", va="center")
        _route(ax, [(xp, r1[0]), (xp, r2[1])])
        _label(ax, xp + 0.05, ymid, r"$\theta(x)$ on probes", ha="left", va="center")
        yb = mid(samp)[1]
        _route(ax, [(probe[0], yb), (samp[2], yb)])
        _label(ax, (probe[0] + samp[2]) / 2, yb + 0.03, r"$\hat s_c$")
        _route(ax, [(xs, r2[1]), (xs, r1[0])])
        _label(ax, xs + 0.05, ymid, "re-weight $q$", ha="left", va="center")

        # ---- bottom panel: inference on a real dataset
        bot1 = 1.55
        _panel(ax, (px0, 0.03, px1, bot1), "Inference on a real dataset")
        ppfn = (c[1][0], 0.70, c[1][1], 1.22)
        expt = (c[1][0], 0.12, c[1][1], 0.60)
        yc = (expt[1] + ppfn[3]) / 2
        split = (c[0][0], yc - 0.26, c[0][1], yc + 0.26)
        rout = (c[2][0], yc - 0.36, c[2][1], yc + 0.36)
        _card(ax, split, "gray", "Training split", ["train-only preprocessing;", "≤ 32 features for the PFN"])
        _card(ax, ppfn, "blue", "Pretrained MAPLE-Net", ["in-context prediction,", "4 feature permutations"])
        _card(ax, expt, "amber", "Classical experts", ["ridge, kNN, ExtraTrees,", "HistGB, RBF kernel ridge"])
        _card(ax, rout, "purple", "Competence routing",
              [r"$w_k \propto \exp(-\beta\, e_k / e_{\min})$,", r"$e_k$: out-of-fold error;",
               r"global (E) or $\kappa$-nearest (F);", r"$\beta$, $\kappa$ set on synthetic tasks"])
        xa, xb = (c[0][1] + c[1][0]) / 2, (c[1][1] + c[2][0]) / 2
        for box in (ppfn, expt):
            yy = mid(box)[1]
            _route(ax, [(split[2], yc), (xa, yc), (xa, yy), (box[0], yy)])
            _route(ax, [(box[2], yy), (xb, yy), (xb, yc), (rout[0], yc)])

        # ---- checkpoint hand-over: PFN (top) -> pretrained PFN (bottom), via the right-hand lane
        yh = (bot1 + top0) / 2 - 0.02
        _route(ax, [(pfn[2], ym - 0.12), (lane, ym - 0.12), (lane, yh), (mid(ppfn)[0], yh), (mid(ppfn)[0], ppfn[3])],
               dashed=True)
        _label(ax, mid(rout)[0], yh + 0.025, r"pretrained checkpoint $\theta$")
        if return_fig:
            fig.canvas.draw()
            S.save(fig, out)
            return fig
        S.save(fig, out)
    return None


FAMILY_GROUPS = {  # presentation grouping of the training families (every training family appears once)
    "Linear &\npolynomial": ["linear", "polynomial", "interaction"],
    "Nonlinear": ["exponential", "logarithmic", "power_law", "rational", "trigonometric", "rbf"],
    "Saturation\n& growth": ["logistic", "michaelis_menten", "piecewise"],
    "Physical &\ndynamical": ["physics", "damped_oscillator", "logistic_ode", "lotka_volterra"],
    "Regime\nchanges": ["conditional", "discontinuous"],
}
FAMILY_NAMES = {"linear": "linear", "polynomial": "polynomial", "interaction": "pairwise products",
                "exponential": "exponential", "logarithmic": "logarithmic", "power_law": "power law",
                "rational": "rational", "trigonometric": "trigonometric", "rbf": "Gaussian bumps",
                "logistic": "logistic", "michaelis_menten": "Michaelis–Menten", "piecewise": "hinge",
                "physics": "physical laws", "damped_oscillator": "damped oscillator", "logistic_ode": "logistic ODE",
                "lotka_volterra": "Lotka–Volterra", "conditional": "regime switch", "discontinuous": "step jumps",
                "gompertz": "Gompertz", "sir": "SIR", "hyperbolic": "hyperbolic", "interaction3": "three-way products",
                "voronoi": "Voronoi pieces", "mixed_regime": "mixed regimes"}


def figure2_generator(out: Path, return_fig: bool = False):
    """Mathematical task generator: family groups (top) and the assembly of one task (bottom).

    Drawn at print width. Families, counts and thresholds come from maple.priors / maple.tasks.generator.
    """
    from maple.model.train import default_train_config
    from maple.priors.families import FAMILIES, train_families
    from maple.tasks.generator import default_generator_config
    gcfg = default_generator_config()
    train = train_families()
    grouped = [f for fams in FAMILY_GROUPS.values() for f in fams]
    assert sorted(grouped) == sorted(train), "every training family must appear in exactly one group"
    held_fam = [k for k, v in FAMILIES.items() if v.split == "ood"]
    held_struct = [k for k, v in FAMILIES.items() if v.split == "ood_structure"]
    n_levels = default_train_config()["sampler"]["max_complexity"] + 1
    lo_rel, hi_rel = gcfg["noise_rel_range"]
    W, H = TEXT_WIDTH_IN, 3.44
    with plt.rc_context(D_RC):
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")

        # ---- top: training families (green) and held-out structure (rose)
        t0, t1 = 1.95, H - 0.03
        tx1 = 4.30
        _panel(ax, (0.03, t0, tx1, t1),
               f"{len(train)} training families × {n_levels} complexity levels = {len(train) * n_levels} cells",
               role="green")
        _panel(ax, (tx1 + 0.10, t0, W - 0.03, t1), "Held out", role="rose")
        gap, xl, xr = 0.07, 0.11, tx1 - 0.08
        w = (xr - xl - gap * (len(FAMILY_GROUPS) - 1)) / len(FAMILY_GROUPS)
        cy0, cy1 = t0 + 0.09, t1 - 0.27
        for i, (title, fams) in enumerate(FAMILY_GROUPS.items()):
            x0 = xl + i * (w + gap)
            _card(ax, (x0, cy0, x0 + w, cy1), "green", title, [FAMILY_NAMES[f] for f in fams], small=True, valign="top")
        hx0, hx1 = tx1 + 0.18, W - 0.11
        hm = (cy0 + cy1) / 2
        _card(ax, (hx0, hm + 0.035, hx1, cy1), "rose", "OOD families", [FAMILY_NAMES[f] for f in held_fam],
              small=True)
        _card(ax, (hx0, cy0, hx1, hm - 0.035), "rose", "OOD structures", [FAMILY_NAMES[f] for f in held_struct],
              small=True)

        # ---- bottom: assembly of one task for a cell (family, complexity)
        b1 = 1.85
        _panel(ax, (0.03, 0.03, W - 0.03, b1), "Assembly of one task for cell (family, complexity level)")
        steps = [("1. Features", ["copula-correlated,", "mixed marginals;", "irrelevant, redundant,",
                                  "categorical columns"]),
                 ("2. Target", ["primary component", "+ 0–2 additive", "components from", "other families"]),
                 ("3. Noise", ["Gaussian or", "heteroscedastic,", f"relative level {lo_rel:g}–{hi_rel:g}"]),
                 ("4. Missing + split", ["MCAR 2–20% in", f"{gcfg['p_missing']:.0%} of tasks;",
                                         "context / query"]),
                 ("5. Standardise", ["with context", "statistics only"])]
        sg, sx0, sx1 = 0.17, 0.15, W - 0.15
        sw = (sx1 - sx0 - sg * (len(steps) - 1)) / len(steps)
        sy0, sy1 = 0.84, b1 - 0.29
        boxes = []
        for i, (title, lines) in enumerate(steps):
            x0 = sx0 + i * (sw + sg)
            boxes.append((x0, sy0, x0 + sw, sy1))
            _card(ax, boxes[-1], "green", title, lines, small=True, valign="top")
            if i:
                ym = (sy0 + sy1) / 2
                _route(ax, [(boxes[-2][2], ym), (x0, ym)])
        chk = (sx0, 0.13, sx1, 0.55)
        _card(ax, chk, "purple", f"Validity checks (any failure: resample from step 1, at most "
                                 f"{gcfg['max_attempts']} attempts)",
              [f"after step 2: finite target, range ≤ {gcfg['max_range_over_std']:g} SD, "
               f"≥ {gcfg['min_unique_frac']:.0%} distinct values",
               f"after step 5: every target within ±{gcfg['max_abs_ctx_z']:g} context SD"], small=True)
        for i in (1, 4):
            x = (boxes[i][0] + boxes[i][2]) / 2
            _route(ax, [(x, sy0), (x, chk[3])], dashed=True)
            _label(ax, x + 0.04, (sy0 + chk[3]) / 2, "check", ha="left", va="center")
        x = (boxes[0][0] + boxes[0][2]) / 2
        _route(ax, [(x, chk[3]), (x, sy0)], dashed=True)
        _label(ax, x + 0.04, (sy0 + chk[3]) / 2, "resample", ha="left", va="center")
        if return_fig:
            fig.canvas.draw()
            S.save(fig, out)
            return fig
        S.save(fig, out)
    return None
