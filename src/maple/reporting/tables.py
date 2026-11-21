"""Publication tables in CSV, XLSX, Markdown and LaTeX, with best (bold) / second-best (underlined) marking."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def mark_best(df: pd.DataFrame, lower_is_better: bool = True, axis: int = 1) -> pd.DataFrame:
    """'best'/'second' labels per row (axis=1) or per column (axis=0); NaN never wins."""
    marks = pd.DataFrame("", index=df.index, columns=df.columns)
    it = df.iterrows() if axis == 1 else df.items()
    for lab, s in it:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            continue
        order = s.sort_values(ascending=lower_is_better)
        vals = order.unique()
        for rank, tag in ((0, "best"), (1, "second")):
            if rank < len(vals):
                for idx in order[order == vals[rank]].index:
                    if axis == 1:
                        marks.loc[lab, idx] = tag
                    else:
                        marks.loc[idx, lab] = tag
    return marks


def _cell(v, sd, fmt, mark, style):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    s = fmt.format(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
    if style == "tex" and not isinstance(v, (int, float, np.floating, np.integer)):
        s = _latex_escape(s)
    if sd is not None and not (isinstance(sd, float) and np.isnan(sd)):
        s_sd = fmt.format(sd)
        s = f"{s} ± {s_sd}" if style == "md" else f"{s} $\\pm$ {s_sd}"
    if mark == "best":
        core = s.split(" ")[0]
        s = s.replace(core, f"**{core}**" if style == "md" else f"\\textbf{{{core}}}", 1)
    elif mark == "second":
        core = s.split(" ")[0]
        s = s.replace(core, f"_{core}_" if style == "md" else f"\\underline{{{core}}}", 1)
    return s


_TEX_UNICODE = {"✓": "$\\checkmark$", "✗": "$\\times$", "τ": "$\\tau$", "γ": "$\\gamma$", "λ": "$\\lambda$",
                "Δ": "$\\Delta$", "χ": "$\\chi$", "ρ": "$\\rho$", "β": "$\\beta$", "κ": "$\\kappa$", "≤": "$\\le$",
                "≥": "$\\ge$", "−": "$-$", "∈": "$\\in$", "±": "$\\pm$", "²": "$^2$", "×": "$\\times$", "‖": "$\\|$",
                "θ": "$\\theta$", "σ": "$\\sigma$", "≈": "$\\approx$"}


def _latex_escape(s: str) -> str:
    out = str(s).replace("\\", "\\textbackslash{}")
    for a, b in (("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#"), ("{", "\\{"), ("}", "\\}")):
        out = out.replace(a, b)
    out = out.replace("\\textbackslash\\{\\}", "\\textbackslash{}")
    out = out.replace("^", "\\textasciicircum{}").replace("~", "\\textasciitilde{}")
    for a, b in _TEX_UNICODE.items():
        out = out.replace(a, b)
    return out


def _breakable_tex(s: str) -> str:
    return _latex_escape(s).replace(r"\_", r"\_\allowbreak{}").replace("/", r"/\allowbreak{}")


_CENTER = r">{\centering\arraybackslash}X"
_LEFT = r">{\raggedright\arraybackslash}X"
_ROWS_PER_PART = {"table01_datasets": 24, "table10_statistics": 27}


def _table_columns(name: str, cols: list, wrap: dict[str, str] | None) -> str:
    if wrap:
        return "l" + "".join(wrap.get(c, "l") for c in cols)
    if name == "table12_generator":
        return r"p{0.15\linewidth}c" + _LEFT + "c" + _CENTER * 3
    if name == "table01_datasets":
        return r">{\raggedright\arraybackslash}p{0.17\linewidth}" + _CENTER * 8 + r"p{0.125\linewidth}" + _CENTER
    if name == "table10_statistics":
        return r">{\raggedright\arraybackslash}p{0.29\linewidth}rr" + _CENTER * 6
    if len(cols) >= 6:
        width = "0.13" if len(cols) >= 14 else "0.15" if len(cols) >= 10 else "0.20"
        return rf">{{\raggedright\arraybackslash}}p{{{width}\linewidth}}" + _CENTER * len(cols)
    return "l" + (_LEFT if len(cols) == 1 else _CENTER * len(cols))


def _tex_heading(value: str, narrow: bool) -> str:
    text = _breakable_tex(value)
    if narrow:
        for name, lines in {"ExtraTrees": r"Extra\\Trees", "CatBoost": r"Cat\\Boost", "Stack(experts)": r"Stack\\(experts)",
                            "XGBoost": r"XG\\Boost", "LightGBM": r"Light\\GBM",
                            "TabPFN": r"Tab\\PFN", "TabICL": r"Tab\\ICL"}.items():
            if text.startswith(name):
                return rf"\shortstack{{{lines}{text[len(name):]}}}"
        if text.startswith("MAPLE-Net"):
            lines = _latex_escape(value).replace("MAPLE-Net", r"MAPLE\\-Net", 1).replace("/", r"/\\")
            return rf"\shortstack{{{lines}}}"
    return text


def write_table(df: pd.DataFrame, path_stem: str | Path, caption: str, notes: str = "", sd: pd.DataFrame | None = None,
                lower_is_better: bool | None = True, axis: int = 1, fmt: str = "{:.3f}", label: str | None = None,
                index_name: str | None = None, mark_cols: list[str] | None = None,
                wrap: dict[str, str] | None = None) -> list[Path]:
    """wrap: {column: tabularx spec, e.g. 'X'} -> full-width tabularx with wrapped text columns, no resizebox."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    marks = mark_best(df, lower_is_better, axis) if lower_is_better is not None else pd.DataFrame("", index=df.index, columns=df.columns)
    if mark_cols is not None:  # best/second-best marks only where "lower is better" is meaningful
        marks.loc[:, [c for c in marks.columns if c not in mark_cols]] = ""
    out_csv = df.copy()
    if sd is not None:
        out_csv = pd.concat({"value": df, "sd": sd.reindex_like(df)}, axis=1)
    out_csv.to_csv(stem.with_suffix(".csv"), index_label=index_name or df.index.name or "")
    with pd.ExcelWriter(stem.with_suffix(".xlsx")) as xw:
        out_csv.to_excel(xw, sheet_name="table")
        pd.DataFrame({"caption": [caption], "notes": [notes]}).to_excel(xw, sheet_name="caption", index=False)
    cols = list(df.columns)
    get_sd = (lambda r, c: sd.loc[r, c] if sd is not None and r in sd.index and c in sd.columns else None)
    md = [f"**{caption}**", "", "| " + (index_name or df.index.name or "") + " | " + " | ".join(map(str, cols)) + " |",
          "|" + "---|" * (len(cols) + 1)]
    for r in df.index:
        md.append(f"| {r} | " + " | ".join(_cell(df.loc[r, c], get_sd(r, c), fmt, marks.loc[r, c], "md") for c in cols) + " |")
    if notes:
        md += ["", notes]
    stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    full_width = (len(cols) >= 6 or stem.name == "table13_rq_verdicts"
                  or stem.name.startswith(("table03_rmse_", "table04_mae_", "table05_r2_")))
    narrow = len(cols) >= 10 or stem.name == "table_novelty"
    font = r"\fontsize{8}{10}\selectfont" if narrow else r"\fontsize{9}{11}\selectfont"
    width = r"\linewidth" if full_width else r"\textwidth"
    spec = _table_columns(stem.name, cols, wrap)
    padding = "1" if len(cols) >= 14 else "1.5" if narrow else "2"
    header = _latex_escape(index_name or df.index.name or "") + " & " + " & ".join(_tex_heading(c, narrow) for c in cols)
    rows = []
    for r in df.index:
        cells = [_cell(df.loc[r, c], get_sd(r, c), fmt, marks.loc[r, c], "tex") for c in cols]
        if narrow and sd is not None:
            cells = [r"\shortstack{" + cell.replace(r" $\pm$ ", r"\\$\pm$", 1) + "}"
                     if r" $\pm$ " in cell else cell for cell in cells]
        row_label = _breakable_tex(r) if full_width else _latex_escape(r)
        rows.append(row_label + " & " + " & ".join(cells) + r" \\")
    part_size = _ROWS_PER_PART.get(stem.name, 24 if narrow and sd is not None else len(rows) or 1)
    tex = []
    for start in range(0, len(rows) or 1, part_size):
        tex.append(r"\begin{table}[!htb]")
        if full_width:
            tex.append(r"\begin{adjustwidth}{-\extralength}{0cm}")
        if start:
            tex.extend([r"\ContinuedFloat", rf"\caption[]{{{_latex_escape(caption)} (continued)}}"])
        else:
            tex.extend([rf"\caption{{{_latex_escape(caption)}}}", rf"\label{{{label or 'tab:' + stem.name}}}"])
        tex += [r"\centering", font, rf"\setlength{{\tabcolsep}}{{{padding}pt}}",
                rf"\begin{{tabularx}}{{{width}}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule",
                *rows[start:start + part_size], r"\bottomrule", r"\end{tabularx}"]
        if notes and start + part_size >= len(rows):
            tex.append(rf"\par\vspace{{2pt}}{{\justifying\noindent {_latex_escape(notes)}\par}}")
        if full_width:
            tex.append(r"\end{adjustwidth}")
        tex.append(r"\end{table}")
    stem.with_suffix(".tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return [stem.with_suffix(s) for s in (".csv", ".xlsx", ".md", ".tex")]
