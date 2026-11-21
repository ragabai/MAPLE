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
    tex = ["\\begin{table}[H]", f"\\caption{{{_latex_escape(caption)}}}", f"\\label{{{label or 'tab:' + stem.name}}}", "\\centering",
           *(["\\begin{tabularx}{\\textwidth}{l" + "".join(wrap.get(c, "l") for c in cols) + "}"] if wrap else
             ["\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l" + "c" * len(cols) + "}"]), "\\toprule",
           _latex_escape(index_name or df.index.name or "") + " & " + " & ".join(_latex_escape(c) for c in cols) + " \\\\",
           "\\midrule"]
    for r in df.index:
        tex.append(_latex_escape(r) + " & " + " & ".join(_cell(df.loc[r, c], get_sd(r, c), fmt, marks.loc[r, c], "tex")
                                                         for c in cols) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabularx}" if wrap else "\\end{tabular}}"]
    if notes:
        tex.append(f"\\par\\vspace{{2pt}}{{\\footnotesize {_latex_escape(notes)}}}")
    tex.append("\\end{table}")
    stem.with_suffix(".tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return [stem.with_suffix(s) for s in (".csv", ".xlsx", ".md", ".tex")]
