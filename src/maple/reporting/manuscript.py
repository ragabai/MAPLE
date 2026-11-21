"""Manuscript generation: one Jinja2 Markdown source -> manuscript.md and an MDPI LaTeX manuscript.tex.

Numbers enter only through the ``S`` (summary) context built from results/statistics/statistics.json.
Raw LaTeX in tex mode is wrapped in ⟦ ⟧ and passed through unescaped.
"""
from __future__ import annotations

import re
from pathlib import Path

RAW_OPEN, RAW_CLOSE = "⟦", "⟧"
_ESC = [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"), ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}")]
_UNI = {"≤": "$\\le$", "≥": "$\\ge$", "±": "$\\pm$", "×": "$\\times$", "→": "$\\rightarrow$", "≈": "$\\approx$",
        "−": "--", "–": "--", "—": "---", "τ": "$\\tau$", "γ": "$\\gamma$", "λ": "$\\lambda$", "β": "$\\beta$",
        "κ": "$\\kappa$", "θ": "$\\theta$", "Δ": "$\\Delta$", "ρ": "$\\rho$", "χ": "$\\chi$", "∝": "$\\propto$",
        "‖": "$\\|$", "∈": "$\\in$", "²": "$^2$", "δ": "$\\delta$", "σ": "$\\sigma$", "μ": "$\\mu$", "ε": "$\\varepsilon$"}


def _esc_text(s: str) -> str:
    for a, b in _ESC:
        s = s.replace(a, b)
    for a, b in _UNI.items():
        s = s.replace(a, b)
    return s


def _inline(s: str) -> str:
    """Convert one line of Markdown inline syntax to LaTeX, protecting math and raw blocks."""
    keep: list[str] = []

    def stash(x: str) -> str:
        keep.append(x)
        return f"\x00{len(keep) - 1}\x00"

    s = re.sub(re.escape(RAW_OPEN) + r"(.*?)" + re.escape(RAW_CLOSE), lambda m: stash(m.group(1)), s)
    s = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", lambda m: stash(f"${m.group(1)}$"), s)
    s = re.sub(r"\[((?:@[\w.:-]+(?:;\s*)?)+)\]",
               lambda m: stash("\\cite{" + ",".join(k.strip()[1:] for k in m.group(1).split(";")) + "}"), s)
    s = re.sub(r"`([^`]+)`", lambda m: stash("\\texttt{" + _esc_text(m.group(1)) + "}"), s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", lambda m: stash("\\href{" + m.group(2) + "}{" + _esc_text(m.group(1)) + "}"), s)
    s = re.sub(r"(https?://[^\s)]+)", lambda m: stash("\\url{" + m.group(1) + "}"), s)
    bold = re.findall(r"\*\*(.+?)\*\*", s)
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: stash("\\textbf{" + _esc_text(m.group(1)) + "}"), s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", lambda m: stash("\\emph{" + _esc_text(m.group(1)) + "}"), s)
    s = _esc_text(s)
    for _ in range(3):  # nested stashes
        s = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)
    del bold
    return s


def _is_block_start(ln: str) -> bool:
    st = ln.strip()
    return (st.startswith("$$") or bool(re.match(r"^#{2,4}\s", ln)) or bool(re.match(r"^\s*[-*]\s+", ln))
            or bool(re.match(r"^\s*\d+\.\s+", ln)))


def md_to_tex(md: str) -> str:
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    env = None
    while i < len(lines):
        ln = lines[i]
        st = ln.strip()
        if st.startswith("$$"):
            body = []
            if st != "$$" and st.endswith("$$") and len(st) > 4:
                body = [st[2:-2]]
            else:
                i += 1
                while i < len(lines) and lines[i].strip() != "$$":
                    body.append(lines[i])
                    i += 1
            if env:
                out.append(f"\\end{{{env}}}")
                env = None
            out += ["\\begin{equation}", *body, "\\end{equation}"]
            i += 1
            continue
        m = re.match(r"^(#{2,4})\s+(?:[\dA-Z]+(?:\.\d+)*\.?\s+)?(.*)$", ln)
        if m:
            if env:
                out.append(f"\\end{{{env}}}")
                env = None
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[len(m.group(1))]
            out.append(f"\\{cmd}{{{_inline(m.group(2).strip())}}}")
            i += 1
            continue
        mi = re.match(r"^\s*[-*]\s+(.*)$", ln)
        me = re.match(r"^\s*\d+\.\s+(.*)$", ln)
        if mi or me:
            want = "itemize" if mi else "enumerate"
            if env != want:
                if env:
                    out.append(f"\\end{{{env}}}")
                out.append(f"\\begin{{{want}}}")
                env = want
            item = [(mi or me).group(1)]
            i += 1
            while i < len(lines) and lines[i].strip() and lines[i][:1].isspace() and not _is_block_start(lines[i]):
                item.append(lines[i].strip())  # indented continuation of the same item
                i += 1
            out.append("\\item " + _inline(" ".join(item)))
            continue
        if env and st == "":
            out.append(f"\\end{{{env}}}")
            env = None
        if not st:
            out.append("")
            i += 1
            continue
        para = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() and not _is_block_start(lines[i]):
            para.append(lines[i])  # soft line breaks: inline markup may span them
            i += 1
        out.append(_inline(" ".join(p.strip() for p in para)))
    if env:
        out.append(f"\\end{{{env}}}")
    return "\n".join(out) + "\n"


def validate_markdown(md: str, bib_keys: set[str], tables: set[int], figures: set[int]) -> list[str]:
    problems = []
    cited = set()
    for grp in re.findall(r"\[((?:@[\w.:-]+(?:;\s*)?)+)\]", md):
        cited |= {k.strip()[1:] for k in grp.split(";")}
    for k in sorted(cited - bib_keys):
        problems.append(f"citation key not in references.bib: {k}")
    for left in re.findall(r"\{\{.*?\}\}|\{%.*?%\}", md):
        problems.append(f"unrendered template expression: {left}")
    if "TODO" in md:
        problems.append("TODO marker present")
    ref_t = {int(n) for n in re.findall(r"Table[s]?\s+(\d+)", md)}
    ref_f = {int(n) for n in re.findall(r"Figure[s]?\s+(\d+)", md)}
    for n in sorted(ref_t - tables):
        problems.append(f"Table {n} referenced but not generated")
    for n in sorted(ref_f - figures):
        problems.append(f"Figure {n} referenced but not generated")
    for n in sorted(figures - ref_f):
        problems.append(f"Figure {n} generated but never referenced in the text")
    for n in sorted(tables - ref_t):
        problems.append(f"Table {n} generated but never referenced in the text")
    return problems


def bib_keys(path: Path) -> set[str]:
    return set(re.findall(r"^@\w+\{([^,]+),", Path(path).read_text(encoding="utf-8"), re.M))
