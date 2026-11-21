"""Formatting helpers available inside manuscript templates (numbers always come from the summary context)."""
from __future__ import annotations

import math


def _get(d, path: str, default=None):
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, dict) and k.isdigit() and int(k) in cur:
            cur = cur[int(k)]
        else:
            return default
    return cur


def helpers(S: dict) -> dict:
    def get(path, default=None):
        return _get(S, path, default)

    def num(x, d=3):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"

    def pct(x, d=1, signed=False):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        return f"{100 * x:+.{d}f}%" if signed else f"{100 * x:.{d}f}%"

    def pval(p):
        if p is None or (isinstance(p, float) and math.isnan(p)):
            return "n/a"
        return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"

    def gain(rec, d=1):
        """Median relative NRMSE difference of a _pair() record; positive = first-named method has lower error."""
        return pct(rec.get("median_diff") if rec else None, d)

    def reduction(rec, d=1):
        """'a reduced b's error by X' for a _pair(a, b) record: median of (b - a)/b = m/(1+m), m = median((b-a)/a)
        (the map r -> r/(1+r) is monotone, so it commutes with the median)."""
        if not rec or rec.get("median_diff") is None:
            return "n/a"
        m = rec["median_diff"]
        return pct(m / (1 + m), d)

    def negligible(rec):
        """|median relative difference| below the pre-registered 1% practical-significance threshold."""
        return abs(rec.get("median_diff", 0.0)) < 0.01

    def stat(rec, flip=False):
        """W/L/T, p and r_rb; flip=True orients them to the record's SECOND method (when the sentence names it first)."""
        if not rec:
            return "n/a"
        if flip:
            rec = {**rec, "wins": rec["losses"], "losses": rec["wins"], "r_rb": -rec["r_rb"]}
        if "p_holm_family" in rec:
            p = rec["p_holm_family"]
            ptxt = "$p_{\\mathrm{Holm}}$ < 0.001" if p < 1e-3 else f"$p_{{\\mathrm{{Holm}}}}$ = {p:.3f}"
        else:
            p = rec["p"]
            ptxt = "raw p < 0.001" if p < 1e-3 else f"raw p = {p:.3f}"
        return f"W/L/T = {rec['wins']}/{rec['losses']}/{rec['ties']}, {ptxt}, $r_{{rb}}$ = {rec['r_rb']:.2f}"

    def mean(d):
        vals = [v for v in (d.values() if isinstance(d, dict) else d) if v is not None and not
                (isinstance(v, float) and math.isnan(v))]
        return sum(vals) / len(vals) if vals else float("nan")

    def thm1(B, tau, gamma, M, m, delta):
        """Theorem 1: with prob. >= 1 - delta, ||q(s_hat) - q(s)||_1 <= (1-gamma)(B/tau) sqrt(log(2M/delta)/(2m))."""
        return (1 - gamma) * (B / tau) * math.sqrt(math.log(2 * M / delta) / (2 * m))

    def thm1_m(B, tau, gamma, M, delta, target):
        """Smallest number of probe tasks per cell m for which the Theorem 1 bound is <= target."""
        m = math.ceil(((1 - gamma) * B / (tau * target)) ** 2 * math.log(2 * M / delta) / 2)
        while thm1(B, tau, gamma, M, m, delta) > target:
            m += 1
        while m > 1 and thm1(B, tau, gamma, M, m - 1, delta) <= target:
            m -= 1
        return m

    return {"get": get, "num": num, "pct": pct, "pval": pval, "gain": gain, "stat": stat, "mean": mean,
            "reduction": reduction, "negligible": negligible,
            "thm1": thm1, "thm1_m": thm1_m}


def load_tables(table_dir) -> dict:
    """Flat result tables (results/tables/*.csv) as {name: {column: {row: value}}}; tables written with a
    value/SD two-level header are skipped (their means are available from the statistics files)."""
    from pathlib import Path

    import pandas as pd
    out = {}
    for p in sorted(Path(table_dir).glob("*.csv")):
        df = pd.read_csv(p, index_col=0)
        if {str(c).split(".")[0] for c in df.columns} <= {"value", "sd"}:
            continue
        out[p.stem] = df.to_dict()
    return out
