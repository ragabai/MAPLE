"""Library of mathematical function families used as prior components.

Each family maps standardised inputs ``X`` of shape (n, d_in) to a target of shape (n,). Families draw their own
parameters from ``rng``; ``complexity`` in {0, 1, 2} controls the number of terms, curvature and interaction
order; ``param_scale`` stretches frequency/rate-like parameters (values > 1 are used for out-of-distribution
"extrapolated parameter" suites).

Split tags:
  * ``train``          — available to the pretraining prior.
  * ``ood``            — held-out families (never seen in training): gompertz, sir, hyperbolic.
  * ``ood_structure``  — structures used only by OOD suites: 3-way interactions, Voronoi regions, soft mixed regimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Array = np.ndarray
EXP_CLIP = 30.0


@dataclass
class Component:
    f: Callable[[Array], Array]
    meta: dict = field(default_factory=dict)


@dataclass
class Family:
    name: str
    split: str
    sampler: Callable[[np.random.Generator, int, int, float], tuple[Callable[[Array], Array], dict]]
    description: str = ""

    def sample(self, rng: np.random.Generator, d_in: int, complexity: int, param_scale: float = 1.0) -> Component:
        if d_in < 1:
            raise ValueError("d_in must be >= 1")
        if complexity not in (0, 1, 2):
            raise ValueError("complexity must be 0, 1 or 2")
        f, params = self.sampler(rng, d_in, complexity, float(param_scale))
        return Component(f=f, meta={"family": self.name, "complexity": complexity, "param_scale": param_scale,
                                    "params": params})


# ----------------------------------------------------------------------------- helpers

def _dir(rng: np.random.Generator, d_in: int, max_support: int = 3) -> Array:
    """Sparse unit direction: a random linear combination of 1..max_support input features."""
    m = int(rng.integers(1, min(d_in, max_support) + 1))
    idx = rng.choice(d_in, size=m, replace=False)
    w = np.zeros(d_in)
    w[idx] = rng.normal(size=m)
    n = np.linalg.norm(w)
    if n < 1e-8:
        w[idx[0]] = 1.0
        n = 1.0
    return w / n


def _pos(z: Array, lo: float, hi: float) -> Array:
    """Smooth map R -> (lo, hi) via the logistic function."""
    return lo + (hi - lo) / (1.0 + np.exp(-np.clip(z, -EXP_CLIP, EXP_CLIP)))


def _exp(z: Array) -> Array:
    return np.exp(np.clip(z, -EXP_CLIP, EXP_CLIP))


def _terms(c: int, table=(1, 2, 3)) -> int:
    return table[c]


def _rk4(deriv: Callable[[Array], Array], s0: Array, T: Array, steps: int) -> Array:
    """Vectorised fixed-step RK4; ``s0`` (n, k), per-sample horizon ``T`` (n,)."""
    h = (T / steps)[:, None]
    s = s0.copy()
    for _ in range(steps):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * h * k1)
        k3 = deriv(s + 0.5 * h * k2)
        k4 = deriv(s + h * k3)
        s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s = np.clip(s, -1e6, 1e6)
    return s


# ----------------------------------------------------------------------------- linear / polynomial

def _linear(rng, d, c, s):
    w = rng.normal(size=d)
    b = rng.normal()
    return (lambda X: X @ w + b), {"w": w.round(3).tolist()}


def _polynomial(rng, d, c, s):
    n_terms = (2, 4, 7)[c]
    deg_max = min(5, (2, 3, 3)[c] + max(0, int(round(s)) - 1))
    terms = []
    for _ in range(n_terms):
        deg = int(rng.integers(1, deg_max + 1))
        vars_ = rng.integers(0, d, size=deg)
        terms.append((vars_, rng.normal()))

    def f(X):
        out = np.zeros(X.shape[0])
        for vars_, a in terms:
            out += a * np.prod(X[:, vars_], axis=1)
        return out
    return f, {"terms": [(v.tolist(), round(a, 3)) for v, a in terms], "deg_max": deg_max}


# ----------------------------------------------------------------------------- nonlinear

def _exponential(rng, d, c, s):
    k = _terms(c)
    dirs = [_dir(rng, d) for _ in range(k)]
    b = rng.uniform(0.3, 1.0, size=k) * s * (1 + 0.5 * c) * rng.choice([-1, 1], size=k)
    a = rng.normal(size=k)

    def f(X):
        return sum(a[j] * _exp(b[j] * (X @ dirs[j])) for j in range(k))
    return f, {"rates": b.round(3).tolist(), "kind": "growth/decay"}


def _logarithmic(rng, d, c, s):
    k = _terms(c)
    dirs = [_dir(rng, d) for _ in range(k)]
    eps = 10 ** rng.uniform(-2, -0.5, size=k) / s
    a = rng.normal(size=k)
    hi = rng.uniform(3, 20, size=k)

    def f(X):
        return sum(a[j] * np.log(_pos(s * (X @ dirs[j]), eps[j], hi[j])) for j in range(k))
    return f, {"eps": eps.round(4).tolist()}


def _power_law(rng, d, c, s):
    k = min(d, _terms(c))
    dirs = [_dir(rng, d, 1) for _ in range(k)]
    p = rng.uniform(-2, 2, size=k) * s
    a = rng.choice([-1, 1]) * rng.uniform(0.5, 2)

    def f(X):
        out = np.full(X.shape[0], a)
        for j in range(k):
            out = out * _pos(X @ dirs[j], 0.1, 5.0) ** p[j]
        return out
    return f, {"powers": p.round(3).tolist()}


def _rational(rng, d, c, s):
    variant = int(rng.integers(0, 2))
    d1, d2 = _dir(rng, d), _dir(rng, d)
    a, b = rng.normal(size=2)
    cc = rng.uniform(0.5, 3) * s
    delta = rng.uniform(0.2, 1.0)
    z0 = rng.normal(0, 0.7)
    k = _terms(c)
    extra = [(_dir(rng, d), rng.normal(), rng.normal(0, 0.7)) for _ in range(k - 1)]

    def f(X):
        z1, z2 = X @ d1, X @ d2
        out = (a * z1 + b) / (1 + cc * z2 ** 2) if variant == 0 else a / (delta + cc * (z1 - z0) ** 2)
        for w, e, t in extra:
            out = out + e / (delta + cc * (X @ w - t) ** 2)
        return out
    return f, {"variant": variant, "c": round(cc, 3)}


def _trigonometric(rng, d, c, s):
    k = _terms(c)
    dirs = [_dir(rng, d) for _ in range(k)]
    om = rng.uniform(0.5, 2.0, size=k) * (1 + c) * s
    ph = rng.uniform(0, 2 * np.pi, size=k)
    a = rng.normal(size=k)
    use_cos = rng.random(k) < 0.5

    def f(X):
        out = np.zeros(X.shape[0])
        for j in range(k):
            arg = om[j] * (X @ dirs[j]) + ph[j]
            out += a[j] * (np.cos(arg) if use_cos[j] else np.sin(arg))
        return out
    return f, {"omega": om.round(3).tolist()}


def _rbf(rng, d, c, s):
    K = (1, 3, 6)[c]
    m = min(d, (1, 2, 3)[c])
    idx = rng.choice(d, size=m, replace=False)
    centers = rng.normal(0, 1, size=(K, m))
    widths = rng.uniform(0.3, 1.2, size=K) / s
    amps = rng.normal(size=K)

    def f(X):
        Z = X[:, idx]
        d2 = ((Z[:, None, :] - centers[None]) ** 2).sum(-1)
        return (amps[None] * np.exp(-d2 / (2 * widths[None] ** 2))).sum(1)
    return f, {"n_bumps": K, "widths": widths.round(3).tolist()}


# ----------------------------------------------------------------------------- saturation / growth

def _logistic(rng, d, c, s):
    k = _terms(c)
    dirs = [_dir(rng, d) for _ in range(k)]
    L = rng.normal(size=k)
    kk = rng.uniform(1, 5, size=k) * s
    z0 = rng.normal(0, 0.7, size=k)

    def f(X):
        return sum(L[j] / (1 + _exp(-kk[j] * (X @ dirs[j] - z0[j]))) for j in range(k))
    return f, {"steepness": kk.round(3).tolist()}


def _michaelis_menten(rng, d, c, s):
    k = _terms(c)
    dirs = [_dir(rng, d) for _ in range(k)]
    vmax = rng.normal(size=k)
    km = rng.uniform(0.2, 5, size=k) / s

    def f(X):
        out = np.zeros(X.shape[0])
        for j in range(k):
            u = _pos(1.5 * (X @ dirs[j]), 0.0, 10.0)
            out += vmax[j] * u / (km[j] + u)
        return out
    return f, {"Km": km.round(3).tolist()}


def _piecewise(rng, d, c, s):
    k = (1, 2, 4)[c]
    dirs = [_dir(rng, d) for _ in range(k)]
    t = rng.normal(0, 0.8, size=k)
    a = rng.normal(size=k) * s
    base = rng.normal(size=d) * 0.3

    def f(X):
        out = X @ base
        for j in range(k):
            out = out + a[j] * np.maximum(0.0, X @ dirs[j] - t[j])
        return out
    return f, {"knots": t.round(3).tolist()}


# ----------------------------------------------------------------------------- physical relationships

def _physics(rng, d, c, s):
    kind = ["inverse_square", "energy", "multiplicative"][int(rng.integers(0, 3))]
    cols = [rng.integers(0, d) for _ in range(4)]
    g = rng.uniform(0.5, 2.0) * s

    def u(X, j, lo, hi):
        return _pos(X[:, cols[j]], lo, hi)

    if kind == "inverse_square":
        def f(X):
            return g * u(X, 0, 0.5, 5) * u(X, 1, 0.5, 5) / u(X, 2, 0.5, 3) ** 2
    elif kind == "energy":
        def f(X):  # kinetic + potential energy: 1/2 m v^2 + m g h
            m = u(X, 0, 0.5, 5)
            return 0.5 * m * (X[:, cols[1]] * s) ** 2 + m * g * u(X, 2, 0, 10)
    else:
        p = rng.uniform(0.5, 1.5, size=3)

        def f(X):
            return g * u(X, 0, 0.2, 4) ** p[0] * u(X, 1, 0.2, 4) ** p[1] * u(X, 3, 0.2, 4) ** p[2]
    return f, {"kind": kind}


# ----------------------------------------------------------------------------- dynamical systems

def _damped_oscillator(rng, d, c, s):
    om = rng.uniform(0.5, 2.0) * s * (1 + 0.5 * c)
    A = rng.normal()
    ph = rng.uniform(0, 2 * np.pi)
    ct, cz = rng.integers(0, d), rng.integers(0, d)
    zeta_const = rng.uniform(0.05, 0.6)

    def f(X):
        t = _pos(X[:, ct], 0.0, 8.0)
        zeta = _pos(X[:, cz], 0.05, 0.8) if (d > 1 and cz != ct) else zeta_const
        wd = om * np.sqrt(1 - zeta ** 2)
        return A * np.exp(-zeta * om * t) * np.cos(wd * t + ph)
    return f, {"omega": round(om, 3)}


def _logistic_ode(rng, d, c, s):
    """Closed-form solution of dN/dt = r N (1 - N/K)."""
    ct, cr, ck = rng.integers(0, d, size=3)
    r0 = rng.uniform(0.3, 2.0) * s
    n0 = rng.uniform(0.02, 0.3)
    use_r, use_k = c >= 1 and d > 1, c >= 2 and d > 2

    def f(X):
        t = _pos(X[:, ct], 0.0, 10.0)
        r = r0 * _pos(X[:, cr], 0.5, 1.5) if use_r else r0
        K = _pos(X[:, ck], 0.5, 2.0) if use_k else 1.0
        N0 = n0 * K
        return K / (1 + ((K - N0) / N0) * np.exp(-np.clip(r * t, 0, EXP_CLIP)))
    return f, {"r": round(r0, 3)}


def _lotka_volterra(rng, d, c, s):
    al, be, ga, de = rng.uniform(0.6, 1.4, size=4)
    cT, ca, cg = rng.integers(0, d, size=3)
    x0, y0 = rng.uniform(0.5, 2.0, size=2)
    steps = 40 + 20 * c

    def f(X):
        n = X.shape[0]
        T = _pos(X[:, cT], 0.5, 6.0 * s)
        a = al * _pos(X[:, ca], 0.5, 1.5)
        g = ga * _pos(X[:, cg], 0.5, 1.5) if c >= 1 else np.full(n, ga)

        def deriv(st):
            xx, yy = st[:, 0], st[:, 1]
            return np.stack([a * xx - be * xx * yy, de * xx * yy - g * yy], 1)
        s0 = np.tile([x0, y0], (n, 1))
        return _rk4(deriv, s0, T, steps)[:, 0]
    return f, {"alpha": round(al, 3), "beta": round(be, 3), "steps": steps}


# ----------------------------------------------------------------------------- complex structures

_SIMPLE = ("linear", "trigonometric", "polynomial", "logistic", "rbf")


def _conditional(rng, d, c, s):
    gdir = _dir(rng, d, 1)
    t = rng.normal(0, 0.5)
    sub_c = max(0, c - 1)
    f1 = FAMILIES[_SIMPLE[int(rng.integers(len(_SIMPLE)))]].sample(rng, d, sub_c, s).f
    f2 = FAMILIES[_SIMPLE[int(rng.integers(len(_SIMPLE)))]].sample(rng, d, sub_c, s).f

    def f(X):
        return np.where(X @ gdir > t, f1(X), f2(X))
    return f, {"threshold": round(t, 3)}


_G = (lambda z: z, np.sin, lambda z: z ** 2, lambda z: np.abs(z))


def _interaction(rng, d, c, s):
    k = (1, 2, 4)[c]
    pairs = [(int(rng.integers(d)), int(rng.integers(d)), int(rng.integers(len(_G))), int(rng.integers(len(_G))))
             for _ in range(k)]
    a = rng.normal(size=k)

    def f(X):
        return sum(a[j] * _G[g1](s * X[:, i]) * _G[g2](s * X[:, kk]) for j, (i, kk, g1, g2) in enumerate(pairs))
    return f, {"pairs": [p[:2] for p in pairs]}


def _discontinuous(rng, d, c, s):
    k = _terms(c)
    dirs = [_dir(rng, d, 1) for _ in range(k)]
    t = rng.normal(0, 0.8, size=k)
    h = rng.normal(size=k) * (1 + 0.5 * s)
    w = rng.normal(size=d) * 0.5

    def f(X):
        out = X @ w
        for j in range(k):
            out = out + h[j] * (X @ dirs[j] > t[j])
        return out
    return f, {"jumps": h.round(3).tolist()}


# ----------------------------------------------------------------------------- held-out (OOD) families

def _gompertz(rng, d, c, s):
    k = _terms(c)
    dirs = [_dir(rng, d) for _ in range(k)]
    a = rng.normal(size=k)
    b = rng.uniform(0.5, 3.0, size=k)
    kk = rng.uniform(0.5, 2.5, size=k) * s

    def f(X):
        return sum(a[j] * np.exp(-b[j] * _exp(-kk[j] * (X @ dirs[j]))) for j in range(k))
    return f, {"rates": kk.round(3).tolist()}


def _sir(rng, d, c, s):
    cb, cg, cT = rng.integers(0, d, size=3)
    i0 = rng.uniform(0.001, 0.02)
    steps = 60 + 20 * c

    def f(X):
        n = X.shape[0]
        beta = _pos(X[:, cb], 0.2, 2.5 * s)
        gamma = _pos(X[:, cg], 0.05, 0.8) if c >= 1 else np.full(n, 0.3)
        T = _pos(X[:, cT], 2.0, 30.0)

        def deriv(st):
            S, I = st[:, 0], st[:, 1]
            return np.stack([-beta * S * I, beta * S * I - gamma * I], 1)
        s0 = np.tile([1 - i0, i0], (n, 1))
        return _rk4(deriv, s0, T, steps)[:, 1]
    return f, {"i0": round(i0, 4), "steps": steps}


def _hyperbolic(rng, d, c, s):
    d1, d2 = _dir(rng, d), _dir(rng, d)
    a = rng.normal()
    b = rng.uniform(0.3, 1.0) * s
    cc = rng.uniform(0.3, 1.0) * s
    with_cosh = c >= 1
    e = rng.normal() * 0.5
    d3 = _dir(rng, d)

    def f(X):
        out = a * np.sinh(np.clip(b * (X @ d1), -8, 8)) / np.cosh(np.clip(cc * (X @ d2), -8, 8))
        if with_cosh:
            out = out + e * np.cosh(np.clip(0.7 * cc * (X @ d3), -8, 8))
        return out
    return f, {"b": round(b, 3)}


def _interaction3(rng, d, c, s):
    k = (1, 2, 3)[c]
    triples = [rng.integers(0, d, size=3) for _ in range(k)]
    gs = [rng.integers(0, len(_G), size=3) for _ in range(k)]
    a = rng.normal(size=k)

    def f(X):
        out = np.zeros(X.shape[0])
        for j in range(k):
            term = np.ones(X.shape[0])
            for v, g in zip(triples[j], gs[j]):
                term = term * _G[g](s * X[:, v])
            out += a[j] * term
        return out
    return f, {"order": 3, "n_terms": k}


def _voronoi(rng, d, c, s):
    K = (3, 5, 8)[c]
    m = min(d, (2, 3, 4)[c])
    idx = rng.choice(d, size=m, replace=False)
    centers = rng.normal(0, 1, size=(K, m))
    W = rng.normal(size=(K, d)) * 0.7
    b = rng.normal(size=K) * (1 + 0.5 * s)

    def f(X):
        Z = X[:, idx]
        region = ((Z[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        return (X * W[region]).sum(1) + b[region]
    return f, {"n_regions": K}


def _mixed_regime(rng, d, c, s):
    R = (2, 3, 4)[c]
    names = [_SIMPLE[int(i)] for i in rng.integers(0, len(_SIMPLE), size=R)] + ["exponential", "michaelis_menten"]
    names = names[:R]
    comps = [FAMILIES[nm].sample(rng, d, max(0, c - 1), s).f for nm in names]
    G = rng.normal(size=(R, d)) * rng.uniform(1.5, 4.0)

    def f(X):
        logits = X @ G.T
        logits -= logits.max(1, keepdims=True)
        w = np.exp(logits)
        w /= w.sum(1, keepdims=True)
        outs = np.stack([comp(X) for comp in comps], 1)
        sd = outs.std(0) + 1e-8
        return (w * (outs - outs.mean(0)) / sd).sum(1)
    return f, {"regimes": names}


# ----------------------------------------------------------------------------- registry

FAMILIES: dict[str, Family] = {}


def _reg(name, split, sampler, desc):
    FAMILIES[name] = Family(name, split, sampler, desc)


_reg("linear", "train", _linear, "w^T x + b")
_reg("polynomial", "train", _polynomial, "sparse multivariate polynomial with interactions")
_reg("exponential", "train", _exponential, "sum a exp(b z): growth and decay")
_reg("logarithmic", "train", _logarithmic, "sum a log(u), u in (eps, hi)")
_reg("power_law", "train", _power_law, "a prod u_j^p_j")
_reg("rational", "train", _rational, "(a z1 + b)/(1 + c z2^2) or Lorentzian peaks")
_reg("trigonometric", "train", _trigonometric, "sum a sin/cos(omega z + phi)")
_reg("rbf", "train", _rbf, "sum of Gaussian bumps")
_reg("logistic", "train", _logistic, "L/(1+exp(-k(z-z0)))")
_reg("michaelis_menten", "train", _michaelis_menten, "Vmax u/(Km+u)")
_reg("piecewise", "train", _piecewise, "hinge (piecewise-linear growth)")
_reg("physics", "train", _physics, "inverse-square / energy / multiplicative laws")
_reg("damped_oscillator", "train", _damped_oscillator, "A exp(-zeta omega t) cos(omega_d t + phi)")
_reg("logistic_ode", "train", _logistic_ode, "closed-form logistic ODE N(t)")
_reg("lotka_volterra", "train", _lotka_volterra, "RK4 predator-prey, prey at time T")
_reg("conditional", "train", _conditional, "hard two-regime switch on one feature")
_reg("interaction", "train", _interaction, "pairwise products of g(x_i) g(x_k)")
_reg("discontinuous", "train", _discontinuous, "linear trend plus step jumps")
_reg("gompertz", "ood", _gompertz, "a exp(-b exp(-k z))")
_reg("sir", "ood", _sir, "RK4 SIR model, infected fraction at time T")
_reg("hyperbolic", "ood", _hyperbolic, "a sinh(b z1)/cosh(c z2) (+ cosh term)")
_reg("interaction3", "ood_structure", _interaction3, "three-way products")
_reg("voronoi", "ood_structure", _voronoi, "piecewise-linear on Voronoi regions")
_reg("mixed_regime", "ood_structure", _mixed_regime, "softmax-gated mixture of regimes")


def train_families() -> list[str]:
    return [k for k, v in FAMILIES.items() if v.split == "train"]


def ood_families() -> list[str]:
    return [k for k, v in FAMILIES.items() if v.split == "ood"]
