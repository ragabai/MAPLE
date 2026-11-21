"""Evaluation suites: in-distribution (ID) and structurally out-of-distribution (OOD) task sets, plus the
conventional random-MLP/SCM prior used by baseline P0.

Suite tasks use the RNG namespace ("suite", name, i) which never overlaps training streams ("train", ...).
"""
from __future__ import annotations

import numpy as np

from maple.priors.features import apply_missing
from maple.priors.families import ood_families
from maple.tasks.generator import Task, TaskGenerator, cells, default_generator_config
from maple.tasks.preprocess import preprocess_arrays
from maple.utils import rng as mk_rng

SUITE_N_CTX = (32, 64, 128, 256, 512)
SUITE_D = (2, 16)
SUITE_N_QRY = 128

SUITES: dict[str, dict] = {
    "id": {"description": "training families, fresh draws"},
    "ood_families": {"description": "held-out families: Gompertz, SIR, hyperbolic"},
    "ood_nested": {"description": "nested compositions g(h(x)); training uses additive compositions only"},
    "ood_params": {"description": "training families with parameter ranges stretched x2.5"},
    "ood_interact3": {"description": "three-way interactions; training has at most pairwise"},
    "ood_voronoi": {"description": "piecewise-linear functions on Voronoi regions"},
    "ood_noise": {"description": "heavy-tailed Student-t (df=2.5) and Laplace noise"},
    "ood_mixed": {"description": "soft (softmax-gated) mixtures of 2-4 regimes"},
}


def _suite_generator(name: str) -> tuple[TaskGenerator, list, tuple | None]:
    cfg = default_generator_config()
    noise = None
    if name == "id":
        cs = cells(cfg)
    elif name == "ood_families":
        cs = [(f, c) for f in ood_families() for c in (0, 1, 2)]
    elif name == "ood_nested":
        cfg["composition"] = "nested"
        cs = cells(cfg)
    elif name == "ood_params":
        cfg["param_scale"] = 2.5
        cs = cells(cfg)
    elif name == "ood_interact3":
        cs = [("interaction3", c) for c in (0, 1, 2)]
    elif name == "ood_voronoi":
        cs = [("voronoi", c) for c in (0, 1, 2)]
    elif name == "ood_noise":
        cs = cells(cfg)
        noise = ("student_t", "laplace")
    elif name == "ood_mixed":
        cs = [("mixed_regime", c) for c in (0, 1, 2)]
    else:
        raise KeyError(name)
    return TaskGenerator(cfg), cs, noise


def make_suite(name: str, n_tasks: int, seed: int) -> list[Task]:
    gen, cs, noise = _suite_generator(name)
    out = []
    for i in range(n_tasks):
        r = mk_rng(seed, "suite", name, i)
        cell = cs[int(r.integers(len(cs)))]
        n_ctx = int(r.choice(SUITE_N_CTX))
        d = int(r.integers(SUITE_D[0], SUITE_D[1] + 1))
        t = gen.sample(r, cell, n_ctx, SUITE_N_QRY, d, noise_types=noise)
        t.meta["suite"] = name
        t.meta["task_index"] = i
        out.append(t)
    return out


# ------------------------------------------------------------------ conventional prior (baseline P0)

_ACTS = (np.tanh, lambda z: np.maximum(z, 0), np.sin, lambda z: z)


def random_mlp_task(rng: np.random.Generator, n_ctx: int, n_qry: int, d: int, max_attempts: int = 20) -> Task:
    """TabPFN-v1-style SCM prior: a random MLP is a causal graph; features and the target are random nodes."""
    n = n_ctx + n_qry
    for attempt in range(max_attempts):
        n_layers = int(rng.integers(2, 5))
        width = int(rng.integers(max(8, d + 1), 65))
        n_cause = int(rng.integers(2, 11))
        act = _ACTS[int(rng.integers(len(_ACTS)))]
        sig_node = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
        h = rng.normal(size=(n, n_cause)) if rng.random() < 0.5 else rng.uniform(-1.7, 1.7, size=(n, n_cause))
        nodes = []
        for _ in range(n_layers):
            W = rng.normal(size=(h.shape[1], width)) / np.sqrt(h.shape[1])
            W *= rng.random(W.shape) < rng.uniform(0.3, 1.0)  # sparse edges
            h = act(h @ W + rng.normal(size=width) * 0.5) + rng.normal(size=(n, width)) * sig_node
            nodes.append(h)
        allnodes = np.concatenate(nodes, 1)
        sd = allnodes.std(0)
        ok = np.where(sd > 1e-6)[0]
        last_ok = [j for j in ok if j >= allnodes.shape[1] - width]
        if len(ok) < d + 1 or not last_ok:
            continue
        tgt = int(rng.choice(last_ok))
        feats = rng.choice([j for j in ok if j != tgt], size=d, replace=len(ok) - 1 < d)
        X = allnodes[:, feats]
        f = allnodes[:, tgt]
        f = (f - f.mean()) / f.std()
        if not np.isfinite(f).all() or (f.max() - f.min()) > 25:
            continue
        y = f + rng.normal(size=n) * float(np.exp(rng.uniform(np.log(0.01), np.log(0.5))))
        if rng.random() < 0.3:  # discretise a few features
            for j in rng.choice(d, size=int(rng.integers(1, min(3, d) + 1)), replace=False):
                X[:, j] = np.digitize(X[:, j], np.quantile(X[:, j], np.sort(rng.uniform(0.05, 0.95, size=int(rng.integers(1, 7))))))
        Xm, miss = apply_missing(rng, X, {"p_missing": 0.2})
        Xc, Xq = preprocess_arrays(Xm[:n_ctx], Xm[n_ctx:])
        ym, ys = y[:n_ctx].mean(), y[:n_ctx].std()
        if ys < 1e-8 or np.abs((y - ym) / ys).max() > 12.0 or np.abs((f[n_ctx:] - ym) / ys).max() > 12.0:
            continue
        meta = {"cell": ("random_mlp", 0), "family": "random_mlp", "complexity": 0, "d": d, "n_ctx": n_ctx,
                "n_qry": n_qry, "n_layers": n_layers, "width": width, "missing_rate": miss,
                "composition": "scm", "noise": {"type": "gaussian"}}
        return Task(Xc, (y[:n_ctx] - ym) / ys, Xq, (y[n_ctx:] - ym) / ys, (f[n_ctx:] - ym) / ys, meta)
    raise RuntimeError("random_mlp_task: no valid task")
