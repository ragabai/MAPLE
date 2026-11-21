"""Deterministic, persisted train/test splits shared by all methods.

A regime ``n_train`` is feasible only if at least ``MIN_TEST`` rows remain for testing; otherwise ``None`` is
returned and the caller logs the skip. Test sets are capped at ``MAX_TEST`` rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from maple.utils import atomic_write_text, rng as mk_rng

MIN_TEST = 200
MAX_TEST = 2000


def make_splits(name: str, n: int, n_train: int, repeats: int, seed: int,
                cache_dir: str | Path | None = None) -> list[tuple[np.ndarray, np.ndarray]] | None:
    if n - n_train < MIN_TEST:
        return None
    n_test = min(MAX_TEST, n - n_train)
    splits = []
    for r in range(repeats):
        perm = mk_rng(seed, "split", name, n_train, r).permutation(n)
        splits.append((np.sort(perm[:n_train]), np.sort(perm[n_train:n_train + n_test])))
    if cache_dir is not None:
        p = Path(cache_dir) / f"{name}__n{n_train}__s{seed}.json"
        payload = {"name": name, "n": n, "n_train": n_train, "seed": seed,
                   "splits": [[tr.tolist(), te.tolist()] for tr, te in splits]}
        if p.exists():
            old = json.loads(p.read_text())
            if old["n"] != n:
                raise ValueError(f"cached splits {p} were made for n={old['n']}, not n={n}")
            cached = [(np.array(a), np.array(b)) for a, b in old["splits"]]
            for (a, b), (c, d) in zip(cached, splits):
                if not (np.array_equal(a, c) and np.array_equal(b, d)):
                    raise ValueError(f"cached splits {p} differ from regenerated ones")
            if len(cached) >= repeats:
                return cached[:repeats]
            # fewer cached repeats: the regenerated list has an identical prefix (checked above) -> extend the cache
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(p, json.dumps(payload))
    return splits
