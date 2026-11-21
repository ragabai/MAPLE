"""Adaptive cell sampler: q = (1 - gamma) * Gibbs(p0, s / tau) + gamma * uniform  (theory.md, Props. 1-3)."""
from __future__ import annotations

from typing import Literal

import numpy as np

Mode = Literal["uniform", "loss", "disagreement", "ambiguity", "pairwise", "combined"]
MODES = ("uniform", "loss", "disagreement", "ambiguity", "pairwise", "combined")


def gibbs(p0: np.ndarray, s: np.ndarray, tau: float) -> np.ndarray:
    """Numerically stable q_c ∝ p0_c exp(s_c / tau)."""
    z = np.log(p0) + np.asarray(s, dtype=float) / tau
    z -= z.max()
    q = np.exp(z)
    return q / q.sum()


class CellSampler:
    def __init__(self, cells: list, mode: Mode, tau: float, gamma: float, ema: float = 0.5, clip_B: float = 4.0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not (tau > 0 and 0 <= gamma < 1 and 0 <= ema < 1):
            raise ValueError("need tau > 0, 0 <= gamma < 1, 0 <= ema < 1")
        self.cells = [tuple(c) for c in cells]
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.mode, self.tau, self.gamma, self.ema, self.clip_B = mode, float(tau), float(gamma), float(ema), float(clip_B)
        M = len(self.cells)
        self.p0 = np.full(M, 1.0 / M)
        self.scores = np.zeros(M)
        self.seen = np.zeros(M, dtype=bool)
        self.history: list[dict] = []
        self._q = self.p0.copy()

    def probs(self) -> np.ndarray:
        return self._q.copy()

    def sample(self, rng: np.random.Generator) -> tuple:
        return self.cells[int(rng.choice(len(self.cells), p=self._q))]

    def update(self, scores: dict, step: int | None = None) -> None:
        for c, v in scores.items():
            i = self.index[tuple(c)]
            v = float(np.clip(v, 0.0, self.clip_B))
            self.scores[i] = v if not self.seen[i] else self.ema * self.scores[i] + (1 - self.ema) * v
            self.seen[i] = True
        if self.mode == "uniform":
            self._q = self.p0.copy()
        else:
            self._q = (1 - self.gamma) * gibbs(self.p0, self.scores, self.tau) + self.gamma / len(self.cells)
        self.history.append({"step": step, "probs": self._q.tolist(), "scores": self.scores.tolist(),
                             "raw": {f"{c[0]}|{c[1]}": float(v) for c, v in scores.items()}})

    def state_dict(self) -> dict:
        return {"cells": [list(c) for c in self.cells], "mode": self.mode, "tau": self.tau, "gamma": self.gamma,
                "ema": self.ema, "clip_B": self.clip_B, "scores": self.scores.tolist(), "seen": self.seen.tolist(),
                "q": self._q.tolist(), "history": self.history}

    def load_state_dict(self, st: dict) -> None:
        if [tuple(c) for c in st["cells"]] != self.cells:
            raise ValueError("cell set mismatch")
        self.scores = np.asarray(st["scores"], dtype=float)
        self.seen = np.asarray(st["seen"], dtype=bool)
        self._q = np.asarray(st["q"], dtype=float)
        self.history = list(st["history"])
