"""Prior-data fitted network for regression: a transformer over (context, query) points.

Every token attends only to context tokens (keys/values come from the context), so query predictions are
independent of each other and invariant to the order of context points. No positional encodings are used.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dropout = dropout

    def forward(self, h: torch.Tensor, n_ctx: int) -> torch.Tensor:
        B, T, D = h.shape
        a = self.ln1(h)
        q, k, v = self.qkv(a).chunk(3, dim=-1)
        k, v = k[:, :n_ctx], v[:, :n_ctx]
        H = self.n_heads

        def split(x):
            return x.view(B, x.shape[1], H, D // H).transpose(1, 2)
        att = F.scaled_dot_product_attention(split(q), split(k), split(v),
                                             dropout_p=self.dropout if self.training else 0.0)
        h = h + self.out(att.transpose(1, 2).reshape(B, T, D))
        return h + self.ff(self.ln2(h))


class PFN(nn.Module):
    def __init__(self, d_max: int = 32, d_model: int = 256, n_layers: int = 6, n_heads: int = 8, d_ff: int = 768,
                 dropout: float = 0.0):
        super().__init__()
        self.d_max = d_max
        self.config = {"d_max": d_max, "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads, "d_ff": d_ff,
                       "dropout": dropout}
        self.x_emb = nn.Linear(d_max, d_model)
        self.y_emb = nn.Linear(1, d_model)
        self.qry_tag = nn.Parameter(torch.zeros(d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def _pad(self, X: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] > self.d_max:
            raise ValueError(f"{X.shape[-1]} features > d_max={self.d_max}")
        X = F.pad(X, (0, self.d_max - X.shape[-1]))
        scale = torch.sqrt(self.d_max / d.to(X.dtype).clamp(min=1)).view(-1, 1, 1)
        return X * scale

    def forward(self, X_ctx: torch.Tensor, y_ctx: torch.Tensor, X_qry: torch.Tensor,
                d: torch.Tensor | None = None) -> torch.Tensor:
        B, N, _ = X_ctx.shape
        if d is None:
            d = torch.full((B,), X_ctx.shape[-1], device=X_ctx.device)
        ctx = self.x_emb(self._pad(X_ctx, d)) + self.y_emb(y_ctx.unsqueeze(-1))
        qry = self.x_emb(self._pad(X_qry, d)) + self.qry_tag
        h = torch.cat([ctx, qry], dim=1)
        for blk in self.blocks:
            h = blk(h, N)
        return self.head(self.ln_f(h[:, N:])).squeeze(-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def cosine_lr(step: int, total: int, warmup: int, base: float, min_frac: float = 0.02) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))
