"""PFN pretraining engine with adaptive (disagreement-steered) cell sampling.

Data generation and probe-committee fitting run in CPU worker processes; the PFN trains on the GPU. Committee
predictions on probe tasks do not depend on the PFN, so probe rounds are prepared ahead of time; at each
adaptation round only a PFN forward pass over the probe tasks is needed.

Restartable: ``out_dir/ckpt.pt`` (+ ``ckpt.opt.pt``) holds model, optimiser, sampler state, pending batch
specifications and the step counter. Batch contents depend only on (seed, step, spec), so a resumed run is
identical to an uninterrupted one on CPU.
"""
from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from maple.disagreement.measures import pfn_gap, task_score, variance
from maple.experts.registry import COMMITTEE_DEFAULT, fit_predict_committee
from maple.model.pfn import PFN, cosine_lr, count_params
from maple.sampling.sampler import CellSampler
from maple.tasks.generator import TaskGenerator, cells as gen_cells, default_generator_config
from maple.tasks.suites import random_mlp_task
from maple.utils import rng as mk_rng, stable_hash

_RUNTIME_KEYS = ("workers", "device", "stop_after", "prefetch", "log_every")


def default_train_config() -> dict:
    return {
        "seed": 0, "steps": 20000, "stop_after": None, "device": "auto", "workers": 8, "prefetch": 24,
        "prior": "math", "generator": default_generator_config(),
        "model": {"d_max": 32, "d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 768},
        "n_ctx_range": [16, 1024], "d_range": [1, 32], "n_qry": 128, "token_budget": 32768,
        "batch_range": [4, 64], "target": "f",
        "lr": 3e-4, "weight_decay": 0.01, "warmup": 500, "grad_clip": 1.0,
        "log_every": 50, "ckpt_every": 1000,
        "sampler": {"mode": "disagreement", "tau": 0.2, "gamma": 0.1, "ema": 0.5, "clip_B": 4.0,
                    "update_every": 1000, "probes_per_cell": 4, "probe_n_ctx": 128, "probe_n_qry": 128,
                    "committee": list(COMMITTEE_DEFAULT), "max_complexity": 2, "probe_always": True},
    }


# ----------------------------------------------------------------------------------------- worker side
_W: dict[str, Any] = {}


def _init_worker(gen_cfg: dict, prior: str) -> None:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:  # pragma: no cover
        pass
    _W["gen"] = TaskGenerator(gen_cfg)
    _W["prior"] = prior


def _make_task(r: np.random.Generator, cell, n_ctx: int, n_qry: int, d: int):
    if _W["prior"] == "scm":
        return random_mlp_task(r, n_ctx, n_qry, d)
    return _W["gen"].sample(r, tuple(cell), n_ctx, n_qry, d)


def gen_batch(spec: dict) -> dict:
    tasks = [_make_task(mk_rng(spec["seed"], "train", spec["step"], i), c, spec["n_ctx"], spec["n_qry"], spec["d"])
             for i, c in enumerate(spec["cells"])]
    st = lambda k: np.stack([getattr(t, k) for t in tasks]).astype(np.float32)  # noqa: E731
    return {"X_ctx": st("X_ctx"), "y_ctx": st("y_ctx"), "X_qry": st("X_qry"), "y_qry": st("y_qry"),
            "f_qry": st("f_qry"), "cells": spec["cells"]}


def gen_probe(spec: dict) -> list[dict]:
    out = []
    for i in range(spec["m"]):
        r = mk_rng(spec["seed"], "probe", spec["round"], spec["cell"][0], spec["cell"][1], i)
        d = int(r.integers(spec["d_range"][0], spec["d_range"][1] + 1))
        t = _make_task(r, spec["cell"], spec["n_ctx"], spec["n_qry"], d)
        P = fit_predict_committee(spec["committee"], t.X_ctx, t.y_ctx, t.X_qry, seed=spec["seed"] + i, probe=True)
        out.append({"cell": tuple(spec["cell"]), "d": d, "X_ctx": t.X_ctx.astype(np.float32),
                    "y_ctx": t.y_ctx.astype(np.float32), "X_qry": t.X_qry.astype(np.float32),
                    "y_qry": t.y_qry, "f_qry": t.f_qry, "P": P, "noise_rel": t.meta.get("noise", {}).get("rel")})
    return out


class _Inline:
    """Synchronous stand-in for a process pool (workers=0)."""

    def __init__(self, gen_cfg, prior):
        _init_worker(gen_cfg, prior)

    def submit(self, fn, *a):
        f: Future = Future()
        f.set_result(fn(*a))
        return f

    def shutdown(self, **kw):
        pass


# ----------------------------------------------------------------------------------------- helpers

def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _identity_cfg(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    for k in _RUNTIME_KEYS:
        c.pop(k, None)
    # max_attempts only decides whether generation fails, never the content of a valid task
    c.get("generator", {}).pop("max_attempts", None)
    return c


def _shape_spec(cfg: dict, step: int) -> tuple[int, int, int]:
    r = mk_rng(cfg["seed"], "shape", step)
    lo, hi = cfg["n_ctx_range"]
    n_ctx = int(round(np.exp(r.uniform(np.log(lo), np.log(hi)))))
    dlo, dhi = cfg["d_range"]
    d = int(min(dhi, max(dlo, round(np.exp(r.uniform(np.log(dlo), np.log(dhi + 1)) - 0.5)))))
    B = int(np.clip(cfg["token_budget"] // (n_ctx + cfg["n_qry"]), *cfg["batch_range"]))
    return n_ctx, d, B


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _append(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


@torch.no_grad()
def predict_tasks(model: PFN, items: list[dict], device: torch.device, chunk: int = 64) -> list[np.ndarray]:
    """Predict a list of same-shape tasks (dicts with X_ctx, y_ctx, X_qry, d) in padded chunks."""
    was_training = model.training
    model.eval()
    preds = []
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        dmax = max(p["X_ctx"].shape[1] for p in part)
        pad = lambda a: np.pad(a, ((0, 0), (0, dmax - a.shape[1])))  # noqa: E731
        Xc = torch.tensor(np.stack([pad(p["X_ctx"]) for p in part]), device=device)
        yc = torch.tensor(np.stack([p["y_ctx"] for p in part]), device=device)
        Xq = torch.tensor(np.stack([pad(p["X_qry"]) for p in part]), device=device)
        d = torch.tensor([p["d"] for p in part], device=device)
        with torch.autocast(device.type, dtype=_amp_dtype(device), enabled=device.type == "cuda"):
            out = model(Xc, yc, Xq, d=d).float().cpu().numpy()
        preds.extend(list(out))
    model.train(was_training)
    return preds


def _amp_dtype(device: torch.device):
    # Turing (sm_75) reports bf16 as "supported" via emulation, but fused attention kernels lack bf16 there.
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float16


# ----------------------------------------------------------------------------------------- main loop

def train(cfg: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "final.pt"
    if final.exists():
        return final
    cfg = copy.deepcopy(cfg)
    scfg = cfg["sampler"]
    device = _device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg["prior"] == "scm":
        cell_list = [("random_mlp", 0)]
        if scfg["mode"] != "uniform":
            raise ValueError("the SCM prior has a single cell; use sampler.mode='uniform'")
    else:
        cell_list = [c for c in gen_cells(cfg["generator"]) if c[1] <= scfg["max_complexity"]]
    sampler = CellSampler(cell_list, scfg["mode"], scfg["tau"], scfg["gamma"], scfg["ema"], scfg["clip_B"])
    model = PFN(**cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp and _amp_dtype(device) == torch.float16)

    step, pending = 0, {}
    ckpt = out_dir / "ckpt.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        if stable_hash(_identity_cfg(st["cfg_identity"])) != stable_hash(_identity_cfg(cfg)):
            raise ValueError(f"checkpoint in {out_dir} was produced by a different configuration")
        model.load_state_dict(st["model"])
        opt.load_state_dict(torch.load(out_dir / "ckpt.opt.pt", map_location="cpu", weights_only=False))
        sampler.load_state_dict(st["sampler"])
        if st.get("scaler") is not None:
            scaler.load_state_dict(st["scaler"])
        step, pending = st["step"], {int(k): v for k, v in st["pending"].items()}
    else:
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    pool = (_Inline(cfg["generator"], cfg["prior"]) if cfg["workers"] == 0 else
            ProcessPoolExecutor(cfg["workers"], initializer=_init_worker, initargs=(cfg["generator"], cfg["prior"])))
    futs: dict[int, Future] = {}
    probe_futs: dict[int, list[Future]] = {}
    U = scfg["update_every"]
    probing = scfg["probe_always"] or scfg["mode"] != "uniform"
    probing = probing and cfg["prior"] == "math"

    def new_spec(s: int) -> dict:
        n_ctx, d, B = _shape_spec(cfg, s)
        r = mk_rng(cfg["seed"], "cells", s)
        cl = [list(sampler.sample(r)) for _ in range(B)]
        return {"seed": cfg["seed"], "step": s, "n_ctx": n_ctx, "d": d, "n_qry": cfg["n_qry"], "cells": cl}

    def submit_probe(rnd: int):
        if rnd in probe_futs or rnd * U > cfg["steps"]:
            return
        probe_futs[rnd] = [pool.submit(gen_probe, {"seed": cfg["seed"], "round": rnd, "cell": list(c),
                                                   "m": scfg["probes_per_cell"], "n_ctx": scfg["probe_n_ctx"],
                                                   "n_qry": scfg["probe_n_qry"], "d_range": cfg["d_range"],
                                                   "committee": scfg["committee"]}) for c in cell_list]

    for s, spec in pending.items():
        futs[s] = pool.submit(gen_batch, spec)
    stop_at = cfg["steps"] if cfg.get("stop_after") is None else min(cfg["steps"], cfg["stop_after"])
    first_probe_pending = probing
    t_last, wait_s, loss_acc, n_acc = time.perf_counter(), 0.0, 0.0, 0
    model.train()
    try:
        while step < stop_at:
            # ---- adaptation round
            if probing and step > 0 and step % U == 0:
                rnd = step // U
                submit_probe(rnd)
                t0 = time.perf_counter()
                items = [it for f in probe_futs.pop(rnd) for it in f.result()]
                wait_s += time.perf_counter() - t0
                preds = predict_tasks(model, items, device)
                by_cell: dict[tuple, list[float]] = {}
                diag = {}
                for it, p in zip(items, preds):
                    kind = scfg["mode"] if scfg["mode"] != "uniform" else "disagreement"
                    by_cell.setdefault(it["cell"], []).append(task_score(kind, p, it["P"], it["y_qry"], scfg["clip_B"]))
                    dg = diag.setdefault(f"{it['cell'][0]}|{it['cell'][1]}", {"loss": [], "excess": [], "gap": [],
                                                                            "ambiguity": [], "committee_excess": []})
                    dg["loss"].append(float(np.mean((p - it["y_qry"]) ** 2)))
                    dg["excess"].append(float(np.mean((p - it["f_qry"]) ** 2)))
                    dg["gap"].append(float(pfn_gap(p, it["P"]).mean()))
                    dg["ambiguity"].append(float(variance(it["P"]).mean()))
                    dg["committee_excess"].append(float(np.mean((it["P"].mean(0) - it["f_qry"]) ** 2)))
                if scfg["mode"] != "uniform":
                    sampler.update({c: float(np.mean(v)) for c, v in by_cell.items()}, step=step)
                _append(out_dir / "sampler_log.jsonl",
                        {"step": step, "round": rnd, "probs": sampler.probs().tolist(),
                         "cells": [f"{c[0]}|{c[1]}" for c in cell_list], "scores": sampler.scores.tolist(),
                         "diag": {k: {m: float(np.mean(v)) for m, v in dd.items()} for k, dd in diag.items()},
                         "per_task": {k: dd for k, dd in diag.items()}})
                submit_probe(rnd + 1)  # one round ahead: ready well before it is needed
            # ---- keep the prefetch window full (specs decided with the current sampler)
            for s in range(step, min(cfg["steps"], step + cfg["prefetch"])):
                if s not in futs:
                    pending[s] = new_spec(s)
                    futs[s] = pool.submit(gen_batch, pending[s])
            if first_probe_pending:  # queue the first probe round *behind* the initial batches
                submit_probe(step // U + 1)
                first_probe_pending = False
            t0 = time.perf_counter()
            b = futs.pop(step).result()
            pending.pop(step, None)
            wait_s += time.perf_counter() - t0
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, cfg["steps"], cfg["warmup"], cfg["lr"])
            tt = {k: torch.from_numpy(b[k]).to(device, non_blocking=True) for k in ("X_ctx", "y_ctx", "X_qry")}
            target = torch.from_numpy(b["f_qry" if cfg["target"] == "f" else "y_qry"]).to(device)
            with torch.autocast(device.type, dtype=_amp_dtype(device), enabled=amp):
                pred = model(tt["X_ctx"], tt["y_ctx"], tt["X_qry"])
            loss = torch.mean((pred.float() - target) ** 2)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            step += 1
            lv = float(loss.detach())
            loss_acc += lv
            n_acc += 1
            if step % cfg["log_every"] == 0 or step == stop_at:
                now = time.perf_counter()
                _append(out_dir / "train_log.jsonl",
                        {"step": step, "loss": lv, "loss_avg": loss_acc / n_acc, "lr": opt.param_groups[0]["lr"],
                         "sec": now - t_last, "wait_sec": wait_s, "n_ctx": b["X_ctx"].shape[1],
                         "d": b["X_ctx"].shape[2], "batch": b["X_ctx"].shape[0],
                         "cuda_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if amp else 0.0})
                t_last, wait_s, loss_acc, n_acc = now, 0.0, 0.0, 0
            if step % cfg["ckpt_every"] == 0 or step == stop_at:
                _atomic_save(opt.state_dict(), out_dir / "ckpt.opt.pt")
                _atomic_save({"model": model.state_dict(), "step": step, "sampler": sampler.state_dict(),
                              "pending": {s: pending[s] for s in sorted(pending)},
                              "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                              "cfg_identity": _identity_cfg(cfg)}, ckpt)
    finally:
        pool.shutdown(wait=False, cancel_futures=True) if cfg["workers"] else None
    if step >= cfg["steps"]:
        _atomic_save({"model": model.state_dict(), "model_config": model.config, "cfg": cfg,
                      "sampler_history": sampler.history, "cells": cell_list, "n_params": count_params(model),
                      "final_probs": sampler.probs().tolist(), "device": str(device),
                      "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                      "amp_dtype": ({torch.bfloat16: "bf16", torch.float16: "fp16"}[_amp_dtype(device)]
                                    if amp else "fp32")}, final)
        return final
    return ckpt


def load_pfn(path: str | Path, device: str | torch.device = "cpu") -> PFN:
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = st.get("model_config") or st["cfg_identity"]["model"]
    m = PFN(**cfg)
    m.load_state_dict(st["model"])
    return m.to(device).eval()
