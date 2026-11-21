"""Benchmark runners. All results are appended to an append-only RawStore; completed keys are skipped.

Real data, three phases (each resumable):
  1. standalone baselines (CPU, parallel)              -> records  "<ds>|n<regime>|r<rep>|<method>"
  2. caches: expert OOF/test predictions (CPU, parallel) and PFN OOF/test predictions per checkpoint (GPU)
  3. derived methods from caches: pfn_<v>, E/F routing, stacking, averaging, single experts -> records
Synthetic suites:
  committee/baseline predictions per task (CPU, parallel, cached) + PFN predictions per checkpoint (GPU) -> records
"""
from __future__ import annotations

import json
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from maple.config import env_versions
from maple.data import datasets as D
from maple.data.splits import make_splits
from maple.disagreement.measures import pfn_gap, variance
from maple.evaluation import baselines
from maple.evaluation.metrics import compute
from maple.experts.registry import COMMITTEE_DEFAULT, make_expert
from maple.utils import RawStore, atomic_savez, rss_mb, seed_int, stable_hash

VERSIONS = env_versions()
PRED_REPEAT = 0  # test predictions are stored for this repeat only (residual/prediction figures)


class PeakMemory:
    """Peak RSS increase (MB) during a block, sampled every 5 ms by a background thread."""

    def __enter__(self):
        self.base = rss_mb()
        self.peak = self.base
        self._stop = False

        def run():
            while not self._stop:
                self.peak = max(self.peak, rss_mb())
                time.sleep(0.005)
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join()
        self.peak = max(self.peak, rss_mb())
        self.mb = self.peak - self.base


def key(ds: str, regime: int, rep: int, method: str) -> str:
    return f"{ds}|n{regime}|r{rep}|{method}"


def build_splits(datasets, regimes, repeats, loader, split_dir, seed, log: list | None = None) -> list[dict]:
    out = []
    for ds in datasets:
        X, y, _ = loader(ds)
        for reg in regimes:
            sp = make_splits(ds, len(y), reg, repeats[reg], seed, split_dir)
            if sp is None:
                if log is not None:
                    log.append({"dataset": ds, "regime": reg, "n": len(y), "reason": "fewer than 200 test rows"})
                continue
            for rep, (tr, te) in enumerate(sp):
                out.append({"dataset": ds, "regime": reg, "repeat": rep, "train": tr, "test": te,
                            "seed": seed_int(seed, ds, reg, rep)})
    return out


@lru_cache(maxsize=4)
def _cached_load(loader: Callable, name: str):
    return loader(name)


TIMING_PROTOCOL = ("wall-clock (perf_counter) and CPU time (process_time) of the job's own process; CPU models "
                   "single-threaded; machine shared with other jobs, so wall-clock is an upper bound; PFN on a shared GPU")


def _record(sp, method, y_te, pred, fit_s, pred_s, mem, extra=None, fit_cpu=None, pred_cpu=None,
            n_trials=None) -> dict:
    r = {"key": key(sp["dataset"], sp["regime"], sp["repeat"], method), "dataset": sp["dataset"],
         "regime": sp["regime"], "repeat": sp["repeat"], "seed": sp["seed"], "method": method,
         "config_hash": stable_hash({"method": method, "n_trials": n_trials, "seed": sp["seed"], "code": "v2"}),
         "metrics": compute(y_te, pred), "fit_cpu_s": fit_cpu, "predict_cpu_s": pred_cpu,
         "timing_protocol": TIMING_PROTOCOL,
         "fit_s": fit_s, "predict_s": pred_s, "peak_mem_mb": mem, "n_train": len(sp["train"]),
         "n_test": len(sp["test"]), "versions": VERSIONS, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        r.update(extra)
    return r


def _save_pred(pred_dir, sp, method, pred):
    if pred_dir is not None and sp["repeat"] == PRED_REPEAT:
        p = Path(pred_dir) / f"{sp['dataset']}__n{sp['regime']}__{method}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            np.save(p, pred.astype(np.float32))


def _standalone_job(sp, method, loader, n_trials, pred_dir):
    X, y, _ = _cached_load(loader, sp["dataset"])
    Xtr, Xte = X.iloc[sp["train"]].reset_index(drop=True), X.iloc[sp["test"]].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = baselines.make(method, seed=sp["seed"] % (2**31 - 1), n_trials=n_trials)
        with PeakMemory() as pm:
            t0, c0 = time.perf_counter(), time.process_time()
            m.fit(Xtr, y[sp["train"]])
            t1, c1 = time.perf_counter(), time.process_time()
            pred = m.predict(Xte)
            t2, c2 = time.perf_counter(), time.process_time()
    _save_pred(pred_dir, sp, method, pred)
    extra = {"best_params": getattr(m, "best_params_", None)}
    return _record(sp, method, y[sp["test"]], pred, t1 - t0, t2 - t1, pm.mb, extra, c1 - c0, c2 - c1, n_trials)


def _safe_job(sp, method, loader, n_trials, pred_dir):
    try:
        return _standalone_job(sp, method, loader, n_trials, pred_dir)
    except Exception as e:  # a failing job must not discard the finished records of its batch
        return {"__error__": True, "key": key(sp["dataset"], sp["regime"], sp["repeat"], method),
                "error": f"{type(e).__name__}: {e}"}


def run_real(methods, datasets, regimes, repeats, store: RawStore, loader=D.load, split_dir=None, n_jobs=1, seed=0,
             max_jobs=None, n_trials=10, pred_dir=None, skip_log=None, batch=64) -> int:
    splits = build_splits(datasets, regimes, repeats, loader, split_dir, seed, skip_log)
    done = store.completed_keys()
    jobs = [(sp, m) for sp in splits for m in methods if key(sp["dataset"], sp["regime"], sp["repeat"], m) not in done]
    # expensive tuned jobs first so the pool stays busy at the end
    jobs.sort(key=lambda j: (not j[1].endswith("_tuned"), j[1] != "catboost"))
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    n = 0
    # records are appended as each job finishes (no per-batch barrier, so stragglers do not idle the pool);
    # `batch` bounds the number of jobs dispatched ahead of completion
    if n_jobs != 1:
        recs = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered",
                        pre_dispatch=max(batch, 2 * n_jobs))(
            delayed(_safe_job)(sp, m, loader, n_trials, pred_dir) for sp, m in jobs)
    else:
        recs = (_safe_job(sp, m, loader, n_trials, pred_dir) for sp, m in jobs)
    for r in recs:
        if r.get("__error__"):
            err = store.path.with_name(store.path.stem + ".errors.jsonl")
            with err.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": r["key"], "error": r["error"], "time": time.strftime("%H:%M:%S")}) + "\n")
            continue
        store.append(r)
        n += 1
    return n


# --------------------------------------------------------------------------------- caches for routing/derived

def _cache_path(cache_dir, sp, what) -> Path:
    return Path(cache_dir) / f"{sp['dataset']}__n{sp['regime']}__r{sp['repeat']}__{what}.npz"


def _expert_cache_job(sp, loader, cache_dir, experts, n_folds=5, tag="experts"):
    from maple.routing.router import _expert_matrix, _expert_matrix_fit

    path = _cache_path(cache_dir, sp, tag)
    if path.exists():
        return str(path)
    X, y, _ = _cached_load(loader, sp["dataset"])
    Xtr, Xte = X.iloc[sp["train"]].reset_index(drop=True), X.iloc[sp["test"]].reset_index(drop=True)
    ytr = y[sp["train"]]
    K = len(experts)
    O = np.zeros((K, len(ytr)))
    oof_s, oof_c = np.zeros(K), np.zeros(K)
    s = sp["seed"] % (2**31 - 1)
    for tr, va in KFold(n_folds, shuffle=True, random_state=s).split(Xtr):
        enc, p, Ztr = _expert_matrix_fit(Xtr.iloc[tr])
        Zva = _expert_matrix(enc, p, Xtr.iloc[va])
        for j, e in enumerate(experts):
            t0, c0 = time.perf_counter(), time.process_time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                O[j, va] = make_expert(e, s).fit(Ztr, ytr[tr]).predict(Zva)
            oof_s[j] += time.perf_counter() - t0
            oof_c[j] += time.process_time() - c0
    enc, p, Ztr = _expert_matrix_fit(Xtr)
    Zte = _expert_matrix(enc, p, Xte)
    T = np.zeros((K, len(Xte)))
    fit_s, pred_s, mem, fit_c, pred_c = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)
    for j, e in enumerate(experts):
        with warnings.catch_warnings(), PeakMemory() as pm:
            warnings.simplefilter("ignore")
            t0, c0 = time.perf_counter(), time.process_time()
            m = make_expert(e, s).fit(Ztr, ytr)
            t1, c1 = time.perf_counter(), time.process_time()
            T[j] = m.predict(Zte)
            t2, c2 = time.perf_counter(), time.process_time()
        fit_s[j], pred_s[j], mem[j], fit_c[j], pred_c[j] = t1 - t0, t2 - t1, pm.mb, c1 - c0, c2 - c1
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(path, O=O, T=T, Ztr=Ztr.astype(np.float32), Zte=Zte.astype(np.float32), ytr=ytr,
                        yte=y[sp["test"]], oof_s=oof_s, fit_s=fit_s, pred_s=pred_s, mem=mem, fit_cpu=fit_c,
                        pred_cpu=pred_c, oof_cpu=oof_c,
                        experts=np.array(experts))
    return str(path)


def run_expert_cache(splits, loader, cache_dir, n_jobs=1, experts=None, tag="experts"):
    experts = list(experts or COMMITTEE_DEFAULT)
    todo = [sp for sp in splits if not _cache_path(cache_dir, sp, tag).exists()]
    Parallel(n_jobs=n_jobs, backend="loky")(delayed(_expert_cache_job)(sp, loader, cache_dir, experts, 5, tag)
                                            for sp in todo)


def run_pfn_cache(splits, ckpts: dict, loader, cache_dir, device="auto", n_folds=5, n_ensemble=4, progress=None,
                  oof_for=None, n_ensemble_oof=None):
    """For each split and checkpoint: PFN OOF predictions on the training split + test predictions (GPU)."""
    import torch

    from maple.model.predictor import MapleRegressor, _load, resolve_device

    n_ensemble_oof = n_ensemble if n_ensemble_oof is None else n_ensemble_oof  # review finding 3: matched ensembles
    dev = resolve_device(device)
    for i, sp in enumerate(splits):
        X, y, _ = _cached_load(loader, sp["dataset"])
        Xtr, Xte = X.iloc[sp["train"]].reset_index(drop=True), X.iloc[sp["test"]].reset_index(drop=True)
        ytr = y[sp["train"]]
        s = sp["seed"] % (2**31 - 1)
        for name, ck in ckpts.items():
            path = _cache_path(cache_dir, sp, f"pfn_{name}")
            if path.exists():
                continue
            O = np.full(len(ytr), np.nan)
            t0, c0 = time.perf_counter(), time.process_time()
            if oof_for is None or name in oof_for:
                for tr, va in KFold(n_folds, shuffle=True, random_state=s).split(Xtr):
                    O[va] = MapleRegressor(ck, n_ensemble_oof, device=device, seed=s).fit(
                        Xtr.iloc[tr], ytr[tr]).predict(Xtr.iloc[va])
            oof_s, oof_c = time.perf_counter() - t0, time.process_time() - c0
            _load(str(Path(ck).resolve()), str(dev))  # load/cache weights outside the timed block
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            with PeakMemory() as pm:
                t0, c0 = time.perf_counter(), time.process_time()
                m = MapleRegressor(ck, n_ensemble, device=device, seed=s).fit(Xtr, ytr)
                t1, c1 = time.perf_counter(), time.process_time()
                T = m.predict(Xte)
                t2, c2 = time.perf_counter(), time.process_time()
            cuda_mb = torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else 0.0
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_savez(path, O=O, T=T, oof_s=oof_s, fit_s=t1 - t0, pred_s=t2 - t1, mem=pm.mb,
                                cuda_mb=cuda_mb, fit_cpu=c1 - c0, pred_cpu=c2 - c1, oof_cpu=oof_c,
                                n_ensemble_oof=n_ensemble_oof)
        if progress and i % 25 == 0:
            progress(f"pfn cache {i}/{len(splits)}")


def derive_records(splits, ckpt_names, routing: dict, cache_dir, store: RawStore, pred_dir=None,
                   with_classical=True, expected_experts=None, tag="experts") -> int:
    """Records for pfn_<v>, E_<v> (global), F_<v> (local), stackpfn_<v>, and (once) classical combinations.

    Memory of combined methods is the SUM of the components' peaks (all models are resident together).
    """
    from maple.routing.router import combine, stacking_weights

    expected = list(expected_experts or COMMITTEE_DEFAULT)
    done = store.completed_keys()
    skipped_log = store.path.with_name(store.path.stem + ".skipped.jsonl")
    n = 0
    for sp in splits:
        ep = _cache_path(cache_dir, sp, tag)
        if not ep.exists():
            continue
        E = np.load(ep)
        experts = [str(e) for e in E["experts"]]
        if with_classical and experts != expected:
            raise ValueError(f"{ep} holds committee {experts}, expected {expected}")
        yte, ytr = E["yte"], E["ytr"]
        ecpu = lambda k, j=None: float(E[k][j] if j is not None else E[k].sum()) if k in E else None  # noqa: E731
        if with_classical:
            cls = {f"expert_{e}": (E["T"][j], E["fit_s"][j], E["pred_s"][j], E["mem"][j], ecpu("fit_cpu", j),
                                   ecpu("pred_cpu", j)) for j, e in enumerate(experts)}
            cls["avg_experts"] = (E["T"].mean(0), E["fit_s"].sum(), E["pred_s"].sum(), E["mem"].sum(),
                                  ecpu("fit_cpu"), ecpu("pred_cpu"))
            w = stacking_weights(E["O"], ytr)
            cls["stack_experts"] = (w @ E["T"], E["fit_s"].sum() + E["oof_s"].sum(), E["pred_s"].sum(), E["mem"].sum(),
                                    (ecpu("fit_cpu") or 0) + (ecpu("oof_cpu") or 0), ecpu("pred_cpu"))
            for m, (pred, fs, ps, mem, fc, pc) in cls.items():
                k = key(sp["dataset"], sp["regime"], sp["repeat"], m)
                if k not in done:
                    store.append(_record(sp, m, yte, pred, float(fs), float(ps), float(mem), None, fc, pc))
                    _save_pred(pred_dir, sp, m, pred)
                    n += 1
        for v in ckpt_names:
            pp = _cache_path(cache_dir, sp, f"pfn_{v}")
            if not pp.exists():
                continue
            P = np.load(pp)
            pc = lambda k: float(P[k]) if k in P else None  # noqa: E731
            O = np.vstack([P["O"][None], E["O"]])
            T = np.vstack([P["T"][None], E["T"]])
            base_fit = float(P["fit_s"] + E["fit_s"].sum() + P["oof_s"] + E["oof_s"].sum())
            base_pred = float(P["pred_s"] + E["pred_s"].sum())
            cpu_fit = sum(x for x in (pc("fit_cpu"), pc("oof_cpu"), ecpu("fit_cpu"), ecpu("oof_cpu")) if x is not None)
            cpu_pred = sum(x for x in (pc("pred_cpu"), ecpu("pred_cpu")) if x is not None)
            mem = float(P["mem"] + E["mem"].sum())
            meths = {f"pfn_{v}": (P["T"], float(P["fit_s"]), float(P["pred_s"]), float(P["mem"]),
                                  {"cuda_peak_mb": float(P["cuda_mb"])}, pc("fit_cpu"), pc("pred_cpu"))}
            if v in routing:
                if np.isfinite(P["O"]).all():
                    for mode, rtag in (("global", "E"), ("local", "F")):
                        prm = routing[v][mode]
                        pred = combine(O, T, ytr, E["Ztr"], E["Zte"], mode, prm["beta"], prm["k"])
                        meths[f"{rtag}_{v}"] = (pred, base_fit, base_pred, mem,
                                               {"routing": {"beta": prm["beta"], "k": prm["k"]},
                                                "cuda_peak_mb": float(P["cuda_mb"])}, cpu_fit, cpu_pred)
                    w = stacking_weights(O, ytr)
                    meths[f"stackpfn_{v}"] = (w @ T, base_fit, base_pred, mem, {"weights": w.tolist()}, cpu_fit, cpu_pred)
                else:
                    with skipped_log.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"split": key(sp["dataset"], sp["regime"], sp["repeat"], v),
                                             "reason": "PFN OOF predictions missing/non-finite"}) + "\n")
            for m, (pred, fs, ps, mm, extra, fc, pcu) in meths.items():
                k = key(sp["dataset"], sp["regime"], sp["repeat"], m)
                if k not in done:
                    store.append(_record(sp, m, yte, pred, fs, ps, mm, extra, fc, pcu))
                    _save_pred(pred_dir, sp, m, pred)
                    n += 1
    return n


# --------------------------------------------------------------------------------- synthetic suites

SYN_BASELINES = ["mean", "ridge", "knn", "rf", "extratrees", "hgb", "rbf", "xgboost", "lightgbm", "catboost"]


def _syn_pred(name, t, seed):
    if name == "mean":
        return np.full(len(t.y_qry), t.y_ctx.mean())
    if name in ("xgboost", "lightgbm", "catboost"):
        m = baselines.make(name, seed)
        cols = [f"x{i}" for i in range(t.X_ctx.shape[1])]
        return m.fit(pd.DataFrame(t.X_ctx, columns=cols), t.y_ctx).predict(pd.DataFrame(t.X_qry, columns=cols))
    return make_expert(name, seed).fit(t.X_ctx, t.y_ctx).predict(t.X_qry)


def _syn_cache_job(suite, i, seed, cache_dir, n_tasks):
    from maple.tasks.suites import make_suite  # noqa: F401  (regenerated deterministically)
    path = Path(cache_dir) / f"{suite}__t{i}.npz"
    if path.exists():
        return
    t = _suite_task(suite, i, seed)
    preds, times = [], []
    for name in SYN_BASELINES:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds.append(_syn_pred(name, t, seed + i))
        times.append(time.perf_counter() - t0)
    P = np.stack(preds)
    C = np.stack([P[SYN_BASELINES.index(c)] for c in COMMITTEE_DEFAULT])
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(path, P=P, times=np.array(times), committee_mean=C.mean(0), ambiguity=variance(C))


def run_synthetic_external(suites, n_tasks, seed, methods, store: RawStore) -> int:
    """Extra synthetic records for models outside the cached baseline set (e.g. TabPFN v2, TabICL), on exactly the
    same tasks and inputs as the cached gradient-boosting baselines (DataFrame columns x0.., seed = seed + task)."""
    done = store.completed_keys()
    n = 0
    for s in suites:
        for i in range(n_tasks):
            todo = [m for m in methods if f"{s}|t{i}|{m}" not in done]
            if not todo:
                continue
            t = _suite_task(s, i, seed)
            cols = [f"x{j}" for j in range(t.X_ctx.shape[1])]
            Xc, Xq = pd.DataFrame(t.X_ctx, columns=cols), pd.DataFrame(t.X_qry, columns=cols)
            base = {"suite": s, "task": i, "seed": seed, **_syn_meta(t), "versions": VERSIONS}
            for m in todo:
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = baselines.make(m, seed + i).fit(Xc, t.y_ctx).predict(Xq)
                store.append({"key": f"{s}|t{i}|{m}", "method": m, **base, "metrics": _syn_metrics(t, pred),
                              "fit_predict_s": time.perf_counter() - t0})
                n += 1
    return n


def _suite_task(suite, i, seed):
    from maple.tasks.suites import SUITE_D, SUITE_N_CTX, SUITE_N_QRY, _suite_generator
    from maple.utils import rng as mk_rng
    gen, cs, noise = _suite_generator(suite)
    r = mk_rng(seed, "suite", suite, i)
    cell = cs[int(r.integers(len(cs)))]
    n_ctx = int(r.choice(SUITE_N_CTX))
    d = int(r.integers(SUITE_D[0], SUITE_D[1] + 1))
    t = gen.sample(r, cell, n_ctx, SUITE_N_QRY, d, noise_types=noise)
    t.meta.update({"suite": suite, "task_index": i})
    return t


def _syn_meta(t) -> dict:
    m = t.meta
    return {"family": m.get("family"), "complexity": m.get("complexity"), "n_ctx": m.get("n_ctx"), "d": m.get("d"),
            "noise_type": m.get("noise", {}).get("type"), "noise_rel": m.get("noise", {}).get("rel"),
            "composition": m.get("composition"), "missing_rate": m.get("missing_rate")}


def _syn_metrics(t, pred) -> dict:
    out = compute(t.y_qry, pred)
    rf = float(np.sqrt(np.mean((pred - t.f_qry) ** 2)))
    out.update({"rmse_f": rf, "nrmse_f": rf / (t.f_qry.std() + 1e-12)})
    return out


def run_synthetic(suites, n_tasks, seed, ckpts: dict, store: RawStore, cache_dir, n_jobs=1, device="auto",
                  baseline_records=True) -> int:
    from maple.model.predictor import _load, pfn_predict_arrays, resolve_device

    todo = [(s, i) for s in suites for i in range(n_tasks) if not (Path(cache_dir) / f"{s}__t{i}.npz").exists()]
    Parallel(n_jobs=n_jobs, backend="loky", batch_size=4)(
        delayed(_syn_cache_job)(s, i, seed, cache_dir, n_tasks) for s, i in todo)
    done = store.completed_keys()
    dev = resolve_device(device)
    models = {v: _load(str(Path(p).resolve()), str(dev)) for v, p in ckpts.items()}
    n = 0
    for s in suites:
        for i in range(n_tasks):
            want = ([f"{s}|t{i}|{b}" for b in SYN_BASELINES] if baseline_records else []) + \
                   [f"{s}|t{i}|pfn_{v}" for v in ckpts]
            if all(k in done for k in want):
                continue
            t = _suite_task(s, i, seed)
            C = np.load(Path(cache_dir) / f"{s}__t{i}.npz")
            base = {"suite": s, "task": i, "seed": seed, **_syn_meta(t), "versions": VERSIONS}
            if baseline_records:
                for j, b in enumerate(SYN_BASELINES):
                    k = f"{s}|t{i}|{b}"
                    if k not in done:
                        store.append({"key": k, "method": b, **base, "metrics": _syn_metrics(t, C["P"][j]),
                                      "fit_predict_s": float(C["times"][j])})
                        n += 1
            for v, model in models.items():
                k = f"{s}|t{i}|pfn_{v}"
                if k in done:
                    continue
                t0 = time.perf_counter()
                p = pfn_predict_arrays(model, t.X_ctx, t.y_ctx, t.X_qry, dev)
                el = time.perf_counter() - t0
                store.append({"key": k, "method": f"pfn_{v}", **base, "metrics": _syn_metrics(t, p),
                              "fit_predict_s": el, "gap": float(np.mean((p - C["committee_mean"]) ** 2)),
                              "ambiguity": float(C["ambiguity"].mean()),
                              "committee_excess": float(np.mean((C["committee_mean"] - t.f_qry) ** 2)),
                              "excess": float(np.mean((p - t.f_qry) ** 2))})
                n += 1
    return n
