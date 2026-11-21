"""YAML configuration loading with ``include:`` deep-merge, and environment capture."""
from __future__ import annotations

import copy
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inc = cfg.pop("include", None)
    if inc is None:
        return cfg
    includes = [inc] if isinstance(inc, str) else list(inc)
    base: dict = {}
    for item in includes:
        base = deep_merge(base, load_config(path.parent / item))
    return deep_merge(base, cfg)


_PKGS = ["numpy", "scipy", "pandas", "scikit-learn", "torch", "xgboost", "lightgbm", "catboost", "openml",
         "matplotlib", "statsmodels"]


def env_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in _PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "missing"
    return out
