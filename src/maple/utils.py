"""Seeding, hashing, timing, memory and append-only raw record storage."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def stable_hash(obj: Any, length: int = 12) -> str:
    """Order-independent short SHA-1 of a JSON-serialisable object."""
    return hashlib.sha1(_canonical(obj).encode()).hexdigest()[:length]


def rng(seed: int, *keys: str | int) -> np.random.Generator:
    """Independent generator for (seed, *keys); identical inputs give identical streams."""
    digest = hashlib.sha256(_canonical([int(seed), *map(str, keys)]).encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(list(np.frombuffer(digest[:16], dtype=np.uint32))))


def seed_int(seed: int, *keys: str | int) -> int:
    """Derived 31-bit integer seed, for libraries that need an int."""
    return int(rng(seed, *keys).integers(0, 2**31 - 1))


class Timer:
    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._t0


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (peak_wset on Windows)."""
    info = psutil.Process(os.getpid()).memory_info()
    peak = getattr(info, "peak_wset", None) or info.rss
    return peak / 2**20


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


class RawStore:
    """Append-only JSONL store. Each record needs a unique ``key``.

    Existing files are only ever opened in append mode; a torn final line from a
    killed process is ignored on read (and left in place, never rewritten).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: set[str] | None = None

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def completed_keys(self) -> set[str]:
        if self._keys is None:
            self._keys = {r["key"] for r in self.records() if "key" in r}
        return set(self._keys)

    def append(self, record: dict) -> None:
        if "key" not in record:
            raise ValueError("record must contain 'key'")
        keys = self.completed_keys()
        if record["key"] in keys:
            raise ValueError(f"duplicate key {record['key']!r} in {self.path}")
        # Ensure we start on a fresh line if a previous writer was killed mid-line.
        needs_nl = self.path.exists() and self.path.stat().st_size > 0 and not self._ends_with_newline()
        with self.path.open("a", encoding="utf-8") as fh:
            if needs_nl:
                fh.write("\n")
            fh.write(json.dumps(record, default=_json_default) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        assert self._keys is not None
        self._keys.add(record["key"])

    def _ends_with_newline(self) -> bool:
        with self.path.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) == b"\n"


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _atomic(path: Path, write) -> None:
    """Write to a sibling temp file, then os.replace: a kill mid-write never leaves a truncated target that a resumed
    run would mistake for a finished result."""
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    _atomic(Path(path), lambda t: t.write_text(text, encoding="utf-8"))


def atomic_savez(path: str | Path, **arrays) -> None:
    import numpy as np

    def w(t):
        with open(t, "wb") as fh:  # file handle, so numpy does not append ".npz" to the temp name
            np.savez_compressed(fh, **arrays)
    _atomic(Path(path), w)


def write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=_json_default))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
