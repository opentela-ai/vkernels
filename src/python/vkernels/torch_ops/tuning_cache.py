"""Persistent autotune: best launch configs survive the process.

Triton's ``@triton.autotune`` re-benchmarks every config in every fresh
process and throws the result away at exit — warmup paths pay the full
sweep (and its capture-pollution hazards) on every boot, and a hand-tuned
config table can only be produced by copy-pasting benchmark output into
source. This module keeps the sweep's winner:

* **Store** — one JSON file per (kernel, device) under
  ``$VKERNELS_TUNING_CACHE`` (default ``~/.cache/vkernels/tuning``),
  schema ``vk-tuning-store/1``: per-key chosen config, measured time,
  device block (name/arch/CU count), software block (torch/triton/HIP),
  and sha256 fingerprints of the producing sources.
* **Lookup-first** — ``persistent_autotune`` is a drop-in for
  ``triton.autotune``: on a store hit the persisted config is replayed
  with zero benchmarking; on a miss the sweep runs once and the winner is
  persisted for the next process.
* **Lenient by design** — a device/software mismatch or a changed source
  fingerprint is a *cache miss* (re-tune), never an error. These are
  local performance accelerators, not deployment artifacts; the frozen,
  fail-loud artifact contract lives in :mod:`.tuning_manifest` (#67).
  ``strict=True`` (or a frozen store dir) flips mismatches to raises for
  auditable shared stores.
* **Off switch** — ``VKERNELS_TUNING_CACHE=off`` (or ``off=True``)
  restores pure in-process autotune; the Triton JIT binary cache
  (``TRITON_CACHE_DIR``) is orthogonal and unaffected.

Concurrency: writes are atomic (temp file + ``os.replace``); concurrent
sweeps last-writer-wins per key, which is benign for benchmark winners.
Torch and Triton load lazily and only when a GPU-touching API is called;
store plumbing (record/lookup/schema) is testable without either.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "vk-tuning-store/1"

logger = logging.getLogger(__name__)


class TuningCacheError(RuntimeError):
    """A strict-mode store mismatch (device, software, or fingerprint)."""


# ---------------------------------------------------------------------------
# environment / device identity
# ---------------------------------------------------------------------------


def tuning_enabled() -> bool:
    """False when ``VKERNELS_TUNING_CACHE=off`` — pure in-process autotune."""
    return os.environ.get("VKERNELS_TUNING_CACHE", "").lower() != "off"


def default_store_dir() -> Path:
    """The store root: ``$VKERNELS_TUNING_CACHE`` or ``~/.cache/vkernels/tuning``."""
    env = os.environ.get("VKERNELS_TUNING_CACHE", "").strip()
    return Path(env) if env else Path.home() / ".cache" / "vkernels" / "tuning"


def device_fingerprint() -> dict:
    """JSON-safe identity of the current CUDA device (arch, CU count, software).

    Deliberately excludes the device *index*: the store is shared per host
    and per-arch, and ordinals are not stable across boots or
    ``CUDA_VISIBLE_DEVICES`` settings. Key-level device discrimination
    (e.g. mhc's ``DEVICE`` constexpr) stays in the record keys.
    """
    import torch
    import triton

    props = torch.cuda.get_device_properties(0)
    # HIP exposes gcnArchName (e.g. gfx942:sramecc+:xnack-); NVIDIA falls
    # back to the compute capability.
    arch = getattr(props, "gcnArchName", None) or f"sm{props.major}{props.minor}"
    software = {"torch": torch.__version__, "triton": triton.__version__}
    if torch.version.hip:  # ROCm builds; absent on CUDA builds
        software["hip"] = torch.version.hip
    return {
        "name": props.name,
        "arch": arch.split(":")[0],  # strip the feature-suffix flags
        "cu_count": props.multi_processor_count,
        "software": software,
    }


def fingerprint_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


class TuningCache:
    """A per-kernel, per-device JSON store of tuned launch configs.

    Parameters
    ----------
    kernel_name:
        Store file stem. One file per kernel per device is written:
        ``<store_dir>/<kernel_name>.<arch>.json``.
    source_files:
        Files fingerprinted as producer provenance. A changed fingerprint
        marks every record stale (miss, or raise in strict mode) — configs
        tuned by different source bytes are not this build's configs.
    store_dir:
        Overrides ``default_store_dir()``. Pass a repo path for a frozen
        shared store (pair with ``strict=True``).
    device:
        Injectable fingerprint (tests); defaults to ``device_fingerprint()``.
    strict:
        Raise :class:`TuningCacheError` on any mismatch instead of missing.
    """

    def __init__(self, kernel_name, source_files=(), *, store_dir=None,
                 device=None, strict=False):
        self.kernel_name = kernel_name
        self.source_files = tuple(str(p) for p in source_files)
        self.store_dir = Path(store_dir) if store_dir else default_store_dir()
        self.strict = strict
        self._device = device
        self._records = None
        self._store_path = None

    # -- identity -----------------------------------------------------------

    @property
    def device(self) -> dict:
        if self._device is None:
            self._device = device_fingerprint()
        return self._device

    def _path(self) -> Path:
        if self._store_path is None:
            safe_arch = "".join(c if c.isalnum() else "-" for c in self.device["arch"])
            self._store_path = self.store_dir / f"{self.kernel_name}.{safe_arch}.json"
        return self._store_path

    def _fingerprint_mismatch(self, fingerprints) -> bool:
        return fingerprints != {
            path: fingerprint_file(path) for path in self.source_files
        }

    # -- records ------------------------------------------------------------

    def _load(self) -> dict:
        if self._records is None:
            path = self._path()
            if path.exists():
                doc = json.loads(path.read_text())
                if doc.get("schema") != SCHEMA:
                    if self.strict:
                        raise TuningCacheError(
                            f"{path}: schema {doc.get('schema')!r} != {SCHEMA!r}")
                    logger.warning("tuning cache: %s has schema %r; ignoring",
                                   path, doc.get("schema"))
                    doc = {"records": {}}
                self._records = doc["records"]
            else:
                self._records = {}
        return self._records

    def lookup(self, key) -> dict | None:
        """Return the persisted record for ``key``, or None on any miss.

        Misses: no record, disabled store, device/software mismatch,
        or a producer fingerprint mismatch. Strict mode raises instead
        of missing on the mismatch classes.
        """
        if not tuning_enabled():
            return None
        record = self._load().get(_key_json(key))
        if record is None:
            return None
        problems = []
        if record.get("device") != self.device:
            problems.append(f"device {record.get('device')} != current {self.device}")
        if self._fingerprint_mismatch(record.get("producer", {}).get("fingerprints", {})):
            problems.append("producer sources changed since the record was written")
        if problems:
            if self.strict:
                raise TuningCacheError(
                    f"{self._path()}[{_key_json(key)}]: " + "; ".join(problems))
            logger.info("tuning cache: %s[%s] stale (%s); re-tuning",
                        self.kernel_name, _key_json(key), "; ".join(problems))
            return None
        return record

    def record(self, key, *, kwargs, num_warps, num_stages, time_ms) -> dict:
        """Persist the winning config for ``key`` (atomic write)."""
        if not tuning_enabled():
            return {}
        records = self._load()
        doc_record = {
            "config": {"kwargs": kwargs, "num_warps": num_warps, "num_stages": num_stages},
            "time_ms": time_ms,
            "device": self.device,
            "producer": {"fingerprints": {
                path: fingerprint_file(path) for path in self.source_files
            }},
            "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        records[_key_json(key)] = doc_record
        self._write(records)
        return doc_record

    def _write(self, records) -> None:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema": SCHEMA,
            "kernel": self.kernel_name,
            "device": self.device,
            "records": records,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as out:
                json.dump(doc, out, indent=2, sort_keys=True)
                out.write("\n")
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
        self._records = records

    def clear(self) -> None:
        """Drop this kernel's store file and the in-memory records."""
        self._records = {}
        self._path().unlink(missing_ok=True)


def _key_json(key) -> str:
    return json.dumps(list(key), separators=(",", ":"))


# ---------------------------------------------------------------------------
# the drop-in autotuner
# ---------------------------------------------------------------------------


class PersistentAutotuner:
    """``triton.autotune`` with a persistent store behind the cache.

    Use exactly like ``triton.autotune(configs=..., key=...)``, stacked
    over ``@triton.jit``::

        @persistent_autotune(
            configs=[triton.Config({"ROWS": r}, num_warps=w) for ...],
            key=["TOKENS", "DEVICE"],
            kernel_name="mhc_projection",
            source_files=[__file__],
        )
        @triton.jit
        def partial(...): ...

    Launch as before: ``partial[grid](*args, **launch_kwargs)``. The grid
    callable receives each trial config's constexprs, like Triton's own
    autotuner. ``.cache`` maps key tuples to the winning
    ``triton.Config`` (the same introspection surface
    ``triton.autotune`` exposes).
    """

    def __init__(self, fn, configs, key, *, kernel_name, source_files=(),
                 store_dir=None, strict=False, reprioritize=None):
        self.fn = fn
        self.configs = list(configs)
        self.keys = list(key)
        self.cache = {}  # key tuple -> triton.Config (in-process)
        arg_names = getattr(fn, "arg_names", None)
        if arg_names is None:
            raise TypeError("persistent_autotune must wrap a @triton.jit function")
        missing = [k for k in self.keys if k not in arg_names]
        if missing:
            raise ValueError(f"autotune key(s) {missing} not in kernel args {arg_names}")
        self._cache = TuningCache(kernel_name, source_files,
                                  store_dir=store_dir, strict=strict)
        self._bench = None  # injectable for tests

    def _config_from_record(self, record):
        import triton

        config = record["config"]
        return triton.Config(dict(config["kwargs"]),
                             num_warps=config["num_warps"],
                             num_stages=config["num_stages"])

    def _key_of(self, args, kwargs):
        bound = dict(zip(self.fn.arg_names, args))
        bound.update(kwargs)
        return tuple(bound[k] for k in self.keys)

    def _pick(self, key, grid, args, kwargs):
        """Store hit, else benchmark every config and persist the winner."""
        record = self._cache.lookup(key)
        if record is not None:
            return self._config_from_record(record)
        import triton

        bench = self._bench or triton.testing.do_bench
        best, best_ms = None, float("inf")
        for config in self.configs:
            call = self._GridCall(self, grid, args, kwargs, config)
            time_ms = bench(call)
            if time_ms < best_ms:
                best, best_ms = config, time_ms
        self._cache.record(
            key, kwargs=dict(best.kwargs), num_warps=best.num_warps,
            num_stages=best.num_stages, time_ms=best_ms)
        return best

    class _GridCall:
        """``self.fn[grid](...)`` binding that re-evaluates a callable grid
        against one trial config's constexprs (Triton's autotuner behavior)."""

        def __init__(self, tuner, grid, args, kwargs, config):
            self._tuner, self._grid = tuner, grid
            self._args, self._kwargs = args, kwargs
            self._config = config

        def __call__(self):
            launch = dict(self._config.kwargs)
            launch["num_warps"] = self._config.num_warps
            launch["num_stages"] = self._config.num_stages
            launch.update(self._kwargs)
            self._tuner.fn[self._grid](*self._args, **launch)

    def __getitem__(self, grid):
        return _BoundLaunch(self, grid)


class _BoundLaunch:
    def __init__(self, tuner, grid):
        self._tuner, self._grid = tuner, grid

    def __call__(self, *args, **kwargs):
        tuner = self._tuner
        key = tuner._key_of(args, kwargs)
        config = tuner.cache.get(key)
        if config is None:
            if tuning_enabled():
                config = tuner._pick(key, self._grid, args, kwargs)
            else:  # off switch: time every config, keep nothing
                import triton

                bench = tuner._bench or triton.testing.do_bench
                best, best_ms = None, float("inf")
                for candidate in tuner.configs:
                    call = PersistentAutotuner._GridCall(tuner, self._grid, args, kwargs, candidate)
                    time_ms = bench(call)
                    if time_ms < best_ms:
                        best, best_ms = candidate, time_ms
                config = best
            tuner.cache[key] = config
        PersistentAutotuner._GridCall(tuner, self._grid, args, kwargs, config)()


def persistent_autotune(configs, key, *, kernel_name, source_files=(),
                        store_dir=None, strict=False):
    """Decorator factory: ``@persistent_autotune(...)`` over ``@triton.jit``.

    See :class:`PersistentAutotuner` for the contract. The winner is read
    from (or written to) the per-kernel, per-device tuning store; with
    ``VKERNELS_TUNING_CACHE=off`` the decorator degrades to benchmark-only
    in-process autotune.
    """

    def decorate(fn):
        return PersistentAutotuner(fn, configs, key, kernel_name=kernel_name,
                                   source_files=source_files,
                                   store_dir=store_dir, strict=strict)

    return decorate
