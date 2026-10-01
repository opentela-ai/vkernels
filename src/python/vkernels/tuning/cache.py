"""Op-config cache: tune once per (op, arch, shape class), persist, replay.

The kernel tier (:mod:`vkernels.torch_ops.tuning_cache`) persists per-*kernel*
autotune winners keyed by autotune key args — often exact sizes, so a new
size re-sweeps. This module is the *op* tier above it: per-op launch configs
keyed by

    (op name, kernel-source fingerprint, arch identity, shape-class bucket)

where the shape class is a COARSE tier (the tree's per-shape gating
conventions — decode T<=8 / T<=64 / prefill row tiers, head-dim tiers,
(O, I) GEMV tiers — never exact shapes), so one tuned config serves a whole
bucket and a new size inside a known bucket is a hit, not a sweep.

* **Store** — one JSON per (op, arch capability) under ``$VKERNELS_CACHE``
  (default ``~/.cache/vkernels/``), ``schema vk-op-config-cache/1``: per
  shape-class records carrying the config, the key fields (source
  fingerprint, device identity), the tune timestamp, and the bench numbers
  captured at tune time. Human-readable, auditable, safe to hand-inspect.
* **Lookup-first** — on a store hit the persisted config replays with zero
  benchmarking; on a miss the op's own sweep runs ONCE under a bounded
  budget (default 2 s per op+bucket: ``VKERNELS_CACHE_BUDGET_MS``) and the
  winner persists for every later process. Budget exhaustion, a failed
  bench, or an empty candidate space falls back to the op's DECLARED
  default config, persisted with ``status: "default"`` and the reason —
  auditable, and never re-paid per process.
* **Capture contract** — configs must be static when a CUDA graph capture
  observes the op. floe's serve boot warms eagerly per shape BEFORE any
  capture (``engine/runner/warmup.py``: eager ``startup-eager`` passes,
  then ``capture_buckets``); the tune-on-miss path runs exactly there.
  Enforcement is in here, not just convention: while a stream is
  capturing, :func:`op_config` NEVER benches and NEVER writes — it serves
  the process memo or the declared default instantly (a ``do_bench`` sync
  mid-capture would abort the capture). A bucket first seen inside a
  capture therefore stays on the default config for the process lifetime,
  and the captured graph is self-consistent.
* **Off switch** — ``VKERNELS_CACHE=off`` disables the store AND the sweep:
  the declared default config returns unchanged (today's behavior — the
  status-quo audit found no in-process ``@triton.autotune`` left in the
  tree to fall back to). The kernel tier's ``VKERNELS_TUNING_CACHE`` is
  orthogonal and unaffected, as is Triton's own JIT binary cache.
* **Population entries** — :func:`seed` lands offline-produced tables
  (e.g. lane-29 H100 sweeps) in the SAME schema with ``status: "seeded"``;
  a locally tuned record always outranks a seed for the same bucket.

Source fingerprints (module files by default) make code changes a cache
miss: configs tuned by different source bytes are not this build's
configs, so an edit re-tunes once under the same bounded budget.

Lenient by design, like the kernel tier: a corrupt/foreign store file, a
failed bench, or a missing device block is a miss or a default — never an
error on the op-launch hot path. Only programmer errors (a non-JSON-
serializable config vocabulary, a missing ``default``) raise, loudly.

Concurrency: writes are atomic (temp file + ``os.replace``) under the
per-file ``store_lock`` shared with the kernel tier; a writer re-reads the
disk state under the lock and merges only its own record, so concurrent
processes last-writer-wins per RECORD (not per file). Within a process a
module lock serializes resolution and a memo freezes every resolved
(config, status) for the process lifetime — a bucket never changes config
under a running serve. Torch imports lazily; the store plumbing is
testable without torch (inject ``device=``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ..torch_ops.store_lock import store_lock
from ..torch_ops.tuning_cache import fingerprint_file

SCHEMA = "vk-op-config-cache/1"
DEFAULT_BUDGET_MS = 2000.0

logger = logging.getLogger(__name__)

# statuses a persisted record can carry
TUNED, SEEDED, DEFAULT = "tuned", "seeded", "default"
# statuses op_config can resolve to (the first three come from records;
# "off"/"capture" never touch the store)
OFF, CAPTURE = "off", "capture"


def enabled() -> bool:
    """False when ``VKERNELS_CACHE=off`` — declared defaults, no store, no sweep."""
    return os.environ.get("VKERNELS_CACHE", "").strip().lower() != "off"


def store_root(store_dir=None) -> Path:
    """The store root: ``$VKERNELS_CACHE`` or ``~/.cache/vkernels/``.

    The kernel tier lives one level down (``~/.cache/vkernels/tuning/``);
    op-tier files sit beside it as ``<op>.<capability>.json``.
    """
    if store_dir is not None:
        return Path(store_dir)
    env = os.environ.get("VKERNELS_CACHE", "").strip()
    if env:
        return Path(env)
    return Path.home() / ".cache" / "vkernels"


def tune_budget(override=None) -> float:
    """Per-(op, bucket) tune budget: arg > ``VKERNELS_CACHE_BUDGET_MS`` > 2 s."""
    if override is not None:
        return float(override)
    env = os.environ.get("VKERNELS_CACHE_BUDGET_MS", "").strip()
    return float(env) if env else DEFAULT_BUDGET_MS


def arch_identity(device=None) -> dict:
    """The arch identity a record is keyed/validated against.

    ``capability`` (``torch.cuda.get_device_capability`` as ``sm<maj><min>``
    / stripped ``gcnArchName``), ``sm_count`` (multiprocessor count) and the
    device ``name`` — exactly the triple that moves GEMV/tile optima between
    parts. Software versions ride along as audit-only metadata (a triton
    upgrade may stale a config, but that is a perf question, not identity;
    the source fingerprint owns code-change invalidation). Reuses the
    kernel tier's device fingerprint so every tier names a device the same
    way.
    """
    from ..torch_ops.tuning_cache import device_fingerprint

    fp = device_fingerprint(device)
    return {
        "capability": fp["arch"],
        "sm_count": fp["cu_count"],
        "name": fp["name"],
        "software": fp["software"],
    }


def _identity_key(device: dict) -> tuple:
    """The (capability, sm_count, name) triple a record must match."""
    return (device.get("capability"), device.get("sm_count"), device.get("name"))


def source_fingerprint(files=(), objs=()) -> str:
    """``sha256:...`` over the op's producing sources — auto-invalidation.

    Module files are hashed byte-for-byte (the kernel + its config plumbing
    + the heuristic that shapes the default). ``objs`` are hashed by
    ``inspect.getsource`` after unwrapping (``functools.lru_cache`` /
    triton JIT wrappers) for finer granularity when a caller wants it; an
    unhashable object falls back to a stable repr. A changed fingerprint is
    a cache miss (re-tune), never an error.
    """
    import hashlib
    import inspect

    parts = []
    for path in files:
        parts.append(f"{path}={fingerprint_file(path)}")
    for obj in objs:
        fn = obj
        for attr in ("__wrapped__", "fn", "__func__"):
            fn = getattr(fn, attr, fn)
        try:
            src = inspect.getsource(fn)
        except Exception:
            src = repr(obj)
        digest = hashlib.sha256(src.encode()).hexdigest()
        parts.append(f"{getattr(fn, '__qualname__', repr(obj))}={len(src)}:sha256:{digest}")
    joined = hashlib.sha256("\n".join(parts).encode()).hexdigest()
    return f"sha256:{joined}"


def _capturing() -> bool:
    """True when the current thread's CUDA stream is capturing a graph.

    The capture contract's enforcement point: tune-on-miss and store writes
    are illegal here (``do_bench`` synchronizes; a sync mid-capture aborts
    the capture). Monkeypatched by tests; False wherever torch is absent.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _config_key(config: dict) -> str:
    """Canonical JSON key for a config (bench-number table index)."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _sanitize(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "-" for c in name)


_MEMO: dict = {}  # (op, identity triple, source fp, shape_class) -> (config, status)
_FILES: dict = {}  # (op, capability, resolved store root) -> records dict
_LOCK = threading.RLock()


def reset_memo(op: str | None = None) -> None:
    """Forget in-process resolutions (tests; simulating a fresh process).

    The memo is the process-lifetime freeze that keeps a bucket's config
    static across capture and replay; clearing it makes the next
    :func:`op_config` re-read the store (and re-tune on a miss).
    """
    with _LOCK:
        if op is None:
            _MEMO.clear()
        else:
            for key in [k for k in _MEMO if k[0] == op]:
                _MEMO.pop(key, None)


def _store_path(op, device, store_dir) -> Path:
    capability = _sanitize(str(device.get("capability") or "unknown"))
    return store_root(store_dir) / f"{_sanitize(op)}.{capability}.json"


def _load_records(op, device, store_dir) -> dict:
    """The op's records from disk (empty on any absent/foreign/corrupt file)."""
    path = _store_path(op, device, store_dir)
    cache_key = (op, str(device.get("capability")), str(path))
    if cache_key in _FILES:
        return _FILES[cache_key]
    records: dict = {}
    if path.exists():
        try:
            doc = json.loads(path.read_text())
            if doc.get("schema") == SCHEMA and isinstance(doc.get("records"), dict):
                records = doc["records"]
            else:
                logger.info("op-config cache: %s has foreign schema; re-tuning", path)
        except (OSError, ValueError):
            logger.info("op-config cache: %s unreadable; re-tuning", path)
    _FILES[cache_key] = records
    return records


def _write_records(op, device, records, store_dir) -> bool:
    """Atomically persist one op's whole record map. Lenient on I/O errors."""
    path = _store_path(op, device, store_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with store_lock(path):
            # merge the latest disk state: other processes may have added
            # records for other buckets (or even this one) meanwhile
            disk = {}
            if path.exists():
                try:
                    doc = json.loads(path.read_text())
                    if doc.get("schema") == SCHEMA:
                        disk = doc.get("records") or {}
                except (OSError, ValueError):
                    disk = {}
            merged = dict(disk)
            merged.update(records)
            doc = {
                "schema": SCHEMA,
                "op": op,
                "device": device,
                "records": merged,
                "updated": _now(),
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
        cache_key = (op, str(device.get("capability")), str(path))
        _FILES[cache_key] = merged
        return True
    except OSError as exc:
        logger.info("op-config cache: persisting %s failed: %s", path, exc)
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def op_config(op, shape_class, *, default, candidates=(), bench=None,
              source_files=(), source_objs=(), budget_ms=None,
              store_dir=None, device=None):
    """Resolve one op's launch config for a shape class — cache-first.

    Resolution order (first hit wins, then frozen for the process):

    1. the process memo (static across capture and replay by construction);
    2. ``VKERNELS_CACHE=off`` → ``(default, "off")`` — no store, no sweep;
    3. an active CUDA graph capture → ``(default, "capture")`` — never
       bench, never write mid-capture (the floe ordering contract's
       enforcement half);
    4. a valid store record for (op, arch, source fingerprint, shape
       class) → its config with the record's status (``tuned``/``seeded``/
       ``default``) — zero benchmarking;
    5. tune-on-miss: bench each candidate at the LIVE shape via ``bench``,
       under the bounded budget; persist and return the winner as
       ``tuned`` — or, on budget exhaustion / total bench failure / an
       empty candidate space, persist and return the declared default as
       ``default`` with the reason.

    Parameters
    ----------
    op:
        Store identity, e.g. ``"glm_gemv.dense_gemv"`` — file stem.
    shape_class:
        Coarse bucket tag from the op's own tier function (tree gating
        conventions: ``t8``/``t64``/``prefill``-style tiers, dim tiers —
        NOT exact shapes).
    default:
        The op's DECLARED default config (its pin-or-heuristic choice —
        the fallback for ``off``, capture, budget exhaustion and bench
        failure; also what a seeded/tuned record must beat to be useful).
        A flat JSON-serializable dict.
    candidates:
        The sweep space, default first by convention (the declared default
        is usually candidate 0, so a sweep that confirms it still records
        its measured cost). Every candidate must be JSON-serializable —
        this is the store's config vocabulary.
    bench:
        ``callable(candidate) -> milliseconds`` benchmarking ONE candidate
        at the live shape (usually a ``triton.testing.do_bench`` closure
        over the op's own launch). Called only during step 5, only outside
        capture. Exceptions from one candidate prune it; if every
        candidate fails, the default is recorded (``bench-failed``).
    source_files, source_objs:
        Producer fingerprints (module files / functions) — a change is a
        miss. Ops pass ``( __file__, )``.
    budget_ms:
        Sweep budget in milliseconds for THIS (op, bucket) resolution
        (default from :func:`tune_budget`). Per-op totals are the sum over
        buckets the warmup actually touches.
    store_dir:
        Overrides :func:`store_root` (tests, frozen shared stores).
    device:
        Injectable arch identity (tests / multi-device callers); defaults
        to :func:`arch_identity`.

    Returns ``(config, status)``. The config dict is shared — treat it as
    read-only. Never raises for operational failures (store I/O, bench);
    raises only for a non-serializable config vocabulary or a missing
    ``default`` (programmer errors).
    """
    if not isinstance(default, dict):
        raise TypeError(f"op_config({op!r}): default must be a config dict")
    fp = source_fingerprint(source_files, source_objs)
    dev = dict(device) if device is not None else arch_identity()
    key = (op, _identity_key(dev), fp, str(shape_class))
    hit = _MEMO.get(key)
    if hit is not None:
        return hit
    with _LOCK:
        hit = _MEMO.get(key)
        if hit is not None:
            return hit
        resolved = _resolve(op, str(shape_class), default, candidates, bench,
                            fp, dev, budget_ms, store_dir)
        _MEMO[key] = resolved
        return resolved


def _resolve(op, shape_class, default, candidates, bench, fp, dev, budget_ms, store_dir):
    # validate the config vocabulary loudly: the store is JSON, and a
    # config that cannot round-trip would silently lose the sweep winner
    json.dumps(default)
    cands = []
    for cand in candidates:
        json.dumps(cand)
        cands.append(dict(cand))
    if not enabled():
        return dict(default), OFF
    if _capturing():
        # capture contract: static config, instantly — benching here would
        # sync mid-capture and abort it; writing would stall the capture
        return dict(default), CAPTURE

    records = _load_records(op, dev, store_dir)
    record = records.get(shape_class)
    if record is not None and record.get("source") == fp:
        stale = _identity_key(record.get("device") or {}) != _identity_key(dev)
        if not stale and record.get("status") in (TUNED, SEEDED, DEFAULT) \
                and isinstance(record.get("config"), dict):
            status = record["status"]
            logger.debug("op-config cache: %s[%s] %s (store hit)", op, shape_class, status)
            return dict(record["config"]), status
        logger.info("op-config cache: %s[%s] record stale (device/source); re-tuning",
                    op, shape_class)

    space = (not cands) or bench is None
    if space:
        _persist(op, shape_class, dev, fp, default, DEFAULT, None,
                 {"reason": "no-candidates" if not cands else "no-bench",
                  "candidates": {}, "elapsed_ms": 0.0, "budget_ms": budget_ms or 0.0},
                 store_dir)
        return dict(default), DEFAULT

    limit = tune_budget(budget_ms)
    t0 = time.monotonic()
    measured: dict[str, float] = {}
    winner, winner_ms, last_err, exhausted = None, math.inf, None, False
    for cand in cands:
        if (time.monotonic() - t0) * 1e3 >= limit:
            exhausted = True
            break  # budget spent mid-sweep: the sweep did not finish
        try:
            ms = float(bench(cand))
        except Exception as exc:  # a config that cannot run is pruned, not fatal
            last_err = exc
            continue
        if not math.isfinite(ms) or ms <= 0:
            last_err = ValueError(f"bench returned {ms!r}")
            continue
        measured[_config_key(cand)] = ms
        if ms < winner_ms:
            winner, winner_ms = cand, ms
    elapsed = (time.monotonic() - t0) * 1e3
    if exhausted or winner is None:
        # spec semantic: an unfinished sweep never ships a partial winner
        # (machine-speed-dependent) — the declared default wins,
        # deterministically; the partial numbers stay in the record for audit
        reason = "budget-exhausted" if exhausted else "bench-failed"
        logger.info("op-config cache: %s[%s] default (%s; %.0f ms budget %.0f)",
                    op, shape_class, reason, elapsed, limit)
        _persist(op, shape_class, dev, fp, default, DEFAULT, None,
                 {"reason": reason, "candidates": measured,
                  "elapsed_ms": elapsed, "budget_ms": limit,
                  "error": repr(last_err) if last_err else None}, store_dir)
        return dict(default), DEFAULT
    logger.info("op-config cache: %s[%s] tuned %s (%.4f ms; %d candidate(s), %.0f ms)",
                op, shape_class, _config_key(winner), winner_ms, len(measured), elapsed)
    _persist(op, shape_class, dev, fp, winner, TUNED, winner_ms,
             {"reason": None, "candidates": measured,
              "elapsed_ms": elapsed, "budget_ms": limit}, store_dir)
    return dict(winner), TUNED


def _persist(op, shape_class, dev, fp, config, status, time_ms, bench_block, store_dir):
    """Merge one record into the op's store file (lenient; memo unfrozen)."""
    record = {
        "config": dict(config),
        "status": status,
        "time_ms": time_ms,
        "device": dict(zip(("capability", "sm_count", "name"), _identity_key(dev))),
        "source": fp,
        "tuned_at": _now(),
        "origin": "op-config-tune",
        "bench": bench_block,
    }
    _write_records(op, dev, {shape_class: record}, store_dir)


def seed(op, entries, *, source_files=(), source_objs=(), producer="seeded",
         store_dir=None, device=None, force=False) -> dict:
    """Land offline-produced population entries (lane-29 H100 tables, pins).

    ``entries`` maps shape-class → ``{"config": {...}, "time_ms": <ms or
    None>, "origin": "<producing run>", ...}`` — the SAME record schema the
    tuner writes, with ``status: "seeded"``. A seeded record serves the
    bucket until a local tune overwrites it; a locally ``tuned`` record is
    never downgraded by a seed unless ``force=True``. The source
    fingerprint is computed against the CURRENT tree — seed tables must be
    produced against the tree they seed (they are priors, and any code
    change re-tunes over them).

    Returns the persisted record map (for audit/tests).
    """
    fp = source_fingerprint(source_files, source_objs)
    dev = dict(device) if device is not None else arch_identity()
    if not enabled():
        return {}
    with _LOCK:
        records = dict(_load_records(op, dev, store_dir))
        landed = {}
        for shape_class, entry in entries.items():
            existing = records.get(str(shape_class))
            if existing is not None and existing.get("status") == TUNED and not force:
                continue
            record = {
                "config": dict(entry["config"]),
                "status": SEEDED,
                "time_ms": entry.get("time_ms"),
                "device": dict(zip(("capability", "sm_count", "name"), _identity_key(dev))),
                "source": fp,
                "tuned_at": _now(),
                "origin": str(entry.get("origin") or producer),
                "bench": {"reason": "seeded", "candidates": {},
                          "elapsed_ms": 0.0, "budget_ms": 0.0},
            }
            records[str(shape_class)] = record
            landed[str(shape_class)] = record
        if landed and _write_records(op, dev, landed, store_dir):
            reset_memo(op)
        return records


def stored_records(op, *, store_dir=None, device=None) -> dict:
    """Read one op's persisted record map (inspection, tests, ``vkl tune``)."""
    dev = dict(device) if device is not None else arch_identity()
    return dict(_load_records(op, dev, store_dir))
