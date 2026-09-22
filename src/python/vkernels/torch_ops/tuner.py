"""The single tuner: one surface for every tunable kernel in the codebase.

Two tiers write into the same store root (``$VKERNELS_TUNING_CACHE``,
default ``~/.cache/vkernels/tuning``, ``=off`` disables both):

* **triton** — ``@persistent_autotune`` kernels (``tuning_cache.py``);
  the sweep runs in-process on first launch of each key shape, winners
  land in ``<kernel>.<arch>.json``.
* **native** — the C++/HIP/CUDA kernels (``core/tuning.hpp``); the sweep
  runs in a bench binary (``meta/benchmarks/*`` ``--persist``), winners
  land in ``<kernel>.<arch>.tune`` line-format sidecars that the C++
  config selectors read back.

This module is the layer that makes them ONE tuner:

* :data:`REGISTRY` catalogs every tunable kernel with its tier and the
  artifact that tunes it (a sweep callable for Triton, a bench binary
  for native);
* :func:`tune` drives the sweep for one or all kernels;
* :func:`status` reads both tiers' artifacts into one report;
* :func:`clear` removes a kernel's records from both tiers.

The CLI exposes all three as ``vkl tune status|run|clear``. Store I/O for
the native tier lives in :class:`NativeStore`, which mirrors the C++
parser in ``core/tuning.cpp`` rule for rule (a sidecar written here must
be readable there and vice versa).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .tuning_cache import TuningCacheError, default_store_dir, device_fingerprint

NATIVE_SCHEMA = "vk-native-tuning/1"


def native_arch() -> str:
    """The native tier's arch token — the same string C++ writes.

    ``VKERNELS_TUNING_ARCH`` pins it (tests, cross-arch artifact
    inspection); otherwise the torch device query. ``""`` when neither
    is available (host-only box).
    """
    env = os.environ.get("VKERNELS_TUNING_ARCH", "").strip()
    if env:
        return env
    try:
        return device_fingerprint()["arch"]
    except Exception:
        return ""


def _safe_arch(arch: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in arch)


# ---------------------------------------------------------------------------
# The native (.tune) sidecar — mirrors vkernels/core/tuning.cpp exactly.
# ---------------------------------------------------------------------------


def _parse_key_list(text: str) -> tuple[int, ...] | None:
    cells = text.split(",")
    key = []
    for cell in cells:
        try:
            key.append(int(cell))
        except ValueError:
            return None  # malformed key line: anchors nothing
    return tuple(key) if key else None


def _parse_param(text: str) -> tuple[str, int] | None:
    name, sep, raw = text.partition("=")
    if not sep or not name:
        return None
    try:
        return name, int(raw)
    except ValueError:
        return None  # malformed value line: skipped (lenient)


def parse_tune_body(text: str) -> dict[tuple[int, ...], dict[str, int]]:
    """Parse one ``.tune`` body — same semantics as C++ ``parse_body``.

    ``key=`` lines start records; ``name=value`` lines attach to the
    current record. A malformed key line anchors nothing, so orphan
    params after it are DROPPED, never glued onto the record above.
    Comments and blank lines are ignored.
    """
    records: dict[tuple[int, ...], dict[str, int]] = {}
    cur: tuple[int, ...] | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip(" \t\r")
        if not line:
            continue
        if line.startswith("key="):
            parsed = _parse_key_list(line[4:])
            cur = None
            if parsed is not None:
                records[parsed] = {}
                cur = parsed
            continue
        param = _parse_param(line)
        if param is not None and cur is not None:
            records[cur][param[0]] = param[1]
    return records


def parse_tune_arch(text: str, filename: str) -> str:
    """The arch a sidecar carries: its ``# arch=`` header, else the filename."""
    for raw in text.splitlines():
        comment = raw.find("#")
        if comment < 0:
            break
        marker = raw.find("arch=", comment)
        if marker >= 0:
            return raw[marker + 5:].split()[0]
    stem = filename[: filename.rfind(".tune")] if filename.endswith(".tune") else filename
    dot = stem.find(".")
    return stem[dot + 1:] if 0 <= dot else ""


class NativeStore:
    """Read/write one native kernel's ``.tune`` sidecars.

    Same file contract as ``core/tuning.hpp``: the arch-matching file is
    authoritative; a single foreign-arch file is honored (one-machine
    rule); multiple foreign files are ambiguous and read as empty.
    Writes are atomic (temp file + ``os.replace``) and keep every record
    the file already carries.
    """

    def __init__(self, kernel_name, *, store_dir=None, arch=None):
        self.kernel_name = kernel_name
        self.store_dir = Path(store_dir) if store_dir else default_store_dir()
        self.arch = arch if arch is not None else native_arch()

    # -- files ---------------------------------------------------------------

    def _files(self) -> dict[str, Path]:
        """filename -> path for every sidecar of this kernel."""
        if not self.store_dir.is_dir():
            return {}
        prefix = self.kernel_name + "."
        return {
            p.name: p
            for p in self.store_dir.iterdir()
            if p.name.startswith(prefix) and p.name.endswith(".tune")
        }

    def path(self) -> Path | None:
        """The authoritative sidecar for this arch (see the class docstring)."""
        archs = {}
        for name, path in self._files().items():
            archs[parse_tune_arch(path.read_text(errors="replace"), name)] = path
        if self.arch and self.arch in archs:
            return archs[self.arch]
        return next(iter(archs.values())) if len(archs) == 1 else None

    # -- records -------------------------------------------------------------

    def records(self) -> dict[tuple[int, ...], dict[str, int]]:
        path = self.path()
        if path is None:
            return {}
        return parse_tune_body(path.read_text(errors="replace"))

    def upsert(self, key, params, *, written_by="tuner") -> dict[str, int]:
        """Merge one winner (atomic whole-file rewrite)."""
        if not self.arch:
            raise TuningCacheError(
                f"{self.kernel_name}: no device arch to name the sidecar "
                "(set VKERNELS_TUNING_ARCH or run on a GPU host)")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        path = self.store_dir / f"{self.kernel_name}.{_safe_arch(self.arch)}.tune"
        records = parse_tune_body(path.read_text(errors="replace")) if path.exists() else {}
        records[tuple(int(k) for k in key)] = {str(n): int(v) for n, v in params.items()}

        try:
            cu_count = device_fingerprint()["cu_count"]
        except Exception:
            cu_count = 0
        header = f"# {NATIVE_SCHEMA}\n# arch={self.arch}"
        if cu_count > 0:
            header += f" cu_count={cu_count}"
        header += f" written_by={written_by}\n"
        body = [header]
        for key_tuple, params_map in records.items():
            body.append("key=" + ",".join(str(v) for v in key_tuple))
            body.extend(f"{name}={value}" for name, value in params_map.items())

        fd, tmp = tempfile.mkstemp(dir=self.store_dir, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as out:
                out.write("\n".join(body) + "\n")
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
        return dict(records[tuple(int(k) for k in key)])

    def clear(self) -> bool:
        """Remove every sidecar of this kernel. True when one was removed."""
        removed = False
        for path in self._files().values():
            path.unlink()
            removed = True
        return removed


# ---------------------------------------------------------------------------
# The registry: every tunable kernel, one row per tuning artifact.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tunable:
    name: str                # the store kernel name (both tiers use it)
    tier: str                # "triton" | "native"
    description: str
    toolkit: str = "any"     # "cuda" | "hip" — hardware the sweep needs
    sweep: str | None = None  # triton: "module:callable" driving the sweep
    bench: str | None = None  # native: meta/benchmarks executable (persist-capable)
    persists: bool = True     # False: the harness cannot write the store yet
    store_names: tuple[str, ...] = ()  # triton: store files a pipeline sweep
                                      # fills (defaults to (name,))


REGISTRY = (
    Tunable(
        "mhc_projection", "triton",
        "mHC projection (BF16 GEMV+reduce); keys are (device, rows in {1,2})",
        sweep="vkernels.torch_ops.mhc_projection:mhc_projection_tune",
    ),
    Tunable(
        "dsa_topk_logits_split_for", "native", toolkit="cuda",
        description="DSA indexer split_kv selector (padded-MQA top-k logits)",
        bench="dsa_topk_logits_bench_cuda",
    ),
    Tunable(
        "dsa_sparse_fwd_split_for", "native", toolkit="hip",
        description="DSA sparse-MLA split_kv selector (issue #137 optima)",
        bench="dsa_bench_split137", persists=False,
    ),
    Tunable(
        "glm_projection", "triton",
        "GLM decode projection GEMV+reduce; keys are (N,K,TOKENS,DEVICE)",
        sweep="vkernels.torch_ops.glm_projection:glm_projection_tune",
    ),
    Tunable(
        "qkv_projection", "triton",
        "QKV decode projection GEMV; keys are (TOKENS,DEVICE)",
        sweep="vkernels.torch_ops.qkv_projection:qkv_projection_tune",
    ),
    Tunable(
        "kda_chunk_nv", "triton", toolkit="cuda",
        description=(
            "chunked-KDA pipeline (NVIDIA): one end-to-end sweep populates "
            "the store files for chunk_local_cumsum_vector_kernel, "
            "solve_tril_16x16_kernel, merge_16x16_to_{32,64}x64_inverse_kernel, "
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64, "
            "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_{intra,inter}, "
            "kda_recompute_w_u, chunk_gla_fwd_kernel_o"),
        sweep="vkernels.torch_ops.vllm_kda:kda_tune",
        store_names=(
            "chunk_local_cumsum_vector_kernel", "solve_tril_16x16_kernel",
            "merge_16x16_to_32x32_inverse_kernel",
            "merge_16x16_to_64x64_inverse_kernel",
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            "kda_scaled_dot_kkt_intra_sub_intra",
            "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter",
            "kda_recompute_w_u", "chunk_gla_fwd_kernel_o",
        ),
    ),
    Tunable(
        "kda_chunk_amd", "triton", toolkit="hip",
        description=(
            "KDA pipelines (ROCm): chunked sweep populates the AMD chunk "
            "kernel stores (incl. the shared ones) + l2norm_fwd_kernel{,1} "
            "and fused_recurrent_kda_fwd_kernel on the decode leg"),
        sweep="vkernels.torch_ops.vllm_kda_amd:kda_tune",
        store_names=(
            "l2norm_fwd_kernel1", "l2norm_fwd_kernel",
            "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter",
            "chunk_gla_fwd_kernel_o", "kda_gate_cumsum_fwd_kernel",
            "kda_gate_fwd_kernel", "kda_scaled_dot_kkt_intra_sub_intra",
            "kda_recompute_w_u",
        ),
    ),
)


def repo_root() -> Path:
    """The repository root — same resolution rules as the ``vkl`` CLI."""
    from vkernels import discovery

    return discovery.resolve_root(None)


def find_bench_binary(name: str) -> Path | None:
    """A built bench executable under ``build/*/meta/benchmarks/``, newest first."""
    candidates = [
        p
        for p in (repo_root() / "build").glob(f"*/meta/benchmarks/{name}")
        if p.is_file() and os.access(p, os.X_OK)
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


# ---------------------------------------------------------------------------
# Driver: tune / status / clear
# ---------------------------------------------------------------------------


def _resolve_sweep(spec: str):
    import importlib

    module_name, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _toolkit_mismatch(toolkit: str) -> str | None:
    """Reason string when ``toolkit`` cannot run on this host, else None."""
    if toolkit == "any":
        return None
    try:
        import torch

        is_hip = bool(torch.version.hip)
    except Exception:
        return f"{toolkit} toolkit unavailable (no torch)"
    if (toolkit == "hip") != is_hip:
        return (f"sweep needs {toolkit}; this torch is "
                + ("ROCm" if is_hip else "CUDA"))
    return None


def tune(names=(), *, store_dir=None, all=False) -> list[dict]:
    """Run the sweep for the named (or all) registry kernels.

    Triton kernels sweep in-process (the bench is the launch itself);
    native kernels run their bench binary with ``--persist``. Returns
    one report row per kernel: ``{"name", "tier", "ok", "detail"}``.
    """
    targets = list(REGISTRY)
    if names and not all:
        wanted = set(names)
        unknown = wanted - {t.name for t in targets}
        if unknown:
            raise TuningCacheError(
                f"unknown tunable kernel(s): {sorted(unknown)}; "
                f"registry has {[t.name for t in targets]}")
        targets = [t for t in targets if t.name in wanted]

    store = Path(store_dir) if store_dir else default_store_dir()
    reports = []
    for entry in targets:
        row = {"name": entry.name, "tier": entry.tier}
        try:
            if not entry.persists:
                if all:
                    # Batch runs tune what they can: a harness that cannot
                    # write the store is a skip, not a failure.
                    row.update(ok=True, skipped=True, detail=(
                        "skipped: sweep harness does not write the store yet "
                        f"({entry.bench}, {entry.toolkit} on-site step)"))
                    reports.append(row)
                    continue
                row.update(ok=False, detail=(
                    "sweep harness does not write the store yet "
                    f"({entry.bench}, {entry.toolkit} on-site step)"))
                reports.append(row)
                continue
            mismatch = _toolkit_mismatch(entry.toolkit)
            if mismatch is not None:
                if all:
                    row.update(ok=True, skipped=True, detail=f"skipped: {mismatch}")
                    reports.append(row)
                    continue
                row.update(ok=False, detail=mismatch)
                reports.append(row)
                continue
            if entry.tier == "triton":
                _resolve_sweep(entry.sweep)()
                names = entry.store_names or (entry.name,)
                n = sum(_triton_record_count(name, store) for name in names)
                row.update(ok=True, detail=(
                    f"{n} record(s) in {store}" + ("" if len(names) == 1
                    else f" across {len(names)} kernel store(s)")))
            else:
                binary = find_bench_binary(entry.bench)
                if binary is None:
                    row.update(ok=False, detail=(
                        f"bench not built: {entry.bench} (configure with "
                        "VKERNELS_BUILD_BENCHMARKS=ON; needs "
                        f"{entry.toolkit} toolkit)"))
                else:
                    env = dict(os.environ, VKERNELS_TUNING_CACHE=str(store))
                    proc = subprocess.run(
                        [str(binary), f"--persist={store}"],
                        env=env, capture_output=True, text=True)
                    if proc.returncode != 0:
                        row.update(ok=False, detail=(
                            f"{binary.name} exited {proc.returncode}: "
                            f"{proc.stderr.strip()[-400:]}"))
                    else:
                        n = len(NativeStore(entry.name, store_dir=store).records())
                        row.update(ok=True, detail=f"{n} record(s); bench {binary}")
        except TuningCacheError as exc:
            row.update(ok=False, detail=str(exc))
        except Exception as exc:  # a failed sweep must not block the rest
            row.update(ok=False, detail=f"{type(exc).__name__}: {exc}")
        reports.append(row)
    return reports


def _triton_record_count(kernel_name, store_dir) -> int:
    arch = _safe_arch(native_arch() or "?")
    path = Path(store_dir) / f"{kernel_name}.{arch}.json"
    if not path.exists():
        return 0
    try:
        return len(json.loads(path.read_text())["records"])
    except (ValueError, KeyError):
        return 0


def status(*, store_dir=None) -> dict:
    """One report over both tiers' artifacts plus registry coverage."""
    store = Path(store_dir) if store_dir else default_store_dir()
    arch = native_arch()
    files = {"triton": [], "native": []}
    if store.is_dir():
        for path in sorted(store.iterdir()):
            if path.name.endswith(".json"):
                tier, stem = "triton", path.name[: -len(".json")]
            elif path.name.endswith(".tune"):
                tier, stem = "native", path.name[: -len(".tune")]
            else:
                continue
            dot = stem.rfind(".")
            if dot <= 0:
                continue
            try:
                records = (
                    _triton_doc(path) if tier == "triton"
                    else parse_tune_body(path.read_text(errors="replace"))
                )
            except (ValueError, KeyError):
                records = {}
            files[tier].append({
                "kernel": stem[:dot],
                "tier": tier,
                "file_arch": stem[dot + 1:],
                "records": len(records),
                "path": str(path),
            })

    registry_rows = []
    for entry in REGISTRY:
        stored = [f for f in files[entry.tier] if f["kernel"] == entry.name]
        registry_rows.append({
            "name": entry.name,
            "tier": entry.tier,
            "tuned": bool(stored),
            "persists": entry.persists,
            "detail": entry.description,
        })
    return {
        "store": str(store),
        "arch": arch,
        "enabled": os.environ.get("VKERNELS_TUNING_CACHE", "").lower() != "off",
        "stores": files["triton"] + files["native"],
        "registry": registry_rows,
    }


def _triton_doc(path: Path) -> dict:
    """The records dict of a Triton-tier store (lenient about schemas)."""
    doc = json.loads(path.read_text())
    if doc.get("schema") != "vk-tuning-store/1":
        return {}
    return doc["records"]


def clear(names=(), *, store_dir=None, all=False) -> list[dict]:
    """Remove records for the named (or all) kernels, both tiers."""
    targets = [t.name for t in REGISTRY] if all else list(names)
    store = Path(store_dir) if store_dir else default_store_dir()
    rows = []
    for name in targets:
        removed = {"triton": False, "native": False}
        for suffix, tier in ((".json", "triton"), (".tune", "native")):
            for path in store.glob(f"{name}.*{suffix}"):
                path.unlink()
                removed[tier] = True
        rows.append({"name": name, "cleared": any(removed.values()),
                     "tiers": [t for t, hit in removed.items() if hit]})
    return rows
