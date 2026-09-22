# Persistent autotune (`vkernels.torch_ops.tuning_cache`)

Triton's `@triton.autotune` re-benchmarks every config in every fresh
process and discards the result at exit. Warmup paths pay the full sweep
on every boot (and inherit its graph-capture pollution hazards —
`mhc_projection` and `glm_projection` both document first-call trials
dirtying capture); a hand-tuned config table can only be produced by
copy-pasting benchmark output into source comments. The tuning cache
keeps the sweep's winner.

## The store

One JSON file per (kernel, device):

```
$VKERNELS_TUNING_CACHE/<kernel>.<arch>.json     # default ~/.cache/vkernels/tuning
```

Schema `vk-tuning-store/1`, in the spirit of the frozen-artifact
manifests (#67, `tuning_manifest.py`):

| Field | Meaning |
| --- | --- |
| `kernel`, `schema` | store identity + format version |
| `device` | name, arch (gcnArchName or `smXY`), CU count, torch/triton/HIP versions — **no device index** (ordinals are not stable across boots or `CUDA_VISIBLE_DEVICES`) |
| `records` | per-key: chosen config (kwargs + num_warps + num_stages), measured `time_ms`, device block, producer fingerprints (sha256 of the producing sources), `measured_at` |

Writes are atomic (temp file + `os.replace`); concurrent sweeps
last-writer-wins per key, benign for benchmark winners.

## The decorator

`persistent_autotune` is a drop-in for `triton.autotune`:

```python
from .tuning_cache import persistent_autotune

@persistent_autotune(
    configs=[triton.Config({"ROWS": r}, num_warps=w) for r in (1, 2, 4) for w in (4, 8)],
    key=["TOKENS", "DEVICE"],
    kernel_name="mhc_projection",
    source_files=[__file__],
)
@triton.jit
def partial(...): ...
```

* **Store hit** → the persisted config is replayed with zero
  benchmarking. A fresh process (or a re-created kernel, e.g. after
  `lru_cache` eviction) costs one JSON read.
* **Store miss** → every config is benchmarked (`triton.testing.do_bench`)
  once, the winner is persisted, and the launch proceeds. Grid callables
  receive each trial config's constexprs, exactly like Triton's own
  autotuner.
* `.cache` maps key tuples to the winning `triton.Config` — the same
  introspection surface `triton.autotune` exposes (reports keep working).

## Lenient by design

A device/software mismatch, a changed producer fingerprint, or an
unknown schema is a **cache miss (re-tune)**, never an error: these are
local performance accelerators, not deployment artifacts, and a stale
entry must degrade to the historical behavior, not break the boot. The
fail-loud contract for *frozen, shared* artifacts lives in
[tuning-qualification](tuning-qualification.md) — pass
`strict=True` (optionally with a repo-path `store_dir`) to get
`TuningCacheError` on any mismatch for auditable stores.

## Knobs

* `VKERNELS_TUNING_CACHE=<dir>` — store root.
* `VKERNELS_TUNING_CACHE=off` — pure in-process autotune (historical
  behavior); nothing is read or written.
* The Triton JIT **binary** cache (`TRITON_CACHE_DIR`) is orthogonal and
  unaffected: this module persists *choices*, Triton persists *cubins*.
  Pin both for fully warm boots.

## Adoption

Exemplar: `torch_ops/mhc_projection.py`. To migrate another kernel,
replace `@triton.autotune(configs=..., key=...)` with
`@persistent_autotune(configs=..., key=..., kernel_name=<unique stem>,
source_files=[__file__])`. Keys must be JSON-able scalars (ints, bools,
strings — the constexprs that shape the launch); the per-key values are
stored verbatim, so do not key on tensor arguments.

Deleting a kernel's stored choices: `TuningCache(kernel_name).clear()`.
