# Persistent autotune (`vkernels.torch_ops.tuning_cache`)

Triton's `@triton.autotune` re-benchmarks every config in every fresh
process and discards the result at exit. Warmup paths pay the full sweep
on every boot (and inherit its graph-capture pollution hazards —
`mhc_projection` and `glm_projection` both document first-call trials
dirtying capture); a hand-tuned config table can only be produced by
copy-pasting benchmark output into source comments. The tuning cache
keeps the sweep's winner.

The same idea covers the **native tier**: the C++/HIP/CUDA kernels in
`src/c/` don't autotune at all — their launch configs are heuristic
formulas frozen in the headers, re-fitted by hand after each measured
sweep (`docs/performance/dsa/gfx942.md`). The native store replaces the
copy-paste step: the sweep benches write the winner, the launchers read
it back. See [Native tier](#native-tier-c-hipcudakernels) below.

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

## Native tier (C++/HIP/CUDA kernels)

`src/c/vkernels/core/tuning.{hpp,cpp}` implements the same store for the
compiled kernels. One line-format sidecar per (kernel, device arch):

```
$VKERNELS_TUNING_CACHE/<kernel>.<arch>.tune     # e.g. dsa_topk_logits_split_for.sm121.tune
```

```
# vk-native-tuning/1
# arch=sm121 cu_count=48 written_by=bench_dsa_topk_logits
key=1,512,64
split=32
```

| Piece | What it does |
| --- | --- |
| `tuning::find(kernel, key)` | the persisted record, or `nullptr` on any miss. The arch token matches the Python tier (`gcnArchName` on HIP, `smXY` on CUDA); a single foreign-arch file is honored (one-machine rule), two are ambiguous and miss. `VKERNELS_TUNING_ARCH` pins the token (tests, cross-arch inspection). |
| `tuning::persist(kernel, key, params, written_by)` | merge-write the winner (atomic whole-file rewrite, unknown records kept). |
| `VKERNELS_TUNING_CACHE=off` | disables reads *and* writes — pure compiled-in formulas. |

**Seam contract:** config-selector formulas consult the store BEFORE their
heuristics; a persisted winner is ground truth for the device arch,
including prefill early-outs. The keys mirror the formula's arguments, so
a record is self-describing and a stale one is just a re-tune. First
adopter: `dsa_topk_logits_split_for` / `dsa_sparse_fwd_split_for`
(`kernels/dsa.cpp`) — the two formulas that previously had their measured
sweeps copy-pasted into comments (#137).

**Sweep harness:** the benches own the tuning loop. `bench_dsa_topk_logits
--persist[=<dir>]` sweeps split_kv per decode shape, prints the winner vs
the formula, and persists (`meta/benchmarks/bench_dsa_topk_logits.cu`).
On GB10 the measured winners override the formula at several decode
shapes (e.g. `bs=1, msl=512`: split 32, 10.8 us vs formula's 8 at 47 us).
Extend the same `--persist` pattern to the other bench binaries as their
kernels get tunable knobs.

Everything here is lenient — a stale/malformed/foreign record is a
re-tune, never an error (the fail-loud frozen-artifact contract stays in
`tuning_manifest`, and the auditable Python tier is `torch_ops/`).
Unit tests: `tests/core/test_tuning.cpp` (C++) and
`tests/python/test_tuner.py` (the Python mirror — a sidecar written by
either tier must parse byte-compatibly in the other).

## One tuner (`vkl tune`)

`torch_ops/tuner.py` is the single surface over both tiers. Its
`REGISTRY` catalogs every tunable kernel with the artifact that tunes it
(a sweep callable for Triton kernels, a persist-capable bench binary for
native ones); the CLI drives it:

```
vkl tune status               # every store artifact, both tiers + registry coverage
vkl tune run <name>|--all     # sweep and persist (in-process for Triton, bench for native)
vkl tune clear <name>|--all   # delete stored configs (both tiers)
```

Adding a kernel to the tuner means two edits: register it in
`REGISTRY` (with its sweep hook or bench binary), and make the sweep
write the store — `@persistent_autotune` for Triton kernels,
`--persist` in the bench for native ones. Kernels whose harness cannot
persist yet are cataloged as `formula-only` and skipped with a reason
rather than mis-run.

The arch token is canonical across tiers: HIP `gcnArchName` (feature
flags stripped), CUDA `sm<major><minor>` — pinned by
`VKERNELS_TUNING_ARCH`. (Torch on CUDA >= 13 exposes a `gcnArchName`
attribute even on NVIDIA; `device_fingerprint()` ignores it there so
both tiers name a device identically.)

## Build cache

Two orthogonal caches; pin both for fully warm builds:

* **Object cache** — `ccache` in front of the compilers:
  `VKERNELS_USE_CCACHE` (ON by default; auto-detects `ccache`, logs it in
  the configure banner). CMake sets `CMAKE_{C,CXX,CUDA}_COMPILER_LAUNCHER`
  for every target, so rebuilds across presets/branches share
  `$CCACHE_DIR` (default `~/.cache/ccache`). Install ccache and the first
  build populates it; `ccache -s` for hit rates.
* **Build tree** — every preset pins `binaryDir` to `build/<preset>` and
  the Makefile wraps it (`make configure|build|test P=cuda`), so branch
  switches reuse the configured tree instead of re-running CMake from
  scratch. The build dir is disposable; the object cache above is what
  makes recreating it cheap.
