# Persistent tuning cache — how to use it

vkernels kernels come in two tiers, and both used to lose their tuning:

* **Triton** (`torch_ops/`) — `@triton.autotune` re-benchmarks every
  config in every fresh process and discards the result at exit. Warmup
  paths pay the full sweep on every boot (and inherit its graph-capture
  pollution hazards — `mhc_projection` and `glm_projection` both document
  first-call trials dirtying capture).
* **Native** (`src/c/`, the C++/HIP/CUDA kernels) — no autotune at all;
  launch configs are heuristic formulas re-fitted by hand after each
  measured sweep, with the results copy-pasted into source comments
  (issue #137).

The tuning cache persists the sweep's winner for both tiers in one store,
and `vkl tune` is the one command that drives it.

## Quick start (new box / new hardware)

```bash
# 0. Build once with the benches (native sweeps live in meta/benchmarks):
cmake --preset cuda -DVKERNELS_BUILD_BENCHMARKS=ON && cmake --build --preset cuda -j8

# 1. See what is tunable and what is already tuned:
vkl tune status

# 2. Sweep every registered kernel and persist the winners (one-time
#    per device arch; GPU required, takes ~minutes):
vkl tune run --all

# 3. Point serving at the store. Every launch now replays the stored
#    config — no benchmarking, no code changes:
export VKERNELS_TUNING_CACHE=~/.cache/vkernels/tuning   # this is the default
```

That's the whole workflow. The store is a deployable artifact: tar the
directory and ship it with boxes of the same GPU arch, or leave it local
— nothing needs `vkl tune` at serving time, launchers read the store on
their own (Triton kernels on first launch of each key shape, native
config selectors on every launch).

Measured stores captured in-tree: `meta/tuning-stores/` carries the
sidecars recorded on real hardware (MI300A, A100, GB10). Copy the file
for your arch into your store directory to start from measured configs
instead of the compiled-in formulas:

```bash
mkdir -p ~/.cache/vkernels/tuning
cp meta/tuning-stores/<kernel>.<arch>.tune ~/.cache/vkernels/tuning/
```

(In-repo, `vkl` is `make vkl ARGS='tune status'` or
`python3 -m vkernels.cli tune status`; the console script comes from
`pip install -e ./src`.)

**When to re-tune:** the records self-invalidate. A different device
arch, a torch/triton/HIP upgrade, or changed producer sources turns the
record into a **miss** — the next launch re-sweeps that key and
re-persists. It never errors and never picks a config tuned for other
hardware; delete the store (`vkl tune clear --all`) only to force a full
clean sweep sooner.

## Commands

```
vkl tune status               # every store artifact, both tiers + registry coverage
vkl tune run <name>|--all     # sweep and persist (in-process for Triton, bench for native)
vkl tune clear <name>|--all   # delete stored configs (both tiers)
```

`status` output on a tuned GB10 (plain formatter; with
[rich](https://rich.readthedocs.io/) installed the same report renders
as aligned tables — the CLI uses rich when importable and falls back to
the stdlib formatter otherwise, mirroring `_backend.py`'s fallback
pattern):

```
store: ~/.cache/vkernels/tuning  arch: sm121  enabled
  tier    kernel                           arch       records  path
  triton  mhc_projection                   sm121            2  mhc_projection.sm121.json
  native  dsa_topk_logits_split_for        sm121            4  dsa_topk_logits_split_for.sm121.tune
  registry:
    triton  mhc_projection                   tuned         mHC projection (BF16 GEMV+reduce); ...
    native  dsa_topk_logits_split_for        tuned         DSA indexer split_kv selector ...
    native  dsa_sparse_fwd_split_for         formula-only  DSA sparse-MLA split_kv selector ...
```

* `tuned` — a store artifact exists for this kernel.
* `untuned` — registered but never swept; `vkl tune run <name>` fills it.
* `formula-only` — registered, but the sweep harness cannot write the
  store yet (needs on-site hardware); `run` skips it with a reason
  instead of mis-running. The kernel still uses its compiled-in formula.

`run` is idempotent and incremental: an existing record for a key is
re-measured and replaced only by the (possibly same) winner. A failing
kernel's sweep never blocks the others — the report marks it `FAIL` with
the reason and moves on. A `formula-only` entry is a `FAIL` when named
explicitly (you asked for it, it can't run) but only a `SKIP` under
`--all` (batch runs tune what they can), so the quick-start above exits
clean.

## Environment variables

| Variable | Effect |
| --- | --- |
| `VKERNELS_TUNING_CACHE=<dir>` | store root. Default `~/.cache/vkernels/tuning`. |
| `VKERNELS_TUNING_CACHE=off` | disable reads *and* writes — pure in-process autotune / compiled-in formulas (historical behavior). |
| `VKERNELS_TUNING_ARCH=<token>` | pin the arch token used in file names (tests, cross-arch artifact inspection). Otherwise: HIP `gcnArchName` (feature flags stripped), CUDA `sm<major><minor>`. |
| `TRITON_CACHE_DIR` | orthogonal: Triton's **binary** (cubin) cache. This module persists *choices*, Triton persists *cubins* — pin both for fully warm boots. |
| `CCACHE_DIR` / `VKERNELS_USE_CCACHE` | the build cache (see [Build cache](#build-cache)). |

The arch token is canonical across tiers — both name a device
identically. (Torch on CUDA >= 13 exposes a `gcnArchName` attribute even
on NVIDIA holding the marketing name; the Python tier ignores it there.)

## Store layout

One artifact per (kernel, device arch), two formats in one directory:

```
$VKERNELS_TUNING_CACHE/
  mhc_projection.sm121.json                  # triton tier (schema vk-tuning-store/1)
  dsa_topk_logits_split_for.sm121.tune       # native tier (vk-native-tuning/1)
```

The native sidecar is line-based (the C++ reader takes no dependencies):

```
# vk-native-tuning/1
# arch=sm121 cu_count=48 written_by=bench_dsa_topk_logits
key=1,512,64
split=32
```

Writes are atomic (temp + rename) in both tiers; concurrent sweeps
last-writer-wins per key, benign for benchmark winners. The JSON tier
also stores the measured time, the full device/software block, and
sha256 producer fingerprints — see [The decorator](#the-decorator) and
[Lenient by design](#lenient-by-design).

---

# Reference

The sections below are the design/implementation detail behind the
workflow above.

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

## The Triton store schema

```
$VKERNELS_TUNING_CACHE/<kernel>.<arch>.json
```

Schema `vk-tuning-store/1`, in the spirit of the frozen-artifact
manifests (#67, `tuning_manifest.py`):

| Field | Meaning |
| --- | --- |
| `kernel`, `schema` | store identity + format version |
| `device` | name, arch, CU count, torch/triton/HIP versions — **no device index** (ordinals are not stable across boots or `CUDA_VISIBLE_DEVICES`) |
| `records` | per-key: chosen config (kwargs + num_warps + num_stages), measured `time_ms`, device block, producer fingerprints (sha256 of the producing sources), `measured_at` |

## Native tier (C++/HIP/CUDA kernels)

`src/c/vkernels/core/tuning.{hpp,cpp}` implements the store for the
compiled kernels. Reading:

| Piece | What it does |
| --- | --- |
| `tuning::find(kernel, key)` | the persisted record as an owned snapshot (`std::shared_ptr<const Entry>`), or null on any miss. A single foreign-arch file is honored (one-machine rule), two are ambiguous and miss. |
| `tuning::persist(kernel, key, params, written_by)` | merge-write the winner (atomic whole-file rewrite, unknown records kept). |
| `VKERNELS_TUNING_CACHE=off` | disables reads *and* writes — pure compiled-in formulas. |

**Seam contract:** config-selector formulas consult the store BEFORE
their heuristics; a persisted winner is ground truth for the device
arch, including prefill early-outs. The keys mirror the formula's
arguments, so a record is self-describing and a stale one is just a
re-tune. First adopters: `dsa_topk_logits_split_for` /
`dsa_sparse_fwd_split_for` (`kernels/dsa.cpp`) — the two formulas that
previously had their measured sweeps copy-pasted into comments (#137) —
since joined by `dsa_sparse_fwd_tile` (`dsa.hip`/`dsa.cpp`, consulted
by the HIP dispatch itself), `mla_fwd_split_for` (`kernels/mla.cpp`,
on every MLA launch), and `glm_fp8_gemv_pick_sk` (`kernels/glm_moe.cpp`,
host arithmetic shared by the public API and the internal pick).
`find` returns an owned snapshot, so a `persist` from another thread
cannot invalidate a dispatch site's read. Measured effect on GB10: the
sweep's winners override the formula at several decode shapes (e.g.
`bs=1, msl=512`: split 32, 10.8 us vs the formula's 8 at 47 us).

**Sweep harnesses:** the benches own the tuning loop —
`bench_dsa_topk_logits --persist[=<dir>]` sweeps split_kv per decode
shape, prints the winner vs the formula, and writes the store
(`meta/benchmarks/bench_dsa_topk_logits.cu`). `vkl tune run` invokes the
registered bench for you. Extend the same `--persist` pattern to the
other bench binaries as their kernels get tunable knobs.

Unit tests: `tests/core/test_tuning.cpp` (C++) and
`tests/python/test_tuner.py` (the Python mirror — a sidecar written by
either tier must parse byte-compatibly in the other).

## One tuner (`torch_ops/tuner.py`)

The `REGISTRY` catalogs every tunable kernel with the artifact that
tunes it — a sweep callable for Triton kernels, a persist-capable bench
binary for native ones. Adding a kernel means two edits: register it in
`REGISTRY`, and make its sweep write the store
(`@persistent_autotune` or bench `--persist`).

## Adopting in torch_ops

Exemplar: `torch_ops/mhc_projection.py` (including
`mhc_projection_tune`, the sweep hook the registry calls). To migrate
another kernel, replace `@triton.autotune(configs=..., key=...)` with
`@persistent_autotune(configs=..., key=..., kernel_name=<unique stem>,
source_files=[__file__])`, then register it in
`torch_ops/tuner.py::REGISTRY` with a `tune` hook that launches the
kernel once per key shape on synthetic tensors. Keys must be JSON-able
scalars (ints, bools, strings — the constexprs that shape the launch);
the per-key values are stored verbatim, so do not key on tensor
arguments.

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
