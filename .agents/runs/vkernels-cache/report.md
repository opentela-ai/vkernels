# LANE 30 — VKERNELS_CACHE: the op-config cache (tune-once, persist-always)

**Repo**: `/local/home/xiayao/Documents/code/vkernels` (main @ 6f6de90). **Scope**: local
code + GB10 (sm_121, this box) only. All lane files uncommitted; other lanes'
in-flight files untouched. **Lane files**:

- `src/python/vkernels/tuning/__init__.py`, `src/python/vkernels/tuning/cache.py` — the module (new)
- `src/python/vkernels/torch_ops/glm_gemv.py` — integration 1 (dense GEMV)
- `src/python/vkernels/torch_ops/glm_kda_decode.py` — integration 2 (KDA decode)
- `src/python/vkernels/torch_ops/elementwise.py` — integration 3 (rms_norm rung)
- `tests/python/conftest.py` — session default `VKERNELS_CACHE=off`
- `tests/python/test_tuning_cache.py` — 25 tests (21 CPU-safe + 4 GPU), all passing

Full suite: `1325 passed, 101 skipped, 499 subtests` — the only failure
(`test_sgl_fused_moe_matches_eager_serving_shape[8]`) fails identically on the
clean baseline (verified via `git stash`); one transient `test_mhc_chain` GPU
failure in the first full run does not reproduce (passes 3/3 isolated, and a
second full-suite run is clean).

## 1. Status-quo audit (required before the design lands)

What the ops actually do today, per family:

| op family | status quo |
|---|---|
| `glm_router.py`, `glm_router_dense.py`, `glm_kda_decode.py`, `elementwise.py`, `mhc_projection.py`, `qkv_projection.py`, `glm_projection.py` | **Pinned configs** (constants in the launch call: e.g. kda `BV=32, num_warps=4`; rms_norm implicit `num_warps=4`; router `BLOCK_E=next_pow2(E), warps=4`). No `@triton.autotune` anywhere in `src/` (grep clean). Zero per-process re-tuning. |
| `glm_gemv.py` (bf16 dense GEMV) | **Exact-shape pins + heuristic**: `_CFG` dict of measured (O,I) winners for the GLM decode shapes (+ a MI300A override table); unseen shapes take a next-pow2/rows heuristic. Pins are code — a new GPU re-measures nothing automatically. |
| `sgl_moe.py` (fused MoE) | **Reads packaged JSON sidecars** (vLLM-style per-M-bucket `configs/`, moved in-tree by the sgl-fuse campaign) — tuned offline, shipped with the wheel, keyed by exact M buckets. |
| `glm_fp8_blockwise_gemm.py` / `glm_dense_fp8_gemv.py` | **Three-layer precedence**: `VK_FP8GEMM_TILES` env override > M-bucketed JSON sidecar via `tile_configs.lookup_tiles` > heuristic default. Sidecar production is a manual `configs/README.md` flow (one-time tune). |
| `tuning_cache.py` (pre-existing, **kernel tier**) | `persistent_autotune` + Triton key functions: caches Triton **compilation** artifacts keyed by autotune key args — often exact sizes, so a new size re-sweeps. Gated by its own env (`VKERNELS_TUNING_CACHE`); orthogonal to this lane. |
| `tuner.py` (pre-existing) | A global tuner daemon abstraction (rig-driven), not wired into the op hot paths. |

**Bottom line**: no in-process `@triton.autotune` survives in the tree; the
tree's convention is pinned-or-sidecar configs with offline production. The
gap this lane closes: the offline sidecar flow is manual and per-repo-harness,
and per-(machine, code-version) re-tuning of a new GPU or edited kernel
requires someone to run a bench script by hand. `VKERNELS_CACHE` automates
exactly that, under a bounded budget, with the pins as the declared fallback.

## 2. Design (as settled)

### Cache key — `(op name, kernel-source fingerprint, arch identity, shape class)`

- **op name**: store identity, e.g. `"glm_gemv.dense_gemv"` (file stem).
- **source fingerprint**: `sha256` over the op's producing module files by
  default (`(__file__,)` — kernel + wrapper + the heuristic that shapes the
  default all live in one file for these ops), `source_objs=` (hashed via
  `inspect.getsource` after unwrapping `__wrapped__`/triton JIT) for finer
  granularity. A code change ⇒ different fingerprint ⇒ miss ⇒ one bounded
  re-tune. This is the auto-invalidation the lane asked for.
- **arch identity**: `torch.cuda.get_device_capability` → `sm<maj><min>`,
  SM count, device name — the triple that actually moves GEMV/tile optima
  between parts. Reuses the kernel tier's `device_fingerprint` so both tiers
  name a device identically. Software versions ride along as **audit-only
  metadata** (a triton upgrade may stale a config, but that is a perf
  question, not identity; the source fingerprint owns code-change
  invalidation). Injectable (`device=`) so the whole store lifecycle is
  testable without a GPU.
- **shape class**: coarse tiers per op, mirroring the tree's per-shape gating
  conventions — never exact shapes:
  - `glm_gemv.dense_gemv`: O tiers (`o64/o512/o2k/o8k/oX`) × I tiers
    (`i512/i2k/iX`) — the grid and reduction axes of a GEMV;
  - `glm_kda_decode.kda_decode`: head-dim tag (already an eligibility tier:
    `{32,64,128}`) × batch·heads tiers (`bh8/bh64/bhx`) — the decode ladder;
  - `elementwise.rms_norm`: row tiers (`r8/r64/rX`, the decode ladder's
    occupancy axis) × width pow2 tiers (`d2k/d4k/d8k/dX`).
  Consequence: one tuned config serves a whole bucket, so a new size inside a
  known tier is a **store hit, not a sweep**.

### Storage

`$VKERNELS_CACHE` (default `~/.cache/vkernels/`), **one JSON per
(op, capability)** — `glm_gemv.dense_gemv.sm121.json` — schema
`vk-op-config-cache/1`, `indent=2, sort_keys=True` (human-readable,
diffable). One file per op (not per op+arch) keyed by capability inside
records: buckets from different parts coexist and merge (see Concurrency).
Each record:

```json
"o64-i512": {
  "config":  {"rows": 4, "block_i": 512, "warps": 8},
  "status":  "tuned" | "seeded" | "default",
  "time_ms": 12.3,                        // winner's measured cost (null for default)
  "device":  {"capability": "sm121", "sm_count": 1, "name": "..."},
  "source":  "sha256:...",                // producing-sources fingerprint
  "tuned_at": "2026-09-28T21:51:39+00:00",
  "origin":  "op-config-tune",
  "bench":   {"candidates": {"{\"rows\":4}": 12.3, ...},   // full measured table
              "elapsed_ms": 890.1, "budget_ms": 2000.0,
              "reason": null | "budget-exhausted" | "bench-failed" | "no-candidates",
              "error": null}
}
```

Writes: temp file + `os.replace` under the per-file `store_lock` (fcntl)
**shared with the kernel tier**; a writer re-reads disk under the lock and
merges only its own record — concurrent processes last-writer-wins per
RECORD, never per file (tested).

### Load path + the floe capture-ordering contract

**The contract (documented, and enforced, not just conventional):** configs
must be static when any graph capture observes the op. Floe's serve boot
(`floe/engine/runner/warmup.py::warmup_model`) runs eager `startup-eager`
passes over the prefill/decode shape ladders **before**
`model.capture_buckets(...)` — tune-on-miss runs exactly there: first eager
call per bucket sweeps (≤2 s), persists, memoizes; by capture time every
bucket is memo-or-store, so captures see static configs. The reusable
capture pattern (`floe/engine/graph_capture/__init__.py`) additionally runs
warmup iterations of the graph body on a capture stream *before* the
`torch.cuda.graph` region (for autotuners/workspaces) with an idle window
before the capture pass — a first-seen-bucket sweep may also run there, and
completes before capture starts.

**Enforcement**: `op_config` checks `torch.cuda.is_current_stream_capturing()`
before anything expensive: mid-capture it **never benches and never writes** —
it returns memo-or-default instantly (a `do_bench` sync mid-capture would
abort the capture; tested via monkeypatched `_capturing`). A bucket first
seen *inside* a capture stays on the default for the process lifetime (the
resolution is memoized) — a captured graph is self-consistent and a config
can never flip under a running serve. This composes with floe's
`--compile-freeze` (no lazy mid-serving captures at all) and with lazy
captures of truly-unseen buckets (those stay default — conservative and
capture-consistent; a later process warms them eagerly and tunes).

### Tune-on-miss, bounded

Per (op, bucket): each candidate benched at the **live shape** via a
`triton.testing.do_bench(median)` closure over the op's own launch. Budget
gate runs **between** candidates: arg `budget_ms` >
`VKERNELS_CACHE_BUDGET_MS` env > **2 s default**. On budget exhaustion the
sweep never ships a partial winner (that would be machine-speed-dependent) —
the op's **declared default** config wins deterministically, persisted with
`status: "default"`, `reason: "budget-exhausted"`, and the partial measured
table kept for audit; the miss is never re-paid (a later process replays the
default record with zero benchmarking). Same for a totally failed bench
(`bench-failed`, per-candidate exceptions prune, not crash) and an empty
candidate space (`no-candidates`). Per-op totals = sum over the buckets the
warmup actually touches (GLM decode: ~15 GEMV shapes fall into ~5 buckets,
one kda bucket, a few rms_norm buckets ⇒ ≲4×2 s worst case across the whole
model boot).

### Off switch

`VKERNELS_CACHE=off` disables **both** the store and the sweep: `op_config`
returns `(declared default, "off")` with zero benchmarking and zero disk
traffic — i.e. today's behavior bit-for-bit (the pins are the declared
defaults). The kernel tier's `VKERNELS_TUNING_CACHE` and Triton's JIT binary
cache are orthogonal and unaffected. Test sessions default to `off`
(conftest `pytest_configure`), so a developer's real `~/.cache/vkernels` is
never touched by a test run; cache tests re-enable per-test against a tmp
store.

### Population entries (seeds)

`seed(op, {shape_class: {"config":..., "time_ms":..., "origin":...}})` lands
offline-produced tables (e.g. lane-29 H100 rig sweeps) in the **same** schema
with `status: "seeded"` — served with zero benchmarking, never downgraded by
a later seed; a locally `tuned` record always outranks a seed. This is the
bridge from the tree's current manual sidecar flow to per-machine tuning.

### API surface

```python
from vkernels.tuning import op_config, seed, stored_records, reset_memo

cfg, status = op_config("glm_gemv.dense_gemv", shape_class,      # "tuned"|"seeded"|"default"|"off"|"capture"
                        default={"rows": 4, "block_i": 512, "warps": 8},   # declared default
                        candidates=[...],                 # sweep space, default first
                        bench=lambda cand: do_bench(...), # one candidate, live shape
                        source_files=(__file__,))
```

Ops keep a per-exact-shape host memo over the per-bucket cache so the
per-launch Python cost is one dict lookup (the tier string is built once per
exact shape, on the cache miss — never on the hot path).

## 3. Integrations (2–3 representative ops, end-to-end)

Chosen for real config sensitivity, one per op family:

1. **`glm_gemv.dense_gemv`** (exact-shape pins + heuristic today): sweep of
   ≤5 candidates (rows ∈ {1,4,8} filtered to `O % rows == 0` — the kernel's
   store is unmasked on the row axis, so a row group must divide O — plus
   warps ∈ {4,8}); default = today's pin/heuristic path verbatim
   (`_gemv_default`). GPU test: tune → persist → fresh-memo reload → same
   config, **bit-identical output** across the reload boundary, parity vs
   `x.float() @ w.float().T` intact.
2. **`glm_kda_decode.kda_decode`** (pinned `BV=32, warps=4`): sweep of
   `{(32,4) default, (64,4), (64,8), (16,4), (32,8)}` with `BV ≤ dim`; the
   launch call now takes the resolved `(bv, warps)`. GPU test: parity vs
   `kda_decode_reference` at the tree's own tolerances (fp32 vectors, same
   distribution as the op's existing parity test) with the cache ON, and the
   off switch keeps the exact pre-cache launch (`BV=32/warps=4`).
3. **`elementwise.rms_norm`** (implicit `num_warps=4`): BLOCK stays pinned
   to `next_power_of_2(d)` (the row reduction needs the whole row in one
   program), so `num_warps ∈ {2,4,8}` is the free knob. GPU test: parity vs
   `rms_norm_reference` with the cache ON; record `r8-d2k` landed.

Docstrings of all three ops document the integration and the fallback
semantics (off/capture/budget-exhaustion/bench-failure → declared default).

## 4. Tests (`tests/python/test_tuning_cache.py`, 25 passing)

CPU-safe (injected device + scripted bench, no GPU): tune-on-miss persists
winner with full record fields; store replay costs zero benchmarks (a second
sweep would raise); same-config frozen for the process even if the store
file vanishes; buckets share one file; budget exhaustion → default +
`budget-exhausted` + partial table, never re-paid; bench failure →
`bench-failed` with error text; no-candidates; env budget override; source
edit invalidates once (and the re-tune persists — not re-paid per process);
arch identity moves the bucket; foreign-arch file ignored; off switch (and
seed honors it); capture guard never benches/writes and freezes the
resolution; corrupt store = miss not crash; foreign schema ignored;
concurrent-record merge; non-serializable config raises loudly; seeds:
served without benching, never downgrade a local tune, survive reload.

GPU (real loop on GB10): the three op integrations above + the off-switch
behavior test.

Notable semantic pinned by a test that originally failed against the
implementation: **budget exhaustion mid-sweep falls back to the declared
default** (the lane spec), not the partial winner.

## 5. Notes / rollout

- First boot on a new machine: warmup pays ~2 s per first-seen (op, bucket);
  every later process replays from disk. `VKERNELS_CACHE_BUDGET_MS` caps it;
  `VKERNELS_CACHE=off` reverts everything.
- Delete the JSON (or edit the source) to force a re-tune; records are
  per-capability so mixed fleets sharing a store dir never poison each other.
- `stored_records(op)` exists for a future `vkl tune` CLI / dump tooling;
  `seed()` is the rig-table intake (lane-29 shape).
- Natural follow-ups (not done here): wire `glm_router_dense.fused_router_dense`
  and `glm_fp8_blockwise_gemm` tiles through the same `op_config` (the fp8
  sidecar flow becomes a seed + local re-tune); add a `--tune-warmup` floe
  flag that widens the shape ladder at boot so every serving bucket tunes
  before freeze.
