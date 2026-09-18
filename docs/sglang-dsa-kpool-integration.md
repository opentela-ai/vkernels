# Integrating the vkernels DSA kpool-cache kernels into sglang (SM80 / A100)

**Audience**: whoever wires GLM-5.3-Flash's DeepseekSparseAttn (DSA) kpool
indexer cache path in the **sglang CUDA serving image** (A100 / SM80) to the
native vkernels kernels — issue #60. Read together with
[`docs/kernels/dsa_kpool.md`](kernels/dsa_kpool.md) (kernel math + contract)
and [`docs/performance/dsa-kpool/A100.md`](performance/dsa-kpool/A100.md)
(A100 measurements).

## Why this exists

sglang's kpool indexer cache path is two Triton kernels in
`kpool_fp8_index.py`:

* `_kpool_assemble_softmax_rotate_write_cache_kernel` — prefill /
  short-context compress+write,
* `_kpool_decode_update_and_maybe_write_cache_kernel` — decode compress+write
  and live-tail maintenance.

Both declare `tl.float8e4nv` in their signatures (the cache is
`torch.float8_e4m3fn`). **SM80 Triton cannot declare `*fp8e4nv`** — there is
no native FP8 on A100 — so the kernels JIT-fail on the *first forward*
(`ValueError("type fp8e4nv ...")`), before MoE is ever reached. There is no
config-only escape: `_compress_write` runs on every forward.

vkernels ships native replacements with **bf16 storage + fp32 accumulation
and no `fp8e4nv` in any kernel signature** (the fp8-layout entries that
remain quantize in software — pure IEEE bit manipulation — which compiles on
any arch). They were verified device-vs-oracle **14/14 on MI300A** (gfx942)
and **14/14 on A100-SXM4-80GB** (sm_80, CUDA 12.6); bf16-store residual
≈ 3.4e-3 against a 2e-2 tolerance, identical magnitude on both archs.

## The mapping (sglang entry point → vkernels ABI)

| sglang (`dsa/dsa_indexer_kpool.py`, `kpool_fp8_index.py`) | vkernels C ABI (`libvkernels_c.so` on CUDA, `libvkernels_hip.so` on ROCm) | vkernels python wrapper (`vkernels.dsa_kpool_device`) |
| --- | --- | --- |
| `kpool_assemble_softmax_rotate_write_cache` (prefill; Triton `_kpool_assemble_softmax_rotate_write_cache_kernel`) | `vk_hip_dsa_kpool_assemble` | `dsa_kpool_assemble(...)` |
| `kpool_decode_update_and_maybe_write_cache` (decode; Triton `_kpool_decode_update_and_maybe_write_cache_kernel`) | `vk_hip_dsa_kpool_decode_update` | `dsa_kpool_decode_update(...)` |
| same two, **legacy uint8 fp8+scale cache** (the gfx942 drop-in layout `[num_pages, ssp*(128+4)]`) | `vk_hip_dsa_kpool_assemble_fp8` / `vk_hip_dsa_kpool_decode_update_fp8` | `dsa_kpool_assemble_fp8(...)` / `dsa_kpool_decode_update_fp8(...)` |
| `scatter_kpool_tail_updates` / `_scatter_kpool_tail_updates_kernel` | **not affected** — index-only, keep sglang's implementation | — |
| host-side helpers `kpool_topk_group_topk_supported` / `kpool_max_closed_pools` | `vk_dsa_kpool_group_topk_supported` / `vk_dsa_kpool_max_closed_pools` (static `vkernels` / host capi) | — |

The **bf16 entries are the SM80 serving path**; the `*_fp8` entries exist so
the same ABI also serves the legacy fp8+scale cache on gfx942 (byte-identical
store math to the Triton kernels, including the optional power-of-two
`round_scale`).

### Argument mapping (prefill, `dsa_kpool_assemble`)

All tensors device-resident; the wrappers accept sglang's natural dtypes and
cast the int arrays (one tiny kernel each):

| vkernels arg | shape / dtype | sglang source |
| --- | --- | --- |
| `out_bf16` (cache) | `[num_pages, ssp, 128]` bf16 | the kpool cache (allocate bf16, see below) |
| `chunk_k`, `chunk_score` | `[num_chunks, 128]` bf16 | the per-chunk keys / gate logits of this forward |
| `tail_k`, `tail_score` | `[n_reqs, tail_size, 128]` bf16 | the live `kpool-1` tail buffers |
| `ape` | `[pool_size, 128]` fp32 | the per-pool-slope gate weight (`ape_cache` / `compute_ape`) — **stays fp32 end-to-end** |
| `req_pool_idx` | `[n_pools]` int32 | per-pool request index (`writes.*` plan) |
| `n_from_tail` | `[n_pools]` int32 | how many of the pool's slots come from the tail |
| `chunk_src_start` | `[n_pools]` int32 | first chunk slot of the pool |
| `tail_logical_base` | `[n_pools]` int32 | ring-buffer logical base of the tail |
| `loc` | `[n_pools]` int32 | flat destination slot `page*ssp + slot` |
| `write_mask` | `[n_pools]` int32 or `None` | per-row gate (non-zero writes); `None` = write all |
| `stream` | torch Stream / handle / `None` | `None` launches on the **caller's current torch stream** (correct under `enable_dual_stream`) |

Decode (`dsa_kpool_decode_update`) takes `key/slot_score [batch,128]` bf16,
the same `ape`, `block_tables [batch, btc]` int32, `req_pool_indices`,
`positions`, `seq_lens`, `out_cache_loc` (all int32 after cast) — the names
match sglang's forward batch fields one-to-one. `pool_size` is derived from
`ape.shape[0]` (= `pool.index_kpool`).

### Behavioral contract (read before wiring)

* **The cache is NOT zeroed by the kernel.** Only rows addressed by `loc`
  (prefill) / pool-complete valid rows (decode) are written; untouched slots
  keep their content. Do **not** zero the cache between incremental forwards
  — this is exactly the Triton kernels' behavior and what makes the
  replacement a drop-in.
* **Decode updates the live tail IN PLACE** (`tail_k`/`tail_score` must
  already be bf16 + contiguous — the wrappers raise `TypeError` rather than
  silently cast, which would fork the buffer and drop the tail update).
* **Sync-free / enqueue-only**: no D2H or H2D, launches on the caller's
  current stream when `stream=None`; `round_scale` (fp8 entries only) is
  materialized as a cached device int32 so the call stays
  graph-capturable.
* **bf16 storage, no per-vector scale.** The `[num_pages, ssp, 128+4]`-style
  fp8+scale layout collapses to flat `[num_pages, ssp, 128]` bf16: bf16's 8
  exponent bits cover the gated-softmax-weighted mean without a scale, and
  the scale region is dropped entirely (that is the point of leaving fp8).
  Cache footprint roughly doubles vs the uint8 layout but is noise against
  the FP8 MoE weights / KV cache.
* Empty launches (`n_pools == 0` / `batch == 0`) are no-ops in both the ABI
  and the wrappers — no special-casing needed in sglang.

## How to load

```python
from vkernels import dsa_kpool_device as kpool

kpool.available()            # True when a device library is found
lib = kpool.load_libvkernels()   # ctypes handle, prototypes bound
```

`find_libvkernels()` resolution order: `$VKERNELS_LIB` (explicit path —
use this in the serving image) → `$K3/home/pylib/libvkernels_{c,hip}.so` →
newest `build/**/libvkernels_{c,hip}.so` under the repo → `ctypes.util.
find_library`. One ABI serves both backends: on a CUDA build the four
`vk_hip_dsa_kpool_*` entry points are exported by `libvkernels_c.so`
(`capi/cuda_capi_kpool.cpp`; the `.hip` sources compile with nvcc through the
`kernels/cuda_compat` shim, so `hipStream_t` IS `cudaStream_t` and the
trailing-`void* stream` convention is identical).

## A100 go/no-go checklist (bristen / CUDA 12.6, sm_80)

1. **Build**: on a CUDA-only machine `cmake` finds nvcc and compiles
   `dsa_kpool.hip` into the static lib; confirm the shared ABI exports:
   `nm -D build/cuda/src/c/libvkernels_c.so | grep vk_hip_dsa_kpool`
   → four hits. (Reference build: `meta/scripts/build_dsa_kpool_a100_cuda.sh`.)
2. **Offline correctness** (no sglang needed): `ctest --test-dir build/cuda
   -R dsa_kpool_device_cuda` (or run `meta/benchmarks/test_dsa_kpool_correct`
   directly) → device-vs-oracle **14/14 PASS** on the GPU. Registered as a
   CTest test on every CUDA build; CI runs it on every PR.
3. **Wire the indexer**: point `dsa/dsa_indexer_kpool.py::_compress_write` /
   `_compress_write_extend` at the python wrappers above (bf16 cache,
   fp32 `ape`, int32 index arrays via the wrappers' casts; keep
   `scatter_kpool_tail_updates` on sglang's path).
4. **First-forward gate**: `flashmla_sparse` backend, short-context
   generation completes through `_compress_write` **without**
   `ValueError("type fp8e4nv ...")` — the original blocker.
5. **Serving-parity gate**: 2-node PP2 `gen_correctness.py` reaches
   `verdict=PASS pass=5/6 crisp=3/3` — parity with the Clariden runs.
6. **Perf sanity** (optional): `meta/benchmarks/dsa_kpool_bench` — expect
   6–22 µs per step, launch-bound (see A100.md); noise against a ms-scale
   decode pass. No regression risk expected from the swap itself.

Steps 3–5 touch the sglang image and are the remaining open items of issue
#60; steps 1–2 are already automated in this repo (CI `cuda` job + the
registered CTest test).

## Test coverage map (what pins this contract)

* Host oracles + C ABI wiring: `tests/kernels/attn/test_dsa_kpool{,_fp8}.cpp`,
  `tests/capi/test_capi.cpp` (run in every CI job, host and CUDA).
* Device-vs-oracle on the actual GPU: `meta/benchmarks/test_dsa_kpool_correct.hip`,
  registered as CTest `dsa_kpool_device_cuda` (CUDA builds; skips with code
  77 on GPU-less boxes), executed by the CI `cuda` job.
* Device-ABI python wrappers + bf16/no-fp8e4nv contract:
  `tests/python/test_dsa_kpool_device.py` (stdlib unittest; runs anywhere,
  against whichever `libvkernels_{c,hip}.so` is present).
