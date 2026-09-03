# dsa_kpool — GLM-5.3 DSA kpool-cache compress/write (gfx942 / MI300A, SM80 / A100)

The two **kpool-cache** kernels of GLM-5.3-Flash's `DeepseekSparseAttn`
layer, the path that runs on *every* forward when `index_kpool > 1`: a
`kpool-1` live tail of recent tokens is appended to the fixed `index_topk`
columns and held in a persistent cache, which is **compressed** (one pool →
one rotated mean key) and **written** back on each forward. This is the
kpool-cache path PR #52 did *not* touch (PR #52 landed `dsa_sparse_fwd`, the
attention **forward**; the compress/write kernels are a *separate* set of
Triton kernels in `kpool_fp8_index.py`). The canonical shape/contract table
lives in `src/c/vkernels/kernels/dsa_kpool.hpp`; this page only restates it.

On **A100 / SM80** the cache is hardcoded `torch.float8_e4m3fn` → `tl.float8e4nv`,
and SM80 Triton cannot even declare `*fp8e4nv` in a kernel signature, so the
JIT fails on the first forward (before MoE is ever reached). This issue ports
the two FP8 Triton kernels to native vkernels with **bf16 storage + fp32
accumulation and no `fp8e4nv` in any kernel signature** — the same
two-implementation model (`dsa.{hpp,cpp,hip}`) and `gemm_bf16` storage
convention PR #52 established. The two Triton kernels replaced are

- `kpool_fp8_index.py::_kpool_assemble_softmax_rotate_write_cache_kernel`
  (prefill / short-context) → `dsa_kpool_assemble`
- `kpool_fp8_index.py::_kpool_decode_update_and_maybe_write_cache_kernel`
  (decode; first arg `buf_fp8_ptr`) → `dsa_kpool_decode_update`

(`scatter_kpool_tail_updates` / `_scatter_kpool_tail_updates_kernel` is
index-only and is **not** affected.)

- **Source (CPU oracle)**: `src/c/vkernels/kernels/dsa_kpool.cpp`
- **Source (HIP)**: `src/c/vkernels/kernels/dsa_kpool.hip`
- **Header**: `src/c/vkernels/kernels/dsa_kpool.hpp`
- **C ABI**: `vk_dsa_kpool_assemble` / `vk_dsa_kpool_decode_update` (host),
  `vk_hip_dsa_kpool_assemble` / `vk_hip_dsa_kpool_decode_update` (device) in
  `src/c/vkernels/capi/`
- **Python**: `vkernels.kernels.dsa_kpool_assemble`,
  `vkernels.kernels.dsa_kpool_decode_update` (fp32 CPU reference; the
  device ABI is bf16 and the integrator converts — mirrors `dsa_sparse_fwd`)
- **Tests (host)**: `tests/kernels/attn/test_dsa_kpool.cpp`
  (`DsaKpoolAssemble`, `DsaKpoolDecodeUpdate` — boundary, null-arg,
  out-of-range, no-op, hand-checked, and randomized matches-reference sweeps
  over both kernels)
- **Tests (HIP)**: `meta/benchmarks/test_dsa_kpool_correct.hip`
  (both device kernels vs the fp32 CPU oracles, bf16-tolerant, run on
  gfx942 — `head_dim == 128`, the 128-pt involution-normalised Hadamard, the
  GLM-5.3-Flash shape)

---

## The math every kernel parallelises (verified by hand, the host oracle)

For a pool of `pool_size` participating keys `k[s]` (dim `H = 128`) with
per-token gate logits `score[s]` and per-token (per-slot) gate weights
`ape[s]` (the `compute_ape` / `ape_cache` tensor), the compressed key is the
**`ape`-gated online-softmax-weighted mean**, then rotated by the 128-point
involution-normalised Hadamard `H`:

```
logit[s, d]   = score[s, d] + ape[s, d]                 (s = 0..pool_size-1)
m[d]          = max_s logit[s, d]                        (online running max)
w[s, d]       = exp(logit[s, d] - m[d])                  (online, rescalled)
mean[d]       = ( Σ_s w[s,d] · k[s,d] ) / ( Σ_s w[s,d] )
out[d]        = (H @ mean)[d]                            (128-pt Hadamard)
```

The online softmax (`m`/`acc`/`denom` rescalled on every `fmaxf` update) is
numerically identical to a naive two-pass softmax (max, then weighted sum
under the fixed max) — verified against a NumPy golden reference. The
Hadamard `H[i,j] = ±1/√128` with the sign given by the parity of
`popcount(i & j)`; `H @ H = I` in fp32 (involutory), verified by the same
golden reference. The kernel accumulates in **fp32** end-to-end; the cache
is **bf16** storage (input dequant on load via `bf16_to_f32`, output quant on
store via `f32bits_to_bf16`). There is **no** per-vector scale (bf16's range
is ample for the gated softmax weights and the `k` means — see
`dsa_kpool.hpp`).

### PREFILL (`dsa_kpool_assemble`)

For each of `n_pools` logical pools, the `pool_size` participating keys are
gathered from a **TAIL prefix** (`n_from_tail` of them, taken from the live
tail at logical offsets
`[tail_logical_base, tail_logical_base + n_from_tail)` via
`phys = (base + s) % tail_size`) and a **CHUNK suffix** (the remaining
`pool_size - n_from_tail`, taken from `chunk_k` / `chunk_score` starting at
`chunk_src_start`; only the suffix is read, so
`chunk_src_start + (pool_size - n_from_tail) <= num_chunks` is the caller
invariant). The compressed key is written to the persistent page cache at
`out[ loc[r] // ssp, loc[r] % ssp ]`. `write_mask[r] == 0` skips the whole
row (the integrator pre-filters); the output is zero-initialised so skipped
rows stay zero. One warp per pool, 64 lanes, each lane owns head elements
`{2l, 2l+1}` (the two-pass per-lane online softmax mirrors the Triton).

### DECODE (`dsa_kpool_decode_update`)

For each of `batch` decode requests, the current token's `key` / `slot_score`
is **substituted** into slot `pos % pool_size` of the pool (the remaining
slots are read from the live tail), the pool is compressed as above, and the
result is written to the persistent page cache at
`out[ block_tables[r, clamp(pool_page_group * pool_size, 0, btc-1)], pos % ssp ]`
where `pool_id = pos // pool_size`, `pool_page_group = pool_id // ssp` — but
**only when** the request is valid (`req_pool_idx` in range,
`out_cache_loc != 0`, `0 <= pos < seq_len`) **and** the pool is full
(`pos % pool_size == pool_size - 1`). **Unconditionally** (valid or not) the
current `key` / `slot_score` is written to the live tail at
`[req, pos % tail_size]` — masked to a no-op when the row is invalid, so the
tail is left untouched for invalid rows (mirrors the Triton
`_kpool_decode_update_and_maybe_write_cache_kernel`). One warp per request.

---

## Launch / dimensions

| kernel | grid | block | shmem |
| --- | --- | --- | --- |
| `dsa_kpool_assemble_kernel` | `(n_pools, 1, 1)` | `64` (1 warp) | `128 * sizeof(float)` |
| `dsa_kpool_decode_kernel` | `(batch, 1, 1)` | `64` (1 warp) | `128 * sizeof(float)` |

One warp per pool/request is sufficient — the work is a `pool_size × 128`
reduction (at most `8 × 128 = 1024` fp32 ops), and the warp specialises
naturally: lanes `0..63` each own two head elements, lane 0 performs the
in-place `hadamard128` over the staged `mean`, and all lanes store the two
quantised bf16 elements. No `__syncthreads` is needed (single warp); the two
`__syncwarp()` around the Hadamard stage the staged `mean` (lane 0 reads all
128, every other lane wrote two).

---

## SM80 / A100 integration note

The native kernel removes the `*fp8e4nv` JIT blocker: the upstream CUDA
image's `dsa/dsa_indexer_kpool.py::_compress_write` /
`_compress_write_extend` is wired (separately, in the deployment recipe) to
call `vk_dsa_kpool_assemble` (prefill) / `vk_dsa_kpool_decode_update`
(decode) instead of the raw-FP8 Triton kernels, passing the cache as **bf16**
(`torch.bfloat16` ↔ the `uint16_t` device ABI) and keeping `ape` as fp32.
The kpool cache footprint doubles relative to FP8 (2× bytes) but is small
relative to the FP8 MoE weights / KV cache; the per-token `ape` gate stays
fp32 end-to-end (mirroring `dsa_sparse_fwd`'s bf16→fp32 device ABI).
