# dsa — DeepseekSparseAttn sparse-MLA forward (gfx942 / MI300A)

The sparse multi-head-latent-attention forward of GLM-5.3-Flash /
DeepSeek-V3 (issue #51). An external *indexer* has already picked, per query
token, the top-`topk` most relevant KV tiles (`indices`); this kernel scores
each query against exactly those selected keys and produces the combined
attention output. It replaces the tilelang
`sparse_mla_fwd_decode_partial` (+`_combine`) decode kernels and the
`_v1`/`_v2` prefill kernels, which GPU-fault on gfx942 (the decode JIT fault
is the blocker this issue asks vkernels to close with gfx942 HIP kernels).
The shape tilelang *cannot even compile* — GLM-5.3-Flash
(`qk_rope_head_dim = 0` → `tail_dim = 0`) — is a first-class, hand-checked
case: when `tail_dim == 0` the rope-tail dot is skipped at runtime, so there
is no zero-size GEMM to lower.

- **Source (CPU)**: `src/c/vkernels/kernels/dsa.cpp`
- **Source (HIP)**: `src/c/vkernels/kernels/dsa.hip`
- **Header**: `src/c/vkernels/kernels/dsa.hpp`
- **C ABI**: `vk_dsa_config`, `vk_dsa_sparse_fwd` (host),
  `vk_hip_dsa_sparse_fwd` (device) in `src/c/vkernels/capi/`
- **Tests (host)**: `tests/kernels/attn/test_dsa.cpp` (10 cases, incl. a
  hand-checked `tail_dim == 0` GLM-5.3-Flash case and a `tail_dim > 0`
  DeepSeek-V3 case, plus a randomized matches-reference sweep over both
  shape families)
- **Tests (HIP)**: `meta/benchmarks/test_dsa_correct.hip` (device kernel vs
  `dsa_sparse_fwd_cpu`, bf16-tolerant, run on gfx942)
- **Benchmark**: `meta/benchmarks/bench_dsa.hip` (+ `bench_dsa.sh` driver)

---

## Per-token oracle

The math every kernel parallelises (verified by hand, the host oracle). For
query token `i`, head `h`, selected key index `idx = indices[i,0,k]` (masked
when `idx < 0` or `idx >= S_kv`):

```
d_v       = dim - tail_dim                      ( > 0; tail_dim may be 0 )
raw[i,h,k]   = q_main[i,h] · k_main[idx]
             + (tail_dim>0) q_tail[i,h] · k_tail[idx]
score[i,h,k] = sm_scale · raw[i,h,k]            ( sm_scale includes log2(e) )
w[i,h,k]     = 2^(score[i,h,k] - lse[i,h])      ( masked keys -> weight 0 )
out[i,h]     = Σ_k w[i,h,k] · v[idx]            ( d_v wide )
lse[i,h]     = log2( Σ_k 2^score[i,h,k] )        ( base-2, when return_lse )
```

`sm_scale = (1 / sqrt(dim + tail_dim)) · log2(e)`, so the weights are the
standard natural-exp softmax `exp(raw / sqrt(dim + tail_dim)) / Σ` expressed
in base-2 (FlashAttention's log2 trick). Masked keys contribute weight 0;
when *every* selected key is masked the output row is 0 and the LSE is
`-inf`. `kv_group == 1` (all heads share one key/value head); the
per-group `partial_o` / `partial_lse` the tilelang kernel emits are an
internal parallelism detail — the combined result is grouping-independent,
which the host test validates directly.

## Two-implementation model

Mirrors `mla.{hpp,cpp,hip}` from issue #21:

| Operation | CPU (`dsa.cpp`) | HIP (`dsa.hip`) |
|---|---|---|
| `sparse_mla_fwd_decode_partial` (+ combine) | `dsa_sparse_fwd_cpu` | `dsa_sparse_fwd` |
| `sparse_attention_fwd_kernel_v1/v2` (prefill) | `dsa_sparse_fwd_cpu` | `dsa_sparse_fwd` |
| tile selector | `dsa_config_for` | `dsa_config_for` (decode ≤8 q/block; prefill BQ q/block × 256 th) |

The CPU reference is a numerically-stable two-pass base-2 softmax, fp32
throughout (always compiled; the oracle on host CI). The HIP kernel is an
**online softmax**: each block owns one (query, head) pair — `bq` query rows
in prefill — streams its `topk` selected keys in `block_I` tiles repeated
`inner_iter` times, and accumulates the combined output directly in fp32
with bf16 storage. It matches the oracle to bf16 tolerance for both
`tail_dim == 0` and `tail_dim > 0`.

## C ABI

```c
// Per-shape tile selector (decode vs prefill).
int  vk_dsa_config(int S_q, int H, int dim, int topk,
                   int* bq, int* threads, int* block_I, int* inner_iter);

// Host entry: delegates to the CPU oracle.
int  vk_dsa_sparse_fwd(int S_q, int S_kv, int H, int dim, int tail_dim,
                       int topk, int kv_group, float sm_scale,
                       int return_lse,
                       const void* q, const void* kv, const void* indices,
                       void* out, void* lse);

// Device entry: dispatches the HIP kernel (gfx942). 0 on success, non-zero
// with a captured last-error string on validation failure.
int  vk_hip_dsa_sparse_fwd(int S_q, int S_kv, int H, int dim, int tail_dim,
                           int topk, int kv_group, float sm_scale,
                           int return_lse,
                           const void* q, const void* kv, const void* indices,
                           void* out, void* lse);
```

`q`/`kv` are bf16; `indices` is int32 (padded to a multiple of 64; entries
`< 0` or `>= S_kv` are masked kpool tail tokens); `out` is bf16; `lse` is
fp32 and may be null when `return_lse` is false.

## Acceptance

* `vk_dsa_sparse_fwd` matches `dsa_sparse_fwd_cpu` within bf16 tolerance for
  both `tail_dim == 0` (GLM-5.3-Flash) and `tail_dim > 0` (DeepSeek-V3)
  shapes. The host tests are the oracle on host CI (no GPU available); the
  device kernel is validated on gfx942 against `dsa_sparse_fwd_cpu` by
  `test_dsa_correct.hip`.
* The GLM-5.3-Flash `tail_dim == 0` shape — which tilelang cannot compile —
  is served by the HIP kernel with no custom tilelang overlay.

## Benchmark (MI300A, gfx942)

`meta/benchmarks/bench_dsa.hip` (+ `bench_dsa.sh` driver). Roof: 1307
TFLOP/s bf16, 5300 GB/s HBM3, ridge ~247 FLOP/B.

The kernel scores `S_q · H · topk` (query·key) dots, each `W = dim + tail_dim`
wide (the key dot) plus `d_v = dim - tail_dim` wide (the value gather). At
the GLM-5.3-Flash decode shape (`dim = 256, tail_dim = 0, W = 256, d_v = 256,
topk ≤ 256`) the arithmetic intensity is `~2·(W+d_v)/(bytes/q + bytes/kv) ≈
0.25`, so the kernel is **memory-bound** on the index gather + key/value
reads — the target of a future tiled-key prefetch and partial-output
reduction. Numbers are collected on a gfx942 box via `bench_dsa.hip`.
