# mhc — Multi-head hybrid-attention pre/post (gfx942 / MI300A)

Two of the tilelang MHC kernels from
`sglang/kernels/ops/layernorm/mhc.py` — the split-k
`mhc_pre_gemm_sqrsum` stage-0 kernel and the `mhc_post` combine — GPU-fault /
JIT-abort on gfx942 (MI300A, issue #51, part 2). `mhc_pre_gemm_sqrsum`
aborts JIT `init()` with *Requested dynamic shared memory 98304 exceeds
device limit 65536* because tilelang's HIP codegen never calls
`hipFuncSetAttribute` to opt in to the larger dynamic-shared cap (a per-stage
double-buffer pushes a 256-wide hidden block past MI300A's 64 KB *non-optin*
shared cap). The sglang-side workaround is to patch the kernel's
`hidden_block 256 -> 128`.

vkernels re-implements these two kernels as portable host references + HIP
kernels that run on gfx942 **without** that workaround: the HIP kernels use
*static* shared memory sized to fit MI300A's non-optin cap (no pipelined
double-buffer, no dynamic-shared request, no `hipFuncSetAttribute` opt-in).

The complex fused kernels (`mhc_pre_big_fuse` — RMSNorm + mix extraction +
Sinkhorn normalisation, and `mhc_fused_post_pre_fma`) remain on the
tilelang path; issue #51 accepts either this vkernels HIP MHC path **or** an
upstream tilelang `hipFuncSetAttribute` opt-in for them.

- **Source (CPU)**: `src/c/vkernels/kernels/mhc.cpp`
- **Source (HIP)**: `src/c/vkernels/kernels/mhc.hip`
- **Header**: `src/c/vkernels/kernels/mhc.hpp`
- **C ABI**: `vk_mhc_pre_gemm_sqrsum`, `vk_mhc_post` (host),
  `vk_hip_mhc_pre_gemm_sqrsum`, `vk_hip_mhc_post` (device) in
  `src/c/vkernels/capi/`
- **Tests (host)**: `tests/kernels/attn/test_mhc.cpp` (7 cases, incl. a
  hand-checked identity for `mhc_pre_gemm_sqrsum` over `hc_hidden_size` up
  to the static-smem limit, a hand-checked `mhc_post` identity, a
  zero-`a`-broadcasts-`c` case, plus randomized
  matches-`numpy`/`torch` sweeps for both kernels)
- **Tests (HIP)**: `meta/benchmarks/test_mhc_correct.hip` (device kernels vs
  `mhc_pre_gemm_sqrsum_cpu` / `mhc_post_cpu`, bf16-tolerant, run on
  gfx942)
- **Benchmark**: `meta/benchmarks/bench_mhc.hip`

---

## Per-token oracle

The math every kernel parallelises (verified by hand, the host oracle).

### `mhc_pre_gemm_sqrsum` — pre-norm GEMM + squared sum

```
hc_hidden_size = hc_mult * hidden_size          ( the per-token feature width )
hc_mult3       = hc_mult * (2 + hc_mult)        ( <= 32; the GEMM output width )

out[n, o]   = Σ_h  x[n, h] · fn[o, h]           ( o in [0, hc_mult3), h in [0, hc_hidden_size) )
sqrsum[n]   = Σ_h  x[n, h] · x[n, h]
```

`x` is `[num_tokens, hc_hidden_size]` bf16 on the device (fp32 on the host
oracle); `fn` is `[hc_mult3, hc_hidden_size]` fp32; `out` is
`[num_tokens, hc_mult3]` fp32; `sqrsum` is `[num_tokens]` fp32. The GEMM is
`x @ fn^T`; the squared sum is the per-token L2 energy of `x`. `out` must
not alias any input.

### `mhc_post` — post-attention combine

```
out[n, j, h] = c[n, j] · d[n, h]  +  Σ_{k=0}^{hc-1}  a[n, k, j] · b[n, k, h]
```

`a` (`comb_res_mix`) is `[num_tokens, hc, hc]` fp32; `b` (`residual`) is
`[num_tokens, hc, hidden]` bf16; `c` (`post_layer_mix`, squeezed over the
last axis) is `[num_tokens, hc]` fp32; `d` (`x`, the per-token input) is
`[num_tokens, hidden]` bf16; `out` is `[num_tokens, hc, hidden]` bf16. The
device kernel does a fp32 accumulation with a bf16 round-trip on storage —
the only divergence from the fp32 host oracle. `out` must not alias any
input.

## Two-implementation model

Mirrors `mla.{hpp,cpp,hip}` (issue #21) and `dsa.{hpp,cpp,hip}`:

| Operation | CPU (`mhc.cpp`) | HIP (`mhc.hip`) |
|---|---|---|
| `mhc_pre_gemm_sqrsum` (split-k stage 0) | `mhc_pre_gemm_sqrsum_cpu` | `mhc_pre_gemm_sqrsum` |
| `mhc_post` (post-attention combine) | `mhc_post_cpu` | `mhc_post` |

The CPU reference is fp32 throughout (always compiled; the oracle on host
CI). The HIP kernel is an online fp32 accumulator with bf16 storage:

- `mhc_pre_gemm_sqrsum` keeps a single static `hc_hidden_size`-wide bf16
  staging buffer in shared memory (<= 56 KB for the 28672-wide case), well
  within MI300A's 64 KB *non-optin* shared cap — so there is no
  dynamic-shared request and no `hipFuncSetAttribute` opt-in, the exact
  thing the tilelang codegen fails to emit. One block owns `hc_mult3`
  output columns (split across the `hc_hidden_size` accumulation with an
  inner tile), accumulating `out` and `sqrsum` in fp32 registers.
- `mhc_post` uses one block per `(token, output-head j)`, streaming the
  `hc` reduction over `k` and the `hidden` output over `h` with fp32
  accumulators, writing bf16 output.

Both match the oracle to bf16 tolerance.

## C ABI

```c
// Host entries: delegate to the CPU oracle. 0 on success, non-zero on
// validation failure.
int32_t vk_mhc_pre_gemm_sqrsum(int num_tokens, int hc_mult, int hidden_size,
                               const float* x, const float* fn,
                               float* out, float* sqrsum);

int32_t vk_mhc_post(int num_tokens, int hc, int hidden,
                    const float* a, const float* b, const float* c,
                    const float* d, float* out);

// Device entries: dispatch the HIP kernels (gfx942).
void vk_hip_mhc_pre_gemm_sqrsum(int num_tokens, int hc_mult3,
                                int hc_hidden_size,
                                const void* x, const void* fn,
                                void* out, void* sqrsum);

void vk_hip_mhc_post(int num_tokens, int hc, int hidden,
                     const void* a, const void* b, const void* c,
                     const void* d, void* out);
```

`x`/`b`/`d` are bf16; `fn`/`a`/`c` are fp32; `out`/`sqrsum` are fp32 for
`mhc_pre_gemm_sqrsum` and bf16 for `mhc_post`. The host C ABI takes fp32 for
every buffer (the bf16 round-trip is a device-only detail).

## Acceptance

* `vk_mhc_pre_gemm_sqrsum` and `vk_mhc_post` match the `mhc_*_cpu` oracle
  within bf16 tolerance. The host tests are the oracle on host CI (no GPU
  available); the device kernels are validated on gfx942 against the
  `mhc_*_cpu` oracle by `test_mhc_correct.hip`.
* The GLM-5.3-Flash shapes that the tilelang `mhc_pre_gemm_sqrsum` /
  `mhc_post` kernels cannot JIT (dynamic shared > 64 KB non-optin cap) are
  served by the HIP kernels with no `hidden_block 256 -> 128` patch and no
  custom tilelang overlay — the blocker issue #51 part 2 asks vkernels to
  close with gfx942 HIP kernels.
