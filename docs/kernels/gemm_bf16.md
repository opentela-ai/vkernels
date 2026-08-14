# gemm_bf16 — Tiled bf16 MFMA GEMM (gfx942 / MI300A)

A tiled bf16 dense matrix multiply built on the AMD K16 bf16 MFMA
(`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`) for the **Kimi-K3 projection
shapes** (issue #29). Today these shapes fall back to AITER's untuned
"torch solution:0" because `bf16_tuned_gemm.csv` has no gfx942 entries;
this kernel provides a real, validated HIP path so the fallback is no
longer the only option.

- **Source (CPU)**: `src/c/vkernels/kernels/gemm_bf16.cpp`
- **Source (HIP)**: `src/c/vkernels/kernels/gemm_bf16.hip`
- **Header**: `src/c/vkernels/kernels/gemm_bf16.hpp`
- **Tests**: `tests/kernels/gemm/test_gemm_bf16.cpp` (host),
  `meta/benchmarks/test_gemm_bf16_correct.hip` (device vs CPU)
- **Benchmarks**: `meta/benchmarks/bench_gemm_bf16.hip` (latency, TFLOP/s,
  effective GB/s, and a per-shape tile autotuner)
- **Performance**: `docs/performance/gemm-bf16/gfx942.md`

---

## Computation

```
C[i][j] = α · Σₖ A[i][k] · B[k][j]  +  β · C[i][j]

for i ∈ [0, M),  j ∈ [0, N),  k ∈ [0, K)
```

All matrices are dense, row-major, and stored as `uint16_t` IEEE-754 bf16
bit patterns:

| Matrix | Shape | Meaning |
|---|---|---|
| A | M × K | input activations (bf16) |
| B | K × N | the **transposed** projection weight `W[N, K].T` (bf16) |
| C | M × N | output (bf16) |

`α`/`β` are `float32` scalars. The accumulation is **fp32**; the store is a
single round-to-nearest-even back to bf16 — the same contract as the MFMA
kernel, so host and device agree to the bit on simple cases and to ~1 bf16
ULP in general.

### Kimi-K3 projection shapes

The issue lists each projection as an `N × K` weight shape (output × input).
The serving recipe transposes the `[N, K]` weight once (or keeps it
pre-transposed) and calls this kernel with `(M, N, K) = (batch, N_issue,
K_issue)`:

| N | K | projection |
|---:|---:|---|
| 6288 | 7168 | QKV |
| 3584 | 7168 | gate/up (split) |
| 896 | 7168 | |
| 2112 | 7168 | |
| 1536 | 7168 | |
| 7168 | 1536 | |
| 7168 | 768 | |
| 7168 | 3584 | |
| 2304 | 1536 | |
| 3072 | 512 | |
| 1536 | 128 | |

Serving batches are `M ≈ 5–64`; warmup/profiling uses the same `(N, K)` at
`M = 8192`. **Every `K` is a multiple of 64** (so the fixed `BK = 64`
K-tile needs no K bounds-check for K3), and **every `N` is a multiple of 16** (so
the 16-column MFMA fragment is exact; only the last `BN`-tile of `N` needs
a column bounds-check, e.g. `N = 6288` is 393 × 16 but not a multiple of
64).

---

## Contract

| Condition | Error |
|---|---|
| `M == 0 \|\| N == 0 \|\| K == 0 \|\| A != null` | `std::invalid_argument` |
| `M == 0 \|\| N == 0 \|\| K == 0 \|\| B != null` | `std::invalid_argument` |
| `M == 0 \|\| N == 0 \|\| C != null` | `std::invalid_argument` |

Empty dimensions are no-ops: with `M == 0` nothing is read or written
(null `A`/`B`/`C` are allowed); with `N == 0` the inner loop is empty
(B/C untouched); with `K == 0` the accumulator is zero so `C = β · C`.

---

## CPU reference

`gemm_bf16_cpu` is the oracle. It converts each bf16 operand to fp32,
accumulates a per-output dot product in fp32, and stores with the same
round-to-nearest-even as the device (`f32bits_to_bf16_local`, identical to
`f32bits_to_bf16` in `moe_device.hip`):

```cpp
void gemm_bf16_cpu(std::size_t M, std::size_t N, std::size_t K, float alpha,
                   const uint16_t* A, const uint16_t* B, float beta,
                   uint16_t* C) {
  VK_EXPECTS(M == 0 || N == 0 || K == 0 || A != nullptr, "A must not be null");
  VK_EXPECTS(M == 0 || N == 0 || K == 0 || B != nullptr, "B must not be null");
  VK_EXPECTS(M == 0 || N == 0 || C != nullptr, "C must not be null");
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (std::size_t k = 0; k < K; ++k)
        acc += bf16_to_f32_local(A[i*K+k]) * bf16_to_f32_local(B[k*N+j]);
      float prev = beta != 0.0f ? bf16_to_f32_local(C[i*N+j]) : 0.0f;
      C[i*N+j] = f32_to_bf16_local(alpha * acc + beta * prev);
    }
}
```

`gemm_bf16_config_for(M, N, K, &bm, &bn, &bk, &threads)` picks the launch
tile. Both shapes are bf16-memory-bound at serving `M ≤ 64` and
bf16-compute-bound at warmup `M > 64`; `BK` is fixed at 64. The serving
`BN = 16` (not `64`) is **measured**: the on-device autotuner found
`(16,16)` beats `(16,64)` by 1.4–2.9× on every serving shape because it
launches `⌈N/16⌉ × ⌈M/16⌉` blocks (e.g. 1572 at `M=64, N=6288`) and
saturates the 228 CUs, whereas `BN = 64` launches only 396.

| M | BM | BN | BK | threads | regime |
|---:|---:|---:|---:|---:|---|
| ≤ 64 | 16 | 16 | 64 | 64 | serving (one wavefront per 16-row fragment) |
| > 64 | 64 | 64 | 64 | 256 | warmup (4 wavefronts; max B reuse) |

---

## HIP kernel — tiled bf16 MFMA

Each block owns an output tile `[BM, BN]`; one wavefront (64 threads) per
16-row fragment, so `THREADS = (BM/16) * 64` and each wavefront computes
`(BN/16)` column fragments via `BK/kMfmaK = 4` MFMAs per K-tile.

```cpp
template <int BM, int BN>
__global__ void gemm_bf16_kernel(const uint16_t* A, const uint16_t* B,
                                 uint16_t* C, int M, int N, int K,
                                 float alpha, float beta) {
  constexpr int kNF = BN/16, kWarps = BM/16, kTh = kWarps*64;
  __shared__ uint16_t sA[BM][BK], sB[BK][BN];
  const int warp = threadIdx.x >> 6, lane = threadIdx.x & 63;
  const int m = lane & 15, n = lane & 15, kq = lane >> 4;
  const int row_base = warp * 16;
  v4f acc[kNF] = {0};
  for (int kt = 0; kt < (K+BK-1)/BK; ++kt) {
    const int k_off = kt * BK;
    // cooperative uint2 (4 bf16) loads into sA / sB, M/N/K bounds-checked
    // ... __syncthreads();
    for (int mf = 0; mf < BK/kMfmaK; ++mf) {
      const int k0 = mf*kMfmaK + kq*4;
      v4s a = {sA[row_base+m][k0], sA[row_base+m][k0+1],
               sA[row_base+m][k0+2], sA[row_base+m][k0+3]};
      for (int nf = 0; nf < kNF; ++nf) {
        const int nb = nf*16 + n;
        v4s b = {sB[k0][nb], sB[k0+1][nb], sB[k0+2][nb], sB[k0+3][nb]};
        acc[nf] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc[nf], 0, 0, 0);
      }
    }
    // __syncthreads();
  }
  // epilogue: col=lane%16, row=warp*16+(lane/16)*4+i, bounds-checked
}
```

### Verified fragment layout

The A/B/C operand distribution mirrors the empirically verified
`mfma_k64_pf` helper in `moe_fused.hip` (confirmed on gfx90a with one-hot
matrices), here generalized to arbitrary `(BM, BN)`:

| Operand | lane → index | element |
|---|---|---|
| A (row `m`, K-group `kq`) | `m = lane%16`, `k0 = (lane/16)*4` | `a[i] = sA[row_base+m][k0+i]` |
| B (col `n`, K-group `kq`) | `n = lane%16`, `k0 = (lane/16)*4` | `b[i] = sB[k0+i][nf*16+n]` |
| C (row, col) | `col = lane%16`, `row = (lane/16)*4 + i` | `c[i] = C[m_tile*BM + row][n_tile*BN + col]` |

### Cooperative loads

`sA[BM][BK]` and `sB[BK][BN]` are filled cooperatively: each thread loads
one `uint2` (4 bf16) per stride, with M/N/K bounds-checking so that
non-exact tiles (M not a multiple of 16/64; N not a multiple of 64;
K not a multiple of 64) read
zero for out-of-range elements instead of going out of bounds. Every K3
`K` is a multiple of 64, so the K bounds are always satisfied; they are
checked defensively so the kernel is correct for any `(M, N, K)`.

**Two index facts that are easy to get wrong** (both caught by line-by-line
audit before any on-device run; `test_gemm_bf16_correct` would catch them
too):

- The `sB` *source row* carries the K-tile offset: load from
  `B[k_off + row][...]`, not `B[row][...]`, because `B`'s K-tile changes
  every iteration. `sA`'s M-tile is fixed for the block, so only `sA`'s K
  column carries `k_off`. Omitting it silently re-reads the first `BK`
  rows of `B` every K-tile, giving wrong output for `K > 64`.
- The grid is `(⌈M/BM⌉, ⌈N/BN⌉)` with `m_tile = blockIdx.x` (M-tile) and
  `n_tile = blockIdx.y` (N-tile), matching `moe_fused.hip`. Swapping the
  two silently writes only the first `BN` columns for single-M-tile
  shapes (every K3 serving shape).

### Two entry points

- `vkernels::kernels::hip::gemm_bf16(M, N, K, α, A, B, β, C)` — the public
  entry point; calls `gemm_bf16_config_for` and dispatches.
- `vkernels::kernels::hip::gemm_bf16_with_config(... bm, bn, bk, threads)`
  — the explicit-tile dispatcher (not in the public header, so it is not a
  vkl discovery entry; forward-declared by the benchmark and correctness
  harnesses). It is the hook the offline autotuner uses to sweep the seven
  compiled tiles: `(16,16), (16,64), (16,128), (32,64), (32,128),
  (64,64), (64,128)`.

---

## Host tests

`tests/kernels/gemm/test_gemm_bf16.cpp` (100% line coverage of
`gemm_bf16.cpp`):

- A hand-checked `2×2 × identity` GEMM (`α=1, β=0`).
- A non-square `2×3 × 3×2` GEMM with `β = 2` accumulation, bit-exact against
  an independent fp32+RNE reference.
- A bit-exact cross-check against the independent reference across the K3
  shapes (including `N = 6288`, not a multiple of 64).
- `gemm_bf16_config_for` for both branches (`M ≤ 64` → `16×16/64`;
  `M > 64` → `64×64/256`) and the `M == 64` boundary.
- Null-argument contracts throw `std::invalid_argument`.
- `M == 0` / `N == 0` are no-ops.

---

## Device validation and tuning

- **Correctness** (`test_gemm_bf16_correct.hip`): the public entry matches
  the CPU oracle within `max_rel < 2e-2` for every K3 shape at serving
  `M = {8, 64}` and warmup `M = 8192`, and every compiled tile is
  cross-checked via the explicit-config dispatcher on an awkward
  `(M=16, N=100, K=128)` shape. **Run on MI300A (gfx942) at CSCS beverin:
  0 failures, `max_rel ≤ 0.78%`.**
- **Benchmark / autotuner** (`bench_gemm_bf16.hip`): per-shape median
  latency, achieved TFLOP/s, effective GB/s, the binding resource against
  the MI300A roof, and a full `7-tile × 11-shape × 5-M` sweep that
  regenerates `gemm_bf16_config_for` on device.

Both run on gfx942 (ROCm 6.3, `hipcc` AMD clang 18) under
`srun --partition=mi300`; see `docs/performance/gemm-bf16/gfx942.md` for
the **measured** roofline analysis, the autotuner matrix, and the
documented limitations (no cross-tile B reuse; 45% effective HBM from
the two-phase tile).

---

## File layout

```
src/c/vkernels/kernels/
├── gemm_bf16.hpp   # public API (gemm_bf16_cpu, gemm_bf16_config_for,
│                   #                  hip::gemm_bf16)
├── gemm_bf16.cpp   # CPU oracle + config selector (always compiled)
└── gemm_bf16.hip   # HIP MFMA kernel + dispatcher (VKERNELS_HAS_HIP)
```
