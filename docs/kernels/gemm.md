# GEMM — Tiled SGEMM (Single-Precision General Matrix Multiply)

A textbook tiled SGEMM: `C = α · A @ B + β · C`, all `float32`, row-major
layout. This is the baseline dense matrix multiply — simple, readable, and
bit-identical between CPU and GPU — that later optimized variants (TMA,
warp-specialized, Tensor Core) will be measured against.

- **Source (CPU)**: `src/c/vkernels/kernels/gemm.cpp`
- **Source (CUDA)**: `src/c/vkernels/kernels/gemm.cu`
- **Header**: `src/c/vkernels/kernels/gemm.hpp`
- **Tests**: `tests/kernels/gemm/test_gemm.cpp`
- **Python**: `vkernels.kernels.gemm(A, B, alpha=1.0, beta=0.0)`
- **Rust**: `vkernels::kernels::gemm(M, N, K, alpha, A, B, beta, C)`

---

## Computation

```
C[i][j] = α · Σₖ A[i][k] · B[k][j]  +  β · C[i][j]

for i ∈ [0, M),  j ∈ [0, N),  k ∈ [0, K)
```

All matrices are row-major `float32`. Dimensions:

| Matrix | Shape | Size (elements) |
|---|---|---|
| A | M × K | M · K |
| B | K × N | K · N |
| C | M × N | M · N |

`α` and `β` are scalars converted to `float32`. When `β = 0` the previous
contents of `C` are ignored (pure `α · A @ B`).

---

## Contract

| Condition | Error |
|---|---|
| `A.size() == M * K` | `std::invalid_argument` / `ValueError` |
| `B.size() == K * N` | `std::invalid_argument` / `ValueError` |
| `C.size() == M * N` | `std::invalid_argument` / `ValueError` |

Empty dimensions (`M == 0`, `N == 0`, or `K == 0`) are allowed: the loops
execute zero iterations and `C` is unchanged (except for the `β · C` term,
which is always applied even when `K == 0`).

---

## CPU reference

```cpp
void gemm(std::size_t M, std::size_t N, std::size_t K, float alpha,
          Span<const float> A, Span<const float> B, float beta, Span<float> C) {
  VK_EXPECTS(A.size() == M * K, "A must be M*K");
  VK_EXPECTS(B.size() == K * N, "B must be K*N");
  VK_EXPECTS(C.size() == M * N, "C must be M*N");

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (std::size_t k = 0; k < K; ++k)
        acc += A[i * K + k] * B[k * N + j];
      C[i * N + j] = alpha * acc + beta * C[i * N + j];
    }
  }
}
```

Triple-nested loop in `M, N, K` order. The innermost loop is a dot product
over the `K` dimension; the accumulator `acc` is a `float32` local variable.

**Why this order?** `M`-outer / `N`-middle / `K`-inner means `A` is accessed
with stride-1 (cache-friendly) and `B` is accessed with stride-`N` (not
ideal, but matches the tiled GPU kernel's access pattern). It's the simplest
correct reference, not the fastest CPU implementation.

---

## CUDA kernel — tiled 16×16 shared-memory SGEMM

```cuda
constexpr int kTile = 16;

__global__ void gemm_kernel(const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  __shared__ float sA[kTile][kTile];
  __shared__ float sB[kTile][kTile];

  float acc = 0.0f;
  for (int t = 0; t < (K + kTile - 1) / kTile; ++t) {
    // Cooperative load: each thread loads one element of sA and one of sB.
    sA[threadIdx.y][threadIdx.x] =
        (t * kTile + threadIdx.x < K && row < M)
            ? A[row * K + t * kTile + threadIdx.x] : 0.0f;
    sB[threadIdx.y][threadIdx.x] =
        (t * kTile + threadIdx.y < K && col < N)
            ? B[(t * kTile + threadIdx.y) * N + col] : 0.0f;
    __syncthreads();

    for (int k = 0; k < kTile; ++k)
      acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    __syncthreads();
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}
```

### How it works

1. **Grid**: 2-D grid of `(⌈N/16⌉, ⌈M/16⌉)` blocks, each a 2-D
   `(16, 16)` thread block. Thread `(tx, ty)` computes output element
   `C[row][col]`.

2. **Tiling over K**: The K dimension is traversed in `kTile=16`-wide
   strips. For each K-tile `t`:
   - All 256 threads cooperatively load a 16×16 tile of A into `sA` and
     a 16×16 tile of B into `sB`. Each thread loads exactly one element
     of each. Out-of-bounds elements (when K is not a multiple of 16, or
     edge blocks where M/N are not multiples) are padded with zero.
   - `__syncthreads()` ensures the entire tile is visible before the
     compute phase.
   - Each thread computes a partial dot product: 16 multiply-adds over
     the K-tile, reading from `sA` (row-wise) and `sB` (column-wise).
   - A second `__syncthreads()` prevents the next tile load from
     overwriting shared memory that warp-level compute may still be
     reading.

3. **Epilogue**: After all K-tiles are consumed, the accumulated sum is
   scaled by `α`, the existing `C` element scaled by `β`, and the result
   written back.

### Thread-to-data mapping

```
Thread (tx, ty) computes C[blockIdx.y*16 + ty][blockIdx.x*16 + tx].

sA[ty][tx] ← A[row][blockIdx.x*16 + tx]  (row-major A, column = K dimension)
sB[ty][tx] ← B[blockIdx.y*16 + ty][col]  (row-major B, row = K dimension)
```

The `sA` load is coalesced (threads in a warp load consecutive addresses).
The `sB` load is also coalesced because threads `(tx, ty)` within a warp
share the same `ty` and differ in `tx`, loading consecutive elements of
`B` in row-major order.

### Shared-memory bank conflicts

- `sA[ty][tx]`: thread `(tx, ty)` accesses bank `tx` (assuming 32-bit
  words, 32 banks). All 32 threads in a warp have different `tx`, so
  zero bank conflicts on load. During compute, thread `(tx, ty)` reads
  `sA[ty][k]` for `k=0..15` — same `ty`, varying `tx` via `k` — so each
  thread reads from a different bank. Zero conflicts.
- `sB[ty][tx]`: similar analysis. Zero bank conflicts on both load and
  compute phases. This is the default layout advantage when both tiles
  are square and accessed in the natural order.

---

## Usage

### Python

```python
import numpy as np
import vkernels

A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
B = np.array([[1.0], [1.0]], dtype=np.float32)
vkernels.kernels.gemm(A, B)              # → [[3.], [7.]]
vkernels.kernels.gemm(A, B, alpha=2.0)   # → [[6.], [14.]]
# In-place accumulation:
C = np.ones((2, 1), dtype=np.float32)
vkernels.kernels.gemm(A, B, beta=1.0, out=C)  # C += A@B
```

The Python wrapper infers `M, N, K` from the shapes of `A` and `B`. The
`out` parameter can be a flat `(M*N,)` array or a `(M, N)` array.

### Rust

```rust
use vkernels::kernels;
let a = [1.0_f32, 2.0, 3.0, 4.0];   // 2x2 row-major
let b = [1.0, 1.0];                   // 2x1
let mut c = [0.0_f32; 2];
kernels::gemm(2, 1, 2, 1.0, &a, &b, 0.0, &mut c)?;  // c = [3.0, 7.0]
```

Dimensions are explicit in Rust.

---

## Performance characteristics

- **FLOPs**: `2 · M · N · K` (one multiply + one add per inner product
  term).
- **Arithmetic intensity**: `(2·M·N·K) / (4·(M·K + K·N + M·N))` bytes.
  For large square matrices `M=N=K=n`, this is `2n³ / 12n² = n/6`, which
  grows with `n` — large GEMMs are **compute-bound**.
- **Ridge point** on an A100 (19.5 TFLOPS fp32, 2.0 TB/s HBM): `n/6 ≈
  19.5/2.0 = 9.75 FLOP/byte`, crossing at `n ≈ 60`. Matrices smaller than
  ~60×60 are memory-bound; larger ones are compute-bound.
- **Shared memory per block**: `2 × 16 × 16 × 4 bytes = 2048 bytes`. At
  48 KB shared memory per SM, occupancy is limited by registers (each
  thread uses ~10 registers → ~40 KB per block), giving roughly 2–3
  blocks per SM.
- **Current limitations** (this is the baseline):
  - No double-buffering of shared-memory tiles (compute stalls waiting
    for loads).
  - No vectorized loads (each thread loads one `float`, not `float4`).
  - No warp-level matrix multiply (Tensor Core / MFMA).
  - No TMA (Tensor Memory Access) for asynchronous tile movement.

Each of these is a documented optimization step in the GEMM tuning ladder.

---

## Edge cases

| Case | Behavior |
|---|---|
| K = 0 | `acc = 0.0`, so `C = β · C` (pure scaling of C) |
| M = 0 or N = 0 | No output elements written (grid is empty) |
| α = 0.0, β = 0.0 | `C` zeroed |
| Non-multiple dimensions | Padding zeros fill incomplete edge tiles |
| NaN in A or B | Propagates through multiply-add to C |

---

## File layout

```
src/c/vkernels/kernels/
├── gemm.hpp       # public API
├── gemm.cpp       # CPU triple-loop reference
└── gemm.cu        # CUDA tiled 16×16 SGEMM + cuda:: launcher
```
