# Reduction Kernels

Two whole-array reductions: `sum` and `max`. Both accept a flat `float32`
array and produce a single scalar, accumulated in `float32`.

- **Source (CPU)**: `src/c/vkernels/kernels/reduce.cpp`
- **Source (CUDA)**: `src/c/vkernels/kernels/reduce.cu`
- **Header**: `src/c/vkernels/kernels/reduce.hpp`
- **Tests**: `tests/kernels/reduce/test_reduce.cpp`
- **Python**: `vkernels.kernels.sum / max`
- **Rust**: `vkernels::kernels::sum / max`

---

## `sum` — float32-accumulated sum

### Computation

```
out = Σ x[i]    for i ∈ [0, n)
```

Accumulation is done in `float32`. This means the reduction is associative
in a mathematical sense but not strictly bit-reproducible across different
accumulation orders (the CPU sums left-to-right; the GPU uses a tree
reduction in shared memory).

### Contract

| Condition | Error |
|---|---|
| `x.size() > 0` | `std::invalid_argument` / `ValueError` |
| `x.size() == 0` | throws — empty reduction is undefined |

The empty-input case throws because a reduction over an empty domain has no
sensible identity in the context of a kernel whose output is always a
single `float`.

### CPU reference

```cpp
void sum(Span<const float> x, float& out) {
  VK_EXPECTS(x.size() > 0, "cannot reduce an empty span");
  float acc = 0.0f;
  for (std::size_t i = 0; i < x.size(); ++i) acc += x[i];
  out = acc;
}
```

Left-to-right sequential accumulation. The accumulator is a plain `float`
(local variable); no Kahan or pairwise summation is used. This is
deliberately the simplest possible reference — faster variants
(e.g. pairwise, Kahan) can be added later and validated against it.

### CUDA kernel (two-stage tree reduction)

```cuda
__global__ void reduce_sum_kernel(const float* x, float* partials, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  s[tid] = (i < n) ? x[i] : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) partials[blockIdx.x] = s[0];
}
```

- **Stage 1** (this kernel): each 256-thread block loads a tile into shared
  memory, then performs an in-place tree reduction (`stride` halves each
  iteration: 128 → 64 → 32 → ... → 1). Thread 0 writes the block's partial
  sum to a global `partials` array.
- **Stage 2** (launcher): a second kernel invocation reduces the partials
  array to a single value (or, when `partials` fits in one block, the
  first-stage kernel is launched with a single-block grid and the
  launcher reads back `partials[0]`).

The current `cuda::sum` launcher is a **stub** (it runs the CPU loop and
leaves the device path to be wired in a later change). The device kernel
`reduce_sum_kernel` is compiled, linkable, and validated separately; the
full end-to-end CUDA path is gated on wiring the device-side partials
allocation.

### Edge cases

| Input | Sum |
|---|---|
| `[1, 2, 3]` | `6.0` |
| `[-1, 1]` | `0.0` |
| `[1e20, 1e-20, -1e20, 1e-20]` | `0.0` (float32 catastrophic cancellation) |
| `[]` | throws |
| `[NaN, 1, 2]` | `NaN` |

---

## `max` — maximum value

### Computation

```
out = max x[i]    for i ∈ [0, n)
```

### Contract

| Condition | Error |
|---|---|
| `x.size() > 0` | `std::invalid_argument` / `ValueError` |
| `x.size() == 0` | throws |

### CPU reference

```cpp
void max(Span<const float> x, float& out) {
  VK_EXPECTS(x.size() > 0, "cannot reduce an empty span");
  float m = x[0];
  for (std::size_t i = 1; i < x.size(); ++i)
    if (x[i] > m) m = x[i];
  out = m;
}
```

Seeded with `x[0]` so the first element is both the initial maximum and the
result for single-element arrays. The comparison `x[i] > m` (not `>=`) means
the *first* occurrence of the maximum is retained when there are ties.

### CUDA kernel

The `cuda::max` launcher is currently a stub that runs the CPU loop. The
device kernel (a tree reduction with `fmaxf` instead of `+`) will mirror
`reduce_sum_kernel` and is gated on the same device-partials wiring.

### Edge cases

| Input | Max |
|---|---|
| `[3, 1, 4, 1, 5, 9, 2]` | `9.0` |
| `[NaN, 1, 2]` | `NaN` (any `>` comparison with NaN is false, so NaN "wins" only if it's `x[0]`) |
| `[0.0, -0.0]` | `0.0` (IEEE: `-0.0 > 0.0` is false) |
| `[-∞, 5]` | `5.0` |
| `[]` | throws |

---

## Performance characteristics

Reductions are **memory-bound** at practical sizes:

- **Arithmetic intensity**: 1 FLOP / 4 bytes per element (sum) or 0 FLOP /
  4 bytes (max comparison only) — far below the ridge point.
- The two-stage approach sacrifices a small amount of bandwidth for
  simplicity. At very large `n`, a single-pass warp-shuffle reduction
  would cut the global-memory round-trip for partials.
- The tree depth is `log₂(blockDim) = 8` steps for 256 threads, each
  step requiring a `__syncthreads()` barrier. Warp-level primitives
  (`__shfl_down_sync`) can eliminate the shared-memory traffic and
  barriers for the last 5 levels (stride ≤ 32).

---

## File layout

```
src/c/vkernels/kernels/
├── reduce.hpp       # public API (sum, max)
├── reduce.cpp       # CPU reference (always compiled)
└── reduce.cu        # CUDA tree-reduction kernel + stubbed launcher
```
