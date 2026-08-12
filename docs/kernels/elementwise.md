# Element-wise Kernels

Three embarrassingly-parallel pointwise operations: `add`, `scale`, and
`relu`. They share the same execution model — one thread per element, a 1-D
grid mapped linearly over the input — so they differ only in the arithmetic
each thread performs.

- **Source (CPU)**: `src/c/vkernels/kernels/elementwise.cpp`
- **Source (CUDA)**: `src/c/vkernels/kernels/elementwise.cu`
- **Header**: `src/c/vkernels/kernels/elementwise.hpp`
- **Tests**: `tests/kernels/elementwise/test_elementwise.cpp`
- **Python**: `vkernels.kernels.add / scale / relu`
- **Rust**: `vkernels::kernels::add / scale / relu`

---

## `add` — element-wise vector addition

### Computation

```
out[i] = a[i] + b[i]    for i ∈ [0, n)
```

All operands are `float32`.

### Contract

| Condition | Error |
|---|---|
| `a.size() == b.size()` | `std::invalid_argument` / `ValueError` |
| `out.size() == a.size()` | `std::invalid_argument` / `ValueError` |
| `a.size() == 0` | OK (no-op, writes nothing) |

### CPU reference

```cpp
void add(Span<const float> a, Span<const float> b, Span<float> out) {
  VK_EXPECTS(a.size() == b.size(), "a and b must have equal length");
  VK_EXPECTS(a.size() == out.size(), "out must have the same length as inputs");
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
}
```

A straight-line loop. The `VK_EXPECTS` macros throw `std::invalid_argument`
on violation. The empty-input case is handled implicitly (loop body never
executes).

### CUDA kernel

```cuda
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
```

- Launch: `(n + 255) / 256` blocks × 256 threads.
- Each thread handles one element. The `if (i < n)` guard handles
  non-multiple-of-256 `n` cleanly.
- Occupancy is limited by the simple register footprint (~4 registers), so
  the kernel is latency-bound on global-memory reads; performance is
  essentially DRAM bandwidth.

### Usage

```python
# Python
import vkernels
vkernels.kernels.add([1.0, 2.0], [10.0, 20.0])  # → [11., 22.]
```

```rust
// Rust
use vkernels::kernels;
kernels::add(&[1.0_f32, 2.0], &[10.0, 20.0], &mut [0.0; 2])?;
```

---

## `scale` — scalar-vector multiply

### Computation

```
out[i] = α · x[i]    for i ∈ [0, n)
```

`α` is converted to `float32`. All vector operands are `float32`.

### Contract

| Condition | Error |
|---|---|
| `x.size() == out.size()` | `std::invalid_argument` / `ValueError` |

### CPU reference

```cpp
void scale(Span<const float> x, float alpha, Span<float> out) {
  VK_EXPECTS(x.size() == out.size(), "x and out must have equal length");
  for (std::size_t i = 0; i < x.size(); ++i) out[i] = alpha * x[i];
}
```

### CUDA kernel

```cuda
__global__ void scale_kernel(const float* x, float alpha, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = alpha * x[i];
}
```

Launch parameters are identical to `add`: `(n + 255) / 256` × 256. A fused
multiply-add pattern could be used here, but the simple multiply stays closer
to the reference and is sufficient for the baseline.

### Usage

```python
vkernels.kernels.scale([1.0, 2.0, 3.0], 2.0)  # → [2., 4., 6.]
```

---

## `relu` — Rectified Linear Unit

### Computation

```
out[i] = max(x[i], 0)    for i ∈ [0, n)
```

### Contract

| Condition | Error |
|---|---|
| `x.size() == out.size()` | `std::invalid_argument` / `ValueError` |

### CPU reference

```cpp
void relu(Span<const float> x, Span<float> out) {
  VK_EXPECTS(x.size() == out.size(), "x and out must have equal length");
  for (std::size_t i = 0; i < x.size(); ++i)
    out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
```

### CUDA kernel

```cuda
__global__ void relu_kernel(const float* x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmaxf(x[i], 0.0f);
}
```

Uses `fmaxf` (the CUDA built-in) rather than a branch so every warp lane
executes the same instruction — no warp divergence on mixed-sign inputs.

### Edge cases

| Input | Output |
|---|---|
| `+0.0` | `+0.0` |
| `-0.0` | `+0.0` (IEEE 754: `fmaxf(-0.0, 0.0)` = `+0.0`) |
| `NaN` | `NaN` (`fmaxf(NaN, 0.0)` = `NaN` — IEEE propagation) |
| `+∞` | `+∞` |
| `-∞` | `0.0` |

The CPU branch `x[i] > 0.0f ? x[i] : 0.0f` agrees with `fmaxf` on all
non-NaN values but diverges on NaN (the branch takes the `false` leg,
returning `0.0f`). For consistency, treat NaN inputs as undefined behaviour.

### Usage

```python
vkernels.kernels.relu([-1.0, 0.5, 2.0])  # → [0., 0.5, 2.]
```

---

## Performance characteristics

All three kernels are **memory-bound** at any practical problem size:

- **Arithmetic intensity**: 1 FLOP / 4 bytes (add), 1 FLOP / 8 bytes
  (scale), 0 FLOP / 4 bytes (relu) — all well below the ridge point.
- **Binding resource**: HBM bandwidth.
- **Theoretical peak**: DRAM bandwidth × (bytes moved ÷ element size).
- The 256-thread block and tiny register footprint mean these kernels
  achieve high occupancy automatically; tuning effort should go to
  bandwidth-saving measures (vectorised loads, cache hints) rather than
  occupancy.

The CPU reference is always available and bit-identical to the CUDA path
for non-NaN float32 inputs.

---

## File layout

```
src/c/vkernels/kernels/
├── elementwise.hpp     # public API (add, scale, relu)
├── elementwise.cpp     # CPU reference (always compiled)
└── elementwise.cu      # CUDA kernel + cuda:: launchers (VKERNELS_HAS_CUDA)
```
