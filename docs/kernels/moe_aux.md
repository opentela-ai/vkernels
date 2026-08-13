# MXFP4 MoE Aux — Quantization, Sort & Routed Scatter-Reduce

The five orchestration ops from issue #22 that bracket a grouped MXFP4
GEMM on gfx942. On gfx950 these are provided by AITER's
`module_moe_mxfp4_aux` (JIT kernels whose ~82 KB LDS requirement exceeds
the 64 KB limit of gfx942 / MI300A). vkernels re-implements them as
portable host references plus HIP kernels so a self-contained W4A4 MoE
serving path can be assembled on gfx942.

The five ops are the small data-movement primitives that turn a routed
MoE layer into something a grouped GEMM can consume. They are **not** the
GEMM itself — that is [`fused_moe_mxfp4`](moe_fused.md).

- **Source (CPU)**: `src/c/vkernels/kernels/moe_aux.cpp`
- **Source (HIP)**: `src/c/vkernels/kernels/moe_aux.hip`
- **Header**: `src/c/vkernels/kernels/moe_aux.hpp`
- **Tests**: `tests/kernels/moe/test_moe_aux.cpp` (host),
  `tests/python/test_kernels.py::MoeAuxTest` (compiled vs fallback, K3 pipeline)

---

## Pipeline

```
            moe_align_block_size (see moe_fused.hpp)
   topk_ids ───────────────────────► sorted_ids [EM], expert_ids [EM]
            │
            ▼
            mxfp4_moe_sort          gather A          [M, hidden] → [EM, hidden]
            mxfp4_moe_sort_scales   gather scales     [M, n_groups] → [EM, n_groups]
            mxfp4_moe_quant         per-token / block E2M1 + ue8m0
            │
            ▼
            fused_moe_mxfp4 (see moe_fused.md)
            │   partial [EM, hidden]  (fp32, expert-local)
            ▼
            mxfp4_moe_scatter_reduce[_q]   routed combine → out [M, hidden]
```

`moe_align_block_size` produces the index map (`sorted_ids`, `expert_ids`,
`EM`). The two **sort** ops gather the activation and its per-token scales
into expert-grouped, block-aligned rows. `mxfp4_moe_quant` quantizes the
gathered activation to MXFP4. After the grouped GEMM, the **scatter-reduce**
op folds the `top_k` partials back onto each output token (bias-free
weighted add; the bias is folded into the W4A4 down-projection separately).
`mxfp4_moe_scatter_reduce_q` is the bandwidth-reduced form: its partial is
already in MXFP4 layout and is dequantized **inline** during the scatter,
matching the AITER `mxfp4_moe_scatter_reduce_q` signature.

---

## Data formats

### Activations — bf16

`A` is `uint16_t*` holding bf16 bit patterns, shape `[M, hidden]`,
row-major. bf16 ↔ float32 uses the standard `bits << 16` / round-to-nearest
`>> 16` pair (shared with `moe.cpp` / `moe_fused.cpp`).

### Packed MXFP4 — E2M1 + ue8m0

| Tensor | Shape | Element | Packing |
|---|---|---|---|
| `packed` | `[M, hidden / 2]` | uint8 | 2 E2M1 nibbles per byte |
| `scales` | `[M, hidden / group_size]` | uint8 | ue8m0 per group |

E2M1 is `sign | 2-bit-exponent | 1-bit-mantissa`, two values per byte, low
nibble first (even K index). The 16 nibble codes decode to
`{0, ±0.25, ±1.0, ±1.5, ±2.0, ±3.0, ±inf, NaN}`, matching `fp4_nibble_to_float`
in `moe.cpp`. The largest **finite** value is `FP4_MAX = 3.0f`.

ue8m0 is an unsigned 8-bit exponent with no mantissa: `2^(s - 127)`, or
`0.0` when `s == 0xFF`. One scale is shared across `group_size` consecutive
hidden elements.

### Scale selection

For each `(token m, group g)` with group amplitude `amax = max_i |x_i|`:

```
if amax == 0 or non-finite:   scales[m, g] = 0xFF, all nibbles = 0
else:                         e  = ceil(log2(amax / FP4_MAX))
                              sb = clamp(e + 127, 1, 254)
                              scale = 2^(sb - 127)
```

Each element `x_i / scale` is then rounded to the nearest representable
E2M1 value, **ties round to the larger magnitude** (breakpoints
`0.125, 0.625, 1.25, 1.75, 2.5`). The scale byte is clamped to `[1, 254]`
— never `0` — so that a group with a tiny-but-nonzero `amax` (for example a
bf16 minimum-normal) still dequantizes through a finite `2^(sb - 127)`
rather than `ue8m0_to_float(0) = 0.0`. `0xFF` is reserved as the explicit
zero-group flag.

### Constraints

```
hidden % 2 == 0
hidden % group_size == 0
group_size > 0
EM >= M * top_k            (moe_align_block_size pads to a block boundary)
```

---

## Operations

All C++ entry points live in `namespace vkernels::kernels` (the CPU
references, in `moe_aux.cpp`). The HIP kernels live in
`vkernels::kernels::hip` (`moe_aux.hip`) with identical signatures; they
are self-declared in the `.hip` (not re-declared in the header) so the host
reference names stay unique in the `vkl` discovery list.

### `mxfp4_moe_quant`

```cpp
void mxfp4_moe_quant(const uint16_t* A, uint8_t* packed, uint8_t* scales,
                     int M, int hidden, int group_size);
```

Per-token, per-group MXFP4 activation quantization. For each group of
`group_size` consecutive hidden elements: compute `amax`, pick a ue8m0
scale so that no element exceeds `FP4_MAX`, and pack the two nearest-E2M1
nibbles per byte (low nibble = even K index). Zero/non-finite groups emit
`scale = 0xFF` and all-zero nibbles, dequantizing to exactly zero.

### `mxfp4_moe_sort`

```cpp
void mxfp4_moe_sort(const uint16_t* A, const int32_t* sorted_ids,
                    uint16_t* A_sorted, int M, int hidden, int top_k, int EM);
```

Gather `A` into the expert-grouped, block-aligned layout the grouped GEMM
consumes. For each sorted row `r`, `A_sorted[r, :] = A[sorted_ids[r] // top_k, :]`.
Padding rows (`r >= M * top_k`, up to `EM`) are zeroed.

### `mxfp4_moe_sort_scales`

```cpp
void mxfp4_moe_sort_scales(const uint8_t* scales, const int32_t* sorted_ids,
                           uint8_t* scales_sorted, int M, int n_groups,
                           int top_k, int EM);
```

The same gather as `mxfp4_moe_sort`, applied to the per-token ue8m0 scales
`[M, n_groups]`. This is the bridge between `mxfp4_moe_quant` run in token
order and the grouped GEMM, which reads scales in sorted order. Running
`mxfp4_moe_sort_scales(scales_tok, sorted_ids)` must therefore yield
exactly the scales that `mxfp4_moe_quant(A_sorted)` would produce — both
express the per-group scale of token `sorted_ids[r] // top_k` at sorted
row `r`.

### `mxfp4_moe_scatter_reduce`

```cpp
void mxfp4_moe_scatter_reduce(const float* partial, const float* topk_w,
                              const int32_t* sorted_ids, float* out,
                              int M, int width, int top_k, int EM);
```

Routed combine of the expert-local fp32 partials back onto each output
token — a bias-free weighted scatter-add. For each sorted row `r`,
`out[sorted_ids[r] // top_k, :] += topk_w[r // top_k, r % top_k] * partial[r, :]`
(the routing weight is looked up by token, not by sorted row). Multiple
sorted rows map to the same output token (`top_k > 1`), so the HIP kernel
uses `atomicAdd`; the caller must zero-initialise `out`. This is the
bias-free form of `moe_combine_cpu` (`moe_fused.cpp`), which adds bias +
weight + scatter in one pass.

### `mxfp4_moe_scatter_reduce_q`

```cpp
void mxfp4_moe_scatter_reduce_q(const uint8_t* partial_q,
                                const uint8_t* partial_s,
                                const float* topk_w,
                                const int32_t* sorted_ids, float* out,
                                int M, int width, int top_k, int EM,
                                int group_size);
```

The bandwidth-reduced combine: the partial is stored in MXFP4 layout
(`partial_q` E2M1 + `partial_s` ue8m0, `group_size` per group) and is
dequantized **inline** during the scatter, element-by-element. The
accumulation and routing weight lookup are identical to
`mxfp4_moe_scatter_reduce`; only the partial is quantized, so the
down-projection result travels at quarter precision on the way back.

---

## HIP kernels

The host reference (`moe_aux.cpp`) is always compiled and is the oracle.
The HIP kernels (`moe_aux.hip`, gated on `VKERNELS_HAS_HIP`) mirror the
algorithms:

- **`mxfp4_moe_quant`**: grid `M * n_groups`, block `group_size` (≤ 256),
  shared-memory amax tree reduction with a nibble-staging buffer.
- **`mxfp4_moe_sort`** / **`mxfp4_moe_sort_scales`**: one block per sorted
  row, a generic gather over `elem = sizeof(element)` bytes (bf16 → 2,
  ue8m0 → 1), strided across the row.
- **`mxfp4_moe_scatter_reduce`** / **`mxfp4_moe_scatter_reduce_q`**:
  `atomicAdd` into `out` so multiple sorted rows mapping to the same token
  accumulate correctly, mirroring `down_combine_kernel` in `moe_fused.hip`.
  The `_q` kernel dequantizes each group against its ue8m0 scale before
  the atomic add.

---

## K3 configuration

The orchestration path is exercised end-to-end at the K3 token shape
`(M = 112, hidden = 7168, ispp = 3072, top_k = 16)` with `E = 64`
experts. `MoeAuxTest.test_pipeline_k3` runs the full
`align → sort → quant → sort_scales → (oracle GEMM) → scatter_reduce`
sequence against the CPU oracle and checks that the routed combine
reconstructs the weighted token output. The grouped GEMM stage uses the
host reference; on hardware it is replaced by `hip::fused_moe_mxfp4`.

---

## Python API

All five ops are exposed through `vkernels.kernels` with a compiled
backend (pybind11, `src/python/vkernels/_core.cpp`) and a pure-Python
fallback (`src/python/vkernels/_fallback.py`). The two implementations are
cross-checked bit-exactly (`mxfp4_moe_quant`, the two sorts, the two
scale/sort outputs) or allclose (`mxfp4_moe_scatter_reduce[_q]`) by
`MoeAuxTest.test_backend_consistency`.

```python
packed, scales = vk.mxfp4_moe_quant(A, group_size=32)          # A: bf16 uint16
A_sorted        = vk.mxfp4_moe_sort(A, sorted_ids, top_k=top_k)
scales_sorted   = vk.mxfp4_moe_sort_scales(scales, sorted_ids, top_k=top_k)
out             = vk.mxfp4_moe_scatter_reduce(partial, topk_w, sorted_ids,
                                              M=M, width=hidden, top_k=top_k)
out             = vk.mxfp4_moe_scatter_reduce_q(partial_q, partial_s, topk_w,
                                                sorted_ids, M=M, width=hidden,
                                                top_k=top_k, group_size=32)
```

`moe_align_block_size` (from `moe_fused`) produces `sorted_ids` /
`expert_ids` and is the entry point for the pipeline above.
