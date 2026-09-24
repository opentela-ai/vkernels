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
  `tests/kernels/moe/test_moe_aux_fused.cpp` (fused sorted-quant, host),
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
            │   (fused alternative: mxfp4_moe_sorted_quant does
            │    gather + quantize in one launch, no A_sorted round-trip)
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

### `mxfp4_moe_sorted_quant`

```cpp
void mxfp4_moe_sorted_quant(const uint16_t* A, const int32_t* sorted_ids,
                            uint8_t* packed, uint8_t* scales, int M,
                            int hidden, int group_size, int top_k, int EM);
```

The fused gather+quantize: produces `mxfp4_moe_quant(mxfp4_moe_sort(A,
sorted_ids, ...))` in a single launch — `packed`/`scales` are the
expert-grouped `[EM, hidden / 2]` / `[EM, n_groups]` quantized activation
consumed directly by the grouped GEMM, with **no `A_sorted` materialization**.
This removes the largest intermediate write+read of the pre-GEMM chain
(`EM x hidden` bf16 out and in, ~58 MB round-trip at K3).

Bit-exact composition contract (checked by
test_moe_aux_fused.cpp and the bench parity gate):

```
mxfp4_moe_sorted_quant(A, sorted_ids)[r] == mxfp4_moe_quant(mxfp4_moe_sort(A, sorted_ids))[r]
```

for every sorted row `r`, real or padding. Padding rows quantize exactly
like real zero activations (`scale = 0xFF`, zero nibbles), matching the
`sort -> quant` route. Note the standalone `quant -> sort_scales` route
instead writes literal `0` scale bytes for padding rows — a pre-existing
asymmetry between the two standalone routes (harmless: the grouped GEMM
never reads padding rows); the fused op follows `sort -> quant`, the route
it replaces.

On the device path the launcher dispatches to the fused kernel by default;
`VK_MOE_AUX_FUSED_QUANT=0` falls back to `sort -> quant` (the proven
two-launch path), so the fusion is an env-gated behavioral change with a
bit-identical fallback.

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
sorted rows map to the same output token (`top_k > 1`). The CPU reference
accumulates into (pre-zeroed) `out`; the HIP kernel (#145) implements the
same math as an atomic-free token-major gather and fully overwrites `out`
(a well-formed `sorted_ids` contributes exactly `top_k` rows per token,
so the caller-side memset is unnecessary on the device path). This is the
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
- **`mxfp4_moe_sorted_quant`**: (#157) same grid/block shape as the quant
  kernel (`EM * n_groups` blocks of `group_size` threads) but each block
  gathers its group's `group_size` bf16 elements directly from `A` via
  `sorted_ids` (row `sorted_ids[r] / top_k`, zero for padding rows) instead
  of reading the materialized `A_sorted`. The amax tree reduction, E2M1
  rounding and ue8m0 packing are the quant kernel's code, so the fused op
  is bit-identical to `sort -> quant` on device by construction.
- **`mxfp4_moe_scatter_reduce`** / **`mxfp4_moe_scatter_reduce_q`**:
  (#145) **token-major gather** — an inverse permutation `inv[sorted_ids[r]] = r`
  (micro-kernel + cached device scratch, or prebuilt via
  `mxfp4_moe_build_inv`) lets each block own one output token and read its
  `top_k` partial rows contiguously; plain stores, no atomics, no
  pre-zeroed `out`. The `_q` kernel dequantizes each group against its
  ue8m0 scale before accumulating. Output is deterministic (fixed
  k-ascending order per token).
  Before #145 the kernels used `atomicAdd` (one block per sorted row,
  16-way same-address contention), running at 45.4% of L2 vs sort's 70%;
  the gather form closes that gap (see the benchmark section).

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
packed_s, scales_s = vk.mxfp4_moe_sorted_quant(A, sorted_ids, group_size=32,
                                               top_k=top_k)   # fused gather+quant
out             = vk.mxfp4_moe_scatter_reduce(partial, topk_w, sorted_ids,
                                              M=M, width=hidden, top_k=top_k)
out             = vk.mxfp4_moe_scatter_reduce_q(partial_q, partial_s, topk_w,
                                                sorted_ids, M=M, width=hidden,
                                                top_k=top_k, group_size=32)
```

`moe_align_block_size` (from `moe_fused`) produces `sorted_ids` /
`expert_ids` and is the entry point for the pipeline above.

---

## Validation on gfx942 (MI300A)

`meta/benchmarks/test_moe_aux_correct.hip` cross-checks the five HIP kernels
(`src/c/vkernels/kernels/moe_aux.hip`, `vkernels::kernels::hip`) against the
CPU oracle (`moe_aux.cpp`, `vkernels::kernels`) on a real MI300A compute node.
The two configs exercised:

| op                          | small (M8 H256 tk2 E4) | K3 (M112 H7168 tk16 E64) |
|-----------------------------|------------------------|---------------------------|
| `mxfp4_moe_quant.packed`    | bit-exact (1024 B)     | bit-exact (401408 B)      |
| `mxfp4_moe_quant.scales`    | bit-exact (64 B)       | bit-exact (25088 B)       |
| `mxfp4_moe_sort`            | bit-exact (16384)      | bit-exact (14565376)      |
| `mxfp4_moe_sort_scales`     | bit-exact (512 B)      | bit-exact (455168 B)      |
| `mxfp4_moe_scatter_reduce`  | rel<1e-5 (max 0)       | rel<1e-5 (max 2.38e-07)   |
| `mxfp4_moe_scatter_reduce_q`| rel<1e-5 (max 0)       | rel<1e-5 (max 1.19e-07)   |

The quant and two sort ops are pure data movement / integer quantization (no
floating accumulation across blocks) and so are bit-exact. The two
scatter-reduce ops reduce `top_k` addends per output token; the device
gather (#145) accumulates in k-ascending order per token while the CPU
oracle walks rows r-ascending, so orders can differ. With `top_k = 2` the
two addends commute (bit-exact), while the K3 shape (`top_k = 16`) has
sixteen addends per token and so genuinely exercises the relative
tolerance — the observed `~2e-7` is two orders of magnitude inside `1e-5`
(and the device order is now deterministic, cf. #132).

Build & run (bare-metal, ROCm 6.3; the compute node has the runtime, so no
container is needed):

```sh
cmake --preset hip -DVKERNELS_BUILD_BENCHMARKS=ON
cmake --build build/hip --target test_moe_aux_correct
srun --partition=mi300 --ntasks=1 --cpus-per-task=16 --time=00:05:00 \
     ./build/hip/meta/benchmarks/test_moe_aux_correct
```

A latent buffer-overflow was caught at the K3 scale: the sort / scales /
scatter buffers were sized `E * BLOCK_M`, which is too small whenever an
expert receives more than `BLOCK_M` assignments (K3 has `~28/expert`, giving
`EM = 2032` against the old bound of `1024`). They are now sized to the safe
upper bound `EM_max = M * top_k + E * BLOCK_M` with a runtime assert.

---

## Latency & bandwidth on gfx942 (MI300A)

`meta/benchmarks/bench_moe_aux.hip` times all five ops at the K3 decode
shape (`M ∈ {8, 32, 64, 112}`, `hidden=7168`, `top_k=16`, `E=64`,
`group_size=32`) with `hipEvent` timing (warmup + median over 100 iters).

Because the K3 problem (1–58 MB per op) fits comfortably in the MI300A's
~256 MB L2, the **warm** timed iterations measure L2 bandwidth, not raw
HBM. Two streaming-copy references bracket the roof:

| reference | size | bandwidth |
|---|---|---|
| HBM copy (L2-bypassed, 4096 blocks) | 512 MB | 2868 GB/s |
| L2 copy (K3-sized, 1024 blocks) | 28 MB | 3219 GB/s |

K3 row (`M=112`, `EM=2048`), latency primary, effective GB/s secondary.
Post-#145 numbers (native ROCm 6.3 run, 2026-09-18, this run's L2 roof
2754 GB/s / HBM 2938 GB/s) — `scatter_reduce` improved 42.4 → 20.2 µs
(2.1×, 95% of the 3219 GB/s L2 roof recorded above) and `_q`
43.7 → 22.6 µs (1.9×, atomic traffic gone; the residual is the E2M1
dequant round-trip), and the caller-side `out` memset left the path:

| op | lat (µs) | GB/s | %L2 |
|---|---|---|---|
| `mxfp4_moe_quant` | 20.8 | 98 | 3.6% |
| `mxfp4_moe_sort` | 26.7 | 2060 | 74.8% |
| `mxfp4_moe_sort_scales` | 8.6 | 100 | 3.6% |
| `mxfp4_moe_scatter_reduce` | **20.2** | **3066** | **95.2% of 3219** |
| `mxfp4_moe_scatter_reduce_q` | **22.6** | **487** | 17.7% |

(pre-#145, same shapes: quant 17.8, sort 24.4, sort_scales 5.6,
scatter_reduce 42.4 @ 45.4% L2, scatter_reduce_q 43.7 @ 7.8%.)

Interpretation:

- **`mxfp4_moe_sort`** (69.9% L2 pre-#145) and **`mxfp4_moe_scatter_reduce`**
  (45.4% L2 pre-#145 → **95% post**) are the pure data-movement paths and
  approach the bandwidth roof; the latter's gap was `atomicAdd` contention
  (16 sorted rows per output token), removed by the #145 token-major
  gather.
- **`mxfp4_moe_quant`** and **`mxfp4_moe_scatter_reduce_q`** are
  partially **compute-bound**: the per-element E2M1 round-trip (branchy
  nibble quantize / dequantize + shared-memory amax reduction) dominates
  over the small memory footprint, so GB/s is low despite fast latency
  (post-#145 the `_q` combine's atomic cost is gone; the remaining 22.6 µs
  is the dequant round-trip).
- **`mxfp4_moe_sort_scales`** (0.8 MB, 5.6 µs) is **launch-bound** — the
  kernel runtime is dominated by dispatch overhead.

All five ops complete in 6–44 µs at K3, a fraction of the fused-GEMM step
they orchestrate (milliseconds), so they add negligible end-to-end latency
while keeping the activation in the quantized layout for bandwidth
reduction.

```sh
cmake --preset hip -DVKERNELS_BUILD_BENCHMARKS=ON
cmake --build build/hip --target moe_aux_bench
srun --partition=mi300 --ntasks=1 --cpus-per-task=16 \
     ./build/hip/meta/benchmarks/moe_aux_bench
```

---

## Fusion candidate — `mxfp4_moe_sorted_quant` (GB10 bring-up, gfx942 pending)

The chain `sort → quant → sort_scales` materializes the `[EM, hidden]` bf16
`A_sorted` (written by sort, read by quant) and re-derives sorted scales
that the quant of sorted rows already produced. The fused op eliminates
both. Validated so far on the GB10 via the CUDA-on-NVIDIA shim path (the
`.hip` compiles under nvcc 13 with `cuda_compat/hip/hip_runtime.h`):

- **Parity**: bit-exact vs the CPU `sort → quant` composition on 11 shape
  classes up to full K3 (`M=112, hidden=7168, top_k=16, EM=2048` →
  7.3 MB packed + 448 KiB scales, 0 mismatched bytes), with
  `VK_MOE_AUX_FUSED_QUANT` in **both** positions. One shape (`gs=4` with a
  mixed finite+NaN group) diverges from the *oracle* identically in both
  gate positions — a pre-existing device-vs-oracle NaN-propagation
difference in the shared amax tree (`(a > b) ? a : b` keeps a NaN that the
  oracle's `if (aa > amax)` drops); the fused kernel is bit-identical to
  the proven device quant kernel by construction.
- **Latency (informative, GB10 — not gfx942)**: K3 shape, 100-iter best-of
  CUDA events: `sort+quant+sort_scales` chain 652 µs → fused 496 µs
  (**−24%**), consistent with dropping the `EM·hidden` bf16 round-trip and
  the extra launch. gfx942 numbers to be recorded with the srun run below.

```sh
# gfx942 A/B (build on login node, run on compute node)
cmake --preset hip -DVKERNELS_BUILD_BENCHMARKS=ON
cmake --build build/hip --target moe_aux_bench test_moe_aux_correct
srun --partition=mi300 --ntasks=1 --gpus=1 ./build/hip/meta/benchmarks/test_moe_aux_correct
srun --partition=mi300 --ntasks=1 --gpus=1 ./build/hip/meta/benchmarks/moe_aux_bench   # fused (default)
srun --partition=mi300 --ntasks=1 --gpus=1 VK_MOE_AUX_FUSED_QUANT=0 \
     ./build/hip/meta/benchmarks/moe_aux_bench                                          # gate-off A/B
```

The bench prints a `parity OK` gate line per shape and refuses to time if
the fused output differs from the `sort → quant` composition by even one
byte, in either gate position.
