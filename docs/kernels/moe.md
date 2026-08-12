# MoE Low-Level Primitives — AMD gfx942 / CDNA3

Four low-level GPU primitives that fill gaps where CDNA4-only (gfx950)
instructions in the AITER flydsl MXFP4 fused-MoE path do **not** lower on
gfx942 (CDNA3, e.g. MI300X/MI300A). Together they provide the building
blocks used by the fused-MoE grouped GEMM kernel (`moe_fused.hip`).

- **Source (CPU)**: `src/c/vkernels/kernels/moe.cpp`
- **Source (HIP)**: `src/c/vkernels/kernels/moe.hip`
- **Header**: `src/c/vkernels/kernels/moe.hpp`
- **Tests**: `tests/kernels/moe/test_moe.cpp`
- **Python**: `vkernels.kernels.direct_lds_fill_bf16 / fp4_to_bf16_dequant / mfma_f32_16x16x16bf16 / use_async_copy_default`

---

## #12 — Software direct-to-LDS fill (`direct_lds_fill_bf16`)

Replaces the CDNA4-only `rocdl.raw_ptr_buffer_load_lds` intrinsic with a
software path that works on gfx942.

### Computation

```
lds_dst[lane_id · ept + k] = global_src[lane_id · ept + k]
    for k ∈ [0, ept)
    where ept = elements_per_thread = 8 (16 bytes / 128 bits)
```

Each thread loads 8 bf16 values (128 bits, `uint4`) from global memory and
writes them to LDS at the same lane-major offset the hardware path would
have produced.

### HIP implementation

```cpp
__device__ void direct_lds_fill_bf16(void* lds_dst, const void* global_src,
                                     int elements_per_thread) {
  int tid = threadIdx.x;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(global_src)
                        + tid * (elements_per_thread / 2);
  uint32_t* dst = reinterpret_cast<uint32_t*>(lds_dst)
                  + tid * (elements_per_thread / 2);

  uint4 v = *reinterpret_cast<const uint4*>(src);
  dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
}
```

- Uses `uint4` (128-bit) loads: the compiler emits `global_load_dwordx4`
  for aligned 128-bit pointer dereferences on AMD GPUs.
- Each thread loads exactly 4 dwords = 8 bf16 values = 16 bytes.
- The address in global is lane-major: thread `N` loads elements
  `[N*ept .. N*ept+ept-1]`.

### CPU reference

```cpp
void direct_lds_fill_bf16(void* lds_dst, const void* global_src,
                          std::size_t elements) {
  VK_EXPECTS(lds_dst != nullptr && global_src != nullptr || elements == 0,
             "lds_dst and global_src must not be null for non-empty copy");
  std::size_t bytes = elements * sizeof(uint16_t);
  std::memcpy(lds_dst, global_src, bytes);
}
```

Plain `memcpy` — the host doesn't have LDS, so the copy is to/from regular
memory.

### Contract

| Condition | Behavior |
|---|---|
| `elements == 0` | No-op (pointers may be null) |
| `elements > 0`, null pointers | `std::invalid_argument` |

---

## #13 — Software fp4→bf16 dequant (`fp4_to_bf16_dequant`)

Converts packed MXFP4 E2M1 (microscaling) values to bf16 bit patterns,
with an optional per-block scale factor.

### E2M1 format

Each byte holds **two** fp4 values: low nibble first, high nibble second.

```
  bits:  [s] [e1 e0] [m]
         sign  exp    mantissa

  Normal  (e ∈ {1, 2}):    (-1)^s × 2^(e-1) × (1 + m/2)
  Subnorm (e = 0, m = 1):  (-1)^s × 0.25
  Zero    (e = 0, m = 0):  (-1)^s × 0
  Inf     (e = 3, m = 0):  (-1)^s × ∞
  NaN     (e = 3, m = 1):  NaN
```

### Representable values

| Nibble | Value | Nibble | Value |
|---|---|---|---|
| `0000` | +0.0 | `1000` | -0.0 |
| `0001` | +0.25 | `1001` | -0.25 |
| `0010` | +1.0 | `1010` | -1.0 |
| `0011` | +1.5 | `1011` | -1.5 |
| `0100` | +2.0 | `1100` | -2.0 |
| `0101` | +3.0 | `1101` | -3.0 |
| `0110` | +∞ | `1110` | -∞ |
| `0111` | NaN | `1111` | NaN |

### Computation

```
for each byte b in packed:
    out[2i]   = round_to_bf16( fp4_to_float(b & 0x0F) · scale )
    out[2i+1] = round_to_bf16( fp4_to_float(b >> 4)    · scale )
```

The conversion to bf16 uses round-to-nearest-even (RNE).

### HIP kernel

```cuda
__global__ void fp4_to_bf16_kernel(const uint8_t* packed, uint16_t* out,
                                   std::size_t n_packed, float scale) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_packed) return;

  uint8_t byte = packed[i];
  uint32_t flo = fp4nib_to_f32bits(byte & 0x0F);
  uint32_t fhi = fp4nib_to_f32bits((byte >> 4) & 0x0F);

  float vlo, vhi;
  std::memcpy(&vlo, &flo, sizeof(float));
  std::memcpy(&vhi, &fhi, sizeof(float));
  vlo *= scale; vhi *= scale;

  uint32_t bits_lo, bits_hi;
  std::memcpy(&bits_lo, &vlo, sizeof(float));
  std::memcpy(&bits_hi, &vhi, sizeof(float));

  out[i * 2]     = f32bits_to_bf16(bits_lo);
  out[i * 2 + 1] = f32bits_to_bf16(bits_hi);
}
```

- Launch: `(n_packed + 255) / 256` blocks × 256 threads.
- One thread per packed byte → two output bf16 values.
- The `fp4nib_to_f32bits` function is `__host__ __device__` (callable from
  both CPU and GPU) and uses pure integer bit manipulation to construct
  the float32 bit pattern without a `switch` or branch table — critical
  for GPU performance (no warp divergence).

### Round-to-nearest-even for bf16

```cpp
__host__ __device__ uint16_t f32bits_to_bf16(uint32_t bits) {
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7FFFu + lsb;   // add half-ULP of the truncated 16 LSBs
  return static_cast<uint16_t>(bits >> 16);
}
```

Adds `0x7FFF` (half the LSB of the kept 16 bits) plus the low bit of the
result for tie-breaking to even.

### Contract

| Condition | Error |
|---|---|
| `out.size() == packed.size() * 2` | `std::invalid_argument` |

### Usage

```python
import numpy as np
from vkernels.kernels import fp4_to_bf16_dequant

# Two fp4 values in one byte: low nibble = +1.0, high nibble = -2.0
packed = np.array([0xC2], dtype=np.uint8)  # low=0x2 (+1.0), high=0xC (-2.0)
bf16 = fp4_to_bf16_dequant(packed, scale=1.0)
# bf16[0] ≈ 0x3F80 (bf16 representation of 1.0)
# bf16[1] ≈ 0xC000 (bf16 representation of -2.0)
```

---

## #14 — Platform async-copy gate (`use_async_copy_default`)

A boolean gate that tells the fused-MoE pipeline whether to use hardware
async-copy (copy-engine overlap) or fall back to synchronous loads.

### Behavior

| Platform | Default |
|---|---|
| gfx942 (CDNA3, MI300X/A) | `false` — async copy misbehaves |
| gfx950 (CDNA4) | `true` — async copy works correctly |
| CPU (host build) | `true` |
| All others | `true` |

Override with environment variable: `K3_NO_ASYNC=0` forces ON, `=1` forces OFF.

### HIP implementation

```cpp
bool use_async_copy_default() {
  const char* env = std::getenv("K3_NO_ASYNC");
  if (env) return env[0] != '1';

  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, 0);
  if (std::strstr(props.gcnArchName, "gfx942") != nullptr)
    return false;   // CDNA3
  return true;
}
```

### Contract

Always returns a `bool`. Never throws. The function is a pure query with
no side effects.

---

## #15 — K16 bf16 MFMA (`mfma_f32_16x16x16bf16`)

Issues a single `v_mfma_f32_16x16x16bf16_1k` instruction on gfx942. On
gfx942 the CDNA4-only `v_mfma_f32_16x16x32_bf16` (K32 bf16 MFMA) does not
select; the K32 is emulated by calling this K16 function **twice** — once
for K=0..15 (low halves of A and B fragments), once for K=16..31 (high
halves).

### Computation

```
C[0..3] += A[0..1] ⊗ B[0..1]    (16×16×16 bf16, fp32 accumulator)
```

This is a **warp-group** instruction: all 64 lanes in a warp cooperate to
compute a 16×16 output tile, with each lane holding 4 float32 accumulators.

### Fragment layout

Each lane within a 64-thread warp holds:

| Register | Content | Shape |
|---|---|---|
| `C[0..3]` | 4 float32 accumulators | A 4×4 sub-tile of the 16×16 output |
| `A[0..1]` | 4 bf16 values packed into 2 uint32 | 1×4 vector from A |
| `B[0..1]` | 4 bf16 values packed into 2 uint32 | 4×1 vector from B |

The lane's 4 accumulators correspond to a 4×4 block within the 16×16 output
tile. Each accumulator `C[i]` is the dot product of the lane's 4 A-values
and 4 B-values.

### HIP inline assembly

```cpp
__device__ void mfma_f32_16x16x16bf16(float c[4], const uint32_t a[2],
                                      const uint32_t b[2],
                                      int cbsz, int abid, int blgp) {
  register float r0 asm("v0"), r1 asm("v1"), r2 asm("v2"), r3 asm("v3");
  register uint32_t s0 asm("v4"), s1 asm("v5");
  register uint32_t t0 asm("v6"), t1 asm("v7");

  // Load from memory into named registers
  __builtin_memcpy(&r0, &c[0], 4); /* ... r1,r2,r3 similarly */
  s0 = a[0]; s1 = a[1]; t0 = b[0]; t1 = b[1];

  __asm__ __volatile__(
      "v_mfma_f32_16x16x16bf16_1k v[0:3], v[4:5], v[6:7], v[0:3]"
      : "+v"(r0), "+v"(r1), "+v"(r2), "+v"(r3)
      :  "v"(s0),  "v"(s1),  "v"(t0),  "v"(t1));

  __builtin_memcpy(&c[0], &r0, 4); /* ... write back */
}
```

- Uses named VGPR variables (`asm("v0")` through `asm("v7")`) to guarantee
  the consecutive register ranges the instruction demands.
- The accumulator registers are both inputs and outputs (`+v` constraint).
- `cbsz`, `abid`, `blgp` are the MFMA control flags that set the
  precision/rounding mode; they are accepted but ignored in the assembly
  (the `_1k` variant hardcodes these).

### CPU reference

```cpp
void mfma_f32_16x16x16bf16(float c[4], const uint32_t a[2],
                           const uint32_t b[2],
                           int /*cbsz*/, int /*abid*/, int /*blgp*/) {
  // Unpack bf16 pairs to float
  float a_f32[4], b_f32[4];
  for (int i = 0; i < 2; ++i) {
    uint16_t lo = a[i] & 0xFFFF, hi = a[i] >> 16;
    /* reinterpret as float */
  }
  for (int i = 0; i < 4; ++i)
    c[i] += a_f32[i] * b_f32[i];
}
```

The CPU reference unpacks the packed bf16 and computes the per-thread dot
products in float32. The MFMA control flags are ignored on the host path.

### K32 emulation pattern

```cpp
// One K32 MFMA = two K16 MFMAs
// A is [64 lanes, K=32] → A_lo = first 16 K, A_hi = second 16 K
// B is [64 lanes, K=32] → B_lo = first 16 K, B_hi = second 16 K
mfma_f32_16x16x16bf16(c, a_lo, b_lo, 0, 0, 0);  // K = 0..15
mfma_f32_16x16x16bf16(c, a_hi, b_hi, 0, 0, 0);  // K = 16..31
// Result: C += A_lo·B_lo + A_hi·B_hi ≡ one K32 dot product
```

### Usage

```python
from vkernels.kernels import mfma_f32_16x16x16bf16

c = [0.0, 0.0, 0.0, 0.0]
# bf16(1.0) = 0x3F80; pack two into each uint32
a = [0x3F803F80, 0x3F803F80]
b = [0x3F803F80, 0x3F803F80]
mfma_f32_16x16x16bf16(c, a, b)
# c → [1.0, 1.0, 1.0, 1.0]
```

---

## File layout

```
src/c/vkernels/kernels/
├── moe.hpp       # public API (#12–#15)
├── moe.cpp       # CPU references (always compiled)
└── moe.hip       # HIP kernels + hip:: launchers (VKERNELS_HAS_HIP)
```
