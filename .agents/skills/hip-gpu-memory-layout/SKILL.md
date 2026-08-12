---
name: hip-gpu-memory-layout
description: Arrange GPU data in AMD physical memory so accesses coalesce, LDS bank conflicts vanish, and tiles arrive in the layout a Matrix Core expects. Use when choosing a memory space (global, LDS, VGPRs, scratch), designing a tile layout, describing a layout with shape and strides, using named hardware axes (lane id, VGPR, Matrix Core fragment), fixing bank conflicts with swizzling on AMD's 32-bank LDS, placing MFMA accumulators in VGPRs, or sharding across multiple GCDs.
---

# HIP GPU Memory and Layout

The same values can differ in performance by an order of magnitude on the same
GPU depending only on how they are physically arranged. A tensor's logical
shape says nothing about where its bytes live; the **layout** supplies that
physical information. This skill covers the AMD memory spaces, the layout
notation, and swizzling — the three things that decide whether accesses
coalesce, conflict, or match the hardware unit that reads them.

## Memory spaces and their tradeoffs

| Space | Ownership | Role | Notes |
|---|---|---|---|
| Global (HBM) | Device-wide | Persistent tensor storage | Large HBM; shared by all CUs; slow (~300+ cycles) |
| LDS (Local Data Share) | Per-workgroup | Tile staging | Low-latency scratchpad; 64 KB per CU on MI300X; 32 banks |
| VGPR (Vector Register File) | Per-thread | Scalars, per-thread fragments, MFMA accumulators | Fastest; epilogue/temp values; excess cuts occupancy |
| SGPR (Scalar Register File) | Per-wavefront | Uniform constants, addresses, control | One copy shared by 64 threads; 104 SGPRs max |
| Scratch (VGPR spill) | Per-thread | VGPR overflow | Invisible to programmer but kills performance |

> **AMD vs NVIDIA key difference:** AMD has separate SGPR and VGPR register
> files. There is no separate "Tensor Memory" — MFMA accumulators live in
> VGPRs (specifically `acc` VGPRs on CDNA). LDS is the shared memory, 32 banks
> of 4 bytes each (same as NVIDIA SMEM). AMD CUs share an L2 cache but LDS is
> strictly per-CU — no "distributed shared memory" over multiple CUs (no
> direct equivalent of NVIDIA's DSMEM).

On MI300X, there is additional hierarchy: the device has 8 GCDs (Graphics
Compute Dies, akin to 8 separate GPUs on one package), each with its own HBM
partition, L2, and CUs. Programming across GCDs adds another level of memory
management — see [references/multi-gcd.md](references/multi-gcd.md).

## The shape-stride layout model

A layout is written `S[(shape) : (strides)]`. The shape decomposes a flat
logical index; the strides map the resulting coordinates to a physical
location by dot product. A row-major `4×4` matrix is:

```
S[(4, 4) : (4, 1)]
addr(i, j) = i·4 + j·1
```

Most view-producing operations (`permute`, `view`, compatible `reshape`) only
change shape and strides — no data moves. Tiling is the same model after
splitting each index into a tile coordinate and an in-tile coordinate: an `8×8`
matrix tiled `2×4` has shape `(4, 2, 2, 4)` (tile_row, row_in_tile, tile_col,
col_in_tile) and strides chosen so that a tile occupies contiguous storage.

## Named axes: beyond one address

Linear memory has one address axis, written `@m`. Some hardware storage needs
**more than one** coordinate to identify a physical location, so we tag axes:

- **VGPR fragments** (MFMA operands) are distributed across a wavefront's 64
  lanes; a position needs both `@laneid` (lane within the wavefront) and
  `@reg` (per-lane VGPR slot). On AMD, the 64-lane wavefront is the
  fundamental width — unlike NVIDIA's 32-lane warp.
- **MFMA register packing** differs by instruction: `v_mfma_f32_16x16x16f16`
  distributes a `16×16` tile across 64 lanes with specific per-lane element
  patterns. See [references/mfma-fragments.md](references/mfma-fragments.md)
  for the exact packing of each MFMA variant.
- **MI300X multi-GCD** adds a `@gcd` axis (0–7).

## Coalescing and bank conflicts on AMD

### Global memory coalescing

A wavefront of 64 threads accesses global memory in **segments** of 64 bytes
(one cacheline on most CDNA parts). The hardware inspects the addresses across
all active lanes and issues the minimum number of cacheline fetches:

- **Coalesced:** all 64 lanes hit consecutive 4-byte words → 4 cacheline
  fetches (256 bytes / 64 bytes per line). Ideal.
- **Strided:** stride-N access → N× as many cacheline requests.
- **Random:** each lane hits a unique cacheline → up to 64 fetches.

> **AMD vs NVIDIA:** AMD's 64-thread wavefront needs 256-byte aligned accesses
> for ideal coalescing; NVIDIA's 32-thread warp needs 128-byte. The principle
> is identical — the widths differ.

### LDS bank conflicts

LDS is divided into 32 banks; for 4-byte granularity, `bank = (addr // 4) % 32`.
- **Bank conflict on AMD:** in one access cycle, ≥2 threads in the *same
  half-wavefront* (lanes 0–31 OR lanes 32–63) hit different addresses in the
  same bank → the accesses serialize. A wavefront of 64 threads reads LDS in
  two cycles of 32 threads each.
- **Broadcast:** threads reading the *same* address get the word free, no
  conflict.
- A plain row-major tile coalesces row reads but piles a column read onto one
  bank (an `8×8` fp16 tile read by column → 8-way conflict).

## Swizzling for LDS

A **swizzle** rearranges LDS addresses without changing the tile's logical
shape, so the *same* tile supports both row-wise and column-wise access without
conflicts. The common technique XORs part of the row index into the column
index:

```
mapped_col = logical_col XOR (row % atom_rows)
bank = mapped_col % 32
```

An `8×8` example: column 0 now maps rows 0–7 to banks 0–7, so all eight reads
go in parallel instead of serializing on bank 0.

### AMD LDS swizzle patterns

AMD provides hardware-accelerated LDS swizzle instructions:

```
ds_swizzle_b32 v[dst], v[src], 0x8000  ; XOR swizzle pattern
```

Common patterns (encoded in the swizzle pattern bits):
- `0x8000` — `(lane_id XOR 1)` — swap adjacent pairs
- `0x80FF` — broadcast lane 0's value to all
- `0x8140` — rotate 8 lanes
- Custom patterns can XOR, OR, AND, or shift lane IDs

For tile swizzling, the canonical AMD approach is to apply the XOR in the
address calculation before the `ds_write`/`ds_read`:

```cpp
// Thread gathers 8 fp16 values for a column access pattern
// Swizzled address to avoid bank conflicts on column read
uint32_t col = thread_id % 16;
uint32_t row = thread_id / 16;
uint32_t swizzled_col = col ^ (row % 8);  // XOR swizzle, 8-row atom
uint32_t lds_addr = (row * 16 + swizzled_col) * 2;  // 2 bytes per fp16
reinterpret_cast<half*>(&lds[lds_addr])[0] = value;
```

> **AMD vs NVIDIA:** NVIDIA's TMA can apply the swizzle on write
> transparently. On AMD, you must apply the swizzle yourself in the address
> calculation or use `ds_swizzle_b32`. The swizzle pattern must agree between
> the producer and consumer — a mismatch reads correct bytes as wrong
> elements.

### Rules that catch people out:
- The swizzle permutes *only within an atom*. Choose atom sizes that divide
  your tile's contiguous dimension cleanly.
- **Every** operation accessing a swizzled tile — the writer, the reader, and
  the MFMA that reads it — must agree on the swizzle mode.
- AMD's 64-thread wavefront splits LDS access into two 32-thread cycles.
  Bank conflict analysis should consider each half-wavefront separately.

## MFMA accumulator layout (AMD-specific)

Unlike NVIDIA where `tcgen05` writes to a separate TMEM, AMD MFMA instructions
write to VGPRs — specifically **accumulator VGPRs** (`acc0`–`acc255` on CDNA)
that the instruction defines:

```
; v_mfma_f32_16x16x16f16: 16x16 fp32 output, K=16 fp16 inputs
; Accumulator: 4 VGPRs (v[acc:acc+3]) for the full 16x16 tile
; A operand: 2 VGPRs (v[a:a+1]) holding a 16x16 fp16 fragment
; B operand: 2 VGPRs (v[b:b+1]) holding a 16x16 fp16 fragment
v_mfma_f32_16x16x16f16 v[0:3], v[4:5], v[6:7], v[0:3]
```

The exact element-to-lane-to-VGPR mapping depends on the MFMA variant and is
documented in [references/mfma-fragments.md](references/mfma-fragments.md).
rocWMMA handles this packing automatically; raw MFMA requires you to pack
elements correctly.

## Multi-GCD (MI300X)

MI300X has 8 GCDs, each accessible as a separate HIP device. Cross-GCD access
uses `hipDeviceEnablePeerAccess` and `hipMemcpy` between device pointers.
Programs can use all 8 GCDs as:
- **Data-parallel sharding:** split tensors along one axis, each GCD
  processes its slice independently.
- **Model-parallel pipelining:** each GCD processes one layer/stage, passing
  activations to the next GCD.

See [references/multi-gcd.md](references/multi-gcd.md) for full GCD topology
and communication patterns.

## Completion criterion

Every access to your tile is conflict-free (or a broadcast) for its actual
access pattern; the tile is in the physical layout its consumer expects; and
you picked the right swizzle for the contiguous dimension. You can state, for
each element, which device/GCD / bank / lane+VGPR holds it.
