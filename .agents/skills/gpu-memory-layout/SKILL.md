---
name: gpu-memory-layout
description: Arrange GPU data in physical memory so accesses coalesce, shared-memory bank conflicts vanish, and tiles arrive in the layout a Tensor Core expects. Use when choosing a memory space (global, shared, Tensor Memory, registers, distributed shared memory), designing a tile layout, describing a layout with shape and strides, using named hardware axes (lane id, register, TMEM lane/column), fixing bank conflicts with swizzling, placing MMA accumulators or block-scaled scale factors, or sharding across GPUs.
---

# GPU Memory and Layout

The same values can differ in performance by an order of magnitude on the same
GPU depending only on how they are physically arranged. A tensor's logical
shape says nothing about where its bytes live; the **layout** supplies that
physical information. This skill covers the spaces, the layout notation, and
swizzling — the three things that decide whether accesses coalesce, conflict,
or match the hardware unit that reads them.

## Memory spaces and their tradeoffs

| Space | Ownership | Role | Notes |
|---|---|---|---|
| Global (GMEM) | Device-wide | Persistent tensor storage | Large HBM; shared by all SMs; slow |
| Shared (SMEM) | Per-CTA | Tile staging | Low-latency scratchpad; ≤228 KB/SM on B200; 32 banks |
| Tensor Memory (TMEM) | Per-CTA | MMA accumulator storage | Blackwell; 128 lanes × ≤512 32-bit cols; used by `tcgen05` |
| Register file (RF) | Per-thread | Scalars, per-thread fragments | Fastest; epilogue/temp values; excess cuts occupancy |

A cluster adds **distributed shared memory (DSMEM)**: a CTA can read another
CTA's SMEM without a GMEM round trip. A high-performance kernel's central task
is moving data efficiently *between* these spaces.

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

- **TMEM** is inherently 2-D: 128 lane rows × ≤512 32-bit columns. A `128×256`
  accumulator is `S[(128, 256) : (1@TLane, 1@TCol)]` — the layout returns both
  a lane and a column, not one integer.
- **Register fragments** (Tensor Core operands) are distributed across a warp's
  32 lanes; a position needs both `@laneid` (lane within the warp) and `@reg`
  (per-lane slot). An `m8n8` fragment is `S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]`
  with `laneid = row·4 + col//2`, `reg = col%2`. Low-precision packs multiple
  elements per 32-bit register.

See [references/tmem-and-fragments.md](references/tmem-and-fragments.md) for
TMEM allocation, accumulator readback, block-scaled SFA/SFB packing, and
multi-GPU mesh layouts.

## Replication and offset

A layout returns one physical location per logical element, but hardware
sometimes holds several **copies**. Append a replication term:

- `R[n : s@axis]` — `n` independent copies at stride `s` along `axis`. E.g.
  Blackwell `.warpx4` replicates one 32-lane TMEM base tile across the four
  partitions: `S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]`.
- `O[s@axis]` — a fixed *translation* (shift by `s`), not a copy. Used for
  sharding an element to one device, or for fixed lane/column offsets.

## Coalescing and bank conflicts

SMEM is divided into 32 banks; for 4-byte granularity, `bank = (addr // 4) % 32`.
- **Bank conflict**: in one processing batch (a "wavefront"), ≥2 lanes hit
  *different* addresses in the same bank → the accesses serialize. A wavefront
  moves ≤128 bytes (4 B × 32 banks); access width sets the grouping (4 B/lane
  → 32 lanes/wavefront, 8 B → groups of 16, 16 B → groups of 8). Conflicts are
  scored only within a wavefront.
- **Broadcast**: lanes reading the *same* address get the word free, no
  conflict.
- A plain row-major tile coalesces row reads but piles a column read onto one
  bank (an `8×8` fp16 tile read by column → 8-way conflict).

## Swizzling

A **swizzle** rearranges SMEM addresses without changing the tile's logical
shape, so the *same* tile supports both row-wise and column-wise access without
conflicts. The common technique XORs part of the row index into the column
index:

```
mapped_col = logical_col XOR (row % atom_rows)
bank = mapped_col
```

An `8×8` example: column 0 now maps rows 0–7 to banks 0–7, so all eight reads
go in parallel instead of serializing on bank 0.

**Atom sizes** are `8 rows × W bytes`: `SWIZZLE_128B` (`W=128`), `SWIZZLE_64B`
(`W=64`), `SWIZZLE_32B` (`W=32`). Choose the largest row width the tile's
contiguous dimension supports (and is preferably divisible by): if the
contiguous dimension is ≥128 bytes (64 fp16), use `SWIZZLE_128B`.

**Rules that catch people out:**
- The swizzle permutes *only within an atom*. The innermost contiguous
  dimension of a TMA box must not exceed the swizzle width. If the data is
  narrower than the swizzle width, allocate the full width anyway.
- Row stride matters even with swizzling. A 256-byte row stride (two 128-byte
  spans) can cause 2-way conflicts across rows; reshape into explicit 128-byte
  groups so adjacent rows within a group are 128 bytes apart.
- **Every** operation accessing a tile — the TMA descriptor that writes it, the
  SMEM layout, and the MMA that reads it — must use the *same* swizzle mode.
  TMA can apply the swizzle on write; if the consumer reads it as linear, the
  bytes are present but interpreted as the wrong matrix elements.

The swizzle is a non-affine XOR transform composed with the affine layout, not
part of it. Full detail with the bank arithmetic is in
[references/swizzle.md](references/swizzle.md).

## Completion criterion

Every access to your tile is conflict-free (or a broadcast) for its actual
access pattern; the tile is in the physical layout its consumer (TMA on write,
MMA on read) expects; and you picked the largest swizzle atom the contiguous
dimension allows. You can state, for each element, which device / bank /
lane+register / TMEM-cell holds it.
