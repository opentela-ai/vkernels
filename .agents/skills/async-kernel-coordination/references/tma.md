# TMA: Asynchronous Tile Movement

TMA (Tensor Memory Accelerator) asynchronously moves a tile between global and
shared memory. **One thread** of a warp issues the copy and supplies two
things; the rest of the warp is masked off until the request is submitted. The
engine performs the remaining address calculation and transfer, then reports
completion through a barrier — the issuing warp and the other warps in the CTA
can continue with other work in the meantime.

## The two inputs to a copy

1. **Tensor-map descriptor** — the global tensor's element type, per-dimension
   shape and strides, the tile shape for one copy, and the swizzle mode to
   apply when writing SMEM. One descriptor is usually reused across many
   copies. ("How is this tensor organized?")
2. **Per-copy arguments** — the tile's starting coordinates in the global
   tensor and its destination SMEM address. ("Where does this copy begin, and
   where should it land?")

## Swizzle on write

TMA can apply the swizzle while writing SMEM, so the tile arrives in the
physical layout the later MMA expects — the issuing thread does not compute
each swizzled address. For a `16×128` fp16 slice with `SWIZZLE_128B`, each row
is eight 16-byte sectors; logical sector `col` in row `row` is written to:

```
physical_sector = col XOR (row % 8)
```

The descriptor, the SMEM tile layout, and the consumer MMA must all describe
the **same** physical arrangement. If TMA writes a 128-byte swizzle but the
MMA reads the data as linear, the bytes reached SMEM but the Tensor Core
interprets them as the wrong matrix elements.

## 3-D TMA for multiple swizzle atoms

A `SWIZZLE_128B` atom is `8 rows × 128 B`. Swizzle permutes **only within an
atom**, so the innermost contiguous dimension of a TMA box cannot exceed
128 bytes (64 fp16). A `16×128` fp16 slice has a 256-byte row, so split each
row into two groups:

```
group = j // 64
col   = j % 64
global[row, j] = global3[group, row, col]      # (group=2, row=16, col=64)
```

This reshape only changes how the tensor map interprets coordinates; it does
not move data in global memory. Each group is now ≤128 bytes and contains two
atoms (rows 0–7, 8–15); the whole slice is four atoms, swizzled within each.

## The row-stride trap

A 256-byte row stride (two 128-byte spans per row) can cause a **2-way
conflict** even though the swizzle permutes within each 128-byte span:

```
# plain 256B stride:  bank_sector = local_col XOR ((2·row + span) % 8)  → 4 distinct sectors
# explicit 128B groups: bank_sector = local_col XOR (row % 8)            → 8 distinct sectors
```

Reshape so adjacent rows within a group are 128 bytes apart, not 256. The
distinction is how the spans are arranged in memory, not whether the swizzle
itself groups sectors in sets of eight.

When the tile's contiguous dimension is narrower than the swizzle width, the
SMEM allocation must still reserve the full width; choose among 128/64/32-byte
modes by both the tile width and the access pattern.

## Waiting for a load

A TMA load is asynchronous: issuing it only starts the transfer. The consumer
waits on the **mbarrier** that tracked the bytes. One phase tracks both an
arrival count and a pending **tx-count**; the phase completes only when both
reach zero.

Example: load two 2048-byte operand tiles A and B on one barrier initialized
with expected arrival count 1. The issuing thread associates both loads with
the barrier and executes:

```
mbarrier.arrive.expect_tx(4096)        # 1 arrival + 4096 pending bytes
# engine applies complete-tx updates as each transfer finishes
consumer: try_wait(phase) until arrival count == 0 AND tx-count == 0
```

Only then may the consumer read A and B. Detail on the barrier mechanics is in
[mbarrier.md](mbarrier.md).

## Waiting for a store

A TMA store (SMEM → GMEM) asks the opposite question: the *producer* needs to
know when the **source** buffer is safe to reuse, not when the destination is
readable. Stores use a bulk async group:

```
issue one or more TMA stores
cp.async.bulk.commit_group        # bundle uncommitted stores
cp.async.bulk.wait_group 0        # until all committed groups are done
# now Dsmem may be overwritten
```

So: **a load's consumer waits for data through an mbarrier with byte tracking;
a store's producer waits for source reuse through a commit group and wait
group.**

## Putting TMA into a pipeline

TMA's bigger benefit than fewer copy instructions is **overlap**. With two
SMEM stages:

```
time t:   MMA reads stage 0 | TMA fills stage 1
time t+1: MMA reads stage 1 | TMA fills stage 0
```

Before MMA reads a stage it waits the matching TMA load; before TMA overwrites
a stage the kernel confirms the previous computation no longer uses it. TMA
performs the transfer, the barrier hands each stage from producer to consumer,
and the time spent waiting for future data is hidden behind computation on the
current tile.
