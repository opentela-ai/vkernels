# Swizzling and Bank Conflicts

Shared memory is divided into **32 banks**. For the 4-byte granularity used in
this chapter, an element's bank is:

```
bank = (addr // 4) % 32
```

A warp instruction may be split into several **processing batches**, which
Nsight Compute calls *wavefronts*. A contiguous, aligned access moves at most
**128 bytes per wavefront** — 4 bytes from each of the 32 banks. The access
width therefore sets the wavefront grouping:

| Access / lane | Lanes per wavefront |
|---|---|
| 4 bytes | 32 |
| 8 bytes | 16 |
| 16 bytes | 8 |

Bank conflicts are scored **only within a wavefront**. Lanes in different
wavefronts never conflict, even on the same bank. Lanes reading the **same**
address get a free broadcast — no conflict.

## The conflict a plain layout causes

A row-major tile coalesces row reads (adjacent elements hit adjacent banks),
but a column read separates elements by the row stride. If that stride equals
the bank period, several lanes land on one bank and serialize.

The `8×8` example below reduces 32 banks to 8 for clarity. With plain
row-major, `bank = logical_col`, so reading column 3 sends all eight accesses
to bank 3 — an **8-way conflict** (eight cycles instead of one).

## The XOR swizzle

A swizzle rearranges physical addresses while preserving the tile's logical
shape. The common technique XORs part of the row index into the column index:

```
mapped_col = logical_col XOR (row % atom_rows)
bank = mapped_col
```

For an 8-row atom, `col = 0` now maps rows 0–7 to banks 0–7 — all eight reads
go in parallel. The same tile now serves both row-wise and column-wise access
without conflicts.

## Atoms and modes

A swizzle **atom** is the smallest repeating block of the address permutation,
with shape `8 rows × W bytes`:

| Mode | Atom (`8 × W`) | When to use |
|---|---|---|
| `SWIZZLE_128B` | `8 × 128 B` | contiguous dim ≥ 128 B (≥64 fp16) |
| `SWIZZLE_64B` | `8 × 64 B` | contiguous dim ≥ 64 B |
| `SWIZZLE_32B` | `8 × 32 B` | contiguous dim ≥ 32 B |

Choose the **largest** row width the tile's contiguous dimension supports and
is preferably divisible by. For a row ≥128 bytes of fp16, `SWIZZLE_128B` makes
both row reads and 8-row column reads conflict-free — but only for that
element width, alignment, and access pattern. Changing any of them can
reintroduce conflicts.

## Traps

1. **The swizzle permutes only within an atom.** The innermost contiguous
   dimension of a TMA box must not exceed the swizzle width (128 B holds 64
   fp16). If the data is narrower than the swizzle width, the SMEM allocation
   must still reserve the full width.
2. **Row stride can defeat a group structure.** A 256-byte row stride puts
   two 128-byte spans per row; even though the swizzle permutes within each
   128-byte span, reading one column across eight rows reaches only four
   distinct banks → a 2-way conflict. The fix is to reshape so the two
   128-byte spans are **explicit groups** (`g0`, `g1`): adjacent rows within a
   group are 128 bytes apart, so
   `bank_sector = local_col XOR (row % 8)` reaches eight distinct banks.
3. **Everyone must agree.** The TMA descriptor that writes the tile, the SMEM
   layout, and the MMA that reads it must all use the *same* swizzle mode. TMA
   can apply the swizzle on write; if the consumer reads it as linear, the
   bytes are present but interpreted as the wrong matrix elements.

## How it composes with the layout

You do not compute swizzled addresses by hand. The full mapping is two steps:

1. `S[...]` maps a logical element to a **linear `@m` address** (affine).
2. The swizzle (a **non-affine XOR**) transforms that address into the final
   SMEM location.

Because XOR is not affine, the swizzle is a *separate* address transformation
composed with the layout, not part of the `S[...]` layout itself. Different
hardware units impose different swizzle requirements, and those requirements
also change across GPU generations — check the descriptor each consumer expects.
