# Tensor Memory and Register Fragments

## Tensor Memory (TMEM) — Blackwell

TMEM is an on-chip storage space introduced with Blackwell. Pre-Blackwell MMA
accumulators lived in registers; as MMA tiles grew, those accumulators ate a
large share of the register file. Blackwell's `tcgen05` writes accumulators to
**TMEM instead**, cutting register pressure.

- Logically belongs to the **CTA**; physically resides on the SM.
- **128 lane rows × up to 512 32-bit columns.** A position needs two
  coordinates, so it uses named axes:
  ```
  S[(128, 256) : (1@TLane, 1@TCol)]
  (row, col) = unflatten(x; 128, 256)
  f_D(x) = row@TLane + col@TCol
  ```
  `f_D(x)` returns both a `TLane` and a `TCol`, not one integer.
- **Explicitly managed.** A kernel allocates and frees TMEM, and the epilogue
  must explicitly read the MMA accumulator back into registers. To read a full
  128-lane accumulator, the four warps of a warpgroup each load their own
  32-lane TMEM window.

## Register fragments

Tensor-Core operands are distributed across a warp's 32 lanes, so a lane id
alone is not enough to identify an element — you also need the per-lane slot.
Use `@laneid` (lane within the warp) and `@reg` (a lane-local slot).

For an `m8n8`-style fragment (an `8×8` tile, 64 elements, two slots per lane):

```
laneid = row·4 + col//2
reg    = col%2
S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]
(c0, c1, c2) = unflatten(x; 8, 4, 2) = (row, col//2, col%2)
f_D(x) = (c0·4 + c1)@laneid + c2@reg
```

Click-style example: logical element `(5, 3)` is owned by **lane 21** at
**slot 1**. A specific instruction may still pack multiple low-precision
elements into one 32-bit hardware register.

## Replication and offset

`f_D(x)` returns one location, but hardware sometimes holds several copies.
Append a replication or offset term:

- **`R[n : s@axis]`** — `n` independent copies at stride `s` along `axis`. No
  new logical data; it records the physical locations of the copies. E.g.
  Blackwell `.warpx4` replicates one 32-lane TMEM base tile into the four
  partitions:
  ```
  S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]   // TLane += {0,32,64,96}
  ```
- **`O[s@axis]`** — a fixed *translation* by `s` along `axis` (one copy, not
  replicated). Used to shard an element to one device, or for fixed
  lane/column offsets.

## Block-scaled scale factors in TMEM

Block-scaled MMA (MXFP8, NVFP4) is not a dtype — it is a family of low-precision
MMAs that use per-block scale factors. MXFP8 shares one E8M0 scale per 32 K
elements; NVFP4 shares one E4M3 scale per 16 E2M1 elements. Before the MMA:

```
A_real[m, k] = A_low[m, k] · SFA[m, k // K_blk]
B_real[k, n] = B_low[k, n] · SFB[n, k // K_blk]
D = C + A_real · B_real
```

Packing a `128×4` SFA (each scale = 1 byte → 512 bytes total) into one
`tcgen05.cp.32x128b.warpx4` base tile (32 lanes × 16 bytes = 512 bytes):

```
local_lane = m % 32     // 32 lanes own the 128 rows
Mgroup     = m // 32    // → TCol 0..3
byte       = sfk        // → byte 0..3 within the 32-bit cell
byte_offset = TCol·4 + byte
```

`.warpx4` then multicasts that packed base tile into all four TMEM partitions
(`TLane = local_lane + 32·p`, `p = 0..3`). PTX therefore requires **both SFA
and SFB duplicated across all four partitions.** SFB follows the same rule
with B's column index `n` replacing A's row index `m`.

## Multi-GPU sharding

The same replication/offset notation describes a multi-GPU layout. A `2×2` GPU
mesh identifies devices by `(@gpuid_x, @gpuid_y)`. Start with a base sharded
along `@gpuid_y`:

```
base = S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)]
```

Element `(1, 2, 3)` → device `(_, 1)`, local offset `19`. Then:

- `base + R[2 : 1@gpuid_x]` → replicated on devices `{(0,1),(1,1)}` (two copies).
- `base + O[1@gpuid_x]` → one copy translated to device `(1,1)`.

`R` makes copies; `O` translates. The distinction is what makes a layout
express "this element lives in several places" versus "this element lives
here, shifted."
