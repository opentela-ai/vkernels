# FlashAttention (the GEMM ladder applied to attention)

Attention is `O = softmax(QKᵀ / √d) · V`. A direct implementation materialises
the full score matrix `S = QKᵀ` to HBM and reads it back — that round trip
sinks the arithmetic intensity as the sequence length grows. FlashAttention
processes K and V in blocks and keeps only the current tiles plus per-row
softmax state on chip, never writing the full score matrix.

Versions differ mainly in how the algorithm maps to hardware. FA-2 improved
work partitioning across CTAs/warps; FA-3 used TMA + WGMMA + warp
specialisation on Hopper; **FA-4 targets Blackwell** and reorganises the
pipeline around `tcgen05` and TMEM, so it is the canonical application of the
ladder in the GEMM skill.

## Algorithm structure

For query row `i`, `s_ij = q_i · k_j` is row `i` of `S = QKᵀ`. Let `mᵢ` be the
exact largest score in the row. **Online softmax** keeps, per row, an exponent
reference `rᵢ`, a running denominator `ℓᵢ`, and a running weighted sum `oᵢ`,
all accumulated relative to the current `rᵢ`. Once a K/V block's scores are
consumed they are discarded.

Basic online softmax updates `rᵢ` to the running maximum whenever it sees a
larger score. **FA-4 first checks the gap** `δ = new_ref − old_ref` against a
threshold `τ = log₂(256) = 8`:

- `δ ≥ −τ` → retain the old reference; `acc_scale = 1`; accumulate `block_O`
  directly (no rescale of the existing `O`).
- `δ < −τ` → adopt the candidate reference, convert old `ℓᵢ` and `O` by
  `acc_scale` to the new reference, then accumulate.

This trades fewer rescaling operations against bounded exponent growth (the
largest unnormalised weight is capped at `2⁸ = 256` while the old reference
holds). Note `row_max` need **not** equal the exact `mᵢ` at every iteration
while the threshold permits it. The actual kernel applies the rescale test
separately to the 32 rows each warp in WG2 owns; only after every K/V block
does it compute the final `O / ℓ`.

## Tile primitive data flow

`S`, `P`, and `O` live in **TMEM**, handed between roles:

```
Q, K : GMEM --TMA load--> SMEM --QKᵀ MMA--> S in TMEM
S    : TMEM --tcgen05.ld--> registers --softmax--> P in registers
P    : registers --TMEM store--> P in TMEM
P, V : P in TMEM + V in SMEM --PV MMA--> O in TMEM
when the exponent reference changes: O --tcgen05.ld--> registers --rescale/TMEM store--> O
at the end: O --tcgen05.ld--> registers --normalize/cast--> O in SMEM --TMA store--> O in GMEM
```

So three TMEM tiles (`S`, `P`, `O`) connect two MMAs (QKᵀ, PV) with softmax
between them — the same `load → MMA → epilogue/store` structure as the GEMM
ladder, with an extra MMA and a softmax in the middle.

## Warp roles and scope

- **Producer** warps load Q, K, V via TMA (and load `V` once per K/V block).
- **WG2** runs the **QKᵀ MMA** (writes `S` to TMEM) and the **PV MMA**
  (accumulates `O` in TMEM from `P` in TMEM and `V` in SMEM), with softmax
  (CUDA cores) turning `S` into `P` between them.
- **Writeback** reads the final `O` from TMEM into registers, normalises
  (`O / ℓ`) and casts to the output dtype, stages through SMEM, and issues a
  TMA store.

Each `S → P → O` handoff is gated by the matching mbarrier phase, with the
**tcgen05 fence** before any TMEM read.

## Key barrier protocols

- **QK MMA → softmax:** `tcgen05.commit` on an mbarrier; softmax waits that
  phase, then `tcgen05.ld` of `S`.
- **Softmax → PV MMA:** `P` written back to TMEM; PV MMA waits its own barrier
  phase before reading `P` and `V`.
- **Reference change:** `O` is read from TMEM into registers, rescaled by
  `acc_scale`, written back to TMEM — guarded by its own wait/fence — *before*
  the next PV MMA accumulates into it.

## Exponential evaluation is split

FA-4 divides `exp2` between two execution paths so they run concurrently and
the kernel does not bottleneck on a single unit: some elements use hardware
`exp2`, others a cubic polynomial evaluated with FP32 FMA (`ex2_emulation_2`).
This changes *how* the exponential is evaluated, not the online-softmax
recurrence above.

## Causal masking and GQA

- **Causal masking:** mask future positions within the `S` tile before softmax
  (predicated or masked MMA), so no query attends to keys that follow it.
- **GQA:** Q has more heads than K/V. Tile so each Q-head group shares the
  same K/V, loading K/V **once per group** and replicating Q heads across the
  shared keys/values.

## Why it is the ladder

- **Fusion** (no score round-trip) raises arithmetic intensity — the algorithm
  change that, per the roofline skill, is the only thing that helps once the
  memory roof is hit.
- **TMA + pipelining + warp specialisation** (Steps 4–9) cut the idle time
  between the two MMAs and the softmax, keeping the Tensor Cores busy.
- **TMEM** holds `S`, `P`, and `O`, so the large score/output accumulators do
  not eat the register file.
