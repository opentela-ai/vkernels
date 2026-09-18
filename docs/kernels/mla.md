# mla — Multi-head Latent Attention forward (gfx942 / MI300A)

The latent-attention path of Kimi-K3 (issue #21). MLA compresses the KV
cache into a low-rank latent (`k_c`/`v_c`, both `kv_lora_rank`) and a
decoupled RoPE part (`k_pe`, `qk_rope_head_dim`), then attends with a
per-head query that carries its own rope slice. This matches vLLM's
TRITON_MLA / AITER semantics so a K3 forward served on gfx942 no longer
needs a vendor-specific fallback.

- **Source (CPU)**: `src/c/vkernels/kernels/mla.cpp`
- **Source (HIP)**: `src/c/vkernels/kernels/mla.hip`
- **Header**: `src/c/vkernels/kernels/mla.hpp`
- **Tests**: `tests/kernels/attn/test_mla.cpp` (host, 9 cases incl. the
  K3-shaped cross-check, the fused-head sizing helper and the split-K
  recommendation helper) and
  `meta/benchmarks/test_mla_correct.hip` (on-device, 16 cases incl. the
  fused-head prefill and the split-K decode paths)

---

## Computation

For each query row `i` (global index `q_start + i`) and key `j`
(global index `kv_start + j`), with a **causal** mask `kv_start + j <= q_start + i`:

```
score[i][j] = scale · ( q_nope[i] · k_c[j]  +  q_rope[i] · k_pe[j] )

out[i]      = Σ_j  softmax_causal(score)[i][j] · v_c[j]
```

where `q[i] = [q_nope[i] | q_rope[i]]` is `Dq = kv_lora_rank + qk_rope_head_dim`
wide, `k_c[j]` and `v_c[j]` are `kv_lora_rank`, and `k_pe[j]` is
`qk_rope_head_dim`. The output is `kv_lora_rank`-wide (the no-ropes
projection of the value latent).

| Tensor | Shape | Meaning |
|---|---|---|
| `q` | B × H × S_q × Dq | per-head query (no-ropes ‖ rope) |
| `k_c` | B × S_kv × kv_lora_rank | compressed key latent (shared across heads) |
| `k_pe` | B × S_kv × qk_rope_head_dim | decoupled RoPE key (shared across heads) |
| `v_c` | B × S_kv × kv_lora_rank | compressed value latent (shared across heads) |
| `out` | B × H × S_q × kv_lora_rank | attention output |

`q_start`/`kv_start` are global sequence offsets; causality is
`kv_start + j <= q_start + i`, so chunked prefill (where `q_start` ≠ 0) is
masked correctly.

---

## CPU reference (`mla.cpp`)

A two-pass softmax (fp32) with the standard numerically-stable max-shift.
Each (b, h, i) is computed independently against the full visible key range,
which is the unambiguous oracle for the online-softmax kernel.

## HIP kernel (`mla.hip`)

One block owns `BQ` query rows for one (b, h) and streams the shared K/V
latent **one key at a time**, accumulating a weighted output in fp32 with an
**online softmax** (a running row-max and row-sum, lock-step across the
warp). The head dimension is laid out **strided across the 64 lanes** (lane
`d` owns elements `{d, d+64, d+128, ...}`); at the K3 config
(`kv_lora_rank=512`, `qk_rope_head_dim=64` → `Dq=576`) that is 9 elements
per lane and 8 output accumulators per lane — entirely in registers, exact
(no bounds-checks fire). The per-key score is a warp-shuffle reduction of
the lane-local partial dot, broadcast to every lane, which keeps the
online state identical across the warp with no LDS synchronisation on the
hot path.

The baseline streams one key per iteration (no double-buffered key tile);
a tiled key prefetch is a follow-on optimisation, not required for
correctness against the oracle.

### Fused-head prefill (issue #151)

The plain prefill grid (`(S_q/BQ)·H·B` blocks) re-reads the **head-shared**
K/V latent once per (query tile, head) — ~36 GB vs ~290 MB ideal at the
H=128 S_q=512 K3 shape — and its one-key-at-a-time loop is latency-bound
(~1300 cycles/key measured, because every key serialises two strided global
row loads through the score reduction).

The fused-head kernel batches `BH` heads into one block alongside its `BQ`
query rows: a block owns `BQ·BH` (row, head) pairs — one wavefront each —
all streaming the SAME key sequence. Since `k_c`/`k_pe`/`v_c` are identical
for every head, the block-wide key reads are absorbed by L1 and the latent
is read from DRAM once per (query tile, head group) instead of once per
(query tile, head) — a `BH`-fold traffic cut — while each wavefront keeps
the plain kernel's inner loop (which allocates cleanly: a BN-tiled LDS
variant of this kernel measured 57 spilled SGPRs on gfx942 and ran ~50×
slower per key; the serial-head-loop variant spilled the same way).

Wide ranks (`kv_lora_rank = 512`, `qk_rope_head_dim <= 64` — the K3 recipe)
run a pipelined variant (`mla_fwd_headfused_wide_kernel`): exact lane
slices (8 per lane, no bounds guards), consecutive-lane `float4` loads
(2 per row instead of 8 scalars), a BN=2-key register pipeline that issues
a tile's loads up front so the score→expf→accumulate chain runs ALU-only,
and an online-softmax fast path that skips the accumulator rescale when a
key does not advance the running max (one `expf` per key). BH is capped so
the block stays within the CU's 256 KB register file (12 wavefronts).

Chunked / short prefill (`S_q > 8` grids that still leave CUs idle) split
their `S_kv` window through the #82 split+combine kernels (BQ=4 split
variant), same as decode.

### Split-K decode (issue #82)

A decode grid of `B·H·S_q` blocks (one 64-thread block per row) leaves CUs
idle when `B·H·S_q << kMlaCus` (228, MI300A) — the H=1 S_q=1 S_kv=8192 case
ran on ONE block: 5.5 ms = 6.5 GB/s. `mla_fwd_split_for(B, H, S_q, S_kv)`
recommends a split count (occupancy rule, as in `dsa_topk_logits_split_for`:
fill the CUs the plain grid leaves idle, capped so every split keeps ≥
`kMlaMinSplitKeys` = 32 keys), and `mla_fwd` routes such decode shapes
through the split path:

* each split block streams its own key slice and writes the per-split
  **normalized partial-softmax output** (softmax restricted to its visible
  keys) + a per-split **lse** (`row_max + log(row_sum)`);
* a combine kernel folds the splits in FIXED ascending order:
  `out = Σ_sp exp(lse_sp − lse_g) · partial_out_sp` with
  `lse_g = m_g + log(Σ exp(lse_sp − m_g))` — the same partial+combine scheme
  proven on `dsa_sparse_fwd_split` (4.5–20×) and `dsa_topk_logits` (4.7×);
  with normalized partials the combine is exact (no second normalisation),
  all-masked rows still produce zeros (oracle parity), and the fixed order
  keeps the result deterministic;
* prefill shapes (`S_q > 8`), grids that already fill the CUs, and a failed
  workspace allocation all fall back to the plain single-block path —
  `mla_fwd_with_tile` keeps the exact pre-split semantics for the autotuner.

The partial buffers live in a grow-only, mutex-guarded device workspace
inside `mla.hip` (allocations happen only on first use / shape growth, so
CUDA-graph capture after a warmup needs no allocation inside capture).

---

## Config selection

`mla_config_for(S_q, kv_lora_rank, qk_rope_head_dim, &bq, &bn, &th)`
picks a query-tile size (`BQ` ∈ {1,2,4,8}) for the serving recipe;
`mla_fwd_with_tile` exposes the explicit tile for the offline autotuner.
`mla_fwd_split_for(B, H, S_q, S_kv)` adds the split-K decode decision
(issue #82): 1 = plain path, >1 = split the `S_kv` window across blocks.

## Benchmark (MI300A, gfx942)

`meta/benchmarks/bench_mla.hip`. Roof: 1307 TFLOP/s bf16, 5300 GB/s HBM3,
ridge ~247 FLOP/B. K3 config: `kv_lora_rank=512`, `qk_rope_head_dim=64`
→ `Dq=576`, `scale=1/√576`.

| H | S_q | S_kv | us(med) | TFLOP/s | GB/s | AI | bound |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 1 | 1 | 8192 | 66.8 | 0.267 | 533 | 0.5 | mem (split-K, split=228) |
| 1 | 1 | 8192 | 5507 | 0.003 | 6.5 | 0.5 | mem (plain baseline, pre-split) |
| 1 | 64 | 8192 | 5284 → 413 | 0.216 → 2.76 | 108 → 1380 | 2.0 | mem (split path, issue #151) |
| 8 | 64 | 64 | 217 → 118 | 0.328 → 0.60 | 174 → 320 | 1.9 | mem (fused heads, issue #151) |
| 16 | 512 | 512 | 3381 → 1146 | 2.700 → 7.96 | 1360 → 2022 | 2.0 | mem (fused heads, issue #151) |
| 128 | 512 | 512 | 15898 → 8322 | 4.593 → 8.77 | 2314 → 2228 | 2.0 | mem (fused heads, issue #151) |
| 1 | 8192 | 8192 | 59741 → 39234 | 2.444 → 3.72 | 1223 → 1862 | 2.0 | mem (fused heads, issue #151) |

The prefill numbers are `before → after` issue #151 (job 641391,
2026-09-18). Note the GB/s columns are NOT comparable across the arrow for
prefill rows: the fused-head kernel reads the shared latent once per
(query tile, head group) instead of once per (query tile, head), so it
counts ~2× fewer redundant bytes at H=128 and H=16 while running 1.9–3.0×
faster in wall time. All configs remain memory-bound (AI << ridge 247).
The worst case was **decode** (H=1, S_q=1, S_kv=8192): one block re-reads
8192 keys for a single query → only **6.5 GB/s (0.12% of roof)**.
Documented tuning targets:

* **Decode** — split-K: multiple blocks per query row with a partial-softmax
  reduction, so the 8192 keys are spread across CUs. **DONE (issue #82,
  job 640246, 2026-09-17): 5507 → 66.8 µs median (63.3 µs min) = 82×,
  ~549 GB/s effective with the partial-buffer traffic (~10% of the HBM
  roof)** — near the 32-keys-per-split serial-chain floor; prefill rows
  are within ~2% node noise of the pre-split table above.
* **Prefill** — cross-head KV reuse + a pipelined per-key loop. **DONE
  (issue #151, job 641391, 2026-09-18): fused-head blocks (BQ·BH (row,
  head) wavefront pairs sharing one key stream) + wide-rank BN=2 register
  pipeline with float4 row loads: chunked prefill H=1 S_q=64 5284 → 413 µs
  (12.9×, via the #82 split path), H=16 S_q=512 3381 → 1146 µs (3.0×),
  H=128 S_q=512 15898 → 8322 µs (1.9×, at half the counted KV bytes),
  H=1 S_q=8192 59741 → 39234 µs (1.5×)** — all at ≤ 42% of the HBM roof
  on counted bytes; the residual gap is the L2-latency serial chain, a
  candidate for a deeper (BN=4+) pipeline or persistent-CTA KV reuse.
  Register-allocation note for future tiling work: an LDS-staged BN-tile
  variant of this kernel spills ~57 SGPRs on gfx942 and runs ~50× slower
  per key — keep the streaming loads and pipeline in registers instead.
  Decode rows are within ~2% node noise of the pre-split table above.

