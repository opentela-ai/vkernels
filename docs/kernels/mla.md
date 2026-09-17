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
- **Tests**: `tests/kernels/attn/test_mla.cpp` (host, 8 cases incl. the
  K3-shaped cross-check and the split-K recommendation helper) and
  `meta/benchmarks/test_mla_correct.hip` (on-device, 9 cases incl. the
  split-K decode paths)

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
| 1 | 64 | 8192 | 5284 | 0.216 | 108 | 2.0 | mem |
| 8 | 64 | 64 | 217 | 0.328 | 174 | 1.9 | mem |
| 16 | 512 | 512 | 3381 | 2.700 | 1360 | 2.0 | mem |
| 128 | 512 | 512 | 15898 | 4.593 | 2314 | 2.0 | mem |
| 1 | 8192 | 8192 | 59741 | 2.444 | 1223 | 2.0 | mem |

All configs are **memory-bound** (AI ≤ 2.0 << ridge 247). The wide prefill
(H=128, S_q=S_kv=512) reaches **2314 GB/s — 44% of the HBM roof**, still the
family best. The worst case was **decode** (H=1, S_q=1, S_kv=8192): one
block re-reads 8192 keys for a single query → only **6.5 GB/s (0.12% of
roof)**. Documented tuning targets:

* **Decode** — split-K: multiple blocks per query row with a partial-softmax
  reduction, so the 8192 keys are spread across CUs. **DONE (issue #82,
  job 640246, 2026-09-17): 5507 → 66.8 µs median (63.3 µs min) = 82×,
  ~549 GB/s effective with the partial-buffer traffic (~10% of the HBM
  roof)** — near the 32-keys-per-split serial-chain floor; prefill rows
  are within ~2% node noise of the pre-split table above.
* **Prefill** — persistent kernel / tiled-key loop that reuses a KV tile
  across query tiles; vectorised `float4` loads.

