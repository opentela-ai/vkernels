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
- **Tests**: `tests/kernels/attn/test_mla.cpp` (host, 6 cases incl. a
  K3-shaped cross-check against an independent reference)

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

---

## Config selection

`mla_config_for(S_q, kv_lora_rank, qk_rope_head_dim, &bq, &bn, &th)`
picks a query-tile size (`BQ` ∈ {1,2,4,8}) for the serving recipe;
`mla_fwd_with_tile` exposes the explicit tile for the offline autotuner.

## Benchmark (MI300A, gfx942)

`meta/benchmarks/bench_mla.hip`. Roof: 1307 TFLOP/s bf16, 5300 GB/s HBM3,
ridge ~247 FLOP/B. K3 config: `kv_lora_rank=512`, `qk_rope_head_dim=64`
→ `Dq=576`, `scale=1/√576`.

| H | S_q | S_kv | us(med) | TFLOP/s | GB/s | AI | bound |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 1 | 1 | 8192 | 5510 | 0.003 | 6.5 | 0.5 | mem |
| 1 | 64 | 8192 | 5315 | 0.215 | 107 | 2.0 | mem |
| 8 | 64 | 64 | 216 | 0.330 | 175 | 1.9 | mem |
| 16 | 512 | 512 | 3387 | 2.695 | 1358 | 2.0 | mem |
| 128 | 512 | 512 | 15875 | 4.599 | 2318 | 2.0 | mem |
| 1 | 8192 | 8192 | 58727 | 2.487 | 1244 | 2.0 | mem |

All configs are **memory-bound** (AI ≤ 2.0 << ridge 247). The kernel
re-reads all KV per query tile (no cross-tile reuse), so for the wide
prefill (H=128, S_q=S_kv=512) it reaches **2318 GB/s — 44% of the HBM
roof**, the best case. The worst case is **decode** (H=1, S_q=1,
S_kv=8192): one block re-reads 8192 keys for a single query → only
**6.5 GB/s (0.12% of roof)**. Documented tuning targets:

* **Decode** — split-K: multiple blocks per query row with a partial-softmax
  reduction, so the 8192 keys are spread across CUs.
* **Prefill** — persistent kernel / tiled-key loop that reuses a KV tile
  across query tiles; vectorised `float4` loads.

