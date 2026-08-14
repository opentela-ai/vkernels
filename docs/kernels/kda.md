# kda — Kimi Delta Attention (gated delta-rule layer, gfx942 / MI300A)

The delta-rule hybrid-attention layer of Kimi-K3 (issue #21). KDA is a
**gated delta-rule linear attention**: a per-head state matrix `S_t`
(`head_dim × head_dim`) is updated each token by a *delta correction*
`β_t (v_t − S_{t−1} k_t) k_tᵀ` and decayed by a forget gate `g_t`, and the
output is `o_t = S_t q_t`. On gfx942 today the AITER / Triton chunked
kernels GPU-fault (job 586165), so K3 serving sets `K3_DISABLE_KDA=1`.
vkernels re-implements the seven faulting kernels as portable host
references + HIP kernels so the KDA path is correct on MI300A **without**
that flag.

- **Source (CPU)**: `src/c/vkernels/kernels/kda.cpp`
- **Source (HIP)**: `src/c/vkernels/kernels/kda.hip`
- **Header**: `src/c/vkernels/kernels/kda.hpp`
- **Tests**: `tests/kernels/attn/test_kda.cpp` (host, 12 cases incl. a
  chunked-vs-naive cross-check at K3 head shapes)

---

## Per-token oracle

The math every kernel parallelises (verified by hand, the host oracle):

```
S_0 = 0
for t = 0 .. S-1:
    a_t = S_{t-1} · k_t                       # prediction from prior state
    S_t = g_t · S_{t-1} + β_t (v_t - a_t) ⊗ k_t   # forget + delta update
    o_t = S_t · q_t
```

`k` is L2-normalised by the caller (delta-net convention); `g` (forget gate)
and `β` (delta gate) are scalar per token, broadcast across the head dim.

## Chunked algorithm

The naive recurrence is `O(S · D²)` per head — correct but too slow. The
chunked algorithm (Yang et al., *Parallelizing Linear Transformers with the
Delta Rule over Sequence Length*) splits the sequence into chunks of size
`C` and recovers three parallel pieces, joined by within-chunk gate
products `G_{a,b} = ∏_{l=a}^{b} g_l = exp(L_b − L_{a−1})` (`L_{−1}=0`):

1. **gate cumsum** (`kda_gate_chunk_cumsum`) — within-chunk inclusive
   `L_{c,t} = Σ_{l≤t} log g_{c,l}` and cross-chunk exclusive
   `I_c = Σ_{c'<c} Σ_l log g_{c',l}`.
2. **intra solve** (`kda_delta_rule_intra`) — per chunk, solve the
   lower-triangular system for the delta-corrected values
   `u_t = v_t − G_{0,t−1}(C_{c−1} k_t) − Σ_{j<t} G_{j+1,t−1} β_j (k_j·k_t) u_j`.
3. **inter propagation** (`kda_delta_rule_inter`) —
   `C_c = G_{0,C−1} C_{c−1} + Σ_t G_{t+1,C−1} β_t u_t k_tᵀ`.
4. **output combine** (`kda_gla_fwd_o`) —
   `o_t = G_{0,t}(C_{c−1} q_t) + Σ_{j≤t} G_{j+1,t} β_j (k_j·q_t) u_j`.

Because chunk `c`'s intra solve reads `C_{c−1}` (the state leaving chunk
`c−1`), the CPU reference interleaves intra and inter **chunk by chunk**
(carrying `C` serially). `kda_delta_rule_fwd` is the orchestrator; it is
cross-checked against `kda_naive_delta_rule_fwd_cpu` at K3 head shapes
(`B,H,S,D,chunk` up to `{1,1,64,8,16}`) and matches to within fp32
round-off.

## Two-implementation model

| Operation | CPU (`kda.cpp`) | HIP (`kda.hip`) |
|---|---|---|
| `layer_norm_gated_fwd` | `kda_layer_norm_gated_cpu` | `kda_layer_norm_gated` |
| `kda_gate_chunk_cumsum_vector_kernel` | `kda_gate_chunk_cumsum_cpu` | `kda_gate_chunk_cumsum` |
| `chunk_gated_delta_rule_fwd_kernel` | `kda_naive_delta_rule_fwd_cpu` (oracle) + `kda_delta_rule_fwd_cpu` (chunked) | `kda_delta_rule_fwd` (cooperative recurrence) |
| `chunk_kda_fwd_kernel_intra_sub_chunk` | `kda_delta_rule_intra_cpu` | (subsumed by the cooperative forward) |
| `chunk_kda_fwd_kernel_inter_solve_fused` | `kda_delta_rule_inter_cpu` | (subsumed by the cooperative forward) |
| `chunk_gla_fwd_kernel_o` | `kda_gla_fwd_o_cpu` | (subsumed by the cooperative forward) |
| `pack_bitmatrix` | `kda_pack_bitmatrix_cpu` | `kda_pack_bitmatrix` |

The HIP forward is a **cooperative per-token recurrence** (one block per
`(b,h)`, `D×D` state in LDS, all threads cooperate on the matrix-vector
products and the rank-1 outer-product update, three barrier-separated
phases per token). It runs the per-token recurrence in the same order as
the oracle, so the only divergence is fp round-off — a correctness-first
baseline. The chunked intra/inter/output passes (which parallelise across
chunks via fla's `C_{c−1}`-decoupled solve) are a documented follow-on for
throughput; they are validated separately against the same oracle.

## Acceptance

A K3-shaped forward (MLA + KDA layers) runs on gfx942 and matches the
CPU/torch reference; `K3_DISABLE_KDA=1` is no longer required. The host
tests are the oracle on host CI (no GPU available); the device kernels are
validated on GPU machines against `kda_naive_delta_rule_fwd_cpu` /
`mla_fwd_cpu`.

## Benchmark (MI300A, gfx942)

`meta/benchmarks/bench_kda.hip` (+ `bench_kda.sh` driver). Roof: 1307
TFLOP/s bf16, 5300 GB/s HBM3, ridge ~247 FLOP/B.

### `kda_delta_rule_fwd` (DxD state in gmem, row-parallel)

| H | S | D | us(med) | TFLOP/s | GB/s | AI | bound |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 1 | 64 | 16 | 49 | 0.002 | 5.6 | 0.41 | mem |
| 1 | 64 | 32 | 96 | 0.005 | 11.2 | 0.42 | mem |
| 1 | 64 | 64 | 140 | 0.013 | 30.3 | 0.43 | mem |
| 16 | 64 | 64 | 143 | 0.205 | 477 | 0.43 | mem |
| 1 | 512 | 64 | 1023 | 0.014 | 33 | 0.43 | mem |
| 1 | 512 | 128 | 2640 | 0.022 | 51 | 0.43 | mem |

Every config is **memory-bound** (AI ~0.43 << ridge 247). The baseline
writes the full DxD state to HBM every token (~16 KB/token/head at D=128),
so the achievable GB/s is low because a single (b,h) block re-reads its
own state three times per token — the dominant cost and the target of the
LDS-resident-state and chunked intra/inter/output follow-ons.

### Supporting kernels

| kernel | shape | us(med) | GB/s |
|---|---|--:|--:|
| `kda_layer_norm_gated` | N=8192 D=128 | 174 | 72 |
| `kda_gate_chunk_cumsum` | B=1 H=16 nc=8 cs=64 | 13 | 5 |

`layer_norm_gated` at N=8192 launches only 32 blocks (one per 256 rows) on
304 CUs — a clear occupancy target (raise the block count or vectorise the
row reduction).
