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
  chunked-vs-naive cross-check at K3 head shapes),
  `tests/kernels/attn/test_kda_k3_chunked.cpp` (the per-key-dim chunked WY
  derivation), `tests/kernels/attn/test_kda_cp.cpp` (the state-handoff /
  context-parallel composition contract)

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

## Context-parallel state handoff (chunked-scan state API, #CP)

Because the delta-rule recurrence is Markov in `S_t`, KDA layers compose
across sequence shards: running shard `[S0,S1)` with initial state `S_in`
and shard `[S1,S2)` seeded with the exported `S_out` equals the monolithic
`[S0,S2)` run. That is exactly what a context-parallel (CP) rank ring needs:
each rank processes its sequence shard and hands the `D×D` per-head state
to the next rank (the attention KV still crosses the wire per layer; the
linear-attention state is the O(1) summary that makes the KDA path
ring-composable).

Two CPU entry points (state layout: the canonical `[B, H, D, D]` float of
the HIP state scratch, row-major `S[v][k]`):

- `kda_naive_delta_rule_fwd_state_cpu` — the per-token oracle with an
  explicit `S_0 = state_in` and exported `S_S = state_out` (the state-carrying
  counterpart of `kda_naive_delta_rule_fwd_cpu`, which hard-codes `S_0 = 0`).
- `kda_delta_rule_fwd_state_cpu` — the **chunked-scan state API**: the affine
  WY form (the math of `hip::kda_delta_rule_fwd_chunked_with_scratch`) with
  `C_{-1} = state_in` and the final `C` exported to `state_out`. The caller
  hands a whole shard; chunking is internal (`chunk_size` divides the shard
  length). `state_out` may alias `state_in` (one ring buffer per `(b,h)`;
  the seed is fully read before the export).

Contract: same input regime as the HIP chunked kernel (`k` L2-normalised,
`g` in (0,1], `β ≤ 1`). `tests/kernels/attn/test_kda_cp.cpp` checks: the
WY state forward vs the state-carrying oracle at zero and random seeded
states (incl. the K3 head shapes `D=64/128`, `cs=64`), the ring-handoff
composition (4 shards == monolithic, in outputs AND final state), in-place
state aliasing, and the null/chunk contracts.

**HIP status: no HIP change is required.**
`hip::kda_delta_rule_fwd_with_scratch` and
`hip::kda_delta_rule_fwd_chunked_with_scratch` already implement the
seed/export contract (the `state` buffer is in-out: pre-fill with the
gathered initial state, read the final state back after the call — the
multi-turn decode-with-scratch path). The new CPU functions are the host
oracle for exactly that contract, which until now was validated only
on GPU machines against the cooperative kernel. A CP rank ring therefore
needs only a transport for the `B·H·D·D` state buffer (see
[comm-context-parallel.md](../comm-context-parallel.md)); per rank shard
the GPU path is unchanged.

## Two-implementation model

| Operation | CPU (`kda.cpp`) | HIP (`kda.hip`) |
|---|---|---|
| `layer_norm_gated_fwd` | `kda_layer_norm_gated_cpu` | `kda_layer_norm_gated` |
| `kda_gate_chunk_cumsum_vector_kernel` | `kda_gate_chunk_cumsum_cpu` | `kda_gate_chunk_cumsum` |
| `chunk_gated_delta_rule_fwd_kernel` | `kda_naive_delta_rule_fwd_cpu` (oracle) + `kda_delta_rule_fwd_cpu` (chunked) + `kda_naive_delta_rule_fwd_state_cpu` (state-carrying oracle) + `kda_delta_rule_fwd_state_cpu` (chunked WY with explicit state) | `kda_delta_rule_fwd` (cooperative recurrence) |
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

### `kda_delta_rule_fwd` (LDS-resident state, row-parallel)

| H | S | D | us(med) | TFLOP/s | GB/s | AI | bound |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 1 | 64 | 16 | 49 | 0.002 | 5.6 | 0.41 | mem |
| 1 | 64 | 32 | 96 | 0.005 | 11.2 | 0.42 | mem |
| 1 | 64 | 64 | 140 | 0.013 | 30.3 | 0.43 | mem |
| 16 | 64 | 64 | 143 | 0.205 | 477 | 0.43 | mem |
| 1 | 512 | 64 | 1023 | 0.014 | 33 | 0.43 | mem |
| 1 | 512 | 128 | 2640 | 0.022 | 51 | 0.43 | mem |

Every config is tagged **memory-bound**, but that verdict is the roofline
MODEL's legacy gmem bytes: the model still counts the per-token DxD state
HBM round-trip that the committed kernel no longer performs. The kernel
caches this block's `Db` state rows in LDS (`srow[Db·D]` — 4 KB at
D=64/Db=16, 16 KB at D=128/Db=32, both far under the 64 KB static-smem
cap), runs all four per-token phases on-chip, and touches HBM state only
once per 256-token chunk (the between-chunk handoff; the decode-with-
scratch multi-turn path keeps the same once-per-call residency). Actual
state traffic is ~250x lower than the model; the residual cost of each
(b,h) D-row block is the serial 64-token × 4-barrier recurrence, whose
intra-block headroom is exhaustively disproven (4→3 barriers = cross-
thread GATE-vs-OUTPUT race; parallel dots regress low-H — see
[kda-lds-optimization](kda-lds-optimization.md)).

### `kda_delta_rule_fwd_chunkedWY` — chunked-vs-baseline at the bench_kda shapes (issue #83, job 640235)

`meta/benchmarks/bench_kda_chunked.hip` (+ `meta/scripts/ab_kda_chunked_mi300.sh`)
re-measures the committed cooperative per-token kernel (above) against
`kda_delta_rule_fwd_chunked[_with_scratch]` (#70) with identical inputs/events
and the same roofline columns (`ab` rows; one short-lived process per shape,
MI300A per-context fault workaround; `sbatch -A a-infra02`). beverin
nid002768, 2026-09-17 — `setperflevel` not permitted on that node, so treat
small deltas as ±few %; raw log:
`meta/benchmarks/artifacts/issue-83/run-640235-ab-k3-shapes.out`.

K3 six-config set:

| H | S | D | coop us(med) | chunked us(med) | speedup | chunked GB/s |
|--:|--:|--:|--:|--:|--:|--:|
| 16 | 64 | 64 | 150.2 | 384.8 | 0.39x | 177 |
| 1 | 64 | 16 | 64.2 | 348.5 | 0.18x | 0.8 |
| 1 | 64 | 32 | 113.9 | 345.9 | 0.33x | 3.1 |
| 1 | 64 | 64 | 149.1 | 379.1 | 0.39x | 11.2 |
| 1 | 512 | 64 | 1032.6 | 529.8 | 1.95x | 64.4 |
| 1 | 512 | 128 | 1897.7 | 719.0 | 2.64x | 188.2 |

H sweep at S=512 D=128 + long-S:

| H | S | D | coop us(med) | chunked us(med) | speedup | chunked GB/s |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 512 | 128 | 1894.8 | 720.3 | 2.63x | 188 |
| 8 | 512 | 128 | 1920.0 | 738.2 | 2.60x | 1466 |
| 16 | 512 | 128 | 2892.9 | 740.0 | 3.91x | 2925 |
| 32 | 512 | 128 | 3024.3 | 1107.4 | 2.73x | 3909 |
| 64 | 512 | 128 | 2928.3 | 1754.6 | 1.67x | 4935 |
| 128 | 512 | 128 | 3201.5 | 2935.5 | 1.09x | 5899 |
| 32 | 1024 | 128 | 5826.7 | 1825.1 | 3.19x | 4744 |
| 32 | 2048 | 128 | 11628.2 | 3311.1 | 3.51x | 5230 |

Phase decomposition (`kda_chunked_bench phases`): the gram (M/N) launch is
the largest single block cost (269 us of 748 at H=1 S=512 D=128; 1324 us of
2922 at H=128 S=512), then the state pass (339–1333 us); cumsum is ~23–36 us.

**Finding — the chunked WY kernel LOSES at the decode shapes.** At S=64
(cs=64, nc=1) it is 2.5–5x slower than the row-parallel kernel
(0.18–0.39x): the four fixed launches (cumsum, gram, inverse+GEMMs, state)
plus the 8·S·D scratch traffic and the gmem M/N round-trip cannot amortize
over a single 64-token chunk, and the B·H-block precompute (16–32 blocks at
these shapes) has no occupancy to hide it. The S≥512 prefill shapes are
where the WY form wins (1.95–3.91x; 3.19x/3.51x at S=1024/2048), consistent
with the S=512+ table in [kda-lds-optimization](kda-lds-optimization.md).
The re-measured coop entries also update the single-shape numbers of the
record table above (150.2 vs 143 us at 16 64 64; 1897.7 vs 2640 us at
1 512 128 — different node/day; the record table stays as bench_kda
measured it).

Correctness on the same job: `test_kda_chunked` **12/12 PASS** — chunked vs
`kda_naive_delta_rule_fwd_cpu` at 10 shapes incl. H=128 S=512 D=128
(max_rel ≤ 6e-6) and the nonzero-initial-state (multi-turn decode-with-
scratch) contract vs the cooperative kernel (out + final state,
max_rel ≤ 5e-6).

### Fused cumsum launch (`VK_KDA_CHUNKED_FUSED=1`, fusion lane)

The chain's first launch, `kda_k3_cumsum_kernel` (per-key within-chunk
log-cumsum, 23–36 us of the 748–2922 us phase decomposition above), only
feeds the gram/local kernels **this chunk's** `L` — there is no cross-chunk
dependency. `kda.hip` therefore also ships `kda_k3_gram_cumsum_kernel`, the
gram kernel with the cumsum statements as a prologue (one thread per key
column, same ascending order, same log(0) clamp): the chain runs **3
launches instead of 4** and the gram kernel stages `sL` from the computed
registers instead of re-reading `L` from gmem (the `L` gmem scratch is still
written — the (2b) local kernel reads it).

Bit-identity with the proven chain holds by construction (same statements,
same order → same `L` → same every float downstream) and is enforced by
`tests/kernels/attn/test_kda_chunked_fused_gpu.cpp`: fused-vs-proven out AND
final state must compare `memcmp`-equal, plus oracle parity of both chains
against `kda_delta_rule_fwd_state_cpu`.

- **Gate**: env `VK_KDA_CHUNKED_FUSED=1` enables the fused chain; DEFAULT
  (unset/0) is the proven 4-launch chain. Flip the default only after the
  MI300A A/B (`meta/scripts/ab_kda_chunked_mi300.sh` extension, worklog
  NOTES-fusion-dsa-kda-launches.md).
- **GB10 shim validation** (HIP-on-NVIDIA, D≤64 templates — NVIDIA's 48 KB
  static-shared cap excludes the 64 KB D=128 gram staging; D=128 stays
  HIP-only): bit-identical at {1,1,64,16}, {1,2,128,64}, {2,1,128,32};
  oracle max_rel ≤ 4.4e-7. Wall A/B (sync-per-call, 200 iters):
  32% faster at H=16 S=64 (launch-bound decode shape), ~5% at H=16 S=512,
  ~1% at H=32 S=512 — GB10 is launch-overhead-dominated; the MI300A numbers
  are the ones that matter for serving.

### Supporting kernels

| kernel | shape | us(med) | GB/s |
|---|---|--:|--:|
| `kda_layer_norm_gated` | N=8192 D=128 | 4.8 (was 174) | ~2600 (was 72) |
| `kda_gate_chunk_cumsum` | B=1 H=16 nc=8 cs=64 | 13 | 5 |

`layer_norm_gated` was occupancy-bound at N=8192: 32 blocks (one per 256
rows) on 228 CUs, 72 GB/s = 1.4% of HBM. Issue #144 rewrote it as
**one warp per token row with `float4` loads** (256-thread block → 8 rows
→ `N/8 = 1024` blocks; lane `l` holds elements `4l..4l+3`; sum-of-squares
via a 5-step `__shfl_xor` butterfly; normalize+SiLU write also `float4`).
Measured on beverin MI300A (nid003020, ROCm 6.3): **170.7 → 4.8 µs
(74 → ~2600 GB/s ≈ 49% of the 5300 GB/s HBM roof, ~35×)**, batched
1000-launch event timing (`meta/scripts` A/B standalone, job 641303);
rocprof per-launch average 6.15 µs. Output identical to the one-thread
layout (max_abs_diff 3e-7) and to the CPU oracle (max_rel 1e-6,
test_kda_correct 11/11 PASS). The one-thread-per-row kernel remains as
the fallback for `D % 4 != 0`.
