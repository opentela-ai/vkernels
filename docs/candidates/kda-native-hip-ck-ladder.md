# KDA delta rule — the native HIP/CK ladder (perf3/kda-hip)

**Status: lane C (skeleton + oracle harness) landed. The rung-1 standalone
primitive passed gfx942 device/oracle parity on 2026-10-19. Full-chain and
performance qualification remain outstanding; rungs 2–3 are planned.**

## Rung-1 device check, 2026-10-19

Beverin job **687212**, a separate MI300A allocation on `nid002682`, compiled
the native HIP implementation with ROCm 7 and compared the standalone MFMA
grams with `kda_k3_grams_blocked_cpu` for D=16/32/64/128, two heads, two
64-token chunks, and both no forgetting and random gates in [0.1, 0.99].
All eight cases passed the existing 2e-4 absolute tolerance; the largest
observed M/N error was 2.38419e-7. The cases check the strict/inclusive
triangles as well as values. The permanent HIP CTest is `kda_grams_gpu`. The final formatted test and
CUDA-shim timing correction were rechecked by job **687229** on `nid002672`,
with the same eight passing cases and maximum error. See the
[raw results and source hashes](evidence/2026-10-19-kda-grams-mi300a/validation.json).

This checks the standalone primitive, not the full chunked-WY opt-in route or
its speed. `VK_KDA_GRAM_MFMA` remains off by default. The factorisation reduces
the analytical exponent count by about 6.5x at cs=64/blk=16, not 65x; this is
not a measured end-to-end speedup. The consolidation also keeps the D=128
scalar timing path available to CUDA-shim builds.

Branch `perf3/kda-hip` (off 137d590). This doc is the working plan for
moving the K3 per-key-dim gated-delta-rule forward from the committed
scalar chunked-WY chain (`kda.hip` #L8) to matrix-core (MFMA) kernels,
with the option of CK tile primitives where hand-tuning stalls. It is the
forward-side ladder the trainer's DESIGN.md K4 entry (backward) will
mirror.

## 1. Why: where the chain spends its time

Measured on gfx942 (`meta/benchmarks/artifacts/issue-83/run-640235`,
`phases` mode, D=128, cs=64):

| shape | cumsum | **grams** | local | **state** | sum |
|---|---|---|---|---|---|
| H=1 S=512 | 22.6 µs | **269.4 µs** | 116.8 µs | 339.2 µs | 748 µs |
| H=32 S=2048 | 33.1 µs | **1325.1 µs** | 607.2 µs | **1332.8 µs** | 3298 µs |
| H=128 S=512 | 36.3 µs | **1324.0 µs** | 623.4 µs | 938.5 µs | 2922 µs |

The 8B training microbatch (MICRO_BS=1 ACCUM=4, H=32 KDA heads,
D=128, cs=64, S=8192 → nc=128) is ~4× the S=2048 row per layer, ×24 KDA
layers. Two phases dominate at ~40% each: **the gate-weighted grams** and
**the serial state pass**. The grams are exp/SFU-bound (the LDS-staging
work in `docs/kda-lds-optimization.md` already removed the gmem-latency
bound); the state pass is chunk-serial with barrier-heavy scalar GEMMs.

## 2. What already exists (do not rebuild)

- **CPU oracle + chunked CPU WY** (`kda.cpp`): `kda_naive_delta_rule_fwd
  _cpu` (per-token oracle), `kda_delta_rule_fwd_state_cpu` (chunked WY,
  explicit state in/out — the math the device chain implements), plus the
  standard-rule L2–L6 pipeline (different recurrence, kept for non-K3).
- **Device chain** (`kda.hip` #L8): 4 launches — per-key log-cumsum;
  scalar per-pair grams (LDS-staged k/L, streamed q); chunk-local
  artifacts (explicit `(I+tril(M,−1))⁻¹` by warp-column substitution +
  level-scheduled off-diagonal blocks, then U_v/W/T/Opar GEMMs); serial
  state pass rowblock-split over D. Env `VK_KDA_CHUNKED_FUSED` folds the
  cumsum launch into the gram prologue.
- **Golden-tensor harness** (this lane): `bench/kda_trainer_golden.py`
  imports the trainer's `vtrainer.model.kda` (read-only) and emits
  deterministic CPU goldens for end-to-end out/state in BOTH layout
  conventions, plus the trainer's piece tensors. See §6.
- **Block-anchored gram oracle** (this lane): `kda_k3_grams_blocked_cpu`
  + host tests (`KdaK3GramsBlocked.*` in `tests/kernels/attn/test_kda.cpp`).
- **MFMA gram kernel skeleton** (this lane): `kda_k3_gram_mfma_kernel` +
  `hip::kda_k3_grams_blocked` + `VK_KDA_GRAM_MFMA=1` dispatch (default
  off) + `kda_chunked_phase_times` `out_us[10]` A/B hook.

## 3. Piece → kernel map (the ladder)

Rung numbering = suggested landing order (oracle-first, one rung at a
time; every rung gates on parity vs `kda.cpp` + the golden file before
any perf work, then on `bench_kda_chunked phases` before flipping the
default).

| # | piece (per chunk, cs=64) | math shape | current | planned | effort* |
|---|---|---|---|---|---|
| — | per-key log-cumsum | [cs×D] scan | `kda_k3_cumsum_kernel` (+ fused prologue variant) | keep (1% of chain) | — |
| 1 | grams M/N | per (P,Q≤P) 16-row block pair: `M,N[16×16] = ±(A·Bᵀ)`, K=D | scalar per-pair, exp/SFU-bound (1325 µs) | **block-anchored MFMA** (skeleton landed; the anchor factorisation kills 6.5× of the exp work, MFMA kills the FMA work) | 1–2 d |
| 2 | chunk-local: Ainv, U_v/W, T/Opar | `Ainv@X`: [64×64]·[64×D]; `N@X` tril-incl | scalar loops (607 µs) | MFMA GEMMs (M=64→4 row frags, N=D→8 col frags, K=64); Ainv diagonal blocks stay warp-substitution, off-diagonal level products become 16×16 MFMA chains | 2–3 d |
| 3 | state pass: u, o, C-update | `W@Cᵀ` [64×128]·[128×Rb]; `Kgbᵀ@u` [128×64]·[64×Rb], Rb=32 | chunk-serial scalar, 3 barriers/chunk (1333 µs) | MFMA per chunk (still serial over nc; LDS C tile with D+4 padding stays) | 2–3 d |
| 4 | cross-chunk scan (optional) | per-chunk affine maps A_c [D×D], B_c [D×D_v]; log2(nc) associative combine | not built (trainer ships `_scan_linear` in torch) | only if the serial pass still binds after rung 3: the D×D map products are plain [128×128]² GEMMs | 3–5 d |

\* focused days on a gfx942 box, incl. parity + bench gates, on top of
this lane.

**Cumsum-vs-grams fusion note:** the fused prologue (`VK_KDA_CHUNKED
_FUSED`) is orthogonal to rung 1 — it applies unchanged to the MFMA
kernel's staging loop if the A/B says the launch overhead matters.

## 4. Rung 1 detail (this lane's skeleton)

Math (identical cells to the scalar grams; fp32-equivalent, not
bit-equal): for the pair (row-block P, col-block Q ≤ P) with anchor
`S[d] = L[col_end(Q)−1][d]` (L = per-key within-chunk inclusive
log-cumsum, monotone decreasing),

```
a^k_t[d] = exp(L_t − S)[d] · k_t[d]      row operand (M)
a^q_t[d] = exp(L_t − S)[d] · q_t[d]      row operand (N)
b_j[d]   = β_j · exp(S − L_j)[d] · k_j[d] col operand (both)
M[t][j] = Σ_d a^k_t[d]·b_j[d]  (j<t)     N[t][j] = Σ_d a^q_t[d]·b_j[d] (j≤t)
```

Overflow analysis (from the trainer's `_wy_grams`, reproduced in the
kda.hpp header comment): both exp arguments are ≤ 0 except on the
diagonal block P==Q, where the exponent is bounded by ONE block's decay
(≤ blk·5 = 80 nats at the gate bound — inside fp32 exp). This is the
same factorisation the trainer measured at 2.9e-5 abs vs its mask
reference.

Kernel structure (`kda_k3_gram_mfma_kernel<D>`, skeleton):
grid `(B·H, nc)`, one 64-thread wavefront per block, 10 block pairs
sequential per block; LDS stages the three exp-folded operand tiles
(3 × 16×D fp32 = 24 KB at D=128); K-loop of D/4 `MFMA_F32_16x16x4K1`
(fp32 operands, fp32 accumulate — same numerics class as the scalar
rung); one B-fragment serves both the M and N accumulators; epilogue
writes strict-lower M / tril-inclusive N, upper cells pre-zeroed.
Later variants to A/B: 2 block-pairs resident per wavefront (amortise
staging), 32×32 C tiles (4 MFMAs per K step sharing B fragments), and —
**only after the tolerance gate** — bf16 operands
(`mfma_f32_16x16x16bf16_1k`, 4× the MACs/instruction), which rounds the
operand tiles and must be cleared against the golden end-to-end
tolerance first (the explicit Ainv amplifies Gram errors; |M| ≤ 1 by
Cauchy–Schwarz keeps it bounded, but the measured number decides).

**Skeleton risks (explicit):** the MFMA fragment layouts follow the
empirically verified 16×16 discipline of `gemm_bf16.hip` but the
fp32 16×16×4 variant was never run here — the first device step is a
layout confirmation (tiny probe or direct parity). fp32-MFMA throughput
on gfx942 (K=4 per instruction vs K=16 bf16) must be measured; if the
wavefront is issue-bound rather than MAC-bound, the bf16 rung becomes
the primary path and rung 1's value collapses to the exp reduction —
either way the `phases` bench answers it in one run.

## 5. Where the triangular solve lives

In the chunk-local kernel, as the **explicit inverse**
`Ainv = (I + tril(M,−1))⁻¹`, exactly where it is today. Rationale:

1. `Ainv` is needed as a **GEMM operand** (U_v = Ainv·v, W = Ainv·Kgw);
   a library batched trsm (rocBLAS/rocSolvers) would round-trip it
   through gmem and split the chunk-local kernel into launch pieces —
   the repo just spent issue #70/#83 removing launch overhead.
2. The trainer measured the alternatives at the 8B shape
   (`_wy_inverse` docstring): Neumann product form diverges at cs=64
   (rel 8.7e-3 at |M|≤0.5, 8.2e4 at |M|≤2); blocked-16 series form is
   numerically fine but 2.5–3× slower than the substitution.
3. The current decomposition (warp-per-column forward substitution for
   the four 16×16 diagonal blocks + level-scheduled off-diagonal
   products) is already blocked at 16 — rung 2 upgrades the
   off-diagonal products to MFMA without moving the solve.

CPU-side the solve is forward substitution inside
`kda_delta_rule_fwd_state_cpu` — the parity reference. (Note the
convention difference: the trainer's WY matrix is the NEGATED,
row-β-folded form `(I − M)⁻¹`; vkernels stores M positive with β on the
column side. Piece tensors from the two repos are not directly
comparable — only end-to-end out/state are. §6.)

## 6. Oracle harness (landed): `bench/kda_trainer_golden.py`

CPU-only, torch; locates the trainer via `$VTRAINER_SRC` (default the
sibling `../../trainer/src`). Per shape (default `B1_S256_H4_D128_cs64`,
`--with-8b` adds `1,8192,32,128,64`) it generates deterministic
module-contract inputs (q/k L2-normalised, log-gates in [−5,0], sigmoid
β), runs the trainer's `kda_recurrent` (per-token oracle),
`kda_chunked` (mask-form) and `kda_chunked_fast` (`_wy_grams` path),
plus the piece tensors (g_cum, attn, attn_intra, Ainv, u, k_cumdecay,
per-chunk state trace) — all fp32 CPU — and writes `scratch/kda-trainer
-golden.pt`. Internal gates: chunked/fast/piece-replay vs the recurrent
oracle at rel ≤ 1e-3 (measured ~2e-6).

The file also exports **vkernels-layout twins** (`*_vk`): [B,H,S,D]
inputs with q pre-scaled by D^-1/2, normal-space gates `exp(g_log)`,
β as [B,H,S], and out/state transposed to the canonical `S[v][k]`
state layout. A device-parity test (next lane) feeds `*_vk` to the HIP
entry points and compares `out`/`state` against `out_vk`/`state_vk` —
tolerance from the measured CPU-vs-CPU spread plus device rounding
(start at rel 1e-3; the state layout transpose is already done).

Gate-space warning for device tests: the trainer g is LOG-space;
vkernels APIs take NORMAL-space g in (0,1] and log internally. Feed
`g_vk = exp(g_log)` — never re-log it.

## 7. Bench hooks (landed)

- `bench_kda_chunked phases H S D` now prints
  `rung1 gramsab: scalar=…us mfma=…us (…x)` (out_us[10], the rung-1 A/B
  hook) alongside the per-launch table.
- `hip::kda_k3_grams_blocked(...)` is a standalone entry (same contract
  as the CPU oracle, `blk==16` on device) so a parity harness can drive
  just launch (2a).
- `VK_KDA_GRAM_MFMA=1` selects the MFMA grams inside the full chunked
  chain (default off; the scalar chain stays the proven default until
  the A/B + parity gates pass — the `VK_KDA_CHUNKED_FUSED` discipline).

## 8. Lane C status / next steps

Landed: CPU oracle (`kda_k3_grams_blocked_cpu`) + host tests (3 cases,
17/17 suite pass); golden harness (validated against the live trainer);
HIP skeleton + dispatch + bench hook; this plan. Host CI-equivalent
(`cmake --preset host` + full ctest) passes 47/47; the nvcc HIP-shim
build compiles `kda.hip` (MFMA region excluded by design — gfx942-only).

Completed on 2026-10-19: hipcc compile on gfx942 and standalone
`hip::kda_k3_grams_blocked` parity for all four admitted widths.

Next lane (device, gfx942):
1. Full chain vs `kda_delta_rule_fwd_state_cpu`; end-to-end vs the golden
   file (`*_vk` tensors), including restored initial/final state.
2. `phases` A/B at H∈{1,32} S∈{512,2048,8192} D=128; decide rung-1
   variant (pair batching, 32×32 tiles, bf16 study).
3. Then rung 2 (local GEMMs), rung 3 (state pass), re-gating each.
