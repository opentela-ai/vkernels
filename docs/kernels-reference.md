# vkernels — Kernel Reference: Achieved Performance & Gap to Speed of Light

> **One-stop reference** of every kernel and communication primitive in the
> repo, with the hardware it has been tested on, its headline measured
> performance, and its gap to "speed of light" (the roof of whichever
> resource actually binds it). Sources are the per-kernel records under
> [`docs/performance/`](performance/) and the inline benchmark sections of
> the kernel docs. Every number links back to a raw log or a reproduce
> script; nothing here is projected.
>
> **How to read "gap to SOL":** each kernel is classified by its *binding*
> constraint. A memory-bound kernel is compared against the HBM (or L2)
> roof; a compute-bound kernel against the Matrix-Core/Tensor-Core roof; a
> **launch/occupancy-bound** kernel is compared against a *floor*
> (dispatch cost or CU count), not a bandwidth roof — its "% of HBM roof"
> is tiny by construction and is **not a defect**. Reading a 0.1%-of-HBM
> number on a launch-bound 20 µs bookkeeping kernel as "640× headroom"
> would be wrong; the honest SOL for that kernel is the ~3 µs dispatch
> floor it already sits near.

---

## 1. Speed-of-light reference (roofs used throughout)

Unless noted, roofs are the in-binary measured/quoted values from the
benchmark records, not datasheet marketing numbers.

| Device | Arch | Compute roof (bf16) | Memory roof | Ridge (AI) | Notes |
|---|---|---:|---:|---:|---|
| AMD MI300A | gfx942 | 1307 TFLOP/s | 5300 GB/s HBM3 | ~247 FLOP/B | 228 CU; L2 copy measured 3219 GB/s, HBM copy 2868 GB/s ([moe_aux](kernels/moe_aux.md)) |
| AMD MI250X (1 GCD) | gfx90a | ~191 TFLOP/s | ~3.3 TB/s HBM2e (package ~6.6) | — | benches ran device 0 only (one GCD) |
| NVIDIA A100-SXM4-80GB | sm_80 | 19.5 TFLOP/s (fp32 CUDA-core; kpool kernels use no tensor units) | 2039 GB/s HBM2e | ~10 FLOP/B | via HIP→CUDA shim |
| NVIDIA GB10 | sm_121 | 252 TFLOP/s (measured wmma micro-kernel) | 227 GB/s LPDDR (measured copy) | ~1108 FLOP/B | 48 SM; desktop SoC, 23× less BW than MI300A |
| NVIDIA H100 NVL | sm_90 | — | ~265 GB/s unidirectional NVLink (measured); 600 GB/s bidirectional datasheet; HBM3 ~3.35 TB/s | — | sgs-gpu07 (4×, NV12 pairs) |
| GH200 + JSC IB HDR | sm_90 + CXI | — | same-node NVLink ~220–243 GB/s; fused restore 88.5 GB/s; cross-node one HDR-200 port ~24 GB/s | — | [comm-cross-node-kv.md](comm-cross-node-kv.md) |

---

## 2. Master table — every measured kernel

Sorted by family. "Best" = best measured configuration at the headline
serving shape, not a theoretical optimum.

| # | Kernel | HW tested | Headline shape | Achieved | Binding roof | Gap to SOL | Binding constraint | Record |
|---|---|---|---|---|---|---|---|---|
| 1 | `gemm_bf16` (HIP MFMA) | MI300A | K3 QKV 6288×7168, M=64 (serving, split-K #146) | 244.8 µs, 2949 GB/s | 5300 GB/s HBM | **56% of HBM** | memory (AI≈8 ≪ ridge 247; S=8 split-K, combine round-trip included) | [gemm-bf16/gfx942](performance/gemm-bf16/gfx942.md) |
| 2 | `gemm_bf16` (HIP MFMA) | MI300A | same, M=8192 (warmup) | 76.2 TFLOP/s, 2400 GB/s | 1307 TFLOP/s / 5300 GB/s | **5.8% compute, 45% HBM** | memory (B re-read ⌈M/BM⌉×, AI≈31) | same |
| 3 | `gemm_bf16` (CUDA wmma) | GB10 | same shape, M=64 | 547 µs, 826 GB/s* | 227 GB/s LPDDR | ~100% of LPDDR roof | memory (pinned to LPDDR ceiling; compute units 30× idle) | [gemm-bf16/gb10](performance/gemm-bf16/gb10.md) |
| 4 | `glm_fp8_block_gemv` | MI300A | GLM gate/up [4096,4096], M=1 | 22.4 µs, 751 GB/s | 5300 GB/s | **14% of HBM** | instruction-issue (all paths plateau 500–800 GB/s after 3 structural rounds) | [glm53-decode-kernels](glm53-decode-kernels.md) |
| 5 | `fused_moe_mxfp4` (decode cfg) | MI250X | E=256 h4096 i512 k6, M=48 | 10.33 ms, 351 GFLOP/s | 191 TFLOP/s/GCD | 0.18% of compute | padding waste (EM 3456 vs 288 real rows) + inline dequant ALU | [moe-fused/gfx90a](performance/moe-fused/gfx90a.md) |
| 6 | `fused_moe_mxfp4` (decode cfg) | MI300A | same, M=48 | 4.07 ms, 890 GFLOP/s | 1307 TFLOP/s | 0.07% of compute | same | [moe-fused/gfx942](performance/moe-fused/gfx942.md) |
| 7 | `fused_moe_mxfp4` (prefill cfg) | MI250X | E=8 h4096 i512 k2, M=2048 | 5.78 ms, 8.9 TFLOP/s | 191 TFLOP/s/GCD | **4.7% of compute** | occupancy (dequant ALU; 3 blocks/CU) | same gfx90a |
| 8 | `fused_moe_mxfp4` (prefill cfg) | MI300A | same, M=2048 | 1.98 ms, 26.0 TFLOP/s | 1307 TFLOP/s | **2.0% of compute** | occupancy (inline E2M1 dequant dominates MFMA) | same gfx942 |
| 9 | `fused_moe_mxfp4` (K3 routing) | MI300A | E=256 h7168 k16, M=1 | 1.108 ms, 0.020 TFLOP/s | 1307 TFLOP/s | ≪0.01% | decode padding (93.8% padded rows) | same gfx942 |
| 10 | `mxfp4_moe_sort` | MI300A | K3 M=112, EM=2048 | 24.4 µs, 2253 GB/s | 3219 GB/s L2 | **70% of L2** | data movement, near roof | [kernels/moe_aux.md](kernels/moe_aux.md) |
| 11 | `mxfp4_moe_scatter_reduce` | MI300A | same | 42.4 µs, 1461 GB/s | 3219 GB/s L2 | 45% of L2 | atomicAdd contention (16 rows/token) | same |
| 12 | `mxfp4_moe_quant` / `_q` / `_sort_scales` | MI300A | same | 17.8 / 43.7 / 5.6 µs | L2 / launch floor | 3.6–7.8% L2; sort_scales launch-bound | compute (E2M1 round-trip) / launch | same |
| 13 | `mla_fwd` | MI300A | H=128 S_q=S_kv=512 (prefill, best case) | 15.9 ms, 2318 GB/s | 5300 GB/s | **44% of HBM** | memory (AI 2.0; KV re-read per query tile) | [kernels/mla.md](kernels/mla.md) |
| 14 | `mla_fwd` split-K decode | MI300A | H=1 S_q=1 S_kv=8192, split=228 | **66.8 µs, 533 GB/s** (was 5.5 ms, 6.5 GB/s pre-split) | 5300 GB/s | ~10% of HBM | **82× vs the single-block baseline**; 32-keys/split serial chain | [kernels/mla.md](kernels/mla.md) |
| 15 | `kda_delta_rule_fwd` | MI300A | H=16 S=64 D=64 | 143 µs, 477 GB/s | 5300 GB/s | 9.0% of HBM | memory (D×D state → HBM per token, 3× re-read) | [kernels/kda.md](kernels/kda.md) |
| 16 | `kda_layer_norm_gated` | MI300A | N=8192 D=128 | 174 µs, 72 GB/s | HBM | 1.4% | occupancy (32 blocks on 228 CUs) | same |
| 17 | `dsa_sparse_fwd` (plain) | MI300A | GLM decode, topk=2048 (full) | 3.12 ms, 21.5 GB/s | HBM/L2 | 0.4% HBM | occupancy (64 blocks / 228 CU, serial key chain) | [dsa/gfx942](performance/dsa/gfx942.md) |
| 18 | `dsa_sparse_fwd_split` | MI300A | same, split=64 | **0.156 ms**, 430 GB/s | 5300 GB/s | **8.1% HBM** (20.0× vs unsplit) | key-stream latency (8–32 keys/split serial chain) | same |
| 19 | `dsa_sparse_fwd` (plain, prefill) | MI300A | H=1 S_q=8192 topk=128, GLM | 0.49 ms, 1116 GB/s | 5300 GB/s | **21% of HBM** | memory | same |
| 20 | `dsa_sparse_fwd` (plain, prefill DSv3) | MI300A | H=1 S_q=8192 topk=256, W=640 | 1.85 ms, 1464 GB/s | 5300 GB/s | **28% of HBM** (best in family) | memory | same |
| 21 | `dsa_topk_logits` (indexer, mfma/auto) | MI300A | GLM H=32 mt=8 (msl=512) | 264 µs | one-wavefront floor | **10.3× faster than fp32-Q scalar**; occupancy-bound | occupancy (1 wavefront; split_kv scales) | [dsa-topk/gfx942](performance/dsa-topk/gfx942.md) + mk2 log |
| 22 | `dsa_topk_logits` + split_kv=64 | MI300A | H=32, msl=4096 | 190 µs (4.7× vs split=1) | 228 CU | near-linear split scaling to 64 CU | occupancy | same |
| 23 | `dsa_kpool_assemble` / `_decode_update` | MI300A | ps=8 t128, 256 pools / bs=512 | 12.2–21.5 µs, ≤214 GB/s | 5300 GB/s / 2.8 µs dispatch floor | 4% HBM; ~4–7× dispatch floor | **launch/occupancy** (sub-µs kernels; ~15 ns/request marginal) | [dsa-kpool/gfx942](performance/dsa-kpool/gfx942.md) |
| 24 | `dsa_kpool_*` (A100 port) | A100 (via HIP→CUDA shim) | same | 6.0–15.7 µs, ≤294 GB/s | 2039 GB/s | 14.5% HBM | launch | [dsa-kpool/A100](performance/dsa-kpool/A100.md) |
| 25 | `dsa_kpool_*_fp8` | MI300A + A100 | same | +10–20% vs bf16 | — | deliberate: half cache footprint + graph capture | launch (2 extra smem reductions) | [dsa-kpool/fp8](performance/dsa-kpool/fp8.md) |
| 26 | `mhc_pre_gemm_sqrsum` | MI300A | GLM decode n=1, hc_hidden=16384 | 278 µs, 5.8 GB/s | HBM | 0.1% HBM | occupancy (**1 block on 228 CUs**; serial hc_hidden walk) | [mhc/gfx942](performance/mhc/gfx942.md) |
| 27 | `mhc_post` | MI300A | n=1..7, hc=4, hidden=4096 | < 0.5 µs | event-timer floor | free | negligible — no work warranted | same |
| 28 | torch_ops `qkv_projection` (Triton) | MI300A | [3×8192,4096], M=1 | 70.1 µs vs 517.5 BLAS | — | **7.4× vs default BLAS** (TunableOp 76 µs) | BLAS algorithm choice | [qkv-projection](qkv-projection.md) |
| 29 | torch_ops `mhc_projection` (Triton) | MI300A | [24,16384], M=1 | 3.94 µs vs 146.2 BLAS | — | **37× vs default BLAS** | single-GEMV dispatch overhead | [mhc-projection](mhc-projection.md) |
| 30 | torch_ops `glm_projection` (Triton) | MI300A | N∈{64..1536}, M=1 | 5.9–9.7 µs | — | **no win** (BLAS already near-optimal; honest negative) | none | [glm53-decode-kernels](glm53-decode-kernels.md) |
| 31 | `p2p_gather_runs` (adaptive) | H100 NVL | 48 MiB, real NVLink peer | ~210 µs flat (240 GB/s) | 265 GB/s NVLink unidir | **~90% of NVLink roof** | NVLink bandwidth | [p2p-gather/h100-nvl](performance/p2p-gather/h100-nvl.md) |
| 32 | `p2p_gather_runs` | GB10 | 2048 runs × 2 KiB | 53.3 µs vs 4096.5 µs loop | dispatch floor | **76.8× vs per-run loop** | per-run driver cost removed | [p2p-gather/gb10](performance/p2p-gather/gb10.md) |
| 33 | `p2p_kv_donate` (prepared plan) | H100 NVL (D2D) | 48 MiB/layer × 40 layers | 36.9 µs/layer = 1.37 TB/s payload (2.73 TB/s touched) | measured D2D copy roof ~1.46 TB/s payload | **~94% of D2D copy roof** | copy/gather bound; page grid-stride landed, fixed >65535-page truncation (#84) | [p2p-kv-donate/h100-nvl](performance/p2p-kv-donate/h100-nvl.md) |
| 34 | `pipeline_boundary` (PP transfer) | H100 NVL | NVLink pair | 265 GB/s unidirectional | 600 GB/s bidir datasheet | **44% of NVLink roof** | NVLink; graph replay 3.3 µs, 1.6–1.7× host win | [comm-pipeline-boundary](comm-pipeline-boundary.md) |
| 35 | cross-node KV (donate/restore) | GH200 + IB HDR | 2 nodes | ~24 GB/s per hop | one HDR-200 port ~25 GB/s | **~96% of one port** | fabric (2-rank p2p can't stripe 4 HCAs) | [comm-cross-node-kv](comm-cross-node-kv.md) |
| 36 | K3 PP=3 serving profile | MI300A ×6 | Kimi-K3 dummy, PP3/TP8 | breakable cudagraph 35.2 tok/s vs eager 25.3 | — | **1.39× throughput**; MoE region unchanged | host sync (`topk_ids.cpu()`) on PP0 | [moe-fused/gfx942-pp-pipeline](performance/moe-fused/gfx942-pp-pipeline.md) |

---

## 3. Detail by family

### 3.1 Dense GEMM — `gemm_bf16` (K3 projection shapes)

Two ports of the *same algorithm* (two-phase tiled loop, 16×16×16
matrix-core instruction) — with an increasingly arch-specific tail on
CUDA: a `cp.async` double-buffered pipeline for serving, and an M-grouped
cross-tile B-reuse kernel for warmup. The HIP port keeps the synchronous
form; issue #77 ported the reuse + LDS double-buffer kernel to HIP and the
on-device M=8192 autotuner measured it ~2× slower than the flat `(64,64)`
tile on MI300A (occupancy collapse from the 24 KB LDS ring — see
[gemm-bf16/gfx942](performance/gemm-bf16/gfx942.md)), so the AMD warmup
path stays on the flat tile and the reuse kernel remains an
offline-autotuner / correctness-sweep entry only.

| Config | HW | M | Time | TFLOP/s | GB/s | AI | % of binding roof |
|---|---|---:|---:|---:|---:|---:|---|
| QKV (6288×7168), (16,16) tile | MI300A | 5 | 346 µs | 1.30 | 342 | 3.8 | 6.5% HBM |
| | MI300A | 64 | 376 µs | 15.3 | 1919 | 8.0 | 36% HBM |
| QKV (6288×7168), split-K S=8 (#146) | MI300A | 5 | **99.6 µs** | 4.5 | 1189 | 3.8 | 22% HBM |
| | MI300A | 64 | **244.8 µs** | 23.6 | 2949 | 8.0 | **56% HBM** |
| | GB10 (16,16) tile (pre-fix) | 5 | 636 µs | 0.71 | 186 | 3.8 | 82% LPDDR |
| | GB10 (16,16) tile (pre-fix) | 64 | 2248 µs | 2.57 | 321 | 8.0 | ~100% LPDDR |
| | GB10 **(16,64)** cp.async (per-arch default) | 5 | 442 µs | 1.02 | 220* | 4.6 | see gb10.md |
| | GB10 **(16,64)** cp.async (per-arch default) | 64 | **547 µs** | 10.54 | 826* | 12.8 | see gb10.md |
| Warmup (64,64) tile, all shapes | MI300A | 8192 | 322–9700 µs | 61–78 | 2282–2474 | ≈31 | **45% HBM / 5.8% compute** |
| Warmup (16,64,RM4) reuse, all shapes | GB10 | 8192 | 210–53588 µs | 13.7–20.5 | 426–659* | ≈31 | see gb10.md |

\*GB/s is the no-reuse *model* byte rate; L2 absorbs most modeled re-reads, so
it is not a DRAM-roof fraction.

* The two chips run the same memory-bound kernel at their respective
  effective-memory ceilings — the ~4–5.6× TFLOP/s gap at warmup is now
  mostly the ~10× effective-bandwidth gap, plus a residual latency term on
  GB10's 48 SMs.
* Highest-leverage next rung for MI300A serving: the #146 split-K landed
  (M ≤ 64 routed to `gemm_bf16_splitk` at `S = min(8, K/64)`; QKV serving
  346 → 99.6 µs at M=5, 376 → 244.8 µs at M=64 = 56% of HBM) — the
  remaining serving gap is the workspace round-trip plus the two-phase
  tile's exposed load latency inside each split. Warmup (`M = 8192`, 45%
  effective HBM) is unchanged by split-K: cross-tile B reuse (persistent
  kernel) would lift AI from ~31 toward ~2378. On GB10 the `cp.async`
  double-buffer landed first (1.3–1.8×) and cross-tile reuse then landed as
  an **L1TEX/register-blocking** lever for warmup (2–52%); serving is
  already at the DRAM roof, so reuse is gated on `M > 64`. On MI300A the
  #77 reuse + LDS double-buffer port was **measured and rejected** (≈2×
  regression, occupancy-bound — the flat `(64,64)` tile already runs at
  45% effective HBM with 3× the reuse kernel's occupancy); closing the
  45%→100% effective-HBM gap there needs a schedule that keeps
  ≥1536 resident threads/CU while pipelining (e.g. `(64,64)` with a
  single-buffered sB pipeline or wider `BN`), not the GB10 ring layout.
* GB10's *synchronous* kernel preferred `(32,64)` (1.4–2.3× over `(16,16)`);
  the **cp.async double-buffer** (1.3–1.8×) then re-tuned it to `(16,64)` at
  serving and, with the **M-grouped `(16,64,RM4)` reuse** kernel, gave the
  warmup small-K shapes another 26–52%. `gemm_bf16_config_for` is per-arch
  (GB10 `(16,64)`/`(64,64)`, MI300A `(16,16)`), so the QKV `M=64` default is
  now 547 µs (was 2248 µs pre-fix, 986 µs after the arch config).
* GB10 warmup is now ~13.7 TFLOP/s on the large-K shapes and ~19.5–20.4
  TFLOP/s on the small-K ones (QKV 95458 → 53588 µs); the earlier "`(32,64)`
  beats `(64,64)` at large K" finding was a latency effect that the pipeline
  removed.

### 3.2 GLM-5.3 block-FP8 expert GEMV — `glm_fp8_block_gemv`

| Shape | M | Fused | Materialized BF16 GEMV (steady) | Win |
|---|---:|---:|---:|---:|
| gate/up [4096,4096] | 1 | 22.4 µs (751 GB/s) | 42.0 µs | **1.88×** |
| gate/up [4096,4096] | 2 | 29.5 µs | 62.2 µs | 2.11× |
| down [4096,2048] | 1 | 16.4 µs | 17.4 µs | 1.06× |
| Per decode token (top-8 experts) | — | ~324 µs | ~529 µs + dequant buffer | 1.6× |

Instruction-issue bound at 500–800 GB/s (9–15% of HBM) after three
structural rounds (split-K, branchless `2^120` decode, SEGS template).
Further gains need a different design (hardware fp8 converts, MFMA, fused
top-k gather), not tuning.

### 3.3 Fused MXFP4 MoE — `fused_moe_mxfp4`

Decode (E=256, h=4096, i=512, k=6), latency and useful GFLOP/s:

| M | MI250X | MI300A | xkernels torch (MI250X / MI300A) | MI300A speedup vs torch |
|---:|---:|---:|---:|---:|
| 1 | 0.65 ms | 0.45 ms | 6.0 / 4.4 ms | 9.8× |
| 8 | 3.4 ms | 0.89 ms | ~30 / ~15 ms | ~17× |
| 48 | 10.3 ms | 4.07 ms | 162 / 127 ms | 31.2× |

Prefill (E=8, top_k=2, dense routing), MI300A:

| M | decode ms | prefill ms | prefill TFLOP/s | % of 1307 roof |
|---:|---:|---:|---:|---:|
| 512 | 1.209 | 0.804 | 16.0 | 1.2% |
| 2048 | 4.982 | 1.981 | 26.0 | **2.0%** |

Verdict: ~1.5–2% of the compute roof, bound by the **inline E2M1 dequant
ALU**, not the MFMA. Occupancy was the proven lever (BN 128→64 → ~1.5×);
wavefront-specialised dequant is the follow-on. The fused kernel's win over
the torch loop is never touching the 138 GB materialized bf16 buffer.
(Baseline is torch, not Triton — xkernels' Triton backend SIGSEGVs under
ROCm 6.2.4 on both gfx90a and gfx942.)

Micro-primitives (`bench_moe.hip`, single wavefront — MFMA row is a
dependent-chain floor, not a device roof):

| Primitive | gfx90a | gfx942 |
|---|---:|---:|
| K16 MFMA dependent chain | 0.431 TFLOP/s | 0.539 TFLOP/s |
| fp4→bf16 dequant | 279 GElem/s | 652 GElem/s |
| LDS fill (`global_load_dwordx4`) | 4898 GB/s | 7090 GB/s |

### 3.4 Attention — MLA / KDA / DSA / MHC (all MI300A, gfx942)

**MLA** (`bench_mla.hip`, all memory-bound, AI ≤ 2.0):

| H | S_q | S_kv | µs (med) | GB/s | % HBM |
|--:|--:|--:|--:|--:|--:|
| 128 | 512 | 512 | 15875 | 2318 | **44%** |
| 16 | 512 | 512 | 3387 | 1358 | 26% |
| 1 | 8192 | 8192 | 58727 | 1244 | 23% |
| 1 | 64 | 8192 | 5315 | 107 | 2% |
| 1 | 1 | 8192 | 66.8 | 533 | ~10% (split=228, issue #82; was 5510 µs, 0.12%) |

**KDA** delta rule (all memory-bound, AI ≈ 0.43): best 477 GB/s (9% HBM,
H=16 S=64 D=64); worst 5.6 GB/s (H=1). The D×D state hits HBM every token
3× — the LDS-resident-state / chunked follow-on's target. Supporting
kernels: `kda_layer_norm_gated` 174 µs / 72 GB/s (occupancy: 32 blocks on
228 CUs), `kda_gate_chunk_cumsum` 13 µs / 5 GB/s (launch).

**DSA sparse forward** (decode, per layer, `split_for` recommendation):

| Shape | unsplit | split | speedup | split GB/s |
|---|---:|---:|---:|---:|
| GLM topk=128 | 203 µs | **45 µs** | 4.5× | 94 |
| GLM topk=256 | 397 µs | **59 µs** | 6.7× | 144 |
| GLM topk=2048 (full) | 3125 µs | **156 µs** | 20.0× | 430 (8.1% HBM) |
| DSv3 topk=256 | 742 µs | **94 µs** | 7.9× | 56 |

Prefill: 21–28% of HBM (1116–1464 GB/s) — the closest-to-roof attention
configurations in the repo. Cross-check vs FlyDSL (the ROCm DSL): our split
kernel moves ~2.7× more bytes/second at the same decode pattern; FlyDSL is
numerically wrong at D=256 anyway.

**DSA indexer** (`dsa_topk_logits`, mfma/auto variant): 264 µs at
msl=512 (10.3× vs the fp32-Q scalar baseline's 2717 µs), 893 µs at
msl=4096; `split_kv=64` cuts that to 190 µs (4.7×). Occupancy-bound (one
wavefront); batch scaling free to ~228.

**MHC**: `mhc_pre` 278 µs at decode (single-block floor — split-column
redesign documented); `mhc_post` sub-µs (free). n=7 costs the same as n=1.

**kpool bookkeeping** (`dsa_kpool_*`): 6–22 µs per step at serving shapes,
~15 ns/request marginal on MI300A (~7 ns on A100). Launch/occupancy-bound
by construction; 4% (MI300A) / 14.5% (A100) of HBM at worst. fp8+scale
overloads: +10–20% latency for half the persistent cache footprint and
graph-capturable `round_scale`.

### 3.5 torch_ops projections (MI300A, Torch 2.9 + Triton 3.4/3.5)

| Operator | Shape (M=1) | Default BLAS | TunableOp | Fused Triton | Triton win |
|---|---|---:|---:|---:|---:|
| `qkv_projection` | 3×[8192,4096] | 517.5 µs | 76.0 µs | **70.1 µs** | 7.4× |
| `mhc_projection` | [24,16384] | 146.2 µs | 6.6 µs | **3.94 µs** | 37× |
| `glm_projection` | N=64..1536 | 5.8–11.7 µs | 5.4–8.0 µs | 5.9–9.7 µs | none |

At M=2 the tuned BLAS wins qkv; mhc Triton stays flat (3.95 µs). The
`glm_projection` result is an honest negative: BLAS is already
near-optimal at these isolated shapes, so the operator stays in-tree for
correctness/graph-safety, not speed.

### 3.6 Communication primitives

**p2p_gather (adaptive, H100 NVL, 48 MiB real peer):**

| Runs | copy loop | kernel | adaptive | speedup |
|---:|---:|---:|---:|---:|
| 1–2 | 200–205 µs | 209 µs | copy | 1.00× |
| 4–16 | 221–308 µs | ~210 µs | kernel | 1.06–1.47× |
| 192 | 1824 µs | 214 µs | kernel | **8.5×** |

Kernel is flat ~210 µs (≈240 GB/s ≈ 90% of the measured 265 GB/s NVLink
unidirectional roof) regardless of fragmentation, idle or under concurrent
memory-bound compute (immune up to 4 blocks/SM). GB10: up to 76.8× at 2048
runs. Prepared plan: 0.158 ms prepare once, 206.5 µs/execute, 4.2 µs host
enqueue.

**p2p_kv_donate (H100 NVL):** same-device D2D prepared plan: 6.0 µs/layer
(1 page) to 36.9 µs/layer (48 MiB) = 1.37 TB/s payload / 2.73 TB/s touched.
The binding roof is a copy, not the HBM read peak: a pure D2D streaming
copy kernel measures 34.5 µs (1.46 TB/s payload) on the same box, so the
plan is ~94% of the copy roof (the remaining gap is the two-stream K/V
gather + slot indirection). "~40% of HBM" counted payload against the
read-only peak. Over an idle NVLink pair the plan is 207.6 µs = 100% of
the measured 206.7 µs peer copy-kernel roof. The page grid-stride (#84)
removed a latent `num_pages > 65535` truncation. One-shot fused beats the
two-stage gather+copy by 4–48×.

**pipeline_boundary:** peer copy saturates at ~265 GB/s (44% of the
600 GB/s bidirectional NVLink roof); captured graph replay is
payload-independent at ~3.3 µs with zero host enqueue (1.6–1.7× host-side
vs eager).

**Cross-node KV (GH200, JSC IB HDR):** 2-rank point-to-point caps at
~24 GB/s = ~96% of one HDR-200 port (measured), vs 3.3 TB/s same-node HBM
ceiling — the all-gather draft targets the 4-port ~100 GB/s aggregate
(unmeasured hypothesis).

**RCCL/OFI transport:** cost model verified host-side (OFI 3.75×/3.04× vs
Socket at 1/16 MiB); the on-hardware cross-node A/B over Slingshot needs a
multi-node reservation — model-only today.

### 3.7 End-to-end serving data points

| Workload | HW | Metric | Eager | Breakable cudagraph |
|---|---|---|---:|---:|
| Kimi-K3 PP3/TP8 (dummy) | 6× MI300A | agg tok/s | 25.3 | **35.2 (1.39×)** |
| | | p50 latency | 92.0 s | 62.5 s |
| GLM-5.3 fp8 real ckpt | MI300A | prefill tok/s | 163.8 (bf16 287.9 is the fp8 arm) | — |

The K3 win is entirely the captured non-MoE graph; the MoE region is
unchanged and gated by the PP0 `topk_ids.cpu()` host sync (97–100% of its
MoE step) — on-device `moe_align_block_size` is the named follow-up. The
GLM fp8 grouped-GEMM NaN was a silently-unwritten-output-rows tile-map bug
(fixed, 73bb30d; 0/16.7M mismatches post-fix).

---

## 4. Kernels without measured performance records

Present, correct (CPU-oracle-validated, many with GPU test harnesses), but
**no benchmark numbers recorded yet**:

| Kernel / primitive | Backend | Status |
|---|---|---|
| `add` / `scale` / `relu`, `sum` / `max`, `gemm` (fp32 SGEMM) | CUDA | tier-1 infrastructure; no perf doc (A100 ridge math only in `gemm.md`) |
| `kda_delta_rule_fwd_chunked*` (WY chunked, #70) | HIP | correctness harness `test_kda_chunked.hip`; perf via `bench_kda_chunked.hip` not recorded here |
| `dsa_topk_transform` (pool-level top-k, #58) | HIP | correctness only in this table's sources |
| `moe_align_block_size` | CPU today | device kernel is the top PP-pipeline follow-up |
| `kv_gather_layer` / `kv_scatter_layer` (#1/#2) | CUDA | benches exist (`kv_gather_bench.cu`), no perf doc |
| `p2p_kv_restore` (fused + plan, #27) | CUDA | roofline constants (88.5 GB/s fused restore on sgs-gpu07) quoted via the donate/cross-node docs; no dedicated perf record |
| ring `allreduce`, `OverlapExecutor`, `Channel`/topology | host | correctness surface; overlap behavior exercised in tests |
| `dist` MoE (TP/EP/PP) | host + HIP correctness harness | matched-oracle on device; throughput belongs to the serving-level records above |
| cross-node KV **all-gather** (#49 draft) | CUDA | implemented, multi-node perf unmeasured (hypothesis ~100 GB/s) |
| RCCL OFI/CXI net plugin | HIP build | build + host-verified on beverin; transport hooks skeleton, no hardware A/B |

---

## 5. Caveats that apply across the table

- **Clocks unpinned on MI300A clusters** (`rocm-smi --setperflevel`
  denied): absolute µs there are upper bounds; ratios/scalings are the
  robust signals. MI300A decode timings vary ±30% node-to-node (APU clock
  throttling).
- **Timing floors:** MI300A `hipEvent` resolution ~0.5 µs + DVFS quirks
  (zero-sample discard now built into all harnesses; sub-µs kernels are
  batched 2000 launches/sample). GB10/CUDA-13 has a rapid-relaunch
  `elapsed=0` glitch (capped at 80 iters).
- **Byte models:** DSA-family GB/s uses the repo convention (kv counted
  once per head per pass); the dsa doc explains the 430-vs-860 GB/s
  bookkeeping difference vs FlyDSL's model. L2-resident benches (moe_aux,
  dsa-topk) report L2 bandwidth, not HBM.
- **Baselines:** tilelang/Triton baselines are unavailable on gfx942 by
  construction (GPU-fault / JIT-abort — the reason vkernels exists), so
  "delta to baseline" is n/a there; MoE baselines are torch loops; the
  honest per-kernel comparison is the roofline column.
- **One GCD on MI250X**; peer-write H100 numbers taken under contention;
  GB10 perf docs predate the adaptive p2p-gather dispatch (re-run to
  refresh).

## 6. Provenance / how to update

Every row's "Record" link contains the reproduce script and raw log. To add
a row: benchmark on target HW per
[`docs/performance/README.md`](performance/README.md) (one directory per
kernel, one file per arch), then update the master table here with the
headline number, the binding roof, and the % gap.
