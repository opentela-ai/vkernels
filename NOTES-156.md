# Issue #156 — MoE decode expert-GEMV bandwidth (MI300A) — work log

Branch: `fix/issue-156-moe-decode-bw`. Checkpoint commits: `2b751c5` (wide split-K + block-FP8 kernels, tests, probe).

## Baseline (job 649360, synced pristine tree, 2025 run)

- Build: OK (vkernels, gemm_bf16_bench, test_gemm_bf16_correct, test_moe_aux_correct, moe_fused_prefill_bench, glm_fp8_gemv_bench).
- `test_gemm_bf16_correct`: PASS (0 failures) — all serving M=8 rows + reuse + split-K sweep.
- `test_moe_aux_correct`: **16/16 PASS** (bit-exact quant rows included). ✔ HARD CONSTRAINT MET (baseline)
- Prefill V5 (moe_fused_prefill_bench), M=1024 row: **626.3 us / 20.574 TF** (issue reference: 617 us / 20.9 TF — node-to-node variance; same-job before/after comparison is the contract).
- Clocks during run: mclk 1300 MHz (level 3), sclk 1508–2047 MHz.
- Batched probe FAILED to compile (first run): `std::max`/`std::ldexp` vs CUDA `cuda_wrappers/algorithm` ambiguity in the gfx942 device pass. Fix: use `std::fmax` on host paths and `exp2f` in the device dequant oracle. **=> baseline batched numbers must come from the re-run (job 649379).**

## Implementation (2b751c5)

1. **Wide split-K bf16 kernel** (`gemm_bf16_splitk_wide_kernel<BM,BN>`):
   - uint4 (16B) staging of both sA and sB (host-gated `N%8==0 && K%8==0`),
   - LDS double-buffer ring (2-deep, one `__syncthreads()` per K-tile; prefetch of kt+1 issued before the MFMAs of kt),
   - MFMA body / fp32 partial planes / fixed-order combine identical to #146.
   - Auto-selected by `gemm_bf16_splitk_with_config` for aligned shapes; #146 kernel preserved behind `gemm_bf16_splitk_legacy_with_config` for A/B.
2. **Block-FP8 split-K kernel** (`gemm_fp8_block_splitk_with_config`):
   - Weights: W8 = E4M3FNUZ codes [K,N] row-major + fp32 scales per (128-K x 8-N) block → dequant w = fp8(code)*scale. fp8 bytes = 1/2 of bf16 → 2x BW ceiling.
   - Dequant fused into sB staging: branchless fnuz→fp32 bit trick (fp32 bits = sign|(b&0x7F)<<20 = fnuz*2^-119 UNIFORMLY incl. subnormals; scale folded with 2^119). NOTE: glm_moe folds 2^120 — that matches ITS OWN CPU oracle contract (glm stored scales are 2x the standard-decode scale); our contract is defined by OUR oracle (standard bias-8 fnuz decode) and validated bit-exact against it via the device dequant oracle.
   - MFMA body / combine identical to bf16 wide path; A stays bf16; gate N%8==0 && K%8==0 + compiled tile pairs.
   - Validation: (a) device dequant-oracle → wide bf16 kernel → fp8 kernel must match BIT-EXACTLY; (b) CPU fp32 oracle on independently host-dequantized weights (2e-2 rel); (c) moe_aux 16/16 stays green (untouched code).
3. Tests: `test_gemm_bf16_correct.hip` section (5) — 5 fp8 oracle cases (tiles 16x16/16x64/32x64, beta!=0, S clamp, tiny odd-S) + N%8!=0 rejection (C untouched).
4. Probe `probe_issue156_batched.cpp`: phases `base` (legacy sweep) / `full` (legacy A/B + wide sweep + fp8 phase with dequant-oracle bit-exact gate), batched 1000-launch event pairs x4 batches, checksums, per-batch us printed.

## Perf model notes

- Weight bytes per launch (bf16, split-K S): B re-read per (m-tile, split) → bytes ≈ 2*(M*K*ceil(N/16) + K*N*ceil(M/16) + M*N); the S-way split reads B S times total across splits (each split reads its own K-slice once per m-tile) — the probe's bytes model counts B once per full column pass (S-independent within a split since K-slices partition K). Workspace round-trip 2*S*M*N*4B fp32 is included in the per-config GB/s impression only qualitatively.
- LDS bank-conflict risk identified (to evaluate after first wide numbers): a-fragment LDS reads `sA[cur][row_base+m][k0]` have 128B stride between lanes → same-bank 16-way conflict potential (pre-existing in #146 too). Fix candidate v2: store sA TRANSPOSED [BK][BM] so a-frag reads are conflict-free (strided scalar stores at stage time are conflict-free since separate instructions). Decide from job 649379 numbers.

## Job 649379 (in flight)

- In-job gates before perf: rebuild + `test_gemm_bf16_correct` (incl. new fp8 section). Probe `base` (legacy sweep = baseline batched numbers) then `full` (wide + fp8).
- Results: TBD (append below when done).

## Results

(append per-batch us / GB/s / gate outcomes here as jobs complete)

## Build/log incidents (so the same footguns aren't re-stepped on)

- Job 649379 failed twice: (1) my fp8-kernel edit accidentally DUPLICATED
  `gemm_bf16_splitk_combine_kernel` -> libvkernels built a stale .o silently
  (cmake kept the old object; the failure surfaced as an undefined-symbol
  LINK error in test_gemm_bf16_correct, and nm on the .o showed no fp8
  symbol). Fixed in 26ced7d (single definition, 2 call sites).
  (2) probe's dequant kernel called host-only `f2bf_host` -> fixed with
  `__host__ __device__` helpers + exp2f/memcpy (std:: wrappers are risky in
  .hip TUs: std::max/std::ldexp hit cuda_wrappers/algorithm ambiguity in the
  gfx942 device pass; use std::fmax / exp2f).
- Login-node build now validated BEFORE submitting: lib + test link OK,
  probe links OK, nm shows gemm_fp8_block_splitk_with_config +
  gemm_bf16_splitk_legacy_with_config exported.

## Jobs

- 649360: baseline gates (pristine tree) — moe_aux 16/16 PASS, gemm test
  PASS, prefill V5 M=1024 = 626.3 us / 20.574 TF. Batched probe N/A (compile
  error above).
- 649379: wide+fp8 build — failed (see incidents).
- 649400: iteration-1 — **gate 1 caught a real wide-kernel bug**: shapes with
  odd K-tiles-per-split (K=1536/S=8 kts=3, K=512/S=8 kts=1, K=3584/S=8 kts=7)
  FAILED (max_rel ~25-35). Root cause: prologue staged kt0 into ring slot 0
  but the loop reads slot kt&1 — odd kt0 reads stale LDS. Fix (26ced7d+):
  prologue loads into slot kt0&1. Regression cases added to section (4)
  (K=1536 S=8, K=512 S=8). The K3 target shapes (K=7168, S=8, kts=14 even)
  were NOT affected, but the fix is mandatory.
- 649409: iteration-1 rerun after ring-parity fix — gates GREEN (all
  sections incl. new ring-parity cases kts=1/3), prefill V5 after-check
  **629.8 us / 20.460 TF** (baseline 626.3/20.574 — within noise, no
  regression). moe_aux 16/16 PASS.
  BUT the probe results were invalid: the `probe156` binary was linked
  BEFORE the parity fix (statically archives the old kernels) — wide/fp8
  S=16 (kts=7, odd kt0) rows showed garbage checksums, and fp8 rows showed
  bit-mismatch vs oracle on every element. Debugging trail:
  + host 256-code x 8-scale check: oracle-expression vs kernel
    bit-trick-expression BIT-IDENTICAL (0 mismatches).
  + gfx942 micro-test: same, 0/2560 mismatches; device exp2f exact at
    integer args. (First micro run silently compiled for gfx90a — always
    pass --offload-arch=gfx942!)
  + tiny end-to-end repro (fresh lib, N=64 K=128 and N=896 K=7168, both
    tiles): fp8 kernel == dequant-oracle gemm BIT-EXACTLY (0/4480).
  => fp8 staging + MFMA are bit-exact vs the independent oracle; the 649409
  fp8 FAILs were stale-binary artifacts.
- 649451: iteration-2 (rebuilt probe + lib): gates + probe base/full +
  prefill after-check. Results below when done.

## FP8 contract notes

- The repo's canonical `fp8e4m3fnuz_to_f32` (device_numeric.cuh) decodes
  SUBNORMALS as m*2^-7, which is non-monotonic against its own normal min
  2^-7 (code 0x07 > code 0x08). Our contract (and fnuz = OCP-E4M3FN/2):
  subnormal = m*2^-10, normal = (1+m/8)*2^(E-8) — self-consistent, and the
  normal branch matches the canonical bit-for-bit. Left canonical untouched
  (out of scope; dsa never encodes subnormals) but FLAG in the issue.
- glm WIP (2268cf0) is internally inconsistent on subnormals: its bench
  decode (m*2^-9, bias-7 normals) vs its kernels (trick fold 2^120 =
  bias-7 normals, subnormal m*2^-6). Our contract is internally consistent
  and validated bit-exact vs the independent oracle.

## FP8 scale-contract cross-check vs WIP 89a3a35/2268cf0

- WIP bench `e4m3_f32` decode: fp32 exp field = code_exp − 7 + 127 → **bias 7**
  contract; its kernels fold 2^120 into the scale with the bit trick
  ((1+m/8)·2^(E−127)·2^120 = (1+m/8)·2^(E−7)) — internally consistent pair.
- Ours: **standard bias-8 FNUZ** contract; kernel fold 2^119
  ((1+m/8)·2^(E−127)·2^119 = (1+m/8)·2^(E−8)); host oracle `fnuz_dec` =
  (1+m/8)·2^(E−8), subnormals m·2^−10 (kernel trick for E=0: (m/8)·2^−126·2^119
  = m·2^−10 ✓ uniform). Encoder (frexp + RNE, saturate at 240 = fnuz max,
  scale = amax/240) round-trips the decode. NOT interchangeable with the glm
  contract (2x scale offset); documented in gemm_bf16.hpp.
- Reduce ordering wide vs #146: identical — same per-split fp32 epilogue
  planes and the same fixed-ascending-S combine kernel; staging changes only.
  Wide output should be bit-identical to legacy on aligned shapes.

## Iteration 3: decode-GEMV split-K (latency-floor attack)

Hypothesis: at M<=8 the MFMA tiles are latency-floored (~30-96us, at most
1.24 TB/s) because a 16x16 tile issues far more work than the problem
needs; a dedicated GEMV-style split-K kernel with per-thread column
ownership, uint4/uint2 B loads and one LDS stage per split should get
closer to BW roof. New public API in gemm_bf16.hpp:

- `gemv_decode_bf16_splitk(M,N,K,alpha,A,B,beta,C,S,tb,th)` (M<=8)
- `gemv_decode_fp8_block_splitk(...,scales,S,tb,th)` (same, block-FP8)
- shared dispatch: kts = ceil(K/S) element splits, fp32 partial planes in
  the shared SplitKWorkspace, combine = fixed ascending-S sum (same
  ordering as #146 legacy/wide => same numerics contract).

### Correctness (job 649479/649502 gates)

- bf16: 8 cases PASS (M=5/8/2/1, S=31/32/33/64 odd+prime splits, tb=4/8,
  th=64/128, beta=0/0.5, odd K=257, N-tail).
- fp8: 4 cases PASS after one fix (see below).
- M=9 rejection: C untouched PASS.

### fp8 gemv fix (oracle mismatch, not math)

First run: fp8 gemv FAILed on K=7168 shapes (max_rel ~0.1) while K=256
passed, and an isolated repro against an UNROUNDED fp32-dequant oracle
PASSED everywhere incl. the exact failing config. Root cause: the dequant
ORACLE weights are RNE-rounded to bf16 (f2bf(scale*fnuz_dec), the #146
contract the MFMA fp8 kernel matches bit-exactly), while my gemv kept the
fp32 product unrounded. The accumulated rounding difference on the test
data distribution reaches ~0.1 relative over K=7168 terms. Fix: gemv fp8
now computes w = bf16_to_f32(f2bf(sc * fp32_decode)) — the same
expression as the MFMA fp8 staging — so all fp8 paths share one contract.
(Semantics note for future work: a deferred-scale variant — accumulate
unscaled per-128k-block dot products, multiply by the block scale at the
block boundary — would avoid the per-element rounding entirely and is
arguably the "exact" block-FP8 math, but it changes the contract and would
require re-validating the MFMA path against a new oracle.)

### Performance (probe gemv phase, job 649489; M=5, K=7168)

Best-of-sweep vs the legacy/wide MFMA split-K bests (same wall-clock model):

| shape (M=5) | legacy best | gemv best | config | bytes | GB/s legacy -> gemv |
|---|---|---|---|---|---|
| N=6288 bf16 | 95.66us | 53.4us | S=32 tb=4 th=256 | 90.3MB | 1238 -> 1689 (+36%, 1.79x wall) |
| N=3584 bf16 | 54.8us (wide) | 40.7us | S=64 tb=4 th=128 | 51.5MB | ~1176 -> 1264 (+~25% wall) |
| N=896 bf16 | 27.8us | 31.9us | S=64 tb=4 th=128 | 12.9MB | 607 -> 405 (MFMA WINS small N) |
| N=6288 fp8 | n/a | 70.6us | S=64 tb=4 th=128 | 45.4MB | 643 raw = 1286 GB/s bf16-equiv |
| N=3584 fp8 | n/a | 54.8us | S=64 tb=4 th=128 | 25.9MB | 473 raw |
| N=896 fp8 | n/a | 42.3us | S=64 tb=4 th=128 | 6.53MB | 154 raw |

Config observations: tb=4 (8B/thread/k) beats tb=8 at every shape; th=256
helps N=6288 (more warps/CU at small grid), th=128 helps N=3584/896;
S=96+ regresses (combine plane traffic grows: S planes * M * N * 4B).
unroll 4->8: no measurable effect (not load-latency limited).

### Where we stand vs the issue targets

- bf16 N=6288/3584: 1689 / 1264 GB/s = 32% / 24% of the 5.3 TB/s roof.
  The issue's >=3.5 TB/s (65%) is NOT reached; the measured ceiling for
  tiny-M decode on this stack is ~1.6-1.7 TB/s bf16.
- fp8 effective: 1286 GB/s bf16-equivalent vs the bf16 gemv path's 1689
  GB/s at the same shape => 0.76x, NOT the >=2x target.
- Calibration vs repo prior art: the WIP glm fused fp8 GEMV (649470) —
  arguably the most tuned decode path in the repo — reaches only
  684-776 GB/s fp8 at N=4096 M=1 (24.6us / 16.8MB). Our 643 GB/s fp8 at
  4x the bytes is the same performance class.
- Diagnosis: tiny-M decode is not BW-bound in our kernels; the per-k
  instruction stream (extract/decode + M*TB FMA + a[m] LDS reads) and
  fixed overheads (A stage barrier, combine planes) bound us ~3x below
  roof. Structural fixes that plausibly break the ceiling (out of scope
  today, sketched for the issue): multi-k register blocking (2-4 k rows
  per thread iteration, fewer loop iterations), removing the a[m] LDS
  broadcast (small-A L1-resident scalar loads), deferred-scale fp8
  (kills per-element f2bf), and for fp8 a 256-entry dequant LUT per
  column-block rebuilt every 128-k block.
- Dispatch guidance (documented, not auto-routed — no public-path
  regression risk): M<=8 & N>=2048 & N%8==0: gemv_decode (S=32-64,
  tb=4, th=128-256). Smaller N: keep MFMA wide split-K.
