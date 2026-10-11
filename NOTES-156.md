# Issue #156 — MoE decode expert-GEMV bandwidth (MI300A) — work log (v2)

Lane branch: `issue-156/gemv-bw`. Prior delivery: `7bf4773` on main (v1
`gemv_decode_{bf16,fp8_block}_splitk`, ceiling ~1.6-1.7 TB/s at tiny-M,
instruction/latency-bound — see the issue comment and the NOTES in that
commit). This lane implements the structural follow-up sketched there.

## v2 design (commit 9a9555b)

`gemv_decode_{bf16,fp8}_splitk_v2` in `src/c/vkernels/kernels/gemm_bf16.{hip,hpp}`:

1. **TB fixed at 8 (exact uint4).** The v1 default config tb=4 issued a
   16 B uint4 load per thread per K row but consumed only 8 B (4 columns),
   so every B byte was requested by FOUR threads; wavefronts overlapped
   4x on the same lines. v2 gives each thread 8 columns; a 64-thread
   wavefront covers 1 KiB of B contiguously with zero overlap.
2. **Multi-K register blocking (KK ∈ {2,4,8}).** KK independent in-flight
   uint4 B loads per thread per iteration, KK·8·M FMAs amortise the loop.
3. **Optional `a_gmem` A path.** A[m][k] read straight from global memory
   (uniform across threads; the split's slice is M·kl·2 B, L1-resident) —
   removes the LDS stage and its `__syncthreads`.
4. **Deferred-scale fp8 (new contract).** Raw fnuz codes are accumulated
   UNSCALED (uniform value·2^-119 domain of the branchless bit decode) and
   each (128-K × 8-N) block scale is applied ONCE at the block boundary.
   Removes the per-element `f2bf` bf16 rounding of the v1/MFMA contract.
   The v1 per-element-rounded path is kept unchanged. Validation: exact
   fp64 deferred oracle (1e-3 gate) + bf16-rounded dequant oracle (2e-2).

Same fp32 partial planes + fixed ascending-S combine as v1 ⇒ combine-side
numerics contract unchanged; prefill V5 path untouched.

## Probe / tests

- `probe_issue156_batched.cpp` phase `v2`: sweep (S, th, kk, a_gmem) on
  N=6288/3584/896 K=7168 M=5, bf16 + fp8-deferred; batched 1000-launch
  event pairs × 4 batches + checksums + per-row CPU-oracle gates.
- `test_gemm_bf16_correct.hip` section (6b): 11 v2 cases (odd/prime S,
  odd K=257 tails, beta≠0, a_gmem, fp8-deferred incl. splits spanning
  scale blocks) + M=9 / kk=3 rejection (C untouched).

## Job 676517 (beverin mi300, --exclusive -N1)

- Login-node build: lib + test + moe benches OK (ROCm 6.3.4, gfx942);
  probe156 linked with `--offload-arch=gfx942` (the gfx90a silent-default
  footgun from the v1 notes is still live).
- Gates / probe / prefill V5 / moe_aux results: TBD (append below).

## Results

(append when job 676517 completes)
