# NOTES-fusion-dsa-topk-tailfold — the deferred candidate (i): cooperative logits→transform fold

Continuation of NOTES-fusion-dsa-kda-launches.md, candidate (i), which was
assessed and deferred: folding `dsa_topk_transform` into the logits tail
needs a grid-wide barrier because the logits kernels split one score row
across `split_kv` blocks while the transform needs the COMPLETE row. This
pass implements the recommended design: a cooperative-launch variant where
the transform runs as phase 2 of the SAME grid after `grid.sync()`, gated
`VK_DSA_TOPK_FUSED=1` DEFAULT OFF.

## Shim feasibility (probed for real, nvcc 13, GB10)

- `dsa_topk.hip` compiles CLEAN under the HIP-on-NVIDIA shim: the scalar
  logits arms are vendor-intrinsic-free. Added to
  `VKERNELS_CUDA_HIP_SOURCES` (+1 line) behind
  `VKERNELS_DSA_TOPK_CUDA_SHIM`; the fused path is GB10-validatable.
- `dsa.hip` is a DOCUMENTED NEGATIVE — do not add it to the shim list:
  3 trivially-mappable API symbols (`hipGetDevice`,
  `hipDeviceGetAttribute`, `hipDeviceAttributeMultiprocessorCount` — all
  now mapped in the shim header anyway) plus 2 AMD-only MFMA builtins
  (`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`,
  `__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8`) with no CUDA mapping short
  of an mma.sync rewrite. Consequence: the fused path covers the AUTO
  dispatcher's SCALAR arms only (`q_variant` 0/1/2); the MFMA fast paths
  stay unfused. On gfx942 H=32 picks fp32-Q and H=64 fp8-Q (64 KB
  non-optin cap); on GB10 both fit fp32-Q (101,376 B opt-in ceiling).

## Co-residency (the make-or-break question)

Cooperative launch requires every block co-resident. Analysis:

- The AUTO dispatcher's serving geometry: grid = (batch, split_kv), one
  1024-thread block per (row, split). At 1024 threads/block:
  gfx942 = 1 block/CU (2048 threads/SM budget vs 1024 used → 1 at the
  64 KB LDS cap), GB10 = 1 block/SM (1536 threads/SM).
- MI300A: the serving split formula (`dsa_topk_logits_split_for`,
  NUM_CU=228) already keeps batch·split_kv ≤ 228 → the whole serving
  matrix is co-resident BY CONSTRUCTION. No clamp needed in practice.
- GB10: 48 SMs; the launcher clamps split_kv to floor(capacity/batch)
  when batch·split_kv > capacity, and refuses (falls back) when even
  batch > capacity. The clamp is legal because the grouped logit is
  grouping-independent (pinned by test_dsa_topk_correct case split=2);
  perf-only parameter.
- Runtime guards in `launch_logits_transform_coop`: occupancy query
  (`hipOccupancyMaxActiveBlocksPerMultiprocessor`), cooperative-launch
  device attribute, capacity check, then the launch itself — ANY failure
  silently runs the proven two-launch chain.

## Change

Additive, env-gated, default = proven chain (per-call getenv,
`VK_MHC_PRE_STRICT` pattern, no static caching):

- `dsa_topk_device.cuh` (+262): the radix machinery + scalar logits
  kernels moved VERBATIM from `dsa_topk.hip` (only `static` linkage added;
  verified body-identical: coarse_float_key, ordered_float_key,
  radix_topk, transform_token) — one definition shared by the standalone
  kernel, the fused tail, and the CUDA TU dsa.cu.
- `dsa_topk.hip` (+476 net): `dsa_topk_logits_transform_coop_kernel<K,
  kFp8Q>` — phase 1 scalar logits row slices, `cg::this_grid().sync()`,
  phase 2 (`blockIdx.y == 0` blocks) the verbatim transform row body;
  `launch_logits_transform_coop` with the guard stack above;
  `dsa_topk_logits_transform_fused` public entry (signature = the
  two-launch chain's arguments plus `q_variant`) that IS the proven chain
  when the gate is off — callers can wire it unconditionally.
  `logits` doubles as the transform's score buffer (same
  `score_stride == max_seq_len` contract the two-launch chain has).
- Shim header: additive mappings (coop launch, occupancy,
  `hipFuncSetAttribute` dynamic-LDS opt-in, device attributes). Real HIP
  needs no dynamic-LDS opt-in (gfx942 64 KB cap is the ceiling — KB note
  mi300a-dynamic-lds-no-optin); the CUDA build queries the opt-in
  ceiling, mirroring dsa.cu.
- `dsa_topk.hpp`: contract comment on the fused entry (comment-only).
- Test: `tests/kernels/attn/test_dsa_topk_fused_gpu.cpp` (self-skips
  without the shim build).

## Validation (GB10, HIP-on-NVIDIA shim)

- Byte-identity gates: fused logits memcmp-equal to the proven chain over
  the WHOLE buffer (canary cells included — both runs memset to the same
  pattern, kernels write only `t < seq_len[b]`); transform rows
  set-identical after canonical sort; BOTH paths bit-exact (memcmp)
  against the CPU oracle's transform.
- Shape matrix mt=8…64 × split=2…4 clean; paged and ragged-offset remap
  variants covered; unsupported group_topk falls back silently and stays
  bit-identical.
- Launch-overhead probe (print-only) at GLM decode geometry (bs=2 H=32
  D=128 B=64 mt=64 seq=4096): gate-off 2944 µs vs fused 3086 µs per
  iteration. Launch-neutral on GB10 — per-iteration sync dominates, and a
  cooperative launch carries its own setup cost. The A/B target is
  MI300A, where per-launch overhead is the quantity being removed.
- ctest: cuda 57/57, host 47/47 (host build exercises the oracle paths;
  the GPU test self-skips there by design).

## Incidents (for the record)

1. First worker run died mid-implementation (clean exit, no commit) —
   recovered the findings from its transcript and relaunched with them
   front-loaded.
2. A 4-minute hang during debug-repro linking: nvcc was fed
   `libvkernels.a` as a SOURCE file (gcc -E on a binary archive). Killed
   the process, steered the correct link form. The standalone repro was
   then abandoned anyway — device-linking the full archive under nvcc
   hung — and the probe went into the already-linked test harness.
3. A compute-sanitizer OOB that looked like a kernel bug was the probe's
   own bug: `seq_lens` built as `std::vector<float>` and bit-copied into
   the int32 buffer (512.0f = 0x44000000 → seq_len 1,140,850,688 →
   ~35 MB past the page table, exactly the sanitizer's offset). Caught
   with an in-kernel debug print, fixed in the test, scaffold removed.

## Pending (for the MI300A lane)

- A/B on gfx942: `VK_DSA_TOPK_FUSED=0` vs `=1` on the serving matrix,
  decode-step-dominated workload (the launch saving is per step per
  layer). Repro: `meta/benchmarks/bench_dsa_topk.hip` extended to the
  fused entry, `srun --partition=mi300`.
- If the A/B wins, consider wiring the fused entry into the AUTO
  dispatcher behind the same gate so the MFMA arms can keep their
  two-launch chain while scalar shapes fuse.
- Risks: cooperative launch occupancy is sensitive to future shared-memory
  growth in the scalar arms (a silent fallback, never a wrong answer —
  the fallback contract makes degradation benign); CUDA 13 coop-launch
  semantics verified on sm_121 only.

## Commits

This worklog accompanies the `fusion/dsa-topk-tailfold` change: dsa_topk
fused cooperative chain (VK_DSA_TOPK_FUSED=1 default off), verbatim
shared-kernel extraction, shim coop mappings, GPU byte-identity suite,
docs/kernels/dsa_topk.md fused-path section.
