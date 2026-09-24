# NOTES — fusion candidate rank 3: fold the split-K combine into the gemm_bf16 launch

Branch: `fusion/gemm-splitk-combine` (managed worktree off main HEAD).
Scope touched: `src/c/vkernels/kernels/gemm_bf16.{hpp,hip}`,
`meta/benchmarks/test_gemm_bf16_correct.hip`, `docs/kernels/gemm_bf16.md`.
No shared files (top-level CMakeLists, CAPI registry, python `__init__`)
were touched — **no merge-conflict surface with the four sibling lanes.**

## Baseline (from the docs, not re-measured here)

- docs/kernels-reference.md §3.1 / row 1: QKV `6288×7168` split-K `S=8`
  (issue #146 two-kernel path) at `M=5` = **99.6 us / 1189 GB/s (22% HBM)**;
  at `M=64` = **244.8 us / 2949 GB/s = 56% of HBM** — the best serving row
  in the master table. The named remaining gap: **"the workspace round-trip
  plus the exposed load latency"** (this candidate) and the AI-bound
  ceiling (cross-tile B reuse, out of scope here).
- docs/performance/gemm-bf16/gfx942.md: `gemm_bf16_splitk_kernel<16,16>` +
  `gemm_bf16_splitk_combine_kernel`; fp32 partial planes `P[S][M][N]` in
  the grow-only workspace (mla_fwd #82 pattern), combined in **fixed
  ascending-s order** (deterministic). S sweep at M=5: `S=2` 181 us,
  `S=4` 111.6 us, `S=8` 100.1 us — the workspace round-trip "stays ~us",
  i.e. the split-K win is latency, not bandwidth-limited. Small shapes:
  M=5 N=2304 K=1536 = 15.6 us, M=5 N=3072 K=512 = **8.3 us** (two launches
  + a grid drain for ~1-2 MB of combine work).
- Cost structure of the two-kernel path per launch:
  1. split kernel writes `S*M*N*4` B of fp32 partials (<= 12.9 MB at
     M=64 QKV) with the MFMA fragment store pattern;
  2. full grid drain (device-wide serialization between the kernels);
  3. combine kernel re-reads the same `S*M*N*4` B fully coalesced, applies
     alpha/beta, single RNE store (~5 us of DRAM at roof at M=64, plus one
     kernel-launch gap ~3-5 us).

## What changed (all additive; default behavior bit-identical)

1. **`gemm_bf16_splitk_fused_with_config`** (new entry, gemm_bf16.hip +
   public header decl): single-launch split-K. Same grid, same WIDE
   staging, same fp32 partial planes as `gemm_bf16_splitk_with_config`.
   The combine is folded into the split kernel:
   - after storing its partial plane, each block does `__threadfence()`
     then `atomicAdd` on the arrival counter of its `(m, n)` output tile
     (canonical threadFenceReduction pattern => the last arriver observes
     every other split's plane stores);
   - the **last-arriving** block for the tile sums all `S` planes for that
     tile in the **SAME fixed ascending-s order** as
     `gemm_bf16_splitk_combine_kernel` (`sum += P[s][row][col]` per
     element), applies alpha / `beta*C` once, single RNE bf16 store.
     Identical fp32 addition order over identical plane values =>
     **bit-exact with the two-kernel path**, a strictly stronger gate than
     the 2e-2 oracle tolerance.
   - the combining block resets its tile counter to 0 (`atomicExch`), so
     no per-launch memset is needed; counters live in the same grow-only
     `SplitKWorkspace` (extended with `counters/counter_cap`), zeroed at
     (re)allocation.
2. **Kill-switch `VK_GEMM_SPLITK_FUSED`** (default **0** = the proven
   two-kernel path): when `=1`, `gemm_bf16_splitk_with_config` routes
   WIDE-eligible shapes (`N%8==0 && K%8==0`) to the fused entry. The
   explicit fused entry is always callable for probes. Unaligned shapes
   and uncompiled tiles fall back exactly as before.
3. The wide kernel (`gemm_bf16_splitk_wide_kernel`) gained a
   `bool Fused = false` template parameter (default preserves every
   existing instantiation/call site — one definition, no duplicated symbol,
   per the NOTES-156 incident) and 4 extra launch args. The #146 legacy
   kernel, the fp8 split-K kernel, and the decode-GEMV kernels are
   untouched (fp8/decode paths keep the separate combine).
4. Tests: new section (4b) in `meta/benchmarks/test_gemm_bf16_correct.hip`
   — same case matrix as the split-K section (4): odd S, K=128
   minimum-split corner, S clamping, beta != 0, unaligned N fallback, the
   ring-parity cases (kts=1/3/7). Per case: **bit-exact vs the two-kernel
   path** AND the fused path run TWICE back-to-back (proves the counters
   self-reset between launches) AND oracle tolerance `max_rel < 2e-2`.

## Validated HERE (GB10 box) vs NOT validated

- Host build (build/host): **45/45 ctest PASS** (incl.
  `vkernels_test_gemm_bf16`, the CPU-oracle suite — oracle untouched).
- CUDA build (build/cuda, nvcc 13.0, sm_121): **55/55 ctest PASS**. The
  CUDA port has no split-K path, so this validates only that the shared
  `gemm_bf16.hpp/.cpp` changes compile and nothing regressed on the
  NVIDIA serving/warmup path.
- **The HIP path is UNVALIDATED on this box: there is no HIP toolchain
  here; `gemm_bf16.hip` (the fused kernel, the fence/atomic epilogue, and
  the device test section) has NOT been compiled or run on gfx942.** The
  static checks done here: brace/paren balance, unique-definition audit
  (3 references to the fused entry in the .hip = fwd-decl + env-routed
  call + definition; no duplicated kernel symbols), and the param/
  template-default review of the single `launch_splitk` call site.
  First MI300A job MUST run the correctness gate before any perf number.

## MI300A A/B plan (exact commands)

```bash
# login-node build gate (compile catches what this box cannot):
cmake --preset hip && cmake --build --preset hip -j
nm build/hip/lib*/libvkernels* | grep splitk_fused   # symbol exported, once

# 1) correctness gate FIRST (fused section (4b) + all prior sections):
srun --partition=mi300 -N1 -G1 ./build/hip/meta/benchmarks/test_gemm_bf16_correct
#    section (4b) must print bit_exact=1 on every case (2 reps each).

# 2) same-job A/B (probe_issue156_batched methodology; 20 warmup +
#    1000-launch event pairs x4 batches, per-batch us + checksums):
#    phase "base" rows = gemm_bf16_splitk_with_config (two-kernel);
#    new phase "fused" rows = gemm_bf16_splitk_fused_with_config, same
#    shapes (K3 serving, M=5/64), tiles (16,16), S in {4,8,16}.
#    Gate: bit-exact checksums fused vs two-kernel before quoting speedups.
#    Env-flag form (public path): A: unset VK_GEMM_SPLITK_FUSED,
#    B: VK_GEMM_SPLITK_FUSED=1, same binary, interleaved batches.

# 3) bench_gemm_bf16.hip split-K sweep with VK_GEMM_SPLITK_FUSED=1 vs 0
#    for the full S sweep at M=5/64 (regenerates the S-sweep table).
```

## Expected win (byte/roofline model, QKV 6288x7168, S=8, tile (16,16))

Traffic per launch is UNCHANGED (the partial-plane write `S*M*N*4` and the
combine read `S*M*N*4` both remain; 12.9 MB each at M=64, 2.0 MB each at
M=5). What is removed:

- one kernel launch + the inter-kernel device drain: ~3-5 us fixed;
- the tail serialization: the two-kernel combine cannot start until the
  LAST split block of the LAST tile finishes; the fused path combines each
  tile on the heels of its own last split, overlapping the combine reads
  with the remaining blocks' MFMA work.
- net add: the fused combine reads planes at fragment granularity (16
  lanes x 4 B = 64 B contiguous segments per read instruction vs the
  combine kernel's fully coalesced 256-column sweeps) — worst case ~4x
  read-instruction inflation on <= 12.9 MB, i.e. a few us if the tail
  wave is occupancy-thin.

Predicted: M=64 QKV 244.8 -> ~235-240 us (~2-4%); M=5 QKV 99.6 -> ~96-98 us;
the SMALL shapes are where the fixed cost dominates — M=5 N=3072 K=512
8.3 us could drop ~3-5 us (~40-60% of the combine phase) and M=5 N=2304
K=1536 15.6 -> ~12-13 us. If the A/B shows the fragment-pattern read
penalty eating the overlap win at M=64, the honest outcome is "adopt for
K <= ~2048 only" (a K-gate in the env-routed dispatch, same shape as the
existing `K >= 4*BK` gate). Success criterion: bit-exact gate PASS + no
regression on any serving shape + measurable win on the small-K rows.

## Risks / open questions

- **gfx942 memory model**: the fence+atomic pattern is the canonical CUDA
  threadFenceReduction scheme and HIP documents `__threadfence()` as the
  device-scope equivalent, but it is unproven on CDNA3 in this repo. The
  bit-exact test (2 reps, 10 case shapes, S up to 16) is the detector; a
  failure mode would be non-deterministic plane tails in the combine.
- **Counter self-reset** assumes same-stream serialization between
  launches (true for the current default-stream dispatch; the workspace
  itself already assumes serialized consumers under its mutex). Concurrent
  multi-stream split-K launches sharing the workspace would need a
  per-launch `hipMemsetAsync` of the counters (~6 KB, ~1-2 us) — cheap
  fallback if the A/B probe ever runs streams concurrently.
- **Aborted launches** (hipError after some blocks arrived) leave stale
  counters; the next launch would mis-detect "last". Same mitigation
  (per-launch memset) if this ever matters; today an errored launch is
  already a serving failure.
- Occupancy/register cost of the fused epilogue is negligible (reuses the
  epilogue's `col`/`row_off`, +4 B shared, branch predicated off at
  compile time for `Fused=false`), but the CTA count is unchanged so the
  combine work inflates the LAST wave's runtime per tile.
- Open question for the A/B: is `S=16` (more, smaller planes, more counter
  traffic) changed by the fusion? The fixed-order contract makes any S
  legal; only perf moves.
- Not addressed here (rank-3 scope): the workspace round-trip bytes
  themselves (atomic fp32 C accumulation would remove the planes but
  break the fixed-order bit-exactness contract — explicitly NOT shipped;
  the brief's stop-condition).

## Jobs

- (this box) host 45/45 PASS, cuda 55/55 PASS; HIP uncompiled (no
  toolchain).
- (pending) MI300A job: correctness gate -> batched A/B probe -> S sweep.
  Append per-batch us / checksum outcomes here.
