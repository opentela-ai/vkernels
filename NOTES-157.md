# NOTES — fusion candidate 2: `mxfp4_moe_sorted_quant` (fused MoE gather+quantize)

Branch: `fusion/moe-align-device` (worktree, sequential lanes). This
commit (0483eb6) is stacked **intentionally** on top of the sibling
align-device candidate (74e34c8, also on this branch): the harness runs
the five fusion lanes sequentially through one managed worktree/branch,
so each lane lands on the previous lane's HEAD. The parent should plan
the merge as a stack: 74e34c8 (align-device) → 0483eb6 (this lane).
Slug: moe-aux-fused-sorted-quant. Follows NOTES-155 / NOTES-156 format.

## Baseline

The pre-GEMM MoE aux chain for the W4A4 grouped GEMM (docs/kernels/moe_aux.md):

    align → mxfp4_moe_sort → mxfp4_moe_quant → mxfp4_moe_sort_scales → grouped GEMM

- `mxfp4_moe_sort` writes `A_sorted [EM, hidden]` bf16 (K3: 2048×7168×2 B
  ≈ 29.4 MB written).
- `mxfp4_moe_quant` reads `A_sorted` back (another 29.4 MB) and writes the
  quantized activation; it computes per-group ue8m0 scales **in sorted row
  order** — exactly what `mxfp4_moe_sort_scales` then re-derives from the
  token-order scales (K3: 0.45 MB, launch-bound per the doc's perf table).
- gfx942 measured (doc, post-#145): sort 26.7 µs, quant 20.8 µs,
  sort_scales 8.6 µs. On the pp-pipeline serving profile the aux chain is
  a small fraction of the GEMM step, but the `A_sorted` round-trip is the
  largest pure-data-movement item that carries no new information.

## Change

1. **CPU oracle** (`moe_aux.cpp`, `moe_aux.hpp`): new op
   `mxfp4_moe_sorted_quant(A, sorted_ids, packed, scales, M, hidden,
   group_size, top_k, EM)`. The per-row quant loop was factored into a
   file-local `quant_row()` helper and `mxfp4_moe_quant` now calls it —
   the fused op quantizes `A[sorted_ids[r] / top_k]` with the **same
   function**, so bit-identity with `sort → quant` holds by construction
   (same code, not merely same spec). Padding rows (`sorted_ids[r]` outside
   `[0, M·top_k)`) emit the quant-of-zero-row encoding (`0xFF` scale +
   zero nibbles) directly. The five existing ops keep their signatures and
   behavior; the only edit to shared code is the mechanical `quant_row`
   refactor (output verified unchanged by the existing suites).
2. **HIP kernel** (`moe_aux.hip`): `mxfp4_moe_sorted_quant_kernel` — same
   grid/block shape as the proven quant kernel (`EM·n_groups` blocks of
   `group_size` threads), but each block gathers its group straight from
   `A` via `sorted_ids` (padding rows read the implicit zero row). The
   amax tree reduction, sb/scale expression and nibble staging are the
   quant kernel's statements verbatim. Launcher: fused by default,
   **`VK_MOE_AUX_FUSED_QUANT=0`** (VK_DSA_DECODE_SPLIT naming precedent)
   falls back to the proven two-launch `sort → quant` composition through
   a cached device scratch, so the fusion is an env-gated change with a
   bit-identical fallback in the same entry point.
3. **CUDA shim wiring** (`src/c/CMakeLists.txt` + one define in
   `cuda_compat/hip/hip_runtime.h`): `moe_aux.hip` joins `dsa_kpool.hip`
   in the CUDA-on-NVIDIA path, which made it possible to actually execute
   the HIP kernels on this box's GB10 (nvcc 13, sm_121) instead of
   shipping unvalidated device code.
4. **Tests** (`tests/kernels/moe/test_moe_aux_fused.cpp`, new target
   `moe_aux_fused` in tests/CMakeLists.txt): oracle case matrix (H∈{16…256},
   gs∈{2,4,16,32,64}, tk∈{1…8}, M∈{1…16}), sort→quant composition parity
   (packed AND scales), fused vs `quant(A)→sort_scales` for real rows,
   padding rows == quantized zero activations, zero/±inf/NaN group
   encodings, tie-break breakpoints (`0.125, 0.625, 1.25, 1.75, 2.5`),
   scale-clamp edge (sb=127 exactly), and VK_EXPECTS arg validation.
5. **Python** (`kernels.py` + `_fallback.py` + `_core.cpp`):
   `vk.mxfp4_moe_sorted_quant(A, sorted_ids, group_size=32, top_k=…)`; the
   pure-Python backend implements it as the composition of the two per-op
   oracles; discovery lists updated.
6. **Bench** (`meta/benchmarks/bench_moe_aux.hip`): fused op + the
   3-launch chain added as timing rows, plus a hard parity gate
   (bit-exact vs host `sort → quant`, both gate positions) that aborts the
   run on any mismatch before timing.

## Jobs / validation (GB10 box, no ROCm toolchain)

- `cmake --build build/host` + `ctest --preset host`: **45/45 pass**
  (incl. new `moe_aux_fused` 6/6, existing `moe_aux` 9/9 unchanged).
- `cmake --build build/cuda` (HIP-on-NVIDIA): compiles clean; ctest
  `moe_aux`, `moe_aux_fused` pass.
- Ad-hoc GPU parity driver on the GB10 (11 shape classes → K3 full):
  bit-exact vs CPU `sort → quant` with the gate **on and off**; gate-on
  output byte-identical to gate-off output on every dumped shape
  (`cmp`-verified), so the fused kernel ≡ the proven device composition.
- Python: `tests.python.test_discovery` (DiscoveryTest 9/9 — CliTest
  failures are pre-existing/environmental, reproduced on a clean stash),
  `test_backend` 11 OK, `tests.python.test_kernels` 123 OK (23 skipped:
  compiled-backend tests, no pybind build here); fallback
  `mxfp4_moe_sorted_quant` cross-checked bit-exact against the per-op
  fallback composition.
- Informative GB10 timing (CUDA events, best-of-100, K3 M=112 H=7168
  tk=16 EM=2048): `sort+quant+sort_scales` 652 µs → fused **496 µs**
  (−24%); gate-off (`sort→quant`) 655 µs, i.e. the win is the removed
  `A_sorted` round-trip + one launch, not the fused math. gfx942 numbers
  pending the srun plan in docs/kernels/moe_aux.md.

## Incidents / findings

- **Composition contract subtlety (worth knowing for sibling lanes)**: the
  docs claimed `sort_scales(quant(A).scales) == quant(A_sorted)` — true
  for real rows only. On **padding rows** `sort_scales` writes literal `0`
  scale bytes while `quant` of the zeroed row writes `0xFF`. The fused op
  follows the `sort → quant` route (the one it replaces); the Python
  backends' own test already compared real rows only. Documented in
  moe_aux.hpp / docs/kernels/moe_aux.md; test asserts the real-row part
  and the padding-row `0xFF` encoding explicitly.
- **Pre-existing device-vs-oracle NaN divergence** (found by the GB10
  parity driver, NOT introduced here; supervisor directive: record only,
  do NOT fix in this lane): a group mixing finite values with a NaN
  diverges — the device amax tree (`(a > b) ? a : b`) propagates the NaN
  (→ `0xFF` scale) while the oracle's `if (aa > amax)` ignores it
  (→ quantizes with the finite amax). Exact repro: gs=4 case of the GB10
  driver (M=4, H=32, gs=4, tk=2, EM=64), sorted row r=1 group g=2 holding
  `{-2.64062, -6.875, 2.14062, NaN}` — oracle scale byte 129, device
  0xFF; 12/512 scale bytes differ in that shape. Identical in the proven
  quant kernel and the fused kernel (shared expression), identical in
  both gate positions, so no contract is weakened by this lane. All-NaN
  groups agree (`0xFF` both sides). Proposed one-line follow-up for a
  separate change: `fmaxf` in both device kernels (`fmaxf(NaN, x) = x`,
  matching the oracle) in `moe_aux.hip`'s `mxfp4_moe_quant_kernel` and
  `mxfp4_moe_sorted_quant_kernel` amax reductions. Left untouched here:
  changing the proven quant kernel's numerics is outside this lane's
  additive scope.
  **[RESOLVED in the follow-up change]** Both amax trees now reduce with
  `fmaxf` (NaN-ignoring in both operand positions, matching the oracle);
  the shim header gained `#ifndef` guards on the `__host__`/`__device__`
  neutralizers (CUDA 13's `crt/host_defines.h` defines them in host TUs
  too). New committed GPU test
  `MoeAuxFused.DeviceMatchesOracleOnMixedNaNGroups`
  (tests/kernels/moe/test_moe_aux_fused.cpp) runs BOTH device kernels on
  GB10 vs the CPU oracle with NaN/inf-mixed groups and pins the recorded
  repro (scale byte 129, not 0xFF). Negative control verified: reverting
  to the ternary makes the test fail. cuda ctest 56/56, host 46/46.
- First parity run showed a gs=4 "failure" that turned out to be the NaN
  item above; a second debug harness quantized token-order A with EM rows
  (out-of-bounds garbage) before the real cause was isolated — both
  scratch harnesses were disposable and are not committed.

## Risks / open questions

- gfx942 unvalidated (no ROCm here): the doc's srun sequence
  (test_moe_aux_correct + bench, gate on/off) is mandatory before the
  fusion is default-ON anywhere; until then the kill-switch precedent
  covers rollback (`VK_MOE_AUX_FUSED_QUANT=0`).
- `group_size > 256` silently truncates block dim — inherited verbatim
  from the proven quant launcher (`gs = min(group_size, 256)`), not
  worsened; K3 uses gs=32.
- Merge-conflict points for sibling lanes: `tests/CMakeLists.txt` (+1
  line), `tests/python/test_discovery.py` (expected-kernels list),
  `src/c/CMakeLists.txt` (CUDA HIP sources list +1),
  `cuda_compat/hip/hip_runtime.h` (+1 define),
  `src/python/vkernels/{kernels.py,_fallback.py,_core.cpp}` (new op in
  each). All additive.
- Open: should `mxfp4_moe_sorted_quant` also replace the
  `quant → sort_scales` pair inside `vllm_experts.py` serving (the
  binding currently feeds token-order quant through sort_scales)? Left to
  the serving lane; the op is ready for it.
