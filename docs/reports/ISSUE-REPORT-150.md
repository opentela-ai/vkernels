# ISSUE-REPORT — lane issue-150/bm128-prefill (issue #150)

## What was done

Root-caused the variant-6 (`VK_MOE_PF_VARIANT=6`, 128-row prefill tile)
first-launch fault (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`) and the
128-row numeric corruption on MI300A / ROCm 6.3.0, fixed the dispatch, and
verified the full suite green on-cluster. The lane ran a fault bisect
(standalone repro matrix, link-composition matrix, rocgdb precise fault
analysis, allocation-history discriminators) down to two independent
defects — one platform, one real dispatch bug — then fixed both and
re-verified on the mi300 partition.

### PART A — fault bisect (link-composition phase)

Jobs 676508–676556 (`meta/scripts/issue150_bisect{,2..7}_mi300.sh`) narrowed
the fault:

- Resource-spill theory eliminated: VGPR 114 (under the 128 cap for 2
  blocks/CU), LDS in budget — `<128,16>` / `<128,32>` instantiations are
  clean (`bisect.676515`).
- Launch-bounds / occupancy probe still faults; `-O2` also faults (not an
  O3-only miscompile) (`bisect2.676518`, `bisect3.676530`).
- Standalone repro (`meta/benchmarks/repro_issue150_fault.hip`) passes at
  block=128 on first launch — RDC and non-RDC, sequenced, and under alloc
  pressure (`bisect3d.676540`, `bisect4.676542`, `bisect5.676547`).
- Link composition exonerated: repro linked against the full
  `libvkernels.a`, against the CMake-compiled `moe_fused` objects only, and
  in manual `-fPIC`/`-fgpu-rdc` compositions — the kernels fault nowhere;
  only the CMake-built test binary faults, and only for the exact 4-case
  sequence (`bisect5/6`, `dbg2_run.txt`). The bisect7 `clang++`-link matrix
  cells did not complete as designed (clang++ PATH + repro link failures in
  the bisect7b script) — superseded by the harness-side evidence above.
- rocgdb precise analysis: faulting kernel was the **v0 baseline down
  kernel** (`down_combine_kernel_prefill<64>`), faulting *after* its sync
  returned; fault VAs ≡ `w2_base + garbage_expert × 0x20000` with a garbage
  expert index (e.g. `0x2010007E`); pre-launch device state verified clean
  (`eids=[0,1,2,3]`) (`gdb_precise.txt`, `eids_run.txt`).

**Two independent root causes:**

1. *Platform:* per-case `hipMalloc`/`hipFree` churn triggers a ROCm 6.3.0
   VM-deallocation fault on MI300A at phantom addresses, asynchronously —
   even in waves of earlier cases whose `hipDeviceSynchronize()` already
   returned success. Discriminators: single cases pass; repeated cases
   pass; churn-free sequences pass with 52 distinct allocations
   (`seqB` 3/3); fixed max-size reuse passes 3/3 (`seqR`); the churn
   sequence faults 100%. Immune to all serialization knobs.
2. *Real dispatch bug:* with variant 6, the blk64 cases fell through all
   v5 else-ifs into the **v0 baseline** branch (the test comment promises a
   v5 fallback), and the 128-row path launched both stages with 128-row
   tiles but then fell through into the generic down section, launching the
   down GEMM a **second time as the v0 baseline** at grid `(EM/64, …)` —
   re-reading `expert_ids[4..7]` past the end of the 128-row expert_ids
   array → garbage experts → wild weight loads and numeric corruption
   (clean128/padded128 `max_rel ≈ 280`).

### Fixes

- `src/c/vkernels/kernels/moe_fused.hip` — variant 6 joins the v5 pipe
  branch for gateup; the down dispatch skips the second launch entirely at
  128-row alignment (both stages already launched above) and falls back to
  the v5 down path at 64-row alignment, matching the v5-fallback gateup.
  `VK150_DEBUG`-guarded expert-guard/pointer-print diagnostics added to the
  pipe kernels (off unless `VK150_DEBUG` is defined).
- `meta/benchmarks/test_moe_fused_prefill_correct.hip` — test restructured
  to the churn-free pattern: persistent `DevBufs` allocated once at
  worst-case geometry (`alloc_dev_bufs`, freed at the end), aliases per
  case; validation logic unchanged. Avoids the platform fault without
  weakening the test.

### PART B — on-cluster verification (green)

The two terminal in-band ctest jobs hit their walls without persisting
results (stdout rode the SSH pipe and was consumed by the tool timeout;
ctest finalized only a truncated `Testing/Temporary/LastTest.log.tmp` start
banner — 676611 t90 TIMEOUT 1:30:26, 676635 t60 TIMEOUT 1:00:19, plus a
t30 retry 676646; 676604, the in-band build+ctest step, was cancelled at
1:03 elapsed under the lane's 1 TB memory guard regime). Verdicts were
re-drawn from adjusted **sbatch arms with persistent `-o` artifacts,
steady-state monitor samples every 60s, per-test timeouts, and a 900 GB
self-cancel threshold (respecting the 1 TB guard — peak RSS ≈ 30 MB)**:

- `verify8.676660` (`issue150_verify_mi300.sbatch`) — incremental rebuild
  on the fixed source (md5 `dcb95d56` provenance in the artifact):
  variant-6 sweep **all PASS** — default v5-fallback, variants 0–4, variant
  6 swiglu ×3 reps (`max_rel=0.010937`, bf16 tolerance) and variant 6 situ
  ×3 reps (`max_rel=0.000023`), no fault; MOE ctest family
  **7/7 passed in 1.83s**.
- `suite8.676661` (`issue150_suite_mi300.sbatch`) — full ctest suite:
  **100% tests passed, 0 tests failed out of 50, 20.73s** (incl. the MOE
  family and the previously-missing `kda_chunked_fused_gpu` /
  `dsa_topk_fused_gpu` GPU tests, now built).

The earlier wall-timeouts are understood as an in-band execution artifact:
the same suite completes in seconds under a proper sbatch GPU allocation
with `ctest --timeout` bounding each test; the wall-kill design (monitor
samples landing in-band) is what the adjusted arms make safe.

## Deliverables

- `src/c/vkernels/kernels/moe_fused.hip` — variant-6 down-dispatch fix +
  guarded diagnostics
- `meta/benchmarks/test_moe_fused_prefill_correct.hip` — churn-free alloc
  restructure (`DevBufs`), validation logic unchanged
- `meta/benchmarks/repro_issue150_fault.hip` — standalone fault repro
  (isolates the platform churn trigger from kernel logic)
- `meta/scripts/issue150_bisect{,2..7}_mi300.sh`, `issue150_debug_mi300.sh`
  — fault bisect matrix
- `meta/scripts/issue150_verify_mi300.sbatch`, `issue150_suite_mi300.sbatch`
  — adjusted verification arms (persistent output, monitor samples, guard)
- `ISSUE-REPORT-150.md` — this report

## Validation

```
# Phase 1 build (incremental, -j 16, guard-friendly)
cmake --build cmbuild -j 16 --target vkernels test_moe_fused_prefill_correct \
  vkernels_test_{moe,moe_fused,moe_align_oracle,moe_aux,moe_aux_fused,glm_moe,dist_moe}
#   → Built target vkernels + 7 MOE test targets, BUILD_EXIT=0

# Phase 2 sweep (verify8.676660)
VK_MOE_PF_VARIANT=6 test_moe_fused_prefill_correct      # clean128/padded128 max_rel=0.010937 PASS
VK_MOE_PF_VARIANT=6 test_moe_fused_prefill_correct situ # max_rel=0.000023 PASS
# variants 0–4 + default v5: PASS; variant-6 reps ×3: PASS (no fault)

# Phase 3 MOE ctest family (verify8.676660)
ctest -R '^moe$|^moe_fused$|^moe_align_oracle$|^moe_aux$|^moe_aux_fused$|^glm_moe$|^dist_moe$' --timeout 300
#   → 100% tests passed, 0 tests failed out of 7 (1.83s)

# Full suite (suite8.676661)
ctest --timeout 300 --output-on-failure -LE "multi|node"
#   → 100% tests passed, 0 tests failed out of 50 (20.73s)
```

Residual: the `bisect7` clang++ link-matrix script cells never produced
fault/no-fault rows (tooling only — clang++ PATH and repro link failures);
the link-composition verdict rests on the bisect3d–6 repro matrix and the
`seqR`/`seqB` allocation discriminators. The platform half of the root
cause (ROCm 6.3.0 churn fault) is avoided in the test rather than fixed —
any other test in this repo with per-case hipMalloc/hipFree cycles remains
exposed to the same driver behavior and may need the same restructure.
