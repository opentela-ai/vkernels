# ISSUE-REPORT — issue #148 (fp8 MFMA decode), lane `issue-148/fp8-mfma-decode`

## What was done

Revived lane (two prior attempts died on transient provider 400s before any
work). STEP 0 recon found that PR **#166** (merged 2026-10-11, "Reconcile
Issue 148 FNUZ and MFMA experiments") had already landed the issue's code on
main: native HIP `glm_fp8_fnuz_block_gemv` / `glm_fp8_fnuz_mfma_block_gemv`,
CPU oracles, the Triton `_triton_decode_kernel`/`glm_decode_fp8_gemm`, the
secondary `GLM53_GEMV_FNUZ_NATIVE` float8e4b8 bitcast in `glm_expert_gemv`,
diagnostic probes, and `meta/scripts/issue148_decode_mfma_mi300.sh`. What
remained — exactly what #166's reconciliation doc listed — was the gfx942
qualification: compile, native correctness, and fresh MI300A timing. This
lane executed that; **no new kernel code was needed**.

1. Local gates: `cmake --preset host` + build + `ctest --preset host` →
   **47/47 passed** (2 GPU tests disabled as expected).
2. Synced the merged tree to beverin scratch and ran the issue's sbatch
   methodology.
3. Fixed one bench bug found en route (see below).
4. Updated perf records, pushed, opened draft PR, commented on the issue.

## Evidence

- **Job 676509** (nid003018, 2026-10-11): built all HIP targets; `test_glm_fp8_gemv`
  **24/24 PASS** (manual / fnuz-cvt / MFMA vs CPU oracles, both token counts,
  incl. K=1152 odd segments); same-node HIP timing (median of 200 event pairs,
  unpinned clocks):
  - gate/up [4096,4096] M=1: manual 27.3 µs / 616 GB/s; fnuz-cvt 27.5 µs
    (0.99×); **MFMA 23.3 µs / 722 GB/s (1.17×)**
  - gate/up M=2: manual 31.6; fnuz 27.3 (1.16×); MFMA 26.5 (1.19×)
  - down [4096,2048] M=1: manual 14.8; fnuz 14.6 (1.01×); MFMA 16.8 (0.88×)
  - down M=2: manual 16.4; fnuz 16.0 (1.03×); MFMA 18.4 (0.89×)
  - (this node's manual baseline ran ~0.8× the 22.4 µs record's clock state;
    ratios are the robust signal per the issue methodology)
- **Job 676512** (nid003028, 2026-10-11): Triton MFMA decode `--sweep`
  (BM=16, BN∈{64,128} × stages∈{2,3} × sk∈{2,4,8}) after the bench fix:
  best 363 GB/s gate/up M=1 (BN=64, s=2, sk=2), 183 GB/s down M=1;
  e2e 74–76 µs (fused activation quantize alone ~36 µs); parity rel ≤ 4.4e-3
  vs torch blockwise oracle on the same quantized operands; full JSON at
  `work/glm53/issue148_decode.json` on beverin. Secondary
  `expert_gemv` FNUZ_NATIVE: **bit-identical** to manual decode in all four
  cells, perf-neutral (±3%) — the "LLVM lowers to the same bit-twiddle"
  outcome the issue anticipated. No win.

## Verdict

The issue's ≥2× acceptance (≥1502 GB/s on gate/up M=1) is **not met** by any
variant. Best measured is HIP MFMA at 722 GB/s — 1.17× the same-node manual
decode, still below the 751 GB/s record in absolute GB/s. Removing the
per-byte decode does not make these kernels bandwidth-bound at M=1/BM=16;
MFMA regresses on down (K=2048). The manual-decode kernel remains the
serving record; serving dispatch unchanged; the MFMA/fnuz kernels stay
in-tree as validated (oracle-parity) experiments, not adopted. Full-model
quality gate (+0.01 nats/token vs SGLang) is moot at this speedup and was
not executed.

## Bench fix (the only code change)

`meta/benchmarks/bench_glm_issue148_decode.py`: pop `VK_DECODE_GEMM_TILES`
before each shape's parity call. A previous shape's stages=3 best-config
export otherwise leaks into the next shape's parity launch → shared-memory
OutOfResources crash (jobs 641621 and 676509 pre-fix) → no results JSON.
Post-fix the bench completes all four cells (job 676512).

## Deliverables

- Lane branch commit: `a92f337` on `issue-148/fp8-mfma-decode` (pushed,
  up-to-date with origin).
- Docs: `docs/glm53-decode-kernels.md` §#148 (full measured record) and
  `docs/issue148-reconciliation.md` (qualification checklist items 1–3).
- Draft PR: https://github.com/opentela-ai/vkernels/pull/169 (results table,
  "Refs #148").
- Issue comment: https://github.com/opentela-ai/vkernels/issues/148#issuecomment-6110119731

## Reproduce

```bash
SRC=/capstor/scratch/cscs/xyao/issue148 sbatch --exclusive -A a-infra01 \
  -o issue148.%j.out \
  --export=ALL,SRC=/capstor/scratch/cscs/xyao/issue148,PY=/usr/bin/python3.11 \
  meta/scripts/issue148_decode_mfma_mi300.sh
```

(Python section alone: `issue148_pyonly.sbatch` on beverin scratch; torch
env is `/usr/bin/python3.11` with user site-packages torch 2.9.1+rocm6.3,
triton 3.5.1. sbatch requires `-A a-infra01`; the historical `-A root` no
longer works via CLI.)

## What remains / follow-ups

- Full-model quality A/B (reconciliation item 4) — unexecuted, moot for
  adoption at the measured speedup.
- If the decode record is ever re-attacked: the measured op-mix evidence
  points at scale handling + LDS staging + BM=16 padded-tile overhead, not
  weight decode — a fused scale/lds redesign (not more decode variants)
  would be the angle; out of this lane's scope.
- Graph/stream adapter for the native wrappers (reconciliation item 2) —
  only relevant if the kernels are ever adopted; they were not.
