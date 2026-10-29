# ISSUE-REPORT — issue #60 (lane `issue-60/sm80-kpool`)

**Issue**: feat(cuda): GLM-5.3 DSA kpool-cache kernels for SM80 (A100) — bf16
storage to avoid fp8e4nv JIT
**Branch**: `issue-60/sm80-kpool` (pushed, PR #171 draft)
**PR**: https://github.com/opentela-ai/vkernels/pull/171

## What was already on main when this lane started (reconciled, not redone)

- `1de4eec` — kpool-cache compress/write kernels + HIP-to-CUDA A100 port;
  `8b4ac08` — fp8e4m3fn+scale cache + shared bf16/fp8 store core; `cc14083`/
  `155d40a`/`b547888` — kpool device C ABI on the CUDA build + ctypes loader;
  `607554e` (PR #139) — closed the in-repo SM80/CUDA gaps (CTest registration,
  bf16-storage contract test, CI, `docs/sglang-dsa-kpool-integration.md`).
- Device-vs-oracle **14/14 PASS on MI300A and on A100-SXM4-80GB** (bristen,
  CUDA 12.6/sm_80) — recorded in `docs/performance/dsa-kpool/{gfx942,A100}.md`.
- Cookbook (`opentela-ai/cookbook`,
  `deployments/llm/bristen/glm-53-flash/`) already carries the sglang-side
  wiring: dtype-dispatched bridge in `kpool_fp8_index.py` with tier-1 C-ABI
  device dispatch, tier-2 torch port, tier-3 numpy; SM80 `act_quant` +
  deep_gemm shims; local wiring tests T0–T11 green. First-forward smoke on
  bristen `flashmla_sparse` already PASSED (jobs 83115/83120/83152/83166).

## What remained and what this lane did

The only remaining gap: the **bristen serving deployment still ran the
tier-2 (torch) bridge** — the deployed `patches_full/kpool_fp8_index.py`
predates the tier-1 dispatch, the deployed `vkernels-python` package lacked
`dsa_kpool_device`, and the deployed sbatch never exported `VKERNELS_LIB`.
This lane:

1. **`meta/scripts/issue60_kpool_a100_bringup.sh`** (commit `956393c`) —
   idempotent two-phase bring-up: `stage` (source export, python package
   refresh, cookbook tier-1 `kpool_fp8_index.py` into `patches_full`,
   `VKERNELS_LIB` block into the deployed sbatch) and `build` (in-container
   cmake CUDA build → device-vs-oracle gate → install
   `libvkernels_c.so` → `nm`/ctypes probe of the 4 exports **inside the
   serving container**).
2. **Executed `stage` on bristen** — deploy dir
   `/capstor/scratch/cscs/xyao/glm-53-flash-bristen`: `vkernels-src/` at main
   tip + lane commit, `vkernels-python/vkernels` refreshed, tier-1 patched
   file + sbatch block installed (backups `.pre-tier1`).
3. **Verified the build path off-cluster** (SSH to bristen was down
   ~15:00–15:41Z due to a revoked SSH cert; GB10 dev box used meanwhile):
   - cmake configure + build of `vkernels_c` + `test_dsa_kpool_correct`
     under **CUDA 13.0** (no ROCm): clean.
   - Device-vs-oracle gate: **PASS (0 failures)** (bf16 + fp8+scale cache,
     prefill + decode cases).
   - `nm -D libvkernels_c.so` → exactly
     `vk_hip_dsa_kpool_{assemble,decode_update}{,_fp8}` (4/4).
   - `tests/python/test_dsa_kpool_device.py` against the built `.so`:
     **13/13 OK** (5 torch-dependent skips).
   - tier-1 patched `kpool_fp8_index.py`: byte-compiles.
4. **Queued the on-cluster jobs** (bristen, account `infra02`):
   - **84637** `vk60-kpool-build` — 1 GPU node, 40 min, runs `build` inside
     the `sglang-dev-cu13` serving image (same CUDA ABI as serving),
     installs `libvkernels_c.so` to `$VKERNELS_PYTHON/lib/`.
   - **84638** serving + probe — `--dependency=afterok:84637`, 2-node PP2
     `serve_glm_53_flash_sglang.sbatch` with `GEN_PROBE_TIMEOUT=900`,
     `GEN_CORRECTNESS_PER_REQ_TIMEOUT=300`. Logs:
     `logs/glm53-bristen-84638.out`; go/no-go lines:
     `[kpool] device tier: ...` (tier-1 active) and `GEN_PROBE_OK` +
     `run/84638/gen_correctness_*.json` verdict (acceptance target:
     `PASS pass=5/6 crisp=3/3`).
   Both were still **PENDING** at report time (queue at ~83 jobs,
   `ReqNodeNotAvail` on the excluded/bad-boot node set). They need no
   further action — 84637 installs the library and 84638 then serves with
   tier 1 automatically; `scancel 84638` / `scancel 84637` to abort.
5. **Docs**: `docs/sglang-dsa-kpool-integration.md` — new "Tier-1 on-cluster
   bring-up (bristen)" section with the staged state, off-cluster evidence
   and exact reproduction commands.

## Honest status vs the acceptance list

| Acceptance item | Status |
|---|---|
| Host CI: `vkernels_test_dsa_kpool` + full ctest no regressions | **done** (pre-existing on main; re-verified at main tip + lane commit) |
| Real A100 first-forward on `flashmla_sparse` w/o `fp8e4nv` JIT | **already passed earlier** (jobs 83115/83120/83152/83166, torch tier) |
| Tier-1 device kernels serving on bristen | **staged + gated; in-container build queued (job 84637)** |
| 2-node PP2 `gen_correctness.py` `PASS 5/6 crisp=3/3` | **queued (job 84638, chained after 84637)** — best prior on-cluster result: 6/6 correct on a warm server (manual probe, 83120); in-job probe budget historically passed 2/6 at 0.37 tok/s, now ~5.1 tok/s bs=1 |

Nothing was fabricated: every number above is from a completed run or a
logged job; the two queued jobs are pending, not passed.

## Reproduction (after the queued jobs, or from scratch)

```bash
# from a vkernels checkout (this branch)
meta/scripts/issue60_kpool_a100_bringup.sh stage /path/to/cookbook
# build + gate + install (~20 min, 1 GPU node)
sbatch vk60-build.sbatch            # see script header for the inline srun form
# serving + full-budget probe (2 nodes)
sbatch --dependency=afterok:<build-job> \
  --export=ALL,GEN_PROBE_TIMEOUT=900,GEN_CORRECTNESS_PER_REQ_TIMEOUT=300 \
  $DEPLOY_DIR/serve_glm_53_flash_sglang.sbatch
grep -E "kpool\] device tier|GEN_PROBE" logs/glm53-bristen-<job>.out
cat run/<job>/gen_correctness_*.json   # verdict / pass / crisp
```
