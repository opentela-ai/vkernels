# ISSUE-137 lane report — issue-137/dsa-fulltopk

## What was done

Step-0 reconciliation found that **the entire #137 scope was already implemented and
merged to main via PR #140** (commit 3044526, merged 2026-09-18): the vectorized
lane-contiguous partial kernel (`dsa_sparse_fwd_partial_vec_kernel`, uint2 per lane,
one load per key, admission tail_dim==0 / dim%4==0 / dim<=256), shipped unfused with
`split_for` re-derived to two bands (topk>=1024 -> 32, else min(topk,16)), the perf
record section in `docs/performance/dsa/gfx942.md`, and the focused harness
`meta/benchmarks/bench_dsa_split137.hip` + `meta/scripts/run_issue137_vec_mi300.sh`.
The fused-combine and register-batch (G>1) levers were implemented, measured slower,
and shipped OFF (documented). Nothing remained to implement; the issue sat OPEN with
zero comments.

The lane therefore became: **independent re-verification of the acceptance criteria
from merged main**, then the record/comment deliverables.

1. **Local quick gates**: `cmake --preset host`, build, `ctest --preset host` —
   47/47 passed (2 GPU tests disabled on host, expected).
2. **Fresh hardware verification** on beverin: worktree rsynced to
   `/capstor/scratch/cscs/xyao/vkernels-i137-verify/` (note: first attempt dropped
   `src/c/vkernels/core/` because `rsync -C` cvs-excludes the name `core` — job
   676507 failed at configure; re-synced without `-C`), then
   `sbatch -A a-infra01 -p mi300 ... run_issue137_vec_mi300.sh` with `SRC` pointed at
   the fresh copy (fresh build dir).
3. **Perf record updated** with a verification note in the #137 section + raw log
   stored as `docs/performance/dsa/beverin-137-verify.log` (commit ad9b57f,
   conventional: `docs(dsa): ...`), pushed as `issue-137/dsa-fulltopk`.
4. **Draft PR** #168 with results table and "Refs #137".
5. **Issue comment** posted with the measured-results table and acceptance status:
   https://github.com/opentela-ai/vkernels/issues/137#issuecomment-6110100394

## Evidence

- **Job**: 676510 (`vk-i137v`, nid003028, mi300, 2026-10-11, COMPLETED, 41 s —
  same wall time as the original #137 jobs 641060/641089/641098, i.e. the identical
  variance-driven HIP-event-pair harness: warmup 3, <=60 event-paired iters, median,
  exact-zero samples discarded). Log: `docs/performance/dsa/beverin-137-verify.log`.
- **Full-topk decode (H=64, S_q=1, topk=2048, dim=256, tail=0), shipped vec dispatch:**
  - split=32 (`split_for` recommendation): 94.0 / **98.8 us** (min/med) / **680 GB/s**
  - split=64: 123.4 / **126.2 us** / **532 GB/s**
  - Acceptance (<156 us, >430 GB/s): **met at both splits**.
  - Cross-check under the batched 1000-launch methodology: #155's same-job A/B
    (job 649392) measured full-topk decode at 85.2 us/launch.
- **Correctness**: 19-row device-vs-oracle suite, **0 failures under all 8 env
  variants** (default, VK_DSA_VEC=0, VK_DSA_DB=2, VK_DSA_FUSE=1, VK_DSA_VECG=4/8,
  fused+G=4, 3-rep nondeterminism probe); max rel <= 3.9e-3, identical magnitudes to
  pre-#137 rows. 17/17 hard gate (+vec coverage rows) holds; db-kernel history
  (e6499e6, c6b9866) untouched.
- **`dsa_sparse_fwd_split_for` re-validated**: 2048->32 is the measured best;
  256->16 and DSv3->16 match their optima; 128->16 remains the known ~5 us
  conservative band (split=8 = 30.9 us). Bands unchanged.
- Known pre-existing ROCm 6.3 {0,0,0} event-timer quirk hit 3 wandering rows
  (topk=128 split=32 serial; topk=256 split=16 vec; topk=128 split=4 fused); all
  neighbors measured normally — same documented artifact as the original runs.

## Links

- PR (draft): https://github.com/opentela-ai/vkernels/pull/168
- Issue comment: https://github.com/opentela-ai/vkernels/issues/137#issuecomment-6110100394
- Original implementation PR: https://github.com/opentela-ai/vkernels/pull/140 (merged)
- Branch: `issue-137/dsa-fulltopk` (commit ad9b57f on top of main 100398f)

## What remains

- Issue #137 is ready to **close** (all four acceptance checkboxes verified; stated
  in the comment). Closing left to maintainers/orchestrator.
- No code changes were needed on the lane; the only commit is the verification
  doc/log. If a non-doc artifact is desired, none is warranted by the acceptance list.
