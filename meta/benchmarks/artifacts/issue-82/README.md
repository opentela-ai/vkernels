# mla_fwd split-K decode — issue #82 artifacts

Beverin (MI300A, gfx942), `mi300` partition, account `a-infra02`, one job at
a time (only this issue's jobs — others untouched), scratch
`/capstor/scratch/cscs/xyao/vk-issue-82` synced from the
`issue-82-mla-splitk-decode2` worktree.

| Slurm job | what | outcome |
|---|---|---|
| 640241 | first submission | failed in 3 s: runner defaulted `SRC=$HOME/vk-i82` (no scratch checkout there) |
| 640242 | resubmission with a script-path-derived SRC | failed in 3 s: sbatch copies the script to `/var/spool`, so BASH_SOURCE does not resolve; SRC pinned explicitly afterwards |
| 640244 | full build + correctness + bench | build OK; **correctness gate FAILED (4/5 split-path cases)**: the split kernel stored the UNSCALED partial numerator while the combine applies `exp(lse_sp − lse_g)` weights, which is exact only for the NORMALIZED per-split output (an `s_sp` factor leaked into every combined row). Fixed the same day. |
| 640246 | full run on the fixed kernel | **all 9 device cases PASS** (max_rel ≤ 3e-6, incl. masked-slice splits and an all-masked row); bench: decode H=1 S_q=1 S_kv=8192 **5507 → 66.8 µs median (63.3 µs min) = 82×, ~549 GB/s**; prefill unchanged |

- `run.640246.out` — the passing run: correctness gate + acceptance bench +
  (autotuner; the host-test step found no binary because the runner did not
  build `vkernels_test_mla` that day — the formula test passed locally 8/8).
- `run.640244.out` — the failing gate (kept as the correctness-first record).
- `run.640241-640242-startup-failures.log` — the two runner misfires.
