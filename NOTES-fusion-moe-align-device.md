# NOTES — fusion candidate `moe-align-device` (on-device moe_align_block_size)

Branch: `fusion/moe-align-device` (worktree, off main @ 8e2bc1b). Do NOT
push/merge. Slug: moe-align-device, rank-1 candidate.

## Honest scope statement (read this first)

The repo state at main already contains the bulk of this candidate — the
on-device align landed via issues **#46** (kernel + C ABI +
`meta/benchmarks/test_capi_moe_align.hip` parity harness) and **#78**
(binding-layer default-ON + `VKERNELS_GPU_ALIGN=0` kill-switch), after the
`gfx942-pp-pipeline.md` profile called the `topk_ids.cpu()` round-trip the
single highest-value follow-up. This lane therefore (a) audited the full
chain end-to-end against the brief's contract, (b) added the **brief-named
kill-switch `VK_MOE_ALIGN_DEVICE=0`** (the VK_DSA_DECODE_SPLIT /
VK_MHC_PRE_STRICT precedent naming) as an independent gate at the binding
layer, (c) added the missing host-oracle test coverage (padding sentinels,
garbage-id skipping, prefill block_size=64, E=256, and the
`_align_em_bound` ≥ oracle-EM safety property), and (d) documented the
device path (the kernel doc had NO section for it) + the MI300A A/B plan.
Nothing existing was weakened; all changes additive.

## Baseline (numbers from the docs, not re-measured here)

- `docs/performance/moe-fused/gfx942-pp-pipeline.md` (jobs 603394/603395,
  K3 dummy, PP3/TP8/EP, PP0 gate, 31 layers/step):
  - PP0 `moe:vkernel_apply` mean **4199.4 µs (eager) / 3953.3 µs
    (breakable)**, of which `moe:apply.cpu_copy` — the `topk_ids.cpu()`
    host round-trip feeding the CPU-only align — is **97–100%**
    (4185.1 / 3844.4 µs). PP1/PP2 vk_apply is only ~234–802 µs.
  - MoE/step on PP0: **130.2 ms (41% of the 316.8 ms step)**; the pure GPU
    `launch` floor is ~7–9 ms/step. Breakable captured only the non-MoE
    graph (1.39x throughput); the MoE region stayed eager (0.94x) because
    of this sync.
  - PP1/PP2 regressed ~3x under breakable (234 → 758/802 µs) — hidden
    under PP0's gate today.
- `docs/kernels-reference.md` row 36 carries the same 1.39x / host-sync
  framing. This is the highest-value fusion follow-up in the repo.

## What exists on main (audited, unchanged by this lane)

- `moe_fused.hip`: `moe_align_block_size_kernel` — single block,
  threads = max(N, local_n) rounded to a wavefront (≤1024). Phase A
  parallel histogram (atomicAdd per flat index, after `expert_map`
  global→local remap, garbage ids → skip); Phase B thread-0 padded counts
  + exclusive prefix + EM (ascending experts, min one block, EM clamped
  to max_EM); Phase C thread-0 ordered scatter + per-block expert_ids.
  Outputs bit-identical to the CPU/Python oracle including padding
  encoding (`sorted_ids` pad = N = M*top_k, `expert_ids` pad = −1).
- C ABI `vk_hip_moe_align_block_size` (hip_capi.cpp): validation only,
  enqueues on the caller stream, no alloc/sync/D2H → capture-safe;
  `VK_ERROR_UNSUPPORTED` for N > 1024 (prefill stays on CPU align).
- `vllm_experts.py`: on-device default (issue #78), GEMM grid sized at the
  host-constant `_align_em_bound` so `out_em` is never read back;
  caller-owned `CaptureSafeScratch` buffers.

## Change (this lane)

1. `src/python/vkernels/vllm_experts.py` — **`VK_MOE_ALIGN_DEVICE=0`**
   kill-switch (default ON), independent of `VKERNELS_GPU_ALIGN`; either
   one forces the CPU align path at the binding layer. `use_gpu` now
   requires both flags. Comment blocks updated. (Merge-conflict point:
   small, additive — flag for the sibling lanes.)
2. `tests/python/test_moe_align_device.py` (NEW) — default-ON, `0`/`false`
   disables, other values keep ON, and independence of the two switches
   (fresh-interpreter subprocess probes of `_MOE_ALIGN_DEVICE` /
   `_GPU_ALIGN` / combined gate).
3. `tests/kernels/moe/test_moe_align_oracle.cpp` (NEW, registered as
   `moe_align_oracle` in tests/CMakeLists.txt — one additive line, the
   only shared-file touch besides vllm_experts.py):
   - `PaddingSentinelsExact` — sorted pad is exactly N, expert_ids pad −1,
     untouched tails preserved (the contract the fused/moe_aux kernels
     guard on: `flat < N`, `expert_ids[b] < 0`).
   - `GarbageAndNegativeIdsSkipped` — negative / ≥ num_experts ids never
     echo into sorted_ids (device kernel implements the same rule).
   - `BlockSize64AndManyExpertsMatchRef` — prefill block_size=64 and the
     E=256 serving shape vs an independent in-test reference.
   - `ServingEmBoundUpperBoundsOracle` — `_align_em_bound` (mirrored in
     C++) ≥ oracle EM and a block multiple, over uniform / hot-expert /
     power-law-skewed routings on 6 shape classes. If this bound were ever
     below the real EM, the capture-safe max_EM grid would silently
     truncate real tokens — this is the property test that guards it.
   - `M0EmptyRouting` — C++ oracle returns EM=0 (documented divergence:
     the Python/with_map path and the GPU kernel emit one padding block;
     handled at the binding layer).
4. `docs/kernels/moe_fused.md` — NEW "On-device alignment" section
   (device entry, C ABI, output/max_EM contract, capture-safety,
   kill-switches, parity + oracle tests).
5. `docs/performance/moe-fused/gfx942-pp-pipeline.md` — appended
   "Implementation notes — on-device align landed" (status of the
   Future-work sketch + the MI300A A/B plan for the new switch).

## Validation run HERE (GB10 box, no HIP toolchain)

- Host build + ctest: **all green** — `moe`, `moe_fused`, `moe_aux`,
  `glm_moe`, `dist_moe`, plus the new `moe_align_oracle` (6/6).
- `.venv/bin/python -m pytest tests/python/test_moe_align_device.py
  tests/python/test_vllm_experts.py`: 33 passed, 1 skipped
  (env-dependent skip), 11 subtests.
- **NOT validated here**: the HIP kernel cannot compile/run on this box
  (no HIP toolchain). It is unchanged from the main-landing that
  `meta/benchmarks/test_capi_moe_align.hip` validates on gfx942; this
  lane changed no device code. No CUDA port of moe_fused exists
  (no `.cu`), so the build/cuda tree cannot exercise this op either.

## Exact MI300A A/B plan (unvalidated on gfx942 by this lane)

```bash
# build (login node)
cmake --preset hip && cmake --build --preset hip -j

# 1. parity gate — must pass with the switch in BOTH positions
srun --partition=mi300 --gpus=1 ./build/hip/meta/benchmarks/test_capi_moe_align

# 2. serving A/B — same job shape as 603394/603395 (K3 dummy, PP3/TP8/EP,
#    breakable on), only VK_MOE_ALIGN_DEVICE toggled
srun --partition=mi300 --gpus=1 VK_MOE_ALIGN_DEVICE=0 <pp-serving-launch>   # A: CPU align
srun --partition=mi300 --gpus=1 VK_MOE_ALIGN_DEVICE=1 <pp-serving-launch>   # B: device align

# 3. profile — regions moe:apply.{cpu_copy,cpu_align,gpu_copy,gpu_align,launch}
python3 meta/benchmarks/moe_profile.py --label A <A-traces rank0/8/16> \
                                       --label B <B-traces rank0/8/16> --head-to-head
```

Batched-probe micro option: wrap `vk_hip_moe_align_block_size` +
`vk_hip_fused_moe_mxfp4` in a `meta/benchmarks/probe_*.cpp` 1000-launch
event-pair loop (NOTES-155 style) comparing the CPU-align feed vs the
device-align feed end-to-end at the decode shapes (M ∈ {1,8,64},
top_k ∈ {8,16}, E=256, block 16), checksums equal.

## Expected win — byte/roofline model

Device traffic of the align step (decode, worst case N = M·top_k = 1024,
local_n = 256, block_size = 16, max_EM = 4864):

- read `topk_ids`: 4·N = 4 KiB
- write `sorted_ids`: 4·max_EM = 19.4 KiB; `expert_ids`: 4·(max_EM/16) =
  1.2 KiB; `out_em`: 4 B
- total ≈ **25 KiB** → at even 1% of the 5.3 TB/s HBM roof (53 GB/s) this
  is ~0.5 µs of traffic. The single-block kernel is purely
  **launch/latency-bound** (~2–4 µs, the ~2.8 µs dispatch-floor class of
  `dsa_kpool_*`, kernels-reference rows 23–25) — the honest SOL for it is
  the dispatch floor, not the HBM roof. Compare against what it removes:
  a **3.8–4.2 ms stream-synchronizing host round-trip per call** (≥1000×
  the kernel's entire cost), 31× per step on PP0.
- Serving-level expectation (citing the baseline table): PP0 MoE/step
  130.2 ms → ~7–9 ms launch floor + ~31·4 µs align ≈ **8–9 ms**
  (~15x on the MoE region); step wall 316.8 ms → ~195 ms, i.e. up to
  ~1.6x on the PP0 gate **and** the whole MoE region becomes
  graph-capturable (removing the per-op Python dispatch the breakable
  path still pays on PP1/PP2).

## Risks / open questions

- **gfx942 status**: kernel/ABI untouched from the validated main landing,
  but THIS lane's changes (env gate, tests, docs) have never run on a ROCm
  box — the srun sequence above is mandatory before defaulting anything
  new in a deployment cookbook.
- N > 1024 (prefill) still pays the host round-trip → the natural follow-up
  is a multi-block decoupled-lookback variant; deliberately out of scope
  ("simple and correct first").
- PP1/PP2 ~3x breakable regression (finding 4 of the pp-pipeline doc)
  becomes the new floor once PP0 stops gating — next lane's problem, not
  addressed here.
- Serial thread-0 phases are O(N + local_n) inside one block — fine at
  N ≤ 1024, dominant cost if the limit is ever raised.
- C++ oracle `moe_align_block_size` returns EM=0 for M=0 while the
  with_map Python reference and the device kernel emit one padding block
  (min-EM). Documented in the new test; unifying is a contract change and
  was not done.
- Merge-conflict points: `tests/CMakeLists.txt` (+1 line) and
  `vllm_experts.py` (`use_gpu` gate + adjacent comments). Both minimal and
  additive.
- Open: should the CUDA tree get a `.cu` port of the align kernel for the
  GB10/A100 path? Should `VK_MOE_ALIGN_DEVICE` also be honored inside the
  C++ library (it is binding-layer-only, per the brief)?
