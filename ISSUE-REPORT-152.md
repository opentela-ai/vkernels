# ISSUE-REPORT — lane issue-152/crossnode-stripe (issues #152 + #153)

## What was done

The striped multi-lane plan implementation was **already merged to main**
(`cad17f6`: `compute_cross_node_kv_stripes`, striped donate/restore plans,
`nccl-stripe` bench mode, 20 striped-plan tests). This lane verified that
state and handled the remaining acceptance criterion.

### PART A — host gate (DONE, green)

On the lane worktree at HEAD (`d668099`, level with `origin/main`):

```
cmake --preset host && cmake --build --preset host   # OK
ctest --preset host                                  # 47/47 passed
```

Includes `cross_node_kv_striped` (layout tiling/remainder/single-lane
collapse, per-lane traffic distribution, byte-exact equality vs the
single-lane bounce oracle, chunk-size-mismatch/too-few-lanes rejection,
stream one-task semantics, graph eager-break). Line coverage on
`src/c/vkernels/comm/cross_node_kv.cpp`: 277/277 (100%), gate green.

### PART B — JSC 2×GH200 HDR-200 on-fabric measurement (BLOCKED, fully staged)

Fabric survey per the lane brief (timeout-10 ssh each, then
`nvidia-smi -L` / `ibstat -l` / `ibv_devices` / `ls /sys/class/infiniband`
/ `sinfo` / `scontrol show node`):

| host | finding |
|---|---|
| ela | login `ela4`; no IB tools, no GPU fabric |
| bristen | Alps (CSCS) GPU nodes, Slingshot — not InfiniBand |
| beverin | CSCS Slurm, partitions mi300 (MI300A) / mi200 (MI250X) — AMD only |
| clariden | GH200 ×4 nodes, but Alps/Slingshot — no IB devices |
| ds6 | `dgx-spark-06.inf.ethz.ch` — single GB10 DGX Spark |
| euler | A100 nodes on `ibfabric2`, mlx4 on login — not GH200 / not HDR |

**No reachable host provides 2-node GH200 + InfiniBand HDR.** Per lane
discipline, the bench was NOT run on a non-HDR fabric and presented as a
JSC result. Instead:

- `meta/scripts/bench_cross_node_nccl_jsc.sh` — staged build + submit
  driver (module loads CUDA/13.0 + NCCL/2.29.7 EasyBuild; `EBROOTNCCL`
  is consumed by `meta/cmake/NcclSupport.cmake`; `cmake --preset cuda`;
  builds target `cross_node_nccl_bench`; then sbatches).
- `meta/scripts/cross_node_nccl_bench_jsc.sbatch` — staged job: 2 nodes,
  1 GPU/rank, `--exclusive`, `srun --mpi=pmix -N2 -n2`, full 256→65536-tok
  sweep of `--iters 101 --stripe 4` (produces `nccl-xfer` / `nccl-pipe` /
  `nccl-stripe` / `d2d-local` rows under identical conditions) + an
  `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET --quick` pass verifying each
  stripe communicator pinned a distinct `mlx5_<l>`. Partition name in the
  sbatch carries a FIXME to verify at the JSC system at submission time.
- `docs/comm-cross-node-kv.md` §8 — added the "Striped rows (issue #152) —
  on-fabric numbers still pending" note with the probe evidence, the staged
  recipe pointer, and the success criterion. **No on-fabric number is
  claimed.**

## Deliverables

1. Conventional commit on the lane branch: `d668099`
   `docs(comm): stage JSC striped-striping bench recipe, record pending on-fabric rows (issue #152)`
2. Doc/perf record: `docs/comm-cross-node-kv.md` §8 updated (pending note;
   the new striped table itself requires the JSC run).
3. Pushed: `git push -u origin issue-152/crossnode-stripe` — done.
4. Draft PR with results table + "Refs #152":
   https://github.com/opentela-ai/vkernels/pull/167
5. Issue comments:
   - #152: https://github.com/opentela-ai/vkernels/issues/152#issuecomment-6110037902
   - #153 (what is staged / what access is needed): https://github.com/opentela-ai/vkernels/issues/153#issuecomment-6110037905
6. This report.

## Evidence

- Host gate: 47/47 ctest pass (run on the lane worktree, this branch).
- No cluster/GPU job ids: no compute-node job was launched — the on-fabric
  criterion requires the JSC testbed, which was unreachable from every
  probed host.
- Fabric-survey commands (reproducible):
  `timeout 10 ssh <host> hostname`; then per host
  `nvidia-smi -L; ibstat -l; ibv_devices; ls /sys/class/infiniband; sinfo -h; scontrol show node <nid>`.

## What remains

1. Run `bash meta/scripts/bench_cross_node_nccl_jsc.sh` on a JSC 2×GH200
   HDR-200 slot (issue #49 setup: 4× mlx5, GPUDirect RDMA, NCCL
   2.29.7+cuda13.0, GPFS /e/scratch bootstrap).
2. Success criterion (unchanged): `nccl-stripe` ≥3× `nccl-xfer` `usefGB`
   at 16384+ toks (expect ~24 → ~70–90 GB/s), all rows `check=ok`; small
   transfers (<16384 tok) verify the single-QP auto-fallback.
3. Record the striped rows in `docs/comm-cross-node-kv.md` §8, undraft PR
   #167, close #152 and #153.
4. If the `NCCL_IB_HCA` per-init pinning proves fragile at JSC, bind lanes
   through libfabric/NIXL endpoints instead; the plan layer is
   transport-agnostic.
