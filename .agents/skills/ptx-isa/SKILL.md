---
name: ptx-isa
description: Write and read NVIDIA PTX, the virtual ISA layered between CUDA C++ and machine (SASS) code. Use when authoring inline asm PTX, lowering a kernel to PTX, choosing warp / warpgroup / Tensor-Core instructions (mma, wgmma, tcgen05, shfl), reading or using special registers (%tid, %laneid, %ctaid, %cluster_ctarank), setting performance directives (.maxnreg, .maxntid, .reqnctapersm, .reqnctapercluster), or reasoning about PTX state spaces and the memory-consistency model.
---

# PTX ISA

PTX (Parallel Thread Execution) is a stable, target-independent virtual ISA.
CUDA C++ compiles to PTX, which the driver lowers to a GPU's machine code
(SASS). Writing PTX directly (inline `asm` or a full kernel) gives explicit
control over registers, state spaces, the instruction mix, and the thread
hierarchy — the levers a high-performance kernel actually pulls.

## The thread hierarchy

Every PTX operation has a **scope** — the group of threads that participates in
or is affected by it. Analysing a kernel means knowing each operation's scope
together with its data layout and dispatch mechanism.

- **Thread** — scalar unit of execution; own program counter, own registers,
  identified by a lane ID within its warp.
- **Warp** — 32 threads executing in SIMT (single instruction, multiple
  threads). All lanes issue one instruction together, but each keeps its own
  registers and can be masked off independently — this is what lets the lanes
  of a single warp follow different branches (with a cost per taken path).
- **Warpgroup** — four consecutive warps (128 threads). Introduced by Hopper as
  the unit that issues warpgroup MMA (`wgmma`); on Blackwell its four warps also
  cover the four 32-lane windows of Tensor Memory.
- **CTA** (Cooperative Thread Array, a CUDA "thread block") — the basic unit the
  hardware schedules; runs on one SM and owns a private SMEM allocation. Several
  CTAs can be resident on one SM and share its SMEM budget.
- **Cluster** — a group of co-scheduled CTAs, possibly on different SMs, that
  synchronise at cluster scope and access one another's SMEM (distributed shared
  memory, DSMEM).
- **Grid** — all CTAs (or clusters) of a single kernel launch.

## State spaces

A variable's state space says where it physically lives and who can reach it.
The important ones:

- **register** (`.reg`) — per-thread, fastest. Scalars and per-thread tile
  fragments. Excess use cuts occupancy.
- **special** (`.sreg`) — read-only, per-thread platform registers: thread/block
  indices, lane id, cluster ids, clocks, perf counters, dynamic-SMEM size (see
  [references/special-registers-and-directives.md](references/special-registers-and-directives.md)).
- **constant** (`.const`) — read-only, all threads; fast broadcast.
- **global** (`.global`) — device-wide HBM, persistent across kernels, visible
  to host. The slow common case.
- **shared** (`.shared`) — per-CTA scratchpad on the SM; low latency, high
  bandwidth, but 32 banks and bank conflicts (see the
  [gpu-memory-layout](../gpu-memory-layout/SKILL.md) skill).
- **local** (`.local`) — per-thread, spills to memory; treat as a cost signal.
- **param** (`.param`) — kernel/function parameters and return.

Blackwell adds **Tensor Memory (TMEM)** as MMA accumulator storage; it is
described with named axes in the
[gpu-memory-layout](../gpu-memory-layout/SKILL.md) skill.

## Types

Fundamental scalars (`.s8/.u16/.f32/...`); vectors up to 4 elements; half and
bf16 (`.f16`/`.bf16`); alternate FP formats (`.e4m3`/`.e5m2`/`.f8f6f4`); fixed
point; and packed types. Mixed-precision and block-scaled MMA (MXFP8, NVFP4)
operate on packed low-precision operands with per-block scale factors.

## Instruction families

A high-performance kernel is built from a handful of families. The Tensor-Core
and warp-coordination families are the ones that change across generations; the
full set is in [references/instructions.md](references/instructions.md).

- **Tensor Core MMA** (the dominant FLOP source, ~10× a CUDA core):
  - `mma.sync` (Volta+) — warp scope, fragments in registers.
  - `wgmma.mma_async` (Hopper) — warpgroup scope, async, A from registers or
    SMEM (descriptor), B from SMEM; committed by
    `wgmma.commit_group` / `wgmma.wait_group`.
  - `tcgen05.mma` (Blackwell, 5th gen) — accumulator lives in **TMEM**, not
    registers, cutting register pressure. One designated thread commits via
    `tcgen05.commit...mbarrier::arrive`; the epilogue waits that barrier and
    applies the required tcgen05 fence before reading TMEM.
- **`shfl.sync.{bfly,up,down,idx}`** — warp-level register exchange with no
  SMEM; the workhorse for broadcasts, reductions, and indexing within a warp.
- **`redux.sync.{add,min,max}`** — one-instruction integer warp reduction.
- **Atomics & fences** — `atom{.scope}.{relaxed/acq_rel}` on global/shared;
  `fence.sc.sync` for sequentially-consistent ordering. Async operations go
  through the **async proxy** while ordinary code uses the **generic proxy** —
  PTX requires proxy fences to order accesses between them (see the
  [async-kernel-coordination](../async-kernel-coordination/SKILL.md) skill).
- **`cp.async.cg.shared.global`** (Ampere) — per-thread async load, the
  non-bulk predecessor of TMA, committed by `cp.async.commit_group` /
  `cp.async.wait_group`.
- **Control flow & predication** — most instructions accept a guard predicate
  (`@p` / `@!p`); `setp`/`selp` manipulate predicates; `bra`, `call`,
  `bar.sync` (CTA barrier), `bar.warp.sync`. Divergence costs a pass per taken
  path — prefer warp-uniform control flow or explicit masking.

## Special registers and directives

`%tid`/`%ntid` (thread and block dimensions), `%laneid` (0–31), `%ctaid`/
`%nctaid` (block and grid indices), `%smid`/`%nsmid` (physical SM id), and
under an explicit cluster: `%clusterid`, `%cluster_ctaid`/`%cluster_nctaid`
(local CTA index/count within the cluster),
`%cluster_ctarank`/`%cluster_nctarank`, `%clock64`, performance counters
`%pm0..7`, and the dynamic/aggregate SMEM-size registers. The full table and
the directive reference (`.maxnreg`, `.maxntid`/`.reqntid`, `.minnctapersm`,
`.reqnctapercluster`/`.explicitcluster`/`.maxclusterrank`, `.blocksareclusters`,
`.pragma` strings such as `"nounroll"`, `"used_bytes_mask"`,
`"enable_smem_spilling"`, `"frequency"`, `"mma_throughput"`) are in
[references/special-registers-and-directives.md](references/special-registers-and-directives.md).

## Completion criterion

The PTX you emit uses the correct **state space** and **type** for every
operand, the correct **scope** for every multi-thread instruction, and the
**fences** the consistency model requires (especially around the async proxy).
Special registers are read at the hierarchy level that matches the operation,
and the performance directives reflect the occupancy and cluster shape you
intend.
