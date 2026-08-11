# PTX Special Registers and Directives

## Special registers (`.sreg`, read-only, per-thread)

| Register | Meaning |
|---|---|
| `%tid` | Thread index within CTA, `.b32` or `.v4.b32` (x,y,z) |
| `%ntid` | CTA dimensions `.v4.b32` (x,y,z,pad) |
| `%laneid` | Lane index within the warp, 0–31 |
| `%warpid` / `%nwarpid` | Warp index within CTA / number of warps |
| `%ctaid` | CTA index within grid, `.v4.b32` (x,y,z) |
| `%nctaid` | Grid dimensions `.v4.b32` |
| `%smid` / `%nsmid` | Physical SM id / count of SMs |
| `%gridid` | Grid launch id |
| `%is_explicit_cluster` | 1 if the CTA runs under an explicit cluster |
| `%clusterid` / `%nclusterid` | Cluster index / grid cluster count |
| `%cluster_ctaid` / `%cluster_nctaid` | Local CTA index / count *within* the cluster |
| `%cluster_ctarank` / `%cluster_nctarank` | Rank-based variant; only set under explicit cluster |
| `%clock` / `%clock_hi` / `%clock64` | Cycle counters (use `%clock64`; the 32-bit pair wraps) |
| `%pm0..%pm7` / `%pm0_64..%pm7_64` | Performance counters |
| `%envreg<32>` | Environment registers (driver/platform scratch) |
| `%total_smem_size` / `%aggr_smem_size` | Total / aggregate SMEM per CTA |
| `%dynamic_smem_size` | Dynamically-requested SMEM (used in occupancy math) |
| `%reserved_smem_offset_begin/end/cap` | Reserved SMEM range (driver use) |
| `%current_graph_exec` | CUDA graph execution context |

Use `%tid`/`%ctaid`/`%cluster_ctaid` to compute per-element and per-tile
indices; `%laneid` for warp-internal addressing; `%dynamic_smem_size` when
calculating whether a launch will fit.

## Performance directives

Apply to an `.entry`/`.func` to shape occupancy, the launch, and codegen:

- **`.maxnreg N`** — max registers per thread. Lowering raises occupancy but
  may spill to local memory (watch for it).
- **`.maxntid x[,y[,z]]`** — max threads per CTA (an upper bound; the launch
  may be smaller).
- **`.reqntid x[,y[,z]]`** — exact threads per CTA (the launch must match).
- **`.minnctapersm N`** — minimum CTAs per SM (occupancy floor; the compiler
  may cut `.maxnreg` to meet it).

### Cluster directives (Hopper+, `.target` must enable sm90+)

- **`.reqnctapercluster x[,y[,z]]`** — exact CTAs per cluster (the grid is
  divided into clusters of this size). Must be ≤ `.maxclusterrank`.
- **`.explicitcluster`** — the kernel launches under an explicit cluster
  (enables `%cluster_ctaid`/`%cluster_ctarank`).
- **`.maxclusterrank x[,y[,z]]`** — upper bound on cluster size.
- **`.blocksareclusters`** — treat the launch's blocks as clusters.

### `.pragma` strings (hints to the compiler backend)

- `"nounroll"` — do not unroll the following loop.
- `"used_bytes_mask <hex>"` — report which bytes of an output are live
  (affects store coalescing).
- `"enable_smem_spilling"` — allow spilling to shared memory.
- `"frequency <GHz>"` — assumed clock for latency modelling.
- `"mma_throughput <n>"` — assumed Tensor-Core issue rate for scheduling.

## Other directives worth knowing

- **Module:** `.version`, `.target` (e.g. `sm_90a` — `a` suffix enables
  arch-specific instructions), `.address_size 64`.
- **Linkage:** `.extern`, `.visible`, `.weak`, `.common` (default for
  uninitialized globals).
- **Debugging:** `.file`, `.loc`, `.section`, `@@dwarf`.
- **Control-flow hints:** `.branchtargets`, `.calltargets`, `.callprototype`
  (help the assembler with indirect branches/calls).
