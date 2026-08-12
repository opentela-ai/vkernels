---
name: hip-async-coordination
description: Overlap an AMD GPU kernel's stages — load, compute, store — with LDS double-buffering and the wait-counter model, and coordinate the handoffs so no unit waits on another. Use when copying tiles with vectorized global loads, synchronising with s_waitcnt (vmcnt, lgkmcnt, expcnt counters), building a LDS software pipeline, splitting producer and consumer wavefronts (wavefront specialisation), or running a persistent kernel across multiple GCDs on MI300X.
---

# HIP Async Coordination

A global load or an MFMA operation only *starts* the unit; the hardware runs
it independently. Program order (you issued it before the read) proves the
transfer *started*, not that it *finished* — the consumer may read incomplete
data. The cure is an explicit completion signal: the producer **arrives**
(writes data), the consumer **waits** (drains the relevant counter). On AMD,
this is the `s_waitcnt` model, not NVIDIA's mbarrier. This skill is the
machinery for that, and for using it to keep load, compute, and store busy at
the same time.

## The AMD wait-counter model

AMD GPU memory operations increment hardware counters as they complete.
`s_waitcnt` blocks until the specified counter is **at or below** the given
value. Three independent counters track three different memory pipelines:

| Counter | Full name | Tracks | Operations that increment |
|---|---|---|---|
| `vmcnt` | Vector Memory Count | Global/flat/buffer loads | `global_load_*`, `flat_load_*`, `buffer_load_*` |
| `lgkmcnt` | L/G/K Memory Count | LDS, scalar memory, constants | `ds_read/write`, `s_load_*`, `s_buffer_load_*` |
| `expcnt` | Export Count | Export (pixel/interp) ops | `exp_*`, compute exports |

```
s_waitcnt vmcnt(0)             ; wait until ALL vector memory ops complete
s_waitcnt vmcnt(1)             ; allow 1 in-flight op
s_waitcnt lgkmcnt(0)           ; wait until ALL LDS/scalar ops complete
s_waitcnt vmcnt(0) & lgkmcnt(0)  ; wait on both
```

> **Key insight:** `s_waitcnt vmcnt(N)` blocks until `vmcnt ≤ N`. A value of
> 0 means "all done"; a value of 1 means "allow 1 in-flight op." This enables
> pipelines: wait for batch k to complete while batch k+1 is already in
> flight.

### The basic handoff pattern

```
; Step 1: Producer loads from HBM to VGPRs
global_load_dword v[0], v[addr0], s[desc]   ; vmcnt++
global_load_dword v[1], v[addr1], s[desc]   ; vmcnt++

; Step 2: Wait for all loads to complete
s_waitcnt vmcnt(0)                           ; block until vmcnt==0

; Step 3: Write VGPRs to LDS
ds_write_b128 v[lds_addr], v[0:3]            ; lgkmcnt++
ds_write_b128 v[lds_addr+4], v[4:7]          ; lgkmcnt++

; Step 4: Wait for LDS writes to complete
s_waitcnt lgkmcnt(0)                         ; block until lgkmcnt==0

; Step 5: Consumer reads LDS safely
ds_read_b128 v[8:11], v[lds_addr]            ; data is guaranteed ready
```

> **AMD vs NVIDIA:** There is no TMA. All data movement is explicit: HBM →
> VGPRs (via `global_load`) → LDS (via `ds_write`) → consumer. There is no
> `mbarrier`/phase/arrive/wait model. The handoff is purely through
> `s_waitcnt` on `vmcnt` and `lgkmcnt`. There is no `expect_tx` — you cannot
> track bytes-in-flight per individual load; you track completion counts.

## The `__builtin_amdgcn_s_waitcnt` intrinsic (HIP/C++)

In HIP C++, you rarely write raw assembly. Use the compiler intrinsic:

```cpp
// Wait for all vector memory operations to complete
__builtin_amdgcn_s_waitcnt(0, 0, 0);  // wait vmcnt(0), lgkmcnt(0), expcnt(0)

// Wait for vmcnt(1) — allow 1 in-flight global load
__builtin_amdgcn_s_waitcnt(1, 0, 0);

// Wait for lgkmcnt(0) — ensure LDS writes are visible
__builtin_amdgcn_s_waitcnt(0, 0x7F, 0);  // max value (~127) for lgkmcnt
```

**Arguments:** `(vmcnt, lgkmcnt, expcnt)` — each is the maximum allowed count.
Values beyond the hardware counter width are clamped to the max (effectively
"don't care").

### The four standard waits in HIP kernels

```cpp
// 1. HBM → VGPR: wait for all global loads (after TILE_LOAD)
__builtin_amdgcn_s_waitcnt(0, 0, 0);

// 2. VGPR → LDS: wait for LDS writes to complete before reading
__builtin_amdgcn_s_waitcnt(0, 0x7F, 0);  // lgkmcnt drain

// 3. LDS → VGPR: wait for LDS reads (before MFMA uses them)
__builtin_amdgcn_s_waitcnt(0, 0, 0);

// 4. VGPR → HBM: wait for global stores before reusing buffer
__builtin_amdgcn_s_waitcnt(0, 0, 0);
```

## LDS double-buffering (software pipeline)

Overlap is the payoff once a kernel is compute-bound. With ≥2 LDS stages:

```
time t:   MFMA reads LDS stage 0   |  global load fills LDS stage 1
time t+1: MFMA reads LDS stage 1   |  global load fills LDS stage 0
```

While the Matrix Core reads stage k, the copy engine writes stage k+1 and the
epilogue processes k−1.

```cpp
// Simplified LDS double-buffering
__shared__ half lds_a[2][TILE_M][TILE_K];  // 2 stages
__shared__ half lds_b[2][TILE_K][TILE_N];

int stage = 0;
// Prefetch stage 0
load_tile_to_lds(lds_a[0], lds_b[0], k=0);
__syncthreads();  // ensures LDS writes visible (also drains lgkmcnt)

for (int k = 0; k < K; k += B_K) {
    // Issue load for next stage (if not last iteration)
    if (k + B_K < K) {
        load_tile_to_lds(lds_a[1-stage], lds_b[1-stage], k + B_K);
    }

    // Compute current stage (MFMA reads LDS)
    mfma_compute(lds_a[stage], lds_b[stage]);
    // __syncthreads() might be needed depending on load/compute wavefront split

    // If next stage load was issued, wait for it to arrive in LDS
    if (k + B_K < K) {
        __syncthreads();  // or: wait only on lgkmcnt
    }

    stage = 1 - stage;  // toggle
}
```

### Fine-grained wait: skip the `__syncthreads`

`__syncthreads()` is a heavy barrier — it blocks all wavefronts in the
workgroup. For wavefront-specialised pipelines, use targeted `s_waitcnt`:

```cpp
// Producer wavefront: writes to LDS
ds_write_b128(v[lds_addr], v[vgpr_data]);  // lgkmcnt++
__builtin_amdgcn_s_waitcnt(0, 0, 0);       // optional: consume the lgkmcnt

// Consumer wavefront: waits only for lgkmcnt, not for other wavefronts
__builtin_amdgcn_s_waitcnt(0, 0x7F, 0);    // drain just LDS traffic
// Now reads LDS — other wavefronts may still be loading to other stages
```

## Wavefront specialisation

Instead of one wavefront doing load *and* compute, dedicate whole wavefronts
to roles — **producer** wavefronts issue global loads and LDS writes,
**consumer** wavefronts run the MFMA. This may temporarily increase resource
use and lower occupancy, but it is the structure that enables the deep
multi-stage overlap that pays off later.

On AMD, a workgroup can contain up to 16 wavefronts (1024 threads / 64).
Typical specialisation:

- **Workgroup 1 (compute heavy):** 8 consumer wavefronts + 2 producer
  wavefronts.
- **Workgroup 2 (balanced):** 4 consumer + 4 producer.
- **Workgroup 3 (memory heavy):** 4 consumer + 8 producer.

Producers use `global_load` + `ds_write`; consumers use `ds_read` + `v_mfma`.
Coordination between them uses `s_waitcnt lgkmcnt(0)` on the consumer to
ensure the producer's LDS writes are visible.

> **AMD limitation:** No direct equivalent of NVIDIA's warpgroup MMA
> (`wgmma`). Each MFMA occupies one 64-thread wavefront. For larger tiles,
> multiple consumer wavefronts compute adjacent output sub-tiles and
> accumulate independently. There is no "cooperative MMA" across wavefronts
> within a workgroup — they must coordinate through LDS.

## Persistent kernels

Launch a fixed pool of long-lived workgroups that each compute several output
tiles in a loop, cutting launch and setup overhead:

```cpp
__global__ void persistent_gemm(...) {
    // Grid-stride loop: each workgroup computes multiple tiles
    for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        int m_tile = tile_idx / N_TILES;
        int n_tile = tile_idx % N_TILES;
        // ... full GEMM on this tile, same as Steps 1-5 ...
    }
}
```

This is the simple grid-stride variant. More advanced work-stealing requires a
global work queue with `atomicAdd` to claim work items.

## Multi-GCD considerations (MI300X)

MI300X has 8 GCDs. For data-parallel GEMM across GCDs:

- Each GCD gets a partition of the output (e.g., 1/8 of M rows).
- Each GCD runs the same kernel on its partition, loading its share of A/B.
- Cross-GCD communication for the forward pass is zero (each GCD loads its A
  rows and all of B). For the backward pass, an allreduce across GCDs is
  needed.
- Use `hipSetDevice(gcd_id)` before launching and manage separate streams per
  GCD.

See [references/multi-gcd-persistent.md](references/multi-gcd-persistent.md)
for full MI300X multi-GCD patterns.

## Four common handoffs

1. **HBM → VGPRs.** `global_load` increments `vmcnt`. Consumer reads VGPRs
   after `s_waitcnt vmcnt(0)` (or `__builtin_amdgcn_s_waitcnt(0, 0, 0)`).

2. **VGPRs → LDS.** `ds_write` increments `lgkmcnt`. Consumer reads LDS after
   `s_waitcnt lgkmcnt(0)` (or `__builtin_amdgcn_s_waitcnt(0, 0x7F, 0)`).

3. **LDS → MFMA.** `ds_read` increments `lgkmcnt` (reads count too).
   Consumer MFMA reads VGPR operands after `__builtin_amdgcn_s_waitcnt(0, 0, 0)`.

4. **MFMA → epilogue (HBM store).** The epilogue reads accumulator VGPRs
   after the MFMA completes. MFMA is a synchronous instruction — no counter
   needed within a wavefront. But if multiple wavefronts wrote to the output
   tile, `__syncthreads()` or per-wavefront coordination is needed before the
   store.

## Completion criterion

Every asynchronous read is preceded by the matching `s_waitcnt` at the correct
counter value; every reused LDS buffer waits on `lgkmcnt(0)` before the next
write; the LDS layout, the swizzle, and the consumer MFMA agree; and the
pipeline keeps the copy engine, the Matrix Cores, and the store path active
rather than serialising `load → wait → compute → wait → store`.
