# p2p-kv-donate — fused indexed-KV-to-peer donation (NVIDIA H100 NVL, sm_90)

Issue #36: the donation-side mirror of `p2p_kv_restore` (issue #27). KVAAS
materializes a full all-layer packed scratch tensor before peer DMA
(`pack_pages`, then scratch-to-peer copies with the scratch pinned until
the completion ACK). This kernel fuses the gather and the peer store: one
SM kernel reads arbitrary local paged-KV slots and writes K/V directly
into the layer-major peer-page destination through peer-accessible UVA
pointers — no scratch allocation, no extra local-HBM read/write pass, no
separate peer copy.

## Environment

- sgs-gpu07: 4x NVIDIA H100 NVL (95,830 MiB), driver 580.82.07,
  CUDA 13.0, Ubuntu 24.04
- topology: GPU0<->GPU1 over 12 NVLinks (NV12); GPU0/1<->GPU2/3 over PCIe
- benchmark: `p2p_kv_donate_bench --dst-device 1` — local K/V sources on
  device 0, peer page destinations on device 1 (real NVLink peer writes,
  peer access enabled by the bench), 50-iteration medians
- machine state at first measurement (2026-08-14): GPU0 idle; GPUs 1-3 running
  an unrelated training job at ~100% util, so the NVLink peer-write numbers
  land in a contended GPU1 HBM. The SAME-DEVICE D2D table (idle GPU0,
  `--quick`) is the clean measurement; the peer table demonstrates real
  NVLink writes with the contention caveat.
- **2026-09-17 re-measurement:** idle NVLink pair GPU2<->GPU3 (NV12, no
  unrelated load), full 50-iteration medians — see
  [Idle re-measurement and roof calibration](#idle-re-measurement-and-roof-calibration-sgs-gpu07-2026-09-17)
  below. That section supersedes the peer numbers above and adds the
  measured copy roofs.

## Workload

Qwen3-14B KV geometry (one model layer per donation):

| Parameter | Value |
|---|---|
| Page size (tokens/page) | 64 |
| KV heads | 8 |
| Head dim | 128 |
| Element | BF16 (2 bytes) |
| Bytes per page (one layer) | 0.25 MiB |
| Model layers | 40 |

The constants below are the pre-fit used by the in-code adaptive model
(seed values from the restore's measured line); the measured tables in
the next section are what govern the conclusions. Retune the constants
from the bench output when a truly idle NVLink pair is available.

| Path | Model |
|---|---|
| Direct-plan SM kernel | `max(8.6 us, 4.20 us/MiB)` flat in page count (page grid-stride since #84, no 65535-page cap) |
| Copy-engine two stage | gather kernel `max(8.6, 4.20*MiB)` + one copy `max(20.0, 4.20*MiB)` + `7.37 us` per extra page |

The gather kernel pays no NVLink traffic (local -> local scratch), the
single copy moves the full payload over the link, and each additional page
adds a per-copy overhead — the same split measured on the restore side.
Because the fallback runs the gather kernel regardless, the direct store
wins from ONE page; the copy-engine path remains available (and is
byte-identical) for systems where a GPU cannot store into the peer's
memory directly.

## Measured results (sgs-gpu07, 2026-08-14)

### Same-device D2D over HBM — idle GPU0 (`--quick` medians)

One-shot per-page sweep: `two_stage` = `kv_gather` scratch kernel +
per-page `cudaMemcpyAsync`; `fused` = one direct-store kernel. The
one-shot path pays async-alloc + H2D slot-map/descriptor upload +
async-free per launch — the same overhead class as KVAAS's "pack_pages
each layer" baseline:

| pages | MiB | two_stage us | fused us | speedup |
|---:|---:|---:|---:|---:|
| 1   | 0.25 | 120.4  | 118.3  | 1.02x |
| 4   | 1    | 192.2  | 118.2  | 1.63x |
| 16  | 4    | 481.6  | 118.8  | 4.05x |
| 64  | 16   | 1937.3 | 132.5  | 14.6x |
| 192 | 48   | 7924.6 | 164.9  | 48.1x |

The one-shot fused line is FLAT (~118 us) from 0.25 to 4 MiB: it is
dominated by the per-launch host-side cost, not the kernel. This is
exactly the per-layer overhead the prepared plan removes — plan sweep
across the 40-layer KVAAS workload (`total` includes the one-time
create amortized over 40 layers):

| pages | MiB/lyr | one-shot us/lyr | plan us/lyr | plan prepare | 40-layer total us | plan vs 1-shot |
|---:|---:|---:|---:|---:|---:|---:|
| 1   | 0.25 | 118-138 |  6.0  | 0.112 ms |  352  | 15.6x |
| 16  | 4.00 |   138   |  8.9  | 0.106 ms |  460  | 12.0x |
| 64  | 16.00|   144   | 16.5  | 0.107 ms |  769  |  7.5x |
| 192 | 48.00|   159   | 36.5  | 0.113 ms | 1575  |  4.0x |

The plan moves the entire payload in a single kernel per layer: 36.5 us
for 48 MiB D2D = ~1.34 TB/s effective (96 MiB touched). Per execute:
one kernel launch, zero allocation, zero H2D, zero D2H sync.

### Real NVLink peer writes (GPU0 -> GPU1, GPU1 under 100% load)

Same bench with `--dst-device 1` while GPU1 ran an unrelated job at
100% util / 76 GiB resident. Absolute numbers are therefore INFLATED;
the relative ordering is the content:

| pages | MiB | two_stage us | fused us | speedup |
|---:|---:|---:|---:|---:|
| 1   | 0.25 | 129.5  | 119.9  | 1.08x |
| 16  | 4.00 | 618.4  | 123.6  | 5.01x |
| 64  | 16.00| 2200.4 | 179.7  | 12.2x |
| 192 | 48.00| 8509.5 | 331.1  | 25.7x |

Prepared plan at 48 MiB/layer: 207 us/layer (vs 315 one-shot; create
0.14 ms). The ~110 us/layer one-shot-vs-plan gap is the peer-independent
allocation + upload overhead the plan removes.

### Byte-exactness

The bench asserts `cudaMemcmp` equality of the fused kernel against the
`kv_gather` + per-page copy reference on every page count, and the test
suites check byte-exactness against the host reference for page sizes
16/32/64, head dims 64/128/256, on both the host and CUDA paths.

## Idle re-measurement and roof calibration (sgs-gpu07, 2026-09-17)

Issue #84 asked whether the plan-execute grid under-covers the SMs and
whether a grid-stride loop would lift the 48 MiB point above 1.34 TB/s.
Re-measured on an idle GPU (same-device) and an idle NVLink pair
(`CUDA_VISIBLE_DEVICES=2,3`, GPU2->GPU3, NV12), full 50-iteration medians,
with a **copy-roof calibration** so the achieved number can be compared to
the resource that actually binds a copy.

### Is the 48 MiB point at the roof?

A copy of 48 MiB moves 48 MiB of payload but *touches* 96 MiB (read +
write); the plan additionally reads a small slot map. The right roof is
therefore the device copy roof, not the HBM read peak:

| Path (48 MiB, distinct layer buffers, no L2 reuse) | med us | payload | touched | vs payload copy-kernel roof |
|---|---:|---:|---:|---:|
| D2D streaming copy kernel (grid-stride, `__ldcs`/`__stcs`) | 34.50 | 1.459 TB/s | 2.918 TB/s | 1.00x |
| D2D `cudaMemcpy` | 33.79 | 1.489 TB/s | 2.979 TB/s | 1.02x |
| **donate prepared plan** (page sweep, 192 pages) | **36.86** | **1.365 TB/s** | **2.730 TB/s** | **0.94x (93.6%)** |
| **restore prepared plan** (page sweep, 192 pages) | **35.62** | **1.413 TB/s** | **2.826 TB/s** | **0.97x (96.8%)** |

The plan is within 3-6% of the pure copy kernel and 7-9% of `cudaMemcpy`.
The residual gap is the gather itself — two independent K/V source
streams, a per-token slot indirection, and `dst_k`/`dst_v` interleaving —
not grid coverage. Reported as "% of HBM read peak" it looks like ~40%
(donate) because 1.365 TB/s is measured against the 3.35 TB/s read-only
peak; against the copy roof it is 94%.

### NVLink peer writes (idle GPU2 -> GPU3)

| pages | MiB | two_stage us | fused us |
|---:|---:|---:|---:|
| 1   | 0.25 | 128.3 | 118.2 |
| 16  | 4.00 | 624.1 | 119.2 |
| 64  | 16.0 | 2208.8 | 172.0 |
| 192 | 48.0 | 8595.1 | 324.2 |

Prepared plan, 40-layer KVAAS sweep (median/layer):

| pages | MiB/lyr | one-shot us | plan us/layer | plan create | plan vs one-shot |
|---:|---:|---:|---:|---:|---:|
| 1   | 0.25 | 118.8 |  7.0 | 0.145 ms | 12.8x |
| 16  | 4.00 | 137.2 | 23.3 | 0.111 ms |  5.3x |
| 64  | 16.0 | 177.7 | 73.6 | 0.113 ms |  2.3x |
| 192 | 48.0 | 310.2 | **207.6** | 0.132 ms |  1.47x |

Restore prepared plan at 48 MiB/layer: **207.1 us**.

Peer roof calibration (48 MiB, distinct buffers): a pure peer streaming
copy kernel is **206.7 us (243.6 GB/s)** and peer `cudaMemcpy` is
**195.1 us (258.0 GB/s)**. So at 48 MiB the donate/restore plan
(207.6 / 207.1 us) is **at the SM peer-copy roof (100.2-100.4%)**; only
the copy-engine/`cudaMemcpy` path is ~6% faster. This matches the
`p2p_gather` finding of a ~240-265 GB/s NVLink unidirectional roof on this
box, so the peer plan is not an optimization target either.

### Grid-stride change (issue #84)

Even though it does not move the 48 MiB number, the plan kernels were
switched to stride over PAGES (`grid.y`, capped at `kPlanTargetBlocks = 264`,
blockDim 256), matching the existing `kv_gather` / `kv_scatter` idiom:

- **Correctness:** the old launch capped `grid.y = min(num_pages, 65535)`
  and the kernel did `if (p >= num_pages) return;` with no loop, so any plan
  with **more than 65535 pages silently dropped every page past 65534**.
  The page loop makes coverage independent of page count. Regression tests
  `KvDonatePlan.CoversMorePagesThanGridYLimit` and
  `KvRestorePlan.CoversMorePagesThanGridYLimit` build a 70000-page plan and
  check every byte (they fail on the pre-#84 kernel).
- **Coverage:** the grid is now `(ceil(page_size*units_per_token/256), 264)`
  instead of growing to `num_pages` blocks, so it is bounded and still
  fills the SMs. At the KVAAS point the two shapes are identical
  (`(32, 192)`), which is why the measured 36.5-36.9 us is unchanged.
- A full grid-stride over BOTH axes (inner unit loop) was measured and is
  **~4% slower** at 48 MiB (38.4 us vs 36.6 us) because the inner loop
  executes one iteration but still emits its induction/branch; the
  page-only stride keeps the loop-invariant unit computation outside the
  loop and stays within ~1% of the pre-#84 kernel.

## Dispatch model (for reference)

## Caveats

- The plan's per-layer kernel now uses a grid of
  `(page_size*units_per_token/256, min(num_pages, 264))` blocks and
  grid-strides over pages (issue #84). At the KVAAS point this is the same
  `(32, 192)` grid the original one-block-per-page kernel launched, so the
  36.5 us D2D number is unchanged; it was never SM-starved there. The change
  fixes a real latent bug — `num_pages > 65535` was silently truncated by
  the `grid.y` cap — and bounds the grid for all shapes. The binding
  resource at 48 MiB is the D2D/peer copy roof (measured 34.5 us D2D,
  206.7 us peer), which the plan reaches to 94% / 100%; the restore plan
  shares this shape and the same caveat. See the 2026-09-17 section above.
- Peer-write absolute numbers above were measured against a GPU1 running
  someone else's job (100% util); NVLink peer writes land in its HBM and
  are slowed. Re-measure on an idle pair before quoting absolute peer
  latency; the same-device table is contention-free.
- `execute_via_scratch(...)` measured slower than one-shot fused at
  >=16 MiB/layer (0.64x at 16 MiB, 0.25x at 48 MiB) — it serializes the
  gather kernel and N copies; it exists for byte-exact fallback when
  direct peer stores are unavailable, not as a fast path.
- Direct peer stores require peer access (or IPC mapping) established by
  the CALLER and held until the stream completes; the C ABI and the plan
  keep `execute_via_scratch` as the documented fallback when that is
  unavailable.
- Order is not preserved across pages written concurrently from different
  blocks; completion/publish ordering stays with the caller (same
  contract as KVAAS's pack + DMA).
- The fused kernel is byte-exact with `pack_pages` + peer copy for BF16 /
  FP16 (elem_size == 2) and validated for page sizes 16/32/64 and head
  dims 64/128/256 on both the host and CUDA paths.
- Slot indices must be non-negative and (host-input plan) `< num_slots`;
  repeated source slots are legal in any order (gather semantics).

## Reproduce

```sh
cmake --preset cuda -DVKERNELS_BUILD_BENCHMARKS=ON
cmake --build --preset cuda
# idle GPU, real NVLink peer (dst on GPU1, sources on GPU0):
./build/cuda/meta/benchmarks/p2p_kv_donate_bench --dst-device 1
# same-device reference (D2D over HBM):
./build/cuda/meta/benchmarks/p2p_kv_donate_bench
# 10 iterations instead of 50:
./build/cuda/meta/benchmarks/p2p_kv_donate_bench --dst-device 1 --quick
```

The bench prints the two-stage vs fused table (one-shot forced to the
direct path for an apples-to-apples comparison) and then the prepared-plan
sweep across 40 layers: fused one-shot, adaptive one-shot, prepared plan
(always direct), and plan-via-scratch, each with prepare-once reported
separately.

## Journal

2026-08-14 — Issue #36 implementation: fused indexed-KV-to-peer donation
(host reference + CUDA kernel + C ABI) mirroring the restore's plan API
with the data flow reversed; adaptive one-shot dispatch (forced-direct /
forced-copy-engine for A/B); prepared plan with host / device-int32 /
device-int64 slot-map inputs (device-int64 converted once at create, no
D2H sync); 38 host tests + 27 CUDA C ABI tests; benchmark harness.

2026-08-14 (later) — Benched on sgs-gpu07 (H100 NVL, CUDA 13,
driver 580.82.07): same-device D2D table + 40-layer plan sweep on idle
GPU0 (clean); NVLink peer table measured GPU0->GPU1 while GPU1 was under
100% load from another job (inflated absolute numbers, valid ordering).
All numbers above are MEASURED with the bench binary from the Reproduce
section; the fitted-model predictions they replace matched qualitatively
(fused wins everywhere, no copy-engine crossover) but overstated
one-shot absolute latency by ~6x because the model lacks the per-launch
alloc/upload overhead — the prepared-plan measurement captures it. 27/27
donate C ABI + 38/38 host donate tests pass on GPU; the one failing test
in the repo suite (p2p_kv_restore_c
NonUniqueSlotsReturnsInvalidArgument) fails identically on unmodified
main — pre-existing, unrelated. A CUDA-only compile fix (missing
`#pragma once`) and two donate C ABI test bugs were found and fixed on
GPU; remaining follow-up: re-run the peer table on an idle NVLink pair
and retune the dispatch constants.

2026-09-17 — Issue #84 (grid-stride loop). Switched both plan kernels
(`p2p_kv_donate.cu`, `p2p_kv_restore.cu`) to stride over pages with
`grid.y` capped at `kPlanTargetBlocks = 264`, matching `kv_gather` /
`kv_scatter`. This fixes a latent correctness bug: the old
`grid.y = min(num_pages, 65535)` plus a non-looping `if (p >= num_pages)
return` silently dropped every page past 65534 (regression tests added for
70000 pages on both sides, which fail on the old kernel). Re-measured on
idle hardware with a copy-roof calibration: the 48 MiB plan is 36.9 us
(donate) / 35.6 us (restore) D2D = 1.37-1.41 TB/s payload, i.e. 94-97% of
the measured D2D copy-kernel roof; and 207.6 / 207.1 us over an idle
GPU2->GPU3 NVLink pair = 100% of the 206.7 us peer copy-kernel roof. The
"~40% of HBM" framing counted payload against the HBM READ peak while a
copy touches 2x and is bound by the copy roof; a full both-axis
grid-stride was measured ~4% slower at 48 MiB, so only the page axis
strides. Tests: 39 donate + 27 donate C ABI + 36 restore + 18 restore C
ABI, all pass.
