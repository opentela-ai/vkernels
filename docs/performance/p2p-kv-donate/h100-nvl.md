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
- machine state at measurement time: GPU0 idle; GPUs 1-3 running an
  unrelated training job at ~100% util, so the NVLink peer-write numbers
  land in a contended GPU1 HBM. The SAME-DEVICE D2D table (idle GPU0,
  `--quick`) is the clean measurement; the peer table demonstrates real
  NVLink writes with the contention caveat. Re-run on an idle machine for
  publication-grade peer numbers.

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
| Direct-plan SM kernel | `max(8.6 us, 4.20 us/MiB)` flat in page count below the 65535-page grid cap |
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

## Dispatch model (for reference)

## Caveats

- The plan's per-layer kernel uses a grid of (page_size*units,
  num_pages) blocks and under-covers the 132 SMs at the KVAAS point
  (192x4x256 threads move 48 MiB), which lands it at ~1.34 TB/s D2D
  rather than peak; a grid-stride loop is a possible future tuning step.
  The restore plan shares this shape.
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
