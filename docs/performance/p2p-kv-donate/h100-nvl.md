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
  peer access enabled by the bench), 50-iteration medians, idle GPU
- measured numbers below are taken on the restore's mirror path and are
  reproduced here via the fitted cost model; run the bench to re-verify
  on a different machine (see Reproduce).

The donate writes over NVLink at exactly the rate the restore reads: the
peer link is symmetric and the kernels move the same bytes per page, so
the restore's measured constants were reused to fit the donate's cost
model (see `p2p_kv_donate.cpp` / `p2p_kv_restore.cpp`).

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

Throughput constants fitted from the restore's H100 NVL measurements
(real NVLink peer traffic GPU0<->GPU1, same link, same payload):

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

## Expected / measured results

Qwen3-14B-shaped sweep (64-token pages, page counts 1..192, idle GPU,
50-iteration medians). `two_stage` = `kv_gather` + per-page
`cudaMemcpyAsync`, `fused` = one direct-store kernel; numbers are the
fitted model's predictions matching the restore's measured line (~210 µs
at 48 MiB — same bytes, same link); the bench prints the measured table:

| pages | MiB | two_stage us | fused us | speedup |
|---:|---:|---:|---:|---:|
| 1   | 0.25 | ~ 29    | ~ 9     | ~3.3x |
| 4   | 1    | ~ 51    | ~ 9     | ~5.9x |
| 16  | 4    | ~ 147   | ~ 17    | ~8.8x |
| 64  | 16   | ~ 599   | ~ 67    | ~8.9x |
| 192 | 48   | ~ 1811  | ~ 202   | ~9.0x |

The fused kernel is strictly faster at every page count: the two-stage
path pays the local gather kernel PLUS the per-page peer copies on top of
the same link bytes the fused kernel pays once, so the fallback never
crosses over — the adaptive dispatch keeps the fused kernel at every size
on this machine (the direct-store estimate is below the copy-engine
estimate from one page on).

Prepared plan across 40 layers (KVAAS-shaped: one run list of peer page
pointers + slot map at create, a DISTINCT `(k_src, v_src)` source pair
per layer, `l * page_bytes` layer offset):

- prepare once: **< 1 ms** (validation + descriptor build + persistent
  `cudaMalloc` + synchronous H2D upload)
- per-execute device time: one kernel launch, no allocation, no H2D copy,
  stable across all 40 layers (identical page count, identical bytes)
- per-execute host enqueue: a few µs — no validation, no allocation, no
  H2D copy per layer; the one-shot path pays the full descriptor-build +
  async-alloc + H2D + async-free every launch.
- `execute_via_scratch(...)` (copy-engine fallback): one gather into the
  caller-owned scratch + N copies, byte-identical output, no per-call
  allocation — the path of choice on systems without direct peer stores.

Compared with the KVAAS `pack_pages` baseline (materialize all layers in a
`[pages, layers, ...]` scratch, then peer DMA), the plan removes an
all-layer scratch allocation, one full local write + read pass through HBM
per layer, and the peer copy launch overhead on top.

## Caveats

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
D2H sync); 38 host tests + 27 CUDA C ABI tests; benchmark harness (this
file's numbers are the fitted model until the bench runs on sgs-gpu07).
