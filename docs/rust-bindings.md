# Rust bindings

The Rust workspace under [`src/rust/`](../src/rust/) is the Rust counterpart of the
Python bindings under [`src/python/vkernels/`](../src/python/vkernels/): it drives the same
kernels and communication primitives from [`src/c/vkernels/`](../src/c/vkernels/)
with an idiomatic, safe API.

It has two layers:

* **`vkernels-sys`** ([`src/rust/vkernels-sys/`](../src/rust/vkernels-sys)) — the
  unsafe FFI layer. Its `build.rs` configures the repository's own CMake
  build (the single source of truth for the library) with the `cmake` crate
  and links the resulting static library. The C ABI it binds to lives in
  [`src/c/vkernels/capi/`](../src/c/vkernels/capi/) — `extern "C"` wrappers
  around the C++ API that fold every C++ exception into a status code plus a
  thread-local message (exceptions cannot cross the ABI). The C shim is
  compiled into the `vkernels` static library itself, so there is nothing
  extra to link.
* **`vkernels`** ([`src/rust/vkernels/`](../src/rust/vkernels)) — the safe,
  idiomatic API built on `vkernels-sys`, mirroring the Python modules
  `vkernels.kernels`, `vkernels.comm` and `vkernels.core` function for
  function. Contract violations (length mismatches, empty inputs, invalid
  topology, out-of-capacity runs, …) surface as
  [`Error`](https://docs.rs/vkernels/latest/vkernels/enum.Error.html)
  (`Result`), exactly like the `ValueError`s of the Python bindings and the
  `std::invalid_argument`s of the C++ library.

## Building and testing

The crate is host-buildable on any machine with a Rust toolchain — the
library is compiled by CMake in the host (CPU-reference) configuration by
default, so no CUDA toolkit is required:

```sh
cargo test --manifest-path src/rust/Cargo.toml    # from the repository root
# or: make rust-test
# or, via the CMake presets (registers `rust_bindings` with CTest):
cmake --preset rust
cmake --build --preset rust
ctest --preset rust
```

With a CUDA toolkit, set `VKERNELS_RUST_CUDA=ON` when building to compile
the CUDA kernels too (the same entry points then drive the GPU path):

```sh
VKERNELS_RUST_CUDA=ON cargo test --manifest-path src/rust/Cargo.toml
```

`vkernels::has_cuda()` reports how the linked library was built.

### Serving-runtime CUDA ABI

The CUDA serving boundary (`libvkernels_c.so`) is exposed separately from the
host/static `vk_*` API. Raw declarations live in
`vkernels_sys::serving` behind the `serving-c-abi` feature. Downstream serving
runtimes should enable `external-c-abi`, which implies those declarations,
skips the bundled CMake build, and links a prebuilt `libvkernels_c.so` from
`VKERNELS_LIB_DIR` (or the system linker path). `KVAAS_VKERNELS_LIB_DIR` is
accepted temporarily as a migration alias.

`vkernels_serving_abi_version()` must equal
`VKERNELS_SERVING_ABI_VERSION` before a runtime uses serving structs or opaque
handles. The raw crate owns ABI layout; consumers own higher-level stream,
allocation, and completion lifetimes.

## Usage

```rust
use vkernels::kernels;
use vkernels::comm;
use vkernels::core::{Device, Stream};

// Element-wise -----------------------------------------------------------
let a = [1.0_f32, 2.0];
let b = [3.0_f32, 4.0];
let mut out = [0.0_f32; 2];
kernels::add(&a, &b, &mut out)?;                 // out == [4.0, 6.0]
kernels::scale(&a, 2.0, &mut out)?;              // out == [2.0, 4.0]
kernels::relu(&[-1.0, 2.0], &mut out)?;          // out == [0.0, 2.0]

// Reductions -------------------------------------------------------------
kernels::sum(&[1.0, 2.0, 3.0])?;                 // 6.0
kernels::max(&[1.0, 5.0, 3.0])?;                 // 5.0

// SGEMM: C = alpha * A @ B + beta * C -------------------------------------
let a = [1.0, 2.0, 3.0, 4.0];                    // 2x2 row-major
let b = [1.0, 1.0];                              // 2x1
let mut c = [0.0_f32; 2];
kernels::gemm(2, 1, 2, 1.0, &a, &b, 0.0, &mut c)?;  // c == [3.0, 7.0]

// Ring all-reduce (host simulation) --------------------------------------
let a = vec![1.0_f32, 2.0];
let b = vec![3.0_f32, 4.0];
let out = comm::ring_allreduce(&[a, b])?;        // two ranks, each [4.0, 6.0]

// Compute/communication overlap -------------------------------------------
let ex = comm::OverlapExecutor::new();
let res = ex.run(4, |i| i as i32 * 2, |_, _| {})?;   // compute_count == comm_count == 4

// P2P run-list gather -----------------------------------------------------
let src: Vec<u8> = (0..6).collect();
let mut dst = vec![0u8; 6];
comm::p2p_gather_runs(&mut dst, &[src.as_ptr() as usize], &[2], &[4], None)?;
assert_eq!(dst, vec![0, 0, 0, 1, 2, 3]);

// Streams ------------------------------------------------------------------
let s = Stream::new();
s.submit(move || println!("on the stream"))?;
s.wait();

# Ok::<(), vkernels::Error>(())
```

## API reference

### `vkernels::kernels`

#### Element-wise, reduction, GEMM

| Function | Semantics | Notes |
|---|---|---|
| `add(a, b, out) -> Result<()>` | `out = a + b` | `a.len() == b.len() == out.len()` |
| `scale(x, alpha, out) -> Result<()>` | `out = alpha * x` | |
| `relu(x, out) -> Result<()>` | `out = max(x, 0)` | |
| `sum(x) -> Result<f32>` | float32-accumulated sum | `Err` on empty input |
| `max(x) -> Result<f32>` | maximum | `Err` on empty input |
| `gemm(M, N, K, alpha, A, B, beta, C) -> Result<()>` | `C = alpha*A@B + beta*C` | `A` is `M*K`, `B` is `K*N`, `C` is `M*N` |

Buffers are `&[f32]` / `&mut [f32]`; `out` is always written in place (never
silently copied). Contract violations return
`Err(Error::InvalidArgument)`.

#### bf16 helpers (`kernels::bf16`)

There is no `half` crate dependency: all bf16 / MXFP4 APIs below take and
return raw `u16` bit patterns. Convert to/from `f32` with
[`kernels::bf16`](https://docs.rs/vkernels/latest/vkernels/kernels/bf16/index.html):

| Function | Semantics |
|---|---|
| `bf16::to_f32(bits: u16) -> f32` | lossless zero-extend |
| `bf16::from_f32(v: f32) -> u16` | round to nearest-even (mirrors `float_to_bf16`) |
| `bf16::to_f32_slice(bits) -> Vec<f32>` | vectorised `to_f32` |
| `bf16::from_f32_slice(v) -> Vec<u16>` | vectorised `from_f32` |

#### gfx942 primitives

| Function | Semantics | Notes |
|---|---|---|
| `use_async_copy_default() -> bool` | host: `false` if `K3_NO_ASYNC=1` | reads the env once |
| `mfma_f32_16x16x16bf16(c, a, b)` | 16×16×16 MFMA, `c:&mut [f32;4]`, `a,b:&[u32;2]` | `a`/`b` are 2×bf16 packed in a `u32` |
| `fp4_to_bf16_dequant(packed, out, scale) -> Result<()>` | E2M1 → bf16 dequant | `out.len() == 2*packed.len()` |
| `unsafe direct_lds_fill_bf16(lds, m, n, lda, rs)` | raw `void*` memcpy | caller guarantees `lds`/`rs` validity |

#### bf16 GEMM & MLA attention

| Function | Semantics | Notes |
|---|---|---|
| `gemm_bf16(M, N, K, a, b, c) -> Result<()>` | `C = A@B`, bf16 in → bf16 out | `a`=`M*K`, `b`=`K*N`, `c`=`M*N` bit patterns |
| `gemm_bf16_config(M, N, K) -> (i32,i32,i32,i32)` | `(bm,bn,bk,rstep)` tile config | `bk==64`; `M<=64`→`(16,16,64,64)` |
| `mla_fwd(b,h,q,k_c,k_pe,v_c,s_q,s_kv,q_s,kv_s,l,rhd,scale,out) -> Result<()>` | MLA attention, causal from `q_s`/`kv_s` | `q`=`b*h*s_q*(l+rhd)`, etc. |
| `mla_config(s_q, kv_lora_rank, qk_rope_head_dim) -> (i32,i32,i32)` | `(wg_s_q,bm_kv,rstep_kv)` | `s_q<=8`→`(1,64,64)` |

#### KDA (gated delta attention)

| Function | Semantics | Notes |
|---|---|---|
| `kda_layer_norm_gated(x, w, gate, out, N, D, eps) -> Result<()>` | gated RMSNorm × SiLU(gate) | `x/gate`=`N*D`, `w`=`D` |
| `kda_gate_chunk_cumsum(g, B,H,n_chunks,chunk_size, intra, inter) -> Result<()>` | intra inclusive, inter **exclusive** log-cumsum | `n_chunks,chunk_size>0` |
| `kda_naive_delta_rule_fwd(q,k,v,g,beta,B,H,S,D,out) -> Result<()>` | full O(S²) delta-rule forward | `D>0` |
| `kda_delta_rule_fwd(q,k,v,g,beta,B,H,S,D,chunk_size,out) -> Result<()>` | chunked forward, `S%chunk_size==0` | drives intra+inter |
| `kda_delta_rule_intra(q,k,v,g,beta,intra_log,inter_state,B,H,S,D,chunk_size,chunk_idx,u) -> Result<()>` | within-chunk solve `u` for chunk `chunk_idx` | `inter_state`=`B*H*(n_chunks+1)*D*D`, read-only |
| `kda_delta_rule_inter(intra_log,inter_state,B,H,S,D,chunk_size,chunk_idx) -> Result<()>` | cross-chunk state update `C_c += u_c C_{c-1}ᵀ` | writes `inter_state[...,chunk_idx+1]` |
| `kda_gla_fwd_o(q,k,v,g,beta,B,H,S,D,out) -> Result<()>` | gated-linear-attention output only | `D>0` |
| `kda_pack_bitmatrix(bits, packed, n_bits) -> Result<()>` | MSB-first bit packing | `n_bits<=bits.len()`, `packed>=(n_bits+7)/8` |

#### MoE orchestration (MXFP4)

| Function | Semantics | Notes |
|---|---|---|
| `mxfp4_moe_quant(a, packed, scales, M, hidden, group_size) -> Result<()>` | bf16→MXFP4 per-group quant | `hidden%group_size==0`, `%2==0` |
| `mxfp4_moe_sort(a, sorted_ids, M, hidden, top_k, a_sorted) -> Result<()>` | gather + zero-pad padding rows | padding rows have `sorted_ids[r] >= M*top_k` |
| `mxfp4_moe_sort_scales(scales, sorted_ids, M, n_groups, top_k, scales_sorted) -> Result<()>` | gather + zero-pad scales | `EM = sorted_ids.len()` |
| `mxfp4_moe_scatter_reduce(partial, topk_w, M, width, top_k, out) -> Result<()>` | weighted scatter-reduce | `out`=`M*width`, caller zero-inits |
| `mxfp4_moe_scatter_reduce_q(partial_q, partial_s, topk_w, M, width, top_k, group_size, out) -> Result<()>` | scatter-reduce + per-group dequant | `width%2==0`, `%group_size==0` |
| `moe_align_block_size(topk_ids, M, top_k, block_size, num_experts) -> Result<(Vec<i32>, Vec<i32>, usize)>` | block-aligned sort (allocates output) | returns `(sorted_ids, expert_ids, em)` |
| `moe_align_block_size_max_em(M, top_k, block_size, num_experts) -> usize` | upper bound on `em` (pure) | `((M+BS-1)/BS + num_experts) * BS` |
| `fused_moe_mxfp4(a, w13, b13, w2, b2, a_scales, topk_ids, topk_scales, act_scratch, out, M, hidden, ispp, group_size, activation) -> Result<()>` | end-to-end MXFP4 MoE | `EM%16==0`; `b13`/`b2` optional via `Option<&[f32]>` |

`MoeActivation` is `pub enum MoeActivation { SwiGLU, SiTU }` (`#[derive(Clone,Copy,PartialEq,Eq,Debug)]`);
`fused_moe_mxfp4` infers `num_experts` from `w13.len() / (2*ispp*(hidden/2))`, and `em` from `topk_ids.len() / top_k`.

Example — a tiny MXFP4 MoE:

```rust
use vkernels::kernels::{self, MoeActivation};

let (m, hidden, ispp, gs, e, top_k) = (16, 128, 64, 32, 1, 1);
let a = vec![1.0_f32; m * hidden];
let w13 = vec![1.0_f32; e * 2 * ispp * (hidden / 2)];
let w2 = vec![1.0_f32; e * ispp * hidden];
let a_scales = vec![127u8; m * hidden / gs];
let topk_ids = vec![0i32; m * top_k];
let topk_scales = vec![1.0_f32; m * top_k];
let mut act_scratch = vec![0.0_f32; m * ispp];
let mut out = vec![0.0_f32; m * hidden];
kernels::fused_moe_mxfp4(
    &a, &w13, None, &w2, None, &a_scales,
    &topk_ids, &topk_scales, &mut act_scratch, &mut out,
    m, hidden, ispp, gs, MoeActivation::SwiGLU,
)?;
# Ok::<(), vkernels::Error>(())
```

All K3-family functions take flat slices with explicit dimension parameters
(Rust cannot infer shapes from a `numpy` array the way the Python bindings
do) and mirror the `VK_EXPECTS` contracts of the C++ kernels as
`Err(Error::InvalidArgument)`. `moe_align_block_size` is the sole function
that allocates internally and returns an owned `Vec`.

### `vkernels::comm`

| Function / type | Semantics |
|---|---|
| `Topology` | `rank`/`world`/`next`/`prev` ring slot |
| `ring_rank(rank, world) -> Result<Topology>` | one ring slot |
| `build_ring_topology(world) -> Result<Vec<Topology>>` | one entry per rank |
| `BlockingQueue` | thread-safe queue of float32 chunks (`push`/`pop`/`close`/`closed`) |
| `MockChannel::new(out, in)` | in-process channel (`send`/`recv`/`closed`) |
| `make_ring_channels(world)` | `world` channels in a ring (`r` sends to `r+1`) |
| `ring_allreduce_rank(local, rank, world, next, prev)` | one rank's all-reduce, `local` summed in place |
| `ring_allreduce(locals) -> Result<Vec<Vec<f32>>>` | all ranks simulated in one process |
| `OverlapExecutor::run(iters, compute, comm) -> Result<OverlapResult>` | compute on stream A, comm on stream B, per-iteration future |
| `stage_runs_1d(dst, src_ptrs, dst_offsets, lengths)` | validate + stage 1-D runs (`StagedRun1D`) |
| `stage_runs_2d(dst, runs)` | validate + stage 2-D tiles (`StagedRun2D`) |
| `p2p_gather_runs(dst, src_ptrs, dst_offsets, lengths, stream)` | single-launch 1-D gather |
| `p2p_gather_runs_2d(dst, runs, stream)` | single-launch strided-tile gather |
| `memcpy_peer_batch_async(dst, src_ptrs, dst_offsets, lengths, stream)` | legacy per-run seam (benchmarks) |

`Gather2DRun` / `StagedRun1D` / `StagedRun2D` mirror the C++ structs; run
fields are `usize` byte addresses / offsets, matching the raw-address API of
the Python bindings (e.g. `arr.as_ptr() as usize`). The p2p functions
validate the run list up front (capacity, disjoint output runs, src/dst
non-overlap) and return `Err(Error::InvalidArgument)` on violation; a
`num_runs == 0` list is a valid no-op.

When `stream` is `None` the p2p work runs to completion before returning;
with a `core::Stream` the work is enqueued and the caller owns ordering and
completion via `stream.wait()`.

**Lifetime contract:** with a `stream`, `dst` and every source must stay
alive until `stream.wait()` completes the enqueued copy — the borrow checker
cannot see the asynchronous use, so this is on the caller, exactly as in the
C++ and Python APIs.

### `vkernels::core`

* `Device::new(index)` — `index()`, `set_current()`, `sync()`,
  `supports_peer(&other)` (all no-ops on a host build; real device semantics
  under CUDA), `PartialEq`. `default_device()` returns `Device::new(-1)`.
* `Stream::new()` — `submit(task)`, `wait()`, `submitted()`; one worker
  thread per stream, in-order execution within a stream, concurrency across
  streams. Tasks are `FnOnce() + Send + 'static`; a panicking task is caught
  and does not abort the process. Dropping a stream first runs every
  still-queued task.

## Error model

`vkernels::Error` has four variants mirroring `vkernels::Code`:
`InvalidArgument` (the C++ `VK_EXPECTS` contract checks, like Python's
`ValueError`), `OutOfRange`, `Unsupported` and `Internal` (the `VK_ENSURES`
invariant checks and allocation failures). Every fallible call returns
`Result`; only handle *construction* (`Device::new`, `Stream::new`, ...)
panics, and only on an out-of-memory failure of the C++ side.

## Testing

```sh
cargo test --manifest-path src/rust/Cargo.toml   # unit + integration tests
ctest --preset rust                          # same, via CTest (rust_bindings)
```

The tests live inline in the crate modules (`core.rs`, `kernels.rs`,
`comm.rs`) and mirror `tests/python/` scenario for scenario: contract
violations, cross-thread ring all-reduce, overlap ordering, stream
semantics and the p2p validation rules. The crate builds and tests on any
machine — the CPU-reference path always works, the same philosophy as the
rest of the repository.

## Layout

```
src/rust/                     # Rust workspace
├── Cargo.toml                #   members: vkernels-sys, vkernels
├── vkernels-sys/             #   unsafe FFI (build.rs drives CMake, links src/c)
│   ├── build.rs              #     cmake crate + link lines (CUDA opt-in)
│   └── src/lib.rs            #     extern "C" declarations of the C ABI
└── vkernels/                 #   safe API
    ├── src/{lib,core,kernels,comm}.rs
    └── tests/                #   integration tests (mirror tests/python)
src/c/vkernels/capi/           # C ABI shim compiled into the vkernels library
├── capi.hpp                  #   the C interface (also usable from plain C)
└── capi.cpp                  #   exception -> status-code translation
```
