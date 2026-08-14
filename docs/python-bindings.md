# Python bindings

The Python package under [`src/python/vkernels/`](../src/python/vkernels/) is the documented
way to drive the kernels and communication primitives in
[`src/c/vkernels/`](../src/c/vkernels/) from Python. It has two layers:

* **Discovery / CLI** — [`vkernels.discovery`](../src/python/vkernels/discovery.py)
  scans the C++/CUDA sources and [`vkernels.cli`](../src/python/vkernels/cli/) backs
  the `vkl` command (`python3 -m vkernels.cli list`). No third-party
  dependencies.
* **Bindings** — [`vkernels.kernels`](../src/python/vkernels/kernels.py),
  [`vkernels.comm`](../src/python/vkernels/comm.py) and
  [`vkernels.core`](../src/python/vkernels/core.py) implement the callable
  interface to the kernels. These require numpy.

This document covers the bindings.

## Two backends, one interface

The public modules dispatch to one of two interchangeable backends:

| Backend | Implementation | When used |
|---|---|---|
| **compiled** | [`vkernels._core`](../src/python/vkernels/_core.cpp) — a pybind11 extension that calls the C++ library directly | whenever the extension can be loaded (see below) |
| **fallback** | [`vkernels._fallback`](../src/python/vkernels/_fallback.py) — pure-Python (numpy) reference mirroring the C++ CPU oracles | when the extension is not built / not loadable |

The active backend is exposed as `vkernels.backend` (`"compiled"` or
`"fallback"`). Both backends are bit-identical for element-wise ops, GEMM
and the reductions (the fallback replicates the C++ float32 operation order),
so tests and user code do not need to care which one is loaded. The compiled
backend is the fast path and the one that exercises the real kernels; the
fallback keeps the package importable and testable on any laptop — the same
"CPU reference always works" philosophy as the C++ side.

The loader ([`vkernels/_backend.py`](../src/python/vkernels/_backend.py)) resolves
the extension in this order:

1. an importable `vkernels._core` (installed copy, or `PYTHONPATH` pointing
   at a build tree), then
2. a freshly built `build/*/python/vkernels/_core*.so` under the repository
   root (the default CMake output location), newest first, then
3. the pure-Python fallback.

## Building the compiled backend

The extension is opt-in, matching the `VKERNELS_BUILD_PYTHON` CMake option:

```sh
cmake --preset python        # host build + bindings
cmake --build --preset python
ctest --preset python        # includes the Python test suite
```

This compiles [`src/python/vkernels/_core.cpp`](../src/python/vkernels/_core.cpp) into
`build/python/python/vkernels/_core.cpython-*.so`. pybind11 is found via
`find_package` or fetched at configure time (pinned release); a CUDA build
(`--preset cuda -DVKERNELS_BUILD_PYTHON=ON`) links the same extension against
the CUDA-enabled library, so the Python API automatically becomes the entry
point to the GPU kernels as well.

The Python test suite runs under CTest as `python_bindings`; point it at an
interpreter that has numpy if the system Python does not:

```sh
cmake --preset python -DVKERNELS_PYTHON_EXECUTABLE=/path/to/venv/bin/python
```

Without a build, everything still works through the fallback:

```sh
python3 -c "import numpy, sys; sys.path.insert(0, 'src/python'); import vkernels; print(vkernels.backend)"  # fallback
```

## Usage

```python
import numpy as np
import vkernels
from vkernels import kernels, comm, core

print("backend:", vkernels.backend)          # "compiled" or "fallback"

# Element-wise kernels -----------------------------------------------------
kernels.add([1.0, 2.0], [3.0, 4.0])          # array([4., 6.], dtype=float32)
kernels.scale([1.0, 2.0], 2.0)               # array([2., 4.], dtype=float32)
kernels.relu([-1.0, 2.0])                    # array([0., 2.], dtype=float32)

# Reductions ----------------------------------------------------------------
kernels.sum([1.0, 2.0, 3.0])                 # 6.0
kernels.max([1.0, 5.0, 3.0])                 # 5.0

# SGEMM: C = alpha * A @ B + beta * C --------------------------------------
A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
B = np.array([[1.0], [1.0]], dtype=np.float32)
kernels.gemm(A, B)                           # array([[3.], [7.]], dtype=float32)

# Ring all-reduce (host simulation) ----------------------------------------
a = np.array([1.0, 2.0], dtype=np.float32)
b = np.array([3.0, 4.0], dtype=np.float32)
comm.ring_allreduce([a, b])                  # [array([4., 6.]), array([4., 6.])]

# Compute/communication overlap ---------------------------------------------
ex = comm.OverlapExecutor()
res = ex.run(4, lambda i: i * 2, lambda i, v: None)   # Result(4, 4)

# P2P run-list gather -------------------------------------------------------
src = np.arange(6, dtype=np.uint8)
dst = np.zeros(6, dtype=np.uint8)
addr = src.__array_interface__["data"][0]
comm.p2p_gather_runs(dst, [addr], [2], [4])  # src[0:4] -> dst[2:6]: [0, 0, 0, 1, 2, 3]

# Streams --------------------------------------------------------------------
s = core.Stream()
s.submit(lambda: None)
s.wait()
```

## API reference

### `vkernels.kernels`

All kernels accept anything convertible to a C-contiguous `float32` numpy
array as input (a copy is made only when the dtype/layout differs) and write
into a caller-provided `out` or a freshly allocated array.

| Function | Semantics | Notes |
|---|---|---|
| `add(a, b, out=None)` | `out = a + b` | `a.size == b.size` |
| `scale(x, alpha, out=None)` | `out = alpha * x` | `alpha` converted to float32 |
| `relu(x, out=None)` | `out = max(x, 0)` | |
| `sum(x) -> float` | float32-accumulated sum | raises on empty input |
| `max(x) -> float` | maximum | raises on empty input |
| `gemm(A, B, alpha=1.0, beta=0.0, out=None)` | `C = alpha*A@B + beta*C` | shapes `(M,K)`, `(K,N)`, `(M,N)` |
| `fp4_to_bf16_dequant(packed, scale=1.0)` | fp4 (E2M1) → bf16 | returns `uint16[2·len]` |
| `mfma_f32_16x16x16bf16(c, a, b, cbsz=0, abid=0, blgp=0)` | K16 bf16 MFMA: `c[0..3] += a[0..1]·b[0..1]` | `c` 4 floats in-place; `a`,`b` 2 packed uint32 |
| `direct_lds_fill_bf16(lds_dst, global_src, elements)` | global → LDS bf16 copy | raw byte addresses |
| `use_async_copy_default() -> bool` | async-copy gate (host: always `True`) | `K3_NO_ASYNC` overrides |
| `moe_align_block_size(topk_ids, num_experts, block_size=16)` | `[M, top_k]` routing → block-aligned layout | returns `(sorted_ids, expert_ids, EM)` |
| `fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale, sorted_ids, topk_w, expert_ids, *, top_k=1, group_size=32, swiglu_limit=0.0, b13=None, b2=None, act_scratch=None, out=None)` | fused MXFP4 MoE grouped GEMM (gate_up + SwiGLU, then down + combine) | CPU-reference oracle; returns `out` `(M, hidden)` |

Contract violations raise `ValueError`; a badly-typed `out` raises
`TypeError`. `out` must be writable, C-contiguous `float32` and exactly the
right length — never passed by value expecting a silent copy.

#### MoE fused usage

The fused MoE kernel consumes the *sorted* layout produced by
`moe_align_block_size` (flat topk indices, per-expert padded). The routing
weights must be gathered into the same order before the call:

```python
sorted_ids, expert_ids, EM = kernels.moe_align_block_size(topk_ids, E)
# topk_w is [M, top_k]; gather into sorted order, zeroing padding entries.
N = M * top_k
tw = np.where(sorted_ids < N, topk_w.ravel()[np.clip(sorted_ids, 0, N - 1)], 0.0)
out = kernels.fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale,
                              sorted_ids, tw, expert_ids, top_k=top_k)
```

`A` is bf16 (`uint16`) `[M, hidden]`; `w13`/`w2` are packed E2M1 `uint8`;
`w13_scale`/`w2_scale` are ue8m0 `uint8`. These bindings call the CPU
reference (`fused_moe_mxfp4_cpu`), matching the C++ oracle bit-for-bit up to
float32 operation order; the HIP GPU path is invoked from the C++ API
(`vkernels::kernels::hip::fused_moe_mxfp4`).

#### bf16 GEMM

`gemm_bf16` is a bf16-storage GEMM that mirrors `gemm_bf16_cpu`: the inputs
and output are packed bf16 (`uint16`) arrays, the multiply-accumulate runs
in float32, and the result is rounded back to bf16 with round-to-nearest-even.
The fallback replicates the C++ triple-loop operation order, so the two
backends are bit-identical.

```python
A = np.array([_f2bf(1), _f2bf(2), _f2bf(3), _f2bf(4)],
             dtype=np.uint16).reshape(2, 2)
I = np.array([_f2bf(1), _f2bf(0), _f2bf(0), _f2bf(1)],
             dtype=np.uint16).reshape(2, 2)
kernels.gemm_bf16(A, I)                       # [[bf16(1), bf16(2)],
                                             #  [bf16(3), bf16(4)]]
# C = alpha*A@B + beta*C  (C pre-filled with bf16(1), beta=2)
C = np.full((2, 2), _f2bf(1), dtype=np.uint16)
kernels.gemm_bf16(A, I, alpha=1.0, beta=2.0, out=C)
```

`gemm_bf16_config(M, N, K) -> (block_m, block_n, block_k, threads)` selects the
serving tile when `M <= 64` (`(16, 16, 64, 64)`) and the warmup/prefill tile
otherwise (`(64, 64, 64, 256)`), matching `gemm_bf16_config_for`.

#### MLA — Multi-head Latent Attention (absorbed form)

`mla_fwd(q, k_c, k_pe, v_c, *, q_start=0, kv_start=0, scale=None, out=None)`
implements absorbed-form MLA on the host. `q` is `[B, H, S_q, lr+rhd]`, split
into the RoPE-routed part `q_pe = q[..., :rhd]` and the content part
`q_c = q[..., rhd:]`. The per-head query position is `gqi = q_start + i` and a
key `j` is attended iff `kv_start + j <= gqi` (causal). The attention score is

```
s_j = scale * (q_c[i] . k_c[j] + q_pe[i] . k_pe[j])
```

with `scale` defaulting to `1/sqrt(lr+rhd)`. Rows with no attended key yield
zero. `mla_config(S_q, lr, rhd) -> (block_q, block_n, threads)` mirrors
`mla_config_for` (`S_q <= 8` -> `(1, 64, 64)`, otherwise `(4, 64, 256)`).

```python
q = np.array([1,0,0,0, 0,1,1,0], dtype=np.float32).reshape(1,1,2,4)
k_c  = np.array([1,0, 0,1], dtype=np.float32).reshape(1,2,2)
k_pe = np.array([0,1, 1,0], dtype=np.float32).reshape(1,2,2)
v_c  = np.array([5,6, 7,8], dtype=np.float32).reshape(1,2,2)
kernels.mla_fwd(q, k_c, k_pe, v_c, scale=0.5)
# [[5., 6.],            # q0 attends only k0
#  [w0*5+w1*7, w0*6+w1*8]]   # q1 attends k0,k1; w1 = sigmoid(1)
```

#### KDA — Kimi Delta Attention

The KDA family mirrors `src/c/vkernels/kernels/kda.{cpp,hpp}`. The
per-token recurrence is

```
S_t = g_t * S_{t-1} + beta_t * (v_t - a_t) * k_t^T     (a_t = S_{t-1} . k_t)
o_t = S_t . q_t
```

with `g` (forgetting) and `beta` in `[B, H, S]`, and `q`/`k`/`v` in
`[B, H, S, D]`. Two entry points cover the whole compute graph:

| Function | Semantics |
|---|---|
| `kda_layer_norm_gated(x, weight, gate, *, eps=1e-6, out=None)` | gated RMSNorm followed by SiLU(gate): `silu(gate) * rmsnorm(x) * weight`. Returns `float32[B,H,S,D]` (infers `(N,D)` for any 2-D `x` with a matching `gate`). |
| `kda_gate_chunk_cumsum(g) -> (intra, inter)` | log-space cumsum of `g` per chunk and across chunks. `g` is `[B,H,n_chunks,chunk_size]`; returns `intra[B,H,n_chunks,chunk_size]` and `inter[B,H,n_chunks]` (prefix-sum of each chunk's last logit). A `g==0` chunk is clamped to a large negative logit so it contributes nothing. |
| `kda_naive_delta_rule_fwd(q, k, v, g, beta, out=None)` | per-token oracle (above), sequential. Bit-exact against the C++ kernel. |
| `kda_delta_rule_intra(q, k, v, g, beta, intra, inter_state, *, u, chunk_size, chunk_idx)` | one chunk's intra-chunk recurrence into `u[B,H,S,D]`, reading `u[j<t]` from the same call. |
| `kda_delta_rule_inter(k, v, g, beta, intra, u, *, inter_state, chunk_size, chunk_idx)` | one chunk's inter-chunk state update into `inter_state[B,H,n_chunks+1,D,D]` (reads `inter_state[..,chunk_idx]`, the previous chunk's state). |
| `kda_gla_fwd_o(q, k, g, beta, intra, inter_state, u, *, chunk_size, out=None)` | gated linear-attention output assembly: multiplies the per-chunk outputs by the gate cumsum and sums across chunks. |
| `kda_delta_rule_fwd(q, k, v, g, beta, *, chunk_size, out=None)` | chunked orchestrator: `gate_chunk_cumsum` -> intra/inter loop -> `gla_fwd_o`. `S` must be divisible by `chunk_size`. |
| `kda_pack_bitmatrix(bits, n_bits=None) -> uint8` | MSB-first bit packing of a 1-D `uint8` {0,1} array into `ceil(n_bits/8)` bytes (defaults to `bits.size`). |

The standalone intra/inter/gla pieces are stateful: callers supply fresh
zeroed `u` and `inter_state` buffers (the orchestrator allocates them
internally). `kda_pack_bitmatrix` is the bit-packing used to store the
`q @ k^T` sign matrix compactly. On the compiled backend the standalone
stages compose bit-identically to `kda_delta_rule_fwd`; the chunked forward
matches the naive oracle to `1e-3*(1+maxabs)` (the same tolerance as the C++
test in `tests/kernels/attn/test_kda.cpp`).

```python
B = H = 1; S = 64; D = 16; cs = 16
rng = np.random.default_rng(0)
q = rng.standard_normal((B, H, S, D)).astype(np.float32)
k = rng.standard_normal((B, H, S, D)).astype(np.float32)
v = rng.standard_normal((B, H, S, D)).astype(np.float32)
g = (0.3 + 0.7*rng.random((B, H, S))).astype(np.float32)
beta = (0.3 + 0.7*rng.random((B, H, S))).astype(np.float32)

naive  = kernels.kda_naive_delta_rule_fwd(q, k, v, g, beta)
chunky = kernels.kda_delta_rule_fwd(q, k, v, g, beta, chunk_size=cs)
assert np.allclose(naive, chunky, atol=1e-3*(1+np.abs(naive).max()))

# bit packing: 1,0,1,1,0,0,0,1,0,1  ->  0xB1, 0x40
kernels.kda_pack_bitmatrix(np.array([1,0,1,1,0,0,0,1,0,1], np.uint8), n_bits=10)
# array([177,  64], dtype=uint8)
```

### `vkernels.comm`

| Function / class | Semantics |
|---|---|
| `ring_rank(rank, world) -> Topology` | `(rank, world, next, prev)` ring slot |
| `build_ring_topology(world) -> list[Topology]` | one entry per rank |
| `BlockingQueue` | thread-safe queue of float32 chunks (`push/pop/close/closed`) |
| `MockChannel(out, in_)` | in-process channel (`send/recv/closed`) |
| `make_ring_channels(world)` | `world` channels in a ring (`r` sends to `r+1`) |
| `ring_allreduce_rank(local, rank, world, next, prev)` | one rank's all-reduce, `local` summed in place |
| `ring_allreduce(locals) -> list[array]` | all ranks simulated in one process |
| `OverlapExecutor.run(iters, compute, comm) -> Result` | compute on stream A, comm on stream B, per-iteration future |
| `stage_runs_1d(dst, src_ptrs, dst_offsets, lengths)` | validate + stage 1-D runs (`StagedRun1D`) |
| `stage_runs_2d(dst, runs)` | validate + stage 2-D tiles (`StagedRun2D`) |
| `p2p_gather_runs(dst, src_ptrs, dst_offsets, lengths, *, stream=None)` | single-launch 1-D gather |
| `p2p_gather_runs_2d(dst, runs, *, stream=None)` | single-launch strided-tile gather |
| `memcpy_peer_batch_async(dst, src_ptrs, dst_offsets, lengths, *, stream=None)` | legacy per-run seam (benchmarks) |

`Gather2DRun` is a dataclass with fields `(src, src_stride, dst_offset,
dst_stride, width, height)`; 6-tuples are accepted wherever a run list is
expected. `src_ptrs` are raw byte addresses — e.g.
`arr.__array_interface__["data"][0]` or `ctypes.addressof(...)` — of
peer-accessible memory under CUDA or simply readable memory on the host.
The p2p functions validate the run list up front (capacity, disjoint output
runs, src/dst non-overlap) and raise `ValueError` on violation; a
`num_runs == 0` list is a valid no-op.

When ``stream=None`` the p2p work runs to completion before returning; with a
:class:`~vkernels.core.Stream` the work is enqueued and the caller owns
ordering and completion via `stream.wait()`. The run-metadata arrays are read
before returning, but the *source buffers* must stay alive until the stream
completes.

### `vkernels.core`

* `Device(index=-1)` — `index()`, `set_current()`, `sync()`,
  `supports_peer(other)` (all no-ops on a host build; real device semantics
  under CUDA), equality. `default_device()` returns `Device(-1)`.
* `Stream()` — `submit(task)`, `wait()`, `submitted()`; one worker thread
  per stream, in-order execution within a stream, concurrency across
  streams. A destroyed stream first drains its queue.

## Testing

```sh
uv sync                       # one-time: .venv/ + editable install + deps
uv run pytest                 # full suite (discovery + bindings)
uv run python -m unittest discover -s tests/python -v   # same via unittest
# or, from the CMake build:
ctest --preset python         # vkl_smoke + python_bindings
```

The Python tests live in ``tests/python/`` (``test_discovery.py`` for the
CLI, plus ``test_kernels.py``, ``test_comm.py``, ``test_core.py`` and
``test_backend.py`` for the bindings). The binding tests run against
whichever backend is loaded and, when both are available, cross-check them
bit-for-bit on random float32 data. Under CTest the interpreter can be
overridden with ``-DVKERNELS_PYTHON_EXECUTABLE`` (e.g. a venv that has
numpy when the system Python does not).

## Layout

```
pyproject.toml                # uv-managed Python project (src-layout, numpy dep)
├── src/
│   ├── python/
│   │   ├── CMakeLists.txt    # extension build + Python tests under CTest
│   │   └── vkernels/
│   │       ├── __init__.py   # version, backend flag, lazy submodules
│   │       ├── _backend.py   # extension loader (compiled vs fallback)
│   │       ├── _core.cpp     # pybind11 bindings to src/c (the caller)
│   │       ├── _fallback.py  # pure-Python reference (numpy)
│   │       ├── _types.py     # shared dataclasses (Topology, runs, ...)
│   │       ├── core.py       # public Device/Stream API
│   │       ├── kernels.py    # public elementwise/reduce/gemm API
│   │       ├── comm.py       # public collectives/overlap/p2p API
│   │       ├── discovery.py  # C++/CUDA source scanner (no deps)
│   │       └── cli/          # the `vkl` command
│   └── rust/                 # Rust bindings (workspace: vkernels-sys FFI + vkernels)
└── tests/python/             # unittest suites (CTest, uv, make py-test)
```
