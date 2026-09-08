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
| `fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale, sorted_ids, topk_w, expert_ids, act_scratch=None, out=None, *, top_k=1, group_size=32, swiglu_limit=0.0, activation="swiglu", beta=4.0, linear_beta=25.0, b13=None, b2=None)` | fused MXFP4 MoE grouped GEMM (gate_up + `activation`∈{`swiglu`,`situ`}, then down + combine) | CPU-reference oracle; returns `out` `(M, hidden)` |

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

`activation` selects the gate/up epilogue: `"swiglu"` (default, the
`silu(clamp(gate,L)) · clamp(up,L)` SwiGLU clamp) or `"situ"` (Kimi-K3,
matching vLLM's `situ_and_mul`: `beta·tanh(gate/beta)·sigmoid(gate) ·
linear_beta·tanh(up/linear_beta)`, no `swiglu_limit` clamp). `beta` and
`linear_beta` are the SiTU softcaps. `act_scratch` is an optional
`uint16 [EM * ispp]` buffer for the activation intermediate (allocated when
omitted); `b13`/`b2` are optional fp32 biases. `A` is bf16 (`uint16`)
`[M, hidden]`; `w13`/`w2` are packed E2M1 `uint8`; `w13_scale`/`w2_scale`
are ue8m0 `uint8`. These bindings call the CPU reference
(`fused_moe_mxfp4_cpu`), matching the C++ oracle bit-for-bit up to float32
operation order; the HIP GPU path is invoked from the C++ API
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

#### DSA — DeepseekSparseAttn sparse-MLA

`dsa_sparse_fwd(q, kv, indices, *, dim, tail_dim, topk=None, kv_group=1,
block_I=64, inner_iter=1, sm_scale=None, return_lse=False, out=None,
lse=None)` is the CPU reference for the gfx942 sparse-MLA forward
(GLM-5.3-Flash / DeepSeek-V3, issue #51). An external indexer has already
selected, per query token, the `topk` most relevant KV tiles (`indices`);
the kernel scores each query against exactly those keys with a two-pass
base-2 stable softmax and produces the combined attention output.

* `q` is `[1, S_q, H, dim + tail_dim]` fp32 — `[q_main (dim) | q_tail (tail_dim)]` (tail may be 0).
* `kv` is `[1, S_kv, kv_group, dim + tail_dim]` fp32 (`kv_group == 1`):
  `v = kv[j][0:d_v]`, `k_main = kv[j][0:dim]`, `k_tail = kv[j][dim:dim+tail_dim]`, `d_v = dim − tail_dim` (> 0).
* `indices` is `[1, S_q, kv_group, topk]` int32; entries `< 0` or `>= S_kv` are masked kpool tails (weight 0).
* `sm_scale` defaults to `(1/sqrt(dim + tail_dim)) * log2(e)`. `tail_dim == 0`
  skips the rope-tail dot entirely — the exact case the tilelang code path
  cannot compile.

Returns `out` `[1, S_q, H, d_v]` fp32, or `(out, lse)` when `return_lse=True`
(`lse` is `[1, S_q, H]` fp32, base-2). `dsa_config(S_q, H, dim, topk)`
returns `(bq, threads, block_I, inner_iter)` (decode `S_q<=8` →
`(1, 64, 64, inner_iter)`; prefill → `(4, 256, 64, inner_iter)`). The
device ABI is bf16 (the integrator converts); the paged-MQA gated
**top-k logits** indexer stage (`dsa_topk_logits`, fp8 e4m3fnuz) is HIP/C
only and is not exposed as a CPU-path Python binding.

```python
q  = np.array([1,0,0,0, 0,1,1,0], np.float32).reshape(1, 2, 1, 4)  # dim=4, tail=0
dim, tail_dim, topk = 4, 0, 2
kv = np.array([[1,0, 0,1, 1,1, 1,0]], np.float32).reshape(1, 2, 1, 4)
id = np.array([[[0, 1]]], np.int32).reshape(1, 2, 1, 2)
kernels.dsa_sparse_fwd(q, kv, id, dim=dim, tail_dim=tail_dim, topk=topk)
# array([[[[..], [..]]]], dtype=float32)  # shape (1, 2, 1, 4)
```

`dsa_kpool_assemble(chunk_k, chunk_score, tail_k, tail_score, ape,
req_pool_idx, n_from_tail, chunk_src_start, tail_logical_base, loc, *,
slots_per_page, num_pages, write_mask=None, out=None)` is the CPU
reference for the GLM-5.3 DSA **kpool-cache compress + rotate + write**
path (issue #60), replacing `_kpool_assemble_softmax_rotate_write_cache_kernel`.
For each of `n_pools` logical pools it gathers `pool_size` participating
keys from a TAIL prefix (`n_from_tail`, at `tail_logical_base + s` via
`phys = (base + s) % tail_size`) and a CHUNK suffix (the remaining
`pool_size - n_from_tail`, from `chunk_k`/`chunk_score` at
`chunk_src_start`), computes the `ape`-gated online-softmax weighted
mean, rotates it by the 128-point involution-normalised Hadamard, and
writes the bf16-quantised result to the persistent page cache at
`out[loc[r] // ssp, loc[r] % ssp]`. `write_mask[r] == 0` skips the whole
row; `out` is zero-initialised so skipped rows stay zero.

`dsa_kpool_decode_update(key, slot_score, tail_k, tail_score, ape,
block_tables, req_pool_idx, pos, seq_len, out_cache_loc, *, tail_size,
slots_per_page, num_pages, out=None, tail_k_out=None, tail_score_out=None)`
is the CPU reference for the GLM-5.3 DSA **kpool-cache decode update +
maybe-write** path (issue #60), replacing
`_kpool_decode_update_and_maybe_write_cache_kernel`. For each of `batch`
requests it substitutes the current token's `key`/`slot_score` into slot
`pos % pool_size` of the pool (the rest read from the live tail),
compresses as above, and writes the persistent cache at
`out[bt[r, clamp(pool_page_group*pool_size, 0, btc-1)], pos % ssp]` —
but only when the request is valid (`req_pool_idx` in range,
`out_cache_loc != 0`, `0 <= pos < seq_len`) **and** the pool is full
(`pos % pool_size == pool_size - 1`). **Unconditionally** the current
`key`/`slot_score` is written to the live tail at `[r, pos % tail_size]`
(masked to a no-op for invalid rows, leaving the tail untouched).

Both are `head_dim == 128` (the 128-pt involution-normalised Hadamard);
the device ABI is bf16 (input dequant on load, output quant on store,
fp32 accumulation end-to-end), mirroring `dsa_sparse_fwd`. Returns `out`
`[num_pages, ssp, 128]` (assemble) / `[num_pages, ssp, 128]` (decode);
`dsa_kpool_decode_update` also writes the live tail **in place** when
`tail_k_out`/`tail_score_out` are omitted (otherwise to the provided
buffers). `dsa_kpool_group_topk_supported(group_topk)` and
`dsa_kpool_max_closed_pools(num_draft_tokens, pool_size)` are the host
planning helpers (mirroring the C ABI).

```python
import numpy as np
rng = np.random.default_rng(0)
ck = rng.standard_normal((64, 128)).astype(np.float32)   # chunk_k
ck_scr = rng.standard_normal((64, 128)).astype(np.float32)
tk = rng.standard_normal((2, 64, 128)).astype(np.float32)  # live tail
tk_scr = rng.standard_normal((2, 64, 128)).astype(np.float32)
ape = rng.standard_normal((4, 128)).astype(np.float32)     # pool_size=4
rpi = np.array([0, 0, 1, 1], np.int32)        # req per pool
nft = np.array([3, 2, 4, 1], np.int32)        # n_from_tail per pool
css = np.array([28, 29, 28, 30], np.int32)    # chunk_src_start per pool
tlb = np.array([10, 20, 30, 40], np.int32)    # tail_logical_base per pool
loc = np.array([5, 9, 21, 30], np.int32)      # output page-slot
wm  = np.array([1, 1, 1, 1], np.int32)        # write_mask (none skipped)
out = kernels.dsa_kpool_assemble(ck, ck_scr, tk, tk_scr, ape, rpi, nft,
                                 css, tlb, loc, slots_per_page=8,
                                 num_pages=16, write_mask=wm)
assert out.shape == (16, 8, 128) and out.dtype == np.float32
```

#### MHC — multi-head hybrid-attention pre/post

Two CPU references for the GLM-5.3-Flash MHC kernels (issue #51, part 2):

| Function | Semantics |
|---|---|
| `mhc_pre_gemm_sqrsum(x, fn, *, hc_mult, hidden_size, out=None, sqrsum=None)` | pre-norm GEMM + squared-sum: `out[n,o] = Σ_h x[n,h]·fn[o,h]` (`hc_mult3 = hc_mult·(2+hc_mult)` cols) **plus** `sqrsum[n] = Σ_h x[n,h]²` |
| `mhc_post(a, b, c, d, *, hc, hidden, out=None)` | post-attention combine: `out[n,j,h] = c[n,j]·d[n,h] + Σ_k a[n,k,j]·b[n,k,h]` |

`x`/`b`/`d` are fp32 on the host path (the device path is bf16 —
`vk_hip_mhc_pre_gemm_sqrsum` / `vk_hip_mhc_post` — checked against these
fp32 references). `mhc_pre_gemm_sqrsum` infers `num_tokens` from
`x.size / (hc_mult*hidden_size)`; `mhc_post` infers it from
`a.size / (hc*hc)`. Both reject inconsistent shapes and non-writable
`out`. Returns the reshaped output
(`(num_tokens, hc_mult3)` / `(num_tokens, hc, hidden)`).

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
| `kv_gather_layer(k_src, v_src, slot_ids, dst, *, stream=None) -> None` | fused indexed K/V gather for one layer (#2); `dst[:,:,0]=k_src[slot_ids]`, `dst[:,:,1]=v_src[slot_ids]` |
| `kv_scatter_layer(k_dst, v_dst, slot_ids, src, *, stream=None) -> None` | fused indexed K/V scatter for one layer (#1); reverse of `kv_gather_layer`, **unique** destination slots |

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

### `vkernels.dist`

The host reference for the distributed MoE path (issue #18), mirroring
`src/c/vkernels/dist/dist_moe.{cpp,hpp}`. The fused kernel is
single-device; `vkernels.dist` shards the weights so per-rank shards are
consumed verbatim by the stage functions and provides the orchestration
around them.

| Function | Semantics |
|---|---|
| `tp_plan(hidden, ispp, tp, group_size=32) -> dict` | validate a TP split (`hidden%tp`, `ispp%tp`, shards `%64`/`%group_size`) and return the per-rank shard geometry + byte counts |
| `tp_shard_weights(w13, w13_scale, w2, w2_scale, tp, group_size=32)` | split full weights into `tp` per-rank shards (layout-preserving) |
| `dist_moe_tp(A, w13, w13_scale, w2, w2_scale, topk_ids, topk_w, b13=None, b2=None, *, top_k, tp, group_size=32, swiglu_limit=0.0, activation="swiglu", beta=4.0, linear_beta=25.0, block_size=16)` | tensor-parallel fused MoE over `tp` simulated ranks; the two linear stages are all-reduced *before* the nonlinear epilogues |
| `ep_plan(num_experts, ep, rank) -> dict` | expert-parallel placement for `rank` |
| `ep_dispatch(topk_ids, plan, num_experts, block_size=16) -> (local_ids, recv_counts, offsets, local_expert_ids)` | expert-parallel all-to-all / sort re-layout with local expert ids |
| `dist_moe_ep(...)` | expert-parallel fused MoE forward |
| `pp_boundary_send(state, queue)` / `pp_boundary_recv(queue, M, hidden)` | pipeline-parallel boundary transfer (graph-capturable primitive, ties into #10) |
| `round_bf16(x)` | re-quantise the bf16 stage input (matches `vkernels::round_bf16`) |

The TP/EP forward matches `vkernels.kernels.fused_moe_mxfp4` (the
single-rank oracle) after the all-reduces. The `activation` knob
(`"swiglu"`/`"situ"`) is threaded into the same per-stage functions the
single-rank path uses.

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
│   │       ├── dist.py        # distributed MoE: TP/EP/PP sharding (#18)
│   │       ├── discovery.py  # C++/CUDA source scanner (no deps)
│   │       ├── vllm_experts.py # OPTIONAL vLLM integration (torch + vLLM)
│   │       └── cli/          # the `vkl` command
│   └── rust/                 # Rust bindings (workspace: vkernels-sys FFI + vkernels)
└── tests/python/             # unittest suites (CTest, uv, make py-test)
```

## vLLM integration (optional): `vkernels.vllm_experts`

`vkernels.vllm_experts` is an **opt-in** integration that drives the gfx942
HIP C ABI (`vk_hip_fused_moe_mxfp4`, PR #44) from a vLLM `FusedMoE` expert
layer, replacing the broken AITER/Triton MoE path for Kimi-K3 on MI300A.
It is **not** imported by `import vkernels` (which stays numpy-only); pull
it in explicitly:

```python
from vkernels.vllm_experts import VkernelFusedExperts, CaptureSafeScratch
```

The module has two layers:

* **`CaptureSafeScratch`** — the reusable, torch-only persistent-scratch
  manager (issue #41, item 1). The HIP launcher performs **no device
  allocation of its own**; every scratch buffer (`act_scratch`, `out`,
  `sorted_ids`, `expert_ids`) is caller-provided and **reused** across
  calls. `CaptureSafeScratch` enforces that contract: each
  `(device, key)` buffer is sized ONCE (on the eager profile/warmup run,
  before any CUDA-graph capture) and sliced into forever after, so the
  storage address is stable and a captured graph replays correctly.
  **Growing a buffer while a capture session is active is refused**
  (the freed storage is referenced by the captured graph and would fault
  on replay — the root cause of the "Memory access fault by GPU node-X"
  crashes on the per-call `torch.empty()` wrapper). Usable without vLLM:
  any callable `capture_probe` can be injected (defaults to
  `torch.cuda.is_current_stream_capturing()`).

* **`VkernelFusedExperts`** — the vLLM `UnfusedOAITritonExperts` subclass
  that issues the kernels on PyTorch's *current* stream (no per-launch
  device sync, so each MoE layer is no longer a cross-stream TP barrier)
  and backs all C-ABI scratch with `CaptureSafeScratch`. Built lazily:
  `from vkernels.vllm_experts import VkernelFusedExperts` imports vLLM on
  first use (use the `_vllm_capture_probe` to also cover the breakable
  cudagraph eager-break window, issue #42).

`find_libvkernels_hip` resolves the shared library via `VKERNELS_LIB`,
`$K3/home/pylib/`, `$VKERNELS_DIR/build/...`, then
`ctypes.util.find_library`. `moe_align_block_size_with_map` is the
`expert_map`-aware (TP-sharded) CPU helper that matches
`vkernels.kernels.moe_align_block_size` for the no-map case.
