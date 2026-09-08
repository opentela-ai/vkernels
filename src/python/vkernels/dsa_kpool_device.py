"""ctypes device access to the DSA kpool-cache kernels (issue #60).

The compiled pybind backend (:mod:`vkernels._core`) binds the **CPU oracles**
only; the device kernels are consumed through the C ABI shared library,
exactly like :func:`vllm_experts.load_libvkernels_hip`. ONE ABI serves both
backends — the entry points, argument order and stream convention are
identical (the "HIP C ABI" is the portable device ABI; on the CUDA build the
``.hip`` sources compile with nvcc via the ``kernels/cuda_compat`` shim and
``hipStream_t`` IS ``cudaStream_t``, see
``docs/performance/dsa-kpool/A100.md``):

* **HIP/ROCm (gfx942)** — ``libvkernels_hip.so`` (``capi/hip_capi.cpp``).
* **CUDA (SM80 A100, bristen)** — ``libvkernels_c.so`` (the CUDA branch
  compiles ``dsa_kpool.hip`` with nvcc and exports the kpool entries from
  ``capi/cuda_capi_kpool.cpp``).

Exported entry points (see ``capi/hip_capi.hpp`` for the full contract):

===========================  ==============================================
``vk_hip_dsa_kpool_assemble``        prefill, bf16 store ``[num_pages, ssp, 128]``
``vk_hip_dsa_kpool_assemble_fp8``    prefill, legacy uint8 cache ``[num_pages, ssp*(128+4)]``
``vk_hip_dsa_kpool_decode_update``   decode, bf16 store
``vk_hip_dsa_kpool_decode_update_fp8`` decode, legacy uint8 cache
===========================  ==============================================

The torch-facing wrappers here are **serving-grade**: they are sync-free
(no D2H/H2D, no per-call host scratch), launch on the caller's current
torch stream when none is given (the sglang DSA indexer runs the compress
path on an alternate stream under ``enable_dual_stream``), and enforce the
device ABI's dtype/layout contract with cheap on-device casts:

* ``chunk_k/chunk_score/tail_k/tail_score/key/slot_score`` — bf16 device
  (``tail_k/tail_score`` are updated IN PLACE by the decode kernel, so they
  must already be bf16; a cast would silently fork the buffer).
* ``ape`` — fp32 device (the gate weight stays fp32 end-to-end).
* all index/validity arrays — int32 device (sglang hands some of them over
  as int64/bool; those are cast — each cast is one tiny kernel launch).
* ``round_scale`` — a host bool is materialized as a cached device int32
  (the kernel reads ``*it > 0`` on-device; NULL selects raw absmax/448), so
  the call stays graph-capturable.

The kernels write ONLY the slots addressed by ``loc`` / the pool-complete
rows — untouched cache slots keep their content, so the caller must not
zero the cache between incremental forwards (this is what makes the
leg fp8+scale layout usable as a drop-in for the Triton store path).
"""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path

__all__ = [
    "find_libvkernels",
    "load_libvkernels",
    "available",
    "dsa_kpool_assemble",
    "dsa_kpool_assemble_fp8",
    "dsa_kpool_decode_update",
    "dsa_kpool_decode_update_fp8",
]

_LIB_NAMES = ("libvkernels_c.so", "libvkernels_hip.so")

_lib_cache: dict = {}


def _repo_root() -> Path | None:
    """Repository root (a directory containing ``src/c/vkernels``), or None."""
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / "src" / "c" / "vkernels").is_dir():
            return cand
    return None


def find_libvkernels() -> str | None:
    """Locate a shared library exporting the ``vk_hip_dsa_kpool_*`` C ABI.

    Resolution order (mirrors ``vllm_experts.find_libvkernels_hip``):

    1. ``$VKERNELS_LIB`` — explicit path (highest precedence).
    2. ``$K3/home/pylib/libvkernels_hip.so`` — the per-model image layout.
    3. ``$VKERNELS_DIR`` / the repo root — newest ``build/**/libvkernels_
       {c,hip}.so`` (dev checkout).
    4. ``$LD_LIBRARY_PATH`` via :func:`ctypes.util.find_library`.
    5. ``None``.
    """
    env_path = os.environ.get("VKERNELS_LIB")
    if env_path and os.path.exists(env_path):
        return env_path

    k3 = os.environ.get("K3", "")
    if k3:
        for name in _LIB_NAMES:
            p = os.path.join(k3, "home/pylib", name)
            if os.path.exists(p):
                return p

    vdir = os.environ.get("VKERNELS_DIR") or ""
    if not vdir:
        root = _repo_root()
        vdir = str(root) if root else ""
    if vdir:
        cands: list[str] = []
        for name in _LIB_NAMES:
            cands += glob.glob(
                os.path.join(vdir, "build", "**", name), recursive=True
            )
        if cands:
            return max(cands, key=os.path.getmtime)

    try:
        from ctypes.util import find_library

        for soname in ("vkernels_c", "vkernels_hip"):
            found = find_library(soname)
            if found:
                return found
    except Exception:  # noqa: S110, BLE001 - no library finder; fall through
        pass

    return None


def _set_kpool_prototypes(lib: ctypes.CDLL) -> None:
    """ctypes prototypes for the four kpool device entry points.

    All tensor arguments arrive as ``c_void_p`` (raw device addresses from
    ``torch.Tensor.data_ptr()``); dims are ``c_int``; every entry takes a
    trailing ``void* stream`` (NULL -> default stream).
    """
    d = ctypes.c_int
    v = ctypes.c_void_p

    assemble_common = [
        d, d, d, d,  # n_pools, pool_size, head_dim, tail_size
        d, d, d, d,  # slots_per_page, num_pages, num_chunks, n_reqs
    ] + [v] * 10  # chunk_k, chunk_score, tail_k, tail_score, ape,
    #                 req_pool_idx, n_from_tail, chunk_src_start,
    #                 tail_logical_base, loc
    lib.vk_hip_dsa_kpool_assemble.argtypes = (
        assemble_common + [v, v, v]  # write_mask, out(bf16), stream
    )
    lib.vk_hip_dsa_kpool_assemble.restype = None
    lib.vk_hip_dsa_kpool_assemble_fp8.argtypes = (
        assemble_common + [v, v, v, v]  # write_mask, cache_u8, round_scale, stream
    )
    lib.vk_hip_dsa_kpool_assemble_fp8.restype = None

    decode_common = [
        d, d, d, d,  # batch, pool_size, head_dim, tail_size
        d, d, d, d,  # slots_per_page, block_table_cols, n_reqs, num_pages
    ] + [v] * 6  # key, slot_score, tail_k(in-place), tail_score(in-place), ape,
    #              block_tables
    lib.vk_hip_dsa_kpool_decode_update.argtypes = (
        decode_common + [v] * 5 + [v, v]  # req_pool_indices, positions,
        #                                   seq_lens, out_cache_loc,
        #                                   out(bf16), stream
    )
    lib.vk_hip_dsa_kpool_decode_update.restype = None
    lib.vk_hip_dsa_kpool_decode_update_fp8.argtypes = (
        decode_common + [v] * 5 + [v, v, v]  # + cache_u8, round_scale, stream
    )
    lib.vk_hip_dsa_kpool_decode_update_fp8.restype = None


def load_libvkernels() -> ctypes.CDLL:
    """Load and cache the shared library via ctypes.

    Returns the loaded :class:`ctypes.CDLL` with the kpool prototypes bound.
    Raises ``RuntimeError`` with guidance when the library cannot be found.
    """
    if "lib" in _lib_cache:
        return _lib_cache["lib"]
    path = find_libvkernels()
    if path is None:
        raise RuntimeError(
            "vkernels device library not found (looked for "
            f"{' / '.join(_LIB_NAMES)}). Set VKERNELS_LIB to the exact path "
            "of a CUDA or HIP build of libvkernels_c.so / "
            "libvkernels_hip.so (e.g. build/<preset>/src/c/libvkernels_c.so)."
        )
    lib = ctypes.CDLL(path)
    try:
        _set_kpool_prototypes(lib)
    except AttributeError as e:  # pragma: no cover - stale library
        raise RuntimeError(
            f"{path} does not export the vk_hip_dsa_kpool_* C ABI "
            f"({e}); rebuild the library from this vkernels checkout."
        ) from e
    _lib_cache["lib"] = lib
    _lib_cache["path"] = path
    return lib


def available() -> bool:
    """True when the device library loads (cached; never raises)."""
    try:
        load_libvkernels()
    except Exception:
        return False
    return True


# --- torch-facing wrappers --------------------------------------------------
# (torch is imported lazily: the vkernels package itself stays torch-free.)


def _torch():
    import torch

    return torch


def _as_device_stream_arg(tensors, stream):
    """Resolve the launch stream: an explicit handle wins (a raw int handle
    or a torch Stream object); otherwise the CALLER'S current torch stream
    on the tensors' device (the sglang DSA indexer launches the compress
    path on an alternate stream — the NULL default stream would order the
    writes against the wrong stream)."""
    import torch

    if stream is not None:
        if isinstance(stream, torch.cuda.Stream):
            return stream.cuda_stream
        return stream
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            return torch.cuda.current_stream(t.device).cuda_stream
    return None


def _ptr(t):
    return None if t is None else ctypes.c_void_p(t.data_ptr())


def _prep(t, dtype, name, in_place_ok=False):
    """dtype/contiguity enforcement for one tensor argument.

    Returns the tensor to take ``data_ptr()`` from. Non-in-place arguments
    may be cast (one tiny kernel); IN-PLACE arguments (the decode live tail)
    must already match the ABI dtype — a cast would fork the buffer and
    silently drop the kernel's tail update.
    """
    import torch

    if t.dtype != dtype:
        if in_place_ok:
            raise TypeError(
                f"dsa_kpool device path: {name} must be {dtype} (the decode "
                f"kernel updates it in place); got {t.dtype}"
            )
        t = t.to(dtype)
    if not t.is_contiguous():
        if in_place_ok:
            raise TypeError(
                f"dsa_kpool device path: {name} must be contiguous "
                "(updated in place by the kernel)"
            )
        t = t.contiguous()
    return t


_round_scale_buf: dict = {}


def _round_scale_ptr(round_scale: bool, device):
    """Cached device int32 for the graph-capturable round_scale flag."""
    import torch

    if not round_scale:
        return None
    key = str(device)
    buf = _round_scale_buf.get(key)
    if buf is None:
        buf = torch.ones(1, dtype=torch.int32, device=device)
        _round_scale_buf[key] = buf
    return buf


def _i32(t, name):
    return _prep(t, _torch().int32, name)


def dsa_kpool_assemble_fp8(
    cache_u8,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    ape,
    req_pool_idx,
    n_from_tail,
    chunk_src_start,
    tail_logical_base,
    loc,
    write_mask=None,
    round_scale: bool = False,
    stream=None,
) -> None:
    """PREFILL compress/write into the legacy uint8 fp8+scale cache.

    ``cache_u8``: ``[num_pages, ssp*(128+4)]`` uint8 CUDA (NOT zeroed —
    untouched slots keep their content). ``chunk_k/chunk_score``:
    ``[num_chunks, 128]`` bf16. ``tail_k/tail_score``: ``[n_reqs, tail_size,
    128]`` bf16 (read-only here). ``ape``: ``[pool_size, 128]`` fp32. The
    int arrays are ``[n_pools]`` / ``[n_pools]``-shaped index/validity
    vectors (sglang's ``writes.*`` plan, see
    ``kpool_assemble_softmax_rotate_write_cache``). Sync-free; enqueue-only.
    """
    torch = _torch()
    n_pools = int(req_pool_idx.shape[0])
    if n_pools == 0:
        return
    ck = _prep(chunk_k, torch.bfloat16, "chunk_k")
    cs = _prep(chunk_score, torch.bfloat16, "chunk_score")
    tk = _prep(tail_k, torch.bfloat16, "tail_k")
    ts = _prep(tail_score, torch.bfloat16, "tail_score")
    ap = _prep(ape, torch.float32, "ape")
    rp = _i32(req_pool_idx, "req_pool_idx")
    nf = _i32(n_from_tail, "n_from_tail")
    csrc = _i32(chunk_src_start, "chunk_src_start")
    tlb = _i32(tail_logical_base, "tail_logical_base")
    lc = _i32(loc, "loc")
    wm = None if write_mask is None else _i32(write_mask, "write_mask")
    rs = _round_scale_ptr(round_scale, cache_u8.device)
    s = _as_device_stream_arg([ck, cache_u8], stream)
    load_libvkernels().vk_hip_dsa_kpool_assemble_fp8(
        n_pools, int(ap.shape[0]), int(ck.shape[1]), int(tk.shape[1]),
        int(cache_u8.shape[1] // (ck.shape[1] + 4)), int(cache_u8.shape[0]),
        int(ck.shape[0]), int(tk.shape[0]),
        _ptr(ck), _ptr(cs), _ptr(tk), _ptr(ts), _ptr(ap), _ptr(rp),
        _ptr(nf), _ptr(csrc), _ptr(tlb), _ptr(lc), _ptr(wm),
        _ptr(cache_u8), _ptr(rs), ctypes.c_void_p(s),
    )


def dsa_kpool_decode_update_fp8(
    cache_u8,
    key,
    slot_score,
    tail_k,
    tail_score,
    ape,
    block_tables,
    req_pool_indices,
    positions,
    seq_lens,
    out_cache_loc,
    round_scale: bool = False,
    stream=None,
) -> None:
    """DECODE update + maybe-write into the legacy uint8 fp8+scale cache.

    ``key/slot_score``: ``[batch, 128]`` bf16 (read-only). ``tail_k/
    tail_score``: ``[n_reqs, tail_size, 128]`` bf16, updated IN PLACE (must
    already be bf16 + contiguous). ``ape``: ``[pool_size, 128]`` fp32.
    ``block_tables``: ``[batch, btc]`` int32 page table. The compressed K is
    written only for pool-complete valid rows; every valid row's live tail
    is updated. Sync-free; enqueue-only. Pass ``pool_size`` via
    ``ape.shape[0]`` (== sglang's ``pool.index_kpool``).
    """
    torch = _torch()
    batch = int(key.shape[0])
    if batch == 0:
        return
    k = _prep(key, torch.bfloat16, "key")
    ss = _prep(slot_score, torch.bfloat16, "slot_score")
    tk = _prep(tail_k, torch.bfloat16, "tail_k", in_place_ok=True)
    ts = _prep(tail_score, torch.bfloat16, "tail_score", in_place_ok=True)
    ap = _prep(ape, torch.float32, "ape")
    bt = _i32(block_tables, "block_tables")
    rp = _i32(req_pool_indices, "req_pool_indices")
    pos = _i32(positions, "positions")
    sl = _i32(seq_lens, "seq_lens")
    ocl = _i32(out_cache_loc, "out_cache_loc")
    rs = _round_scale_ptr(round_scale, cache_u8.device)
    s = _as_device_stream_arg([k, cache_u8], stream)
    load_libvkernels().vk_hip_dsa_kpool_decode_update_fp8(
        batch, int(ap.shape[0]), int(k.shape[1]), int(tk.shape[1]),
        int(cache_u8.shape[1] // (k.shape[1] + 4)), int(bt.shape[1]),
        int(tk.shape[0]), int(cache_u8.shape[0]),
        _ptr(k), _ptr(ss), _ptr(tk), _ptr(ts), _ptr(ap), _ptr(bt), _ptr(rp),
        _ptr(pos), _ptr(sl), _ptr(ocl),
        _ptr(cache_u8), _ptr(rs), ctypes.c_void_p(s),
    )


def dsa_kpool_assemble(
    out_bf16,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    ape,
    req_pool_idx,
    n_from_tail,
    chunk_src_start,
    tail_logical_base,
    loc,
    write_mask=None,
    stream=None,
) -> None:
    """PREFILL compress/write, bf16 store ``[num_pages, ssp, 128]``.

    Only the rows addressed by ``loc`` (and not masked out) are written;
    pass a zeroed buffer when a fully-defined output is expected (the
    vkernels bench contract), the live cache for incremental sglang-style
    updates. ``pool_size`` is taken from ``ape.shape[0]``.
    """
    torch = _torch()
    n_pools = int(req_pool_idx.shape[0])
    if n_pools == 0:
        return
    ck = _prep(chunk_k, torch.bfloat16, "chunk_k")
    cs = _prep(chunk_score, torch.bfloat16, "chunk_score")
    tk = _prep(tail_k, torch.bfloat16, "tail_k")
    ts = _prep(tail_score, torch.bfloat16, "tail_score")
    ap = _prep(ape, torch.float32, "ape")
    rp = _i32(req_pool_idx, "req_pool_idx")
    nf = _i32(n_from_tail, "n_from_tail")
    csrc = _i32(chunk_src_start, "chunk_src_start")
    tlb = _i32(tail_logical_base, "tail_logical_base")
    lc = _i32(loc, "loc")
    wm = None if write_mask is None else _i32(write_mask, "write_mask")
    s = _as_device_stream_arg([ck, out_bf16], stream)
    load_libvkernels().vk_hip_dsa_kpool_assemble(
        n_pools, int(ap.shape[0]), int(ck.shape[1]), int(tk.shape[1]),
        int(out_bf16.shape[1] // ck.shape[1]), int(out_bf16.shape[0]),
        int(ck.shape[0]), int(tk.shape[0]),
        _ptr(ck), _ptr(cs), _ptr(tk), _ptr(ts), _ptr(ap), _ptr(rp),
        _ptr(nf), _ptr(csrc), _ptr(tlb), _ptr(lc), _ptr(wm),
        _ptr(out_bf16), ctypes.c_void_p(s),
    )


def dsa_kpool_decode_update(
    out_bf16,
    key,
    slot_score,
    tail_k,
    tail_score,
    ape,
    block_tables,
    req_pool_indices,
    positions,
    seq_lens,
    out_cache_loc,
    stream=None,
) -> None:
    """DECODE update + maybe-write, bf16 store ``[num_pages, ssp, 128]``.

    Same contract as :func:`dsa_kpool_decode_update_fp8` but the compressed
    K lands as bf16 (no requantization) in ``out_bf16``.
    """
    torch = _torch()
    batch = int(key.shape[0])
    if batch == 0:
        return
    k = _prep(key, torch.bfloat16, "key")
    ss = _prep(slot_score, torch.bfloat16, "slot_score")
    tk = _prep(tail_k, torch.bfloat16, "tail_k", in_place_ok=True)
    ts = _prep(tail_score, torch.bfloat16, "tail_score", in_place_ok=True)
    ap = _prep(ape, torch.float32, "ape")
    bt = _i32(block_tables, "block_tables")
    rp = _i32(req_pool_indices, "req_pool_indices")
    pos = _i32(positions, "positions")
    sl = _i32(seq_lens, "seq_lens")
    ocl = _i32(out_cache_loc, "out_cache_loc")
    s = _as_device_stream_arg([k, out_bf16], stream)
    load_libvkernels().vk_hip_dsa_kpool_decode_update(
        batch, int(ap.shape[0]), int(k.shape[1]), int(tk.shape[1]),
        int(out_bf16.shape[1] // k.shape[1]), int(bt.shape[1]),
        int(tk.shape[0]), int(out_bf16.shape[0]),
        _ptr(k), _ptr(ss), _ptr(tk), _ptr(ts), _ptr(ap), _ptr(bt), _ptr(rp),
        _ptr(pos), _ptr(sl), _ptr(ocl),
        _ptr(out_bf16), ctypes.c_void_p(s),
    )
