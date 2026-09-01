"""vLLM ``FusedMoE`` integration: capture-safe vkernels HIP backend.

This is an OPTIONAL integration module. Importing it requires ``torch``;
:class:`VkernelFusedExperts` additionally requires ``vllm`` (imported
lazily, so the torch-only :class:`CaptureSafeScratch` is usable without
vLLM). ``import vkernels`` stays dependency-free; only an explicit
``from vkernels.vllm_experts import ...`` pulls these in.

Two layers:

* :class:`CaptureSafeScratch` — the reusable, torch-only persistent
  scratch manager. The vkernels HIP launcher performs **no device
  allocation of its own** (issue #41, item 1): every scratch buffer
  (``act_scratch``, ``out``, ``sorted_ids``, ``expert_ids``) is
  caller-provided and **reused** across calls. This class enforces that
  contract by sizing each ``(device, key)`` buffer ONCE — on the eager
  profile/warmup run, before any CUDA-graph capture — and slicing into it
  forever after, so the storage address is stable and a captured graph
  replays correctly. Growing a buffer while a capture session is active
  is *refused* (the freed storage is referenced by the captured graph and
  would fault on replay).

* :class:`VkernelFusedExperts` — the vLLM expert backend that drives the
  gfx942 HIP C ABI :c:func:`vk_hip_fused_moe_mxfp4` (PR #44) via ctypes on
  MI300A, replacing the broken AITER/Triton MoE path for Kimi-K3.
  ``moe_align_block_size`` is done on CPU (routing metadata is small,
  ``O(M*top_k)``) and ``apply`` issues the kernels on PyTorch's *current*
  stream (no per-launch device sync), so each MoE layer is no longer a
  cross-stream TP barrier.

Validated on MI300A (gfx942): all 8 C++ GPU tests + 4 Python ctypes tests
pass, and the breakable piecewise cudagraph path (issue #42) runs 0 faults
over 58 min at 1.4–1.7× the eager baseline (see the cookbook BENCHMARK.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # torch is an OPTIONAL runtime dependency (see _require_torch); it is
    # only needed for type annotations here, so guard it under
    # TYPE_CHECKING to keep `import vkernels.vllm_experts` torch-free.
    import torch

__all__ = [
    "CaptureSafeScratch",
    "VkernelFusedExperts",  # noqa: F822 - provided via __getattr__
    "max_em_count",
    "max_num_tokens_hint",
    "moe_align_block_size_with_map",
]

import ctypes
import glob
import importlib
import os
from pathlib import Path

import numpy as np


# torch is an OPTIONAL dependency of this module: the numpy-only helpers
# (moe_align_block_size_with_map, max_em_count, max_num_tokens_hint,
# find_libvkernels_hip) and the CaptureSafeScratch *class* are importable
# without it. torch is imported lazily by CaptureSafeScratch.get and the
# VkernelFusedExperts factory (the parts that actually touch tensors), so
# `from vkernels.vllm_experts import CaptureSafeScratch` works on a
# torch-free host (e.g. CI) and only `CaptureSafeScratch().get(...)` /
# `VkernelFusedExperts` raise if torch is missing.
def _require_torch():
    """Import and return torch, or raise a clear ImportError."""
    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - optional integration dep
        raise ImportError(
            "vkernels.vllm_experts needs torch for this operation. Install it "
            "with `pip install torch` (or your ROCm torch wheel). `import "
            "vkernels` does not need torch — only this optional integration "
            "module's tensor operations do."
        ) from exc


# ---------------------------------------------------------------------------
# Capture detection
# ---------------------------------------------------------------------------


def _default_capture_probe() -> bool:
    """True iff a CUDA-graph capture is active on the current stream.

    The default probe (no vLLM). ``CaptureSafeScratch`` uses this to decide
    whether it may grow a buffer: growth is only safe when *not* capturing,
    because a capture-time address is not replay-stable (PyTorch's caching
    allocator may recycle it before the graph replays).
    """
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return True
    except Exception:  # noqa: S110, BLE001 - best-effort; "not capturing" on failure
        pass
    return False


def _vllm_capture_probe() -> bool:
    """Capture detection that also covers the vLLM *breakable* cudagraph
    eager-break window.

    ``@eager_break_during_capture`` runs ``apply`` via
    ``BreakableCUDAGraphCapture.add_eager``, which does
    ``_end_segment()`` (stream leaves capture) → ``fn()`` (apply, on a
    momentarily *non*-capturing stream) → ``_begin_segment()`` (re-enters
    capture). During that window ``torch.cuda.is_current_stream_capturing()``
    is False even though a capture session is on the call stack — and an
    allocation here is still unsafe (its address will not replay). This
    probe also asks ``BreakableCUDAGraphCapture.is_active()``, falling back
    to the plain torch flag for the non-breakable path.
    """
    try:
        from vllm.compilation.breakable_cudagraph import (
            BreakableCUDAGraphCapture,
        )

        if BreakableCUDAGraphCapture.is_active():
            return True
    except Exception:  # noqa: S110, BLE001 - older vLLM / non-breakable path
        pass
    return _default_capture_probe()


# ---------------------------------------------------------------------------
# CaptureSafeScratch — the reusable, capture-safe persistent-scratch fix
# ---------------------------------------------------------------------------


class CaptureSafeScratch:
    """Persistent, capture-safe device scratch buffers (issue #41, item 1).

    The vkernels HIP C ABI expects **caller-provided, persistent** scratch:
    ``act_scratch`` is ``[EM, ispp]`` bf16 and ``out`` is ``[M, hidden]``
    fp32, both supplied by the caller and reused across calls. A naive
    wrapper that does ``torch.empty()`` per call faults under CUDA-graph
    capture/replay: the fresh caching-allocator storage is not
    replay-stable, so when the captured graph replays the C kernel is
    launched against memory the allocator has recycled
    ("``Memory access fault by GPU node-X on address 0x...``").

    This manager sizes each ``(device, key)`` buffer ONCE — at the absolute
    maximum token budget (``capacity``, computed by the caller from
    ``MAX_NUM_BATCHED_TOKENS`` on the eager profile/warmup run, before any
    capture begins) — and slices into it on every subsequent call,
    including the capture-time eager break and replay. It therefore
    allocates exactly once per key and never again.

    Args:
        capture_probe: zero-arg callable returning ``True`` while a
            CUDA-graph capture is active. Defaults to
            :func:`_default_capture_probe` (plain torch). vLLM callers pass
            :func:`_vllm_capture_probe` to also cover the breakable
            eager-break window.

    Example:
        >>> import torch  # doctest: +SKIP
        >>> dev = torch.device("cuda", 0)
        >>> s = CaptureSafeScratch()
        >>> # eager warmup: pin the high-water mark (M_max*hidden)
        >>> out = s.get("out", M*hidden, dtype=torch.float32,
        ...             device=dev, capacity=M_max*hidden)
        >>> # capture / replay: slice the same storage (no alloc)
        >>> out = s.get("out", M*hidden, dtype=torch.float32, device=dev)
    """

    def __init__(self, capture_probe=None):
        # (device.index, key) -> torch.Tensor (over-allocated once)
        self._buffers: dict[tuple[int, object], torch.Tensor] = {}
        self._capture_probe = capture_probe or _default_capture_probe

    def keys(self):
        """The ``(device.index, key)`` pairs currently held."""
        return list(self._buffers.keys())

    def get(
        self,
        key,
        count: int,
        *,
        dtype,
        device,
        capacity: int | None = None,
        name: str = "scratch",
    ):
        """Return a persistent ``[count]`` view of ``dtype`` on ``device``.

        On first use the storage is allocated at ``capacity`` (defaults to
        ``count``) so the high-water mark is pinned on the eager warmup;
        later calls only slice (``store[:count]``). If a later call needs
        *more* than the pinned capacity, the storage is reallocated at the
        new ``capacity`` — but **only when not capturing**: a too-small
        buffer during capture raises ``RuntimeError`` instead, because
        growing would free storage the captured graph still references.

        Args:
            key: hashable key identifying the buffer (e.g. ``"out"`` or
                ``("act", ispp)``); storage is keyed by ``(device, key)``.
            count: number of elements required *this* call.
            dtype: torch dtype (e.g. ``torch.float32``).
            device: torch device.
            capacity: total elements to allocate on first use (the
                high-water mark). Defaults to ``count``.
            name: human-readable name for the refusal error message.
        """
        if count < 0:
            raise ValueError(f"{name}: count must be non-negative, got {count}")
        k = (device.index, key)
        store = self._buffers.get(k)
        if store is not None and store.numel() >= count:
            return store[:count]

        # Need a new or larger buffer. Refuse while capturing (growing
        # frees storage the captured graph references -> replay fault).
        if self._capture_probe():
            cap = 0 if store is None else store.numel()
            raise RuntimeError(
                f"[vkernels] persistent {name} (dev={device.index}, "
                f"key={key!r}, cap={cap}) is too small for {count} elements "
                f"AND a CUDA-graph capture is active — refusing to grow "
                f"(would corrupt replay). Size it on the eager profile/"
                f"warmup run (before capture) via a larger max-token budget, "
                f"or disable cudagraph capture for this layer."
            )

        torch = _require_torch()
        cap = max(count, capacity) if capacity is not None else count
        store = torch.empty(cap, dtype=dtype, device=device)
        self._buffers[k] = store
        return store[:count]

    def reset(self) -> None:
        """Drop all buffers (e.g. between engine recreations)."""
        self._buffers.clear()


# ---------------------------------------------------------------------------
# Max-token budget + EM high-water mark
# ---------------------------------------------------------------------------


def max_num_tokens_hint() -> int:
    """Best-effort global token budget for sizing persistent scratch.

    ``MAX_NUM_BATCHED_TOKENS`` (the vLLM knob) is the natural high-water
    mark; ``VLLM_MAX_NUM_BATCHED_TOKENS`` is the older name. Falls back to
    a conservative constant so the first eager warmup call still
    over-allocates and we effectively never allocate again.
    """
    for k in ("MAX_NUM_BATCHED_TOKENS", "VLLM_MAX_NUM_BATCHED_TOKENS"):
        v = os.environ.get(k)
        if v:
            try:
                n = int(v)
                if n > 0:
                    return n
            except ValueError:
                pass
    return 8192


def max_em_count(max_M: int, top_k: int, local_n: int) -> int:
    """High-water mark for ``EM`` = ``padded(M*top_k + local_experts, 64)``.

    ``local_n`` is the number of experts on this rank (for TP sharding);
    the padding matches :func:`moe_align_block_size_with_map` (block_size
    is a multiple of 16/64, so 64 is a safe upper bound).
    """
    return ((max_M * top_k + local_n + 63) // 64) * 64


def _local_n(local_num_experts, global_num_experts) -> int:
    return (
        local_num_experts
        if local_num_experts and local_num_experts > 0
        else global_num_experts
    )


# ---------------------------------------------------------------------------
# moe_align_block_size (CPU, expert_map-aware)
# ---------------------------------------------------------------------------


def moe_align_block_size_with_map(
    topk_ids_flat: np.ndarray,
    num_experts: int,
    block_size: int,
    expert_map: np.ndarray | None = None,
):
    """Map the flat ``[M*top_k]`` token→expert routing to the block-aligned
    ``sorted_ids`` / ``expert_ids`` layout consumed by
    :c:func:`vk_hip_fused_moe_mxfp4`, with optional global→local expert
    remapping for TP sharding.

    This is the ``expert_map``-aware variant of
    :func:`vkernels.kernels.moe_align_block_size` (which is global and
    2-D). Tokens are grouped by *local* expert (``expert_map[global_e]``;
    ``-1`` = skip / unmapped), block-aligned per expert, and the routing
    index space is padded with ``M*top_k`` (the flat index one past the
    last real token, used as a sentinel by the kernel).

    Args:
        topk_ids_flat: ``[M*top_k]`` int32 *global* expert IDs.
        num_experts: total (global) number of experts ``E``.
        block_size: GEMM block size (16 decode, 64 prefill).
        expert_map: ``[num_experts]`` int32 (global→local, ``-1`` = skip),
            or ``None`` (1:1, no sharding).

    Returns:
        ``(sorted_ids, expert_ids, EM)`` —
        ``sorted_ids`` is ``[EM]`` int32 (flat token indices, padded with
        ``M*top_k``); ``expert_ids`` is ``[EM // block_size]`` int32
        (local expert per block, ``-1`` for pure-padding blocks); ``EM``
        is the padded row count (a multiple of ``block_size``).
    """
    ids = np.ascontiguousarray(topk_ids_flat, dtype=np.int32).ravel()
    N = ids.size
    per_local: dict[int, list[int]] = {}
    for i in range(N):
        global_e = int(ids[i])
        if expert_map is not None:
            local_e = int(expert_map[global_e])
            if local_e == -1:
                continue
        else:
            local_e = global_e
        per_local.setdefault(local_e, []).append(i)

    local_experts = sorted(per_local.keys())
    EM = sum(
        ((len(per_local[e]) + block_size - 1) // block_size) * block_size
        for e in local_experts
    )
    # At least block_size (avoids zero-length arrays when nothing routes
    # to this rank — e.g. an all-padded expert_map on a cold step).
    if EM == 0:
        EM = block_size

    sorted_ids = np.full(EM, N, dtype=np.int32)
    expert_ids = np.full(EM // block_size, -1, dtype=np.int32)

    idx = 0
    blk = 0
    for e in local_experts:
        tokens = per_local[e]
        for t in tokens:
            sorted_ids[idx] = t
            idx += 1
        padded = ((len(tokens) + block_size - 1) // block_size) * block_size
        num_blocks = padded // block_size
        for b in range(num_blocks):
            if b * block_size < len(tokens):
                expert_ids[blk] = e
            blk += 1
        for _ in range(len(tokens), padded):
            sorted_ids[idx] = N
            idx += 1

    return sorted_ids, expert_ids, EM


# ---------------------------------------------------------------------------
# libvkernels_hip.so loader (ctypes)
# ---------------------------------------------------------------------------

_lib_cache: dict = {}


def find_libvkernels_hip():
    """Locate ``libvkernels_hip.so`` (the gfx942 HIP C ABI library).

    Resolution order:

    1. ``$VKERNELS_LIB`` — explicit path (highest precedence).
    2. ``$K3/home/pylib/libvkernels_hip.so`` — the per-model image layout.
    3. ``$VKERNELS_DIR/build/.../libvkernels_hip.so`` — a dev checkout;
       defaults to this repository's root (so a freshly built
       ``build/<preset>/.../libvkernels_hip.so`` is found automatically).
    4. ``$LD_LIBRARY_PATH`` via :func:`ctypes.util.find_library` — a
       system-installed copy.
    5. ``None`` — :func:`load_libvkernels_hip` raises with guidance.
    """
    env_path = os.environ.get("VKERNELS_LIB")
    if env_path and os.path.exists(env_path):
        return env_path

    k3 = os.environ.get("K3", "")
    if k3:
        k3_path = os.path.join(k3, "home/pylib/libvkernels_hip.so")
        if os.path.exists(k3_path):
            return k3_path

    vdir = os.environ.get("VKERNELS_DIR")
    if not vdir:
        # Default to this repository's root so a dev build is found without
        # an env var (matches src/python/vkernels/_backend.py).
        here = Path(__file__).resolve()
        for cand in (here, *here.parents):
            if (cand / "src" / "c" / "vkernels").is_dir():
                vdir = str(cand)
                break
    if vdir:
        cands = sorted(
            glob.glob(os.path.join(vdir, "build", "**", "libvkernels_hip.so"),
                      recursive=True)
        )
        if cands:
            # Newest build wins.
            return max(cands, key=lambda p: os.path.getmtime(p))

    try:
        from ctypes.util import find_library

        found = find_library("vkernels_hip")
        if found:
            return found
    except Exception:  # noqa: S110, BLE001 - find_library unavailable; fall through
        pass

    return None


def load_libvkernels_hip():
    """Load and cache ``libvkernels_hip.so`` via ctypes.

    Returns the loaded :class:`ctypes.CDLL`. Raises ``RuntimeError`` with
    guidance if the library cannot be found.
    """
    if "lib" not in _lib_cache:
        path = find_libvkernels_hip()
        if path is None:
            raise RuntimeError(
                "libvkernels_hip.so not found. Set VKERNELS_LIB to its path, "
                "place it in $K3/home/pylib/, or build it under "
                "$VKERNELS_DIR/build/ (cmake --preset cuda && "
                "cmake --build --preset cuda)."
            )
        _lib_cache["lib"] = ctypes.CDLL(path)
    return _lib_cache["lib"]


def resolve_moe_fn(lib):
    """Return the HIP fused-MoE C ABI function from ``lib``.

    PR #44 names it ``vk_hip_fused_moe_mxfp4`` (namespaced away from the
    CPU reference). Note the CPU reference ``vk_fused_moe_mxfp4`` is a
    *live* symbol in ``libvkernels`` with a materially different signature
    (it returns ``int32`` and takes ``sorted_ids``/``topk_w_sorted``/
    ``group_size`` rather than ``topk_ids``/``topk_w``/``block_size``), so
    it must NOT be used as a fallback — calling it with the HIP arg layout
    would be a silent ABI mismatch. Raise clearly if the HIP symbol is
    absent instead.
    """
    fn = getattr(lib, "vk_hip_fused_moe_mxfp4", None)
    if fn is None:
        raise RuntimeError(
            "vk_hip_fused_moe_mxfp4 not found in libvkernels_hip.so — "
            "rebuild with PR #44 (the CPU vk_fused_moe_mxfp4 has a "
            "different signature and must not be used here)."
        )
    return fn


def resolve_align_fn(lib):
    """Return the HIP on-device ``moe_align_block_size`` C ABI function.

    ``vk_hip_moe_align_block_size`` (issue #46 follow-up) reads ``topk_ids``
    on-device on the caller's stream — ordered after the expert-dispatch
    all-to-all — and writes the block-aligned ``sorted_ids`` / ``expert_ids``
    WITHOUT the ``topk_ids.cpu()`` host round-trip that dominated PP0's MoE
    (97-100% of its per-call ``moe:vkernel_apply``). Returns ``None`` when the
    symbol is absent (older build) so the caller can fall back to the CPU
    path; an absent symbol is NOT fatal (unlike ``resolve_moe_fn``) because
    the CPU path remains correct, just slower.
    """
    fn = getattr(lib, "vk_hip_moe_align_block_size", None)
    if fn is None:
        return None
    import ctypes
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_void_p,  # topk_ids        [N] int32  (device)
        ctypes.c_void_p,  # expert_map      [num_experts] int32 (device), or NULL
        ctypes.c_int,     # M
        ctypes.c_int,     # top_k
        ctypes.c_int,     # block_size
        ctypes.c_int,     # num_experts  (global)
        ctypes.c_int,     # local_n      (experts on this rank)
        ctypes.c_int,     # max_EM       (host high-water bound)
        ctypes.c_void_p,  # sorted_ids   [max_EM] int32 (device)
        ctypes.c_void_p,  # expert_ids   [max_EM/block_size] int32 (device)
        ctypes.c_void_p,  # out_EM       [1] int32 (device; NOT read fast-path)
        ctypes.c_void_p,  # stream
    ]
    return fn


# Opt-in: do moe_align_block_size on the GPU (no topk_ids.cpu() host
# round-trip). Defaults to the CPU path; the GPU path requires the
# on-device align symbol (PR #47) AND M*top_k <= 1024 (single-block
# shared memory). Larger N falls back to CPU automatically.
_GPU_ALIGN = os.environ.get("VKERNELS_GPU_ALIGN", "0") not in ("0", "", "false")


def _align_em_bound(M: int, top_k: int, local_n: int, block_size: int) -> int:
    """Tight, host-computed upper bound on ``EM`` (no device sync).

    Worst-case routing (max padding): each of ``min(N, local_n)`` experts
    gets exactly one token, each padded to ``block_size``; any remaining
    tokens land in one expert, padded to a multiple of ``block_size``.
    This matches the actual ``EM`` from :func:`moe_align_block_size_with_map`
    at the extreme and is a valid upper bound for every routing — and it is
    a *constant* for a given batch shape (M, top_k, local_n, block_size are
    all host integers), so the GEMM grid (``bound / block_size``) is
    capture-safe and the kernels early-out the padding blocks/rows.
    """
    N = M * top_k
    if N <= 0:
        return block_size
    head = min(N, local_n) * block_size
    if N <= local_n:
        return head
    rem = ((N - local_n) + block_size - 1) // block_size * block_size
    return head + rem


# Module singleton, keyed by (device.index, <key>), shared across all MoE
# layers on a rank (matches the validated cookbook wrapper). Uses the vLLM
# capture probe so the breakable eager-break window is covered.
_scratch = CaptureSafeScratch(capture_probe=_vllm_capture_probe)


# ---------------------------------------------------------------------------
# VkernelFusedExperts (vLLM integration; vllm imported lazily)
# ---------------------------------------------------------------------------

_VkernelFusedExperts_cls = None


def _build_vllm_experts():
    """Build :class:`VkernelFusedExperts` on first use (imports vLLM).

    Deferred so the torch-only :class:`CaptureSafeScratch` and helpers
    above remain usable without vLLM.
    """
    global _VkernelFusedExperts_cls
    if _VkernelFusedExperts_cls is not None:
        return _VkernelFusedExperts_cls

    from vllm.model_executor.layers.fused_moe.config import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
        UnfusedOAITritonExperts,
    )
    from vllm.platforms import current_platform

    # torch is guaranteed present here (vLLM import above already
    # requires it); bind it locally so the class body closes over it.
    torch = _require_torch()

    class VkernelFusedExperts(UnfusedOAITritonExperts):
        """vkernels HIP C ABI backend for MXFP4 MoE on gfx942 (MI300A).

        Calls :c:func:`vk_hip_fused_moe_mxfp4` (PR #44) via ctypes. The
        kernel does the full computation: gate-up GEMM → activation
        (SwiGLU / SiTU) → down GEMM → routing-weight application → top-k
        summation. Output is fp32, converted to bf16.

        ``moe_align_block_size`` is done on CPU (routing metadata is
        small). Weight format: ``[E, 2*ispp, hidden/2]`` uint8 (w13),
        ``[E, hidden, ispp/2]`` uint8 (w2) — matches vLLM's
        ``Mxfp4MoEMethod.create_weights()`` exactly.

        All C-ABI scratch (``act_scratch``, ``out_fp32``, ``sorted_ids``,
        ``expert_ids``) is backed by the capture-safe persistent manager
        (``_scratch``): sized once on the eager warmup at
        ``MAX_NUM_BATCHED_TOKENS`` and reused forever, so the layer is
        safe under CUDA-graph capture/replay (issue #42).
        """

        @staticmethod
        def _supports_current_device() -> bool:
            if not current_platform.is_rocm():
                return False
            # The library was compiled for gfx942 (MI300A). Check the
            # actual device arch to avoid a segfault on gfx90a.
            try:
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    gcn = getattr(props, "gcnArchName", "")
                    return "gfx942" in gcn
            except Exception:  # noqa: S110, BLE001 - arch sniff; refuse on failure
                pass
            return False

        @staticmethod
        def _supports_activation(activation: MoEActivation) -> bool:
            return activation in [
                MoEActivation.SILU,
                MoEActivation.SWIGLUOAI,
                MoEActivation.SWIGLUOAI_UNINTERLEAVE,
                MoEActivation.SWIGLUSTEP,
                MoEActivation.SITU,  # Kimi-K3 SiTU
            ]

        def workspace_shapes(
            self,
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        ):
            """Size ``workspace13`` for ``act_scratch [EM_max, ispp]`` bf16.

            The framework allocates workspace tensors with the dtype from
            :meth:`workspace_dtype` (bf16 by default). ``workspace13`` is
            the *preferred* act_scratch (it lives in the cudagraph pool, so
            its address is replay-stable) when already large enough;
            ``out_fp32`` and any too-small-``workspace13`` fallback are
            backed by the capture-safe persistent manager — never by
            per-call ``torch.empty()``.
            """
            ispp = N // 2
            block_size = 16 if M <= 32 else 64
            # EM_max: padded(M*topk + local_num_experts) — matches
            # moe_align_block_size_with_map's output size.
            EM_max = (
                (M * topk + local_num_experts + block_size - 1) // block_size
            ) * block_size
            ws13 = (EM_max, ispp)
            ws2 = (0,)  # out_fp32 is backed by the persistent manager
            output = (M, K)
            return (ws13, ws2, output)

        def apply(
            self,
            output: torch.Tensor,
            hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            activation: MoEActivation,
            global_num_experts: int,
            expert_map: torch.Tensor | None,
            a1q_scale: torch.Tensor | None,
            a2_scale: torch.Tensor | None,
            workspace13: torch.Tensor,
            workspace2: torch.Tensor,
            expert_tokens_meta,
            apply_router_weight_on_input: bool,
        ) -> None:
            from torch.profiler import record_function

            lib = load_libvkernels_hip()
            moe_fn = resolve_moe_fn(lib)
            dev = hidden_states.device

            M, K = hidden_states.shape
            E, N, _ = w1.shape  # E=local experts, N=2*ispp
            ispp = N // 2
            top_k = topk_ids.size(1)

            if global_num_experts == -1:
                global_num_experts = E

            # Block size: 16 for decode (small M), 64 for prefill (large M)
            block_size = 16 if M <= 32 else 64

            # Launch on PyTorch's *current* compute stream so every device
            # kernel (the on-device moe_align below when enabled, plus the
            # fused GEMM) is ordered with the buffers the caller allocated /
            # will read on that same stream. Acquired ONCE, up here, so the
            # on-device align (when enabled) runs on this stream too —
            # ordered after the expert-dispatch all-to-all that produced
            # topk_ids, WITHOUT the topk_ids.cpu() host round-trip that
            # previously serialised every TP all-to-all and made each MoE
            # layer a per-step barrier (issue #46).
            stream = torch.cuda.current_stream().cuda_stream
            topk_ids_dev = topk_ids.contiguous().view(-1)

            # Persistent sids/eids are read by the C kernel via raw ctypes
            # pointers PyTorch's allocator cannot track, so they are sized
            # once on the eager warmup (at MAX_NUM_BATCHED_TOKENS) and
            # reused forever — reallocating here would let the allocator
            # recycle the storage mid-capture/replay -> "Memory access
            # fault by GPU node-X". ----------------------------------------
            max_M = max_num_tokens_hint()
            # Local expert count: prefer this layer's view when set,
            # else fall back to the local w1 row count.
            try:
                local_n = _local_n(self.num_experts, global_num_experts)
            except AttributeError:
                local_n = _local_n(E, global_num_experts)
            cap_em = max_em_count(max_M, top_k, local_n)

            # On-device moe_align_block_size (issue #46 follow-up): when
            # the symbol is present (PR #47), VKERNELS_GPU_ALIGN is set, and
            # M*top_k <= 1024 (single-block shared memory — the decode
            # regime that dominates PP0), do the routing sort on the GPU on
            # `stream` and read topk_ids/expert_map on-device — removing the
            # ~4 ms topk_ids.cpu() host sync (97-100% of PP0's per-call
            # moe:vkernel_apply). Larger N (prefill) falls back to the CPU
            # path automatically. The GEMM is launched with max_EM/block_size
            # blocks (a constant for the batch shape; the kernels early-out
            # the padding blocks/rows), so NO host read of out_em is needed.
            align_fn = resolve_align_fn(lib)
            use_gpu = (
                _GPU_ALIGN
                and align_fn is not None
                and (M * top_k) <= 1024
            )
            if use_gpu:
                with record_function("moe:apply.gpu_align"):
                    max_EM_host = min(
                        _align_em_bound(M, top_k, local_n, block_size), cap_em
                    )
                    d_sids = _scratch.get(
                        ("sids", top_k, local_n), max_EM_host,
                        dtype=torch.int32, device=dev, capacity=cap_em,
                        name="sids",
                    )
                    d_eids = _scratch.get(
                        ("eids", top_k, local_n), max_EM_host // block_size,
                        dtype=torch.int32, device=dev, capacity=cap_em,
                        name="eids",
                    )
                    # out_em is written by the GPU but NOT read here (the
                    # GEMM uses max_EM_host). Persistent + capacity 1 so the
                    # device pointer is capture-stable and never reallocates.
                    d_out_em = _scratch.get(
                        ("out_em", top_k, local_n), 1,
                        dtype=torch.int32, device=dev, capacity=1,
                        name="out_em",
                    )
                    expert_map_dev = (
                        expert_map.contiguous() if expert_map is not None else None
                    )
                    rc = align_fn(
                        ctypes.c_void_p(topk_ids_dev.data_ptr()),
                        ctypes.c_void_p(expert_map_dev.data_ptr())
                        if expert_map_dev is not None else None,
                        ctypes.c_int(M), ctypes.c_int(top_k),
                        ctypes.c_int(block_size),
                        ctypes.c_int(global_num_experts),
                        ctypes.c_int(local_n),
                        ctypes.c_int(max_EM_host),
                        ctypes.c_void_p(d_sids.data_ptr()),
                        ctypes.c_void_p(d_eids.data_ptr()),
                        ctypes.c_void_p(d_out_em.data_ptr()),
                        ctypes.c_void_p(stream),
                    )
                if rc == 0:  # VK_OK
                    EM = max_EM_host
                else:
                    # ABI rejected the inputs (e.g. local_n > num_experts);
                    # fall back to the proven CPU path below.
                    use_gpu = False

            if not use_gpu:
                with record_function("moe:apply.cpu_copy"):
                    topk_ids_flat = (
                        topk_ids_dev.cpu().numpy().astype(np.int32)
                    )
                    expert_map_np = (
                        expert_map.cpu().numpy().astype(np.int32)
                        if expert_map is not None
                        else None
                    )
                with record_function("moe:apply.cpu_align"):
                    sids_np, eids_np, EM = moe_align_block_size_with_map(
                        topk_ids_flat, global_num_experts, block_size,
                        expert_map_np,
                    )
                with record_function("moe:apply.gpu_copy"):
                    d_sids = _scratch.get(
                        ("sids", top_k, local_n), EM, dtype=torch.int32,
                        device=dev, capacity=cap_em, name="sids",
                    )
                    d_eids = _scratch.get(
                        ("eids", top_k, local_n), EM, dtype=torch.int32,
                        device=dev, capacity=cap_em, name="eids",
                    )
                    d_sids.copy_(torch.from_numpy(sids_np))
                    d_eids[: EM // block_size].copy_(torch.from_numpy(eids_np))

            # Routing weights: if already applied on input, use a
            # persistent ones buffer (sized once at MAX_NUM_BATCHED_TOKENS
            # * top_k on the eager warmup). A per-call torch.ones_like()
            # would allocate fresh caching-allocator storage that - during
            # the breakable cudagraph eager-break window (issue #42), where
            # is_current_stream_capturing() is False but a session is live
            # - is not replay-stable and faults like the old
            # torch.empty() out_fp32 wrapper did.
            if apply_router_weight_on_input:
                ones_buf = _scratch.get(
                    ("ones", top_k), M * top_k, dtype=torch.float32, device=dev,
                    capacity=max_M * top_k, name="topk_ones",
                )
                topk_w = ones_buf[: M * top_k].fill_(1.0)
            else:
                topk_w = topk_weights.contiguous().view(-1)

            # topk_ids_dev was acquired before the align branch (no sync):
            # it is the on-device routing the GPU align reads when enabled,
            # and the input the CPU fallback .cpu()'s otherwise.

            # Activation parameters.
            if activation == MoEActivation.SITU:
                act_code = 1  # SiTU
                beta_raw = getattr(self.moe_config, "activation_situ_beta", None)
                beta = float(beta_raw) if beta_raw is not None else 1.0
                linear_beta_raw = getattr(
                    self.moe_config, "activation_situ_linear_beta", None
                )
                linear_beta = (
                    float(linear_beta_raw) if linear_beta_raw is not None else 25.0
                )
            else:
                act_code = 0  # SwiGLU
                beta = 0.0
                linear_beta = 0.0

            swiglu_limit_raw = getattr(self, "gemm1_clamp_limit", None)
            swiglu_limit = (
                float(swiglu_limit_raw) if swiglu_limit_raw is not None else 4.0
            )

            # Scales and biases (from quant_config properties).
            w13_scale = self.w1_scale
            w2_scale = self.w2_scale
            b13 = self.w1_bias
            b2 = self.w2_bias

            # --- act_scratch [EM, ispp] bf16 --------------------------------
            # Prefer the framework-provided workspace13 (cudagraph-pool,
            # replay-stable) when already large enough; otherwise fall
            # back to the capture-safe persistent scratch.
            _ws13_flat = workspace13.view(-1)
            need_act = EM * ispp
            if _ws13_flat.numel() >= need_act:
                act_scratch = _ws13_flat[:need_act].view(EM, ispp)
            else:
                act_scratch = _scratch.get(
                    ("act", ispp), need_act, dtype=torch.bfloat16, device=dev,
                    capacity=max_em_count(max_M, top_k, local_n) * ispp,
                    name="act_scratch",
                )

            # --- out_fp32 [M, K] fp32 - persistent, keyed by ("out", K)
            # so MoE layers with different hidden dimensions on a rank
            # do not collide on a single buffer sized by the first
            # layer's K (which would then RuntimeError during capture
            # for a larger-K layer, or silently truncate). The per-call
            # torch.empty() here was the primary capture/replay fault. -
            out_fp32 = _scratch.get(
                ("out", K), M * K, dtype=torch.float32, device=dev,
                capacity=max_M * K, name="out_fp32",
            )
            # The down/combine kernels accumulate into ``out`` via
            # atomicAdd (see moe_fused.hip: down_combine_kernel,
            # down_combine_kernel_prefill, combine_kernel), so the buffer
            # MUST be zeroed before every launch - otherwise this call's
            # result accumulates onto the previous pass's residuals (the
            # documented caller contract; cf. test_capi_moe.hip's
            # hipMemset(dOut, 0, ...) before each call). The zero runs on
            # torch.cuda.current_stream() - the same stream the kernels
            # launch on below - so under cudagraph capture the memset is
            # recorded into the graph and replays correctly.
            out_fp32.zero_()

            # `stream` was acquired before the align branch above so the
            # on-device moe_align (when enabled) is ordered on the SAME
            # stream as this GEMM launch and the fp32->bf16 copy below.
            # This removes the device-wide synchronize the wrapper
            # previously needed as a cross-stream correctness guard (that
            # drain serialised every TP all-to-all and made each MoE layer
            # a per-step barrier).

            with record_function("moe:apply.launch"):
                moe_fn(
                    ctypes.c_void_p(hidden_states.data_ptr()),
                    ctypes.c_void_p(w1.data_ptr()),
                    ctypes.c_void_p(w13_scale.data_ptr()),
                    ctypes.c_void_p(w2.data_ptr()),
                    ctypes.c_void_p(w2_scale.data_ptr()),
                    ctypes.c_void_p(topk_ids_dev.data_ptr()),
                    ctypes.c_void_p(topk_w.data_ptr()),
                    ctypes.c_void_p(act_scratch.data_ptr()),
                    ctypes.c_void_p(out_fp32.data_ptr()),
                    ctypes.c_int(M),
                    ctypes.c_int(K),
                    ctypes.c_int(ispp),
                    ctypes.c_int(top_k),
                    ctypes.c_void_p(d_sids.data_ptr()),
                    ctypes.c_void_p(d_eids.data_ptr()),
                    ctypes.c_int(EM),
                    ctypes.c_float(swiglu_limit),
                    ctypes.c_int(act_code),
                    ctypes.c_float(beta),
                    ctypes.c_float(linear_beta),
                    ctypes.c_void_p(b13.data_ptr()) if b13 is not None else None,
                    ctypes.c_void_p(b2.data_ptr()) if b2 is not None else None,
                    ctypes.c_int(block_size),
                    ctypes.c_void_p(stream),
                )

            # Convert fp32 output to bf16 and write to vLLM's output tensor.
            output.copy_(out_fp32.view(M, K).to(torch.bfloat16))

    _VkernelFusedExperts_cls = VkernelFusedExperts
    return VkernelFusedExperts


def __getattr__(name):
    """PEP 562: build :class:`VkernelFusedExperts` lazily (imports vLLM
    only on first access, so the torch-only helpers stay usable without
    vLLM)."""
    if name == "VkernelFusedExperts":
        return _build_vllm_experts()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
