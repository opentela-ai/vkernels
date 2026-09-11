"""cutedsl kernels.

Adopted verbatim from floe's ``engine/runner/kernels/cutedsl.py`` (the
#64/#65 thin-adapter model inverts here: vkernels owns the kernel, floe
imports ``vkernels.cutedsl`` behind its existing try/except dispatch).
The ``FLOE_CUTEDSL_*`` environment-variable names below are floe's CLI
contract (``--cutedsl-swiglu`` / ``--cutedsl-attention``) and are kept
unchanged so both repos stay flag-compatible.

This is the **NVIDIA/GB10** kernel track for
:mod:`floe.engine.runner.models.qwen35`. It is gated behind
:func:`cutedsl_available` (GB10 is SM121 = Blackwell, GeForce-class). The
MMA path implemented here is **SM120 warp-level**
``mma.sync.m16n8k16`` + ``ldmatrix`` (Ampere-class ``TensorOpGemm``,
vendored in ``cutedsl_tensorop_gemm.py``) — *not* SM90 wgmma or SM100
tcgen05: see the dead-end notes below for why the other CUTLASS DSL
kernels don't run on this box (CUTLASS DSL 4.5.2). When the DSL/runtime is
unavailable (no CUDA, AMD, CPU tests, or a shape the fixed-tile kernel
can't cover), the dispatchers here fall back to the eager PyTorch path,
which remains the cross-checked oracle
(see ``tests/test_qwen35_self_consistency.py``).

Design follows the same two-implementation model as ``runner/kernels/``: the
eager expression in ``qwen35_arch`` is the reference; a cutedsl device kernel
is "correct" precisely because it matches that reference under
:func:`torch.testing.assert_close`. No kernel is wired in unconditionally —
the model is never broken by an in-progress port.

On-ds5 references for each rung (paths inside the ``floe-qwen35-27b``
container, which has ``cutlass.cute`` 4.5.2 + torch 2.10a0 NVIDIA):

* Rung 1 (this module, SwiGLU = 2 GEMMs, gate+up fused): the implemented
  kernel is the **Ampere** warp-level-MMA GEMM, vendored verbatim into
  ``cutedsl_tensorop_gemm.py`` from
  ``/opt/pytorch/pytorch/third_party/cutlass/examples/python/CuTeDSL/
  ampere/tensorop_gemm.py``. This is the SM120/SM121 (GB10) path
  (``mma.sync.m16n8k16`` + ``ldmatrix`` + ``cp.async``). The *other*
  SM-class kernels in the tree are dead ends on CUTLASS DSL 4.5.2 / GB10
  and are kept here only as algorithmic references:

  - ``quack/gemm_sm120.py`` + ``quack/gemm_interface.py`` — full SM120
    GEMM, but ``import quack.gemm`` fails:
    ``from cutlass.cute.experimental import iket`` (absent on 4.5.2).
  - ``.../blackwell_geforce/dense_gemm.py`` — TMA+persistent; *compiles*
    on GB10 but fails in its epilogue at ``cute.arch.ProxyKind`` (absent
    on 4.5.2).
  - ``.../blackwell/tutorial_gemm/fp16_gemm_*.py`` — uses ``tcgen05``
    MMA, which rejects ``sm_121a`` (GeForce Blackwell) outright;
    SM100-only.
  - ``.../sglang/jit_kernel/cutedsl_dsv3_fused_a_gemm.py`` — production
    fused gate+up A GEMM; SM90 hand-rolled MMA (raw ``mma.sync`` +
    ``cp.async`` + ``ldmatrix``), shape-constrained (K%1024==0, N in
    {2112,6144}). Algorithm only.
* Rung 1 (this module, SwiGLU): DONE — gate+up FUSED into one GEMM
  over a concatenated ``[2*inter, H]`` weight (the DeepSeekV3/sglang
  canonical MLP), so SwiGLU is 2 GEMMs (fused gate+up, then down) + one
  bf16 ``silu(gate)*up``, bit-identical to the eager oracle. The SM90
  ``cutedsl_dsv3_fused_a_gemm.py`` above is now an *algorithm* reference
  only — the fusion here reuses the proven Ampere ``TensorOpGemm``.
* Rung 2 (next): full-attention SDPA (online softmax, SM120 MMA) +
  Q/K/V/O.
* Rung 3 (GatedDeltaNet delta-rule scan):
  - ``.../sglang/jit_kernel/cutedsl_gdn.py``
    (production GDN decode kernel = qwen35's linear-attention core)

The cutedsl SwiGLU (2 GEMMs + bf16 ``silu(gate)*up``, bit-identical to the
eager oracle at qwen35 serving shapes) is dispatched from
:func:`floe.engine.runner.kernels.qwen35_primitives.SwiGLU.forward` behind
the ``FLOE_CUTEDSL_SWIGLU=1`` env switch. Run the parity gate on ds5 with::

    FLOE_CUTEDSL_SWIGLU=1 python -m pytest tests/test_cutedsl_swiglu.py -x
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "cutedsl_available",
    "sm_version",
    "swiglu_enabled",
    "swiglu",
]

# A single global "the cutedsl path blew up once; don't keep trying" flag, so
# an in-progress kernel can't turn every forward into a noisy traceback. The
# first failure is logged; subsequent calls fall back to the eager oracle.
_swiglu_cutedsl_broken = threading.Event()


class _SwigluUnsupported(Exception):
    """This call's shape/dtype/contiguity is outside the (fixed-tile) kernel.

    Distinct from a real compile/runtime bug: the dispatcher falls back to
    the eager oracle FOR THIS CALL ONLY (no global disable), so the tiny
    test config and oddly-shaped inputs keep working while real qwen35
    serving shapes go through the kernel.
    """


# Compiled-kernel cache, keyed by (M, N, K, dtype). The vendored
# ``TensorOpGemm`` (``cutedsl_tensorop_gemm.py``) uses a *fixed* CTA tile
# (128, 128, 32) and is compiled per distinct (M, N, K) via ``cute.compile``;
# compile is the dominant cost (~1s) so the cache makes a warm decode loop
# free. ``M`` is part of the key because the warp-level-MMA tile scheduling
# bakes M in at compile time (a dynamic-M variant is a later rung).
_gemm_cache: dict = {}
_gemm_lock = threading.Lock()


def sm_version() -> Optional[int]:
    """Return the major CUDA compute capability of device 0, or ``None``.

    GB10 reports ``(12, 1)`` → SM121 (Blackwell, GeForce-class). The SM120
    warp-level-MMA cutedsl path compiles on SM120/SM121; SM90/SM100 kernels
    (wgmma / tcgen05) are a *different* port target and are not what this
    module dispatches to on GB10.
    """
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.get_device_capability(0)[0])
    except Exception:
        return None


def cutedsl_available() -> bool:
    """True iff the CuTe DSL runtime + a Blackwell (SM≥12) device are usable.

    Mirrors the guard the qwen35 container honours: ``cutlass.cute`` and its
    ``from_dlpack`` torch bridge must import, and we must be on an NVIDIA
    device with a Blackwell-or-later compute capability (the only arch for
    which the SM120 warp-level MMA kernels here are emitted).
    """
    try:
        import cutlass.cute as cute  # noqa: F401  (import-time JIT plumbing)
        from cutlass.cute.runtime import from_dlpack  # noqa: F401
    except Exception:
        return False
    # cute.gemm is the load-bearing primitive every rung builds on; make sure
    # the installed DSL actually exposes it before promising anything.
    if not hasattr(cute, "gemm") or not hasattr(cute, "kernel"):
        return False
    return sm_version() is not None and sm_version() >= 12


@dataclass
class CutedslConfig:
    """Runtime opt-ins for the cutedsl dispatchers (see :func:`configure`).

    ``None`` means "not explicitly set": the gate falls back to its legacy
    environment variable (``FLOE_CUTEDSL_*``), then to off. An explicit
    ``True``/``False`` (from a ``floe serve`` CLI flag or a benchmark) always
    wins over the environment.
    """

    swiglu: Optional[bool] = None
    attention: Optional[bool] = None


CONFIG = CutedslConfig()


def configure(swiglu: Optional[bool] = None, attention: Optional[bool] = None) -> None:
    """Set the cutedsl opt-ins for this process (CLI/benchmark entry point).

    Pass only the knobs you want to force; untouched knobs (and knobs passed
    as ``None``) keep consulting their legacy ``FLOE_CUTEDSL_*`` environment
    variable and default to off. ``floe serve`` calls this once at model-load
    time from the ``--cutedsl-swiglu`` / ``--cutedsl-attention`` flags.
    """
    CONFIG.swiglu = swiglu
    CONFIG.attention = attention


def _resolve(opt: Optional[bool], env_name: str) -> bool:
    if opt is not None:
        return opt
    return os.environ.get(env_name, "0") == "1"


def swiglu_enabled() -> bool:
    """Opt-in for the cutedsl SwiGLU path.

    Off by default so ``SwiGLU.forward`` is byte-identical to the eager
    reference unless a caller asks for it (``cutedsl.configure(swiglu=True)``
    or, legacy, ``FLOE_CUTEDSL_SWIGLU=1``). Requires
    :func:`cutedsl_available`.
    """
    return _resolve(CONFIG.swiglu, "FLOE_CUTEDSL_SWIGLU") and cutedsl_available()


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


def _swiglu_eager(x, gate_weight, up_weight, down_weight):
    """The PyTorch oracle. ``weight`` is an ``nn.Linear`` weight ([out, in]),
    i.e. the projection is ``x @ weight.T`` — identical to
    :class:`SwiGLU <floe.engine.runner.kernels.qwen35_primitives.SwiGLU>`.
    """
    import torch.nn.functional as F

    g = x @ gate_weight.t()
    u = x @ up_weight.t()
    act = F.silu(g) * u
    return act @ down_weight.t()


def _swiglu_cutedsl(x, gate_weight, up_weight, down_weight):
    """cutedsl SwiGLU on GB10 (SM121): two SM120 warp-level-MMA GEMMs.

    Gate and up are FUSED into a single GEMM over a concatenated
    ``[2*inter, H]`` weight (the DeepSeekV3/sglang canonical MLP), so the
    whole SwiGLU is two GEMMs (fused gate+up, then down) + one bf16
    ``silu(gate) * up`` -- one fewer GEMM and one fewer x read than Rung 1.

    Mirrors :func:`_swiglu_eager` exactly -- ``silu(gate) * up`` is computed in
    ``x.dtype`` (bf16), matching :meth:`SwiGLU.forward`, so the path is
    bit-identical to the eager oracle at qwen35 serving shapes (verified on
    ds5: H=5120, inter=17408, M in 1..16 -> max|diff| = 0.0).

    Each GEMM drives the vendored ``TensorOpGemm`` (Ampere warp-level
    ``mma.sync.m16n8k16`` + ``ldmatrix`` + ``cp.async``; see
    ``cutedsl_tensorop_gemm.py``), compiled once per (M, N, K) and bridged
    from torch with ``from_dlpack``. The weight arguments are ``nn.Linear``
    weights ``[out, in]`` i.e. the projection is ``x @ weight.T``; because the
    GEMM computes ``C[m,n] = sum_k A[m,k] * B[n,k]``, the ``[N, K]`` weights go
    in *untransposed* (K = the nn.Linear ``in`` dim, contiguous).

    Raises :class:`_SwigluUnsupported` for inputs the fixed-tile kernel can't
    handle (non-bf16, non-contiguous, K not %32, N not %128 -- e.g. the tiny
    test config), so :func:`swiglu` can fall back to eager per call.
    """
    import torch  # noqa: F401  (gated; CPU/AMD must still import this module)
    import torch.nn.functional as F
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack
    from .cutedsl_tensorop_gemm import TensorOpGemm

    # --- shape/dtype/contiguity guards -> per-call eager fallback --------
    if x.dtype != torch.bfloat16:
        raise _SwigluUnsupported(f"cutedsl SwiGLU needs bf16, got {x.dtype}")
    if not (x.is_contiguous() and gate_weight.is_contiguous() and up_weight.is_contiguous() and down_weight.is_contiguous()):
        raise _SwigluUnsupported("cutedsl SwiGLU needs contiguous tensors")
    if x.dim() != 2:
        raise _SwigluUnsupported(f"cutedsl SwiGLU needs 2-D x, got {x.dim()}-D")
    if not (gate_weight.dim() == up_weight.dim() == down_weight.dim() == 2):
        raise _SwigluUnsupported("cutedsl SwiGLU needs 2-D weights")

    m, hidden = x.shape
    inter = gate_weight.shape[0]
    # TensorOpGemm CTA tile is fixed (128, 128, 32): every contraction K must
    # be a multiple of 32 and every output N a multiple of 128. Both hold for
    # real qwen35 (H=5120=32*160, inter=17408=32*544=128*136) but NOT for the
    # tiny test config (H=16, inter=32) -- which is exactly the case we want
    # to fall back to eager, not crash.
    for label, k in (("gate/up K", hidden), ("down K", inter)):
        if k % 32 != 0:
            raise _SwigluUnsupported(f"{label}={k} not a multiple of 32 (TensorOpGemm CTA bK)")
    for label, n in (("gate/up N", inter), ("down N", hidden)):
        if n % 128 != 0:
            raise _SwigluUnsupported(f"{label}={n} not a multiple of 128 (TensorOpGemm CTA bN)")
    # Fused gate+up GEMM outputs N=2*inter, which needs 2*inter % 128 == 0.
    # inter % 128 == 0 (just checked) implies inter % 64 == 0 implies
    # 2*inter % 128 == 0, so this holds whenever the gate/up N check does.
    if gate_weight.shape != (inter, hidden) or up_weight.shape != (inter, hidden) or down_weight.shape != (hidden, inter):
        raise _SwigluUnsupported(f"weight shapes inconsistent with SwiGLU (gate/up [inter,{hidden}], down [{hidden},inter])")

    atom_layout_mnk = (2, 2, 1)
    acc_dtype = cutlass.Float32
    bf16 = cutlass.BFloat16
    # divisibility for mark_compact_shape_dynamic == 128 // dtype.width == 8
    # for bf16 (matches the upstream example's derivation).
    divisibility = 128 // bf16.width

    def _wrap(t2):
        # Inference kernel: the weights arrive with requires_grad=True from a
        # plain nn.Linear, and torch's __dlpack__ refuses grad-tracking
        # tensors. detach() is free (same storage) and safe -- the kernel
        # never participates in autograd.
        t2 = t2.detach()
        # 2-D [X, Y] (Y contiguous, K-major for x/w or N-major for c) -> the
        # 3-D cute tensor [X, Y, 1] the kernel expects, with the contiguous
        # leading dim marked dynamic (so N/K needn't match the compile shape
        # as long as the divisibility above holds).
        return from_dlpack(t2.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=divisibility)

    def _gemm(a, b):
        # a:[M,K] K-major, b:[N,K] K-major -> c:[M,N] N-major ; = a @ b.T
        mm, k = a.shape
        n = b.shape[0]
        c = torch.empty(mm, n, dtype=cutlass_torch.dtype(bf16), device=a.device)
        with _gemm_lock:
            compiled = _gemm_cache.get((mm, n, k))
            if compiled is None:
                compiled = cute.compile(TensorOpGemm(bf16, bf16, acc_dtype, atom_layout_mnk), _wrap(a), _wrap(b), _wrap(c))
                _gemm_cache[(mm, n, k)] = compiled
        compiled(_wrap(a), _wrap(b), _wrap(c))
        return c

    # Fuse gate + up into ONE GEMM (the DeepSeekV3/sglang canonical MLP):
    # concat the two [inter, H] weights along their output dim -> [2*inter, H],
    # a single GEMM reads x once and yields [M, 2*inter] = [gate | up]. This
    # halves the x reads and removes one kernel launch vs two separate GEMMs;
    # at decode (M<=16) the GEMMs are ~3 us and launch-bound so the win is
    # marginal (~1.0-1.1x on ds5), but it matches the production pattern and
    # sets up the fused-epilogue rung. cat keeps H contiguous (K-major).
    gate_up_w = torch.cat((gate_weight, up_weight), dim=0)  # [2*inter, H]
    gate_up = _gemm(x, gate_up_w)  # [M, 2*inter]
    gate = gate_up[:, :inter]
    up = gate_up[:, inter:]
    act = F.silu(gate) * up  # bf16, == eager SwiGLU
    return _gemm(act, down_weight)  # [M, hidden]


def swiglu(x, gate_weight, up_weight, down_weight):
    """Full SwiGLU: ``down(silu(x@gate_w.T) * (x@up_w.T))``.

    Dispatches to the cutedsl kernel when :func:`swiglu_enabled`; otherwise,
    or once the cutedsl path has failed with a real bug, returns the eager
    PyTorch oracle. A :class:`_SwigluUnsupported` (shape/dtype/contiguity
    outside the fixed-tile kernel, e.g. the tiny test config) falls back to
    the eager oracle for *that call only* -- the global switch stays on so
    real qwen35 serving shapes keep using the kernel. The return is always a
    fresh tensor in ``x``'s dtype/device.
    """
    if not swiglu_enabled() or _swiglu_cutedsl_broken.is_set():
        return _swiglu_eager(x, gate_weight, up_weight, down_weight)
    try:
        return _swiglu_cutedsl(x, gate_weight, up_weight, down_weight)
    except _SwigluUnsupported:
        # Outside the kernel's fixed-tile envelope -- not a bug, just punt
        # this call to eager and stay armed for the next (supported) one.
        return _swiglu_eager(x, gate_weight, up_weight, down_weight)
    except Exception:
        # A genuine compile/runtime failure: disable globally so an in-progress
        # kernel can't turn every forward into a noisy traceback.
        if not _swiglu_cutedsl_broken.is_set():
            _swiglu_cutedsl_broken.set()
            import traceback

            print("[cutedsl] SwiGLU kernel error; using eager oracle:\n" + traceback.format_exc(), flush=True)
        return _swiglu_eager(x, gate_weight, up_weight, down_weight)


# ---------------------------------------------------------------------------
# Full attention (FlashAttention-V2 core)
# ---------------------------------------------------------------------------

# The vendored kernel (``cutedsl_flash_attn.py``, Ampere warp-level MMA +
# cp.async + online softmax) writes the real per-block dynamic shared memory
# budget into its tile sizes. The example's ``can_implement`` checks the
# *sm_80* capacity (164 KB), but SM121's actual launch limit is 101376 B:
# a 131072 B config (head_dim=256, m=128/n=64) compiles and then FAILS at
# launch. Verified block configs per head_dim (smem = (m*hd + n*hd*2)*2 B):
_FA_BLOCK = {
    # head_dim: (m_block, n_block)  smem
    256: (64, 64),  # 98304 B  (qwen35: head_dim=256)
    128: (128, 128),  # 98304 B
    64: (128, 128),  # 49152 B
    32: (128, 128),  # 24576 B
}
_FA_SMEM_MAX = 101376  # bytes; real SM121 per-block dynamic-smem cap

_attention_cutedsl_broken = threading.Event()
# Compiled FA2 kernels. The kernel computes its grid from mQ.shape at CALL
# time and the wrapped layouts are dynamic, so one compile per
# (head_dim, block config, is_causal, heads) serves every seqlen -- verified
# on ds6 (S=512 kernel reused at S=768 / S=500 / decode S_q=1).
_fa_cache: dict = {}
_fa_lock = threading.Lock()


def attention_enabled() -> bool:
    """Opt-in for the cutedsl FlashAttention-V2 core.

    Off by default so ``FullAttention.forward`` stays byte-identical to the
    eager reference unless a caller asks for it
    (``cutedsl.configure(attention=True)`` or, legacy,
    ``FLOE_CUTEDSL_ATTENTION=1``). Requires :func:`cutedsl_available`.
    """
    return _resolve(CONFIG.attention, "FLOE_CUTEDSL_ATTENTION") and cutedsl_available()


class _AttentionUnsupported(Exception):
    """Shape/dtype the FA2 kernel can't cover -- per-call eager fallback."""


def _attention_eager(q, k_kv, v_kv, *, is_causal, scale):
    """The exact eager SDPA path ``FullAttention.forward`` runs today.

    q ``[S, H, D]`` -> ``[1, H, S, D]`` view; K/V ``[n_kv, S_kv, D]``
    expanded to the Q-head count via ``repeat_interleave`` when grouped.
    Returns the ``F.scaled_dot_product_attention`` output ``[1, H, S, D]``.
    """
    import torch.nn.functional as F

    q4 = q.transpose(0, 1).unsqueeze(0)
    k_a = k_kv.unsqueeze(0)
    v_a = v_kv.unsqueeze(0)
    group = q.shape[1] // k_kv.shape[0]
    if group != 1:
        k_a = k_a.repeat_interleave(group, dim=1)
        v_a = v_a.repeat_interleave(group, dim=1)
    return F.scaled_dot_product_attention(q4, k_a, v_a, is_causal=is_causal, scale=scale)


def _attention_cutedsl(q, k_kv, v_kv, *, is_causal, scale):
    """FlashAttention-V2 core (online softmax) on GB10 via the vendored kernel.

    ``q`` is ``[S, H, D]`` *contiguous* bf16 -- the qwen35-native layout, and
    exactly the ``[B=1, S, H, D]`` the kernel wants (the eager path needs a
    transpose; this one doesn't). ``k_kv``/``v_kv`` are ``[n_kv, S_kv, D]`` in
    any strides (kv-cache views are fine). The kernel has **no GQA indexing**
    (K/V are read by Q-head id), so K/V are expanded to the Q-head count with
    one grouped broadcast copy each -- the same data volume the eager path's
    ``repeat_interleave`` materialises. Causal masking is only dispatched for
    *pure prefill* (``S_q == S_kv``): PyTorch's ``is_causal`` alignment for
    ``S_q != S_kv`` (continued prefill off a prefix) is a different contract,
    so those calls fall back to eager rather than risk a subtle mismatch.

    Returns ``[1, H, S, D]`` (the ``F.scaled_dot_product_attention``
    convention) as a free permute of the kernel's ``[1, S, H, D]`` output, so
    the caller's ``transpose(1, 2).squeeze(0)`` lands back on the contiguous
    layout and the final ``reshape`` is free.

    Raises :class:`_AttentionUnsupported` for inputs outside the kernel's
    envelope so :func:`attention` can fall back per call.
    """
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda_drv
    from .cutedsl_flash_attn import FlashAttentionForwardAmpere

    # --- guards -> per-call eager fallback ------------------------------
    if not (q.dtype == k_kv.dtype == v_kv.dtype == torch.bfloat16):
        raise _AttentionUnsupported(f"cutedsl attention needs bf16, got {q.dtype}/{k_kv.dtype}/{v_kv.dtype}")
    if q.dim() != 3 or k_kv.dim() != 3 or v_kv.dim() != 3:
        raise _AttentionUnsupported("cutedsl attention needs 3-D q/k/v")
    if not q.is_contiguous():
        raise _AttentionUnsupported("cutedsl attention needs contiguous q")
    s_q, n_heads, head_dim = q.shape
    n_kv, s_kv = int(k_kv.shape[0]), int(k_kv.shape[1])
    if tuple(v_kv.shape) != tuple(k_kv.shape) or int(k_kv.shape[2]) != head_dim:
        raise _AttentionUnsupported("k/v shapes inconsistent with q")
    if n_kv == 0 or n_heads % n_kv != 0:
        raise _AttentionUnsupported(f"GQA requires n_heads % n_kv == 0 (H={n_heads}, n_kv={n_kv})")
    if head_dim not in _FA_BLOCK:
        raise _AttentionUnsupported(f"no verified block config for head_dim={head_dim} (have {sorted(_FA_BLOCK)})")
    if is_causal and s_kv != s_q:
        raise _AttentionUnsupported(f"causal dispatch covers pure prefill only (S_q={s_q} != S_kv={s_kv}; continued-prefill alignment differs)")
    if s_q == 0 or s_kv == 0:
        raise _AttentionUnsupported("empty sequence")
    m_block, n_block = _FA_BLOCK[head_dim]
    if (m_block * head_dim + n_block * head_dim * 2) * 2 > _FA_SMEM_MAX:
        raise _AttentionUnsupported("block config exceeds SM121 smem budget")

    group = n_heads // n_kv
    # q: [1, S, H, D] is a free view; K/V: one grouped broadcast copy each
    # ([S_kv, n_kv, group, D] view <- [S_kv, n_kv, 1, D] source), giving
    # head h <- kv head h // group -- exactly repeat_interleave's grouping.
    q4 = q.unsqueeze(0)
    k4 = torch.empty(1, s_kv, n_heads, head_dim, dtype=q.dtype, device=q.device)
    k4.view(s_kv, n_kv, group, head_dim).copy_(k_kv.transpose(0, 1).unsqueeze(2))
    v4 = torch.empty_like(k4)
    v4.view(s_kv, n_kv, group, head_dim).copy_(v_kv.transpose(0, 1).unsqueeze(2))
    o4 = torch.empty_like(q4)

    def _wrap(t):
        # Inference kernel: detach so grad-tracking q/k/v (e.g. a caller
        # outside torch.no_grad) can still cross the __dlpack__ bridge; the
        # kernel never participates in autograd.
        t = t.detach()
        # The example's create_tensor recipe: dynamic layout, contiguous
        # leading (head_dim) axis, divisibility 8 for bf16.
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=3).mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(), divisibility=8)

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (head_dim, m_block, n_block, bool(is_causal), n_heads, n_kv)
    with _fa_lock:
        compiled = _fa_cache.get(key)
        if compiled is None:
            compiled = cute.compile(FlashAttentionForwardAmpere(head_dim, m_block, n_block, 128, is_causal), _wrap(q4), _wrap(k4), _wrap(v4), _wrap(o4), scale, stream)
            _fa_cache[key] = compiled
    compiled(_wrap(q4), _wrap(k4), _wrap(v4), _wrap(o4), scale, stream)
    # kernel output [1, S, H, D] -> F.sdpa convention [1, H, S, D] (free view).
    return o4.permute(0, 2, 1, 3)


def attention(q, k_kv, v_kv, *, is_causal, scale):
    """Attention core ``softmax(scale * q k^T) v`` (``FullAttention.forward``).

    Dispatches to the cutedsl FlashAttention-V2 kernel when
    :func:`attention_enabled`; otherwise -- or once the kernel has failed with
    a real bug -- the exact eager PyTorch path. An
    :class:`_AttentionUnsupported` (shape outside the kernel's envelope, e.g.
    the tiny test config or continued prefill) falls back for *that call
    only*; the global switch stays armed for supported shapes. Always returns
    a ``[1, H, S_q, D]`` tensor, matching ``F.scaled_dot_product_attention``.
    """
    if not attention_enabled() or _attention_cutedsl_broken.is_set():
        return _attention_eager(q, k_kv, v_kv, is_causal=is_causal, scale=scale)
    try:
        return _attention_cutedsl(q, k_kv, v_kv, is_causal=is_causal, scale=scale)
    except _AttentionUnsupported:
        # Outside the kernel's envelope -- not a bug, punt this call to eager
        # and stay armed for the next (supported) one.
        return _attention_eager(q, k_kv, v_kv, is_causal=is_causal, scale=scale)
    except Exception:
        # A genuine compile/runtime failure: disable globally so an in-progress
        # forward can't turn every step into a noisy traceback.
        if not _attention_cutedsl_broken.is_set():
            _attention_cutedsl_broken.set()
            import traceback

            print("[cutedsl] attention kernel error; using eager oracle:\n" + traceback.format_exc(), flush=True)
        return _attention_eager(q, k_kv, v_kv, is_causal=is_causal, scale=scale)
