"""Distributed (TP / EP / PP) orchestration for the fused MXFP4 MoE.

Python-side host reference for the distributed layer defined in
``src/c/vkernels/dist/dist_moe.hpp`` (issue #18): shards the fused-MoE
weights so per-rank shards are consumed verbatim by the fused kernel, and
runs the multi-rank forwards the single-device kernel cannot:

* **TP** — gate_up weights split along ``hidden`` (column-parallel), down
  weights along ``ispp`` (row-parallel).  The two linear stages are
  separated so rank partials can be all-reduced *before* the nonlinear
  epilogues — the all-reduce points where the caller runs its own
  collective.  ``dist_moe_tp`` runs the whole TP forward in-process over
  ``tp`` simulated ranks and matches :func:`vkernels.kernels.fused_moe_mxfp4`
  (the single-rank oracle).
* **EP** — experts are partitioned across ranks; each token is dispatched to
  the rank owning its chosen experts (all-to-all of activations), computed
  locally, and the weighted partial outputs are summed back for the routed
  combine (all-to-all of outputs).  ``dist_moe_ep`` returns per-rank partial
  outs; their element-wise sum is the MoE output.
* **PP** — the MoE layer is stage-local; ``pp_boundary_send`` / ``recv``
  define the stage-boundary transfer (host reference for the graph-capturable
  primitive of issue #10), and :func:`round_bf16` re-quantises an fp32 stage
  output to the bf16 activation format the next stage consumes.

Everything is pure numpy (plus the public :mod:`vkernels.kernels` and
:mod:`vkernels.comm` APIs), so it runs with or without the compiled
extension and is usable as the semantic reference for a vLLM-style
integration.

Kimi-K3 layout (issue #18): E=112, D_INTER (ispp)=3072, hidden=7168, TP8 →
per-rank shards 896 x 384, both multiples of 64 (BLOCK_K) and 32
(ue8m0 group) — see :func:`tp_plan`.
"""

from __future__ import annotations

import numpy as np

from vkernels import kernels, comm

_F32 = np.dtype(np.float32)


# ---------------------------------------------------------------------------
# bf16 / fp4 E2M1 / ue8m0 helpers (mirror moe_fused.cpp)
# ---------------------------------------------------------------------------

# E2M1 nibble → value (index = nibble).  Subnormals: 0b0001 → 0.25, 0b1001 →
# -0.25; 0b0110/0b1110 → ±inf; 0b0111/0b1111 → NaN.
_FP4_VALUES = np.array(
    [0.0, 0.25, 1.0, 1.5, 2.0, 3.0, np.inf, np.nan,
     0.0, -0.25, -1.0, -1.5, -2.0, -3.0, -np.inf, np.nan],
    dtype=np.float32,
)


def _f32_to_bf16(v: np.ndarray) -> np.ndarray:
    """Round fp32 to bf16 (RNE), returning uint16 bit patterns."""
    bits = v.astype(np.float32).view(np.uint32)
    lsb = (bits >> 16) & 1
    bits = bits + np.uint32(0x7FFF) + lsb
    return (bits >> 16).astype(np.uint16)


def _bf16_to_f32(v: np.ndarray) -> np.ndarray:
    return (v.astype(np.uint32) << 16).view(np.float32)


def _ue8m0_to_float(s: np.ndarray) -> np.ndarray:
    """ue8m0 scale bytes → float (2^(s-127), 0xFF → 0.0)."""
    out = np.where(s == 0xFF, 0.0, (s.astype(np.int32) - 127).astype(np.float32))
    return np.exp2(out)


def _dequant_full(packed: np.ndarray, scale: np.ndarray, n_rows: int, k: int,
                  stride_packed: int, stride_scale: int,
                  group_size: int) -> np.ndarray:
    """Dequant a full [n_rows, k] weight slice (packed [n_rows, k/2] + scale
    [n_rows, k/group]) to bf16 (uint16 [n_rows, k]), rounding once — mirrors
    ``dequant_weight_tile`` on a row-strided buffer."""
    pb = packed.reshape(n_rows, stride_packed)
    lo = (pb & 0x0F).astype(np.intp)
    hi = (pb >> 4).astype(np.intp)
    vals = np.stack([_FP4_VALUES[lo], _FP4_VALUES[hi]], axis=-1).reshape(n_rows, k)
    sc = np.repeat(_ue8m0_to_float(scale.reshape(n_rows, stride_scale)),
                   group_size, axis=-1)
    return _f32_to_bf16(vals * sc)


# ---------------------------------------------------------------------------
# Stage split of the fused MoE (mirrors moe_fused.cpp stage functions)
# ---------------------------------------------------------------------------
# The fused call is split at the linear/nonlinear boundary so TP ranks can
# all-reduce before the activation:
#
#   stage 0a  _moe_gateup        gate_up[EM, 2*ispp] = A_slice @ w13^T
#   stage 0b  _moe_act_epilogue  act[EM, ispp] = act(gate_up + b13)
#   stage 1a  _moe_down          partial[EM, hidden] = act_slice @ w2^T
#   stage 1b  _moe_combine       out[M, hidden] += (partial + b2) * topk_w
#
# The GEMMs accumulate in fp32 (numpy matmul — order differs from the C++
# oracle's scalar loops, so results agree within fp32 tolerance, not
# bit-for-bit).

_BLOCK_M, _BLOCK_N, _BLOCK_K = 16, 64, 64


def _moe_gateup(A, w13, w13_scale, sorted_ids, expert_ids, gate_up,
                M, a_stride, k_base, hidden_k, ispp, top_k, EM, group_size):
    """gate_up [EM, 2*ispp] fp32 = A[:, k_base:k_base+hidden_k] @ w13^T."""
    A = np.asarray(A).ravel()
    N = M * top_k
    w13_e = 2 * ispp * (hidden_k // 2)
    w13s_e = 2 * ispp * (hidden_k // group_size)
    stride_packed = hidden_k // 2
    stride_scale = hidden_k // group_size
    for mb in range(EM // _BLOCK_M):
        expert = int(expert_ids[mb])
        if expert < 0:
            continue
        tb = mb * _BLOCK_M
        # Dequant this expert's whole shard once; slice 64-col tiles from it.
        p0 = expert * w13_e
        s0 = expert * w13s_e
        wf = _bf16_to_f32(_dequant_full(w13[p0:p0 + w13_e], w13_scale[s0:s0 + w13s_e],
                                        2 * ispp, hidden_k, stride_packed,
                                        stride_scale, group_size))
        wg = wf[:ispp]
        wu = wf[ispp:]
        for nb in range(ispp // _BLOCK_N):
            tA = np.zeros((_BLOCK_M, hidden_k), dtype=np.uint16)
            for m in range(_BLOCK_M):
                f = int(sorted_ids[tb + m])
                if f < N:
                    token = f // top_k
                    base = token * a_stride + k_base
                    tA[m] = A[base:base + hidden_k]
            af = _bf16_to_f32(tA)
            n0, n1 = nb * _BLOCK_N, (nb + 1) * _BLOCK_N
            acc_g = af @ wg[n0:n1].T
            acc_u = af @ wu[n0:n1].T
            for m in range(_BLOCK_M):
                r0 = (tb + m) * 2 * ispp + n0
                gate_up[r0:r0 + _BLOCK_N] = acc_g[m]
                gate_up[r0 + ispp:r0 + ispp + _BLOCK_N] = acc_u[m]


def _moe_act_epilogue(gate_up, b13, act, sorted_ids, expert_ids,
                      M, top_k, EM, ispp, swiglu_limit, activation, beta,
                      linear_beta):
    """act [EM, ispp] bf16 = activation(gate_up + b13), padding rows skipped."""
    N = M * top_k
    for mb in range(EM // _BLOCK_M):
        expert = int(expert_ids[mb])
        if expert < 0:
            continue
        tb = mb * _BLOCK_M
        for nb in range(ispp // _BLOCK_N):
            for m in range(_BLOCK_M):
                f = int(sorted_ids[tb + m])
                if f >= N:
                    continue
                g = gate_up[(tb + m) * 2 * ispp + nb * _BLOCK_N:
                            (tb + m) * 2 * ispp + nb * _BLOCK_N + _BLOCK_N].copy()
                u = gate_up[(tb + m) * 2 * ispp + ispp + nb * _BLOCK_N:
                            (tb + m) * 2 * ispp + ispp + nb * _BLOCK_N + _BLOCK_N].copy()
                if b13 is not None:
                    g += b13[expert, nb * _BLOCK_N:nb * _BLOCK_N + _BLOCK_N]
                    u += b13[expert, ispp + nb * _BLOCK_N:
                             ispp + nb * _BLOCK_N + _BLOCK_N]
                if activation == "situ":
                    sig = 1.0 / (1.0 + np.exp(-g))
                    gate_out = beta * np.tanh(g / beta) * sig
                    up_out = (linear_beta * np.tanh(u / linear_beta)
                              if linear_beta > 0 else u)
                    result = gate_out * up_out
                else:
                    gl = g.copy()
                    ul = u.copy()
                    if swiglu_limit > 0:
                        gl = np.minimum(gl, swiglu_limit)
                        ul = np.minimum(np.maximum(ul, -swiglu_limit), swiglu_limit)
                    result = gl / (1.0 + np.exp(-gl)) * ul
                act[(tb + m) * ispp + nb * _BLOCK_N:
                    (tb + m) * ispp + nb * _BLOCK_N + _BLOCK_N] = (
                    _f32_to_bf16(result.astype(np.float32)))


def _moe_down(act, w2, w2_scale, sorted_ids, expert_ids, partial,
              M, a_stride, k_base, ispp_k, hidden, top_k, EM, group_size):
    """partial [EM, hidden] fp32 = act[:, k_base:k_base+ispp_k] @ w2^T."""
    act = np.asarray(act).ravel()
    N = M * top_k
    w2_e = hidden * (ispp_k // 2)
    w2s_e = hidden * (ispp_k // group_size)
    stride_packed = ispp_k // 2
    stride_scale = ispp_k // group_size
    for mb in range(EM // _BLOCK_M):
        expert = int(expert_ids[mb])
        if expert < 0:
            continue
        tb = mb * _BLOCK_M
        wf = _bf16_to_f32(_dequant_full(w2[expert * w2_e:(expert + 1) * w2_e],
                                        w2_scale[expert * w2s_e:(expert + 1) * w2s_e],
                                        hidden, ispp_k, stride_packed,
                                        stride_scale, group_size))
        for nb in range(hidden // _BLOCK_N):
            tA = np.zeros((_BLOCK_M, ispp_k), dtype=np.uint16)
            for m in range(_BLOCK_M):
                f = int(sorted_ids[tb + m])
                if f < N:
                    base = (tb + m) * a_stride + k_base
                    tA[m] = act[base:base + ispp_k]
            n0, n1 = nb * _BLOCK_N, (nb + 1) * _BLOCK_N
            acc = _bf16_to_f32(tA) @ wf[n0:n1].T
            for m in range(_BLOCK_M):
                r0 = (tb + m) * hidden + n0
                partial[r0:r0 + _BLOCK_N] = acc[m]


def _moe_combine(partial, b2, topk_w_sorted, sorted_ids, expert_ids, out,
                 M, hidden, top_k, EM):
    """out[M, hidden] += (partial + b2) * topk_w_sorted (padding rows skipped)."""
    partial = np.asarray(partial).ravel()
    out = np.asarray(out).ravel()
    N = M * top_k
    for mb in range(EM // _BLOCK_M):
        expert = int(expert_ids[mb])
        if expert < 0:
            continue
        tb = mb * _BLOCK_M
        for nb in range(hidden // _BLOCK_N):
            for m in range(_BLOCK_M):
                f = int(sorted_ids[tb + m])
                if f >= N:
                    continue
                token = f // top_k
                weight = np.float32(topk_w_sorted[tb + m])
                val = partial[(tb + m) * hidden + nb * _BLOCK_N:
                              (tb + m) * hidden + nb * _BLOCK_N + _BLOCK_N].copy()
                if b2 is not None:
                    val += b2[expert, nb * _BLOCK_N:nb * _BLOCK_N + _BLOCK_N]
                val *= weight
                out[token * hidden + nb * _BLOCK_N:
                    token * hidden + nb * _BLOCK_N + _BLOCK_N] += val


def _align(topk_ids: np.ndarray, num_experts: int, block_size: int):
    """moe_align_block_size via the public API; returns (sorted, eids, EM)."""
    return kernels.moe_align_block_size(topk_ids, num_experts, block_size)


# ---------------------------------------------------------------------------
# TP
# ---------------------------------------------------------------------------


def tp_plan(hidden: int, ispp: int, tp: int, group_size: int = 32) -> dict:
    """Validate a TP split against the fused-kernel constraints and return the
    per-rank shard geometry (mirrors ``dist::moe_tp_plan``).

    The fused kernel's stage functions tile K in BLOCK_K=64 chunks and ue8m0
    scales span group_size=32 consecutive K elements, so each rank's shard
    must keep those multiples: ``hidden % tp == 0``, ``ispp % tp == 0`` and
    the shards divisible by 64 (hence by 32).  Kimi-K3 satisfies all for
    TP8: 7168/8 = 896, 3072/8 = 384.

    Returns a dict with ``hidden_shard``, ``ispp_shard`` and the per-expert
    shard byte counts ``w13_shard_bytes`` / ``w13s_shard_bytes`` /
    ``w2_shard_bytes`` / ``w2s_shard_bytes``.

    Raises:
        ValueError: on any constraint violation.
    """
    if tp <= 0:
        raise ValueError("tp must be positive")
    if hidden % tp or ispp % tp:
        raise ValueError("hidden and ispp must be divisible by tp")
    hidden_shard, ispp_shard = hidden // tp, ispp // tp
    if hidden_shard % 64 or ispp_shard % 64:
        raise ValueError("hidden/tp and ispp/tp must be multiples of 64 "
                         "(BLOCK_K); got %d, %d" % (hidden_shard, ispp_shard))
    if hidden_shard % group_size or ispp_shard % group_size:
        raise ValueError("hidden/tp and ispp/tp must be multiples of "
                         "group_size %d" % group_size)
    return {
        "tp": tp,
        "hidden": hidden,
        "ispp": ispp,
        "hidden_shard": hidden_shard,
        "ispp_shard": ispp_shard,
        "w13_shard_bytes": 2 * ispp * (hidden_shard // 2),
        "w13s_shard_bytes": 2 * ispp * (hidden_shard // group_size),
        "w2_shard_bytes": hidden * (ispp_shard // 2),
        "w2s_shard_bytes": hidden * (ispp_shard // group_size),
    }


def tp_shard_weights(w13, w13_scale, w2, w2_scale, tp: int, group_size: int = 32):
    """Split full weights into ``tp`` per-rank shards (layout-preserving).

    Each rank's shards are the full-weight layouts with the sharded dimension
    narrowed, so ``fused_moe_mxfp4`` / the stage functions consume them
    verbatim with ``hidden=hidden_shard`` / ``ispp=ispp_shard``:

    * ``w13``      ``[E, 2*ispp, hidden/2]``    → ``[E, 2*ispp, hidden_shard/2]``
    * ``w13_scale````[E, 2*ispp, hidden/32]``   → ``[E, 2*ispp, hidden_shard/32]``
    * ``w2``       ``[E, hidden, ispp/2]``      → ``[E, hidden, ispp_shard/2]``
    * ``w2_scale`` ``[E, hidden, ispp/32]``     → ``[E, hidden, ispp_shard/32]``

    Biases are not sharded (they are added post-all-reduce).  Returns a list
    of per-rank ``(w13, w13_scale, w2, w2_scale)`` uint8 arrays.
    """
    w13 = np.ascontiguousarray(w13, dtype=np.uint8)
    w13_scale = np.ascontiguousarray(w13_scale, dtype=np.uint8)
    w2 = np.ascontiguousarray(w2, dtype=np.uint8)
    w2_scale = np.ascontiguousarray(w2_scale, dtype=np.uint8)
    E, two_ispp, h2 = w13.shape
    hidden = 2 * h2
    ispp = two_ispp // 2
    plan = tp_plan(hidden, ispp, tp, group_size)

    out = []
    for r in range(tp):
        cs = slice(r * plan["hidden_shard"], (r + 1) * plan["hidden_shard"])
        ds = slice(r * plan["ispp_shard"], (r + 1) * plan["ispp_shard"])
        out.append((
            np.ascontiguousarray(w13[:, :, cs.start // 2:cs.stop // 2]),
            np.ascontiguousarray(w13_scale[:, :, cs.start // group_size:cs.stop // group_size]),
            np.ascontiguousarray(w2[:, :, ds.start // 2:ds.stop // 2]),
            np.ascontiguousarray(w2_scale[:, :, ds.start // group_size:ds.stop // group_size]),
        ))
    return out


def dist_moe_tp(A, w13, w13_scale, w2, w2_scale, topk_ids, topk_w,
                b13=None, b2=None, *, top_k: int, tp: int, group_size: int = 32,
                swiglu_limit: float = 0.0, activation: str = "swiglu",
                beta: float = 4.0, linear_beta: float = 25.0,
                block_size: int = 16):
    """Tensor-parallel fused MoE forward over ``tp`` simulated ranks.

    Each rank holds the full (replicated) input ``A [M, hidden]`` and its
    weight shard (from :func:`tp_shard_weights`).  The two linear stages are
    all-reduced across ranks *before* the nonlinear epilogues — the
    all-reduce points where a real deployment runs its own collective (this
    host reference uses a plain sum, the result of a ring/NCCL/RCCL
    all-reduce).

    Returns a list of ``tp`` per-rank outputs ``[M, hidden]`` fp32 — all
    identical after the all-reduces, and matching
    :func:`vkernels.kernels.fused_moe_mxfp4` (the single-rank oracle).

    Args:
        A: uint16 bf16 activations ``(M, hidden)`` (replicated on every rank).
        w13/w13_scale/w2/w2_scale: full (unsharded) weights, as consumed by
            :func:`vkernels.kernels.fused_moe_mxfp4`; sharded internally.
        topk_ids: int32 ``(M, top_k)`` token→expert routing (replicated).
        topk_w: float32 ``(M, top_k)`` routing weights.
        b13/b2: optional biases (full, shared across ranks).
        top_k/tp/group_size: as in the fused kernel; ``tp`` ranks.
        activation: ``"swiglu"`` or ``"situ"``.
    """
    A = np.ascontiguousarray(A, dtype=np.uint16)
    M, hidden = A.shape
    w13 = np.ascontiguousarray(w13, dtype=np.uint8)
    E, two_ispp, _ = w13.shape
    ispp = two_ispp // 2
    topk_ids = np.ascontiguousarray(topk_ids, dtype=np.int32)
    topk_w = np.ascontiguousarray(topk_w, dtype=np.float32).ravel()
    if topk_ids.ndim != 2 or topk_ids.shape[1] != top_k:
        raise ValueError("topk_ids must be [M, top_k]")
    plan = tp_plan(hidden, ispp, tp, group_size)

    sorted_ids, expert_ids, EM = _align(topk_ids, E, block_size)
    nflat = M * top_k
    # Padding sentinels (== nflat) fall outside topk_w; clamp before indexing
    # and let the where() mask them to 0.0.
    clamped = np.minimum(sorted_ids, nflat - 1)
    sorted_w = np.where(
        (sorted_ids >= 0) & (sorted_ids < nflat), topk_w[clamped], 0.0).astype(np.float32)

    b13_f = np.ascontiguousarray(b13, dtype=np.float32) if b13 is not None else None
    b2_f = np.ascontiguousarray(b2, dtype=np.float32) if b2 is not None else None
    act_key = activation.lower()
    if act_key not in ("swiglu", "situ"):
        raise ValueError(f"activation must be 'swiglu' or 'situ', got {activation!r}")

    shards = tp_shard_weights(w13, w13_scale, w2, w2_scale, tp, group_size)

    # Stage 0a: every rank's gate/up GEMM over its hidden K-slice.
    gate_up_partials = []
    for rank, (w13_r, w13s_r, _w2, _w2s) in enumerate(shards):
        gate_up = np.zeros(EM * 2 * ispp, dtype=np.float32)
        _moe_gateup(A, w13_r.ravel(), w13s_r.ravel(), sorted_ids, expert_ids,
                    gate_up, M, hidden, rank * plan["hidden_shard"],
                    plan["hidden_shard"], ispp, top_k, EM, group_size)
        gate_up_partials.append(gate_up)
    # Caller's TP collective #1: sum the linear gate/up partials.  In a real
    # deployment this is a ring/NCCL/RCCL all-reduce; the host reference
    # sums elementwise (each rank then holds the same full result).
    gate_up = sum(gate_up_partials)

    # Stage 0b: bias + activation (rank-independent; computed once).
    act = np.empty(EM * ispp, dtype=np.uint16)
    _moe_act_epilogue(gate_up, b13_f, act, sorted_ids, expert_ids,
                      M, top_k, EM, ispp, swiglu_limit, act_key,
                      float(beta), float(linear_beta))

    # Stage 1a: every rank's down GEMM over its ispp K-slice.
    down_partials = []
    for rank, (_w13, _w13s, w2_r, w2s_r) in enumerate(shards):
        partial = np.zeros(EM * hidden, dtype=np.float32)
        _moe_down(act, w2_r.ravel(), w2s_r.ravel(), sorted_ids, expert_ids,
                  partial, M, ispp, rank * plan["ispp_shard"],
                  plan["ispp_shard"], hidden, top_k, EM, group_size)
        down_partials.append(partial)
    # Caller's TP collective #2: sum the linear down partials.
    partial = sum(down_partials)

    # Stage 1b: routed combine — identical on every rank after the all-reduce.
    out = np.zeros(M * hidden, dtype=np.float32)
    _moe_combine(partial, b2_f, sorted_w, sorted_ids, expert_ids, out,
                 M, hidden, top_k, EM)
    return [out.reshape(M, hidden)] * tp


# ---------------------------------------------------------------------------
# EP
# ---------------------------------------------------------------------------


def ep_plan(num_experts: int, ep: int, rank: int) -> dict:
    """Expert range owned by one EP rank (contiguous blocks; tail experts go
    to the last ranks when ``num_experts % ep != 0``)."""
    if ep <= 0:
        raise ValueError("ep must be positive")
    if rank < 0 or rank >= ep:
        raise ValueError("rank out of range")
    if num_experts < ep:
        raise ValueError("num_experts must be at least ep")
    base, rem = divmod(num_experts, ep)
    begin = rank * base + (rank if rank < rem else rem)
    end = begin + base + (1 if rank < rem else 0)
    return {"ep": ep, "rank": rank, "expert_begin": begin, "expert_end": end,
            "num_local": end - begin}


def ep_dispatch(topk_ids: np.ndarray, plan: dict, num_experts: int,
                block_size: int = 16):
    """Build the local block-aligned sorted layout for one EP rank.

    Masks the ``[M, top_k]`` routing down to the locally-owned experts and
    block-aligns the survivors (exactly ``moe_align_block_size`` restricted
    to this rank).  ``sorted_ids`` carry flat ``token*top_k + sel`` indices
    (the token owner is recoverable for the output all-to-all) and
    ``expert_ids`` carry *local* expert ids (``0 .. num_local-1``) so a
    compact per-rank weight buffer is indexed without a global offset.

    Returns ``(sorted_ids, expert_ids, EM)``.
    """
    ids = np.ascontiguousarray(topk_ids, dtype=np.int32).ravel()
    masked = ids.copy()
    owned = (ids >= plan["expert_begin"]) & (ids < plan["expert_end"])
    masked[~owned] = -1
    sorted_ids, expert_ids, EM = _align(masked.reshape(topk_ids.shape),
                                        num_experts, block_size)
    local = expert_ids.copy()
    mask = local >= 0
    local[mask] -= plan["expert_begin"]
    return sorted_ids, local, EM


def dist_moe_ep(A, w13, w13_scale, w2, w2_scale, topk_ids, topk_w,
                b13=None, b2=None, *, top_k: int, ep: int, group_size: int = 32,
                swiglu_limit: float = 0.0, activation: str = "swiglu",
                beta: float = 4.0, linear_beta: float = 25.0,
                block_size: int = 16):
    """Expert-parallel fused MoE forward over ``ep`` simulated ranks.

    Each rank holds the full input ``A`` (replicated) and the weights of its
    expert range (pointers into the full buffers are offset by the plan).
    Tokens are dispatched to the rank owning their chosen experts, computed
    locally with the full hidden dimension (no all-reduce needed in EP), and
    the weighted partial outputs are sent back for the routed combine.

    Returns per-rank partial outputs ``[M, hidden]`` fp32; the true MoE
    output is their element-wise sum (the all-to-all-back + home-rank
    combine).  Validated against the single-rank oracle in tests.
    """
    if ep <= 0:
        raise ValueError("ep must be positive")
    A = np.ascontiguousarray(A, dtype=np.uint16)
    M, hidden = A.shape
    w13 = np.ascontiguousarray(w13, dtype=np.uint8)
    E, two_ispp, _ = w13.shape
    ispp = two_ispp // 2
    topk_ids = np.ascontiguousarray(topk_ids, dtype=np.int32)
    topk_w = np.ascontiguousarray(topk_w, dtype=np.float32).ravel()
    b13_f = np.ascontiguousarray(b13, dtype=np.float32) if b13 is not None else None
    b2_f = np.ascontiguousarray(b2, dtype=np.float32) if b2 is not None else None

    outs = []
    for rank in range(ep):
        plan = ep_plan(E, ep, rank)
        sorted_ids, expert_ids, EM = ep_dispatch(topk_ids, plan, E, block_size)
        nflat = M * top_k
        clamped = np.minimum(sorted_ids, nflat - 1)
        sorted_w = np.where(
            (sorted_ids >= 0) & (sorted_ids < nflat), topk_w[clamped], 0.0).astype(np.float32)

        # Local weights: this rank's experts (axis 0 = experts).
        b0, b1 = plan["expert_begin"], plan["expert_end"]
        w13_r = w13[b0:b1]
        w13s_r = w13_scale[b0:b1]
        w2_r = w2[b0:b1]
        w2s_r = w2_scale[b0:b1]
        b13_r = b13_f[b0:b1] if b13_f is not None else None
        b2_r = b2_f[b0:b1] if b2_f is not None else None

        out = np.zeros(M * hidden, dtype=np.float32)
        _moe_local_forward(
            A, w13_r.ravel(), w13s_r.ravel(), w2_r.ravel(), w2s_r.ravel(),
            sorted_ids, sorted_w, expert_ids, out, EM,
            M, hidden, ispp, top_k, group_size, swiglu_limit,
            activation.lower(), float(beta), float(linear_beta),
            b13_r, b2_r)
        outs.append(out.reshape(M, hidden))
    return outs


def _moe_local_forward(A, w13, w13_scale, w2, w2_scale, sorted_ids, sorted_w,
                       expert_ids, out, EM, M, hidden, ispp, top_k,
                       group_size, swiglu_limit, activation, beta,
                       linear_beta, b13=None, b2=None):
    """One rank's local fused MoE on its own experts with the full hidden
    dimension — the single-rank forward the distributed schemes build on.
    ``w13/w2`` are this rank's compact expert weights; ``sorted_ids`` /
    ``expert_ids`` (local ids) are its block-aligned layout."""
    gate_up = np.zeros(EM * 2 * ispp, dtype=np.float32)
    _moe_gateup(A, w13, w13_scale, sorted_ids, expert_ids, gate_up,
                M, hidden, 0, hidden, ispp, top_k, EM, group_size)
    act = np.empty(EM * ispp, dtype=np.uint16)
    _moe_act_epilogue(gate_up, b13, act, sorted_ids, expert_ids,
                      M, top_k, EM, ispp, swiglu_limit, activation,
                      beta, linear_beta)
    partial = np.zeros(EM * hidden, dtype=np.float32)
    _moe_down(act, w2, w2_scale, sorted_ids, expert_ids, partial,
              M, ispp, 0, ispp, hidden, top_k, EM, group_size)
    _moe_combine(partial, b2, sorted_w, sorted_ids, expert_ids, out,
                 M, hidden, top_k, EM)


# ---------------------------------------------------------------------------
# PP: stage boundary interface (graph-capturable transfer, issue #10)
# ---------------------------------------------------------------------------


def round_bf16(x):
    """Round an fp32 hidden state to the bf16 activation format the MoE
    consumes (RNE), returning a uint16 array."""
    return _f32_to_bf16(np.ascontiguousarray(x, dtype=np.float32))


def pp_boundary_send(state, queue: comm.BlockingQueue):
    """Stage-boundary transfer of an fp32 hidden state [M, hidden].

    Host reference for the graph-capturable transfer primitive of issue #10:
    on gfx942 a deployment replaces the queue with a stream-ordered device
    copy / peer transfer captured into the graph segment so a replay needs no
    host progress.  This wrapper fixes the interface (fp32 [M, hidden]) that
    primitive must satisfy.
    """
    queue.push(list(np.ascontiguousarray(state, dtype=np.float32).ravel()))


def pp_boundary_recv(queue: comm.BlockingQueue, M: int, hidden: int) -> np.ndarray:
    """Receive a stage-boundary hidden state as fp32 ``(M, hidden)``."""
    got = np.asarray(queue.pop(), dtype=np.float32)
    if got.size != M * hidden:
        raise ValueError("PP boundary state size mismatch")
    return got.reshape(M, hidden)
