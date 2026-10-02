"""Vendored SGLang Triton fused-MoE — fp8_w8a8, block_shape=[128,128] ONLY.

Provenance (Apache-2.0; see floe's design.md for the full trail):
  * ``fused_moe_kernel`` + ``invoke`` skeleton from
    ``sglang.kernels.ops.moe.fused_moe_triton_kernels`` (SGLang v0.5.20,
    /tmp/sgl-kernels/ops/moe/fused_moe_triton_kernels.py), which is itself the
    vLLM fused-MoE Triton kernel (vllm/model_executor/layers/fused_moe) with
    SGLang's blockwise-fp8 additions.
  * ``_moe_align_small_numel_kernel`` verbatim (minus the PDL/GDC hooks) from
    ``sglang.kernels.ops.moe.moe_align_small_numel`` — the single-CTA Triton
    align that covers ``numel = T*top_k <= 64`` (the entire decode bucket:
    T <= 8, top_k = 8), any expert count.
  * ``_per_token_group_quant_fp8_kernel`` from
    ``sglang.kernels.ops.quantization.fp8_kernel._per_token_group_quant_8bit``
    (row-major-scales variant; fp8 branch only).
  * config resolution semantics (nearest-M buckets, filename convention) from
    ``sglang.moe_runner.triton_utils.fused_moe_triton_config``.

Adopted from floe (``floe/engine/runner/kernels/sgl_moe.py``, the
sgl-fuse campaign; the tuned ``configs/`` sidecars moved with it) under
the #64/#65 re-export model. Adoption deltas besides the move: the
between-stages swiglu and the weighted combine reuse THIS package's ops
(``elementwise.swiglu_limit`` / ``moe_combine.moe_weighted_sum``) instead
of floe's ``vkl_ops`` shims, and the tile-config sidecar directory is
resolved here — ``VK_SGL_MOE_CONFIG_DIR`` overrides, else the packaged
``configs/`` beside this module (the ``VK_TILE_CONFIGS_DIR`` precedent;
floe's ``FLOE_SGL_MOE_CONFIG_DIR`` knob retired with the move).

Stripped from the donor (deliberately — this is the minimal decode set):
TMA descriptors (a_desc/b_desc + allocator + cache), GDC/PDL, LORA variants
(FUSE_ADD_TO_OUTPUT / MASK_OUTPUT / FUSE_SUM_ALL_REDUCE / LORA_PRESERVE_BASE /
add_mask), c_sorted, bias, FUSE_SWIGLU epilogue, int8/int8w8a16/int4/gptq/awq
paths, per-channel/tensorwise scales, filter_expert (floe TP-runs all 288
experts locally — no EP filtering), torch.compile combine variants, and the
moe_sum_reduce kernel (the weighted adaptation ships as
``moe_combine.moe_weighted_sum`` — reused).

Shape mapping (floe GLM-5.3-Flash fp8 TP4, per rank — see floe's design.md
§2):
  SGLang w1/w13 := gate_up_proj [E=288, 2*I_tp=1024, K=4096] fp8e4m3
                   gate_up_scale [288, 1024/128, 4096/128] = [288, 8, 32] fp32
  SGLang w2      := down_proj    [E=288, K=4096, I_tp=512]  fp8e4m3
                   down_scale    [288, 4096/128, 512/128]  = [288, 32, 4] fp32
  The kernel sees stage-1 as B[N=1024, K=4096] and stage-2 as B[N=4096,
  K=512] — SGLang's w2 convention is [E, hidden, intermediate] (row = output
  dim), which is exactly floe's down_proj layout. Gate rows are the FIRST
  half of w13's output dim, up the second (both floe's chunk(2) and SGLang's
  silu_and_mul assume this). No transposes or re-packs needed.

CUDA-graph capture contract (how SGLang keeps this capture-safe; preserved):
  * every buffer is sized from HOST-known static shapes — the alignment
    upper bound ``max_num_tokens_padded = numel + (E+1)*(BLOCK_M-1)`` (or
    ``numel*BLOCK_M`` when ``numel < E+1``, the decode case) comes from
    ``topk_ids.shape``, never from device data;
  * ``num_tokens_post_padded`` stays a device int32[1]; the kernel reads it
    IN-KERNEL (``tl.load``) and excess CTAs early-return — the launch grid
    uses the static upper bound. No ``.item()``, no host sync, anywhere;
  * the config lookup is host-side and lru_cached on static shape keys;
  * the Triton JIT must compile during eager warmup (floe's decode warmup
    precedes capture; a cold compile under capture would fail);
  * ``torch.empty`` intermediates inside capture ride the graph memory pool.

Stdlib+torch at import (triton imported lazily, moe_combine house style —
CPU-only hosts can import this module for the eligibility tests).
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

__all__ = [
    "SMALL_NUMEL_LIMIT",
    "sgl_fused_moe",
    "sgl_fused_moe_eligible",
    "try_get_moe_config",
]

logger = logging.getLogger(__name__)

# The decode alignment kernel's register budget limit (donor constant): its
# [NP, NP] pairwise tensors fit in registers at NP=64 but spill at NP=256.
# floe's decode bucket is T <= moe_decode_max_tokens = 8 with top_k = 8,
# so numel <= 64 — the whole bucket rides the single-launch Triton path.
SMALL_NUMEL_LIMIT = 64

_FP8_DTYPE = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# eligibility (cheap, no device sync — the call site gates on this and keeps
# the eager per-expert loop / expert_gemv path as the fallback, moe_combine
# house style)
# ---------------------------------------------------------------------------
def sgl_fused_moe_eligible(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_scale: Optional[torch.Tensor],
    down_proj: torch.Tensor,
    down_scale: Optional[torch.Tensor],
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> bool:
    """Contract for :func:`sgl_fused_moe`.

    CUDA bf16 activations [T, H]; fp8-e4m3fn expert stacks [E, 2I, H] /
    [E, H, I] (contiguous) with fp32 128x128-block scales [E, 2I/128, H/128]
    / [E, H/128, I/128]; int routing [T, K] with float [T, K] weights
    (unit inner stride); H, I multiples of 128. e4m3fn storage only — CDNA3
    keeps its fnuz stacks on the vkernels grouped path (that lane's
    e4m3fn_to_fnuz rewrite is orthogonal to this H100-targeted vendor).
    """
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and gate_up_proj.dim() == 3
        and down_proj.dim() == 3
        and gate_up_proj.dtype == _FP8_DTYPE
        and down_proj.dtype == _FP8_DTYPE
        and gate_up_proj.is_contiguous()
        and down_proj.is_contiguous()
        and gate_up_scale is not None
        and down_scale is not None
        and gate_up_scale.dtype == torch.float32
        and down_scale.dtype == torch.float32
        and gate_up_proj.shape[0] == down_proj.shape[0]
        and gate_up_proj.shape[2] == x.shape[1]
        and down_proj.shape[1] == x.shape[1]
        and down_proj.shape[2] * 2 == gate_up_proj.shape[1]
        and gate_up_proj.shape[1] % 128 == 0
        and gate_up_proj.shape[2] % 128 == 0
        and down_proj.shape[1] % 128 == 0
        and down_proj.shape[2] % 128 == 0
        and gate_up_scale.shape
        == (
            gate_up_proj.shape[0],
            gate_up_proj.shape[1] // 128,
            gate_up_proj.shape[2] // 128,
        )
        and down_scale.shape
        == (down_proj.shape[0], down_proj.shape[1] // 128, down_proj.shape[2] // 128)
        and top_k_index.shape == top_k_weights.shape
        and top_k_index.shape[0] == x.shape[0]
        and not top_k_index.dtype.is_floating_point
        and top_k_weights.is_floating_point()
        and (top_k_weights.dim() == 1 or top_k_weights.stride(-1) == 1)
        and top_k_index.shape[1] > 0
    )


# ---------------------------------------------------------------------------
# JIT-scoped kernels (import triton lazily — moe_combine/vkernels pattern)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _per_token_group_quant_fp8_kernel(
        y_ptr,
        y_q_ptr,
        y_s_ptr,
        # Stride between consecutive GROUPS (the flat group index steps the
        # row pointer by group_size elements for a contiguous input).
        y_stride,
        # Columns of one group.
        N,
        # Avoid dividing zero.
        eps,
        fp8_min,
        fp8_max,
        BLOCK: tl.constexpr,
    ):
        """Per-token-group fp8 quantization (donor: _per_token_group_quant_8bit).

        One program per (row, group): scale = max(amax, eps) / 448 stored to
        the row-major fp32 scale buffer, payload clamped to the e4m3fn range.
        """
        g_id = tl.program_id(0)
        y_ptr += g_id * y_stride
        y_q_ptr += g_id * y_stride
        y_s_ptr += g_id

        cols = tl.arange(0, BLOCK)  # N <= BLOCK
        mask = cols < N

        y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        # Quant
        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        y_s = _absmax / fp8_max
        y_s_inv = 1.0 / y_s
        y_q = tl.clamp(y * y_s_inv, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + cols, y_q, mask=mask)
        tl.store(y_s_ptr, y_s)

    @triton.jit
    def _moe_align_small_numel_kernel(
        topk_ids_ptr,  # [numel] int, flattened (token, slot) expert ids
        sorted_token_ids_ptr,  # [max_num_tokens_padded] int32
        expert_ids_ptr,  # [max_num_m_blocks] int32
        num_tokens_post_pad_ptr,  # [1] int32
        num_experts,  # E + 1 (the "+1 offset" convention's bucket count)
        block_size,
        numel,
        NP: tl.constexpr,  # power-of-2 >= numel
        NB: tl.constexpr,  # power-of-2 >= max blocks used
        route_weights_ptr,
        SKIP_ZERO: tl.constexpr,
    ):
        """Single-CTA moe_align for tiny batches with MANY experts (donor:
        _moe_align_small_numel_kernel, PDL hooks removed).

        Everything works on the PAIR axis ([NP, NP] pairwise comparisons plus
        a rank-0 representative per bucket). Reference semantics reproduced:
        - "+1 offset" convention: expert -1 (EP-filtered) maps to bucket 0 and
          its blocks get expert_ids = -1 (floe never routes -1, so every
          written id is the real expert id);
        - every bucket is padded to a block_size multiple, offsets in bucket
          order;
        - pad slots inside [0, num_tokens_post_pad) hold `numel`.
        Intended deviations, both invisible to fused_moe_kernel: intra-bucket
        order is stable in pair index (the CUDA reference's atomicAdd order is
        scheduling-dependent); sorted_token_ids beyond num_tokens_post_pad is
        left unwritten (consumers only read below the published total).
        """
        offs_p = tl.arange(0, NP)
        mask_p = offs_p < numel
        if SKIP_ZERO:
            mask_p = mask_p & (tl.load(route_weights_ptr + offs_p, mask=mask_p, other=0.0) != 0.0)
        ids = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=-2)
        # Padded lanes get an out-of-range bucket and are masked out everywhere.
        bucket = tl.where(mask_p, (ids + 1).to(tl.int32), num_experts)

        # Pairwise stats: stable rank within the bucket and bucket population.
        same = (bucket[None, :] == bucket[:, None]) & mask_p[None, :] & mask_p[:, None]
        earlier = offs_p[None, :] < offs_p[:, None]
        rank = tl.sum((same & earlier).to(tl.int32), axis=1)  # [NP]
        cnt = tl.sum(same.to(tl.int32), axis=1)  # [NP], own-bucket population
        padded_cnt = ((cnt + block_size - 1) // block_size) * block_size
        is_rep = (rank == 0) & mask_p  # one representative pair per bucket

        # Bucket-ordered exclusive offsets: sum the padded counts of every
        # representative with a strictly smaller bucket id.
        smaller_rep = (bucket[None, :] < bucket[:, None]) & is_rep[None, :]
        excl = tl.sum(smaller_rep.to(tl.int32) * padded_cnt[None, :], axis=1)  # [NP]

        total = tl.sum(tl.where(is_rep, padded_cnt, 0), axis=0)
        tl.store(num_tokens_post_pad_ptr, total.to(tl.int32))

        # expert_ids per used block: representative r owns blocks
        # [excl[r], excl[r] + padded_cnt[r]); the written id is bucket - 1
        # (bucket 0 = filtered -> -1).
        offs_b = tl.arange(0, NB)
        block_start = offs_b * block_size
        in_range = (
            (block_start[:, None] >= excl[None, :])
            & (block_start[:, None] < (excl + padded_cnt)[None, :])
            & is_rep[None, :]
        )
        eid = tl.sum(in_range.to(tl.int32) * (bucket[None, :] - 1), axis=1)
        tl.store(expert_ids_ptr + offs_b, eid.to(tl.int32), mask=block_start < total)

        # Fill the used region's pad slots with `numel`, then scatter the real
        # pair indices over them. The barrier is required: fill and scatter
        # run on different warps of this CTA, and a scatter store must not be
        # overtaken by a later-warp fill store to the same address.
        n_fill = (total + NP - 1) // NP
        for it in range(n_fill):
            f_offs = it * NP + offs_p
            tl.store(
                sorted_token_ids_ptr + f_offs,
                tl.full([NP], 0, tl.int32) + numel,
                mask=f_offs < total,
            )
        tl.debug_barrier()
        pos = excl + rank
        tl.store(sorted_token_ids_ptr + pos, offs_p.to(tl.int32), mask=mask_p)

    @triton.jit
    def fused_moe_kernel(
        # Pointers to matrices (plain pointer interface — no TMA descriptors).
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions.
        N,
        K,
        EM,
        num_valid_tokens,
        # Strides (elements). B is indexed [expert][n, k] via (stride_be,
        # stride_bn, stride_bk) — SGLang passes (B.stride(0), B.stride(1),
        # B.stride(2)) as (stride_be, stride_bn, stride_bk) so a row-major
        # [E, N, K] stack reads n-major tiles.
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Block size for block-wise quantization (the ONLY quant path here).
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        # Meta-parameters.
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        swap_ab: tl.constexpr,
        even_Ks: tl.constexpr,
    ):
        """Fused MoE GEMM (donor: fused_moe_kernel, fp8_w8a8 + block scales
        branch only).

        - A: activations (*, K). Stage 1: per-token fp8 [T, K] (rows gathered
          by ``offs_token // top_k``); stage 2: per-slot fp8 [T*top_k, K]
          (top_k == 1).
        - B: stacked expert weights [E, N, K] fp8.
        - C: [M, topk, N]-shaped buffer addressed by the flat slot id.
        - sorted_token_ids / expert_ids / num_tokens_post_padded: the
          block-aligned routing from moe_align (all consumed on-device; excess
          CTAs early-return so the launch grid can use the static upper bound
          EM — this is what makes the launch CUDA-graph capturable).
        Blockwise scales are applied per k-iteration: BLOCK_K <= group_k means
        one scalar group per dot; BLOCK_SIZE_N > group_n loads a per-column
        scale vector instead.
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        offs_token = offs_token.to(tl.int64)
        token_mask = offs_token < num_valid_tokens

        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

        # block-wise fp8 scales (the only quant path vendored)
        a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
        if BLOCK_SIZE_N > group_n:
            offs_bsn = offs_bn // group_n
        else:
            offs_bsn = pid_n * BLOCK_SIZE_N // group_n
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
        )

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix, accumulating into
        # fp32 and applying the per-(k-group, n-block) scales per iteration.
        if swap_ab:
            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_SIZE_K):
            if even_Ks:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None],
                    other=0.0,
                )
            else:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                    other=0.0,
                )
            if even_Ks:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

            offs_ks = k_start // group_k
            a_scale = tl.load(
                a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
            )
            b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
            if swap_ab:
                a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                a_scale, b_scale = b_scale, a_scale
            if BLOCK_SIZE_N > group_n:
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if swap_ab:
            accumulator = tl.trans(accumulator, (1, 0))

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
            accumulator *= moe_weight[:, None]

        accumulator = accumulator.to(compute_type)
        # -----------------------------------------------------------
        # Write back the block of the output.
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    return _Namespace(
        quant=_per_token_group_quant_fp8_kernel,
        align=_moe_align_small_numel_kernel,
        moe=fused_moe_kernel,
        triton=triton,
        tl=tl,
    )


class _Namespace:
    """Tiny attribute bag (avoids re-importing triton pieces per call)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------------------------------------------------------------------------
# per-token-group fp8 quant (row-major scales)
# ---------------------------------------------------------------------------
def per_token_group_quant_fp8(
    x: torch.Tensor, group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp16 [T, K] -> (fp8-e4m3 [T, K], fp32 [T, K/group] scales).

    Donor: ``_per_token_group_quant_8bit_raw`` (fp8 + row-major branch).
    K must be a multiple of group_size; x must be contiguous (the kernel
    steps by flat group index).
    """
    import triton

    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dim {x.shape[-1]} not divisible by group_size {group_size}"
        )
    if not x.is_contiguous():
        raise ValueError("per_token_group_quant_fp8 needs contiguous x")
    x_q = torch.empty(x.shape, device=x.device, dtype=_FP8_DTYPE)
    x_s = torch.empty(
        (*x.shape[:-1], x.shape[-1] // group_size),
        device=x.device,
        dtype=torch.float32,
    )
    num_groups = x.numel() // group_size
    if num_groups == 0:
        return x_q, x_s
    info = torch.finfo(_FP8_DTYPE)
    block = triton.next_power_of_2(group_size)
    _kernels().quant[(num_groups,)](
        x,
        x_q,
        x_s,
        group_size,
        group_size,
        1e-10,
        info.min,
        info.max,
        BLOCK=block,
        num_warps=min(max(block // 256, 1), 8),
        num_stages=1,
    )
    return x_q, x_s


# ---------------------------------------------------------------------------
# moe_align_block_size (graph-safe: static shapes, device-only reads)
# ---------------------------------------------------------------------------
def _moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int,
    *, route_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align routing to block_size multiples (donor: moe_align_block_size).

    Returns (sorted_token_ids [max_pad], expert_ids [max_blocks],
    num_tokens_post_padded [1] int32 — never read on the host). Decode
    (numel <= SMALL_NUMEL_LIMIT) rides the donor's single-CTA Triton kernel;
    larger numels take a static-shape torch fallback (scatter_add histogram +
    padded cumsum + stable argsort scatter — the vkernels ``_route_counts`` /
    ``_tile_map_static`` pattern, no bincount (ROCm), no unique_consecutive,
    no host syncs, so it stays CUDA-graph capturable too).
    """
    import triton

    numel = topk_ids.numel()
    if route_weights is not None:
        if (numel > SMALL_NUMEL_LIMIT or route_weights.shape != topk_ids.shape
                or route_weights.device != topk_ids.device
                or not route_weights.is_floating_point()
                or not route_weights.is_contiguous()):
            from ._dispatch import OpNotEligible
            raise OpNotEligible("zero-route alignment requires <=64 pairs and contiguous same-device floating weights")
    if numel < num_experts + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + (num_experts + 1) * (block_size - 1)
    device = topk_ids.device
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)

    flat = topk_ids.reshape(-1).to(torch.int32)
    if numel <= SMALL_NUMEL_LIMIT:
        _kernels().align[(1,)](
            flat,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            num_experts + 1,  # the donor's "+1 offset" bucket count
            block_size,
            numel,
            NP=triton.next_power_of_2(max(numel, 2)),
            NB=triton.next_power_of_2(max(max_num_m_blocks, 2)),
            route_weights_ptr=flat if route_weights is None else route_weights,
            SKIP_ZERO=route_weights is not None,
            num_warps=4,
        )
        return sorted_ids, expert_ids, num_tokens_post_pad

    # Static-shape torch fallback (numel > SMALL_NUMEL_LIMIT).
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    counts.scatter_add_(
        0, flat.long(), torch.ones_like(flat, dtype=torch.int32)
    )
    padded_counts = (
        (counts + block_size - 1) // block_size
    ) * block_size
    ends = padded_counts.cumsum(0)  # inclusive padded end per expert
    starts = ends - padded_counts  # exclusive padded start per expert
    num_tokens_post_pad.copy_(ends[-1].reshape(1))
    sorted_ids.fill_(numel)  # pad fill (in-place, no host read)
    order = torch.argsort(flat.long(), stable=True)  # ascending expert order
    e_sorted = flat.long()[order]
    seg = counts.cumsum(0) - counts  # exclusive RAW-count cumsum
    rank = torch.arange(numel, device=device) - seg[e_sorted]
    sorted_ids[(starts[e_sorted] + rank).long()] = order.to(torch.int32)
    # Block b (start s = b*block_size) belongs to the expert whose padded
    # window covers s; blocks past num_tokens_post_pad are never read.
    block_starts = (
        torch.arange(max_num_m_blocks, device=device, dtype=torch.int64)
        * block_size
    )
    owner = torch.searchsorted(ends, block_starts, right=True).clamp(
        max=num_experts - 1
    )
    expert_ids.copy_(owner.to(torch.int32))
    return sorted_ids, expert_ids, num_tokens_post_pad


# ---------------------------------------------------------------------------
# config resolution (heuristic default + optional JSON sidecar)
# ---------------------------------------------------------------------------
def _config_dir() -> Path:
    """Where tuned JSON sidecars live (``VK_SGL_MOE_CONFIG_DIR`` wins).

    This is a data-file location, NOT a dispatch knob — it cannot change
    which code runs, only the tile shape of the same kernel (the
    ``VK_TILE_CONFIGS_DIR`` precedent for tuning artifacts); the default
    is the packaged ``configs/`` directory beside this module.
    """
    env_dir = os.environ.get("VK_SGL_MOE_CONFIG_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parent / "configs"


def _config_file_name(
    E: int, N: int, dtype: str = "fp8_w8a8", block_shape=(128, 128)
) -> str:
    """Donor filename convention (fused_moe_triton_config.get_config_file_name).

    N is SGLang's config key: ``E, _, N = w2.shape`` — i.e. the INTERMEDIATE
    size per rank (512 for GLM-5.3-Flash TP4), NOT the GEMM N of either
    stage. One file therefore serves both stages (the donor's optional
    ``_down`` selector is not vendored; both stages share one config and one
    alignment — see floe's design.md §4).
    """
    device = torch.cuda.get_device_name(0).replace(" ", "_") if torch.cuda.is_available() else "unknown"
    return f"E={E},N={N},device_name={device},dtype={dtype},block_shape=[{block_shape[0]}, {block_shape[1]}].json"


@functools.lru_cache(maxsize=8)
def _load_configs(E: int, N: int) -> Optional[dict]:
    path = _config_dir() / _config_file_name(E, N)
    if not path.is_file():
        return None
    logger.info("sgl_moe: using tuned config %s", path)
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def _get_default_config(M: int) -> dict:
    """Heuristic default for fp8_w8a8 + block_shape=[128,128].

    The donor's blockwise default is BLOCK_M=64 — sized for prefill and
    exactly wrong at decode (per-expert assignment counts are <= top_k=8 at
    T<=8). The decode default below is seeded from the donor's own tuned
    H100 fp8-blockwise profiles at small M (e.g. E=257,N=128 H100:
    BLOCK_M=16/BLOCK_N=128/BLOCK_K=128, warps=4, stages=3-4 — BLOCK_M<64
    with BLOCK_N>=64 also engages swap_ab on SM90). Replace it with the
    one-time tune (configs/README.md).
    """
    if M <= 64:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 3,
        }
    # donor get_default_config, fp8_w8a8 + block_shape branch
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3,
    }


def try_get_moe_config(E: int, N: int, M: int) -> dict:
    """Config for (E, N=intermediate, M=num tokens): nearest tuned bucket,
    else the heuristic default (donor try_get_optimal_moe_config semantics,
    M-keyed by T; cached host-side — static per capture bucket)."""
    configs = _load_configs(E, N)
    if configs:
        return dict(configs[min(configs.keys(), key=lambda x: abs(x - M))])
    return _get_default_config(M)


# swap_ab benefits SM90 GPUs (H100/H200) for certain block shapes (donor
# note + should_enable_swap_ab; batch-invariant-mode hook not vendored).
@functools.lru_cache(maxsize=8)
def _should_enable_swap_ab(BLOCK_SIZE_M: int, BLOCK_SIZE_N: int) -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9 and BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64


# ---------------------------------------------------------------------------
# kernel invocation (donor: invoke_fused_moe_kernel, fp8-blockwise branch)
# ---------------------------------------------------------------------------
def _invoke_fused_moe_kernel(
    A: torch.Tensor,  # fp8 activations (pre-quantized by the caller)
    B: torch.Tensor,  # fp8 expert stack [E, N, K]
    C: torch.Tensor,  # bf16 output buffer
    A_scale: torch.Tensor,  # fp32 [rows, K/128]
    B_scale: torch.Tensor,  # fp32 [E, N/128, K/128]
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
) -> None:
    import triton.language as tl

    assert topk_weights.stride(-1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert A.dtype == _FP8_DTYPE and A_scale.ndim == 2 and A_scale.dtype == torch.float32
    triton = _kernels().triton

    grid = lambda META: (  # noqa: E731 - donor form
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    N, K = B.shape[1], B.shape[2]
    even_ks = K % config["BLOCK_SIZE_K"] == 0
    swap_ab = _should_enable_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])
    _kernels().moe[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(-2),
        C.stride(-1),
        A_scale.stride(0),
        A_scale.stride(1),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        group_n=128,
        group_k=128,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16,
        swap_ab=swap_ab,
        even_Ks=even_ks,
        **config,
    )


# ---------------------------------------------------------------------------
# the wrapper
# ---------------------------------------------------------------------------
def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """GLM swiglu (clamp(gate, max=limit), clamp(up, +/-limit), silu, mul).

    Reuses the package's Triton op (:func:`~vkernels.torch_ops.elementwise.swiglu_limit`
    — bit-identical to the eager chain, already shipped behind
    ``fused_swiglu`` at 84 sites/step); falls back to the eager chain on a
    contract miss (the non-contiguous gate/up views below — e.g. the
    half-splits of the stage-1 cache — miss the op's contiguity floor) —
    which is also the numerical oracle. NOT vendored as a duplicate Triton
    silu_and_mul_clamp: one launch either way, and a second implementation
    would fork the rounding contract for zero launch savings.
    """
    try:
        from .elementwise import swiglu_limit

        return swiglu_limit(gate, up, limit)
    except (ValueError, TypeError):  # OpNotEligible subclasses both
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return torch.nn.functional.silu(gate) * up


def sgl_fused_moe(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_proj: torch.Tensor,
    down_scale: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    swiglu_limit: float = math.inf,
    skip_zero_weights: bool = False,
) -> torch.Tensor:
    """Routed-expert MLP in 5 launches: quant -> stage1 -> swiglu -> quant ->
    stage2, then the weighted top-k combine (the package's
    ``moe_combine.moe_weighted_sum``; eager expression as fallback).

    Inputs (per rank, fp8-resident serving mode): x bf16 [T, H];
    gate_up_proj fp8-e4m3fn [E, 2I, H] + gate_up_scale fp32 [E, 2I/128, H/128];
    down_proj [E, H, I] + down_scale [E, H/128, I/128]; top_k_index int
    [T, K] (values in [0, E)); top_k_weights float [T, K] (routed scaling
    already folded in by floe's router — no extra factor here). Returns
    bf16 [T, H]. Raises :class:`~vkernels.torch_ops._dispatch.OpNotEligible`
    when :func:`sgl_fused_moe_eligible` fails — callers gate on the
    eligibility check and keep their eager path.

    Routing weights are applied at the COMBINE (stage 2 runs with
    MUL_ROUTED_WEIGHT=False), matching the landed decode path's rounding
    class: bf16 GEMM store, fp32 multiply-accumulate in the reduce, one
    round at the final store (SGLang instead multiplies in the stage-2
    epilogue and runs an unweighted sum — same class, one fewer load; the
    MUL_ROUTED_WEIGHT constexpr stays available for that A/B).
    """
    if not sgl_fused_moe_eligible(
        x, gate_up_proj, gate_up_scale, down_proj, down_scale,
        top_k_index, top_k_weights,
    ):
        from ._dispatch import OpNotEligible

        raise OpNotEligible(
            "sgl_fused_moe contract: CUDA bf16 x [T,H] with fp8-e4m3fn "
            "[E,2I,H]/[E,H,I] expert stacks, fp32 128x128-block scales, int "
            "routing and float weights (see sgl_fused_moe_eligible)"
        )

    T, H = x.shape
    E, N2, _ = gate_up_proj.shape
    I = N2 // 2
    top_k = top_k_index.shape[1]
    numel = T * top_k

    config = try_get_moe_config(E, I, T)
    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
        top_k_index, config["BLOCK_SIZE_M"], E,
        route_weights=top_k_weights if skip_zero_weights else None,
    )

    # stage 1: quant once per token (shared across its K experts — the
    # kernel gathers rows via offs_token // top_k), then the gate/up GEMM.
    xq, xs = per_token_group_quant_fp8(x, 128)
    # Skipped slots are not written by either GEMM. Zero them on EVERY
    # invocation/replay, not merely warmup: routes can change in a graph.
    allocate = torch.zeros if skip_zero_weights else torch.empty
    cache1 = allocate((numel, N2), device=x.device, dtype=torch.bfloat16)
    _invoke_fused_moe_kernel(
        xq, gate_up_proj, cache1, xs, gate_up_scale,
        top_k_weights, top_k_index, sorted_token_ids, expert_ids,
        num_tokens_post_padded, mul_routed_weight=False, top_k=top_k,
        config=config,
    )

    # swiglu over the REAL slot rows [0, numel) (pad rows in the aligned
    # buffer are never real slots; every real slot was written by stage 1).
    gate, up = cache1[:, :I], cache1[:, I:]
    act = _swiglu(gate, up, swiglu_limit)  # [numel, I] bf16

    # stage 2: per-slot activations (top_k == 1 — A rows are the slots).
    aq, asc = per_token_group_quant_fp8(act, 128)
    cache3 = allocate((T, top_k, H), device=x.device, dtype=torch.bfloat16)
    _invoke_fused_moe_kernel(
        aq, down_proj, cache3, asc, down_scale,
        top_k_weights, top_k_index, sorted_token_ids, expert_ids,
        num_tokens_post_padded, mul_routed_weight=False, top_k=1,
        config=config,
    )

    # weighted combine: the package's vendored reduce (fp32 MAC, one
    # rounding at the store — the moe_combine contract).
    try:
        from .moe_combine import moe_weighted_sum

        return moe_weighted_sum(cache3, top_k_weights.float())
    except (ValueError, TypeError):  # OpNotEligible subclasses both
        return (cache3 * top_k_weights.to(cache3.dtype).unsqueeze(-1)).sum(dim=1)
