"""Five-launch decode MoE with the vendored SGLang GEMMs unchanged.

Fuse routing alignment with input quantization, and SwiGLU with activation
quantization and down-buffer initialization. Skipped routes never read stale
gate outputs. The final torch reduction retains the incumbent rounding order.
This module owns only invocation-local scratch, not weights or KV state.
"""

from functools import lru_cache

import torch

from ._dispatch import OpNotEligible


def _gemm_configs(config, *, tuned, capability, device_name, tokens,
                  experts, hidden, intermediate, top_k, skip):
    """Shape/device qualification, with independent copies of donor configs.

    Only GH200 EP4 checkpoint geometry at B1/B4 has measured coverage. B2/B8
    regress in parts of the screen and deliberately keep the donor config.
    Numerical block order (BM16/K128) and SM90 swap direction stay fixed.
    """
    gate, down = dict(config), dict(config)
    if (tuned and capability == (9, 0) and device_name.startswith("NVIDIA GH200")
            and tokens in (1, 4) and experts == 72 and hidden == 4096
            and intermediate == 2048 and top_k == 8 and skip
            and config == {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128,
                           "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16,
                           "num_warps": 4, "num_stages": 3}):
        gate.update(BLOCK_SIZE_N=64, GROUP_SIZE_M=4)
        down.update(BLOCK_SIZE_N=64, num_stages=4)
    return gate, down


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    from .sgl_moe import _kernels as donor_kernels

    align = donor_kernels().align

    @triton.jit
    def route_quant(X, Q, S, IDS, ROUTES, SORTED, EXPERTS, TOTAL,
                    H: tl.constexpr, E: tl.constexpr, SLOTS: tl.constexpr,
                    BM: tl.constexpr, NP: tl.constexpr, NB: tl.constexpr,
                    SKIP: tl.constexpr):
        pid = tl.program_id(0)
        if pid == 0:
            align(IDS, SORTED, EXPERTS, TOTAL, E + 1, BM, SLOTS,
                  NP, NB, ROUTES, SKIP)
        else:
            group = pid - 1
            col = tl.arange(0, 128)
            x = tl.load(X + group * 128 + col).to(tl.float32)
            scale = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-10) / 448.0
            q = tl.clamp(x * (1.0 / scale), -448.0, 448.0)
            tl.store(Q + group * 128 + col, q)
            tl.store(S + group, scale)

    @triton.jit
    def activation_quant(GU, ROUTES, Q, S, DOWN, LIMIT,
                         I: tl.constexpr, H: tl.constexpr, SKIP: tl.constexpr):
        row = tl.program_id(0)
        group = tl.program_id(1)
        col = group * 128 + tl.arange(0, 128)
        if group * 128 < I:
            live = tl.full((), True, tl.int1)
            if SKIP:
                live = tl.load(ROUTES + row) != 0.0
            g = tl.load(GU + row * (2 * I) + col, live, 0).to(tl.float32)
            u = tl.load(GU + row * (2 * I) + I + col, live, 0).to(tl.float32)
            # Match elementwise._swiglu_limit, including all BF16 stores.
            g = tl.minimum(g, LIMIT, propagate_nan=tl.PropagateNan.ALL).to(tl.bfloat16).to(tl.float32)
            u = tl.minimum(tl.maximum(u, -LIMIT, propagate_nan=tl.PropagateNan.ALL),
                           LIMIT, propagate_nan=tl.PropagateNan.ALL).to(tl.bfloat16).to(tl.float32)
            silu = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
            act = (silu * u).to(tl.bfloat16).to(tl.float32)
            scale = tl.maximum(tl.max(tl.abs(act), 0), 1.0e-10) / 448.0
            q = tl.clamp(act * (1.0 / scale), -448.0, 448.0)
            tl.store(Q + row * I + col, q)
            tl.store(S + row * (I // 128) + group, scale)
        if SKIP and group * 128 < H:
            # A separate kernel boundary orders this before the down GEMM.
            tl.store(DOWN + row * H + col, 0.0)

    return triton, route_quant, activation_quant


def sgl_fused_moe_decode_glue(
    x, gate_up_proj, gate_up_scale, down_proj, down_scale, top_k_index,
    top_k_weights, *, swiglu_limit=float("inf"), skip_zero_weights=False,
    tuned_gemm=False,
):
    """Same W8A8/weighted-down rounding as sgl_fused_moe, in five kernels.

    Inference-only CUDA E4M3FN; 1..8 tokens and at most 64 route slots.
    IDs must address valid local experts, including zero-weight slots.
    Warm eagerly before graph capture; scratch uses the graph memory pool.
    """
    from .sgl_moe import (
        _invoke_fused_moe_kernel, sgl_fused_moe_eligible, try_get_moe_config,
    )

    values = (x, gate_up_proj, gate_up_scale, down_proj, down_scale,
              top_k_index, top_k_weights)
    if not (
        not torch.version.hip
        and all(isinstance(v, torch.Tensor) for v in values)
        and x.ndim == 2
        and top_k_index.ndim == top_k_weights.ndim == 2
        and top_k_index.dtype in (torch.int32, torch.int64)
        and top_k_weights.dtype == torch.float32
        and all(v.device == x.device and v.is_contiguous() for v in values)
        and not any(v.requires_grad for v in values)
        and 0 < x.shape[0] <= 8 and 0 < top_k_index.numel() <= 64
        and sgl_fused_moe_eligible(*values)
    ):
        raise OpNotEligible("decode glue requires bounded contiguous CUDA W8A8 inference inputs")
    t, h = x.shape
    e, twice_i, _ = gate_up_proj.shape
    i = twice_i // 2
    k = top_k_index.shape[1]
    slots = t * k
    with torch.cuda.device(x.device):
        triton, route_quant, activation_quant = _kernels()
        config = try_get_moe_config(e, i, t)
        gate_config, down_config = _gemm_configs(
            config, tuned=tuned_gemm, capability=torch.cuda.get_device_capability(x.device),
            device_name=torch.cuda.get_device_name(x.device), tokens=t, experts=e,
            hidden=h, intermediate=i, top_k=k, skip=skip_zero_weights)
        bm = config["BLOCK_SIZE_M"]
        max_pad = slots * bm if slots < e + 1 else slots + (e + 1) * (bm - 1)
        sorted_ids = torch.empty(max_pad, device=x.device, dtype=torch.int32)
        experts = torch.empty(triton.cdiv(max_pad, bm), device=x.device, dtype=torch.int32)
        total = torch.empty(1, device=x.device, dtype=torch.int32)
        xq = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        xs = torch.empty((t, h // 128), device=x.device, dtype=torch.float32)
        route_quant[(1 + t * h // 128,)](
            x, xq, xs, top_k_index, top_k_weights, sorted_ids, experts, total,
            h, e, slots, bm, triton.next_power_of_2(max(slots, 2)),
            triton.next_power_of_2(max(experts.numel(), 2)), skip_zero_weights,
            num_warps=4, num_stages=1)
        gate = torch.empty((slots, twice_i), device=x.device, dtype=torch.bfloat16)
        _invoke_fused_moe_kernel(
            xq, gate_up_proj, gate, xs, gate_up_scale, top_k_weights, top_k_index,
            sorted_ids, experts, total, False, k, gate_config)
        aq = torch.empty((slots, i), device=x.device, dtype=torch.float8_e4m3fn)
        asc = torch.empty((slots, i // 128), device=x.device, dtype=torch.float32)
        down = torch.empty((t, k, h), device=x.device, dtype=torch.bfloat16)
        activation_quant[(slots, max(i, h) // 128)](
            gate, top_k_weights, aq, asc, down, float(swiglu_limit), i, h,
            skip_zero_weights, num_warps=1, num_stages=1)
        _invoke_fused_moe_kernel(
            aq, down_proj, down, asc, down_scale, top_k_weights, top_k_index,
            sorted_ids, experts, total, True, 1, down_config)
        return down.sum(dim=1)
