# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""mHC prenorm split-K GEMM: the producer for ``mhc.gluon_mhc_pre_reduce_apply``.

Vendored (verbatim kernel body) from lightseekorg/tokenspeed
``tokenspeed-kernel`` ``main`` (MIT, fetched 2026-11):
``ops/residual/triton.py`` — ``_mhc_prenorm_gemm_triton_kernel``,
``_mhc_prenorm_gemm_launch_config``, ``_mhc_prenorm_gemm_triton``.

Adapted (import point only): their ``tokenspeed_kernel._triton`` shim became
the local one (:mod:`vkernels.torch_ops.tokenspeed_amd._gluon`, which binds
stock triton). Unlike the rest of the vendored package this kernel is plain
Triton (no gluon), so it runs on any Triton the repo already supports.

Contract: ``x`` [T, K] (the flattened mHC residual streams, K = hc*D),
``fn`` [N, K] fp32 (the ``[hc | hc | hc*hc]`` projection rows, N = 24 for
hc=4) — the same ``fn`` layout ``aiter_ops.aiter_mhc_pre`` consumes. The
kernel writes per-split partials ``out_mul [n_splits, T, N]`` and
``out_sqrsum [n_splits, T]`` for the vendored reduce/apply kernel.
"""

from __future__ import annotations

import torch

from vkernels.torch_ops.tokenspeed_amd._gluon import tl, triton

__all__ = ["mhc_prenorm_gemm", "gluon_mhc_pre"]


@triton.jit
def _mhc_prenorm_gemm_triton_kernel(
    x,
    fn,
    out_mul,
    out_sqrsum,
    num_tokens,
    K: tl.constexpr,
    N: tl.constexpr,
    # Derived from the per-batch split count; runtime so every batch shape
    # shares one binary.
    SPLIT_K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    split_id = tl.program_id(0)
    token_block = tl.program_id(1)
    n_block = tl.program_id(2)
    offs_m = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    split_start = split_id * SPLIT_K
    dot_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    square_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, SPLIT_K, BLOCK_K):
        offs_k = split_start + k_start + tl.arange(0, BLOCK_K)
        x_values = tl.load(
            x + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < num_tokens) & (offs_k[None, :] < K),
            other=0.0,
        )
        fn_values = tl.load(
            fn + offs_k[:, None] + offs_n[None, :] * K,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        dot_acc = tl.dot(
            x_values.to(tl.float32),
            fn_values,
            dot_acc,
            input_precision="ieee",
        )
        x_fp32 = x_values.to(tl.float32)
        square_acc += tl.sum(x_fp32 * x_fp32, axis=1)

    tl.store(
        out_mul + split_id * num_tokens * N + offs_m[:, None] * N + offs_n[None, :],
        dot_acc,
        mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < N),
    )
    tl.store(
        out_sqrsum + split_id * num_tokens + offs_m,
        square_acc,
        mask=(offs_m < num_tokens) & (n_block == 0),
    )


def _mhc_prenorm_gemm_launch_config(
    num_tokens: int, k: int, n: int, n_splits: int
) -> tuple[int, int, int, int, int]:
    if (num_tokens, k, n, n_splits) == (64, 16384, 24, 64):
        return 16, 16, 64, 2, 3
    return 16, 32, 64, 4, 1


def mhc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out_mul: torch.Tensor,
    out_sqrsum: torch.Tensor,
    n_splits: int,
) -> None:
    """Split-K projection GEMM producing the reduce/apply kernel's inputs."""
    num_tokens, k = x.shape
    n = fn.shape[0]
    block_m, block_n, block_k, num_warps, num_stages = _mhc_prenorm_gemm_launch_config(
        num_tokens, k, n, n_splits
    )
    split_k = triton.cdiv(triton.cdiv(k, n_splits), block_k) * block_k
    _mhc_prenorm_gemm_triton_kernel[
        (
            n_splits,
            triton.cdiv(num_tokens, block_m),
            triton.cdiv(n, block_n),
        )
    ](
        x,
        fn,
        out_mul,
        out_sqrsum,
        num_tokens,
        K=k,
        N=n,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def gluon_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-4,
    sinkhorn_iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full mHC pre chain: split-K projection GEMM + gluon reduce/apply.

    Mirrors the calling contract of ``vkernels.torch_ops.aiter_ops.
    aiter_mhc_pre`` (``residual`` [T, hc, D] — bf16 or fp32; ``fn``
    [2*hc + hc*hc, hc*D] fp32, the ``[hc | hc | hc*hc]`` rows) so the two
    implementations benchmark apples-to-apples. Returns
    ``(post_mix [T, hc], comb_mix [T, hc*hc], layer_input [T, D] bf16)``.

    The vendored reduce/apply kernel specializes hidden ∈ {4096, 7168} with
    the matching split-K decomposition (``n_splits = hc*D // 64`` lanes) and
    1..64 tokens — decode-shaped batches.
    """
    from vkernels.torch_ops.tokenspeed_amd.mhc import gluon_mhc_pre_reduce_apply_gfx950

    tokens, hc, hidden = residual.shape
    if hc != 4:
        raise ValueError(
            f"gluon_mhc_pre specializes hc=4 (24 projection rows), got {hc}"
        )
    # The reduce/apply kernel's split-K contract: one split per 64 hidden
    # lanes, each covering hc*64 K elements of the hc*D projection.
    n_splits = hidden // 64
    x = residual.reshape(tokens, hc * hidden)
    out_mul = torch.empty(n_splits, tokens, 24, dtype=torch.float32, device=x.device)
    out_sqrsum = torch.empty(
        n_splits, tokens, dtype=torch.float32, device=x.device
    )
    mhc_prenorm_gemm(x, fn, out_mul, out_sqrsum, n_splits)

    layer_input = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=x.device)
    post_mix = torch.empty(tokens, 4, dtype=torch.float32, device=x.device)
    comb_mix = torch.empty(tokens, 16, dtype=torch.float32, device=x.device)
    gluon_mhc_pre_reduce_apply_gfx950(
        out_mul,
        out_sqrsum,
        hc_scale,
        hc_base,
        residual.to(torch.float32),
        layer_input,
        post_mix,
        comb_mix,
        hidden,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        n_splits,
        tokens,
    )
    return post_mix, comb_mix, layer_input
