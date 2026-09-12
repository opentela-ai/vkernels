"""Deferred GDN SSM commit for speculative decoding (DFlash2 verify).

Computes, for one GDN layer, the recurrent gated-delta state after block
token ``j`` -- the committed SSM state at accepted position ``j``:

    s = ssm_state
    for t in 0..j:
        s = s * gamma[t]
        sk = s @ k[t]                     # [hv] matrix-vector
        s = s + beta[t] * (v[t] - sk) ⊗ k[t]

This is exactly the numerically-stable recurrent form used by the qwen35
GDN runner's eager reference, fused into ONE kernel per layer (grid over
value heads x value-row blocks) instead of ~6*(j+1) tiny launches per
layer -- measured ~10 ms/step -> ~0.5 ms/step for 48 GDN layers at
j_avg~1.6.

Only the FINAL state is produced (no per-token intermediates), matching the
deferred-commit design. Numerics agree with the torch reference to ~1e-6
relative (different fp32 reduction order in the matvec).

All three element strides of ``k``/``v`` are passed explicitly: the real
verify path's ``v_f32`` is a NON-CONTIGUOUS view (strides like
``(1, 1024, 8)`` -- innermost stride 8) of the qkv-split buffer, which an
earlier stride-1 assumption silently corrupted.

Adopted from floe's ``engine/runner/kernels/gdn_commit.py`` (vkernels owns
the kernel; floe imports it back through a thin adapter — the #64/#65
thin-adapter model). Torch and Triton load lazily. Inference-only.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _gdn_state_at_j_kernel(
        K_ptr,
        V_ptr,
        BETA_ptr,
        GAMMA_ptr,
        S0_ptr,
        OUT_ptr,
        stride_kt,
        stride_kh,
        stride_kk,
        stride_vt,
        stride_vh,
        stride_vv,
        stride_bt,
        stride_gt,
        J: tl.constexpr,
        HV: tl.constexpr,
        HK: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """One program owns a [BLOCK_V, HK] slice of one value head's state."""
        h = tl.program_id(0)  # value head
        vb = tl.program_id(1)  # value-row block
        offs_v = vb * BLOCK_V + tl.arange(0, BLOCK_V)  # [BLOCK_V] rows
        offs_k = tl.arange(0, HK)  # [HK] cols

        s = tl.load(
            S0_ptr + h * HV * HK + offs_v[:, None] * HK + offs_k[None, :]
        )  # [BV, HK] fp32

        for t in range(J + 1):
            g = tl.load(GAMMA_ptr + t * stride_gt + h)
            b = tl.load(BETA_ptr + t * stride_bt + h)
            kvec = tl.load(
                K_ptr + t * stride_kt + h * stride_kh + offs_k * stride_kk
            )  # [HK]
            vvec = tl.load(
                V_ptr + t * stride_vt + h * stride_vh + offs_v * stride_vv
            )  # [BV]
            s = s * g
            sk = tl.sum(s * kvec[None, :], axis=1)  # [BV] matvec over HK
            s = s + (b * (vvec - sk))[:, None] * kvec[None, :]

        tl.store(OUT_ptr + h * HV * HK + offs_v[:, None] * HK + offs_k[None, :], s)

    return _gdn_state_at_j_kernel


def gdn_commit_state_triton(
    k_exp,  # [seq, nv, hk] fp32 (normalized, expanded)
    v,  # [seq, nv, hv] fp32 (may be a strided view)
    beta,  # [seq, nv] fp32
    gamma,  # [seq, nv] fp32
    ssm_state,  # [nv, hv, hk] fp32 (input, NOT mutated)
    j: int,
):
    """Committed state [nv, hv, hk] after block token ``j`` (Triton path).

    ``beta``/``gamma`` must be innermost-contiguous; ``k``/``v`` may have
    arbitrary strides (all three are passed to the kernel); ``hv`` must be a
    multiple of the 32-row value block (the kernel is unmasked over rows).
    """
    import torch

    seq, NV, HK = k_exp.shape
    HV = v.shape[-1]
    BLOCK_V = 32
    if not (k_exp.is_cuda and v.is_cuda):
        raise ValueError("triton path requires CUDA tensors")
    if seq < j + 1:
        raise ValueError(f"k_exp/v seq ({seq}) must cover j + 1 ({j + 1})")
    if HV % BLOCK_V:
        raise ValueError(f"hv ({HV}) must be a multiple of the {BLOCK_V}-row block")
    if beta.stride(1) != 1 or gamma.stride(1) != 1:
        raise ValueError("beta/gamma must be innermost-contiguous")
    if not ssm_state.is_contiguous():
        ssm_state = ssm_state.contiguous()
    out = torch.empty_like(ssm_state)
    grid = (NV, HV // BLOCK_V)

    _kernel()[grid](
        k_exp,
        v,
        beta,
        gamma,
        ssm_state,
        out,
        k_exp.stride(0),
        k_exp.stride(1),
        k_exp.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        beta.stride(0),
        gamma.stride(0),
        J=int(j),
        HV=HV,
        HK=HK,
        BLOCK_V=BLOCK_V,
        num_warps=4,
    )
    return out


def gdn_commit_state_reference(
    k_exp,  # [seq, nv, hk] fp32 (normalized, expanded)
    v,  # [seq, nv, hv] fp32
    beta,  # [seq, nv] fp32
    gamma,  # [seq, nv] fp32
    ssm_state,  # [nv, hv, hk] fp32 (input, NOT mutated)
    j: int,
):
    """Eager FP32 oracle: the numerically-stable recurrent form."""

    s = ssm_state.clone()
    for t in range(int(j) + 1):
        s = s * gamma[t][:, None, None]
        sk = (s * k_exp[t][:, None, :]).sum(-1)  # [nv, hv] matvec over HK
        s = s + (beta[t][:, None] * (v[t] - sk))[..., None] * k_exp[t][:, None, :]
    return s
