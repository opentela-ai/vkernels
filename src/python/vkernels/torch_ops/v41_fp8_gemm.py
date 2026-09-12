"""DeepSeek-V4.1 fp8 block-scaled GEMM with UE8M0 (power-of-two) scales.

V4.1's backbone projections run fp8 GEMMs whose scales are **UE8M0-rounded**
(power of two) at a configurable block (32 in the reference; §2.9), unlike the
GLM-5 path's 128x128 fp32 scales. The block-scaled GEMM math is the same
DeepSeek scheme (``w = q_fp8 * scale[·//block]``), so this module adds:

* ``quantize_fp8_ue8m0`` — bf16 -> (e4m3, UE8M0 block scales): per (block x
  block) tile, ``scale = 2**ceil(log2(amax/448))`` (a power of two, i.e. an
  E8M0-representable exponent), payload ``round(x/scale)`` clamped to e4m3.
* ``fp8_block_gemm`` / ``*_reference`` — general block-size block-scaled GEMM.
  For ``block == 128`` on GPU it delegates to the proven GLM Hopper kernel
  (``glm_fp8_blockwise_gemm``); otherwise it uses the torch oracle.

CPU ``*_reference`` is the always-tested oracle. Torch/Triton load lazily.
Inference-only, no autograd backward.
"""


def quantize_fp8_ue8m0(x, block: int = 32):
    """BF16/FP32 ``[M, K]`` -> (e4m3 ``[M, K]``, fp32 UE8M0 scales
    ``[ceil(M/block), K//block]``). Scales are exact powers of two."""
    import torch
    import torch.nn.functional as F

    if x.ndim != 2:
        raise ValueError("expected a 2-D [M, K] tensor")
    m, k = x.shape
    if k % block:
        raise ValueError(f"K ({k}) must be a multiple of block ({block})")
    mb = (m + block - 1) // block
    pad = mb * block - m
    xp = F.pad(x, (0, 0, 0, pad)) if pad else x
    xg = xp.view(mb, block, k // block, block).float()
    amax = xg.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    exponent = torch.ceil(torch.log2(amax / 448.0))  # UE8M0: scale is 2**e
    scale = torch.exp2(exponent)
    q = (xg / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(mb * block, k)[:m].contiguous(), scale.reshape(mb, k // block).contiguous()


def _dequant_blocks(q_fp8, scales, block, rows, cols):
    full = scales.float().repeat_interleave(block, 0).repeat_interleave(block, 1)
    return q_fp8.float() * full[:rows, :cols]


def fp8_block_gemm_reference(a_fp8, a_scales, b_fp8, b_scales, *, block: int = 32, out_dtype=None):
    """``D = dequant(A) @ dequant(B).T`` in fp32, rounded to ``out_dtype``
    (bf16). ``A``: ``[M, K]`` with scales ``[ceil(M/block), K//block]``;
    ``B``: ``[N, K]`` (weights, transposed-use) with scales
    ``[ceil(N/block), K//block]``."""
    import torch

    if out_dtype is None:
        out_dtype = torch.bfloat16
    m, k = a_fp8.shape
    n, k2 = b_fp8.shape
    if k != k2:
        raise ValueError(f"K mismatch: A K={k}, B K={k2}")
    a = _dequant_blocks(a_fp8, a_scales, block, m, k)
    b = _dequant_blocks(b_fp8, b_scales, block, n, k)
    return (a @ b.t()).to(out_dtype)


# Backend selection for the decode-shape Triton path below. "auto" (default)
# shape-gates on the measured GB10 crossover (see the harness): the Triton
# kernel wins when N is large enough to fill the grid (wo_b 3.15x, wq_b 1.5x)
# and loses ~11x on small-N shapes (wkv N=512 -> 4 CTAs). "reference" forces
# the torch oracle everywhere; "triton" forces the kernel for all eligible
# shapes (correct, just slower when N is small).
_FP8_GEMM_DEFAULT = "auto"
_FP8_GEMM_TRITON_MIN_N = 2048


def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _e4m3_to_f32(c):
        # e4m3fn bit-exact decode: exp>0 -> (1+man/8)*2^(exp-7); exp==0 ->
        # man*2^-9; 0x7F/0xFF -> NaN (payloads are clamped so this is defensive
        # only, but keeps parity with the hardware cvt the oracle uses).
        sign = tl.where((c & 0x80) != 0, -1.0, 1.0)
        exp = (c & 0x78) >> 3
        man = c & 0x07
        val = tl.where(exp > 0, (1.0 + man.to(tl.float32) / 8.0) * tl.exp2(exp.to(tl.float32) - 7.0),
                       man.to(tl.float32) * tl.exp2(-9.0))
        return sign * tl.where((c & 0x7F) == 0x7F, float("nan"), val)

    @triton.jit
    def _fp8_gemm_decode(A, SA, B, SB, Y,
                         M: tl.constexpr, MR: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         BLOCK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
        # Decode-shape block-scaled GEMM: D[m,n] = sum_k
        # e4m3(A[m,k]) * sa[k//BLOCK] * e4m3(B[n,k]) * sb[n//BLOCK, k//BLOCK].
        # M is the tl.dot-padded tile height (>= 16); MR the REAL row count
        # (all masks key off MR — the padded rows must neither load nor store).
        # Payloads arrive as uint8 (e4m3fn codes) so `.to(dtype)` passes
        # upstream can't cast them; decode is arithmetic (bit-exact for
        # e4m3 -> fp32). A-scale row 0 only (MR <= BLOCK).
        pid = tl.program_id(0)
        m = tl.arange(0, M)
        n = pid * BN + tl.arange(0, BN)
        mm = m < MR
        nm = n < N
        acc = tl.zeros((M, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            km = k < K
            ac = tl.load(A + m[:, None] * K + k[None, :], mm[:, None] & km[None, :], 0).to(tl.int32)
            av = _e4m3_to_f32(ac)
            sa = tl.load(SA + k // BLOCK, km, 0.0)
            bc = tl.load(B + n[:, None] * K + k[None, :], nm[:, None] & km[None, :], 0).to(tl.int32)
            bv = _e4m3_to_f32(bc)
            # Scales are per (BLOCK x BLOCK) tile: SB[n // BLOCK, k // BLOCK].
            sb = tl.load(SB + (n // BLOCK)[:, None] * (K // BLOCK) + (k // BLOCK)[None, :], nm[:, None] & km[None, :], 0.0)
            acc = tl.dot(av * sa[None, :], tl.trans(bv * sb), acc, input_precision="ieee")
        tl.store(Y + m[:, None] * N + n[None, :], acc.to(tl.bfloat16), mm[:, None] & nm[None, :])

    return _fp8_gemm_decode


def _triton_decode_gemm(a_fp8, a_scales, b_fp8, b_scales, *, block: int, out_dtype):
    import torch

    m, k = a_fp8.shape
    n = b_fp8.shape[0]
    if k % block or 64 % block:
        return fp8_block_gemm_reference(a_fp8, a_scales, b_fp8, b_scales, block=block, out_dtype=out_dtype)
    au = a_fp8.contiguous().view(torch.uint8)
    bu = b_fp8.contiguous().view(torch.uint8)
    sa = a_scales.float().contiguous()
    sb = b_scales.float().contiguous()
    out = torch.empty((m, n), device=au.device, dtype=torch.bfloat16)
    import triton

    # tl.dot needs >=16 rows; pad M up to a power of two <= 32.
    mpad = max(16, triton.next_power_of_2(m))
    bn = 128 if n >= 128 else triton.next_power_of_2(n)
    with torch.cuda.device(au.device):
        _kernel()[(triton.cdiv(n, bn),)](
            au, sa, bu, sb, out, mpad, m, n, k, block, 64, bn,
            num_warps=4, num_stages=2,
        )
    return out[:m]


def fp8_block_gemm(a_fp8, a_scales, b_fp8, b_scales, *, block: int = 32, out_dtype=None):
    """Device block-scaled fp8 GEMM. For ``block == 128`` on GPU, delegates to
    the GLM Hopper kernel; decode shapes (``M <= block``) take the Triton
    block-scaled kernel when the measured crossover says it wins (``N >= 2048``
    under the default ``VKERNELS_V41_FP8_GEMM_BACKEND=auto``; ``reference``
    forces the torch oracle, ``triton`` forces the kernel); otherwise the
    torch reference (always correct)."""
    import os

    import torch

    if out_dtype is None:
        out_dtype = torch.bfloat16
    on_gpu = a_fp8.is_cuda and b_fp8.is_cuda
    if block == 128 and on_gpu:
        try:
            from .glm_fp8_blockwise_gemm import fp8_blockwise_gemm

            return fp8_blockwise_gemm(a_fp8, a_scales, b_fp8, b_scales, out_dtype=out_dtype)
        except Exception:
            # house idiom (glm_fp8_blockwise_gemm): no triton, or a shape the
            # GLM kernel validates against, translates to the torch oracle
            pass
    m = a_fp8.shape[0]
    n = b_fp8.shape[0]
    backend = os.environ.get("VKERNELS_V41_FP8_GEMM_BACKEND", _FP8_GEMM_DEFAULT)
    eligible = on_gpu and m <= block and block != 128 and n >= _FP8_GEMM_TRITON_MIN_N
    if (backend == "triton" and on_gpu and m <= block and block != 128) or (backend == "auto" and eligible):
        try:
            import triton  # noqa: F401
        except Exception:
            pass
        else:
            return _triton_decode_gemm(a_fp8, a_scales, b_fp8, b_scales, block=block, out_dtype=out_dtype)
    return fp8_block_gemm_reference(a_fp8, a_scales, b_fp8, b_scales, block=block, out_dtype=out_dtype)
