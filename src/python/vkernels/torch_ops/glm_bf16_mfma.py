"""Default-off BF16 matrix-core projections qualified on the Beverin stack.

Prototype pins matched native hipBLASLt results on gfx942. Operand
order and K packing are part of the numerical policy; do not tune them at
runtime. Two prototype operator stages tested these 27 cells, including
1024 changing inputs/four new weights per cell and signed-zero/cancellation
cases. The full-vocabulary single-row head also matched 30 trained MTP
hidden rows and 4096 cells across four new weights, including signed zero
and cancellation. This API and full-model speed/quality require separate
qualification.
"""
from functools import lru_cache
from ._dispatch import OpNotEligible

_QUALIFIED_TORCH = "2.9.0a0+git7bcbafe"
_QUALIFIED_HIP = "7.0.51831-a3e329ad8"
# (outputs, inputs, rows): (block_k, weight_first, kpack). All use
# 16x16 matrix instructions, two warps, one stage, FP32 accumulation.
_PINS = {
    (16, 4096, 1): (512, True, 1),
    (16, 4096, 2): (512, True, 1),
    (16, 4096, 3): (512, True, 1),
    (16, 4096, 4): (512, True, 1),
    (24, 16384, 1): (512, True, 1),
    (24, 16384, 2): (512, True, 1),
    (24, 16384, 3): (512, True, 1),
    (24, 16384, 4): (512, True, 1),
    (32, 4096, 1): (512, True, 1),
    (32, 4096, 2): (512, True, 1),
    (32, 4096, 3): (512, True, 1),
    (32, 4096, 4): (512, True, 1),
    (128, 4096, 1): (512, True, 1),
    (128, 4096, 2): (512, True, 1),
    (128, 4096, 3): (512, True, 1),
    (128, 4096, 4): (512, True, 1),
    (512, 4096, 1): (512, False, 2),
    (512, 4096, 2): (512, False, 2),
    (512, 4096, 3): (512, False, 2),
    (512, 4096, 4): (512, False, 2),
    (2048, 128, 1): (256, False, 2),
    (2048, 128, 3): (256, False, 2),
    (2048, 128, 4): (128, False, 2),
    (38720, 4096, 1): (512, False, 2),
    (38720, 4096, 2): (512, False, 2),
    (38720, 4096, 3): (128, False, 2),
    (38720, 4096, 4): (512, False, 2),
    (154880, 4096, 1): (512, False, 2),
}


@lru_cache(maxsize=1)
def _kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _gemm_mfma_input(X, W, Y, M: tl.constexpr, N: tl.constexpr,
                       K: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr = 16,
                       BN: tl.constexpr = 16):
        rm = tl.arange(0, BM)
        rn = tl.program_id(0) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for offset in range(0, K, BK):
            a = tl.load(X + rm[:, None] * K + offset + rk[None, :],
                        (rm[:, None] < M) & (offset + rk[None, :] < K), other=0)
            b = tl.load(W + rn[:, None] * K + offset + rk[None, :],
                        (rn[:, None] < N) & (offset + rk[None, :] < K), other=0)
            acc = tl.dot(a, tl.trans(b), acc)
        tl.store(Y + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
                 (rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _gemm_mfma_weight(X, W, Y, M: tl.constexpr, N: tl.constexpr,
                        K: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr = 16,
                        BN: tl.constexpr = 16):
        rm = tl.arange(0, BM)
        rn = tl.program_id(0) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BN, BM), tl.float32)
        for offset in range(0, K, BK):
            a = tl.load(X + rm[:, None] * K + offset + rk[None, :],
                        (rm[:, None] < M) & (offset + rk[None, :] < K), other=0)
            b = tl.load(W + rn[:, None] * K + offset + rk[None, :],
                        (rn[:, None] < N) & (offset + rk[None, :] < K), other=0)
            acc = tl.dot(b, tl.trans(a), acc)
        tl.store(Y + rm[None, :] * N + rn[:, None], acc.to(tl.bfloat16),
                 (rm[None, :] < M) & (rn[:, None] < N))

    return triton, _gemm_mfma_input, _gemm_mfma_weight


def bf16_mfma_linear(x, weight):
    """Return F.linear-shaped output for the measured inference cells.

    Inputs are read-only. Unmeasured shapes, layouts, devices, software or
    BLAS policies decline. No process-global backend setting is changed.
    Eager warmup must precede graph capture for each admitted geometry.
    """
    import torch

    if x.ndim < 1 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise OpNotEligible("expected x [..., K] and weight [N, K]")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise OpNotEligible("requires BF16 inputs and weights")
    if not (x.is_cuda and weight.is_cuda and x.device == weight.device):
        raise OpNotEligible("inputs must share a HIP device")
    if not (x.is_contiguous() and weight.is_contiguous()):
        raise OpNotEligible("requires contiguous inputs and weights")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise OpNotEligible("inference only")
    if str(torch.__version__) != _QUALIFIED_TORCH or torch.version.hip != _QUALIFIED_HIP:
        raise OpNotEligible("unqualified Torch/HIP stack")
    if str(torch.backends.cuda.preferred_blas_library()) != "_BlasBackend.Cublaslt":
        raise OpNotEligible("pins require native hipBLASLt policy")
    import torch.cuda.tunable as tunable
    if tunable.is_enabled():
        raise OpNotEligible("native BLAS tuning policy changed")
    if torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0] != "gfx942":
        raise OpNotEligible("qualified on gfx942 only")
    n, k = weight.shape
    if k <= 0:
        raise OpNotEligible("empty reduction")
    rows = x.numel() // k
    pin = _PINS.get((n, k, rows))
    if pin is None:
        raise OpNotEligible("unmeasured projection geometry")
    block_k, weight_first, kpack = pin
    triton, input_kernel, weight_kernel = _kernels()
    output = torch.empty((rows, n), device=x.device, dtype=x.dtype)
    kernel = weight_kernel if weight_first else input_kernel
    with torch.cuda.device(x.device):
        kernel[(triton.cdiv(n, 16),)](x.view(rows, k), weight, output,
            rows, n, k, block_k, num_warps=2, num_stages=1,
            matrix_instr_nonkdim=16, kpack=kpack)
    return output.view(*x.shape[:-1], n)
