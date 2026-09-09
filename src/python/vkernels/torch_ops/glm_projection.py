"""GLM batch-one small-output projection (issues #66 / #68).

One operator for the GLM decode GEMV family where default BLAS picks a
one-workgroup algorithm: DSA input projections (q_a_proj [1536,4096],
kv_a_proj_with_mqa [512,4096]) and KDA gate projections (f_a_proj /
g_a_proj [128,4096], b_proj [64,4096]). x has one or two rows; each
program computes ROWS output rows over a 1/SPLITK slice of the reduction
dim in FP32 through an inner 256-wide loop, partials are reduced
deterministically, and the result is rounded to BF16 once. Reduction
order differs from BLAS. The partial buffer is zero-filled immediately
before each launch (a small memset that captures and replays), so results
never depend on scratch history or autotune trials.

Contract: K a positive multiple of 4096 (the GLM family; every SPLITK in
{1..16} divides 4096, so all autotune configs are valid for every
supported K and no runtime config pruning is needed). The partial buffer
is zero-filled each call so the fixed 32-split reduce is correct for any
autotuned SPLITK; the zeroing is one small memset (<= 384 KB) that
captures and replays like the mHC operator's scratch.

Torch and Triton load lazily. Warm up each (device, rows, N, K) eagerly
before graph capture. Tuning is in-process only. Inputs are read-only and
this inference-only operator has no autograd backward.
"""

from functools import lru_cache

_WARMED = set()


def _validate(x, weight, *, gpu):
    import torch

    if x.ndim < 1 or x.shape[-1] % 4096 != 0:
        raise ValueError("expected x [...,K] with K a positive multiple of 4096")
    if weight.ndim != 2 or weight.shape[1] != x.shape[-1]:
        raise ValueError("expected weight [N,K] with K matching x")
    tokens = x.numel() // x.shape[-1]
    if tokens not in (1, 2):
        raise ValueError("projection supports one or two activation rows")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("projection requires BF16 inputs")
    if x.device != weight.device or (gpu and not x.is_cuda):
        raise ValueError("inputs must share a GPU device" if gpu else "inputs must share a device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("projection requires contiguous inputs")
    return tokens


def glm_projection_reference(x, weight):
    """FP32-accumulating Torch reference, available on CPU and GPU."""
    import torch

    _validate(x, weight, gpu=False)
    return torch.nn.functional.linear(x.float(), weight.float()).to(torch.bfloat16)


@lru_cache(maxsize=1)
def _kernels():
    # Put tl in the module namespace for Triton's annotation resolution.
    global tl
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[triton.Config({"ROWS": rows, "SPLITK": split}, num_warps=warps)
                 for rows in (1, 2, 4, 8, 16, 32)
                 for split in (1, 2, 4, 8, 16)
                 for warps in (4, 8)],
        key=["N", "K", "TOKENS", "DEVICE"],
    )
    @triton.jit
    def partial(X, W, P, N, K: tl.constexpr, TOKENS: tl.constexpr,
                DEVICE: tl.constexpr, ROWS: tl.constexpr, SPLITK: tl.constexpr):
        token = tl.program_id(2)
        split = tl.program_id(1)
        row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        slice_k: tl.constexpr = K // SPLITK          # 256 .. 4096, power of 2
        base = split * slice_k
        acc = tl.zeros([ROWS], dtype=tl.float32)
        for i in tl.static_range(0, slice_k, 256):   # <= 8 unrolled steps
            col = base + i + tl.arange(0, 256)
            x = tl.load(X + token * K + col).to(tl.float32)
            w = tl.load(W + row[:, None] * K + col[None, :],
                        row[:, None] < N, 0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        # Only this program's own slot: other slots of the same row belong
        # to concurrent programs on other splits — zeroing them here would
        # race. The launcher zero-fills the buffer right before the final
        # launch so unused slots read 0 deterministically.
        tl.store(P + (token * N + row) * 32 + split, acc, row < N)

    @triton.jit
    def reduce(P, Y, TOTAL, BN: tl.constexpr):
        idx = tl.program_id(0) * BN + tl.arange(0, BN)
        split = tl.arange(0, 32)
        mask = idx < TOTAL
        value = tl.load(P + idx[:, None] * 32 + split[None, :], mask[:, None], 0)
        tl.store(Y + idx, tl.sum(value, axis=1), mask)

    return partial, reduce


def glm_projection(x, weight):
    """Return BF16 [...,N]; eager warmup is required before GPU capture."""
    import torch

    tokens = _validate(x, weight, gpu=True)
    n = weight.shape[0]
    k = x.shape[-1]
    with torch.cuda.device(x.device):
        device = x.device.index
        key = (device, tokens, n, k)
        if torch.cuda.is_current_stream_capturing() and key not in _WARMED:
            raise RuntimeError("warm up glm_projection eagerly on this device/row-count/shape before capture")
        partial, reduce = _kernels()
        scratch = torch.empty((tokens, n, 32), dtype=torch.float32, device=x.device)
        out = torch.empty((*x.shape[:-1], n), dtype=torch.bfloat16, device=x.device)
        grid = lambda meta: ((n + meta["ROWS"] - 1) // meta["ROWS"], meta["SPLITK"], tokens)
        if key not in _WARMED:
            # First call on this key: autotune trials pollute the partial
            # buffer (every trial writes its own config's slots), so run
            # the tuning pass, then start from a clean slate below.
            partial[grid](x, weight, scratch, n, k, tokens, device,
                          enable_fp_fusion=False)
        # Deterministic zero fill right before the real launch: unused
        # slots read 0 for any cached SPLITK, and the memset captures and
        # replays inside a graph.
        scratch.zero_()
        partial[grid](x, weight, scratch, n, k, tokens, device,
                      enable_fp_fusion=False)
        total = tokens * n
        reduce[((total + 15) // 16,)](scratch, out, total, 16, enable_fp_fusion=False)
        _WARMED.add(key)
    return out


def glm_projection_tuning_metadata():
    """Return JSON-safe in-process tuning choices for benchmark reports."""
    if not _kernels.cache_info().currsize:
        return {"choices": []}
    return {"choices": [
        {"key": repr(key), "kwargs": dict(config.kwargs), "num_warps": config.num_warps}
        for key, config in _kernels()[0].cache.items()
    ]}
