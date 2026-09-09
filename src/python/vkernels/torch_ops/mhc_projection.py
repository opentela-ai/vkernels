"""Small-output BF16 mHC projection, with no activation normalization.

The fixed GLM shape is [..., 16384] @ [24, 16384].T with one or two
activation rows. Inputs are read-only. A 32-way split-K writes FP32 partials;
a second kernel reduces them deterministically and rounds once to BF16.
Reduction order differs from BLAS, so bitwise BLAS equivalence is not promised.

Torch and Triton are optional and loaded only on invocation. Call once eagerly
for each device/row-count before graph capture: this compiles and autotunes six
launch configurations. The warmed path allocates graph-pool-compatible scratch.
Tuning is in-process, keyed by device ordinal and row count, not a portable
deployment artifact. Retune in a fresh process after changing hardware/software.
"""

from functools import lru_cache

_WARMED = set()


def _validate(x, weight, *, gpu):
    import torch

    if x.ndim < 1 or x.shape[-1] != 16384 or tuple(weight.shape) != (24, 16384):
        raise ValueError("expected x [...,16384] and weight [24,16384]")
    rows = x.numel() // 16384
    if rows not in (1, 2):
        raise ValueError("mHC projection supports one or two activation rows")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("mHC projection requires BF16 inputs")
    if x.device != weight.device or (gpu and not x.is_cuda):
        raise ValueError("inputs must share a GPU device" if gpu else "inputs must share a device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("mHC projection requires contiguous inputs")
    return rows


def mhc_projection_reference(normalized_x, weight):
    """FP32-accumulating Torch reference, available on CPU and GPU."""
    import torch

    _validate(normalized_x, weight, gpu=False)
    return torch.nn.functional.linear(normalized_x.float(), weight.float()).to(torch.bfloat16)


@lru_cache(maxsize=1)
def _kernels():
    # Put tl in the module namespace for Triton's annotation/source resolution.
    global tl
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[triton.Config({"ROWS": rows}, num_warps=warps)
                 for rows in (1, 2, 4) for warps in (4, 8)],
        key=["TOKENS", "DEVICE"],
    )
    @triton.jit
    def partial(X, W, P, TOKENS: tl.constexpr, DEVICE: tl.constexpr, ROWS: tl.constexpr):
        token = tl.program_id(2)
        split = tl.program_id(1)
        row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        col = split * 512 + tl.arange(0, 512)
        x = tl.load(X + token * 16384 + col).to(tl.float32)
        w = tl.load(W + row[:, None] * 16384 + col[None, :], row[:, None] < 24, 0).to(tl.float32)
        value = tl.sum(w * x[None, :], axis=1)
        tl.store(P + (token * 24 + row) * 32 + split, value, row < 24)

    @triton.jit
    def reduce(P, Y, TOKENS: tl.constexpr):
        row = tl.program_id(0) * 4 + tl.arange(0, 4)
        split = tl.arange(0, 32)
        value = tl.load(P + row[:, None] * 32 + split[None, :], row[:, None] < TOKENS * 24, 0)
        out = tl.sum(value, axis=1)
        tl.store(Y + row, out, row < TOKENS * 24)

    return partial, reduce


def mhc_projection(normalized_x, weight):
    """Return BF16 linear output; eager warmup is required before GPU capture.

    This is an inference-only operator without an autograd backward. It does
    not normalize activations or fuse the subsequent mHC gate/Sinkhorn math.
    """
    import torch

    tokens = _validate(normalized_x, weight, gpu=True)
    with torch.cuda.device(normalized_x.device):
        device = normalized_x.device.index
        key = (device, tokens)
        if torch.cuda.is_current_stream_capturing() and key not in _WARMED:
            raise RuntimeError("warm up mhc_projection eagerly on this device/row-count before capture")
        partial, reduce = _kernels()
        scratch = torch.empty((tokens, 24, 32), dtype=torch.float32, device=normalized_x.device)
        out = torch.empty((*normalized_x.shape[:-1], 24), dtype=torch.bfloat16, device=normalized_x.device)
        partial[lambda meta: ((24 + meta["ROWS"] - 1) // meta["ROWS"], 32, tokens)](
            normalized_x, weight, scratch, tokens, device, enable_fp_fusion=False,
        )
        reduce[(tokens * 6,)](scratch, out, tokens, num_warps=4, enable_fp_fusion=False)
        _WARMED.add(key)
    return out


def mhc_projection_tuning_metadata():
    """Return JSON-safe in-process tuning choices for benchmark reports."""
    if not _kernels.cache_info().currsize:
        return {"split_k": 32, "choices": []}
    partial, _ = _kernels()
    return {
        "split_k": 32,
        "choices": [
            {"key": repr(key), "kwargs": dict(config.kwargs), "num_warps": config.num_warps}
            for key, config in partial.cache.items()
        ],
    }
