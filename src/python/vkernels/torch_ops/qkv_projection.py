"""Decode-only BF16 Q/K/V projection without packing or copying weights.

Each weight is [8192,4096]; x has one or two rows. One launch computes all
three projections with FP32 tree reductions and rounds the concatenated
Q/K/V result once to BF16. Reduction order can differ from BLAS.

Torch and Triton load lazily. Warm up each device/row-count eagerly before
graph capture. Eight configurations are autotuned in process, not persisted;
retune in a fresh process after hardware or software changes. Inputs are
read-only, and this inference-only operator has no autograd backward.
"""

from functools import lru_cache

_WARMED = set()


def _validate(x, weights, *, gpu):
    import torch

    if x.ndim < 1 or x.shape[-1] != 4096 or any(tuple(w.shape) != (8192, 4096) for w in weights):
        raise ValueError("expected x [...,4096] and three weights [8192,4096]")
    tokens = x.numel() // 4096
    if tokens not in (1, 2):
        raise ValueError("QKV projection supports one or two activation rows")
    if x.dtype != torch.bfloat16 or any(w.dtype != torch.bfloat16 for w in weights):
        raise TypeError("QKV projection requires BF16 inputs")
    if any(w.device != x.device for w in weights) or (gpu and not x.is_cuda):
        raise ValueError("inputs must share a GPU device" if gpu else "inputs must share a device")
    if not x.is_contiguous() or any(not w.is_contiguous() for w in weights):
        raise ValueError("QKV projection requires contiguous inputs")
    return tokens


def qkv_projection_reference(x, q_weight, k_weight, v_weight):
    """FP32-accumulating Torch reference, available on CPU and GPU."""
    import torch

    weights = (q_weight, k_weight, v_weight)
    _validate(x, weights, gpu=False)
    return torch.cat([
        torch.nn.functional.linear(x.float(), weight.float()).to(torch.bfloat16)
        for weight in weights
    ], dim=-1)


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[triton.Config({"ROWS": rows}, num_warps=warps)
                 for rows in (1, 2, 4, 8) for warps in (4, 8)],
        key=["TOKENS", "DEVICE"],
    )
    @triton.jit
    def project(X, Q, K, V, Y, TOKENS: tl.constexpr, DEVICE: tl.constexpr, ROWS: tl.constexpr):
        projection = tl.program_id(1)
        token = tl.program_id(2)
        row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, 4096)
        # A whole program reads exactly one original matrix. No packed weight
        # allocation, activation normalization, or intermediate tensor is used.
        if projection == 0:
            weight = Q
        elif projection == 1:
            weight = K
        else:
            weight = V
        x = tl.load(X + token * 4096 + col).to(tl.float32)
        w = tl.load(weight + row[:, None] * 4096 + col[None, :], row[:, None] < 8192, 0).to(tl.float32)
        value = tl.sum(w * x[None, :], axis=1)
        tl.store(Y + token * 24576 + projection * 8192 + row, value, row < 8192)

    return project


def qkv_projection(x, q_weight, k_weight, v_weight):
    """Return contiguous BF16 [...,24576], ordered Q then K then V."""
    import torch

    tokens = _validate(x, (q_weight, k_weight, v_weight), gpu=True)
    with torch.cuda.device(x.device):
        device = x.device.index
        key = (device, tokens)
        if torch.cuda.is_current_stream_capturing() and key not in _WARMED:
            raise RuntimeError("warm up qkv_projection eagerly on this device/row-count before capture")
        project = _kernel()
        out = torch.empty((*x.shape[:-1], 24576), dtype=torch.bfloat16, device=x.device)
        project[lambda meta: ((8192 + meta["ROWS"] - 1) // meta["ROWS"], 3, tokens)](
            x, q_weight, k_weight, v_weight, out, tokens, device, enable_fp_fusion=False,
        )
        _WARMED.add(key)
    return out


def qkv_projection_tuning_metadata():
    """Return JSON-safe in-process tuning choices for benchmark reports."""
    if not _kernel.cache_info().currsize:
        return {"choices": []}
    return {"choices": [
        {"key": repr(key), "kwargs": dict(config.kwargs), "num_warps": config.num_warps}
        for key, config in _kernel().cache.items()
    ]}
