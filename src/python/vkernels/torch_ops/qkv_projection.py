"""Decode-only BF16 Q/K/V projection without packing or copying weights.

The three weights share one shape ``[O, I]`` (any equal-shape triple:
the original 8192-row unsharded envelope, the TP4 per-rank ``[2048, 4096]``
triples, or any other); ``I`` must be a power of two (``tl.arange``) and
matches ``x``'s last dim. ``x`` has one to eight rows (the decode-GEMV
``m <= 8`` policy, matching ``fp8_dense_gemv_max_m``). One launch computes
all three projections with FP32 tree reductions and rounds the concatenated
Q/K/V result once to BF16. Reduction order can differ from BLAS.

Torch and Triton load lazily. Warm up each (device, row-count) key eagerly
before capture. Configurations are autotuned per ``(O, tokens)`` on first
launch; with the tuning cache the winner persists per device
 (``tuning_cache.py``), so a fresh process replays the stored choice. Inputs
are read-only, and this inference-only operator has no autograd backward.
"""

from ._dispatch import OpNotEligible
from .tuning_cache import persistent_autotune
from functools import lru_cache

_WARMED = set()
_TOKENS = (1, 2, 4, 8)  # the decode-GEMV m<=8 policy (fp8_dense_gemv_max_m)


def _validate(x, weights, *, gpu):
    import torch

    if x.ndim < 1 or any(w.ndim != 2 for w in weights):
        raise OpNotEligible("expected x [...,I] and three 2-D weights")
    i = x.shape[-1]
    shapes = {tuple(w.shape) for w in weights}
    if len(shapes) != 1:
        raise OpNotEligible(f"expected three equal-shape weights, got {sorted(shapes)}")
    (o, wi), = shapes
    if wi != i:
        raise OpNotEligible(f"expected x [...,{wi}] with matching weights, got x width {i}")
    if i < 1 or i & (i - 1):
        raise OpNotEligible("expected a power-of-two weight width (tl.arange constraint)")
    tokens = x.numel() // i
    if tokens not in _TOKENS:
        raise OpNotEligible(f"QKV projection supports one to {_TOKENS[-1]} activation rows")
    if x.dtype != torch.bfloat16 or any(w.dtype != torch.bfloat16 for w in weights):
        raise OpNotEligible("QKV projection requires BF16 inputs")
    if any(w.device != x.device for w in weights) or (gpu and not x.is_cuda):
        raise OpNotEligible("inputs must share a GPU device" if gpu else "inputs must share a device")
    if not x.is_contiguous() or any(not w.is_contiguous() for w in weights):
        raise OpNotEligible("QKV projection requires contiguous inputs")
    return tokens, o, i


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

    @persistent_autotune(
        configs=[triton.Config({"ROWS": rows}, num_warps=warps)
                 for rows in (1, 2, 4, 8) for warps in (4, 8)],
        key=["TOKENS", "DEVICE", "O"],
        kernel_name="qkv_projection",
        source_files=[__file__],
    )
    @triton.jit
    def project(X, Q, K, V, Y, TOKENS: tl.constexpr, DEVICE: tl.constexpr,
                O: tl.constexpr, I: tl.constexpr, ROWS: tl.constexpr):
        projection = tl.program_id(1)
        token = tl.program_id(2)
        row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, I)
        # A whole program reads exactly one original matrix. No packed weight
        # allocation, activation normalization, or intermediate tensor is used.
        if projection == 0:
            weight = Q
        elif projection == 1:
            weight = K
        else:
            weight = V
        x = tl.load(X + token * I + col).to(tl.float32)
        w = tl.load(weight + row[:, None] * I + col[None, :], row[:, None] < O, 0).to(tl.float32)
        value = tl.sum(w * x[None, :], axis=1)
        tl.store(Y + token * (3 * O) + projection * O + row, value, row < O)

    return project


def qkv_projection(x, q_weight, k_weight, v_weight):
    """Return contiguous BF16 ``[..., 3*O]``, ordered Q then K then V."""
    import torch

    tokens, o, i = _validate(x, (q_weight, k_weight, v_weight), gpu=True)
    with torch.cuda.device(x.device):
        device = x.device.index
        key = (device, tokens)
        if torch.cuda.is_current_stream_capturing() and key not in _WARMED:
            raise RuntimeError("warm up qkv_projection eagerly on this device/row-count before capture")
        project = _kernel()
        out = torch.empty((*x.shape[:-1], 3 * o), dtype=torch.bfloat16, device=x.device)
        project[lambda meta: ((o + meta["ROWS"] - 1) // meta["ROWS"], 3, tokens)](
            x, q_weight, k_weight, v_weight, out, tokens, device, o, i,
            enable_fp_fusion=False,
        )
        _WARMED.add(key)
    return out


def qkv_projection_tune(device="cuda"):
    """Drive the persistent-autotune sweep for every (device, rows) key.

    Registry entry (``vkernels.torch_ops.tuner``) calls this: one launch
    per row-count on synthetic tensors triggers the sweep; with the tuning
    cache enabled the winner of each key lands in the store. Sweeps both
    envelope shapes: the unsharded ``[8192, 4096]`` triple and the TP4
    per-rank ``[2048, 4096]`` triple (the deployed recipe's shape).
    """
    import torch

    swept = []
    for tokens in _TOKENS:
        for o in (8192, 2048):
            x = torch.randn(tokens, 4096, device=device, dtype=torch.bfloat16)
            weights = [torch.randn(o, 4096, device=device, dtype=torch.bfloat16)
                       for _ in range(3)]
            qkv_projection(x, *weights)
            torch.cuda.synchronize(device)
            swept.append((torch.cuda.current_device(), tokens, o))
    return swept


def qkv_projection_tuning_metadata():
    """Return JSON-safe in-process tuning choices for benchmark reports."""
    if not _kernel.cache_info().currsize:
        return {"choices": []}
    return {"choices": [
        {"key": repr(key), "kwargs": dict(config.kwargs), "num_warps": config.num_warps}
        for key, config in _kernel().cache.items()
    ]}
