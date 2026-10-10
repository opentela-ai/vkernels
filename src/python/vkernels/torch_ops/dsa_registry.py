"""Metadata-only selection for the portable Triton DSA (sparse MLA) paths.

The compute kernels are :mod:`vkernels.torch_ops.tokenspeed_dsa` (vendored
from tokenspeed-kernel, MIT); this module registers them through the same
metadata-only machinery as the rest of torch_ops so planning can filter by
format signature and shape traits without importing torch/triton.
"""

from vkernels.registry import KernelImplementation, KernelRegistry, KernelRequest, Priority
from vkernels.signature import FormatSignature

__all__ = ["DSA_REGISTRY"]

_ATTN_DTYPES = frozenset(
    {"bfloat16", "float16", "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz"}
)


def _check(request: KernelRequest) -> tuple[str, ...]:
    reasons = []
    if len(request.tensors) in (2, 3):
        q = request.tensors[0]
        if len(q.shape) != 3:
            reasons.append("q must be [tokens, H, lora+rope]")
        elif q.shape[2] <= 0 or q.shape[1] <= 0:
            reasons.append("q heads and head dim must be positive")
        for tensor in request.tensors:
            if tensor.device.split(":")[0] not in ("cuda", "hip", "rocm"):
                reasons.append("GPU-resident tensors required")
                break
    return tuple(reasons)


DSA_SIGNATURE = FormatSignature(
    roles=("q", "kv"),
    dtypes=(_ATTN_DTYPES, _ATTN_DTYPES),
    layout="paged",  # topk slot addressing, not dense sequence attention
)


def _impl(name: str, entrypoint: str) -> KernelImplementation:
    return KernelImplementation(
        operation="dsa_attention", name=name, backend="triton",
        priority=Priority.PORTABLE,
        entrypoint=entrypoint,
        check=_check, graph_capture=True, signature=DSA_SIGNATURE,
    )


DSA_REGISTRY = KernelRegistry((
    _impl("triton_dense_kv", "vkernels.torch_ops.tokenspeed_dsa:triton_dsa_prefill"),
    _impl("triton_packed_kv", "vkernels.torch_ops.tokenspeed_dsa:triton_dsa_decode"),
))
