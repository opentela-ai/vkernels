"""Metadata-only selection for the portable Triton MLA paths.

Planning uses the same metadata-only registry as the rest of torch_ops:
capability checks are pure functions of :class:`~vkernels.registry.KernelRequest`
metadata, the dtype contract is a :class:`~vkernels.signature.FormatSignature`,
and loading the launcher (which imports torch/triton) happens only at
execution time. Kernels: :mod:`vkernels.torch_ops.tokenspeed_mla`
(vendored from tokenspeed-kernel, MIT).
"""

from vkernels.registry import KernelImplementation, KernelRegistry, KernelRequest, Priority
from vkernels.signature import FormatSignature

__all__ = ["MLA_REGISTRY", "MLA_PREFILL_REGISTRY", "MLA_DECODE_REGISTRY"]

# fp16/bf16 plus the FP8 storage dtypes the kernels read; output is bf16
# for FP8 inputs (the launchers decide that, not the signature).
_ATTN_DTYPES = frozenset(
    {"float16", "bfloat16", "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz"}
)


def _paged_layout_check(request: KernelRequest) -> tuple[str, ...]:
    reasons = []
    if request.tensors:
        for tensor in request.tensors:
            if tensor.device.split(":")[0] not in ("cuda", "hip", "rocm"):
                reasons.append("GPU-resident tensors required")
                break
    return tuple(reasons)


MLA_PREFILL_SIGNATURE = FormatSignature(
    roles=("q", "k", "v"),
    dtypes=(_ATTN_DTYPES, _ATTN_DTYPES, _ATTN_DTYPES),
    layout="dense",
)

MLA_DECODE_SIGNATURE = FormatSignature(
    roles=("q", "kv_cache"),
    dtypes=(_ATTN_DTYPES, _ATTN_DTYPES),
    layout="paged",
)


def _kv_head_grouping(request: KernelRequest) -> tuple[str, ...]:
    if len(request.tensors) == 3:
        q_heads, kv_heads = request.tensors[0].shape[1], request.tensors[1].shape[1]
        if kv_heads and q_heads % kv_heads:
            return (f"num_q_heads ({q_heads}) must be divisible by num_kv_heads ({kv_heads})",)
    return ()


MLA_PREFILL_REGISTRY = KernelRegistry((
    KernelImplementation(
        operation="mla_prefill", name="triton", backend="triton",
        priority=Priority.PORTABLE,
        entrypoint="vkernels.torch_ops.tokenspeed_mla:triton_mla_prefill",
        check=_kv_head_grouping, graph_capture=True, signature=MLA_PREFILL_SIGNATURE,
    ),
))


def _decode_shapes(request: KernelRequest) -> tuple[str, ...]:
    reasons = []
    if len(request.tensors) == 2:
        q, kv = request.tensors
        if len(q.shape) != 4:
            reasons.append("q must be [B, 1, H, lora+rope]")
        elif q.shape[1] != 1:
            reasons.append("triton MLA decode supports q_len == 1")
        if len(kv.shape) not in (3, 4):
            reasons.append("kv_cache must be 3-D or 4-D paged latent cache")
    return tuple(reasons)


MLA_DECODE_REGISTRY = KernelRegistry((
    KernelImplementation(
        operation="mla_decode", name="triton", backend="triton",
        priority=Priority.PORTABLE,
        entrypoint="vkernels.torch_ops.tokenspeed_mla:triton_mla_decode_with_kvcache",
        check=_decode_shapes, graph_capture=True, signature=MLA_DECODE_SIGNATURE,
    ),
))


def _all_impls():
    return (*MLA_PREFILL_REGISTRY.implementations, *MLA_DECODE_REGISTRY.implementations)


#: Combined view for planning tools that want every MLA registration at once.
MLA_REGISTRY = KernelRegistry(_all_impls())
