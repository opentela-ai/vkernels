"""Discover and explain the MoE combine contract without importing torch.

Example startup planning (no GPU runtime needed)::

    request = KernelRequest(
        "moe_weighted_sum",
        (TensorMetadata((4, 8, 4096), "bfloat16", "cuda:0"),
         TensorMetadata((4, 8), "float32", "cuda:0")),
        backends=frozenset({"triton"}), graph_capture=True,
    )
    print(MOE_COMBINE_REGISTRY.explain(request))

The Triton implementation accepts strided inputs and needs no scratch beyond
its output. Graph capture requires eager warmup for the actual shapes/dtypes.
"""

from vkernels.registry import (
    KernelImplementation,
    KernelRegistry,
    KernelRequest,
    Priority,
    TensorMetadata,
)
from vkernels.signature import FormatSignature

__all__ = ["MOE_COMBINE_REGISTRY", "KernelRequest", "TensorMetadata"]


# Per-role dtype contract: bf16/fp16/fp32 activations, fp32 combine weights.
_MOE_COMBINE_SIGNATURE = FormatSignature(
    roles=("activations", "weights"),
    dtypes=(frozenset({"bfloat16", "float16", "float32"}), frozenset({"float32"})),
)


def _check(request: KernelRequest) -> tuple[str, ...]:
    if len(request.tensors) != 2:
        return ("expected activation and weight metadata",)
    out, weights = request.tensors
    reasons = []
    if len(out.shape) != 3 or len(weights.shape) != 2 or out.shape[:2] != weights.shape:
        reasons.append("expected activations [T,K,H] and weights [T,K]")
    if len(out.shape) == 3 and out.shape[2] <= 0:
        reasons.append("hidden dimension must be positive")
    if out.device != weights.device:
        reasons.append("inputs must share a device")
    return tuple(reasons)


def _check_triton(request: KernelRequest) -> tuple[str, ...]:
    reasons = _check(request)
    if request.tensors and request.tensors[0].device.split(":")[0] != "cuda":
        reasons += ("CUDA-resident inputs required (including torch HIP devices)",)
    return reasons


MOE_COMBINE_REGISTRY = KernelRegistry((
    KernelImplementation(
        operation="moe_weighted_sum", name="triton", backend="triton",
        priority=Priority.PERFORMANT,
        entrypoint="vkernels.torch_ops.moe_combine:_moe_weighted_sum_triton",
        check=_check_triton, graph_capture=True, signature=_MOE_COMBINE_SIGNATURE,
    ),
    KernelImplementation(
        operation="moe_weighted_sum", name="torch_reference", backend="torch",
        entrypoint="vkernels.torch_ops.moe_combine:moe_weighted_sum_reference",
        check=_check, reference=True, signature=_MOE_COMBINE_SIGNATURE,
    ),
))
