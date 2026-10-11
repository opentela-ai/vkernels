"""Metadata-only selection for the vendored TokenSpeed-Kernel-AMD ops.

Planning uses the same metadata-only registry as the rest of torch_ops:
capability checks are pure functions of :class:`~vkernels.registry.KernelRequest`
metadata, the dtype contract is a :class:`~vkernels.signature.FormatSignature`,
and loading the launcher (which imports torch/gluon) happens only at execution
time. Kernels: :mod:`vkernels.torch_ops.tokenspeed_amd` (vendored from
tokenspeed-kernel-amd, MIT).

Host-capability discovery follows the same rule as the Triton MLA/DSA ports:
the registry gates on tensor metadata only (GPU residency, ranks, dtypes);
whether Gluon — and, for the CDNA4 kernels, ``gluon.language.amd`` — actually
loads is discovered at launch, where the vendored launchers fail loudly.
"""

from vkernels.registry import KernelImplementation, KernelRegistry, KernelRequest, Priority
from vkernels.signature import FormatSignature

__all__ = [
    "ROUTER_TOPK_REGISTRY",
    "MOE_SORTING_REGISTRY",
    "SAMPLING_ARGMAX_REGISTRY",
    "HADAMARD_TRANSFORM_REGISTRY",
    "MHC_PRE_REGISTRY",
    "TOKENSPEED_AMD_REGISTRY",
]

_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_GPU_DEVICES = ("cuda", "hip", "rocm")


def _gpu_check(request: KernelRequest) -> tuple[str, ...]:
    reasons = []
    for tensor in request.tensors:
        if tensor.device.split(":")[0] not in _GPU_DEVICES:
            reasons.append("GPU-resident tensors required")
            break
    return tuple(reasons)


def _router_check(request: KernelRequest) -> tuple[str, ...]:
    reasons = list(_gpu_check(request))
    if request.tensors:
        logits = request.tensors[0]
        if len(logits.shape) != 2:
            reasons.append("router logits must be [M, E]")
        # Single-CTA kernel limits: E <= 1024 and M*topk <= 128 (the sigmoid
        # prefill variant lifts the M limit but needs CDNA4 Triton).
        elif not 0 < logits.shape[1] <= 1024:
            reasons.append("num_experts must be in (0, 1024]")
    return tuple(reasons)


ROUTER_TOPK_SIGNATURE = FormatSignature(
    roles=("logits",),
    dtypes=(_FLOAT_DTYPES,),
    layout="dense",
)

ROUTER_TOPK_REGISTRY = KernelRegistry((
    KernelImplementation(
        operation="router_topk", name="gluon-softmax", backend="gluon",
        priority=Priority.PORTABLE,
        entrypoint="vkernels.torch_ops.tokenspeed_amd.routing:invoke_softmax_topk_route_gluon",
        check=_router_check, graph_capture=True, signature=ROUTER_TOPK_SIGNATURE,
    ),
    KernelImplementation(
        operation="router_topk", name="gluon-sigmoid-bias", backend="gluon",
        priority=Priority.PORTABLE,
        entrypoint="vkernels.torch_ops.tokenspeed_amd.routing:invoke_sigmoid_bias_topk_route_gluon",
        check=_router_check, graph_capture=True, signature=ROUTER_TOPK_SIGNATURE,
    ),
))


def _sorting_check(request: KernelRequest) -> tuple[str, ...]:
    reasons = list(_gpu_check(request))
    if len(request.tensors) != 2:
        reasons.append("moe sorting needs (topk_ids, topk_weights)")
    else:
        ids, weights = request.tensors
        if ids.shape != weights.shape or len(ids.shape) != 2:
            reasons.append("topk_ids/topk_weights must be matching [M, TOPK]")
    return tuple(reasons)


MOE_SORTING_SIGNATURE = FormatSignature(
    roles=("topk_ids", "topk_weights"),
    dtypes=(frozenset({"int32"}), frozenset({"float32"})),
    layout="dense",
)

MOE_SORTING_REGISTRY = KernelRegistry((
    KernelImplementation(
        operation="moe_sorting", name="gluon", backend="gluon",
        priority=Priority.PORTABLE,
        entrypoint="vkernels.torch_ops.tokenspeed_amd.moe_sorting:gluon_moe_sorting",
        check=_sorting_check, graph_capture=True, signature=MOE_SORTING_SIGNATURE,
    ),
))


def _argmax_check(request: KernelRequest) -> tuple[str, ...]:
    reasons = list(_gpu_check(request))
    if request.tensors:
        logits = request.tensors[0]
        if len(logits.shape) != 2:
            reasons.append("argmax logits must be [M, N]")
    return tuple(reasons)


SAMPLING_ARGMAX_SIGNATURE = FormatSignature(
    roles=("logits",),
    dtypes=(_FLOAT_DTYPES,),
    layout="dense",
)

SAMPLING_ARGMAX_REGISTRY = KernelRegistry((
    KernelImplementation(
        operation="sampling_argmax", name="gluon", backend="gluon",
        priority=Priority.PORTABLE,
        entrypoint="vkernels.torch_ops.tokenspeed_amd.argmax:launch_gluon_argmax_gfx950",
        check=_argmax_check, graph_capture=True, signature=SAMPLING_ARGMAX_SIGNATURE,
    ),
))


def _hadamard_check(request: KernelRequest) -> tuple[str, ...]:
    reasons = list(_gpu_check(request))
    if request.tensors:
        x = request.tensors[0]
        if not x.shape or x.shape[-1] != 128:
            reasons.append("hadamard_128 requires contiguous length-128 rows")
    return tuple(reasons)


HADAMARD_TRANSFORM_SIGNATURE = FormatSignature(
    roles=("x",),
    dtypes=(frozenset({"bfloat16", "float16", "float32"}),),
    layout="dense",
)

HADAMARD_TRANSFORM_REGISTRY = KernelRegistry((
    KernelImplementation(
        operation="hadamard_transform", name="gluon-128", backend="gluon",
        priority=Priority.PORTABLE,
        entrypoint="vkernels.torch_ops.tokenspeed_amd.hadamard:launch_gluon_hadamard_transform_128_gfx950",
        check=_hadamard_check, graph_capture=True, signature=HADAMARD_TRANSFORM_SIGNATURE,
    ),
))


def _mhc_check(request: KernelRequest) -> tuple[str, ...]:
    reasons = list(_gpu_check(request))
    if len(request.tensors) >= 4:
        residual, layer_input = request.tensors[2], request.tensors[3]
        if len(residual.shape) != 3 or residual.shape[1] != 4:
            reasons.append("mhc residual must be [tokens, 4, hidden]")
        elif residual.shape[2] not in (4096, 7168):
            reasons.append("mhc gluon specialization requires hidden 4096 or 7168")
        elif residual.shape[0] > 64:
            reasons.append("mhc gluon specialization is decode-shaped (<=64 tokens)")
        if len(layer_input.shape) != 2:
            reasons.append("mhc layer_input must be [tokens, hidden]")
    return tuple(reasons)


MHC_PRE_SIGNATURE = FormatSignature(
    roles=("gemm_out_mul", "gemm_out_sqrsum", "residual", "layer_input"),
    dtypes=(_FLOAT_DTYPES, _FLOAT_DTYPES, _FLOAT_DTYPES, frozenset({"bfloat16"})),
    layout="dense",
)

MHC_PRE_REGISTRY = KernelRegistry((
    KernelImplementation(
        operation="mhc_pre", name="gluon", backend="gluon",
        priority=Priority.PORTABLE,
        entrypoint="vkernels.torch_ops.tokenspeed_amd.mhc:gluon_mhc_pre_reduce_apply_gfx950",
        check=_mhc_check, graph_capture=True, signature=MHC_PRE_SIGNATURE,
    ),
))


def _all_impls():
    return (
        *ROUTER_TOPK_REGISTRY.implementations,
        *MOE_SORTING_REGISTRY.implementations,
        *SAMPLING_ARGMAX_REGISTRY.implementations,
        *HADAMARD_TRANSFORM_REGISTRY.implementations,
        *MHC_PRE_REGISTRY.implementations,
    )


#: Combined view for planning tools that want every tokenspeed-amd
#: registration at once.
TOKENSPEED_AMD_REGISTRY = KernelRegistry(_all_impls())
