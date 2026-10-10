"""Metadata-only selection for the actual selected-expert FP8 GEMV paths."""

from vkernels.registry import KernelImplementation, KernelRegistry, KernelRequest, Priority
from vkernels.signature import FormatSignature

# The shared per-role dtype contract: BF16 activations, FP8 block-scaled
# weights (either storage), FP32 scales, int64 expert indices.
_EXPERT_GEMV_SIGNATURES = FormatSignature(
    roles=("x", "weights", "scales", "indices"),
    dtypes=(
        frozenset({"bfloat16"}),
        frozenset({"float8_e4m3fn", "float8_e4m3fnuz"}),
        frozenset({"float32"}),
        frozenset({"int64"}),
    ),
)


def _check(request: KernelRequest) -> tuple[str, ...]:
    if len(request.tensors) != 4:
        return ("expected activations, weights, scales and expert indices",)
    x, weights, scales, indices = request.tensors
    if len(weights.shape) != 3 or len(indices.shape) != 2:
        return ("expected weights [E,O,I] and indices [T,K]",)
    e, o, i = weights.shape
    t, k = indices.shape
    cap = max(2, int(dict(request.parameters).get("t_cap", 2)))
    reasons = []
    if min(e, o, i) <= 0 or o % 128 or i % 128 or t > cap:
        reasons.append(f"requires positive block-128 dimensions and T<={cap}")
    if x.shape not in ((t, i), (t, k, i)) or scales.shape != (e, o // 128, i // 128):
        reasons.append("activation or block-scale shape mismatch")
    if any(tensor.device != x.device for tensor in request.tensors):
        reasons.append("inputs must share a device")
    if any(not tensor.contiguous for tensor in request.tensors):
        reasons.append("inputs must be contiguous")
    return tuple(reasons)


def _gpu(request: KernelRequest) -> tuple[str, ...]:
    reasons = _check(request)
    if request.tensors and request.tensors[0].device.split(":")[0] != "cuda":
        reasons += ("GPU inputs required",)
    return reasons


def _native(request: KernelRequest) -> tuple[str, ...]:
    reasons = _gpu(request)
    if len(request.tensors) == 4 and request.tensors[1].dtype != "float8_e4m3fn":
        reasons += ("native CUDA decoding requires E4M3FN storage",)
    return reasons


def _reference(request: KernelRequest) -> tuple[str, ...]:
    reasons = _check(request)
    if len(request.tensors) == 4 and request.tensors[1].dtype != "float8_e4m3fn":
        reasons += ("reference oracle requires E4M3FN storage",)
    return reasons


def _check_fnuz(request: KernelRequest) -> tuple[str, ...]:
    reasons = _gpu(request)
    if len(request.tensors) == 4 and request.tensors[1].dtype != "float8_e4m3fnuz":
        reasons += ("fnuz variant requires in-place converted E4M3FNUZ storage",)
    return reasons


EXPERT_GEMV_REGISTRY = KernelRegistry((
    KernelImplementation(
        operation="expert_gemv", name="cuda_native", backend="triton_cuda",
        priority=Priority.SPECIALIZED,
        entrypoint="vkernels.torch_ops.glm_expert_gemv:_launch_native",
        check=_native, graph_capture=True, signature=_EXPERT_GEMV_SIGNATURES,
    ),
    KernelImplementation(
        operation="expert_gemv", name="portable", backend="triton",
        priority=Priority.PORTABLE,
        entrypoint="vkernels.torch_ops.glm_expert_gemv:_launch_portable",
        check=_gpu, graph_capture=True, signature=_EXPERT_GEMV_SIGNATURES,
    ),
    # gfx942 serving: the checkpoint's e4m3fn bytes are rewritten in place
    # (payloads halved, scales doubled) once at load — declared here so the
    # loading side can discover the transform from the selection instead
    # of hard-coding it next to the model code.
    KernelImplementation(
        operation="expert_gemv", name="portable_fnuz", backend="triton",
        priority=Priority.PORTABLE + 1,
        entrypoint="vkernels.torch_ops.glm_expert_gemv:_launch_portable",
        check=_check_fnuz, graph_capture=True, signature=_EXPERT_GEMV_SIGNATURES,
        weight_preprocessor=(
            "vkernels.torch_ops.glm_fp8_blockwise_gemm:e4m3fn_to_fnuz_inplace"
        ),
    ),
    KernelImplementation(
        operation="expert_gemv", name="torch_reference", backend="torch",
        entrypoint="vkernels.torch_ops.glm_expert_gemv:expert_gemv_reference",
        check=_reference, reference=True,
    ),
))
