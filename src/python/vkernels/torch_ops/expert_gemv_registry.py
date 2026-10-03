"""Metadata-only selection for the actual selected-expert FP8 GEMV paths."""

from vkernels.registry import KernelImplementation, KernelRegistry, KernelRequest


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
    if (x.dtype != "bfloat16" or weights.dtype not in ("float8_e4m3fn", "float8_e4m3fnuz")
            or scales.dtype != "float32" or indices.dtype != "int64"):
        reasons.append("requires BF16 activations, E4M3FN/FNUZ weights, FP32 scales and int64 indices")
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


EXPERT_GEMV_REGISTRY = KernelRegistry((
    KernelImplementation(
        operation="expert_gemv", name="cuda_native", backend="triton_cuda", priority=100,
        entrypoint="vkernels.torch_ops.glm_expert_gemv:_launch_native",
        check=_native, graph_capture=True,
    ),
    KernelImplementation(
        operation="expert_gemv", name="portable", backend="triton",
        entrypoint="vkernels.torch_ops.glm_expert_gemv:_launch_portable",
        check=_gpu, graph_capture=True,
    ),
    KernelImplementation(
        operation="expert_gemv", name="torch_reference", backend="torch",
        entrypoint="vkernels.torch_ops.glm_expert_gemv:expert_gemv_reference",
        check=_reference, reference=True,
    ),
))
