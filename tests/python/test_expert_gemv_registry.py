"""Actual native/portable expert GEMV dispatch and offline capability checks."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vkernels.registry import KernelRequest, KernelSelectionError, TensorMetadata
from vkernels.torch_ops.expert_gemv_registry import EXPERT_GEMV_REGISTRY


def _request(dtype="float8_e4m3fn", backends=frozenset({"triton", "triton_cuda"})):
    return KernelRequest("expert_gemv", (
        TensorMetadata((2, 128), "bfloat16", "cuda:0"),
        TensorMetadata((4, 128, 128), dtype, "cuda:0"),
        TensorMetadata((4, 1, 1), "float32", "cuda:0"),
        TensorMetadata((2, 2), "int64", "cuda:0"),
    ), backends=backends, graph_capture=True)


def test_native_and_portable_format_selection():
    assert EXPERT_GEMV_REGISTRY.select(_request()).name == "cuda_native"
    assert EXPERT_GEMV_REGISTRY.select(_request(backends=frozenset({"triton"}))).name == "portable"
    # fnuz storage routes to its dedicated variant, which declares the
    # one-time load-time weight transform.
    selected = EXPERT_GEMV_REGISTRY.select(_request("float8_e4m3fnuz"))
    assert selected.name == "portable_fnuz"
    assert selected.weight_preprocessor == (
        "vkernels.torch_ops.glm_fp8_blockwise_gemm:e4m3fn_to_fnuz_inplace"
    )
    assert EXPERT_GEMV_REGISTRY.select(_request(), override="portable").name == "portable"
    with pytest.raises(KernelSelectionError, match="requires E4M3FN storage"):
        EXPERT_GEMV_REGISTRY.select(_request("float8_e4m3fnuz"), override="cuda_native")
    with pytest.raises(KernelSelectionError, match="unavailable"):
        EXPERT_GEMV_REGISTRY.select(_request(backends=frozenset({"triton"})), override="cuda_native")
    with pytest.raises(KernelSelectionError, match="contiguous"):
        request = _request()
        EXPERT_GEMV_REGISTRY.select(replace(request, tensors=(replace(request.tensors[0], contiguous=False), *request.tensors[1:])))


def test_weight_preprocessor_resolves_to_the_inplace_conversion():
    selected = EXPERT_GEMV_REGISTRY.select(_request("float8_e4m3fnuz"))
    preprocess = selected.load_weight_preprocessor()
    assert preprocess.__name__ == "e4m3fn_to_fnuz_inplace"
    # Selection itself stays free of the transform: impls without one
    # resolve to None (cuda_native decodes FN bytes in hardware).
    assert EXPERT_GEMV_REGISTRY.select(_request()).load_weight_preprocessor() is None


def test_offline_registry_and_operator_import_stay_lazy():
    script = """
import sys
class BlockGPU:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton'}:
            raise AssertionError('unexpected GPU import: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
from vkernels.torch_ops.expert_gemv_registry import EXPERT_GEMV_REGISTRY
from vkernels.torch_ops.glm_expert_gemv import expert_gemv
assert not {'torch', 'triton'}.intersection(sys.modules)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src/python"))
    # Disable installed editable finders so this offline probe imports the
    # checkout named by PYTHONPATH, even inside another serving worktree's venv.
    subprocess.run([sys.executable, "-S", "-c", script], check=True, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("invalid,reason", [
    ("shape", "expected weights"),
    ("device", "GPU device"),
    ("dtype", "BF16 activations"),
    ("storage", "unknown weight storage"),
])
def test_invalid_override_contract_cannot_silently_fallback(invalid, reason):
    torch = pytest.importorskip("torch")
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv

    x = torch.ones((1, 128), dtype=torch.bfloat16)
    weights = torch.zeros((1, 128, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
    scales = torch.ones((1, 1, 1))
    indices = torch.zeros((1, 1), dtype=torch.int64)
    kwargs = {}
    if invalid == "shape":
        weights = weights[0]
    elif invalid == "dtype":
        x = x.float()
    elif invalid == "storage":
        kwargs["storage"] = "bad-storage"
    # Every case is CPU-only. The device case reaches device validation with
    # otherwise valid inputs; the other cases fail earlier for their own cause.
    with pytest.raises(OpNotEligible, match=reason):
        expert_gemv(x, weights, scales, indices, **kwargs)
    with pytest.raises(KernelSelectionError, match=reason) as caught:
        expert_gemv(x, weights, scales, indices, implementation="portable", **kwargs)
    assert not isinstance(caught.value, OpNotEligible)


@pytest.mark.parametrize("implementation", ["cuda_native", "portable"])
def test_real_cuda_implementations_reference_and_graph_replay(implementation):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("CUDA parity test")
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv, expert_gemv_reference
    torch.manual_seed(391)
    x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    weights = (torch.randn((4, 128, 128), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    scales = torch.full((4, 1, 1), 0.05, device="cuda")
    indices = torch.tensor([[0, 2], [1, 3]], device="cuda")
    expected = expert_gemv_reference(x, weights, scales, indices)
    actual = expert_gemv(x, weights, scales, indices, implementation=implementation)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-3)
    with pytest.raises(KernelSelectionError, match="unknown implementation"):
        expert_gemv(x, weights, scales, indices, implementation="missing")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            expert_gemv(x, weights, scales, indices, implementation=implementation)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = expert_gemv(x, weights, scales, indices, implementation=implementation)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(captured, expert_gemv_reference(x, weights, scales, indices), rtol=0.01, atol=1e-3)


def test_fnuz_native_override_fails_before_launch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv
    x = torch.ones((1, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.zeros((1, 128, 128), device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fnuz)
    scales = torch.ones((1, 1, 1), device="cuda")
    indices = torch.zeros((1, 1), device="cuda", dtype=torch.int64)
    with pytest.raises(KernelSelectionError, match="requires E4M3FN storage"):
        expert_gemv(x, weights, scales, indices, storage="e4m3fnuz", implementation="cuda_native")
    assert torch.count_nonzero(expert_gemv(x, weights, scales, indices, storage="e4m3fnuz")) == 0


def test_zero_route_dispatch_survives_registry_integration_and_changed_graph_inputs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("CUDA-native zero-route contract")
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv, expert_gemv_skip_zero

    torch.manual_seed(492)
    x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    weights = (torch.randn((4, 128, 128), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    scales = torch.full((4, 1, 1), 0.05, device="cuda")
    indices = torch.tensor([[0, 2], [1, 3]], device="cuda")
    routes = torch.tensor([[0.0, 0.25], [1.0, 0.0]], device="cuda")

    def expected():
        ordinary = expert_gemv(x, weights, scales, indices)
        return torch.where(routes[..., None] == 0, 0, ordinary)

    actual = expert_gemv_skip_zero(x, weights, scales, indices, routes)
    torch.testing.assert_close(actual, expected(), rtol=0, atol=0)
    with pytest.raises(KernelSelectionError, match="cuda_native"):
        expert_gemv(x, weights, scales, indices, implementation="portable", route_weights=routes)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expert_gemv_skip_zero(x, weights, scales, indices, routes)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = expert_gemv_skip_zero(x, weights, scales, indices, routes)
    routes.copy_(torch.tensor([[0.5, 0.0], [0.0, 0.75]], device="cuda"))
    x.normal_()
    graph.replay()
    torch.testing.assert_close(captured, expected(), rtol=0, atol=0)
