"""Decode QKV contract checks and optional GPU parity/capture checks."""

import importlib
import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run([sys.executable, "-c", "import sys; import vkernels.torch_ops.qkv_projection; "
                    "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("shape", [(4096,), (1, 4096), (1, 2, 4096)])
def test_reference_shape_order(torch, shape):
    from vkernels.torch_ops.qkv_projection import qkv_projection_reference

    x = torch.ones(shape, dtype=torch.bfloat16)
    weights = [torch.full((8192, 4096), scale / 4096, dtype=torch.bfloat16) for scale in (1, 2, 3)]
    actual = qkv_projection_reference(x, *weights)
    assert actual.shape == (*shape[:-1], 24576)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    for expected, part in enumerate(actual.split(8192, dim=-1), 1):
        assert torch.equal(part, torch.full_like(part, expected))


def test_contract(torch):
    from vkernels.torch_ops.qkv_projection import qkv_projection, qkv_projection_reference

    x = torch.ones((1, 4096), dtype=torch.bfloat16)
    w = torch.ones((8192, 4096), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="GPU"):
        qkv_projection(x, w, w, w)
    with pytest.raises(TypeError, match="BF16"):
        qkv_projection_reference(x.float(), w, w, w)
    with pytest.raises(ValueError, match="one or two"):
        qkv_projection_reference(x.expand(3, -1), w, w, w)
    with pytest.raises(ValueError, match="contiguous"):
        qkv_projection_reference(x, w, w.T.contiguous().T, w)
    with pytest.raises(ValueError, match="expected"):
        qkv_projection_reference(x, w, w, w[:4096])
    with pytest.raises(ValueError, match="share a device"):
        qkv_projection_reference(x, w, w, torch.empty_like(w, device="meta"))


@pytest.mark.parametrize("tokens", [1, 2])
def test_gpu_parity_graph_and_readonly(torch, tokens):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.qkv_projection import qkv_projection, qkv_projection_reference

    torch.manual_seed(173)
    x = torch.randn((1, tokens, 4096), device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16) / 64 for _ in range(3)]
    before = [tensor.clone() for tensor in (x, *weights)]
    eager = qkv_projection(x, *weights)
    torch.testing.assert_close(eager, qkv_projection_reference(x, *weights), rtol=0.008, atol=0.008)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qkv_projection(x, *weights)
    graph.replay()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    for original, tensor in zip(before, (x, *weights)):
        assert torch.equal(original, tensor)
    x.copy_(before[0] * 0.5)
    graph.replay()
    torch.testing.assert_close(captured, qkv_projection_reference(x, *weights), rtol=0.008, atol=0.008)


def test_gpu_zero_cancellation_and_projection_order(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.qkv_projection import qkv_projection

    x = torch.zeros((2, 4096), device="cuda", dtype=torch.bfloat16)
    weights = [torch.ones((8192, 4096), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    assert torch.count_nonzero(qkv_projection(x, *weights)).item() == 0
    x.fill_(1)
    for weight, residual in zip(weights, (0, 0.0078125, 0.015625)):
        weight[:, 1::2] = -1
        weight[:, -1] = -1 + residual
    result = qkv_projection(x, *weights)
    for part, residual in zip(result.split(8192, dim=-1), (0, 0.0078125, 0.015625)):
        torch.testing.assert_close(part, torch.full_like(part, residual), rtol=0, atol=0)


def test_gpu_cold_capture_guard(torch, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    module = importlib.import_module("vkernels.torch_ops.qkv_projection")
    x = torch.ones((1, 4096), device="cuda", dtype=torch.bfloat16)
    w = torch.ones((8192, 4096), device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(module, "_WARMED", set())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warm up.*eagerly"):
        module.qkv_projection(x, w, w, w)
    assert module._WARMED == set()


def test_gpu_device_guard(torch):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    pytest.importorskip("triton")
    from vkernels.torch_ops.qkv_projection import qkv_projection

    original = torch.cuda.current_device()
    x = torch.ones((1, 4096), device="cuda:1", dtype=torch.bfloat16)
    w = torch.full((8192, 4096), 1 / 4096, device="cuda:1", dtype=torch.bfloat16)
    with torch.cuda.device(0):
        result = qkv_projection(x, w, w, w)
        assert result.device == torch.device("cuda:1")
        assert torch.cuda.current_device() == 0
        torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)
    assert torch.cuda.current_device() == original
