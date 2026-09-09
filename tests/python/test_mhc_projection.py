"""CPU contract checks and optional device-native projection validation."""

import importlib
import subprocess
import sys

import pytest


def test_import_does_not_load_torch_or_triton():
    subprocess.run([sys.executable, "-c", "import sys; import vkernels.torch_ops; "
                    "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("shape", [(16384,), (1, 16384), (1, 2, 16384)])
def test_cpu_reference(torch, shape):
    from vkernels.torch_ops import mhc_projection_reference
    x = torch.ones(shape, dtype=torch.bfloat16)
    w = torch.full((24, 16384), 1 / 16384, dtype=torch.bfloat16)
    y = mhc_projection_reference(x, w)
    assert y.shape == (*shape[:-1], 24)
    assert y.dtype == torch.bfloat16
    assert torch.equal(y, torch.ones_like(y))


def test_input_contract(torch):
    from vkernels.torch_ops import mhc_projection, mhc_projection_reference
    x = torch.ones((1, 16384), dtype=torch.bfloat16)
    w = torch.ones((24, 16384), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="GPU"):
        mhc_projection(x, w)
    with pytest.raises(TypeError, match="BF16"):
        mhc_projection_reference(x.float(), w)
    with pytest.raises(ValueError, match="one or two"):
        mhc_projection_reference(x.expand(3, -1), w)
    with pytest.raises(ValueError, match="contiguous"):
        mhc_projection_reference(x, w.T.contiguous().T)
    with pytest.raises(ValueError, match="expected"):
        mhc_projection_reference(x[:, :8192], w)


@pytest.mark.parametrize("tokens", [1, 2])
def test_gpu_projection_and_graph(torch, tokens):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops import mhc_projection, mhc_projection_reference
    torch.manual_seed(71)
    x = torch.randn((1, tokens, 16384), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((24, 16384), device="cuda", dtype=torch.bfloat16) / 128
    x_before, w_before = x.clone(), w.clone()
    expected = mhc_projection_reference(x, w)
    eager = mhc_projection(x, w)
    torch.testing.assert_close(eager, expected, rtol=0.008, atol=0.008)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mhc_projection(x, w)
    graph.replay()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    x.copy_(x_before * 0.5)
    graph.replay()
    torch.testing.assert_close(captured, mhc_projection_reference(x, w), rtol=0.008, atol=0.008)
    assert torch.equal(w, w_before)
    assert torch.equal(x, x_before * 0.5)


def test_gpu_zero_and_cancellation(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops import mhc_projection, mhc_projection_reference

    x = torch.zeros((2, 16384), device="cuda", dtype=torch.bfloat16)
    w = torch.ones((24, 16384), device="cuda", dtype=torch.bfloat16)
    assert torch.equal(mhc_projection(x, w), torch.zeros((2, 24), device="cuda", dtype=torch.bfloat16))
    x.fill_(1)
    w[:, 1::2] = -1
    # Exact cancellation, followed by a small exactly representable residual.
    for last in (-1.0, -0.9921875):
        w[:, -1] = last
        actual = mhc_projection(x, w)
        expected = mhc_projection_reference(x, w)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, torch.full_like(actual, 1 + last), rtol=0, atol=0)


def test_gpu_cold_capture_guard(torch, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    module = importlib.import_module("vkernels.torch_ops.mhc_projection")
    x = torch.ones((1, 16384), device="cuda", dtype=torch.bfloat16)
    w = torch.ones((24, 16384), device="cuda", dtype=torch.bfloat16)
    # Simulate only the capture predicate: intentionally raising inside a real
    # capture could invalidate its stream and contaminate subsequent GPU tests.
    monkeypatch.setattr(module, "_WARMED", set())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warm up.*eagerly"):
        module.mhc_projection(x, w)
    assert module._WARMED == set()


def test_gpu_noncurrent_device_restored(torch):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    pytest.importorskip("triton")
    from vkernels.torch_ops import mhc_projection, mhc_projection_reference

    original = torch.cuda.current_device()
    x = torch.ones((1, 16384), device="cuda:1", dtype=torch.bfloat16)
    w = torch.full((24, 16384), 1 / 16384, device="cuda:1", dtype=torch.bfloat16)
    with torch.cuda.device(0):
        expected = mhc_projection_reference(x, w)
        actual = mhc_projection(x, w)
        assert actual.device == torch.device("cuda:1")
        assert torch.cuda.current_device() == 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.cuda.current_device() == original
