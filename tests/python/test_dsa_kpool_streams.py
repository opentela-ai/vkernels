"""Ordering and allocation lifetime at the native kpool boundary."""

import gc

import pytest

torch = pytest.importorskip("torch")

from vkernels import dsa_kpool_device as dev  # noqa: E402


def _inputs(decode, fp8, device="cuda"):
    # Noncontiguous float32 keys/scores and int64 indices force preparation.
    key = torch.arange(256, device=device, dtype=torch.float32).reshape(128, 2).T / 100
    score = torch.zeros_like(key)
    tail_k = torch.zeros((2, 2, 128), device=device, dtype=torch.bfloat16)
    tail_score = torch.zeros_like(tail_k)
    ape = torch.zeros((1, 128), device=device)
    out = torch.zeros((2, 264) if fp8 else (2, 2, 128), device=device,
                      dtype=torch.uint8 if fp8 else torch.bfloat16)
    def index(values):
        return torch.tensor(values, device=device, dtype=torch.int64)
    if decode:
        indices = [index([[0], [1]]), index([0, 1]), index([1, 1]),
                   index([2, 2]), index([1, 3])]
    else:
        indices = [index([0, 1]), index([0, 0]), index([0, 1]),
                   index([0, 0]), index([1, 3])]
    return [out, key, score, tail_k, tail_score, ape, *indices]


def _operation(decode, fp8):
    name = "dsa_kpool_decode_update" if decode else "dsa_kpool_assemble"
    return getattr(dev, name + ("_fp8" if fp8 else ""))


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_cpu_tensors_rejected_before_native_call(monkeypatch, decode, fp8):
    def unexpected():
        pytest.fail("native library must not be called for CPU tensors")
    monkeypatch.setattr(dev, "load_libvkernels", unexpected)
    with pytest.raises(ValueError, match="CUDA/HIP"):
        _operation(decode, fp8)(*_inputs(decode, fp8, device="cpu"))


@pytest.mark.parametrize("shape,fp8,expected", [
    ((2, 16, 128), False, 16), ((2, 2048), False, 16), ((2, 2112), True, 16),
])
def test_page_geometry(shape, fp8, expected):
    out = torch.empty(shape, dtype=torch.uint8 if fp8 else torch.bfloat16)
    assert dev._slots_per_page(out, 128, fp8=fp8) == expected


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


@gpu
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("raw_stream", [False, True])
def test_explicit_stream_orders_conversions_and_retains_inputs(decode, fp8, raw_stream):
    if not dev.available():
        pytest.skip("native kpool library required")
    operation = _operation(decode, fp8)
    expected = _inputs(decode, fp8)
    kwargs = {"round_scale": True} if fp8 else {}
    operation(*expected, **kwargs)
    torch.cuda.synchronize()
    expected_out = expected[0].clone()
    expected_tail = expected[3].clone()
    assert torch.count_nonzero(expected_out).item() > 0

    launch = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    # Ensure input creation is pending when the wrapper switches streams.
    torch.cuda._sleep(5_000_000)
    actual = _inputs(decode, fp8)
    output, tail = actual[0], actual[3]
    with torch.cuda.stream(launch):
        torch.cuda._sleep(10_000_000)
    operation(*actual, stream=launch.cuda_stream if raw_stream else launch, **kwargs)
    assert torch.cuda.current_stream() == current
    del actual
    gc.collect()
    # Reuse the source allocator's size classes while launch is still queued.
    churn = [torch.empty((256,), device="cuda", dtype=torch.float32).fill_(-99)
             for _ in range(64)]
    current.wait_stream(launch)
    torch.testing.assert_close(output, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(tail, expected_tail, rtol=0, atol=0)
    del churn


@gpu
@pytest.mark.parametrize("fp8", [False, True])
def test_graph_replay_after_stream_warmup(fp8):
    if not dev.available():
        pytest.skip("native kpool library required")
    operation = _operation(False, fp8)
    args = _inputs(False, fp8)
    stream = torch.cuda.Stream()
    kwargs = {"round_scale": True} if fp8 else {}
    operation(*args, stream=stream, **kwargs)
    stream.synchronize()
    expected = args[0].clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        operation(*args, stream=stream, **kwargs)
    for _ in range(3):
        args[0].zero_()
        graph.replay()
        torch.testing.assert_close(args[0], expected, rtol=0, atol=0)


@gpu
def test_mixed_devices_rejected():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    args = _inputs(False, False)
    args[1] = args[1].to("cuda:1")
    with pytest.raises(ValueError, match="one GPU"):
        dev.dsa_kpool_assemble(*args)
