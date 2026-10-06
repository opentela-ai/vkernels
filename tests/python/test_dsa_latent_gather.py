"""Copy-only gather parity, bit payloads, capture reuse and eligibility."""

import pytest
import torch

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.dsa_latent_gather import dsa_latent_gather


def reference(latent, idx):
    return latent[torch.arange(latent.shape[0], device=latent.device)[:, None, None], idx.clamp(min=0)]


def test_cpu_declines():
    with pytest.raises(OpNotEligible, match="CUDA"):
        dsa_latent_gather(torch.zeros(1, 5, 32), torch.zeros(1, 3, 7, dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA gather")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 19, 35, 512), (1, 128, 131, 512), (1, 7, 9, 31)])
def test_exact_bytes_with_duplicates_and_negative_ids(dtype, shape):
    batch, seq, width, dim = shape
    torch.manual_seed(413)
    latent = torch.randn(batch, 73, dim, device="cuda", dtype=dtype)
    idx = torch.randint(0, 73, (batch, seq, width), device="cuda", dtype=torch.int64)
    idx[..., ::3] = -17
    idx[..., 2] = idx[..., 1]
    out = dsa_latent_gather(latent, idx)
    expected = reference(latent, idx)
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA gather")
def test_nan_payloads_signed_zero_and_changed_capture_inputs():
    payload = torch.tensor([0, -32768, 32704, 32641, 16256, -16512, 1, -1], dtype=torch.int16, device="cuda")
    latent = payload.repeat(1, 13, 64).view(torch.bfloat16)
    idx = torch.randint(0, 13, (1, 9, 11), device="cuda", dtype=torch.int32)
    dsa_latent_gather(latent, idx)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = dsa_latent_gather(latent, idx)
    torch.cuda.current_stream().wait_stream(stream)
    for step in range(3):
        latent.view(torch.int16).add_(step)
        idx.copy_(torch.randint(-1, 13, idx.shape, device="cuda", dtype=idx.dtype))
        graph.replay()
        assert torch.equal(out.view(torch.uint8), reference(latent, idx).view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA gather")
def test_noncontiguous_declines_and_empty_output():
    latent = torch.zeros(2, 5, 32, device="cuda", dtype=torch.bfloat16)
    idx = torch.zeros(2, 3, 7, device="cuda", dtype=torch.int32)
    with pytest.raises(OpNotEligible, match="contiguous"):
        dsa_latent_gather(latent, idx[:, :2])
    assert dsa_latent_gather(latent, idx[:, :0]).shape == (2, 0, 7, 32)


@pytest.mark.skipif(torch.cuda.device_count() < 2 or torch.version.hip is not None, reason="two CUDA devices")
def test_noncurrent_device_owns_launch_and_restores_caller():
    previous = torch.cuda.current_device()
    with torch.cuda.device(0):
        latent = torch.randn(2, 73, 512, device="cuda:1", dtype=torch.bfloat16)
        idx = torch.randint(-1, 73, (2, 19, 35), device="cuda:1", dtype=torch.int32)
        out = dsa_latent_gather(latent, idx)
        assert out.device == latent.device
        assert torch.cuda.current_device() == 0
        assert torch.equal(out.view(torch.uint8), reference(latent, idx).view(torch.uint8))
    assert torch.cuda.current_device() == previous
