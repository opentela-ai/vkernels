"""V4.1 sparse attention with sink: reference oracle + naive check + GPU parity."""

import importlib.util
import math
import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [sys.executable, "-c",
         "import sys; import vkernels.torch_ops.v41_sparse_attention as m; assert 'torch' not in sys.modules"],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_reference_concrete_with_sink(torch):
    from vkernels.torch_ops.v41_sparse_attention import sparse_attention_reference

    q = torch.tensor([[[[1.0, 0.0]]]])   # [B=1,H=1,S=1,D=2]
    kv = torch.tensor([[[1.0, 0.0]]])    # [B=1,N=1,D=2]
    mask = torch.tensor([[[1.0]]])       # keep the single key
    sink = torch.tensor([0.0])           # sink logit 0
    out = sparse_attention_reference(q, kv, mask, sink, scale=1.0)
    # softmax([score=1, sink=0]) -> key prob e/(e+1); output = that * kv[0]
    w = math.e / (math.e + 1.0)
    assert torch.allclose(out.flatten(), torch.tensor([w, 0.0]), atol=1e-5)


def test_reference_masked_key_is_dropped(torch):
    from vkernels.torch_ops.v41_sparse_attention import sparse_attention_reference

    q = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[5.0, 5.0]]])
    mask = torch.tensor([[[0.0]]])       # drop the only key -> all weight to sink -> zero output
    sink = torch.tensor([0.0])
    out = sparse_attention_reference(q, kv, mask, sink, scale=1.0)
    assert torch.allclose(out.flatten(), torch.zeros(2), atol=1e-6)


def test_reference_matches_naive(torch):
    from vkernels.torch_ops.v41_sparse_attention import sparse_attention_reference

    torch.manual_seed(0)
    B, H, S, N, D = 2, 3, 4, 6, 8
    q = torch.randn(B, H, S, D)
    kv = torch.randn(B, N, D)
    mask = (torch.rand(B, S, N) > 0.3).float()
    sink = torch.randn(H)
    scale = 1.0 / math.sqrt(D)
    got = sparse_attention_reference(q, kv, mask, sink, scale)
    exp = torch.zeros(B, H, S, D)
    for b in range(B):
        for h in range(H):
            for s in range(S):
                logits = []
                for n in range(N):
                    sc = (q[b, h, s] * kv[b, n]).sum() * scale if mask[b, s, n] > 0 else torch.tensor(-1e30)
                    logits.append(sc)
                logits.append(sink[h])  # sink column
                probs = torch.softmax(torch.stack(logits), dim=0)
                exp[b, h, s] = sum(probs[n] * kv[b, n] for n in range(N))
    assert torch.allclose(got, exp, atol=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton required")
def test_gpu_matches_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.v41_sparse_attention import sparse_attention, sparse_attention_reference

    torch.manual_seed(1)
    B, H, S, N, D = 2, 8, 4, 192, 128
    q = torch.randn(B, H, S, D)
    kv = torch.randn(B, N, D)
    mask = (torch.rand(B, S, N) > 0.5).float()
    sink = torch.randn(H)
    scale = 1.0 / math.sqrt(D)
    ref = sparse_attention_reference(q, kv, mask, sink, scale)
    got = sparse_attention(q.cuda(), kv.cuda(), mask.cuda(), sink.cuda(), scale).cpu()
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3)



def test_wrapper_falls_back_on_cpu(torch):
    from vkernels.torch_ops.v41_sparse_attention import sparse_attention, sparse_attention_reference

    torch.manual_seed(2)
    B, H, S, N, D = 2, 3, 4, 6, 8
    q, kv = torch.randn(B, H, S, D), torch.randn(B, N, D)
    mask = (torch.rand(B, S, N) > 0.3).float()
    sink = torch.randn(H)
    scale = 1.0 / math.sqrt(D)
    got = sparse_attention(q, kv, mask, sink, scale)  # CPU -> reference
    assert torch.equal(got, sparse_attention_reference(q, kv, mask, sink, scale))


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton required")
def test_gpu_backend_gate_and_prefill(torch):
    """Shape-gated dispatch (measured GB10 crossover): decode-sized S takes the
    Triton kernel, prefill-sized S takes the batched-cuBLAS reference — both
    must match the oracle; forced-triton stays correct at prefill too."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import os

    from vkernels.torch_ops.v41_sparse_attention import sparse_attention, sparse_attention_reference

    torch.manual_seed(2)
    H, D, N = 64, 512, 640
    for s in (1, 64):
        q = torch.randn(1, H, s, D, device="cuda", dtype=torch.bfloat16) * 0.1
        kv = torch.randn(1, N, D, device="cuda", dtype=torch.bfloat16) * 0.1
        mask = (torch.rand(1, s, N, device="cuda") < 0.5).float()
        sink = torch.rand(H, device="cuda")
        ref = sparse_attention_reference(q, kv, mask, sink, 0.088)
        auto = sparse_attention(q, kv, mask, sink, 0.088)
        assert torch.allclose(auto, ref, atol=1e-2, rtol=1e-2), s
        os.environ["VKERNELS_V41_SPARSE_ATTN_BACKEND"] = "triton"
        try:
            forced = sparse_attention(q, kv, mask, sink, 0.088)
        finally:
            os.environ.pop("VKERNELS_V41_SPARSE_ATTN_BACKEND", None)
        assert torch.allclose(forced, ref, atol=1e-2, rtol=1e-2), s
        if s == 1:
            assert torch.equal(auto, forced), s  # auto picked the kernel

