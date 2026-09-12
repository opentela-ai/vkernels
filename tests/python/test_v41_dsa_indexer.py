"""DSA indexer scores: reference oracle (naive-loop cross-check) + GPU parity."""

import importlib.util
import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [sys.executable, "-c",
         "import sys; import vkernels.torch_ops.v41_dsa_indexer as m; assert 'torch' not in sys.modules"],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_reference_concrete(torch):
    from vkernels.torch_ops.v41_dsa_indexer import indexer_scores_reference

    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # [B=1,S=1,H=2,D=2]
    k = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])     # [B=1,T=2,D=2]
    w = torch.tensor([[[1.0, 1.0]]])                 # [B=1,S=1,H=2]
    out = indexer_scores_reference(q, k, w)          # [1,1,2]
    # t0: relu(h0.t0=1)+relu(h1.t0=0)=1 ; t1: relu(h0.t1=1)+relu(h1.t1=1)=2
    assert out.flatten().tolist() == [1.0, 2.0]


def test_reference_relu_zeroes_negative_dots(torch):
    from vkernels.torch_ops.v41_dsa_indexer import indexer_scores_reference

    q = torch.tensor([[[[1.0, 0.0]]]])   # h0 = [1,0]
    k = torch.tensor([[[-1.0, 0.0]]])    # t0 = [-1,0] -> dot -1 -> relu 0
    w = torch.tensor([[[2.0]]])
    assert indexer_scores_reference(q, k, w).item() == 0.0


def test_reference_matches_naive_loop(torch):
    from vkernels.torch_ops.v41_dsa_indexer import indexer_scores_reference

    torch.manual_seed(0)
    B, S, H, D, T = 2, 3, 4, 8, 5
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, T, D)
    w = torch.randn(B, S, H)
    got = indexer_scores_reference(q, k, w)
    exp = torch.zeros(B, S, T)
    for b in range(B):
        for s in range(S):
            for t in range(T):
                acc = 0.0
                for h in range(H):
                    dot = (q[b, s, h] * k[b, t]).sum()
                    acc += w[b, s, h] * torch.clamp(dot, min=0.0)
                exp[b, s, t] = acc
    assert torch.allclose(got, exp, atol=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton required")
def test_gpu_matches_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.v41_dsa_indexer import indexer_scores, indexer_scores_reference

    torch.manual_seed(1)
    B, S, H, D, T = 2, 4, 32, 128, 320  # real-ish indexer geometry
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, T, D)
    w = torch.randn(B, S, H)
    ref = indexer_scores_reference(q, k, w)
    got = indexer_scores(q.cuda(), k.cuda(), w.cuda()).cpu()
    assert torch.allclose(got, ref, atol=1e-2, rtol=1e-3)

    # regression: non-pow-2 geometry must fall back to the reference, not
    # crash in tl.arange (H/D aranges need power-of-two sizes). Compare against
    # the reference on the SAME device: cross-device bitwise equality (GPU
    # cuBLAS vs CPU) is not guaranteed even when every printed digit matches.
    B, S, H, D, T = 2, 2, 32, 96, 64  # D=96: not a power of two
    q, k = torch.randn(B, S, H, D), torch.randn(B, T, D)
    w = torch.randn(B, S, H)
    got = indexer_scores(q.cuda(), k.cuda(), w.cuda()).cpu()
    ref = indexer_scores_reference(q.cuda(), k.cuda(), w.cuda()).cpu()
    assert torch.equal(got, ref)


def test_wrapper_falls_back_on_cpu(torch):
    from vkernels.torch_ops.v41_dsa_indexer import indexer_scores, indexer_scores_reference

    torch.manual_seed(3)
    B, S, H, D, T = 2, 3, 4, 8, 5
    q, k = torch.randn(B, S, H, D), torch.randn(B, T, D)
    w = torch.randn(B, S, H)
    # CPU wrapper routes to the exact reference (the "always correct" promise)
    got = indexer_scores(q, k, w)
    assert torch.equal(got, indexer_scores_reference(q, k, w))


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton required")
def test_gpu_triton_backend_matches_reference(torch, monkeypatch):
    """The fused Triton kernel is off the hot path by default (it loses to the
    cuBLAS einsum at decode shapes, ~6x on GB10); it stays reachable via
    VKERNELS_DSA_INDEXER_BACKEND=triton and must keep matching the oracle."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.v41_dsa_indexer import indexer_scores, indexer_scores_reference

    monkeypatch.setenv("VKERNELS_DSA_INDEXER_BACKEND", "triton")
    torch.manual_seed(2)
    B, S, H, D, T = 2, 4, 32, 128, 320  # real-ish indexer geometry
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, T, D)
    w = torch.randn(B, S, H)
    ref = indexer_scores_reference(q, k, w)
    got = indexer_scores(q.cuda(), k.cuda(), w.cuda()).cpu()
    assert torch.allclose(got, ref, atol=1e-2, rtol=1e-3)

