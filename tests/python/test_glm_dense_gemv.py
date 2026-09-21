"""Tests for the dense BF16 decode GEMV (torch_ops.glm_gemv).

The op is GPU-gated: on CUDA it must match ``F.linear`` within the
fp32-reduction-order tolerance at the real GLM-5.3-Flash decode shapes
(and exactly at tiny sizes where the sum order cannot differ), and it
must raise ``OpNotEligible`` — not corrupt — on every contract miss the
caller's fallback relies on. CPU boxes skip the kernel tests.
"""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402
from vkernels.torch_ops.glm_gemv import dense_gemv  # noqa: E402

# The real per-rank TP4 decode shapes (loader output, bench 3460776).
REAL_SHAPES = [
    (2048, 4096),  # kda q/k/v
    (4096, 2048),  # kda o
    (16, 4096),  # kda b_proj
    (128, 4096),  # kda g_a / indexer wk
    (2048, 128),  # kda g_b / forget f_b
    (1536, 4096),  # dsa q_a
    (4096, 1536),  # dsa q_b / indexer wq_b
    (512, 4096),  # dsa kv_a / shared gate+up
    (8192, 512),  # dsa kv_b
    (4096, 512),  # shared down
    (3072, 4096),  # dense mlp gate/up
    (4096, 3072),  # dense mlp down
    (32, 4096),  # indexer weights_proj
]


def _cuda():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda(), reason="kernel dispatch is CUDA-gated")
@pytest.mark.parametrize("o,i", REAL_SHAPES)
def test_parity_vs_linear_real_shapes(o, i):
    """Graph-replay parity at the shapes the serve will capture."""
    gen = torch.Generator(device="cuda").manual_seed(o * 7 + i)
    w = (torch.randn(o, i, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    x = (torch.randn(1, i, generator=gen, device="cuda")).to(torch.bfloat16)
    y = dense_gemv(x, w)
    ref = torch.nn.functional.linear(x, w)
    # fp32 accumulation both sides; only the sum order can differ
    assert torch.allclose(y.float(), ref.float(), atol=2e-2, rtol=0), (
        (y.float() - ref.float()).abs().max().item()
    )


@pytest.mark.skipif(not _cuda(), reason="kernel dispatch is CUDA-gated")
def test_parity_exact_small_no_tail():
    """BLOCK_I == I fast path, O % 4 == 0 — full tile, no masking."""
    w = torch.eye(8, device="cuda", dtype=torch.bfloat16)
    x = torch.tensor([[1.0, 2, 3, 4, 5, 6, 7, 8]], device="cuda", dtype=torch.bfloat16)
    y = dense_gemv(x, w)
    assert torch.equal(y, x)


@pytest.mark.skipif(not _cuda(), reason="kernel dispatch is CUDA-gated")
def test_tail_rows_and_cols():
    """O % 4 != 0 (tail program) and non-pow2 I (masked K tail) both exact."""
    w = torch.eye(10, 12, device="cuda", dtype=torch.bfloat16)
    x = torch.arange(12, device="cuda", dtype=torch.bfloat16)[None]
    y = dense_gemv(x, w)
    assert torch.equal(y, x[:, :10])
    w2 = torch.randn(2050, 3000, device="cuda", dtype=torch.bfloat16) * 0.02
    x2 = torch.randn(1, 3000, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x2, w2)
    assert torch.allclose(dense_gemv(x2, w2).float(), ref.float(), atol=2e-2, rtol=0)


@pytest.mark.skipif(not _cuda(), reason="kernel dispatch is CUDA-gated")
def test_1d_activation_returns_1d():
    w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    y = dense_gemv(x, w)
    assert y.shape == (64,)
    ref = torch.nn.functional.linear(x[None], w)[0]
    assert torch.allclose(y.float(), ref.float(), atol=2e-2, rtol=0)


@pytest.mark.skipif(not _cuda(), reason="kernel dispatch is CUDA-gated")
def test_graph_capture_two_plays():
    """Capture-safe: warm up, capture two plays, replay — values stable."""
    w = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w)
    dense_gemv(x, w)  # JIT warmup outside capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        dense_gemv(x, w)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y1 = dense_gemv(x, w)
        y2 = dense_gemv(x, w)
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(y1.float(), ref.float(), atol=2e-2, rtol=0)
    assert torch.equal(y1, y2)


@pytest.mark.skipif(not _cuda(), reason="kernel dispatch is CUDA-gated")
def test_contract_misses_raise_not_corrupt():
    w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(OpNotEligible):  # M > 1
        dense_gemv(torch.randn(2, 128, device="cuda", dtype=torch.bfloat16), w)
    with pytest.raises(OpNotEligible):  # dtype
        dense_gemv(x.float(), w)
    with pytest.raises(OpNotEligible):  # non-contiguous weight
        dense_gemv(x, w.t())
    with pytest.raises(OpNotEligible):  # shape mismatch
        dense_gemv(x, torch.randn(64, 256, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(OpNotEligible):  # cpu tensor
        dense_gemv(x.cpu(), w.cpu())


def test_cpu_rejection_everywhere():
    """No CUDA at all -> OpNotEligible (caller falls back to BLAS)."""
    if _cuda():
        pytest.skip("CUDA present; the GPU-gated tests cover this")
    w = torch.randn(64, 128, dtype=torch.bfloat16)
    x = torch.randn(1, 128, dtype=torch.bfloat16)
    with pytest.raises(OpNotEligible):
        dense_gemv(x, w)
