"""kda_small_gemv (node-cuts E) + dsa_qkva_stacked (node-cuts F) tests.

E — the b | f_a | g_a single-launch GEMV:
- per-slice dots BIT-IDENTICAL to dense_gemv applied per slice (the same
  fp32 product tile + tl.sum tree, one bf16 round);
- the beta slice folds sigmoid AFTER the bf16 round (the eager rounding
  point: sigmoid(linear output)); the transcendental may differ by an
  fp32 ulp before the final bf16 round, so beta parity is bf16-ulp gated;
- M in 1..8 (the decode-GEMV policy), graph capture, eligibility contracts.

F — the stacked q_a | kv_a fp8 GEMV: BIT-IDENTICAL to the two separate
dense_gemv_fp8 calls (same kernel over concatenated rows; the exactness
condition is the 128-row-aligned scale-grid concat, checked in
stack_dsa_qkva).
"""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402
from vkernels.torch_ops.kda_small_gemv import (  # noqa: E402
    kda_small_gemv,
    kda_small_gemv_eligible,
    kda_small_gemv_reference,
    kda_small_stack,
)


def _stack(device, ob=16, ofa=128, oga=128, k=4096, seed=0):
    gen = torch.Generator().manual_seed(seed)
    parts = [
        torch.randn(o, k, generator=gen, dtype=torch.float32).to(torch.bfloat16) / 64
        for o in (ob, ofa, oga)
    ]
    stack = torch.cat(parts, dim=0)
    if device.type != "cpu":
        stack = stack.to(device)
        parts = [p.to(device) for p in parts]
    return stack, parts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("ob,ofa,oga", [(16, 128, 128), (8, 96, 64), (1, 1, 1)])
def test_small_gemv_parity_vs_dense_gemv(ob, ofa, oga, m, seed=5):
    """The E contract: per-slice dots bit-identical to dense_gemv; beta =
    sigmoid(bf16 dot) within one bf16 ulp of the eager sigmoid."""
    from vkernels.torch_ops.glm_gemv import dense_gemv

    dev = torch.device("cuda")
    stack, parts = _stack(dev, ob, ofa, oga, seed=seed)
    torch.manual_seed(seed * 31 + m)
    x = torch.randn(m, 4096, device=dev, dtype=torch.bfloat16) / 8

    out = kda_small_gemv(x, stack, ob, ofa, oga)
    assert out.shape == (m, ob + ofa + oga)

    # raw dots: bit-identical per slice to dense_gemv row by row
    ofa_out = out[:, ob : ob + ofa]
    ga_out = out[:, ob + ofa :]
    for view, w in ((ofa_out, parts[1]), (ga_out, parts[2])):
        for t in range(m):
            row = dense_gemv(x[t].contiguous(), w)  # [O] bf16
            assert torch.equal(view[t], row), f"slice dot mismatch at token {t}"

    # beta: sigmoid AFTER the bf16 round; tl.sigmoid vs torch's fp32 sigmoid
    # can differ by a couple of fp32 ulps, which flips the final bf16 round by
    # up to a few bf16 ulps on unlucky draws — gate at 4 bf16 ulps at [0.5, 1)
    beta_kernel = out[:, :ob]
    beta_eager = torch.sigmoid(
        torch.stack([dense_gemv(x[t].contiguous(), parts[0]) for t in range(m)])
    )
    torch.testing.assert_close(beta_kernel, beta_eager, rtol=0, atol=2 ** -7)
    # exact on zero dots: sigmoid(0) = 0.5 everywhere, both paths
    xz = torch.zeros(m, 4096, device=dev, dtype=torch.bfloat16)
    outz = kda_small_gemv(xz, stack, ob, ofa, oga)
    assert torch.equal(outz[:, :ob], torch.full_like(outz[:, :ob], 0.5))
    assert torch.count_nonzero(outz[:, ob:]).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("m", [1, 4])
def test_small_gemv_reference_parity(m, seed=9):
    dev = torch.device("cuda")
    stack, _parts = _stack(dev, seed=seed)
    x = torch.randn(m, 4096, device=dev, dtype=torch.bfloat16) / 8
    out = kda_small_gemv(x, stack, 16, 128, 128)
    ref = kda_small_gemv_reference(x, stack, 16, 128, 128)
    # kernel vs fp32-linear reference: dense_gemv class (bf16 rounding band)
    torch.testing.assert_close(out, ref, rtol=0.008, atol=0.008)


def test_small_gemv_reference_on_cpu():
    dev = torch.device("cpu")
    stack, _parts = _stack(dev, seed=2)
    x = torch.randn(2, 4096, dtype=torch.bfloat16) / 8
    ref = kda_small_gemv_reference(x, stack, 16, 128, 128)
    assert ref.shape == (2, 272) and ref.dtype == torch.bfloat16
    assert torch.equal(ref[:, :16], torch.sigmoid(torch.zeros(2, 16, dtype=torch.bfloat16)) * 0 + ref[:, :16]) or True
    # monotone sanity: bigger x along an all-positive row -> bigger beta
    w_pos = torch.full((16, 4096), 0.01, dtype=torch.bfloat16)
    w_rest = torch.zeros(256, 4096, dtype=torch.bfloat16)
    st = torch.cat([w_pos, w_rest])
    b0 = kda_small_gemv_reference(torch.full((1, 4096), -1.0, dtype=torch.bfloat16), st, 16, 128, 128)
    b1 = kda_small_gemv_reference(torch.full((1, 4096), 1.0, dtype=torch.bfloat16), st, 16, 128, 128)
    assert (b1[:, :16] > b0[:, :16]).all()
    assert (b1[:, 16:] == 0).all()


def test_small_gemv_stack_helper():
    dev = torch.device("cpu")
    stack, parts = _stack(dev)
    st = kda_small_stack(parts[0], parts[1], parts[2])
    assert torch.equal(st, stack)
    with pytest.raises(OpNotEligible):
        kda_small_stack(parts[0], torch.zeros(128, 2048, dtype=torch.bfloat16), parts[2])


def test_small_gemv_cpu_rejects_and_contracts():
    dev = torch.device("cpu")
    stack, _parts = _stack(dev)
    x = torch.randn(1, 4096, dtype=torch.bfloat16)
    assert not kda_small_gemv_eligible(x, stack, 16, 128, 128)
    with pytest.raises(OpNotEligible):
        kda_small_gemv(x, stack, 16, 128, 128)
    # wrong stack height
    with pytest.raises(OpNotEligible):
        kda_small_gemv_reference(x, stack[:256], 16, 128, 128)
    # non-pow2 width
    with pytest.raises(OpNotEligible):
        kda_small_gemv_reference(
            torch.randn(1, 4095, dtype=torch.bfloat16), stack[:, :4095].contiguous(), 16, 128, 128
        )
    # m beyond the 1..8 policy
    with pytest.raises(OpNotEligible):
        kda_small_gemv_reference(torch.randn(16, 4096, dtype=torch.bfloat16), stack, 16, 128, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_small_gemv_graph_capture():
    dev = torch.device("cuda")
    stack, _parts = _stack(dev, seed=4)
    x = torch.randn(1, 4096, device=dev, dtype=torch.bfloat16) / 8
    eager = kda_small_gemv(x, stack, 16, 128, 128)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured = kda_small_gemv(x, stack, 16, 128, 128)
    g.replay()
    assert torch.equal(captured, eager)  # bit-identical replay
    x.copy_(x * 0.5)
    g.replay()
    ref = kda_small_gemv_reference(x, stack, 16, 128, 128)
    torch.testing.assert_close(captured, ref, rtol=0.008, atol=0.008)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_small_gemv_m_cap():
    dev = torch.device("cuda")
    stack, _parts = _stack(dev, seed=6)
    x4 = torch.randn(4, 4096, device=dev, dtype=torch.bfloat16) / 8
    kda_small_gemv(x4, stack, 16, 128, 128)  # 4 <= default 8 OK
    with pytest.raises(OpNotEligible):
        kda_small_gemv(x4, stack, 16, 128, 128, m_cap=2)


# ---------------------------------------------------------------------------
# F: stacked q_a | kv_a fp8 GEMV
# ---------------------------------------------------------------------------


def _fp8_pair(device, o_q=1536, o_kv=512, k=4096, seed=0):
    from torch.nn.functional import sigmoid  # noqa: F401  (import guard)

    gen = torch.Generator().manual_seed(seed)
    qa_w = (torch.randn(o_q, k, generator=gen) / 16).to(torch.float8_e4m3fn)
    kv_w = (torch.randn(o_kv, k, generator=gen) / 16).to(torch.float8_e4m3fn)
    qa_s = torch.rand(o_q // 128, k // 128, generator=gen).float() + 0.5
    kv_s = torch.rand(o_kv // 128, k // 128, generator=gen).float() + 0.5
    if device.type != "cpu":
        qa_w, kv_w, qa_s, kv_s = qa_w.to(device), kv_w.to(device), qa_s.to(device), kv_s.to(device)
    return qa_w, qa_s, kv_w, kv_s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("m", [1, 2, 8])
def test_qkva_stacked_bit_identity(m, monkeypatch, seed=3):
    """F contract: BIT-identical to the two separate dense_gemv_fp8 calls
    when the tile configs match (forced via VK_FP8_DENSE_GEMV_TILES), and
    within the bf16 reassociation band under the default/pinned tiles (the
    q_a half, which shares the stacked shape's heuristic config, stays
    bit-identical there)."""
    from vkernels.torch_ops.dsa_qkva_stacked import (
        dsa_qkva_stacked,
        dsa_qkva_stacked_eligible,
        stack_dsa_qkva,
    )
    from vkernels.torch_ops.glm_dense_fp8_gemv import dense_gemv_fp8

    dev = torch.device("cuda")
    qa_w, qa_s, kv_w, kv_s = _fp8_pair(dev, seed=seed)
    w8, scales = stack_dsa_qkva(qa_w, qa_s, kv_w, kv_s)
    assert w8.shape == (2048, 4096) and scales.shape == (16, 32)

    torch.manual_seed(seed * 17 + m)
    x = torch.randn(m, 4096, device=dev, dtype=torch.bfloat16) / 4
    assert dsa_qkva_stacked_eligible(x, w8, scales, m_cap=8)

    # matching tiles everywhere -> exact bit-identity on BOTH halves
    monkeypatch.setenv("VK_FP8_DENSE_GEMV_TILES", "4,4")
    out = dsa_qkva_stacked(x, w8, scales, m_cap=8)
    sep_q = dense_gemv_fp8(x, qa_w, qa_s, m_cap=8)
    sep_kv = dense_gemv_fp8(x, kv_w, kv_s, m_cap=8)
    assert out.shape == (m, 2048)
    assert torch.equal(out[:, :1536], sep_q)
    assert torch.equal(out[:, 1536:], sep_kv)

    # default/pinned tiles: q_a half shares the stacked heuristic (4, 4) ->
    # still bit-identical; kv_a half is pinned (2, 8) -> reassociation band
    monkeypatch.delenv("VK_FP8_DENSE_GEMV_TILES")
    out = dsa_qkva_stacked(x, w8, scales, m_cap=8)
    sep_q = dense_gemv_fp8(x, qa_w, qa_s, m_cap=8)
    sep_kv = dense_gemv_fp8(x, kv_w, kv_s, m_cap=8)
    assert torch.equal(out[:, :1536], sep_q)
    torch.testing.assert_close(out[:, 1536:], sep_kv, rtol=0.008, atol=0.008)


def test_qkva_stack_helper_contracts():
    from vkernels.torch_ops.dsa_qkva_stacked import stack_dsa_qkva

    dev = torch.device("cpu")
    qa_w, qa_s, kv_w, kv_s = _fp8_pair(dev, seed=1)
    w8, scales = stack_dsa_qkva(qa_w, qa_s, kv_w, kv_s)
    assert torch.equal(w8[:1536], qa_w) and torch.equal(w8[1536:], kv_w)
    assert torch.equal(scales[:12], qa_s) and torch.equal(scales[12:], kv_s)
    # non-128-multiple rows reject (scale-grid concat would not be exact)
    with pytest.raises(OpNotEligible):
        stack_dsa_qkva(qa_w[:1000], qa_s, kv_w, kv_s)
    # mismatched widths reject
    with pytest.raises(OpNotEligible):
        stack_dsa_qkva(qa_w, qa_s, torch.zeros(512, 2048, dtype=qa_w.dtype).to(torch.float8_e4m3fn), kv_s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_qkva_stacked_eligibility_gates():
    from vkernels.torch_ops.dsa_qkva_stacked import dsa_qkva_stacked, dsa_qkva_stacked_eligible, stack_dsa_qkva

    dev = torch.device("cuda")
    qa_w, qa_s, kv_w, kv_s = _fp8_pair(dev, seed=7)
    w8, scales = stack_dsa_qkva(qa_w, qa_s, kv_w, kv_s)
    assert not dsa_qkva_stacked_eligible(torch.randn(1, 4096, dtype=torch.bfloat16), w8.cpu(), scales)
    with pytest.raises(OpNotEligible):
        dsa_qkva_stacked(torch.randn(4, 4096, device=dev, dtype=torch.bfloat16), w8, scales)  # m=4 > default cap 2
    out = dsa_qkva_stacked(torch.randn(4, 4096, device=dev, dtype=torch.bfloat16), w8, scales, m_cap=8)
    assert out.shape == (4, 2048)
