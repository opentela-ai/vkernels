"""Preserve original cache-off decode arithmetic across stores and graph replay."""
import pytest

torch = pytest.importorskip("torch")


def same_bits(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def reject_tuning(*args, **kwargs):
    raise AssertionError("A numerically different reduction layout must not be tuned")


@pytest.mark.parametrize("shape,default,hip_default", [
    ((6144, 4096), (4, 4096, 4), (1, 4096, 2)),
    ((2048, 128), (8, 128, 8), (4, 128, 8)),
    ((24, 16384), (1, 16384, 8), (1, 16384, 8)),
])
def test_gemv_original_layout_ignores_stores_and_stale_proposals(monkeypatch, tmp_path, shape, default, hip_default):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    pytest.importorskip("triton")
    from vkernels.torch_ops import glm_gemv as gemv
    from vkernels.tuning import cache

    o, i = shape
    torch.manual_seed(1123)
    x = torch.randn(1, i, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(o, i, device="cuda", dtype=torch.bfloat16) * .01
    before = [x.clone(), w.clone()]
    expected = torch.empty(1, o, device="cuda", dtype=torch.bfloat16)
    rows, block, warps = hip_default if torch.version.hip else default
    kernel = gemv._kernel()[2]

    def control():
        kernel[((o + rows - 1) // rows,)](x, w, expected, o, i, rows, block,
                                         INDEPENDENT_ROWS=False, num_warps=warps)
        return expected

    monkeypatch.setattr(gemv, "_GEMV_CFG", {shape: (1, 4096, 8)})
    monkeypatch.setattr(cache, "op_config", reject_tuning)
    for store in ("off", str(tmp_path)):
        monkeypatch.setenv("VKERNELS_CACHE", store)
        assert same_bits(gemv.dense_gemv(x, w), control())
        assert gemv._GEMV_CFG[shape] == (rows, block, warps)
        assert all(same_bits(a, b) for a, b in zip(before, (x, w), strict=True))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gemv.dense_gemv(x, w)
    x.copy_(x.roll(1, 1) * .75)
    changed = control().clone()
    graph.replay()
    assert same_bits(actual, changed)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("out_fp32", [False, True])
@pytest.mark.parametrize("batch", [1, 4])
def test_kda_original_layout_ignores_stores_and_stale_proposals(monkeypatch, tmp_path, dtype, out_fp32, batch):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    import triton
    from vkernels.torch_ops import glm_kda_decode as kda
    from vkernels.tuning import cache

    torch.manual_seed(417)
    h, d = 16, 128
    q, k, v = [torch.randn(batch, 1, h, d, device="cuda", dtype=dtype) for _ in range(3)]
    g = -torch.rand_like(q)
    beta = torch.rand(batch, 1, h, device="cuda", dtype=dtype)
    state = torch.randn(batch, h, d, d, device="cuda") * .1
    args = (q, k, v, g, beta, state)
    before = [x.clone() for x in args]
    expected = torch.empty_like(q, dtype=torch.float32 if out_fp32 else dtype)
    next_state = torch.empty_like(state)

    def control():
        kda._kernel()[(batch * h, triton.cdiv(d, 32))](
            *args, expected, next_state, d, 1e-6, 32,
            num_warps=4, enable_fp_fusion=False,
        )
        return expected, next_state

    monkeypatch.setattr(kda, "_KDA_CFG", {(batch * h, d): (16, 4)})
    monkeypatch.setattr(cache, "op_config", reject_tuning)
    for store in ("off", str(tmp_path)):
        monkeypatch.setenv("VKERNELS_CACHE", store)
        actual = kda.kda_decode(*args, out_fp32=out_fp32)
        assert all(same_bits(a, b) for a, b in zip(actual, control(), strict=True))
        assert kda._KDA_CFG[(batch * h, d)] == (32, 4)
        assert all(same_bits(a, b) for a, b in zip(before, args, strict=True))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = kda.kda_decode(*args, out_fp32=out_fp32)
    q.copy_(q.roll(1, -1) * .75)
    changed = [x.clone() for x in control()]
    graph.replay()
    assert all(same_bits(a, b) for a, b in zip(actual, changed, strict=True))
