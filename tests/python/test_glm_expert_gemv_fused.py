"""Fused-epilogue expert GEMV (silu(gate)*up in-kernel) contract + parity.

Three tiers, mirroring tests/python/test_glm_expert_gemv.py:

* contract checks on CPU (``OpNotEligible`` envelope, opt-in knob default);
* GPU **bit-exact** parity against the unfused chain it replaces
  (``expert_gemv`` bf16 store + ``elementwise.silu_mul``) — the fused
  epilogue rounds the gate/up dots to bf16 in registers exactly where the
  two-kernel chain stores/reloads, so equality is exact, NaN included;
* GPU parity vs the composed gather-dequant reference at the repo's
  3e-3 relative gate, plus a CUDA-graph capture/replay check.
"""

import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run([sys.executable, "-c",
                    "import sys; import vkernels.torch_ops.glm_expert_gemv_fused; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'triton' not in sys.modules"], check=True)


def test_silu_fused_default_off(torch, monkeypatch):
    from vkernels.torch_ops.glm_expert_gemv_fused import silu_fused_enabled

    monkeypatch.delenv("GLM53_MOE_SILU_FUSED", raising=False)
    assert silu_fused_enabled() is False          # old path stays default
    monkeypatch.setenv("GLM53_MOE_SILU_FUSED", "1")
    assert silu_fused_enabled() is True


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def make_inputs(torch, e=8, ia=256, i=256, t=1, k=8, seed=7, device="cpu"):
    """Stacked gate/up block-FP8 experts: weights [E, 2*IA, I], x[T,...,I].

    IA (act width) and I (input width) are independent — the GLM serving
    shape is [E, 4096, 4096] with IA=2048, I=4096."""
    torch.manual_seed(seed)
    raw = torch.randint(0, 256, (e, 2 * ia, i), dtype=torch.uint8)
    raw.masked_fill_((raw & 127) == 127, 0)          # drop NaN encodings
    weights = raw.view(torch.float8_e4m3fn)
    scales = 0.0005 + torch.rand((e, 2 * ia // 128, i // 128)) * 0.001
    indices = torch.rand((t, e)).topk(k, dim=-1).indices.to(torch.int64)
    return weights, scales, indices


def unfused_chain(torch, x, weights, scales, indices, storage="e4m3fn"):
    """The two-launch path the fused kernel replaces."""
    from vkernels.torch_ops.elementwise import silu_mul
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv

    gu = expert_gemv(x, weights, scales, indices, storage=storage)
    ia = weights.shape[1] // 2
    return silu_mul(gu[..., :ia], gu[..., ia:])


@pytest.mark.parametrize("shape", [(t, k) for t in (1, 2) for k in (1, 8)])
@pytest.mark.parametrize("broadcast", [True, False])
def test_gpu_bitexact_vs_unfused_chain(torch, shape, broadcast):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_expert_gemv_fused import expert_gemv_silu

    t, k = shape
    weights, scales, indices = make_inputs(torch, t=t, k=k)
    weights, scales, indices = (v.to("cuda") for v in (weights, scales, indices))
    width = 256 if broadcast else k
    x = torch.randn((t, width, 256), device="cuda", dtype=torch.bfloat16)
    if broadcast:
        x = x[:, 0, :].contiguous()               # x[T, I] (contract: contiguous)
    actual = expert_gemv_silu(x, weights, scales, indices)
    expected = unfused_chain(torch, x, weights, scales, indices)
    assert actual.shape == (t, k, 256)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    # The headline gate: NOT a tolerance — bit-identical to the chain.
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("ia,i", [(128, 256), (192, 256), (256, 512)])
def test_gpu_bitexact_asymmetric(torch, ia, i):
    """IA != I (the GLM serving shape is IA=2048, I=4096) and an IA that
    straddles a 128 scale-row boundary mid-tile (IA=192: the up rows' scale
    row-block (row+IA)//128 is NOT IA//128 + row//128)."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_expert_gemv_fused import expert_gemv_silu

    t, k = 2, 8
    weights, scales, indices = make_inputs(torch, ia=ia, i=i, t=t, k=k)
    weights, scales, indices = (v.to("cuda") for v in (weights, scales, indices))
    x = torch.randn((t, k, i), device="cuda", dtype=torch.bfloat16)
    actual = expert_gemv_silu(x, weights, scales, indices)
    assert actual.shape == (t, k, ia)
    assert torch.equal(actual, unfused_chain(torch, x, weights, scales, indices))


def test_gpu_bitexact_fnuz_storage(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_expert_gemv_fused import expert_gemv_silu
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import e4m3fn_to_fnuz_inplace

    weights, scales, indices = make_inputs(torch, t=2, k=8)
    weights, scales, indices = (v.to("cuda") for v in (weights, scales, indices))
    x = torch.randn((2, 8, 256), device="cuda", dtype=torch.bfloat16)
    # PRISTINE copies first: e4m3fn_to_fnuz_inplace rewrites bytes and
    # DOUBLES scales in place, so clones taken afterwards are not e4m3fn.
    weights_e4, scales_e4 = weights.clone(), scales.clone()
    fnuz_w, fnuz_s = e4m3fn_to_fnuz_inplace(weights, scales)
    actual = expert_gemv_silu(x, fnuz_w, fnuz_s, indices, storage="e4m3fnuz")
    expected = unfused_chain(torch, x, fnuz_w, fnuz_s, indices,
                             storage="e4m3fnuz")
    assert torch.equal(actual, expected)
    # fnuz storage is lossless vs the original e4m3fn weights (the halving
    # is an exact exponent decrement; scales double exactly): the same
    # activation must also match the plain-storage chain.
    assert torch.equal(actual, unfused_chain(torch, x, weights_e4, scales_e4,
                                             indices))
    assert fnuz_w.data_ptr() == weights.data_ptr()  # sanity: in-place


def test_gpu_parity_vs_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_expert_gemv_fused import (
        expert_gemv_silu, expert_gemv_silu_reference)

    weights, scales, indices = make_inputs(torch, e=32, i=256, t=2, k=8)
    weights, scales, indices = (v.to("cuda") for v in (weights, scales, indices))
    x = torch.randn((2, 8, 256), device="cuda", dtype=torch.bfloat16)
    actual = expert_gemv_silu(x, weights, scales, indices)
    expected = expert_gemv_silu_reference(x, weights, scales, indices)
    assert actual.dtype == torch.bfloat16
    diff = actual.float() - expected.float()
    rel = (diff.norm() / expected.float().norm().clamp_min(1e-12)).item()
    # Secondary gate (the PRIMARY contract is the bit-exact chain equality
    # above): the silu product can locally amplify the GEMV's bf16-level
    # dot difference where the activation is near zero, so the composed
    # reference is held to the same 3e-3 norm gate on a full expert pool.
    assert rel < 3e-3, rel


def test_gpu_graph_capture_replay(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_expert_gemv_fused import expert_gemv_silu

    weights, scales, indices = make_inputs(torch, t=1, k=8)
    weights, scales, indices = (v.to("cuda") for v in (weights, scales, indices))
    x = torch.randn((1, 8, 256), device="cuda", dtype=torch.bfloat16)
    expert_gemv_silu(x, weights, scales, indices)      # eager warmup per shape
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        static_out = expert_gemv_silu(x, weights, scales, indices)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        static_out = expert_gemv_silu(x, weights, scales, indices)

    # The captured graph reads the STATIC buffers: feed new values in
    # place, replay, and compare against the chain on the new values.
    x2 = torch.randn_like(x) / 2.0
    idx2 = torch.randint(0, weights.shape[0], (1, 8),
                         device="cuda", dtype=torch.int64)
    x.copy_(x2)
    indices.copy_(idx2)
    graph.replay()
    torch.cuda.synchronize()
    expected = unfused_chain(torch, x2, weights, scales, idx2)
    assert torch.equal(static_out, expected)


def test_contract(torch):
    from vkernels.torch_ops.glm_expert_gemv_fused import expert_gemv_silu

    # Pin the T<=cap contract against the DEFAULT cap (2).
    weights, scales, indices = make_inputs(torch, t=1, k=8)
    x = torch.randn(1, 8, 256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="GPU"):
        expert_gemv_silu(x, weights, scales, indices)
    with pytest.raises(ValueError, match="T<=2"):
        expert_gemv_silu(x[:3].expand(3, 8, 256), weights, scales,
                         torch.randint(0, 8, (3, 8), dtype=torch.int64))
    with pytest.raises(TypeError, match="BF16"):
        expert_gemv_silu(x.float(), weights, scales, indices)
    with pytest.raises(ValueError, match="expected scales"):
        expert_gemv_silu(x, weights, scales[:, :, :-1], indices)
    with pytest.raises(ValueError, match="x\\[T,I\\]"):
        expert_gemv_silu(torch.randn(1, 128, dtype=torch.bfloat16),
                         weights, scales, indices)
    with pytest.raises(ValueError, match="stacked gate/up"):
        # 448 rows: not a multiple of 128 -> not a block-128 stacked
        # gate/up stack (any even O multiple of 128 IS a valid stack).
        expert_gemv_silu(x, weights[:, :-64].contiguous(),
                         scales[:, :-1].contiguous(), indices)
    with pytest.raises(ValueError, match="unknown weight storage"):
        expert_gemv_silu(x, weights, scales, indices, storage="e5m2")
