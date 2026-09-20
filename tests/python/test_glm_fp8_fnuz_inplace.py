"""e4m3fn -> e4m3fnuz IN-PLACE conversion (issue #71).

The grouped native-fp8 path's copy-based conversion cache doubled
expert-weight residency (~9.7 GiB/layer permanent) and its ~19 GiB fp32
transient OOM-swallowed conversions under a fixed HBM floor. The halving
map is a pure per-byte function, so ``e4m3fn_to_fnuz_inplace`` rewrites the
checkpoint bytes in place through a 256-entry LUT: the converted stack
shares the checkpoint's storage (no second resident copy) and no fp32
materialization (peak extra memory is chunk-sized). The fnuz-storage decode
variants (``expert_gemv`` / ``gather_dequant`` with ``storage="e4m3fnuz"``)
consume it with bit-identical values.
"""

import pytest


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _byte_stack(torch, e=4, o=256, i=256, seed=11):
    """Full-range byte pattern: subnormals, both NaN encodings, -0, maxima."""
    g = torch.Generator().manual_seed(seed)
    weights = torch.randint(
        0, 256, (e, o, i), dtype=torch.uint8, generator=g
    ).view(torch.float8_e4m3fn)
    scales = torch.rand((e, o // 128, i // 128), generator=g) * 0.1 + 0.01
    return weights, scales


def _value_stack(torch, e=4, o=256, i=256, seed=5):
    """Realistic NaN-free weights for output-bit-exactness comparisons."""
    g = torch.Generator().manual_seed(seed)
    weights = (
        torch.randn((e, o, i), generator=g).clamp(-4, 4).to(torch.float8_e4m3fn)
    )
    scales = torch.rand((e, o // 128, i // 128), generator=g) * 0.1 + 0.01
    return weights, scales


def test_lut_matches_reference_exhaustive(torch):
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import _fnuz_byte_lut

    lut = _fnuz_byte_lut(torch.device("cpu"))
    raw = torch.arange(256, dtype=torch.uint8)
    fn = raw.view(torch.float8_e4m3fn)
    expected = (fn.to(torch.float32) * 0.5).to(torch.float8_e4m3fnuz).view(torch.uint8)
    assert torch.equal(lut, expected)


def test_halving_is_value_exact(torch):
    """fnuz_decode(LUT[b]) * 2 == fn_decode(b) for every finite byte and NaN
    maps to NaN — consumers multiplying by the DOUBLED scales decode
    bit-identical values to the e4m3fn path."""
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import _fnuz_byte_lut

    lut = _fnuz_byte_lut(torch.device("cpu"))
    raw = torch.arange(256, dtype=torch.uint8)
    fn_val = raw.view(torch.float8_e4m3fn).to(torch.float32)
    fz_val = lut.view(torch.float8_e4m3fnuz).to(torch.float32)
    finite = torch.isfinite(fn_val)
    assert finite.sum() == 254  # only 0x7f/0xff are fn NaN
    assert torch.equal(fz_val[finite] * 2, fn_val[finite])
    assert torch.isnan(fz_val[raw == 0x7F]).all()
    assert fz_val[raw == 0x80] == 0.0  # fn -0 -> fnuz +0 (no -0 in fnuz)


def test_inplace_matches_reference_and_shares_storage(torch):
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
        e4m3fn_to_fnuz,
        e4m3fn_to_fnuz_inplace,
    )

    weights, scales = _byte_stack(torch)
    ref_w, ref_s = e4m3fn_to_fnuz(weights.clone(), scales.clone())
    ptr = weights.data_ptr()
    w_fnuz, s2 = e4m3fn_to_fnuz_inplace(weights, scales)
    assert w_fnuz.data_ptr() == ptr  # same storage — nothing extra resident
    assert w_fnuz.dtype == torch.float8_e4m3fnuz
    assert s2 is scales  # doubled in place
    assert torch.equal(w_fnuz.view(torch.uint8), ref_w.view(torch.uint8))
    assert torch.equal(s2, ref_s)


def test_chunked_conversion_matches_single_shot(torch):
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import e4m3fn_to_fnuz_inplace

    weights, scales = _byte_stack(torch, seed=3)
    w1, s1 = e4m3fn_to_fnuz_inplace(weights.clone(), scales.clone())
    w2, s2 = e4m3fn_to_fnuz_inplace(
        weights.clone(), scales.clone(), chunk_bytes=997
    )
    assert torch.equal(w1.view(torch.uint8), w2.view(torch.uint8))
    assert torch.equal(s1, s2)


def test_guards(torch):
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import e4m3fn_to_fnuz_inplace

    weights, scales = _byte_stack(torch, e=2)
    with pytest.raises(TypeError, match="already e4m3fnuz"):
        e4m3fn_to_fnuz_inplace(weights.view(torch.float8_e4m3fnuz), scales)
    with pytest.raises(TypeError, match="e4m3fn"):
        e4m3fn_to_fnuz_inplace(weights.to(torch.float32), scales)
    with pytest.raises(ValueError, match="contiguous"):
        e4m3fn_to_fnuz_inplace(weights.transpose(0, 1), scales)


def test_wrapper_storage_validation(torch):
    """The consumer wrappers validate the storage kwarg before anything
    device-specific (so this runs on CPU-only hosts too)."""
    from vkernels.torch_ops.glm_expert_gather_dequant import gather_dequant
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv

    weights, scales = _value_stack(torch, e=2)
    indices = torch.zeros((1, 1), dtype=torch.int64)
    x = torch.randn(1, 256).to(torch.bfloat16)
    with pytest.raises(ValueError, match="unknown weight storage"):
        gather_dequant(weights, scales, indices, storage="fnuz")
    with pytest.raises(ValueError, match="unknown weight storage"):
        expert_gemv(x, weights, scales, indices, storage="fp8")
    with pytest.raises(TypeError, match="E4M3FNUZ"):
        gather_dequant(weights, scales, indices, storage="e4m3fnuz")
    with pytest.raises(TypeError, match="E4M3FNUZ"):
        expert_gemv(x, weights, scales, indices, storage="e4m3fnuz")


def test_device_guard(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires two devices")
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import e4m3fn_to_fnuz_inplace

    weights, scales = _byte_stack(torch, e=2)
    with pytest.raises(ValueError, match="device"):
        e4m3fn_to_fnuz_inplace(weights.to("cuda"), scales)


def test_gpu_inplace_matches_cpu_reference(torch):
    """The production path: the LUT gather runs on-device."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
        e4m3fn_to_fnuz,
        e4m3fn_to_fnuz_inplace,
    )

    weights, scales = _byte_stack(torch)
    ref_w, ref_s = e4m3fn_to_fnuz(weights.clone(), scales.clone())
    gpu_w, gpu_s = weights.to("cuda"), scales.to("cuda")
    w_fnuz, s2 = e4m3fn_to_fnuz_inplace(gpu_w, gpu_s, chunk_bytes=1 << 20)
    assert w_fnuz.data_ptr() == gpu_w.data_ptr()
    assert torch.equal(w_fnuz.view(torch.uint8).cpu(), ref_w.view(torch.uint8))
    assert torch.equal(s2.cpu(), ref_s)


def test_expert_gemv_fnuz_storage_bit_exact(torch):
    """fnuz-storage GEMV on the in-place converted stack must decode the
    SAME values as the e4m3fn path on the original stack (halving is exact,
    scales are doubled) — outputs bit-identical on every backend."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import e4m3fn_to_fnuz_inplace

    weights, scales = _value_stack(torch)
    x = torch.randn(
        2, 256, generator=torch.Generator().manual_seed(1)
    ).to(torch.bfloat16)
    indices = (
        torch.rand((2, 4), generator=torch.Generator().manual_seed(2))
        .topk(4, dim=-1)
        .indices.to(torch.int64)
    )
    expected = expert_gemv(x.cuda(), weights.cuda(), scales.cuda(), indices.cuda())
    w_nz, s2 = e4m3fn_to_fnuz_inplace(weights, scales)
    actual = expert_gemv(
        x.cuda(), w_nz.cuda(), s2.cuda(), indices.cuda(), storage="e4m3fnuz"
    )
    assert torch.equal(actual, expected)


def test_gather_dequant_fnuz_storage_bit_exact(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_expert_gather_dequant import gather_dequant
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import e4m3fn_to_fnuz_inplace

    weights, scales = _value_stack(torch)
    indices = torch.rand((2, 4), generator=torch.Generator().manual_seed(2)).topk(
        4, dim=-1
    ).indices.to(torch.int64)
    expected = gather_dequant(weights.cuda(), scales.cuda(), indices.cuda())
    w_nz, s2 = e4m3fn_to_fnuz_inplace(weights, scales)
    actual = gather_dequant(
        w_nz.cuda(), s2.cuda(), indices.cuda(), storage="e4m3fnuz"
    )
    assert torch.equal(actual, expected)


def test_grouped_native_inplace_matches_reference(torch):
    """The grouped fp8 path consumes the in-place views exactly as it
    consumed the copy-based conversion — outputs bit-identical."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
        e4m3fn_to_fnuz,
        e4m3fn_to_fnuz_inplace,
        fnuz_required,
        glm_moe_grouped_gemm_native,
    )
    if not fnuz_required():
        pytest.skip(
            "the fnuz grouped-gemm flavour is CDNA-only; NVIDIA fp8 is "
            "e4m3fn (glm_fp8_blockwise_gemm.fnuz_required)"
        )

    e, i, h, t, k = 4, 256, 256, 8, 2
    g = torch.Generator().manual_seed(9)
    gate_up = torch.randn((e, 2 * i, h), generator=g).clamp(-4, 4).to(torch.float8_e4m3fn)
    down = torch.randn((e, h, i), generator=g).clamp(-4, 4).to(torch.float8_e4m3fn)
    gu_sc = torch.rand((e, (2 * i) // 128, h // 128), generator=g) * 0.1 + 0.01
    dn_sc = torch.rand((e, h // 128, i // 128), generator=g) * 0.1 + 0.01
    gate_up, down, gu_sc, dn_sc = gate_up.cuda(), down.cuda(), gu_sc.cuda(), dn_sc.cuda()
    x = torch.randn((t, h), generator=g).to(torch.bfloat16).cuda()
    topk_index = torch.rand((t, e), generator=g).topk(k, dim=-1).indices.to(torch.int64).cuda()
    topk_weights = torch.softmax(torch.rand((t, k), generator=g), dim=-1).cuda()

    ref = glm_moe_grouped_gemm_native(
        x, *e4m3fn_to_fnuz(gate_up.clone(), gu_sc.clone()),
        *e4m3fn_to_fnuz(down.clone(), dn_sc.clone()),
        topk_index, topk_weights,
    )
    gu_nz, gu_sc2 = e4m3fn_to_fnuz_inplace(gate_up, gu_sc)
    dn_nz, dn_sc2 = e4m3fn_to_fnuz_inplace(down, dn_sc)
    actual = glm_moe_grouped_gemm_native(
        x, gu_nz, gu_sc2, dn_nz, dn_sc2, topk_index, topk_weights
    )
    assert torch.equal(actual, ref)
