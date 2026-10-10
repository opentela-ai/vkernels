"""glm_moe_grouped: the two-launch grouped decode pair vs the expert_gemv ladder.

The kernels (torch_ops/glm_moe_grouped.py) fold the decode MoE segment —
``expert_gemv`` (gate/up) + ``swiglu_limit`` + ``expert_gemv`` (down) +
``moe_weighted_sum`` — into two Triton launches that consume the stacked
fp8-e4m3fn expert tensors directly (no moe_align/sort, no merged step, no
host syncs). Contract: bit-exact with the ladder on a given device — the
ladder (and its eager-combine variant) is the oracle here, asserted with
``torch.equal``, the strictest gate in the kernel suite.

CPU environments exercise the eligibility/fallback surface only; the
kernels need CUDA (skipped otherwise, house pattern). Ported from floe's
test_moe_grouped.py (the arch-knob forward test stays in floe).
"""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops.glm_moe_grouped import (  # noqa: E402
    moe_grouped_decode,
    moe_grouped_decode_eligible,
)
from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402


def _stack(e, rows, cols, seed, scale_std=0.02):
    """Random fp8-e4m3fn stack + positive block-[128,128] fp32 scales."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(e, rows, cols, device="cuda", generator=gen) * 0.3).to(torch.float8_e4m3fn)
    s = torch.exp(torch.randn(e, rows // 128, cols // 128, device="cuda", generator=gen)) * scale_std
    return w, s.contiguous()


def _case(t, k, i, ia, h, e=None, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if e is None:
        e = max(4 * k, 32)
    w13, s13 = _stack(e, 2 * ia, i, seed * 4 + 1)
    w2, s2 = _stack(e, h, ia, seed * 4 + 2)
    idx = torch.stack([torch.randperm(e, device="cuda", generator=gen)[:k] for _ in range(t)]).to(torch.int64)
    w = torch.rand(t, k, device="cuda", generator=gen)
    w = (w / w.sum(-1, keepdim=True)).contiguous()
    x = torch.randn(t, i, device="cuda", generator=gen, dtype=torch.bfloat16)
    return x, w13, s13, w2, s2, idx, w


def _ladder(x, w13, s13, w2, s2, idx, w, limit, combine="eager"):
    """The current decode segment, composed from the shipped ops."""
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv
    from vkernels.torch_ops.elementwise import swiglu_limit
    from vkernels.torch_ops.moe_combine import moe_weighted_sum

    gu = expert_gemv(x, w13, s13, idx, t_cap=x.shape[0])
    gate, up = gu.chunk(2, dim=-1)
    act = swiglu_limit(gate, up, limit)
    out = expert_gemv(act, w2, s2, idx, t_cap=x.shape[0])
    if combine == "eager":
        return (out * w.to(out.dtype).unsqueeze(-1)).sum(dim=1)
    return moe_weighted_sum(out, w)


LIMIT = 10.0
SHAPES = [
    # (T, K, I, IA, H) — serving shape is (<=8, 8, 4096, 512, 4096); the
    # small ones keep the test fast while covering the T sweep + a tiny
    # non-pow-2-COLS case.
    (1, 8, 4096, 512, 4096),
    (2, 8, 4096, 512, 4096),
    (4, 8, 4096, 512, 4096),
    (8, 8, 4096, 512, 4096),
    (3, 2, 256, 128, 256),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rows2", [4, 8])
def test_bit_exact_vs_ladder_production_chain(shape, rows2, monkeypatch):
    """Bit-exact vs expert_gemv + swiglu_limit + expert_gemv + moe_weighted_sum."""
    monkeypatch.setenv("VK_MOE_GROUPED_CFG", f'{{"rows1": 4, "warps1": 4, "rows2": {rows2}, "warps2": 4}}')
    x, w13, s13, w2, s2, idx, w = _case(*shape, seed=shape[0])
    got = moe_grouped_decode(x, w13, s13, w2, s2, idx, w, LIMIT, t_cap=x.shape[0])
    want = _ladder(x, w13, s13, w2, s2, idx, w, LIMIT, combine="combine")
    assert got.dtype == torch.bfloat16 and got.shape == (shape[0], shape[4])
    if not torch.equal(got, want):
        bad = (got != want).any(-1).nonzero().flatten()
        raise AssertionError(f"rows2={rows2} shape={shape}: {len(bad)}/{shape[0]} rows differ; first diffs {[(int(r), got[r, :4].tolist(), want[r, :4].tolist()) for r in bad[:3]]}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("shape", SHAPES)
def test_eager_chain_tolerance(shape, monkeypatch):
    """vs the eager oracle (broadcast-mul + sum): NOT bit-exact by contract —
    the eager pair rounds ``w`` and every product at bf16, while this kernel
    and moe_weighted_sum accumulate in fp32 with ONE rounding at the store
    (moe_combine's documented parity contract, which production runs under
    fused_moe_combine=1). Gate: bf16-ulp tolerance."""
    monkeypatch.delenv("VK_MOE_GROUPED_CFG", raising=False)
    x, w13, s13, w2, s2, idx, w = _case(*shape, seed=shape[0] + 100)
    got = moe_grouped_decode(x, w13, s13, w2, s2, idx, w, LIMIT, t_cap=x.shape[0])
    want = _ladder(x, w13, s13, w2, s2, idx, w, LIMIT, combine="eager")
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_nan_propagation_matches_ladder(monkeypatch):
    """A NaN in the gate path must stay NaN on both sides (propagate_nan).
    Oracle: the production chain (fp32-accumulate combine; the eager pair
    would differ by the documented combine contract, not NaN handling)."""
    monkeypatch.delenv("VK_MOE_GROUPED_CFG", raising=False)
    x, w13, s13, w2, s2, idx, w = _case(2, 2, 256, 128, 256, 8, seed=7)
    w13 = w13.view(torch.uint8).clone()
    w13.view(-1)[12345] = 0x7F  # e4m3fn NaN byte
    w13 = w13.view(torch.float8_e4m3fn)
    got = moe_grouped_decode(x, w13, s13, w2, s2, idx, w, LIMIT, t_cap=2)
    want = _ladder(x, w13, s13, w2, s2, idx, w, LIMIT, combine="combine")
    assert torch.equal(got, want) | bool(((got == want) | (torch.isnan(got) & torch.isnan(want))).all())
    assert torch.isnan(got).any() and torch.isnan(want).any()


# -----------------------------------------------------------------------
# Storage flavours (issue #71): e4m3fn checkpoint bytes vs the in-place
# fnuz rewrite. Mirrors test_glm_fp8_fnuz_inplace.py's parity pattern —
# the SAME weights converted both ways must decode bit-identically (the
# halving map is value-exact and the fnuz scales are DOUBLED).


def _fnuz_decode_emulation(raw_uint8):
    """CPU emulation of the grouped kernels' FNUZ=1 manual bit-decode
    (glm_moe_grouped._decode_w8): bias 8, subnormals m*2^-10, NaN ONLY at
    0x80, sign bit recombined — the exact arithmetic the Triton kernel
    runs, in torch (the repo's CPU-emulation harness pattern)."""
    ri = raw_uint8.to(torch.int32)
    exponent, mantissa = (ri >> 3) & 15, ri & 7
    bits = ((exponent + 119) << 23) | (mantissa << 20)
    value = torch.where(
        exponent == 0,
        mantissa.to(torch.float32) * 0.0009765625,
        bits.view(torch.float32),
    )
    value = torch.where(ri == 128, torch.tensor(float("nan")), value)
    return (value.view(torch.int32) | ((ri & 128) << 24)).view(torch.float32)


def _segment_emulation(x, w8, s, idx, storage):
    """CPU emulation of one expert-GEMV stage's weight decode + dot:
    gather the selected stacks, decode per flavour against the flavour's
    scales, round to bf16 (the kernel's ``value * scale`` boundary) and
    take the fp32 dot — enough to pin the DECODE parity, which is what
    the flavour change touches (the dot/epilogue contract is unchanged
    and GPU-covered by the bit-exact tests above). x is [T,I] (stage 1,
    broadcast over the K slots) or [T,K,I] (stage 2)."""
    fnuz = storage == "e4m3fnuz"
    want = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    w = w8.view(want).view(torch.uint8)[idx]  # [T,K,O,I] raw bytes
    sc = s[idx]
    expand = sc.repeat_interleave(128, dim=2).repeat_interleave(128, dim=3)
    if fnuz:
        vals = _fnuz_decode_emulation(w.reshape(-1)).reshape(w.shape)
    else:
        vals = w.view(torch.float8_e4m3fn).to(torch.float32)
    x3 = x if x.ndim == 3 else x[:, None, :].expand(w.shape[0], w.shape[1], w.shape[3])
    return torch.einsum(
        "tki,tkoi->tko",
        x3.float(),
        (vals * expand).to(torch.bfloat16).float(),
    ).to(torch.bfloat16)


def test_cpu_fnuz_decode_matches_native_fn_exhaustive():
    """CPU harness: the kernel's manual fnuz bit-decode must agree with
    torch's fnuz->fp32 conversion over ALL 256 bytes (NaN position
    included — fnuz NaN is 0x80, fn NaN is 0x7f/0xff)."""
    raw = torch.arange(256, dtype=torch.uint8)
    emu = _fnuz_decode_emulation(raw)
    native = raw.view(torch.float8_e4m3fnuz).to(torch.float32)
    finite = torch.isfinite(emu) & torch.isfinite(native)
    assert torch.equal(emu[finite], native[finite])
    assert torch.isnan(emu[raw == 0x80]).all() and torch.isnan(native[raw == 0x80]).all()
    assert finite.sum() == 255  # fnuz has exactly ONE NaN encoding (0x80)


def test_cpu_fnuz_vs_fn_value_scale_parity():
    """CPU harness: after e4m3fn_to_fnuz_inplace, fnuz_decode(w_nz) *
    s_doubled == fn_decode(w) * s BITWISE for full-range byte stacks —
    the documented doubled-scale contract the kernel relies on."""
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import e4m3fn_to_fnuz_inplace

    g = torch.Generator().manual_seed(11)
    w = torch.randint(0, 256, (4, 256, 256), dtype=torch.uint8, generator=g).view(torch.float8_e4m3fn)
    s = torch.rand((4, 2, 2), generator=g) * 0.1 + 0.01
    w_nz, s2 = e4m3fn_to_fnuz_inplace(w.clone(), s.clone())
    s_big = s.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    s2_big = s2.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    fn_prod = w.view(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32) * s_big
    fz_prod = _fnuz_decode_emulation(w_nz.view(torch.uint8)) * s2_big
    finite = torch.isfinite(fn_prod) & torch.isfinite(fz_prod)
    assert finite.sum() > 0 and torch.equal(fn_prod[finite], fz_prod[finite])
    # NaN maps to NaN under the conversion (fn 0x7f/0xff -> fnuz 0x80)
    assert (torch.isnan(fn_prod) == torch.isnan(fz_prod)).all()


def test_cpu_segment_emulation_flavour_parity():
    """CPU harness: the decode segment on the ORIGINAL e4m3fn stack and on
    the in-place fnuz rewrite (same storage, doubled scales) produce
    bit-identical bf16 outputs — the property that makes the ladder oracle
    flavour-independent."""
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import e4m3fn_to_fnuz_inplace

    g = torch.Generator().manual_seed(5)
    e, i, ia, h, t, k = 8, 256, 128, 256, 2, 4
    w13 = (torch.randn((e, 2 * ia, i), generator=g) * 0.3).to(torch.float8_e4m3fn)
    w2 = (torch.randn((e, h, ia), generator=g) * 0.3).to(torch.float8_e4m3fn)
    s13 = torch.rand((e, 2 * ia // 128, i // 128), generator=g) * 0.05 + 0.005
    s2 = torch.rand((e, h // 128, ia // 128), generator=g) * 0.05 + 0.005
    idx = torch.stack([torch.randperm(e, generator=g)[:k] for _ in range(t)]).to(torch.int64)
    x = torch.randn((t, i), generator=g).to(torch.bfloat16)

    gu_fn = _segment_emulation(x, w13, s13, idx, "e4m3fn")
    # stage-2 input: swiglu would sit between stages; decode parity per
    # stage is what matters here — feed an arbitrary bf16 "act" to both.
    act = torch.randn((t, k, ia), generator=g).to(torch.bfloat16)
    out_fn = _segment_emulation(act, w2, s2, idx, "e4m3fn")

    w13_nz, s13_2 = e4m3fn_to_fnuz_inplace(w13.clone(), s13.clone())
    w2_nz, s2_2 = e4m3fn_to_fnuz_inplace(w2.clone(), s2.clone())
    gu_nz = _segment_emulation(x, w13_nz, s13_2, idx, "e4m3fnuz")
    out_nz = _segment_emulation(act, w2_nz, s2_2, idx, "e4m3fnuz")
    assert torch.equal(gu_fn, gu_nz)
    assert torch.equal(out_fn, out_nz)


def test_eligibility_accepts_fnuz_storage():
    """The flavour-agnostic contract: fnuz-dtype stacks pass every
    non-device clause of the eligibility check (the fn stack's twin
    clauses); unknown storage strings are rejected. CPU tensors can't
    pass the CUDA gate, so clause parity is probed by comparing the
    check against the fn flavour's verdict on otherwise-equal inputs."""
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import e4m3fn_to_fnuz_inplace

    cpu_x = torch.randn(1, 256, dtype=torch.bfloat16)
    cpu_w = torch.randn(8, 512, 256).to(torch.float8_e4m3fn)
    cpu_s = torch.rand(8, 4, 2)
    cpu_idx = torch.zeros(1, 2, dtype=torch.int64)
    cpu_rw = torch.ones(1, 2)
    assert not moe_grouped_decode_eligible(cpu_x, cpu_w, cpu_s, cpu_w, cpu_s, cpu_idx, cpu_rw, t_cap=2)
    assert not moe_grouped_decode_eligible(
        cpu_x, cpu_w.view(torch.float8_e4m3fnuz), cpu_s, cpu_w.view(torch.float8_e4m3fnuz), cpu_s,
        cpu_idx, cpu_rw, t_cap=2, storage="e4m3fnuz")
    # a flavour/dtype MISMATCH is rejected on the clause level either way:
    # on a CUDA box the fn check below would be True, so assert only the
    # mismatch asymmetry that holds on every host.
    if torch.cuda.is_available():
        x, w13, s13, w2, s2, idx, w = _case(1, 2, 256, 128, 256, 8)
        assert moe_grouped_decode_eligible(x, w13, s13, w2, s2, idx, w, t_cap=2)
        assert not moe_grouped_decode_eligible(x, w13, s13, w2, s2, idx, w, t_cap=2, storage="e4m3fnuz")
        w13n, s13n = e4m3fn_to_fnuz_inplace(w13.clone(), s13.clone())
        w2n, s2n = e4m3fn_to_fnuz_inplace(w2.clone(), s2.clone())
        assert moe_grouped_decode_eligible(x, w13n, s13n, w2n, s2n, idx, w, t_cap=2, storage="e4m3fnuz")
        assert not moe_grouped_decode_eligible(x, w13n, s13n, w2n, s2n, idx, w, t_cap=2)
        with pytest.raises(OpNotEligible, match="e4m3fn-or-e4m3fnuz"):
            moe_grouped_decode(x, w13n, s13n, w2n, s2n, idx, w, LIMIT, t_cap=2)
        with pytest.raises(OpNotEligible, match="storage"):
            moe_grouped_decode(x, w13, s13, w2, s2, idx, w, LIMIT, t_cap=2, storage="fp8")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("t", [1, 2, 4, 8])
@pytest.mark.parametrize("seed", [0, 204])
def test_default_geometry_bit_exact_t_sweep(t, seed, monkeypatch):
    """The DEFAULT launch geometry must reproduce the ladder's tl.sum
    reduction order on this backend, at production dims, across the decode
    T sweep. Regression for the GB10 sm_121 failure: a CUDA-side default of
    rows1=2/warps1=8 (the MI300A sweep pick) split the stage-1 fp32 dot
    reduction differently from expert_gemv's ROWS=4/num_warps=4 and broke
    bit-parity on every tested seed, faulting
    test_fnuz_storage_bit_exact_vs_ladder[shape2] before the FNUZ
    conversion (diagnostics/2026-10-06-merged-main-failures, serving-sys
    workspace). The default is backend-aware (CUDA = ladder geometry,
    HIP = the MI300A-measured pick); this pins the CUDA side."""
    from vkernels.torch_ops.glm_moe_grouped import _default_cfg

    if torch.version.hip:
        pytest.skip("pins the CUDA default geometry")
    monkeypatch.delenv("VK_MOE_GROUPED_CFG", raising=False)
    cfg = _default_cfg()
    assert (cfg["rows1"], cfg["warps1"]) == (4, 4), cfg
    x, w13, s13, w2, s2, idx, w = _case(t, 8, 4096, 512, 4096, seed=seed)
    got = moe_grouped_decode(x, w13, s13, w2, s2, idx, w, LIMIT, t_cap=t)
    assert torch.equal(got, _ladder(x, w13, s13, w2, s2, idx, w, LIMIT, combine="combine"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("shape", SHAPES)
def test_fnuz_storage_bit_exact_vs_ladder(shape, monkeypatch):
    """fnuz-storage grouped decode on the IN-PLACE converted stacks must be
    bit-identical to the e4m3fn grouped decode on the original stacks (and
    to the expert_gemv ladder): halving is value-exact, scales are doubled
    (mirrors test_glm_fp8_fnuz_inplace.py's expert_gemv parity case)."""
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import e4m3fn_to_fnuz_inplace

    monkeypatch.delenv("VK_MOE_GROUPED_CFG", raising=False)
    x, w13, s13, w2, s2, idx, w = _case(*shape, seed=shape[0] + 200)
    expected = moe_grouped_decode(x, w13, s13, w2, s2, idx, w, LIMIT, t_cap=x.shape[0])
    ladder = _ladder(x, w13, s13, w2, s2, idx, w, LIMIT, combine="combine")
    assert torch.equal(expected, ladder)
    w13n, s13_2 = e4m3fn_to_fnuz_inplace(w13.clone(), s13.clone())
    w2n, s2_2 = e4m3fn_to_fnuz_inplace(w2.clone(), s2.clone())
    got = moe_grouped_decode(x, w13n, s13_2, w2n, s2_2, idx, w, LIMIT, t_cap=x.shape[0], storage="e4m3fnuz")
    assert torch.equal(got, expected)


def test_eligibility_contract():
    if torch.cuda.is_available():
        x, w13, s13, w2, s2, idx, w = _case(1, 2, 256, 128, 256, 8)
        assert moe_grouped_decode_eligible(x, w13, s13, w2, s2, idx, w, t_cap=2)
        x2, w13b, s13b, w2b, s2b, idx2, wb = _case(2, 2, 256, 128, 256, 8)
        assert not moe_grouped_decode_eligible(x2, w13b, s13b, w2b, s2b, idx2, wb, t_cap=1)  # T > cap
        assert not moe_grouped_decode_eligible(x, w13.to(torch.bfloat16), s13, w2, s2, idx, w, t_cap=2)
        assert not moe_grouped_decode_eligible(x, w13, s13, w2, s2, idx, w.to(torch.bfloat16), t_cap=2)
        assert not moe_grouped_decode_eligible(x, w13, s13, w2, s2, idx.to(torch.int32), w, t_cap=2)
        with pytest.raises(OpNotEligible):
            moe_grouped_decode(x, w13, s13, w2, s2, idx, w.to(torch.bfloat16), LIMIT, t_cap=2)
    cpu_x = torch.randn(1, 256, dtype=torch.bfloat16)
    cpu_w = torch.randn(8, 256, 256).to(torch.float8_e4m3fn)
    cpu_s = torch.rand(8, 2, 2)
    cpu_idx = torch.zeros(1, 2, dtype=torch.int64)
    cpu_rw = torch.ones(1, 2)
    assert not moe_grouped_decode_eligible(cpu_x, cpu_w, cpu_s, cpu_w, cpu_s, cpu_idx, cpu_rw, t_cap=2)
    with pytest.raises(OpNotEligible):
        moe_grouped_decode(cpu_x, cpu_w, cpu_s, cpu_w, cpu_s, cpu_idx, cpu_rw, LIMIT, t_cap=2)
