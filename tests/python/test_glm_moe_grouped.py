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
