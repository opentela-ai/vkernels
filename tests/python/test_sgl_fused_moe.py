"""sgl_fused_moe: the vendored SGLang Triton fused-MoE vs the eager oracle.

Modeled on the moe_combine test (the house pattern for a borrowed kernel:
CUDA-gated parity vs the eager reference, eligibility contract tests).

Parity class (NOT bit-identical — tolerance-gated, bf16-ulp scale):
the kernel computes fp8 x fp8 dots with fp32 block scales and bf16
stores, plus per-token-group fp8 activation quant — the same class as
the grouped path and the expert_gemv path, one rounding per boundary.
The oracle runs bf16 GEMMs over the DEQUANTIZED weights (the same weight
values the kernel sees), so the gate isolates kernel numerics from
weight-quant error; activation-quant noise remains and is covered by the
rtol/atol below (rtol=2e-2, the house bf16 gate).

Ported from floe's test_sgl_fused_moe.py (the knobs-wired forward test
stays in floe; tolerances measured on sgs-gpu07 H100).
"""

import math

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops.sgl_moe import (  # noqa: E402
    per_token_group_quant_fp8,
    sgl_fused_moe,
    sgl_fused_moe_eligible,
)
from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402

SWIGLU_LIMIT = 7.0  # GLM-5.3-Flash checkpoint value


def _fp8_block_stack(w: torch.Tensor):
    """bf16 [E, O, I] -> (fp8-e4m3fn [E, O, I], fp32 [E, O/128, I/128]).

    The checkpoint quantization scheme: per-128x128-block amax/448 scales.
    """
    e, o, i = w.shape
    assert o % 128 == 0 and i % 128 == 0
    wb = w.float().view(e, o // 128, 128, i // 128, 128)
    amax = wb.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12)
    scale = amax / 448.0
    q = (wb / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(e, o, i), scale.squeeze(2).squeeze(3).contiguous()


def _dequant(w8: torch.Tensor, s: torch.Tensor, dtype=torch.bfloat16):
    """The Glm53Experts._dq reference: w8.float() * block scales -> dtype."""
    e, o, i = w8.shape
    w = w8.float().view(e, o // 128, 128, i // 128, 128) * s.float()[:, :, None, :, None]
    return w.reshape(e, o, i).to(dtype)


def _make_case(device, e, h, i, topk, t, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(t, h, device=device, dtype=torch.bfloat16, generator=gen)
    w13 = (torch.randn(e, 2 * i, h, device=device, generator=gen) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(e, h, i, device=device, generator=gen) * 0.02).to(torch.bfloat16)
    w13_8, w13_s = _fp8_block_stack(w13)
    w2_8, w2_s = _fp8_block_stack(w2)
    # distinct experts per token (the decode router's contract)
    top_k_index = torch.stack(
        [torch.randperm(e, device=device, generator=gen)[:topk] for _ in range(t)]
    ).to(torch.int64)
    top_k_weights = torch.softmax(
        torch.randn(t, topk, device=device, generator=gen), dim=-1)
    return x, w13_8, w13_s, w2_8, w2_s, top_k_index, top_k_weights


def _eager_oracle(x, w13_8, w13_s, w2_8, w2_s, idx, w, limit=SWIGLU_LIMIT,
                  act_quant=False):
    """Per-expert eager MoE over DEQUANTIZED weights (the floe loop, bf16).

    ``act_quant=True`` additionally emulates the kernel's fp8-w8a8 activation
    quantization (per-token-group 128, deq(quant(x)) feeding each GEMM stage)
    — the reference the tight house gate applies to, isolating kernel
    numerics from the quant noise that is inherent to the w8a8 design (and
    absent from the landed bf16-activation paths, hence the separate L2 gate
    in the serving-shape test)."""
    t, _ = idx.shape
    e, n2, h = w13_8.shape
    i = n2 // 2
    out = torch.zeros(t, h, device=x.device, dtype=torch.float32)
    w13 = _dequant(w13_8, w13_s)
    w2 = _dequant(w2_8, w2_s)

    def _aq(v):
        if not act_quant:
            return v
        q, s = per_token_group_quant_fp8(v.reshape(-1, v.shape[-1]).contiguous(), 128)
        return (q.float() * s.repeat_interleave(128, -1)).to(v.dtype).view_as(v)

    x = _aq(x)
    for expert in range(e):
        tok, pos = torch.where(idx == expert)
        if tok.numel() == 0:
            continue
        gu = torch.nn.functional.linear(x[tok], w13[expert])
        gate, up = gu[:, :i], gu[:, i:]
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        act = torch.nn.functional.silu(gate) * up
        cur = torch.nn.functional.linear(_aq(act), w2[expert])
        out.index_add_(0, tok, (cur.float() * w[tok, pos, None]))
    return out.to(torch.bfloat16)


# --- parity: the fused kernel vs the eager per-expert oracle ---------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("t", [1, 2, 3, 4, 5, 6, 7, 8])
def test_sgl_fused_moe_matches_eager_small_shape(t):
    """Fast iteration shape (E=16, H=256, I=128, topk=4) across T=1..8."""
    x, w13_8, w13_s, w2_8, w2_s, idx, w = _make_case("cuda", 16, 256, 128, 4, t)
    fused = sgl_fused_moe(x, w13_8, w13_s, w2_8, w2_s, idx, w,
                          swiglu_limit=SWIGLU_LIMIT)
    oracle = _eager_oracle(x, w13_8, w13_s, w2_8, w2_s, idx, w)
    torch.testing.assert_close(fused, oracle, rtol=2e-2, atol=2e-2)
    assert fused.shape == (t, 256) and fused.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("t", [1, 4, 8])
def test_sgl_fused_moe_matches_eager_serving_shape(t):
    """The real per-rank dims: E=288, H=4096, I=512, topk=8 (TP4 decode).

    Two gates (measured on sgs-gpu07 H100): (1) vs the PLAIN eager oracle —
    scale-free L2 < 5e-2, the fp8-w8a8 activation-quant class (per-token-
    group e4m3 quant error is ~3-4% RMS of the dot and INHERENT to the
    donor design; the landed bf16-activation paths carry none, which is
    exactly why this knob stays opt-in and A/B'd); (2) vs the QUANT-AWARE
    oracle (same per-token-group quantization emulated in torch) at the
    house bf16 gate rtol=atol=2e-2 — this one pins KERNEL numerics
    (mapping, scales, swiglu, combine) independent of the quant scheme."""
    x, w13_8, w13_s, w2_8, w2_s, idx, w = _make_case("cuda", 288, 4096, 512, 8, t)
    fused = sgl_fused_moe(x, w13_8, w13_s, w2_8, w2_s, idx, w,
                          swiglu_limit=SWIGLU_LIMIT)
    oracle = _eager_oracle(x, w13_8, w13_s, w2_8, w2_s, idx, w)
    rel = (fused.float() - oracle.float()).norm() / oracle.float().norm()
    assert rel.item() < 5e-2, f"L2 rel error {rel.item():.4f} > 5e-2"
    oracle_q = _eager_oracle(x, w13_8, w13_s, w2_8, w2_s, idx, w, act_quant=True)
    torch.testing.assert_close(fused, oracle_q, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_sgl_fused_moe_duplicate_experts_and_alignment():
    """All tokens routing the SAME expert (worst-case alignment: one full
    bucket) and the numel>64 torch-align fallback at T=9, topk=8."""
    x, w13_8, w13_s, w2_8, w2_s, idx, w = _make_case("cuda", 16, 256, 128, 4, 4)
    idx = idx[:, :1].expand(-1, 4).contiguous()  # every token -> one expert x4
    fused = sgl_fused_moe(x, w13_8, w13_s, w2_8, w2_s, idx, w,
                          swiglu_limit=SWIGLU_LIMIT)
    oracle = _eager_oracle(x, w13_8, w13_s, w2_8, w2_s, idx, w)
    torch.testing.assert_close(fused, oracle, rtol=2e-2, atol=2e-2)
    # numel = 9*8 = 72 > SMALL_NUMEL_LIMIT: the static-shape torch align path
    x9, w13_8_9, w13_s_9, w2_8_9, w2_s_9, idx9, w9 = _make_case("cuda", 16, 256, 128, 8, 9, seed=3)
    fused9 = sgl_fused_moe(x9, w13_8_9, w13_s_9, w2_8_9, w2_s_9, idx9, w9,
                           swiglu_limit=SWIGLU_LIMIT)
    oracle9 = _eager_oracle(x9, w13_8_9, w13_s_9, w2_8_9, w2_s_9, idx9, w9)
    torch.testing.assert_close(fused9, oracle9, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_sgl_fused_moe_no_limit():
    """swiglu_limit=inf (plain swiglu) also matches the oracle."""
    x, w13_8, w13_s, w2_8, w2_s, idx, w = _make_case("cuda", 8, 256, 128, 4, 4)
    fused = sgl_fused_moe(x, w13_8, w13_s, w2_8, w2_s, idx, w, swiglu_limit=math.inf)
    oracle = _eager_oracle(x, w13_8, w13_s, w2_8, w2_s, idx, w, limit=math.inf)
    torch.testing.assert_close(fused, oracle, rtol=2e-2, atol=2e-2)


# --- closed-form reference (lane moe-b4 requirement) -----------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("t", [1, 4, 8])
def test_sgl_fused_moe_matches_closed_form(t):
    """Synthetic weights vs a CLOSED-FORM reference: the exact math (fp64
    dequantized-weight matmuls, exact clamp/silu/mul, fp64 weighted sum),
    not just the eager bf16 oracle — pins the shape mapping and the
    routing/combine semantics independently of any floe code path."""
    e, h, i, topk = 16, 256, 128, 4
    x, w13_8, w13_s, w2_8, w2_s, idx, w = _make_case("cuda", e, h, i, topk, t,
                                                     seed=11)
    fused = sgl_fused_moe(x, w13_8, w13_s, w2_8, w2_s, idx, w,
                          swiglu_limit=SWIGLU_LIMIT)
    w13 = _dequant(w13_8, w13_s, torch.float64)
    w2 = _dequant(w2_8, w2_s, torch.float64)
    xf = x.double()
    ref = torch.zeros(t, h, dtype=torch.float64, device=x.device)
    for tok in range(t):
        for k in range(topk):
            expert = int(idx[tok, k])
            gu = w13[expert] @ xf[tok]
            gate = gu[:i].clamp(max=SWIGLU_LIMIT)
            up = gu[i:].clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            act = gate * torch.sigmoid(gate) * up
            ref[tok] += float(w[tok, k]) * (w2[expert] @ act)
    torch.testing.assert_close(fused.double(), ref, rtol=2e-2, atol=2e-2)


# --- capture safety ---------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_sgl_fused_moe_cuda_graph_capturable():
    """The whole wrapper records into a CUDA graph at a fixed T bucket and
    replays bit-identically (static shapes/strides, device-only align reads
    — the capture contract in torch_ops/sgl_moe.py)."""
    x, w13_8, w13_s, w2_8, w2_s, idx, w = _make_case("cuda", 16, 256, 128, 4, 4)
    sgl_fused_moe(x, w13_8, w13_s, w2_8, w2_s, idx, w,
                  swiglu_limit=SWIGLU_LIMIT)  # eager warmup: JIT compiles here
    x2, idx2, w2t = x.clone(), idx.clone(), w.clone()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sgl_fused_moe(x2, w13_8, w13_s, w2_8, w2_s, idx2, w2t,
                      swiglu_limit=SWIGLU_LIMIT)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        replay = sgl_fused_moe(x2, w13_8, w13_s, w2_8, w2_s, idx2, w2t,
                               swiglu_limit=SWIGLU_LIMIT)
    eager = sgl_fused_moe(x2, w13_8, w13_s, w2_8, w2_s, idx2, w2t,
                          swiglu_limit=SWIGLU_LIMIT)
    g.replay()
    torch.testing.assert_close(replay, eager, rtol=0, atol=0)  # bit-identical replay


# --- eligibility contract ---------------------------------------------------
def test_eligibility_contract():
    if torch.cuda.is_available():
        x, w13_8, w13_s, w2_8, w2_s, idx, w = _make_case("cuda", 8, 256, 128, 4, 2)
        assert sgl_fused_moe_eligible(x, w13_8, w13_s, w2_8, w2_s, idx, w)
        assert not sgl_fused_moe_eligible(x, w13_8.float(), w13_s, w2_8, w2_s, idx, w)  # fp8 stacks required
        assert not sgl_fused_moe_eligible(x, w13_8, None, w2_8, w2_s, idx, w)  # scales required
        assert not sgl_fused_moe_eligible(x, w13_8, w13_s, w2_8, w2_s, idx, w.to(torch.int32))  # floating weights required
        assert not sgl_fused_moe_eligible(x, w13_8, w13_s, w2_8, w2_s, idx[:, :2], w)  # shapes must agree
        with pytest.raises(OpNotEligible):
            sgl_fused_moe(x, w13_8.float(), w13_s, w2_8, w2_s, idx, w)
    cpu_x = torch.randn(2, 256, dtype=torch.bfloat16)
    cpu_w8 = torch.zeros(4, 256, 256, dtype=torch.float8_e4m3fn)
    cpu_s = torch.ones(4, 2, 2)
    cpu_idx = torch.zeros(2, 4, dtype=torch.int64)
    cpu_w = torch.ones(2, 4)
    assert not sgl_fused_moe_eligible(cpu_x, cpu_w8, cpu_s, cpu_w8, cpu_s, cpu_idx, cpu_w)  # CUDA required
    if not torch.cuda.is_available():
        with pytest.raises(OpNotEligible):
            sgl_fused_moe(cpu_x, cpu_w8, cpu_s, cpu_w8, cpu_s, cpu_idx, cpu_w)
