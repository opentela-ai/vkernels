"""mHC chain-fuse kernel (``mhc_compose_pre``): parity + incumbent + capture.

The kernel replaces the three-launch decode chain between two hyper-
connection sites (``_compose`` → ``_norm_uw`` → mix GEMV) with one Triton
launch. The binding oracle is the VERBATIM eager chain
(:func:`mhc_compose_pre_reference` — the production eager expressions from
``_mhc_compose`` + ``Glm53UnweightedRMSNorm`` + ``F.linear``); against it
the contract is the same documented class ``mhc_compose`` and
``mhc_pre_gemv`` already carry (fp32 reduction order / rsqrt approximation
differences that survive at most one storage-dtype ulp per rounding
boundary). Against the FUSED incumbent chain (``mhc_compose`` →
``mhc_pre_gemv`` — the ``mhc_big_fuse`` arm this lane stacks on) the
element-wise structure is reproduced verbatim, so the strong assertion is
bit-equality. Ported from floe's test_glm5_mhc_chain.py (the site-dispatch
tests stay in floe).
"""

import contextlib

import pytest

torch = pytest.importorskip("torch")

pytest.importorskip("triton")

from vkernels.torch_ops.mhc_chain import (  # noqa: E402
    mhc_compose_pre,
    mhc_compose_pre_reference,
)
from vkernels.torch_ops.mhc_compose import mhc_compose  # noqa: E402
from vkernels.torch_ops.mhc_pre_gemv import mhc_pre_gemv  # noqa: E402
from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

# Rig-realistic envelope: the deployed site (hc=4, hidden 4096 → fn
# [24, 16384]) plus the pow-2 hidden ladder; the kernel's correctness
# envelope is the 8-row decode ladder (swept under the eligibility
# override), while the shipped policy caps the fusion at rows <= 2 (the
# measured crossover — see mhc_chain._max_rows).
HC_D = ((4, 4096), (2, 512), (4, 2048))
TOKENS = (1, 2, 4, 8)
POLICY_TOKENS = (1, 2)


@contextlib.contextmanager
def _exact_fp32():
    """Pin fp32 matmul to IEEE for the reference legs (tf32 would move the
    eager oracle itself by ~2**-11, an order more than any kernel drift)."""
    backend = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    previous = getattr(backend, "allow_tf32", None)
    if previous is None:
        yield
        return
    backend.allow_tf32 = False
    try:
        yield
    finally:
        backend.allow_tf32 = previous


def _bf16_ulp_distance(got, want):
    """Element count of >0 ulp drift between two same-dtype tensors, plus
    the max ulp distance (bf16 grid; treat inf/nan mismatches as fatal)."""
    assert got.dtype == want.dtype, (got.dtype, want.dtype)
    gi = got.view(torch.int16).to(torch.int32)
    wi = want.view(torch.int16).to(torch.int32)
    # map sign-magnitude to a monotone ladder
    gi = torch.where(gi < 0, -32768 - gi, gi)
    wi = torch.where(wi < 0, -32768 - wi, wi)
    d = (gi - wi).abs()
    return int((d > 0).sum().item()), int(d.max().item())


def _inputs(hc, hidden, tokens, seed=7, dtype=torch.bfloat16):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k, width = hc * hidden, hc * (hc + 2)
    post = (torch.rand(tokens, hc, generator=g) * 2).to("cuda", torch.float32)
    comb = torch.rand(tokens, hc, hc, generator=g).to("cuda", torch.float32)
    sub = (torch.randn(tokens, hidden, generator=g) * 0.7).to("cuda", dtype)
    streams = (torch.randn(tokens, hc, hidden, generator=g) * 0.7).to("cuda", dtype)
    fn = (torch.randn(width, k, generator=g) * 0.02).to("cuda", dtype)
    return post, comb, sub, streams, fn


def test_contract():
    post, comb, sub, streams, fn = _inputs(4, 4096, 2)
    with pytest.raises(OpNotEligible, match="hc must be"):
        mhc_compose_pre(post, comb, sub, streams, fn, hc=3, hidden_size=4096)
    with pytest.raises(OpNotEligible, match="streams must be"):
        mhc_compose_pre(post, comb, sub, streams[:, :, :100].contiguous(), fn, hc=4, hidden_size=4096)
    with pytest.raises(OpNotEligible, match="fn "):
        mhc_compose_pre(post, comb, sub, streams, fn[:3].contiguous(), hc=4, hidden_size=4096)
    with pytest.raises(OpNotEligible, match="share the streams' dtype"):
        mhc_compose_pre(post, comb, sub.float(), streams, fn, hc=4, hidden_size=4096)
    with pytest.raises(OpNotEligible, match="bf16/fp16"):
        mhc_compose_pre(post, comb, sub.double(), streams.double(), fn.double(),
                        hc=4, hidden_size=4096)
    with pytest.raises(OpNotEligible, match="decode-sized"):
        big = _inputs(4, 4096, 9)
        mhc_compose_pre(*big, hc=4, hidden_size=4096)
    # the shipped POLICY cap (rows <= 2): the T=4/8 buckets keep the
    # incumbent arm where the fat-CTA regime measured slower
    with pytest.raises(OpNotEligible, match="decode-sized"):
        mhc_compose_pre(*_inputs(4, 4096, 4), hc=4, hidden_size=4096)
    with pytest.raises(OpNotEligible, match="power of two"):
        mhc_compose_pre(*_inputs(4, 4032, 2), hc=4, hidden_size=4032)
    cpu = tuple(t.cpu() for t in _inputs(4, 4096, 2))
    with pytest.raises(OpNotEligible, match="GPU"):
        mhc_compose_pre(*cpu, hc=4, hidden_size=4096)
    with pytest.raises(OpNotEligible, match="contiguous"):
        p = _inputs(4, 4096, 2)
        mhc_compose_pre(p[0], p[1], p[2], p[3].transpose(1, 2).contiguous().transpose(1, 2), p[4],
                        hc=4, hidden_size=4096)


@pytest.mark.parametrize("hc,hidden", HC_D)
@pytest.mark.parametrize("tokens", TOKENS)
def test_kernel_vs_verbatim_eager_chain(hc, hidden, tokens, monkeypatch):
    monkeypatch.setenv("VK_MHC_COMPOSE_PRE_ROWS", "8")
    post, comb, sub, streams, fn = _inputs(hc, hidden, tokens, seed=hc * 100 + hidden + tokens)
    with _exact_fp32():
        ref_streams, ref_logits = mhc_compose_pre_reference(
            post, comb, sub, streams, fn, hc=hc, hidden_size=hidden)
        got_streams, got_logits = mhc_compose_pre(
            post, comb, sub, streams, fn, hc=hc, hidden_size=hidden)
    assert got_streams.shape == ref_streams.shape and got_streams.dtype == ref_streams.dtype
    assert got_logits.shape == ref_logits.shape and got_logits.dtype == ref_logits.dtype
    # compose leg: same fp32 association, cuBLAS blocking drift only — the
    # mhc_compose contract (<= 1 storage ulp)
    n_bad, max_ulp = _bf16_ulp_distance(got_streams, ref_streams)
    assert max_ulp <= 1, (hc, hidden, tokens, n_bad, max_ulp)
    # head leg: the mhc_pre_gemv contract class — a bf16-rounded rstd factor
    # that flips one storage step scales every logit by (1 ± 2**-8), so the
    # drift is a RELATIVE class (pinned 2e-2; measured identical to
    # the incumbent chain's own drift at these shapes, see the bit-equality
    # test).
    diff = (got_logits.float() - ref_logits.float()).abs()
    scale = ref_logits.float().abs().max().item()
    assert diff.max().item() <= max(2e-2, scale * 2e-2), (hc, hidden, tokens,
                                                          diff.max().item(), scale)


def test_bit_equal_vs_fused_incumbent_chain(monkeypatch):
    monkeypatch.setenv("VK_MHC_COMPOSE_PRE_ROWS", "8")
    """The chain this lane replaces in the ``mhc_big_fuse`` arm is
    ``mhc_compose`` → ``mhc_pre_gemv``; the kernel reproduces that
    element-wise structure, so bit-equality must hold (the reduction trees
    share the flat hc·d vector shape)."""
    for hc, hidden in HC_D:
        for tokens in TOKENS:  # under the envelope override below
            post, comb, sub, streams, fn = _inputs(hc, hidden, tokens, seed=hc * 7 + tokens)
            got_streams, got_logits = mhc_compose_pre(
                post, comb, sub, streams, fn, hc=hc, hidden_size=hidden)
            inc_streams = mhc_compose(post, comb, sub, streams, hc=hc)
            k = hc * hidden
            inc_logits = mhc_pre_gemv(
                inc_streams.reshape(tokens, k), fn, hc=hc, hidden_size=hidden)
            assert torch.equal(got_streams, inc_streams), (hc, hidden, tokens)
            assert torch.equal(got_logits, inc_logits.view_as(got_logits)), (hc, hidden, tokens)


def test_graph_capture_replay():
    hc, hidden, tokens = 4, 4096, 1
    post, comb, sub, streams, fn = _inputs(hc, hidden, tokens)
    mhc_compose_pre(post, comb, sub, streams, fn, hc=hc, hidden_size=hidden)  # JIT warm
    torch.cuda.synchronize()
    want = mhc_compose_pre(post, comb, sub, streams, fn, hc=hc, hidden_size=hidden)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        got = mhc_compose_pre(post, comb, sub, streams, fn, hc=hc, hidden_size=hidden)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cap = mhc_compose_pre(post, comb, sub, streams, fn, hc=hc, hidden_size=hidden)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(cap[0], want[0]) and torch.equal(cap[1], want[1])
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
