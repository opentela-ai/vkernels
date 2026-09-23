"""mHC pre-GEMV fusion: contract checks, eager-chain parity, capture.

Parity bar: bf16 logits against the exact two-launch eager chain the
kernel replaces (``Glm53UnweightedRMSNorm`` formula + ``F.linear``) at
the deployed [24, 16384] shape (GLM-5.3-Flash: hc=4, hidden 4096). Both
sides accumulate the GEMV in FP32 and round once on store, so the
residual drift is bounded by fp32 reduction order; the test reports the
observed max abs diff per bucket (tolerance 2e-2 on the bf16 logit
scale, far above the observed ~1e-3-class drift and below anything that
could move a gate).
"""

import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.mhc_pre_gemv; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


# Rig-realistic envelope: hc=4, hidden 4096 (fn [24, 16384]); decode graph
# buckets are 1/2/4 tokens. A non-pow2 K (hc=2, hidden 96) and an odd
# hidden exercise the masked tail path.
HC_D = ((4, 4096), (2, 96), (2, 1536))
TOKENS = (1, 2)  # the op's decode envelope (rows <= 2; mhc_projection's line)


def _inputs(torch, hc, hidden, tokens, seed=13, dtype=None):
    dtype = dtype or torch.bfloat16
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device("cuda")
    k = hc * hidden
    x = torch.randn(tokens, k, generator=g).to(dev, dtype) * 0.5
    fn = (torch.randn(hc * (hc + 2), k, generator=g) * 0.02).to(dev, dtype)
    return x, fn


def _ref(torch, x, fn, hc, hidden, eps=1e-6):
    from vkernels.torch_ops.mhc_pre_gemv import mhc_pre_gemv_reference

    return mhc_pre_gemv_reference(x, fn, hc=hc, hidden_size=hidden)


def test_contract(torch):
    from vkernels.torch_ops.mhc_pre_gemv import mhc_pre_gemv

    x, fn = _inputs(torch, 4, 4096, 2)
    with pytest.raises(ValueError, match="hc must be"):
        mhc_pre_gemv(x, fn, hc=3, hidden_size=4096)
    with pytest.raises(ValueError, match="streams_flat"):
        mhc_pre_gemv(x[:, :100], fn, hc=4, hidden_size=4096)
    with pytest.raises(ValueError, match="fn "):
        mhc_pre_gemv(x, fn[:10], hc=4, hidden_size=4096)
    with pytest.raises(ValueError, match="bf16/fp16"):
        mhc_pre_gemv(x.float(), fn.float(), hc=4, hidden_size=4096)
    with pytest.raises(ValueError, match="contiguous"):
        mhc_pre_gemv(x.t().contiguous().t(), fn, hc=4, hidden_size=4096)
    with pytest.raises(ValueError, match="decode-sized"):
        mhc_pre_gemv(x.repeat(2, 1), fn, hc=4, hidden_size=4096)  # 4 rows
    cpu = _inputs(torch, 4, 4096, 2)
    with pytest.raises(ValueError, match="GPU"):
        mhc_pre_gemv(cpu[0].cpu(), cpu[1].cpu(), hc=4, hidden_size=4096)


def test_gpu_parity(torch):
    from vkernels.torch_ops.mhc_pre_gemv import mhc_pre_gemv

    for hc, hidden in HC_D:
        for tokens in TOKENS:
            x, fn = _inputs(torch, hc, hidden, tokens, seed=hc * 100 + hidden + tokens)
            got = mhc_pre_gemv(x, fn, hc=hc, hidden_size=hidden)
            want = _ref(torch, x, fn, hc, hidden)
            assert got.shape == want.shape
            assert got.dtype == want.dtype
            diff = (got.float() - want.float()).abs().max().item()
            scale = want.float().abs().max().item()
            # bf16 logit grid + fp32-accum reduction-order drift only
            assert diff <= max(2e-2, scale * 2e-2), (hc, hidden, tokens, diff)
            print(f"hc={hc} hidden={hidden} tokens={tokens}: max abs diff {diff:.3e}")


def test_gpu_tokens_shape_roundtrip(torch):
    """The floe call site passes [B, S, hc*hidden] and views [..., mix]
    back — the op must preserve leading dims exactly."""
    from vkernels.torch_ops.mhc_pre_gemv import mhc_pre_gemv

    hc, hidden = 4, 4096
    x, fn = _inputs(torch, hc, hidden, 2)
    x3 = x.view(2, 1, -1)
    got = mhc_pre_gemv(x3, fn, hc=hc, hidden_size=hidden)
    assert got.shape == (2, 1, hc * (hc + 2))
    want = _ref(torch, x, fn, hc, hidden).view(2, 1, -1)
    diff = (got.float() - want.float()).abs().max().item()
    scale = want.float().abs().max().item()
    assert diff <= max(2e-2, scale * 2e-2), diff


def test_graph_capture_replay(torch):
    from vkernels.torch_ops.mhc_pre_gemv import mhc_pre_gemv

    hc, hidden = 4, 4096
    x, fn = _inputs(torch, hc, hidden, 1)
    mhc_pre_gemv(x, fn, hc=hc, hidden_size=hidden)  # warm the JIT
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mhc_pre_gemv(x, fn, hc=hc, hidden_size=hidden)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        out = mhc_pre_gemv(x, fn, hc=hc, hidden_size=hidden)
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    want = _ref(torch, x, fn, hc, hidden)
    diff = (out.float() - want.float()).abs().max().item()
    scale = want.float().abs().max().item()
    assert diff <= max(2e-2, scale * 2e-2), diff
