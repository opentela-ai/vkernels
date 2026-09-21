"""mHC pre big-fuse: contract checks, eager-chain parity, drift probe, capture.

Parity bar: the fp32 control planes against the mhc_mix bar (atol/rtol
1e-5, test_glm_mhc_mix) and the bf16 layer input against the composed
eager chain (mix reference -> eager collapse -> eager RMSNorm) — the
kernel pays exactly the chain's rounding points, so the residual drift is
bounded by fp32 reduction order (documented in the module docstring) and
the test reports the observed max abs diff per bucket.
"""

import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.glm_mhc_big_fuse; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


# Rig-realistic envelope: GLM-5.3-Flash runs hc=4, hidden 4096 (the mix fn
# is [24, 16384]); decode graph buckets are 1/2/4/8 tokens.
HC = 4
D = 4096
WIDTH = HC * (HC + 2)
TOKENS = (1, 2, 4, 8)


def _rig_inputs(torch, tokens, seed=11, sinkhorn_iters=20):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device("cuda")
    logits = torch.randn(tokens, WIDTH, generator=g).to(dev, torch.bfloat16)
    base = torch.randn(WIDTH, generator=g).to(dev)
    scale = torch.randn(3, generator=g).to(dev)
    streams = torch.randn(tokens, HC, D, generator=g).to(dev, torch.bfloat16)
    norm_weight = torch.randn(D, generator=g).to(dev, torch.bfloat16)
    return logits, base, scale, streams, norm_weight, sinkhorn_iters


def test_contract(torch):
    from vkernels.torch_ops.glm_mhc_big_fuse import mhc_pre_big_fuse

    logits, base, scale, streams, norm_weight, iters = _rig_inputs(torch, 2)
    with pytest.raises(ValueError, match="hc values"):
        mhc_pre_big_fuse(logits, base, scale, streams, norm_weight, hc=3)
    with pytest.raises(ValueError, match="width"):
        mhc_pre_big_fuse(torch.randn(2, 23, device="cuda"), base, scale, streams, norm_weight)
    with pytest.raises(ValueError, match="FP32"):
        mhc_pre_big_fuse(logits, base.double(), scale, streams, norm_weight)
    with pytest.raises(ValueError, match="GPU"):
        mhc_pre_big_fuse(logits.cpu(), base, scale, streams.cpu(), norm_weight)
    with pytest.raises(ValueError, match="power of two"):
        bad = torch.randn(2, HC, 4000, device="cuda").to(torch.bfloat16)
        mhc_pre_big_fuse(logits, base, scale, bad, torch.randn(4000, device="cuda"))

def test_gpu_parity_and_drift(torch):
    """Per-bucket parity vs the composed eager chain + drift probe."""
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_mhc_big_fuse import (
        mhc_pre_big_fuse,
        mhc_pre_big_fuse_reference,
    )

    for tokens in TOKENS:
        logits, base, scale, streams, norm_weight, iters = _rig_inputs(
            torch, tokens, seed=17 + tokens
        )
        post, comb, layer_input = mhc_pre_big_fuse(
            logits, base, scale, streams, norm_weight,
            hc=HC, sinkhorn_iters=iters, norm_eps=1e-6,
        )
        post_r, comb_r, layer_r = mhc_pre_big_fuse_reference(
            logits, base, scale, streams, norm_weight,
            hc=HC, eps=1e-6, sinkhorn_iters=iters, norm_eps=1e-6,
        )
        assert post.shape == (tokens, HC) and post.dtype == torch.float32
        assert comb.shape == (tokens, HC, HC) and comb.dtype == torch.float32
        assert layer_input.shape == (tokens, D)
        assert layer_input.dtype == streams.dtype
        torch.testing.assert_close(post, post_r, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(comb, comb_r, atol=1e-5, rtol=1e-5)
        drift = (layer_input.float() - layer_r.float()).abs().max().item()
        # bf16 store: the fused kernel and the chain round at identical
        # points, so the residual is reduction-order dust — a few bf16 ulps
        # at unit scale at most. Bar: the compose tests' bf16 tolerance.
        assert drift <= 2 ** -6, f"drift {drift} at tokens={tokens}"
        print(f"tokens={tokens}: max|dlayer_input| = {drift:.3e}")


def test_gpu_parity_random_sweeps(torch):
    """Many seeds incl. extreme gate scales — the Sinkhorn loop amplifies
    scale[2] outliers, so sweep rather than spot-check."""
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_mhc_big_fuse import (
        mhc_pre_big_fuse,
        mhc_pre_big_fuse_reference,
    )

    for seed in range(8):
        logits, base, scale, streams, norm_weight, iters = _rig_inputs(
            torch, 4, seed=seed
        )
        if seed % 2:
            scale = scale * 8.0  # stress the gate/softmax dynamic range
        post, comb, layer_input = mhc_pre_big_fuse(
            logits, base, scale, streams, norm_weight, hc=HC, sinkhorn_iters=6
        )
        post_r, comb_r, layer_r = mhc_pre_big_fuse_reference(
            logits, base, scale, streams, norm_weight, hc=HC, sinkhorn_iters=6
        )
        torch.testing.assert_close(post, post_r, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(comb, comb_r, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            layer_input.float(), layer_r.float(), atol=2 ** -6, rtol=2 ** -6
        )


def test_graph_capture_replay(torch):
    """CUDA-graph capture/replay with static per-bucket shapes: the decode
    graphs bank (lane D) requires launches with no host syncs and static
    shapes; replay must track new inputs without re-capture."""
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_mhc_big_fuse import mhc_pre_big_fuse

    tokens = 4
    logits, base, scale, streams, norm_weight, iters = _rig_inputs(torch, tokens)
    # Warm the JIT specialization outside capture (prewarmed-bucket model).
    mhc_pre_big_fuse(
        logits, base, scale, streams, norm_weight, hc=HC, sinkhorn_iters=iters
    )
    g = torch.cuda.CUDAGraph()
    static = (logits, base, scale, streams, norm_weight)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        p, c, o = mhc_pre_big_fuse(
            *static, hc=HC, sinkhorn_iters=iters
        )
    # fresh data in the same static buffers
    logits.copy_(torch.randn(tokens, WIDTH, device="cuda"))
    streams.copy_(torch.randn(tokens, HC, D, device="cuda"))
    g.replay()
    torch.cuda.synchronize()
    post2, comb2, out2 = mhc_pre_big_fuse(
        logits, base, scale, streams, norm_weight, hc=HC, sinkhorn_iters=iters
    )
    torch.testing.assert_close(p, post2, atol=0, rtol=0)
    torch.testing.assert_close(c, comb2, atol=0, rtol=0)
    torch.testing.assert_close(o, out2, atol=0, rtol=0)
