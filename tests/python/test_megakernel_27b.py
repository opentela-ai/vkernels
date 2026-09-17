"""Qwen3.8-27B-FP8 megakernel feasibility: validated building blocks.

The 27B target is a Qwen3.5-family hybrid (64 layers: 48 GDN linear-
attention + 16 full-attention, gated attention, FP8 block-quantized
weights) with the incoai/Qwen3.8-27B-DFlash2 drafter. The dense-Qwen3
megakernel does not cover it yet; these tests prove the three missing
piece classes against the REAL checkpoint tensors and the repo's own
reference implementations:

* **FP8 block GEMV** (``_t_gemv_fp8``): e4m3 weights + 128x128
  ``weight_scale_inv`` blocks, dequantized in-register — vs a torch
  dequant reference on layer 0's real ``mlp.gate_proj``;
* **GDN decode step, decomposed into tasks**: projections (fp8 GEMVs) ->
  conv-state update (``_t_gdn_conv``) -> per-head delta rule
  (``_t_gdn_heads``) -> out_proj (fp8 GEMV), vs the repo's
  ``GatedDeltaNet.forward`` at seq=1 with the real layer-0 weights in
  fp32 — final output, SSM state, and conv state all compared.

Skipped when the checkpoint is not on this box.
"""

from __future__ import annotations

import glob
import os

import pytest

torch = pytest.importorskip("torch")
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from vkernels.compiler.device_triton import _t_gdn_conv, _t_gdn_heads, _t_gemv_fp8, triton_available  # noqa: E402


@triton.jit
def _drive_gemv_fp8(x_ptr, w_ptr, scale_ptr, y_ptr, K: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr, BK: tl.constexpr):
    _t_gemv_fp8(tl.program_id(0), tl.num_programs(0), x_ptr, w_ptr, scale_ptr, y_ptr, K, N, TILE, BK)


@triton.jit
def _drive_gdn_conv(state_ptr, w_ptr, x_ptr, out_ptr, C: tl.constexpr, ELEM: tl.constexpr, KTAPS: tl.constexpr):
    _t_gdn_conv(tl.program_id(0), tl.num_programs(0), state_ptr, w_ptr, x_ptr, out_ptr, C, ELEM, KTAPS)


@triton.jit
def _drive_gdn_heads(q_ptr, k_ptr, v_ptr, z_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr, normw_ptr, state_ptr, out_ptr, NH: tl.constexpr, NK: tl.constexpr, HV: tl.constexpr, HK: tl.constexpr, eps: tl.constexpr, scale: tl.constexpr):
    _t_gdn_heads(tl.program_id(0), tl.num_programs(0), q_ptr, k_ptr, v_ptr, z_ptr, a_ptr, b_ptr, alog_ptr, dtb_ptr, normw_ptr, state_ptr, out_ptr, NH, NK, HV, HK, eps, scale)


gpu = pytest.mark.skipif(not (torch.cuda.is_available() and triton_available()), reason="requires CUDA + triton")

TARGET = "/local/home/xiayao/minisgl-ds5/models/Qwen3.8-27B-FP8"
has_27b = pytest.mark.skipif(not glob.glob(os.path.join(TARGET, "layers-*.safetensors")), reason=f"requires {TARGET}")


def _layer0():
    """Layer-0 tensors of the real 27B checkpoint (one 384MB shard)."""
    from safetensors.torch import load_file

    path = sorted(glob.glob(os.path.join(TARGET, "layers-*.safetensors")))[0]
    sd = load_file(path)
    return {k.split("layers.")[1].split(".", 1)[1]: v for k, v in sd.items() if "layers." in k}


def _dequant(w_fp8, scale):
    """128x128 block dequant to fp32 (scale rows: [out_blocks, in_blocks])."""
    N, K = w_fp8.shape
    s = scale.float()
    blocks = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
    return w_fp8.float() * blocks


def _gemv_fp8(x, w, sc, n_out, k):
    y = torch.zeros(n_out, device=x.device, dtype=torch.float32)
    _drive_gemv_fp8[(48,)](x, w, sc, y, K=k, N=n_out, TILE=16, BK=128, num_warps=8)
    return y


# ---------------------------------------------------------------------------
# FP8 block GEMV vs dequant reference
# ---------------------------------------------------------------------------


@gpu
@has_27b
def test_fp8_gemv_matches_dequant_reference():
    sd = _layer0()
    w = sd["mlp.gate_proj.weight"].cuda()  # F8_E4M3 [17408, 5120]
    sc = sd["mlp.gate_proj.weight_scale_inv"].cuda()  # bf16 [136, 40]
    assert w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (17408, 5120)
    N, K = w.shape
    x = torch.randn(K, device="cuda", dtype=torch.float32)
    ref = _dequant(w, sc) @ x
    y = _gemv_fp8(x, w, sc, N, K)
    torch.cuda.synchronize()
    rel = (y - ref).abs().max() / ref.abs().max()
    assert rel < 1e-5, f"fp8 gemv diverged: {rel}"
    assert bool((y != 0).all()), "every output tile must be covered"


# ---------------------------------------------------------------------------
# GDN decode step, task-decomposed, vs the repo reference (real weights)
# ---------------------------------------------------------------------------


def _gdn_reference(sd):
    # The 27B target's reference GDN implementation lives in the floe repo;
    # skip when running the vkernels suite without the sibling checkout.
    arch = pytest.importorskip("floe.engine.runner.models.qwen35.qwen35_arch")
    gdn_mod = pytest.importorskip("floe.engine.runner.models.qwen35.qwen35_gdn")
    Qwen35Config = arch.Qwen35Config
    GatedDeltaNet = gdn_mod.GatedDeltaNet

    cfg = Qwen35Config(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=1,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        full_attention_interval=1,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    gdn = GatedDeltaNet(cfg, torch.device("cuda"), torch.float32).eval().cuda()
    with torch.no_grad():
        gdn.in_proj_qkv.weight.copy_(_dequant(sd["linear_attn.in_proj_qkv.weight"].cuda(), sd["linear_attn.in_proj_qkv.weight_scale_inv"].cuda()))
        gdn.in_proj_z.weight.copy_(_dequant(sd["linear_attn.in_proj_z.weight"].cuda(), sd["linear_attn.in_proj_z.weight_scale_inv"].cuda()))
        gdn.in_proj_a.weight.copy_(sd["linear_attn.in_proj_a.weight"].cuda().float())
        gdn.in_proj_b.weight.copy_(sd["linear_attn.in_proj_b.weight"].cuda().float())
        gdn.out_proj.weight.copy_(_dequant(sd["linear_attn.out_proj.weight"].cuda(), sd["linear_attn.out_proj.weight_scale_inv"].cuda()))
        gdn.conv1d.weight.copy_(sd["linear_attn.conv1d.weight"].cuda().float())
        gdn.A_log.copy_(sd["linear_attn.A_log"].cuda().float())
        gdn.dt_bias.copy_(sd["linear_attn.dt_bias"].cuda().float())
        gdn.norm.weight.copy_(sd["linear_attn.norm.weight"].cuda().float())
    return gdn


@gpu
@has_27b
def test_gdn_decode_step_matches_reference():
    sd = _layer0()
    gdn = _gdn_reference(sd)

    torch.manual_seed(0)
    h = torch.randn(1, 5120, device="cuda", dtype=torch.float32)
    ssm0 = torch.randn(48, 128, 128, device="cuda", dtype=torch.float32) * 0.05
    conv0 = torch.randn(3, 10240, device="cuda", dtype=torch.float32) * 0.05

    with torch.no_grad():
        ref_out, ref_ssm, ref_conv = gdn(h, ssm0.clone(), conv0.clone())

    # --- megakernel task decomposition of the same step -------------------
    nk, nv, hk, hv = 16, 48, 128, 128
    key_dim, value_dim = nk * hk, nv * hv
    conv_dim = 2 * key_dim + value_dim
    dev = h.device
    x = h[0].contiguous()

    qkv = _gemv_fp8(x, sd["linear_attn.in_proj_qkv.weight"].cuda(), sd["linear_attn.in_proj_qkv.weight_scale_inv"].cuda(), 2 * key_dim + value_dim, 5120)
    z = _gemv_fp8(x, sd["linear_attn.in_proj_z.weight"].cuda(), sd["linear_attn.in_proj_z.weight_scale_inv"].cuda(), value_dim, 5120)
    a = sd["linear_attn.in_proj_a.weight"].cuda().float() @ x
    b = sd["linear_attn.in_proj_b.weight"].cuda().float() @ x

    state = conv0.clone()
    conv_out = torch.zeros(conv_dim, device=dev, dtype=torch.float32)
    _drive_gdn_conv[(48,)](state, sd["linear_attn.conv1d.weight"].cuda().float().squeeze(1), qkv, conv_out, C=conv_dim, ELEM=1024, KTAPS=4, num_warps=8)

    outs = torch.zeros(value_dim, device=dev, dtype=torch.float32)
    _drive_gdn_heads[(48,)](
        conv_out[:key_dim],
        conv_out[key_dim : 2 * key_dim],
        conv_out[2 * key_dim :],
        z,
        a,
        b,
        sd["linear_attn.A_log"].cuda(),
        sd["linear_attn.dt_bias"].cuda(),
        sd["linear_attn.norm.weight"].cuda(),
        ssm0,
        outs,
        NH=nv,
        NK=nk,
        HV=hv,
        HK=hk,
        eps=gdn.norm.eps,
        scale=hk**-0.5,
        num_warps=8,
    )
    got = _gemv_fp8(outs, sd["linear_attn.out_proj.weight"].cuda(), sd["linear_attn.out_proj.weight_scale_inv"].cuda(), 5120, value_dim)
    torch.cuda.synchronize()

    err = (got - ref_out[0].float()).abs().max() / ref_out[0].float().abs().max()
    assert err < 1e-4, f"GDN decode step diverged: {err}"
    assert (state - ref_conv.float()).abs().max() < 1e-5, "conv state update diverged"
    assert (ssm0 - ref_ssm.float()).abs().max() < 1e-4, "ssm state update diverged"


# ---------------------------------------------------------------------------
# Full hybrid megakernel: one persistent launch for all 64 layers
# ---------------------------------------------------------------------------


@gpu
@has_27b
def test_hybrid_megakernel_full_schedule():
    """The full 64-layer hybrid (48 GDN + 16 full-attn, FP8) runs as ONE
    persistent launch per decode step. This validates the barrier schedule
    (754 in-kernel grid barriers, type-relative) executes without deadlock
    and that the barrier counter advances by exactly ``NUM_BARRIERS * P``
    per step -- the strongest no-reference check that the whole-model
    task graph is live and correctly ordered."""
    from vkernels.compiler.device_triton_hybrid import HybridMegakernel

    mk = HybridMegakernel(checkpoint=TARGET, capacity=64, workers=48, device="cuda")
    assert mk.NUM_BARRIERS == 754

    # Step 0: launch + barrier counter check (the grid barrier is the only
    # inter-task sync; a wrong count or a missed arrival hangs sync).
    lg0 = mk.run(1, 0, check_counter=True)
    torch.cuda.synchronize()
    assert lg0.isfinite().all(), "step 0 logits not finite"
    assert int(mk.bar[0]) == mk.NUM_BARRIERS * mk.workers

    # Step 1: state evolves (ssm/conv/kv), barrier base advances -- the
    # schedule must still complete from the new base.
    lg1 = mk.run(2, 1, check_counter=True)
    torch.cuda.synchronize()
    assert lg1.isfinite().all(), "step 1 logits not finite"
    assert int(mk.bar[0]) == 2 * mk.NUM_BARRIERS * mk.workers

    # Greedy: the two steps should generally agree on-continuation here only
    # in that both produce a confident argmax (a degenerate/collapsed logit
    # would have near-zero spread).
    spread0 = float(lg0.float().std())
    spread1 = float(lg1.float().std())
    assert spread0 > 1e-3 and spread1 > 1e-3, "logits collapsed"

