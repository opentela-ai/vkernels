"""Contract tests for the AITER bridge (``vkernels.torch_ops.aiter_ops``).

Pins fix #3 (OpNotEligible migration): every public op raises
``OpNotEligible`` on an eligibility miss — aiter unavailable, CPU tensor,
wrong dtype/shape/contiguity — instead of silently returning ``None``, and
genuine aiter kernel failures are no longer swallowed into ``None``.
The availability probes (``aiter_available`` / ``available`` / ``report``)
stay non-raising.

Everything here runs CPU-safely: the structural checks (dtype, shape,
contiguity) fire before the aiter-availability and CUDA-device checks, so
each validation path is reachable with CPU tensors. The aiter module
itself is monkeypatched (present/absent) so the tests are deterministic on
any host, ROCm or not. Kernel-launch parity needs a gfx942 device and is
covered by ``bench/bench_aiter_ab.py`` on the MI300A box — nothing here
touches a real device.
"""

from __future__ import annotations

import types

import pytest
import torch

from vkernels.torch_ops import aiter_ops
from vkernels.torch_ops._dispatch import OpNotEligible

# ---------------------------------------------------------------------------
# fixtures: pin the aiter probe so tests are host-independent
# ---------------------------------------------------------------------------


@pytest.fixture
def no_aiter(monkeypatch):
    """aiter importable nowhere: the availability probe returns None."""
    monkeypatch.setattr(aiter_ops, "_aiter", lambda: None)


@pytest.fixture
def fake_aiter(monkeypatch):
    """aiter 'available' (dummy module) but no CUDA device around."""
    monkeypatch.setattr(aiter_ops, "_aiter", lambda: types.ModuleType("aiter"))


# ---------------------------------------------------------------------------
# structurally valid CPU arguments per op (pass every dtype/shape/contiguity
# check, so the raise comes from the aiter/CUDA eligibility checks)
# ---------------------------------------------------------------------------


def mhc_pre_args():
    residual = torch.zeros(2, 4, 256, dtype=torch.bfloat16)
    fn = torch.zeros(2 * 4 + 4 * 4, 4 * 256, dtype=torch.float32)
    hc_scale = torch.zeros(4, dtype=torch.float32)
    hc_base = torch.zeros(4, dtype=torch.float32)
    return dict(residual=residual, fn=fn, hc_scale=hc_scale, hc_base=hc_base)


def mhc_post_args():
    return dict(
        x=torch.zeros(2, 256, dtype=torch.bfloat16),
        residual=torch.zeros(2, 4, 256, dtype=torch.bfloat16),
        post_layer_mix=torch.zeros(2, 4, 1, dtype=torch.float32),
        comb_res_mix=torch.zeros(2, 4, 4, dtype=torch.float32),
    )


def quant_args():
    return dict(x=torch.zeros(3, 128, dtype=torch.bfloat16))


def align_args():
    return dict(topk_ids=torch.zeros(5, 3, dtype=torch.int32),
                num_experts=8, block_size=32)


def stage_args():
    sorted_token_ids = torch.zeros(8, dtype=torch.int32)
    sorted_expert_ids = torch.zeros(2, dtype=torch.int32)
    num_valid_ids = torch.zeros(1, dtype=torch.int32)
    return sorted_token_ids, sorted_expert_ids, num_valid_ids


def stage1_args():
    s, e, v = stage_args()
    return dict(x_q=torch.zeros(6, 256, dtype=torch.bfloat16),
                w1=torch.zeros(4, 512, 256),
                w2=torch.zeros(4, 256, 256),
                sorted_token_ids=s, sorted_expert_ids=e, num_valid_ids=v,
                topk=2)


def stage2_args():
    s, e, v = stage_args()
    return dict(inter_q=torch.zeros(6, 256, dtype=torch.bfloat16),
                w1=torch.zeros(4, 512, 256),
                w2=torch.zeros(4, 256, 256),
                sorted_token_ids=s, sorted_expert_ids=e, num_valid_ids=v,
                topk=2)


def experts_args():
    def swiglu(g, u):
        return g * u

    return dict(x=torch.zeros(3, 256, dtype=torch.bfloat16),
                gate_up=torch.zeros(4, 512, 256),
                gate_up_scale=torch.zeros(4, 4, 2),
                down=torch.zeros(4, 256, 128),
                down_scale=torch.zeros(4, 2, 1),
                topk_index=torch.zeros(3, 2, dtype=torch.int64),
                topk_weights=torch.zeros(3, 2),
                swiglu=swiglu)


OPS = [
    ("aiter_mhc_pre", aiter_ops.aiter_mhc_pre, mhc_pre_args),
    ("aiter_mhc_post", aiter_ops.aiter_mhc_post, mhc_post_args),
    ("per_group_quant_fp8", aiter_ops.per_group_quant_fp8, quant_args),
    ("moe_align_block_size", aiter_ops.moe_align_block_size, align_args),
    ("ck_moe_stage1", aiter_ops.ck_moe_stage1, stage1_args),
    ("ck_moe_stage2", aiter_ops.ck_moe_stage2, stage2_args),
    ("fp8_blockscale_experts", aiter_ops.fp8_blockscale_experts, experts_args),
]


# ---------------------------------------------------------------------------
# probes never raise
# ---------------------------------------------------------------------------


def test_probes_never_raise():
    assert isinstance(aiter_ops.aiter_available(), bool)
    assert isinstance(aiter_ops.available(), bool)
    report = aiter_ops.report()
    assert isinstance(report, dict)
    assert report["aiter"] is aiter_ops.aiter_available()


def test_aiter_available_false_when_probe_none(no_aiter):
    assert aiter_ops.aiter_available() is False


def test_aiter_available_true_when_probe_hits(fake_aiter):
    assert aiter_ops.aiter_available() is True


# ---------------------------------------------------------------------------
# the OpNotEligible migration, op by op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,op,args", OPS, ids=[o[0] for o in OPS])
def test_missing_aiter_raises_op_not_eligible(no_aiter, name, op, args):
    """aiter unavailable -> OpNotEligible (was: silent None)."""
    with pytest.raises(OpNotEligible, match="aiter is not importable"):
        op(**args())


@pytest.mark.parametrize("name,op,args", OPS, ids=[o[0] for o in OPS])
def test_cpu_tensor_raises_op_not_eligible(fake_aiter, name, op, args):
    """aiter 'available' but CPU tensors -> OpNotEligible (was: silent None)."""
    with pytest.raises(OpNotEligible, match="CUDA tensor"):
        op(**args())


def test_op_not_eligible_dual_inherits():
    """The contract's dual ValueError/TypeError inheritance holds here too."""
    assert issubclass(OpNotEligible, ValueError)
    assert issubclass(OpNotEligible, TypeError)


# ---------------------------------------------------------------------------
# per-op structural checks (all CPU-evaluable, deterministic messages)
# ---------------------------------------------------------------------------


def test_mhc_pre_rejects_wrong_residual_dtype(fake_aiter):
    a = mhc_pre_args()
    a["residual"] = a["residual"].to(torch.float16)
    with pytest.raises(OpNotEligible, match="residual must be bf16"):
        aiter_ops.aiter_mhc_pre(**a)


def test_mhc_pre_rejects_wrong_fn_shape(fake_aiter):
    a = mhc_pre_args()
    a["fn"] = a["fn"][:, :-1]  # [2hc+hc^2, hc*D - 1]
    with pytest.raises(OpNotEligible, match="fn must be"):
        aiter_ops.aiter_mhc_pre(**a)


def test_mhc_pre_rejects_wrong_fn_dtype(fake_aiter):
    a = mhc_pre_args()
    a["fn"] = a["fn"].to(torch.float64)
    with pytest.raises(OpNotEligible, match="fn must be fp32"):
        aiter_ops.aiter_mhc_pre(**a)


def test_mhc_pre_rejects_non_contiguous(fake_aiter):
    a = mhc_pre_args()
    a["residual"] = a["residual"].transpose(0, 1)  # [hc, n, D], non-contiguous
    assert not a["residual"].is_contiguous()
    with pytest.raises(OpNotEligible, match="contiguous"):
        aiter_ops.aiter_mhc_pre(**a)


def test_mhc_post_rejects_wrong_x_dtype(fake_aiter):
    a = mhc_post_args()
    a["x"] = a["x"].to(torch.float32)
    with pytest.raises(OpNotEligible, match="x must be bf16"):
        aiter_ops.aiter_mhc_post(**a)


def test_mhc_post_rejects_undivisible_hidden(fake_aiter):
    a = mhc_post_args()  # D=256 is fine; rebuild with D=100
    a = dict(
        x=torch.zeros(2, 100, dtype=torch.bfloat16),
        residual=torch.zeros(2, 4, 100, dtype=torch.bfloat16),
        post_layer_mix=torch.zeros(2, 4, 1, dtype=torch.float32),
        comb_res_mix=torch.zeros(2, 4, 4, dtype=torch.float32),
    )
    with pytest.raises(OpNotEligible, match="256"):
        aiter_ops.aiter_mhc_post(**a)


def test_mhc_post_rejects_wrong_out(fake_aiter):
    a = mhc_post_args()
    a["out"] = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
    with pytest.raises(OpNotEligible, match="out must be"):
        aiter_ops.aiter_mhc_post(**a)


def test_quant_rejects_undivisible_last_dim(fake_aiter):
    with pytest.raises(OpNotEligible, match="multiple"):
        aiter_ops.per_group_quant_fp8(torch.zeros(3, 100, dtype=torch.bfloat16))


def test_quant_rejects_non_contiguous(fake_aiter):
    x = torch.zeros(3, 256, dtype=torch.bfloat16)[:, ::2]
    assert not x.is_contiguous()
    with pytest.raises(OpNotEligible, match="contiguous"):
        aiter_ops.per_group_quant_fp8(x)


def test_align_rejects_wrong_rank(fake_aiter):
    with pytest.raises(OpNotEligible, match=r"\[T, topk\]"):
        aiter_ops.moe_align_block_size(torch.zeros(5, dtype=torch.int32), 8, 32)


def test_stage1_rejects_wrong_weight_rank(fake_aiter):
    a = stage1_args()
    a["w1"] = a["w1"][0]
    with pytest.raises(OpNotEligible, match="3-D"):
        aiter_ops.ck_moe_stage1(**a)


def test_stage1_rejects_odd_gate_up_dim(fake_aiter):
    a = stage1_args()
    a["w1"] = torch.zeros(4, 511, 256)
    with pytest.raises(OpNotEligible, match="even"):
        aiter_ops.ck_moe_stage1(**a)


def test_experts_rejects_wrong_x_dtype(fake_aiter):
    a = experts_args()
    a["x"] = a["x"].to(torch.float32)
    with pytest.raises(OpNotEligible, match="x must be bf16"):
        aiter_ops.fp8_blockscale_experts(**a)


def test_experts_rejects_expert_count_mismatch(fake_aiter):
    a = experts_args()
    a["down"] = torch.zeros(5, 256, 128)
    with pytest.raises(OpNotEligible, match="E=4"):
        aiter_ops.fp8_blockscale_experts(**a)


def test_experts_rejects_undivisible_dims(fake_aiter):
    a = experts_args()
    a["x"] = torch.zeros(3, 100, dtype=torch.bfloat16)
    with pytest.raises(OpNotEligible, match="multiples of 128"):
        aiter_ops.fp8_blockscale_experts(**a)


def test_experts_rejects_wrong_topk_weights_shape(fake_aiter):
    a = experts_args()
    a["topk_weights"] = torch.zeros(3, 3)
    with pytest.raises(OpNotEligible, match="topk_weights"):
        aiter_ops.fp8_blockscale_experts(**a)


# ---------------------------------------------------------------------------
# gfx942-only: real kernel launches / numerics parity.
# Covered on the MI300A harness by bench/bench_aiter_ab.py; nothing to run
# here (importing aiter off-ROCm fails, and the launches need a real device).
# ---------------------------------------------------------------------------
