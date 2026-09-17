"""GPU tests for the Triton megakernel device backend.

Chain (§15.1 levels, now including the device):

* Milestone 0 (§8.4): the grid-barrier microprogram passes on this stack —
  repeated barriers, cross-block visibility, idle-worker participation,
  repeated invocations, several worker counts — and flips the strict-mode
  capability flag;
* task/schedule fidelity: the device backend's phase and task counts match
  the compiler's schedule for the same config (17 ops/layer, §6.4 tiles);
* whole-model: the GPU megakernel matches the HF-checked NumPy oracle over
  a full-capacity sequential decode for several worker counts (fp32), and
  at the real 0.6B dims against the real checkpoint;
* §15.2 launch accounting: exactly one CUDA kernel event per step.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vkernels.compiler.device_triton import TritonMegakernel, triton_available  # noqa: E402
from vkernels.compiler.model_qwen3 import (  # noqa: E402
    QWEN3_06B,
    Qwen3KVCache,
    qwen3_reference_forward,
    random_qwen3_weights,
    tiny_qwen3_config,
)

gpu = pytest.mark.skipif(not (torch.cuda.is_available() and triton_available()), reason="requires CUDA + triton")

_HF_GLOB = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*")
real_ckpt = pytest.mark.skipif(not glob.glob(_HF_GLOB), reason="requires the local Qwen3-0.6B checkpoint")


# ===========================================================================
# Milestone 0 (§8.4)
# ===========================================================================


@gpu
def test_milestone0_verification_passes_and_flips_flag():
    from vkernels.compiler.runtime import synchronization as sync

    saved = sync.GRID_SYNC_BACKEND_VERIFIED
    sync.GRID_SYNC_BACKEND_VERIFIED = False
    try:
        report = sync.verify_milestone0(rounds=60, worker_counts=(1, 7, 48))
        assert report["P=48,idle=True"] == 60 * 48 * 3
        assert sync.GRID_SYNC_BACKEND_VERIFIED is True
    finally:
        # keep process-global gate state untouched for other test modules
        # (the strict-mode refusal test must still see the unverified flag).
        sync.GRID_SYNC_BACKEND_VERIFIED = saved


# ===========================================================================
# Schedule fidelity
# ===========================================================================


@gpu
def test_device_schedule_matches_compiled_program():
    """The Triton backend runs the compiler's phase structure (17 ops/layer,
    §6.4 tile counts, GQA head counts)."""
    from vkernels.compiler import compile_model

    cfg = tiny_qwen3_config()
    w = random_qwen3_weights(cfg)
    exe = compile_model(model_config=cfg, weights=w, workers=4)
    mk = TritonMegakernel(cfg, w, capacity=cfg.cache_capacity, workers=4, dtype=torch.float32)

    assert mk.num_phases == exe.schedule.num_phases == 1 + 17 * cfg.layers + 2
    assert mk.num_barriers == mk.num_phases - 1
    counts = mk.task_counts()
    fams = {f.op.source_location: f for f in exe.families}
    assert counts["l0_qkv"] == fams["l0_qkv"].task_count
    assert counts["l0_gate_up"] == fams["l0_gate_up"].task_count
    assert counts["l0_down"] == fams["l0_down"].task_count
    assert counts["l0_o_proj"] == fams["l0_o_proj"].task_count
    assert counts["l0_scores"] == fams["layer 0 attention scores"].task_count  # B*H query heads
    assert counts["l0_append"] == fams["layer 0 kv cache append"].task_count  # B*KVH
    assert counts["logits"] == fams["logits"].task_count


# ===========================================================================
# Whole-model: GPU megakernel vs the HF-checked NumPy oracle
# ===========================================================================


@gpu
def test_tiny_full_capacity_decode_matches_oracle():
    cfg = tiny_qwen3_config()
    w = random_qwen3_weights(cfg)
    ids = np.random.default_rng(21).integers(0, cfg.vocab, size=cfg.cache_capacity)
    for P in (1, 3, 4, 7):
        mk = TritonMegakernel(cfg, w, capacity=cfg.cache_capacity, workers=P, dtype=torch.float32)
        ocache = Qwen3KVCache(cfg)
        worst = 0.0
        for p in range(cfg.cache_capacity):
            ref, _ = qwen3_reference_forward(w, ids[p : p + 1], ocache, p, cfg)
            lg = mk.run(int(ids[p]), p, check_counter=(p == 0)).cpu().numpy()
            worst = max(worst, float(np.abs(lg - ref[0]).max()))
        kc = np.nan_to_num(mk.k_cache.cpu().numpy()).transpose(0, 2, 1, 3)[:, None]  # -> [L,1,KVH,T,D]
        vc = np.nan_to_num(mk.v_cache.cpu().numpy()).transpose(0, 2, 1, 3)[:, None]
        assert np.abs(kc - np.nan_to_num(ocache.k)).max() < 1e-4
        assert np.abs(vc - np.nan_to_num(ocache.v)).max() < 1e-4
        assert worst < 1e-5, f"P={P}: logits diverged ({worst})"
        assert mk.barrier_base == mk.num_barriers * P * cfg.cache_capacity


@gpu
@real_ckpt
def test_real_checkpoint_decode_matches_oracle():
    from vkernels.compiler.model_qwen3 import weights_from_hf

    transformers = pytest.importorskip("transformers")
    path = sorted(glob.glob(_HF_GLOB))[-1]
    hf = transformers.AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16)
    cfg = QWEN3_06B(cache_capacity=8)
    w = weights_from_hf(hf.state_dict(), cfg)
    del hf
    torch.cuda.empty_cache()

    mk = TritonMegakernel(cfg, w, capacity=8, workers=48, dtype=torch.bfloat16)
    ocache = Qwen3KVCache(cfg)
    ids = np.random.default_rng(3).integers(0, cfg.vocab, size=4)
    worst = 0.0
    for p in range(4):
        ref, _ = qwen3_reference_forward(w, ids[p : p + 1], ocache, p, cfg)
        lg = mk.run(int(ids[p]), p, check_counter=(p == 0)).cpu().numpy()
        worst = max(worst, float(np.abs(lg - ref[0]).max() / np.abs(ref[0]).max()))
    assert worst < 3e-2, f"real-checkpoint decode diverged ({worst})"


# ===========================================================================
# §15.2: launch accounting
# ===========================================================================


@gpu
def test_exactly_one_kernel_launch_per_step():
    from torch.profiler import ProfilerActivity, profile

    cfg = tiny_qwen3_config()
    mk = TritonMegakernel(cfg, random_qwen3_weights(cfg), capacity=cfg.cache_capacity, workers=4, dtype=torch.float32)
    mk.run(3, 0)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        mk.run(3, 1)
        torch.cuda.synchronize()
    n = sum(1 for e in prof.events() if str(getattr(e, "device_type", "")) == "DeviceType.CUDA" and "memcpy" not in str(getattr(e, "name", "")).lower() and "memset" not in str(getattr(e, "name", "")).lower())
    assert n == 1, f"expected exactly 1 CUDA kernel event per step, got {n}"
    # The step's tokens/positions travel as two tiny H2D copies (disclosed
    # in run()); they are not kernel launches and do not count under §15.2.


@gpu
def test_position_bounds_are_guarded():
    cfg = tiny_qwen3_config()
    mk = TritonMegakernel(cfg, random_qwen3_weights(cfg), capacity=8, workers=2, dtype=torch.float32)
    with pytest.raises(ValueError):
        mk.run(1, -1)
    with pytest.raises(ValueError):
        mk.run(1, 8)
