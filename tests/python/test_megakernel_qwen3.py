"""Qwen3-0.6B frontend tests for the megakernel compiler.

Chain of evidence for the Qwen3 path (§15.1 levels):

* IR/capture: the dense-Qwen3 body records the expected 17-ops-per-layer
  phase structure (RMSNorm, RoPE, QK-norm, GQA attention, SwiGLU);
* oracle-vs-HF: the NumPy oracle matches HF ``Qwen3ForCausalLM`` on a tiny
  faithful fixture (exact match modulo HF's fp32 RoPE tables);
* whole-model: the compiled phase schedule, executed on the CPU reference
  executor, matches the oracle across the full cache capacity for several
  worker counts, with one simulated launch per step;
* GQA precision: per-task regions map q head h to kv head h // group;
* real dims: the published 0.6B config compiles (479 phases, ~34k tasks).
"""

from __future__ import annotations

import numpy as np
import pytest

from vkernels.compiler import compile_model
from vkernels.compiler.model_qwen3 import (
    QWEN3_06B,
    Qwen3KVCache,
    build_qwen3_forward,
    qwen3_reference_forward,
    random_qwen3_weights,
    rope_tables,
    tiny_qwen3_config,
)

EXPECTED_LAYER_KINDS = [
    "rms_norm",  # ln1
    "linear",  # qkv (no bias)
    "rms_norm",  # q_norm (per head)
    "rms_norm",  # k_norm (per head)
    "rope",  # q
    "rope",  # k
    "cache_append",
    "attention_scores",
    "softmax",
    "attention_values",
    "linear",  # o_proj (input width H*D)
    "add",
    "rms_norm",  # ln2
    "linear",  # fused gate_up [C, 2F]
    "swiglu",
    "linear",  # down
    "add",
]


@pytest.fixture(scope="module")
def tiny():
    cfg = tiny_qwen3_config()
    return cfg, random_qwen3_weights(cfg)


@pytest.fixture(scope="module")
def compiled_tiny(tiny):
    cfg, w = tiny
    return compile_model(model_config=cfg, weights=w, workers=4)


def test_capture_records_qwen3_phase_structure(compiled_tiny, tiny):
    cfg, _ = tiny
    kinds = [op.kind for op in compiled_tiny.graph.ops]
    expected = ["embedding"] + EXPECTED_LAYER_KINDS * cfg.layers + ["rms_norm", "linear"]
    assert kinds == expected
    assert len(kinds) == 1 + 17 * cfg.layers + 2


def test_rope_tables_follow_hf_convention(tiny):
    cfg, _ = tiny
    cos, sin = rope_tables(cfg)
    D, S = cfg.head_dim, cfg.max_positions
    assert cos.shape == (S, D) and sin.shape == (S, D)
    # cos = cat([f, f]): both halves identical.
    assert np.allclose(cos[3, : D // 2], cos[3, D // 2 :])
    # theta=1e6: inv_freq[0]=1 -> cos(0)=1 for all positions at index 0.
    assert np.allclose(cos[:, 0], np.cos(np.arange(S) * 1.0))


def test_gqa_region_mapping(compiled_tiny, tiny):
    """Per-task regions map q head h -> kv head h // group (§10.3 precision)."""
    cfg, _ = tiny
    # kv heads = 2, group = 2: q head 3 reads kv head 1's rows only.
    from vkernels.compiler import Region

    fams = {f.family_id: f for f in compiled_tiny.families}
    scores = fams["ph08_attn_scores"]  # layer 0 attention scores
    k_cache = compiled_tiny.graph.tensor("l0_kcache")
    reads = [r for r in scores.reads(3) if r.storage_id == k_cache.storage_id]
    kv1 = Region.tile(k_cache, ((0, 1), (1, 2), (0, 16), (0, cfg.head_dim)))
    kv0 = Region.tile(k_cache, ((0, 1), (0, 1), (0, 16), (0, cfg.head_dim)))
    assert any(r.overlaps(kv1) for r in reads)
    assert not any(r.overlaps(kv0) for r in reads)


def test_compiled_schedule_matches_oracle_full_capacity(compiled_tiny, tiny):
    cfg, w = tiny
    ids_seq = np.random.default_rng(5).integers(0, cfg.vocab, size=cfg.cache_capacity)
    worst = 0.0
    for workers in (1, 3, 4):
        cache_ref, cache_run = Qwen3KVCache(cfg), Qwen3KVCache(cfg)
        for p in range(cfg.cache_capacity):
            ids = ids_seq[p : p + 1]
            ref, _ = qwen3_reference_forward(w, ids, cache_ref, p, cfg)
            lg, tr = compiled_tiny.run(ids, cache_run, p, workers=workers)
            assert tr.kernel_launches == 1  # §15.2: one kernel event per step
            assert tr.grid_barriers == compiled_tiny.schedule.num_phases
            worst = max(worst, float(np.abs(lg - ref).max()))
            # §5.3: NaN tails must never be read.
            assert np.isnan(cache_run.k[:, :, :, p + 1 :, :]).all()
        assert np.allclose(cache_run.k, cache_ref.k, equal_nan=True)
        assert np.allclose(cache_run.v, cache_ref.v, equal_nan=True)
    assert worst < 1e-9


def test_real_0_6b_dims_compile():
    """The published Qwen3-0.6B config compiles (shapes only; no CPU exec)."""
    exe = compile_model(model_config=QWEN3_06B(cache_capacity=128), workers=48)
    r = exe.report
    assert r.num_phases == 1 + 17 * 28 + 2 == 479
    # Per-layer GEMM tile counts at real dims (16x16 tiles, full-K):
    fams = {f.op.source_location: f for f in exe.families}
    assert fams["l0_qkv"].task_count == 4096 // 16  # [1,1024]x[1024,4096]
    assert fams["l0_gate_up"].task_count == 6144 // 16
    assert fams["l0_down"].task_count == 1024 // 16
    assert fams["l0_o_proj"].task_count == 1024 // 16
    assert fams["logits"].task_count == 151936 // 16
    assert fams["layer 0 attention scores"].task_count == 16  # B*H query heads
    assert fams["layer 0 kv cache append"].task_count == 8  # B*KVH
    # GQA halves the KV cache bytes vs an MHA layout at hidden width.
    assert r.kv_cache_bytes == 2 * 28 * 1 * 128 * (8 * 128) * 4
    # Generated source stays syntactically valid at this size.
    compile(exe.source, "<generated>", "exec")


def test_qwen3_oracle_matches_hf_implementation(tiny):
    """The oracle is checked against HF Qwen3ForCausalLM itself (tiny fixture).

    The residual ~1e-7 is HF's fp32 RoPE tables (verified: rounding our
    fp64 tables to fp32 reproduces the same gap). Semantics — RMSNorm,
    QK-norm, rotate-half RoPE, GQA, SwiGLU, tied head — match exactly.
    """
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    cfg, w = tiny

    hf_cfg = transformers.Qwen3Config(
        vocab_size=cfg.vocab,
        hidden_size=cfg.hidden,
        num_hidden_layers=cfg.layers,
        num_attention_heads=cfg.heads,
        num_key_value_heads=cfg.kv_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate,
        rms_norm_eps=cfg.rms_eps,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_positions,
        tie_word_embeddings=True,
        attention_bias=False,
    )
    model = transformers.Qwen3ForCausalLM(hf_cfg).to(torch.float64).eval()
    sd = {
        "model.embed_tokens.weight": torch.from_numpy(w.token_emb),
        "model.norm.weight": torch.from_numpy(w.final_gamma),
    }
    H, KVH, D, F = cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.intermediate
    for li, lw in enumerate(w.layers):
        p = f"model.layers.{li}."
        qkv = lw.qkv_w.T
        sd[p + "input_layernorm.weight"] = torch.from_numpy(lw.ln1_gamma)
        sd[p + "self_attn.q_proj.weight"] = torch.from_numpy(qkv[: H * D])
        sd[p + "self_attn.k_proj.weight"] = torch.from_numpy(qkv[H * D : H * D + KVH * D])
        sd[p + "self_attn.v_proj.weight"] = torch.from_numpy(qkv[H * D + KVH * D :])
        sd[p + "self_attn.q_norm.weight"] = torch.from_numpy(lw.q_norm_gamma)
        sd[p + "self_attn.k_norm.weight"] = torch.from_numpy(lw.k_norm_gamma)
        sd[p + "self_attn.o_proj.weight"] = torch.from_numpy(lw.o_proj_w.T)
        sd[p + "post_attention_layernorm.weight"] = torch.from_numpy(lw.ln2_gamma)
        sd[p + "mlp.gate_proj.weight"] = torch.from_numpy(lw.gate_up_w.T[:F])
        sd[p + "mlp.up_proj.weight"] = torch.from_numpy(lw.gate_up_w.T[F:])
        sd[p + "mlp.down_proj.weight"] = torch.from_numpy(lw.down_w.T)
    model.load_state_dict(sd, strict=False)

    ids = np.random.default_rng(11).integers(0, cfg.vocab, size=5)
    with torch.no_grad():
        hf_logits = model(input_ids=torch.from_numpy(ids)[None]).logits[0].numpy()
    cache = Qwen3KVCache(cfg)
    ours = np.stack([qwen3_reference_forward(w, ids[p : p + 1], cache, p, cfg)[0][0] for p in range(5)])
    rel = np.abs(ours - hf_logits).max() / np.abs(hf_logits).max()
    assert rel < 1e-6, f"oracle vs HF diverged: {rel}"


def test_body_runs_under_recording_backend_only(tiny):
    """The same body captures under a fresh recorder (frontend reusability)."""
    from vkernels.compiler import RecordingBackend, capture_model
    from vkernels.compiler.model_qwen3 import Qwen3ModelArgs

    cfg, _ = tiny
    recorder = RecordingBackend()
    position = recorder.define_position(cfg.cache_capacity)
    ids = recorder.external_tensor("ids", (cfg.batch,), storage_id=10**6)
    args = Qwen3ModelArgs(recorder, cfg)
    graph, _ = capture_model(build_qwen3_forward, args, ids, position, cfg, backend=recorder)
    assert len(graph.ops) == 1 + 17 * cfg.layers + 2
    # RoPE consumed the symbolic position and stayed runtime (§4.2).
    assert graph.scalars["p"].name == "p"
