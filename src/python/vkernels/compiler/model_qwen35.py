"""Qwen3.5 hybrid compiler frontend for the megakernel compiler (issue #102 Stage 1).

The model definition (config, weights, pools, fp64 whole-step mirror) lives in
:mod:`.qwen35_arch` — this module provides the compiler-specific capture
layer: ``Qwen35ModelArgs`` (symbolic externals incl. per-row positions #93,
paged slot tables #94 and the GDN state pools) and ``build_qwen35_forward``
(the hybrid decode-step body expressed against the §4.1 ``ops`` interface so
the capture → legality → lowering → schedule → memory pipeline applies).

Stage-1 scope: Qwen3.5 hybrid decode only — rope partial/NeoX (#88),
gdn_conv (#89), gdn_delta (#90), gemv_fp8 projections (#91), per-row valid
lengths (#93), slot-table indirection (#94), gated attention_values (#92).
Stage 2/3 families (DeepSeek-V4, GLM-5.3) are out of scope here.
"""

from __future__ import annotations

from .capture import RecordingBackend
from .model_gpt2 import _stable_storage_id
from .operator_ir import F8_E4M3
from .qwen35_arch import Qwen35Config

__all__ = ["Qwen35ModelArgs", "build_qwen35_forward"]


class Qwen35ModelArgs:
    """External tensors for capture: parameters, rope tables, state pools."""

    def __init__(self, ops: RecordingBackend, config: Qwen35Config):
        self.config = config
        ops._externals = getattr(ops, "_externals", {})
        sid = _stable_storage_id
        C = config.hidden

        self.token = ops.external_tensor("token_emb", (config.vocab, C), storage_id=sid("token_emb"))
        self.final_gamma = ops.external_tensor("final_ln_gamma", (C,), storage_id=sid("final_ln_gamma"))
        half = config.rotary_dim // 2
        self.rope_cos = ops.external_tensor("rope_cos", (config.max_positions, half), storage_id=sid("rope_cos"))
        self.rope_sin = ops.external_tensor("rope_sin", (config.max_positions, half), storage_id=sid("rope_sin"))

        # #93: per-row decode positions arrive as the ``position`` argument
        # to ``build_qwen35_forward`` (compile_model defines them via
        # ``define_row_positions``) — every position consumer below takes
        # that row-form tensor, so the whole step is ragged-batch safe.
        self.row_positions = None

        # paged plumbing (#94): one shared slot table (token → pool slot),
        # per-paged-FA-layer K/V pools [slots, KVH, D].
        self.slot_table = ops.external_tensor(
            "slot_table", (config.batch, config.cache_capacity), storage_id=sid("slot_table")
        )
        n_fa_paged = 1  # first FA layer is paged (see build_qwen35_forward)
        self.k_pool = ops.external_tensor(
            "k_pool", (n_fa_paged, config.n_slots, config.kv_heads, config.head_dim), storage_id=sid("k_pool")
        )
        self.v_pool = ops.external_tensor(
            "v_pool", (n_fa_paged, config.n_slots, config.kv_heads, config.head_dim), storage_id=sid("v_pool")
        )

        self.layers = []
        for li, lt in enumerate(config.layer_types):
            views = {}
            if lt == "gdn":
                CC = config.gdn_kv_dim
                shapes = {
                    "ln1_gamma": (C,),
                    "in_a_w": (C, config.gdn_v_heads),
                    "in_b_w": (C, config.gdn_v_heads),
                    "conv_w": (CC, config.conv_kernel),
                    "A_log": (config.gdn_v_heads,),
                    "dt_bias": (config.gdn_v_heads,),
                    "norm_w": (config.head_v_dim,),
                    "ln2_gamma": (C,),
                    "gate_up_w": (C, 2 * config.intermediate),
                    "down_w": (config.intermediate, C),
                    "o_proj_w": (config.gdn_v_heads * config.head_v_dim, C),
                }
                for fname, shape in shapes.items():
                    views[fname] = ops.external_tensor(f"l{li}_{fname}", shape, storage_id=sid(f"l{li}_{fname}"))
                views["in_qkvz_bytes"] = ops.external_tensor(
                    f"l{li}_in_qkvz_bytes", (CC, C), F8_E4M3, storage_id=sid(f"l{li}_in_qkvz_bytes")
                )
                views["in_qkvz_scale"] = ops.external_tensor(
                    f"l{li}_in_qkvz_scale",
                    (-(-CC // config.fp8_quant_block), C // config.fp8_quant_block),
                    storage_id=sid(f"l{li}_in_qkvz_scale"),
                )
            else:
                D = config.head_dim
                shapes = {
                    "ln1_gamma": (C,),
                    "q_norm_gamma": (D,),
                    "k_norm_gamma": (D,),
                    "ln2_gamma": (C,),
                    "gate_up_w": (C, 2 * config.intermediate),
                    "down_w": (config.intermediate, C),
                    "o_proj_w": (config.heads * D, C),
                }
                for fname, shape in shapes.items():
                    views[fname] = ops.external_tensor(f"l{li}_{fname}", shape, storage_id=sid(f"l{li}_{fname}"))
                views["qkv_gate_bytes"] = ops.external_tensor(
                    f"l{li}_qkv_gate_bytes", (config.fa_proj_dim, C), F8_E4M3, storage_id=sid(f"l{li}_qkv_gate_bytes")
                )
                views["qkv_gate_scale"] = ops.external_tensor(
                    f"l{li}_qkv_gate_scale",
                    (-(-config.fa_proj_dim // config.fp8_quant_block), C // config.fp8_quant_block),
                    storage_id=sid(f"l{li}_qkv_gate_scale"),
                )
            self.layers.append(type("LayerViews", (), views))

        # GDN persistent pools: stacked [n_gdn, ...] with per-layer narrow views.
        n_gdn = len(config.gdn_layers)
        self.conv_state = ops.external_tensor(
            "conv_state", (n_gdn, config.batch, config.conv_kernel - 1, config.gdn_kv_dim),
            storage_id=sid("conv_state"),
        )
        self.ssm_state = ops.external_tensor(
            "ssm_state", (n_gdn, config.batch, config.gdn_v_heads, config.head_v_dim, config.head_k_dim),
            storage_id=sid("ssm_state"),
        )
        # dense KV caches for the gated (non-paged) FA layers
        n_fa_dense = len(config.fa_layers) - n_fa_paged
        self.k_cache_dense = ops.external_tensor(
            "k_cache_dense", (max(n_fa_dense, 1), config.batch, config.kv_heads, config.cache_capacity, config.head_dim),
            storage_id=sid("k_cache_dense"),
        )
        self.v_cache_dense = ops.external_tensor(
            "v_cache_dense", (max(n_fa_dense, 1), config.batch, config.kv_heads, config.cache_capacity, config.head_dim),
            storage_id=sid("v_cache_dense"),
        )


def build_qwen35_forward(ops, args, ids, position, config: Qwen35Config):
    """One Qwen3.5 hybrid decode step: F(W, ids, pools, row_pos) -> logits.

    ``position`` is the #93 row-positions tensor — every position consumer
    (embedding, rope, append, attention) records per-row valid-length
    regions, so the step is ragged-batch safe end to end.
    """
    cfg = config
    B = cfg.batch
    NK, HK, NV, HV = cfg.gdn_k_heads, cfg.head_k_dim, cfg.gdn_v_heads, cfg.head_v_dim
    H, KVH, D = cfg.heads, cfg.kv_heads, cfg.head_dim
    F = cfg.intermediate
    CC = cfg.gdn_kv_dim

    hidden = ops.embedding(ids, args.token, None, position)  # token only (rope carries position)

    gdn_i = 0
    fa_dense_i = 0
    for li, lt in enumerate(cfg.layer_types):
        lv = args.layers[li]
        if lt == "gdn":
            # --- projections: fp8 qkvz + small a/b on the normed hidden ---
            n1 = ops.rms_norm(hidden, lv.ln1_gamma, cfg.rms_eps, name=f"l{li}_rms1")
            qkvz = ops.linear_fp8(
                n1, lv.in_qkvz_bytes, lv.in_qkvz_scale, quant_block=cfg.fp8_quant_block, name=f"l{li}_in_qkvz"
            )  # [B, CC]
            a = ops.linear(n1, lv.in_a_w, None, name=f"l{li}_in_a")  # [B, NV]
            b = ops.linear(n1, lv.in_b_w, None, name=f"l{li}_in_b")  # [B, NV]
            # --- short conv over the packed row, persistent state RMW ---
            conv_view = ops.view_of(
                ops.narrow(args.conv_state, f"l{li}_conv_l", axis=0, start=gdn_i, length=1),
                f"l{li}_conv_state", (B, cfg.conv_kernel - 1, CC),
            )
            conv_out, conv_post = ops.gdn_conv(conv_view, qkvz, lv.conv_w, layer=li)
            # --- q|k|v|z views of the post-conv row (uniform head width:
            # HV == HK in the fixture, so one 3D tiling covers all four) ---
            conv3 = ops.view_of(conv_out, f"l{li}_conv3", (B, 2 * NK + 2 * NV, HK))
            q = ops.view_of(ops.narrow(conv3, f"l{li}_q_n", axis=1, start=0, length=NK), f"l{li}_q", (B, NK, HK))
            k = ops.view_of(ops.narrow(conv3, f"l{li}_k_n", axis=1, start=NK, length=NK), f"l{li}_k", (B, NK, HK))
            v = ops.view_of(
                ops.narrow(conv3, f"l{li}_v_n", axis=1, start=2 * NK, length=NV), f"l{li}_v", (B, NV, HV)
            )
            z = ops.view_of(
                ops.narrow(conv3, f"l{li}_z_n", axis=1, start=2 * NK + NV, length=NV), f"l{li}_z", (B, NV, HV)
            )
            # --- gated delta rule over the fp32 SSM pool ---
            ssm_view = ops.view_of(
                ops.narrow(args.ssm_state, f"l{li}_ssm_l", axis=0, start=gdn_i, length=1),
                f"l{li}_ssm_state", (B, NV, HV, HK),
            )
            out, ssm_post = ops.gdn_delta(
                ssm_view, q, k, v, z, a, b, lv.A_log, lv.dt_bias, lv.norm_w,
                layer=li, scale=cfg.delta_scale, eps=cfg.delta_eps,
            )
            attn = ops.linear(ops.view_of(out, f"l{li}_gdn_flat", (B, NV * HV)), lv.o_proj_w, None, name=f"l{li}_gdn_o_proj")
            hidden = ops.add(hidden, attn, name=f"l{li}_gdn_res")
            gdn_i += 1
        else:
            # --- chunked q|gate|k|v projection (fp8) ---
            n1 = ops.rms_norm(hidden, lv.ln1_gamma, cfg.rms_eps, name=f"l{li}_rms1")
            qkv_gate = ops.linear_fp8(
                n1, lv.qkv_gate_bytes, lv.qkv_gate_scale, quant_block=cfg.fp8_quant_block, name=f"l{li}_qkv_gate"
            )  # [B, (2H+2KVH)*D]
            qkv3 = ops.view_of(qkv_gate, f"l{li}_qkv3", (B, 2 * H + 2 * KVH, D))
            q = ops.view_of(ops.narrow(qkv3, f"l{li}_q_n", axis=1, start=0, length=H), f"l{li}_q", (B, H, D))
            gate = ops.view_of(
                ops.narrow(qkv3, f"l{li}_gate_n", axis=1, start=H, length=H), f"l{li}_gate", (B, H, D)
            )
            k = ops.view_of(
                ops.narrow(qkv3, f"l{li}_k_n", axis=1, start=2 * H, length=KVH), f"l{li}_k", (B, KVH, D)
            )
            v = ops.view_of(
                ops.narrow(qkv3, f"l{li}_v_n", axis=1, start=2 * H + KVH, length=KVH), f"l{li}_v", (B, KVH, D)
            )
            # QK-norm, then partial NeoX rope at the per-row position
            q = ops.rms_norm(q, lv.q_norm_gamma, cfg.rms_eps, name=f"l{li}_qnorm")
            k = ops.rms_norm(k, lv.k_norm_gamma, cfg.rms_eps, name=f"l{li}_knorm")
            q = ops.rope(
                q, args.rope_cos, args.rope_sin, position, layer=li, which="q",
                rotary_dim=cfg.rotary_dim, convention="neox_partial",
            )
            k = ops.rope(
                k, args.rope_cos, args.rope_sin, position, layer=li, which="k",
                rotary_dim=cfg.rotary_dim, convention="neox_partial",
            )
            if li == cfg.fa_layers[0]:
                # --- paged plumbing (#94): slot-table indirection ---
                k_view = ops.view_of(
                    ops.narrow(args.k_pool, f"l{li}_kp_l", axis=0, start=0, length=1),
                    f"l{li}_kpool", (cfg.n_slots, KVH, D),
                )
                v_view = ops.view_of(
                    ops.narrow(args.v_pool, f"l{li}_vp_l", axis=0, start=0, length=1),
                    f"l{li}_vpool", (cfg.n_slots, KVH, D),
                )
                k_post, v_post = ops.cache_append_paged(k_view, v_view, args.slot_table, k, v, position, layer=li)
                scores = ops.attention_scores_paged(
                    q, k_post, args.slot_table, position, scale=cfg.attention_scale, layer=li, kv_heads=KVH
                )
                probs = ops.softmax(scores, position, layer=li)
                ctx = ops.attention_values_paged(probs, v_post, args.slot_table, position, layer=li, kv_heads=KVH)
            else:
                # --- dense cache + fused sigmoid output gate (#92) ---
                k_view = ops.view_of(
                    ops.narrow(args.k_cache_dense, f"l{li}_kc_l", axis=0, start=fa_dense_i, length=1),
                    f"l{li}_kcache", (B, KVH, cfg.cache_capacity, D),
                )
                v_view = ops.view_of(
                    ops.narrow(args.v_cache_dense, f"l{li}_vc_l", axis=0, start=fa_dense_i, length=1),
                    f"l{li}_vcache", (B, KVH, cfg.cache_capacity, D),
                )
                k_post, v_post = ops.cache_append(k_view, v_view, k, v, position, layer=li)
                scores = ops.attention_scores(q, k_post, position, scale=cfg.attention_scale, layer=li, kv_heads=KVH)
                probs = ops.softmax(scores, position, layer=li)
                ctx = ops.attention_values(probs, v_post, position, layer=li, kv_heads=KVH, gate=gate)
            attn = ops.linear(ops.view_of(ctx, f"l{li}_ctx_flat", (B, H * D)), lv.o_proj_w, None, name=f"l{li}_o_proj")
            hidden = ops.add(hidden, attn, name=f"l{li}_attn_res")
            if li != cfg.fa_layers[0]:
                fa_dense_i += 1  # only the dense-gated branch consumes the dense-cache slot

        # --- SwiGLU MLP (shared shape across layer kinds) ---
        n2 = ops.rms_norm(hidden, lv.ln2_gamma, cfg.rms_eps, name=f"l{li}_rms2")
        gu = ops.linear(n2, lv.gate_up_w, None, name=f"l{li}_gate_up")  # [B, 2F]
        gu3 = ops.view_of(gu, f"l{li}_gu3", (B, 2, F))
        gate_mlp = ops.view_of(ops.narrow(gu3, f"l{li}_mlpgate_n", axis=1, start=0, length=1), f"l{li}_mlp_gate", (B, F))
        up = ops.view_of(ops.narrow(gu3, f"l{li}_up_n", axis=1, start=1, length=1), f"l{li}_mlp_up", (B, F))
        act = ops.swiglu(gate_mlp, up, name=f"l{li}_swiglu")
        down = ops.linear(act, lv.down_w, None, name=f"l{li}_down")
        hidden = ops.add(hidden, down, name=f"l{li}_mlp_res")

    hidden = ops.rms_norm(hidden, args.final_gamma, cfg.rms_eps, name="final_rms")
    tied_w = ops.transpose_view(args.token, "token_emb.T")  # [C, V], tied head
    return ops.linear(hidden, tied_w, None, name="logits")
