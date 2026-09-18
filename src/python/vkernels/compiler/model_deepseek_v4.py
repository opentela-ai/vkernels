"""DeepSeek-V4 compiler frontend for the megakernel compiler (issue #102 Stage 2).

The model definition (config, weights, paged/entry pools, fp64 whole-step
mirror) lives in :mod:`.deepseek_v4_arch` — this module provides the
compiler-specific capture layer: ``DeepseekV4ModelArgs`` (symbolic externals:
per-row positions #93, global+local slot tables #94, latent/entry pools, the
per-step compressor window/gates, MoE routing scratch) and
``build_deepseek_v4_forward`` (the MLA + DSA + MoE + mHC decode-step body
expressed against the §4.1 ``ops`` interface).

Per layer, in program order: mhc_pre → compressor_append (boundary rows) →
paged latent append (#94) → indexer_scores + index_topk (block bias) →
mla_scores/mla_values + conjugate_rope + grouped o-proj (#95) → mhc_post;
then the second mHC pair wraps the routed-MoE sublayer (#98). The head is a
flattened-stream RMSNorm + tied linear over the token embedding table.
"""

from __future__ import annotations

from .capture import RecordingBackend
from .deepseek_v4_arch import DeepseekV4Config
from .model_gpt2 import _stable_storage_id
from .operator_ir import F32, I32

__all__ = ["DeepseekV4ModelArgs", "build_deepseek_v4_forward"]


class DeepseekV4ModelArgs:
    """External tensors for capture: parameters, pools, per-step inputs."""

    def __init__(self, ops: RecordingBackend, config: DeepseekV4Config):
        self.config = config
        ops._externals = getattr(ops, "_externals", {})
        sid = _stable_storage_id
        B, C, D = config.batch, config.hidden, config.latent_dim
        S, L = config.cache_capacity, config.layers
        H, HI = config.heads, config.index_heads
        E, I = config.n_experts, config.moe_intermediate
        Is = config.shared_intermediate
        M = config.entry_capacity
        R = config.entries_per_series

        self.token = ops.external_tensor("token_emb", (config.vocab, config.hc * C), F32, storage_id=sid("token_emb"))
        self.final_gamma = ops.external_tensor("final_ln_gamma", (config.hc * C,), F32, storage_id=sid("final_ln_gamma"))
        self.rope_cos = ops.external_tensor("rope_cos", (config.max_positions, D // 2), F32, storage_id=sid("rope_cos"))
        self.rope_sin = ops.external_tensor("rope_sin", (config.max_positions, D // 2), F32, storage_id=sid("rope_sin"))

        # #93 row positions arrive via ``build_deepseek_v4_forward``'s
        # ``position`` argument (compile_model defines them per-row).
        self.row_positions = None

        # #94 slot tables: global (paged appends) + row-local (mla gathers).
        self.slot_table_global = ops.external_tensor("slot_table_global", (B, S), I32, storage_id=sid("slot_table_global"))
        self.slot_table_local = ops.external_tensor("slot_table_local", (B, S), I32, storage_id=sid("slot_table_local"))

        # paged latent pools (token-slot-major, KVH=1): [layers, B*S, 1, D]
        self.latent_k = ops.external_tensor("latent_k", (L, B * S, 1, D), F32, storage_id=sid("latent_k"))
        self.latent_v = ops.external_tensor("latent_v", (L, B * S, 1, D), F32, storage_id=sid("latent_v"))

        # DSA compressor pools (#96): per-layer two-series entry pool + series
        # state (the [B, le=1, 2, R, D] layout keeps each layer's slab
        # contiguous so the flat [B, M, D] entry view is a legal reshape)
        self.entry_pool = [
            ops.external_tensor(f"l{li}_entry_pool", (B, 1, 2, R, D), F32, storage_id=sid(f"l{li}_entry_pool"))
            for li in range(L)
        ]
        self.series_state = [
            ops.external_tensor(f"l{li}_series_state", (B, 1, 2), I32, storage_id=sid(f"l{li}_series_state"))
            for li in range(L)
        ]
        # per-step compressor inputs (host-maintained ring window + gates +
        # per-row emission rope rows — the #96 landed feeding pattern)
        self.comp_window = ops.external_tensor("comp_window", (B, config.compress_m, D), F32, storage_id=sid("comp_window"))
        self.comp_gates = ops.external_tensor("comp_gates", (B, config.compress_m), F32, storage_id=sid("comp_gates"))
        self.comp_cos = ops.external_tensor("comp_cos", (B, D // 2), F32, storage_id=sid("comp_cos"))
        self.comp_sin = ops.external_tensor("comp_sin", (B, D // 2), F32, storage_id=sid("comp_sin"))
        # #97: per-row valid candidate count (emitted entries so far)
        self.valid_counts = ops.external_tensor("valid_counts", (B,), I32, storage_id=sid("valid_counts"))

        # #98 routing-table scratch (the router WRITES these; expert/combine
        # phases consume the returned post-views RAW)
        self.moe_ids = []
        self.moe_weights = []
        for li in range(L):
            self.moe_ids.append(ops.external_tensor(f"l{li}_moe_ids", (B, config.moe_top_k), I32, storage_id=sid(f"l{li}_moe_ids")))
            self.moe_weights.append(ops.external_tensor(f"l{li}_moe_weights", (B, config.moe_top_k), F32, storage_id=sid(f"l{li}_moe_weights")))

        self.layers = []
        for li in range(L):
            v = {}
            shapes = {
                "mhc_attn_fn": ((2 + config.hc) * config.hc, config.hc * C),
                "mhc_attn_base": ((2 + config.hc) * config.hc,),
                "mhc_attn_scale": (3,),
                "mhc_moe_fn": ((2 + config.hc) * config.hc, config.hc * C),
                "mhc_moe_base": ((2 + config.hc) * config.hc,),
                "mhc_moe_scale": (3,),
                "attn_ln_gamma": (C,),
                "q_proj_w": (C, H * D),
                "kv_a_w": (C, D),
                "o_proj_w": (H * D, H * (C // H)),
                "idx_q_w": (C, HI * D),
                "idx_mix_w": (C, HI),
                "compressor_rms_w": (D,),
                "mla_sink": (H,),
                "moe_ln_gamma": (C,),
                "router_w": (E, C),
                "expert_gate_up": (E, 2 * I, C),
                "expert_down": (E, C, I),
                "shared_gate_up": (C, 2 * Is),
                "shared_down": (Is, C),
            }
            for fname, shape in shapes.items():
                v[fname] = ops.external_tensor(f"l{li}_{fname}", shape, F32, storage_id=sid(f"l{li}_{fname}"))
            self.layers.append(type("LayerViews", (), v))


def build_deepseek_v4_forward(ops, args, ids, position, config: DeepseekV4Config):
    """One DeepSeek-V4 decode step: F(W, ids, pools, row_pos) -> logits."""
    cfg = config
    B, C, D = cfg.batch, cfg.hidden, cfg.latent_dim
    S, H, HI = cfg.cache_capacity, cfg.heads, cfg.index_heads
    M, R = cfg.entry_capacity, cfg.entries_per_series

    # embedding → initial stream stack [B, hc, C] (the table is [V, hc·C])
    emb = ops.embedding(ids, args.token, None, position)
    streams = ops.view_of(emb, "streams0", (B, cfg.hc, C))

    for li in range(cfg.layers):
        lv = args.layers[li]
        # ================= attention sublayer =================
        h_in, post_a, comb_a = ops.mhc_pre(
            streams, lv.mhc_attn_fn, lv.mhc_attn_base, lv.mhc_attn_scale,
            layer=li, iters=cfg.mhc_iters, eps=cfg.mhc_eps, rms_eps=cfg.mhc_rms_eps,
        )
        n1 = ops.rms_norm(h_in, lv.attn_ln_gamma, cfg.rms_eps, name=f"l{li}_attn_rms")

        # --- DSA compressor emission (#96): boundary rows only ---
        pool_post, _state_post = ops.compressor_append(
            args.entry_pool[li], args.series_state[li], args.comp_window, args.comp_gates,
            lv.compressor_rms_w, args.comp_cos, args.comp_sin, position,
            m=cfg.compress_m, r=cfg.compress_r, eps=cfg.compressor_eps, name=f"l{li}_compress",
        )
        entries = ops.view_of(pool_post, f"l{li}_entries", (B, M, D))

        # --- paged latent append (#94): the current token's latent row ---
        kv_k_pool = ops.view_of(
            ops.narrow(args.latent_k, f"l{li}_lk_l", axis=0, start=li, length=1),
            f"l{li}_kpool", (B * S, 1, D),
        )
        kv_v_pool = ops.view_of(
            ops.narrow(args.latent_v, f"l{li}_lv_l", axis=0, start=li, length=1),
            f"l{li}_vpool", (B * S, 1, D),
        )
        lat = ops.linear(n1, lv.kv_a_w, None, name=f"l{li}_kv_a")  # [B, D]
        lat_kv = ops.view_of(lat, f"l{li}_lat_kv", (B, 1, D))
        k_post, v_post = ops.cache_append_paged(
            kv_k_pool, kv_v_pool, args.slot_table_global, lat_kv, lat_kv, position, layer=li
        )
        kv_k = ops.view_of(k_post, f"l{li}_kv_k", (B, S, D))
        kv_v = ops.view_of(v_post, f"l{li}_kv_v", (B, S, D))

        # --- Lightning indexer (#97): scores + fixed-count top-k ---
        q_idx = ops.view_of(ops.linear(n1, lv.idx_q_w, None, name=f"l{li}_idx_q"), f"l{li}_idx_q3", (B, HI, D))
        mix = ops.linear(n1, lv.idx_mix_w, None, name=f"l{li}_idx_mix")
        idx_scores = ops.indexer_scores(q_idx, entries, mix, name=f"l{li}_indexer")
        comp_idx, block_bias = ops.index_topk(idx_scores, args.valid_counts, k=cfg.index_k, name=f"l{li}_topk")

        # --- MLA (#95): rope'd query over window ∪ selected entries ∪ sink ---
        q = ops.view_of(ops.linear(n1, lv.q_proj_w, None, name=f"l{li}_q_proj"), f"l{li}_q3", (B, H, D))
        q_r = ops.rope(q, args.rope_cos, args.rope_sin, position, layer=li, which="q", convention="interleaved", rotary_dim=D)
        probs = ops.mla_scores(
            q_r, kv_k, args.slot_table_local, entries, comp_idx, lv.mla_sink, position,
            scale=cfg.attention_scale, layer=li, window=cfg.window, bias=block_bias,
        )
        ctx = ops.mla_values(probs, kv_v, args.slot_table_local, entries, comp_idx, position, layer=li, window=cfg.window)
        ctx_c = ops.conjugate_rope(ctx, args.rope_cos, args.rope_sin, position, layer=li, which="o", convention="interleaved", rotary_dim=D)
        attn = ops.linear(
            ops.view_of(ctx_c, f"l{li}_ctx_flat", (B, H * D)), lv.o_proj_w, None,
            grouped_heads=H, name=f"l{li}_o_proj",
        )
        streams = ops.mhc_post(streams, attn, post_a, comb_a, layer=li)

        # ================= MoE sublayer =================
        h_in2, post_m, comb_m = ops.mhc_pre(
            streams, lv.mhc_moe_fn, lv.mhc_moe_base, lv.mhc_moe_scale,
            layer=li, iters=cfg.mhc_iters, eps=cfg.mhc_eps, rms_eps=cfg.mhc_rms_eps,
        )
        n2 = ops.rms_norm(h_in2, lv.moe_ln_gamma, cfg.rms_eps, name=f"l{li}_moe_rms")
        r_ids, r_w = ops.moe_route(
            n2, lv.router_w, args.moe_ids[li], args.moe_weights[li],
            mode="learned", score_fn="sqrtsoftplus", top_k=cfg.moe_top_k,
            routed_scaling_factor=cfg.routed_scaling_factor, name=f"l{li}_route",
        )
        partials = ops.moe_expert(n2, lv.expert_gate_up, lv.expert_down, r_ids, name=f"l{li}_experts")
        # dense shared-expert path (ordinary ops, composed in moe_combine)
        sgu = ops.linear(n2, lv.shared_gate_up, None, name=f"l{li}_shared_gu")
        s_gate = ops.narrow(sgu, f"l{li}_sgate", axis=1, start=0, length=cfg.shared_intermediate)
        s_up = ops.narrow(sgu, f"l{li}_sup", axis=1, start=cfg.shared_intermediate, length=cfg.shared_intermediate)
        shared = ops.linear(ops.swiglu(s_gate, s_up, name=f"l{li}_shared_swiglu"), lv.shared_down, None, name=f"l{li}_shared_down")
        moe = ops.moe_combine(partials, r_w, shared=shared, name=f"l{li}_combine")
        streams = ops.mhc_post(streams, moe, post_m, comb_m, layer=li)

    # ---- head: flattened streams → RMSNorm → tied embedding linear ----
    flat = ops.view_of(streams, "streams_flat", (B, cfg.hc * C))
    n = ops.rms_norm(flat, args.final_gamma, cfg.rms_eps, name="final_rms")
    tied_w = ops.transpose_view(args.token, "token_emb.T")  # [hc*C, V]
    return ops.linear(n, tied_w, None, name="logits")
