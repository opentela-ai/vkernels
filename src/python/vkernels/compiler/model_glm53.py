"""GLM-5.3 Flash compiler frontend (issue #102 Stage 3).

Model definition + fp64 mirror live in :mod:`.glm53_arch`; this module
provides the capture layer: ``Glm53ModelArgs`` (symbolic externals for the
recording backend — token-row streams, per-layer conv/ssm state pools and
every weight) and ``build_glm53_forward`` (the decode-step body expressed
against the §4.1 ``ops`` interface so the capture → lowering → schedule →
codegen pipeline applies).

Scope (Stage 3, per issue #102): the KDA linear-attention spine
(gdn_conv #89 → kda_delta #101 → rms_norm_gated #100) wrapped in mHC
hyper-connection mixing (#99), with sparse-MoE decode (#98: noaux_tc
route ≺ indirected experts ≺ combine + shared expert) and the dense-MLP
variant. The DSA indexer (#97) is exercised by the hybrid capture test
(see ``build_glm53_hybrid_capture``); the sparse-attention consumer of
its top-k table is the documented Stage-2/3 seam: the landed paged
attention ops read ``table[b, : p+1]`` (KV-window convention) while the
indexer emits a fixed-width ``[B, k]`` selection — compiling the absorbed
latent attention needs a table-width mode on the paged ops (follow-up).
"""

from __future__ import annotations

import numpy as np

from .capture import RecordingBackend, SymbolicTensor
from .operator_ir import F32, I32
from .glm53_arch import Glm53Config, Glm53Weights


class Glm53ModelArgs:
    """Symbolic externals for one GLM-5.3 decode step (batch B)."""

    def __init__(self, ops: RecordingBackend, cfg: Glm53Config, weights: Glm53Weights, batch: int = 3, *,
                 materialize_hosts: bool = True):
        B, C = batch, cfg.hidden_size
        H, D = cfg.linear_num_heads, cfg.linear_head_dim
        hc = cfg.hc_mult
        K, Cc = cfg.linear_conv_kernel_dim, cfg.conv_dim
        self.cfg, self.batch = cfg, B

        def ext(name: str, arr: np.ndarray, dtype=F32) -> SymbolicTensor:
            t = ops.external_tensor(name, tuple(arr.shape), dtype=dtype, storage_id=next(self._sid))
            if not materialize_hosts:
                # compile-only mode (real-dims smoke): shapes only — the
                # 288-expert fp32 weight externals are ~18 GB per layer and
                # never executed on the host
                self.host[name] = None
                self.host_by_sid[t.value.storage_id] = None
                return t
            # host mirror of the storage: fp32 (the compiled-pool convention;
            # the fp64 oracle in glm53_arch consumes these via astype)
            self.host[name] = (arr.astype(np.float32) if dtype is F32 else arr)
            self.host_by_sid[t.value.storage_id] = self.host[name]
            return t

        self.host_by_sid: dict[int, np.ndarray] = {}

        self.host: dict[str, np.ndarray] = {}
        self._sid = iter(range(1000, 100000))

        # token-row streams (host tiles the token embedding across hc)
        self.streams_in = ops.external_tensor("glm53_streams_in", (B, hc, C), storage_id=next(self._sid))
        self.host_by_sid[self.streams_in.value.storage_id] = None  # filled per test (the decode input)
        # per-layer state pools (external, §4.3 RMW)
        self.conv_state: list[SymbolicTensor] = []
        self.ssm_state: list[SymbolicTensor] = []
        for i in range(cfg.num_hidden_layers):
            cs = ops.external_tensor(f"glm53_conv_state_l{i}", (B, K - 1, Cc), storage_id=next(self._sid))
            ss = ops.external_tensor(f"glm53_ssm_state_l{i}", (B, H, D, D), storage_id=next(self._sid))
            self.conv_state.append(cs)
            self.ssm_state.append(ss)
            if materialize_hosts:
                self.host_by_sid[cs.value.storage_id] = np.zeros((B, K - 1, Cc), np.float32)
                self.host_by_sid[ss.value.storage_id] = np.zeros((B, H, D, D), np.float32)
        # weights (floe layout unless the op contract says otherwise;
        # ``linear`` takes [Cin, Cout] — floe's [out, in] is transposed here)
        L = cfg.num_hidden_layers
        self.hc_attn_fn = [ext(f"hc_a_fn_{i}", weights.hc_attn_fn[i]) for i in range(L)]
        self.hc_attn_base = [ext(f"hc_a_base_{i}", weights.hc_attn_base[i]) for i in range(L)]
        self.hc_attn_scale = [ext(f"hc_a_scale_{i}", weights.hc_attn_scale[i]) for i in range(L)]
        self.hc_ffn_fn = [ext(f"hc_f_fn_{i}", weights.hc_ffn_fn[i]) for i in range(L)]
        self.hc_ffn_base = [ext(f"hc_f_base_{i}", weights.hc_ffn_base[i]) for i in range(L)]
        self.hc_ffn_scale = [ext(f"hc_f_scale_{i}", weights.hc_ffn_scale[i]) for i in range(L)]
        self.ln1 = [ext(f"ln1_{i}", weights.ln1[i]) for i in range(L)]
        self.ln2 = [ext(f"ln2_{i}", weights.ln2[i]) for i in range(L)]
        self.qkv_proj = [ext(f"qkv_proj_{i}", weights.qkv_proj[i].T.copy()) for i in range(L)]  # [Cin, 3qkv]
        self.conv_w = [ext(f"conv_w_{i}", weights.conv_w[i]) for i in range(L)]  # [Cc, K]
        self.f_a = [ext(f"f_a_{i}", weights.f_a[i].T.copy()) for i in range(L)]  # [C, D]
        self.f_b = [ext(f"f_b_{i}", weights.f_b[i].T.copy()) for i in range(L)]  # [D, qkv]
        self.dt_bias = [ext(f"dt_bias_{i}", weights.dt_bias[i].reshape(H, D)) for i in range(L)]  # [H, K]
        self.A_log = [ext(f"A_log_{i}", weights.A_log[i]) for i in range(L)]  # [H]
        self.b_proj = [ext(f"b_proj_{i}", weights.b_proj[i].T.copy()) for i in range(L)]  # [C, H]
        self.g_a = [ext(f"g_a_{i}", weights.g_a[i].T.copy()) for i in range(L)]
        self.g_b = [ext(f"g_b_{i}", weights.g_b[i].T.copy()) for i in range(L)]
        self.o_norm = [ext(f"o_norm_{i}", weights.o_norm[i]) for i in range(L)]
        self.o_proj = [ext(f"o_proj_{i}", weights.o_proj[i].T.copy()) for i in range(L)]  # [qkv, C]
        self.router_w = [ext(f"router_w_{i}", weights.router_w[i]) for i in range(L)]  # [E, C]
        self.router_bias = [ext(f"router_bias_{i}", weights.router_bias[i]) for i in range(L)]
        self.expert_gate_up = [ext(f"expert_gu_{i}", weights.expert_gate_up[i]) for i in range(L)]  # [E, 2I, C]
        self.expert_down = [ext(f"expert_dn_{i}", weights.expert_down[i]) for i in range(L)]  # [E, C, I]
        self.shared_gate = [ext(f"sh_g_{i}", weights.shared_gate[i].T.copy()) for i in range(L)]
        self.shared_up = [ext(f"sh_u_{i}", weights.shared_up[i].T.copy()) for i in range(L)]
        self.shared_down = [ext(f"sh_d_{i}", weights.shared_down[i].T.copy()) for i in range(L)]
        self.gate_proj = [ext(f"gate_proj_{i}", weights.gate_proj[i].T.copy()) for i in range(L)]  # [C, I]
        self.up_proj = [ext(f"up_proj_{i}", weights.up_proj[i].T.copy()) for i in range(L)]
        self.down_proj = [ext(f"down_proj_{i}", weights.down_proj[i].T.copy()) for i in range(L)]  # [I, C]
        # HyperHead mean over hc as a [C, hc·C] 0.5-tile GEMV (no reduce-mean op):
        # y = flat @ W with W[j, c] = 0.5 iff j % C == c (flat is the row-major
        # flatten of [B, hc, C], so y[c] = 0.5·Σ_h flat[h·C + c]).
        mean_w = np.zeros((hc * C, C))
        for h_i in range(hc):
            mean_w[h_i * C : (h_i + 1) * C, :] = 0.5 * np.eye(C)
        self.head_mean = ext("glm53_head_mean", mean_w)
        self.final_norm = ext("glm53_final_norm", weights.final_norm[0])
        # top-k scratch for sparse MoE (routing table: i32 ids + f32 weights)
        self.route_ids = [
            ops.external_tensor(f"route_ids_{i}", (B, cfg.num_experts_per_tok), I32, storage_id=next(self._sid))
            for i in range(L)
        ]
        self.route_weights = [
            ops.external_tensor(f"route_w_{i}", (B, cfg.num_experts_per_tok), storage_id=next(self._sid))
            for i in range(L)
        ]
        if materialize_hosts:
            for i in range(L):
                self.host_by_sid[self.route_ids[i].value.storage_id] = np.zeros((B, cfg.num_experts_per_tok), np.int32)
                self.host_by_sid[self.route_weights[i].value.storage_id] = np.zeros((B, cfg.num_experts_per_tok), np.float32)

def build_glm53_forward(ops: RecordingBackend, args: Glm53ModelArgs) -> SymbolicTensor:
    """GLM-5.3 decode step body (KDA layers + mHC + MoE) → normed hidden [B, C]."""
    cfg, B = args.cfg, args.batch
    C, H, D, qkv = cfg.hidden_size, cfg.linear_num_heads, cfg.linear_head_dim, cfg.qkv_dim
    hc = cfg.hc_mult
    streams = args.streams_in
    for i in range(cfg.num_hidden_layers):
        # ---- attention sub-block (mHC-wrapped) ----
        h_in, post_a, comb_a = ops.mhc_pre(
            streams, args.hc_attn_fn[i], args.hc_attn_base[i], args.hc_attn_scale[i],
            layer=i, iters=cfg.hc_sinkhorn_iters, eps=cfg.hc_eps, rms_eps=cfg.rms_norm_eps,
        )
        h = ops.rms_norm(h_in, args.ln1[i], cfg.rms_norm_eps, name=f"ln1_l{i}")
        mixed = ops.linear(h, args.qkv_proj[i], name=f"qkv_proj_l{i}")  # [B, 3qkv]
        qkv_out, conv_post = ops.gdn_conv(args.conv_state[i], mixed, args.conv_w[i], layer=i)
        # split q|k|v thirds, view as [B, H, D]: (B, 3qkv) → (B, 3, H, D)
        # (contiguous reshape), narrow the size-1 third, squeeze back — a
        # strided head view per the layout-exact reshape contract.
        thirds = ops.view_of(qkv_out, f"qkv thirds_l{i}", (B, 3, H, D))
        qh = ops.view_of(ops.narrow(thirds, f"qn_l{i}", axis=1, start=0, length=1), f"qh_l{i}", (B, H, D))
        kh = ops.view_of(ops.narrow(thirds, f"kn_l{i}", axis=1, start=1, length=1), f"kh_l{i}", (B, H, D))
        vh = ops.view_of(ops.narrow(thirds, f"vn_l{i}", axis=1, start=2, length=1), f"vh_l{i}", (B, H, D))
        f_mid = ops.linear(h, args.f_a[i], name=f"f_a_l{i}")
        f = ops.view_of(ops.linear(f_mid, args.f_b[i], name=f"f_b_l{i}"), f"f_l{i}", (B, H, D))
        b_row = ops.linear(h, args.b_proj[i], name=f"b_proj_l{i}")  # [B, H]
        gate = ops.view_of(
            ops.linear(ops.linear(h, args.g_a[i], name=f"g_a_l{i}"), args.g_b[i], name=f"g_b_l{i}"),
            f"gate_l{i}", (B, H, D),
        )
        attn, ssm_post = ops.kda_delta(
            args.ssm_state[i], qh, kh, vh, f, b_row, args.dt_bias[i], args.A_log[i],
            layer=i, scale=D ** -0.5, lower_bound=cfg.linear_lower_bound,
        )
        on = ops.rms_norm_gated(attn, gate, args.o_norm[i], cfg.rms_norm_eps, name=f"o_norm_l{i}")
        body = ops.linear(ops.view_of(on, f"on_flat_l{i}", (B, qkv)), args.o_proj[i], name=f"o_proj_l{i}")
        streams = ops.mhc_post(streams, body, post_a, comb_a, layer=i)
        args.conv_state[i] = conv_post
        args.ssm_state[i] = ssm_post
        # ---- ffn sub-block (mHC-wrapped) ----
        h_in2, post_f, comb_f = ops.mhc_pre(
            streams, args.hc_ffn_fn[i], args.hc_ffn_base[i], args.hc_ffn_scale[i],
            layer=i + 100, iters=cfg.hc_sinkhorn_iters, eps=cfg.hc_eps, rms_eps=cfg.rms_norm_eps,
        )
        h2 = ops.rms_norm(h_in2, args.ln2[i], cfg.rms_norm_eps, name=f"ln2_l{i}")
        if cfg.mlp_layer_types[i] == "dense":
            g = ops.linear(h2, args.gate_proj[i], name=f"mlp_gate_l{i}")
            u = ops.linear(h2, args.up_proj[i], name=f"mlp_up_l{i}")
            # NOTE: GLM's dense-MLP swiglu_limit clamp is inactive at tiny
            # magnitudes; a ``limit`` attribute on the swiglu op is the
            # one-parameter follow-up (mirrors #96's rope_slice note).
            act = ops.swiglu(g, u, name=f"mlp_act_l{i}")
            body = ops.linear(act, args.down_proj[i], name=f"mlp_down_l{i}")
        else:
            ids, weights = ops.moe_route(
                h2, args.router_w[i], args.route_ids[i], args.route_weights[i],
                mode="learned", score_fn="sigmoid_noaux_tc",
                top_k=cfg.num_experts_per_tok, routed_scaling_factor=cfg.routed_scaling_factor,
                bias=args.router_bias[i], n_group=cfg.n_group, topk_group=cfg.topk_group,
                norm_topk_prob=cfg.norm_topk_prob, name=f"moe_route_l{i}",
            )
            partials = ops.moe_expert(
                h2, args.expert_gate_up[i], args.expert_down[i], ids,
                swiglu_limit=cfg.swiglu_limit, name=f"moe_expert_l{i}",
            )
            sg = ops.linear(h2, args.shared_gate[i], name=f"sh_gate_l{i}")
            su = ops.linear(h2, args.shared_up[i], name=f"sh_up_l{i}")
            sh_body = ops.linear(ops.swiglu(sg, su, name=f"sh_act_l{i}"), args.shared_down[i], name=f"sh_down_l{i}")
            body = ops.moe_combine(partials, weights, shared=sh_body, name=f"moe_combine_l{i}")
        streams = ops.mhc_post(streams, body, post_f, comb_f, layer=i + 100)
    # HyperHead mean + final norm
    streams_flat = ops.view_of(streams, "streams_flat", (B, hc * C))
    head = ops.linear(streams_flat, args.head_mean, name="hc_head_mean")
    return ops.rms_norm(head, args.final_norm, cfg.rms_norm_eps, name="final_norm")
