"""Lane I diagnostic: localize the GH200 final-state drift (7.3e-4) of the
fused KDA chunk pipeline vs the eager fp32 chunked oracle.

Headline suspect after the first bisect job: the floe-clariden NGC container
defaults torch matmuls to tf32 (fp32_precision='tf32') — so lane C's "eager
fp32 oracle" was itself tf32-degraded on GH200, while the fused kernel pins
every fp32 dot to ieee. The first section therefore A/Bs the ORACLE's matmul
precision; the rest bisects whatever residual remains:

  1. per-stage diffs   fused intermediates vs the oracle's (Ainv, w, u, kg,
     v_new, out, per-chunk h);
  2. isolations        torch-recompute the chunk update and the w*H product
     from the FUSED intermediates (exact fp32 torch arithmetic) and compare
     against what the kernel produced -> a gap beyond ~1e-6 is in-kernel dot
     lowering, not upstream rounding;
  3. micro-repro       the fwd_h state-update dot standalone (ieee) vs torch,
     plus variants (explicit FMA loop, split dot, warps, tf32 for scale);
  4. ablations         exp2-vs-exp and blockwise-vs-substitution inverse in
     pure torch;
  5. fwd_h autotune config sweep (BV x warps x stages).

Run: python3 kda_state_bisect.py  (GH200, floe-clariden container)
"""
import math
import sys

import torch

sys.path.insert(0, "/iopsstor/scratch/cscs/xyao/kvaas-clariden/vkernels-src")
from vkernels.torch_ops.vllm_kda import (  # noqa: E402
    RCP_LN2,
    chunk_gated_delta_rule_fwd_h,
    chunk_kda_scaled_dot_kkt_fwd,
    chunk_gla_fwd_o_gk,
    recompute_w_u_fwd,
    solve_tril,
)

DEV = "cuda"
CHUNK = 64


def hdr(msg):
    print(f"\n=== {msg} ===", flush=True)


def make_inputs(b, h, s, d, seed, nonzero_state):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, s, d, generator=g)
    k = torch.randn(b, h, s, d, generator=g)
    v = torch.randn(b, h, s, d, generator=g)
    gate = -torch.rand(b, h, s, d, generator=g)
    beta = torch.rand(b, h, s, generator=g)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    state = torch.randn(b, h, d, d, generator=g) * 0.1 if nonzero_state else None
    return q, k, v, gate, beta, state


def dmax(a, b):
    return (a.float() - b.float()).abs().max().item()


def set_matmul_precision(tf32):
    """Pin the torch matmul fp32 path. The NGC container defaults to tf32
    (fp32_precision='tf32') — lane C's oracle ran tf32-degraded there."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32" if tf32 else "ieee"


# --------------------------------------------------------------------------
# instrumented eager oracle (kda_chunk_reference's loop + per-chunk captures)
# --------------------------------------------------------------------------
def ref_instrumented(q, k, v, g, beta, chunk_size, initial_state):
    bsz, num_heads, seq_len, k_dim = q.shape
    scale = 1.0 / math.sqrt(q.shape[-1])
    total = seq_len  # shapes here are chunk multiples
    query = q * scale
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    shape = lambda t: t.reshape(bsz, num_heads, -1, chunk_size, t.shape[-1])  # noqa: E731
    query, key, value, g = shape(query), shape(k), shape(v), shape(g)
    k_beta, v_beta = shape(k_beta), shape(v_beta)
    g = g.cumsum(dim=-2)
    tri0 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(tri0, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())
    state = initial_state.to(value.dtype).clone()
    tri1 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    cap = {n: [] for n in ("attn", "kcd", "value", "v_new", "state_out", "out")}
    for c in range(total // chunk_size):
        q_i, k_i, v_i, g_i = query[:, :, c], key[:, :, c], value[:, :, c], g[:, :, c]
        attn_inter = (q_i * g_i.exp()) @ state
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, c]).sum(dim=-1).masked_fill(tri1, 0)
        v_new = v_i - k_cumdecay[:, :, c] @ state
        cap["attn"].append(attn[:, :, c].clone())
        cap["kcd"].append(k_cumdecay[:, :, c].clone())
        cap["value"].append(v_i.clone())
        cap["v_new"].append(v_new.clone())
        cap["out"].append((attn_inter + attn_intra @ v_new).clone())
        state = state * g_i[:, :, -1].exp().unsqueeze(-1) \
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        cap["state_out"].append(state.clone())
    return cap, g


def fused_pipeline(q, k, v, gate, beta, initial_state, chunk_size=CHUNK):
    """The exact glm_kda_chunk prelude + pipeline calls, capturing everything."""
    b, h, s, k_dim = k.shape
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    beta_t = beta.transpose(1, 2).contiguous()
    # bit-identical cumsum (the kda_chunk_floe path)
    g_cum = (
        gate.reshape(b, h, s // chunk_size, chunk_size, k_dim)
        .cumsum(-2)
        .reshape(b, h, s, k_dim)
        .transpose(1, 2)
        .contiguous()
    )
    g2 = g_cum * RCP_LN2
    h0 = initial_state.transpose(-1, -2).contiguous().float() if initial_state is not None else None
    scale = k_dim ** -0.5

    A, Aqk = chunk_kda_scaled_dot_kkt_fwd(q=q_t, k=k_t, gk=g2, beta=beta_t, scale=scale,
                                          chunk_size=chunk_size, output_dtype=torch.float32)
    Ai = solve_tril(A=A, output_dtype=k_t.dtype)
    w, u, kg = recompute_w_u_fwd(k=k_t, v=v_t, beta=beta_t, A=Ai, gk=g2)
    h_buf, v_new, final = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g2, initial_state=h0, output_final_state=True,
        chunk_size=chunk_size, use_exp2=True)
    o = chunk_gla_fwd_o_gk(q=q_t, v=v_new, g=g2, A=Aqk, h=h_buf, o=v_t, scale=scale,
                           chunk_size=chunk_size)
    return dict(Ai=Ai, w=w, u=u, kg=kg, h=h_buf, v_new=v_new,
                final=final, o=o, g2=g2, g_cum=g_cum)


# layout bridges -------------------------------------------------------------
# fused [B,T,H,X] <-> ref chunk layout [B,H,nc,cs,X]
def to_ref(t, b, h, nc, cs):
    """fused [B,T,H,X] -> [B,H,nc,cs,X]"""
    x = t.shape[-1]
    return t.reshape(b, nc, cs, h, x).permute(0, 3, 1, 2, 4).contiguous()


def to_fused(t, b, s, h):
    """ref [B,H,nc,cs,X] -> [B,T,H,X]"""
    x = t.shape[-1]
    return t.permute(0, 2, 3, 1, 4).reshape(b, s, h, x).contiguous()


# --------------------------------------------------------------------------
def bisect_shape(b, h, s, d, seed, nonzero_state):
    hdr(f"shape B={b} H={h} S={s} D={d} seed={seed} h0={'yes' if nonzero_state else 'no'}")
    q, k, v, gate, beta, state0 = make_inputs(b, h, s, d, seed, nonzero_state)
    q, k, v, gate, beta = (x.to(DEV) for x in (q, k, v, gate, beta))
    state0 = state0.to(DEV) if state0 is not None else None

    cap, g_ref = ref_instrumented(q, k, v, gate, beta, CHUNK, state0)
    fu = fused_pipeline(q, k, v, gate, beta, state0)

    nc = s // CHUNK
    g_ref_t = g_ref  # [B,H,nc,cs,K] nat cumsum
    g_fused_ref_layout = to_ref(fu["g_cum"], b, h, nc, CHUNK)
    print(f"gates: cumsum bit-identical: {torch.equal(g_fused_ref_layout, g_ref_t)}")

    # exp2-vs-exp factor diff (ref layout, per element)
    fac = (torch.exp2(fu["g2"].reshape(b, nc, CHUNK, h, d).permute(0, 3, 1, 2, 4))
           - g_ref_t.exp())
    rel = fac.abs() / g_ref_t.exp().abs().clamp_min(1e-30)
    print(f"decay factors exp2(g*log2e) vs exp(g): max abs {fac.abs().max().item():.2e}, "
          f"max rel {rel.max().item():.2e}")

    print("-- per-stage fused-vs-oracle max abs diffs (ref chunk layout) --")
    print(f"Ainv  (solve_tril vs attn):      {dmax(to_ref(fu['Ai'], b, h, nc, CHUNK), torch.stack(cap['attn'], 2)):.3e}")
    print(f"w     (vs k_cumdecay):           {dmax(to_ref(fu['w'], b, h, nc, CHUNK), torch.stack(cap['kcd'], 2)):.3e}")
    print(f"u     (vs value):                {dmax(to_ref(fu['u'], b, h, nc, CHUNK), torch.stack(cap['value'], 2)):.3e}")
    print(f"v_new (vs oracle v_new):         {dmax(to_ref(fu['v_new'], b, h, nc, CHUNK), torch.stack(cap['v_new'], 2)):.3e}")
    print(f"kg    (vs oracle k*exp(gn-g)):   "
          f"{dmax(fu['kg'], to_fused(k.reshape(b, h, nc, CHUNK, d) * (g_ref[..., -1:, :] - g_ref).exp(), b, s, h)):.3e}")
    print(f"out   (vs oracle out):           {dmax(fu['o'], torch.stack(cap['out'], 2).permute(0, 2, 3, 1, 4).reshape(b, s, h, v.shape[-1])):.3e}")

    state_out_all = torch.stack(cap["state_out"], dim=2)  # [B,H,nc,K,V]
    print("-- per-chunk: |fused h(c+1)-oracle state| | torch-recompute(fused pieces)-fused h(c+1) | --")
    worst = 0.0
    for c in range(nc):
        h_in = fu["h"][:, c]                       # [B,H,V,K] state entering c
        h_next_fused = fu["final"] if c == nc - 1 else fu["h"][:, c + 1]
        d_state = dmax(h_next_fused.transpose(-1, -2), state_out_all[:, :, c])
        # torch recompute of the update from FUSED pieces (fp32 torch arithmetic)
        gk_last = fu["g2"][:, c * CHUNK + CHUNK - 1]        # [B,H,K] log2 space
        vn_c = fu["v_new"][:, c * CHUNK:(c + 1) * CHUNK].permute(0, 2, 1, 3)  # [B,H,cs,V]
        kg_c = fu["kg"][:, c * CHUNK:(c + 1) * CHUNK].permute(0, 2, 1, 3)     # [B,H,cs,K]
        h_recomp = h_in * torch.exp2(gk_last).unsqueeze(-2) \
            + vn_c.transpose(-1, -2) @ kg_c                                 # [B,H,V,K]
        d_recomp = dmax(h_recomp, h_next_fused)
        # v_new recompute (w*H product check)
        w_c = fu["w"][:, c * CHUNK:(c + 1) * CHUNK].permute(0, 2, 1, 3)
        u_c = fu["u"][:, c * CHUNK:(c + 1) * CHUNK].permute(0, 2, 1, 3)
        d_vnew = dmax(u_c - w_c @ h_in, vn_c)
        worst = max(worst, d_state)
        print(f"chunk {c:2d}: state Δ {d_state:.3e} | update-recompute Δ {d_recomp:.3e} | v_new-recompute Δ {d_vnew:.3e}")
    print(f"worst per-chunk state Δ: {worst:.3e}")
    print(f"FINAL state Δ vs oracle: {dmax(fu['final'].transpose(-1, -2), cap['state_out'][-1]):.3e}")
    return q, k, v, gate, beta, state0, fu, cap


# --------------------------------------------------------------------------
# 3. micro-repro of the state-update dot
# --------------------------------------------------------------------------
def micro_repro(b=2, h=8, seed=7):
    import triton
    import triton.language as tl

    hdr(f"micro-repro: fwd_h update dot  (triton {triton.__version__})")
    torch.manual_seed(seed)
    K = V = BT = 64
    kg = torch.randn(b * h, BT, K, device=DEV, dtype=torch.float32)
    vv = torch.randn(b * h, BT, V, device=DEV, dtype=torch.float32)
    ref = torch.bmm(vv.transpose(1, 2), kg)  # [N, V, K] exact fp32 cublas

    @triton.jit
    def _upd_dot(kg_ptr, v_ptr, o_ptr, BT: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                 PREC: tl.constexpr, FMA: tl.constexpr):
        i_n = tl.program_id(0)
        i_v = tl.program_id(1)
        o_v = i_v * 64 + tl.arange(0, 64)
        o_k = tl.arange(0, 64)
        if FMA:
            b_h = tl.zeros([64, 64], dtype=tl.float32)
            for t in tl.static_range(BT):
                b_kt = tl.load(kg_ptr + i_n * BT * K + t * K + o_k)
                b_vt = tl.load(v_ptr + i_n * BT * V + t * V + o_v)
                b_h += b_vt[:, None] * b_kt[None, :]
        else:
            o_t = tl.arange(0, BT)
            b_k = tl.load(kg_ptr + i_n * BT * K + o_k[:, None] + o_t[None, :] * K)
            b_v = tl.load(v_ptr + i_n * BT * V + o_t[:, None] * V + o_v[None, :])
            b_h = tl.trans(tl.dot(b_k, b_v, input_precision=PREC))
        tl.store(o_ptr + i_n * V * K + o_v[:, None] * K + o_k[None, :], b_h)

    @triton.jit
    def _upd_split(kg_ptr, v_ptr, o_ptr, BT: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                   PREC: tl.constexpr):
        i_n = tl.program_id(0)
        i_v = tl.program_id(1)
        o_v = i_v * 64 + tl.arange(0, 64)
        o_k = tl.arange(0, 64)
        o_t = tl.arange(0, 32)
        base_k = kg_ptr + i_n * BT * K
        base_v = v_ptr + i_n * BT * V
        b_k1 = tl.load(base_k + o_k[:, None] + o_t[None, :] * K)
        b_v1 = tl.load(base_v + o_t[:, None] * V + o_v[None, :])
        b_h = tl.trans(tl.dot(b_k1, b_v1, input_precision=PREC))
        b_k2 = tl.load(base_k + o_k[:, None] + (o_t[None, :] + 32) * K)
        b_v2 = tl.load(base_v + (o_t[:, None] + 32) * V + o_v[None, :])
        b_h += tl.trans(tl.dot(b_k2, b_v2, input_precision=PREC))
        tl.store(o_ptr + i_n * V * K + o_v[:, None] * K + o_k[None, :], b_h)

    grid = (b * h, V // 64)

    def run(prec="ieee", warps=4, split=False, fma=False):
        o = torch.empty_like(ref)
        if fma:
            _upd_dot[grid](kg, vv, o, BT, K, V, prec, 1, num_warps=warps)
        elif split:
            _upd_split[grid](kg, vv, o, BT, K, V, prec, num_warps=warps)
        else:
            _upd_dot[grid](kg, vv, o, BT, K, V, prec, 0, num_warps=warps)
        return dmax(o, ref)

    e = {f"dot ieee w{w}": run("ieee", w) for w in (1, 2, 4)}
    e["split-32 ieee"] = run(split=True)
    e["explicit-FMA"] = run(fma=True)
    e["dot tf32"] = run("tf32")
    for name, val in e.items():
        print(f"  {name:16s}: {val:.3e}")
    print("(a ~5e-4-class ieee dot = lowering bug; explicit-FMA ~1e-6 = fix direction)")


# --------------------------------------------------------------------------
# 4. pure-torch ablations
# --------------------------------------------------------------------------
def ablations(b=1, h=8, s=512, d=128, seed=100):
    hdr("ablations (pure torch): exp2-vs-exp, blockwise-inverse-vs-substitution")
    q, k, v, gate, beta, state0 = make_inputs(b, h, s, d, seed, True)
    q, k, v, gate, beta, state0 = (x.to(DEV) for x in (q, k, v, gate, beta, state0))
    cap, g_ref = ref_instrumented(q, k, v, gate, beta, CHUNK, state0)
    ref_final = cap["state_out"][-1]

    def run_ref(use_exp2=False, blockinv=False):
        bsz, num_heads, seq_len, k_dim = q.shape
        scale = 1.0 / math.sqrt(d)
        query = q * scale
        v_beta = v * beta.unsqueeze(-1)
        k_beta = k * beta.unsqueeze(-1)
        shape = lambda t: t.reshape(bsz, num_heads, -1, CHUNK, t.shape[-1])  # noqa: E731
        query, key, value, g = shape(query), shape(k), shape(v), shape(gate)
        k_beta, v_beta = shape(k_beta), shape(v_beta)
        g = g.cumsum(dim=-2)
        ex = (lambda x: torch.exp2(x * RCP_LN2)) if use_exp2 else torch.exp
        tri0 = torch.triu(torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=q.device), 0)
        decay_mask = ex(g.unsqueeze(-2) - g.unsqueeze(-3))
        attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(tri0, 0)
        if blockinv:
            # fla order: 16x16 sequential inverse + block merges, in torch
            Ai = torch.zeros_like(attn)
            o_i = torch.arange(16, device=q.device)
            mA = (o_i[:, None] > o_i[None, :]).float()
            mI = (o_i[:, None] == o_i[None, :]).float()
            for bi in range(4):
                sl = slice(bi * 16, (bi + 1) * 16)
                blk = -attn[..., sl, sl] * mA
                for i in range(2, 16):
                    b_a = -blk[..., i, :]
                    b_a = b_a + (b_a.unsqueeze(-1) * blk).sum(-2)
                    blk = torch.where((o_i == i).view(1, 1, 1, 16, 1), b_a, blk)
                Ai[..., sl, sl] = blk + mI
            for size in (32, 64):
                for bi in range(64 // size):
                    i1, i2 = bi * size, (bi + 1) * size
                    m = size // 2
                    A11 = Ai[..., i1:i1 + m, i1:i1 + m]
                    A22 = Ai[..., i1 + m:i2, i1 + m:i2]
                    A21 = attn[..., i1 + m:i2, i1:i1 + m]
                    Ai[..., i1 + m:i2, i1:i1 + m] = -(A22 @ A21) @ A11
            attn_inv = Ai
        else:
            for i in range(1, CHUNK):
                row = attn[..., i, :i].clone()
                sub = attn[..., :i, :i].clone()
                attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            attn_inv = attn + torch.eye(CHUNK, dtype=attn.dtype, device=q.device)
        value = attn_inv @ v_beta
        k_cumdecay = attn_inv @ (k_beta * ex(g))
        state = state0.to(value.dtype).clone()
        for c in range(s // CHUNK):
            g_i = g[:, :, c]
            v_new = value[:, :, c] - k_cumdecay[:, :, c] @ state
            state = state * ex(g_i[:, :, -1]).unsqueeze(-1) \
                + (key[:, :, c] * ex(g_i[:, :, -1:] - g_i)).transpose(-1, -2) @ v_new
        return state

    print(f"exp2-everywhere vs oracle:   {dmax(run_ref(use_exp2=True), ref_final):.3e}")
    print(f"blockwise-inverse vs oracle: {dmax(run_ref(blockinv=True), ref_final):.3e}")
    print(f"both combined vs oracle:     {dmax(run_ref(use_exp2=True, blockinv=True), ref_final):.3e}")
    print("(if 'both' ~1e-5 but the real fused pipeline is 7e-4, the residual is")
    print(" in-kernel dot lowering -> see micro-repro)")


# --------------------------------------------------------------------------
# 5. fwd_h autotune config sweep
# --------------------------------------------------------------------------
def fwd_h_config_sweep(q, k, v, gate, beta, state0, fu, cap):
    import vkernels.torch_ops.vllm_kda as V

    hdr("fwd_h autotune config sweep")
    kernel = V.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    full = list(kernel.configs)
    print(f"configs ({len(full)}): " + ", ".join(
        f"BV{c.kwargs.get('BV')}/w{c.num_warps}/s{c.num_stages}" for c in full))
    b, hh, s, d = k.shape
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    beta_t = beta.transpose(1, 2).contiguous()
    g2 = fu["g2"]
    w, u, kg = recompute_w_u_fwd(k=k_t, v=v_t, beta=beta_t, A=fu["Ai"], gk=g2)
    h0 = state0.transpose(-1, -2).contiguous().float()
    ref_final = cap["state_out"][-1]
    try:
        for cfg in full:
            kernel.configs = [cfg]
            kernel.cache = {}
            _, _, final = chunk_gated_delta_rule_fwd_h(
                k=kg, w=w, u=u, gk=g2, initial_state=h0, output_final_state=True,
                chunk_size=CHUNK, use_exp2=True)
            dd = dmax(final.transpose(-1, -2), ref_final)
            print(f"  BV{cfg.kwargs.get('BV')}/w{cfg.num_warps}/s{cfg.num_stages}: final Δ {dd:.3e}")
    finally:
        kernel.configs = full
        kernel.cache = {}


def main():
    print(f"device: {torch.cuda.get_device_name(0)}  cc: {torch.cuda.get_device_capability(0)}")
    import triton
    print(f"torch {torch.__version__}  triton {triton.__version__}")
    print(f"container default matmul allow_tf32: {torch.backends.cuda.matmul.allow_tf32}  "
          f"fp32_precision: {getattr(torch.backends.cuda.matmul, 'fp32_precision', 'n/a')}")

    # A/B: reproduce lane C's 7.3e-4 with the container default (tf32 oracle),
    # then the same comparison with a true-fp32 (ieee) oracle.
    hdr("A/B: oracle matmul precision (container default tf32 vs pinned ieee)")
    set_matmul_precision(True)
    q, k, v, gate, beta, state0 = make_inputs(1, 8, 512, 128, 100, True)
    q, k, v, gate, beta, state0 = (x.to(DEV) for x in (q, k, v, gate, beta, state0))
    cap32, _ = ref_instrumented(q, k, v, gate, beta, CHUNK, state0)
    fu32 = fused_pipeline(q, k, v, gate, beta, state0)
    print(f"tf32-oracle vs fused(ieee): final Δ "
          f"{dmax(fu32['final'].transpose(-1, -2), cap32['state_out'][-1]):.3e}")
    set_matmul_precision(False)
    print(f"matmul allow_tf32 now: {torch.backends.cuda.matmul.allow_tf32}  "
          f"fp32_precision: {getattr(torch.backends.cuda.matmul, 'fp32_precision', 'n/a')}")
    cap_ie, _ = ref_instrumented(q, k, v, gate, beta, CHUNK, state0)
    fu_ie = fused_pipeline(q, k, v, gate, beta, state0)
    print(f"ieee-oracle vs fused(ieee): final Δ "
          f"{dmax(fu_ie['final'].transpose(-1, -2), cap_ie['state_out'][-1]):.3e}")
    print(f"ieee-oracle vs tf32-oracle: final Δ "
          f"{dmax(cap_ie['state_out'][-1], cap32['state_out'][-1]):.3e}   "
          "<- if this carries the 7e-4, the drift was the container's tf32 oracle")

    q2, k2, v2, gate2, beta2, state02, fu, cap = bisect_shape(1, 8, 512, 128, 100, True)
    fwd_h_config_sweep(q2, k2, v2, gate2, beta2, state02, fu, cap)
    bisect_shape(1, 64, 2048, 128, 13, True)
    micro_repro()
    ablations()
    print("\ndone rc=0", flush=True)


if __name__ == "__main__":
    main()
