"""Parity + timing gate: vendored ``vkernels.torch_ops.vllm_kda.kda_chunk_floe``
vs floe's eager chunked KDA reference.

Reference below is copied VERBATIM from
``floe/engine/runner/models/glm53flash/forward.py::1467 _kda_chunk``
(fp32 pure-torch chunked per-dim-gate delta rule; provenance pinned — if
floe's reference changes, re-copy). Input recipe mirrors
``docker/beverin/glm5-smoke/bench_kda_chunk_hip.py::make_inputs``:
q/k L2-normalized fp32, v ~ N(0,1), g_log ~ U[-5, 0] per (head, dim),
beta ~ U[0, 1]. Serving dims: B=1, H=64, D=128, chunk 64.

Pass criteria: fp32-vs-fp32 so diffs are association-order only —
out rel <= 5e-4 vs the fp32 reference, final-state max-abs <= 1e-3
(decayed state entries underflow toward 0; rel is meaningless there,
reported anyway).
"""
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/iopsstor/scratch/cscs/xyao/floe-beverin/aiter-mig/vkernels-py")
from vkernels.torch_ops.vllm_kda import kda_chunk_floe  # noqa: E402

DEV = "cuda"


# ---- verbatim floe reference (forward.py::1467) --------------------------
def _kda_chunk_ref(query, key, value, g, beta, chunk_size=64,
                   initial_state=None, output_final_state=False):
    initial_dtype = query.dtype
    bsz, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    scale = 1.0 / math.sqrt(query.shape[-1])
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    total = seq_len + pad

    query = F.pad(query, (0, 0, 0, pad)) * scale
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    g = F.pad(g, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    shape = lambda t: t.reshape(bsz, num_heads, -1, chunk_size, t.shape[-1])  # noqa: E731
    query, key, value, g, k_beta, v_beta = (
        shape(query), shape(key), shape(value), shape(g), shape(k_beta), shape(v_beta),
    )
    beta = beta.reshape(bsz, num_heads, -1, chunk_size)

    g = g.cumsum(dim=-2)
    tri0 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), 0)
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(tri0, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    state = (torch.zeros(bsz, num_heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
             if initial_state is None else initial_state.to(value.dtype))
    core_attn_out = torch.zeros_like(value)
    tri1 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), 1)
    for c in range(total // chunk_size):
        q_i, k_i, v_i, g_i = query[:, :, c], key[:, :, c], value[:, :, c], g[:, :, c]
        attn_inter = (q_i * g_i.exp()) @ state
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, c]).sum(dim=-1).masked_fill(tri1, 0)
        v_prime = k_cumdecay[:, :, c] @ state
        v_new = v_i - v_prime
        core_attn_out[:, :, c] = attn_inter + attn_intra @ v_new
        state = (state * g_i[:, :, -1].exp().unsqueeze(-1)
                 + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new)

    core_attn_out = core_attn_out.reshape(bsz, num_heads, total, v_dim)[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, (state if output_final_state else None)
# ---- end verbatim ---------------------------------------------------------


def make_inputs(b, h, s, d, seed):
    torch.manual_seed(seed)
    q = torch.randn(b, h, s, d, device=DEV, dtype=torch.float32)
    k = torch.randn(b, h, s, d, device=DEV, dtype=torch.float32)
    q = q / q.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
    k = k / k.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
    v = torch.randn(b, h, s, d, device=DEV, dtype=torch.float32)
    g_log = torch.rand(b, h, s, d, device=DEV, dtype=torch.float32) * -5.0
    beta = torch.rand(b, h, s, device=DEV, dtype=torch.float32)
    return q, k, v, g_log, beta


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def parity_case(b, h, s, d, chunk, seed, with_state, state=None):
    q, k, v, g_log, beta = make_inputs(b, h, s, d, seed)
    h0 = None
    if with_state:
        h0 = (torch.randn(b, h, d, d, device=DEV, dtype=torch.float32) * 0.1
              if state is None else state)

    ref_out, ref_st = _kda_chunk_ref(q, k, v, g_log, beta, chunk_size=chunk,
                                     initial_state=h0, output_final_state=True)
    new_out, new_st = kda_chunk_floe(q, k, v, g_log, beta, chunk_size=chunk,
                                     initial_state=h0, output_final_state=True)
    ok = True
    r_out, a_out = rel(new_out, ref_out), (new_out.float() - ref_out.float()).abs().max().item()
    r_st = a_st = float("nan")
    if ref_st is not None and new_st is not None:
        r_st = rel(new_st, ref_st)
        a_st = (new_st.float() - ref_st.float()).abs().max().item()
        ok &= a_st <= 1e-3
    ok &= r_out <= 5e-4
    tag = f"B={b} H={h} S={s} D={d} cs={chunk} h0={with_state}"
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:44s} "
          f"out: rel {r_out:.2e} abs {a_out:.2e} | state: rel {r_st:.2e} abs {a_st:.2e}",
          flush=True)
    return ok


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def main():
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    print("\n=== 1. PARITY (vendored fla Triton vs floe eager fp32) ===", flush=True)
    ok = True
    ok &= parity_case(1, 64, 512, 128, 64, 11, False)
    ok &= parity_case(1, 64, 512, 128, 64, 12, True)
    ok &= parity_case(1, 64, 2048, 128, 64, 13, True)
    ok &= parity_case(1, 64, 8192, 128, 64, 14, True)
    ok &= parity_case(1, 64, 200, 128, 64, 15, True)   # ragged S (S % 64 != 0)
    ok &= parity_case(1, 4, 64, 32, 16, 16, True)      # small-D template + cs=16
    print(f"\nparity gate: {'PASS' if ok else 'FAIL'}", flush=True)

    print("\n=== 2. PERF (mean us, CUDA events) ===", flush=True)
    print(f"{'shape':30s} {'eager_us':>10s} {'triton_us':>10s} {'speedup':>8s}", flush=True)
    for b, h, s, d in ((1, 64, 512, 128), (1, 64, 2048, 128), (1, 64, 8192, 128),
                       (2, 64, 1024, 128)):
        q, k, v, g_log, beta = make_inputs(b, h, s, d, 99)
        iters = 10 if s <= 2048 else 5
        t_ref = timed(lambda: _kda_chunk_ref(q, k, v, g_log, beta, chunk_size=64),
                      iters, 3)
        t_new = timed(lambda: kda_chunk_floe(q, k, v, g_log, beta, chunk_size=64),
                      iters, 3)
        print(f"B={b} H={h} S={s} D={d:<22d} {t_ref:10.1f} {t_new:10.1f} "
              f"{t_ref / max(t_new, 1e-9):8.2f}x", flush=True)
        if s == 2048 and b == 1:
            print(f"  -> extrapolated 34-layer prefill: eager {t_ref*34/1e3:.1f} ms "
                  f"vs triton {t_new*34/1e3:.1f} ms "
                  f"(saves {(t_ref - t_new)*34/1e3:.1f} ms/prompt)", flush=True)

    print("\ndone rc=0", flush=True)


if __name__ == "__main__":
    main()
