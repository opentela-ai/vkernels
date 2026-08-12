#!/usr/bin/env python3
"""Measure xkernels reference torch loop (no Triton) for comparison with vkernels."""
import sys, time, argparse
import warnings; warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/xkernels/src")
from xkernels.ops.moe.mxfp4 import dequant_mxfp4_weight, make_mxfp4_moe_weights

V4 = dict(E=256, HIDDEN=4096, ISPP=512, TOP_K=6, LIMIT=10.0)

def torch_moe(A, w, topk_ids, topk_w, L):
    M, hidden = A.shape
    top_k = topk_ids.shape[1]
    out = torch.zeros(M, hidden, dtype=torch.float32, device=A.device)
    flat = topk_ids.reshape(-1)
    tok = torch.arange(M, device=A.device).repeat_interleave(top_k)
    fw = topk_w.reshape(-1).float()
    for e in torch.unique(flat).tolist():
        sel = flat == e
        t = tok[sel]
        wt = fw[sel]
        w13e = dequant_mxfp4_weight(w["w13"][e], w["w13_scale"][e], 32)
        gu = A[t] @ w13e.T + w["b13"][e]
        g, u = gu.float().chunk(2, -1)
        g = torch.clamp(g, max=L)
        u = torch.clamp(u, -L, L)
        act = (F.silu(g) * u).to(torch.bfloat16)
        w2e = dequant_mxfp4_weight(w["w2"][e], w["w2_scale"][e], 32)
        d = (act @ w2e.T).float() + w["b2"][e].float()
        out.index_add_(0, t, d * wt.unsqueeze(-1))
    return out


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def main():
    assert torch.cuda.is_available(), "needs GPU"
    dev = "cuda"
    E, hidden, ispp, top_k, L = V4["E"], V4["HIDDEN"], V4["ISPP"], V4["TOP_K"], V4["LIMIT"]
    w = make_mxfp4_moe_weights(E, hidden, ispp, group_size=32, with_bias=True, device=dev, seed=1)

    ms = [1, 2, 4, 8, 16, 32, 48]
    print(f"xkernels torch-loop MoE  E={E} hidden={hidden} ispp={ispp} top_k={top_k}")
    print(f"{'M':>6} {'dequant+exp(ms)':>18} {'gemm_w13(ms)':>14} {'gemm_w2(ms)':>13} {'combine(ms)':>12} {'total(ms)':>11}")
    for M in ms:
        A = (torch.randn(M, hidden, device=dev) * 0.1).to(torch.bfloat16)
        topk_ids = torch.stack(
            [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
        ).to(torch.int32)
        topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)

        # Time the full loop
        t_total = bench(lambda: torch_moe(A, w, topk_ids, topk_w, L), iters=5, warmup=2)
        print(f"{M:>6} {'-':>18} {'-':>14} {'-':>13} {'-':>12} {t_total:>10.2f}")


if __name__ == "__main__":
    main()
