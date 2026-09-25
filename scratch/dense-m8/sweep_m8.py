#!/usr/bin/env python3
"""H100 dense-m8 GEMV tile sweep: M in {1,2,4,8} x dense family x bf16/fp8.

Adapted from the M=1 lane's /tmp/sweep_gemv_tiles.py (same bench()
CUDA-graph min-over-replays methodology, same JSON row schema). The bf16
M>1 arm uses glm_gemv's new _gemv_bf16_m kernel (staged worktree shadows
the baked /opt/vkernels via PYTHONPATH); the fp8 kernel already supports
M <= m_cap.

For each (O, I) shape and M it records:
  * cublas        : torch.mm(x, w.t()) on bf16 weights (fp8: the
                    dequant-on-use equivalent — mm on the bf16-ref weight),
                    the routing floor the floe policy must beat,
  * current       : the committed wrapper config (M=1) / heuristic (M>1),
  * sweep         : the full ROWS x BLOCK_I x num_warps grid at num_stages=3
                    (the wrapper passes only num_warps; triton default is 3).

Output: stdout table + /ws/m8_sweep_results.json (same schema as the
M=1 lane's gemv-sweep_results.json, so the analysis tooling is reusable).
"""
import os
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache_m8")
import itertools
import json
import time

import torch

DEV = "cuda:0"
torch.cuda.set_device(0)
DEADLINE = time.time() + 40 * 60  # serialized GPU window; keep < ~45 min

# (O, I): calls/decode-step at B=1 (per rank, TP4; census trace 3498744).
# count 0 = pinned in _CFG but no census count — measured for completeness.
BF16_SHAPES = {
    (2048, 4096): 102,   # KDA q/k/v x34 layers
    (16, 4096): 34,      # KDA b_proj
    (128, 4096): 45,     # KDA g_a x34 + indexer wk x11
    (2048, 128): 68,     # KDA f_b/g_b x34 x2
    (4096, 2048): 34,    # KDA o_proj
    (4096, 1536): 11,    # indexer wq_b
    (32, 4096): 11,      # indexer weights_proj
    (24, 16384): 90,     # mHC mix
    (1024, 1536): 11,    # DSA wq_b (rank-local)
    (8, 4096): 11,       # DSA wproj (rank-local)
    (38720, 4096): 1,    # lm_head (vocab slice)
    (8192, 512): 0,      # DSA kv_b (in _CFG)
    (3072, 4096): 0,     # dense mlp gate/up (in _CFG)
    (4096, 3072): 0,     # dense mlp down (in _CFG)
    (6144, 4096): 0,     # (in _CFG)
}
FP8_SHAPES = {
    (1536, 4096): 11,    # DSA q_a
    (4096, 1536): 11,    # DSA q_b
    (512, 4096): 84,     # DSA kv_a + shared gate/up
    (4096, 4096): 11,    # DSA o_proj
    (4096, 512): 42,     # shared down
    (3072, 4096): 6,     # dense MLP gate/up
    (4096, 3072): 3,     # dense MLP down
}
MS = [1, 2, 4, 8]

RESULTS = []


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def bi_candidates(I):
    np2 = next_pow2(I)
    ps = [64, 128, 256, 512, 1024, 2048, 4096]
    if np2 > 4096:
        ps += [8192, 16384]
    return sorted({p for p in ps if p <= np2})


def bench(fn, n=64, reps=25):
    """min-over-replays CUDA-graph timing; us/call."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    del g
    return best / n * 1e6


def rec(shape, dtype, m, tag, cfg, us, gbps):
    RESULTS.append(dict(o=shape[0], i=shape[1], dtype=dtype, m=m, tag=tag,
                        rows=cfg[0], block_i=cfg[1], warps=cfg[2],
                        stages=cfg[3] if len(cfg) > 3 else None,
                        us=round(us, 3), gbps=round(gbps, 1)))


import triton
from vkernels.torch_ops.glm_gemv import _kernel as bf16_kernel
from vkernels.torch_ops.glm_dense_fp8_gemv import _kernel as fp8_kernel

_, _, KERN_BF16, KERN_BF16_M = bf16_kernel()
_, _, KERN_FP8 = fp8_kernel()


def bf16_bytes(o, i, m):
    return o * i * 2 + m * (i * 2 + o * 2)


def fp8_bytes(o, i, m):
    return o * i + (o // 128) * (i // 128) * 4 + m * (i * 2 + o * 2)


def heuristic_bf16(o, i):
    rows = 4 if (o % 4 == 0 and (o > 2048 or i <= 512)) else 1
    return (rows, triton.next_power_of_2(min(i, 4096)), 4)


def run_bf16():
    print("\n==== bf16 dense family, M in {1,2,4,8} ====", flush=True)
    for (o, i), cnt in BF16_SHAPES.items():
        w = (torch.randn(o, i, device=DEV) * 0.05).to(torch.bfloat16)
        for m in MS:
            if time.time() > DEADLINE:
                print("!! deadline hit, aborting bf16 sweep"); return
            x = (torch.randn(m, i, device=DEV) * 0.5).to(torch.bfloat16)
            y = torch.empty(m, o, device=DEV, dtype=torch.bfloat16)
            yc = torch.empty(m, o, device=DEV, dtype=torch.bfloat16)
            b = bf16_bytes(o, i, m)
            us_c = bench(lambda: torch.mm(x, w.t(), out=yc))
            rec((o, i), "bf16", m, "cublas", (0, 0, 0), us_c, b / us_c / 1e3)
            if m == 1:
                print(f"bf16 ({o:6d},{i:6d}) x{cnt:3d} cuBLAS {us_c:8.2f} us"
                      f"  {b/us_c/1e3:6.0f} GB/s", flush=True)
            wt = w.t()  # mm(x, w.t()) == F.linear
            best = None
            for rows_, bi_, wp_ in itertools.product((1, 2, 4, 8),
                                                     bi_candidates(i), (2, 4, 8)):
                if o % rows_ or rows_ * bi_ > 32768:
                    continue
                if rows_ * m * bi_ > 1 << 21:  # fp32 acc tile guard
                    continue
                if time.time() > DEADLINE:
                    break
                mp = triton.next_power_of_2(m)
                if m == 1:
                    fn = (lambda r=rows_, bi_=bi_, wp_=wp_: KERN_BF16[((o + r - 1) // r,)](
                        x.reshape(i), w, y, o, i, r, bi_, num_warps=wp_, num_stages=3))
                else:
                    fn = (lambda r=rows_, bi_=bi_, wp_=wp_, mp=mp: KERN_BF16_M[(o // r,)](
                        x, w, y, m, o, i, mp, r, bi_, num_warps=wp_, num_stages=3))
                try:
                    us = bench(fn)
                except Exception:
                    continue
                torch.cuda.synchronize()
                ref = torch.mm(x, wt)
                err = (y.float() - ref.float()).abs().max().item()
                if err > 0.1:  # garbage guard (masking bug), not a perf candidate
                    print(f"    [BAD] bf16 ({o},{i}) m={m} ({rows_},{bi_},{wp_}) err={err}")
                    continue
                rec((o, i), "bf16", m, "sweep", (rows_, bi_, wp_), us, b / us / 1e3)
                if best is None or us < best[0]:
                    best = (us, (rows_, bi_, wp_))
            if best:
                print(f"  bf16 m={m} ({o:6d},{i:6d}) BEST {best[1]} {best[0]:8.2f} us"
                      f"  {b/best[0]/1e3:6.0f} GB/s   cuBLAS {us_c:8.2f} us"
                      f"  {'GEMV' if best[0] < us_c else 'cuBLAS'}", flush=True)
        del w
        torch.cuda.empty_cache()


def run_fp8():
    print("\n==== fp8 block-scale dense family, M in {1,2,4,8} ====", flush=True)
    for (o, i), cnt in FP8_SHAPES.items():
        w8 = ((torch.randn(o, i, device=DEV) * 0.1).clamp(-0.4, 0.4)).to(torch.float8_e4m3fn)
        s = (torch.rand(o // 128, i // 128, device=DEV) * 0.4 + 0.8)
        w8u8 = w8.view(torch.uint8)
        wref = (w8.float().view(o // 128, 128, i // 128, 128)
                * s.float()[:, None, :, None]).reshape(o, i).to(torch.bfloat16)
        for m in MS:
            if time.time() > DEADLINE:
                print("!! deadline hit, aborting fp8 sweep"); return
            x = (torch.randn(m, i, device=DEV) * 0.5).to(torch.bfloat16)
            y = torch.empty(m, o, device=DEV, dtype=torch.bfloat16)
            yc = torch.empty(m, o, device=DEV, dtype=torch.bfloat16)
            b = fp8_bytes(o, i, m)
            # cuBLAS floor = dequant-on-use: the bf16 weight GEMM the routing
            # falls back to (dequant cost excluded, as in the M=1 lane).
            us_c = bench(lambda: torch.mm(x, wref.t(), out=yc))
            rec((o, i), "fp8", m, "cublas", (0, 0, 0), us_c, b / us_c / 1e3)
            if m == 1:
                print(f"fp8  ({o:6d},{i:6d}) x{cnt:3d} cuBLAS(deq) {us_c:8.2f} us"
                      f"  {b/us_c/1e3:6.0f} GB/s", flush=True)
            mp = triton.next_power_of_2(m)
            best = None
            for rows_, bi_, wp_ in itertools.product((1, 2, 4, 8),
                                                     bi_candidates(i), (2, 4, 8)):
                if o % rows_ or rows_ * bi_ > 32768:
                    continue
                if rows_ * m * bi_ > 1 << 21:
                    continue
                if time.time() > DEADLINE:
                    break
                fn = (lambda r=rows_, bi_=bi_, wp_=wp_: KERN_FP8[(triton.cdiv(o, r),)](
                    x, w8u8, s, y, m, o, i, MP=mp, ROWS=r, BLOCK_I=bi_,
                    num_warps=wp_, num_stages=3))
                try:
                    us = bench(fn)
                except Exception:
                    continue
                torch.cuda.synchronize()
                ref = torch.mm(x, wref.t())
                err = (y.float() - ref.float()).abs().max().item()
                if err > 0.15:
                    print(f"    [BAD] fp8 ({o},{i}) m={m} ({rows_},{bi_},{wp_}) err={err}")
                    continue
                rec((o, i), "fp8", m, "sweep", (rows_, bi_, wp_), us, b / us / 1e3)
                if best is None or us < best[0]:
                    best = (us, (rows_, bi_, wp_))
            if best:
                print(f"  fp8  m={m} ({o:6d},{i:6d}) BEST {best[1]} {best[0]:8.2f} us"
                      f"  {b/best[0]/1e3:6.0f} GB/s   cuBLAS(deq) {us_c:8.2f} us"
                      f"  {'GEMV' if best[0] < us_c else 'cuBLAS'}", flush=True)
        del w8, s, wref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)
    print(f"device={torch.cuda.get_device_name(0)} cc={p.major}.{p.minor} "
          f"torch={torch.__version__} triton={triton.__version__}", flush=True)
    t0 = time.time()
    run_bf16()
    run_fp8()
    with open("/ws/m8_sweep_results.json", "w") as f:
        json.dump(RESULTS, f, indent=1)
    print(f"\n[sweep done in {time.time()-t0:.0f}s; {len(RESULTS)} timings"
          f" -> /ws/m8_sweep_results.json]", flush=True)
