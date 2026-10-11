"""Microbench: fused_router_dense (decode-regime fold) vs the eager chain.

The node-cut-D incumbent per MoE layer (decode, T = batch <= 8):
  1. x.float() cast kernel          [T, 4096] bf16 -> fp32 materialization
  2. cuBLAS gemvx/gemmSN (F.linear) [288, 4096] fp32 GEMV -> [T, 288]
  3. incumbent _router kernel       sigmoid/bias/top-8/normalize/scale

The fold replaces all three with TWO launches: the dense GEMV (bf16 x
read directly, fp32 logits out) + the incumbent router kernel verbatim.

GB10 measured (profiler device time per call, production [288, 4096] fp32
weight, bf16 x):

    T=1: chain cast 1.8 + gemvx 5.5 + router 2.8 = 10.1 us / 3 nodes
         fold  gemv 9.2              + router 2.8 = 12.0 us / 2 nodes
    T=4: chain 1.8 + 15.1 + 2.8                 = 19.7 us / 3 nodes
         fold 13.4              + 2.8           = 16.2 us / 2 nodes
    T=8: chain 2.5 + 19.9 + 3.2                 = 25.6 us / 3 nodes
         fold 21.8              + 3.2           = 24.9 us / 2 nodes

At T >= 16 cuBLAS's register-tiled fp32 GEMM is compute-bound (~75% of
GB10's fp32 peak, 48 SMs) and 3-7x faster than Triton fp32-FMA GEMV
variants (no weight reuse per row; ieee tl.dot variants measured no
better, with a wider error band) — hence the op's decode-regime
eligibility cap: prefill keeps the eager chain via the OpNotEligible
fallback.
"""

import torch
import torch.nn.functional as F

from vkernels.torch_ops.glm_router import fused_router
from vkernels.torch_ops.glm_router_dense import fused_router_dense

E, H, K, SCALING = 288, 4096, 8, 2.5
WARM, ITERS = 200, 1000


def wall_time(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    import time
    w0 = time.perf_counter_ns()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (w1 := time.perf_counter_ns()) and (w1 - w0) / ITERS / 1e3  # us


def main():
    assert torch.cuda.is_available()
    print(f"{'T':>4}  {'chain (cast+gemv+router)':>26}  {'fold (gemv+router)':>20}  ratio")
    for T in (1, 2, 4, 8, 9, 16, 32, 64):
        torch.manual_seed(0)
        x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(E, H, device="cuda")
        bias = torch.randn(E, device="cuda")

        def chain():
            fused_router(F.linear(x.float(), w), bias, K, SCALING)

        tc = wall_time(chain)
        try:
            tf = wall_time(lambda: fused_router_dense(x, w, bias, K, SCALING))
            ratio = f"{tc / tf:5.2f}x"
        except Exception as e:  # noqa: BLE001 - the T>8 cap is the point
            tf = float("nan")
            ratio = type(e).__name__
        print(f"{T:>4}  {tc:24.2f}us  {tf:18.2f}us  {ratio}")

    # parity spot-check inside the eligible regime
    for T in (1, 8):
        torch.manual_seed(1)
        x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(E, H, device="cuda")
        bias = torch.randn(E, device="cuda")
        ri, _ = fused_router(F.linear(x.float(), w), bias, K, SCALING)
        fi, _ = fused_router_dense(x, w, bias, K, SCALING)
        flips = sum(set(ri[r].tolist()) != set(fi[r].tolist()) for r in range(T))
        print(f"parity T={T}: selection flips {flips}/{T}")


if __name__ == "__main__":
    main()
