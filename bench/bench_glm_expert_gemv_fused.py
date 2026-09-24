"""A/B benchmark: fused-epilogue expert GEMV+silu vs the two-launch chain.

Measures, under the repo benchmarking discipline (.agents/skills/
kernel-benchmarking: warmup discard, batched 1000-launch CUDA-event pairs,
median + spread over 4 batches):

  UNFUSED: expert_gemv (writes gu[T,K,2*IA]) + elementwise.silu_mul
  FUSED:   glm_expert_gemv_fused.expert_gemv_silu (writes act[T,K,IA] only)

Byte model per decode step (t tokens, k slots, serving shape IA=2048
I=4096): both arms read the same 8*k... k*2*IA*I fp8 weight bytes (the
dominant term, ~128 MiB/slot-set at k=8); the unfused arm adds
k*(4*O... precisely 2*O*2 write + 2*O*2 read + IA*2 write bytes of bf16
activation round-trip = k*O*6 bytes ~ 196 KB at k=8 — 0.15% of the weight
stream. The expected win is therefore LAUNCH COUNT and the elimination of
the gate/up store->load dependency, not bandwidth (see
NOTES-fusion-glm-decode-fused.md for the roofline argument).

GB10 numbers are indicative only; MI300A is the serving target.

Usage: .venv/bin/python bench/bench_glm_expert_gemv_fused.py [--batches 4]
[--iters 1000] [--t 1 2]
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

import torch  # noqa: E402


def make_serving_inputs(e, ia, i, t, k, device):
    torch.manual_seed(7)
    raw = torch.randint(0, 256, (e, 2 * ia, i), dtype=torch.uint8)
    raw.masked_fill_((raw & 127) == 127, 0)
    weights = raw.view(torch.float8_e4m3fn).to(device)
    scales = (0.0005 + torch.rand((e, 2 * ia // 128, i // 128)) * 0.001).to(device)
    indices = torch.rand((t, e)).topk(k, dim=-1).indices.to(torch.int64).to(device)
    x = torch.randn((t, i), device=device, dtype=torch.bfloat16)
    return weights, scales, indices, x


def time_batched(fn, iters, batches, warmup=25):
    """Warmup discard, then `batches` event-pair timings of `iters` launches."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per_launch = []
    for _ in range(batches):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        torch.cuda.synchronize()
        per_launch.append(start.elapsed_time(stop) * 1000.0 / iters)  # us
    return per_launch


def report(name, per_launch, bytes_per_call):
    med = statistics.median(per_launch)
    spread = (min(per_launch), max(per_launch))
    gbs = bytes_per_call / (med * 1e-6) / 1e9
    print(f"  {name:<28} {per_launch[0]:8.2f} {per_launch[1]:8.2f} "
          f"{per_launch[2]:8.2f} {per_launch[3]:8.2f} | med {med:8.2f} us "
          f"(spread {spread[1]-spread[0]:5.2f}) | {gbs:7.1f} GB/s-model")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--t", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()

    from vkernels.torch_ops.elementwise import silu_mul
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv
    from vkernels.torch_ops.glm_expert_gemv_fused import expert_gemv_silu

    assert torch.cuda.is_available(), "GB10 GPU required"
    device = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"torch {torch.__version__} (indicative-only numbers; "
          f"MI300A is the serving target)")

    # GLM-5.3-Flash decode shapes (docs/glm53-decode-kernels.md): stacked
    # gate/up [4096,4096] per expert => IA=2048, I_in=4096; top-8 routing.
    for e, ia, i in ((32, 2048, 4096), (32, 1024, 2048)):
        for t in args.t:
            k = 8
            weights, scales, indices, x = make_serving_inputs(e, ia, i, t, k, device)
            o = 2 * ia

            def unfused():
                gu = expert_gemv(x, weights, scales, indices)
                return silu_mul(gu[..., :ia], gu[..., ia:])

            def fused():
                return expert_gemv_silu(x, weights, scales, indices)

            # parity sanity inside the bench (bit-exact chain contract)
            assert torch.equal(fused(), unfused())

            w_bytes = t * k * o * i                      # fp8 weights, both arms
            u_bytes = w_bytes + t * k * (o * 2 + o * 2 + ia * 2)   # + gu round trip
            f_bytes = w_bytes + t * k * ia * 2           # + act write only
            print(f"\nE={e} IA={ia} I={i} t={t} k={k}  "
                  f"weights {w_bytes/2**20:.1f} MiB/call; "
                  f"activation round-trip saved "
                  f"{t*k*(o*4 + ia*2)/1024:.0f} KiB/call "
                  f"({100*(u_bytes-f_bytes)/u_bytes:.2f}% of bytes)")

            print(f"  {'arm':<28} {'b1':>8} {'b2':>8} {'b3':>8} {'b4':>8} "
                  "| median | GB/s-model")
            mu = report("unfused (2 launches)", time_batched(unfused, args.iters,
                                                             args.batches), u_bytes)
            mf = report("fused (1 launch)", time_batched(fused, args.iters,
                                                         args.batches), f_bytes)
            print(f"  => fused/unfused wall: {mf/mu:.3f}x  "
                  f"(saved {mu-mf:.2f} us/call, {100*(mu-mf)/mu:.1f}%)")


if __name__ == "__main__":
    main()
