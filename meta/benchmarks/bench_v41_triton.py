"""Sweep benchmark for the five DeepSeek-V4.1 Triton ops on CUDA/HIP.

Latency + effective logical bandwidth vs a measured copy/read roofline, at
real V4.1 geometries (E=384 experts, K=6/token, hidden 5120, moe intermediate
2304, 64 attn heads x D 512, 32 indexer heads x D 128, top-k 512). Mirrors
``bench_roofline.py``'s honesty rules: compilation/allocation/validation are
outside GPU timings; bandwidth is *logical* bytes / event time, not measured
physical HBM traffic.

``mxfp4_expert_gemv`` is benchmarked against TWO references:
  * ``dequant-all``  — the module oracle (decodes all 384 experts; the figure
    that produced the inflated "41x" claim);
  * ``dequant-selected`` — dequant only the K selected experts, then einsum
    (what a production fallback would actually do). The honest win figure is
    vs THIS baseline.

Timing: CUDA events, ``--warmup`` then median of ``--samples`` (defaults 15/50,
matching the first microbenchmark in the floe model TODO).
"""

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl

from vkernels.torch_ops.v41_dsa_indexer import indexer_scores, indexer_scores_reference
from vkernels.torch_ops.v41_fp8_gemm import fp8_block_gemm, fp8_block_gemm_reference, quantize_fp8_ue8m0
from vkernels.torch_ops.v41_mxfp4_dequant import mxfp4_dequant, mxfp4_dequant_reference
from vkernels.torch_ops.v41_mxfp4_gemv import mxfp4_expert_gemv, mxfp4_expert_gemv_reference
from vkernels.torch_ops.v41_sparse_attention import sparse_attention, sparse_attention_reference

E, K = 384, 6
HIDDEN, MOE_I = 5120, 2304
ATTN_H, ATTN_D = 64, 512
IDX_H, IDX_D, TOPK = 32, 128, 512


@triton.jit
def _read_sum(x, partials, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < N, other=0)
    tl.store(partials + tl.program_id(0), tl.sum(values, axis=0))


@triton.jit
def _copy(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < N, other=0)
    tl.store(y + offsets, values, offsets < N)


def time_op(operation, validate, *, warmup, samples):
    """Event-timed median. Compilation + validation are outside the timing."""
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    validate()
    times = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    validate()
    return {"median_us": statistics.median(times) * 1e3, "min_us": min(times) * 1e3,
            "samples_us": [round(t * 1e3, 1) for t in times]}


def roofline(samples):
    """512 MiB read/copy (exceeds the ~256 MiB LLC per bench_roofline's caveat)."""
    n = 512 * 1024 * 1024 // 4
    x = torch.ones(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    partials = torch.empty(triton.cdiv(n, 4096), device="cuda", dtype=torch.float32)
    out = {}
    for kind in ("read", "copy"):
        if kind == "read":
            op = partial(_read_sum[(triton.cdiv(n, 4096),)], x, partials, n, 4096, num_warps=8)
            validate = lambda: None  # noqa: E731
            logical = n * 4
        else:
            op = partial(_copy[(triton.cdiv(n, 4096),)], x, y, n, 4096, num_warps=8)
            validate = lambda: torch.testing.assert_close(y, x)  # noqa: E731
            logical = 2 * n * 4
        timing = time_op(op, validate, warmup=5, samples=samples)
        out[kind] = {"logical_gb_s": logical / (timing["median_us"] * 1e3), **timing}
    return out


def dequant_selected_reference(x, weights, scales, indices, *, group=32):
    """Honest baseline: dequant only the K selected experts with the DEVICE
    dequant op (the production-quality path a fallback would use), gather,
    then einsum. Not the slow torch reference — that would flatter the Triton
    GEMV."""
    import torch

    t, k = indices.shape
    flat = indices.reshape(-1)
    uniq, inv = torch.unique(flat, return_inverse=True)
    w = mxfp4_dequant(weights[uniq], scales[uniq], group=group, dtype=torch.bfloat16)
    w = w[inv].reshape(t, k, w.shape[1], w.shape[2])
    x3 = x if x.ndim == 3 else x[:, None, :].expand(t, k, w.shape[3])
    return torch.einsum("tki,tkoi->tko", x3.float(), w.float()).to(torch.bfloat16)


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def bench_mxfp4_dequant(results, args):
    for tag, o, i in (("w1w3", MOE_I, HIDDEN), ("w2", HIDDEN, MOE_I)):
        packed = torch.randint(0, 255, (E, o, i // 2), device="cuda", dtype=torch.uint8)
        scale = torch.rand(E, o, i // 32, device="cuda") * 0.01
        logical = E * o * i // 2 + E * o * (i // 32) * 4 + E * o * i * 2
        op = partial(mxfp4_dequant, packed, scale, dtype=torch.bfloat16)
        validate = lambda op=op, packed=packed, scale=scale: torch.testing.assert_close(  # noqa: E731
            op(), mxfp4_dequant_reference(packed, scale, dtype=torch.bfloat16), atol=2e-2, rtol=2e-2)
        results.append({"op": "mxfp4_dequant", "shape": f"[{E},{o},{i//2}] -> bf16", "tag": tag,
                        "logical_mb": logical / 1e6, **time_op(op, validate, warmup=args.warmup, samples=args.samples)})


def bench_expert_gemv(results, args):
    for tag, o, i in (("w1w3", MOE_I, HIDDEN), ("w2", HIDDEN, MOE_I)):
        packed = torch.randint(0, 255, (E, o, i // 2), device="cuda", dtype=torch.uint8)
        scale = torch.rand(E, o, i // 32, device="cuda") * 0.01
        for t in (1, 8):
            x = torch.randn(t, i, device="cuda", dtype=torch.bfloat16) * 0.1
            idx = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(t)])
            oracle = partial(mxfp4_expert_gemv_reference, x, packed, scale, idx)
            triton = partial(mxfp4_expert_gemv, x, packed, scale, idx)
            selected = partial(dequant_selected_reference, x, packed, scale, idx)
            validate = lambda: (torch.testing.assert_close(triton(), oracle(), atol=2e-2, rtol=2e-2),  # noqa: E731
                                torch.testing.assert_close(selected(), oracle(), atol=2e-2, rtol=2e-2))
            w_bytes = K * (o * i // 2 + o * (i // 32) * 4) * t  # K experts per token
            logical = w_bytes + t * i * 2 + t * K * o * 2
            base_samples = max(1, args.samples // 2)
            results.append({"op": "mxfp4_expert_gemv", "shape": f"x[{t},{i}] K{K} O{o}", "tag": tag,
                            "logical_mb": logical / 1e6,
                            **time_op(triton, validate, warmup=args.warmup, samples=args.samples),
                            "baseline_dequant_all_us": time_op(oracle, lambda: None, warmup=args.warmup // 2, samples=base_samples)["median_us"],
                            "baseline_dequant_selected_us": time_op(selected, lambda: None, warmup=args.warmup, samples=base_samples)["median_us"]})


def bench_indexer(results, args):
    import os
    for s, t in ((1, 1024), (1, 4096), (1, 32768), (64, 1024)):
        q = torch.randn(1, s, IDX_H, IDX_D, device="cuda", dtype=torch.bfloat16) * 0.1
        k = torch.randn(1, t, IDX_D, device="cuda", dtype=torch.bfloat16) * 0.1
        w = torch.rand(1, s, IDX_H, device="cuda") + 0.5
        logical = s * IDX_H * IDX_D * 2 + t * IDX_D * 2 + s * IDX_H * 4 + s * t * 4
        for backend in ("reference", "triton"):
            os.environ["VKERNELS_DSA_INDEXER_BACKEND"] = backend
            op = partial(indexer_scores, q, k, w)
            validate = lambda op=op, q=q, k=k, w=w: torch.testing.assert_close(  # noqa: E731
                op(), indexer_scores_reference(q, k, w), atol=1e-3, rtol=1e-3)
            results.append({"op": "indexer_scores", "shape": f"q[1,{s},{IDX_H},{IDX_D}] T{t}",
                            "backend": backend, "logical_mb": logical / 1e6,
                            **time_op(op, validate, warmup=args.warmup, samples=args.samples)})
    os.environ.pop("VKERNELS_DSA_INDEXER_BACKEND", None)


def bench_sparse_attn(results, args):
    import os

    for s, n in ((1, TOPK + 128), (1, 2048), (64, TOPK + 128)):
        q = torch.randn(1, ATTN_H, s, ATTN_D, device="cuda", dtype=torch.bfloat16) * 0.1
        kv = torch.randn(1, n, ATTN_D, device="cuda", dtype=torch.bfloat16) * 0.1
        mask = (torch.rand(1, s, n, device="cuda") < 0.5).float()
        sink = torch.rand(ATTN_H, device="cuda")
        logical = ATTN_H * s * ATTN_D * 2 + n * ATTN_D * 2 + s * n * 4 + ATTN_H * 4 + ATTN_H * s * ATTN_D * 4
        for backend in ("reference", "triton"):
            os.environ["VKERNELS_V41_SPARSE_ATTN_BACKEND"] = backend
            op = partial(sparse_attention, q, kv, mask, sink, 0.088)
            validate = lambda op=op, q=q, kv=kv, mask=mask, sink=sink: torch.testing.assert_close(  # noqa: E731
                op(), sparse_attention_reference(q, kv, mask, sink, 0.088), atol=1e-2, rtol=1e-2)
            results.append({"op": "sparse_attention", "shape": f"q[1,{ATTN_H},{s},{ATTN_D}] N{n}",
                            "backend": backend, "logical_mb": logical / 1e6,
                            **time_op(op, validate, warmup=args.warmup, samples=args.samples)})
    os.environ.pop("VKERNELS_V41_SPARSE_ATTN_BACKEND", None)


def bench_fp8_gemm(results, args):
    import os

    for tag, n, kk, m in (("wq_b", ATTN_H * ATTN_D, 1280, 1), ("wkv", 512, HIDDEN, 1),
                          ("wo_b", HIDDEN, ATTN_H * ATTN_D, 1), ("wo_b_m8", HIDDEN, ATTN_H * ATTN_D, 8),
                          ("wo_b_prefill", HIDDEN, ATTN_H * ATTN_D, 64)):
        a = torch.randn(m, kk, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(n, kk, device="cuda", dtype=torch.bfloat16) * 0.1
        aq, asc = quantize_fp8_ue8m0(a)
        bq, bsc = quantize_fp8_ue8m0(b)
        logical = m * kk + n * kk + (m // 32 + n // 32) * (kk // 32) * 4 + m * n * 2
        # "triton" forces the decode kernel; unset -> auto/reference dispatch.
        for backend in ("reference", "triton"):
            os.environ["VKERNELS_V41_FP8_GEMM_BACKEND"] = backend
            op = partial(fp8_block_gemm, aq, asc, bq, bsc, block=32)
            validate = lambda op=op, aq=aq, asc=asc, bq=bq, bsc=bsc: torch.testing.assert_close(  # noqa: E731
                op(), fp8_block_gemm_reference(aq, asc, bq, bsc, block=32), atol=1e-2, rtol=1e-2)
            results.append({"op": "fp8_block_gemm", "shape": f"[{m},{kk}]x[{n},{kk}]^T block32", "tag": tag,
                            "backend": backend, "logical_mb": logical / 1e6,
                            **time_op(op, validate, warmup=args.warmup, samples=args.samples)})
    os.environ.pop("VKERNELS_V41_FP8_GEMM_BACKEND", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("a CUDA/HIP GPU is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    report = {"device_name": props.name, "torch": torch.__version__, "triton": triton.__version__,
              "bandwidth_definition": "logical bytes / GPU event time; not measured physical HBM traffic",
              "roofline": roofline(args.samples), "results": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for bench in (bench_mxfp4_dequant, bench_expert_gemv, bench_indexer, bench_sparse_attn, bench_fp8_gemm):
        bench(report["results"], args)
        save()

    for r in report["results"]:
        print(json.dumps({k: v for k, v in r.items() if k != "samples_us"}), flush=True)

    roof_gb = report["roofline"]["copy"]["logical_gb_s"]
    print(f"\ncopy roofline: {roof_gb:.0f} GB/s | read: {report['roofline']['read']['logical_gb_s']:.0f} GB/s")
    for r in report["results"]:
        eff = r["logical_mb"] / 1e3 / (r["median_us"] / 1e6)
        line = f"{r['op']:<20} {r['shape']:<28} {r['median_us']:>9.1f} us  {eff:>7.0f} GB/s"
        if "baseline_dequant_selected_us" in r:
            sel = r["baseline_dequant_selected_us"]
            line += f"  | vs dequant-selected {sel:.1f} us ({sel / r['median_us']:.2f}x) vs dequant-all {r['baseline_dequant_all_us']:.1f} us"
        if r.get("backend"):
            line += f" [{r['backend']}]"
        print(line)


if __name__ == "__main__":
    main()
