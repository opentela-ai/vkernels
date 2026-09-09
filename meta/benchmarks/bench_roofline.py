"""Calibrate logical streaming bandwidth, BF16 GEMM throughput, and launch cost.

These are achieved, workload-specific roofs, not hardware-counter measurements
of physical HBM traffic. In particular the 64 MiB case may benefit from cache.
Compilation, allocation, initialization, and validation are outside GPU timings.
"""

import argparse
from functools import partial
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


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


def graph_time(operation, validate, *, iterations=1, samples=15):
    # Warm up on a non-default stream before capture (also compiles Triton).
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    validate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            operation()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / iterations)
    validate()
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "samples_ms": times,
        "graph_iterations": iterations,
        "warm_graph_replays": 3,
        "validation": "passed_before_and_after_timing",
    }


def check_constant(tensor, expected):
    # Only two scalars cross to the host; no full-size host copy is needed.
    actual_min = tensor.min().item()
    actual_max = tensor.max().item()
    if actual_min != expected or actual_max != expected:
        raise AssertionError((actual_min, actual_max, expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes-mib", nargs="+", type=int, default=[64, 512, 2048])
    parser.add_argument("--gemm-size", type=int, choices=[4096, 8192], default=8192)
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    if args.samples < 1 or any(size < 1 for size in args.sizes_mib):
        parser.error("sample count and sizes must be positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA/HIP GPU is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "device": str(props),
        "device_name": props.name,
        "total_memory_bytes": props.total_memory,
        "multiprocessor_count": props.multi_processor_count,
        "architecture": getattr(props, "gcnArchName", None),
        "bandwidth_definition": "logical bytes / GPU event time; not measured physical HBM bytes",
        "bandwidth_caveat": "64 MiB may be cache-resident; 512 MiB and 2 GiB exceed 256 MiB LLC",
        "configuration_search": "bounded sweep, not a claim of globally optimal configurations",
        "streams": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for size_mib in args.sizes_mib:
        n = size_mib * 1024 * 1024 // 4
        x = torch.ones(n, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        for block in (1024, 4096):
            partials = torch.empty(triton.cdiv(n, block), device="cuda", dtype=torch.float32)
            for warps in (4, 8):
                for kind in ("read_sum", "copy"):
                    if kind == "read_sum":
                        operation = partial(_read_sum[(triton.cdiv(n, block),)],
                                            x, partials, n, block, num_warps=warps)
                        validate = partial(check_constant, partials, block)
                        logical_bytes = (n + partials.numel()) * 4
                    else:
                        operation = partial(_copy[(triton.cdiv(n, block),)],
                                            x, y, n, block, num_warps=warps)
                        validate = partial(check_constant, y, 1)
                        logical_bytes = 2 * n * 4
                    timing = graph_time(operation, validate, samples=args.samples)
                    result = {
                        "kind": kind, "input_mib": size_mib, "block": block,
                        "num_warps": warps, "logical_bytes": logical_bytes,
                        "effective_logical_gb_s": logical_bytes / (timing["median_ms"] * 1e6),
                        **timing,
                    }
                    report["streams"].append(result)
                    save()
                    print(json.dumps(result), flush=True)
            del partials
        del x, y

    x = torch.ones(1, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    report["launch_floor"] = graph_time(
        lambda: _copy[(1,)](x, y, 1, 1, num_warps=4),
        lambda: check_constant(y, 1), iterations=20, samples=args.samples,
    )
    report["launch_floor"]["interpretation"] = "amortized one-element copy GPU time within a 20-node graph"
    save()
    n = args.gemm_size
    a = torch.ones((n, n), device="cuda", dtype=torch.bfloat16)
    b = torch.ones_like(a)
    c = torch.empty_like(a)
    timing = graph_time(lambda: torch.mm(a, b, out=c),
                        lambda: check_constant(c, n), samples=args.samples)
    report["bf16_gemm"] = {
        "n": n, "flops": 2 * n**3, "input_dtype": "bfloat16",
        "output_dtype": "bfloat16", "implementation": "default torch.mm BLAS",
        "achieved_tflops": 2 * n**3 / (timing["median_ms"] * 1e9), **timing,
    }
    save()
    print(json.dumps({"launch_floor": report["launch_floor"],
                      "bf16_gemm": report["bf16_gemm"]}), flush=True)


if __name__ == "__main__":
    main()
