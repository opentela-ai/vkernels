"""MI300 mHC projection bakeoff; tune eagerly, then time immutable graphs.

Run with PYTHONPATH=src/python python meta/benchmarks/bench_mhc_projection.py.
Synthetic BF16 inputs, rotating 90 weights (~68 MiB), no normalization included.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.cuda.tunable as tunable
import triton

from vkernels.torch_ops.mhc_projection import mhc_projection, mhc_projection_tuning_metadata


def parity(actuals, references):
    max_abs, relative_l2 = 0.0, 0.0
    for actual, expected in zip(actuals, references):
        error = actual.float() - expected.float()
        relative = (error.norm() / expected.float().norm().clamp_min(1e-12)).item()
        max_abs = max(max_abs, error.abs().max().item())
        relative_l2 = max(relative_l2, relative)
        assert torch.isfinite(actual).all()
        assert relative < 0.005
    return {"max_abs": max_abs, "max_relative_l2": relative_l2}


def graph_time(fn, pairs, references, repeats=15):
    for x, w in pairs:
        fn(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [fn(x, w) for x, w in pairs]
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    graph_parity = parity(outputs, references)
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / len(pairs))
    return {"graph_parity": graph_parity, "median_us": statistics.median(samples) * 1000,
            "min_us": min(samples) * 1000, "samples_ms_per_projection": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tunable.set_filename(str(args.output.with_suffix(".tunable.csv")), insert_device_ordinal=True)
    tunable.enable(False)
    report = {"torch": torch.__version__, "hip": torch.version.hip,
              "triton": triton.__version__, "device": str(torch.cuda.get_device_properties(0)),
              "weights": 90, "shapes": {}}
    for tokens in (1, 2):
        print(f"tokens={tokens}: baseline", flush=True)
        pairs = [(torch.randn(1, tokens, 16384, device="cuda", dtype=torch.bfloat16),
                  torch.randn(24, 16384, device="cuda", dtype=torch.bfloat16) * 0.01)
                 for _ in range(90)]
        references = [F.linear(x, w) for x, w in pairs]
        result = {"blas": graph_time(F.linear, pairs, references)}
        tunable.enable(True)
        tunable.tuning_enable(True)
        print(f"tokens={tokens}: tuning BLAS", flush=True)
        F.linear(*pairs[0])
        torch.cuda.synchronize()
        tunable.tuning_enable(False)
        result["tuned_blas"] = graph_time(F.linear, pairs, references)
        result["tunable_results"] = tunable.get_results()
        tunable.write_file()
        tunable.enable(False)
        print(f"tokens={tokens}: validating/timing Triton", flush=True)
        result["parity"] = parity([mhc_projection(x, w) for x, w in pairs], references)
        result["triton"] = graph_time(mhc_projection, pairs, references)
        result["triton_tuning"] = mhc_projection_tuning_metadata()
        report["shapes"][str(tokens)] = result
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"tokens": tokens, **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
