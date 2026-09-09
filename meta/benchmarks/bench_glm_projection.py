"""MI300 GLM small-projection bakeoff (#66 / #68); tune eagerly, then time
immutable graphs.

Run with PYTHONPATH=src/python python meta/benchmarks/bench_glm_projection.py.
Default BLAS vs TunableOp-selected BLAS vs the vkernels Triton split-K
operator at the GLM decode shapes:

  #68 KDA gate projections:  b_proj [64,4096], f_a/g_a [128,4096]
  #66 DSA input projections: kv_a_with_mqa [512,4096], q_a [1536,4096]

x is M=1 (decode) and M=2. Rotating 64 weight sets per shape group so the
graph streams real weight bytes; parity is checked before timing.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.cuda.tunable as tunable

from vkernels.torch_ops.glm_projection import glm_projection, glm_projection_tuning_metadata


def parity(actuals, references):
    max_abs, relative_l2 = 0.0, 0.0
    for actual, expected in zip(actuals, references):
        error = actual.float() - expected.float()
        relative = (error.norm() / expected.float().norm().clamp_min(1e-12)).item()
        max_abs = max(max_abs, error.abs().max().item())
        relative_l2 = max(relative_l2, relative)
        assert torch.isfinite(actual).all()
        assert relative < 0.005
    return {"max_abs": max(max_abs, 0.0), "max_relative_l2": max(relative_l2, 0.0)}


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
            "min_us": min(samples) * 1000}


SHAPES = [  # (N, K, tag)
    (64, 4096, "#68 b_proj"),
    (128, 4096, "#68 f_a/g_a"),
    (512, 4096, "#66 kv_a_mqa"),
    (1536, 4096, "#66 q_a"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tunable.set_filename(str(args.output.with_suffix(".tunable.csv")), insert_device_ordinal=True)
    tunable.enable(False)
    report = {"torch": torch.__version__, "hip": torch.version.hip,
              "device": str(torch.cuda.get_device_properties(0)),
              "weights": 64, "shapes": {}}
    print(f"device: {torch.cuda.get_device_name(0)}")
    for tokens in (1, 2):
        for n, k, tag in SHAPES:
            pairs = [(torch.randn(1, tokens, k, device="cuda", dtype=torch.bfloat16),
                      torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 8)
                     for _ in range(64)]
            references = [F.linear(x, w) for x, w in pairs]
            result = {"blas": graph_time(F.linear, pairs, references)}
            tunable.enable(True)
            tunable.tuning_enable(True)
            F.linear(*pairs[0])
            torch.cuda.synchronize()
            tunable.tuning_enable(False)
            result["tuned_blas"] = graph_time(F.linear, pairs, references)
            result["tunable_results"] = tunable.get_results()
            result["triton"] = graph_time(glm_projection, pairs, references)
            result["triton_tuning"] = glm_projection_tuning_metadata()
            report["shapes"][f"M={tokens} N={n} K={k}"] = result
            print(f"M={tokens} {tag:16s} N={n:4d}: "
                  f"blas={result['blas']['median_us']:8.2f}us  "
                  f"tuned={result['tuned_blas']['median_us']:8.2f}us  "
                  f"triton={result['triton']['median_us']:8.2f}us", flush=True)
            tunable.enable(False)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
