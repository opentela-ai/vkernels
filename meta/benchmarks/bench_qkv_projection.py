"""GLM decode Q/K/V bakeoff: separate BLAS, tuned BLAS, fused Triton GEMV.

Eight independent real-shape weight triplets (~1.5 GiB) avoid hot-weight timing.
Compilation and tuning happen eagerly before graph capture.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.cuda.tunable as tunable
import torch.nn.functional as F
import triton

from bench_mhc_projection import graph_time
from vkernels.torch_ops.qkv_projection import qkv_projection, qkv_projection_tuning_metadata


def blas(x, weights):
    return torch.cat([F.linear(x, weight) for weight in weights], dim=-1)


def candidate(x, weights):
    return qkv_projection(x, *weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    tunable.set_filename(str(args.output.with_suffix(".tunable.csv")), insert_device_ordinal=True)
    tunable.enable(False)
    report = {"torch": torch.__version__, "hip": torch.version.hip,
              "triton": triton.__version__, "device": str(torch.cuda.get_device_properties(0)),
              "weight_triplets": 8, "shapes": {}}
    weights = [tuple(torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
                     for _ in range(3)) for _ in range(8)]
    for tokens in (1, 2):
        pairs = [(torch.randn(1, tokens, 4096, device="cuda", dtype=torch.bfloat16), w) for w in weights]
        print(f"tokens={tokens}: baseline", flush=True)
        references = [blas(x, w) for x, w in pairs]
        result = {"blas": graph_time(blas, pairs, references)}
        print(f"tokens={tokens}: tuning BLAS", flush=True)
        tunable.enable(True)
        tunable.tuning_enable(True)
        blas(*pairs[0])
        torch.cuda.synchronize()
        tunable.tuning_enable(False)
        result["tuned_blas"] = graph_time(blas, pairs, references)
        result["tunable_results"] = tunable.get_results()
        tunable.write_file()
        tunable.enable(False)
        print(f"tokens={tokens}: tuning and validating Triton", flush=True)
        result["triton"] = graph_time(candidate, pairs, references)
        result["triton_tuning"] = qkv_projection_tuning_metadata()
        report["shapes"][str(tokens)] = result
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"tokens": tokens, **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
