"""Cold counter probes; profiling time is NOT a performance measurement.

Use rocprofv3 kernel filters: 'project' for qkv, 'Cijk_' for gate128.
Discard the first matching dispatch (warmup); the following eight are cold.
"""

import argparse
import importlib
import json

import torch
import torch.nn.functional as F

from bench_roofline import _read_sum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("qkv", "gate128"), required=True)
    args = parser.parse_args()
    x = torch.ones(1, 4096, device="cuda", dtype=torch.bfloat16)
    flush = torch.ones(512 * 1024 * 1024, device="cuda", dtype=torch.float32)
    partials = torch.empty(flush.numel() // 4096, device="cuda")
    if args.case == "qkv":
        weights = [tuple(torch.ones(8192, 4096, device="cuda", dtype=torch.bfloat16)
                         for _ in range(3)) for _ in range(8)]
        module = importlib.import_module("vkernels.torch_ops.qkv_projection")
        # Freeze the actual kernel configuration used on GPUs0–2 in job628645.
        # Direct JIT launch avoids profiling-driven autotuning decisions.
        project = module._kernel().fn
        output = torch.empty(1, 24576, device="cuda", dtype=torch.bfloat16)

        def run(index):
            project[(4096, 3, 1)](x, *weights[index], output, 1, 0, ROWS=2,
                                  num_warps=8, enable_fp_fusion=False)
            return output

        logical_weights = 3 * 8192 * 4096 * 2
        useful_flops = 2 * 3 * 8192 * 4096
    else:
        weights = [torch.ones(128, 4096, device="cuda", dtype=torch.bfloat16) for _ in range(8)]

        def run(index):
            return F.linear(x, weights[index])

        logical_weights = 128 * 4096 * 2
        useful_flops = 2 * 128 * 4096
    run(0)
    torch.cuda.synchronize()
    for index in range(8):
        _read_sum[(partials.numel(),)](flush, partials, flush.numel(), 4096, num_warps=8)
        torch.cuda.synchronize()
        output = run(index)
        torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 4096), rtol=0, atol=0)
    print(json.dumps({"case": args.case, "matching_dispatches": 9, "discard_warmup_dispatches": 1,
                      "cold_dispatches": 8, "flush_bytes": flush.numel() * 4,
                      "weight_bytes_per_dispatch": logical_weights,
                      "useful_flops_per_dispatch": useful_flops,
                      "timing_is_instrumented": True}), flush=True)


if __name__ == "__main__":
    main()
