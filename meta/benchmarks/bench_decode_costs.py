"""Measure preparation, checked decode, prepared decode, and graph replay.

This is a latency benchmark, not a FLOP/s or bandwidth roof measurement.
Use bench_roofline.py for hardware calibration before interpreting throughput.
No clock settings are changed; compare runs only under the same conditions.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from vkernels.compiler import compile_model
from vkernels.compiler.model_qwen3 import tiny_qwen3_config


def distribution(values):
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": mean,
        "std_ms": std,
        "cv": std / mean if mean else 0,
        "samples": len(values),
    }


def measure(operation, *, gpu=False, max_samples=512, target_cv=0.02):
    for _ in range(8):
        operation()
    torch.cuda.synchronize()
    values = []
    while len(values) < max_samples:
        for _ in range(max(16, len(values))):
            if gpu:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                operation()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
            else:
                start = time.perf_counter()
                operation()
                torch.cuda.synchronize()
                values.append((time.perf_counter() - start) * 1e3)
        if distribution(values)["cv"] <= target_cv:
            break
    result = distribution(values)
    result["stable"] = result["cv"] <= target_cv
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(0)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda.synchronize()
        start = time.perf_counter()
        exe = compile_model(model_config=tiny_qwen3_config(layers=args.layers), workers=2)
        runner = exe.prepare_device()
        torch.cuda.synchronize()
        prepare_ms = (time.perf_counter() - start) * 1e3
        ids = torch.tensor([3], dtype=torch.int64, device="cuda")
        pos = torch.tensor([0], dtype=torch.int32, device="cuda")
        reference = runner.run([3], [0]).clone()
        prepared = lambda: runner.run_device(ids, pos)
        checked = measure(lambda: runner.run([3], [0]))
        eager = measure(prepared)
        gpu = measure(prepared, gpu=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            prepared()
        replay = measure(graph.replay, gpu=True)
        torch.testing.assert_close(runner.logits, reference)
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "metric": "latency; no throughput/speedup claim",
        "clocks": "unmodified",
        "prepare_ms_including_compile_and_allocation": prepare_ms,
        "workspace_bytes": exe.workspace_plan.total_bytes,
        "workspace_without_reuse_bytes": exe.workspace_plan.naive_bytes,
        "checked_host_end_to_end": checked,
        "prepared_end_to_end": eager,
        "prepared_gpu_events": gpu,
        "graph_replay_gpu_events": replay,
        "validation": "output parity after timing passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
