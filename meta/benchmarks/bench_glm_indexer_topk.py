"""GLM-5.3 DSA indexer top-k bakeoff: vkernels glm_indexer_topk vs the two
incumbents it must beat (trace kin-3498744: radixSelect + radixSortKVInPlace
~16.6-19 us x 11 indexer layers/step at decode).

Run with PYTHONPATH=src/python python meta/benchmarks/bench_glm_indexer_topk.py.

The ADOPTION GATE (see torch_ops/glm_indexer_topk.py docstring): the fused
op is only worth wiring if it beats BOTH

  1. ``torch.topk(sorted=True)``  — the bit-identical default path, and
  2. ``torch.topk(sorted=False)`` — the zero-code ``indexer_topk_unsorted``
     knob flip (select without the k-wide radix sort),

at the wired decode shapes (T = batch, N = n_pools = kv_len // index_kpool,
k = index_topk // index_kpool = 512 at the shipped config). Parity is
checked before timing; a kernel that loses to the free knob must not ship.

Both score dtypes are swept: fp32 (the exact-reference policy) and bf16
(the deployed ``--model-opt gemm_dtype=bfloat16`` recipe — the op widens
bf16 to fp32 in-kernel; see the DTYPE paragraph in the op's docstring).
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from vkernels.torch_ops.glm_indexer_topk import glm_indexer_topk

# (T, N, k) — decode envelopes across context lengths, then prefill chunks.
SHAPES = [
    (1, 4096, 512, "decode bs=1 @16k ctx"),
    (1, 16384, 512, "decode bs=1 @64k ctx"),
    (8, 32768, 512, "decode bs=8 @128k ctx"),
    (32, 32768, 512, "decode bs=32 @128k ctx"),
    (64, 8192, 512, "decode bs=64 @32k ctx"),
    (512, 8192, 512, "prefill 512 rows @32k ctx"),
    (2048, 32768, 512, "prefill 2048 rows @128k ctx"),
]


def graph_time(fn, scores, k, repeats=50):
    """Median/min us per launch, timed through an immutable CUDA graph
    (the serving decode path replays captured steps)."""
    scores.copy_(torch.randn_like(scores))  # parity on fresh data
    values, indices = fn(scores, k)
    torch.cuda.synchronize()
    tv, ti = torch.topk(scores, k, dim=-1)
    # order-agnostic: the unsorted variants return arbitrary k-axis order
    assert torch.equal(values.sort(-1).values, tv.sort(-1).values), "parity: values differ from torch.topk"
    assert torch.equal(indices.long().sort(-1).values, ti.sort(-1).values), "parity: selection set differs"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(scores, k)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return {"median_us": statistics.median(samples), "min_us": min(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
              "indexer_layers_per_step": 11, "shapes": {}}
    print(f"device: {torch.cuda.get_device_name(0)}")

    def torch_sorted(scores, k):
        return torch.topk(scores, k, dim=-1, sorted=True)

    def torch_unsorted(scores, k):
        v, i = torch.topk(scores, k, dim=-1, sorted=False)
        return v, i

    def op_sorted(scores, k):
        return glm_indexer_topk(scores, k, sorted=True)

    def op_unsorted(scores, k):
        return glm_indexer_topk(scores, k, sorted=False)

    variants = [
        ("torch_sorted", torch_sorted),        # incumbent 1 (default)
        ("torch_unsorted", torch_unsorted),    # incumbent 2 (free knob flip)
        ("vk_sorted", op_sorted),
        ("vk_unsorted", op_unsorted),
    ]
    for tokens, pools, k, tag in SHAPES:
        for dtype in (torch.float32, torch.bfloat16):
            dt_name = str(dtype).split(".")[-1]
            scores = torch.randn(tokens, pools, device="cuda", dtype=dtype)
            result = {"dtype": dt_name}
            medians = {}
            for name, fn in variants:
                # eager warmup JITs before any capture
                fn(scores, k)
                torch.cuda.synchronize()
                timing = graph_time(fn, scores, k)
                result[name] = timing
                medians[name] = timing["median_us"]
            gate = medians["vk_unsorted"] < medians["torch_unsorted"] \
                and medians["vk_sorted"] < medians["torch_sorted"]
            result["adoption_gate"] = "PASS" if gate else "FAIL"
            result["per_step_us_x11"] = {n: round(medians[n] * 11, 1) for n in medians}
            report["shapes"][f"T={tokens} N={pools} k={k} {dt_name}"] = result
            print(f"T={tokens:5d} N={pools:6d} {dt_name} {tag}: " +
                  "  ".join(f"{n}={medians[n]:7.2f}us" for n, _ in variants) +
                  f"  gate={result['adoption_gate']}", flush=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
