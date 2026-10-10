"""Elementwise rungs 2-3 microbench (lane 26): price the kernels that the
two elimination rungs delete, at the g26 census shapes.

Run with python meta/benchmarks/bench_elementwise_rungs2_3.py --output <json>.

Both rungs are FLOE-SIDE eliminations (no new kernel — the ladder's own
prescription), so this bench times the DELETED incumbent chains through
immutable CUDA graphs (the serving replay regime) to bank the kernel-side
half of the win; the other half is the per-node launch-gap law
(~0.74 us/node, node-cuts.md §1):

  L1  KDA constant-true mask mul (g26 census #7+#8, 68 nodes/step):
      `x * mask.unsqueeze(-1).to(x.dtype)` — bool cast + broadcast mul,
      2 launches x 34 KDA layers. The decode graphs' mask is a
      capture-static all-true constant, so the mul is the exact identity
      (x * 1.0 keeps every bf16 bit incl. NaN/inf/-0) and both launches
      are deleted. PARITY GATE: the incumbent chain's output must equal
      x bit-for-bit at every benched shape — that equality IS the
      elimination's correctness argument.

  L7  dead MoE tail fill (g26 census #1's g26 half, 42 nodes/step):
      `torch.zeros_like(x)` at the routed-expert output shape [T, H],
      materialized by Glm53Experts.forward on every call but read only
      by the per-expert loop fallback that decode never reaches —
      42 layers x 1 launch of pure dead work. PARITY GATE: the fill
      produces exact zeros (what the loop's index_add_ starts from).

Per-step columns scale the measured medians by the census layer counts
(34 KDA, 42 MoE) so they can be checked against the ladder's pricing
(L1: -0.106 ms, L7: ~-0.03 ms kernel + 42 x 0.74 us gap).
"""

import argparse
import json
import statistics
import warnings
from pathlib import Path

import torch

# g26 decode shapes: [B, S, H] hidden rows (bs=1 is the priced census;
# bs=4 shown for reference) — and the routed-expert tail [T, H].
KDA_SHAPES = [
    (1, 1, 4096, "decode bs=1"),
    (4, 1, 4096, "decode bs=4"),
]
FILL_SHAPES = [
    (1, 4096, "decode bs=1"),
    (4, 4096, "decode bs=4"),
]
KDA_LAYERS = 34
MOE_LAYERS = 42
GAP_LAW_US_PER_NODE = 0.74  # node-cuts.md §1 measured in-graph dead time


def graph_time(fn, repeats=200):
    """Median/min us per replay, timed through an immutable CUDA graph
    (the serving decode path replays captured steps)."""
    graph = torch.cuda.CUDAGraph()
    fn()  # eager warmup before capture
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        fn()
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
    report = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "gap_law_us_per_node": GAP_LAW_US_PER_NODE,
        "kda_layers_per_step": KDA_LAYERS,
        "moe_layers_per_step": MOE_LAYERS,
        "shapes": {},
    }
    print(f"device: {torch.cuda.get_device_name(0)}")

    def noop(*a, **k):
        return None

    with warnings.catch_warnings():
        # the floor probe is deliberately an empty graph (pure replay cost)
        warnings.simplefilter("ignore", UserWarning)
        empty = graph_time(noop)  # capture/replay floor

    for b, s, h, tag in KDA_SHAPES:
        x = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(b, s, dtype=torch.bool, device="cuda")

        def incumbent():
            return x * mask.unsqueeze(-1).to(x.dtype)

        # PARITY GATE: x * 1.0 == x bit-for-bit — the elimination premise.
        out = incumbent()
        assert torch.equal(out, x), "mask-mul chain is not the exact identity?!"

        t_mul = graph_time(incumbent)
        result = {
            "incumbent_cast_plus_mul_us": t_mul,
            "deleted_us": t_mul["median_us"] - empty["median_us"],
            "per_step_us_x34": round((t_mul["median_us"] - empty["median_us"]) * KDA_LAYERS, 1),
            "gap_banked_x68_nodes_us": round(68 * GAP_LAW_US_PER_NODE, 1),
            "parity": "bit-identical (torch.equal)",
        }
        report["shapes"][f"L1 mask-mul B={b} S={s} H={h} {tag}"] = result
        print(f"L1 B={b} S={s}: mul-chain={t_mul['median_us']:6.2f}us  "
              f"deleted/step x34={result['per_step_us_x34']:6.1f}us  "
              f"+gap x68={result['gap_banked_x68_nodes_us']:.1f}us", flush=True)

    for t, h, tag in FILL_SHAPES:
        x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)

        def fill():
            return torch.zeros_like(x)

        # PARITY GATE: the fill is exact zeros (the loop's index_add base).
        out = fill()
        assert torch.count_nonzero(out).item() == 0

        t_fill = graph_time(fill)
        result = {
            "incumbent_zeros_like_us": t_fill,
            "deleted_us": t_fill["median_us"] - empty["median_us"],
            "per_step_us_x42": round((t_fill["median_us"] - empty["median_us"]) * MOE_LAYERS, 1),
            "gap_banked_x42_nodes_us": round(42 * GAP_LAW_US_PER_NODE, 1),
            "parity": "exact zeros",
        }
        report["shapes"][f"L7 fill T={t} H={h} {tag}"] = result
        print(f"L7 T={t}: zeros_like={t_fill['median_us']:6.2f}us  "
              f"deleted/step x42={result['per_step_us_x42']:6.1f}us  "
              f"+gap x42={result['gap_banked_x42_nodes_us']:.1f}us", flush=True)

    report["capture_floor_us"] = empty
    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
