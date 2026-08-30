#!/usr/bin/env python3
"""Per-kernel profiler for the vkernels MoE path (issue #46 follow-up).

Parses one or more ``torch.profiler`` chrome traces emitted by
``src/python/vkernels/vllm_experts.py`` -- which wraps the eager-break
``VkernelFusedExperts.apply`` body in these ``record_function`` regions:

    moe:vkernel_apply          (outer, the whole apply)
      moe:apply.cpu_copy       topk_ids / expert_map  GPU->host (.cpu())
      moe:apply.cpu_align      pure-Python _moe_align_block_size_cpu
      moe:apply.gpu_copy       sids / eids            host->device
      moe:apply.launch         the C-ABI fused_moe_mxfp4 launch

and prints, **per traced rank**:

  * the per-call mean of each region (reliable: a per-event mean, immune
    to the cross-thread overlap that makes aggregate sums overcount),
  * ``cpu_copy`` as a fraction of ``moe:vkernel_apply`` (the PP0 gate
    indicator -- 97-100% means the apply is dominated by the host sync
    that waits for the expert-dispatch all-to-all),
  * a PP0-vs-PP1 ratio (PP0 computes routing on GPU and pays the full
    GPU->host sync; PP1/PP2 receive it via ``recv_object`` and only do a
    small host memcpy).

With two traces (eager vs breakable) it also emits a head-to-head table
on the PP0 (rank 0) dominant compute thread. Aggregate fractions of the
captured window are NOT reported here because a chrome trace spans many
overlapping threads (vLLM V1 compute + async workers); only per-call
means and per-thread spans are trustworthy. The full request-level
throughput / latency comparison lives in the cookbook's BENCHMARK.md.

Usage:
    # single trace, all traced ranks
    python3 meta/benchmarks/moe_profile.py path/to/step_profile_rank{0,8,16}.json

    # head-to-head (eager vs breakable) on rank 0
    python3 meta/benchmarks/moe_profile.py \\
        --label eager   run-603394/step_profiles/step_profile_rank0.json \\
        --label eager   run-603394/step_profiles/step_profile_rank8.json \\
        --label eager   run-603394/step_profiles/step_profile_rank16.json \\
        --label breakable run-603395/step_profiles/step_profile_rank0.json \\
        --label breakable run-603395/step_profiles/step_profile_rank8.json \\
        --label breakable run-603395/step_profiles/step_profile_rank16.json \\
        --head-to-head

    # machine-readable
    python3 meta/benchmarks/moe_profile.py ... --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

SUB_REGIONS = ("moe:apply.cpu_copy", "moe:apply.cpu_align",
               "moe:apply.gpu_copy", "moe:apply.launch")
OUTER = "moe:vkernel_apply"


def load_events(path):
    with open(path) as f:
        return json.load(f)["traceEvents"]


def mean_us(events, name, tid=None):
    """Per-call mean duration (us) of X-events named ``name``.

    A per-event mean is reliable regardless of cross-thread overlap (unlike
    aggregate sums, which overcount when vLLM's compute + async threads
    both record a region).
    """
    d = [e["dur"] for e in events
         if e.get("name") == name and "dur" in e
         and (tid is None or e.get("tid") == tid)]
    return (sum(d) / len(d), len(d)) if d else (0.0, 0)


def dominant_tid(events, name=OUTER):
    """Thread (tid) that issued the most ``name`` events -- the compute
    thread on a pipeline stage, as opposed to async/profiler threads."""
    c = Counter(e.get("tid") for e in events
                if e.get("name") == name and "dur" in e)
    return c.most_common(1)[0][0] if c else None


def span_us(events, tid=None):
    ts = [e["ts"] for e in events
          if isinstance(e.get("ts"), (int, float))
          and (tid is None or e.get("tid") == tid)]
    return (max(ts) - min(ts)) if ts else 0.0


def rank_row(path, rank):
    """Per-call means on the dominant compute thread of ``path``."""
    ev = load_events(path)
    tid = dominant_tid(ev)
    vm, vc = mean_us(ev, OUTER, tid)
    sub = {s: mean_us(ev, s, tid)[0] for s in SUB_REGIONS}
    cpu_frac = (sub["moe:apply.cpu_copy"] / vm * 100.0) if vm else 0.0
    return dict(path=path, rank=rank, tid=tid, n=vc,
                outer=vm, cpu_copy_frac=cpu_frac, **sub,
                span_us=span_us(ev, tid))


def fmt_table(rows, label=None):
    head = ("%-10s %4s %8s %9s %7s %8s %7s %9s %9s"
            % ((label or "set"), "rank", "n_apply",
               "vk_apply", "cpu_cp", "cpu_al", "gpu_cp", "launch",
               "cpu_cp%"))
    out = [head, "-" * len(head)]
    for r in rows:
        out.append("%-10s %4d %8d %9.1f %7.1f %8.1f %7.1f %9.1f %8.0f%%"
                   % (label or r["set"], r["rank"], r["n"],
                      r["outer"], r["moe:apply.cpu_copy"],
                      r["moe:apply.cpu_align"], r["moe:apply.gpu_copy"],
                      r["moe:apply.launch"], r["cpu_copy_frac"]))
    return "\n".join(out)


def fmt_head_to_head(eager, breakable):
    """Per-call head-to-head on rank 0's dominant compute thread."""
    e = next((r for r in eager if r["rank"] == 0), None)
    b = next((r for r in breakable if r["rank"] == 0), None)
    if not (e and b):
        return "(no rank 0 in one of the sets)"
    lines = ["head-to-head (rank 0 / PP0, dominant compute thread):",
             "  %-22s %9.1f -> %9.1f us  (%.2fx)"
             % ("moe:vkernel_apply", e["outer"], b["outer"],
                b["outer"] / e["outer"] if e["outer"] else 0.0),
             "  %-22s %9.1f -> %9.1f us  (%.2fx)"
             % ("moe:apply.cpu_copy", e["moe:apply.cpu_copy"],
                b["moe:apply.cpu_copy"],
                b["moe:apply.cpu_copy"] / e["moe:apply.cpu_copy"]
                if e["moe:apply.cpu_copy"] else 0.0)]
    for s in ("moe:apply.cpu_align", "moe:apply.gpu_copy",
              "moe:apply.launch"):
        lines.append("  %-22s %9.1f -> %9.1f us  (%.2fx)"
                     % (s, e[s], b[s], b[s] / e[s] if e[s] else 0.0))
    # PP0 vs PP1 within each set (the pipeline-gate signal)
    for lbl, rs in (("eager", eager), ("breakable", breakable)):
        r0 = next((r for r in rs if r["rank"] == 0), None)
        r1 = next((r for r in rs if r["rank"] != 0), None)
        if r0 and r1 and r1["outer"]:
            lines.append("  %-22s %s: PP0/PP1 = %.1fx (PP0 is the gate)"
                         % ("moe:vkernel_apply", lbl,
                            r0["outer"] / r1["outer"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*",
                    help="chrome-trace .json paths "
                         "(step_profile_rank{0,8,16}.json)")
    ap.add_argument("--label", action="append", default=[],
                    help="label for the NEXT trace(s); repeat to build "
                         "named sets, e.g. --label eager a.json b.json "
                         "--label breakable c.json")
    ap.add_argument("--head-to-head", action="store_true",
                    help="emit a per-call eager-vs-breakable table on "
                         "rank 0 (requires exactly two labels)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of tables")
    args = ap.parse_args(argv)

    # Map trace paths to label sets. If --label is used, every trace must
    # be preceded by (or share) a label; bare traces form an unnamed set.
    sets = {}
    if args.label:
        # labels apply positionally to the traces in order; simplest UI:
        # user re-states the label before each group. We accept either
        # `--label X a b --label Y c d` (label owns following bare args)
        # or interleaving. Here: assign by matching count.
        if len(args.label) == 1 and len(args.traces) >= 1:
            # all traces under one label
            sets[args.label[0]] = args.traces
        elif len(args.label) == 2 and len(args.traces) >= 2:
            # split traces evenly between the two labels
            half = len(args.traces) // 2
            sets[args.label[0]] = args.traces[:half]
            sets[args.label[1]] = args.traces[half:]
        else:
            ap.error("with --label, pass 1 label (all traces) or 2 labels "
                     "(split evenly: eager... breakable...)")
    else:
        sets["trace"] = args.traces

    def rank_of(p):
        n = os.path.basename(p)
        for tok in ("rank0", "rank8", "rank16", "rank1", "rank2"):
            if tok in n:
                return int(tok[4:])
        return -1

    all_rows = {}
    for lbl, paths in sets.items():
        rows = []
        for p in paths:
            if not os.path.isfile(p):
                print("warning: missing %s" % p, file=sys.stderr)
                continue
            rows.append(rank_row(p, rank_of(p)))
        rows.sort(key=lambda r: r["rank"])
        all_rows[lbl] = rows

    if args.json:
        out = {lbl: [{k: v for k, v in r.items() if k != "set"}
                     for r in rows]
               for lbl, rows in all_rows.items()}
        if args.head_to_head and len(all_rows) == 2:
            lbls = list(all_rows)
            out["head_to_head"] = fmt_head_to_head(
                all_rows[lbls[0]], all_rows[lbls[1]])
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    for lbl, rows in all_rows.items():
        if rows:
            print(fmt_table(rows, label=lbl))
            print()
    if args.head_to_head and len(all_rows) == 2:
        lbls = list(all_rows)
        print(fmt_head_to_head(all_rows[lbls[0]], all_rows[lbls[1]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
