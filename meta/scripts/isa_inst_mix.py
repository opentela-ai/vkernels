#!/usr/bin/env python3
"""Instruction-mix counter for a kernel's inner loop in an AMDGPU ISA dump.

Answers questions like "how many VALU warp-instructions does the dequant cost
per K-block, against how many MFMAs?" straight off the generated assembly, so
an "issue demand" claim in a comment or doc can quote a static count instead
of an estimate.  This is how the `~30x dequant-ALU / MFMA` figure behind the
issue #76 variant-3 rejection was measured (see
docs/performance/moe-fused/gfx942.md).

Produce the dump with the offload-arch the kernel actually ships for, e.g.

    hipcc -c --offload-arch=gfx942 -O3 -std=c++17 -DVKERNELS_HAS_HIP=1 \
          -I src/c src/c/vkernels/kernels/moe_fused.hip -o /tmp/m.o \
          --save-temps            # -> moe_fused-hip-amdgcn-amd-amdhsa-gfx942.s

then

    meta/scripts/isa_inst_mix.py /tmp/moe_fused-...-gfx942.s prefill_pipe
    meta/scripts/isa_inst_mix.py <dump.s> <name-substring> --loop 1
    meta/scripts/isa_inst_mix.py <dump.s> <name-substring> --range 410 1554

Kernel names are mangled, so <name-substring> is matched against the
`.type <mangled>,@function` directive; the first match wins (pass a longer
substring, e.g. `ILi16EEEvPKtPKh`, to pick one template instantiation).

Without --range, every backward `s_branch` span (a natural loop) is listed and
the *innermost* one is analysed: the block between the branch target and the
branch is the loop body, and since the prefill kb loop is fully unrolled over
the tile, its static instruction counts are its dynamic per-iteration counts.

Derived counters are separated by pipe, because that is what decides whether a
split helps: `VALU/MFMA` is the ratio of ALU issue demand to MFMA issue
demand, and `SALU` on the dequant path is almost always control flow that a
branchless formulation could remove.
"""

import argparse
import collections
import re
import sys

# Instruction prefixes per pipe.  `flat_` covers flat_load/flat_store/flat_atomic.
VMEM_PREFIXES = ("global_", "buffer_", "scratch_", "flat_load", "flat_store",
                 "flat_atomic", "s_buffer_", "tbuffer_")
KIND = [
    ("MFMA", lambda o: "mfma" in o),
    ("LDS", lambda o: o.startswith("ds_")),
    ("VMEM", lambda o: o.startswith(VMEM_PREFIXES)),
    ("SALU", lambda o: o.startswith("s_")),
    ("VALU", lambda o: o.startswith("v_")),
]


def kind_of(op):
    for name, pred in KIND:
        if pred(op):
            return name
    return "OTHER"


def parse_instructions(body):
    """(index, opcode, text) for every instruction line in `body`."""
    out = []
    for k, line in enumerate(body):
        m = re.match(r"^\s+([a-z][a-z0-9_]*)[\s$]", line)
        if m:
            out.append((k, m.group(1), line.strip()))
    return out


def backward_branch_spans(body, insts):
    """Loop spans (lo, hi, label) from backward s_branch to an earlier label."""
    labels = {}
    for k, line in enumerate(body):
        m = re.match(r"^\.?(LBB[0-9_]+):", line.strip())
        if m:
            labels[m.group(1)] = k
    spans = []
    for k, op, text in insts:
        if not op.startswith("s_branch"):
            continue
        m = re.search(r"\b(LBB[0-9_]+)\b", text)
        if m and m.group(1) in labels and labels[m.group(1)] < k:
            spans.append((labels[m.group(1)], k, m.group(1)))
    # innermost first: smallest span
    spans.sort(key=lambda t: t[1] - t[0])
    return spans


def report(body, insts, lo, hi, label):
    seg = [(o, t) for k, o, t in insts if lo <= k <= hi]
    counts = collections.Counter(o for o, _ in seg)
    by_kind = collections.Counter(kind_of(o) for o, _ in seg)
    total = len(seg)
    valu, mfma = by_kind["VALU"], by_kind["MFMA"]
    print("\n=== %s  [line %d .. %d] ===" % (label, lo, hi))
    print("  instructions   : %d" % total)
    for name, _ in KIND:
        if by_kind[name]:
            print("  %-14s : %d" % (name, by_kind[name]))
    if by_kind["OTHER"]:
        print("  %-14s : %d" % ("OTHER", by_kind["OTHER"]))
    if mfma:
        print("  VALU/MFMA      : %.2f" % (valu / mfma))
    print("  top opcodes    : " + ", ".join("%s:%d" % (o, n)
                                            for o, n in counts.most_common(12)))
    return by_kind


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help="AMDGPU .s assembly dump")
    ap.add_argument("kernel", help="substring of the mangled kernel name")
    ap.add_argument("--range", nargs=2, type=int, metavar=("LO", "HI"),
                    help="analyse this line range of the function instead of "
                         "auto-detecting the innermost loop")
    ap.add_argument("--loop", type=int, default=0, metavar="N",
                    help="analyse the N-th loop span (0 = innermost, default)")
    ap.add_argument("--list-loops", action="store_true",
                    help="only list the backward-branch loop spans and exit")
    args = ap.parse_args()

    lines = open(args.dump).read().splitlines()
    starts = [i for i, l in enumerate(lines)
              if re.match(r"\s*\.type\s+.*%s.*,@function" % re.escape(args.kernel), l)]
    if not starts:
        sys.exit("no .type directive matched %r" % args.kernel)
    k0 = starts[0]
    k1 = next((i for i in range(k0 + 1, len(lines))
               if re.match(r"\s*\.size\s+", lines[i])), len(lines))
    body = lines[k0:k1]
    name = re.search(r"\.type\s+(\S+),@function", lines[k0]).group(1)
    print("function : %s" % name)
    print("lines    : %d .. %d (%d)" % (k0, k1, len(body)))

    insts = parse_instructions(body)
    spans = backward_branch_spans(body, insts)
    if args.list_loops or (not args.range and not spans):
        print("backward s_branch loop spans (innermost first):")
        for lo, hi, lab in spans:
            print("  %6d .. %6d  span %6d  %s" % (lo, hi, hi - lo, lab))
        if not spans:
            print("  (none)")
        return

    if args.range:
        report(body, insts, args.range[0], args.range[1], "explicit range")
        return
    lo, hi, lab = spans[args.loop]
    print("loop spans (innermost first): " +
          ", ".join("%d..%d" % (a, b) for a, b, _ in spans[:6]))
    report(body, insts, lo, hi, "loop %s (innermost first index %d)" % (lab, args.loop))


if __name__ == "__main__":
    main()
