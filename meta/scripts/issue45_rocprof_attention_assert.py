#!/usr/bin/env python3
"""issue45_rocprof_attention_assert.py -- AC3 post-processor: assert zero
AITER / Triton *attention* kernel launches in a rocprof capture.

Issue #45 AC3: with VKERNELS MLA + KDA active, every attention kernel must go
through the vkernels C ABI (vk_hip_mla_fwd, vk_hip_kda_delta_rule_fwd); the
AITER MLA decode path is gfx950-only (mla_gluon, job 586002) and the Triton
KDA path GPU-faults on gfx942 (job 586165), so ANY AITER/Triton attention
launch in the capture is a routing failure.  AITER/Triton *MoE* kernels
(fused_moe / gemm / mx*) are the known-good serving path and are explicitly
allowed.

Input: one or more rocprof outputs -- --stats CSV (a "Kernel Name"/"Name"
column is parsed) or raw log text (pattern scan fallback).

Usage:
  python3 issue45_rocprof_attention_assert.py capture.csv [more.csv ...] \
      [--json out.json] [--expect-vk]
Exit 0 iff no forbidden kernel appears (and, with --expect-vk, at least one
vkernels attention kernel was observed).
"""
import argparse
import sys

import issue45_common as C


def main(argv=None):
    ap = argparse.ArgumentParser(description="AC3 zero-AITER/Triton-attention assert")
    ap.add_argument("inputs", nargs="*", help="rocprof CSV / log files")
    ap.add_argument("--json", dest="out_json", default="", help="machine-readable summary")
    ap.add_argument("--expect-vk", action="store_true",
                    help="also require >=1 vkernels attention kernel in the capture")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    summary = {"inputs": args.inputs, "per_file": [], "verdict": "PASS",
               "reason": "zero AITER/Triton attention kernels"}
    vk_seen = False
    for path in args.inputs:
        r = C.analyze_rocprof_file(path)
        r["file"] = path
        summary["per_file"].append(r)
        vk_seen = vk_seen or bool(r["vk_attention"])
        if r["verdict"] == "FAIL":
            summary["verdict"] = "FAIL"
            summary["reason"] = r["reason"]
    if args.expect_vk and not vk_seen and summary["verdict"] == "PASS":
        summary["verdict"] = "FAIL"
        summary["reason"] = ("no vkernels attention kernel observed in capture "
                             "(--expect-vk); the capture may not cover attention")

    for r in summary["per_file"]:
        print(f"[AC3] {r['file']}: kernels={r['n_kernels']} "
              f"forbidden={r['forbidden_attention'] or 'NONE'} "
              f"vk_attention={r['vk_attention'] or 'NONE'}")
    print(f"[AC3] verdict={summary['verdict']} ({summary['reason']})")
    if args.out_json:
        C.dump_json(summary, args.out_json)
    return 0 if summary["verdict"] == "PASS" else 1


def selftest():
    import os
    import tempfile
    td = tempfile.mkdtemp(prefix="i45ac3-")

    def w(name, content):
        p = os.path.join(td, name)
        with open(p, "w") as fh:
            fh.write(content)
        return p

    # 1. clean capture: vkernels MLA + KDA + AITER MoE GEMMs -> PASS
    clean = w("clean.csv",
              '"Index","Kernel Name","Duration_ns"\n'
              '0,"vk_hip_mla_fwd",1234\n'
              '1,"vk_hip_kda_delta_rule_fwd",5678\n'
              '2,"aiter::mx_gemm_fused_moe",999\n')
    C.selftest("ac3", C.analyze_rocprof_file(clean)["verdict"] == "PASS", clean)
    # 2. AITER attention kernel -> FAIL
    bad1 = w("aiter_mla.csv",
             '"Index","Kernel Name","Duration_ns"\n'
             '0,"aiter::mla_decode_mla_gluon",10\n')
    C.selftest("ac3", C.analyze_rocprof_file(bad1)["verdict"] == "FAIL", bad1)
    # 3. Triton attention kernel -> FAIL
    bad2 = w("triton_attn.csv",
             '"Index","Kernel Name","Duration_ns"\n'
             '0,"triton_paged_attention_lpu",10\n'
             '1,"triton_fused_moe_kernel",20\n')
    C.selftest("ac3", C.analyze_rocprof_file(bad2)["verdict"] == "FAIL", bad2)
    # 4. AITER MoE only -> PASS (AC3 scopes attention kernels)
    moe = w("moe_only.csv",
            '"Index","Kernel Name","Duration_ns"\n'
            '0,"aiter::fused_moe_mx_bf16",10\n')
    C.selftest("ac3", C.analyze_rocprof_file(moe)["verdict"] == "PASS", moe)
    # 5. --expect-vk on a capture without vkernels kernels -> FAIL
    r = C.analyze_rocprof_file(moe)
    C.selftest("ac3", r["vk_attention"] == [], "--expect-vk sees no vk kernels")
    # 6. raw log fallback (no CSV header)
    log = w("log.txt", "some log line\n| vk_hip_mla_fwd | 1.2ms |\n"
                       "launch aiter::mla_prefill\n")
    C.selftest("ac3", C.analyze_rocprof_file(log)["verdict"] == "FAIL", log)
    # 7. att. keyword list must catch the two historical offenders
    for kn in ("mla_gluon", "chunk_kda_with_fused_gate", "fused_recurrent_kda"):
        one = w("one.csv", f'"Kernel Name"\n"aiter::{kn}"\n')
        C.selftest("ac3", C.analyze_rocprof_file(one)["verdict"] == "FAIL", kn)
    print("issue45_rocprof_attention_assert selftest: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
