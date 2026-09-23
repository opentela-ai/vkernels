#!/usr/bin/env python3
"""issue45_bench_compare.py -- AC4 comparator: per-request latency of the
VKERNELS arm vs the K3_DISABLE_KDA=1 all-MLA baseline.

Issue #45 AC4: no per-request latency regression <=1.05x at C=1.  Inputs are
two benchmark JSONs in the format produced by the deployment benchmark.py /
the checked-in baseline artifact artifacts/k3-attn-accept/benchmark_638514.json
(per-level entries with concurrency / lat_p50_s / per_req_out_tok_s_med /
optional error entries for timed-out levels).

Usage:
  python3 issue45_bench_compare.py BASELINE.json MEASURED.json \
      [--threshold 1.05] [--level 1] [--json out.json]
Exit 0 iff the C=1 p50 latency ratio measured/baseline <= threshold (and the
measured arm actually completed that level).
"""
import argparse
import sys

import issue45_common as C


def level_map(path):
    """Extract {concurrency: entry} for completed levels."""
    data = C.load_json(path)
    out = {}
    for r in data.get("results", []):
        if "error" in r and r.get("error"):
            continue
        if "lat_p50_s" not in r:
            continue
        out[int(r["concurrency"])] = r
    return data, out


def main(argv=None):
    ap = argparse.ArgumentParser(description="AC4 latency baseline comparator")
    ap.add_argument("baseline", nargs="?")
    ap.add_argument("measured", nargs="?")
    ap.add_argument("--threshold", type=float, default=1.05)
    ap.add_argument("--level", type=int, default=1, help="concurrency level to gate (default C=1)")
    ap.add_argument("--json", dest="out_json", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    base_data, base = level_map(args.baseline)
    meas_data, meas = level_map(args.measured)
    lv = args.level
    out = {"baseline": args.baseline, "measured": args.measured,
           "threshold": args.threshold, "gate_level": lv, "verdict": "FAIL",
           "levels": []}
    for c in sorted(set(base) | set(meas)):
        e = {"concurrency": c}
        if c in base and c in meas:
            e["baseline_p50_s"] = base[c]["lat_p50_s"]
            e["measured_p50_s"] = meas[c]["lat_p50_s"]
            e["ratio"] = round(meas[c]["lat_p50_s"] / base[c]["lat_p50_s"], 4)
            b_tps = base[c].get("per_req_out_tok_s_med")
            m_tps = meas[c].get("per_req_out_tok_s_med")
            if b_tps and m_tps:
                e["tok_s_ratio"] = round(m_tps / b_tps, 4)
        elif c in base:
            e["baseline_p50_s"] = base[c]["lat_p50_s"]
            e["measured"] = "missing-or-timed-out"
        elif c in meas:
            e["measured_p50_s"] = meas[c]["lat_p50_s"]
            e["baseline"] = "missing"
        out["levels"].append(e)

    if lv not in base:
        out["reason"] = f"baseline has no completed C={lv} level"
    elif lv not in meas:
        out["reason"] = f"measured arm has no completed C={lv} level"
    else:
        ratio = meas[lv]["lat_p50_s"] / base[lv]["lat_p50_s"]
        out["c1_ratio"] = round(ratio, 4)
        out["reason"] = f"C={lv} p50 ratio {ratio:.3f} vs threshold {args.threshold}"
        if ratio <= args.threshold:
            out["verdict"] = "PASS"

    print(f"[AC4] baseline={args.baseline} measured={args.measured} threshold={args.threshold}")
    for e in out["levels"]:
        print(f"[AC4] C={e['concurrency']}: " + " ".join(
            f"{k}={v}" for k, v in e.items() if k != "concurrency"))
    print(f"[AC4] verdict={out['verdict']} ({out.get('reason')})")
    if args.out_json:
        C.dump_json(out, args.out_json)
    return 0 if out["verdict"] == "PASS" else 1


def selftest():
    import json
    import os
    import tempfile
    td = tempfile.mkdtemp(prefix="i45ac4-")

    def bench(path, c1, extra=()):
        data = {"results": [{"tag": "measured", "concurrency": 1, "lat_p50_s": c1,
                             "per_req_out_tok_s_med": 3.8}]}
        data["results"] += list(extra)
        with open(path, "w") as fh:
            json.dump(data, fh)
        return path

    b = bench(os.path.join(td, "base.json"), 66.7)
    m_ok = bench(os.path.join(td, "m_ok.json"), 68.0)     # ratio 1.019 <= 1.05
    m_bad = bench(os.path.join(td, "m_bad.json"), 75.0)   # ratio 1.124 > 1.05
    # also: measured arm timed out at C=1 -> FAIL (cannot claim AC4)
    m_timeout = os.path.join(td, "m_to.json")
    with open(m_timeout, "w") as fh:
        json.dump({"results": [{"concurrency": 1, "error": "TimeoutError: "}]}, fh)
    # also: real 638514 baseline vs itself -> ratio 1.0 PASS
    real = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "benchmarks", "artifacts", "k3-attn-accept",
                        "benchmark_638514.json")

    def run(a, b_, lvl=1, thr=1.05):
        rc = main([a, b_, "--level", str(lvl), "--threshold", str(thr)])
        return rc

    C.selftest("ac4", run(b, m_ok) == 0, "1.019 <= 1.05 -> PASS")
    C.selftest("ac4", run(b, m_bad) == 1, "1.124 > 1.05 -> FAIL")
    C.selftest("ac4", run(b, m_timeout) == 1, "measured C=1 timed out -> FAIL")
    C.selftest("ac4", run(real, real) == 0, "638514 baseline vs itself -> PASS")
    print("issue45_bench_compare selftest: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
