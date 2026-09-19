#!/usr/bin/env python3
"""issue45_gen_inspect.py -- gen_correctness report inspector: distinguish a
COHERENT pass from a SMOKE-ONLY pass (and from degenerate garbage).

Issue #45 erratum: jobs 597880/603711 (and 639143/639260) "passed" the
6-probe gate with real weights while emitting `!!!!…` / `.dartampionship…`
-- the smoke matcher is _nonempty, and the deployed gen_correctness.py
(deploy_v2 / cookbook f3a12b04) does not yet write the "smoke": true/false
field into its report.  This inspector therefore:

  1. uses the report's explicit "smoke" field when present (newer builds);
  2. otherwise infers smoke mode from the only reachable signature
     (smoke mode sets min_pass=6 and skips the crisp gate, while a non-smoke
     PASS requires all 3 crisp prompts => PASS with crisp_pass==0 is
     smoke-mode) ;
  3. ALWAYS runs the degeneracy detector (repetition loops / symbol garbage)
     over the raw continuations, independent of the verdict field;
  4. inspects the merged "kda_recall" entry when present (AC2 contrast).

Usage:
  python3 issue45_gen_inspect.py REPORT.json [more.json ...] [--json out.json]
                                     [--fail-on smoke-only-pass,degenerate]
Exit 0 iff every report's classification is outside --fail-on
(default: everything except coherent-pass fails, i.e. the gate only accepts
a non-smoke, non-degenerate PASS).
"""
import argparse
import sys

import issue45_common as C

DEFAULT_FAIL_ON = "smoke-only-pass,degenerate,pass-no-crisp,fail"


def main(argv=None):
    ap = argparse.ArgumentParser(description="gen_correctness smoke-vs-coherent inspector")
    ap.add_argument("reports", nargs="*")
    ap.add_argument("--json", dest="out_json", default="")
    ap.add_argument("--fail-on", default=DEFAULT_FAIL_ON,
                    help="comma-separated classifications that fail the gate "
                         f"(default {DEFAULT_FAIL_ON})")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    fail_on = {s.strip() for s in args.fail_on.split(",") if s.strip()}
    out = {"reports": [], "verdict": "PASS", "fail_on": sorted(fail_on)}
    for path in args.reports:
        r = C.inspect_gen_file(path)
        out["reports"].append(r)
        if r["classification"] in fail_on:
            out["verdict"] = "FAIL"
        print(f"[GEN] {path}: verdict={r['verdict_field']} "
              f"pass={r['pass']}/{6} crisp={r['crisp_pass']} "
              f"min_pass={r['min_pass']} smoke={r['smoke_field']}"
              f"{'(inferred)' if r['smoke_field'] is None and r['smoke_mode'] else ''} "
              f"-> {r['classification']}"
              + (f" DEGEN: {r['degen_reason']}" if r["degenerate"] else ""))
    print(f"[GEN] verdict={out['verdict']} (fail-on={sorted(fail_on)})")
    if args.out_json:
        C.dump_json(out, args.out_json)
    return 0 if out["verdict"] == "PASS" else 1


def selftest():
    import json
    import os
    import tempfile
    td = tempfile.mkdtemp(prefix="i45gen-")

    def report(path, **kw):
        base = {"base_url": "u", "model": "m", "max_tokens": 64,
                "min_pass": 5, "crisp_total": 3, "pass": 6, "crisp_pass": 3,
                "verdict": "PASS", "elapsed_s": 1.0, "results": [
                    {"prompt": "p", "label": "capital -> Paris", "crisp": True,
                     "text": "Paris, the City of Light.", "ok": True,
                     "error": "", "elapsed_s": 1.0}]}
        base.update(kw)
        p = os.path.join(td, path)
        with open(p, "w") as fh:
            json.dump(base, fh)
        return p

    # 1. coherent pass
    good = report("good.json")
    r = C.inspect_gen_file(good)
    C.selftest("gen", r["classification"] == C.COHERENT_PASS, good)
    # 2. explicit smoke field -> smoke-only-pass even though verdict PASS
    sm = report("smoke.json", smoke=True, min_pass=6, crisp_pass=0)
    r = C.inspect_gen_file(sm)
    C.selftest("gen", r["classification"] == C.SMOKE_ONLY_PASS and r["smoke_mode"], sm)
    # 3. inferred smoke: PASS + min_pass==6 + crisp_pass==0 (deployed builds
    #    do not write the field; this is the 638514 artifact signature)
    inf = report("inf.json", min_pass=6, crisp_pass=0)
    r = C.inspect_gen_file(inf)
    C.selftest("gen", r["classification"] == C.SMOKE_ONLY_PASS
               and r["smoke_source"] == "inferred", inf)
    # 4. degenerate real-weight run (the 638514 artifact shape: non-smoke
    #    fields but garbage text)
    dg = report("deg.json", results=[
        {"prompt": "p", "label": "capital -> Paris", "crisp": True,
         "text": ".dartampionship.dartampionship.dartampionship."
                 "dartampionship.dartampionship.dartampionship."
                 "dartampionship.dartampionship.dartampionship.",
         "ok": True, "error": "", "elapsed_s": 1.0}])
    r = C.inspect_gen_file(dg)
    C.selftest("gen", r["classification"] == C.DEGENERATE and r["degenerate"], dg)
    # 5. fail verdict
    fl = report("fail.json", verdict="FAIL", **{"pass": 0}, crisp_pass=0)
    r = C.inspect_gen_file(fl)
    C.selftest("gen", r["classification"] == C.FAIL, fl)
    # 6. real checked-in artifacts (truth anchors from the erratum)
    art = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "benchmarks", "artifacts", "k3-attn-accept")
    real = os.path.join(art, "gen_correctness_638514.json")
    r = C.inspect_gen_file(real)
    C.selftest("gen", r["classification"] in (C.DEGENERATE, C.SMOKE_ONLY_PASS),
               f"638514 artifact -> {r['classification']} (must NOT be coherent-pass)")
    C.selftest("gen", r["smoke_mode"] is True,
               "638514 was GEN_CORRECTNESS_SMOKE=1 -> smoke inferred")
    real2 = os.path.join(art, "recall_baseline_638514.json")
    r = C.inspect_gen_file(real2)
    C.selftest("gen", r["kda_recall_pass"] is False,
               "638514 recall baseline -> kda_recall FAIL (AC2 contrast anchor)")
    # 7. gate exit code: artifacts must not pass the coherent gate
    C.selftest("gen", main([real]) == 1, "638514 artifact fails the coherent gate")
    print("issue45_gen_inspect selftest: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
