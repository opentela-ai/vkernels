#!/usr/bin/env python3
"""issue45_kda_recall_probe.py -- AC2 runner for the gated long-context
associative-recall probe (KDA_RECALL_PROBE).

Issue #45 AC2: the probe must PASS with VKERNELS_KDA=1 (the delta-rule layer
served by vk_hip_kda_delta_rule_fwd) and FAIL on the K3_DISABLE_KDA=1 all-MLA
baseline (job 638514: recall_baseline_638514.json, verdict FAIL).  This
runner is self-contained (stdlib urllib): it does not need the deployment's
gen_correctness.py, but it can merge its result INTO an existing
gen_correctness report (--merge-into) using the same "kda_recall" key the
638514/639143/639740 artifacts use.

The probe embeds a secret access code at a controlled depth inside a long
filler context (default ~3000 words ~ ~4k tokens -- long enough that recall
is carried by the delta-rule state, not by local attention) and checks the
greedy continuation for the code.  The report also carries the degeneracy
verdict from issue45_common so a code match inside repetition-loop garbage is
flagged instead of silently PASSED.

Usage:
  python3 issue45_kda_recall_probe.py BASE_URL MODEL OUT_JSON \
      [--max-tokens 64] [--filler-words 3000] [--needle-pos start|middle|end]
      [--code Q9XZ-7K2P] [--merge-into report.json] [--per-req-timeout 300]
Exit 0 iff the recall probe PASSES (code found AND continuation not
degenerate).
"""
import argparse
import json
import sys
import time
import urllib.request

import issue45_common as C

FILLER_SENTENCES = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium.",
]


def build_prompt(code, filler_words, needle_pos):
    filler = []
    while sum(len(s.split()) for s in filler) < filler_words:
        filler.extend(FILLER_SENTENCES)
    filler_text = " ".join(filler)
    needle = f"The secret access code for the vault is {code}. "
    if needle_pos == "start":
        return needle + filler_text + "\n\nWhat is the secret access code for the vault? Answer with the code only:"
    if needle_pos == "end":
        return filler_text + " " + needle + "\n\nWhat is the secret access code for the vault? Answer with the code only:"
    # middle: split the filler
    half = len(filler_text) // 2
    return filler_text[:half] + " " + needle + filler_text[half:] + \
        "\n\nWhat is the secret access code for the vault? Answer with the code only:"


def post_completion(base_url, model, prompt, max_tokens, timeout):
    body = json.dumps({"model": model, "prompt": prompt,
                       "temperature": 0.0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    choices = resp.get("choices") or [{}]
    return choices[0].get("text", ""), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description="KDA_RECALL_PROBE runner (issue #45 AC2)")
    ap.add_argument("base_url", nargs="?", default="http://127.0.0.1:8080")
    ap.add_argument("model", nargs="?", default="")
    ap.add_argument("out_json", nargs="?", default="")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--filler-words", type=int, default=3000)
    ap.add_argument("--needle-pos", choices=("start", "middle", "end"), default="start")
    ap.add_argument("--code", default=C.DEFAULT_RECALL_CODE)
    ap.add_argument("--per-req-timeout", type=float, default=300)
    ap.add_argument("--merge-into", default="",
                    help="existing gen_correctness report to attach the "
                         "kda_recall result to (key 'kda_recall')")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    prompt = build_prompt(args.code, args.filler_words, args.needle_pos)
    print(f"[KDA_RECALL_PROBE] {args.base_url} model={args.model} "
          f"filler_words={args.filler_words} needle={args.needle_pos} "
          f"prompt_chars={len(prompt)}")
    entry = {"probe": "kda_recall", "base_url": args.base_url.rstrip("/"),
             "model": args.model, "code": args.code,
             "needle_pos": args.needle_pos, "filler_words": args.filler_words,
             "max_tokens": args.max_tokens, "ok": False, "verdict": "FAIL",
             "text": "", "error": "", "elapsed_s": 0.0}
    try:
        text, dt = post_completion(args.base_url, args.model, prompt,
                                   args.max_tokens, args.per_req_timeout)
        entry["elapsed_s"] = round(dt, 3)
        entry["text"] = text
        found, degen, why = C.evaluate_recall_text(args.code, text)
        entry["code_found"] = found
        entry["degenerate"] = degen
        entry["degen_reason"] = why
        entry["ok"] = found and not degen
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"{type(exc).__name__}: {exc}"
    entry["verdict"] = "PASS" if entry["ok"] else "FAIL"
    print(f"[KDA_RECALL_PROBE] verdict={entry['verdict']} "
          f"code_found={entry.get('code_found')} "
          f"degenerate={entry.get('degenerate')} "
          f"elapsed={entry['elapsed_s']}s")
    print(f"[KDA_RECALL_PROBE] text[:120]={entry['text'][:120]!r}")

    C.dump_json(entry, args.out_json)
    if args.merge_into:
        report = C.load_json(args.merge_into)
        report["kda_recall"] = entry
        C.dump_json(report, args.merge_into)
        print(f"[KDA_RECALL_PROBE] merged kda_recall -> {args.merge_into}")
    return 0 if entry["ok"] else 1


def selftest():
    """Offline checks: prompt builder, matcher, degeneracy interaction."""
    p = build_prompt(C.DEFAULT_RECALL_CODE, 3000, "start")
    C.selftest("recall", C.DEFAULT_RECALL_CODE in p, "code embedded at start")
    C.selftest("recall", p.count(C.DEFAULT_RECALL_CODE) == 1, "needle exactly once")
    p_mid = build_prompt("AB12-3CD4", 2000, "middle")
    C.selftest("recall", "AB12-3CD4" in p_mid, "needle embedded in middle")
    # matcher: real hit
    f, d, _ = C.evaluate_recall_text("Q9XZ-7K2P",
                                     "The secret access code for the vault is Q9XZ-7K2P.")
    C.selftest("recall", f and not d, "clean continuation -> found, not degenerate")
    # matcher: case/format variance
    f, _, _ = C.evaluate_recall_text("Q9XZ-7K2P", "code: q9xz 7k2p end")
    C.selftest("recall", f, "case/space variant still found")
    # matcher: miss (the 638514 baseline outcome: garbage, no code; the
    # artifact's continuation repeats "bar nodes" ~30x)
    f, d, _ = C.evaluate_recall_text("Q9XZ-7K2P", "\ufffdbar " + "nodesbar " * 29)
    C.selftest("recall", (not f) and d, "baseline garbage -> miss + degenerate")
    # matcher: code inside garbage is flagged, not passed
    f, d, _ = C.evaluate_recall_text("Q9XZ-7K2P", "Q9XZ-7K2P " * 40)
    C.selftest("recall", f and d, "code inside repetition loop -> found but degenerate")
    print("issue45_kda_recall_probe selftest: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
