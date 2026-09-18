#!/usr/bin/env python3
"""issue45_common.py -- shared helpers for the #45 K3-attn acceptance tooling.

Stdlib only (no torch / numpy / pytest): every tool must run bare-env on the
login node, inside the kimi-k3-vllm container, and under the repo CI.

Contents:
  * DEGENERACY detector -- the issue-#45 erratum lesson: a non-empty
    continuation is NOT evidence of a correct model.  `.dartampionship…` and
    `!!!!…` are both non-empty; both are garbage.  detect_degenerate() flags
    immediate repetition loops (any 1..4-gram repeated R>=DEGEN_MIN_RUN times
    in a row) and near-zero alphabetic-ratio output.
  * SMOKE classification -- gen_correctness.py's smoke matcher (_nonempty)
    passes anything non-empty.  Newer builds record "smoke": true/false in the
    report; the copies deployed on beverin (deploy_v2 / cookbook f3a12b04, and
    every artifact checked in under artifacts/k3-attn-accept/) do NOT.  For
    those, smoke mode is *inferable*: GEN_CORRECTNESS_SMOKE=1 sets
    min_pass=len(prompts)=6 and disables the crisp gate, while a non-smoke
    PASS requires ALL 3 crisp prompts -- so verdict==PASS with crisp_pass==0
    is only reachable in smoke mode.  classify_gen_report() uses the explicit
    field when present, falls back to that signature, and always runs the
    degeneracy detector so a smoke-only / degenerate PASS is self-identifying.
"""

from __future__ import annotations

import json
import re
import sys

# A 1..4-gram repeated this many times consecutively is a degenerate loop
# (" spent. spent. spent." -> 32 repeats; real prose never does this).
DEGEN_MIN_RUN = 8
# Below this fraction of alphabetic characters the text is symbol garbage
# ("!!!!…" -> 0.0).
DEGEN_MIN_ALPHA_RATIO = 0.15

# ---------------------------------------------------------------------------
# degeneracy detection
# ---------------------------------------------------------------------------

def _max_repeat_run(tokens, k):
    """Longest run of consecutive identical k-grams (in repetitions)."""
    if len(tokens) < 2 * k:
        return 0
    grams = [tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]
    best = run = 1
    for i in range(1, len(grams)):
        run = run + 1 if grams[i] == grams[i - 1] else 1
        if run > best:
            best = run
    return best


def detect_degenerate(text):
    """Return (degenerate: bool, reason: str). Empty text is NOT degenerate
    here (an empty/error entry is a transport failure, classified upstream)."""
    if not text or not text.strip():
        return False, ""
    if len(text) < 8:
        return False, ""
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / len(text) < DEGEN_MIN_ALPHA_RATIO:
        return True, f"alpha_ratio={alpha / len(text):.2f} < {DEGEN_MIN_ALPHA_RATIO}"
    tokens = text.split()
    for k in (1, 2, 3, 4):
        if len(tokens) >= (k + 1) * DEGEN_MIN_RUN:
            run = _max_repeat_run(tokens, k)
            if run >= DEGEN_MIN_RUN:
                gram = " ".join(tokens[:k])[:40]
                return True, f"{k}-gram '{gram}' repeated {run}x >= {DEGEN_MIN_RUN}"
    # character-level periodic loop, robust to tokenizer whitespace splits
    # ("bar nodesbar nodesbar nodes" -- the 638514 baseline garbage)
    compact = re.sub(r"\s+", "", text)
    m = re.search(r"(.{4,32}?)\1{5,}", compact)
    if m:
        return True, f"periodic loop {m.group(1)[:20]!r} x{1 + len(m.group(0)) // len(m.group(1))}"
    return False, ""


# ---------------------------------------------------------------------------
# gen_correctness report classification
# ---------------------------------------------------------------------------

# classification values
COHERENT_PASS = "coherent-pass"      # non-smoke PASS, all crisp, not degenerate
SMOKE_ONLY_PASS = "smoke-only-pass"  # PASS from the non-empty smoke matcher
DEGENERATE = "degenerate"            # repetition-loop / symbol-garbage output
FAIL = "fail"                        # report verdict == FAIL
PASS_NO_CRISP = "pass-no-crisp"      # PASS, non-smoke signature impossible,
                                     # crisp_pass==0 -> treat as smoke-like


def classify_gen_report(report):
    """Classify one gen_correctness report dict.

    Returns dict with:
      smoke_field   -- report's explicit "smoke" value (True/False/None=absent)
      smoke_mode    -- True if the report came from GEN_CORRECTNESS_SMOKE=1
      degenerate    -- bool, worst-case over per-prompt continuations
      degen_reason  -- why
      classification-- one of the COHERENT_PASS/... constants
    """
    verdict = str(report.get("verdict", "")).upper()
    smoke_field = report.get("smoke", None)
    pass_n = int(report.get("pass", 0))
    crisp_pass = int(report.get("crisp_pass", 0))
    min_pass = int(report.get("min_pass", 0))

    if isinstance(smoke_field, bool):
        smoke_mode = smoke_field
        smoke_src = "field"
    elif verdict == "PASS" and min_pass == 6 and crisp_pass == 0:
        # non-smoke PASS requires all 3 crisp; only smoke mode can PASS 6/6
        # with crisp_pass==0 (deployed copies do not yet write the field).
        smoke_mode = True
        smoke_src = "inferred"
    else:
        smoke_mode = False
        smoke_src = "inferred"

    degenerate = False
    degen_reason = ""
    for r in report.get("results", []):
        d, why = detect_degenerate(r.get("text", ""))
        if d:
            degenerate = True
            degen_reason = f"{r.get('label', '?')}: {why}"
            break

    if verdict == "FAIL":
        classification = FAIL
    elif degenerate:
        classification = DEGENERATE
    elif smoke_mode:
        classification = SMOKE_ONLY_PASS
    elif crisp_pass == 0:
        classification = PASS_NO_CRISP
    else:
        classification = COHERENT_PASS
    return {
        "verdict_field": verdict,
        "smoke_field": smoke_field,
        "smoke_mode": smoke_mode,
        "smoke_source": smoke_src,
        "degenerate": degenerate,
        "degen_reason": degen_reason,
        "classification": classification,
    }


def inspect_gen_file(path):
    with open(path) as fh:
        report = json.load(fh)
    out = {"file": path}
    out.update(classify_gen_report(report))
    out["pass"] = report.get("pass")
    out["crisp_pass"] = report.get("crisp_pass")
    out["min_pass"] = report.get("min_pass")
    out["kda_recall_pass"] = None
    kr = report.get("kda_recall")
    if isinstance(kr, dict):
        out["kda_recall_pass"] = bool(kr.get("ok", False))
    return out


# ---------------------------------------------------------------------------
# rocprof kernel-name analysis (AC3)
# ---------------------------------------------------------------------------

# AC3 forbids AITER/Triton *attention* kernels specifically: AITER MoE GEMMs
# (fused_moe / gemm / mx*) are the known-good serving path and are allowed.
FORBIDDEN_TRACERS = ("aiter", "triton")
ATTENTION_KEYWORDS = (
    "attn", "attention", "mla", "fmha", "mha", "flash",
    "kda", "delta_rule", "deltarule", "gdn", "mla_gluon", "paged",
)
# The only attention kernels AC3 allows: the vkernels C ABI.
EXPECTED_VK_PATTERNS = ("vk_hip_mla", "vk_hip_kda", "vk_mla_fwd", "vk_kda")

_KNAME_HEADERS = ("kernel name", "kernel_name", "kernelname", "name", "symbol")


def _kernel_names_from_csv(path):
    """Extract kernel names from a rocprof --stats CSV. Returns None if the
    file does not look like a CSV with a recognizable kernel-name column."""
    with open(path, errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not lines:
        return []
    # rocprof quotes the name column; split on ',' respecting quotes.
    def split(line):
        return re.findall(r'"([^"]*)"|([^,]+)', line)

    def cells(line):
        return [a or b for a, b in split(line)]

    header = [c.strip().strip('"').lower() for c in cells(lines[0])]
    idx = None
    for i, h in enumerate(header):
        if h in _KNAME_HEADERS:
            idx = i
            break
    if idx is None:
        return None
    names = []
    for ln in lines[1:]:
        c = cells(ln)
        if len(c) > idx:
            names.append(c[idx].strip().strip('"'))
    return names


def analyze_kernels(names):
    """Classify kernel-name list for AC3. Returns summary dict."""
    forbidden, expected, other = [], [], 0
    for n in names or []:
        low = n.lower()
        is_tracer = any(t in low for t in FORBIDDEN_TRACERS)
        is_attn = any(k in low for k in ATTENTION_KEYWORDS)
        if is_tracer and is_attn:
            forbidden.append(n)
        if any(p in low for p in EXPECTED_VK_PATTERNS):
            expected.append(n)
        else:
            other += 1
    def dedup(seq):
        return sorted(set(seq))
    return {
        "n_kernels": len(names or []),
        "forbidden_attention": dedup(forbidden),
        "vk_attention": dedup(expected),
        "verdict": "PASS" if not forbidden else "FAIL",
        "reason": ("zero AITER/Triton attention kernels"
                   if not forbidden else
                   f"{len(forbidden)} AITER/Triton attention kernel launches"),
    }


def analyze_rocprof_file(path):
    names = _kernel_names_from_csv(path)
    if names is None:  # not a kernel-name CSV: scan as raw log text
        text = open(path, errors="replace").read()
        return analyze_log_text(text)
    return analyze_kernels(names)


def analyze_log_text(text):
    """Fallback: pattern-scan a raw log / non-CSV rocprof output."""
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_:<>]*", text))
    hits = [n for n in names
            if any(t in n.lower() for t in FORBIDDEN_TRACERS + EXPECTED_VK_PATTERNS
                   + ATTENTION_KEYWORDS)]
    return analyze_kernels(sorted(hits))


# ---------------------------------------------------------------------------
# KDA_RECALL_PROBE constants (AC2)
# ---------------------------------------------------------------------------

DEFAULT_RECALL_CODE = "Q9XZ-7K2P"

def evaluate_recall_text(code, text):
    """AC2 recall matcher: the secret code must appear in the continuation
    (case-insensitive; hyphen/space separated digit groups both accepted).
    Returns (found: bool, degenerate: bool, reason: str)."""
    d, why = detect_degenerate(text)
    # normalized substring: ignore case and drop hyphens/spaces so both
    # "Q9XZ-7K2P" and "q9xz 7k2p" match, without a regex tuned to one code.
    norm = re.sub(r"[-\s]", "", (text or "").lower())
    found = code.lower().replace("-", "") in norm
    return found, d, why


# ---------------------------------------------------------------------------
# tiny selftest plumbing
# ---------------------------------------------------------------------------

def selftest(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{name}] {status} {detail}")
    if not cond:
        sys.exit(1)
    return 0


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def dump_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
