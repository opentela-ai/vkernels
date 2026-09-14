"""Three-mode qualification for the frozen TunableOp QKV artifact (#67).

``--mode repro``     Cold/warm artifact load (CSV + manifest), eager/captured
                     forward parity, changed-input replay, exact recorded
                     winners. ``Default`` winners are legitimate here: they
                     are recorded and reported, never silently rewritten.
``--mode quality``   Pre-declared gates from the artifact manifest: FP64-direct
                     BF16 oracle error, argmax-ID agreement (actual IDs with
                     margins and tie flags), baseline determinism, and the
                     model-level NLL-regression gate (fails closed until a
                     floe-side measurement is supplied via --nll-report).
``--mode perf``      Graph timing only. The report is marked performance-only
                     and is explicitly not promotion approval.

Thresholds live in the artifact manifest and were declared before evaluation.
Any failed gate exits nonzero and names the gate. There is no waiver flag.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from vkernels.torch_ops.bf16_oracle import argmax_evidence, qkv_oracle_fp64
from vkernels.torch_ops.qkv_tuned_blas import (configure_qkv_tuned_blas,
                                               qkv_tuned_blas,
                                               qkv_tuned_blas_state)


def _gate(report, name, passed, detail):
    report["gates"].append({"gate": name, "passed": bool(passed), **detail})
    if not passed:
        report["passed"] = False
    return passed


def _blas(x, weights):
    return torch.cat([F.linear(x, weight) for weight in weights], dim=-1)


def _make_inputs(device, tokens, count, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = [torch.randn(8192, 4096, generator=generator).to(device=device, dtype=torch.bfloat16) * 0.01
               for _ in range(3)]
    inputs = [torch.randn(count, tokens, 4096, generator=generator).to(device=device, dtype=torch.bfloat16)
              for _ in range(8)]
    return inputs, weights


def _oracle_error(candidate, oracle):
    difference = (candidate.float() - oracle.float()).abs()
    reference_abs = oracle.float().abs().clamp_min(0.5)  # rel floor at bf16 half-scale
    return {"max_abs": difference.max().item(),
            "max_rel": (difference / reference_abs).max().item(),
            "within_atol_rtol": bool(torch.all(
                difference <= 0.008 + 0.008 * oracle.float().abs()).item())}


def mode_repro(args, report):
    tokens_configured = configure_qkv_tuned_blas(args.csv, manifest=args.manifest)
    state = qkv_tuned_blas_state()
    report["configured"] = state["configured"]
    report["manifest"] = state["manifest"]
    defaults = sorted(t for t, winner in state["configured"].items() if winner == "Default")
    report["default_winners_recorded"] = defaults
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            target = torch.device("cuda", device)
            for tokens in tokens_configured:
                inputs, weights = _make_inputs(target, tokens, 8, seed=42)
                xs = inputs
                references = [_blas(x, weights) for x in xs]
                expected = [qkv_tuned_blas(x, *weights) for x in xs]
                for x, reference, candidate in zip(xs, references, expected):
                    torch.testing.assert_close(candidate, reference, rtol=0.008, atol=0.008)
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream(device=target)
                torch.cuda.synchronize(target)
                with torch.cuda.graph(graph, stream=stream):
                    captured = [qkv_tuned_blas(x, *weights) for x in xs]
                stream.synchronize()
                graph.replay()
                for reference, eager, actual in zip(references, expected, captured):
                    torch.testing.assert_close(actual, eager, rtol=0, atol=0)   # cold capture
                    torch.testing.assert_close(actual, reference, rtol=0.008, atol=0.008)
                for x in xs:
                    x.mul_(0.5)
                graph.replay()
                for x, actual in zip(xs, captured):
                    torch.testing.assert_close(actual, qkv_tuned_blas(x, *weights), rtol=0, atol=0)
                del graph, captured, expected, references
        print(f"device={device}: eager/capture/replay parity PASS (tokens={tokens_configured})", flush=True)
    assert configure_qkv_tuned_blas(args.csv, manifest=args.manifest) == tokens_configured  # warm reload
    _gate(report, "cold_warm_capture_parity", True, {"devices": torch.cuda.device_count()})
    print("repro mode PASS", flush=True)


def mode_quality(args, report):
    configure_qkv_tuned_blas(args.csv, manifest=args.manifest)
    state = qkv_tuned_blas_state()
    gates = (state["manifest"] or {}).get("quality_gates")
    if not gates:
        _gate(report, "manifest_quality_gates", False,
              {"error": "artifact manifest carries no quality_gates block"})
        return
    report["declared_gates"] = gates
    report["numerics"] = {}
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            target = torch.device("cuda", device)
            inputs, weights = _make_inputs(target, 2, args.inputs, seed=1234)
            for tokens in sorted(state["configured"]):
                xs = [x[:, :tokens, :] for x in inputs]
                # baseline determinism floor: the reference must repeat exactly
                repeat_identical = all(torch.equal(_blas(x, weights), _blas(x, weights))
                                       for x in xs)
                candidates = [qkv_tuned_blas(x, *weights) for x in xs]
                oracles = [qkv_oracle_fp64(x, weights) for x in xs]
                errors = [_oracle_error(c, o) for c, o in zip(candidates, oracles)]
                evidence = [argmax_evidence(o, c) for c, o in zip(candidates, oracles)]
                agreement = min(e["agreement"] for e in evidence)
                worst_abs = max(e["max_abs"] for e in errors)
                worst_rel = max(e["max_rel"] for e in errors)
                report["numerics"][f"device{device}_tokens{tokens}"] = {
                    "max_abs_vs_fp64_oracle": worst_abs,
                    "max_rel_vs_fp64_oracle": worst_rel,
                    "min_argmax_agreement": agreement,
                    "changed_argmax": [e["changed"] for e in evidence][:1],
                    "reference_repeat_identical": repeat_identical,
                }
                _gate(report, f"reference_repeat_identical[d{device} t{tokens}]",
                      repeat_identical if gates.get("require_reference_repeat_identical", True) else True,
                      {"observed": repeat_identical})
                _gate(report, f"max_abs_vs_fp64_oracle[d{device} t{tokens}]",
                      worst_abs <= gates["max_abs_vs_fp64_oracle"],
                      {"threshold": gates["max_abs_vs_fp64_oracle"], "observed": worst_abs})
                _gate(report, f"max_rel_vs_fp64_oracle[d{device} t{tokens}]",
                      worst_rel <= gates["max_rel_vs_fp64_oracle"],
                      {"threshold": gates["max_rel_vs_fp64_oracle"], "observed": worst_rel})
                _gate(report, f"argmax_agreement[d{device} t{tokens}]",
                      agreement >= gates["min_argmax_agreement"],
                      {"threshold": gates["min_argmax_agreement"], "observed": agreement})
    # model-level gate: fails closed until a floe-side measurement exists
    ceiling = gates.get("model_nll_regression_max")
    if ceiling is None:
        _gate(report, "model_nll_regression", True, {"observed": "no NLL gate declared"})
    elif args.nll_report is None:
        _gate(report, "model_nll_regression", False,
              {"threshold": ceiling,
               "error": "gate declared but no --nll-report measurement supplied; "
                        "quality mode never silently waives a declared gate"})
    else:
        nll = json.loads(Path(args.nll_report).read_text())
        regression = nll["candidate"] - nll["baseline"]
        _gate(report, "model_nll_regression", regression <= ceiling,
              {"threshold": ceiling, "observed": regression,
               "source": str(args.nll_report)})
    if report["passed"]:
        print("quality mode PASS", flush=True)


def mode_perf(args, report):
    configure_qkv_tuned_blas(args.csv, manifest=args.manifest)
    from bench_mhc_projection import graph_time
    report["promotion"] = "not claimed - performance-only measurement (issue #67)"
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            target = torch.device("cuda", device)
            for tokens in sorted(qkv_tuned_blas_state()["configured"]):
                inputs, weights = _make_inputs(target, tokens, 8, seed=42)
                pairs = [(x, weights) for x in inputs]
                references = [_blas(x, weights) for x, _ in pairs]
                tuned = graph_time(qkv_tuned_blas, pairs, references, repeats=args.samples)
                base = graph_time(_blas, pairs, references, repeats=args.samples)
                report.setdefault("timings", {})[f"device{device}_tokens{tokens}"] = {
                    "default_blas_median_us": base["median_us"],
                    "tuned_blas_median_us": tuned["median_us"],
                    "speedup": base["median_us"] / tuned["median_us"],
                }
                print(f"device={device} tokens={tokens}: "
                      f"blas={base['median_us']:.1f}us tuned={tuned['median_us']:.1f}us "
                      f"({base['median_us'] / tuned['median_us']:.2f}x)", flush=True)
    report["passed"] = True  # perf mode has no quality gates to fail


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="TunableOp CSV (sibling .manifest.json required)")
    parser.add_argument("--manifest", help="explicit manifest path (default: sibling)")
    parser.add_argument("--mode", required=True, choices=("repro", "quality", "perf"))
    parser.add_argument("--report", type=Path, help="report JSON path")
    parser.add_argument("--inputs", type=int, default=64, help="quality-mode activation rows")
    parser.add_argument("--samples", type=int, default=15, help="perf-mode graph timings")
    parser.add_argument("--nll-report", help="JSON {baseline, candidate} mean NLL from the "
                                             "floe-side held-out run (quality mode)")
    args = parser.parse_args()
    args.csv = str(Path(args.csv).resolve())
    if args.report is None:
        args.report = Path(args.csv).with_suffix(f".{args.mode}.report.json")

    report = {"mode": args.mode, "csv": args.csv, "gates": [], "passed": True}
    torch.manual_seed(42)
    {"repro": mode_repro, "quality": mode_quality, "perf": mode_perf}[args.mode](args, report)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"report: {args.report}", flush=True)
    if not report["passed"]:
        failed = [g["gate"] for g in report["gates"] if not g["passed"]]
        print(f"FAILED gates: {failed}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
