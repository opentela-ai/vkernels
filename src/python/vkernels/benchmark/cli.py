"""Command line for the benchmark harness: ``python -m vkernels.benchmark``.

Subcommands:

* ``ops`` — list ops the harness can benchmark.
* ``run [--ops ...] [--sizes ...] [--suite FILE] [--json OUT]`` — run a
  suite and print the table (and optionally write the JSON report).
* ``compare BASELINE CURRENT [--threshold 0.1]`` — the PR CI check: exit 1
  when any case regressed beyond the threshold.

Suite files are JSON: ``{"requests": [{"op": "gemm", "size": 128,
"n": 16, "k": 16}, ...]}`` — one per target platform, kept beside the code
like tokenspeed-kernel's ``benchmarks/<vendor>/<arch>.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vkernels.benchmark.report import format_table, write_json
from vkernels.benchmark.runner import BenchmarkRequest, run_benchmark
from vkernels.numerics.generators import available_ops

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vkernels.benchmark",
        description="Benchmark kernels against their throughput roof.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ops", help="list benchmarkable ops")

    p_run = sub.add_parser("run", help="run a suite and report")
    p_run.add_argument("--ops", nargs="*", default=None, help="ops to run (default: all)")
    p_run.add_argument(
        "--sizes", nargs="*", type=int, default=[1024, 4096], help="sizes to run"
    )
    p_run.add_argument("--n", type=int, default=None, help="gemm N override")
    p_run.add_argument("--k", type=int, default=None, help="gemm K override")
    p_run.add_argument("--iters", type=int, default=20)
    p_run.add_argument("--warmup", type=int, default=5)
    p_run.add_argument("--verify", action="store_true", help="oracle-check before timing")
    p_run.add_argument("--suite", type=Path, default=None, help="JSON suite file")
    p_run.add_argument("--json", type=Path, default=None, help="write JSON report here")

    p_cmp = sub.add_parser("compare", help="PR CI comparison of two JSON reports")
    p_cmp.add_argument("baseline", type=Path)
    p_cmp.add_argument("current", type=Path)
    p_cmp.add_argument("--threshold", type=float, default=0.10)

    return parser


def _requests_from_args(args: argparse.Namespace) -> list[BenchmarkRequest]:
    if args.suite is not None:
        import json

        data = json.loads(args.suite.read_text(encoding="utf-8"))
        return [BenchmarkRequest(**row) for row in data["requests"]]

    ops = args.ops if args.ops else list(available_ops())
    unknown = [op for op in ops if op not in available_ops()]
    if unknown:
        raise SystemExit(
            f"unknown ops {unknown}; available: {', '.join(available_ops())}"
        )
    requests = []
    for op in ops:
        for size in args.sizes:
            requests.append(
                BenchmarkRequest(
                    op=op,
                    size=size,
                    n=args.n if op == "gemm" else None,
                    k=args.k if op == "gemm" else None,
                    warmup=args.warmup,
                    iters=args.iters,
                    verify=args.verify,
                )
            )
    return requests


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ops":
        print(" ".join(available_ops()))
        return 0

    if args.command == "run":
        requests = _requests_from_args(args)
        results = [run_benchmark(request) for request in requests]
        print(format_table(results))
        if args.json is not None:
            write_json(args.json, results)
            print(f"wrote {args.json}")
        return 0 if all(r.error is None for r in results) else 1

    if args.command == "compare":
        report = compare_json_files(args.baseline, args.current, args.threshold)
        print(report)
        return 0 if report.ok else 1

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover


def compare_json_files(baseline: Path, current: Path, threshold: float):
    import json

    from vkernels.benchmark.ci import compare as compare_dicts

    base = json.loads(baseline.read_text(encoding="utf-8"))
    cur = json.loads(current.read_text(encoding="utf-8"))
    return compare_dicts(base, cur, threshold=threshold)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
