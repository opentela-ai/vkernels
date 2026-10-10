"""Command line for the numerics layer: ``python -m vkernels.numerics``.

Subcommands:

* ``ops`` — list the ops with standard generators.
* ``verify <op> [--size N] [--seed S] [--exact] [--n N] [--k K]`` — run the
  active backend against the fallback oracle once and print the verdict.
* ``bisect <op> --lo N --hi N [...]`` — find the smallest failing size.
"""

from __future__ import annotations

import argparse
import sys

from vkernels.numerics.bisect import bisect_size
from vkernels.numerics.generators import available_ops
from vkernels.numerics.verify import verify

__all__ = ["main"]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0, help="input generator seed")
    parser.add_argument(
        "--loose", action="store_true",
        help="use the scaled dtype tolerance instead of exact for float32",
    )
    parser.add_argument(
        "--n", type=int, default=None, help="gemm N (generator extra)"
    )
    parser.add_argument(
        "--k", type=int, default=None, help="gemm K (generator extra)"
    )


def _extra(args: argparse.Namespace) -> dict:
    extra = {}
    if args.n is not None:
        extra["n"] = args.n
    if args.k is not None:
        extra["k"] = args.k
    return extra


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vkernels.numerics",
        description="Verify kernels against their pure-Python oracles.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ops", help="list ops with standard input generators")

    p_verify = sub.add_parser("verify", help="run one kernel-vs-oracle comparison")
    p_verify.add_argument("op", choices=available_ops())
    p_verify.add_argument("--size", type=int, default=1024)
    _add_common(p_verify)

    p_bisect = sub.add_parser(
        "bisect", help="find the smallest size where verification fails"
    )
    p_bisect.add_argument("op", choices=available_ops())
    p_bisect.add_argument("--lo", type=int, default=1)
    p_bisect.add_argument("--hi", type=int, default=4096)
    _add_common(p_bisect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ops":
        print(" ".join(available_ops()))
        return 0

    exact = not args.loose

    if args.command == "verify":
        result = verify(args.op, args.size, seed=args.seed, exact=exact, **_extra(args))
        print(result)
        if not result.ok:
            stats = result.stats
            print(
                f"  max_abs={stats.get('max_abs')}, max_rel={stats.get('max_rel')}, "
                f"calc_diff={stats.get('calc_diff')}, "
                f"first_mismatch={stats.get('first_mismatch')}"
            )
        return 0 if result.ok else 1

    if args.command == "bisect":
        report = bisect_size(
            args.op, args.lo, args.hi, seed=args.seed, exact=exact, **_extra(args)
        )
        print(report)
        for trial in report.trials:
            print(f"  size={trial.size}: {'ok' if trial.ok else 'FAILED'}")
        return 0 if report.smallest_failing is None else 1

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
