"""Tabular and JSON reporting for benchmark results.

JSON is the interchange for the CI comparison flow: run once on the merge
base, run on the candidate, then ``vkernels.benchmark.ci.compare`` decides
regressions. The schema is versioned so old baselines fail loudly instead
of matching keys by accident.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from vkernels.benchmark.runner import BenchmarkResult

__all__ = ["format_table", "to_json", "from_json", "write_json", "read_json"]

SCHEMA_VERSION = 1


def format_table(results: Iterable["BenchmarkResult"]) -> str:
    """Fixed-width table: one row per result, throughput against the roof."""
    rows = list(results)
    if not rows:
        return "(no results)"
    header = (
        f"{'op':<10} {'params':<18} {'backend':<10} {'method':<12} "
        f"{'med ms':>9} {'spread':>7} {'GFLOP/s':>9} {'GB/s':>8} {'ok':>4}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        p = r.request.params()
        params = ",".join(f"{k}={v}" for k, v in sorted(p.items()))
        med_ms = r.seconds * 1e3 if r.seconds == r.seconds else float("nan")
        timing_spread = (max(r.samples) - min(r.samples)) / r.seconds if len(r.samples) > 1 and r.seconds else 0.0
        spread = f"{timing_spread:.0%}"
        verified = "-" if r.verified is None else ("y" if r.verified else "N")
        gflops = f"{r.gflops:.2f}" if r.gflops == r.gflops else "-"
        gbytes = f"{r.gbytes:.2f}" if r.gbytes == r.gbytes else "-"
        error = f"  !! {r.error}" if r.error else ""
        lines.append(
            f"{r.request.op:<10} {params:<18} {r.backend:<10} {r.method:<12} "
            f"{med_ms:>9.3f} {spread:>7} {gflops:>9} {gbytes:>8} {verified:>4}{error}"
        )
    return "\n".join(lines)


def to_json(results: Iterable["BenchmarkResult"]) -> dict:
    rows = []
    for r in results:
        row = asdict(r)
        row["key"] = r.request.key
        row["samples"] = list(r.samples)
        rows.append(row)
    return {"schema": SCHEMA_VERSION, "results": rows}


def from_json(data: dict) -> list["BenchmarkResult"]:
    from vkernels.benchmark.runner import BenchmarkRequest, BenchmarkResult

    if data.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported benchmark JSON schema {data.get('schema')!r}; "
            f"this harness writes {SCHEMA_VERSION}"
        )
    results = []
    for row in data["results"]:
        row = dict(row)
        row["request"] = BenchmarkRequest(**row["request"])
        row["samples"] = tuple(row["samples"])
        row.pop("key", None)
        results.append(BenchmarkResult(**row))
    return results


def write_json(path: str | Path, results: Iterable["BenchmarkResult"]) -> None:
    Path(path).write_text(json.dumps(to_json(results), indent=1) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> list["BenchmarkResult"]:
    return from_json(json.loads(Path(path).read_text(encoding="utf-8")))
