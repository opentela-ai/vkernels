#!/usr/bin/env python3
"""Line-coverage gate for the host (CPU reference) build.

Runs ``gcov`` over every ``.gcno`` produced by a coverage build, merges the
results per source file (so headers included from several translation units
are counted once), and reports per-file and total line coverage. Exits
non-zero when coverage falls below ``--min`` (default 100) for any source
file under ``--source-dir`` (default src/c/).

Branch coverage is computed and reported for information but, unless
``--require-branches`` is set, it does not fail the gate.

Lines whose source carries a ``LCOV_EXCL_LINE`` comment are intentionally
excluded from the gate (used for defensively-dead branches such as malloc
failure handling that a test cannot deterministically trigger). gcov's ``-b``
mode appends ``*`` to the count on lines with an unexecuted branch; that
suffix is ignored so such lines still count as covered.

Example:
    python3 scripts/coverage.py --build-dir build/coverage --source-dir src/c --min 100
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

_SOURCE_LINE = re.compile(r"^\s*([^:]+):\s*(\d+):(.*)$")
_SOURCE_HDR = re.compile(r"^\s*-:\s*0:Source:(.+)$")
_BRANCH_HDR = re.compile(r"^\s*branch\s+\d+\s+(taken|never executed)")
_NUMERIC = re.compile(r"^\d+$")


@dataclass
class FileStats:
    executable: set[int] = field(default_factory=set)
    covered: set[int] = field(default_factory=set)
    branches_total = 0
    branches_taken = 0

    def add_exec(self, ln: int) -> None:
        self.executable.add(ln)

    def add_covered(self, ln: int) -> None:
        self.covered.add(ln)
        self.executable.add(ln)

    @property
    def total(self) -> int:
        return len(self.executable)

    @property
    def hit(self) -> int:
        return len(self.covered & self.executable)

    def pct(self) -> float:
        return 100.0 * self.hit / self.total if self.total else 100.0


def find_gcno(build_dir: str) -> list[str]:
    out: list[str] = []
    for root, _dirs, files in os.walk(build_dir):
        for f in files:
            if f.endswith(".gcno"):
                out.append(os.path.join(root, f))
    return out


def parse_gcov(gcov_path: str) -> tuple[str | None, FileStats]:
    """Return (source_path, stats) for one .gcov file."""
    source: str | None = None
    stats = FileStats()
    with open(gcov_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            mhdr = _SOURCE_HDR.match(line)
            if mhdr and source is None:
                source = mhdr.group(1).strip()
                continue
            mb = _BRANCH_HDR.match(line)
            if mb:
                stats.branches_total += 1
                if mb.group(1) == "taken":
                    stats.branches_taken += 1
                continue
            m = _SOURCE_LINE.match(line)
            if not m:
                continue
            tok, ln, src = m.group(1).strip(), int(m.group(2)), m.group(3)
            if ln == 0 or tok == "-":
                continue
            if "LCOV_EXCL_LINE" in src:
                continue  # explicitly excluded from coverage
            tok = tok.rstrip("*")  # gcov -b: "N*" = executed with a branch not taken
            if tok == "#####":
                stats.add_exec(ln)
            elif _NUMERIC.match(tok):
                stats.add_covered(ln)
    return source, stats


def under_any(path: str, roots: list[str]) -> bool:
    ap = os.path.abspath(path)
    return any(ap.startswith(os.path.abspath(r) + os.sep) or
               ap == os.path.abspath(r) for r in roots)


def merge_stats(build_dir: str, source_roots: list[str]) -> dict[str, FileStats]:
    merged: dict[str, FileStats] = {}
    gcno_files = find_gcno(build_dir)
    if not gcno_files:
        return merged
    with tempfile.TemporaryDirectory(prefix="vkernels-gcov-") as tmp:
        for idx, gcno in enumerate(gcno_files):
            # Run each gcov in its own subdir so same-named headers included
            # from multiple translation units don't clobber each other.
            run_dir = os.path.join(tmp, str(idx))
            os.makedirs(run_dir, exist_ok=True)
            try:
                subprocess.run(
                    ["gcov", "-b", "-c", os.path.abspath(gcno)],
                    cwd=run_dir, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
            for entry in os.listdir(run_dir):
                if not entry.endswith(".gcov"):
                    continue
                src, stats = parse_gcov(os.path.join(run_dir, entry))
                if src is None or not os.path.exists(src):
                    continue
                if not under_any(src, source_roots):
                    continue
                cur = merged.setdefault(src, FileStats())
                cur.executable |= stats.executable
                cur.covered |= stats.covered
                cur.branches_total += stats.branches_total
                cur.branches_taken += stats.branches_taken
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", default="build/coverage", help="coverage build output")
    ap.add_argument("--source-dir", action="append", default=None,
                    help="source root(s) to gate (default src/c); repeatable")
    ap.add_argument("--min", type=float, default=100.0, help="minimum coverage percent")
    ap.add_argument("--require-branches", action="store_true",
                    help="also fail when any branch is never executed")
    args = ap.parse_args()

    roots = args.source_dir if args.source_dir else ["src/c"]
    if not os.path.isdir(args.build_dir):
        print(f"error: build dir not found: {args.build_dir}", file=sys.stderr)
        return 2

    merged = merge_stats(args.build_dir, roots)
    if not merged:
        print("warning: no coverage data found under source dir(s); "
              "did you build with VKERNELS_ENABLE_COVERAGE=ON and run tests?",
              file=sys.stderr)
        return 2

    rows = []
    total_exec = total_hit = 0
    for src in sorted(merged):
        s = merged[src]
        total_exec += s.total
        total_hit += s.hit
        rows.append((os.path.relpath(src), s.total, s.hit, s.pct(),
                     s.branches_total, s.branches_taken))

    width = max((len(r[0]) for r in rows), default=0)
    print(f"{'file':<{width}}  lines(hit/total)   %   branches(taken/total)")
    print("-" * (width + 48))
    for name, tot, hit, pct, bt, bh in rows:
        print(f"{name:<{width}}  {hit:4d}/{tot:<4d}  {pct:6.2f}  {bh}/{bt}")
    print("-" * (width + 48))
    overall = 100.0 * total_hit / total_exec if total_exec else 100.0
    print(f"{'TOTAL':<{width}}  {total_hit:4d}/{total_exec:<4d}  {overall:6.2f}")

    failed = [r for r in rows if r[3] + 1e-9 < args.min]
    if failed:
        by_name = {os.path.relpath(src): s for src, s in merged.items()}
        print(f"\nFAIL: coverage below {args.min:.2f}% in {len(failed)} file(s):",
              file=sys.stderr)
        for name, tot, hit, pct, _, _ in failed:
            miss = tot - hit
            s = by_name.get(name)
            # list the uncovered line numbers for fast fixing
            miss_lines = sorted((s.executable - s.covered) if s else set())[:20]
            print(f"  {name}: {hit}/{tot} ({pct:.2f}%), {miss} uncovered line(s) "
                  f"e.g. {miss_lines}", file=sys.stderr)
        return 1

    if args.require_branches:
        bf = [r for r in rows if r[5] < r[4]]
        if bf:
            print(f"\nFAIL: unexecuted branches in {len(bf)} file(s)", file=sys.stderr)
            for name, _, _, _, bt, bh in bf:
                print(f"  {name}: {bh}/{bt} branches", file=sys.stderr)
            return 1

    print("\nOK: line coverage >= {:.2f}% across {} file(s).".format(args.min, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
