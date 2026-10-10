"""Compare two benchmark JSON reports (the PR CI flow).

The flow (borrowed from tokenspeed-kernel's PR benchmark comparison):
run the suite on the merge base, run it on the candidate, compare — a case
whose median runtime regressed beyond the threshold fails the check; new
and removed cases are reported but do not fail it (a new case has no
baseline; a removed one cannot regress).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["ComparisonRow", "ComparisonReport", "compare"]

Verdict = Literal["ok", "regression", "improved", "new", "missing", "error"]


@dataclass(frozen=True)
class ComparisonRow:
    key: str
    verdict: Verdict
    baseline_ms: float | None = None
    current_ms: float | None = None
    ratio: float | None = None  # current / baseline
    detail: str = ""


@dataclass(frozen=True)
class ComparisonReport:
    threshold: float
    rows: tuple[ComparisonRow, ...]

    @property
    def regressions(self) -> tuple[ComparisonRow, ...]:
        return tuple(r for r in self.rows if r.verdict == "regression")

    @property
    def ok(self) -> bool:
        return not self.regressions

    def __str__(self) -> str:
        lines = [f"benchmark comparison (threshold {self.threshold:.0%})"]
        for row in self.rows:
            base = f"{row.baseline_ms:.3f}" if row.baseline_ms else "-"
            cur = f"{row.current_ms:.3f}" if row.current_ms else "-"
            ratio = f"{row.ratio:.2f}x" if row.ratio else ""
            lines.append(f"  {row.verdict:<10} {row.key}: {base} -> {cur} ms {ratio}")
        summary = (
            f"{len(self.rows)} cases, {len(self.regressions)} regression(s)"
            if self.rows
            else "no comparable cases"
        )
        lines.append(summary)
        return "\n".join(lines)


def compare(baseline: dict, current: dict, *, threshold: float = 0.10) -> ComparisonReport:
    """Compare two :func:`~vkernels.benchmark.report.to_json` documents."""
    if baseline.get("schema") != current.get("schema"):
        raise ValueError("schema mismatch between baseline and current")
    base_rows = {row["key"]: row for row in baseline.get("results", [])}
    cur_rows = {row["key"]: row for row in current.get("results", [])}

    rows: list[ComparisonRow] = []
    for key in sorted(base_rows.keys() | cur_rows.keys()):
        base, cur = base_rows.get(key), cur_rows.get(key)
        if base is None:
            rows.append(ComparisonRow(key, "new", current_ms=_ms(cur)))
            continue
        if cur is None:
            rows.append(ComparisonRow(key, "missing", baseline_ms=_ms(base)))
            continue
        if cur.get("error"):
            rows.append(
                ComparisonRow(key, "error", _ms(base), _ms(cur), detail=cur["error"])
            )
            continue
        base_ms, cur_ms = _ms(base), _ms(cur)
        if base_ms is None or cur_ms is None or base_ms <= 0 or cur_ms <= 0:
            rows.append(ComparisonRow(key, "error", base_ms, cur_ms, detail="invalid timing"))
            continue
        ratio = cur_ms / base_ms
        if ratio > 1 + threshold:
            verdict: Verdict = "regression"
        elif ratio < 1 - threshold:
            verdict = "improved"
        else:
            verdict = "ok"
        rows.append(ComparisonRow(key, verdict, base_ms, cur_ms, ratio))
    return ComparisonReport(threshold, tuple(rows))


def _ms(row: dict | None) -> float | None:
    if row is None:
        return None
    seconds = row.get("seconds")
    return seconds * 1e3 if isinstance(seconds, (int, float)) else None
