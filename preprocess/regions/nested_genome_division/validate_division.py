#!/usr/bin/env python3
"""Invariant checker and stats reporter for the TSS-TES division BED.

Every interval must satisfy, without exception:

* ``length % 128 == 0`` — the only size constraint; there is no floor and no
  ceiling
* every interval inside its chromosome's bounds, and rows sorted within each
  chromosome
* (optionally) no overlap with assembly gaps / the ENCODE blacklist

Rows **do** overlap, and that is not a violation: this is a per-gene division,
so a snoRNA nested in an intron legitimately sits inside its host gene's
interval. ``union_bp`` therefore reports real coverage while ``total_bp`` counts
shared sequence once per row.

``build_division.py`` calls :func:`validate` before writing, so a BED that
reaches disk has already passed. The CLI re-checks a file after the fact:

    python -m preprocess.regions.nested_genome_division.validate_division preprocess/regions/nested_genome_division/out/hg38_tss_tes.bed   # from the repo root
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from preprocess.regions import bedio
from preprocess.regions.nested_genome_division import sizing
from preprocess.regions.nested_genome_division import sources


@dataclass
class ValidationReport:
    """Outcome of :func:`validate` — violations plus descriptive statistics."""

    n_intervals: int = 0
    total_bp: int = 0  # sum of interval lengths (counts shared sequence twice)
    union_bp: int = 0  # distinct bases covered — differs from total_bp only when
                       # intervals overlap, i.e. only in the per-gene mode
    n_chroms: int = 0
    lengths: list[int] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self, *, genome_bp: int | None = None) -> str:
        """Human-readable one-block summary of counts, sizes and coverage."""
        if not self.lengths:
            return "  (no intervals)"
        ordered = sorted(self.lengths)
        n = len(ordered)

        def pct(p: float) -> int:
            return ordered[min(n - 1, int(p * n))]

        lines = [
            f"  intervals : {self.n_intervals:,} across {self.n_chroms} chromosomes",
            f"  total     : {self.total_bp/1e6:,.1f} Mbp (sum of lengths)",
            f"  covered   : {self.union_bp/1e6:,.1f} Mbp (distinct bases)"
            + (f"  ({100*self.union_bp/genome_bp:.1f}% of {genome_bp/1e9:.2f} Gbp)"
               if genome_bp else ""),
            f"  length    : min {ordered[0]:,}  p25 {pct(.25):,}  median {pct(.50):,}"
            f"  p75 {pct(.75):,}  p95 {pct(.95):,}  max {ordered[-1]:,}",
            f"  mean      : {self.total_bp/n:,.0f} bp",
        ]
        if self.total_bp != self.union_bp:
            redundancy = self.total_bp / self.union_bp
            lines.append(
                f"  overlap   : {(self.total_bp - self.union_bp)/1e6:,.1f} Mbp of "
                f"double-counted sequence ({redundancy:.2f}x redundancy)"
            )
        return "\n".join(lines)


def validate(
    rows: list[tuple[str, int, int]],
    chrom_sizes: dict[str, int],
    *,
    excluded: dict[str, list[sources.Span]] | None = None,
    max_report: int = 10,
) -> ValidationReport:
    """Check every invariant over ``(chrom, start, end)`` *rows*.

    *excluded* enables the gap/blacklist overlap check; pass ``None`` to skip it.
    Overlapping rows are expected and never reported — see the module docstring.

    Returns a :class:`ValidationReport`; the caller decides whether to raise.
    """
    report = ValidationReport()
    by_chrom: dict[str, list[sources.Span]] = {}

    def note(msg: str) -> None:
        if len(report.violations) < max_report:
            report.violations.append(msg)
        elif len(report.violations) == max_report:
            report.violations.append("... (further violations suppressed)")

    for chrom, start, end in rows:
        length = end - start
        if length <= 0:
            note(f"{chrom}:{start}-{end} degenerate (end <= start)")
            continue
        if length % sizing.GRID:
            note(f"{chrom}:{start}-{end} length {length} not divisible by {sizing.GRID}")
        if start < 0:
            note(f"{chrom}:{start}-{end} starts before 0")
        size = chrom_sizes.get(chrom)
        if size is None:
            note(f"{chrom} not in chrom.sizes")
        elif end > size:
            note(f"{chrom}:{start}-{end} runs past chromosome end {size}")

        by_chrom.setdefault(chrom, []).append((start, end))
        report.lengths.append(length)
        report.total_bp += length

    report.n_intervals = len(rows)
    report.n_chroms = len(by_chrom)

    for chrom, spans in by_chrom.items():
        ordered = sorted(spans)
        if ordered != spans:
            note(f"{chrom} rows are not in ascending coordinate order")
        holes = (excluded or {}).get(chrom, [])
        report.union_bp += sum(b - a for a, b in sources.merge_spans(ordered))
        if excluded is not None:
            for start, end in ordered:
                if _overlaps(start, end, holes):
                    note(f"{chrom}:{start}-{end} overlaps an excluded "
                         "(gap/blacklist) region")

    return report


def _overlaps(start: int, end: int, holes: list[sources.Span]) -> bool:
    """True if ``[start, end)`` touches any span in sorted disjoint *holes*."""
    lo, hi = 0, len(holes)
    while lo < hi:  # first hole whose end > start
        mid = (lo + hi) // 2
        if holes[mid][1] <= start:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(holes) and holes[lo][0] < end


def read_division_bed(path: Path) -> list[tuple[str, int, int]]:
    """Read a division BED into ``(chrom, start, end)`` rows, in file order."""
    rows: list[tuple[str, int, int]] = []
    for lineno, line, parts in bedio.iter_bed_fields(path, split="tab"):
        if len(parts) < 3:
            raise ValueError(f"{path}:{lineno}: fewer than 3 fields: {line!r}")
        rows.append((parts[0], int(parts[1]), int(parts[2])))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bed", type=Path, help="division BED to check")
    parser.add_argument("--chrom-sizes", type=Path, default=sources.DEFAULT_CHROM_SIZES)
    parser.add_argument("--gaps", type=Path, default=sources.DEFAULT_GAPS)
    parser.add_argument("--blacklist", type=Path, default=sources.DEFAULT_BLACKLIST)
    parser.add_argument("--no-exclusion-check", action="store_true",
                        help="skip the gap/blacklist overlap check")
    args = parser.parse_args(argv)

    rows = read_division_bed(args.bed)
    chrom_sizes = sources.read_chrom_sizes(args.chrom_sizes)
    excluded = (
        None if args.no_exclusion_check
        else sources.load_excluded(gaps=args.gaps, blacklist=args.blacklist)
    )

    report = validate(rows, chrom_sizes, excluded=excluded)
    # Coverage is reported against the chromosomes the BED actually spans, not a
    # fixed list, so a --chroms override still gets a meaningful percentage.
    present = {chrom for chrom, _, _ in rows}
    genome_bp = sum(n for c, n in chrom_sizes.items() if c in present)

    print(f"{args.bed}:")
    print(report.summary(genome_bp=genome_bp))
    if report.ok:
        print("  VALID — all invariants hold")
        return 0
    print(f"  INVALID — {len(report.violations)} violation(s):", file=sys.stderr)
    for violation in report.violations:
        print(f"    {violation}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
