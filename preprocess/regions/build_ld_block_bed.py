#!/usr/bin/env python3
"""Normalize a raw LD-block BED (jmacdon/LDblocks_GRCh38) into a clean 3-column
BED for the create_dataset region pipeline, and emit chromosome-subset pilots.

The raw deCODE/pyrho files are tab-separated but whitespace-padded and carry a
`chr start stop` header row; blocks tile each chromosome contiguously. This script
strips that, validates non-overlap + ascending order, restricts to autosomes, and
writes `chrom<TAB>start<TAB>stop` with no header (0-based half-open, as BED).

Provenance: preprocess/regions/ld_blocks/raw/README.md

Example:
    # full EUR set
    python build_ld_block_bed.py --input ld_blocks/raw/deCODE_EUR_LD_blocks.bed \
        --output ld_blocks_eur_grch38.bed
    # chr21+chr22 pilot
    python build_ld_block_bed.py --input ld_blocks/raw/deCODE_EUR_LD_blocks.bed \
        --output ld_blocks_eur_grch38.chr21_22.bed --chroms chr21 chr22
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

AUTOSOMES = frozenset(f"chr{i}" for i in range(1, 23))

# Header tokens we skip if a row looks like the `chr start stop` label line.
_HEADER_FIRST_TOKENS = {"chr", "chrom", "chromosome", "#chr"}


def normalize_ld_blocks(
    lines: list[str], chroms: set[str] | None = None
) -> list[tuple[str, int, int]]:
    """Parse raw LD-block lines → list of (chrom, start, stop) with int coords.

    Drops a header row, strips whitespace padding, and (if ``chroms`` given)
    keeps only those contigs. Blank lines are ignored.
    """
    blocks: list[tuple[str, int, int]] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 3:
            # tolerate any-whitespace-delimited rows too
            parts = line.split()
        if len(parts) < 3:
            continue
        chrom, start_s, stop_s = parts[0], parts[1], parts[2]
        if chrom.lower() in _HEADER_FIRST_TOKENS or not start_s.lstrip("-").isdigit():
            continue  # header / non-data row
        if chroms is not None and chrom not in chroms:
            continue
        blocks.append((chrom, int(start_s), int(stop_s)))
    return blocks


def validate_blocks(
    blocks: list[tuple[str, int, int]], require_autosomes: bool = False
) -> None:
    """Raise ValueError if any block is degenerate, out of order, or overlapping.

    Checks per contig: start < stop, strictly ascending starts, and no overlap
    (next.start >= prev.stop). Optionally require all contigs to be autosomes.
    """
    if not blocks:
        raise ValueError("no LD blocks parsed (empty input)")
    for chrom, start, stop in blocks:
        if start >= stop:
            raise ValueError(f"degenerate block {chrom}:{start}-{stop} (start >= stop)")
        if require_autosomes and chrom not in AUTOSOMES:
            raise ValueError(f"non-autosome contig/chrom {chrom!r} (expected chr1..chr22)")
    # per-chromosome order + overlap
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for chrom, start, stop in blocks:
        by_chrom.setdefault(chrom, []).append((start, stop))
    for chrom, spans in by_chrom.items():
        prev_start, prev_stop = -1, -1
        for start, stop in spans:  # preserve input order (do NOT re-sort: order is a check)
            if start < prev_start:
                raise ValueError(
                    f"{chrom} blocks not in ascending order: {start} after {prev_start}"
                )
            if start < prev_stop:
                raise ValueError(
                    f"{chrom} blocks overlap: {start} < previous stop {prev_stop}"
                )
            prev_start, prev_stop = start, stop


def format_bed(blocks: list[tuple[str, int, int]]) -> str:
    """Render blocks as tab-separated 3-col BED text (no header, trailing newline)."""
    return "".join(f"{c}\t{s}\t{e}\n" for c, s, e in blocks)


def main(argv: list[str] | None = None) -> int:
    start_ts = datetime.now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="raw LD-block BED")
    parser.add_argument("--output", required=True, type=Path, help="normalized BED out")
    parser.add_argument(
        "--chroms", nargs="+", default=None,
        help="restrict to these contigs (e.g. chr21 chr22); default = all autosomes",
    )
    parser.add_argument(
        "--allow-non-autosomes", action="store_true",
        help="do not require autosome-only contigs",
    )
    args = parser.parse_args(argv)

    print(f"[{start_ts:%Y-%m-%d %H:%M:%S}] build_ld_block_bed START", flush=True)
    print(f"  command: {' '.join(sys.argv)}", flush=True)

    try:
        raw = args.input.read_text().splitlines()
        chroms = set(args.chroms) if args.chroms else None
        blocks = normalize_ld_blocks(raw, chroms=chroms)
        # default: restrict to autosomes unless explicitly allowed or user picked chroms
        if chroms is None and not args.allow_non_autosomes:
            blocks = [b for b in blocks if b[0] in AUTOSOMES]
        validate_blocks(blocks, require_autosomes=not args.allow_non_autosomes and chroms is None)
        args.output.write_text(format_bed(blocks))

        contigs = sorted({b[0] for b in blocks})
        total_bp = sum(e - s for _, s, e in blocks)
        print(
            f"  wrote {len(blocks)} blocks across {len(contigs)} contigs "
            f"({total_bp/1e6:.1f} Mbp) -> {args.output}",
            flush=True,
        )
    except Exception as exc:  # explicit failure message on every path
        print(f"[ERROR] build_ld_block_bed failed: {exc}", file=sys.stderr, flush=True)
        return 1

    end_ts = datetime.now()
    print(
        f"[{end_ts:%Y-%m-%d %H:%M:%S}] build_ld_block_bed OK "
        f"(elapsed {(end_ts - start_ts).total_seconds():.2f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
