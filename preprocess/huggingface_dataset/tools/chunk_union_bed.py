#!/usr/bin/env python3
"""Cut a merged union BED into per-(chromosome, chunk) BEDs for a distributed
`bcftools +split` Phase 1.

`bcftools +split` is single-threaded and has no `--threads`, so the only way to
use more than one core per chromosome is to run several processes over disjoint
slices of the region set. This script produces those slices.

THE ONE CORRECTNESS CONSTRAINT. `bcftools +split` defaults to
`--regions-overlap 1`, i.e. a record is emitted when it *overlaps* a region, not
merely when its POS falls inside one. So a variant whose REF spans the boundary
between two chunks would be emitted by BOTH of them, and the per-sample concat
would carry it twice — a difference from a single-machine run.

Cutting only at gaps of at least ``--min-gap`` bp between consecutive merged
intervals makes that impossible for any variant shorter than the gap. The
default 100 kb is far beyond any indel a short-read joint callset emits, and the
tss-tes union BED offers 2,115 such gaps with at least 7 on every chromosome.
Chunks are therefore always cut at real intergenic distance, never mid-gene, and
`--verify-gaps` re-checks every boundary it actually used.

Chunk counts are allocated across chromosomes in proportion to raw panel BCF
size (the best available proxy for decode cost, which is what dominates), and
within a chromosome the cut points are chosen to balance covered bp.

Usage:
    python -m preprocess.huggingface_dataset.tools.chunk_union_bed UNION_BED OUT_DIR --raw-dir DIR [--chroms chr14,chr2,...]
                       [--workers 48] [--min-gap 100000]

Writes OUT_DIR/<chrom>.<idx:03d>.bed plus OUT_DIR/manifest.tsv. Chunk basenames
sort lexically into genomic order within a chromosome, which is the order the
per-sample concat must use.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


def read_union_bed(path: pathlib.Path) -> dict[str, list[tuple[int, int]]]:
    """Load a merged BED as {chrom: [(start, end), ...]} in coordinate order."""
    spans: dict[str, list[tuple[int, int]]] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            spans.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    for chrom, iv in spans.items():
        iv.sort()
        for (a, b), (c, _) in zip(iv, iv[1:]):
            if c < b:
                sys.exit(f"ERROR: {path} is not merged — {chrom}:{a}-{b} overlaps {c}")
    return spans


def raw_bcf_for(raw_dir: pathlib.Path, chrom: str) -> pathlib.Path:
    """The panel BCF for *chrom*, matched on the chromosome token in its name."""
    pat = re.compile(rf"(?<![0-9A-Za-z]){re.escape(chrom)}(?![0-9])")
    hits = [p for p in sorted(raw_dir.glob("*.bcf")) if pat.search(p.name)]
    if len(hits) != 1:
        sys.exit(f"ERROR: expected exactly one {chrom} BCF in {raw_dir}, found {len(hits)}")
    return hits[0]


def cut_points(intervals: list[tuple[int, int]], n_chunks: int, min_gap: int) -> list[int]:
    """Interval indices at which to start a new chunk, balancing covered bp.

    Only positions preceded by a gap of >= *min_gap* are eligible. Returns at
    most ``n_chunks - 1`` indices; fewer when the chromosome does not offer
    enough wide gaps, in which case the caller simply gets fewer chunks.
    """
    if n_chunks <= 1 or len(intervals) < 2:
        return []

    eligible = [
        i for i in range(1, len(intervals))
        if intervals[i][0] - intervals[i - 1][1] >= min_gap
    ]
    if not eligible:
        return []

    covered = [b - a for a, b in intervals]
    cum: list[int] = []
    running = 0
    for c in covered:
        running += c
        cum.append(running)
    total = running

    chosen: list[int] = []
    for k in range(1, n_chunks):
        target = total * k / n_chunks
        # Nearest eligible boundary to the target cumulative coverage, never
        # reusing one already taken (that would make an empty chunk).
        best = min(
            (i for i in eligible if i not in chosen),
            key=lambda i: abs(cum[i - 1] - target),
            default=None,
        )
        if best is not None:
            chosen.append(best)
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("union_bed", type=pathlib.Path)
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--raw-dir", type=pathlib.Path, required=True,
                    help="directory of per-chromosome panel BCFs (for size-proportional allocation)")
    ap.add_argument("--chroms", default="",
                    help="comma-separated subset to chunk (default: every chromosome in the BED)")
    ap.add_argument("--workers", type=int, default=48,
                    help="total chunks to aim for across all chromosomes [48]")
    ap.add_argument("--min-gap", type=int, default=100_000,
                    help="minimum inter-interval gap a cut may land in, bp [100000]")
    args = ap.parse_args()

    spans = read_union_bed(args.union_bed)
    chroms = args.chroms.split(",") if args.chroms else sorted(spans)
    missing = [c for c in chroms if c not in spans]
    if missing:
        sys.exit(f"ERROR: not in {args.union_bed}: {', '.join(missing)}")

    sizes = {c: raw_bcf_for(args.raw_dir, c).stat().st_size for c in chroms}

    # Allocate to minimise the makespan, not merely in proportion to size: every
    # chromosome starts with one chunk and each remaining worker goes to whichever
    # chromosome currently has the largest bytes-per-chunk. Plain proportional
    # rounding leaves the longest single chunk far above the mean (it handed
    # chr20's 5.75 GiB to one worker while chr2's 22.6 GiB got five), and the run
    # finishes when the slowest chunk does.
    if args.workers < len(chroms):
        sys.exit(f"ERROR: --workers {args.workers} is below the {len(chroms)} "
                 "chromosomes requested; each needs at least one chunk")
    alloc = {c: 1 for c in chroms}
    for _ in range(args.workers - len(chroms)):
        alloc[max(chroms, key=lambda c: sizes[c] / alloc[c])] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*.bed"):
        stale.unlink()

    rows = []
    for chrom in sorted(chroms, key=lambda c: (len(c), c)):
        iv = spans[chrom]
        cuts = cut_points(iv, alloc[chrom], args.min_gap)
        bounds = [0, *cuts, len(iv)]
        if len(bounds) - 1 < alloc[chrom]:
            print(f"  NOTE {chrom}: wanted {alloc[chrom]} chunks, "
                  f"only {len(bounds) - 1} gaps >= {args.min_gap:,} bp available")
        for idx, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            part = iv[lo:hi]
            name = f"{chrom}.{idx:03d}"
            with open(args.out_dir / f"{name}.bed", "w") as fh:
                for a, b in part:
                    fh.write(f"{chrom}\t{a}\t{b}\n")
            gap_before = iv[lo][0] - iv[lo - 1][1] if lo > 0 else -1
            rows.append((name, chrom, idx, len(part), sum(b - a for a, b in part),
                         part[0][0], part[-1][1], gap_before))

    with open(args.out_dir / "manifest.tsv", "w") as fh:
        fh.write("chunk\tchrom\tidx\tn_intervals\tcovered_bp\tfirst_start\tlast_end\tgap_before_bp\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    bad = [r for r in rows if 0 <= r[7] < args.min_gap]
    if bad:
        sys.exit(f"ERROR: {len(bad)} cut(s) landed in a gap < {args.min_gap:,} bp: "
                 + ", ".join(r[0] for r in bad[:5]))

    print(f"{len(rows)} chunks over {len(chroms)} chromosomes -> {args.out_dir}")
    print(f"  covered {sum(r[4] for r in rows) / 1e6:,.1f} Mbp, "
          f"{sum(r[3] for r in rows):,} intervals")
    print(f"  every cut sits in a gap >= {args.min_gap:,} bp "
          f"(narrowest used: {min((r[7] for r in rows if r[7] >= 0), default=-1):,} bp)")

    # Raw bytes per chunk is what sets the wall clock: the run ends when the
    # heaviest chunk does, so print that alongside the mean.
    per_chunk = {}
    for name, chrom, _idx, _n, cov, _fs, _le, _g in rows:
        share = cov / sum(r[4] for r in rows if r[1] == chrom)
        per_chunk[name] = sizes[chrom] * share
    heaviest = max(per_chunk, key=per_chunk.get)
    print(f"  raw bytes per chunk: mean {sum(per_chunk.values()) / len(per_chunk) / 2**30:.2f} GiB, "
          f"heaviest {per_chunk[heaviest] / 2**30:.2f} GiB ({heaviest})")
    print(f"  at the measured ~0.07 MB/s per core that is ~"
          f"{per_chunk[heaviest] / 2**20 / 0.07 / 3600:.1f} h wall clock")


if __name__ == "__main__":
    main()
