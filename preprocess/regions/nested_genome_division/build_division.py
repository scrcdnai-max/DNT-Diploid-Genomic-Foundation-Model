#!/usr/bin/env python3
"""Build the hg38 TSS-TES division consumed by ``create_dataset.py --bed``.

**One GENCODE v50 gene, one interval.** Transcript records of the eleven
biotypes in ``sources.TSS_TES_TX_TYPES`` (``protein_coding`` plus the ten
functional non-coding classes) are merged per gene into a TSS->TES envelope,
then each envelope becomes an interval:

1. **drop** the gene outright if its envelope touches an unreliably-sequenced
   region — UCSC assembly gaps or the ENCODE blacklist v2;
2. add **1 kb upstream** (strand-aware) as a thumb-rule promoter margin;
3. extend **downstream** to the next multiple of **128**;
4. if that window runs into a gap/blacklist wall or the chromosome edge, slide it
   clear — the bases the wall takes back are added on the opposite end, so the
   interval keeps its exact length and stays 128-divisible.

There is no minimum and no maximum length. Genes are placed **independently**,
so intervals overlap wherever the genes do: a snoRNA inside an intron is its own
row inside its host gene's row. That is the point of a per-gene division, and it
is why ``validate_division`` reports ``covered`` (distinct bases) separately from
``total`` (sum of row lengths).

    # from the repo root
    python -m preprocess.regions.nested_genome_division.build_division \\
        --out preprocess/regions/nested_genome_division/out/hg38_tss_tes.bed

**Annotated output** (``--annotated-out``, for the v10 haplotype-pair dataset):
the *same* intervals, written as a headered BED6+3 whose column 4 is the GENCODE
``gene_id`` (unique and stable; the symbol is a label and can repeat) with extra
columns ``gene_name gene_type n_transcripts``, plus a ``<stem>.genes.tsv``
sidecar with one row per gene (transcript ids/names/types, HGNC id, level).
``create_haplotype_dataset.py`` propagates every header-named column beyond
the six BED fields into the Parquet, so adding an annotation later is "add a
column here". The pinned ``hg38_tss_tes.bed`` is untouched — this is a new
output file, byte-identical in columns 1-3 and strand.

    python -m preprocess.regions.nested_genome_division.build_division --out /dev/null \\
        --annotated-out preprocess/regions/nested_genome_division/hg38_tss_tes_v10.bed

This file used to build a second, TAD-based division as well; that mode and its
256-bp / 10-150 kbp size contract are gone (frozen at tag ``pre-reorg``, was ``legacy/``).
"""

from __future__ import annotations

import argparse
import bisect
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from preprocess.regions.nested_genome_division import sizing
from preprocess.regions.nested_genome_division import sources
from preprocess.regions.nested_genome_division import validate_division

#: Region-set name recorded in the annotated BED's ``#region_set=`` header line.
#: Distinct from ``versions.TSSTES_REGION_SET`` on purpose: same coordinates,
#: different file contract (header, gene_id in col 4, extra columns).
ANNOTATED_REGION_SET = "tss_tes_gencode50_tx128_v10"

#: Column header of the annotated BED (written after ``#region_set=``). The six
#: BED fields come first; everything after them is propagated by the v10 dataset
#: builder as a ``region_<col>`` column.
ANNOTATED_COLUMNS = (
    "chrom", "start", "end", "name", "score", "strand",
    "gene_name", "gene_type", "n_transcripts",
)

#: Columns of the ``<stem>.genes.tsv`` sidecar, one row per placed gene.
GENES_TSV_COLUMNS = (
    "gene_id", "gene_name", "gene_type", "strand", "chrom",
    "tss_tes_start", "tss_tes_end",
    "transcript_ids", "transcript_names", "transcript_types",
    "hgnc_id", "level", "gencode_version",
)

_GENCODE_VERSION_RE = re.compile(r"gencode\.(v\d+[a-z]*)", re.IGNORECASE)


def _group_genes(genes: list[sources.Gene]) -> dict[str, list[sources.Gene]]:
    """Bucket genes by chromosome, sorted by span within each."""
    grouped: dict[str, list[sources.Gene]] = {}
    for gene in genes:
        grouped.setdefault(gene.chrom, []).append(gene)
    for chrom_genes in grouped.values():
        chrom_genes.sort(key=lambda g: (g.start, g.end))
    return grouped


def _containing_region(
    regions: list[sources.Span], starts: list[int], start: int, end: int
) -> sources.Span | None:
    """The free region wholly containing ``[start, end)``, or None.

    *regions* is the sorted disjoint complement of the excluded set and *starts*
    its precomputed lower bounds, so a gene that cleared the exclusion gate lies
    inside exactly one of them and the bisect finds it in log time.
    """
    idx = bisect.bisect_right(starts, start) - 1
    if idx < 0:
        return None
    lo, hi = regions[idx]
    return (lo, hi) if end <= hi else None


def build_tss_tes(
    genes_by_chrom: dict[str, list[sources.Gene]],
    excluded: dict[str, list[sources.Span]],
    chrom_sizes: dict[str, int],
    *,
    chroms: tuple[str, ...],
    upstream: int = sizing.DEFAULT_UPSTREAM,
    verbose: bool = True,
    placed_genes: list[sources.Gene] | None = None,
) -> tuple[list[tuple], dict[str, int]]:
    """Turn every gene into one interval. Returns BED6+1 rows and stats.

    A gene is dropped when its envelope touches an excluded region (that is the
    "do not penetrate unreliably-sequenced sequence" rule, applied to the
    annotation itself rather than only to the padding), or in the pathological
    case where the free region it sits in is shorter than the padded window.

    *placed_genes*, when given, receives the :class:`sources.Gene` behind each
    returned row, in row order (``len(placed_genes) == len(rows)``). The
    annotated writer needs ``gene_id``/``transcript_ids``, which the 7-field
    row deliberately does not carry so the plain BED output stays byte-stable.
    """
    rows: list[tuple] = []
    stats = {
        "genes_in": 0,
        "dropped_body_excluded": 0,
        "wall_compensated": 0,
        "dropped_no_room": 0,
        "intervals": 0,
    }

    for chrom in chroms:
        size = chrom_sizes.get(chrom)
        if size is None:
            continue
        holes = excluded.get(chrom, [])
        chrom_genes = genes_by_chrom.get(chrom, [])
        stats["genes_in"] += len(chrom_genes)

        regions = sources.complement_spans(holes, (0, size))
        region_starts = [lo for lo, _ in regions]
        placed: list[tuple] = []

        for gene in chrom_genes:
            region = _containing_region(regions, region_starts, gene.start, gene.end)
            if region is None:
                # The envelope straddles or sits inside a gap/blacklist span.
                stats["dropped_body_excluded"] += 1
                continue

            start, end = sizing.pad_downstream(
                gene.start, gene.end, gene.strand, upstream=upstream
            )
            if start < region[0] or end > region[1]:
                window = sizing.place(
                    start, end, end - start,
                    lo_bound=region[0], hi_bound=region[1],
                )
                if window is None:
                    # Free region shorter than the padded window. Fall back to
                    # the largest grid-multiple that fits, seated on the gene.
                    fit = sizing.snap_down(region[1] - region[0])
                    window = (
                        sizing.place(gene.start, gene.end, fit,
                                     lo_bound=region[0], hi_bound=region[1])
                        if fit else None
                    )
                    if window is None:
                        stats["dropped_no_room"] += 1
                        continue
                stats["wall_compensated"] += 1
                start, end = window

            placed.append(
                ((start, end, gene.name, 0, gene.strand, gene.gene_type), gene)
            )

        # Padding reorders neighbours relative to their gene starts, so sort the
        # emitted windows rather than trusting the input order. Sort on the row
        # only (the Gene is not orderable) — ties are impossible because the
        # row includes the name and gene_type, which pin a gene.
        placed.sort(key=lambda item: item[0])
        rows.extend((chrom, *row) for row, _gene in placed)
        if placed_genes is not None:
            placed_genes.extend(gene for _row, gene in placed)

        if verbose:
            print(f"  {chrom}: {len(placed):,} intervals", flush=True)

    stats["intervals"] = len(rows)
    return rows, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_bed(path: Path, rows: list[tuple]) -> None:
    """Write BED6+1 rows tab-separated, atomically (``.tmp`` -> rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for row in rows:
            fh.write("\t".join(str(field) for field in row) + "\n")
    tmp.replace(path)


def _is_devnull(path: Path | None) -> bool:
    return path is not None and str(path) == os.devnull


def gencode_version_from_path(path: Path) -> str:
    """``gencode.v50.annotation.gtf.gz`` -> ``"v50"``; ``"unknown"`` otherwise."""
    match = _GENCODE_VERSION_RE.search(path.name)
    return match.group(1) if match else "unknown"


def annotated_rows(
    rows: list[tuple], genes: list[sources.Gene]
) -> list[tuple]:
    """BED6+3 rows for the annotated output, aligned with *rows*.

    ``(chrom, start, end, gene_id, 0, strand, gene_name, gene_type,
    n_transcripts)`` — coordinates and strand are copied from the plain row, so
    the two outputs can only ever differ in the label columns.
    """
    if len(rows) != len(genes):
        raise ValueError(
            f"rows/genes length mismatch: {len(rows)} vs {len(genes)}"
        )
    out: list[tuple] = []
    for row, gene in zip(rows, genes):
        chrom, start, end, name, score, strand, gene_type = row
        if gene.chrom != chrom or name != gene.name or strand != gene.strand:
            raise ValueError(
                f"row {row!r} is not aligned with gene {gene!r}"
            )
        out.append((
            chrom, start, end,
            gene.gene_id or gene.name,
            score, strand,
            gene.name, gene_type, len(gene.transcript_ids),
        ))
    return out


def _write_annotated_bed(
    path: Path, rows: list[tuple], *, region_set: str = ANNOTATED_REGION_SET
) -> None:
    """Write the headered BED6+3 atomically.

    Two ``#`` lines precede the data: ``#region_set=<name>`` and the tab-joined
    column header ``#chrom\\tstart\\t...``. Every BED reader in the repo skips
    ``#`` lines, and the v10 loader reads the header to name its extra columns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(f"#region_set={region_set}\n")
        fh.write("#" + "\t".join(ANNOTATED_COLUMNS) + "\n")
        for row in rows:
            if len(row) != len(ANNOTATED_COLUMNS):
                raise ValueError(
                    f"annotated row has {len(row)} fields, header names "
                    f"{len(ANNOTATED_COLUMNS)}: {row!r}"
                )
            fh.write("\t".join(str(field) for field in row) + "\n")
    tmp.replace(path)


def _write_genes_tsv(
    path: Path,
    genes: list[sources.Gene],
    annotations: dict[str, dict[str, object]],
    *,
    gencode_version: str,
) -> None:
    """Write the per-gene sidecar (one row per placed gene, BED row order).

    List-valued fields are ``;``-joined, parallel across ``transcript_ids``,
    ``transcript_names`` and ``transcript_types``. ``tss_tes_start``/``end`` are
    the gene's eligible-transcript envelope (0-based half-open) *before* the
    promoter margin and grid padding, which is what the BED row adds.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        fh.write("\t".join(GENES_TSV_COLUMNS) + "\n")
        for gene in genes:
            ann = annotations.get(gene.gene_id, {})
            names = ann.get("transcript_names", [])
            types = ann.get("transcript_types", [])
            fields = (
                gene.gene_id or gene.name, gene.name, gene.gene_type, gene.strand,
                gene.chrom, gene.start, gene.end,
                ";".join(gene.transcript_ids),
                ";".join(names), ";".join(types),
                ann.get("hgnc_id", ""), ann.get("level", ""), gencode_version,
            )
            fh.write("\t".join(str(f) for f in fields) + "\n")
    tmp.replace(path)


def genes_tsv_path_for(annotated_bed: Path) -> Path:
    """``hg38_tss_tes_v10.bed`` -> ``hg38_tss_tes_v10.genes.tsv`` (same dir)."""
    return annotated_bed.with_name(annotated_bed.stem + ".genes.tsv")


def _report_biotypes(genes: list[sources.Gene], requested: list[str]) -> None:
    """Per-biotype gene counts, naming any requested type that matched nothing.

    Printing the zeroes keeps a mis-typed or out-of-scope biotype from being a
    silent no-op — ``Mt_rRNA``/``Mt_tRNA`` were the standing example before they
    were dropped from the policy, and ``--chroms``/``--tx-types`` can recreate
    the situation at any time.
    """
    counts: dict[str, int] = {}
    for gene in genes:
        counts[gene.gene_type] = counts.get(gene.gene_type, 0) + 1
    for tx_type in sorted(requested, key=lambda t: (-counts.get(t, 0), t)):
        n = counts.get(tx_type, 0)
        note = "   <- requested but absent from these chromosomes" if not n else ""
        print(f"    {tx_type:16s} {n:7,d}{note}", flush=True)


def main(argv: list[str] | None = None) -> int:
    started = datetime.now()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="output BED6+1 (bare symbol in col 4; the pinned "
                             "hg38_tss_tes.bed format). /dev/null skips it. "
                             "Required unless --annotated-out is given.")
    parser.add_argument("--annotated-out", type=Path, default=None,
                        help="also write the headered BED6+3 (gene_id in col 4, "
                             "+gene_name gene_type n_transcripts) and its "
                             "<stem>.genes.tsv sidecar — the v10 dataset input")
    parser.add_argument("--gencode-version", default=None,
                        help="label recorded in the genes.tsv sidecar; default "
                             "is parsed from the --gencode file name (e.g. v50)")
    parser.add_argument("--gencode", type=Path, default=sources.DEFAULT_GENCODE)
    parser.add_argument("--chrom-sizes", type=Path, default=sources.DEFAULT_CHROM_SIZES)
    parser.add_argument("--gaps", type=Path, default=sources.DEFAULT_GAPS)
    parser.add_argument("--blacklist", type=Path, default=sources.DEFAULT_BLACKLIST)
    parser.add_argument("--tx-types", nargs="+",
                        default=sorted(sources.TSS_TES_TX_TYPES),
                        help="GENCODE transcript_type values to keep")
    parser.add_argument("--chroms", nargs="+", default=list(sources.DEFAULT_CHROMS))
    parser.add_argument("--upstream", type=int, default=sizing.DEFAULT_UPSTREAM,
                        help="promoter margin added on the 5' side, in bp")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.out is None and args.annotated_out is None:
        parser.error("at least one of --out / --annotated-out is required")

    verbose = not args.quiet
    print(f"[{started:%Y-%m-%d %H:%M:%S}] build_division tss-tes START", flush=True)
    print(f"  command: {' '.join(sys.argv)}", flush=True)

    try:
        chroms = tuple(args.chroms)
        chrom_sizes = sources.read_chrom_sizes(args.chrom_sizes)
        excluded = sources.load_excluded(gaps=args.gaps, blacklist=args.blacklist)
        annotations: dict[str, dict[str, object]] | None = (
            {} if args.annotated_out is not None else None
        )
        genes = sources.read_transcripts(
            args.gencode, chroms=set(chroms), tx_types=frozenset(args.tx_types),
            annotations_out=annotations,
        )
        print(f"  {len(genes):,} genes from transcripts "
              f"({', '.join(args.tx_types)})", flush=True)
        _report_biotypes(genes, args.tx_types)

        placed_genes: list[sources.Gene] = []
        rows, stats = build_tss_tes(
            _group_genes(genes), excluded, chrom_sizes,
            chroms=chroms, upstream=args.upstream, verbose=verbose,
            placed_genes=placed_genes,
        )
        print("  stats: " + "  ".join(f"{k}={v:,}" for k, v in stats.items()),
              flush=True)

        plain = [(row[0], row[1], row[2]) for row in rows]
        report = validate_division.validate(plain, chrom_sizes, excluded=excluded)
        genome_bp = sum(chrom_sizes[c] for c in chroms if c in chrom_sizes)
        print(report.summary(genome_bp=genome_bp), flush=True)
        if not report.ok:
            print("[ERROR] division failed validation:", file=sys.stderr, flush=True)
            for violation in report.violations:
                print(f"    {violation}", file=sys.stderr, flush=True)
            return 1

        if args.out is not None and not _is_devnull(args.out):
            _write_bed(args.out, rows)
            print(f"  wrote {len(rows):,} intervals -> {args.out}", flush=True)

        if args.annotated_out is not None:
            ann_rows = annotated_rows(rows, placed_genes)
            _write_annotated_bed(args.annotated_out, ann_rows)
            print(f"  wrote {len(ann_rows):,} annotated intervals "
                  f"(#region_set={ANNOTATED_REGION_SET}) -> {args.annotated_out}",
                  flush=True)
            gencode_version = (
                args.gencode_version or gencode_version_from_path(args.gencode)
            )
            tsv = genes_tsv_path_for(args.annotated_out)
            _write_genes_tsv(tsv, placed_genes, annotations or {},
                             gencode_version=gencode_version)
            print(f"  wrote {len(placed_genes):,} gene rows "
                  f"(gencode_version={gencode_version}) -> {tsv}", flush=True)
    except Exception as exc:
        print(f"[ERROR] build_division failed: {exc}", file=sys.stderr, flush=True)
        return 1

    elapsed = (datetime.now() - started).total_seconds()
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] build_division tss-tes OK "
          f"(elapsed {elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
