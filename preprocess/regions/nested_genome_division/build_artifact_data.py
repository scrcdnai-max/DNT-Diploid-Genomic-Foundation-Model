#!/usr/bin/env python3
"""Build the embedded dataset for the "hg38 Per-Gene Division" artifact.

That artifact is a self-contained HTML page (chromosome dropdown, scroll-to-zoom
genome track view) published via Claude's Artifact tool from an ephemeral
scratchpad, so its *page code* does not live in this repo. What does need to
survive across sessions — and what this script owns — is turning the real BED
inputs into the compact JS data blob the page embeds:

    const GENOME_DATA = {"sizes": {...}, "gaps": {...}, "genes": {...},
                         "stats": {...}};

* ``sizes``  chrom -> length, from ``raw/hg38.chrom.sizes``.
* ``gaps``   chrom -> [[start, end, typeCode], ...], from ``raw/gap.txt.gz``
             (typeCode: ct=contig, ht=heterochromatin, sc=scaffold, sa=short_arm,
             tl=telomere).
* ``genes``  chrom -> [[start, end, label, padLeft, padRight, biotype], ...],
             from ``hg38_tss_tes.bed`` (chr1-22 + chrX). This is a **per-gene**
             division, so these rows overlap wherever the genes do — draw them on
             stacked lanes, not one line. padLeft/padRight are NOT read off the
             BED (it only has the post-padding interval). They come from
             re-running the real placement algorithm
             (``build_division.build_tss_tes``, called here through a thin
             tracking wrapper that also records the ``(gene.start, gene.end)``
             each row was placed from) so the page can shade padding vs. the real
             gene span at reduced opacity. Padding is asymmetric by construction:
             1 kb upstream plus a 0-127 bp grid remainder downstream, mirrored on
             the minus strand. This re-run is checked row-for-row against the
             committed BED before being trusted — see ``--verify-only`` below,
             which fails loudly on any mismatch.
* ``stats``  the counts the "what is in, what is out" panel needs: per-biotype
             gene/interval/bp totals for the kept biotypes, the GENCODE
             transcript biotypes deliberately left out, the genes lost because
             none of their transcripts were eligible, and the drop funnel.

Regenerating the artifact end to end (e.g. after a GENCODE version bump):

1. If the GENCODE input changed, update ``sources.DEFAULT_GENCODE`` (or pass
   ``--gencode`` below) to the new GTF, e.g.
   ``gencode.v51.annotation.gtf.gz``.
2. Rebuild the division BED (see this directory's README, "Regenerate"):
       python -m preprocess.regions.nested_genome_division.build_division \\
           --gencode <new.gtf.gz> --out preprocess/regions/nested_genome_division/out/hg38_tss_tes.bed
   The builder runs ``validate_division.py`` before writing, so a BED that lands
   in ``out/`` has already passed every invariant. Copy it over the committed
   file in this directory and update ``TSSTES_BED_SHA256`` in
   ``../../huggingface_dataset/versions.py`` to match.
3. Run this script:
       python -m preprocess.regions.nested_genome_division.build_artifact_data --out /tmp/genome_data.js
   It re-derives padding by re-running the placement algorithm against whichever
   GENCODE file produced the committed ``hg38_tss_tes.bed``, and refuses to write
   output if that re-run doesn't reproduce the committed BED exactly (row count
   and every row must match) — a silent mismatch there would mean the padding
   shading is lying about where the real gene locus is.
4. Fetch the currently-published artifact's HTML as your editing base
   (Artifact tool, ``action: "read"`` with its URL — find the URL with
   ``action: "list"`` if you don't have it handy), replace its
   ``const GENOME_DATA = ...;`` line with the new file's contents (it is a
   single line — a plain line-based replace is enough, see the note below),
   and republish with ``url`` set to the same artifact so the link doesn't
   change. If you no longer have a live URL to read from, just republish
   fresh from scratch — a deleted artifact cannot be updated in place.

Line-based splice (what step 4's "replace the line" means in practice):

    with open("artifact.html") as f:
        lines = f.readlines()
    with open("/tmp/genome_data.js") as f:
        new_line = f.read().strip() + "\\n"
    for i, line in enumerate(lines):
        if line.startswith("const GENOME_DATA = "):
            lines[i] = new_line
            break
    open("artifact.html", "w").writelines(lines)

Run the whole thing with ``--verify-only`` to just check that the current
committed ``hg38_tss_tes.bed`` is still reproducible (useful as a standalone
sanity check, e.g. after any change to ``sizing.py``/``build_division.py``)
without writing a data bundle.

The older "TAD x Gene Browser" artifact this script also used to feed is gone
with the TAD division; see ``legacy/``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

from preprocess import paths  # noqa: E402
from preprocess.regions.nested_genome_division import build_division as bd  # noqa: E402
from preprocess.regions.nested_genome_division import sizing  # noqa: E402
from preprocess.regions.nested_genome_division import sources  # noqa: E402

#: The division's chromosome scope; every track is filtered to it.
IN_SCOPE_CHROMS = frozenset(sources.DEFAULT_CHROMS)


def in_scope(chrom: str) -> bool:
    return chrom in IN_SCOPE_CHROMS


GAP_CODE = {
    "contig": "ct",
    "heterochromatin": "ht",
    "scaffold": "sc",
    "short_arm": "sa",
    "telomere": "tl",
}


def load_gaps(gaps_path: Path) -> dict[str, list[list]]:
    """Per-chromosome assembly-gap rows straight off the UCSC gap table.

    Unlike ``sources.load_excluded`` (which merges gaps with the blacklist for
    the builder's own exclusion logic), this keeps every individual gap row —
    with its original type — for the artifact's mini-map, which draws each gap
    as its own notch rather than one merged blob.
    """
    import gzip

    out: dict[str, list[list]] = {}
    with gzip.open(gaps_path, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, start, end, gap_type = parts[1], parts[2], parts[3], parts[7]
            if not in_scope(chrom):
                continue
            out.setdefault(chrom, []).append(
                [int(start), int(end), GAP_CODE.get(gap_type, gap_type[:2])]
            )
    for c in out:
        out[c].sort(key=lambda r: r[0])
    return out


# ---------------------------------------------------------------------------
# Placement re-run with gene-span tracking
# ---------------------------------------------------------------------------
#
# Placement is independent per gene — pad, then slide clear of a wall if one is
# in the way — so re-deriving each row's core span means repeating a dozen lines
# of the real builder rather than forking a whole algorithm. The verification
# against the committed BED stays anyway: it is what proves the biotype and
# chromosome scope used here match what produced the file, not just that the
# arithmetic agrees.


def build_tss_tes_with_core(
    genes_by_chrom, excluded, chrom_sizes, chroms, *, upstream=sizing.DEFAULT_UPSTREAM
):
    """Division rows plus, for each, the gene span it was padded out from.

    Returns ``(rows, funnel)``. A row is ``(chrom, start, end, name, score,
    strand, biotype, core_start, core_end)``. *funnel* counts what happened to
    every gene on the way in, which is what the artifact's drop panel states.
    """
    rows = []
    funnel = {"genes": 0, "body_excluded": 0, "wall_compensated": 0,
              "no_room": 0, "intervals": 0}
    for chrom in chroms:
        size = chrom_sizes.get(chrom)
        if size is None:
            continue
        holes = excluded.get(chrom, [])
        chrom_genes = genes_by_chrom.get(chrom, [])
        funnel["genes"] += len(chrom_genes)
        regions = sources.complement_spans(holes, (0, size))
        region_starts = [lo for lo, _ in regions]
        placed = []
        for gene in chrom_genes:
            region = bd._containing_region(
                regions, region_starts, gene.start, gene.end
            )
            if region is None:
                funnel["body_excluded"] += 1
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
                    fit = sizing.snap_down(region[1] - region[0])
                    window = (
                        sizing.place(gene.start, gene.end, fit,
                                     lo_bound=region[0], hi_bound=region[1])
                        if fit else None
                    )
                    if window is None:
                        funnel["no_room"] += 1
                        continue
                funnel["wall_compensated"] += 1
                start, end = window
            placed.append((start, end, gene.name, 0, gene.strand, gene.gene_type,
                           gene.start, gene.end))
        placed.sort()
        rows.extend((chrom, *row) for row in placed)
    funnel["intervals"] = len(rows)
    return rows, funnel


def read_committed_bed(path: Path) -> list[tuple]:
    """Rows of the committed BED, coordinates as ints, rest as strings."""
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows.append((parts[0], int(parts[1]), int(parts[2]), *parts[3:]))
    return rows


def derive_gene_padding(
    *,
    gencode: Path,
    verify_against: Path,
    tx_types=sources.TSS_TES_TX_TYPES,
    chroms=sources.DEFAULT_CHROMS,
):
    """Padding-annotated rows + biotype stats, verified against the BED.

    Raises ``SystemExit`` if the re-run doesn't reproduce ``verify_against``
    exactly — that BED is the ground truth, and a mismatch means this script's
    inputs or scope have drifted from ``build_division.py``'s and must not be
    trusted to shade padding.
    """
    chrom_sizes = sources.read_chrom_sizes()
    excluded = sources.load_excluded()
    genes = sources.read_transcripts(gencode, chroms=set(chroms), tx_types=tx_types)
    print(f"  {len(genes):,} genes from {gencode.name}", flush=True)

    rows, funnel = build_tss_tes_with_core(
        bd._group_genes(genes), excluded, chrom_sizes, chroms
    )
    print(f"  {len(rows):,} rows produced (one row per gene, overlaps allowed)",
          flush=True)

    committed = read_committed_bed(verify_against)
    mine = [(c, s, e, name, str(score), strand, biotype)
            for c, s, e, name, score, strand, biotype, _cs, _ce in rows]
    if committed != mine:
        n_mismatch = sum(1 for a, b in zip(committed, mine) if a != b)
        n_mismatch += abs(len(committed) - len(mine))
        print(
            f"[ERROR] re-run does not reproduce {verify_against.name} "
            f"({n_mismatch} mismatching/missing rows out of {len(committed):,}). "
            "This means build_division.py, sources.TSS_TES_TX_TYPES/"
            "DEFAULT_CHROMS, or the inputs (gaps/blacklist/chrom.sizes) differ "
            "from what produced the committed BED. Refusing to emit padding data "
            "that would mis-shade the artifact — fix the drift before rerunning.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"  verified: 0 mismatches against {verify_against.name}", flush=True)

    by_chrom: dict[str, list[list]] = {}
    for c, s, e, name, _score, _strand, biotype, cs, ce in rows:
        core_lo, core_hi = max(s, cs), min(e, ce)
        by_chrom.setdefault(c, []).append(
            [s, e, name, core_lo - s, e - core_hi, biotype]
        )
    for c in by_chrom:
        by_chrom[c].sort(key=lambda r: r[0])
    stats = gene_stats(
        genes, rows, gencode=gencode, tx_types=tx_types, chroms=chroms
    )
    stats["funnel"] = funnel
    return by_chrom, stats


def scan_all_transcript_types(
    gencode: Path, chroms
) -> tuple[dict[str, int], dict[str, int], int]:
    """What every GENCODE transcript biotype would have contributed, in scope.

    ``sources.read_transcripts`` filters, so it can only ever report what was
    kept. The artifact has to state what was *not* kept with the same
    denominator, which means one extra pass over the GTF.

    Returns ``(per_tx_type, no_eligible_transcript, n_eligible_transcripts)``:

    * ``per_tx_type`` — transcript_type -> number of distinct genes carrying at
      least one transcript of that type. A gene with isoforms of several types is
      counted under each, so these do not sum to a gene total.
    * ``no_eligible_transcript`` — gene_type -> number of genes whose *gene* type
      is one we keep but which contribute no transcript of a kept type, so they
      are absent from the division. 357 ``protein_coding`` genes annotated only
      through ``retained_intron`` / ``nonsense_mediated_decay`` /
      ``protein_coding_CDS_not_defined`` land here; that is intended, not a bug,
      and the panel should say so out loud.
    * ``n_eligible_transcripts`` — how many transcript *records* passed the
      biotype filter. This is the top of the artifact's funnel, and counting it
      here rather than writing it into the page by hand is what keeps the page's
      claim that every figure on it is computed true.
    """
    per_type: dict[str, set[str]] = {}
    n_eligible = 0
    gene_type_of: dict[str, str] = {}
    has_eligible: set[str] = set()
    wanted = set(chroms)
    with sources._open_maybe_gzip(gencode) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 9)
            if len(fields) < 9 or fields[2] != "transcript" or fields[0] not in wanted:
                continue
            attrs = fields[8]
            tt = sources._TX_TYPE_RE.search(attrs)
            gid = sources._GENE_ID_RE.search(attrs)
            gt = sources._GENE_TYPE_RE.search(attrs)
            if tt is None or gid is None:
                continue
            per_type.setdefault(tt.group(1), set()).add(gid.group(1))
            if gt is not None:
                gene_type_of[gid.group(1)] = gt.group(1)
            if tt.group(1) in sources.TSS_TES_TX_TYPES:
                has_eligible.add(gid.group(1))
                n_eligible += 1

    missed: dict[str, int] = {}
    for gene_id, gene_type in gene_type_of.items():
        if gene_type in sources.TSS_TES_TX_TYPES and gene_id not in has_eligible:
            missed[gene_type] = missed.get(gene_type, 0) + 1
    return {t: len(v) for t, v in per_type.items()}, missed, n_eligible


def gene_stats(genes, rows, *, gencode: Path, tx_types, chroms) -> dict:
    """Everything the artifact's "what is in, what is out" panel states.

    ``kept`` is per-biotype: genes read, intervals emitted, distinct bases
    covered and the median raw gene span — with an explicit zero row for a
    requested biotype that matched nothing, so the page can show it as
    requested-but-empty rather than omitting it. ``dropped_types`` is every other
    GENCODE transcript biotype with its gene count over the *same* chromosomes,
    so "in" and "out" share a denominator. ``lengths`` is a log2 histogram of
    emitted interval lengths — the division spans 1,152 bp to ~2.5 Mbp with no
    floor or ceiling, so linear buckets would put almost everything in the first
    one.
    """
    import statistics

    kept: dict[str, dict] = {
        tx_type: {"genes": 0, "intervals": 0, "bp": 0, "core_bp": 0,
                  "median_span": 0, "spans": []}
        for tx_type in tx_types
    }
    for gene in genes:
        entry = kept[gene.gene_type]
        entry["genes"] += 1
        entry["spans"].append(gene.end - gene.start)

    # Both the padded footprint and the raw gene footprint, per biotype. The gap
    # between them is what padding costs — for the small-RNA classes the 1 kb
    # promoter margin is most of the interval, which is the honest answer to
    # "what does keeping miRNA in the biotype list actually put in the dataset".
    per_type_spans: dict[str, list[tuple[str, int, int]]] = {}
    per_type_cores: dict[str, list[tuple[str, int, int]]] = {}
    for c, s, e, _n, _sc, _st, biotype, cs, ce in rows:
        kept[biotype]["intervals"] += 1
        per_type_spans.setdefault(biotype, []).append((c, s, e))
        per_type_cores.setdefault(biotype, []).append((c, max(s, cs), min(e, ce)))

    def _union(spans):
        by_chrom: dict[str, list[tuple[int, int]]] = {}
        for c, s, e in spans:
            by_chrom.setdefault(c, []).append((s, e))
        return sum(b - a for v in by_chrom.values() for a, b in sources.merge_spans(v))

    for biotype, spans in per_type_spans.items():
        kept[biotype]["bp"] = _union(spans)
        kept[biotype]["core_bp"] = _union(per_type_cores[biotype])

    for entry in kept.values():
        spans = entry.pop("spans")
        entry["median_span"] = int(statistics.median(spans)) if spans else 0

    all_counts, no_eligible, n_eligible_tx = scan_all_transcript_types(gencode, chroms)
    dropped_types = {t: n for t, n in all_counts.items() if t not in tx_types}

    # Union coverage per chromosome, and the sum-of-lengths it is measured
    # against — they differ only because the rows overlap.
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    total_bp = 0
    lengths: dict[str, int] = {}
    for c, s, e, *_rest in rows:
        by_chrom.setdefault(c, []).append((s, e))
        total_bp += e - s
        bucket = str(int(math.log2(e - s)))
        lengths[bucket] = lengths.get(bucket, 0) + 1
    coverage = {
        c: sum(b - a for a, b in sources.merge_spans(v))
        for c, v in by_chrom.items()
    }

    return {
        "kept": kept,
        "dropped_types": dropped_types,
        "no_eligible_transcript": no_eligible,
        "n_eligible_transcripts": n_eligible_tx,
        "chroms": list(chroms),
        "coverage": coverage,
        "total_bp": total_bp,
        "union_bp": sum(coverage.values()),
        "lengths": lengths,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "out" / "genome_data.js",
                    help="where to write the 'const GENOME_DATA = ...;' JS file")
    p.add_argument("--gencode", type=Path, default=sources.DEFAULT_GENCODE,
                    help="GENCODE GTF that produced --tss-tes-bed (bump this for a version update)")
    p.add_argument("--tss-tes-bed", type=Path, default=paths.TSSTES_BED,
                    help="committed division BED to verify the padding re-run against")
    p.add_argument("--gaps", type=Path, default=sources.DEFAULT_GAPS)
    p.add_argument("--chrom-sizes", type=Path, default=sources.DEFAULT_CHROM_SIZES)
    p.add_argument("--verify-only", action="store_true",
                    help="just check the padding re-run reproduces --tss-tes-bed; write nothing")
    args = p.parse_args(argv)

    print("[1/2] chrom sizes + assembly gaps", flush=True)
    sizes = sources.read_chrom_sizes(args.chrom_sizes)
    sizes = {c: n for c, n in sizes.items() if in_scope(c)}
    gaps = load_gaps(args.gaps)

    print("[2/2] padding re-derivation + verification", flush=True)
    genes, stats = derive_gene_padding(
        gencode=args.gencode, verify_against=args.tss_tes_bed
    )
    if args.verify_only:
        print("--verify-only: not writing a data bundle.", flush=True)
        return 0

    payload = {"sizes": sizes, "gaps": gaps, "genes": genes, "stats": stats}
    js = "const GENOME_DATA = " + json.dumps(payload, separators=(",", ":")) + ";"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(js)
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
