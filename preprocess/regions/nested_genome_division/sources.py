#!/usr/bin/env python3
"""Input parsers and interval algebra for the TSS-TES genome division.

All coordinates here are 0-based half-open (BED convention), matching what
``create_dataset.py``'s ``load_bed_regions`` expects. GENCODE GTF is the one
1-based inclusive input and is converted on read.

Inputs, all already on disk — nothing is fetched:

* GENCODE v50 transcript annotation  ``/storage/data/reference_genome/``
* UCSC assembly gaps                 ``raw/gap.txt.gz``
* ENCODE blacklist v2                ``raw/hg38-blacklist.v2.bed.gz``
* hg38 chromosome sizes              ``raw/hg38.chrom.sizes``

The division reads **transcript** records, not gene records, and merges them per
gene. For the eleven biotypes in :data:`TSS_TES_TX_TYPES` GENCODE v50 happens to
have no row where ``transcript_type != gene_type``, so the merged envelope
matches the gene record for every gene that survives — but the selection is not
equivalent, because a gene whose only transcripts are ``retained_intron`` or
``nonsense_mediated_decay`` has no eligible transcript and drops out entirely
(357 ``protein_coding`` genes do exactly that).
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, replace
from pathlib import Path

from preprocess.regions import bedio

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

DEFAULT_GENCODE = Path("/storage/data/reference_genome/gencode.v50.annotation.gtf.gz")
DEFAULT_GAPS = RAW / "gap.txt.gz"
DEFAULT_BLACKLIST = RAW / "hg38-blacklist.v2.bed.gz"
DEFAULT_CHROM_SIZES = RAW / "hg38.chrom.sizes"

AUTOSOMES: tuple[str, ...] = tuple(f"chr{i}" for i in range(1, 23))

#: The division's chromosome scope. chrY and chrM are deliberately out: chrY is
#: present in only half the cohort and its ampliconic palindromes are mostly
#: blacklisted anyway, and chrM is a 16.6 kb circle whose 37 genes would each
#: wrap most of it.
DEFAULT_CHROMS: tuple[str, ...] = AUTOSOMES + ("chrX",)

#: GENCODE ``transcript_type`` values that define a transcribed locus for our
#: purposes: ``protein_coding`` plus every biotype that makes a real, functional
#: non-coding RNA — the small-RNA classes (miRNA, snoRNA, snRNA, scaRNA, sRNA,
#: vault_RNA, misc_RNA), rRNA and ribozyme.
#:
#: Deliberately excluded:
#:
#: * every pseudogene class (``processed_pseudogene``, ``unprocessed_pseudogene``,
#:   ``transcribed_*_pseudogene``, ``unitary_pseudogene``, ``rRNA_pseudogene``,
#:   ``IG_*_pseudogene``, ``TR_*_pseudogene``) — ~14,800 genes, by far the
#:   largest excluded group;
#: * ``TEC`` (to-be-experimentally-confirmed) and ``artifact``;
#: * the rearranging immunoglobulin/T-cell-receptor segments (``IG_*_gene``,
#:   ``TR_*_gene``) — the reference locus is not what any given B or T cell
#:   actually transcribes;
#: * the coding-adjacent *transcript* classes ``nonsense_mediated_decay``,
#:   ``retained_intron``, ``protein_coding_CDS_not_defined``, ``non_stop_decay``,
#:   ``protein_coding_LoF`` and ``processed_transcript``. These are isoforms of
#:   coding genes, so excluding them narrows a coding gene's envelope to its
#:   coding-transcript extent and drops the 357 genes annotated *only* that way;
#: * ``Mt_rRNA`` and ``Mt_tRNA``, which occur only on chrM — out of
#:   :data:`DEFAULT_CHROMS`, so listing them would be a silent no-op.
TSS_TES_TX_TYPES = frozenset({
    "protein_coding",
    "lncRNA",
    "misc_RNA",
    "snRNA",
    "miRNA",
    "snoRNA",
    "scaRNA",
    "rRNA",
    "ribozyme",
    "sRNA",
    "vault_RNA",
})

_TX_TYPE_RE = re.compile(r'transcript_type "([^"]+)"')
_GENE_TYPE_RE = re.compile(r'gene_type "([^"]+)"')
_GENE_NAME_RE = re.compile(r'gene_name "([^"]+)"')
_GENE_ID_RE = re.compile(r'gene_id "([^"]+)"')
_TX_ID_RE = re.compile(r'transcript_id "([^"]+)"')
# Sidecar-only attributes (``read_transcripts(annotations_out=...)``).
_TX_NAME_RE = re.compile(r'transcript_name "([^"]+)"')
_HGNC_ID_RE = re.compile(r'hgnc_id "([^"]+)"')
_LEVEL_RE = re.compile(r'\blevel (\d+)')

Span = tuple[int, int]
"""A half-open ``(start, end)`` interval on some implied chromosome."""


@dataclass(frozen=True, slots=True)
class Gene:
    """One gene, spanning the union of its eligible transcripts' TSS->TES spans.

    Built by :func:`read_transcripts`, never read straight off a GTF ``gene``
    row — ``start``/``end`` are ``min``/``max`` over the transcripts that passed
    the biotype filter, which is not the same as the gene record's own extent
    when some of the gene's isoforms were filtered out.

    ``gene_id`` is the GENCODE ``gene_id`` **verbatim, version suffix included**
    (``ENSG00000290825.2``) — unique and stable within one GENCODE release,
    unlike ``name`` (a symbol, which can repeat and falls back to the id when
    absent). ``transcript_ids`` lists the *eligible* transcripts that built the
    envelope, in GTF order. Both default to empty so the positional
    ``Gene(chrom, start, end, name, gene_type, strand)`` construction used
    before they existed still works.
    """

    chrom: str
    start: int  # 0-based inclusive
    end: int  # 0-based exclusive
    name: str
    gene_type: str
    strand: str
    gene_id: str = ""
    transcript_ids: tuple[str, ...] = ()


def _open_maybe_gzip(path: Path):
    """Open *path* as text, transparently decompressing a ``.gz``."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def read_chrom_sizes(path: Path = DEFAULT_CHROM_SIZES) -> dict[str, int]:
    """Parse a UCSC ``chrom.sizes`` file into ``{chrom: length}``."""
    sizes: dict[str, int] = {}
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                sizes[parts[0]] = int(parts[1])
    if not sizes:
        raise ValueError(f"no chromosome sizes parsed from {path}")
    return sizes


def read_transcripts(
    path: Path = DEFAULT_GENCODE,
    *,
    chroms: frozenset[str] | set[str] | None = None,
    tx_types: frozenset[str] | set[str] = TSS_TES_TX_TYPES,
    annotations_out: dict[str, dict[str, object]] | None = None,
) -> list[Gene]:
    """Parse GENCODE ``transcript`` rows and merge them per gene.

    One pass over the GTF. Every ``transcript`` record whose ``transcript_type``
    is in *tx_types* and whose chromosome is in *chroms* contributes its span to
    its ``gene_id``; the returned :class:`Gene` for that id spans
    ``min(start)..max(end)`` over those contributions — the gene's TSS->TES
    extent *as annotated by the eligible transcripts* — and lists those
    transcripts' ids in :attr:`Gene.transcript_ids`.

    A gene with no eligible transcript never appears, which is the intended
    behaviour and not the same as filtering GTF ``gene`` rows on ``gene_type``:
    357 ``protein_coding`` genes are annotated only through ``retained_intron``
    / ``nonsense_mediated_decay`` / ``protein_coding_CDS_not_defined``
    transcripts and so are absent from the division.

    *annotations_out*, when given, is filled with the richer per-gene fields
    that only the ``.genes.tsv`` sidecar of the annotated BED needs
    (``build_division.py --annotated-out``): ``gene_id -> {"transcript_names":
    [...], "transcript_types": [...], "hgnc_id": str, "level": str}``, with the
    transcript lists parallel to :attr:`Gene.transcript_ids`. Nothing else
    reads it, so the three extra regexes per row only run when asked for.

    GTF is 1-based inclusive; the returned spans are 0-based half-open. Genes are
    returned in first-seen order, which the builder re-sorts per chromosome.
    """
    if chroms is None:
        chroms = set(DEFAULT_CHROMS)
    merged: dict[str, Gene] = {}
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            chrom = fields[0]
            if chrom not in chroms:
                continue
            attrs = fields[8]
            tt_match = _TX_TYPE_RE.search(attrs)
            if tt_match is None or tt_match.group(1) not in tx_types:
                continue
            id_match = _GENE_ID_RE.search(attrs)
            if id_match is None:
                continue
            gene_id = id_match.group(1)
            start, end = int(fields[3]) - 1, int(fields[4])
            tx_match = _TX_ID_RE.search(attrs)
            tx_id = tx_match.group(1) if tx_match else ""

            existing = merged.get(gene_id)
            if existing is None:
                gt_match = _GENE_TYPE_RE.search(attrs)
                name_match = _GENE_NAME_RE.search(attrs)
                merged[gene_id] = Gene(
                    chrom=chrom,
                    start=start,
                    end=end,
                    name=name_match.group(1) if name_match else gene_id,
                    gene_type=gt_match.group(1) if gt_match else tt_match.group(1),
                    strand=fields[6],
                    gene_id=gene_id,
                    transcript_ids=(tx_id,),
                )
            else:
                merged[gene_id] = replace(
                    existing,
                    start=min(existing.start, start),
                    end=max(existing.end, end),
                    transcript_ids=existing.transcript_ids + (tx_id,),
                )

            if annotations_out is not None:
                ann = annotations_out.get(gene_id)
                if ann is None:
                    hgnc_match = _HGNC_ID_RE.search(attrs)
                    level_match = _LEVEL_RE.search(attrs)
                    ann = annotations_out[gene_id] = {
                        "transcript_names": [],
                        "transcript_types": [],
                        "hgnc_id": hgnc_match.group(1) if hgnc_match else "",
                        "level": level_match.group(1) if level_match else "",
                    }
                name_match = _TX_NAME_RE.search(attrs)
                ann["transcript_names"].append(name_match.group(1) if name_match else "")
                ann["transcript_types"].append(tt_match.group(1))

    if not merged:
        raise ValueError(
            f"no transcripts parsed from {path} (chroms/tx_types too strict?)"
        )
    return list(merged.values())


def read_ucsc_gaps(path: Path = DEFAULT_GAPS) -> dict[str, list[Span]]:
    """Parse UCSC ``gap.txt.gz`` into merged per-chromosome spans.

    Columns are ``bin, chrom, chromStart, chromEnd, ix, n, size, type, bridge``
    — i.e. the coordinates live in fields 1..3, not 0..2 as in a plain BED.
    """
    by_chrom: dict[str, list[Span]] = {}
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            by_chrom.setdefault(parts[1], []).append((int(parts[2]), int(parts[3])))
    return {c: merge_spans(v) for c, v in by_chrom.items()}


def read_bed(path: Path) -> dict[str, list[Span]]:
    """Parse a plain BED into merged per-chromosome spans (cols 1-3 only)."""
    by_chrom: dict[str, list[Span]] = {}
    for _lineno, _line, parts in bedio.iter_bed_fields(
            path, split="tab-then-ws", compressed=path.suffix == ".gz"):
        if len(parts) < 3:
            continue
        by_chrom.setdefault(parts[0], []).append((int(parts[1]), int(parts[2])))
    return {c: merge_spans(v) for c, v in by_chrom.items()}


def load_excluded(
    *,
    gaps: Path | None = DEFAULT_GAPS,
    blacklist: Path | None = DEFAULT_BLACKLIST,
) -> dict[str, list[Span]]:
    """Union of assembly gaps and the ENCODE blacklist, per chromosome.

    These are the stretches no interval of either division may cover: telomeres,
    centromeres and contig gaps (megabases of ``N``), plus low-mappability and
    high-signal regions. Either source can be disabled by passing ``None``.
    """
    combined: dict[str, list[Span]] = {}
    for path in (gaps, blacklist):
        if path is None:
            continue
        source = read_ucsc_gaps(path) if path == gaps else read_bed(path)
        for chrom, spans in source.items():
            combined.setdefault(chrom, []).extend(spans)
    return {c: merge_spans(v) for c, v in combined.items()}


# ---------------------------------------------------------------------------
# Interval algebra (single chromosome, sorted half-open spans)
# ---------------------------------------------------------------------------


def merge_spans(spans: list[Span]) -> list[Span]:
    """Sort and union overlapping/abutting spans into disjoint spans."""
    if not spans:
        return []
    ordered = sorted(spans)
    out: list[Span] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = out[-1]
        if start <= last_end:
            out[-1] = (last_start, max(last_end, end))
        else:
            out.append((start, end))
    return out


def subtract_spans(spans: list[Span], holes: list[Span]) -> list[Span]:
    """Remove *holes* from *spans*, splitting a span that a hole punches through.

    Both inputs must be sorted and disjoint (as :func:`merge_spans` returns).
    The sweep advances a single hole cursor, so this is linear rather than
    quadratic — it runs over every base of the genome in the TAD mode.
    """
    if not holes:
        return list(spans)
    out: list[Span] = []
    hole_idx = 0
    for start, end in spans:
        cursor = start
        # Skip holes entirely left of this span.
        while hole_idx < len(holes) and holes[hole_idx][1] <= start:
            hole_idx += 1
        probe = hole_idx
        while probe < len(holes) and holes[probe][0] < end:
            hole_start, hole_end = holes[probe]
            if hole_start > cursor:
                out.append((cursor, hole_start))
            cursor = max(cursor, hole_end)
            if cursor >= end:
                break
            probe += 1
        if cursor < end:
            out.append((cursor, end))
    return out


def intersect_spans(spans: list[Span], window: Span) -> list[Span]:
    """Clip sorted disjoint *spans* to ``window``, dropping empties."""
    lo, hi = window
    out: list[Span] = []
    for start, end in spans:
        if end <= lo:
            continue
        if start >= hi:
            break
        out.append((max(start, lo), min(end, hi)))
    return out


def complement_spans(spans: list[Span], window: Span) -> list[Span]:
    """The parts of ``window`` not covered by sorted disjoint *spans*."""
    lo, hi = window
    out: list[Span] = []
    cursor = lo
    for start, end in intersect_spans(spans, window):
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < hi:
        out.append((cursor, hi))
    return out


