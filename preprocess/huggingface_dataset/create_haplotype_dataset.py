#!/usr/bin/env python3
"""v10 — per-patient **phased haplotype-pair** dataset (``generate`` / ``finalize-ref``).

Every earlier dataset version (v4–v8, ``-ldb``, ``-tsstes``) collapses a patient's
two haplotypes into one diploid (or ALT-wins haploid) string per ``(sample ×
region)`` and then **dedups + shuffles** at assemble time, so ``sample_id`` is
lossy and a genome cannot be joined back to a patient's EHR record. v10 is the
experimental, patient-preserving format:

* two haploid ``ACGTN^{}`` strings per region — ``hap0`` (left GT allele) and
  ``hap1`` (right) — straight from the BCF phasing, via
  :meth:`GenomeSequenceProcessor.get_haplotype_pair_parallel_str`;
* ``sample_id``, the BED label (``region_id``, column 4) and ``strand`` (column
  6, **recorded, not applied**) on every row, plus one ``region_<col>`` column
  per header-named BED column beyond the six BED fields (the annotation-transfer
  guarantee: everything that leaves the GTF reaches every row);
* **no dedup, no shuffle, no assemble** — one compressed Parquet file per patient
  (zstd by default), optionally a position-synced TSV (``--format tsv|both``);
* an ``HG38_REF`` pseudo-sample per split whose ``sample_weight`` is the number of
  samples in that split with no row for the region, written by ``finalize-ref``
  once every patient shard of the split exists;
* all-null placeholder columns for phyloP/CADD, laid out per haplotype.

**Breaking change vs. v4–v8 (marker grammar).** Every earlier version renders a
*homozygous* deletion by silently dropping the deleted bases — no ``{}`` at all —
because the gap cells never open an INDEL span; a het deletion is visible only
through the diploid ``X/-`` symbol or the haploid ALT-wins collapse. In v10 the
carrying haplotype always shows ``{}`` (or ``{…}`` for a partial span), ``^`` is
placed *per haplotype* (only before a 1-bp substitution on that haplotype), and
the non-carrier of a het indel shows plain reference bases with no braces. So
**marker counts are not comparable** between v10 and earlier haploid output, and
``gather_dataset_stats.py``-style INDEL tallies on v10 will be higher for the
same genomes. Inherited grammar: the indel anchor base stays outside the braces
(MNV ``REF=AC ALT=GT`` renders ``G{T}`` with an unmarked anchor). An unphased
heterozygote (``0/1``) is a **hard error** for that sample (FAILED, exit 1).

Usage (data node)::

    python -m preprocess.huggingface_dataset.create_haplotype_dataset generate \\
        --bcf-dir <per-sample BCFs> --bed hg38_tss_tes_v10.bed --ref hg38.2bit \\
        --output <out> [--test-samples-file ...] [--format parquet|tsv|both]
    python -m preprocess.huggingface_dataset.create_haplotype_dataset finalize-ref \\
        --output <out> --bed hg38_tss_tes_v10.bed --ref hg38.2bit

``run_v10_dataset.sh`` chains the BED build, the BCF prep and these two
subcommands. See ``README.md`` §"v10 — per-patient haplotype pairs".
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import multiprocessing
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent

from preprocess import paths  # noqa: E402
from preprocess.encoder.get_genome_sequence import (  # noqa: E402
    INTEGRITY_COUNTER_KEYS,
    GenomeSequenceProcessor,
    UnphasedHeterozygoteError,
)
from preprocess.huggingface_dataset import versions  # noqa: E402
from preprocess.huggingface_dataset.provenance import (  # noqa: E402
    GENERATION_PROVENANCE_FILENAME,
    append_run_record,
    check_or_write_generation_provenance,
    load_json_strict,
)
from preprocess.huggingface_dataset.create_dataset import (  # noqa: E402
    _CHROM_PAT,
    _MAX_N_FRACTION,
    _REF_SAMPLE_ID,
    _STATUS_CLAIMED,
    _STATUS_DONE,
    _STATUS_FAILED,
    _STATUS_SKIPPED,
    _ClaimHeartbeat,
    _claimed_samples,
    _file_sha256,
    _is_persample_bcf_dir,
    _load_sample_names,
    _missing_samples,
    _n_fraction,
    _tally_worker_results,
    discover_samples,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = versions.HAPLOTYPE_PAIR_SCHEMA_VERSION          # "v10"
REPRESENTATION = versions.HAPLOTYPE_PAIR_REPRESENTATION          # "haplotype_pair"
V10_REPRESENTATION = REPRESENTATION

#: Parquet key-value metadata key carrying the JSON provenance of each shard.
METADATA_KEY = b"v10"

#: Default tokens file (the encoder's 256×256 diploid lookup still runs inside
#: ``_prepare_haplotype_data``; its result is discarded, but the file is required).
DEFAULT_TOKENS = paths.DIPLOID_TOKENS

#: BED strand values accepted (recorded verbatim, never applied).
VALID_STRANDS = frozenset({"+", "-", "."})

#: The six BED fields the annotated-BED header must start with.
BED6_COLUMNS = ("chrom", "start", "end", "name", "score", "strand")

#: Prefix of the Parquet columns carrying header-named extra BED columns.
REGION_COL_PREFIX = "region_"

#: Separator used when BED rows with identical ``(chrom, start, end)`` are
#: merged into one region: their names (and every extra column) are joined
#: with it, so ``region_id == "ENSG…1;ENSG…2"`` says "two genes, one row".
#: The real tss-tes BED has 11 such pairs (e.g. HTD2/RPP14); without the
#: merge a patient gets two rows with the same coordinates and the HG38_REF
#: weight count, which is keyed by coordinates, cannot tell them apart.
DUPLICATE_COORD_SEP = ";"

#: Name of the per-split file listing every sample that belongs to the split
#: (union across resumed runs). ``finalize-ref`` counts exactly these.
SPLIT_SAMPLES_FILENAME = "split_samples.txt"

#: Copied next to the output root from ``<bed stem>.genes.tsv`` when present.
REGION_ANNOTATIONS_FILENAME = "region_annotations.tsv"

FORMATS = ("parquet", "tsv", "both")
TSV_MODES = ("variants", "full")
COMPRESSIONS = ("zstd", "lz4", "snappy", "none")

MARKER_GRAMMAR_NOTE = (
    "reference orientation; strand informational (recorded, not applied) — a "
    "consumer that reverse-complements must re-mark. ^ precedes a 1-bp "
    "substitution on THIS haplotype; {…} encloses INDEL content on THIS "
    "haplotype, a carried deletion is an empty span {} (v4–v8 dropped the bases "
    "silently); indel anchor base stays outside the braces (MNV AC>GT → G{T})."
)

#: TSV columns (``--format tsv|both``), in file order.
TSV_COLUMNS = (
    "sample_id", "region_id", "chrom", "strand", "pos", "offset", "ref",
    "hap0", "hap1", "snp0", "snp1", "indel0", "indel1",
)

# Streaming-writer flush thresholds: rows, or bytes of hap0+hap1 (a 2.47 Mbp
# tss-tes region is ~5 MB per row; LZ4's frame codec dies past ~2 GB).
_FLUSH_ROWS = 5000
_FLUSH_BYTES = 256 * 1024 * 1024

# Shared processor loaded once in the parent and inherited by forked workers.
_shared_processor: Optional[GenomeSequenceProcessor] = None

# Set in cmd_generate BEFORE the Pool forks (from the BED's extra columns).
_SCHEMA: Optional[pa.Schema] = None

# The effective region list, set in cmd_generate BEFORE the Pool forks so the
# workers inherit it instead of unpickling ~60k Region tuples per task.
_REGIONS: Optional[List["Region"]] = None

# Region index for finalize-ref's parallel row count (set before its Pool forks).
_COUNT_INDEX: Optional[Dict[Tuple[str, int, int], int]] = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_PLACEHOLDER_FIELDS = [
    pa.field("hap0_phylop", pa.list_(pa.uint8()), nullable=True),
    pa.field("hap1_phylop", pa.list_(pa.uint8()), nullable=True),
    pa.field("hap0_cadd_phred",
             pa.list_(pa.struct([pa.field("pos", pa.int32()),
                                 pa.field("score", pa.uint8())])), nullable=True),
    pa.field("hap1_cadd_phred",
             pa.list_(pa.struct([pa.field("pos", pa.int32()),
                                 pa.field("score", pa.uint8())])), nullable=True),
]
_PLACEHOLDER_NAMES = tuple(f.name for f in _PLACEHOLDER_FIELDS)


def build_schema(extra_cols: Sequence[str] = ()) -> pa.Schema:
    """The v10 Arrow schema, with one ``region_<col>`` column per extra BED column.

    Base columns (stable): ``sample_id, region_id, [region_<col>…], chrom,
    start, end, strand, hap0, hap1, n_variants, n_het, sample_weight`` + the
    four all-null phyloP/CADD placeholders. Extra columns are dictionary-encoded
    strings (``int32`` indices — a gene-type column has a handful of values, a
    gene-name column has one per region).
    """
    fields = [
        pa.field("sample_id", pa.string()),
        pa.field("region_id", pa.string()),
    ]
    for col in extra_cols:
        fields.append(pa.field(REGION_COL_PREFIX + col,
                               pa.dictionary(pa.int32(), pa.string())))
    fields += [
        pa.field("chrom", pa.dictionary(pa.int16(), pa.string())),
        pa.field("start", pa.int32()),
        pa.field("end", pa.int32()),
        pa.field("strand", pa.string()),
        pa.field("hap0", pa.large_utf8()),
        pa.field("hap1", pa.large_utf8()),
        pa.field("n_variants", pa.int32()),
        pa.field("n_het", pa.int32()),
        pa.field("sample_weight", pa.int32()),
    ]
    fields += _PLACEHOLDER_FIELDS
    return pa.schema(fields)


def region_columns_of(schema: pa.Schema) -> List[str]:
    """Extra BED column names recorded in a v10 schema (``region_<col>`` → ``col``)."""
    return [n[len(REGION_COL_PREFIX):] for n in schema.names
            if n.startswith(REGION_COL_PREFIX) and n != "region_id"]


def set_schema(extra_cols: Sequence[str]) -> pa.Schema:
    """Build and publish the module-level schema (call before the Pool forks)."""
    global _SCHEMA
    _SCHEMA = build_schema(extra_cols)
    return _SCHEMA


# ---------------------------------------------------------------------------
# BED6(+N) loading
# ---------------------------------------------------------------------------


class Region(NamedTuple):
    chrom: str
    start: int
    end: int
    name: str
    strand: str
    extras: Tuple[str, ...]

    @property
    def coords(self) -> Tuple[str, int, int]:
        return (self.chrom, self.start, self.end)


class BedTable(NamedTuple):
    regions: List[Region]
    extra_cols: List[str]
    region_set: Optional[str]
    #: Number of BED rows folded into an earlier row with the same coordinates
    #: (see :func:`merge_duplicate_coordinates`); 0 when every row is distinct.
    merged_duplicates: int = 0


def merge_duplicate_coordinates(regions: Sequence[Region]) -> Tuple[List[Region], int]:
    """Fold BED rows that share ``(chrom, start, end)`` into one region.

    The first row keeps its place; later rows with the same coordinates are
    merged into it by joining ``name`` and every extra column with
    :data:`DUPLICATE_COORD_SEP` (``"G1;G2"``, ``"snoRNA;snoRNA"``), so the
    merge is visible in every downstream row (``region_id`` contains the
    separator) and the weight count in :func:`count_region_rows`, which is
    keyed by coordinates, sees exactly one row per patient per coordinate.
    Strands must agree (a disagreement is a hard error: v10 records one
    strand per row). Returns ``(regions, n_merged)``.
    """
    index: Dict[Tuple[str, int, int], int] = {}
    out: List[Region] = []
    n_merged = 0
    for r in regions:
        i = index.get(r.coords)
        if i is None:
            index[r.coords] = len(out)
            out.append(r)
            continue
        first = out[i]
        if first.strand != r.strand:
            raise ValueError(
                f"BED rows {first.name!r} and {r.name!r} share coordinates "
                f"{r.chrom}:{r.start}-{r.end} but disagree on strand "
                f"({first.strand!r} vs {r.strand!r}); cannot merge them into one region")
        out[i] = Region(
            first.chrom, first.start, first.end,
            first.name + DUPLICATE_COORD_SEP + r.name, first.strand,
            tuple(a + DUPLICATE_COORD_SEP + b for a, b in zip(first.extras, r.extras)),
        )
        n_merged += 1
    return out, n_merged


def load_bed6_regions(path: Path) -> BedTable:
    """Load a BED6 (optionally headered BED6+N) for v10.

    * ``#region_set=<name>`` lines are recorded; a ``#chrom<TAB>start…`` line is
      the column header and must begin with the six BED names; other ``#``,
      ``track`` and ``browser`` lines are skipped.
    * Every data row needs ≥ 6 columns, a non-empty name (col 4) and a strand in
      ``{+,-,.}`` (col 6) — hard errors, since v10 records both.
    * With a header, every row must have exactly as many columns as the header,
      and columns beyond the six BED fields become ``extras`` (→ ``region_<col>``
      Parquet columns). Without a header, columns beyond six are ignored (warned
      once): there is no name to carry them under.
    * Rows with non-integer or invalid coordinates (``start<0``, ``end<=start``)
      are skipped with a warning, like ``create_dataset.load_bed_regions``.
    * Rows with identical ``(chrom, start, end)`` are merged into one region
      whose ``name``/extras are ``;``-joined (:func:`merge_duplicate_coordinates`);
      the count is returned as ``merged_duplicates`` and logged.
    """
    regions: List[Region] = []
    header: Optional[List[str]] = None
    region_set: Optional[str] = None
    skipped_bad = 0
    warned_unnamed_extras = False
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                if line.startswith("#region_set="):
                    region_set = line[len("#region_set="):].strip()
                elif line.startswith("#chrom\t"):
                    if regions:
                        raise ValueError(
                            f"{path}:{lineno}: column header after data rows")
                    header = line[1:].split("\t")
                    if tuple(header[:6]) != BED6_COLUMNS:
                        raise ValueError(
                            f"{path}:{lineno}: header must start with "
                            f"{list(BED6_COLUMNS)}, got {header[:6]}")
                    if len(set(header)) != len(header):
                        raise ValueError(f"{path}:{lineno}: duplicate header columns")
                continue
            if line.startswith(("track", "browser")):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                parts = line.split()
            if len(parts) < 6:
                raise ValueError(
                    f"{path}:{lineno}: v10 needs BED6 (chrom start end name "
                    f"score strand); got {len(parts)} field(s): {line!r}")
            if header is not None and len(parts) != len(header):
                raise ValueError(
                    f"{path}:{lineno}: {len(parts)} field(s) but the header "
                    f"names {len(header)}: {line!r}")
            try:
                start, end = int(parts[1]), int(parts[2])
            except ValueError:
                logger.warning("BED line %d has non-integer coords, skipping: %r",
                               lineno, line)
                skipped_bad += 1
                continue
            if start < 0 or end <= start:
                logger.warning("BED line %d has invalid coords, skipping: %r",
                               lineno, line)
                skipped_bad += 1
                continue
            name = parts[3].strip()
            if not name or name == ".":
                raise ValueError(
                    f"{path}:{lineno}: empty region name (col 4) — v10 records "
                    f"region_id on every row: {line!r}")
            strand = parts[5].strip()
            if strand not in VALID_STRANDS:
                raise ValueError(
                    f"{path}:{lineno}: strand (col 6) must be one of "
                    f"{sorted(VALID_STRANDS)}, got {strand!r}: {line!r}")
            if header is not None:
                extras = tuple(parts[6:])
            else:
                extras = ()
                if len(parts) > 6 and not warned_unnamed_extras:
                    logger.warning(
                        "%s has %d column(s) beyond BED6 but no '#chrom\\t…' "
                        "header naming them; they are ignored. Add a header to "
                        "carry them as region_<col> columns.",
                        path, len(parts) - 6)
                    warned_unnamed_extras = True
            regions.append(Region(parts[0], start, end, name, strand, extras))
    if skipped_bad:
        logger.warning("BED: skipped %d region(s) with invalid coords", skipped_bad)
    extra_cols = list(header[6:]) if header is not None else []
    regions, n_merged = merge_duplicate_coordinates(regions)
    if n_merged:
        logger.warning(
            "%s: merged %d row(s) into an earlier row with identical coordinates; "
            "their region_id/extras are joined with %r", path, n_merged, DUPLICATE_COORD_SEP)
    return BedTable(regions, extra_cols, region_set, n_merged)


def regions_fingerprint(regions: Sequence[Region]) -> str:
    """sha256 over the effective region list (coords + name + strand + extras)."""
    h = hashlib.sha256()
    for r in regions:
        h.update("\t".join([r.chrom, str(r.start), str(r.end), r.name, r.strand,
                            *r.extras]).encode())
        h.update(b"\n")
    return h.hexdigest()


def filter_regions_to_bcf_chromosomes(
    regions: List[Region], bcf_dir: Path
) -> List[Region]:
    """Drop regions on chromosomes with no BCF (same rule as create_dataset)."""
    if _is_persample_bcf_dir(bcf_dir):
        from cyvcf2 import VCF
        rep = sorted(bcf_dir.glob("*.bcf"))[0]
        with VCF(str(rep)) as v:
            available = set(v.seqnames)
    else:
        available = {m.group(1) for p in bcf_dir.glob("*.bcf")
                     if (m := _CHROM_PAT.search(p.name))}
    missing = {r.chrom for r in regions} - available
    if missing:
        counts: Dict[str, int] = defaultdict(int)
        for r in regions:
            if r.chrom in missing:
                counts[r.chrom] += 1
        for chrom in sorted(counts):
            logger.warning("No BCF for '%s' in %s — excluding %d region(s)",
                           chrom, bcf_dir, counts[chrom])
        regions = [r for r in regions if r.chrom not in missing]
    return regions


# ---------------------------------------------------------------------------
# Records + Parquet writer
# ---------------------------------------------------------------------------


def make_record(
    region: Region, sample_id: str, hap0: str, hap1: str,
    n_variants: int, n_het: int, sample_weight: int, extra_cols: Sequence[str],
) -> dict:
    rec = {
        "sample_id": sample_id,
        "region_id": region.name,
        "chrom": region.chrom,
        "start": region.start,
        "end": region.end,
        "strand": region.strand,
        "hap0": hap0,
        "hap1": hap1,
        "n_variants": n_variants,
        "n_het": n_het,
        "sample_weight": sample_weight,
    }
    for col, val in zip(extra_cols, region.extras):
        rec[REGION_COL_PREFIX + col] = val
    for name in _PLACEHOLDER_NAMES:
        rec[name] = None
    return rec


def hap_n_fraction(hap: str) -> float:
    """N fraction of one haplotype string, markers excluded from the denominator."""
    n_bases = len(hap) - hap.count("^") - hap.count("{") - hap.count("}")
    return _n_fraction(hap, "haploid", n_bases)


def _writer_kwargs(compression: str, compression_level: Optional[int],
                   schema: pa.Schema) -> dict:
    kw: dict = {
        "compression": compression,
        "use_dictionary": [n for n in schema.names
                           if pa.types.is_dictionary(schema.field(n).type)],
        "write_statistics": True,
    }
    if compression == "zstd" and compression_level is not None:
        kw["compression_level"] = compression_level
    return kw


def write_v10_parquet(
    records_iter: Iterable[dict],
    temp_path: Path,
    *,
    compression: str = "zstd",
    compression_level: Optional[int] = 3,
    metadata: Optional[dict] = None,
    schema: Optional[pa.Schema] = None,
    metadata_at_close: Optional[Callable[[], dict]] = None,
) -> int:
    """Stream records to Parquet; flush every 5000 rows or 256 MiB of hap bytes.

    Returns the row count. An empty input still writes a valid file with the
    full schema (and metadata) so a resume never trips over a 0-byte shard.

    The ``v10`` metadata is written once, into the file's key-value metadata
    at close (read it with ``pq.read_metadata(path).metadata[b"v10"]``;
    ``read_schema`` does not see it), so values only known after encoding
    can be part of it: ``metadata_at_close``, when given, is called after the
    last record was consumed and its dict is merged in (the integrity
    counters).
    """
    schema = schema if schema is not None else _SCHEMA
    if schema is None:
        raise RuntimeError("v10 schema not set — call set_schema() first")
    kwargs = _writer_kwargs(compression, compression_level, schema)
    n = 0
    batch: List[dict] = []
    nbytes = 0
    writer = pq.ParquetWriter(temp_path, schema, **kwargs)
    try:
        for rec in records_iter:
            batch.append(rec)
            nbytes += len(rec["hap0"]) + len(rec["hap1"])
            if len(batch) >= _FLUSH_ROWS or nbytes >= _FLUSH_BYTES:
                writer.write_batch(pa.RecordBatch.from_pylist(batch, schema=schema))
                n += len(batch)
                batch.clear()
                nbytes = 0
        if batch or n == 0:  # an empty batch still gives the file its schema
            writer.write_batch(pa.RecordBatch.from_pylist(batch, schema=schema))
            n += len(batch)
        meta = {**(metadata or {}), **(metadata_at_close() if metadata_at_close else {})}
        if meta:
            writer.add_key_value_metadata({METADATA_KEY.decode(): json.dumps(meta)})
    finally:
        writer.close()
    return n


def read_v10_metadata(path: Path) -> Optional[dict]:
    """The shard's ``v10`` JSON metadata, or ``None`` when the file has none."""
    raw = (pq.read_metadata(path).metadata or {}).get(METADATA_KEY)
    return None if raw is None else json.loads(raw)


# ---------------------------------------------------------------------------
# TSV writer (position-synced haplotypes)
# ---------------------------------------------------------------------------

_TSV_SCHEMA = pa.schema([
    pa.field("sample_id", pa.string()), pa.field("region_id", pa.string()),
    pa.field("chrom", pa.string()), pa.field("strand", pa.string()),
    pa.field("pos", pa.int32()), pa.field("offset", pa.int32()),
    pa.field("ref", pa.string()), pa.field("hap0", pa.string()),
    pa.field("hap1", pa.string()),
    pa.field("snp0", pa.int8()), pa.field("snp1", pa.int8()),
    pa.field("indel0", pa.int8()), pa.field("indel1", pa.int8()),
])


def _u8_to_str_array(a: np.ndarray) -> pa.Array:
    return pa.array(np.asarray(a, dtype=np.uint8).view("S1").astype("U1"), type=pa.string())


def table_to_arrow(sample_id: str, region: Region, table: Dict[str, np.ndarray]) -> pa.Table:
    """One region's encoder table → Arrow rows in :data:`TSV_COLUMNS` order."""
    n = len(table["pos"])
    return pa.table({
        "sample_id": pa.array([sample_id] * n, type=pa.string()),
        "region_id": pa.array([region.name] * n, type=pa.string()),
        "chrom": pa.array([region.chrom] * n, type=pa.string()),
        "strand": pa.array([region.strand] * n, type=pa.string()),
        "pos": pa.array(table["pos"], type=pa.int32()),
        "offset": pa.array(table["offset"], type=pa.int32()),
        "ref": _u8_to_str_array(table["ref"]),
        "hap0": _u8_to_str_array(table["hap0"]),
        "hap1": _u8_to_str_array(table["hap1"]),
        "snp0": pa.array(table["snp0"].astype(np.int8), type=pa.int8()),
        "snp1": pa.array(table["snp1"].astype(np.int8), type=pa.int8()),
        "indel0": pa.array(table["indel0"].astype(np.int8), type=pa.int8()),
        "indel1": pa.array(table["indel1"].astype(np.int8), type=pa.int8()),
    }, schema=_TSV_SCHEMA)


class TsvWriter:
    """Streaming gzip TSV writer (header written by hand, rows unquoted)."""

    def __init__(self, path: Path):
        self._stream = pa.CompressedOutputStream(str(path), "gzip")
        self._stream.write(("\t".join(TSV_COLUMNS) + "\n").encode())
        self._writer = pcsv.CSVWriter(
            self._stream, _TSV_SCHEMA,
            write_options=pcsv.WriteOptions(delimiter="\t", include_header=False,
                                            quoting_style="none"),
        )
        self.rows = 0

    def write(self, table: pa.Table) -> None:
        if table.num_rows:
            self._writer.write_table(table)
            self.rows += table.num_rows

    def close(self) -> None:
        self._writer.close()
        self._stream.close()


def rebuild_hap_from_table(rows: Iterable[dict], hap: int) -> str:
    """Reproduce one haplotype's marked string from a *full*-mode TSV's rows.

    The inverse direction the TSV is lossless in: concatenate non-``-`` cells
    of ``hap{hap}`` and apply ``^``/``{}`` from the ``snp{hap}``/``indel{hap}``
    flags with the builder's grammar (a gap opens/closes a span; an insertion
    row is INDEL content that opens a span after its anchor).
    """
    out: List[str] = []
    in_indel = False
    for r in rows:
        ch = r[f"hap{hap}"]
        is_indel = bool(int(r[f"indel{hap}"]))
        is_snp = bool(int(r[f"snp{hap}"]))
        if ch == "-":
            if in_indel and not is_indel:
                out.append("}")
                in_indel = False
            elif not in_indel and is_indel:
                out.append("{")
                in_indel = True
            continue
        if in_indel and not is_indel:
            out.append("}")
            in_indel = False
        if not in_indel and is_indel:
            out.append("{")
            in_indel = True
        if is_snp:
            out.append("^")
        out.append(ch)
    if in_indel:
        out.append("}")
    return "".join(out)


# ---------------------------------------------------------------------------
# Per-sample shard writer
# ---------------------------------------------------------------------------


def _outputs_for(sample_name: str, output_dir: Path, formats: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if "parquet" in formats:
        out["parquet"] = output_dir / f"{sample_name}.parquet"
    if "tsv" in formats:
        out["tsv"] = output_dir / f"{sample_name}.tsv.gz"
    return out


def expand_formats(fmt: str) -> Tuple[str, ...]:
    if fmt == "both":
        return ("parquet", "tsv")
    if fmt in ("parquet", "tsv"):
        return (fmt,)
    raise ValueError(f"--format must be one of {FORMATS}, got {fmt!r}")


def _group_by_chrom(regions: Sequence[Region]) -> Dict[str, List[Region]]:
    groups: Dict[str, List[Region]] = defaultdict(list)
    for r in regions:
        groups[r.chrom].append(r)
    return groups


def write_haplotype_shard(
    sample_name: str,
    bcf_dir: Path,
    regions: Sequence[Region],
    output_dir: Path,
    *,
    extra_cols: Sequence[str],
    compression: str = "zstd",
    compression_level: Optional[int] = 3,
    metadata: Optional[dict] = None,
    region_batch_size: int = 200,
    formats: Sequence[str] = ("parquet",),
    tsv_mode: str = "variants",
    ref_path: Optional[Path] = None,
    tokens_path: Optional[Path] = None,
) -> Tuple[str, Dict[str, int]]:
    """Write one patient's shard(s): ``<sample>.parquet`` and/or ``<sample>.tsv.gz``.

    Mirrors ``create_dataset.write_sample_shard``: resume check → ``O_CREAT|O_EXCL``
    claim (``.parquet.tmp``, or ``.tsv.gz.tmp`` when only TSV is requested) →
    heartbeat → encode **per chromosome batch** of ``region_batch_size`` regions
    (each call is one ``VCF()`` open + one CSI query; peak memory is one batch,
    not the whole BED) → rows streamed → atomic rename (Parquet first, then
    TSV). "Already present" means *every* requested output exists.

    Returns ``(status, integrity)``: the status string and the encoder's
    integrity counters for this sample (``INTEGRITY_COUNTER_KEYS``, every key
    present), which are also written into the shard's ``v10`` metadata under
    ``"integrity"`` and printed on the ``[DONE]`` line.

    On any exception — including :class:`UnphasedHeterozygoteError` — the
    sample, chromosome and region span are printed, every partial ``.tmp`` is
    unlinked and ``_STATUS_FAILED`` is returned (never a silent skip).
    """
    outputs = _outputs_for(sample_name, output_dir, formats)
    if not outputs:
        raise ValueError("formats must include 'parquet' and/or 'tsv'")
    integrity = {k: 0 for k in INTEGRITY_COUNTER_KEYS}
    if all(p.exists() for p in outputs.values()):
        logger.info("[SKIP] %s — output already exists", sample_name)
        return _STATUS_SKIPPED, integrity

    claim_path = (outputs["parquet"] if "parquet" in outputs else outputs["tsv"])
    claim_tmp = claim_path.with_name(claim_path.name + ".tmp")
    tmps = {k: p.with_name(p.name + ".tmp") for k, p in outputs.items()}
    try:
        claim_fd = os.open(str(claim_tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        logger.info("[SKIP] %s — .tmp exists (in progress or stale)", sample_name)
        return _STATUS_CLAIMED, integrity

    heartbeat = _ClaimHeartbeat(claim_fd, claim_tmp)
    heartbeat.start()
    t0 = time.perf_counter()
    processor = _shared_processor
    owns_processor = False
    tsv_writer: Optional[TsvWriter] = None
    schema = _SCHEMA if _SCHEMA is not None else build_schema(extra_cols)
    current = {"chrom": "?", "span": "?"}

    def _cleanup_tmps() -> None:
        for p in tmps.values():
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    try:
        if processor is None:
            if ref_path is None:
                raise RuntimeError("no shared processor and no ref_path given")
            print(f"  [{sample_name}] Loading reference genome (no shared processor)…",
                  flush=True)
            processor = GenomeSequenceProcessor(
                ref_path=ref_path, diploid_tokens_path=tokens_path or DEFAULT_TOKENS,
                force_unphased=False, use_direction_markers=False,
            )
            owns_processor = True
        processor.integrity.clear()  # per-sample totals (the processor is reused)

        def _integrity() -> Dict[str, int]:
            return {k: int(processor.integrity.get(k, 0)) for k in INTEGRITY_COUNTER_KEYS}

        drop = {"variant_free": 0, "high_n": 0}
        table_mode = tsv_mode if "tsv" in outputs else None
        if "tsv" in outputs:
            tsv_writer = TsvWriter(tmps["tsv"])

        def _records() -> Iterator[dict]:
            for chrom, chrom_regions in _group_by_chrom(regions).items():
                for i in range(0, len(chrom_regions), region_batch_size):
                    batch = chrom_regions[i:i + region_batch_size]
                    current["chrom"] = chrom
                    current["span"] = f"{batch[0].start}-{batch[-1].end}"
                    results = processor.get_haplotype_pair_parallel_str(
                        [r.coords for r in batch],
                        bcf_dir=bcf_dir, sample_name=sample_name,
                        max_workers=1, region_batch_size=None,
                        table_mode=table_mode,
                    )
                    for region, res in zip(batch, results):
                        if res is None:
                            drop["variant_free"] += 1
                            continue
                        hap0, hap1, n_variants, n_het = res[:4]
                        if (hap_n_fraction(hap0) > _MAX_N_FRACTION
                                or hap_n_fraction(hap1) > _MAX_N_FRACTION):
                            drop["high_n"] += 1
                            continue
                        if tsv_writer is not None:
                            tsv_writer.write(table_to_arrow(sample_name, region, res[4]))
                        yield make_record(region, sample_name, hap0, hap1,
                                          n_variants, n_het, 1, extra_cols)

        print(f"  [{sample_name}] [{time.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"{len(regions)} regions, batches of {region_batch_size} per chromosome, "
              f"formats={','.join(outputs)}…", flush=True)

        if "parquet" in outputs:
            n_rows = write_v10_parquet(
                _records(), tmps["parquet"], compression=compression,
                compression_level=compression_level,
                metadata={**(metadata or {}), "sample_id": sample_name},
                schema=schema,
                metadata_at_close=lambda: {"integrity": _integrity()},
            )
        else:
            n_rows = sum(1 for _ in _records())
        if tsv_writer is not None:
            tsv_writer.close()
            tsv_writer = None
        integrity = _integrity()

        if owns_processor:
            processor.close()
            processor = None

        # Atomic publication: Parquet first, then TSV. A crash between the two
        # leaves a Parquet without its TSV, which the resume check ("every
        # requested output exists") treats as not done and rewrites both.
        if "parquet" in outputs:
            tmps["parquet"].rename(outputs["parquet"])
        if "tsv" in outputs:
            tmps["tsv"].rename(outputs["tsv"])

        elapsed = time.perf_counter() - t0
        mb = sum(p.stat().st_size for p in outputs.values()) / (1024 * 1024)
        integ = ", ".join(f"{k}={v}" for k, v in integrity.items() if v)
        print(f"[DONE] {sample_name} — {n_rows} rows, dropped(variant_free="
              f"{drop['variant_free']}, high_n={drop['high_n']}), "
              f"integrity({integ or 'clean'}), {mb:.1f} MB, {elapsed:.1f}s", flush=True)
        return _STATUS_DONE, integrity

    except UnphasedHeterozygoteError as exc:
        print(f"[FAIL] {sample_name} — unphased heterozygote in "
              f"{current['chrom']}:{current['span']}: {exc}", flush=True)
        logger.error("[FAIL] %s — %s", sample_name, exc)
        _cleanup_tmps()
        return _STATUS_FAILED, integrity
    except Exception as exc:
        print(f"[FAIL] {sample_name} after {time.perf_counter() - t0:.1f}s at "
              f"{current['chrom']}:{current['span']}: {exc!r}", flush=True)
        logger.exception("[FAIL] %s", sample_name)
        _cleanup_tmps()
        return _STATUS_FAILED, integrity
    finally:
        if tsv_writer is not None:
            try:
                tsv_writer.close()
            except Exception:
                pass
        if owns_processor and processor is not None:
            processor.close()
        heartbeat.stop()
        try:
            os.close(claim_fd)
        except OSError:
            pass


def _worker_fn(args: tuple) -> Tuple[str, str, Dict[str, int]]:
    (sample_name, bcf_dir, output_dir, extra_cols, compression,
     compression_level, metadata, region_batch_size, formats, tsv_mode,
     ref_path, tokens_path) = args
    if _REGIONS is None:
        raise RuntimeError("_REGIONS not set — cmd_generate publishes it before the Pool forks")
    status, integrity = write_haplotype_shard(
        sample_name, Path(bcf_dir), _REGIONS, Path(output_dir),
        extra_cols=extra_cols, compression=compression,
        compression_level=compression_level, metadata=metadata,
        region_batch_size=region_batch_size, formats=formats, tsv_mode=tsv_mode,
        ref_path=Path(ref_path) if ref_path else None,
        tokens_path=Path(tokens_path) if tokens_path else None,
    )
    return sample_name, status, integrity


def _drain_pool(pool: Pool, worker_args: Sequence[tuple], timeout_s: float) -> List[tuple]:
    """Run ``_worker_fn`` over *worker_args* and collect what comes back.

    ``apply_async`` + a bounded ``get`` per task instead of
    ``list(pool.imap_unordered(...))``: a task whose worker died (OOM kill,
    segfault) is never re-queued by ``multiprocessing.Pool`` and the latter
    never returns. Here the wait for any one task
    is capped at *timeout_s*; a task that does not report by then is printed
    as ``[LOST]`` and left out, so ``_missing_samples`` marks it FAILED and
    the run ends instead of hanging.
    """
    pending = [pool.apply_async(_worker_fn, (wa,)) for wa in worker_args]
    results: List[tuple] = []
    for wa, ar in zip(worker_args, pending):
        try:
            results.append(ar.get(timeout=timeout_s))
        except multiprocessing.TimeoutError:
            print(f"[LOST] {wa[0]} — no result within {timeout_s / 3600:.1f} h; the worker "
                  f"probably died (OOM?). Counted as failed.", file=sys.stderr, flush=True)
    return results


# ---------------------------------------------------------------------------
# HG38_REF (finalize-ref)
# ---------------------------------------------------------------------------


def read_split_samples(split_dir: Path) -> List[str]:
    p = split_dir / SPLIT_SAMPLES_FILENAME
    if not p.is_file():
        return []
    return _load_sample_names(p)


def update_split_samples(split_dir: Path, names: Iterable[str]) -> List[str]:
    """Union *names* into the split's sample list (atomic rewrite). Returns it."""
    merged = sorted(set(read_split_samples(split_dir)) | set(names))
    p = split_dir / SPLIT_SAMPLES_FILENAME
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(f"{s}\n" for s in merged))
    tmp.replace(p)
    return merged


def _names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{s}\n" for s in names).encode()).hexdigest()


def pending_tmps(split_dir: Path) -> List[Path]:
    return sorted(list(split_dir.glob("*.parquet.tmp")) + list(split_dir.glob("*.tsv.gz.tmp")))


def _count_one_shard(shard: Path) -> np.ndarray:
    """Rows per region of one shard (uses the forked-in ``_COUNT_INDEX``)."""
    index = _COUNT_INDEX
    if index is None:
        raise RuntimeError("_COUNT_INDEX not set — count_region_rows publishes it first")
    counts = np.zeros(len(index), dtype=np.int64)
    t = pq.read_table(shard, columns=["chrom", "start", "end"])
    seen: set = set()
    for key in zip(t.column("chrom").to_pylist(), t.column("start").to_pylist(),
                   t.column("end").to_pylist()):
        i = index.get(key)
        if i is None:
            raise RuntimeError(
                f"{shard}: row for {key} is not in the BED region list — "
                f"the shard was generated from a different BED")
        if i in seen:
            raise RuntimeError(f"{shard}: duplicate row for region {key}")
        seen.add(i)
        counts[i] += 1
    return counts


def count_region_rows(
    split_dir: Path, samples: Sequence[str], regions: Sequence[Region],
    workers: int = 16,
) -> np.ndarray:
    """Rows per region across the split's patient shards.

    Three small columns per shard, read by up to *workers* forked processes
    (0.15–0.5 s per shard on NFS: 11,586 shards take 1–2 h serially).
    """
    global _COUNT_INDEX
    _COUNT_INDEX = {r.coords: i for i, r in enumerate(regions)}
    shards = [split_dir / f"{s}.parquet" for s in samples]
    counts = np.zeros(len(regions), dtype=np.int64)
    n = max(1, min(workers, len(shards)))
    if n == 1:
        for shard in shards:
            counts += _count_one_shard(shard)
        return counts
    with Pool(processes=n) as pool:
        for part in pool.imap_unordered(_count_one_shard, shards, chunksize=8):
            counts += part
    return counts


def write_reference_shard(
    split_dir: Path,
    regions: Sequence[Region],
    expected_samples: Sequence[str],
    processor: GenomeSequenceProcessor,
    *,
    extra_cols: Sequence[str],
    compression: str = "zstd",
    compression_level: Optional[int] = 3,
    metadata: Optional[dict] = None,
    seqs: Optional[List[str]] = None,
) -> dict:
    """Write ``<split>/HG38_REF.parquet`` with per-region weights.

    ``sample_weight`` = number of *expected* samples with no row for the
    region (variant-free **or** skipped for >10% N), so ``sum(sample_weight)``
    over a region across the split always equals the split's sample count.
    Regions every patient has are still written with weight 0. Refuses while
    any ``.tmp`` exists in the split (a run is in flight) or while any expected
    shard is missing (the weights would be wrong). Overwrites a previous
    reference shard — it is derived data. *seqs*, the regions' reference
    strings, may be passed in so several splits share one extraction.
    """
    out = split_dir / f"{_REF_SAMPLE_ID}.parquet"
    tmp = out.with_name(out.name + ".tmp")
    # A .tmp of our own is debris from a killed finalize-ref, not a claim.
    tmps = [p for p in pending_tmps(split_dir) if p != tmp]
    if tmps:
        raise RuntimeError(
            f"{split_dir}: {len(tmps)} in-flight .tmp file(s) "
            f"({', '.join(p.name for p in tmps[:5])}…) — a generate is running "
            f"or left debris; finish it (or --clean-stale) before finalize-ref")
    if tmp.exists():
        tmp.unlink()
    missing = [s for s in expected_samples if not (split_dir / f"{s}.parquet").exists()]
    if missing:
        raise RuntimeError(
            f"{split_dir}: {len(missing)} expected sample shard(s) missing "
            f"({', '.join(missing[:10])}{'…' if len(missing) > 10 else ''}) — "
            f"re-run generate to fill them before finalize-ref")
    n_samples = len(expected_samples)
    t0 = time.perf_counter()
    counts = count_region_rows(split_dir, expected_samples, regions)
    if (counts > n_samples).any():
        raise RuntimeError("a region has more rows than samples — duplicate sample?")
    weights = n_samples - counts
    if seqs is None:
        seqs = processor.get_reference_haploid_sequences_str([r.coords for r in regions])
    schema = _SCHEMA if _SCHEMA is not None else build_schema(extra_cols)
    n_high_n = 0

    def _records() -> Iterator[dict]:
        nonlocal n_high_n
        for region, seq, w in zip(regions, seqs, weights):
            if hap_n_fraction(seq) > _MAX_N_FRACTION:
                n_high_n += 1
                continue
            yield make_record(region, _REF_SAMPLE_ID, seq, seq, 0, 0, int(w), extra_cols)

    try:
        n_rows = write_v10_parquet(
            _records(), tmp, compression=compression, compression_level=compression_level,
            metadata={**(metadata or {}), "sample_id": _REF_SAMPLE_ID,
                      "n_samples_counted": n_samples},
            schema=schema,
        )
        tmp.replace(out)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    stats = {
        "stage": "finalize-ref",
        "n_samples_counted": n_samples,
        "samples_sha256": _names_sha256(list(expected_samples)),
        "region_count": len(regions),
        "rows_written": n_rows,
        "regions_skipped_high_n": n_high_n,
        "regions_weight_zero": int((weights == 0).sum()),
        "finalized_at": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    append_run_record(split_dir / GENERATION_PROVENANCE_FILENAME, stats)
    print(f"[DONE] {_REF_SAMPLE_ID} ({split_dir.name}) — {n_rows} rows over "
          f"{n_samples} samples, {n_high_n} high-N skipped, "
          f"{stats['regions_weight_zero']} regions at weight 0, "
          f"{stats['seconds']}s", flush=True)
    return stats


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class HaplotypePairDataset:
    """Reader for a v10 split dir (or one shard file).

    Verifies the sidecar / file metadata say ``v10`` + ``haplotype_pair`` and
    iterates rows as dicts. ``__len__`` comes from the Parquet footers (no row
    data read). For an output root with ``train/``/``test/`` pass ``split``.
    """

    def __init__(self, path, split: Optional[str] = None):
        path = Path(path)
        if path.is_file():
            self.files = [path]
            self.dir = path.parent
        else:
            if split is not None:
                path = path / split
            elif not (path / GENERATION_PROVENANCE_FILENAME).is_file() and (path / "train").is_dir():
                path = path / "train"
            self.dir = path
            sidecar = path / GENERATION_PROVENANCE_FILENAME
            if not sidecar.is_file():
                raise RuntimeError(f"{path}: no {GENERATION_PROVENANCE_FILENAME} — not a v10 split dir")
            prov = load_json_strict(sidecar)
            self._check(prov.get("schema_version"), prov.get("representation"), sidecar)
            self.files = sorted(path.glob("*.parquet"))
        self.schemas: List[pa.Schema] = []
        for f in self.files:
            info = read_v10_metadata(f)
            if info is None:
                raise RuntimeError(f"{f}: no {METADATA_KEY!r} Parquet metadata — not a v10 shard")
            self._check(info.get("schema_version"), info.get("representation"), f)
            self.schemas.append(pq.read_schema(f))
        self.schema = self.schemas[0] if self.schemas else None

    @staticmethod
    def _check(version, representation, where) -> None:
        if version != SCHEMA_VERSION or representation != REPRESENTATION:
            raise RuntimeError(
                f"{where}: schema_version={version!r} representation="
                f"{representation!r}; HaplotypePairDataset reads only "
                f"{SCHEMA_VERSION!r}/{REPRESENTATION!r} (use DiploidDataset / "
                f"HaploidDataset for v4–v8)")

    @property
    def region_columns(self) -> List[str]:
        return region_columns_of(self.schema) if self.schema is not None else []

    def __len__(self) -> int:
        return sum(pq.read_metadata(f).num_rows for f in self.files)

    def __iter__(self) -> Iterator[dict]:
        for f in self.files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches():
                yield from batch.to_pylist()

    def __repr__(self) -> str:
        return f"HaplotypePairDataset(dir={self.dir}, files={len(self.files)})"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_HERE, check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _split_dirs(output: Path) -> List[Path]:
    """``train/``+``test/`` when present, else the flat output dir."""
    if (output / "train").is_dir():
        dirs = [output / "train"]
        if (output / "test").is_dir():
            dirs.append(output / "test")
        return dirs
    return [output]


def _sidecar_set(sidecar: Path, key: str, value) -> None:
    doc = load_json_strict(sidecar) if sidecar.exists() else {}
    doc[key] = value
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(sidecar)


def _finalize_ref_command(args: argparse.Namespace) -> str:
    return ("python -m preprocess.huggingface_dataset.create_haplotype_dataset finalize-ref "
            f"--output {args.output} --bed {args.bed} --ref {args.ref}"
            + (f" --tokens {args.tokens}" if args.tokens else ""))


def _effective_regions(bed: Path, bcf_dir: Optional[Path], max_regions: Optional[int]) -> BedTable:
    table = load_bed6_regions(bed)
    regions = table.regions
    if bcf_dir is not None and bcf_dir.is_dir():
        regions = filter_regions_to_bcf_chromosomes(regions, bcf_dir)
    if max_regions is not None:
        regions = regions[:max_regions]
    return BedTable(regions, table.extra_cols, table.region_set, table.merged_duplicates)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    started = datetime.datetime.now()
    t_start = time.perf_counter()
    print(f"v10 generate started at {started:%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"Command: {' '.join(sys.argv)}", flush=True)

    for label, p, kind in (("--bcf-dir", args.bcf_dir, "dir"), ("--bed", args.bed, "file"),
                           ("--ref", args.ref, "file"), ("--tokens", args.tokens, "file")):
        ok = p.is_dir() if kind == "dir" else p.is_file()
        if not ok:
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)
    formats = expand_formats(args.format)

    # ── Samples / split ────────────────────────────────────────────────
    samples = (_load_sample_names(args.samples_file) if args.samples_file
               else discover_samples(args.bcf_dir))
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    test_set: set = set()
    if args.test_samples_file:
        test_list = _load_sample_names(args.test_samples_file)
        test_set = set(test_list)
        extra = [s for s in test_list if s not in set(samples)]
        samples = samples + extra
    if test_set:
        train_dir, test_dir = args.output / "train", args.output / "test"
        split_dirs = {"train": train_dir, "test": test_dir}
    else:
        split_dirs = {"train": args.output}
    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def _split_of(s: str) -> str:
        return "test" if s in test_set else "train"

    # ── Regions ────────────────────────────────────────────────────────
    bed_sha256 = _file_sha256(args.bed)
    try:
        bed = _effective_regions(args.bed, args.bcf_dir, None)
    except ValueError as exc:
        print(f"ERROR: malformed BED — {exc}", file=sys.stderr)
        sys.exit(1)
    regions = bed.regions
    if not regions:
        print(f"ERROR: no regions remain after the BCF-chromosome filter — none of the "
              f"BED chromosomes have a BCF in {args.bcf_dir}", file=sys.stderr)
        sys.exit(1)
    if args.max_regions is not None:
        regions = regions[:args.max_regions]
        if not regions:
            print(f"ERROR: --max-regions {args.max_regions} leaves no regions", file=sys.stderr)
            sys.exit(1)
    print(f"  {len(regions):,} regions, extra BED columns: {bed.extra_cols or 'none'}"
          f"{f', region_set={bed.region_set}' if bed.region_set else ''}"
          f"{f', {bed.merged_duplicates} coordinate-identical BED row(s) merged (region_id joined with {DUPLICATE_COORD_SEP!r})' if bed.merged_duplicates else ''}",
          flush=True)
    set_schema(bed.extra_cols)

    # ── genes.tsv sidecar → region_annotations.tsv ─────────────────────
    # The sha recorded in the provenance fingerprint is that of the SOURCE
    # file, so a resume with a different genes.tsv is refused by the guard
    # below; the copy itself happens only after the guard passed.
    genes_tsv = args.bed.with_name(args.bed.stem + ".genes.tsv")
    annotations_sha = _file_sha256(genes_tsv) if genes_tsv.is_file() else None

    # ── Pending / stale ────────────────────────────────────────────────
    def _done(s: str) -> bool:
        return all(p.exists() for p in _outputs_for(s, split_dirs[_split_of(s)], formats).values())
    pending = [s for s in samples if not _done(s)]
    print(f"  {len(samples)} samples total, {len(samples) - len(pending)} already done, "
          f"{len(pending)} pending", flush=True)
    if args.clean_stale:
        removed = _clean_stale(list(split_dirs.values()), args.stale_age_hours * 3600.0)
        if removed:
            print(f"  --clean-stale removed {removed} stale .tmp file(s)", flush=True)

    # ── Provenance guard per split dir ─────────────────────────────────
    inputs_fp = {
        "bed_sha256": bed_sha256,
        "bed_extra_columns": ",".join(bed.extra_cols),
        "region_set": bed.region_set,
        "ref": f"{args.ref.resolve()}|{args.ref.stat().st_size}",
        "bcf_dir": str(args.bcf_dir.resolve()),
        "tokens_sha256": _file_sha256(args.tokens),
        "samples_file_sha256": _file_sha256(args.samples_file),
        "test_samples_file_sha256": _file_sha256(args.test_samples_file),
        "max_samples": args.max_samples,
        "max_regions": args.max_regions,
        "formats": ",".join(formats),
        "tsv_mode": args.tsv_mode if "tsv" in formats else None,
        "region_annotations_sha256": annotations_sha,
    }
    is_resume = (split_dirs["train"] / GENERATION_PROVENANCE_FILENAME).exists()
    regions_sha = regions_fingerprint(regions)
    for d in split_dirs.values():
        check_or_write_generation_provenance(
            d, schema_version=SCHEMA_VERSION, representation=REPRESENTATION,
            force_unphased=False, use_direction_markers=False, marker_policy="n/a",
            annotations=[], inputs=inputs_fp,
        )
        # The effective region list must be identical across resumes: a shard
        # written by an earlier run is trusted as complete ("already present"),
        # and HG38_REF weights are counted from every shard, so a run that
        # covers a different region list (another --max-regions, a BCF dir
        # that now lacks a chromosome) would silently corrupt the weights.
        # `--max-samples` is deliberately NOT held fixed (split_samples.txt is a
        # union); `--max-regions` is. The provenance guard cannot do this: it
        # ignores a field that is None on one side.
        sidecar = d / GENERATION_PROVENANCE_FILENAME
        recorded = load_json_strict(sidecar).get("regions_effective")
        if recorded and recorded.get("sha256") != regions_sha:
            print(f"ERROR: {d}: the effective region list differs from the one the "
                  f"existing shards were generated with — recorded {recorded.get('count')} "
                  f"regions (max_regions={recorded.get('max_regions')}, sha "
                  f"{str(recorded.get('sha256'))[:12]}…), this run {len(regions)} "
                  f"(max_regions={args.max_regions}, sha {regions_sha[:12]}…). Shards "
                  f"already on disk would be trusted as complete and the HG38_REF weights "
                  f"would be wrong. Use the same --max-regions/--bcf-dir, or a different "
                  f"--output.", file=sys.stderr, flush=True)
            sys.exit(1)
        members = update_split_samples(d, [s for s in samples if split_dirs[_split_of(s)] == d])
        _sidecar_set(sidecar, "split_samples",
                     {"file": SPLIT_SAMPLES_FILENAME, "count": len(members),
                      "sha256": _names_sha256(members)})
        _sidecar_set(sidecar, "regions_effective",
                     {"count": len(regions), "sha256": regions_sha,
                      "max_regions": args.max_regions,
                      "merged_duplicate_coordinates": bed.merged_duplicates,
                      "duplicate_coordinate_sep": DUPLICATE_COORD_SEP})

    # ── genes.tsv → region_annotations.tsv (after the guard accepted it) ──
    if annotations_sha is not None:
        dst = args.output / REGION_ANNOTATIONS_FILENAME
        if not dst.exists() or _file_sha256(dst) != annotations_sha:
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copyfile(genes_tsv, tmp)
            tmp.replace(dst)
        print(f"  region annotations: {genes_tsv.name} → {dst}", flush=True)

    metadata = {
        "schema_version": SCHEMA_VERSION, "representation": REPRESENTATION,
        "bed_sha256": bed_sha256, "bed": str(args.bed), "region_set": bed.region_set,
        "bed_extra_columns": list(bed.extra_cols), "ref": str(args.ref),
        "annotations": [], "marker_grammar": MARKER_GRAMMAR_NOTE,
        "formats": list(formats), "tsv_mode": args.tsv_mode if "tsv" in formats else None,
        "merged_duplicate_coordinates": bed.merged_duplicates,
        "duplicate_coordinate_sep": DUPLICATE_COORD_SEP,
        "git_commit": _git_commit(),
    }
    run_params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                  if k != "func"}

    def _finalize_run(status: str, **milestones) -> None:
        rec = {
            "command": " ".join(sys.argv), "started_at": f"{started:%Y-%m-%d %H:%M:%S}",
            "finished_at": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "duration_seconds": round(time.perf_counter() - t_start, 1),
            "status": status, "resume": is_resume, "params": run_params,
            "milestones": {"n_regions": len(regions), "n_samples": len(samples),
                           "n_pending_at_start": len(pending), **milestones},
        }
        for d in split_dirs.values():
            append_run_record(d / GENERATION_PROVENANCE_FILENAME, rec)

    # ── Encode ─────────────────────────────────────────────────────────
    global _shared_processor
    n_done = n_skipped = 0
    failed: List[str] = []
    claimed: List[str] = []
    integrity_totals: Counter = Counter({k: 0 for k in INTEGRITY_COUNTER_KEYS})
    if pending:
        print(f"  loading reference {args.ref} …", flush=True)
        _shared_processor = GenomeSequenceProcessor(
            ref_path=args.ref, diploid_tokens_path=args.tokens,
            force_unphased=False, use_direction_markers=False,
        )
        global _REGIONS
        _REGIONS = regions
        worker_args = [(
            s, str(args.bcf_dir), str(split_dirs[_split_of(s)]),
            tuple(bed.extra_cols), args.compression, args.compression_level, metadata,
            args.region_batch_size, formats, args.tsv_mode, str(args.ref), str(args.tokens),
        ) for s in pending]
        n_workers = max(1, min(args.workers, len(pending)))
        t_enc = time.perf_counter()
        if n_workers == 1:
            results = [_worker_fn(wa) for wa in worker_args]
        else:
            with Pool(processes=n_workers) as pool:
                results = _drain_pool(pool, worker_args, args.sample_timeout_hours * 3600.0)
        statuses = [(name, st) for name, st, _ in results]
        n_done, n_skipped, failed = _tally_worker_results(statuses)
        failed = failed + _missing_samples(pending, statuses)
        claimed = _claimed_samples(statuses)
        for _, st, integ in results:
            if st == _STATUS_DONE:
                integrity_totals.update(integ)
        print(f"\nGeneration complete in {time.perf_counter() - t_enc:.0f}s — done: {n_done}, "
              f"skipped (already present): {n_skipped - len(claimed)}, "
              f"skipped (.tmp claimed elsewhere): {len(claimed)}, failed: {len(failed)}",
              flush=True)
        print("  integrity over done samples: "
              + ", ".join(f"{k}={integrity_totals[k]}" for k in INTEGRITY_COUNTER_KEYS),
              flush=True)
    else:
        print("All samples already present.", flush=True)

    if claimed:
        print(f"NOTE — {len(claimed)} sample(s) produced no shard because a .tmp claim "
              f"existed: {', '.join(claimed[:20])}", file=sys.stderr, flush=True)
    if failed:
        print(f"FAILED — {len(failed)} sample(s) wrote no shard: {', '.join(failed[:20])}"
              f"{' …' if len(failed) > 20 else ''}. Re-run to retry them.",
              file=sys.stderr, flush=True)

    # ── finalize-ref ───────────────────────────────────────────────────
    clean = not failed and not claimed
    ref_done = False
    if "parquet" not in formats:
        # HG38_REF is Parquet-only and its weights are counted from the patients'
        # Parquet shards; a TSV-only run has neither, by design (the TSV's
        # purpose is variant positions, and a reference table is just the BED).
        print("HG38_REF not written: --format tsv produces no Parquet shards to "
              "count weights from (re-run with --format parquet|both for a "
              "reference shard).", flush=True)
    elif clean and not args.no_finalize_ref:
        if _shared_processor is None:
            _shared_processor = GenomeSequenceProcessor(
                ref_path=args.ref, diploid_tokens_path=args.tokens,
                force_unphased=False, use_direction_markers=False,
            )
        try:
            ref_seqs = _shared_processor.get_reference_haploid_sequences_str(
                [r.coords for r in regions])  # once for every split
            for d in split_dirs.values():
                write_reference_shard(
                    d, regions, read_split_samples(d), _shared_processor,
                    extra_cols=bed.extra_cols, compression=args.compression,
                    compression_level=args.compression_level, metadata=metadata,
                    seqs=ref_seqs,
                )
            ref_done = True
        except Exception as exc:
            print(f"ERROR: finalize-ref failed: {exc}", file=sys.stderr, flush=True)
            failed = failed or [f"{_REF_SAMPLE_ID} (finalize-ref)"]
    if not ref_done and "parquet" in formats:
        print(f"HG38_REF not written{' (--no-finalize-ref)' if args.no_finalize_ref else ''}. "
              f"Once every sample shard exists, run:\n  {_finalize_ref_command(args)}",
              flush=True)

    _finalize_run("failed" if failed else "ok", n_done=n_done, n_skipped=n_skipped,
                  n_claimed=len(claimed), n_failed=len(failed), failed_samples=failed[:50],
                  claimed_samples=claimed[:50], ref_finalized=ref_done,
                  integrity={k: int(integrity_totals[k]) for k in INTEGRITY_COUNTER_KEYS})
    print(f"v10 generate finished at {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
          f"(total {time.perf_counter() - t_start:.1f}s)", flush=True)
    if failed:
        sys.exit(1)


def _clean_stale(dirs: List[Path], grace_s: float) -> int:
    """Remove ``*.parquet.tmp`` / ``*.tsv.gz.tmp`` untouched for ``grace_s``.

    Same rule as ``create_dataset._clean_stale_tmp`` (a recent ``.tmp`` is a
    live claim with a heartbeat; leave it), extended to the TSV claim suffix.
    """
    removed = kept = 0
    now = time.time()
    for d in dirs:
        for tmp in pending_tmps(d):
            try:
                age = now - tmp.stat().st_mtime
            except OSError:
                continue
            if age < grace_s:
                kept += 1
                continue
            try:
                tmp.unlink()
                removed += 1
            except OSError:
                pass
    if kept:
        print(f"  --clean-stale: kept {kept} recently-written .tmp claim(s)", flush=True)
    return removed


# ---------------------------------------------------------------------------
# finalize-ref
# ---------------------------------------------------------------------------


def cmd_finalize_ref(args: argparse.Namespace) -> None:
    if not args.output.is_dir():
        print(f"ERROR: --output not found: {args.output}", file=sys.stderr)
        sys.exit(1)
    bed_sha256 = _file_sha256(args.bed)
    dirs = _split_dirs(args.output)
    processor = None
    rc = 0
    for d in dirs:
        sidecar = d / GENERATION_PROVENANCE_FILENAME
        if not sidecar.is_file():
            print(f"ERROR: {d}: no {GENERATION_PROVENANCE_FILENAME}", file=sys.stderr)
            sys.exit(1)
        prov = load_json_strict(sidecar)
        if (prov.get("schema_version"), prov.get("representation")) != (SCHEMA_VERSION, REPRESENTATION):
            print(f"ERROR: {d} is not a v10 split dir", file=sys.stderr)
            sys.exit(1)
        inputs = prov.get("inputs", {})
        if inputs.get("bed_sha256") != bed_sha256:
            print(f"ERROR: {d}: --bed sha256 {bed_sha256} differs from the one generate "
                  f"recorded ({inputs.get('bed_sha256')})", file=sys.stderr)
            sys.exit(1)
        bcf_dir = Path(inputs["bcf_dir"]) if inputs.get("bcf_dir") else None
        bed = _effective_regions(args.bed, bcf_dir, inputs.get("max_regions"))
        eff = prov.get("regions_effective", {})
        if eff and eff.get("sha256") != regions_fingerprint(bed.regions):
            print(f"ERROR: {d}: re-derived region list ({len(bed.regions)}) does not match "
                  f"the one generate used ({eff.get('count')}); is --bcf-dir "
                  f"{bcf_dir} still reachable?", file=sys.stderr)
            sys.exit(1)
        set_schema(bed.extra_cols)
        samples = read_split_samples(d)
        if not samples:
            print(f"ERROR: {d}: no {SPLIT_SAMPLES_FILENAME} — generate never ran here",
                  file=sys.stderr)
            sys.exit(1)
        if args.exclude_samples_file:
            # A sample that will never get a shard (e.g. an unphased het) is
            # dropped from the split: the file stays the weights' denominator,
            # the sidecar records who was dropped.
            excluded = sorted(set(samples) & set(_load_sample_names(args.exclude_samples_file)))
            if excluded:
                samples = [s for s in samples if s not in excluded]
                p = d / SPLIT_SAMPLES_FILENAME
                tmp = p.with_name(p.name + ".tmp")
                tmp.write_text("".join(f"{s}\n" for s in samples))
                tmp.replace(p)
                prev = prov.get("split_samples", {})
                _sidecar_set(sidecar, "split_samples",
                             {"file": SPLIT_SAMPLES_FILENAME, "count": len(samples),
                              "sha256": _names_sha256(samples),
                              "excluded": sorted(set(prev.get("excluded", [])) | set(excluded))})
                print(f"  {d.name}: excluded {len(excluded)} sample(s) from the split: "
                      f"{', '.join(excluded[:10])}", flush=True)
        if processor is None:
            processor = GenomeSequenceProcessor(
                ref_path=args.ref, diploid_tokens_path=args.tokens,
                force_unphased=False, use_direction_markers=False,
            )
        metadata = {
            "schema_version": SCHEMA_VERSION, "representation": REPRESENTATION,
            "bed_sha256": bed_sha256, "bed": str(args.bed), "region_set": bed.region_set,
            "bed_extra_columns": list(bed.extra_cols), "ref": str(args.ref),
            "annotations": [], "marker_grammar": MARKER_GRAMMAR_NOTE,
            "merged_duplicate_coordinates": bed.merged_duplicates,
            "duplicate_coordinate_sep": DUPLICATE_COORD_SEP,
            "git_commit": _git_commit(),
        }
        try:
            write_reference_shard(
                d, bed.regions, samples, processor, extra_cols=bed.extra_cols,
                compression=args.compression, compression_level=args.compression_level,
                metadata=metadata,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr, flush=True)
            rc = 1
    if processor is not None:
        processor.close()
    if rc:
        sys.exit(rc)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp):
        sp.add_argument("--bed", type=Path, required=True,
                        help="BED6(+N) with name (col 4) and strand (col 6); a "
                             "'#chrom\\t…' header names extra columns → region_<col>")
        sp.add_argument("--ref", type=Path, required=True, help="reference (2bit/FASTA)")
        sp.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS,
                        help=f"diploid tokens TSV (default {DEFAULT_TOKENS})")
        sp.add_argument("--output", type=Path, required=True, help="output root")
        sp.add_argument("--compression", choices=COMPRESSIONS, default="zstd")
        sp.add_argument("--compression-level", type=int, default=3,
                        help="zstd level (ignored for other codecs)")
        sp.add_argument("--log-level", default="ERROR")

    g = sub.add_parser("generate", help="write one haplotype-pair shard per sample")
    g.add_argument("--bcf-dir", type=Path, required=True,
                   help="per-sample <sample>.bcf dir (or legacy per-chrom multi-sample)")
    _common(g)
    g.add_argument("--workers", type=int, default=min(60, os.cpu_count() or 1))
    g.add_argument("--samples-file", type=Path, default=None)
    g.add_argument("--test-samples-file", type=Path, default=None)
    g.add_argument("--max-samples", type=int, default=None)
    g.add_argument("--max-regions", type=int, default=None)
    g.add_argument("--region-batch-size", type=int, default=200,
                   help="regions per encoder call (one CSI query each)")
    g.add_argument("--clean-stale", action="store_true")
    g.add_argument("--stale-age-hours", type=float, default=2.0)
    g.add_argument("--sample-timeout-hours", type=float, default=3.0,
                   help="give up waiting for one sample's worker after this long "
                        "(a dead worker's task never returns; it is then counted as failed)")
    g.add_argument("--no-finalize-ref", action="store_true",
                   help="do not write HG38_REF even when the run is clean")
    g.add_argument("--format", choices=FORMATS, default="parquet")
    g.add_argument("--tsv-mode", choices=TSV_MODES, default="variants",
                   help="variants: rows where a hap differs from ref or is inserted; "
                        "full: every position (smoke tests only)")
    g.set_defaults(func=cmd_generate)

    f = sub.add_parser("finalize-ref", help="write HG38_REF.parquet per split with weights")
    _common(f)
    f.add_argument("--exclude-samples-file", type=Path, default=None,
                   help="samples to drop from split_samples.txt before counting "
                        "(e.g. a sample that fails on an unphased heterozygote)")
    f.set_defaults(func=cmd_finalize_ref)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.ERROR),
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
