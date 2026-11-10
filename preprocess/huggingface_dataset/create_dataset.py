#!/usr/bin/env python3
"""
create_dataset.py — Build a HuggingFace Parquet dataset from multi-sample BCF files.

For each sample in the per-chromosome BCF files under BCF_DIR, reads regions from
BED_FILE and creates one dataset record per (sample × region). Each record stores
the per-region sequence as a Unicode string with SNP/INDEL delimiter markers
(^, {, }) baked in.

The dataset version is selected with --version {v4,v5,v6,v6nd,v7,v8} — plus the
region-bound variants {v4-ldb,v6-ldb,v7-ldb} — and is REQUIRED (there is no
default). The version is a behavioral preset (resolved in versions.py — the
single source of truth for SCHEMA_VERSION), not a set of scattered flags:
  v8: v7 content, but the structural markers (^ { } ‹ ›) are moved OUT of `seq`
      into sparse snp/indel_span/direction columns (seq becomes pure base content);
      per-base annotation columns (phyloP/CADD) allowed.
  v7: inline anchor-SNP '^' + optional per-base annotation columns (phyloP/CADD).
  v6: inline anchor-SNP '^', no annotation columns.
  v6nd: v6 without the ‹/› direction markers and with phase dropped (force_unphased);
        seq equals v6's seq minus the ‹/› codepoints.
  v5: pre-v6 bare-anchor form (no compound-het anchor '^'), no annotation columns.
  v4: the original pre-direction-marker encoding — ordered (phased) diploid symbols
      with inline '^'/'{}' markers but NO ‹/› direction tokens, no annotations.
  v4-ldb / v6-ldb / v7-ldb: those same encodings over a different REGION UNIVERSE
      (the deCODE EUR LD blocks). They pin their BED's sha256, so a mismatched
      --bed is a hard error, and no conversion crosses region sets.
Most diploid presets emit the ‹/› three-level direction markers, but v4 (predates
them) and v6nd (drops them) do not — that axis is now owned by --version, so
--use-direction-markers is deprecated and ignored (see below).
--representation and --force-unphased are orthogonal ablation knobs, not part of a
version's lineage.

Two output representations are supported via --representation:
  diploid (default): 256x256-codepoint diploid string with ‹/› direction markers
                     (preset-owned; emitted for every diploid version except v4/v6nd).
                     Requires --tokens.
  haploid:           literal ACGTN string after per-cell ALT-wins-leftmost collapse.
                     No direction markers, no diploid lookup. --tokens is optional
                     (only the meta tokens <SNP>, <INDEL_span_start>, <INDEL_span_end>
                     are consulted).

Per-base annotation columns (v7/v8):
  --phylop-bw PATH   Add a phyloP-per-base column (list<uint8>, quantised; 255 = NaN)
                     aligned to each sequence.
  --cadd-arrow-dir D Add a sparse CADD-phred column (list<struct<pos:int32,
                     score:float32>>, one entry per variant anchor; requires full_arrow/
                     produced by poc_a2_convert_raw_cadd.py). Incompatible with
                     --representation haploid.
  These flags are optional; omitting both yields the six-column schema (same column
  layout as v4, written with the current SCHEMA_VERSION).

The representation and active annotations are recorded in `generation_provenance.json`
next to the shards and in `provenance.json` next to the assembled dataset; the readers
(DiploidDataset / HaploidDataset) refuse to load the wrong representation.

Records that contain no variants are skipped (except for the HG38_REF pseudo-sample,
which includes all BED regions as reference-only sequences).

Output is one Parquet shard per sample. Multiple workers can run in parallel, and
interrupted runs resume automatically (completed shards are skipped).

Usage (diploid — v7):
    python -m preprocess.huggingface_dataset.create_dataset generate \\
        --bcf-dir  /data/1kgenomes/raw_bcf/ \\
        --bed      hg38_pretrain_canonical_expanded.bed \\
        --ref      hg38.2bit \\
        --tokens   diploid_tokens.tsv \\
        --output   /data/diploid_hf_v7/ \\
        --version  v7 \\
        --workers  60 \\
        [--max-samples 100] \\
        [--genes-file genes.txt] \\
        [--samples-file samples.txt] \\
        [--test-samples-file test_samples.txt] \\
        [--compression-level 3]

Usage (diploid + annotations):
    python -m preprocess.huggingface_dataset.create_dataset generate \\
        --bcf-dir  /data/1kgenomes/raw_bcf/ \\
        --bed      hg38_pretrain_canonical_expanded.bed \\
        --ref      hg38.2bit \\
        --tokens   diploid_tokens.tsv \\
        --output   /data/diploid_hf_v7/ \\
        --version  v7 \\
        --workers  60 \\
        --phylop-bw /opt/hpc/data/PhylopP/hg38.phyloP100way.bw \\
        --cadd-arrow-dir /opt/hpc/data/CADD/full_arrow/

Usage (haploid):
    python -m preprocess.huggingface_dataset.create_dataset generate \\
        --representation haploid \\
        --bcf-dir  /data/1kgenomes/raw_bcf/ \\
        --bed      hg38_pretrain_canonical_expanded.bed \\
        --ref      hg38.2bit \\
        --output   /data/haploid_hf_v7/ \\
        --version  v7 \\
        --workers  60

Assemble shards into a HuggingFace dataset:
    python -m preprocess.huggingface_dataset.create_dataset assemble \\
        --shard-dir /data/diploid_hf_v7/train/ \\
        --output    /data/diploid_hf_v7_assembled/ \\
        [--test-shard-dir /data/diploid_hf_v7/test/]
"""

import argparse
import atexit
import datetime
import errno
import functools
import hashlib
import importlib
import json
import logging
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import warnings
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from preprocess.encoder.get_genome_sequence import GenomeSequenceProcessor, split_markers  # noqa: E402
from preprocess.encoder.perf_timer import PerfAccumulator  # noqa: E402
# Marker characters used to detect whether a region contains variants.
from preprocess.markers import INDEL_OPEN as _INDEL_OPEN, SNP_MARKER as _SNP_MARKER  # noqa: E402
from preprocess.regions import bedio  # noqa: E402

# Version presets resolve a required --version {v4,v5,v6,v6nd,v7,v8} into a behavioral
# DatasetVersion (there is no default — see versions.py). provenance.py owns the
# shared sidecar I/O. annotations.py owns the per-base annotation framework.
from preprocess.huggingface_dataset import versions  # noqa: E402
from preprocess.huggingface_dataset.versions import SUPPORTED_VERSIONS  # noqa: E402
# compat owns the cross-version "downhill" conversion matrix + per-shard
# transforms (v7→v6, v8→v6, v8→v7, v6→v6nd). Leaf module — no import cycle.
from preprocess.huggingface_dataset import compat  # noqa: E402
from preprocess.huggingface_dataset.provenance import (  # noqa: E402
    GENERATION_PROVENANCE_FILENAME as _GEN_PROVENANCE_FILENAME,
    append_run_record as _append_run_record,
    check_or_write_generation_provenance as _check_or_write_generation_provenance_impl,
    load_json_strict as _load_json_strict,
)
# dedup machinery. `import dedup` is needed for the mutable cross-fork globals
# (dedup._GEN_DEDUP / dedup._WORKER_DEDUP_TABLES — assigned by cmd_generate before
# the Pool forks). The named re-exports keep the internal call sites bare (so the
# test monkeypatch on create_dataset._hash_shard_pair still works) and preserve
# `from create_dataset import _LiveDedupTable, …` for the dedup test modules.
from preprocess.huggingface_dataset import dedup  # noqa: E402
from preprocess.huggingface_dataset.dedup import (  # noqa: E402,F401
    _LiveDedupTable,
    _LiveSkipFilter,
    _SKIP_WEIGHTS_SUBDIR,
    _VERIFY_HASH_SEED,
    _dedup_by_hash_pair,
    _dedup_table_capacity,
    _get_worker_dedup_table,
    _hash_shard_pair,
    _load_skip_weights,
    _primary_hash_collisions,
    _write_skip_weight_sidecar,
)
from preprocess.huggingface_dataset.annotations import (  # noqa: E402
    ACTIVE_MARKER_POLICY,
    ANNOTATIONS,
    AnnotationPerf,
    annotation_provenance,
    annotation_schema_fields,
    fetch_and_align,
)

logger = logging.getLogger(__name__)

# Shared processor loaded once in the parent process and inherited via fork COW.
_shared_processor: Optional[GenomeSequenceProcessor] = None

# Diploid N tokens, as diploid_tokens.tsv defines them: homozygous N/N is the
# symmetric symbol 'N' (U+004E) — the one the encoder actually emits, including
# for every N base of HG38_REF — and 'n' (U+006E) is the hemizygous single-N
# symbol. Until 2026-10-06 only 'n' was counted, so the >10%-N filter never
# fired on diploid output (see schemas/README.md).
_DIPLOID_N_TOKENS = ("N", "n")

# Haploid N tokens. The reference is upper-cased on load, so 'N' is a reference
# N; 'n' can only come from a hemizygous token. Both count.
_HAPLOID_N_TOKENS = ("N", "n")

# Maximum fraction of N tokens allowed per region before skipping.
_MAX_N_FRACTION = 0.10

# Reference pseudo-sample name and weight.
_REF_SAMPLE_ID = "HG38_REF"
_REF_SAMPLE_WEIGHT = 10

# Outcome of a shard writer (write_sample_shard / write_reference_shard).
# DONE   — shard written this run.
# SKIPPED— the finalized .parquet already exists; there was nothing to do.
# CLAIMED— another run holds the .tmp claim, so THIS run produced no shard for
#          that sample. Distinct from SKIPPED on purpose: SKIPPED means the shard
#          exists, CLAIMED means it does not (yet). If the claim belongs to a live
#          run that is the expected multi-run behaviour, but if the claim is dead
#          debris younger than --stale-age-hours, the sample is simply missing —
#          and folding that into "skipped (already present)" would report a
#          clean run over an incomplete shard set.
# FAILED — an error occurred; the caller must surface it (never silently drop).
_STATUS_DONE = "done"
_STATUS_SKIPPED = "skipped"
_STATUS_CLAIMED = "claimed"
_STATUS_FAILED = "failed"

# Supported representations for generated sequences.
_REPRESENTATIONS = ("diploid", "haploid")

# _GEN_PROVENANCE_FILENAME (the per-shard-dir generation sidecar name) is now
# imported from provenance.py — it is written by `cmd_generate` on first run and
# re-validated on resume to prevent silently mixing incompatible shards.

# Regex to extract a chromosome name (chr…) from a BCF filename.
# Delimiters are '.' and '-'; underscore is NOT a delimiter (some patch chroms
# use it, e.g. chr_X or chr_patch_hapl).
_CHROM_PAT = re.compile(r'(?:^|[.\-])(chr[^.\-]+)(?:[.\-]|$)')


def _is_persample_bcf_dir(bcf_dir: Path) -> bool:
    """True when *bcf_dir* holds one multi-contig BCF per sample.

    The per-sample layout names each file ``<sample>.bcf`` (GT-only, hom-ref
    dropped, all chromosomes); the legacy layout names per-chromosome
    multi-sample files with a ``chrN`` token. Detected by the absence of any
    chromosome token in the top-level ``*.bcf`` filenames.
    """
    bcfs = list(bcf_dir.glob("*.bcf"))
    return bool(bcfs) and not any(_CHROM_PAT.search(p.name) for p in bcfs)


# ---------------------------------------------------------------------------
# Parquet schema — one record per (sample_id × BED region)
#
# Base fields are stable across v4-v7. v7 appends one column per annotation
# registered in annotations.ANNOTATIONS (type depends on annotation: phylop
# → list<uint8>, cadd_phred → list<struct<pos:int32,score:float32>>; uint8 before 2026-10-14).
# Schema is built once at import time.
# ---------------------------------------------------------------------------

_BASE_FIELDS = [
    pa.field("sample_id", pa.dictionary(pa.int8(), pa.string())),
    pa.field("chrom", pa.dictionary(pa.int8(), pa.string())),
    pa.field("start", pa.int32()),
    pa.field("end", pa.int32()),
    pa.field("seq", pa.large_utf8()),
    pa.field("sample_weight", pa.int32()),
]

# v8 structural-marker columns (markers externalized out of ``seq``). ``pos`` /
# ``start`` / ``end`` index the marker-free ``seq``; ``dir`` is 1=fwd ‹, 2=rev ›.
# A -1 start/end sentinel marks an edge-truncated INDEL block (see split_markers).
_SNP_FIELD = pa.field("snp", pa.list_(pa.int32()))
_INDEL_SPAN_FIELD = pa.field(
    "indel_span",
    pa.list_(pa.struct([pa.field("start", pa.int32()), pa.field("end", pa.int32())])),
)
_DIRECTION_FIELD = pa.field(
    "direction",
    pa.list_(pa.struct([pa.field("pos", pa.int32()), pa.field("dir", pa.uint8())])),
)

# Set in cmd_generate BEFORE the worker Pool forks (and before _rebuild_pa_schema
# is called by the annotation registrars). Workers inherit them via fork.
_EXTERNALIZE_MARKERS: bool = False
_EXTERNALIZE_REPRESENTATION: str = "diploid"
_EMIT_REF_INDEX: bool = False


def _marker_schema_fields() -> list:
    """v8 marker columns to append, gated by the externalize flag/representation.

    Diploid gets snp + indel_span + direction; haploid omits direction (haploid
    output never carries direction markers)."""
    if not _EXTERNALIZE_MARKERS:
        return []
    fields = [_SNP_FIELD, _INDEL_SPAN_FIELD]
    if _EXTERNALIZE_REPRESENTATION == "diploid":
        fields.append(_DIRECTION_FIELD)
    return fields


_PA_SCHEMA = pa.schema(_BASE_FIELDS + _marker_schema_fields() + annotation_schema_fields())


def _rebuild_pa_schema() -> None:
    """Rebuild the module-level ``_PA_SCHEMA`` after ``ANNOTATIONS`` or the
    externalize-marker flag is set (e.g. when CADD is registered at run time, or
    when cmd_generate selects a v8 preset). Workers fork after this is called in
    the parent, so they inherit the up-to-date schema."""
    global _PA_SCHEMA
    _PA_SCHEMA = pa.schema(
        _BASE_FIELDS + ([pa.field("seq_ref_idx", pa.list_(pa.int32()))] if _EMIT_REF_INDEX else []) + _marker_schema_fields() + annotation_schema_fields()
    )

# ---------------------------------------------------------------------------
# Shard-dir provenance sidecar (contamination guard)
# ---------------------------------------------------------------------------


def _n_fraction(seq_str: str, representation: str, n_positions: Optional[int] = None) -> float:
    """Fraction of N bases in a per-region sequence string.

    Diploid: counts ``_DIPLOID_N_TOKENS`` — 'N', the symbol diploid_tokens.tsv
    assigns to an N/N pair (and the encoder's fallback for an unknown pair),
    plus 'n', the hemizygous single-N symbol.

    Haploid: the `seq` field holds reference bytes after the ALT-wins-leftmost
    collapse. The reference is upper-cased on load, so 'N' is a reference N;
    'n' can only come from a hemizygous token. Both count.

    n_positions: denominator = number of non-marker base codepoints
    (reference bases + inserted bases). The caller computes this as
    int((seq_ref_idx != -2).sum()). If omitted, falls back to len(seq_str),
    which over-counts by including the ASCII marker codepoints (^, {, }) and so
    understates the fraction — pass n_positions to exclude them.
    """
    if not seq_str:
        return 0.0
    tokens = _HAPLOID_N_TOKENS if representation == "haploid" else _DIPLOID_N_TOKENS
    n_count = sum(seq_str.count(t) for t in tokens)
    denom = n_positions if n_positions is not None else len(seq_str)
    return n_count / denom if denom > 0 else 0.0


def _file_sha256(path: Optional[Path]) -> Optional[str]:
    """SHA-256 of a file's bytes (streamed in 1 MiB chunks), or None if *path*
    is None. Used to fingerprint small content-determining inputs (BED, tokens,
    selection files) for the resume-verification guard. Cheap for these files
    (a few MB → milliseconds); the multi-hundred-MB reference is fingerprinted by
    path+size instead (see ``cmd_generate``)."""
    if path is None:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_region_binding(
    preset: "versions.DatasetVersion", bed_path: Path, bed_sha256: str
) -> None:
    """Enforce a region-bound preset's BED, and flag the inverse mistake.

    A preset that names a region set (e.g. ``v7-ldb`` → the deCODE EUR LD
    blocks) pins that BED's sha256. Running it against a different BED would
    produce a dataset whose ``schema_version`` claims a region universe it does
    not have — the exact ambiguity the ``-ldb`` names exist to remove — so the
    mismatch is fatal, raised here before any encoding work starts.

    The inverse (an UNBOUND version pointed at a bound preset's BED, e.g.
    ``--version v7 --bed ld_blocks_eur_grch38.bed``) is only warned about: it is
    how LD-block datasets were generated before the ``-ldb`` presets existed and
    stays legal, but it is almost always a mistake now.
    """
    if preset.region_bed_sha256 is not None:
        if bed_sha256 != preset.region_bed_sha256:
            raise SystemExit(
                f"ERROR: --version {preset.schema_version} is bound to the "
                f"'{preset.region_set}' region set ({preset.region_bed_hint}), "
                f"but --bed {bed_path} does not match it.\n"
                f"  expected sha256: {preset.region_bed_sha256}\n"
                f"  actual   sha256: {bed_sha256}\n"
                f"Use the correct BED, or use the unbound base version "
                f"({preset.schema_version.split('-')[0]}) if you intend a "
                f"different region universe."
            )
        return

    # Unbound preset: warn if this BED is some other preset's pinned region set.
    matching = [
        (name, other) for name, other in versions.PRESETS.items()
        if other.region_bed_sha256 == bed_sha256
    ]
    if not matching:
        return
    # Suggest the bound preset that encodes the same way as the one requested —
    # a v8 user must not be told "did you mean v4-ldb?". If this version has no
    # bound twin (v5/v6nd/v8 today), say so instead of suggesting a mismatch.
    same_encoding = [
        name for name, other in matching
        if all(
            getattr(other, f) == getattr(preset, f)
            for f in ("emit_anchor_snp_marker", "use_direction_markers",
                      "allow_annotations", "externalize_markers",
                      "force_unphased")
        )
    ]
    region_set = matching[0][1].region_set
    if same_encoding:
        advice = f"Did you mean --version {same_encoding[0]}?"
    else:
        advice = (
            f"There is no '{region_set}' preset for the {preset.schema_version} "
            f"encoding (bound presets: {', '.join(n for n, _ in matching)}); "
            f"add one to versions.py if this run is intentional."
        )
    logger.warning(
        "--bed %s is the pinned BED of the '%s' region set, but --version %s "
        "is region-unbound: the resulting dataset will record "
        "schema_version=%r and be indistinguishable from a canonical-BED run. "
        "%s",
        bed_path, region_set, preset.schema_version, preset.schema_version,
        advice,
    )


def _check_or_write_generation_provenance(
    output_dir: Path,
    representation: str,
    force_unphased: bool,
    use_direction_markers: bool,
    *,
    schema_version: str,
    marker_policy: str,
    externalize_markers: bool = False,
    inputs: Optional[dict] = None,
) -> None:
    """Thin wrapper over :func:`provenance.check_or_write_generation_provenance`.

    ``schema_version`` / ``marker_policy`` / ``externalize_markers`` come from the
    resolved version preset (passed by ``cmd_generate``) — there is no global
    fallback, since there is no default version. ``inputs`` is the caller-built
    fingerprint of content-determining inputs, verified on resume. The shared
    read/write/compare logic lives in ``provenance.py``.
    """
    _check_or_write_generation_provenance_impl(
        output_dir,
        schema_version=schema_version,
        representation=representation,
        force_unphased=force_unphased,
        use_direction_markers=use_direction_markers,
        marker_policy=marker_policy,
        externalize_markers=externalize_markers,
        annotations=annotation_provenance(),
        inputs=inputs,
    )


def _register_phylop_annotation(args: argparse.Namespace) -> None:
    """Append a :class:`PhyloPAnnotation` to ``ANNOTATIONS`` when the user
    provides ``--phylop-bw``. No-op (and no phyloP column) otherwise.

    Hard-fails if the supplied file does not exist.
    """
    if args.phylop_bw is None:
        return

    # Import via the same module path used for ANNOTATIONS (see top-of-file
    # ``from annotations import ...``). Importing from
    # ``huggingface_dataset.annotations`` would load the module a second time
    # and yield a different ``PhyloPAnnotation`` class.
    from preprocess.huggingface_dataset.annotations import PhyloPAnnotation

    path = Path(args.phylop_bw)
    if not path.is_file():
        raise FileNotFoundError(
            f"--phylop-bw not found: {path}"
        )

    ANNOTATIONS.append(PhyloPAnnotation(source_path=str(path)))
    _rebuild_pa_schema()
    logger.info("PhyloP annotation registered: %s", path)


def _register_cadd_annotation(
    args: argparse.Namespace,
    regions: List[Tuple[str, int, int]],
    output_dir: Path,
) -> None:
    """Register a :class:`CADDAnnotation` consuming the per-chrom Arrow
    shards under ``--cadd-arrow-dir`` (whole-genome ``full_arrow/`` layout
    produced by ``poc_a2_convert_raw_cadd.py``).

    Hard-fails before any worker is spawned if:
    * the arrow directory is missing, or
    * a per-chrom ``cadd_snv_<chrom>.arrow`` or ``cadd_indel_<chrom>.arrow``
      is missing for any chrom in the post-filtered region list.

    No BED-sha256 handshake — ``full_arrow/`` is BED-agnostic by construction.

    Forked workers inherit the appended ``CADDAnnotation`` instance via COW;
    each worker calls ``set_sample()`` per sample before its region loop."""
    from preprocess.huggingface_dataset.annotations import CADDAnnotation

    if args.cadd_arrow_dir is None:
        raise ValueError(
            "_register_cadd_annotation requires --cadd-arrow-dir; got None"
        )
    arrow_dir = Path(args.cadd_arrow_dir)

    if not arrow_dir.is_dir():
        raise FileNotFoundError(
            f"--cadd-arrow-dir not found: {arrow_dir}. "
            f"Run preprocess/huggingface_dataset/cadd_arrow/"
            f"convert_all_cadd_to_arrow.sh first."
        )
    # Each chromosome touched by the (post-filter) region list must have
    # both a SNV and an indel shard. Any missing one is a fail-fast.
    chroms = sorted({r[0] for r in regions})
    missing_snv = [
        c for c in chroms
        if not (arrow_dir / f"cadd_snv_{c}.arrow").is_file()
    ]
    missing_indel = [
        c for c in chroms
        if not (arrow_dir / f"cadd_indel_{c}.arrow").is_file()
    ]
    if missing_snv or missing_indel:
        raise FileNotFoundError(
            f"CADD shards missing in {arrow_dir}: "
            f"snv={missing_snv} indel={missing_indel}. "
            f"Wait for convert_all_cadd_to_arrow.sh to finish these chroms."
        )

    cadd = CADDAnnotation(arrow_dir=str(arrow_dir))
    # Per-chrom SNV and indel shards are loaded lazily inside each worker via
    # pa.memory_map — backed by the OS page cache (shared across workers, zero
    # per-worker private heap copy).  No pre-loading in the parent.
    ANNOTATIONS.append(cadd)
    _rebuild_pa_schema()
    logger.info("CADD annotation registered: arrow=%s", arrow_dir)


# ---------------------------------------------------------------------------
# BED loading
# ---------------------------------------------------------------------------


def load_bed_regions(
    bed_path: Path,
    genes: Optional[Set[str]] = None,
) -> List[Tuple[str, int, int]]:
    """Return (chrom, start, end) tuples from a BED file (0-based half-open).

    If *genes* is provided, only regions whose 4th column (gene name) is in the
    set are included.  Regions without a 4th column are excluded when filtering.
    """
    regions: List[Tuple[str, int, int]] = []
    skipped_by_gene = 0
    skipped_no_name = 0
    skipped_bad_coords = 0
    # tab split with a whitespace fallback for space-delimited BED files
    for lineno, line, parts in bedio.iter_bed_fields(bed_path, split="tab-then-ws"):
        if len(parts) < 3:
            raise ValueError(
                f"BED line {lineno} has fewer than 3 fields: {line!r}"
            )
        try:
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            logger.warning("BED line %d has non-integer coords, skipping: %r", lineno, line)
            continue
        if start < 0 or end <= start:
            logger.warning(
                "BED line %d has invalid coords (start<0 or end<=start), "
                "skipping: %r", lineno, line,
            )
            skipped_bad_coords += 1
            continue
        if genes is not None:
            gene_name = parts[3] if len(parts) >= 4 else ""
            if not gene_name:
                skipped_no_name += 1
                continue
            if gene_name not in genes:
                skipped_by_gene += 1
                continue
        regions.append((chrom, start, end))
    if skipped_bad_coords:
        logger.warning(
            "BED: skipped %d region(s) with invalid coords (start<0 or end<=start)",
            skipped_bad_coords,
        )
    if genes is not None:
        logger.info(
            "Gene filter: kept %d region(s), excluded %d (gene not in list), "
            "%d (no gene name in BED)",
            len(regions), skipped_by_gene, skipped_no_name,
        )
    return regions


# ---------------------------------------------------------------------------
# Shared record generator (used by both sample and reference shard writers)
# ---------------------------------------------------------------------------


def _iter_shard_records(
    regions: List[Tuple[str, int, int]],
    seq_results,
    sample_id: str,
    sample_weight: int,
    representation: str,
    skip_variant_free: bool,
    needs_variants: bool,
    perf: "AnnotationPerf",
    seq_skip_filter=None,
    skip_weight_out: Optional[Dict[Tuple[int, int], int]] = None,
    n_skipped_out: Optional[List[int]] = None,
    drop_counts_out: Optional[Dict[str, int]] = None,
):
    """Yield Parquet record dicts, applying skip rules shared by sample and reference shards.

    seq_results elements are either (seq_str, ref_idx) or
    (seq_str, ref_idx, var_idx, variants_list) depending on the encoder used.
    When needs_variants is True but the element is a 2-tuple (reference path),
    seq_var_idx is synthesised as all-(-1) so variant-aware annotations receive a
    no-variant signal.  When skip_variant_free is True, regions without SNP/INDEL
    markers are discarded (sample path); False keeps all regions (reference path).

    v8 (``_EXTERNALIZE_MARKERS`` set pre-fork): the encoder still emits the
    inline-marked string; here it is split via ``split_markers`` into a
    marker-free ``seq`` plus the sparse ``snp``/``indel_span``/``direction``
    columns, and the marker-free ``seq_ref_idx``/``seq_var_idx`` (``-2`` entries
    dropped) drive annotation alignment so all column positions index the same
    marker-free ``seq``.
    """
    externalize = _EXTERNALIZE_MARKERS
    for (chrom, start, end), result in zip(regions, seq_results):
        seq_str = result[0]
        seq_ref_idx = result[1]
        if len(result) > 2:
            seq_var_idx = result[2]
            variants_list = result[3]
        elif needs_variants:
            seq_var_idx = np.full(len(seq_ref_idx), -1, dtype=np.int32)
            variants_list = []
        else:
            seq_var_idx = None
            variants_list = None

        if not seq_str:
            if drop_counts_out is not None:
                drop_counts_out["empty"] += 1
            continue

        # v8: split the inline markers out into sparse columns; everything below
        # (variant-free skip, N fraction, dedup, annotation alignment) operates
        # on the marker-free sequence and its filtered index arrays.
        if externalize:
            marker_mask = seq_ref_idx != -2  # True for kept base codepoints
            if seq_var_idx is not None:
                seq_var_idx = seq_var_idx[marker_mask]
            seq_str, seq_ref_idx, snp_pos, indel_spans, dir_events = split_markers(
                seq_str, seq_ref_idx)
        else:
            snp_pos = indel_spans = dir_events = None

        if skip_variant_free:
            if externalize:
                variant_free = (not snp_pos) and (not indel_spans)
            else:
                variant_free = (
                    _SNP_MARKER not in seq_str and _INDEL_OPEN not in seq_str
                )
            if variant_free:
                if drop_counts_out is not None:
                    drop_counts_out["variant_free"] += 1
                continue
        # Denominator = emitted base codepoints (ref bases + insertions);
        # marker codepoints (seq_ref_idx == -2) are excluded so ^/{/} don't
        # dilute the N fraction. Under v8 seq_ref_idx is already marker-free, so
        # this equals len(seq_str); the expression stays correct either way.
        n_base_positions = int((seq_ref_idx != -2).sum())
        if _n_fraction(seq_str, representation, n_base_positions) > _MAX_N_FRACTION:
            if drop_counts_out is not None:
                drop_counts_out["high_n"] += 1
            continue
        # Inline dedup: skip sequences already written this run (shared live
        # table) or earlier in this sample (filter's local set).  Done BEFORE
        # fetch_and_align so deduped records cost no annotation compute; the
        # skipped occurrence's weight is recorded so assemble can fold it back
        # into the surviving row. Under v8 the key includes the marker columns
        # (seq alone is lossy — opposite phasings / hom-alt SNPs collapse).
        if seq_skip_filter is not None:
            if externalize:
                should_skip, h1, h2 = seq_skip_filter.should_skip(
                    seq_str, snp_pos, indel_spans, dir_events)
            else:
                should_skip, h1, h2 = seq_skip_filter.should_skip(seq_str)
            if should_skip:
                if skip_weight_out is not None:
                    key = (h1, h2)
                    skip_weight_out[key] = skip_weight_out.get(key, 0) + sample_weight
                if n_skipped_out is not None:
                    n_skipped_out[0] += 1
                continue
        rec = {
            "sample_id": sample_id,
            "chrom": chrom,
            "start": start,
            "end": end,
            "seq": seq_str,
            "sample_weight": sample_weight,
        }
        if _EMIT_REF_INDEX:
            rec["seq_ref_idx"] = seq_ref_idx.tolist()
        if externalize:
            rec["snp"] = snp_pos
            rec["indel_span"] = [{"start": s, "end": e} for s, e in indel_spans]
            if representation == "diploid":
                rec["direction"] = [{"pos": p, "dir": d} for p, d in dir_events]
        rec.update(fetch_and_align(
            ANNOTATIONS, chrom, start, end,
            seq_ref_idx, seq_var_idx, variants_list,
            ACTIVE_MARKER_POLICY, perf,
        ))
        yield rec


# ---------------------------------------------------------------------------
# Per-sample shard writer
# ---------------------------------------------------------------------------


def _write_parquet_streaming(
    records_iter,
    temp_path: Path,
    compression: str = "lz4",
    compression_level: Optional[int] = None,
) -> int:
    """Write records to Parquet using streaming ParquetWriter.

    Returns the number of records written.
    """
    # Build writer kwargs — compression_level only applies to zstd.
    writer_kwargs: dict = {
        "compression": compression,
        "use_dictionary": ["sample_id", "chrom"],
        "write_statistics": True,
    }
    if compression == "zstd" and compression_level is not None:
        writer_kwargs["compression_level"] = compression_level

    # LZ4's frame codec fails past ~2GB of uncompressed input in one shot (int32
    # length ceiling in liblz4). A flat 5000-record cap is fine for the small
    # canonical-BED regions (~7kb seq avg) but overflows for LD-block regions
    # (~2Mbp avg, up to 28Mbp) — a 5000-record batch there is tens of GB. Flush
    # on cumulative seq bytes too, well under the limit to leave headroom for
    # per-record/column overhead.
    _MAX_BATCH_SEQ_BYTES = 256 * 1024 * 1024

    n_written = 0
    batch_records: List[dict] = []
    batch_seq_bytes = 0
    writer = None
    try:
        for rec in records_iter:
            batch_records.append(rec)
            batch_seq_bytes += len(rec["seq"])
            if len(batch_records) >= 5000 or batch_seq_bytes >= _MAX_BATCH_SEQ_BYTES:
                # from_pylist does the column transpose in Arrow C++ (selecting
                # by schema field name, so dict key order is irrelevant) instead
                # of the per-column Python comprehension. Byte-identical output.
                batch = pa.RecordBatch.from_pylist(batch_records, schema=_PA_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(temp_path, _PA_SCHEMA, **writer_kwargs)
                writer.write_batch(batch)
                n_written += len(batch_records)
                batch_records.clear()
                batch_seq_bytes = 0

        # Flush remainder
        if batch_records:
            batch = pa.RecordBatch.from_pylist(batch_records, schema=_PA_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, _PA_SCHEMA, **writer_kwargs)
            writer.write_batch(batch)
            n_written += len(batch_records)

        if writer is None:
            # No records at all — write empty Parquet with correct schema
            table = pa.table(
                {col: pa.array([], type=_PA_SCHEMA.field(col).type) for col in _PA_SCHEMA.names},
                schema=_PA_SCHEMA,
            )
            pq.write_table(table, temp_path, **writer_kwargs)
    finally:
        if writer is not None:
            writer.close()

    return n_written


# ---------------------------------------------------------------------------
# Keeping a .parquet.tmp claim visibly alive.
#
# A .tmp is judged abandoned by how long it has sat untouched (_clean_stale_tmp
# / --stale-age-hours). Nothing writes to it between the O_CREAT|O_EXCL claim
# and the first Parquet batch flush — and that gap IS the per-sample encode,
# because every region is encoded before a single record is streamed (~6 min on
# the canonical BED, ~3x that on LD blocks). So the claim that is working
# hardest looks idle for exactly the stretch that matters, and the age heuristic
# is only as sound as that gap is short. This heartbeat advances the mtime while
# the sample is in flight, which is what makes "untouched for N hours" mean what
# --clean-stale assumes it means.
#
# It touches the claim by FILE DESCRIPTOR rather than by path, deliberately: if
# another run's sweep did unlink our .tmp and re-claim the sample, our fd still
# refers to the now-unlinked inode, so the heartbeat can never keep somebody
# else's claim looking alive.
# ---------------------------------------------------------------------------

_CLAIM_HEARTBEAT_S = 60.0


class _ClaimHeartbeat:
    """Advance a ``.parquet.tmp`` claim's mtime until :meth:`stop` is called."""

    def __init__(self, fd: int, path: Path,
                 interval_s: Optional[float] = None):
        self._fd = fd
        self._path = path
        # Resolved here, not as a default argument: a default binds the module
        # constant at def time, which would make the interval untunable and
        # silently ignore any monkeypatch in a test or a debugging session.
        self._interval = _CLAIM_HEARTBEAT_S if interval_s is None else interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _touch(self) -> None:
        if os.utime in os.supports_fd:
            os.utime(self._fd, None)
        else:                       # no futimens here; path is the only handle
            os.utime(self._path, None)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._touch()
            except (OSError, ValueError, TypeError, NotImplementedError):
                return              # claim or fd gone — nothing left to keep alive

    def start(self) -> None:
        if self._thread is not None:
            return
        # daemon: a heartbeat must never delay a worker's exit.
        self._thread = threading.Thread(
            target=self._run, name=f"claim-{self._path.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


def write_sample_shard(
    sample_name: str,
    bcf_dir: Path,
    regions: List[Tuple[str, int, int]],
    ref_path: Path,
    tokens_path: Optional[Path],
    output_dir: Path,
    compression: str = "lz4",
    compression_level: Optional[int] = None,
    force_unphased: bool = False,
    use_direction_markers: bool = False,
    representation: str = "diploid",
    seq_skip_filter=None,
    emit_anchor_snp_marker: bool = True,
) -> str:
    """Write one Parquet shard for a single sample.

    Returns one of ``_STATUS_DONE`` (shard written), ``_STATUS_SKIPPED``
    (the finalized output already exists), ``_STATUS_CLAIMED`` (another run
    holds the .tmp claim, so this run wrote nothing for this sample) or
    ``_STATUS_FAILED`` (an error occurred — the caller must surface it; a
    failed sample is never silently treated as a deliberate skip).
    Atomic: writes to .parquet.tmp first, renames on success.
    Resume-safe: skips if .parquet already exists or .parquet.tmp is claimed.
    """
    output_path = output_dir / f"{sample_name}.parquet"
    temp_path = output_dir / f"{sample_name}.parquet.tmp"

    # Resume check
    if output_path.exists():
        logger.info("[SKIP] %s — output already exists", sample_name)
        return _STATUS_SKIPPED

    # Atomic claim. The fd is kept open for the sample's lifetime so the
    # heartbeat below can touch this exact inode (see _ClaimHeartbeat); it is
    # closed on every exit path.
    try:
        claim_fd = os.open(str(temp_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        logger.info("[SKIP] %s — .tmp exists (in progress or stale)", sample_name)
        return _STATUS_CLAIMED

    t0 = time.perf_counter()
    needs_variants = any(a.requires_variants for a in ANNOTATIONS)
    if needs_variants and representation == "haploid":
        os.close(claim_fd)
        if temp_path.exists():
            temp_path.unlink()
        logger.error(
            "[FAIL] %s — variant-aware annotations (e.g. CADD) are not supported "
            "in haploid representation.", sample_name,
        )
        return _STATUS_FAILED
    heartbeat = _ClaimHeartbeat(claim_fd, temp_path)
    heartbeat.start()
    perf = AnnotationPerf()
    # Per-worker annotation handles (one of each per process; not thread-safe
    # for pyBigWig but we only call them in the single-threaded post-fetch
    # loop below). Variant-aware annotations (e.g. CADD) get a per-sample
    # set_sample() tag so their miss-log sidecar names the sample.
    processor = _shared_processor
    owns_processor = False
    t_open = 0.0
    try:
        t_open_start = time.perf_counter()
        for a in ANNOTATIONS:
            a.open()
            if hasattr(a, "set_sample"):
                a.set_sample(sample_name)
        t_open = time.perf_counter() - t_open_start
        # Reuse shared processor (loaded once in parent, inherited via fork COW)
        # or create a new one if not available (e.g. single-process mode).
        if processor is None:
            print(f"  [{sample_name}] Loading reference genome (no shared processor)…", flush=True)
            t_load = time.perf_counter()
            processor = GenomeSequenceProcessor(
                ref_path=ref_path, diploid_tokens_path=tokens_path,
                force_unphased=force_unphased,
                use_direction_markers=use_direction_markers,
                emit_anchor_snp_marker=emit_anchor_snp_marker,
            )
            print(f"  [{sample_name}] Reference loaded in {time.perf_counter() - t_load:.1f}s", flush=True)
            owns_processor = True

        # Pass ALL regions in one call. Encoder threads (max_workers below) are
        # fixed at 1: measured on LD-block runs (encoder cpu: ~50%/thr at
        # inner_threads=2, i.e. two threads at ~50% utilization sum to ~1x a
        # single thread's throughput — GIL contention in the Python-level
        # encoding hot path), so a second thread bought no speedup while
        # doubling each worker's peak memory (a full in-flight region buffer
        # per thread, up to ~28Mbp for the largest LD blocks).
        print(
            f"  [{sample_name}] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing {len(regions)} regions across all chromosomes…",
            flush=True,
        )

        _cpu_before = os.times()
        t_encoder_start = time.perf_counter()
        try:
            if representation == "haploid":
                seq_results = processor.get_marked_haploid_sequence_with_ref_idx_parallel_str(
                    regions=regions,
                    bcf_dir=bcf_dir,
                    sample_name=sample_name,
                    skip_gaps=True,
                    max_workers=1,
                )
            elif needs_variants:
                seq_results = processor.get_marked_sequence_with_ref_idx_and_variants_parallel_str(
                    regions=regions,
                    bcf_dir=bcf_dir,
                    sample_name=sample_name,
                    skip_gaps=True,
                    max_workers=1,
                )
            else:
                seq_results = processor.get_marked_sequence_with_ref_idx_parallel_str(
                    regions=regions,
                    bcf_dir=bcf_dir,
                    sample_name=sample_name,
                    skip_gaps=True,
                    max_workers=1,
                )
            t_encoder = time.perf_counter() - t_encoder_start
            _t_cpu_after = os.times()
            _cpu_encoder_s = (
                (_t_cpu_after.user - _cpu_before.user) +
                (_t_cpu_after.system - _cpu_before.system)
            )
        except Exception as exc:
            print(
                f"  [{sample_name}] FAILED — sequence generation: {exc}",
                flush=True,
            )
            logger.warning(
                "[FAIL] %s — sequence generation failed: %s",
                sample_name, exc,
            )
            if owns_processor:
                processor.close()
                processor = None
            if temp_path.exists():
                temp_path.unlink()
            return _STATUS_FAILED

        nan_counts: Dict[str, List[int]] = {a.name: [0, 0] for a in ANNOTATIONS}
        t_stream_start = time.perf_counter()
        n_skipped = [0]
        skip_weight: Dict[int, int] = {}
        drop_counts = {"empty": 0, "variant_free": 0, "high_n": 0}

        n_records = _write_parquet_streaming(
            _encode_annotations_and_yield(
                _tally_nan_and_yield(
                    _iter_shard_records(
                        regions, seq_results, sample_name, 1,
                        representation, skip_variant_free=True,
                        needs_variants=needs_variants, perf=perf,
                        seq_skip_filter=seq_skip_filter,
                        skip_weight_out=skip_weight,
                        n_skipped_out=n_skipped,
                        drop_counts_out=drop_counts,
                    ),
                    nan_counts,
                ),
            ),
            temp_path,
            compression=compression,
            compression_level=compression_level,
        )
        t_stream = time.perf_counter() - t_stream_start

        if owns_processor:
            processor.close()
            processor = None

        # Write the skip-weight sidecar BEFORE finalizing the shard so the
        # invariant "<sample>.parquet exists ⟹ its sidecar exists" holds (when
        # any occurrence was skipped).  A re-run rewrites the same path, so a
        # crash before the shard rename can never double-count.
        if skip_weight:
            _write_skip_weight_sidecar(
                skip_weight,
                output_dir / _SKIP_WEIGHTS_SUBDIR / f"{sample_name}.parquet",
            )

        # Atomic rename (shard finalized LAST)
        temp_path.rename(output_path)

        # Publish this sample's kept hashes to the shared live dedup table now
        # that its shard is durable, so other workers skip them from here on.
        # Done AFTER the rename so the table never references unwritten rows.
        if seq_skip_filter is not None:
            seq_skip_filter.publish()
        elapsed = time.perf_counter() - t0
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        nan_stat = _format_nan_stat(nan_counts)
        # Cumulative annotation compute lives in perf.fetch_s (align_s is
        # 0 since compute() fuses both). The parquet-only time is what's
        # left of the streaming phase after annotation compute.
        t_annot = perf.fetch_s
        t_parquet_only = max(0.0, t_stream - t_annot)
        timing = _format_timing_breakdown(
            elapsed=elapsed, t_open=t_open, t_encoder=t_encoder,
            t_annot=t_annot, t_parquet=t_parquet_only,
            per_annot=perf.per_annotation_fetch_s,
        )
        _cpu_ratio = _cpu_encoder_s / t_encoder if t_encoder > 0 else 0.0
        _cpu_stat = (
            f" | encoder cpu: {_cpu_ratio:.0%}"
            f" ({_cpu_encoder_s:.0f}s÷{t_encoder:.0f}s)"
        )
        skip_stat = f", {n_skipped[0]} deduped" if n_skipped[0] > 0 else ""
        drop_stat = _format_drop_stat(drop_counts)
        print(
            f"[DONE] {sample_name} — {n_records} records{skip_stat}{drop_stat}, "
            f"{file_size_mb:.1f} MB, {timing}{_cpu_stat}{nan_stat}",
            flush=True,
        )
        return _STATUS_DONE

    except Exception:
        elapsed = time.perf_counter() - t0
        print(f"[FAIL] {sample_name} after {elapsed:.1f}s", flush=True)
        logger.exception("[FAIL] %s after %.1fs", sample_name, elapsed)
        if owns_processor and processor is not None:
            processor.close()
        if temp_path.exists():
            temp_path.unlink()
        return _STATUS_FAILED
    finally:
        # Stop the heartbeat before releasing the claim fd — the thread touches
        # that fd, and a join here costs at most one interval.
        heartbeat.stop()
        try:
            os.close(claim_fd)
        except OSError:
            pass
        for a in ANNOTATIONS:
            a.close()


def _format_timing_breakdown(
    *,
    elapsed: float,
    t_open: float,
    t_encoder: float,
    t_annot: float,
    t_parquet: float,
    per_annot: Dict[str, float],
) -> str:
    """One-line breakdown of where the per-sample wall clock went.

    Format:
        total=12.3s [encoder=6.1s 50%, annot=2.4s 19% (phylop=1.0s, cadd_phred=1.4s),
                    parquet=3.5s 28%, open=0.1s 1%, other=0.2s 2%]

    The 'other' bucket catches everything not separately timed (e.g. processor
    close, rename, miscellaneous overhead) so the percentages always sum to
    ~100 %."""
    def pct(s: float) -> str:
        return f"{100 * s / elapsed:.0f}%" if elapsed > 0 else "-"

    t_accounted = t_open + t_encoder + t_annot + t_parquet
    t_other = max(0.0, elapsed - t_accounted)
    per_annot_str = ", ".join(
        f"{name}={s:.2f}s {pct(s)}" for name, s in per_annot.items()
    )
    annot_block = (
        f"annot={t_annot:.2f}s {pct(t_annot)}"
        + (f" ({per_annot_str})" if per_annot_str else "")
    )
    return (
        f"total={elapsed:.1f}s ["
        f"encoder={t_encoder:.2f}s {pct(t_encoder)}, "
        f"{annot_block}, "
        f"parquet={t_parquet:.2f}s {pct(t_parquet)}, "
        f"open={t_open:.2f}s {pct(t_open)}, "
        f"other={t_other:.2f}s {pct(t_other)}]"
    )


def _encode_annotations_and_yield(records_iter):
    """Apply each annotation's encode() to convert float32 compute() output to storage format.

    Runs after _tally_nan_and_yield (which works on float32) and before
    _write_parquet_streaming. No-op when ANNOTATIONS is empty."""
    if not ANNOTATIONS:
        yield from records_iter
        return
    for rec in records_iter:
        for annot in ANNOTATIONS:
            rec[annot.name] = annot.encode(rec[annot.name])
        yield rec


def _tally_nan_and_yield(records_iter, nan_counts: Dict[str, List[int]]):
    """Pass records through unchanged while accumulating, per annotation
    column, ``[n_nan, n_total]`` into ``nan_counts``.

    This replaces the previous post-write re-read of every annotation column
    (``pq.read_table`` of the finished shard) with a single pass over the
    float32 arrays as they stream to the writer — no extra I/O, no second
    materialisation. ``nan_counts`` must be pre-seeded with one ``[0, 0]``
    entry per annotation name (empty dict ⇒ no-op pass-through)."""
    names = list(nan_counts.keys())
    for rec in records_iter:
        for name in names:
            arr = rec[name]
            nan_counts[name][0] += int(np.isnan(arr).sum())
            nan_counts[name][1] += arr.size
        yield rec


def _format_nan_stat(nan_counts: Dict[str, List[int]]) -> str:
    """Format accumulated NaN counts as ``" | nan: phylop=0.05 cadd_phred=0.97"``.

    Catches contig-naming flips, missing pyBigWig coverage, and other
    "all-NaN" failures early. Empty string when there are no annotation
    columns."""
    if not nan_counts:
        return ""
    parts = []
    for name, (n_nan, n_total) in nan_counts.items():
        if n_total == 0:
            parts.append(f"{name}=N/A")
        else:
            parts.append(f"{name}={n_nan / n_total:.3f}")
    return " | nan: " + " ".join(parts) if parts else ""


def _format_drop_stat(drop_counts: Dict[str, int]) -> str:
    """Format per-reason region-drop counts for the ``[DONE]`` line.

    Surfaces regions silently discarded before any record is written —
    ``high_n`` (>10% N tokens), ``variant_free`` (no SNP/INDEL marker, sample
    path only) and ``empty`` (empty sequence). Empty string when nothing was
    dropped so the common case stays quiet."""
    labels = (("high_n", "high-N"), ("variant_free", "variant-free"),
              ("empty", "empty"))
    bits = [f"{drop_counts[k]} {label}" for k, label in labels if drop_counts.get(k)]
    return ", " + " / ".join(bits) + " skipped" if bits else ""


def write_reference_shard(
    processor: GenomeSequenceProcessor,
    regions: List[Tuple[str, int, int]],
    output_dir: Path,
    compression: str = "lz4",
    compression_level: Optional[int] = None,
    representation: str = "diploid",
) -> str:
    """Write a Parquet shard for the HG38 reference genome (no variants).

    Includes ALL BED regions (no variant filtering). Sets sample_weight
    to _REF_SAMPLE_WEIGHT for upweighting during training.

    Returns one of ``_STATUS_DONE``, ``_STATUS_SKIPPED``, ``_STATUS_CLAIMED``
    or ``_STATUS_FAILED`` (same contract as :func:`write_sample_shard`).
    """
    output_path = output_dir / f"{_REF_SAMPLE_ID}.parquet"
    temp_path = output_dir / f"{_REF_SAMPLE_ID}.parquet.tmp"

    if output_path.exists():
        print(f"[SKIP] {_REF_SAMPLE_ID} — output already exists", flush=True)
        return _STATUS_SKIPPED

    try:
        claim_fd = os.open(str(temp_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"[SKIP] {_REF_SAMPLE_ID} — .tmp exists (in progress or stale)", flush=True)
        return _STATUS_CLAIMED

    t0 = time.perf_counter()
    needs_variants = any(a.requires_variants for a in ANNOTATIONS)
    if needs_variants and representation == "haploid":
        os.close(claim_fd)
        if temp_path.exists():
            temp_path.unlink()
        logger.error(
            "[FAIL] %s — variant-aware annotations (e.g. CADD) are not supported "
            "in haploid representation.", _REF_SAMPLE_ID,
        )
        return _STATUS_FAILED
    heartbeat = _ClaimHeartbeat(claim_fd, temp_path)
    heartbeat.start()
    perf = AnnotationPerf()
    t_open = 0.0
    try:
        t_open_start = time.perf_counter()
        for a in ANNOTATIONS:
            a.open()
            if hasattr(a, "set_sample"):
                a.set_sample(_REF_SAMPLE_ID)
        t_open = time.perf_counter() - t_open_start
        print(
            f"  [{_REF_SAMPLE_ID}] Generating reference sequences "
            f"for {len(regions)} regions (representation={representation})…",
            flush=True,
        )
        _cpu_before = os.times()
        t_encoder_start = time.perf_counter()
        if representation == "haploid":
            seq_results = processor.get_reference_haploid_sequences_with_ref_idx_str(
                regions=regions, skip_gaps=True,
            )
        else:
            seq_results = processor.get_reference_sequences_with_ref_idx_str(
                regions=regions, skip_gaps=True,
            )
        t_encoder = time.perf_counter() - t_encoder_start
        _t_cpu_after = os.times()
        _cpu_encoder_s = (
            (_t_cpu_after.user - _cpu_before.user) +
            (_t_cpu_after.system - _cpu_before.system)
        )

        nan_counts: Dict[str, List[int]] = {a.name: [0, 0] for a in ANNOTATIONS}
        t_stream_start = time.perf_counter()
        drop_counts = {"empty": 0, "variant_free": 0, "high_n": 0}
        n_records = _write_parquet_streaming(
            _encode_annotations_and_yield(
                _tally_nan_and_yield(
                    _iter_shard_records(
                        regions, seq_results, _REF_SAMPLE_ID, _REF_SAMPLE_WEIGHT,
                        representation, skip_variant_free=False,
                        needs_variants=needs_variants, perf=perf,
                        drop_counts_out=drop_counts,
                    ),
                    nan_counts,
                ),
            ),
            temp_path,
            compression=compression,
            compression_level=compression_level,
        )
        t_stream = time.perf_counter() - t_stream_start

        temp_path.rename(output_path)
        elapsed = time.perf_counter() - t0
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        nan_stat = _format_nan_stat(nan_counts)
        t_annot = perf.fetch_s
        t_parquet_only = max(0.0, t_stream - t_annot)
        timing = _format_timing_breakdown(
            elapsed=elapsed, t_open=t_open, t_encoder=t_encoder,
            t_annot=t_annot, t_parquet=t_parquet_only,
            per_annot=perf.per_annotation_fetch_s,
        )
        _cpu_ratio = _cpu_encoder_s / t_encoder if t_encoder > 0 else 0.0
        _cpu_stat = (
            f" | encoder cpu: {_cpu_ratio:.0%}/thr"
            f" ({_cpu_encoder_s:.0f}s÷1thr÷{t_encoder:.0f}s)"
        )
        drop_stat = _format_drop_stat(drop_counts)
        print(
            f"[DONE] {_REF_SAMPLE_ID} — {n_records} records{drop_stat}, "
            f"{file_size_mb:.1f} MB, {timing}{_cpu_stat}{nan_stat}",
            flush=True,
        )
        return _STATUS_DONE

    except Exception:
        elapsed = time.perf_counter() - t0
        print(f"[FAIL] {_REF_SAMPLE_ID} after {elapsed:.1f}s", flush=True)
        logger.exception("[FAIL] %s after %.1fs", _REF_SAMPLE_ID, elapsed)
        if temp_path.exists():
            temp_path.unlink()
        return _STATUS_FAILED
    finally:
        heartbeat.stop()
        try:
            os.close(claim_fd)
        except OSError:
            pass
        for a in ANNOTATIONS:
            a.close()


# ---------------------------------------------------------------------------
# Worker entry point (for multiprocessing.Pool)
# ---------------------------------------------------------------------------


def _worker_fn(args: tuple) -> Tuple[str, str]:
    """Unpack arguments and call write_sample_shard.

    Returns ``(sample_name, status)`` where status is one of
    ``_STATUS_DONE`` / ``_STATUS_SKIPPED`` / ``_STATUS_FAILED`` — so the
    parent can tell a deliberate skip apart from a failure.
    """
    (sample_name, bcf_dir, regions, ref_path, tokens_path,
     output_dir, compression, compression_level, force_unphased,
     use_direction_markers, representation,
     split, emit_anchor_snp_marker) = args
    skip_filter = None
    if dedup._GEN_DEDUP is not None and split in dedup._GEN_DEDUP:
        # v8: the dedup key must fold in the externalized marker columns (set
        # pre-fork; inherited by this worker).
        skip_filter = _LiveSkipFilter(
            _get_worker_dedup_table(split), externalize=_EXTERNALIZE_MARKERS)
    result = write_sample_shard(
        sample_name=sample_name,
        bcf_dir=Path(bcf_dir),
        regions=regions,
        ref_path=Path(ref_path),
        tokens_path=Path(tokens_path) if tokens_path is not None else None,
        output_dir=Path(output_dir),
        compression=compression,
        compression_level=compression_level,
        force_unphased=force_unphased,
        use_direction_markers=use_direction_markers,
        representation=representation,
        seq_skip_filter=skip_filter,
        emit_anchor_snp_marker=emit_anchor_snp_marker,
    )
    return sample_name, result


def _tally_worker_results(
    results: List[Tuple[str, str]]
) -> Tuple[int, int, List[str]]:
    """Split ``(sample_name, status)`` worker results three ways.

    Returns ``(n_done, n_skipped, failed_sample_names)``. Keeping FAILED
    separate from SKIPPED is the whole point: a crashed sample must never be
    reported (or counted) as a deliberate skip. CLAIMED counts as skipped here
    (this run wrote nothing for it either way); use :func:`_claimed_samples` to
    name the ones that produced no shard at all."""
    n_done = sum(1 for _, st in results if st == _STATUS_DONE)
    n_skipped = sum(
        1 for _, st in results if st in (_STATUS_SKIPPED, _STATUS_CLAIMED)
    )
    failed = [name for name, st in results if st == _STATUS_FAILED]
    return n_done, n_skipped, failed


def _claimed_samples(results: List[Tuple[str, str]]) -> List[str]:
    """Samples this run skipped because someone else held the .tmp claim.

    These have no finalized shard — unlike a plain SKIPPED, which means the
    ``.parquet`` is already on disk. Surfacing them separately is what keeps a
    run that silently produced nothing (because a dead-but-recent ``.tmp`` still
    blocks every sample) from reading as "skipped (already present)"."""
    return [name for name, st in results if st == _STATUS_CLAIMED]


def _missing_samples(
    submitted_names: List[str], results: List[Tuple[str, str]]
) -> List[str]:
    """Submitted samples that produced no worker status at all.

    A worker killed by the OOM killer or a C-extension segfault returns no
    ``(name, status)`` tuple (and ``multiprocessing.Pool`` may drop the result
    on worker death), so such a sample is absent from ``results`` rather than
    marked FAILED. Treat it as a failure so it can't silently vanish from the
    tally. Order follows ``submitted_names``."""
    seen = {name for name, _ in results}
    return [s for s in submitted_names if s not in seen]


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------


def discover_samples(bcf_dir: Path) -> List[str]:
    """Return the sample list for *bcf_dir*.

    Per-sample layout (``<sample>.bcf`` per file): the sample set is the file
    stems. Legacy per-chromosome multi-sample layout: read the sample list
    from the first BCF's header.
    """
    from cyvcf2 import VCF

    bcf_files = sorted(bcf_dir.glob("*.bcf"))
    if not bcf_files:
        raise FileNotFoundError(f"No *.bcf files found in {bcf_dir}")

    if _is_persample_bcf_dir(bcf_dir):
        return [p.stem for p in bcf_files]

    with VCF(str(bcf_files[0])) as vcf:
        return list(vcf.samples)


def _load_sample_names(path: Path) -> List[str]:
    """Read sample names from a text file (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# CLI — generate subcommand
# ---------------------------------------------------------------------------


# How often the resume pre-seed reports progress. It is the one loop in
# `generate` that can run for hours with nothing to say -- 2.5 h on the 9.7 TB
# v7-tsstes train split -- and silence there reads exactly like the worker-pool
# deadlock of 2026-09-02. 60 s is far below the cost of the work it describes.
_SEED_REPORT_S = 60.0


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate per-sample Parquet shards.

    Failure handling: a sample whose worker raises in Python is caught and
    reported as FAILED (non-zero exit). A worker killed by the OOM killer or a
    C-extension segfault returns no status; ``multiprocessing.Pool`` may then
    drop that result or stall. The completeness guard at the end treats any
    submitted sample with no returned status as FAILED so it can't silently
    vanish from the tally, but a hard hang (no result ever produced) can still
    stall the run — re-run to resume (killed samples leave no ``.parquet``).
    """
    # Capture start wall-clock + monotonic clock up front (before the early-exit
    # paths) so every run record — including all-done no-ops — can report when it
    # ran and how long it took.
    gen_started = datetime.datetime.now()
    gen_started_str = f"{gen_started:%Y-%m-%d %H:%M:%S}"
    t_start_gen = time.perf_counter()
    print(f"Generate started at {gen_started_str}", flush=True)
    print(f"Command: {' '.join(sys.argv)}", flush=True)
    # ── Resolve the version preset (single, REQUIRED input; v4/v5/v6/v6nd/v7/v8) ──
    # Drives encoder anchor-SNP emission, whether markers are externalized into
    # sparse columns, whether annotations are allowed (validated in main()), and
    # the schema_version stamped into provenance. There is no default: argparse
    # requires --version, so args.version is always present.
    preset = versions.resolve(args.version)
    # Direction markers are owned by the preset (every shipped diploid version
    # uses them). Haploid output cannot carry inline direction markers, so force
    # OFF there. Under externalize_markers the direction goes to a column, but
    # haploid still has no direction column, so the same gate applies.
    use_direction_markers = (
        preset.use_direction_markers and args.representation == "diploid"
    )
    externalize_markers = preset.externalize_markers
    # Effective force_unphased: the standalone --force-unphased ablation OR a
    # preset that bundles it (v6nd drops phase intrinsically). Used everywhere
    # below in place of args.force_unphased so both sources take effect.
    force_unphased = args.force_unphased or preset.force_unphased
    print(f"Dataset version: {preset.schema_version} "
          f"(emit_anchor_snp_marker={preset.emit_anchor_snp_marker}, "
          f"use_direction_markers={use_direction_markers}, "
          f"allow_annotations={preset.allow_annotations}, "
          f"externalize_markers={externalize_markers})", flush=True)
    # Publish the externalize-marker decision to the module globals BEFORE the
    # worker Pool forks and before any _rebuild_pa_schema() call, so the schema
    # (marker columns) and _iter_shard_records (the split) are consistent across
    # the parent and all forked workers.
    global _EXTERNALIZE_MARKERS, _EXTERNALIZE_REPRESENTATION, _EMIT_REF_INDEX
    _EMIT_REF_INDEX = args.emit_ref_index
    _EXTERNALIZE_MARKERS = externalize_markers
    _EXTERNALIZE_REPRESENTATION = args.representation
    # ── Validate inputs ─────────────────────────────────────────────────
    if not args.bcf_dir.is_dir():
        print(f"ERROR: --bcf-dir does not exist or is not a directory: {args.bcf_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.bed.is_file():
        print(f"ERROR: --bed file not found: {args.bed}", file=sys.stderr)
        sys.exit(1)
    # Region-set binding. Computed once here and reused for the resume
    # fingerprint below, so the BED is hashed a single time.
    bed_sha256 = _file_sha256(args.bed)
    _check_region_binding(preset, args.bed, bed_sha256)
    if not args.ref.is_file():
        print(f"ERROR: --ref file not found: {args.ref}", file=sys.stderr)
        sys.exit(1)
    if args.representation == "diploid":
        if args.tokens is None:
            print(
                "ERROR: --tokens is required when --representation=diploid",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.tokens.is_file():
            print(f"ERROR: --tokens file not found: {args.tokens}", file=sys.stderr)
            sys.exit(1)
    else:
        # Haploid: --tokens is optional. If supplied, validate it exists; if
        # omitted, the processor will fall back to the default token file
        # shipped with the package (only the meta token symbols are used).
        if args.tokens is not None and not args.tokens.is_file():
            print(f"ERROR: --tokens file not found: {args.tokens}", file=sys.stderr)
            sys.exit(1)

    # ── Discover samples ───────────────────────────────────────────────
    if args.samples_file:
        samples = _load_sample_names(args.samples_file)
        logger.info("Loaded %d sample(s) from %s", len(samples), args.samples_file)
    else:
        logger.info("Discovering samples from BCF files in %s …", args.bcf_dir)
        samples = discover_samples(args.bcf_dir)
        logger.info("Found %d sample(s)", len(samples))

    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        logger.info("Limited to %d sample(s)", len(samples))

    # ── Train/test split ───────────────────────────────────────────────
    test_samples_set: set = set()
    if args.test_samples_file:
        test_samples_list = _load_sample_names(args.test_samples_file)
        test_samples_set = set(test_samples_list)
        logger.info("Loaded %d test sample name(s) from %s",
                     len(test_samples_set), args.test_samples_file)
        # Append test samples that are not already in the (train) samples list.
        # --max-samples applies only to train samples; test samples are always
        # included in full so the test output directory is populated.
        train_set = set(samples)
        extra_test = [s for s in test_samples_list if s not in train_set]
        if extra_test:
            samples = samples + extra_test
            logger.info("Added %d test sample(s) to processing queue", len(extra_test))

    # When test split is active, create train/ and test/ subdirs.
    if test_samples_set:
        train_output = args.output / "train"
        test_output = args.output / "test"
        train_output.mkdir(parents=True, exist_ok=True)
        test_output.mkdir(parents=True, exist_ok=True)
    else:
        train_output = args.output
        args.output.mkdir(parents=True, exist_ok=True)

    # ── Load BED regions (with optional gene filter) ────────────────────
    genes_filter: Optional[Set[str]] = None
    if args.genes_file:
        with open(args.genes_file) as gf:
            genes_filter = {
                line.strip() for line in gf
                if line.strip() and not line.startswith("#")
            }
        logger.info("Gene filter: %d gene(s) from %s", len(genes_filter), args.genes_file)
        if not genes_filter:
            print(
                f"ERROR: --genes-file {args.genes_file} contains no usable gene "
                f"names (only comments/blank lines). Provide at least one gene "
                f"symbol, or omit --genes-file to use all BED regions.",
                file=sys.stderr,
            )
            sys.exit(1)
    try:
        regions = load_bed_regions(args.bed, genes=genes_filter)
    except ValueError as exc:
        print(f"ERROR: malformed BED file — {exc}", file=sys.stderr)
        sys.exit(1)
    if not regions:
        logger.error("No regions loaded from %s", args.bed)
        sys.exit(1)
    logger.info("Loaded %d region(s) from %s", len(regions), args.bed)

    # ── Filter regions for chromosomes with BCF files ────────────────────
    if _is_persample_bcf_dir(args.bcf_dir):
        # Per-sample multi-contig files: available chromosomes come from the
        # BCF header (seqnames), not the filename. The full source contig list
        # is retained in the header, so chromosomes with no variants for a
        # given sample still resolve to an empty (non-crashing) query.
        from cyvcf2 import VCF
        _rep = sorted(args.bcf_dir.glob("*.bcf"))[0]
        with VCF(str(_rep)) as _v:
            available_chroms = set(_v.seqnames)
    else:
        available_chroms = {
            m.group(1)
            for p in args.bcf_dir.glob("*.bcf")
            if (m := _CHROM_PAT.search(p.name))
        }
    region_chroms = {r[0] for r in regions}
    missing_chroms = region_chroms - available_chroms
    if missing_chroms:
        chrom_counts = defaultdict(int)
        for r in regions:
            if r[0] in missing_chroms:
                chrom_counts[r[0]] += 1
        for chrom in sorted(chrom_counts):
            logger.warning(
                "No BCF for '%s' in %s — excluding %d region(s)",
                chrom, args.bcf_dir, chrom_counts[chrom],
            )
        regions = [r for r in regions if r[0] not in missing_chroms]
        logger.info("After BCF-chromosome filter: %d region(s) remain", len(regions))

    # Empty after the BCF-chromosome filter is a hard error: proceeding would
    # write a degenerate 0-record HG38_REF shard and empty sample shards that
    # look like a clean run. Fail fast with the cause instead.
    if not regions:
        print(
            "ERROR: no regions remain after the BCF-chromosome filter — none of "
            f"the BED chromosomes have a matching BCF in {args.bcf_dir}. Check "
            "that --bcf-dir and --bed use the same chromosome naming "
            "(e.g. 'chr1' vs '1').",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.max_regions is not None:
        regions = regions[: args.max_regions]
        if not regions:
            print(
                f"ERROR: no regions remain after --max-regions {args.max_regions} "
                f"truncation — it must be a positive integer (or omit it to use "
                f"all regions).",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── Filter already-completed samples ────────────────────────────────
    def _count_existing(output_dir: Path) -> set:
        return {p.stem for p in output_dir.glob("*.parquet")}

    if test_samples_set:
        existing = _count_existing(train_output) | _count_existing(test_output)
    else:
        existing = _count_existing(train_output)
    pending = [s for s in samples if s not in existing]
    logger.info(
        "%d samples total, %d already done, %d pending",
        len(samples), len(existing), len(pending),
    )

    # ── Clean stale .tmp files if requested ─────────────────────────────
    if args.clean_stale:
        if args.stale_age_hours is not None:
            warnings.warn(
                "--stale-age-hours is ignored: an age cannot be read off an NFS "
                "mtime (the /storage filer clock runs ahead of the host); "
                "--clean-stale watches each claim for --stale-observe-seconds.",
                DeprecationWarning, stacklevel=2)
        stale_dirs = [train_output]
        if test_samples_set:
            stale_dirs.append(test_output)
        stale_count = _clean_stale_tmp(stale_dirs, observe_s=args.stale_observe_seconds)
        if stale_count:
            logger.info("Removed %d stale .tmp file(s)", stale_count)

    # ── Register PhyloP (when --phylop-bw is given) ─────────────────────
    # Clear ANNOTATIONS before registering so a programmatic second call to
    # cmd_generate doesn't accumulate duplicate entries.
    ANNOTATIONS.clear()
    _rebuild_pa_schema()
    # Done BEFORE provenance is written so the sidecar records the
    # actually-used source file.
    _register_phylop_annotation(args)

    # ── Register CADD (when --cadd-arrow-dir is given) ─────────────────
    # Presence-enables — no separate flag.
    if args.cadd_arrow_dir is not None:
        _register_cadd_annotation(args, regions, args.output)

    # ── Sidecar contamination guard (refuse to mix representations OR
    #    versions — the preset's schema_version is recorded, so a v5 resume
    #    into a v6 shard dir is rejected). marker_policy is invariant (NAN)
    #    across presets today; it is recorded from the preset so a future
    #    divergence is auditable. (If a future preset sets a non-NAN policy,
    #    the runtime annotation-alignment reads of ACTIVE_MARKER_POLICY in this
    #    module must also be made to follow the preset.) ─────────────────────
    #
    # The guard ALSO verifies a fingerprint of the content-determining inputs
    # (built here from the resolved args): a resume into the same dir with a
    # different BED, reference, variant source, token map, selection file, or
    # truncation knob is rejected rather than silently mixed. Small files use a
    # content hash (move-tolerant); the multi-hundred-MB reference uses
    # path+size; the BCF directory uses its resolved path.
    inputs_fp = {
        "bed_sha256":               bed_sha256,  # computed once at validation
        "ref":                      f"{args.ref.resolve()}|{args.ref.stat().st_size}",
        "bcf_dir":                  str(args.bcf_dir.resolve()),
        "tokens_sha256":            _file_sha256(args.tokens),
        "genes_file_sha256":        _file_sha256(args.genes_file),
        "samples_file_sha256":      _file_sha256(args.samples_file),
        "test_samples_file_sha256": _file_sha256(args.test_samples_file),
        "max_samples":              args.max_samples,
        "max_regions":              args.max_regions,
        "emit_ref_index":           args.emit_ref_index,
    }
    # A resume is any invocation where this output dir already carries a
    # generation sidecar from a prior run. Captured BEFORE the guard call, which
    # creates the sidecar on a fresh run.
    is_resume = (train_output / _GEN_PROVENANCE_FILENAME).exists()
    _check_or_write_generation_provenance(
        train_output,
        representation=args.representation,
        force_unphased=force_unphased,
        use_direction_markers=use_direction_markers,
        schema_version=preset.schema_version,
        marker_policy=preset.marker_policy.value,
        externalize_markers=preset.externalize_markers,
        inputs=inputs_fp,
    )
    if test_samples_set:
        _check_or_write_generation_provenance(
            test_output,
            representation=args.representation,
            force_unphased=force_unphased,
            use_direction_markers=use_direction_markers,
            schema_version=preset.schema_version,
            marker_policy=preset.marker_policy.value,
            externalize_markers=preset.externalize_markers,
            inputs=inputs_fp,
        )

    # ── Run-documentation: params + a _finalize_run closure ──────────────
    # All params (incl. benign ones like workers/compression) are documented in
    # the sidecar's `runs[]` array — one record per invocation, including every
    # resume — so an on-disk shard set is traceable to how it was produced. This
    # is independent of the contamination guard above (which never reads runs).
    run_params = {
        "version": args.version,
        "schema_version": preset.schema_version,
        "representation": args.representation,
        "force_unphased": force_unphased,
        "use_direction_markers": use_direction_markers,
        "externalize_markers": externalize_markers,
        "emit_ref_index": args.emit_ref_index,
        "bed": str(args.bed),
        "ref": str(args.ref),
        "bcf_dir": str(args.bcf_dir),
        "tokens": str(args.tokens) if args.tokens else None,
        "samples_file": str(args.samples_file) if args.samples_file else None,
        "test_samples_file": str(args.test_samples_file) if args.test_samples_file else None,
        "genes_file": str(args.genes_file) if args.genes_file else None,
        "phylop_bw": str(args.phylop_bw) if args.phylop_bw else None,
        "cadd_arrow_dir": str(args.cadd_arrow_dir) if args.cadd_arrow_dir else None,
        "workers": args.workers,
        "inner_threads": 1,  # fixed — see write_sample_shard comment
        "compression": args.compression,
        "compression_level": args.compression_level,
        "max_samples": args.max_samples,
        "max_regions": args.max_regions,
        "profile": bool(getattr(args, "profile", False)),
        "no_dedup_inline": bool(getattr(args, "no_dedup_inline", False)),
    }

    def _finalize_run(status: str, **milestones) -> None:
        """Append one run record to the generation sidecar(s).

        Called at every terminal point of cmd_generate so each invocation —
        success, failure, or an all-done no-op resume — leaves exactly one dated
        record. The same record is written to both the train and (if active)
        test sidecars, since the run produced both.
        """
        finished = datetime.datetime.now()
        record = {
            "command": " ".join(sys.argv),
            "started_at": gen_started_str,
            "finished_at": f"{finished:%Y-%m-%d %H:%M:%S}",
            "duration_seconds": round(time.perf_counter() - t_start_gen, 1),
            "status": status,
            "resume": is_resume,
            "params": run_params,
            "milestones": {
                "n_regions": len(regions),
                "n_samples": len(samples),
                "n_pending_at_start": len(pending),
                **milestones,
            },
        }
        _append_run_record(train_output / _GEN_PROVENANCE_FILENAME, record)
        if test_samples_set:
            _append_run_record(test_output / _GEN_PROVENANCE_FILENAME, record)

    # Skip the 3 GB reference load entirely when there is nothing to do.
    ref_shard_path = train_output / f"{_REF_SAMPLE_ID}.parquet"
    if not pending and ref_shard_path.exists():
        print("All samples already completed. Nothing to do.", flush=True)
        _finalize_run("noop_all_present", n_done=0,
                      n_skipped=len(existing), n_failed=0)
        return

    # ── Pre-load reference genome (shared across workers via fork COW) ──
    global _shared_processor
    logger.info("Pre-loading reference genome from %s …", args.ref)
    _shared_processor = GenomeSequenceProcessor(
        ref_path=args.ref, diploid_tokens_path=args.tokens,
        force_unphased=force_unphased,
        use_direction_markers=use_direction_markers,
        emit_anchor_snp_marker=preset.emit_anchor_snp_marker,
    )
    logger.info("Reference genome loaded (shared across workers)")

    # ── Generate HG38_REF shard (reference genome, no variants) ─────────
    # The reference pseudo-sample is upweighted (×10) and every BED region
    # depends on it, so a failure here must abort the run rather than slip by
    # and silently yield a dataset missing the reference.
    ref_status = write_reference_shard(
        processor=_shared_processor,
        regions=regions,
        output_dir=train_output,
        compression=args.compression,
        compression_level=args.compression_level,
        representation=args.representation,
    )
    if ref_status == _STATUS_FAILED:
        print(
            f"ERROR: {_REF_SAMPLE_ID} reference shard failed to generate — "
            f"aborting (see the [FAIL] {_REF_SAMPLE_ID} log line above).",
            file=sys.stderr, flush=True,
        )
        _finalize_run("failed_ref_shard", n_done=0,
                      n_skipped=len(existing), n_failed=0)
        sys.exit(1)
    if ref_status == _STATUS_CLAIMED:
        # Not fatal (another run is probably writing it), but the reference
        # pseudo-sample backs every BED region — say so rather than let its
        # absence look like a completed run.
        print(
            f"NOTE: {_REF_SAMPLE_ID} was not written by this run — its "
            f".parquet.tmp is claimed by another process. Assemble only once "
            f"that shard exists.",
            file=sys.stderr, flush=True,
        )

    if not pending:
        print("All samples already completed. Nothing to do.", flush=True)
        _finalize_run("noop_all_present", n_done=0,
                      n_skipped=len(existing), n_failed=0)
        return

    # ── Create per-split live cross-worker dedup tables ─────────────────
    # A shared (h1, h2) hash set per split, grown live as samples finalize, so
    # cross-sample dedup works on a FRESH run (not only on resume).  Per split
    # (train vs test) so a skipped occurrence's weight stays in its own split.
    # Set on the dedup module BEFORE the Pool forks so workers inherit it.
    dedup._GEN_DEDUP = None
    _dedup_shms: List[SharedMemory] = []
    # The live tables are POSIX shared memory, i.e. /dev/shm — the same shared,
    # machine-wide tmpfs the assemble path guards so carefully, and multi-GB at
    # canonical-BED scale. Two things follow, and generate previously did
    # neither: the allocation must be admission-checked (a segment larger than
    # free tmpfs faults on first touch, and filling the tmpfs breaks every other
    # process on the box), and it must be released on EVERY exit path, not only
    # the happy one — an exception or Ctrl-C between here and the unlink below
    # used to strand the whole table until reboot.
    _gen_owner_pid = os.getpid()

    def _release_dedup_shms() -> None:
        if os.getpid() != _gen_owner_pid:
            return                      # a forked worker: not ours to unlink
        while _dedup_shms:
            _shm = _dedup_shms.pop()
            try:
                _shm.close()
            except Exception:
                pass
            try:
                _shm.unlink()
            except Exception:
                pass

    def _release_and_reraise(signum, _frame):
        _release_dedup_shms()
        _prev = _dedup_prev_handlers.get(signum)
        signal.signal(signum, _prev if _prev is not None else signal.SIG_DFL)
        os.kill(os.getpid(), signum)    # preserve the true exit status

    _dedup_prev_handlers: Dict[int, object] = {}
    if not getattr(args, "no_dedup_inline", False):
        dedup._GEN_DEDUP = {}
        splits = [("train", train_output)]
        if test_samples_set:
            splits.append(("test", test_output))
        atexit.register(_release_dedup_shms)
        for _sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                _dedup_prev_handlers[_sig] = signal.signal(
                    _sig, _release_and_reraise)
            except (ValueError, OSError):
                pass    # not on the main thread — try/finally + atexit still apply
        for split, split_dir in splits:
            n_pending_split = sum(
                1 for s in pending
                if (bool(test_samples_set and s in test_samples_set)) == (split == "test")
            )
            want_cap = _dedup_table_capacity(
                n_pending_split, len(regions),
                getattr(args, "dedup_capacity", None),
            )
            # Shrink (or give up) rather than fault: a smaller table just defers
            # more duplicates to the authoritative assemble dedup, which is
            # already the documented behaviour when a table fills.
            cap = _fit_dedup_capacity(want_cap)
            if cap == 0:
                print(f"[DEDUP] {split}: skipping the live table — {_SHM_ROOT} "
                      f"cannot hold even {_DEDUP_MIN_CAPACITY:,} slots "
                      f"({_fmt_bytes(_DEDUP_MIN_CAPACITY * 16)}) within its "
                      f"{_SHM_RESERVE_FRAC:.0%} reserve. All dedup falls to the "
                      f"assemble step (no data loss).", flush=True)
                continue
            if cap < want_cap:
                print(f"[DEDUP] {split}: shrank the live table from "
                      f"{want_cap:,} to {cap:,} slots to stay inside "
                      f"{_SHM_ROOT}'s {_SHM_RESERVE_FRAC:.0%} reserve; the "
                      f"overflow dedups at assemble time.", flush=True)
            shm = SharedMemory(create=True, size=cap * 2 * 8)
            _dedup_shms.append(shm)     # registered BEFORE the first page touch
            arr = np.ndarray((cap * 2,), dtype=np.uint64, buffer=shm.buf)
            arr.fill(0)
            lock = multiprocessing.Lock()
            size = multiprocessing.RawValue("q", 0)
            full = multiprocessing.RawValue("b", 0)
            dedup._GEN_DEDUP[split] = {
                "shm_name": shm.name, "capacity": cap,
                "lock": lock, "size": size, "full": full,
            }
            print(f"[DEDUP] {split}: live table {cap:,} slots "
                  f"({cap * 16 / 1e9:.2f} GB)", flush=True)
            # Pre-seed from existing variant shards (resume) so this run's
            # samples dedup against prior ones.  HG38_REF is excluded — its
            # marker-free sequences never match variant sequences.
            existing_shards = [str(p) for p in split_dir.glob("*.parquet")
                               if p.stem != _REF_SAMPLE_ID]
            if existing_shards:
                t_seed = time.perf_counter()
                table = _LiveDedupTable(arr, cap, lock, size, full)
                # imap_unordered, not map: the parent folds each shard's hashes
                # into the table as they arrive, so map()'s full result list —
                # every shard's (h1, h2, weights) held at once, tens of GB on a
                # late resume — buys nothing. Worker count is capped like the
                # assemble hash step: each one materializes a whole shard's
                # `seq` column as Python strings, so this pool is sized by
                # per-worker memory, not by core count.
                n_seed_workers = min(args.workers, _HASH_POOL_MAX_PROC,
                                     len(existing_shards))
                n_seed = len(existing_shards)
                seed_bytes = sum(os.path.getsize(f) for f in existing_shards)
                # TELEMETRY, and it is not decoration. This loop re-reads every
                # existing shard: on the 2026-09-03 v7-tsstes backfill that was
                # 2,559 shards / 9.7 TB / ~2.5 h at 100% CPU on every worker,
                # with NOT ONE line of output until it finished. From outside it
                # is indistinguishable from the finalization deadlock this same
                # command hit the day before. Any loop that can run for hours
                # must say so while it runs -- see MONITOR.md.
                print(f"[DEDUP] {split}: pre-seeding the live table from "
                      f"{n_seed:,} existing shard(s), "
                      f"{seed_bytes / 1e12:.2f} TB, {n_seed_workers} workers "
                      f"— this re-reads every shard and dominates a resume; "
                      f"--no-dedup-inline skips it", flush=True)
                t_report = t_seed
                done_shards = 0
                # Progress is counted in SHARDS, not bytes: imap_unordered
                # yields out of order, so a result cannot be attributed to a
                # particular input path. Shards here are one sample x one region
                # set and near-uniform (~4 GB), so shards-done is a faithful
                # proxy and the MB/s below is derived from the mean shard size
                # rather than pretending to per-shard precision.
                mean_bytes = seed_bytes / n_seed
                with Pool(processes=n_seed_workers) as pool:
                    for h1, h2, _w in pool.imap_unordered(
                            _hash_shard_pair, existing_shards):
                        table.add_batch(zip(h1.tolist(), h2.tolist()))
                        done_shards += 1
                        now = time.perf_counter()
                        if now - t_report >= _SEED_REPORT_S:
                            el = now - t_seed
                            frac = done_shards / n_seed
                            rate = done_shards * mean_bytes / el if el else 0.0
                            eta = (el / frac - el) if frac > 0 else float("nan")
                            print(f"  [DEDUP] {split}: pre-seed "
                                  f"{done_shards:,}/{n_seed:,} shards "
                                  f"({100 * frac:.1f}%), "
                                  f"{size.value:,} hashes, "
                                  f"~{rate / 1e6:.0f} MB/s, "
                                  f"elapsed {el / 60:.0f}m, "
                                  f"ETA {eta / 60:.0f}m", flush=True)
                            t_report = now
                print(f"[DEDUP] {split}: pre-seeded from "
                      f"{len(existing_shards)} shards ({size.value:,} hashes, "
                      f"{time.perf_counter()-t_seed:.1f}s, "
                      f"{n_seed_workers} workers)", flush=True)

    # Everything from here to the release below runs under try/finally: the
    # live-dedup segments are /dev/shm bytes, and an exception or Ctrl-C in the
    # worker pool must not strand them for the rest of the machine's uptime.
    try:
        # ── Build worker arguments ──────────────────────────────────────────
        worker_args = []
        for sample_name in pending:
            if test_samples_set and sample_name in test_samples_set:
                out_dir = test_output
                split = "test"
            else:
                out_dir = train_output
                split = "train"
            worker_args.append((
                sample_name,
                str(args.bcf_dir),
                regions,
                str(args.ref),
                str(args.tokens) if args.tokens is not None else None,
                str(out_dir),
                args.compression,
                args.compression_level,
                force_unphased,
                use_direction_markers,
                args.representation,
                split,
                preset.emit_anchor_snp_marker,
            ))

        # ── Run workers ─────────────────────────────────────────────────────
        t_start = time.perf_counter()
        n_workers = min(args.workers, len(pending))

        # Profiling mode: single-process, first sample only, PerfAccumulator enabled.
        profile_enabled = getattr(args, "profile", False)
        if profile_enabled:
            n_workers = 1
            worker_args = worker_args[:1]
            _shared_processor._perf = PerfAccumulator()
            print(
                "\n[PROFILE] Profiling mode enabled — processing 1 sample with PerfAccumulator.",
                flush=True,
            )

        logger.info("Starting %d worker(s) for %d sample(s)", n_workers, len(pending))

        if n_workers == 1:
            # Single-process mode (easier to debug / profile)
            results = [_worker_fn(wa) for wa in worker_args]
        else:
            with Pool(processes=n_workers) as pool:
                results = list(pool.imap_unordered(_worker_fn, worker_args))

        if profile_enabled and _shared_processor._perf is not None:
            report = _shared_processor._perf.report()
            total_s = sum(v["total_s"] for v in report.values())
            print(f"\n{'=' * 70}")
            print(f"[PROFILE] PerfAccumulator breakdown  (total instrumented: {total_s:.2f}s)")
            print(f"{'=' * 70}")
            hdr = f"  {'Function':<30} {'Count':>6} {'Total(s)':>9} {'Mean(ms)':>9} {'Med(ms)':>9} {'P95(ms)':>9} {'Max(ms)':>9}"
            print(hdr)
            print(f"  {'-'*30} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
            for fn, s in sorted(report.items(), key=lambda x: -x[1]["total_s"]):
                print(
                    f"  {fn:<30} {s['count']:>6} {s['total_s']:>9.3f} "
                    f"{s['mean_ms']:>9.3f} {s['median_ms']:>9.3f} "
                    f"{s['p95_ms']:>9.3f} {s['max_ms']:>9.3f}"
                )
            print("\n  Percentage breakdown:")
            for fn, s in sorted(report.items(), key=lambda x: -x[1]["total_s"]):
                pct = s["total_s"] / total_s * 100 if total_s > 0 else 0
                bar = "#" * int(pct / 2)
                print(f"    {fn:<30} {pct:5.1f}%  {bar}")
            print(f"{'=' * 70}")

        # Surface inline-dedup table overflow. _LiveDedupTable.add_batch logs a
        # WARNING when a split's table fills, but the default log level is ERROR
        # (so it's invisible) and the warning is per-worker. The deferral to the
        # assemble dedup is correct-by-design; make it observable with a stdout
        # notice for any split whose table filled, before the descriptors are torn
        # down below.
        if dedup._GEN_DEDUP:
            for _split, _info in dedup._GEN_DEDUP.items():
                if _info["full"].value:
                    print(
                        f"[WARN] inline dedup table for '{_split}' filled "
                        f"({_info['capacity']:,} slots); remaining duplicates were "
                        f"deferred to the assemble dedup (no data loss). Use a larger "
                        f"--dedup-capacity to dedup more during generate.",
                        flush=True,
                    )
    finally:
        # Release the inline-dedup shared memory — on every exit path, and
        # exactly once (_release_dedup_shms drains the list). Then disarm the
        # atexit/signal failsafes so they cannot fire against a later run.
        dedup._GEN_DEDUP = None
        dedup._WORKER_DEDUP_TABLES.clear()
        _release_dedup_shms()
        atexit.unregister(_release_dedup_shms)
        for _sig, _prev in _dedup_prev_handlers.items():
            if _prev is None:
                continue    # not a Python-installed handler; nothing to restore
            try:
                signal.signal(_sig, _prev)
            except (ValueError, OSError):
                pass

    # Three-way tally: a crashed sample (FAILED) must never be reported as a
    # deliberate skip. Failures are listed by name and force a non-zero exit
    # so a resume/CI run can't mistake a partial generation for a clean one.
    n_done, n_skipped, failed_samples = _tally_worker_results(results)
    # Completeness guard: a worker killed by the OOM killer or a C-extension
    # segfault returns no status, so the sample is absent from `results` rather
    # than FAILED. Count missing submitted samples as failures so they can't
    # vanish from the tally. Use the actually-submitted set (not `pending`) so
    # --profile, which submits a single sample, is handled correctly.
    submitted_names = [wa[0] for wa in worker_args]
    failed_samples = failed_samples + _missing_samples(submitted_names, results)
    n_failed = len(failed_samples)
    claimed_samples = _claimed_samples(results)
    elapsed = time.perf_counter() - t_start

    print(
        f"\nGeneration complete in {elapsed:.0f}s — "
        f"done: {n_done}, skipped (already present): "
        f"{n_skipped - len(claimed_samples)}, "
        f"skipped (.tmp claimed elsewhere): {len(claimed_samples)}, "
        f"failed: {n_failed}",
        flush=True,
    )
    # A claimed sample has NO shard: this run wrote nothing for it. That is
    # expected while a concurrent generate holds the claim, and a silent hole in
    # the dataset when the claim is debris from a dead run. Either way it must
    # not read as "already present".
    if claimed_samples:
        _preview = ", ".join(claimed_samples[:20])
        _more = "" if len(claimed_samples) <= 20 else (
            f" (+{len(claimed_samples) - 20} more)")
        print(
            f"NOTE — {len(claimed_samples)} sample(s) produced no shard because "
            f"a .parquet.tmp claim already existed: {_preview}{_more}. If "
            f"another generate is running against this --output that is "
            f"expected; otherwise those claims are debris — re-run with "
            f"--clean-stale to release them (a claim is removed only after it "
            f"stays untouched for --stale-observe-seconds).",
            file=sys.stderr, flush=True,
        )
    if n_failed:
        _preview = ", ".join(failed_samples[:20])
        _more = "" if n_failed <= 20 else f" (+{n_failed - 20} more)"
        print(
            f"FAILED — {n_failed} sample(s) errored and wrote no shard: "
            f"{_preview}{_more}. Re-run to retry them (failed samples leave no "
            f".parquet, so a resume picks them up).",
            file=sys.stderr, flush=True,
        )
    print(f"Generate finished at {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
          f"(total {elapsed:.1f}s)", flush=True)
    # Document this invocation BEFORE the failure exit, so a failed-but-tallied
    # run is recorded too. n_done vs n_skipped distinguishes what this resume
    # actually produced from what was already present.
    _finalize_run(
        "failed" if n_failed else "ok",
        n_done=n_done,
        n_skipped=n_skipped,
        n_claimed=len(claimed_samples),
        n_failed=n_failed,
        failed_samples=failed_samples[:50],
        claimed_samples=claimed_samples[:50],
    )
    if n_failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI — assemble subcommand
# ---------------------------------------------------------------------------


# Dedup machinery (hashing, assemble dedup, live inline-dedup table, skip-weight
# sidecars) lives in dedup.py and is imported at the top of this module. The
# mutable cross-fork globals _GEN_DEDUP / _WORKER_DEDUP_TABLES are addressed as
# dedup._GEN_DEDUP so the value forked children inherit is the one dedup exposes.

# Step-5 `save_to_disk` parallelism cap.  Unlike `load_dataset` (parquet decode,
# CPU-bound, scales to core count), the save materializes a shuffled dataset via
# random-access gather over a memory-mapped Arrow store larger than page cache.
# That is random-read-IOPS bound, not CPU bound: extra processes raise the I/O
# queue depth (and overlap read-wait with compress/write) only up to the
# storage's parallel-read ceiling, then thrash the page cache.  Cap the save at
# this many workers; `load_dataset` keeps the full `n_workers`.  Tune per storage
# (NVMe arrays may want more; spinning/NAS less).
# Overridable so the cap can be tuned per storage without editing code — the
# comment above says "tune per storage", and on 2026-09-04 the 1kG train save
# was measured at 30,673 s (50.8% of a 16.8 h assemble) with no evidence that
# 16 is right for this NFS filer in either direction. A valid experiment needs
# a dataset larger than page cache per configuration (2 TB+ on B300), so it is
# not a cheap one; do not raise this on intuition.
_SAVE_TO_DISK_MAX_PROC = int(os.environ.get("DIPLOID_SAVE_MAX_PROC", "16"))


# A *.parquet.tmp is not merely leftover state — it IS the O_CREAT|O_EXCL claim
# that keeps two workers off the same sample (see write_sample_shard). Deleting a
# live one lets a second run re-claim that sample while the first is still
# writing: the first then writes to an unlinked inode and dies at rename().
# Multiple generates against one --output are a supported pattern (the module
# docstring advertises it) and every runner script passes --clean-stale, so the
# sweep must distinguish abandoned claims from live ones. Change over a window
# is the proxy: ParquetWriter appends a row group every batch (5000 records or
# _MAX_BATCH_SEQ_BYTES, see write_shard_streaming) and _ClaimHeartbeat touches the
# claim every 60 s, so a live .tmp's (size, mtime) keeps moving, while an
# abandoned claim's is frozen at the moment its process died. Not age: /storage
# mtimes are filer-stamped and run ahead of the host clock. (Assemble's temp
# dirs carry pid stamps instead — they are directories we own end-to-end, not
# single-file claims handed between runs.)
_STALE_OBSERVE_S = 3 * _CLAIM_HEARTBEAT_S   # two consecutive missed heartbeats ⇒ abandoned


def _clean_stale_tmp(dirs: List[Path], observe_s: float = _STALE_OBSERVE_S,
                     sleep=time.sleep) -> int:
    """Remove abandoned *.parquet.tmp shards and skip_weights/*.parquet.tmp sidecars.

    Clock-free: each .tmp is signed as (size, mtime_ns, inode), ``observe_s``
    passes, and only a byte-identical signature is debris. A live claim is
    heartbeated every ``_CLAIM_HEARTBEAT_S`` (and grows as batches land), so it
    cannot read unchanged across three beats. ``now - mtime`` is NOT usable here:
    /storage mtimes are stamped by the filer, which runs minutes ahead of the
    host (HARDENING_PLAN.md C1), so an age threshold was a silent no-op.
    Finalized *.parquet files are never touched. Returns the number removed.
    ``sleep`` is injectable so tests do not wait.
    """
    def _sig(p: Path):
        try:
            st = p.stat()
        except OSError:
            return None                 # vanished under us — nothing to clean
        return (st.st_size, st.st_mtime_ns, st.st_ino)

    tmps: List[Path] = []
    for d in dirs:
        tmps += list(d.glob("*.parquet.tmp"))
        sub = d / _SKIP_WEIGHTS_SUBDIR
        if sub.is_dir():
            tmps += list(sub.glob("*.parquet.tmp"))
    before = {p: sig for p in tmps if (sig := _sig(p)) is not None}
    if not before:
        return 0
    if observe_s > 0:
        msg = (f"--clean-stale: watching {len(before)} .parquet.tmp claim(s) for "
               f"{observe_s:.0f}s (a live claim heartbeats every "
               f"{_CLAIM_HEARTBEAT_S:.0f}s)")
        logger.info(msg)
        print(f"  {msg}", flush=True)
        sleep(observe_s)
    removed = kept = 0
    for tmp, s0 in before.items():
        s1 = _sig(tmp)
        if s1 is None:
            continue
        if s1 != s0:
            kept += 1
            logger.info("Keeping %s — changed during the window (%s -> %s); another "
                        "generate is still writing this sample", tmp.name, s0, s1)
            continue
        logger.info("Removing stale tmp: %s (unchanged over %.0fs, signature %s)",
                    tmp.name, observe_s, s0)
        try:
            tmp.unlink()
        except OSError:
            continue
        removed += 1
    if kept:
        print(f"  --clean-stale: kept {kept} live .tmp claim(s); removed {removed}",
              flush=True)
    return removed


# ---------------------------------------------------------------------------
# Step-4 temp-space accounting.
#
# Step 4 writes one UNCOMPRESSED Parquet file per input shard (see
# _filter_write_shard). The inputs are lz4-compressed, so their on-disk size is
# roughly HALF of what gets written — sizing the free-space preflight on
# `stat().st_size` under-counts by ~2x and lets the check pass when there is not
# enough room. The correct basis is the Parquet footer's uncompressed byte count
# (post-encoding, pre-compression), which is exactly what compression="none"
# writes and is free to read since Step 1 already opens every footer.
#
# /dev/shm is a shared, system-wide tmpfs on multi-user boxes: filling it breaks
# every other process on the machine, not just this job. So we (a) size on the
# right basis, (b) always leave _SHM_RESERVE_FRAC of the tmpfs for other users,
# and (c) re-check that reserve before EVERY shard write rather than once at
# startup, since another process can consume the headroom mid-run.
#
# CONCURRENCY — one assemble per machine, parallelism only inside a run.
# ----------------------------------------------------------------------
# cmd_assemble takes a machine-wide lock (_acquire_assemble_lock) and refuses to
# start if another assemble holds it. Parallelism within a run is unaffected:
# Step 4's worker Pool, load_dataset(num_proc=) and save_to_disk(num_proc=) all
# run exactly as before. Only whole assembles are serialised, which costs
# little — an assemble is minutes against a generate's days.
#
# This is what makes the space arithmetic above sound rather than approximate.
# Every budget here is scoped to ONE run: _estimate_temp_bytes sizes this run's
# shards, and _write_shard_with_fallback charges each write this run's whole
# worker fleet so its N workers cannot collectively exceed the reserve. A second
# concurrent assemble would invalidate all of it — both would size against the
# same 10% headroom and together blow through it. The lock removes that case
# instead of trying to price it.
#
# Two further hazards it forecloses, both previously real:
#   * two runs writing temp files named after their source shard
#     (<sample>.parquet, identical across dataset versions) into one directory,
#     each then loading a mix of both runs' rows — silent corruption;
#   * a starting run clearing a live run's finished temp shards.
# _shm_temp_dir_for (path-digest naming) and _claim_temp_dir (live-owner
# refusal) still guard both, as defence in depth and because
# _process_split_streaming is also called directly, outside cmd_assemble.
#
# To allow concurrent assembles later, the lock is not the only thing to drop:
# temp filenames would have to be per-run unique, the reserve arithmetic in
# _write_shard_with_fallback would have to account for every concurrent run's
# fleet rather than one, and output-directory ownership would need arbitrating.
# ---------------------------------------------------------------------------

_SHM_RESERVE_FRAC = 0.10    # never consume the last 10% of a shared tmpfs
_TEMP_SAFETY_FACTOR = 1.25  # per-shard skew + marker rehydration growth (v8->v6/v7)
_SHM_ROOT = Path("/dev/shm")

# Cap for the pools that hash whole Parquet shards (assemble Step 2 and the
# generate-time dedup pre-seed). Each worker materializes one shard's `seq`
# column as Python strings, which is GB-scale for LD-block regions, so this pool
# is sized by per-worker memory rather than by core count.
_HASH_POOL_MAX_PROC = 32

# Below this the live inline-dedup table stops being worth its /dev/shm bytes;
# the assemble dedup is authoritative regardless. 1M slots = 16 MB.
_DEDUP_MIN_CAPACITY = 1 << 20
_SHM_PREFIX = ".diploid_assemble_"
_DISK_TEMP_SUFFIX = "_dedup_temp"   # <output>.parent/.<split>_dedup_temp

# Split-completion marker (HARDENING_PLAN.md §A4). `dataset_info.json` is NOT a
# completion signal: `save_to_disk(num_proc=…)` writes it incrementally, so a
# crash mid-save leaves a plausible-looking partial output behind. This marker is
# ours, written atomically as the LAST act of a split, and is the only thing the
# split-level resume trusts.
_SPLIT_MARKER_NAME = ".assemble_complete.json"
_SPLIT_MARKER_VERSION = 1

# ---------------------------------------------------------------------------
# int32 list-offset cap on the Step-4 temp shards
#
# Arrow addresses a `list<T>` column's child array with **int32** offsets, so any
# single contiguous list array holds at most 2^31-1 elements. The Parquet format
# has no such limit (it stores repetition levels, not an offset buffer), so the
# cap only bites when a row group is decoded back into Arrow — and pyarrow
# materializes a **whole row group** before it slices it into batches. A row
# group over the cap therefore cannot be read back AT ALL: `to_batches()` raises
# `OSError: List index overflow` before the first batch is yielded, and no
# reader-side `batch_size` (nor a `large_list` cast, which can only run on an
# array that already exists) can rescue it. Both were measured, not assumed.
#
# Which Arrow type a shard comes back as is decided by the pyarrow schema stored
# in the file, and it does round-trip `large_list`. So declaring the annotation
# columns `large_list` at GENERATE time would also work — but it buys nothing
# here: Step 4 rewrites every shard anyway, so the cap has to be respected at
# that write regardless, and regenerating 3,202 shards to reach the same place
# costs days.
#
# Step 4 used to call `pq.write_table(...)` with no `row_group_size`, which
# writes each temp shard as ONE row group. For `tss_tes_gencode50_tx128` that is
# ~2.28e9 `phylop` elements per shard — one uint8 per base, 2.31 Gbp of regions
# per sample — i.e. 1.06x the cap, so most shards became unreadable and the
# first v7-tsstes assemble died 6 h in at Step 5. Shards that happened to lose
# enough rows to dedup fell under the cap and read fine, which is why it failed
# partway rather than immediately.
#
# `_safe_row_group_size` sizes the row group from the widest list actually in
# the shard, so it adapts to the region set instead of encoding one region set's
# geometry as a constant. It returns None — meaning "keep the pyarrow default" —
# whenever the default is already provably safe, so region sets that never
# approached the cap (v4/v6 have no list columns at all) keep byte-identical
# temp shards.
# ---------------------------------------------------------------------------

_LIST_OFFSET_CAP = 2**31 - 1        # max elements in one Arrow list<T> array
_ROW_GROUP_HEADROOM = 2             # target <= half the cap per row group
_PQ_DEFAULT_ROW_GROUP_ROWS = 1 << 20   # pyarrow's own default (1Mi rows)


def _large_list_features(parquet_file: Path):
    """Features for the Step-5 load, with every list column widened to LargeList.

    `_safe_row_group_size` keeps the *temp shards* readable; this keeps the
    *assembled dataset* readable. Both are needed, for different reasons:

      * `datasets.ArrowWriter._write_table` calls `combine_chunks()` on the table
        it is handed, and `Dataset._save_to_disk_single` feeds it a 1000-row
        batch whose size is hardcoded (`config.DEFAULT_MAX_BATCH_SIZE`) with no
        parameter to override it. 1000 of the widest `tss_tes_gencode50_tx128`
        rows is ~2.49e9 `phylop` elements, over the int32 cap — safe today only
        because `.shuffle()` makes an all-giants batch statistically unreachable,
        which is luck, not a guarantee.
      * The saved Arrow files inherit whichever offset width this declares, so
        `list` here would hand the same latent cap to every downstream reader
        (the pretrain dataloader, `Dataset.map`, `compare_assembled`).

    `LargeList` (int64 offsets) removes both. It costs 4 extra bytes per row per
    list column and changes nothing a caller can observe: HF decodes `large_list`
    to the same Python lists, so per-row hashes — and `compare_assembled`
    fingerprints, which hash decoded values — are unchanged.

    Derived from the shard schema rather than hardcoded because the schema is
    built at run time from the active annotations and the v8 marker flag
    (`_rebuild_pa_schema`), so a literal here would silently rot. Note the
    dictionary-encoded `sample_id`/`chrom` widen to plain `string`: that is what
    the parquet builder's own `Features.from_arrow_schema` already does when
    `features=None`, so the assembled columns are byte-for-byte what they were.
    """
    import datasets   # heavy; caller imports it lazily too

    def widen(field: pa.Field) -> pa.Field:
        if pa.types.is_list(field.type):
            return pa.field(field.name, pa.large_list(field.type.value_field),
                            nullable=field.nullable)
        return field

    schema = pq.read_schema(parquet_file)
    return datasets.Features.from_arrow_schema(
        pa.schema([widen(f) for f in schema])
    )


def _safe_row_group_size(table: pa.Table) -> Optional[int]:
    """Rows per Parquet row group that keep every list column under the int32 cap.

    Sized off the **widest single list** in ``table`` (the worst case a row group
    can contain), not the mean, so the bound holds however the long rows happen
    to be distributed. Returns None when pyarrow's default row-group size is
    already safe, so shards that were never at risk are written exactly as
    before.
    """
    widest = 0
    for col in table.columns:
        if not (pa.types.is_list(col.type) or pa.types.is_large_list(col.type)):
            continue
        col_max = pc.max(pc.list_value_length(col)).as_py()
        if col_max:
            widest = max(widest, int(col_max))
    if widest <= 0:
        return None                 # no list columns (v4/v6), or all lists empty
    rows = _LIST_OFFSET_CAP // (widest * _ROW_GROUP_HEADROOM)
    if rows >= _PQ_DEFAULT_ROW_GROUP_ROWS:
        return None                 # the default already clears the cap
    # No floor: a lower bound large enough to be worth having (say 64 rows) is
    # also large enough to re-cross the cap on a wide enough region set, which
    # would quietly reintroduce exactly the bug this guards. Footer overhead is
    # the cheaper thing to lose.
    return max(1, rows)


def _fmt_bytes(n: float) -> str:
    """Human-readable byte count for log lines (1 decimal, binary-ish decades)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000.0 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}TB"  # unreachable; keeps type checkers happy


def _estimate_temp_bytes(
    file_uncompressed_bytes: List[int],
    total_rows: int,
    n_kept: int,
    safety: float = _TEMP_SAFETY_FACTOR,
) -> int:
    """Estimate the bytes Step 4 will write into the temp dir.

    Basis is the *uncompressed* footer size of the source shards (what
    ``compression="none"`` writes), scaled by the fraction of rows that survive
    dedup, times a safety factor covering per-shard skew and the size growth of
    the optional version-conversion transform (v8->v6/v7 rehydrates markers back
    into ``seq``).
    """
    total = sum(file_uncompressed_bytes)
    if total <= 0 or total_rows <= 0:
        return 0
    frac = min(1.0, max(0.0, n_kept / total_rows))
    return int(total * frac * safety)


def _dir_can_hold(
    path, need_bytes: int, reserve_frac: float = _SHM_RESERVE_FRAC
) -> bool:
    """True if ``path``'s filesystem can absorb ``need_bytes`` and keep its reserve.

    Returns False (rather than raising) when the path is unreadable, so callers
    treat an inaccessible location as simply unavailable.
    """
    try:
        usage = shutil.disk_usage(str(path))
    except OSError:
        return False
    return usage.free - need_bytes >= reserve_frac * usage.total


def _fit_dedup_capacity(
    cap: int,
    root: Path = _SHM_ROOT,
    reserve_frac: float = _SHM_RESERVE_FRAC,
) -> int:
    """Largest power-of-two slot count ≤ ``cap`` that ``root`` can actually hold.

    The live inline-dedup table is a POSIX shared-memory segment, i.e. bytes in
    the same machine-wide tmpfs the assemble path budgets so carefully — and at
    canonical-BED scale it is multi-GB. Allocating past the tmpfs does not fail
    at ``SharedMemory(create=True)``: the segment is created sparse and the
    process takes a SIGBUS when ``fill(0)`` touches the missing pages. So size it
    up front against live free space, honouring the same reserve for other users
    of the box.

    Returns 0 when not even ``_DEDUP_MIN_CAPACITY`` fits — the caller then skips
    inline dedup for that split, which only defers work to the authoritative
    assemble dedup (never data loss, same as a table that fills mid-run).
    """
    if not root.is_dir():
        return cap          # not a tmpfs we can measure; leave the ask alone
    # The minimum bounds how far we SHRINK, never the ask itself: a small run
    # legitimately wants a table below it (a smoke test asks for 4096 slots),
    # and refusing that would disable dedup on a tmpfs with room to spare.
    floor = min(cap, _DEDUP_MIN_CAPACITY)
    while cap >= floor:
        if _dir_can_hold(root, cap * 16, reserve_frac):
            return cap
        cap >>= 1
    return 0


def _proc_start_time(pid: int) -> Optional[int]:
    """Process start time in clock ticks since boot, or None if pid is gone.

    Field 22 of /proc/<pid>/stat. Pairing it with the pid makes the ownership
    stamp immune to PID reuse: a recycled pid will not match the recorded start
    time, so its directory is still correctly identified as an orphan.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    # comm (field 2) is parenthesised and may contain spaces — split after it.
    try:
        rest = data[data.rindex(b")") + 2:].split()
        return int(rest[19])  # field 22 overall = index 19 after state
    except (ValueError, IndexError):
        return None


def _shm_temp_dir_for(output_path: Path) -> Path:
    """The /dev/shm temp dir for one split, unique to its output path.

    /dev/shm is flat and machine-wide, so the name must distinguish every
    concurrently-possible assemble. Path components alone do not: every runner
    passes ``--output <dataset>/assembled``, so the split's parent is the
    constant "assembled" and v6/v6nd/v7/v6-ldb would all land on
    ``.diploid_assemble_assembled_test``. Two such runs write temp files named
    after their source shards — ``<sample>.parquet``, identical across versions
    since they cover the same samples — into the same directory and silently
    overwrite each other's rows. The digest of the resolved output path is what
    actually makes the name unique; the readable prefix is only for humans
    reading ``ls /dev/shm``.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]", "_",
                  f"{output_path.parent.parent.name}_{output_path.name}")[:48]
    digest = hashlib.sha256(
        str(output_path.resolve()).encode()
    ).hexdigest()[:8]
    return _SHM_ROOT / f"{_SHM_PREFIX}{slug}_{digest}"


def _owner_stamp() -> dict:
    """This process's identity, for marking a temp dir or lock as ours."""
    return {"pid": os.getpid(), "start_time": _proc_start_time(os.getpid())}


def _owner_alive(stamp: Optional[dict]) -> bool:
    """True only if ``stamp`` names a process that is still running.

    A missing or malformed stamp means "not alive": it either predates the
    stamping code or belongs to a process that died before writing one, and in
    both cases the resource is safe to reclaim.
    """
    if not isinstance(stamp, dict):
        return False
    try:
        pid = int(stamp["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if pid == os.getpid():
        return True
    live_start = _proc_start_time(pid)
    if live_start is None:
        return False                       # no such process
    recorded = stamp.get("start_time")
    if recorded is None:
        return True                        # pid exists, can't disprove — leave it
    return live_start == recorded          # differs => pid was recycled, orphan


def _read_stamp(path: Path) -> Optional[dict]:
    """Parse an owner stamp from ``path``, or None if absent/unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _stamp_temp_dir(temp_dir: Path) -> None:
    """Record which live process owns ``temp_dir`` so orphans can be identified."""
    try:
        (temp_dir / "owner.json").write_text(json.dumps(_owner_stamp()))
    except OSError as exc:  # non-fatal: the dir is still cleaned in-process
        logger.warning("Could not stamp temp dir %s: %s", temp_dir, exc)


def _claim_temp_dir(temp_dir: Path) -> None:
    """Take exclusive ownership of ``temp_dir``, or refuse if it is in use.

    Clearing a leftover directory is right when its owner is gone (an
    interrupted run) and catastrophic when it is not: a second assemble of the
    same output path would delete a live run's finished shards, or interleave
    same-named files into them. Only one assemble per output path can run at a
    time — this turns that from silent mutual destruction into an up-front
    error naming the other process.

    Concurrent assembles of *different* output paths are fine and always were;
    they get different directories via _shm_temp_dir_for.
    """
    if temp_dir.exists() and _temp_dir_owner_alive(temp_dir):
        owner = "unknown"
        try:
            owner = str(json.loads((temp_dir / "owner.json").read_text())["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        raise RuntimeError(
            f"Temp dir {temp_dir} is in use by a live process (pid {owner}). "
            f"Another assemble of this same output path is already running — "
            f"only one at a time is supported, since both would write the same "
            f"per-sample temp filenames into this directory and corrupt each "
            f"other's output. Wait for it to finish, or kill pid {owner} and "
            f"re-run (its temp dir is then reclaimed automatically)."
        )
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    _stamp_temp_dir(temp_dir)


def _temp_dir_owner_alive(temp_dir: Path) -> bool:
    """True only if ``temp_dir``'s stamp names a process that is still running."""
    return _owner_alive(_read_stamp(temp_dir / "owner.json"))


# ---------------------------------------------------------------------------
# Machine-wide assemble lock.
#
# Exactly one assemble runs per machine. Everything below — the tmpfs reserve,
# the per-write admission budget, the temp dir claims — is arithmetic about one
# run's footprint, and a second concurrent assemble invalidates all of it: two
# runs each budgeting their own worker fleet against the same 10% headroom can
# collectively blow through it, which is the failure this whole module exists to
# prevent. Serialising is cheap (an assemble is minutes; a generate is days) and
# makes those guarantees exact rather than approximate.
#
# The lock lives in /dev/shm so it is machine-wide and vanishes on reboot, and
# carries the same pid+start_time stamp as the temp dirs, so a lock left by a
# SIGKILLed run is reclaimed automatically instead of wedging the next one.
# ---------------------------------------------------------------------------

_ASSEMBLE_LOCK_NAME = ".diploid_assemble.lock"


def _assemble_lock_path() -> Path:
    """Where the machine-wide lock lives (tmpdir only if /dev/shm is absent)."""
    root = _SHM_ROOT if _SHM_ROOT.is_dir() else Path(tempfile.gettempdir())
    return root / _ASSEMBLE_LOCK_NAME


def _acquire_assemble_lock() -> Optional[Path]:
    """Take the machine-wide assemble lock, or raise if another run holds it.

    The lock is published ATOMICALLY AND ALREADY STAMPED: the owner stamp is
    written to a private temp file, which is then ``os.link``ed into place.
    ``link`` fails with EEXIST if the target exists, so it is as exclusive as
    ``O_CREAT|O_EXCL`` — but the lock only ever becomes visible with a complete
    stamp inside it. Creating the file first and writing the stamp afterwards
    leaves a window in which a second assemble reads an empty stamp, concludes
    from ``_owner_alive(None)`` that the owner is dead, unlinks the lock and
    takes it: both then run, which is precisely what the lock exists to prevent.

    A lock whose owner is dead is reclaimed once; if the retry also loses,
    another process won the race and we defer to it.

    Returns the lock path, or None if the lock could not be created at all
    (read-only tmpfs, say) — the caller proceeds unlocked rather than failing an
    assemble over a missing lock file, since the per-output-path claim in
    _claim_temp_dir still guards the destructive case.
    """
    path = _assemble_lock_path()
    for attempt in (0, 1):
        # Same-pid name: any leftover can only be this process's own debris.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            tmp.unlink()
        except OSError:
            pass
        try:
            fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except OSError as exc:
            logger.warning("Could not create assemble lock %s: %s — proceeding "
                           "without a machine-wide lock", path, exc)
            return None
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(_owner_stamp(), fh)
            try:
                os.link(str(tmp), str(path))
            except FileExistsError:
                stamp = _read_stamp(path)
                if _owner_alive(stamp):
                    pid = stamp.get("pid") if isinstance(stamp, dict) else "unknown"
                    raise RuntimeError(
                        f"Another dataset assembly is already running on this "
                        f"machine (pid {pid}, lock {path}). No more than one "
                        f"assemble at a time is allowed: each one sizes /dev/shm "
                        f"against its own worker fleet, and two would exceed the "
                        f"shared-tmpfs headroom they each assume they have. Wait "
                        f"for it to finish, or kill pid {pid} — the lock is then "
                        f"reclaimed automatically."
                    ) from None
                if not attempt:
                    try:     # stale lock from a SIGKILLed run — reclaim it
                        path.unlink()
                        print(f"  Reclaimed stale assemble lock {path} "
                              f"(owning process is gone)", flush=True)
                    except OSError:
                        pass
                continue     # retry once; on attempt 1 fall through to the raise
            except OSError as exc:
                logger.warning("Could not create assemble lock %s: %s — "
                               "proceeding without a machine-wide lock",
                               path, exc)
                return None
            return path
        finally:
            try:             # the lock now holds its own link to the inode
                tmp.unlink()
            except OSError:
                pass
    # Both attempts lost to a process that grabbed the lock in between.
    raise RuntimeError(
        f"Another dataset assembly took the lock {path} while this one was "
        f"reclaiming it. Only one assemble at a time is allowed; re-run once "
        f"it finishes."
    )


def _release_assemble_lock(path: Optional[Path]) -> None:
    """Drop the lock, but only if this process still owns it.

    Ownership must be *positively* confirmed: an unreadable stamp is not proof
    the lock is ours. It is also what a lock reclaimed by another run looks like
    while that run is mid-``json.dump``, and deleting it there would hand the
    machine to a third assemble.  If the file is already gone there is nothing
    to release.
    """
    if path is None:
        return
    stamp = _read_stamp(path)
    if not isinstance(stamp, dict) or stamp.get("pid") != os.getpid():
        return                  # someone else's, unconfirmable, or already gone
    try:
        path.unlink()
    except OSError:
        pass


_SWEEP_GRACE_S = 60.0   # don't reclaim a just-created dir that isn't stamped yet


def _sweep_orphan_dirs(root: Path, pattern: str,
                       grace_s: float = _SWEEP_GRACE_S) -> int:
    """Remove assemble temp dirs under ``root`` whose owning process is gone.

    This is the only layer that covers SIGKILL / OOM-kill / reboot — the in-process
    try/finally, atexit and signal handlers cannot run in those cases. Directories
    owned by a live process are never touched, so concurrent assembles are safe.
    A dir created within ``grace_s`` that has no stamp yet is also left alone: that
    is another run in the gap between its mkdir and its stamp write, not an orphan.
    Returns the number of directories reclaimed.
    """
    if not root.is_dir():
        return 0
    removed = 0
    try:
        candidates = list(root.glob(pattern))
    except OSError:
        return 0
    now = time.time()
    for d in candidates:
        if not d.is_dir() or _temp_dir_owner_alive(d):
            continue
        if not (d / "owner.json").exists():
            try:
                if now - d.stat().st_mtime < grace_s:
                    continue
            except OSError:
                continue
        try:
            size = sum(f.stat().st_size for f in d.glob("*") if f.is_file())
        except OSError:
            size = 0
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
        print(f"  Reclaimed orphaned temp dir {d} ({_fmt_bytes(size)}) — "
              f"owning process is gone", flush=True)
    return removed


def _sweep_orphan_shm_dirs(prefix: str = _SHM_PREFIX,
                           grace_s: float = _SWEEP_GRACE_S) -> int:
    """Reclaim orphaned /dev/shm assemble temp dirs. See :func:`_sweep_orphan_dirs`."""
    return _sweep_orphan_dirs(_SHM_ROOT, f"{prefix}*", grace_s)


def _disk_temp_dir_for(output_path: Path) -> Path:
    """The on-disk Step-4 temp dir for one split (also the ENOSPC fallback)."""
    return output_path.parent / f".{output_path.name}{_DISK_TEMP_SUFFIX}"


def _sweep_orphan_disk_dirs(output_parent: Path,
                            grace_s: float = _SWEEP_GRACE_S) -> int:
    """Reclaim orphaned on-disk Step-4 temp dirs next to an assemble output.

    The /dev/shm sweep cannot see these: they live beside the assembled dataset,
    one per split. They are created on every run (they double as the per-shard
    ENOSPC fallback) and are where whole splits land when the tmpfs is too small,
    so a SIGKILL can strand real bulk here — previously reclaimed only if someone
    happened to re-assemble that exact output path again.
    """
    return _sweep_orphan_dirs(output_parent, f".*{_DISK_TEMP_SUFFIX}", grace_s)


# ---------------------------------------------------------------------------
# Split-level resume + partial-output safety  (HARDENING_PLAN.md §A4, §A5)
#
# The 2026-09-03 incident lost a 4h26m assemble at Step 5 and, because assemble
# had no resume of any kind, the retry redid hashing (1,699 s), dedup (245 s) and
# Step 4 (13,830 s) from scratch. The Step-4 cache (§A1/§A2) is sequenced after
# the BioMe run; this slice ships only the coarser half of the resume — a split
# that FINISHED is never redone — which is what stops a test-split failure from
# costing the 16.8 h train split a second time.
# ---------------------------------------------------------------------------


def _digest_files(files: List[Path], *, with_rows: Optional[List[int]] = None) -> str:
    """sha256 over sorted ``name|size[|rows]`` lines — deliberately no mtime.

    /storage is an NFS FlexGroup whose constituents stamp mtimes from clocks that
    run minutes apart from each other and from this host (MONITOR.md §4), so an
    mtime in a fingerprint would invalidate the cache at random. Content hashing
    is not an option either: the train shard set is 9.7 TB, and hashing it is the
    "invisible 2.5 h" mistake the ledger already records. (name, size, rows) is
    what the Step-1 footer scan produces for free.
    """
    h = hashlib.sha256()
    rows = with_rows if with_rows is not None else [None] * len(files)
    for f, n in sorted(zip(files, rows), key=lambda t: t[0].name):
        try:
            size = f.stat().st_size
        except OSError:
            size = -1
        h.update(f"{f.name}|{size}".encode())
        if n is not None:
            h.update(f"|{int(n)}".encode())
        h.update(b"\n")
    return h.hexdigest()


def _build_split_fingerprint(
    shard_dir: Path,
    output_path: Path,
    parquet_files: List[Path],
    file_row_counts: List[int],
    *,
    max_shard_size: str,
    seed: int,
    route_label: Optional[str],
) -> dict:
    """Everything a completed split's rows are a pure function of.

    Note the seed and max_shard_size ARE compared here, unlike in the Step-4
    fingerprint of §A1: Step 4's output is seed-independent, but a finished
    split's row ORDER is not (`.shuffle(seed=…)`), and its Arrow shard count
    follows max_shard_size.

    When §A1 lands, `_build_step4_fingerprint` supersedes this and the two share
    the shard/sidecar digests; bump `_SPLIT_MARKER_VERSION` then so markers
    written by this version are not read under the new rules.
    """
    sidecar_dir = shard_dir / _SKIP_WEIGHTS_SUBDIR
    sidecars = sorted(sidecar_dir.glob("*.parquet")) if sidecar_dir.is_dir() else []
    return {
        "marker_version": _SPLIT_MARKER_VERSION,
        "shard_dir": str(shard_dir.resolve()),
        "output_path": str(output_path.resolve()),
        "n_shards": len(parquet_files),
        "total_rows": int(sum(file_row_counts)),
        "shards_sha256": _digest_files(parquet_files, with_rows=file_row_counts),
        "skip_weights_sha256": _digest_files(sidecars),
        "route_label": route_label,
        "max_shard_size": max_shard_size,
        "seed": int(seed),
    }


def _write_split_marker(output_path: Path, marker: dict) -> Path:
    """Write ``.assemble_complete.json`` atomically (.tmp → rename).

    Same discipline as provenance.py:185-189 — a half-written marker read by the
    next run would be the one failure mode this whole mechanism must not have.
    """
    path = output_path / _SPLIT_MARKER_NAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(marker, f, indent=2)
    tmp.rename(path)
    return path


def _read_split_marker(output_path: Path) -> Optional[dict]:
    """Parse the marker, or None (with a warning) if absent/corrupt.

    A corrupt marker must degrade to "rebuild", never wedge the run.
    """
    path = output_path / _SPLIT_MARKER_NAME
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            marker = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"  WARNING: ignoring unreadable split marker {path}: {exc}",
              flush=True)
        return None
    if not isinstance(marker, dict):
        print(f"  WARNING: ignoring malformed split marker {path} "
              f"(not a JSON object)", flush=True)
        return None
    return marker


def _split_output_complete(
    output_path: Path, fingerprint: dict,
) -> Tuple[Optional[dict], Optional[str]]:
    """Is ``output_path`` a finished save of exactly this fingerprint?

    Returns ``(marker, None)`` on a hit, or ``(None, reason)`` explaining the
    miss so the caller can print why it is rebuilding. Every condition must hold:
    the marker parses at a known version, HF's own two metadata files are there,
    the fingerprint matches field-for-field, the Arrow shard count still matches
    (catches a half-deleted output), and the split is non-empty.
    """
    marker = _read_split_marker(output_path)
    if marker is None:
        return None, "no completion marker"
    if marker.get("marker_version") != _SPLIT_MARKER_VERSION:
        return None, (f"marker_version={marker.get('marker_version')!r}, "
                      f"this build writes {_SPLIT_MARKER_VERSION}")
    for meta in ("dataset_info.json", "state.json"):
        if not (output_path / meta).is_file():
            return None, f"{meta} missing from a marked-complete output"
    recorded = marker.get("fingerprint")
    if not isinstance(recorded, dict):
        return None, "marker carries no fingerprint"
    differing = sorted(
        k for k in set(recorded) | set(fingerprint)
        if recorded.get(k) != fingerprint.get(k)
    )
    if differing:
        return None, "inputs changed: " + ", ".join(
            f"{k} {recorded.get(k)!r} → {fingerprint.get(k)!r}"
            if k in ("shard_dir", "output_path", "route_label", "seed",
                     "max_shard_size", "n_shards", "total_rows",
                     "marker_version")
            else k
            for k in differing
        )
    n_arrow = len(list(output_path.glob("data-*.arrow")))
    if n_arrow != marker.get("n_arrow_shards"):
        return None, (f"{n_arrow} data-*.arrow files present, marker recorded "
                      f"{marker.get('n_arrow_shards')}")
    if not isinstance(marker.get("n_final"), int) or marker["n_final"] <= 0:
        return None, f"marker records n_final={marker.get('n_final')!r}"
    return marker, None


def _clear_partial_output(output_path: Path, shard_dir: Path) -> int:
    """rmtree a non-empty assemble output before `save_to_disk` writes into it.

    `save_to_disk` does not clean its destination, so a resumed (or simply
    re-run) Step 5 would MERGE into whatever a crashed run left behind — stale
    `data-*.arrow` files from a longer run would be silently loaded by every
    downstream reader. Modeled on parquet2assembled_three_levels.py:642-649 (file deleted 2026-10-08; see tag `pre-reorg`).

    Two guards, because this is an rmtree driven by a CLI path:
      * refuse if the output resolves to the shard dir, and
      * refuse if the tree holds any ``*.parquet`` — that means ``--output`` was
        pointed at a shard dir, and this would destroy the generate output.

    Returns the number of bytes removed (0 if there was nothing to remove).
    """
    if not output_path.exists():
        return 0
    if not output_path.is_dir():
        raise RuntimeError(
            f"Assemble output {output_path} exists and is not a directory; "
            f"refusing to touch it."
        )
    if output_path.resolve() == shard_dir.resolve():
        raise RuntimeError(
            f"Refusing to clear the assemble output {output_path}: it resolves "
            f"to the shard dir {shard_dir}. --output must be a separate "
            f"directory from --shard-dir."
        )
    total = 0
    for root, _dirs, files in os.walk(output_path):
        for name in files:
            if name.endswith(".parquet"):
                raise RuntimeError(
                    f"Refusing to clear the assemble output {output_path}: it "
                    f"contains Parquet files (e.g. {Path(root) / name}). That "
                    f"means --output points at a shard directory, and clearing "
                    f"it would destroy the generate output."
                )
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    if total == 0 and not any(output_path.iterdir()):
        return 0
    shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return total


def _choose_temp_dir(
    need_bytes: int,
    shm_dir: Path,
    disk_dir: Path,
    reserve_frac: float = _SHM_RESERVE_FRAC,
) -> Tuple[Path, str]:
    """Pick the Step-4 temp dir, preferring RAM-backed /dev/shm when it truly fits.

    Returns ``(temp_dir, reason)``; ``reason`` is a log-ready explanation of the
    decision including the numbers it was based on.
    """
    need = _fmt_bytes(need_bytes)
    if not shm_dir.parent.is_dir():
        return disk_dir, f"{shm_dir.parent} absent; using disk (need {need})"
    try:
        usage = shutil.disk_usage(str(shm_dir.parent))
    except OSError as exc:
        return disk_dir, f"{shm_dir.parent} unreadable ({exc}); using disk"
    if _dir_can_hold(shm_dir.parent, need_bytes, reserve_frac):
        return shm_dir, (
            f"using {shm_dir} (RAM-backed): need {need}, "
            f"free {_fmt_bytes(usage.free)} of {_fmt_bytes(usage.total)}"
        )
    reason = (
        f"{shm_dir.parent} cannot hold {need} while reserving "
        f"{reserve_frac:.0%} of {_fmt_bytes(usage.total)} "
        f"(free {_fmt_bytes(usage.free)}); using disk {disk_dir}"
    )
    if not _dir_can_hold(disk_dir.parent, need_bytes, 0.0):
        reason += " — WARNING: disk may not hold it either"
    return disk_dir, reason


# ---------------------------------------------------------------------------
# Step-4 worker state: filter + write one Parquet shard in parallel.
# Large arrays (keep_mask, inverse) are placed in POSIX shared memory by the
# parent so all workers share one physical copy with no COW duplication.
# ---------------------------------------------------------------------------

_FILTER_GLOBALS: dict = {}

# Optional per-shard version-conversion transform (pyarrow.Table -> Table),
# applied to each filtered shard before it is written. Set by
# _process_split_streaming BEFORE the Step-4 Pool forks (so workers inherit it
# via fork, exactly like _EXTERNALIZE_MARKERS) and reset to None afterwards so it
# never leaks across the train/test splits. None = no conversion (identity).
_FILTER_TRANSFORM: Optional[Callable[[pa.Table], pa.Table]] = None

# Number of Step-4 workers writing concurrently. Each worker admission-checks
# only its own shard, so without this they can all see room for their own write
# and collectively blow past the reserve. Set by _process_split_streaming BEFORE
# the Pool forks (same fork-inherited-global pattern as _FILTER_TRANSFORM).
_FILTER_CONCURRENCY: int = 1


def _init_filter_worker(
    mask_name: str,
    mask_shape: tuple,
    mask_dtype: np.dtype,
    inv_name: str,
    inv_shape: tuple,
    inv_dtype: np.dtype,
    gw_name: Optional[str],
    gw_shape: Optional[tuple],
    gw_dtype,
) -> None:
    """Pool initializer: attach to shared-memory arrays and cache them.

    All three large arrays (keep_mask, inverse, group_weight) live in POSIX
    shared memory — zero copies, no COW, no IPC pickle overhead.
    """
    shm_m = SharedMemory(name=mask_name)
    shm_i = SharedMemory(name=inv_name)
    _FILTER_GLOBALS["mask"] = np.ndarray(mask_shape, dtype=mask_dtype,
                                         buffer=shm_m.buf)
    _FILTER_GLOBALS["inv"]  = np.ndarray(inv_shape,  dtype=inv_dtype,
                                         buffer=shm_i.buf)
    _FILTER_GLOBALS["_shm_m"] = shm_m  # keep alive for this worker's lifetime
    _FILTER_GLOBALS["_shm_i"] = shm_i
    if gw_name is not None:
        shm_g = SharedMemory(name=gw_name)
        _FILTER_GLOBALS["gw"] = np.ndarray(gw_shape, dtype=gw_dtype,
                                            buffer=shm_g.buf)
        _FILTER_GLOBALS["_shm_g"] = shm_g
    else:
        _FILTER_GLOBALS["gw"] = None


def _is_enospc(exc: BaseException) -> bool:
    """True if ``exc`` is an out-of-space error.

    pyarrow surfaces ENOSPC as a plain OSError whose errno is sometimes unset,
    so fall back to matching the message.
    """
    return (getattr(exc, "errno", None) == errno.ENOSPC
            or "No space left on device" in str(exc))


def _write_shard_with_fallback(
    table: pa.Table, name: str, temp_dir: str, fallback_dir: Optional[str]
) -> str:
    """Write ``table`` to the first location with room, degrading as needed.

    Ladder: temp_dir uncompressed -> fallback_dir uncompressed -> fallback_dir
    lz4.  Each rung is admission-checked against live free space *before* writing
    (so a shared tmpfs is never driven to 100%, which would break other processes
    on the box) and, if it still hits ENOSPC in the race window, its partial file
    is removed immediately before moving on.  The final lz4 rung roughly halves
    the footprint, so a disk that cannot take the uncompressed form usually still
    takes this one.

    Step 5 loads these paths via ``datasets.load_dataset("parquet", data_files=…)``,
    which is codec- and directory-agnostic, so a mixed result set is fine.

    Row groups are sized by :func:`_safe_row_group_size` rather than left to
    pyarrow's default (which would write the whole shard as one row group).
    That default is what made the first v7-tsstes assemble unreadable at Step 5
    — see the int32 list-offset note above the constant.
    """
    # Budget for every concurrent writer, not just this one: N workers each
    # seeing room for their own shard is exactly how the reserve gets blown
    # through. Charging each write the full fleet's footprint keeps the
    # independent per-worker checks collectively safe.
    need_one = table.nbytes
    need = need_one * max(1, _FILTER_CONCURRENCY)
    # Sized once per shard, not per rung: the ladder only changes where and with
    # which codec the shard lands, never its contents.
    row_group_size = _safe_row_group_size(table)
    # (dir, codec, reserve): the reserve protects other users of a shared
    # filesystem, so it is enforced on the preferred rung but relaxed as the
    # ladder descends — the last rung is a genuine last resort, reserving
    # nothing and budgeting only this one write.
    if fallback_dir is None or str(fallback_dir) == str(temp_dir):
        # Already on disk: no relocation possible, but still degrade the codec.
        ladder = [(temp_dir, "none", 0.02), (temp_dir, "lz4", 0.0)]
    else:
        ladder = [(temp_dir, "none", _SHM_RESERVE_FRAC),
                  (fallback_dir, "none", 0.02),
                  (fallback_dir, "lz4", 0.0)]
    attempts: List[str] = []
    for d, codec, reserve in ladder:
        out = str(Path(d) / name)
        if not _dir_can_hold(d, need if reserve > 0 else need_one, reserve):
            attempts.append(f"{d} [{codec}]: insufficient free space")
            continue
        try:
            pq.write_table(table, out, compression=codec,
                           row_group_size=row_group_size)
            if attempts:
                print(f"  WARNING: {name} fell back to {d} [{codec}] after "
                      f"{'; '.join(attempts)}", flush=True)
            return out
        except OSError as exc:
            if not _is_enospc(exc):
                raise                       # a real I/O bug, not a space problem
            # Drop the partial file right away so the space is returned to the
            # filesystem before the next rung (or another worker) needs it.
            try:
                Path(out).unlink()
            except OSError:
                pass
            attempts.append(f"{d} [{codec}]: ENOSPC")
    raise RuntimeError(
        f"Step 4 ran out of space writing {name} "
        f"({_fmt_bytes(need_one)} for this shard, {_fmt_bytes(need)} budgeted "
        f"across {max(1, _FILTER_CONCURRENCY)} concurrent writers); "
        f"tried {len(ladder)} location(s): {'; '.join(attempts)}. "
        f"Free space in /dev/shm or on the output filesystem and re-run "
        f"(no temp files are left behind)."
    )


def _filter_write_shard(task: tuple) -> Optional[str]:
    """Filter one Parquet shard to kept rows, update weights, write temp file.

    Returns the output path string, or None if the shard had no kept rows.
    """
    pf, temp_dir, offset, n, fallback_dir = task
    mask = _FILTER_GLOBALS["mask"]
    inv  = _FILTER_GLOBALS["inv"]
    gw   = _FILTER_GLOBALS["gw"]

    local_mask = mask[offset:offset + n]
    if not local_mask.any():
        return None

    table = pq.read_table(pf)
    filtered = table.filter(pa.array(local_mask))

    if gw is not None:
        local_indices = np.flatnonzero(local_mask)
        local_ws = gw[inv[offset:offset + n][local_indices]]
        col_idx = filtered.schema.get_field_index("sample_weight")
        filtered = filtered.set_column(
            col_idx, "sample_weight",
            pa.array(local_ws.astype(np.int32), type=pa.int32()),
        )

    # On-the-fly version conversion (v7→v6 / v8→v6 / v8→v7): convert AFTER the
    # weight rewrite (so sample_weight is found in the source schema) and BEFORE
    # the write, folding into this existing rewrite pass at no extra disk cost.
    if _FILTER_TRANSFORM is not None:
        filtered = _FILTER_TRANSFORM(filtered)

    return _write_shard_with_fallback(
        filtered, Path(pf).name, temp_dir, fallback_dir,
    )


def _process_split_streaming(
    shard_dir: Path,
    output_path: Path,
    max_shard_size: str,
    seed: int = 42,
    hf_cache_dir: Optional[str] = None,
    shard_transform: Optional[Callable[[pa.Table], pa.Table]] = None,
    seq_normalizer: Optional[Callable[[str], str]] = None,
    *,
    route_label: Optional[str] = None,
) -> Tuple[int, dict]:
    """Process one split: parallel-hash → dedup → parallel-filter → HF shuffle+save.

    Peak memory: two hash arrays (h1, h2) + a 16-byte structured dedup key +
    inverse — roughly 2x the single-hash footprint (the verify hash is the
    collision safeguard, see _dedup_by_hash_pair) — plus POSIX shared memory
    segments for keep_mask/inverse/group_weight.  Step 4 runs with up to 32
    parallel workers sharing those segments with no COW duplication.

    Returns ``(n_final, stats)`` where ``n_final`` is the number of rows in the
    final dataset and ``stats`` is the per-step row/dedup/timing breakdown (the
    same numbers as the grep-able ``[TIMING]`` line) so the caller can persist
    them in the assembled ``provenance.json``.

    A split whose output is already marked complete for exactly these inputs is
    skipped entirely (HARDENING_PLAN.md §A4), so a test-split failure never
    redoes the train split.  ``route_label`` identifies the on-the-fly
    conversion route in that check — two different ``--target-version`` runs into
    the same output must not look interchangeable.
    """
    # Cheap (three stats) and placed here because everything below depends on
    # the venv still being the one main() checked — see §B.
    _verify_venv_fingerprint(f"entering the {output_path.name!r} split")

    # Step 5 rmtrees a partial output before saving (§A5), and `ds` is
    # memory-mapped out of the HF cache at that moment — a cache placed inside
    # the output would be deleted out from under the live dataset. Checked here,
    # not at Step 5, so a bad --hf-cache-dir costs seconds rather than 4 hours.
    if hf_cache_dir is not None:
        _hc, _op = Path(hf_cache_dir).resolve(), output_path.resolve()
        if _hc == _op or _op in _hc.parents:
            raise RuntimeError(
                f"--hf-cache-dir {hf_cache_dir} is inside the assemble output "
                f"{output_path}. Step 5 clears the output directory before "
                f"saving, which would delete the memory-mapped cache the "
                f"dataset is being read from. Put the cache beside the output, "
                f"not under it (the default is "
                f"<output>.parent/.hf_cache_assemble)."
            )

    import datasets   # heavy; imported lazily so the CLI starts fast

    parquet_files = sorted(shard_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: No .parquet files found in {shard_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: Collect per-file row counts + uncompressed sizes ──────────
    # The footer also carries each row group's uncompressed byte size, which is
    # what Step 4 actually writes (compression="none").  Capturing it here costs
    # nothing — the footers are already being read — and gives the Step-4 temp
    # dir preflight the right basis; the shards' on-disk size is lz4-compressed
    # and roughly half the real requirement.
    t_rowcount0 = time.perf_counter()
    file_row_counts: List[int] = []
    file_uncompressed_bytes: List[int] = []
    for pf in parquet_files:
        _md = pq.read_metadata(pf)
        file_row_counts.append(_md.num_rows)
        file_uncompressed_bytes.append(
            sum(_md.row_group(i).total_byte_size
                for i in range(_md.num_row_groups))
        )
    total_rows = sum(file_row_counts)
    t_rowcount = time.perf_counter() - t_rowcount0
    print(f"  Found {len(parquet_files)} shards, {total_rows:,} total rows "
          f"(scanned in {t_rowcount:.1f}s)", flush=True)

    # ── Split-level resume (HARDENING_PLAN.md §A4) ───────────────────────
    # Placed here because the fingerprint needs Step 1's row counts, and before
    # Step 2 because everything from here on is the 4-6 h this exists to skip.
    # The decision is printed either way: a silent resume is how the ledger's
    # "invisible 2.5 h" entry happened.
    split_fingerprint = _build_split_fingerprint(
        shard_dir, output_path, parquet_files, file_row_counts,
        max_shard_size=max_shard_size, seed=seed, route_label=route_label,
    )
    _marker, _miss = _split_output_complete(output_path, split_fingerprint)
    if _marker is not None:
        print(f"  === RESUME === {output_path} is already complete: "
              f"{_marker['n_final']:,} rows in {_marker['n_arrow_shards']} "
              f"Arrow shard(s), finished {_marker.get('finished_at')} by "
              f"pid {_marker.get('written_by', {}).get('pid')}.", flush=True)
        print(f"  Skipping hash/dedup/filter/load/shuffle/save for this split "
              f"(the run that built it spent "
              f"{_marker.get('stats', {}).get('timing_s', {})}).", flush=True)
        print(f"  To force a rebuild, delete "
              f"{output_path / _SPLIT_MARKER_NAME}.", flush=True)
        stats = dict(_marker.get("stats") or {})
        stats["resumed_from"] = "split_complete"
        # Never let provenance claim a 13,830 s step took 0 s: the original
        # timings move aside under their own key rather than being reported as
        # this run's.
        stats["reused_timing_s"] = stats.pop("timing_s", None)
        stats["n_final"] = int(_marker["n_final"])
        return int(_marker["n_final"]), stats
    print(f"  No reusable output for this split ({_miss}) — building it.",
          flush=True)

    # ── Step 2: Parallel-hash all sequences + collect weights ────────────
    # Each worker returns (h1, h2, weights): h1 = primary dedup hash, h2 = an
    # independent verify hash (collision safeguard), weights = per-row weight so
    # step 3 aggregates without a second shard pass.
    # NOTE (on-the-fly conversion): hashing runs on the SOURCE shards, before the
    # Step-4 transform. For v8→v6/v7 the v8 compound key (_record_key_bytes) is an
    # injective function of (seq, snp, indel_span, direction), and rehydrate is a
    # bijection on that tuple, so deduping on the source key yields exactly the
    # unique groups the rehydrated v6/v7 seq would — see compat.py + the
    # dedup-equivalence property test. So the hash step needs no change.
    # For routes with pre_dedup=True (e.g. STRIP_DIRECTION / v6→v6nd), a
    # seq_normalizer is passed here so that records which differ only in the
    # stripped characters hash to the same key and are merged in step 3 BEFORE
    # the Step-4 shard transform runs.
    print(f"  Hashing {total_rows:,} sequences…", flush=True)
    t0 = time.perf_counter()
    n_workers = min(_HASH_POOL_MAX_PROC, os.cpu_count() or 1)
    _hash_fn = (
        functools.partial(_hash_shard_pair, seq_normalizer=seq_normalizer)
        if seq_normalizer is not None else _hash_shard_pair
    )
    with Pool(processes=n_workers) as pool:
        all_results = pool.map(
            _hash_fn, [str(pf) for pf in parquet_files],
        )
    h1_chunks, h2_chunks, weight_chunks = (
        zip(*all_results) if all_results else ([], [], [])
    )
    h1          = np.concatenate(h1_chunks)
    h2          = np.concatenate(h2_chunks)
    all_weights = np.concatenate(weight_chunks)
    del all_results, h1_chunks, h2_chunks, weight_chunks
    t_hash = time.perf_counter() - t0
    print(f"  Hashing complete in {t_hash:.1f}s", flush=True)

    # ── Step 3: Dedup computation ──────────────────────────────────────
    # Dedup on the (h1, h2) hash pair so two distinct sequences that collide on
    # the primary 64-bit hash are kept as separate groups (collision-safe to
    # ~2^-128).  `inverse` maps every row to its group index; kept alive through
    # step 4 so workers can look up aggregated weights per row.
    # v7 note: the v6-era dup_ratio > 0.10 gate is intentionally removed —
    # weights are always aggregated (see `project_v7_planned_cleanups`).
    t0 = time.perf_counter()
    unique_pairs, first_indices, inverse = _dedup_by_hash_pair(h1, h2)
    del h1, h2

    n_dups = total_rows - len(unique_pairs)
    dup_ratio = n_dups / total_rows if total_rows > 0 else 0.0
    print(f"  Dedup: {n_dups:,}/{total_rows:,} duplicate rows ({dup_ratio:.1%})",
          flush=True)

    # Report the (extremely unlikely) event of a primary-hash collision between
    # different sequences.  The verify hash kept them separate, but the user
    # should know it happened.
    n_coll_h1, n_collided_seqs, example_h1 = _primary_hash_collisions(unique_pairs)
    if n_collided_seqs > 0:
        print(f"  ⚠️  PRIMARY-HASH COLLISION DETECTED: {n_coll_h1:,} 64-bit hash "
              f"value(s) map to multiple distinct sequences; {n_collided_seqs:,} "
              f"sequence(s) a hash-only dedup would have silently dropped are "
              f"preserved by the verify hash. Example hash(es): {example_h1}",
              flush=True)
        logger.warning(
            "Primary-hash collision in %s: %d hash value(s), %d distinct "
            "sequence(s) preserved by verify hash",
            shard_dir, n_coll_h1, n_collided_seqs,
        )

    keep_mask = np.zeros(total_rows, dtype=bool)
    keep_mask[first_indices] = True

    # Inline-dedup skip-weight sidecars: weight of occurrences skipped during
    # generate (never written as rows).  Fold them into the per-group weight so
    # assembled weights match the no-inline-dedup baseline.  Sidecars store the
    # full (h1, h2) pair, so the fold is an EXACT match against unique_pairs —
    # correct even under a primary-hash collision.
    sc_h1, sc_h2, sc_weights, n_sidecars = _load_skip_weights(shard_dir)
    have_sidecars = sc_h1.size > 0

    # Aggregate sample_weight per unique group using the weights collected in
    # step 2 — no second shard pass.  np.bincount with weights is a C-speed
    # histogram.  Build it whenever there are duplicate rows OR skip-weight
    # sidecars to fold (with perfect inline dedup n_dups can be 0 yet skip
    # weights still apply).
    group_weight: Optional[np.ndarray] = None
    if n_dups > 0 or have_sidecars:
        group_weight = np.bincount(
            inverse, weights=all_weights, minlength=len(unique_pairs),
        ).astype(np.int64)

    if have_sidecars and seq_normalizer is not None:
        # Sidecar hashes were computed from raw (pre-normalization) seq values, so
        # they cannot be matched against unique_pairs which uses normalized hashes.
        # Silently folding them in would corrupt weights; bail loudly instead.
        logger.warning(
            "STRIP_DIRECTION conversion: seq_normalizer active but %d sidecar entries "
            "use raw (pre-normalization) hashes — cannot reconcile with normalized "
            "unique_pairs. Sidecar weights will NOT be folded in. Re-run generate with "
            "--no-dedup-inline to avoid this weight undercount.",
            len(sc_h1),
        )
        have_sidecars = False  # skip the fold; weight undercount is visible in logs

    if have_sidecars:
        sc_pairs = np.empty(len(sc_h1), dtype=unique_pairs.dtype)
        sc_pairs["h1"] = sc_h1
        sc_pairs["h2"] = sc_h2
        n_up = len(unique_pairs)
        idx = np.searchsorted(unique_pairs, sc_pairs)
        in_range = idx < n_up
        idx_safe = np.where(in_range, idx, 0)
        valid = in_range & (unique_pairs[idx_safe] == sc_pairs)
        np.add.at(group_weight, idx[valid], sc_weights[valid])
        n_orphan = int((~valid).sum())
        print(f"  Folded skip weights from {n_sidecars} sidecar(s): "
              f"+{int(sc_weights[valid].sum()):,} weight, "
              f"{n_orphan:,} orphan pair(s) dropped", flush=True)
        if n_orphan:
            print(f"  WARNING: {n_orphan:,} skip-weight (h1,h2) pair(s) had no "
                  f"surviving row (dropped {int(sc_weights[~valid].sum()):,} "
                  f"weight) — were shards deleted from {shard_dir}?", flush=True)

    # group_weight is int64, but the surviving row's sample_weight column is
    # int32 (see _filter_write_shard). Guard the downcast: at current scale the
    # largest group is far below 2^31, but a silent wrap would corrupt weights
    # (and can go negative). Fail loudly instead — the fix is an int64 column.
    if group_weight is not None and group_weight.size:
        _max_w = int(group_weight.max())
        if _max_w > np.iinfo(np.int32).max:
            raise RuntimeError(
                f"Aggregated sample_weight {_max_w:,} exceeds int32 max "
                f"({np.iinfo(np.int32).max:,}); the Parquet sample_weight "
                f"column would silently overflow. Widen the schema to int64."
            )

    del all_weights, unique_pairs, first_indices
    n_kept = int(keep_mask.sum())
    t_dedup = time.perf_counter() - t0
    print(f"  Dedup computed in {t_dedup:.1f}s, "
          f"keeping {n_kept:,} rows", flush=True)

    # ── Step 4: Parallel filter + write deduplicated Parquet files ────────
    # keep_mask (~321 MB) and inverse (~2.6 GB) are placed in POSIX shared
    # memory so all workers attach to one physical copy (no COW duplication).
    # group_weight is small and pickled normally via the pool initializer.
    t0 = time.perf_counter()
    # Prefer /dev/shm (RAM-backed tmpfs) for the inter-step temp Parquet files
    # so Step 4 writes and Step 5 reads avoid a disk round-trip entirely.
    # Size the decision on the *uncompressed* footer bytes scaled by the kept
    # fraction — the source shards are lz4-compressed, so their on-disk size is
    # about half of what gets written and would let this check pass with no room.
    # Reclaim dirs stranded by a previous SIGKILL/reboot first: that space is
    # ours to return, and on /dev/shm it may be what makes the RAM path viable.
    # Both roots are swept — the tmpfs and the output-side disk dirs — since a
    # run killed while falling back to disk strands its bytes there instead.
    _sweep_orphan_shm_dirs()
    _sweep_orphan_disk_dirs(output_path.parent)
    _need_bytes = _estimate_temp_bytes(
        file_uncompressed_bytes, total_rows, n_kept,
    )
    # The disk dir doubles as the per-shard ENOSPC fallback, so it is always
    # created — empty, it costs nothing — and workers never race to mkdir it.
    disk_dir = _disk_temp_dir_for(output_path)
    temp_dir, _why = _choose_temp_dir(
        _need_bytes, _shm_temp_dir_for(output_path), disk_dir,
    )
    print(f"  Step-4 temp dir: {_why}", flush=True)
    fallback_dir = None if temp_dir == disk_dir else disk_dir

    # Claim both dirs: clears leftovers from an interrupted run, but refuses if
    # another live process owns them (a second assemble of this same output
    # path, which would corrupt both runs — see _claim_temp_dir).
    # Claiming the second dir can fail, so unwind the first — the layered
    # cleanup below is not armed yet.
    _claimed: List[Path] = []
    try:
        for _d in (temp_dir, fallback_dir):
            if _d is not None:
                _claim_temp_dir(_d)
                _claimed.append(_d)
    except BaseException:
        for _d in _claimed:
            shutil.rmtree(_d, ignore_errors=True)
        raise

    # Failsafe cleanup, layered so every way this process can die is covered:
    #   try/finally  -> exceptions and KeyboardInterrupt (SIGINT)
    #   atexit       -> sys.exit() and unwinding to the interpreter top level
    #   signal       -> SIGTERM/SIGHUP (kill, terminal loss, scheduler eviction)
    #   orphan sweep -> SIGKILL/OOM/reboot, reclaimed by the NEXT run (above)
    # Without these, a mid-run failure strands the whole temp set — on /dev/shm
    # that is shared, system-wide memory, and it is how ~1 TB was once left behind.
    # NOTE: the Step-4 Pool and datasets.load_dataset(num_proc=…) both fork, and
    # children inherit both the atexit registration and the signal handlers. A
    # child exiting normally would then delete the temp dirs out from under the
    # still-running parent. Every cleanup path is therefore pinned to the pid
    # that registered it.
    _cleanup_dirs = [d for d in (temp_dir, fallback_dir) if d is not None]
    _owner_pid = os.getpid()

    def _cleanup_temp_dirs() -> None:
        if os.getpid() != _owner_pid:
            return                      # a forked child: not ours to remove
        for d in _cleanup_dirs:
            shutil.rmtree(d, ignore_errors=True)

    def _cleanup_and_reraise(signum, _frame):
        _cleanup_temp_dirs()
        # signal.signal() returns None for a handler that was not installed from
        # Python, and passing None back raises TypeError — inside a signal
        # handler that would leave the process alive with cleanup already done.
        _prev = _prev_handlers.get(signum)
        signal.signal(signum, _prev if _prev is not None else signal.SIG_DFL)
        os.kill(os.getpid(), signum)   # preserve the true exit status

    atexit.register(_cleanup_temp_dirs)
    _prev_handlers = {}
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            _prev_handlers[_sig] = signal.signal(_sig, _cleanup_and_reraise)
        except (ValueError, OSError):
            pass   # not on the main thread — try/finally + atexit still apply

    try:
        # Place all large arrays in POSIX shared memory so workers share one
        # physical copy with no COW duplication and no IPC pickle overhead.
        mask_shape, mask_dtype = keep_mask.shape, keep_mask.dtype
        inv_shape,  inv_dtype  = inverse.shape,   inverse.dtype

        shm_mask = SharedMemory(create=True, size=keep_mask.nbytes)
        shm_inv  = SharedMemory(create=True, size=inverse.nbytes)
        shm_gw: Optional[SharedMemory] = None
        gw_name: Optional[str] = None
        gw_shape: Optional[tuple] = None
        gw_dtype = None
        # Publish the (optional) version-conversion transform to the module global so
        # the workers forked by the Pool below inherit it (mirrors _EXTERNALIZE_MARKERS
        # / _FILTER_GLOBALS). Reset in the finally so it never leaks across splits.
        global _FILTER_TRANSFORM, _FILTER_CONCURRENCY
        _FILTER_TRANSFORM = shard_transform
        _FILTER_CONCURRENCY = n_workers
        try:
            np.ndarray(mask_shape, dtype=mask_dtype, buffer=shm_mask.buf)[:] = keep_mask
            np.ndarray(inv_shape,  dtype=inv_dtype,  buffer=shm_inv.buf)[:] = inverse
            del keep_mask, inverse

            if group_weight is not None:
                shm_gw = SharedMemory(create=True, size=group_weight.nbytes)
                np.ndarray(group_weight.shape, dtype=group_weight.dtype,
                           buffer=shm_gw.buf)[:] = group_weight
                gw_name, gw_shape, gw_dtype = (
                    shm_gw.name, group_weight.shape, group_weight.dtype,
                )
                del group_weight

            offsets = np.concatenate(([0], np.cumsum(file_row_counts[:-1])))
            _fb = None if fallback_dir is None else str(fallback_dir)
            tasks = [
                (str(pf), str(temp_dir), int(off), int(n), _fb)
                for pf, off, n in zip(parquet_files, offsets, file_row_counts)
            ]

            with Pool(
                processes=n_workers,
                initializer=_init_filter_worker,
                initargs=(
                    shm_mask.name, mask_shape, mask_dtype,
                    shm_inv.name,  inv_shape,  inv_dtype,
                    gw_name, gw_shape, gw_dtype,
                ),
            ) as pool:
                results = pool.map(_filter_write_shard, tasks)
        finally:
            shm_mask.close(); shm_mask.unlink()
            shm_inv.close();  shm_inv.unlink()
            if shm_gw is not None:
                shm_gw.close(); shm_gw.unlink()
            _FILTER_TRANSFORM = None
            _FILTER_CONCURRENCY = 1

        temp_files = [Path(r) for r in results if r is not None]
        t_filter = time.perf_counter() - t0
        print(f"  Filter+write complete in {t_filter:.1f}s "
              f"({len(temp_files)} non-empty shards)", flush=True)
        if not temp_files:
            # Degenerate but worth naming: Step 5 reads the schema off the first
            # shard, so an empty list would surface as a bare IndexError here
            # rather than as what actually went wrong upstream.
            print(f"ERROR: every shard in {shard_dir} filtered to zero rows — "
                  f"nothing to assemble", file=sys.stderr)
            sys.exit(1)

        # ── Step 5: Load via HF (memory-mapped), shuffle, save ────────────
        # Each sub-step is timed separately so the per-step cost — and the gain from
        # the `num_proc` parallelism — is visible in the log next run.  load_dataset
        # is CPU-decode bound and takes the full n_workers; save_to_disk is
        # random-read-IOPS bound and is capped (see _SAVE_TO_DISK_MAX_PROC).  Note
        # `.shuffle()` only builds a lazy index, so its random-gather cost is charged
        # to `save` below, not `shuffle`.
        save_num_proc = min(n_workers, _SAVE_TO_DISK_MAX_PROC)

        print(f"  Loading {len(temp_files)} deduplicated shards via HF "
              f"(num_proc={n_workers})…", flush=True)
        t_load0 = time.perf_counter()
        ds = datasets.load_dataset(
            "parquet",
            data_files=[str(f) for f in temp_files],
            split="train",
            cache_dir=hf_cache_dir,
            num_proc=n_workers,
            features=_large_list_features(temp_files[0]),
        )
        t_load = time.perf_counter() - t_load0
        print(f"  load_dataset complete in {t_load:.1f}s (num_proc={n_workers})",
              flush=True)

        print(f"  Shuffling {len(ds):,} records…", flush=True)
        t_shuf0 = time.perf_counter()
        ds = ds.shuffle(seed=seed)
        t_shuf = time.perf_counter() - t_shuf0
        print(f"  shuffle (index build) complete in {t_shuf:.1f}s", flush=True)

        # save_to_disk does NOT clean its destination, so anything a crashed
        # run left here would be MERGED with this run's output — stale
        # data-*.arrow files from a longer run and all (HARDENING_PLAN.md §A5).
        # Runs on every run, not only resumes.
        _freed = _clear_partial_output(output_path, shard_dir)
        if _freed:
            print(f"  Cleared partial output at {output_path} "
                  f"({_fmt_bytes(_freed)} removed)", flush=True)

        # Last chance to name the real cause: save_to_disk(num_proc>1) spawns
        # workers that re-exec sys.executable, so it is the step the 2026-09-03
        # venv deletion would kill — 40 h into the BioMe run.
        _verify_venv_fingerprint("immediately before save_to_disk")

        print(f"  Saving to {output_path} (num_proc={save_num_proc})…", flush=True)
        t_save0 = time.perf_counter()
        ds.save_to_disk(
            str(output_path),
            max_shard_size=max_shard_size,
            num_proc=save_num_proc,
        )
        t_save = time.perf_counter() - t_save0
        print(f"  save_to_disk complete in {t_save:.1f}s (num_proc={save_num_proc})",
              flush=True)

        n_final = len(ds)
        del ds

        # ── The one exact invariant this pipeline has (HARDENING_PLAN.md §A9.3)
        # Step 4 writes exactly the kept rows; load_dataset and shuffle both
        # preserve cardinality.  The stats dict has always recorded both numbers
        # and never compared them — this assertion would have caught the
        # `List index overflow`-class truncation the day it happened instead of
        # six hours in, and it is what makes a resumed split trustworthy.
        if n_final != n_kept:
            raise RuntimeError(
                f"Assembled row count {n_final:,} != deduplicated row count "
                f"{n_kept:,} for {output_path} (difference "
                f"{n_final - n_kept:+,}). Step 4 writes exactly the kept rows "
                f"and Step 5 preserves cardinality, so this means rows were "
                f"lost or duplicated between them — most likely a temp shard "
                f"that failed to write, or a stale shard left in "
                f"{temp_dir}. The output is NOT trustworthy; do not use it."
            )

        # One grep-able per-split timing summary for cross-run comparison.
        print(
            f"  [TIMING {output_path.name}] rowcount={t_rowcount:.1f}s "
            f"hash={t_hash:.1f}s dedup={t_dedup:.1f}s filter_write={t_filter:.1f}s "
            f"load={t_load:.1f}s shuffle={t_shuf:.1f}s save={t_save:.1f}s "
            f"(load_num_proc={n_workers}, save_num_proc={save_num_proc})",
            flush=True,
        )

        # (Temp dirs are removed in the finally below — on every exit path.)

        stats = {
            "n_final": int(n_final),
            "total_rows": int(total_rows),
            "n_dups": int(n_dups),
            "dup_ratio": round(dup_ratio, 4),
            "n_kept": int(n_kept),
            "timing_s": {
                "rowcount": round(t_rowcount, 1),
                "hash": round(t_hash, 1),
                "dedup": round(t_dedup, 1),
                "filter_write": round(t_filter, 1),
                "load": round(t_load, 1),
                "shuffle": round(t_shuf, 1),
                "save": round(t_save, 1),
            },
            "load_num_proc": n_workers,
            "save_num_proc": save_num_proc,
        }

        # ── Split-completion marker, written LAST (HARDENING_PLAN.md §A4) ──
        # `dataset_info.json` is not this signal: save_to_disk writes it
        # incrementally under num_proc, so a crash mid-save leaves a
        # plausible-looking partial.  This file exists only when the split
        # genuinely finished, and it is written atomically so it can never be
        # observed half-formed.
        _marker_path = _write_split_marker(output_path, {
            "marker_version": _SPLIT_MARKER_VERSION,
            "n_final": int(n_final),
            "n_arrow_shards": len(list(output_path.glob("data-*.arrow"))),
            "fingerprint": split_fingerprint,
            "seed": int(seed),
            "max_shard_size": max_shard_size,
            "finished_at": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "command": " ".join(sys.argv),
            "written_by": {"pid": os.getpid()},
            "stats": stats,
        })
        print(f"  Split marked complete: {_marker_path}", flush=True)

        return n_final, stats
    finally:
        # Failsafe: whatever happened above — success, exception, SIGINT — the
        # temp dirs go away here. On /dev/shm they are shared, system-wide
        # memory; stranding them breaks other users, not just this job.
        _cleanup_temp_dirs()
        atexit.unregister(_cleanup_temp_dirs)
        for _sig, _prev in _prev_handlers.items():
            if _prev is None:
                continue        # not a Python-installed handler; nothing to restore
            try:
                signal.signal(_sig, _prev)
            except (ValueError, OSError):
                pass


def _sniff_shard_columns(shard_dir: Path) -> List[str]:
    """Column names of the first ``*.parquet`` shard in *shard_dir*.

    Raises ``RuntimeError`` if the directory holds no parquet shards (so a typo'd
    path or an empty dir is named, not surfaced as an opaque IndexError)."""
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"No .parquet shards found in {shard_dir}.")
    return list(pq.read_schema(shards[0]).names)


def _detect_source_version(shard_dir: Path) -> Tuple[Optional[str], List[str], dict]:
    """Read the generation provenance and cross-check it against the shards.

    Returns ``(schema_version, columns, generation_provenance)``. ``schema_version``
    is ``None`` when no ``generation_provenance.json`` is present (legacy shards).
    Raises ``RuntimeError`` if the recorded provenance disagrees with the columns
    actually on disk — a hand-edited sidecar or a half-written dir — so a wrong
    route is never chosen from a mislabeled directory.
    """
    gen_sidecar = shard_dir / _GEN_PROVENANCE_FILENAME
    gen = _load_json_strict(gen_sidecar) if gen_sidecar.is_file() else {}
    columns = _sniff_shard_columns(shard_dir)
    if gen:
        externalize = bool(gen.get("externalize_markers", False))
        if externalize != ("snp" in columns):
            raise RuntimeError(
                f"Provenance/columns mismatch in {shard_dir}: provenance says "
                f"externalize_markers={externalize} but the shard "
                f"{'has' if 'snp' in columns else 'has no'} 'snp' column."
            )
        for annot in gen.get("annotations", []):
            name = annot.get("name")
            if name and name not in columns:
                raise RuntimeError(
                    f"Provenance/columns mismatch in {shard_dir}: provenance lists "
                    f"annotation {name!r} but the shard has no {name!r} column."
                )
    return gen.get("schema_version"), columns, gen


def cmd_check_compat(args: argparse.Namespace) -> None:
    """Test whether existing shards can feed a target version (read-only).

    Exit codes: 0 convertible (incl. identity), 1 incompatible, 2 cannot
    determine the source version, 3 provenance/columns mismatch.
    """
    print(f"Command: {' '.join(sys.argv)}", flush=True)
    shard_dir = args.shard_dir
    target = args.target_version
    if not shard_dir.is_dir():
        print(f"ERROR: shard dir {shard_dir} does not exist.", file=sys.stderr)
        sys.exit(2)
    try:
        source, columns, _gen = _detect_source_version(shard_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(3)
    if source is None:
        print(
            f"ERROR: no {_GEN_PROVENANCE_FILENAME} in {shard_dir}; cannot "
            f"determine the source version. Regenerate the shards with an "
            f"explicit --version so provenance records it.", file=sys.stderr)
        sys.exit(2)

    decision = compat.conversion_route(source, target)
    print(f"Source version : {source}")
    print(f"Target version : {target}")
    print(f"Shard columns  : {', '.join(columns)}")
    print(f"Route          : {decision.route.value}")
    print(f"Reason         : {decision.reason}")
    if decision.route is compat.Route.INCOMPATIBLE:
        print(f"INCOMPATIBLE — these shards cannot be assembled into {target}; "
              f"regenerate.", flush=True)
        sys.exit(1)
    if decision.route is compat.Route.IDENTITY:
        print(f"OK — shards already are {target}; assemble normally "
              f"(no --target-version needed).", flush=True)
    else:
        print(f"OK — assemble with `--target-version {target}` to convert "
              f"on the fly (no regeneration needed).", flush=True)
    sys.exit(0)


def cmd_assemble(args: argparse.Namespace) -> None:
    """Assemble per-sample Parquet shards into a HuggingFace dataset."""
    t_start = time.perf_counter()
    assemble_started = datetime.datetime.now()
    assemble_started_str = f"{assemble_started:%Y-%m-%d %H:%M:%S}"
    print(f"Assemble started at {assemble_started_str}", flush=True)
    print(f"Command: {' '.join(sys.argv)}", flush=True)

    shard_dir = args.shard_dir
    output_dir = args.output
    max_shard_size = args.max_shard_size
    # A re-assemble is any invocation whose output already carries an assembled
    # provenance.json from a prior run (assemble rebuilds from scratch — it has
    # no incremental resume — but a re-run is still documented as resume=True).
    assemble_is_resume = (
        (output_dir / "provenance.json").exists()
        or (output_dir / "train" / "provenance.json").exists()
    )
    # Optional on-the-fly version conversion. Resolved inside the try block below
    # (so a bad route prints the "Assemble FAILED" banner); these defaults mean
    # "no conversion" (identity), preserving the plain-assemble behavior.
    target_version: Optional[str] = args.target_version
    # source_version is resolved inside the conversion block but referenced by the
    # run record below, so initialise it here (None ⇒ no conversion recorded).
    source_version: Optional[str] = None
    decision: Optional[compat.RouteDecision] = None
    shard_transform: Optional[Callable[[pa.Table], pa.Table]] = None
    seq_normalizer: Optional[Callable[[str], str]] = None
    # One string covering target_version, shard_transform AND seq_normalizer —
    # all three derive from the same RouteDecision — so the split-resume check
    # can tell two different conversion routes into the same output apart.
    route_label: Optional[str] = None
    # Default HF cache to a sibling of the output dir (same filesystem) to
    # avoid filling the home partition with multi-TB Arrow cache files.
    if args.hf_cache_dir:
        hf_cache_dir = str(args.hf_cache_dir)
    else:
        hf_cache_dir = str(output_dir.parent / ".hf_cache_assemble")
    print(f"HF cache dir: {hf_cache_dir}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    def _read_generation_provenance() -> dict:
        """Read generation_provenance.json from the (train) shard dir, if any."""
        gen_sidecar = shard_dir / _GEN_PROVENANCE_FILENAME
        if gen_sidecar.is_file():
            return _load_json_strict(gen_sidecar)
        return {}

    def _write_assembled_provenance(target: Path) -> None:
        """Write provenance.json next to the assembled HF dataset.

        Records schema_version, representation, marker_policy and the
        annotation provenance entries so loaders (DiploidDataset /
        HaploidDataset) can validate the dataset.
        """
        gen = _read_generation_provenance()
        if not gen:
            # No generation sidecar (legacy shards or a deleted file): the
            # fields below fall back to diploid defaults. Make that inference
            # loud — silently stamping "diploid" onto haploid shards would
            # mislabel the dataset for every downstream loader.
            print(
                f"  WARNING: no {_GEN_PROVENANCE_FILENAME} in {shard_dir}; "
                f"inferring provenance from defaults (representation=diploid, "
                f"marker_policy={ACTIVE_MARKER_POLICY.value}). If these shards "
                f"are NOT diploid, the written provenance.json will be wrong.",
                flush=True,
            )
            logger.warning(
                "Assembled provenance inferred from defaults — no %s in %s",
                _GEN_PROVENANCE_FILENAME, shard_dir,
            )
        # Propagate the schema version recorded at generation time (v4/v5/v6/v6nd/v7/v8)
        # — a v5/v6 generate must assemble into a v5/v6-stamped dataset, not v7.
        # There is no default version, so an absent schema_version is a hard
        # error: we refuse to silently invent one.
        gen_schema = gen.get("schema_version")
        if gen_schema is None:
            raise RuntimeError(
                f"Cannot determine schema_version for assembly: it is not "
                f"recorded in {_GEN_PROVENANCE_FILENAME} under {shard_dir}. "
                f"There is no default version — regenerate the shards with an "
                f"explicit `--version` so provenance records it."
            )
        if gen_schema not in SUPPORTED_VERSIONS:
            print(
                f"  WARNING: generation provenance reports schema_version="
                f"{gen_schema!r}, which is not a known preset "
                f"({', '.join(sorted(SUPPORTED_VERSIONS))}). Stamping it as-is.",
                flush=True,
            )
        if target_version is not None and target_version != gen_schema:
            # On-the-fly conversion: the assembled dataset IS the target version,
            # so stamp it and recompute the version-dependent fields from the
            # target preset (not the source generation sidecar). The shards were
            # transformed in Step 4; provenance must describe the OUTPUT.
            tgt_preset = versions.resolve(target_version)
            keep_annotations = (
                decision is not None
                and decision.route is compat.Route.REHYDRATE_REALIGN
            )
            prov = {
                "schema_version": target_version,
                "representation": gen.get("representation", "diploid"),
                "marker_policy": tgt_preset.marker_policy.value,
                "externalize_markers": bool(tgt_preset.externalize_markers),
                "annotations": gen.get("annotations", []) if keep_annotations else [],
            }
        else:
            prov = {
                "schema_version": gen_schema,
                "representation": gen.get("representation", "diploid"),
                "marker_policy": gen.get("marker_policy", ACTIVE_MARKER_POLICY.value),
                "externalize_markers": bool(gen.get("externalize_markers", False)),
                "annotations": gen.get("annotations", annotation_provenance()),
            }
        # Embed the generate lineage (schema + the full generate run history) so
        # the assembled dataset is a single self-contained provenance record —
        # the consumer of the final dataset can trace both steps from one file.
        if gen:
            prov["generation"] = {
                "schema_version": gen.get("schema_version"),
                "inputs": gen.get("inputs", {}),
                "runs": gen.get("runs", []),
            }
        # Atomic .tmp -> rename so a crash mid-write can't truncate the sidecar.
        prov_path = target / "provenance.json"
        tmp = prov_path.parent / (prov_path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(prov, f, indent=2)
        tmp.rename(prov_path)

    # Exactly one assemble per machine — see the machine-wide lock block. Taken
    # before any work so a conflict costs nothing, and held across BOTH splits
    # (train and test run in sequence below) rather than per split. A stale lock
    # from a killed run is reclaimed here automatically.
    try:
        _assemble_lock = _acquire_assemble_lock()
    except RuntimeError as exc:
        print(f"Error: {exc}", flush=True, file=sys.stderr)
        sys.exit(1)

    try:
        # Resolve the on-the-fly conversion route (if --target-version was given)
        # before touching any shards, so an incompatible request fails fast.
        if target_version is not None:
            source_version, _src_cols, _src_gen = _detect_source_version(shard_dir)
            if source_version is None:
                raise RuntimeError(
                    f"--target-version {target_version} was requested but the "
                    f"source version is unknown: no {_GEN_PROVENANCE_FILENAME} in "
                    f"{shard_dir}. Regenerate the shards with an explicit --version."
                )
            decision = compat.conversion_route(source_version, target_version)
            if decision.route is compat.Route.INCOMPATIBLE:
                raise RuntimeError(
                    f"Cannot convert {source_version} shards to {target_version}: "
                    f"{decision.reason}"
                )
            # Region-bound target: cross-check the BED actually used to generate
            # these shards. conversion_route only compares the two presets, which
            # trusts the sidecar's schema_version label. Sidecars are sometimes
            # hand-written (the v6→v6nd conversion required it), so verify the
            # recorded BED hash too before stamping a region-bound label.
            _tgt_preset = versions.resolve(target_version)
            if _tgt_preset.region_bed_sha256 is not None:
                recorded_bed = _src_gen.get("inputs", {}).get("bed_sha256")
                if recorded_bed is None:
                    print(
                        f"  WARNING: {shard_dir} records no inputs.bed_sha256, so "
                        f"the '{_tgt_preset.region_set}' region set of "
                        f"--target-version {target_version} cannot be verified "
                        f"(legacy sidecar). Proceeding on the recorded "
                        f"schema_version alone.", flush=True,
                    )
                elif recorded_bed != _tgt_preset.region_bed_sha256:
                    raise RuntimeError(
                        f"Cannot convert {shard_dir} to {target_version}: those "
                        f"shards were generated from a BED whose sha256 is "
                        f"{recorded_bed}, but {target_version} is bound to the "
                        f"'{_tgt_preset.region_set}' region set "
                        f"({_tgt_preset.region_bed_hint}, sha256 "
                        f"{_tgt_preset.region_bed_sha256}). The shards do not "
                        f"cover that region universe."
                    )
            # The test split must share the source version — converting a mixed
            # pair would silently produce two different schemas.
            if args.test_shard_dir is not None:
                test_source, _, _ = _detect_source_version(args.test_shard_dir)
                if test_source != source_version:
                    raise RuntimeError(
                        f"train shards are {source_version} but test shards are "
                        f"{test_source}; cannot convert a mismatched split pair."
                    )
            shard_transform = compat.route_transform(decision.route)
            seq_normalizer = compat.route_seq_normalizer(decision.route)
            route_label = (
                f"{source_version}->{target_version}:{decision.route.value}:"
                f"pre_dedup={bool(decision.pre_dedup)}"
            )
            if decision.route is compat.Route.IDENTITY:
                print(f"--target-version {target_version} matches the source; "
                      f"assembling without conversion.", flush=True)
            else:
                pre_note = " [pre-dedup normalisation active]" if decision.pre_dedup else ""
                print(f"On-the-fly conversion: {source_version} → "
                      f"{target_version} [{decision.route.value}]{pre_note} — "
                      f"{decision.reason}", flush=True)

        # Process train split.
        print("Processing train split…", flush=True)
        train_path = output_dir / "train" if args.test_shard_dir else output_dir
        n_train, train_stats = _process_split_streaming(
            shard_dir, train_path, max_shard_size, seed=42,
            hf_cache_dir=hf_cache_dir, shard_transform=shard_transform,
            seq_normalizer=seq_normalizer, route_label=route_label,
        )
        _write_assembled_provenance(train_path)

        # Process test split.
        n_test, test_stats = 0, None
        if args.test_shard_dir:
            print("\nProcessing test split…", flush=True)
            test_path = output_dir / "test"
            n_test, test_stats = _process_split_streaming(
                args.test_shard_dir, test_path, max_shard_size, seed=42,
                hf_cache_dir=hf_cache_dir, shard_transform=shard_transform,
                seq_normalizer=seq_normalizer, route_label=route_label,
            )
            _write_assembled_provenance(test_path)

            # Write DatasetDict metadata so load_from_disk sees train/test splits.
            dd_meta = output_dir / "dataset_dict.json"
            with open(dd_meta, "w") as f:
                json.dump({"splits": ["train", "test"]}, f)
            # Also write top-level provenance so loaders pointed at the
            # parent dir (rather than a split) still see the schema info.
            _write_assembled_provenance(output_dir)

        elapsed = time.perf_counter() - t_start
        if args.test_shard_dir:
            print(f"\nAssembled train={n_train:,}, test={n_test:,} "
                  f"records → {output_dir}", flush=True)
        else:
            print(f"\nAssembled {n_train:,} records → {output_dir}", flush=True)
        print(f"Total assembly time: {elapsed:.1f}s", flush=True)
        print(f"Assemble finished successfully at "
              f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
        logger.info("Saved dataset to %s", output_dir)

        # ── Run-documentation: append one record to each provenance.json ──
        # Params, timestamps, per-split row/dedup/timing breakdown (threaded out
        # of _process_split_streaming), and the conversion route if any. Written
        # alongside the schema dict + embedded generate lineage above, so the
        # assembled dataset is a single self-contained provenance record.
        assemble_record = {
            "command": " ".join(sys.argv),
            "started_at": assemble_started_str,
            "finished_at": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "duration_seconds": round(elapsed, 1),
            "status": "ok",
            "resume": assemble_is_resume,
            "params": {
                "shard_dir": str(shard_dir),
                "test_shard_dir": str(args.test_shard_dir) if args.test_shard_dir else None,
                "output": str(output_dir),
                "max_shard_size": max_shard_size,
                "target_version": target_version,
                "hf_cache_dir": hf_cache_dir,
                "seed": 42,
            },
            "splits": {"train": train_stats, "test": test_stats},
            "conversion": (
                None if decision is None or target_version is None
                else {
                    "route": decision.route.value,
                    "source": source_version,
                    "target": target_version,
                }
            ),
        }
        _append_run_record(train_path / "provenance.json", assemble_record)
        if args.test_shard_dir:
            _append_run_record(test_path / "provenance.json", assemble_record)
            _append_run_record(output_dir / "provenance.json", assemble_record)

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t_start
        print(f"\nAssemble INTERRUPTED after {elapsed:.1f}s at "
              f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
        sys.exit(1)
    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        print(f"\nAssemble FAILED after {elapsed:.1f}s at "
              f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
        print(f"Error: {exc}", flush=True, file=sys.stderr)
        raise
    finally:
        # Always hand the lock back — including on the sys.exit() paths above.
        # A lock lost to SIGKILL needs no handling here: the next run sees its
        # owner is dead and reclaims it.
        _release_assemble_lock(_assemble_lock)


# ---------------------------------------------------------------------------
# Startup preflight for lazily-resolved dependencies  (HARDENING_PLAN.md §B)
#
# 2026-09-03: a second host's bare `uv run` deleted and recreated the shared
# /storage venv while an assemble was 4h26m in. The process survived for another
# ~3 minutes on already-imported modules, then `datasets` built its pool and
# `multiprocess.context.SimpleQueue` did `from .queues import SimpleQueue` — a
# LAZY import — and the run died with `ModuleNotFoundError: No module named
# 'multiprocess.queues'`, losing 15,943 s of work.
#
# `multiprocess` is never imported by this module: it arrives transitively when
# `datasets` builds its pool, and `datasets` itself is imported lazily inside
# `_process_split_streaming`. So a venv that disappears at hour 4 stays invisible
# until Step 5 touches it. Three rungs, cheapest first:
#
#   A  Import every module the run will need — including the submodules that are
#      resolved lazily — so `sys.modules` holds them from t=0. This is the rung
#      that actually PREVENTS the incident: with `multiprocess.queues` resident,
#      the venv can be deleted at hour 4 and Step 5 still completes.
#   B  One real fork-pool round trip, because the lazy set cannot be enumerated
#      by hand. Constructing the pool is what triggered the crash import.
#   C  One SPAWN round trip, assemble only. `save_to_disk(num_proc=16)` uses a
#      spawn pool that re-execs sys.executable ~40 h into the BioMe run, and
#      NOTHING else in the run exercises that path. This is the rung that catches
#      the version-mismatched `.venv/bin/python3` the ledger warns about.
#
# Honest limits: rung A cannot save the spawn pool (it re-execs; no in-process
# trick survives that), and the primary control is host-local venvs, already
# asserted in launch_biome_generate.sh. This is defence in depth, plus a venv
# fingerprint that converts "ModuleNotFoundError 40 h in" into a named cause.
# ---------------------------------------------------------------------------

# Submodules of `multiprocess` that `datasets` resolves lazily at pool
# construction time. `queues` is the one that killed the 2026-09-03 assemble;
# the rest are its neighbours on the same lazy path.
_LAZY_MP_SUBMODULES = (
    "context", "queues", "pool", "connection", "reduction",
    "synchronize", "sharedctypes", "resource_tracker", "popen_fork",
)

_VENV_FINGERPRINT: Optional[dict] = None


def _preflight_echo(x):
    """Round-trip payload for rungs B and C — module-level so it is picklable."""
    return x * 2


def _preflight_spawn_child(x):
    """Rung C's child body: prove a re-exec'd interpreter re-imports the venv.

    A spawn child starts from a bare interpreter and imports everything from
    scratch, so this is the only thing in the run that would notice a
    `.venv/bin/python3` whose version no longer matches the installed packages.
    """
    import datasets  # noqa: F401
    import numpy  # noqa: F401
    import pyarrow  # noqa: F401
    return x * 2


def _venv_fingerprint() -> dict:
    """Identity of the interpreter and the two packages that resolve lazily."""
    fp: dict = {
        "sys_prefix": sys.prefix,
        "sys_executable": sys.executable,
        "sys_version": sys.version,
    }
    fp["executable_ids"] = _path_ids(Path(sys.executable))
    for name in ("multiprocess", "datasets"):
        mod = sys.modules.get(name)
        pkg_dir = (
            str(Path(mod.__file__).parent)
            if mod is not None and getattr(mod, "__file__", None) else None
        )
        fp[f"{name}_dir"] = pkg_dir
        fp[f"{name}_dir_ids"] = (
            _path_ids(Path(pkg_dir)) if pkg_dir is not None else None
        )
    return fp


def _path_ids(path: Path) -> Optional[List[int]]:
    """(st_dev, st_ino) as a JSON-friendly list, or None if it is gone.

    Identity, not mtime: /storage mtimes come from clocks that disagree with this
    host by up to 47 minutes (MONITOR.md §4). A venv that `uv` deleted and
    recreated has the same path and a new inode, which is exactly what this
    catches.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_dev, st.st_ino]


def _verify_venv_fingerprint(where: str) -> None:
    """Re-stat the recorded venv (microseconds) and raise if it moved.

    A no-op when no fingerprint was recorded — direct callers of
    `_process_split_streaming` (the ledger's 157 s repro recipe) bypass
    `main()`, and that is correct.
    """
    if _VENV_FINGERPRINT is None:
        return
    now = _venv_fingerprint()
    diffs = [
        f"{k}: {_VENV_FINGERPRINT[k]!r} → {now[k]!r}"
        for k in ("sys_prefix", "sys_executable", "executable_ids",
                  "multiprocess_dir_ids", "datasets_dir_ids")
        if _VENV_FINGERPRINT.get(k) != now.get(k)
    ]
    if not diffs:
        return
    raise RuntimeError(
        f"The Python environment changed underneath this run, detected "
        f"{where}:\n  " + "\n  ".join(diffs) + "\n"
        "This is the 2026-09-03 failure: a `uv` invocation on ANOTHER HOST "
        "judged the shared venv broken and recreated it, which kills every "
        "lazily-resolved import (multiprocess.queues) and the spawn pool "
        "save_to_disk re-execs. Fix: give every non-primary host its own venv "
        "outside /storage — NEVER run `uv` from a second host without "
        "UV_PROJECT_ENVIRONMENT (see preprocess/CLAUDE.md, 'Running Scripts')."
    )


def _import_or_die(name: str, why: str) -> None:
    """Import one module now, or fail with the venv named as the suspect."""
    try:
        importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"Preflight failed to import {name!r} ({why}).\n"
            f"  sys.prefix     = {sys.prefix}\n"
            f"  sys.executable = {sys.executable}\n"
            f"  original error = {exc}\n"
            f"This is checked at startup on purpose: {name!r} would otherwise be "
            f"resolved hours into the run (see DECISION_LEDGER.md 2026-09-03, "
            f"where a recreated venv killed an assemble 4h26m in). If the venv "
            f"lives on /storage and another host ran `uv`, recreate it with the "
            f"interpreter this host uses and set UV_PROJECT_ENVIRONMENT there."
        ) from exc


def _preflight_imports(args: argparse.Namespace) -> None:
    """Resolve every lazily-imported dependency before the long work starts.

    Called from `main()` once `args.command` is known, so `--help` stays fast and
    the read-only `check-compat` pays nothing. Direct callers of `cmd_*` bypass
    it, which is deliberate.
    """
    global _VENV_FINGERPRINT
    level = getattr(args, "preflight", "full")
    if level == "off":
        return
    t0 = time.perf_counter()
    command = args.command

    if command == "assemble":
        # `datasets` first: it is the heavy import the parent needs anyway, and
        # it is what drags `multiprocess` in.
        _import_or_die("datasets", "Step 5 loads and saves through it")
        _import_or_die("multiprocess", "datasets' pools are built on it")
        for sub in _LAZY_MP_SUBMODULES:
            _import_or_die(
                f"multiprocess.{sub}",
                "resolved lazily when datasets builds its pool",
            )
        _import_or_die("pyarrow.ipc", "save_to_disk writes Arrow IPC files")
    else:  # generate
        for sub in _LAZY_MP_SUBMODULES:
            _import_or_die(
                f"multiprocessing.{sub}",
                "resolved lazily when the worker pool is built",
            )
        _import_or_die("cyvcf2", "the per-sample BCF reader in every worker")
        if getattr(args, "phylop_bw", None) is not None:
            _import_or_die("pyBigWig",
                           "--phylop-bw; imported inside each forked worker")
        if getattr(args, "cadd_arrow_dir", None) is not None:
            _import_or_die("pyarrow.ipc",
                           "--cadd-arrow-dir reads Arrow IPC shards")

    # Rung B — one real fork-pool round trip. The lazy set cannot be enumerated
    # by hand, and building the pool is what triggered the 2026-09-03 import.
    mp = sys.modules["multiprocess"] if command == "assemble" else multiprocessing
    try:
        with mp.Pool(1) as p:
            got = p.apply(_preflight_echo, (21,))
    except Exception as exc:
        raise RuntimeError(
            f"Preflight fork-pool round trip failed ({type(exc).__name__}: "
            f"{exc}). sys.prefix={sys.prefix}, sys.executable={sys.executable}. "
            f"The worker pools this run depends on cannot be built."
        ) from exc
    if got != 42:
        raise RuntimeError(
            f"Preflight fork-pool round trip returned {got!r}, expected 42."
        )

    # Rung C — one spawn round trip (assemble only, and only at --preflight full).
    # save_to_disk(num_proc=16) re-execs sys.executable ~40 h into the BioMe run
    # and nothing else in the run touches that path.
    if command == "assemble" and level == "full":
        try:
            with sys.modules["multiprocess"].get_context("spawn").Pool(1) as p:
                got = p.apply(_preflight_spawn_child, (21,))
        except Exception as exc:
            raise RuntimeError(
                f"Preflight SPAWN round trip failed ({type(exc).__name__}: "
                f"{exc}).\n  sys.executable = {sys.executable}\n"
                f"  sys.prefix     = {sys.prefix}\n"
                f"save_to_disk(num_proc>1) spawns workers that re-exec that "
                f"interpreter; a version-mismatched .venv/bin/python3 breaks it "
                f"HOURS into the run (DECISION_LEDGER.md 2026-09-03). Recreate "
                f"the venv with the interpreter this host actually uses: "
                f"`uv sync --python <that interpreter>`. Use --preflight fast "
                f"to skip this check."
            ) from exc
        if got != 42:
            raise RuntimeError(
                f"Preflight spawn round trip returned {got!r}, expected 42."
            )

    _VENV_FINGERPRINT = _venv_fingerprint()
    print(f"Preflight OK ({level}) in {time.perf_counter() - t0:.1f}s — "
          f"venv {sys.prefix}", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── generate ────────────────────────────────────────────────────────
    gen = subparsers.add_parser(
        "generate",
        help="Generate per-sample Parquet shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    gen.add_argument("--bcf-dir", type=Path, required=True,
                     help="Directory of BCFs. Either per-sample single-file "
                          "layout (<sample>.bcf, multi-contig, GT-only; built "
                          "by make_persample_bcf.sh) or legacy per-chromosome "
                          "multi-sample *.bcf files. Per-sample is auto-detected "
                          "and is much faster (no multi-sample genotype decode).")
    gen.add_argument("--bed", type=Path, required=True,
                     help="BED file with regions to process (0-based half-open)")
    gen.add_argument("--ref", type=Path, required=True,
                     help="Reference genome file (.2bit or FASTA)")
    gen.add_argument("--tokens", type=Path, required=False, default=None,
                     help="Diploid token mapping TSV (diploid_tokens.tsv). "
                          "Required when --representation=diploid. Optional "
                          "when --representation=haploid (only the meta "
                          "tokens <SNP>, <INDEL_span_start>, <INDEL_span_end> "
                          "are honoured; defaults '^', '{', '}' are used if "
                          "omitted).")
    gen.add_argument("--version", choices=sorted(versions.SUPPORTED_VERSIONS),
                     required=True, dest="version",
                     help="Dataset version preset (REQUIRED — there is no "
                          "default). Selects a behavioral bundle instead of "
                          "individual encoder knobs: 'v8' (v7 content but "
                          "structural markers moved OUT of seq into sparse "
                          "snp/indel_span/direction columns; per-base annotation "
                          "columns allowed), 'v7' (inline anchor-SNP '^' + "
                          "per-base annotation columns), 'v6' (inline anchor-SNP "
                          "'^', no annotations), 'v6nd' (v6 content with the ‹/› "
                          "direction tokens removed and phase dropped — unordered "
                          "symbols; diploid-only), 'v5' (no compound-het anchor "
                          "'^', the pre-v6 bare-anchor form, no annotations), 'v4' "
                          "(the original pre-direction-marker encoding: ordered/"
                          "phased symbols with inline '^'/'{}' markers but no ‹/› "
                          "direction tokens, no annotations). Stamps the matching "
                          "schema_version into provenance. "
                          "--phylop-bw/--cadd-arrow-dir require v7 or v8. "
                          "The '-ldb' variants (v4-ldb/v6-ldb/v7-ldb) are those "
                          "same encodings over a different REGION UNIVERSE — the "
                          "deCODE EUR LD blocks — and require --bed to be "
                          "preprocess/regions/ld_blocks_eur_grch38.bed (enforced by "
                          "sha256).")
    gen.add_argument("--representation", choices=list(_REPRESENTATIONS),
                     default="diploid",
                     help="Output representation. 'diploid' (default): "
                          "256x256 Unicode codepoints with SNP/INDEL markers "
                          "(current behavior). 'haploid': ALT-wins-leftmost "
                          "ACGTN string with SNP/INDEL markers. Haploid "
                          "destroys phase by design and is incompatible with "
                          "--use-direction-markers / --force-unphased. "
                          "Orthogonal to --version.")
    gen.add_argument("--emit-ref-index", action="store_true",
                     help="Store per-codepoint HG38 offsets (-2 marker, -1 insertion) for marker-aware extraction.")
    gen.add_argument("--output", type=Path, required=True,
                     help="Output directory for Parquet shards")
    _default_workers = max(1, min(60, os.cpu_count() or 1))
    gen.add_argument("--workers", type=int, default=_default_workers,
                     help=f"Number of parallel worker processes (default: {_default_workers}; "
                          f"auto-scaled to min(60, cpu_count))")
    gen.add_argument("--max-samples", type=int, default=None,
                     help="Process at most this many samples (useful for smoke tests)")
    gen.add_argument("--max-regions", type=int, default=None,
                     help="Use at most this many BED regions (useful for smoke tests)")
    gen.add_argument("--genes-file", type=Path, default=None,
                     help="File with gene symbols (one per line). Only BED regions "
                          "whose 4th column matches a listed gene are included. "
                          "Requires a BED file with gene names in column 4.")
    gen.add_argument("--samples-file", type=Path, default=None,
                     help="File with sample names (one per line). If omitted, all "
                          "samples are discovered from BCF headers.")
    gen.add_argument("--test-samples-file", type=Path, default=None,
                     help="File listing test sample names (one per line). "
                          "These samples are written to output/test/ subdirectory; "
                          "remaining go to output/train/.")
    gen.add_argument("--compression", type=str, default="lz4",
                     choices=["lz4", "snappy", "zstd", "none"],
                     help="Parquet compression codec (default: lz4). lz4 is ~5x faster "
                          "than zstd with ~20%% larger files.")
    gen.add_argument("--compression-level", type=int, default=None,
                     help="Compression level (only for zstd, 1-22). Ignored for lz4/snappy.")
    gen.add_argument("--clean-stale", action="store_true",
                     help="Remove abandoned .parquet.tmp files before starting. "
                          "A .tmp is also a live worker's claim on that sample, "
                          "so each one is watched for --stale-observe-seconds and "
                          "only an unchanged one is removed; live ones are kept "
                          "and reported.")
    gen.add_argument("--stale-observe-seconds", type=float, default=_STALE_OBSERVE_S,
                     help=f"How long --clean-stale watches each .parquet.tmp before "
                          f"an unchanged (size, mtime, inode) counts as abandoned "
                          f"(default: {_STALE_OBSERVE_S:g}s = 3 claim heartbeats). "
                          f"Clock-free, so safe on NFS. Pass 0 to remove every .tmp "
                          f"unconditionally — only safe when no other generate is "
                          f"running against this --output.")
    gen.add_argument("--stale-age-hours", type=float, default=None,
                     help="DEPRECATED and ignored (an age threshold cannot be "
                          "evaluated against an NFS mtime); use "
                          "--stale-observe-seconds.")
    gen.add_argument("--force-unphased", "--force-unordered",
                     dest="force_unphased", action="store_true",
                     help="Force unphased diploid symbols even when BCF "
                          "genotypes are phased (legacy alias: --force-unordered, "
                          "emits DeprecationWarning)")
    gen.add_argument("--use-direction-markers", action="store_true",
                     help="DEPRECATED and ignored: the \u2039/\u203a direction-marker "
                          "encoding is now owned by --version (every preset uses "
                          "it except v6nd; haploid forces it off). Kept only so "
                          "existing scripts still parse.")
    gen.add_argument("--profile", action="store_true",
                     help="Enable PerfAccumulator profiling on the first pending sample "
                          "(forces --workers 1; prints per-function timing breakdown)")
    gen.add_argument("--no-dedup-inline", action="store_true", dest="no_dedup_inline",
                     help="Disable hash-based inline deduplication during generation. "
                          "Duplicates are still removed in the assemble step. "
                          "By default, a live cross-worker hash set skips sequences "
                          "already written this run (works on a fresh start).")
    gen.add_argument("--dedup-capacity", type=int, default=None,
                     help="Slots for the live inline-dedup hash table, per split "
                          "(16 bytes/slot). Default: auto-sized from pending work. "
                          "If the table fills, remaining duplicates are deferred to "
                          "the assemble dedup (no error).")
    gen.add_argument("--phylop-bw", type=Path, default=None,
                     help="PhyloP 100-way vertebrate bigWig path. "
                          "Presence enables the phyloP column; absence omits it. "
                          "No default — common value: "
                          "/opt/hpc/data/PhylopP/hg38.phyloP100way.bw")
    gen.add_argument("--cadd-arrow-dir", type=Path, default=None,
                     help="Directory of per-chrom CADD SNV + indel Arrow "
                          "shards produced by poc_a2_convert_raw_cadd.py "
                          "(cadd_snv_<chrom>.arrow and "
                          "cadd_indel_<chrom>.arrow). Presence enables the "
                          "cadd_phred column. No default — common value: "
                          "/opt/hpc/data/CADD/full_arrow/")
    gen.add_argument("--preflight", choices=["full", "fast", "off"],
                     default="full",
                     help="Resolve every lazily-imported dependency at "
                          "startup instead of hours in (default: full). 'full' "
                          "adds one spawn round trip — the only thing in an "
                          "assemble that exercises the interpreter re-exec "
                          "save_to_disk does at the very end; 'fast' skips that "
                          "rung; 'off' skips the check entirely.")
    gen.add_argument("--log-level", default="ERROR",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     help="Logging verbosity (default: ERROR)")

    # ── assemble ────────────────────────────────────────────────────────
    asm = subparsers.add_parser(
        "assemble",
        help="Assemble per-sample Parquet shards into a HuggingFace dataset",
    )
    asm.add_argument("--shard-dir", type=Path, required=True,
                     help="Directory containing per-sample .parquet shards (train split)")
    asm.add_argument("--test-shard-dir", type=Path, default=None,
                     help="Directory containing test split .parquet shards")
    asm.add_argument("--output", type=Path, required=True,
                     help="Output directory for the assembled HuggingFace dataset")
    asm.add_argument("--target-version", choices=sorted(versions.SUPPORTED_VERSIONS),
                     default=None,
                     help="Convert the shards to this version on the fly during "
                          "assembly instead of regenerating. Only 'downhill' "
                          "routes are supported (drop columns / re-inline "
                          "markers / strip ‹›, no source data needed): v7→v6, v8→v6, "
                          "v8→v7, v6→v6nd, and the same within a region set "
                          "(v7-ldb→v6-ldb, v8-tsstes→v7-tsstes). "
                          "Incompatible requests (e.g. v6→v7, which needs "
                          "PhyloP/CADD source) hard-error before any work. "
                          "Default: keep the source version. Use `check-compat` "
                          "to test a route first.")
    asm.add_argument("--max-shard-size", type=str, default="10GB",
                     help="Max shard size for output (e.g. '5GB'). Default: 10GB")
    asm.add_argument("--hf-cache-dir", type=Path, default=None,
                     help="Override HuggingFace datasets cache directory (default: ~/.cache/huggingface/datasets). "
                          "Use this when the default cache partition lacks space.")
    asm.add_argument("--preflight", choices=["full", "fast", "off"],
                     default="full",
                     help="Resolve every lazily-imported dependency at "
                          "startup instead of hours in (default: full). 'full' "
                          "adds one spawn round trip — the only thing in an "
                          "assemble that exercises the interpreter re-exec "
                          "save_to_disk does at the very end; 'fast' skips that "
                          "rung; 'off' skips the check entirely.")
    asm.add_argument("--log-level", default="ERROR",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     help="Logging verbosity (default: ERROR)")

    # ── check-compat ────────────────────────────────────────────────────
    chk = subparsers.add_parser(
        "check-compat",
        help="Test whether existing shards can be assembled into a target "
             "version (read-only; reports the conversion route or why not)",
    )
    chk.add_argument("--shard-dir", type=Path, required=True,
                     help="Directory of per-sample .parquet shards to inspect")
    chk.add_argument("--target-version", required=True,
                     choices=sorted(versions.SUPPORTED_VERSIONS),
                     help="The version you want to assemble into. Exit 0 if "
                          "convertible (incl. identity), 1 if incompatible, 2 if "
                          "the source version cannot be determined, 3 on a "
                          "provenance/columns mismatch.")
    chk.add_argument("--log-level", default="ERROR",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     help="Logging verbosity (default: ERROR)")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # --force-unordered is a deprecated alias for --force-unphased.
    if "--force-unordered" in sys.argv:
        warnings.warn(
            "'--force-unordered' is deprecated; use '--force-unphased' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # --use-direction-markers is deprecated: direction markers are now owned by
    # the --version preset (and forced off for haploid). The flag is ignored.
    if "--use-direction-markers" in sys.argv:
        warnings.warn(
            "'--use-direction-markers' is deprecated and ignored; the "
            "direction-marker encoding is now selected by --version "
            "(every preset uses it except v6nd).",
            DeprecationWarning,
            stacklevel=2,
        )

    # Haploid mode destroys phase by design — reject flag combinations that
    # rely on phase-aware diploid output. (Direction markers are no longer a
    # user flag; the preset forces them off for haploid, so no check needed.)
    if args.command == "generate" and args.representation == "haploid":
        if args.force_unphased:
            parser.error(
                "--force-unphased is incompatible with "
                "--representation=haploid (the haploid path does not use "
                "the diploid 256x256 token table at all)."
            )
        if versions.resolve(args.version).force_unphased:
            parser.error(
                f"--version {args.version} bundles force_unphased (an unordered "
                "DIPLOID encoding) and is incompatible with "
                "--representation=haploid (the haploid path does not use the "
                "diploid token table at all)."
            )
        if getattr(args, "cadd_arrow_dir", None) is not None:
            parser.error(
                "--cadd-arrow-dir is incompatible with "
                "--representation=haploid (variant-aware annotations require "
                "per-haplotype alleles, which haploid output does not preserve)."
            )

    # Per-base annotation columns are a v7+ feature; v5/v6 presets forbid them.
    if args.command == "generate":
        preset = versions.resolve(args.version)
        if not preset.allow_annotations:
            annot_flags = []
            if getattr(args, "phylop_bw", None) is not None:
                annot_flags.append("--phylop-bw")
            if getattr(args, "cadd_arrow_dir", None) is not None:
                annot_flags.append("--cadd-arrow-dir")
            if annot_flags:
                parser.error(
                    f"{' / '.join(annot_flags)} require --version v7 or v8 "
                    f"(per-base annotation columns were introduced in v7); "
                    f"--version {args.version} produces no annotation columns."
                )

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve every lazily-imported dependency now, while failing is cheap —
    # args.command is known so --help stays fast and check-compat pays nothing.
    if args.command in ("generate", "assemble"):
        _preflight_imports(args)

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "assemble":
        cmd_assemble(args)
    elif args.command == "check-compat":
        cmd_check_compat(args)


if __name__ == "__main__":
    main()
