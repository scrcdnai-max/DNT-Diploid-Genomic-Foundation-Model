"""Genomic Sequence Haplotype Library.

Combines reference sequences (hg38 from 2bit or indexed FASTA) with variant data
from BCF/VCF files to produce per-position diploid haplotype data.

Main Class:
    GenomeSequenceProcessor: Thread-safe processor for extracting haplotype
    sequences from reference + VCF/BCF files.

Coordinate Conventions:
    - API inputs (start, end) use 0-based, half-open BED coordinates
    - VCF positions are 1-based (as per VCF spec)
    - Output DataFrame 'pos' column is 1-based
    - 'offset' is 0 for reference positions, 1+ for inserted nucleotides

Diploid Encoding:
    Each position is encoded as a single diploid symbol derived from the
    (hap0, hap1) nucleotide pair. Phased variants use the phased lookup
    table; unphased variants use the unphased table (symmetric with respect
    to haplotype order). The mapping is defined by the diploid tokens TSV file.

Performance:
    The implementation uses numpy arrays for the reference backbone and applies
    variants as sparse updates, making it efficient for chromosome-scale regions
    (hundreds of millions of bases). For 2bit reference files, the entire genome
    is preloaded into memory at initialization (~7 seconds, ~3GB) for instant
    subsequent access.

Thread Safety:
    GenomeSequenceProcessor is safe for concurrent processing of different
    genomic regions across multiple threads.  Batch parallel methods create
    per-thread VCF readers.  The single-region get_sequence_str() serializes
    VCF access via an internal Lock; for true parallelism use
    get_sequence_parallel() instead.

Example Usage:
    # Single region
    with GenomeSequenceProcessor('hg38.2bit', 'diploid_tokens.tsv') as processor:
        df = processor.get_sequence_df('sample.vcf.gz', 'chr1', 1000000, 2000000)
        print(f"Processed {len(df)} positions")

    # Multiple regions (processed in parallel by default)
    with GenomeSequenceProcessor('hg38.2bit', 'diploid_tokens.tsv') as processor:
        regions = [
            ('chr1', 0, 1000000),
            ('chr2', 0, 1000000),
            ('chr3', 0, 1000000),
        ]
        df = processor.get_sequences_df('sample.vcf.gz', regions)
        print(f"Processed {len(df)} positions across {len(regions)} chromosomes")

    # String output instead of DataFrame
    with GenomeSequenceProcessor('hg38.2bit', 'diploid_tokens.tsv') as processor:
        sequence = processor.get_sequence_str('sample.vcf.gz', 'chr1', 0, 1000)
        print(f"Diploid sequence: {sequence[:50]}...")
"""

import logging
import re
import threading
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from cyvcf2 import VCF

from preprocess import paths, tokens
from preprocess.encoder.reference import ReferenceFile
from preprocess.markers import (
    DIR_FWD,
    DIR_REV,
    INDEL_CLOSE,
    INDEL_OPEN,
    SNP_MARKER,
    rehydrate_markers,
)
from preprocess.regions import bedio

__all__ = [
    "GenomeSequenceProcessor",
    "Variant",
    "split_markers",
    "rehydrate_markers",
    "DIR_FWD",
    "DIR_REV",
]

# Pinned to the pre-move name (Tal, reorg step 8-0): callers and tests silence/capture it by this name.
logger = logging.getLogger("preprocess.get_genome_sequence")

# Diploid symbol used when no mapping exists for a haplotype pair.
UNKNOWN_DIPLOID_SYMBOL = "N"

# Canonical allele ordering: A < C < G < T < N < - < *
# Used to determine direction indicators in three-level encoding.
_ALLELE_PRIORITY = np.full(256, 255, dtype=np.uint8)
_ALLELE_PRIORITY[ord("A")] = 0
_ALLELE_PRIORITY[ord("C")] = 1
_ALLELE_PRIORITY[ord("G")] = 2
_ALLELE_PRIORITY[ord("T")] = 3
_ALLELE_PRIORITY[ord("N")] = 4
_ALLELE_PRIORITY[ord("-")] = 5
# NOTE: '*' (spanning deletion) is remapped to '-' at VCF parse time,
# so this entry is never reached at runtime.  Kept for completeness of
# the canonical ordering definition.
_ALLELE_PRIORITY[ord("*")] = 6


def _u1_to_str(a: np.ndarray) -> str:
    """Join a C-contiguous ``<U1`` array into a ``str`` via a zero-copy view.

    A length-``n`` ``<U1`` array stored contiguously is ``n`` consecutive 4-byte
    UCS-4 codepoints — bit-identical to a single ``<Un`` scalar. Reinterpreting
    the buffer materializes the joined string in C, avoiding the per-element
    Python boxing of ``"".join(a)``. Output is byte-for-byte identical.

    Inherits the little-endian assumption already present in ``view("<U1")``
    elsewhere in this module.
    """
    n = a.shape[0]
    if n == 0:
        # .view(f"<U{n}") raises on n == 0 ("size must be a divisor").
        return ""
    return np.ascontiguousarray(a).view(f"<U{n}")[0]


@dataclass(frozen=True)
class Variant:
    """Parsed variant data for a single VCF record."""

    ref: str
    alt: List[str]
    gt: Tuple[int, int]
    phased: bool


@dataclass(frozen=True)
class VariantCall:
    """Per-sample resolved variant call exposed to annotation consumers.

    Companion of :class:`Variant` flattened for downstream readers
    (e.g. :mod:`huggingface_dataset.annotations.CADDAnnotation`):
    each haplotype's actual allele string is materialized, so the
    consumer does not need to know about GT indices or the alt list.

    ``pos`` is 1-based VCF position. ``hap0`` / ``hap1`` are the actual
    allele strings on each haplotype (equal to ``ref`` if that haplotype
    carries no alt). The per-region variant list returned by the encoder
    is ordered by ascending ``pos``.
    """

    pos: int
    ref: str
    hap0: str
    hap1: str
    is_phased: bool


# ---------------------------------------------------------------------------
# Diploid token loading (instance-level, no global cache)
# ---------------------------------------------------------------------------


def _build_diploid_lookup_table(
    lookup: Dict[str, str],
) -> np.ndarray:
    """Build a 256x256 numpy lookup table for fast diploid symbol resolution.

    Maps (ord(hap0_char), ord(hap1_char)) -> diploid symbol ordinal.
    Returns a 2D array of uint32 (Unicode code points).
    """
    table = np.full((256, 256), ord(UNKNOWN_DIPLOID_SYMBOL), dtype=np.uint32)
    for pair, symbol in lookup.items():
        if len(pair) == 2:
            table[ord(pair[0]), ord(pair[1])] = ord(symbol)
    return table


# ---------------------------------------------------------------------------
# Variant parsing (cyvcf2)
# ---------------------------------------------------------------------------


def _build_pos_to_var_idx(
    variants: Dict[int, "Variant"],
    start: int,
    end: int,
) -> Tuple[np.ndarray, List["VariantCall"]]:
    """Build the per-ref-position variant index + the per-region variant list.

    Used by the v7 annotation pipeline (variant-aware annotations like CADD).
    For each ref offset ``i`` in ``[0, end-start)``, ``pos_to_var_idx[i]``
    is the index of the variant covering that ref position in the returned
    ``variant_calls`` list, or ``-1`` if no variant covers it.

    The mapping spans every reference position the variant's REF allele
    covers (anchor + non-anchor for deletions); insertions occupy only the
    anchor ref position. Cross-region clipping: ranges are intersected with
    ``[0, end-start)`` before assignment.

    Overlap policy: variants are iterated in ascending ``pos`` and a later
    variant's range overwrites earlier entries on the overlap (last-writer-
    wins). Adjacent/overlapping variants are uncommon in production VCFs;
    consumers that care can detect overlap by counting unique ``var_idx``
    values inside the region.
    """
    n = end - start
    pos_to_var_idx = np.full(n, -1, dtype=np.int32)
    variant_calls: List["VariantCall"] = []
    for j, pos_1based in enumerate(sorted(variants.keys())):
        v = variants[pos_1based]
        ref_len = len(v.ref)
        anchor_offset = pos_1based - 1 - start
        s = max(0, anchor_offset)
        e = min(n, anchor_offset + ref_len)
        if s < e:
            pos_to_var_idx[s:e] = j
        h0_str = v.ref if v.gt[0] == 0 else v.alt[v.gt[0] - 1]
        h1_str = v.ref if v.gt[1] == 0 else v.alt[v.gt[1] - 1]
        variant_calls.append(
            VariantCall(
                pos=pos_1based,
                ref=v.ref,
                hap0=h0_str,
                hap1=h1_str,
                is_phased=v.phased,
            )
        )
    return pos_to_var_idx, variant_calls


def _validate_gt_index(gt_idx: int, num_alleles: int, context: str) -> int:
    """Validate a GT index is in range; log warning and return 0 if not."""
    if not (0 <= gt_idx < num_alleles):
        logger.warning(
            "GT index %d out of range [0, %d) for %s; falling back to REF (0)",
            gt_idx,
            num_alleles,
            context,
        )
        return 0
    return gt_idx


#: Keys of :attr:`GenomeSequenceProcessor.integrity`, the per-processor
#: ``Counter`` of silent resolutions the encoder makes while reading a
#: sample. Every key is a decision that changes what the strings carry
#: relative to the raw BCF records, so a dataset writer records the totals
#: (``create_haplotype_dataset.py`` puts them in each shard's metadata).
#:
#: * ``same_pos_composed`` — two records at one POS put a 1-bp anchor
#:   substitution and an indel on the same haplotype; merged into one allele.
#: * ``same_pos_conflict_newer_wins`` — two records at one POS put different,
#:   non-composable alleles on the same haplotype; the later record's allele
#:   is kept (e.g. ``G>A 1|1`` and ``G>C 1|1``).
#: * ``gt_index_fallback`` — a GT index beyond the record's ALT list; that
#:   haplotype falls back to REF.
#: * ``unphased_hom`` — a homozygous call written with ``/``; accepted.
#: * ``overlap_trim_nonbenign`` — the overlap resolver dropped a haplotype's
#:   allele whose content the winning variant does not reproduce.
INTEGRITY_COUNTER_KEYS = (
    "same_pos_composed",
    "same_pos_conflict_newer_wins",
    "gt_index_fallback",
    "unphased_hom",
    "overlap_trim_nonbenign",
)


def _compose_anchor_snp_with_indel(a: str, b: str, ref: str) -> Optional[str]:
    """Compose two alleles of one haplotype at one POS, both padded to *ref*.

    When exactly one of them is a 1-bp substitution of the anchor (same length
    as *ref*, differs only at index 0) and the other keeps the reference anchor
    (an insertion or deletion on that anchor), the haplotype carries both: the
    substituted base followed by the indel allele's remaining bases. Returns
    ``None`` when the pair is not of that shape.
    """
    def is_anchor_snp(x: str) -> bool:
        return len(x) == len(ref) and x[0] != ref[0] and x[1:] == ref[1:]

    if is_anchor_snp(a) and b[0] == ref[0]:
        return a[0] + b[1:]
    if is_anchor_snp(b) and a[0] == ref[0]:
        return b[0] + a[1:]
    return None


def _pick_haplotype_allele(
    existing: str,
    new: str,
    ref: str,
    pos: int,
    chrom: str,
    hap: str,
    counters: Optional[Counter],
) -> str:
    """One haplotype's allele from two same-POS records (padded to *ref*).

    The non-reference allele wins over REF. When both records put a
    non-reference allele on the haplotype: identical alleles agree; an anchor
    substitution plus an indel on the same anchor compose
    (:func:`_compose_anchor_snp_with_indel`, counted as ``same_pos_composed``);
    anything else is a contradiction the records cannot both be right about,
    and the newer record wins (``same_pos_conflict_newer_wins``, WARNING).
    """
    if new == ref:
        return existing
    if existing == ref or existing == new:
        return new
    composed = _compose_anchor_snp_with_indel(existing, new, ref)
    if composed is not None:
        if counters is not None:
            counters["same_pos_composed"] += 1
        logger.debug(
            "Composed %s alleles at %s:%d: '%s' + '%s' -> '%s'",
            hap, chrom, pos, existing, new, composed,
        )
        return composed
    if counters is not None:
        counters["same_pos_conflict_newer_wins"] += 1
    logger.warning(
        "Conflicting %s alleles at %s:%d: '%s' vs '%s'; using '%s'",
        hap, chrom, pos, existing, new, new,
    )
    return new


def _merge_same_position_variants(
    existing: Variant,
    new_var: Variant,
    pos: int,
    chrom: str = "",
    counters: Optional[Counter] = None,
) -> Variant:
    """Merge two VCF records at the same position into a single Variant.

    Phased datasets (e.g. 1000 Genomes) commonly split compound heterozygous
    sites into separate biallelic records at the same POS, each describing one
    haplotype's allele.  This function combines them so that both haplotypes
    are represented in a single Variant.

    When the two records have different REF lengths, the shorter record's
    alleles are right-padded with the trailing reference bases so that
    everything is expressed relative to the longer (more complete) REF.
    """
    # Determine the longest REF and the suffix needed to pad the shorter one.
    if len(existing.ref) >= len(new_var.ref):
        merged_ref = existing.ref
        suffix_for_existing = ""
        suffix_for_new = existing.ref[len(new_var.ref) :]
    else:
        merged_ref = new_var.ref
        suffix_for_existing = new_var.ref[len(existing.ref) :]
        suffix_for_new = ""

    # Resolve each record's per-haplotype alleles (padded to merged REF).
    def _padded_alleles(var: Variant, suffix: str) -> Tuple[str, str]:
        alleles = [var.ref + suffix] + [a + suffix for a in var.alt]
        n = len(alleles)
        raw0, raw1 = var.gt[0], var.gt[1]
        idx0 = raw0 if raw0 < n else 0
        idx1 = raw1 if raw1 < n else 0
        if raw0 >= n or raw1 >= n:
            if counters is not None:
                counters["gt_index_fallback"] += 1
            logger.warning(
                "GT index out of range at %s:%d — GT=(%d,%d) but only %d allele(s); "
                "falling back to REF (0) for out-of-range haplotype(s).",
                chrom,
                pos,
                raw0,
                raw1,
                n,
            )
        return alleles[idx0], alleles[idx1]

    ex_hap0, ex_hap1 = _padded_alleles(existing, suffix_for_existing)
    nw_hap0, nw_hap1 = _padded_alleles(new_var, suffix_for_new)

    # One allele per haplotype: non-ref over REF, compose an anchor SNP with
    # an indel on the same anchor, otherwise the newer record wins.
    final_hap0 = _pick_haplotype_allele(ex_hap0, nw_hap0, merged_ref, pos, chrom, "hap0", counters)
    final_hap1 = _pick_haplotype_allele(ex_hap1, nw_hap1, merged_ref, pos, chrom, "hap1", counters)

    # Build merged ALT list and GT from the final per-haplotype alleles.
    merged_alt: List[str] = []
    for allele in (final_hap0, final_hap1):
        if allele != merged_ref and allele not in merged_alt:
            merged_alt.append(allele)

    all_alleles = [merged_ref] + merged_alt
    gt0 = all_alleles.index(final_hap0)
    gt1 = all_alleles.index(final_hap1)

    merged_phased = existing.phased and new_var.phased

    logger.debug(
        "Merged variants at pos %d: (%s>%s GT=%s) + (%s>%s GT=%s) "
        "→ REF=%s ALT=%s GT=(%d,%d)",
        pos,
        existing.ref,
        existing.alt,
        existing.gt,
        new_var.ref,
        new_var.alt,
        new_var.gt,
        merged_ref,
        merged_alt,
        gt0,
        gt1,
    )
    return Variant(
        ref=merged_ref,
        alt=merged_alt,
        gt=(gt0, gt1),
        phased=merged_phased,
    )


# Chromosomes where hemizygous / half-call genotypes are biologically
# expected (sex chromosomes in males).  Used by _process_vcf_record to
# decide whether a half-call should raise or be treated as hemizygous.
_SEX_CHROMS = frozenset({"chrx", "chry", "x", "y"})

# Characters allowed in VCF REF/ALT alleles after * → - mapping.
_VALID_ALLELE_CHARS = frozenset("ACGT-")


def _is_sex_chrom(chrom: str) -> bool:
    """Return True if *chrom* is a sex chromosome (X or Y)."""
    return chrom.lower() in _SEX_CHROMS


def _process_vcf_record(
    record,
    sample_idx: int,
    chrom: str,
    filter_pass_only: bool = True,
) -> Optional[Variant]:
    """Extract a Variant from a single VCF/BCF record, or None to skip it.

    Handles REF/ALT extraction, FILTER checking, structural variant and
    breakend filtering, spanning deletion mapping, allele validation,
    genotype extraction, and hemizygous normalization.

    Returns None for records that should be skipped (empty alleles,
    structural variants, non-PASS filter, hom-ref genotypes, missing
    genotypes).

    Raises:
        ValueError: On autosomal half-calls, missing GT field, or
            non-ACGT allele characters.
    """
    pos = record.POS  # 1-based
    ref_allele = record.REF.upper() if record.REF else ""
    alt_alleles = [a.upper() for a in record.ALT] if record.ALT else []

    if not ref_allele or not alt_alleles:
        return None

    # Skip non-PASS variants when filter_pass_only is enabled.
    # cyvcf2 returns None for FILTER=PASS and FILTER=. (missing).
    if filter_pass_only and record.FILTER is not None:
        return None

    # Skip structural variants (marked with < in ALT) and breakend
    # notation (contains ] or [ brackets, e.g. ]13:123456]T).
    if any("<" in alt or "]" in alt or "[" in alt for alt in alt_alleles):
        logger.debug(
            "Skipping structural/breakend variant at %s:%d", chrom, pos
        )
        return None

    # Map '*' (spanning deletion) allele to '-'
    alt_alleles = ["-" if alt == "*" else alt for alt in alt_alleles]

    # Validate allele characters — only A, C, G, T, - are allowed.
    for allele in [ref_allele] + alt_alleles:
        if allele != "-" and not set(allele) <= _VALID_ALLELE_CHARS:
            invalid_chars = set(allele) - _VALID_ALLELE_CHARS
            raise ValueError(
                f"Non-ACGT character(s) {invalid_chars} in allele '{allele}' "
                f"at {chrom}:{pos}. VCF alleles must contain only A, C, G, T."
            )

    # Extract genotype and phasing
    genotype: Tuple[int, int] = (0, 0)
    is_phased = False
    try:
        gt = record.genotypes[sample_idx]
    except Exception as e:
        raise ValueError(
            f"Cannot read GT field at {chrom}:{pos} (sample_idx={sample_idx}): {e}. "
            f"Ensure the VCF has FORMAT/GT for all records."
        ) from e
    if len(gt) >= 3:
        a1, a2, phased_flag = gt[0], gt[1], gt[2]
        is_phased = bool(phased_flag)
        a1_valid = a1 is not None and a1 >= 0
        a2_valid = a2 is not None and a2 >= 0
        if a1_valid and a2_valid:
            genotype = (a1, a2)
        elif a1_valid:
            # Half-call: a1 present, a2 missing (e.g. 1|. or 1/.)
            if not _is_sex_chrom(chrom):
                raise ValueError(
                    f"Half-call genotype ({a1}/.) on autosome {chrom}:{pos}. "
                    f"Half-calls on autosomes indicate data quality issues."
                )
            # Hemizygous on sex chromosome — mirror to both haplotypes.
            # Future: use dedicated hemizygous ("single") tokens.
            genotype = (a1, a1)
        elif a2_valid:
            # Half-call: a2 present, a1 missing (e.g. ./1 or .|1)
            if not _is_sex_chrom(chrom):
                raise ValueError(
                    f"Half-call genotype (./{a2}) on autosome {chrom}:{pos}. "
                    f"Half-calls on autosomes indicate data quality issues."
                )
            genotype = (a2, a2)
        # else: both missing (./.) — genotype stays (0, 0), handled below
    elif len(gt) >= 2:
        # Single-allele hemizygous genotype (e.g. chrX/Y in males).
        a1, phased_flag = gt[0], gt[1]
        if a1 is not None and a1 >= 0:
            genotype = (a1, a1)
        is_phased = bool(phased_flag)

    # Skip homozygous-ref (caller signals this via return value)
    if genotype == (0, 0):
        return None

    return Variant(
        ref=ref_allele,
        alt=alt_alleles,
        gt=genotype,
        phased=is_phased,
    )


def _collect_variants(
    records,
    sample_idx: int,
    chrom: str,
    context_label: str,
    _perf=None,
    _region_size: int = 0,
    filter_pass_only: bool = True,
) -> Dict[int, Variant]:
    """Iterate VCF records, build and merge Variants into a position dict.

    Args:
        records: Iterable of cyvcf2 record objects.
        sample_idx: Index of the sample to extract.
        chrom: Chromosome name (for logging).
        context_label: Human-readable label for hom-ref warning.
        _perf: Optional PerfAccumulator for sub-step timing.
        _region_size: Region size for timing metadata.
        filter_pass_only: When True, skip non-PASS variants.

    Returns:
        Dict mapping 1-based positions to Variant objects.
    """
    variants: Dict[int, Variant] = {}
    warned_homref = False
    warned_missing_gt = False
    n_records = 0
    n_kept = 0

    for record in records:
        n_records += 1
        var = _process_vcf_record(
            record, sample_idx, chrom, filter_pass_only=filter_pass_only
        )
        if var is None:
            # Distinguish hom-ref skip from missing-genotype skip from
            # other skips.  _process_vcf_record returns None for several
            # reasons; we warn once per category if the record has
            # valid alleles.
            if record.REF and record.ALT:
                try:
                    gt = record.genotypes[sample_idx]
                    if len(gt) >= 3:
                        a1, a2 = gt[0], gt[1]
                        a1_miss = a1 is None or a1 < 0
                        a2_miss = a2 is None or a2 < 0
                        if a1_miss and a2_miss and not warned_missing_gt:
                            logger.warning(
                                "Skipping missing genotype (./.) at "
                                "%s:%d (%s). These will be ignored.",
                                chrom,
                                record.POS,
                                context_label,
                            )
                            warned_missing_gt = True
                        elif (
                            not warned_homref
                            and a1 == 0
                            and a2 == 0
                        ):
                            logger.debug(
                                "Skipping homozygous-ref (0/0 or 0|0) variant at "
                                "%s:%d (%s). These will be ignored.",
                                chrom,
                                record.POS,
                                context_label,
                            )
                            warned_homref = True
                except Exception:
                    logger.debug(
                        "GT access failed during skip-reason check at %s:%d",
                        chrom,
                        record.POS,
                    )
            continue

        n_kept += 1
        pos = record.POS
        if pos in variants:
            variants[pos] = _merge_same_position_variants(
                variants[pos],
                var,
                pos,
                chrom,
            )
        else:
            variants[pos] = var

    if _perf:
        from preprocess.encoder.perf_timer import TimingSample

        with _perf._lock:
            _perf._samples.append(
                TimingSample(
                    function="_record_counts",
                    thread_name="",
                    elapsed_s=0,
                    region_size=n_records,  # total BCF records traversed
                    n_variants=n_kept,  # records kept as variants
                )
            )

    return variants


def _batch_parse_variants_for_chrom(
    indexed_chrom_regions: List[Tuple[int, Tuple[str, int, int]]],
    vcf_reader,
    chrom: str,
    sample_idx: int = 0,
    _perf=None,
    filter_pass_only: bool = True,
    counters: Optional[Counter] = None,
) -> Tuple[Dict[int, Dict[int, "Variant"]], int, int]:
    """Parse variants for all regions on a chromosome with a single BCF query.

    Issues one query spanning from the first region's start to the
    **largest** region end (regions may nest), then assigns records to
    their overlapping regions using a merge-join walk that skips regions
    already ended.  Regions **must** be sorted by start position.

    Args:
        indexed_chrom_regions: List of ``(orig_idx, (chrom, start, end))``
            tuples, sorted by start.
        vcf_reader: Pre-opened cyvcf2.VCF reader (with samples already set
            for multi-sample BCFs).
        chrom: Chromosome name.
        sample_idx: Sample index to extract (0 for single-sample or after
            ``set_samples``).
        _perf: Optional PerfAccumulator for timing.
        filter_pass_only: When True, skip non-PASS variants.
        counters: Optional ``Counter`` receiving the integrity events of
            this batch (see :data:`INTEGRITY_COUNTER_KEYS`).

    Returns:
        Tuple of (all_variants, n_records, n_kept) where all_variants maps
        orig_idx → ``{pos_1based: Variant}`` for each region, n_records is
        the total BCF records scanned, and n_kept is the count after
        filtering.
    """
    n_regions = len(indexed_chrom_regions)
    if n_regions == 0:
        return {}, 0, 0

    # Sorted region boundaries (0-based half-open BED coords). The query must
    # reach the LARGEST end, not the last region's: with nested regions the
    # last region by start can end long before its host, and a window cut
    # at its end silently drops every later record of the host (same
    # 2026-10-05 finding as the r_end check below).
    first_start = indexed_chrom_regions[0][1][1]
    last_end = max(rgn[2] for _orig_idx, rgn in indexed_chrom_regions)

    # Single CSI query spanning all regions on this chromosome.
    region_str = f"{chrom}:{first_start + 1}-{last_end}"
    if _perf:
        _t0 = time.perf_counter()
    records = vcf_reader(region_str)
    if _perf:
        _perf._record_elapsed("pv_csi_seek", _t0, last_end - first_start)

    # Pre-extract region boundaries for fast access in the inner loop.
    # Each entry: (orig_idx, start_0based, end_0based)
    region_bounds = [
        (orig_idx, rgn[1], rgn[2]) for orig_idx, rgn in indexed_chrom_regions
    ]

    # Per-region variant dicts, keyed by orig_idx.
    all_variants: Dict[int, Dict[int, Variant]] = {}

    lo = 0  # Lowest region index that could still overlap a record
    n_records = 0
    n_kept = 0
    warned_homref = False
    warned_missing_gt = False

    if _perf:
        _t1 = time.perf_counter()

    for record in records:
        n_records += 1
        pos = record.POS  # 1-based
        pos0 = pos - 1  # 0-based

        # Advance lo past regions that end at or before this record's position.
        while lo < n_regions and region_bounds[lo][2] <= pos0:
            lo += 1

        if lo >= n_regions:
            break  # No more regions can overlap

        # Determine REF span in 0-based coords: [pos0, pos0 + ref_len).
        ref_allele = record.REF
        ref_len = len(ref_allele) if ref_allele else 1

        # Scan forward from lo to find all overlapping regions.
        # Region [r_start, r_end) overlaps record [pos0, pos0+ref_len)
        # iff r_start < pos0 + ref_len AND r_end > pos0.
        #
        # `lo` only skips a PREFIX of regions that have ended; it cannot skip
        # a region that ended but sits behind a longer one that has not (the
        # per-gene tss-tes BED nests short genes inside long ones, so this is
        # the common case there, not an edge case). The r_end check below is
        # therefore load-bearing: without it a record is assigned to every
        # already-ended region up to the record's position — a nested
        # variant-free gene would receive its host's variants, be written as
        # a plain-reference row instead of being skipped, and report their
        # count as its own. (Bug found on the v10 BioMe smoke, 2026-10-05.)
        has_overlap = False
        for i in range(lo, n_regions):
            r_orig_idx, r_start, r_end = region_bounds[i]
            if r_start >= pos0 + ref_len:
                break  # This and all subsequent regions start after the record
            if r_end <= pos0:
                continue  # Ended before the record (nested behind a longer region)
            has_overlap = True
            break

        # If no region overlaps, skip this record entirely.
        if not has_overlap:
            continue

        # Process the record into a Variant (done once per record).
        var = _process_vcf_record(
            record, sample_idx, chrom, filter_pass_only=filter_pass_only
        )
        if var is None:
            if record.REF and record.ALT:
                try:
                    gt = record.genotypes[sample_idx]
                    if len(gt) >= 3:
                        a1, a2 = gt[0], gt[1]
                        a1_miss = a1 is None or a1 < 0
                        a2_miss = a2 is None or a2 < 0
                        if a1_miss and a2_miss and not warned_missing_gt:
                            logger.warning(
                                "Skipping missing genotype (./.) at "
                                "%s:%d (batch query). These will be ignored.",
                                chrom,
                                pos,
                            )
                            warned_missing_gt = True
                        elif not warned_homref and a1 == 0 and a2 == 0:
                            logger.debug(
                                "Skipping homozygous-ref (0/0 or 0|0) variant at "
                                "%s:%d (batch query). These will be ignored.",
                                chrom,
                                pos,
                            )
                            warned_homref = True
                except Exception:
                    pass
            continue

        n_kept += 1
        if counters is not None and var.gt[0] == var.gt[1] and not var.phased:
            counters["unphased_hom"] += 1

        # Assign variant to every overlapping region.
        for i in range(lo, n_regions):
            r_orig_idx, r_start, r_end = region_bounds[i]
            if r_start >= pos0 + ref_len:
                break
            if r_end <= pos0:
                continue  # see the overlap scan above
            # Ensure the region dict exists.
            region_vars = all_variants.get(r_orig_idx)
            if region_vars is None:
                region_vars = {}
                all_variants[r_orig_idx] = region_vars
            # Insert or merge at this position.
            if pos in region_vars:
                region_vars[pos] = _merge_same_position_variants(
                    region_vars[pos],
                    var,
                    pos,
                    chrom,
                    counters=counters,
                )
            else:
                region_vars[pos] = var

    if _perf:
        _perf._record_elapsed("pv_record_iter", _t1, last_end - first_start, n_kept)
        from preprocess.encoder.perf_timer import TimingSample

        with _perf._lock:
            _perf._samples.append(
                TimingSample(
                    function="_record_counts",
                    thread_name="",
                    elapsed_s=0,
                    region_size=n_records,
                    n_variants=n_kept,
                )
            )

    return all_variants, n_records, n_kept


def _parse_variants_for_region(
    vcf_path: Path,
    chrom: str,
    start: int,
    end: int,
    vcf_reader: Optional[VCF] = None,
    filter_pass_only: bool = True,
) -> Dict[int, Variant]:
    """Parse variants from VCF/BCF for a specific region using cyvcf2.

    Args:
        vcf_path: Path to VCF/BCF file (must be indexed).
        chrom: Chromosome name.
        start: 0-based start position.
        end: 0-based end position (exclusive).
        vcf_reader: Pre-opened ``cyvcf2.VCF`` reader for internal reuse.
            When provided the file at *vcf_path* is **not** reopened.
            For internal use only — obtain via ``_get_or_open_vcf``.
        filter_pass_only: When True, skip non-PASS variants.

    Returns:
        Dict mapping 1-based positions to Variant objects.
    """
    if vcf_reader is not None:
        region = f"{chrom}:{start + 1}-{end}"
        try:
            records = vcf_reader(region)
            variants = _collect_variants(
                records, 0, chrom, "single-sample VCF",
                filter_pass_only=filter_pass_only,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch region {chrom}:{start}-{end} from "
                f"'{vcf_path}'. Ensure the file is indexed (tabix/csi). "
                f"Original error: {e}"
            ) from e
        logger.info(
            "Parsed %d variants in %s:%d-%d",
            len(variants),
            chrom,
            start,
            end,
        )
        return variants

    try:
        vcf = VCF(str(vcf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open VCF/BCF file '{vcf_path}': {e}")

    try:
        # Verify VCF contains exactly one sample
        if len(vcf.samples) == 0:
            raise ValueError(
                f"VCF file '{vcf_path}' has no samples. "
                f"A VCF with exactly one sample is required."
            )
        elif len(vcf.samples) > 1:
            raise ValueError(
                f"VCF file '{vcf_path}' contains {len(vcf.samples)} samples: "
                f"{vcf.samples}. A VCF with exactly one sample is required."
            )

        region = f"{chrom}:{start + 1}-{end}"
        try:
            records = vcf(region)
            variants = _collect_variants(
                records, 0, chrom, "single-sample VCF",
                filter_pass_only=filter_pass_only,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch region {chrom}:{start}-{end} from "
                f"'{vcf_path}'. Ensure the file is indexed (tabix/csi). "
                f"Original error: {e}"
            ) from e
    finally:
        vcf.close()

    logger.info(
        "Parsed %d variants in %s:%d-%d",
        len(variants),
        chrom,
        start,
        end,
    )
    return variants


def _find_bcf_for_chromosome(bcf_dir: Path, chrom: str) -> Path:
    """Find a BCF file in a directory matching the chromosome name.

    Uses word-boundary matching to avoid false positives (e.g. "chr1"
    must not match "chr10.bcf").  The chromosome name must appear as a
    complete token in the filename — bounded by start-of-string, '.', '_',
    '-', or end-of-string on both sides.

    Args:
        bcf_dir: Directory containing per-chromosome BCF files.
        chrom: Chromosome name to search for in filenames.

    Returns:
        Path to the first matching BCF file (sorted alphabetically).

    Raises:
        FileNotFoundError: If directory doesn't exist or no matching file found.
    """
    bcf_dirp = Path(bcf_dir)
    if not bcf_dirp.exists() or not bcf_dirp.is_dir():
        raise FileNotFoundError(f"BCF directory not found: '{bcf_dir}'")

    # Match chrom as a whole token (bounded by start/end or common separators)
    pattern = re.compile(r"(?:^|[._\-])" + re.escape(chrom) + r"(?:[._\-]|$)")

    candidates = [
        p for p in sorted(bcf_dirp.iterdir()) if p.is_file() and pattern.search(p.name)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No BCF file matching '{chrom}' found in directory '{bcf_dir}'"
        )
    return candidates[0]


def _resolve_sample_bcf(
    bcf_dir: Path, chrom: str, sample_name: Optional[str]
) -> Path:
    """Resolve the BCF file for *(chrom, sample_name)* under *bcf_dir*.

    Supports two directory layouts:

    * **Per-sample** (preferred): a single multi-contig ``<sample_name>.bcf``
      holding all chromosomes for one sample (GT-only, hom-ref dropped). When
      present this is used for every chromosome — it avoids decoding the full
      multi-sample genotype block, which is the dominant parse cost on a
      multi-sample panel BCF.
    * **Per-chromosome multi-sample** (legacy): one ``*chrN*`` file shared
      across all samples, resolved via :func:`_find_bcf_for_chromosome`.

    Detection is by file existence: if ``<bcf_dir>/<sample_name>.bcf`` exists
    the per-sample layout is assumed, otherwise the per-chromosome layout.
    """
    if sample_name is not None:
        cand = Path(bcf_dir) / f"{sample_name}.bcf"
        if cand.is_file():
            return cand
    return _find_bcf_for_chromosome(bcf_dir, chrom)


def _parse_variants_for_region_multisample(
    bcf_path: Path,
    sample_name: str,
    chrom: str,
    start: int,
    end: int,
    vcf_reader: Optional[VCF] = None,
    filter_pass_only: bool = True,
) -> Dict[int, Variant]:
    """Parse VCF records for a specific sample from a multi-sample BCF.

    Args:
        bcf_path: Path to multi-sample BCF file (must be indexed).
        sample_name: Sample name to extract.
        chrom: Chromosome name.
        start: 0-based start position.
        end: 0-based end position, exclusive.
        vcf_reader: Pre-opened ``cyvcf2.VCF`` reader already restricted to
            *sample_name* via ``set_samples``.  When provided the file is
            **not** reopened.  For internal use only — obtain via
            ``_get_or_open_vcf``.
        filter_pass_only: When True, skip non-PASS variants.

    Returns:
        Dict mapping 1-based positions to Variant objects.

    Raises:
        RuntimeError: If BCF file cannot be opened.
        ValueError: If sample not found in BCF.
    """
    if vcf_reader is not None:
        region = f"{chrom}:{start + 1}-{end}"
        try:
            records = vcf_reader(region)
            # After set_samples the target sample is always index 0
            variants = _collect_variants(
                records,
                0,
                chrom,
                f"sample '{sample_name}'",
                filter_pass_only=filter_pass_only,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch region {chrom}:{start}-{end} from "
                f"'{bcf_path}'. Ensure the file is indexed (tabix/csi). "
                f"Original error: {e}"
            ) from e
        logger.info(
            "Parsed %d variants for sample %s in %s:%d-%d",
            len(variants),
            sample_name,
            chrom,
            start,
            end,
        )
        return variants

    try:
        vcf = VCF(str(bcf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open BCF file '{bcf_path}': {e}")

    try:
        if sample_name not in vcf.samples:
            raise ValueError(
                f"Sample '{sample_name}' not found in BCF '{bcf_path}'; "
                f"samples={vcf.samples}"
            )
        sample_idx = vcf.samples.index(sample_name)

        # Restrict parsing to the requested sample only — avoids decoding
        # genotype fields for all samples in large multi-sample BCFs.
        vcf.set_samples([sample_name])
        sample_idx = 0  # After set_samples, the target is always index 0

        region = f"{chrom}:{start + 1}-{end}"
        try:
            records = vcf(region)
            variants = _collect_variants(
                records,
                sample_idx,
                chrom,
                f"sample '{sample_name}'",
                filter_pass_only=filter_pass_only,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch region {chrom}:{start}-{end} from "
                f"'{bcf_path}'. Ensure the file is indexed (tabix/csi). "
                f"Original error: {e}"
            ) from e
    finally:
        vcf.close()

    logger.info(
        "Parsed %d variants for sample %s in %s:%d-%d",
        len(variants),
        sample_name,
        chrom,
        start,
        end,
    )
    return variants


def _is_subsumed_variant_benign(
    owner_pos: int,
    owner_ref: str,
    owner_allele: str,
    new_pos: int,
    new_ref: str,
    new_allele: str,
) -> bool:
    """Check whether a subsumed variant's allele is consistent with its owner.

    A subsumed variant (whose REF span fits entirely within the owner's)
    is *benign* when every reference position it modifies is already
    modified identically by the owner.  VCF anchor bases that match the
    reference are not considered modifications.

    Returns False conservatively when the new variant introduces inserted
    bases (allele longer than REF), since those carry information the
    owner may not capture.
    """
    if len(new_allele) > len(new_ref):
        return False

    for k in range(len(new_ref)):
        ref_base = new_ref[k]
        new_base = new_allele[k] if k < len(new_allele) else "-"

        # Position unchanged from reference — benign regardless of owner.
        if new_base.upper() == ref_base.upper():
            continue

        # New variant modifies this position — owner must agree.
        owner_k = (new_pos + k) - owner_pos
        if owner_k < 0 or owner_k >= len(owner_ref):
            return False

        owner_base = owner_allele[owner_k] if owner_k < len(owner_allele) else "-"
        if owner_base.upper() != new_base.upper():
            return False

    return True


def _hap_modified_span(
    pos: int,
    ref: str,
    allele: str,
) -> Optional[Tuple[int, int]]:
    """Return the ``(first, last)`` 1-based REF positions that *allele* changes.

    This is the span a haplotype really occupies, as opposed to
    ``len(ref)`` which is the span of the *record*.  The two differ whenever
    one haplotype's allele is REF-length but only changes the anchor —
    a SNP right-padded by :func:`_merge_same_position_variants` to a
    same-POS deletion's REF, or a native multi-allelic record such as
    ``TGG>T,GGG GT=1|2`` — and for every deletion, whose anchor base is
    unchanged.

    The allele model is :func:`_split_allele`'s (shared with
    :func:`_build_haplotype_arrays`): inserted bases sit on the anchor, so an
    insertion always changes position ``pos``; the tail is compared
    position-wise with ``ref[1:]``; the REF bases a short tail does not cover
    are deleted. Comparison is case-insensitive.

    Returns ``None`` when the allele leaves every REF position unchanged.
    """
    inserted, tail = _split_allele(allele, len(ref))
    changed = []
    if inserted or allele[0].upper() != ref[0].upper():
        changed.append(0)
    changed += [1 + j for j, b in enumerate(tail) if b.upper() != ref[1 + j].upper()]
    changed += range(1 + len(tail), len(ref))  # deleted trailing REF bases
    if not changed:
        return None
    return pos + min(changed), pos + max(changed)


def _split_allele(allele, ref_len: int):
    """Split one haplotype's allele into ``(inserted, tail)`` against a REF of
    ``ref_len`` bases.

    This is the encoder's one allele model, shared by
    :func:`_build_haplotype_arrays` (which writes the arrays) and
    :func:`_hap_modified_span` (which the overlap resolver uses): the anchor is
    ``allele[0]``; an allele longer than REF inserts its next
    ``len(allele) - ref_len`` bases right after the anchor (the left-aligned
    placement VCF normalisation produces); the remaining ``tail`` aligns
    position-wise with ``REF[1:]``, and a tail shorter than ``ref_len - 1``
    deletes the REF bases it does not cover. Works on ``str`` and ``bytes``.
    """
    k = len(allele) - ref_len
    if k > 0:
        return allele[1:1 + k], allele[1 + k:]
    return allele[:0], allele[1:]


def _variant_touches_region(pos: int, var: Variant, start: int, end: int) -> bool:
    """True when *var* changes at least one base inside ``[start, end)``.

    A record is handed to a region whenever its REF span overlaps it, but a
    deletion anchored on the region's last base deletes only bases beyond
    ``end`` and leaves the region's sequence untouched. The v10 counts
    (``n_variants`` / ``n_het``) and the "write a row" decision use this test
    so that a counted variant is always visible in the strings.
    """
    hap0_allele, hap1_allele = _get_alleles_for_variant(var, pos)
    for gt, allele in ((var.gt[0], hap0_allele), (var.gt[1], hap1_allele)):
        if gt == 0:
            continue
        span = _hap_modified_span(pos, var.ref, allele)
        if span is not None and span[0] <= end and span[1] >= start + 1:
            return True
    return False


def _resolve_overlapping_variants(
    variants: Dict[int, Variant],
    chrom: str = "",
    sample_name: str = "",
    counters: Optional[Counter] = None,
) -> Dict[int, Variant]:
    """Resolve overlapping variants in a haplotype-aware manner.

    A variant's span on a haplotype is measured from **that haplotype's own
    allele** (:func:`_hap_modified_span`): the first to the last REF position
    it actually changes.  It is *not* ``[pos, pos + len(ref) - 1]`` — the
    record's REF span — because a REF-length allele that only changes the
    anchor (a SNP padded to a same-POS deletion's REF, or one allele of a
    multi-allelic record) must not shadow later variants on its haplotype,
    and a deletion's unchanged anchor must not collide with the previous
    deletion's last base.  Both shapes occur in real phased WGS at ~0.5–0.7
    per million records (BioMe pilot, 2026-10-06) and silently lost a SNP or a
    whole deletion before this rule.

    Strategy (applied in ascending position order):

    1. **Different haplotypes** – when two overlapping variants modify
       disjoint haplotypes (e.g. one is ``0|1`` and the other ``1|0``),
       both are kept because they can be applied independently to the
       hap0 / hap1 arrays without conflict.

    2. **Same haplotype** – conflicts are resolved **per-haplotype**.
       On each haplotype independently, the variant whose REF span
       extends furthest (highest end position) wins.  The losing
       variant's GT for that haplotype is set to 0 (reference),
       preserving any modification it makes on the other haplotype.
       A variant is only dropped entirely if it loses on *every*
       haplotype it modifies (GT becomes ``(0, 0)``).

    Per-haplotype span tracking ensures that chains of overlapping
    variants are handled correctly.

    *counters*, when given, receives one ``overlap_trim_nonbenign`` per
    haplotype allele dropped whose content the winner does not reproduce
    (every WARNING this function logs; see :data:`INTEGRITY_COUNTER_KEYS`).

    Note:
        ``Variant`` is a frozen dataclass, so modifications produce new
        instances rather than mutating in place.
    """
    if not variants:
        return variants

    # Build context prefix for log messages.
    _ctx_parts = []
    if sample_name:
        _ctx_parts.append(f"[{sample_name}]")
    if chrom:
        _ctx_parts.append(chrom)
    _ctx = " ".join(_ctx_parts)
    _ctx_prefix = f"{_ctx}: " if _ctx else ""

    resolved: Dict[int, Variant] = {}
    sorted_positions = sorted(variants.keys())

    # Track the furthest-reaching kept variant per haplotype.
    hap0_span_end = -1
    hap0_owner_pos = -1
    hap1_span_end = -1
    hap1_owner_pos = -1

    for pos in sorted_positions:
        var = variants[pos]
        hap0_allele, hap1_allele = _get_alleles_for_variant(var, pos)

        # Per-haplotype span: the REF positions this haplotype's allele
        # actually changes.  A haplotype whose allele equals REF (or whose GT
        # is 0) modifies nothing and takes part in no conflict.
        span_h0 = (
            _hap_modified_span(pos, var.ref, hap0_allele) if var.gt[0] != 0 else None
        )
        span_h1 = (
            _hap_modified_span(pos, var.ref, hap1_allele) if var.gt[1] != 0 else None
        )
        modifies_hap0 = span_h0 is not None
        modifies_hap1 = span_h1 is not None
        start_h0, end_h0 = span_h0 if modifies_hap0 else (pos, pos - 1)
        start_h1, end_h1 = span_h1 if modifies_hap1 else (pos, pos - 1)

        # Check per-haplotype overlap with previously kept variants.
        hap0_conflict = modifies_hap0 and start_h0 <= hap0_span_end
        hap1_conflict = modifies_hap1 and start_h1 <= hap1_span_end

        if not hap0_conflict and not hap1_conflict:
            # No same-haplotype overlap — keep this variant as-is.
            resolved[pos] = var
            if modifies_hap0 and end_h0 > hap0_span_end:
                hap0_span_end = end_h0
                hap0_owner_pos = pos
            if modifies_hap1 and end_h1 > hap1_span_end:
                hap1_span_end = end_h1
                hap1_owner_pos = pos
            continue

        # ---- Per-haplotype conflict resolution ----
        # Resolve each haplotype independently: the larger REF span wins
        # on that haplotype.  The loser's GT index for that haplotype is
        # set to 0 (reference) rather than dropping the variant entirely.

        new_gt0 = var.gt[0]
        new_gt1 = var.gt[1]
        hap0_loss_benign = True
        hap1_loss_benign = True

        if hap0_conflict:
            owner = resolved.get(hap0_owner_pos)
            if owner is not None:
                # The tracked end IS the owner's per-haplotype span end.
                owner_span_end = hap0_span_end
                span_end = end_h0
                if owner_span_end >= span_end:
                    # Existing variant wins on hap0 — trim new variant.
                    # Check if the subsumed content is consistent with
                    # the owner (benign redundancy from VCF callers).
                    owner_h0, _ = _get_alleles_for_variant(
                        owner,
                        hap0_owner_pos,
                    )
                    new_h0, _ = _get_alleles_for_variant(var, pos)
                    hap0_loss_benign = _is_subsumed_variant_benign(
                        hap0_owner_pos,
                        owner.ref,
                        owner_h0,
                        pos,
                        var.ref,
                        new_h0,
                    )
                    if not hap0_loss_benign and counters is not None:
                        counters["overlap_trim_nonbenign"] += 1
                    _log = logger.debug if hap0_loss_benign else logger.warning
                    _log(
                        "%sOverlap on hap0: variant at pos %d (REF='%s', span %d) "
                        "trimmed in favor of existing variant at pos %d "
                        "(REF='%s', span %d)%s",
                        _ctx_prefix,
                        pos,
                        var.ref,
                        span_end,
                        hap0_owner_pos,
                        owner.ref,
                        owner_span_end,
                        " [benign: consistent content]" if hap0_loss_benign else "",
                    )
                    new_gt0 = 0
                else:
                    # New variant wins on hap0 — trim existing variant
                    if counters is not None:
                        counters["overlap_trim_nonbenign"] += 1
                    logger.warning(
                        "%sOverlap on hap0: existing variant at pos %d (REF='%s', "
                        "span %d) trimmed in favor of variant at pos %d "
                        "(REF='%s', span %d)",
                        _ctx_prefix,
                        hap0_owner_pos,
                        owner.ref,
                        owner_span_end,
                        pos,
                        var.ref,
                        span_end,
                    )
                    mod_owner = Variant(
                        ref=owner.ref,
                        alt=owner.alt,
                        gt=(0, owner.gt[1]),
                        phased=owner.phased,
                    )
                    if mod_owner.gt == (0, 0):
                        resolved.pop(hap0_owner_pos, None)
                    else:
                        resolved[hap0_owner_pos] = mod_owner

        if hap1_conflict:
            owner = resolved.get(hap1_owner_pos)
            if owner is not None:
                owner_span_end = hap1_span_end
                span_end = end_h1
                if owner_span_end >= span_end:
                    # Existing variant wins on hap1 — trim new variant.
                    _, owner_h1 = _get_alleles_for_variant(
                        owner,
                        hap1_owner_pos,
                    )
                    _, new_h1 = _get_alleles_for_variant(var, pos)
                    hap1_loss_benign = _is_subsumed_variant_benign(
                        hap1_owner_pos,
                        owner.ref,
                        owner_h1,
                        pos,
                        var.ref,
                        new_h1,
                    )
                    if not hap1_loss_benign and counters is not None:
                        counters["overlap_trim_nonbenign"] += 1
                    _log = logger.debug if hap1_loss_benign else logger.warning
                    _log(
                        "%sOverlap on hap1: variant at pos %d (REF='%s', span %d) "
                        "trimmed in favor of existing variant at pos %d "
                        "(REF='%s', span %d)%s",
                        _ctx_prefix,
                        pos,
                        var.ref,
                        span_end,
                        hap1_owner_pos,
                        owner.ref,
                        owner_span_end,
                        " [benign: consistent content]" if hap1_loss_benign else "",
                    )
                    new_gt1 = 0
                else:
                    # New variant wins on hap1 — trim existing variant
                    if counters is not None:
                        counters["overlap_trim_nonbenign"] += 1
                    logger.warning(
                        "%sOverlap on hap1: existing variant at pos %d (REF='%s', "
                        "span %d) trimmed in favor of variant at pos %d "
                        "(REF='%s', span %d)",
                        _ctx_prefix,
                        hap1_owner_pos,
                        owner.ref,
                        owner_span_end,
                        pos,
                        var.ref,
                        span_end,
                    )
                    mod_owner = Variant(
                        ref=owner.ref,
                        alt=owner.alt,
                        gt=(owner.gt[0], 0),
                        phased=owner.phased,
                    )
                    if mod_owner.gt == (0, 0):
                        resolved.pop(hap1_owner_pos, None)
                    else:
                        resolved[hap1_owner_pos] = mod_owner

        # Keep new variant if it still modifies at least one haplotype
        if (new_gt0, new_gt1) != (0, 0):
            modified_var = Variant(
                ref=var.ref,
                alt=var.alt,
                gt=(new_gt0, new_gt1),
                phased=var.phased,
            )
            resolved[pos] = modified_var
            if new_gt0 != 0 and modifies_hap0 and end_h0 > hap0_span_end:
                hap0_span_end = end_h0
                hap0_owner_pos = pos
            if new_gt1 != 0 and modifies_hap1 and end_h1 > hap1_span_end:
                hap1_span_end = end_h1
                hap1_owner_pos = pos
        else:
            _all_benign = hap0_loss_benign and hap1_loss_benign
            _log = logger.debug if _all_benign else logger.warning
            _log(
                "%sDropping variant at pos %d (REF='%s') — %s",
                _ctx_prefix,
                pos,
                var.ref,
                "benign subsumption by existing variants"
                if _all_benign
                else "conflicts on both haplotypes with existing variants",
            )

    return resolved


# ---------------------------------------------------------------------------
# Allele resolution helpers
# ---------------------------------------------------------------------------


def _get_alleles_for_variant(var: Variant, pos: int = 0) -> Tuple[str, str]:
    """Get the allele strings for each haplotype from a variant.

    Returns:
        Tuple of (hap0_allele, hap1_allele).
    """
    alleles = [var.ref] + list(var.alt)
    hap0_idx = _validate_gt_index(var.gt[0], len(alleles), f"hap0 at pos {pos}")
    hap1_idx = _validate_gt_index(var.gt[1], len(alleles), f"hap1 at pos {pos}")
    return alleles[hap0_idx], alleles[hap1_idx]


def _validate_variants_against_reference(
    variants: Dict[int, Variant],
    ref_seq,
    chrom: str,
    start: int,
) -> None:
    """Validate that VCF REF alleles match the actual reference sequence.

    Args:
        variants: Dict mapping 1-based positions to Variants.
        ref_seq: Reference sequence as a string or numpy uint8 array.
        chrom: Chromosome name (for error messages).
        start: 0-based start of the region.

    Raises:
        ValueError: If a REF allele doesn't match the reference sequence.
    """
    if not variants:
        return

    # Convert to numpy uint8 if needed for vectorized comparison.
    if isinstance(ref_seq, np.ndarray):
        ref_arr = ref_seq
    else:
        ref_arr = np.frombuffer(ref_seq.encode("ascii"), dtype=np.uint8)

    n = len(ref_arr)

    # Separate SNPs (ref_len == 1) from multi-base variants for
    # vectorized vs scalar validation.
    snp_indices = []
    snp_ref_bytes = []
    multibase_variants = []

    for pos_1based, var in variants.items():
        ref_len = len(var.ref)
        idx = pos_1based - start - 1
        # Clip to the region: a record straddling either edge is checked on
        # the part that lies inside it.
        lo, hi = max(idx, 0), min(idx + ref_len, n)
        if hi <= lo:
            continue
        if ref_len == 1:
            snp_indices.append(idx)
            snp_ref_bytes.append(ord(var.ref[0]) & 0xDF)  # uppercase mask
        else:
            multibase_variants.append((pos_1based, var, lo, hi, lo - idx))

    # Vectorized SNP validation
    if snp_indices:
        idx_arr = np.array(snp_indices, dtype=np.intp)
        expected = np.array(snp_ref_bytes, dtype=np.uint8)
        actual = ref_arr[idx_arr] & np.uint8(0xDF)  # uppercase mask
        mismatches = actual != expected
        if mismatches.any():
            first = int(np.argmax(mismatches))
            pos_1based = snp_indices[first] + start + 1
            vcf_ref = chr(snp_ref_bytes[first])
            actual_ref = chr(ref_arr[snp_indices[first]])
            raise ValueError(
                f"REF mismatch at {chrom}:{pos_1based}: "
                f"VCF says REF='{vcf_ref}' but reference has '{actual_ref}'. "
                f"Check that VCF and reference genome are compatible."
            )

    # Scalar validation for multi-base variants
    for pos_1based, var, lo, hi, clip in multibase_variants:
        actual_bytes = ref_arr[lo:hi]
        expected_bytes = np.frombuffer(
            var.ref[clip : clip + hi - lo].upper().encode("ascii"), dtype=np.uint8
        )
        if not np.array_equal(
            actual_bytes & np.uint8(0xDF), expected_bytes & np.uint8(0xDF)
        ):
            actual_ref = actual_bytes.tobytes().decode("ascii")
            raise ValueError(
                f"REF mismatch at {chrom}:{pos_1based}: "
                f"VCF says REF='{var.ref}' but reference has '{actual_ref}'. "
                f"Check that VCF and reference genome are compatible."
            )


# ---------------------------------------------------------------------------
# Vectorized core
# ---------------------------------------------------------------------------


def _build_haplotype_arrays(
    ref_seq: str,
    variants: Dict[int, Variant],
    start: int,
    emit_anchor_snp_marker: bool = True,
    per_hap_masks_out: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, List, np.ndarray, np.ndarray
]:
    """Build hap0/hap1 byte arrays and collect insertion rows.

    Args:
        ref_seq: Reference sequence string (or numpy uint8 array).
        variants: Dict mapping 1-based positions to Variants.
        start: 0-based start of the region.
        emit_anchor_snp_marker: When True (v6/v7), compound-het INDEL anchors
            are flagged as SNPs so the marker builder emits ``^`` before the
            anchor. When False (v5), that flag is suppressed, reproducing the
            pre-v6 bare-anchor form. This is the only byte difference between
            v5 and v6 encoding. Pure het SNPs (is_snp set elsewhere) always
            get ``^`` regardless of this flag.
        per_hap_masks_out: Optional dict that, when given, is filled in the
            same pass with four ``bool`` arrays of length ``n`` under the keys
            ``is_snp_h0``, ``is_snp_h1``, ``is_indel_h0``, ``is_indel_h1``.
            These are the *per-haplotype* marker masks used by the v10
            haplotype-pair output (:meth:`GenomeSequenceProcessor.get_haplotype_pair_parallel_str`):
            a cell is flagged on a haplotype only when that haplotype carries
            non-reference content there, so each haplotype string is
            independently interpretable. ``is_snp_hX[i]`` is set for a 1-bp
            substitution on haplotype X at cell ``i`` (SNP fast path, or a
            complex record whose allele on X has REF length, differs only at
            the anchor, and is not clipped). Every other modification on X
            flags ``is_indel_hX`` over the REF span *after* the anchor, or
            *including* the anchor cell when the record was clipped at the
            region start — that is what renders a straddling deletion as a
            left-truncated ``{}`` / ``{C}`` instead of vanishing. Unlike the
            shared ``is_snp``/``is_indel_non_anchor`` outputs, this block is
            not gated by ``emit_anchor_snp_marker`` and compares alleles
            against the *reference*, not against ``hap0``. The pre-existing
            return values are unaffected by this kwarg.

    Returns:
        (hap0, hap1, source_is_vcf, phased_mask, insertion_rows,
        is_snp, is_indel_non_anchor)
        where hap0/hap1 are uint8 numpy arrays of ASCII bytes,
        source_is_vcf is a boolean mask, phased_mask is boolean,
        insertion_rows is a list of tuples for inserted positions,
        is_snp marks SNP positions, and is_indel_non_anchor marks
        non-anchor positions of complex variants (INDELs/MNVs).
    """
    if isinstance(ref_seq, np.ndarray):
        n = len(ref_seq)
        ref_arr = ref_seq
        hap0 = ref_seq.copy()
        hap1 = ref_seq.copy()
    else:
        n = len(ref_seq)
        ref_bytes = ref_seq.encode("ascii")
        ref_arr = np.frombuffer(ref_bytes, dtype=np.uint8)
        hap0 = ref_arr.copy()
        hap1 = hap0.copy()

    source_is_vcf = np.zeros(n, dtype=bool)
    phased_mask = np.zeros(n, dtype=bool)
    is_snp = np.zeros(n, dtype=bool)
    is_indel_non_anchor = np.zeros(n, dtype=bool)

    # Per-haplotype marker masks (v10 haplotype-pair output). Allocated only
    # when the caller asks; the shared masks above are unchanged either way.
    want_per_hap = per_hap_masks_out is not None
    if want_per_hap:
        is_snp_h = (np.zeros(n, dtype=bool), np.zeros(n, dtype=bool))
        is_indel_h = (np.zeros(n, dtype=bool), np.zeros(n, dtype=bool))
        per_hap_masks_out["is_snp_h0"] = is_snp_h[0]
        per_hap_masks_out["is_snp_h1"] = is_snp_h[1]
        per_hap_masks_out["is_indel_h0"] = is_indel_h[0]
        per_hap_masks_out["is_indel_h1"] = is_indel_h[1]

    if not variants:
        return hap0, hap1, source_is_vcf, phased_mask, [], is_snp, is_indel_non_anchor

    # --- Pre-resolve alleles and classify into SNPs vs complex ---
    # SNP: ref_len == 1, both alleles are single characters, no
    # insertions or deletions.  These can be applied via vectorized
    # numpy fancy indexing instead of a Python per-variant loop.

    snp_indices = []
    snp_h0_bytes = []
    snp_h1_bytes = []
    snp_phased = []
    snp_mod_h0 = []
    snp_mod_h1 = []

    # Complex variants: pre-resolved as (pos, h0_bytes, h1_bytes,
    # ref_len, mod_h0, mod_h1, phased, clipped) tuples with alleles encoded
    # as bytes for fast indexing (bytes[i] returns int directly,
    # avoiding per-character ord() calls). ``clipped`` is True when the
    # record's anchor lay before the region start and the alleles were
    # left-truncated to the overlapping portion.
    complex_variants: List[
        Tuple[int, bytes, bytes, int, bool, bool, bool, bool]
    ] = []

    for pos, var in variants.items():
        hap0_allele, hap1_allele = _get_alleles_for_variant(var, pos=pos)
        ref_len = len(var.ref)
        idx = pos - start - 1

        if idx >= n:
            continue

        mod_h0 = var.gt[0] != 0
        mod_h1 = var.gt[1] != 0

        if idx < 0:
            # Variant anchor is before region start.  If the REF span
            # (or allele) extends into the region, clip so that only the
            # overlapping portion is applied.  This handles deletions and
            # complex variants that straddle a BED region boundary.
            if idx + ref_len <= 0:
                continue  # Variant entirely before region — skip

            clip = -idx  # Number of leading allele/REF bases to drop
            hap0_allele = hap0_allele[clip:]
            hap1_allele = hap1_allele[clip:]
            ref_len = ref_len - clip
            idx = 0
            # Clipped variant is always complex (partial overlap).
            complex_variants.append(
                (
                    start + 1,  # adjusted 1-based pos for array idx 0
                    hap0_allele.encode("ascii"),
                    hap1_allele.encode("ascii"),
                    ref_len,
                    mod_h0,
                    mod_h1,
                    var.phased,
                    True,  # clipped
                )
            )
            continue

        if ref_len == 1 and len(hap0_allele) == 1 and len(hap1_allele) == 1:
            # SNP fast path
            snp_indices.append(idx)
            snp_h0_bytes.append(ord(hap0_allele))
            snp_h1_bytes.append(ord(hap1_allele))
            snp_phased.append(var.phased)
            snp_mod_h0.append(mod_h0)
            snp_mod_h1.append(mod_h1)
        else:
            # Complex variant — encode alleles as bytes for fast
            # inner-loop indexing.
            complex_variants.append(
                (
                    pos,
                    hap0_allele.encode("ascii"),
                    hap1_allele.encode("ascii"),
                    ref_len,
                    mod_h0,
                    mod_h1,
                    var.phased,
                    False,  # clipped
                )
            )

    # --- Vectorized SNP application ---
    if snp_indices:
        idx_arr = np.array(snp_indices, dtype=np.intp)
        h0_arr = np.array(snp_h0_bytes, dtype=np.uint8)
        h1_arr = np.array(snp_h1_bytes, dtype=np.uint8)
        mod_h0_mask = np.array(snp_mod_h0, dtype=bool)
        mod_h1_mask = np.array(snp_mod_h1, dtype=bool)

        # Apply haplotype bytes only where the variant modifies that haplotype
        hap0[idx_arr[mod_h0_mask]] = h0_arr[mod_h0_mask]
        hap1[idx_arr[mod_h1_mask]] = h1_arr[mod_h1_mask]

        source_is_vcf[idx_arr] = True
        phased_mask[idx_arr] = np.array(snp_phased, dtype=bool)
        is_snp[idx_arr] = True

        if want_per_hap:
            # Per-haplotype: `^` only on the haplotype that carries the ALT.
            is_snp_h[0][idx_arr[mod_h0_mask]] = True
            is_snp_h[1][idx_arr[mod_h1_mask]] = True

    # --- Complex variant loop (indels, MNVs) using pre-encoded bytes ---
    hap0_del = np.zeros(n, dtype=bool)
    hap1_del = np.zeros(n, dtype=bool)

    insertion_rows: List[Tuple[int, int, int, int, bool]] = []
    dash = ord("-")

    for (
        pos,
        h0_allele,
        h1_allele,
        ref_len,
        mod_h0,
        mod_h1,
        is_phased,
        clipped,
    ) in complex_variants:
        idx = pos - start - 1

        # Mark entire REF span as VCF-sourced with correct phasing.
        span_end = min(idx + ref_len, n)
        source_is_vcf[idx:span_end] = True
        phased_mask[idx:span_end] = is_phased
        # Non-anchor positions within the REF span are INDEL content.
        if idx + 1 < span_end:
            is_indel_non_anchor[idx + 1 : span_end] = True

        # One allele model for both haplotypes (see _split_allele): `ins` is
        # inserted right after the anchor, `tail` aligns with the REF cells
        # from `tail_from`, and a short tail deletes the REF bases it does not
        # cover. A clipped record lost its anchor to the region start, so its
        # whole clipped allele is tail, aligned from idx 0.
        if clipped:
            ins0, tail0, ins1, tail1 = h0_allele[:0], h0_allele, h1_allele[:0], h1_allele
            tail_from = idx
        else:
            ins0, tail0 = _split_allele(h0_allele, ref_len)
            ins1, tail1 = _split_allele(h1_allele, ref_len)
            tail_from = idx + 1

        # Per-haplotype marker classification (v10). Unconditional — NOT
        # gated by emit_anchor_snp_marker — and compared against the
        # reference (`ref_arr`), not `hap0`. On haplotype X: `^` on the
        # anchor when the allele only substitutes it (no insertion, tail equal
        # to the reference inside the region, not clipped). Otherwise every
        # cell after the anchor that the tail changes or deletes is INDEL
        # content, plus the anchor cell itself when the record was clipped at
        # the region start (so a straddling deletion renders as a
        # left-truncated `{}` / `{C}` instead of vanishing). Inserted bases
        # are marked through ins_by_idx, which opens a span at the anchor.
        if want_per_hap:
            _ref_tail = ref_arr[tail_from:span_end].tobytes()
            for _h, _allele, _ins, _tail, _mod in (
                (0, h0_allele, ins0, tail0, mod_h0),
                (1, h1_allele, ins1, tail1, mod_h1),
            ):
                if not _mod:
                    continue
                if (
                    not clipped
                    and not _ins
                    and len(_tail) == ref_len - 1
                    and _allele[0] != int(ref_arr[idx])
                    and _tail[: len(_ref_tail)] == _ref_tail
                ):
                    is_snp_h[_h][idx] = True
                    continue
                if clipped:
                    is_indel_h[_h][idx] = True  # left-truncated block marker
                for _j, _ref_b in enumerate(_ref_tail):
                    if _j >= len(_tail) or _tail[_j] != _ref_b:
                        is_indel_h[_h][tail_from + _j] = True

        # If one haplotype carries a pure anchor substitution (allele length ==
        # ref_len, only position 0 differs from ref, rest matches ref) while the
        # OTHER haplotype has a different length (INDEL), mark the anchor as a
        # SNP so _build_marked_string emits `^`.
        #
        # This handles compound-het cases where:
        #   • SNP (ref_len=1) + insertion (ref_len=1): the merged variant has
        #     ref_len=1 and the SNP haplotype's allele is exactly 1 char.
        #   • SNP (ref_len=1) + deletion (ref_len>1): the merger pads the SNP
        #     allele with ref bases; the padded allele still has only the anchor
        #     char differing from ref.
        #
        # MNV anchors are EXCLUDED because their other haplotype (ref allele)
        # has the same length as the MNV allele, so `len(other) != ref_len` is
        # False and the condition does not fire.
        #
        # v5 mode (emit_anchor_snp_marker=False) skips this entirely: the anchor
        # stays unflagged and the marker builder emits the bare pre-v6 form.
        if emit_anchor_snp_marker:
            for _allele, _mod, _other in (
                (h0_allele, mod_h0, h1_allele),
                (h1_allele, mod_h1, h0_allele),
            ):
                if (
                    _mod
                    and _allele
                    and len(_allele) == ref_len           # substitution (no ins/del on this hap)
                    and _allele[0] != int(hap0[idx])      # anchor differs from ref
                    and len(_other) != ref_len            # other hap has an INDEL
                    and all(                               # rest of allele matches ref within region
                        _allele[k] == int(hap0[idx + k])
                        for k in range(1, min(ref_len, n - idx))
                    )
                ):
                    is_snp[idx] = True
                    break

        # Apply the anchor (a clipped record has none inside the region).
        if not clipped:
            if mod_h0 and h0_allele:
                hap0[idx] = h0_allele[0]
            if mod_h1 and h1_allele:
                hap1[idx] = h1_allele[0]

        # Apply each tail position-wise — only where it really differs from
        # the reference. Bases equal to REF are padding (a SNP right-padded
        # to a same-POS deletion's REF, or the unchanged tail of a
        # multi-allelic record's allele) and must not overwrite a SNP already
        # applied there by the fast path above. A short tail deletes the
        # rest of the REF span.
        for _hap, _del, _tail, _mod in (
            (hap0, hap0_del, tail0, mod_h0),
            (hap1, hap1_del, tail1, mod_h1),
        ):
            if not _mod:
                continue
            for _j, _b in enumerate(_tail):
                _cell = tail_from + _j
                if _cell < n and _b != ref_arr[_cell]:
                    _hap[_cell] = _b
            _del[tail_from + len(_tail) : span_end] = True  # empty when nothing is deleted

        # Collect insertions: the bases each haplotype inserts right after the
        # anchor, paired by inserted-base index (offset 1..k). '-' marks the
        # haplotype that inserts less (or nothing) at this record.
        for _off in range(max(len(ins0), len(ins1))):
            h0 = ins0[_off] if _off < len(ins0) else dash
            h1 = ins1[_off] if _off < len(ins1) else dash
            insertion_rows.append((pos, _off + 1, h0, h1, is_phased))

    # Apply deletion masks
    hap0[hap0_del] = dash
    hap1[hap1_del] = dash

    return (
        hap0,
        hap1,
        source_is_vcf,
        phased_mask,
        insertion_rows,
        is_snp,
        is_indel_non_anchor,
    )


def _vectorized_diploid_lookup(
    hap0: np.ndarray,
    hap1: np.ndarray,
    phased_mask: np.ndarray,
    phased_table: np.ndarray,
    unphased_table: np.ndarray,
) -> np.ndarray:
    """Vectorized diploid symbol lookup using precomputed tables.

    Args:
        hap0: uint8 array of hap0 ASCII bytes.
        hap1: uint8 array of hap1 ASCII bytes.
        phased_mask: boolean array (True = use phased lookup).
        phased_table: 256x256 uint32 lookup table (phased symbols).
        unphased_table: 256x256 uint32 lookup table (unphased symbols).

    Returns:
        Array of uint32 Unicode code points for diploid symbols.
    """
    # Lookup in both tables
    phased_result = phased_table[hap0, hap1]
    unphased_result = unphased_table[hap0, hap1]

    # Select based on phasing
    return np.where(phased_mask, phased_result, unphased_result)


def _compute_direction_array(
    hap0: np.ndarray,
    hap1: np.ndarray,
    phased_mask: np.ndarray,
) -> np.ndarray:
    """Compute per-position direction indicators for phased heterozygotes.

    Uses canonical allele ordering (A < C < G < T < N < - < *) to determine
    whether hap0 holds the lower-priority (forward) or higher-priority
    (reverse) allele.

    Args:
        hap0: uint8 array of hap0 ASCII bytes.
        hap1: uint8 array of hap1 ASCII bytes.
        phased_mask: boolean array (True = phased).

    Returns:
        uint8 array where 0 = no direction (hom/unphased),
        1 = forward (‹, hap0 is canonical-first),
        2 = reverse (›, hap0 is canonical-second).
    """
    n = len(hap0)
    direction = np.zeros(n, dtype=np.uint8)

    # Only phased het positions get direction markers.
    needs_direction = (hap0 != hap1) & phased_mask
    if not needs_direction.any():
        return direction

    h0_pri = _ALLELE_PRIORITY[hap0[needs_direction]]
    h1_pri = _ALLELE_PRIORITY[hap1[needs_direction]]
    direction[needs_direction] = np.where(
        h0_pri <= h1_pri, np.uint8(1), np.uint8(2)
    )
    return direction


def _compute_insertion_direction(
    h0_byte: int, h1_byte: int, is_phased: bool
) -> int:
    """Compute direction for a single insertion position.

    Returns 0 (no direction), 1 (forward ‹), or 2 (reverse ›).
    """
    if h0_byte == h1_byte or not is_phased:
        return 0
    return 1 if _ALLELE_PRIORITY[h0_byte] <= _ALLELE_PRIORITY[h1_byte] else 2


# ---------------------------------------------------------------------------
# v8 marker externalization (inline markers <-> sparse columns). The inverse,
# rehydrate_markers, lives in preprocess.markers (re-exported here).
# ---------------------------------------------------------------------------


def split_markers(
    marked_seq: str,
    seq_ref_idx: np.ndarray,
    *,
    snp_marker: str = SNP_MARKER,
    indel_open: str = INDEL_OPEN,
    indel_close: str = INDEL_CLOSE,
    dir_fwd: str = DIR_FWD,
    dir_rev: str = DIR_REV,
) -> Tuple[str, np.ndarray, List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split an inline-marked sequence into pure base content + sparse columns.

    This is the encoder side of the v8 representation. Given a marked sequence
    string and its parallel ``seq_ref_idx`` (``-2`` flags a marker codepoint,
    ``-1`` an inserted base, ``>=0`` a reference offset), it returns the
    marker-free sequence plus three sparse structures from which
    :func:`rehydrate_markers` can reconstruct ``marked_seq`` byte-for-byte:

      * ``clean_seq``    — ``marked_seq`` with all marker codepoints removed.
      * ``clean_ref_idx``— ``seq_ref_idx`` with the ``-2`` entries dropped;
                           parallel to ``clean_seq`` (values ``>=0`` ref offset,
                           ``-1`` inserted). Use the same mask
                           (``seq_ref_idx != -2``) to align any other per-codepoint
                           array (e.g. ``seq_var_idx``).
      * ``snp_pos``      — sorted ``clean_seq`` indices flagged with ``^``.
      * ``indel_spans``  — half-open ``(start, end)`` over ``clean_seq`` indices;
                           ``{`` opens before ``clean_seq[start]`` and ``}`` closes
                           after ``clean_seq[end-1]`` (``end`` may equal ``len``).
                           Region-edge truncation (the haploid builder can suppress
                           a ``{`` or ``}`` on a block clipped by the region bound)
                           is preserved with ``-1`` sentinels: ``start == -1`` is a
                           close-only block (``}`` with no ``{``); ``end == -1`` is an
                           open-only block (``{`` with no ``}``).
      * ``dir_events``   — ``(pos, dir)`` with ``dir`` 1=fwd ``‹``, 2=rev ``›``.

    Inline markers always immediately precede the base codepoint they annotate
    (``^``/``‹``/``›``/``{``) or follow the last in-span codepoint (``}``), so one
    forward pass recovers the columns exactly. ``seq_ref_idx[k] == -2`` is the
    authoritative marker test (set by every builder for every marker codepoint);
    the glyph only disambiguates which marker it is.
    """
    clean_chars: List[str] = []
    clean_ref: List[int] = []
    snp_pos: List[int] = []
    indel_spans: List[Tuple[int, int]] = []
    dir_events: List[Tuple[int, int]] = []
    pending_open: Optional[int] = None
    pending_snp = False
    pending_dir = 0

    ref_list = (
        seq_ref_idx.tolist() if isinstance(seq_ref_idx, np.ndarray) else list(seq_ref_idx)
    )
    if len(ref_list) != len(marked_seq):
        raise ValueError(
            f"split_markers: seq_ref_idx length {len(ref_list)} != marked_seq "
            f"length {len(marked_seq)} — they must be parallel."
        )

    for ch, ri in zip(marked_seq, ref_list):
        if ri == -2:  # marker codepoint
            if ch == snp_marker:
                pending_snp = True
            elif ch == dir_fwd:
                pending_dir = 1
            elif ch == dir_rev:
                pending_dir = 2
            elif ch == indel_open:
                pending_open = len(clean_chars)
            elif ch == indel_close:
                # pending_open is None => left-truncated block (the haploid
                # builder suppressed the '{' at the region start): record a
                # close-only span with a -1 start sentinel.
                start = pending_open if pending_open is not None else -1
                indel_spans.append((start, len(clean_chars)))
                pending_open = None
            else:
                raise ValueError(
                    f"split_markers: codepoint {ch!r} has seq_ref_idx == -2 but "
                    f"is not a known marker glyph."
                )
            continue
        # Base codepoint (ref or inserted): emit, attaching any pending markers.
        k = len(clean_chars)
        if pending_snp:
            snp_pos.append(k)
            pending_snp = False
        if pending_dir:
            dir_events.append((k, pending_dir))
            pending_dir = 0
        clean_chars.append(ch)
        clean_ref.append(ri)

    if pending_open is not None:
        # Right-truncated block (the haploid builder suppressed the closing
        # '}' at the region end): record an open-only span with a -1 end.
        indel_spans.append((pending_open, -1))
    if pending_snp or pending_dir:
        raise ValueError(
            "split_markers: trailing ^/direction marker with no base codepoint "
            "to attach to (corrupt marked sequence)."
        )

    return (
        "".join(clean_chars),
        np.asarray(clean_ref, dtype=np.int32),
        snp_pos,
        indel_spans,
        dir_events,
    )


# ---------------------------------------------------------------------------
# Haploid collapse (ALT-wins-leftmost rule)
# ---------------------------------------------------------------------------

# Allowed alphabet for the haploid output string. Lowercase variants are
# accepted because the diploid token table emits lowercase letters for some
# unphased symbols; in haploid we mostly emit uppercase ACGTN, but guard
# permissively.
_HAPLOID_ALLOWED = frozenset("ACGTNacgtn^{}")


def _collapse_haploid(
    hap0: np.ndarray, hap1: np.ndarray, ref_arr: np.ndarray
) -> np.ndarray:
    """ALT-wins-leftmost per-cell collapse for haploid output.

    At each cell: prefer hap0 if it differs from ref; else hap1 if it differs
    from ref; else ref. Implements "ALT wins, leftmost (= hap0) breaks
    multi-allelic ties" — a per-cell rule that does not preserve phase and
    produces a per-cell chimera at compound-het sites.

    The dataframe view (returned uint8 array) may contain `-` at deletion
    content cells; the string emitter is responsible for dropping them.
    """
    return np.where(hap0 != ref_arr, hap0, np.where(hap1 != ref_arr, hap1, ref_arr))


def _haploid_ins_by_idx(
    insertion_rows: List[Tuple[int, int, int, int, bool]], start: int
) -> Dict[int, List[Tuple[int, str]]]:
    """Reduce per-haplotype insertion rows to a single byte per offset.

    For each insertion row, pick ``h0_byte`` if it is not '-', else
    ``h1_byte``. Raises RuntimeError if both are '-' (defensive — no
    insertion row should reach here without at least one non-gap byte).

    Returns a dict keyed by 0-based array index of the anchor, value is a
    list of (offset, single-character str) tuples — same shape as the
    diploid ins_by_idx, just with a single character instead of a diploid
    codepoint, and no direction-marker third element.
    """
    dash = ord("-")
    out: Dict[int, List[Tuple[int, str]]] = {}
    for pos_1based, offset, h0_byte, h1_byte, _is_phased in insertion_rows:
        if h0_byte != dash:
            chosen = h0_byte
        elif h1_byte != dash:
            chosen = h1_byte
        else:
            raise RuntimeError(
                f"Insertion row at pos={pos_1based} offset={offset} has both "
                f"haplotype bytes '-': no insertion content to emit. This "
                f"indicates a bug in _build_haplotype_arrays."
            )
        arr_idx = pos_1based - start - 1
        out.setdefault(arr_idx, []).append((offset, chr(chosen)))
    for syms in out.values():
        syms.sort()
    return out


def _per_hap_ins_by_idx(
    insertion_rows: List[Tuple[int, int, int, int, bool]], start: int
) -> Tuple[Dict[int, List[Tuple[int, str]]], Dict[int, List[Tuple[int, str]]]]:
    """Split insertion rows into one ``ins_by_idx`` dict per haplotype.

    Sibling of :func:`_haploid_ins_by_idx` for the v10 haplotype-pair
    output. Each returned dict has the same shape (anchor array index →
    sorted list of ``(offset, single-char str)``), but holds only the bases
    that *that* haplotype actually inserts.

    A key is created for a haplotype **only when its byte is not** ``'-'``.
    This is load-bearing: the marker builders open ``{`` whenever the anchor
    index is *present* in ``ins_by_idx``, regardless of the list content, so
    an empty list for the non-carrier of a one-haplotype insertion would put
    ``{}`` on the haplotype that carries nothing.

    Raises RuntimeError if a row has ``'-'`` on both haplotypes (no content
    to emit — a bug upstream in :func:`_build_haplotype_arrays`).
    """
    dash = ord("-")
    ins_h0: Dict[int, List[Tuple[int, str]]] = {}
    ins_h1: Dict[int, List[Tuple[int, str]]] = {}
    for pos_1based, offset, h0_byte, h1_byte, _is_phased in insertion_rows:
        if h0_byte == dash and h1_byte == dash:
            raise RuntimeError(
                f"Insertion row at pos={pos_1based} offset={offset} has both "
                f"haplotype bytes '-': no insertion content to emit. This "
                f"indicates a bug in _build_haplotype_arrays."
            )
        arr_idx = pos_1based - start - 1
        if h0_byte != dash:
            ins_h0.setdefault(arr_idx, []).append((offset, chr(h0_byte)))
        if h1_byte != dash:
            ins_h1.setdefault(arr_idx, []).append((offset, chr(h1_byte)))
    for syms in ins_h0.values():
        syms.sort()
    for syms in ins_h1.values():
        syms.sort()
    return ins_h0, ins_h1


def _haplotype_pair_table(
    ref_arr: np.ndarray,
    hap0: np.ndarray,
    hap1: np.ndarray,
    insertion_rows: List[Tuple[int, int, int, int, bool]],
    masks: Dict[str, np.ndarray],
    start: int,
    mode: str,
) -> Dict[str, np.ndarray]:
    """Position-synced table of two haplotypes (v10 ``--format tsv``).

    See :meth:`GenomeSequenceProcessor.get_haplotype_pair_parallel_str`
    (``table_mode``) for the column contract. One row per reference cell of
    the region plus one per inserted base, sorted by ``(pos, offset)``;
    ``mode == "variants"`` drops rows where both haplotypes equal the
    reference and nothing is inserted.
    """
    n = len(ref_arr)
    dash = np.uint8(ord("-"))
    pos = np.arange(start + 1, start + 1 + n, dtype=np.int32)
    offset = np.zeros(n, dtype=np.int32)
    ref = np.asarray(ref_arr, dtype=np.uint8)
    h0 = np.asarray(hap0, dtype=np.uint8)
    h1 = np.asarray(hap1, dtype=np.uint8)
    snp0 = np.asarray(masks["is_snp_h0"], dtype=bool)
    snp1 = np.asarray(masks["is_snp_h1"], dtype=bool)
    indel0 = np.asarray(masks["is_indel_h0"], dtype=bool)
    indel1 = np.asarray(masks["is_indel_h1"], dtype=bool)

    if insertion_rows:
        n_ins = len(insertion_rows)
        ins_pos = np.fromiter((r[0] for r in insertion_rows), dtype=np.int32, count=n_ins)
        ins_off = np.fromiter((r[1] for r in insertion_rows), dtype=np.int32, count=n_ins)
        ins_h0 = np.fromiter((r[2] for r in insertion_rows), dtype=np.uint8, count=n_ins)
        ins_h1 = np.fromiter((r[3] for r in insertion_rows), dtype=np.uint8, count=n_ins)
        pos = np.concatenate([pos, ins_pos])
        offset = np.concatenate([offset, ins_off])
        ref = np.concatenate([ref, np.full(n_ins, dash, dtype=np.uint8)])
        h0 = np.concatenate([h0, ins_h0])
        h1 = np.concatenate([h1, ins_h1])
        zeros = np.zeros(n_ins, dtype=bool)
        snp0 = np.concatenate([snp0, zeros])
        snp1 = np.concatenate([snp1, zeros])
        indel0 = np.concatenate([indel0, ins_h0 != dash])
        indel1 = np.concatenate([indel1, ins_h1 != dash])
        order = np.lexsort((offset, pos))
        pos, offset, ref, h0, h1 = pos[order], offset[order], ref[order], h0[order], h1[order]
        snp0, snp1, indel0, indel1 = snp0[order], snp1[order], indel0[order], indel1[order]

    if mode == "variants":
        keep = (h0 != ref) | (h1 != ref) | (offset > 0)
        pos, offset, ref, h0, h1 = pos[keep], offset[keep], ref[keep], h0[keep], h1[keep]
        snp0, snp1, indel0, indel1 = snp0[keep], snp1[keep], indel0[keep], indel1[keep]

    return {
        "pos": pos, "offset": offset, "ref": ref, "hap0": h0, "hap1": h1,
        "snp0": snp0, "snp1": snp1, "indel0": indel0, "indel1": indel1,
    }


class UnphasedHeterozygoteError(ValueError):
    """A heterozygous genotype without phase (``0/1``) in haplotype-pair mode.

    The v10 haplotype-pair output promises two *true* phased haplotypes per
    region, so an unphased het cannot be placed on either. Raised by
    :func:`_require_fully_phased`; ``create_haplotype_dataset.py`` turns it
    into a FAILED sample (and a non-zero exit) rather than guessing.
    """


def _require_fully_phased(
    variants: Dict[int, Variant],
    chrom: str,
    start: int,
    end: int,
    sample: str = "",
) -> None:
    """Raise :class:`UnphasedHeterozygoteError` if any het variant is unphased.

    Must run on the **raw** variants dict as handed to a region function —
    *before* :func:`_resolve_overlapping_variants`, which can turn a
    homozygous ``1/1`` into ``(0, 1)`` while keeping ``phased=False`` (a
    one-haplotype trim), and would thus manufacture a spurious "unphased
    het" from a perfectly legal unphased hom.

    Homozygous and hemizygous (mirrored) calls are phase-neutral and pass
    whatever their ``phased`` flag says.

    Caveat: :func:`_merge_same_position_variants` ANDs the ``phased`` flags
    of co-located records, so an unphased hom record sharing a POS with a
    phased het record arrives here already as an unphased het and is
    rejected. That is strictly correct for a format that promises true
    phased pairs — the merged site's two alleles cannot be assigned.
    """
    for pos in sorted(variants):
        var = variants[pos]
        if var.gt[0] != var.gt[1] and not var.phased:
            _who = f"sample '{sample}' " if sample else ""
            raise UnphasedHeterozygoteError(
                f"Unphased heterozygous genotype {var.gt[0]}/{var.gt[1]} at "
                f"{chrom}:{pos} (REF='{var.ref}' ALT={var.alt}) in region "
                f"{chrom}:{start}-{end} {_who}— haplotype-pair output "
                f"requires every heterozygous call to be phased ('|')."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class GenomeSequenceProcessor:
    """Thread-safe genomic sequence processor.

    Combines reference genome sequences with VCF variant data to produce
    per-position haplotype information. Designed for efficient multi-threaded
    processing. For 2bit reference files, the entire genome is preloaded into
    memory at initialization (~7 seconds, ~3GB for hg38) for instant access.

    Args:
        ref_path: Path to reference genome (FASTA or 2bit).
        diploid_tokens_path: Path to diploid tokens TSV file.

    Attributes:
        ref_path: Path to the reference genome file.
        diploid_tokens_path: Path to the diploid tokens file.

    Thread Safety:
        Reference genome and token tables are read-only after initialization
        and are always safe to share across threads.

        get_sequence_df() and the batch methods (get_sequences_df,
        get_sequences_df_multisample, get_sequence_parallel) each create
        per-thread VCF readers internally and are fully concurrent.

        get_sequence_str() uses a shared cached VCF reader protected by an
        internal Lock, so concurrent calls are safe but serialize on the VCF
        I/O. For high-throughput parallel use, prefer get_sequence_parallel().

    Example:
        >>> with GenomeSequenceProcessor('hg38.2bit', 'tokens.tsv') as processor:
        ...     df = processor.get_sequence_df('sample.vcf.gz', 'chr1', 1000000, 2000000)
        >>>
        >>> # Multiple chromosomes
        >>> processor = GenomeSequenceProcessor('hg38.2bit', 'tokens.tsv')
        >>> for chrom in ['chr1', 'chr2']:
        ...     df = processor.get_sequence_df('sample.vcf.gz', chrom, 0, 100000)
        >>> processor.close()
    """

    def __init__(
        self,
        ref_path: Optional[Path] = None,
        diploid_tokens_path: Optional[Path] = None,
        allow_mixed_phasing: bool = False,
        force_unphased: Optional[bool] = None,
        filter_pass_only: bool = True,
        use_direction_markers: bool = False,
        emit_anchor_snp_marker: bool = True,
        *,
        force_unordered: Optional[bool] = None,
    ):
        if force_unordered is not None:
            warnings.warn(
                "'force_unordered' is deprecated; pass 'force_unphased' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if force_unordered is not None and force_unphased is not None:
            raise ValueError(
                "pass only one of 'force_unphased' / 'force_unordered'"
            )
        force_unphased_effective = bool(
            force_unphased if force_unphased is not None
            else (force_unordered if force_unordered is not None else False)
        )

        self._allow_mixed_phasing = allow_mixed_phasing
        self._force_unphased = force_unphased_effective
        self._filter_pass_only = filter_pass_only
        self._use_direction_markers = use_direction_markers
        # When False (v5 mode), compound-het INDEL anchors are NOT flagged as
        # SNPs, so the encoder emits the pre-v6 bare-anchor form (no `^` before
        # the anchor). Default True reproduces v6/v7 output. This is the ONLY
        # byte difference between v5 and v6 encoding; see _build_haplotype_arrays.
        self._emit_anchor_snp_marker = emit_anchor_snp_marker
        self._ref_file = None
        self._perf = None  # Set to a PerfAccumulator instance to enable timing
        #: Silent resolutions made while reading variants (keys:
        #: INTEGRITY_COUNTER_KEYS). Accumulates across calls; a dataset writer
        #: clears it per sample and records the totals.
        self.integrity: Counter = Counter()
        self._integrity_lock = threading.Lock()

        if use_direction_markers and force_unphased_effective:
            logger.warning(
                "use_direction_markers=True has no effect when "
                "force_unphased=True (no positions will be phased)"
            )

        self.ref_path = Path(ref_path) if ref_path else paths.HG38_2BIT
        self.diploid_tokens_path = (
            Path(diploid_tokens_path) if diploid_tokens_path else paths.DIPLOID_TOKENS
        )

        # Load diploid tokens (instance-level, read-only after init)
        if not self.diploid_tokens_path.exists():
            raise FileNotFoundError(
                f"Diploid tokens file not found: '{self.diploid_tokens_path}'. "
                f"Provide a valid path to the diploid tokens TSV file."
            )

        (
            self._phased_lookup,
            self._unphased_lookup,
            self._phased_table,
            self._unphased_table,
            self._meta_tokens,
        ) = self._load_tokens()

        # Marker meta tokens must be exactly one character: the diploid build
        # paths splice them in as single codepoints and the per-codepoint
        # seq_ref_idx alignment (used by annotations) assumes one marker == one
        # codepoint. A multi-char override would silently break that 1:1 map
        # (and the uint32 path's ord() would raise an opaque TypeError). The
        # haploid path already guards this against _HAPLOID_ALLOWED; mirror it
        # here for diploid, validated once at load time.
        _marker_defaults = {
            "<SNP>": SNP_MARKER,
            "<INDEL_span_start>": INDEL_OPEN,
            "<INDEL_span_end>": INDEL_CLOSE,
            "<DIR_FWD>": DIR_FWD,
            "<DIR_REV>": DIR_REV,
        }
        _resolved_markers = {}
        for _label, _default in _marker_defaults.items():
            _sym = self._meta_tokens.get(_label, _default)
            if len(_sym) != 1:
                raise ValueError(
                    f"Marker meta token {_label}={_sym!r} must be a single "
                    f"character (got length {len(_sym)}) in tokens file "
                    f"'{self.diploid_tokens_path}'. Marker symbols are spliced "
                    f"in as single codepoints; multi-char markers break the "
                    f"per-codepoint annotation alignment."
                )
            _resolved_markers[_label] = _sym

        # Cache the resolved marker symbols as attributes so the per-region
        # _build_marked_* hot paths read one attribute each instead of five
        # dict.get() calls per call (run hundreds of millions of times across a
        # full dataset). _meta_tokens is read-only after init, so these stay in
        # sync with it.
        self._snp_marker = _resolved_markers["<SNP>"]
        self._indel_open = _resolved_markers["<INDEL_span_start>"]
        self._indel_close = _resolved_markers["<INDEL_span_end>"]
        self._dir_fwd = _resolved_markers["<DIR_FWD>"]
        self._dir_rev = _resolved_markers["<DIR_REV>"]

        logger.info(
            "Loaded diploid tokens from %s: %d phased, %d unphased, %d meta",
            self.diploid_tokens_path,
            len(self._phased_lookup),
            len(self._unphased_lookup),
            len(self._meta_tokens),
        )

        # Open reference file (preloads entire genome for 2bit files)
        ref_file = ReferenceFile(self.ref_path)
        try:
            ref_file.__enter__()
        except Exception:
            ref_file.__exit__(None, None, None)
            raise
        self._ref_file = ref_file

        # Internal VCF/BCF reader cache — lazily opened, auto-managed.
        # A single cached reader is kept open across calls; replaced when
        # vcf_path or sample_name changes.  Closed in close()/__exit__.
        self._cached_vcf_path: Optional[Path] = None
        self._cached_vcf_reader: Optional[VCF] = None
        self._cached_vcf_sample: Optional[str] = None  # None = single-sample VCF

        # Lock that serialises all accesses to the cached reader above.
        # get_sequence_str holds this lock
        # for the duration of _get_or_open_vcf + VCF region query so that a
        # concurrent call on a different VCF path cannot close the reader
        # while it is in use.  The parallel batch methods (get_sequences_df,
        # _dispatch_parallel_str, etc.) create per-thread readers and never
        # touch this cache, so they remain fully concurrent.
        self._vcf_lock = threading.Lock()

    def _load_tokens(
        self,
    ) -> Tuple[Dict[str, str], Dict[str, str], np.ndarray, np.ndarray, Dict[str, str]]:
        """Load diploid tokens from TSV file for this instance.

        Returns:
            Tuple of (phased_lookup, unphased_lookup, phased_table,
            unphased_table, meta_tokens).  meta_tokens maps token names
            such as ``<SNP>``, ``<INDEL_span_start>``, ``<INDEL_span_end>``
            to their symbol characters.
        """
        path = self.diploid_tokens_path

        def _warn_malformed(_lineno: int, line: str) -> None:
            logger.warning(
                "Malformed line in diploid tokens file '%s': %r — "
                "expected at least 3 tab-separated fields "
                "(pair, order, symbol); skipping",
                path,
                line,
            )

        # Canonical labels are "phased"/"unphased"; the legacy
        # "ordered"/"unordered" labels are still accepted but emit a
        # DeprecationWarning (preprocess.tokens.normalize_order_label).
        table = tokens.load_token_table(path, on_malformed=_warn_malformed)
        return (
            table.phased,
            table.unphased,
            _build_diploid_lookup_table(table.phased),
            _build_diploid_lookup_table(table.unphased),
            table.meta,
        )

    def _validate_phasing_consistency(
        self,
        variants: Dict[int, Variant],
    ) -> None:
        """Validate that all variants in a region agree on phasing.

        A region must be either fully phased or fully unphased. Mixed
        phasing produces unreliable haplotype assignments and is rejected
        unless ``allow_mixed_phasing=True`` was set at construction time.

        Homozygous and hemizygous variants (both allele indices equal) are
        phasing-neutral — their phase assignment does not affect haplotype
        output — and are excluded from the consistency check.  This allows
        hemizygous sex-chromosome calls (e.g. chrX/Y in males) to coexist
        with phased heterozygous variants (e.g. PAR regions) without error.

        Raises:
            ValueError: If the region contains both phased and unphased
                        heterozygous variants and mixed phasing is not allowed.
        """
        if not variants or self._allow_mixed_phasing or self._force_unphased:
            return
        # Only heterozygous variants carry meaningful phase information.
        het_variants = [var for var in variants.values() if var.gt[0] != var.gt[1]]
        if not het_variants:
            return
        phased_values = {var.phased for var in het_variants}
        if len(phased_values) > 1:
            raise ValueError(
                "Region contains a mix of phased and unphased variants. "
                "All variants must be consistently phased or unphased. "
                "Set allow_mixed_phasing=True to override."
            )

    def _prepare_haplotype_data(
        self,
        variants: Dict[int, Variant],
        chrom: str,
        start: int,
        end: int,
        sample_name: str = "",
        per_hap_masks_out: Optional[Dict[str, np.ndarray]] = None,
        want_diploid: bool = True,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Shared setup for DataFrame and string output paths.

        Resolves overlapping variants, validates phasing consistency,
        fetches and validates reference sequence, builds haplotype arrays,
        and performs diploid lookup (skipped, ``diploid_chars = None``, when
        ``want_diploid`` is False — the v10 haplotype-pair path never reads
        it).

        Args:
            variants: Dict mapping 1-based positions to Variant objects.
            chrom: Chromosome name.
            start: 0-based start position.
            end: 0-based end position, exclusive.
            per_hap_masks_out: Forwarded to :func:`_build_haplotype_arrays`;
                when given, receives the four per-haplotype marker masks
                (``is_snp_h0/h1``, ``is_indel_h0/h1``) used by the v10
                haplotype-pair output. On the no-variants fast path the four
                keys are filled with all-False arrays so callers never see a
                missing key.

        Returns:
            Tuple of (ref_arr, hap0, hap1, source_is_vcf,
            diploid_chars, insertion_rows, is_snp, is_indel_non_anchor,
            phased_mask).
            ref_arr is a numpy uint8 array.
        """
        _p = self._perf
        _rs = end - start
        _nv = len(variants)

        if _p:
            with _p.time_block("resolve_overlapping", _rs, _nv):
                variants = _resolve_overlapping_variants(variants, chrom, sample_name)
            with _p.time_block("validate_phasing", _rs, _nv):
                self._validate_phasing_consistency(variants)
            with _p.time_block("ref_fetch", _rs):
                ref_arr = self._ref_file.fetch(chrom, start, end, as_array=True)
        else:
            variants = _resolve_overlapping_variants(variants, chrom, sample_name)
            self._validate_phasing_consistency(variants)
            ref_arr = self._ref_file.fetch(chrom, start, end, as_array=True)

        expected_len = end - start
        if len(ref_arr) != expected_len:
            raise ValueError(
                f"Reference returned {len(ref_arr)} bases for {chrom}:{start}-{end}, "
                f"expected {expected_len}. Check that coordinates are within "
                f"chromosome bounds."
            )

        logger.debug(
            "Fetched %d reference bases for %s:%d-%d",
            len(ref_arr),
            chrom,
            start,
            end,
        )

        # Fast path: no variants — skip haplotype building and full diploid lookup.
        # For all-reference positions hap0=hap1=ref, phased=False → unphased table.
        if not variants:
            diploid_chars = (
                self._unphased_table[ref_arr, ref_arr].view("<U1") if want_diploid else None
            )
            empty_bool = np.zeros(len(ref_arr), dtype=bool)
            if per_hap_masks_out is not None:
                for _k in ("is_snp_h0", "is_snp_h1", "is_indel_h0", "is_indel_h1"):
                    per_hap_masks_out[_k] = np.zeros(len(ref_arr), dtype=bool)
            return (
                ref_arr,       # ref_seq
                ref_arr,       # hap0
                ref_arr,       # hap1
                empty_bool,    # source_is_vcf
                diploid_chars, # diploid_chars
                [],            # insertion_rows
                empty_bool,    # is_snp
                empty_bool,    # is_indel_non_anchor
                empty_bool,    # phased_mask
            )

        if _p:
            with _p.time_block("validate_ref", _rs, _nv):
                _validate_variants_against_reference(variants, ref_arr, chrom, start)
            with _p.time_block("build_haplotype_arrays", _rs, _nv):
                (
                    hap0,
                    hap1,
                    source_is_vcf,
                    phased_mask,
                    insertion_rows,
                    is_snp,
                    is_indel_non_anchor,
                ) = _build_haplotype_arrays(
                    ref_arr, variants, start,
                    emit_anchor_snp_marker=self._emit_anchor_snp_marker,
                    per_hap_masks_out=per_hap_masks_out,
                )
            if self._force_unphased:
                phased_mask[:] = False
            with _p.time_block("diploid_lookup", _rs):
                if not want_diploid:
                    diploid_chars = None
                elif self._use_direction_markers and not self._force_unphased:
                    # Three-level encoding: always use unphased symbols;
                    # direction is carried separately by ‹/› markers.
                    diploid_chars = self._unphased_table[hap0, hap1].view("<U1")
                else:
                    diploid_chars = _vectorized_diploid_lookup(
                        hap0, hap1, phased_mask,
                        self._phased_table, self._unphased_table,
                    ).view("<U1")
        else:
            _validate_variants_against_reference(variants, ref_arr, chrom, start)
            (
                hap0,
                hap1,
                source_is_vcf,
                phased_mask,
                insertion_rows,
                is_snp,
                is_indel_non_anchor,
            ) = _build_haplotype_arrays(
                ref_arr, variants, start,
                emit_anchor_snp_marker=self._emit_anchor_snp_marker,
                per_hap_masks_out=per_hap_masks_out,
            )
            if self._force_unphased:
                phased_mask[:] = False
            if not want_diploid:
                diploid_chars = None
            elif self._use_direction_markers and not self._force_unphased:
                diploid_chars = self._unphased_table[hap0, hap1].view("<U1")
            else:
                diploid_chars = _vectorized_diploid_lookup(
                    hap0, hap1, phased_mask,
                    self._phased_table, self._unphased_table,
                ).view("<U1")

        return (
            ref_arr,
            hap0,
            hap1,
            source_is_vcf,
            diploid_chars,
            insertion_rows,
            is_snp,
            is_indel_non_anchor,
            phased_mask,
        )

    def _build_region_dataframe(
        self,
        variants: Dict[int, Variant],
        chrom: str,
        start: int,
        end: int,
        debug: bool = False,
        sample_name: str = "",
    ) -> pd.DataFrame:
        """Build haplotype DataFrame from pre-parsed variants.

        Resolves overlapping variants, fetches reference sequence, builds
        haplotype arrays, and constructs the final DataFrame. Shared by
        both single-sample and multi-sample code paths.

        Args:
            variants: Dict mapping 1-based positions to Variant objects.
            chrom: Chromosome name.
            start: 0-based start position.
            end: 0-based end position, exclusive.
            debug: If True, include 'source' column ("ref" or "vcf").
            sample_name: Sample name for log messages.

        Returns:
            DataFrame with columns: chrom, pos, offset, hap0, hap1,
            diploid_symbol, and optionally source.
        """
        if self._use_direction_markers and not self._force_unphased:
            raise ValueError(
                "use_direction_markers=True is not supported for DataFrame output. "
                "Direction information is only available in string/uint32 marked "
                "output paths (e.g. get_marked_sequence_parallel_str)."
            )

        (
            ref_seq,
            hap0,
            hap1,
            source_is_vcf,
            diploid_chars,
            insertion_rows,
            _is_snp,
            _is_indel,
            _phased_mask,
        ) = self._prepare_haplotype_data(variants, chrom, start, end, sample_name)

        n = len(ref_seq)
        hap0_chars = hap0.view("S1").astype("U1")
        hap1_chars = hap1.view("S1").astype("U1")

        pos_array = np.arange(start + 1, start + 1 + n, dtype=np.int64)

        data = {
            "chrom": pd.Categorical([chrom] * n, categories=[chrom]),
            "pos": pos_array,
            "offset": np.zeros(n, dtype=np.int32),
            "hap0": hap0_chars,
            "hap1": hap1_chars,
            "diploid_symbol": diploid_chars,
        }
        if debug:
            source_col = np.where(source_is_vcf, "vcf", "ref")
            data["source"] = pd.Categorical(source_col, categories=["ref", "vcf"])

        main_df = pd.DataFrame(data)

        if insertion_rows:
            n_ins = len(insertion_rows)
            ins_pos = np.empty(n_ins, dtype=np.int64)
            ins_offset = np.empty(n_ins, dtype=np.int32)
            ins_h0 = np.empty(n_ins, dtype="U1")
            ins_h1 = np.empty(n_ins, dtype="U1")
            ins_diploid = np.empty(n_ins, dtype="U1")

            for i, (pos_1based, offset, h0_byte, h1_byte, is_phased) in enumerate(
                insertion_rows
            ):
                h0_char = chr(h0_byte)
                h1_char = chr(h1_byte)
                pair = h0_char + h1_char
                lookup = self._unphased_lookup if self._force_unphased else (self._phased_lookup if is_phased else self._unphased_lookup)

                ins_pos[i] = pos_1based
                ins_offset[i] = offset
                ins_h0[i] = h0_char
                ins_h1[i] = h1_char
                ins_diploid[i] = lookup.get(pair, UNKNOWN_DIPLOID_SYMBOL)

            ins_data = {
                "chrom": pd.Categorical([chrom] * n_ins, categories=[chrom]),
                "pos": ins_pos,
                "offset": ins_offset,
                "hap0": ins_h0,
                "hap1": ins_h1,
                "diploid_symbol": ins_diploid,
            }
            if debug:
                ins_data["source"] = pd.Categorical(
                    ["vcf"] * n_ins, categories=["ref", "vcf"]
                )

            ins_df = pd.DataFrame(ins_data)
            main_df = pd.concat([main_df, ins_df], ignore_index=True)
            main_df = main_df.sort_values(
                ["pos", "offset"], kind="mergesort"
            ).reset_index(drop=True)

        logger.info("Built DataFrame with %d rows", len(main_df))
        return main_df

    def _get_sequence_df_core(
        self,
        vcf_path: Path,
        chrom: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        debug: bool = False,
        vcf_reader: Optional[VCF] = None,
    ) -> pd.DataFrame:
        """Core implementation for building per-position haplotype DataFrame.

        Private method containing the core logic for processing a single genomic
        region. Called by get_sequence_df() and get_sequences_df().

        Args:
            vcf_path: Path to VCF/BCF file (must be indexed, must contain exactly one sample).
            chrom: Chromosome name.
            start: 0-based start position (BED format). If None, defaults to 0.
            end: 0-based end position, exclusive (BED format). If None, defaults to chromosome end.
            debug: If True, include 'source' column ("ref" or "vcf").
            vcf_reader: Pre-opened ``cyvcf2.VCF`` reader for reuse. When
                provided the file at *vcf_path* is **not** reopened — the
                caller is responsible for open/close lifecycle. Sample-count
                validation is skipped (caller must validate beforehand).

        Returns:
            DataFrame with columns: chrom, pos, offset, hap0, hap1,
            diploid_symbol, and optionally source.

        Raises:
            ValueError: If start/end coordinates are invalid or VCF doesn't contain exactly one sample.
            FileNotFoundError: If VCF file not found.
        """
        if start is None:
            start = 0
        if end is None:
            end = self._ref_file.get_reference_length(chrom)

        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"end must be > start, got start={start}, end={end}")

        variants = _parse_variants_for_region(
            vcf_path, chrom, start, end, vcf_reader=vcf_reader,
            filter_pass_only=self._filter_pass_only,
        )
        return self._build_region_dataframe(variants, chrom, start, end, debug)

    def get_sequences_df(
        self,
        vcf_path: Path,
        regions: List[Tuple[str, Optional[int], Optional[int]]],
        max_workers: Optional[int] = None,
        parallel: bool = True,
        debug: bool = False,
    ) -> pd.DataFrame:
        """Build per-position haplotype DataFrame for multiple genomic regions.

        Processes multiple regions and returns a single concatenated DataFrame
        sorted by chromosome, position, and offset. When *parallel=True* and
        regions span multiple chromosomes, each chromosome is processed in its
        own thread with a single reused VCF reader. Regions within the same
        chromosome are always processed sequentially (one worker per chromosome)
        to avoid redundant VCF index loads and tabix seek contention.

        Args:
            vcf_path: Path to VCF/BCF file (must be indexed, must contain exactly one sample).
            regions: List of (chrom, start, end) tuples where start/end use 0-based
                     half-open BED coordinates. If start is None, defaults to 0.
                     If end is None, defaults to chromosome end.
            max_workers: Maximum number of parallel threads. If None, uses min(25, cpu_count).
                         Ignored if parallel=False or if regions span only one chromosome.
            parallel: If True (default), process chromosomes concurrently using
                      ThreadPoolExecutor when regions span multiple chromosomes.
                      If False, process all regions sequentially.
            debug: If True, include 'source' column ("ref" or "vcf") in output.

        Returns:
            DataFrame with columns: chrom, pos, offset, hap0, hap1, diploid_symbol,
            and optionally source. Results are sorted by natural chromosome order
            (chr1, chr2, ..., chr22, chrX, chrY, chrM, then alphabetical),
            then by pos, then by offset.

        Raises:
            ValueError: If regions is empty, coordinates are invalid, or VCF doesn't
                        contain exactly one sample.
            FileNotFoundError: If VCF file not found.
            RuntimeError: If any region processing fails (original exception is wrapped).

        Example:
            >>> # Process different chromosomes in parallel
            >>> with GenomeSequenceProcessor('hg38.2bit', 'tokens.tsv') as processor:
            ...     regions = [
            ...         ('chr1', 1000000, 2000000),
            ...         ('chr2', 1000000, 2000000),
            ...         ('chr3', 1000000, 2000000),
            ...     ]
            ...     df = processor.get_sequences_df('sample.vcf.gz', regions)
            ...     print(f"Processed {len(df)} positions across {len(regions)} regions")

            >>> # Sequential processing (useful for debugging)
            >>> df = processor.get_sequences_df('sample.vcf.gz', regions, parallel=False)

            >>> # Custom thread pool size
            >>> df = processor.get_sequences_df('sample.vcf.gz', regions, max_workers=8)

        Thread Safety:
            Safe for concurrent calls with non-overlapping regions. The method
            internally uses ThreadPoolExecutor when parallel=True. Each chromosome
            is processed independently with its own VCF reader.

        Performance:
            Parallelism is one worker per chromosome (capped at *max_workers*).
            Regions on the same chromosome share a single VCF reader and are
            processed sequentially in start-position order for optimal tabix
            throughput. Single-chromosome region lists are always processed
            sequentially to avoid unnecessary threading overhead.
        """
        import os
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Validate inputs
        if not regions:
            raise ValueError("regions list cannot be empty")

        # Validate VCF has exactly one sample (once, not per-region)
        try:
            _vcf_check = VCF(str(vcf_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open VCF/BCF file '{vcf_path}': {e}") from e
        try:
            if len(_vcf_check.samples) == 0:
                raise ValueError(
                    f"VCF file '{vcf_path}' has no samples. "
                    f"A VCF with exactly one sample is required."
                )
            elif len(_vcf_check.samples) > 1:
                raise ValueError(
                    f"VCF file '{vcf_path}' contains {len(_vcf_check.samples)} "
                    f"samples: {_vcf_check.samples}. "
                    f"A VCF with exactly one sample is required."
                )
        finally:
            _vcf_check.close()

        # Normalize regions (handle None start/end)
        normalized_regions = self._normalize_regions(regions)

        # Group regions by chromosome, sorted by start for sequential tabix access
        chrom_groups = defaultdict(list)
        for chrom, start, end in normalized_regions:
            chrom_groups[chrom].append((chrom, start, end))
        for chrom in chrom_groups:
            chrom_groups[chrom].sort(key=lambda r: r[1])

        n_chromosomes = len(chrom_groups)

        def process_chrom_regions(chrom_regions):
            """Process all regions for one chromosome with a single VCF reader."""
            thread_reader = VCF(str(vcf_path))
            try:
                dfs = []
                for chrom, start, end in chrom_regions:
                    try:
                        df = self._get_sequence_df_core(
                            vcf_path=vcf_path,
                            chrom=chrom,
                            start=start,
                            end=end,
                            debug=debug,
                            vcf_reader=thread_reader,
                        )
                        dfs.append(df)
                    except (ValueError, FileNotFoundError):
                        raise
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to process region {chrom}:{start}-{end}: {e}"
                        ) from e
                return dfs
            finally:
                thread_reader.close()

        # Decide whether to parallelize: only when multiple chromosomes
        # and parallel=True. One worker per chromosome avoids redundant
        # VCF index loads and tabix seek contention.
        use_parallel = parallel and n_chromosomes > 1

        if use_parallel:
            if max_workers is None:
                max_workers = min(25, os.cpu_count() or 1)
            # Cap workers at chromosome count — no benefit from more
            effective_workers = min(max_workers, n_chromosomes)

            logger.info(
                "Processing %d regions across %d chromosomes in parallel "
                "with %d workers",
                len(normalized_regions),
                n_chromosomes,
                effective_workers,
            )

            results = []
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                future_to_chrom = {
                    executor.submit(process_chrom_regions, chrom_regs): chrom
                    for chrom, chrom_regs in chrom_groups.items()
                }
                for future in as_completed(future_to_chrom):
                    chrom = future_to_chrom[future]
                    try:
                        chrom_dfs = future.result()
                        results.extend(chrom_dfs)
                    except Exception as e:
                        logger.error(
                            "Error processing chromosome %s: %s",
                            chrom,
                            e,
                        )
                        raise
        else:
            # Sequential processing with a single reused VCF reader
            reason = "single chromosome" if n_chromosomes == 1 else "parallel=False"
            logger.info(
                "Processing %d regions sequentially (%s)",
                len(normalized_regions),
                reason,
            )
            # Flatten chromosome groups back into sorted order
            all_regions = []
            for chrom_regs in chrom_groups.values():
                all_regions.extend(chrom_regs)
            results = process_chrom_regions(all_regions)

        # Concatenate and sort results
        if not results:
            return pd.DataFrame(
                columns=["chrom", "pos", "offset", "hap0", "hap1", "diploid_symbol"]
            )

        combined_df = pd.concat(results, ignore_index=True)
        combined_df = self._sort_by_natural_chrom_order(combined_df)

        logger.info(
            "Combined %d DataFrames into %d total rows",
            len(results),
            len(combined_df),
        )

        return combined_df

    def get_sequences_df_multisample(
        self,
        bcf_dir: Path,
        sample_name: str,
        regions: List[Tuple[str, Optional[int], Optional[int]]],
        max_workers: Optional[int] = None,
        parallel: bool = True,
        debug: bool = False,
    ) -> pd.DataFrame:
        """Build per-position haplotype DataFrame for multiple regions from multi-sample BCFs.

        Like get_sequences_df but reads from a directory of per-chromosome
        multi-sample BCF files, extracting a single sample. Each chromosome
        is processed with a single reused VCF reader. When *parallel=True*
        and regions span multiple chromosomes, each chromosome gets its own
        worker thread.

        Args:
            bcf_dir: Directory containing per-chromosome BCF files.
            sample_name: Sample name to extract from each BCF.
            regions: List of (chrom, start, end) tuples (BED coordinates).
            max_workers: Maximum number of parallel threads (default: min(25, cpu_count)).
                         Ignored if parallel=False or if regions span only one chromosome.
            parallel: If True (default), process chromosomes concurrently when
                      regions span multiple chromosomes.
            debug: If True, include 'source' column in output.

        Returns:
            DataFrame with columns: chrom, pos, offset, hap0, hap1, diploid_symbol,
            and optionally source.
        """
        import os
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not regions:
            raise ValueError("regions list cannot be empty")

        normalized_regions = self._normalize_regions(regions)

        # Group regions by chromosome, sorted by start position
        chrom_groups = defaultdict(list)
        for chrom, start, end in normalized_regions:
            chrom_groups[chrom].append((chrom, start, end))
        for chrom in chrom_groups:
            chrom_groups[chrom].sort(key=lambda r: r[1])

        n_chromosomes = len(chrom_groups)

        def process_chrom_regions(chrom_regions):
            """Process all regions for one chromosome with a single BCF reader."""
            chrom = chrom_regions[0][0]
            bcf_path = _resolve_sample_bcf(bcf_dir, chrom, sample_name)
            thread_reader = VCF(str(bcf_path))
            try:
                if sample_name not in thread_reader.samples:
                    raise ValueError(
                        f"Sample '{sample_name}' not found in '{bcf_path}'"
                    )
                thread_reader.set_samples([sample_name])
                dfs = []
                for rgn_chrom, start, end in chrom_regions:
                    try:
                        df = self._get_sequence_df_from_multisample_bcf(
                            bcf_dir=bcf_dir,
                            sample_name=sample_name,
                            chrom=rgn_chrom,
                            start=start,
                            end=end,
                            debug=debug,
                            vcf_reader=thread_reader,
                        )
                        dfs.append(df)
                    except (ValueError, FileNotFoundError):
                        raise
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to process region {rgn_chrom}:{start}-{end}: {e}"
                        ) from e
                return dfs
            finally:
                thread_reader.close()

        use_parallel = parallel and n_chromosomes > 1

        if use_parallel:
            if max_workers is None:
                max_workers = min(25, os.cpu_count() or 1)
            effective_workers = min(max_workers, n_chromosomes)

            logger.info(
                "Processing %d regions across %d chromosomes in parallel "
                "with %d workers",
                len(normalized_regions),
                n_chromosomes,
                effective_workers,
            )

            results = []
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                future_to_chrom = {
                    executor.submit(process_chrom_regions, chrom_regs): chrom
                    for chrom, chrom_regs in chrom_groups.items()
                }
                for future in as_completed(future_to_chrom):
                    chrom = future_to_chrom[future]
                    try:
                        chrom_dfs = future.result()
                        results.extend(chrom_dfs)
                    except Exception as e:
                        logger.error(
                            "Error processing chromosome %s: %s",
                            chrom,
                            e,
                        )
                        raise
        else:
            reason = "single chromosome" if n_chromosomes == 1 else "parallel=False"
            logger.info(
                "Processing %d regions sequentially (%s)",
                len(normalized_regions),
                reason,
            )
            results = []
            for chrom_regs in chrom_groups.values():
                results.extend(process_chrom_regions(chrom_regs))

        if not results:
            return pd.DataFrame(
                columns=[
                    "chrom",
                    "pos",
                    "offset",
                    "hap0",
                    "hap1",
                    "diploid_symbol",
                ]
            )

        combined_df = pd.concat(results, ignore_index=True)
        combined_df = self._sort_by_natural_chrom_order(combined_df)

        logger.info(
            "Combined %d DataFrames into %d total rows",
            len(results),
            len(combined_df),
        )

        return combined_df

    def get_sequence_df(
        self,
        vcf_path: Optional[Path] = None,
        chrom: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        debug: bool = False,
        bcf_dir: Optional[Path] = None,
        sample_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Build per-position haplotype DataFrame for a genomic region.

        Combines reference sequence with variant data to produce haplotype
        information for each position in the specified region. Thread-safe
        for concurrent calls with non-overlapping regions.

        Supports two modes:
        1. Single-sample VCF: Provide vcf_path (must contain exactly one sample)
        2. Multi-sample BCF: Provide bcf_dir and sample_name

        Args:
            vcf_path: Path to VCF/BCF file (must be indexed, must contain exactly one sample).
                      Mutually exclusive with bcf_dir/sample_name.
            chrom: Chromosome name.
            start: 0-based start position (BED format). If None, defaults to 0.
            end: 0-based end position, exclusive (BED format). If None, defaults to chromosome end.
            debug: If True, include 'source' column ("ref" or "vcf").
            bcf_dir: Directory containing per-chromosome BCF files.
                     Mutually exclusive with vcf_path. Requires sample_name.
            sample_name: Sample name to extract from multi-sample BCF.
                         Required when using bcf_dir.

        Returns:
            DataFrame with columns: chrom, pos, offset, hap0, hap1,
            diploid_symbol, and optionally source.

        Raises:
            ValueError: If start/end coordinates are invalid, or if neither vcf_path
                        nor (bcf_dir + sample_name) are provided, or if both are provided.
            FileNotFoundError: If VCF/BCF file not found.

        Example:
            >>> # Single-sample VCF
            >>> with GenomeSequenceProcessor('hg38.2bit', 'tokens.tsv') as processor:
            ...     df = processor.get_sequence_df(vcf_path='sample.vcf.gz', chrom='chr1',
            ...                                      start=1000000, end=2000000)

            >>> # Multi-sample BCF
            >>> with GenomeSequenceProcessor('hg38.2bit', 'tokens.tsv') as processor:
            ...     df = processor.get_sequence_df(bcf_dir='/data/bcf/', sample_name='NA12878',
            ...                                      chrom='chr1', start=1000000, end=2000000)
        """
        # Validate mutually exclusive parameters
        if vcf_path is not None and (bcf_dir is not None or sample_name is not None):
            raise ValueError(
                "vcf_path is mutually exclusive with bcf_dir/sample_name. "
                "Provide either vcf_path (for single-sample VCF) or "
                "bcf_dir + sample_name (for multi-sample BCF)."
            )

        if vcf_path is None and (bcf_dir is None or sample_name is None):
            raise ValueError(
                "Must provide either vcf_path (for single-sample VCF) or "
                "both bcf_dir and sample_name (for multi-sample BCF)."
            )

        # Multi-sample BCF mode
        if bcf_dir is not None:
            return self._get_sequence_df_from_multisample_bcf(
                bcf_dir=bcf_dir,
                sample_name=sample_name,
                chrom=chrom,
                start=start,
                end=end,
                debug=debug,
            )

        # Single-sample VCF mode (original behavior)
        regions = [(chrom, start, end)]
        return self.get_sequences_df(
            vcf_path=vcf_path,
            regions=regions,
            parallel=False,  # No threading overhead for single region
            debug=debug,
        )

    def _get_sequence_df_from_multisample_bcf(
        self,
        bcf_dir: Path,
        sample_name: str,
        chrom: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        debug: bool = False,
        vcf_reader: Optional[VCF] = None,
    ) -> pd.DataFrame:
        """Extract variants from a multi-sample BCF for a specific sample.

        Private helper method that reads a multi-sample BCF from a directory
        and extracts variants only for the specified sample, then processes
        them into a DataFrame. This method transparently provides the same
        output as get_sequence_df with a single-sample VCF.

        Args:
            bcf_dir: Directory containing per-chromosome BCF files.
            sample_name: Sample name to extract from the multi-sample BCF.
            chrom: Chromosome name to look for in BCF filename.
            start: 0-based start position (BED format). If None, defaults to 0.
            end: 0-based end position, exclusive (BED format). If None, defaults to chromosome end.
            debug: If True, include 'source' column ("ref" or "vcf").
            vcf_reader: Pre-opened ``cyvcf2.VCF`` reader already restricted to
                *sample_name* via ``set_samples``. When provided the BCF file
                is **not** reopened. Caller manages open/close lifecycle.

        Returns:
            DataFrame with same format as get_sequence_df output.

        Raises:
            ValueError: If coordinates are invalid or sample not found in BCF.
            FileNotFoundError: If BCF directory or file not found.
            RuntimeError: If BCF file cannot be opened.
        """
        if start is None:
            start = 0
        if end is None:
            end = self._ref_file.get_reference_length(chrom)

        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"end must be > start, got start={start}, end={end}")

        bcf_path = _resolve_sample_bcf(bcf_dir, chrom, sample_name)
        variants = _parse_variants_for_region_multisample(
            bcf_path,
            sample_name,
            chrom,
            start,
            end,
            vcf_reader=vcf_reader,
            filter_pass_only=self._filter_pass_only,
        )
        return self._build_region_dataframe(
            variants,
            chrom,
            start,
            end,
            debug,
            sample_name,
        )

    def _normalize_regions(
        self, regions: List[Tuple[str, Optional[int], Optional[int]]]
    ) -> List[Tuple[str, int, int]]:
        """Normalize region tuples, filling in None values for start/end.

        Args:
            regions: List of (chrom, start, end) tuples. start/end may be None.

        Returns:
            List of (chrom, start, end) tuples with all None values resolved.

        Raises:
            ValueError: If any region has invalid coordinates.
        """
        normalized = []
        for region in regions:
            if len(region) != 3:
                raise ValueError(
                    f"Each region must be a 3-tuple (chrom, start, end), got {region}"
                )

            chrom, start, end = region

            # Default start to 0 if None
            if start is None:
                start = 0

            # Default end to chromosome length if None
            if end is None:
                try:
                    end = self._ref_file.get_reference_length(chrom)
                except Exception as e:
                    raise ValueError(
                        f"Cannot determine length of chromosome '{chrom}': {e}"
                    ) from e

            # Validate coordinates
            if start < 0:
                raise ValueError(f"Region {chrom}:{start}-{end} has negative start")
            if end <= start:
                raise ValueError(f"Region {chrom}:{start}-{end} has end <= start")

            normalized.append((chrom, start, end))

        return normalized

    def _sort_by_natural_chrom_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort DataFrame by natural chromosome order, then pos, then offset.

        Natural chromosome order: chr1, chr2, ..., chr22, chrX, chrY, chrM,
        then any other chromosomes alphabetically.

        Args:
            df: DataFrame with 'chrom', 'pos', and 'offset' columns.

        Returns:
            Sorted DataFrame with reset index.
        """
        # Build ordered category list: standard chroms first, then any
        # extra chromosomes present in the data (sorted alphabetically).
        standard_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
        present_chroms = set(df["chrom"].unique())
        extra = sorted(present_chroms - set(standard_order))
        full_order = [c for c in standard_order if c in present_chroms] + extra

        df["chrom"] = pd.Categorical(
            df["chrom"],
            categories=full_order,
            ordered=True,
        )
        df = df.sort_values(
            ["chrom", "pos", "offset"],
            kind="mergesort",
        ).reset_index(drop=True)

        # Restore chrom to plain string (avoid downstream surprises)
        df["chrom"] = df["chrom"].astype(str)

        return df

    def get_sequence_str(
        self,
        vcf_path: Optional[Path] = None,
        chrom: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        skip_gaps: bool = True,
        bcf_dir: Optional[Path] = None,
        sample_name: Optional[str] = None,
        mark_variant_boundaries: bool = False,
    ) -> str:
        """Get diploid sequence as a string for a genomic region.

        Builds the diploid symbol string directly from numpy arrays,
        bypassing DataFrame construction for better performance.

        The VCF/BCF reader is managed internally: the first call for a given
        *vcf_path* (or *bcf_dir* + *sample_name*) opens and caches the reader;
        subsequent calls with the same file reuse it at no extra overhead.

        Supports two modes:
        1. Single-sample VCF: Provide vcf_path (must contain exactly one sample)
        2. Multi-sample BCF: Provide bcf_dir and sample_name

        Args:
            vcf_path: Path to VCF/BCF file (must be indexed, must contain exactly one sample).
                      Mutually exclusive with bcf_dir/sample_name.
            chrom: Chromosome name.
            start: 0-based start position (BED format). If None, defaults to 0.
            end: 0-based end position, exclusive (BED format). If None, defaults to chromosome end.
            skip_gaps: When True, gap characters ('-') are omitted from result.
            bcf_dir: Directory containing per-chromosome BCF files.
                     Mutually exclusive with vcf_path. Requires sample_name.
            sample_name: Sample name to extract from multi-sample BCF.
                         Required when using bcf_dir.
            mark_variant_boundaries: When True, insert special markers around
                variants: ``^`` before each SNP symbol, and ``{`` / ``}``
                around the non-anchor content of INDELs.

        Returns:
            Diploid sequence as a string of diploid symbols.

        Raises:
            ValueError: If both vcf_path and bcf_dir are provided, or if neither is provided.
        """
        if vcf_path is not None and (bcf_dir is not None or sample_name is not None):
            raise ValueError(
                "vcf_path is mutually exclusive with bcf_dir/sample_name. "
                "Provide either vcf_path (for single-sample VCF) or "
                "bcf_dir + sample_name (for multi-sample BCF)."
            )

        if vcf_path is None and (bcf_dir is None or sample_name is None):
            raise ValueError(
                "Must provide either vcf_path (for single-sample VCF) or "
                "both bcf_dir and sample_name (for multi-sample BCF)."
            )

        if start is None:
            start = 0
        if end is None:
            end = self._ref_file.get_reference_length(chrom)

        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"end must be > start, got start={start}, end={end}")

        # Hold _vcf_lock for the entire open + query sequence so that a
        # concurrent call on a different VCF path cannot close this reader
        # while it is in use.  _sequence_str_from_variants works on purely
        # local data and is called after the lock is released.
        with self._vcf_lock:
            if bcf_dir is not None:
                bcf_path = _find_bcf_for_chromosome(bcf_dir, chrom)
                reader = self._get_or_open_vcf(bcf_path, sample_name)
                variants = _parse_variants_for_region_multisample(
                    bcf_path,
                    sample_name,
                    chrom,
                    start,
                    end,
                    vcf_reader=reader,
                    filter_pass_only=self._filter_pass_only,
                )
            else:
                reader = self._get_or_open_vcf(vcf_path)
                variants = _parse_variants_for_region(
                    vcf_path,
                    chrom,
                    start,
                    end,
                    vcf_reader=reader,
                    filter_pass_only=self._filter_pass_only,
                )

        return self._sequence_str_from_variants(
            variants,
            chrom,
            start,
            end,
            sample_name or "",
            skip_gaps,
            mark_variant_boundaries,
        )

    def get_haploid_sequence_str(
        self,
        vcf_path: Optional[Path] = None,
        chrom: str = "",
        start: Optional[int] = None,
        end: Optional[int] = None,
        skip_gaps: bool = True,
        bcf_dir: Optional[Path] = None,
        sample_name: Optional[str] = None,
    ) -> str:
        """Single-region haploid string (analogue of :meth:`get_sequence_str`).

        Used by the create_dataset.py per-region fallback path on chunk
        failure. Serialises VCF access through the shared ``_vcf_lock`` —
        for concurrent throughput, use
        :meth:`get_marked_haploid_sequence_parallel_str` instead.

        Boundary markers (``^``, ``{``, ``}``) are always emitted. The
        haploid alphabet is ``{A,C,G,T,N,^,{,}}`` (plus lowercase variants);
        ``skip_gaps`` must be True.
        """
        if not skip_gaps:
            raise ValueError(
                "Haploid mode requires skip_gaps=True; the output alphabet "
                "is {A,C,G,T,N,^,{,}} and does not allow '-'."
            )
        if vcf_path is not None and (bcf_dir is not None or sample_name is not None):
            raise ValueError(
                "vcf_path is mutually exclusive with bcf_dir/sample_name."
            )
        if vcf_path is None and (bcf_dir is None or sample_name is None):
            raise ValueError(
                "Must provide either vcf_path or both bcf_dir and sample_name."
            )
        if self._use_direction_markers:
            raise ValueError(
                "use_direction_markers=True is incompatible with haploid output."
            )

        if start is None:
            start = 0
        if end is None:
            end = self._ref_file.get_reference_length(chrom)
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"end must be > start, got start={start}, end={end}")

        with self._vcf_lock:
            if bcf_dir is not None:
                bcf_path = _find_bcf_for_chromosome(bcf_dir, chrom)
                reader = self._get_or_open_vcf(bcf_path, sample_name)
                variants = _parse_variants_for_region_multisample(
                    bcf_path, sample_name, chrom, start, end,
                    vcf_reader=reader,
                    filter_pass_only=self._filter_pass_only,
                )
            else:
                reader = self._get_or_open_vcf(vcf_path)
                variants = _parse_variants_for_region(
                    vcf_path, chrom, start, end,
                    vcf_reader=reader,
                    filter_pass_only=self._filter_pass_only,
                )

        return self._sequence_haploid_str_from_variants(
            variants, chrom, start, end, sample_name or "", skip_gaps=skip_gaps,
        )

    def _sequence_str_from_variants(
        self,
        variants: Dict[int, "Variant"],
        chrom: str,
        start: int,
        end: int,
        sample_name: str,
        skip_gaps: bool,
        mark_variant_boundaries: bool,
        *,
        return_ref_idx: bool = False,
        return_var_idx: bool = False,
    ):
        """Build the diploid sequence string from an already-parsed variants dict.

        Shared implementation behind :meth:`get_sequence_str` and the parallel
        string methods, so variant parsing happens once per region.

        When ``return_ref_idx=True``, returns a ``(seq, seq_ref_idx)`` tuple
        instead of just ``seq``; requires ``mark_variant_boundaries=True``.
        ``seq_ref_idx`` is documented on
        :meth:`_build_marked_string_with_ref_idx`.

        When ``return_var_idx=True`` (also requires
        ``mark_variant_boundaries=True``), returns a
        ``(seq, seq_ref_idx, seq_var_idx, variants_list)`` tuple instead.
        ``seq_var_idx`` and ``variants_list`` are documented on
        :meth:`_build_marked_string_with_ref_idx_and_var_idx` /
        :class:`VariantCall`. Implies ``return_ref_idx=True``.
        """
        if return_ref_idx and not mark_variant_boundaries:
            raise ValueError(
                "return_ref_idx=True requires mark_variant_boundaries=True."
            )
        if return_var_idx and not mark_variant_boundaries:
            raise ValueError(
                "return_var_idx=True requires mark_variant_boundaries=True."
            )
        _p = self._perf
        _rs = end - start

        if _p:
            with _p.time_block("prepare_haplotype_data", _rs, len(variants)):
                (
                    _ref_seq,
                    hap0,
                    hap1,
                    _source_is_vcf,
                    diploid_chars,
                    insertion_rows,
                    is_snp,
                    is_indel_non_anchor,
                    phased_mask,
                ) = self._prepare_haplotype_data(
                    variants,
                    chrom,
                    start,
                    end,
                    sample_name,
                )
        else:
            (
                _ref_seq,
                hap0,
                hap1,
                _source_is_vcf,
                diploid_chars,
                insertion_rows,
                is_snp,
                is_indel_non_anchor,
                phased_mask,
            ) = self._prepare_haplotype_data(
                variants,
                chrom,
                start,
                end,
                sample_name,
            )

        _sb_t0 = time.perf_counter() if _p else 0

        _use_dir = (
            mark_variant_boundaries
            and self._use_direction_markers
            and not self._force_unphased
        )

        if self._use_direction_markers and not mark_variant_boundaries:
            raise ValueError(
                "use_direction_markers=True requires mark_variant_boundaries=True. "
                "Direction markers cannot be emitted without variant boundary markers."
            )

        # Resolve insertion symbols and group by 0-based array index.
        # When direction markers are active, use unphased lookup and
        # store (offset, sym, dir_val) tuples; otherwise (offset, sym).
        # defaultdict(list) skips the throwaway [] that setdefault builds on
        # every append; downstream consumers only index ins_by_idx after an
        # `i in ins_by_idx` membership check, so auto-vivification can't add
        # spurious keys (membership / `not ins_by_idx` truthiness match a dict).
        ins_by_idx: Dict[int, List[Tuple]] = defaultdict(list)
        for pos_1based, offset, h0_byte, h1_byte, is_phased in insertion_rows:
            pair = chr(h0_byte) + chr(h1_byte)
            if _use_dir:
                lookup = self._unphased_lookup
            elif self._force_unphased:
                lookup = self._unphased_lookup
            else:
                lookup = self._phased_lookup if is_phased else self._unphased_lookup
            sym = lookup.get(pair, UNKNOWN_DIPLOID_SYMBOL)
            if skip_gaps and sym == "-":
                continue
            arr_idx = pos_1based - start - 1
            if _use_dir:
                dir_val = _compute_insertion_direction(h0_byte, h1_byte, is_phased)
                ins_by_idx[arr_idx].append((offset, sym, dir_val))
            else:
                ins_by_idx[arr_idx].append((offset, sym))

        # Most positions carry a single insertion; sorting a 0/1-element list is
        # a no-op, so only sort when there is something to order.
        for syms in ins_by_idx.values():
            if len(syms) > 1:
                syms.sort()

        if mark_variant_boundaries:
            direction = None
            if _use_dir:
                direction = _compute_direction_array(hap0, hap1, phased_mask)
            if return_var_idx:
                pos_to_var_idx, variant_calls = _build_pos_to_var_idx(
                    variants, start, end,
                )
                seq, seq_ref_idx, seq_var_idx = (
                    self._build_marked_string_with_ref_idx_and_var_idx(
                        diploid_chars,
                        is_snp,
                        is_indel_non_anchor,
                        ins_by_idx,
                        skip_gaps,
                        direction,
                        pos_to_var_idx,
                    )
                )
                if _p:
                    _p._record_elapsed("string_building", _sb_t0, _rs)
                return seq, seq_ref_idx, seq_var_idx, variant_calls
            if return_ref_idx:
                result = self._build_marked_string_with_ref_idx(
                    diploid_chars,
                    is_snp,
                    is_indel_non_anchor,
                    ins_by_idx,
                    skip_gaps,
                    direction,
                )
            else:
                result = self._build_marked_string(
                    diploid_chars,
                    is_snp,
                    is_indel_non_anchor,
                    ins_by_idx,
                    skip_gaps,
                    direction,
                )
            if _p:
                _p._record_elapsed("string_building", _sb_t0, _rs)
            return result

        # --- Original (unmarked) path ---

        if not insertion_rows:
            # Fast path: no insertions
            if skip_gaps:
                result = "".join(diploid_chars[diploid_chars != "-"])
            else:
                result = "".join(diploid_chars)
            if _p:
                _p._record_elapsed("string_building", _sb_t0, _rs)
            return result

        # Build output by splicing insertion strings between segments
        # of the main diploid array.  The O(n) work stays in numpy /
        # C-level string ops; Python only iterates over insertion sites.
        if skip_gaps:
            main_chars = diploid_chars[diploid_chars != "-"]
            # Recompute insertion indices relative to gap-free array.
            # Build a cumulative gap count so we can adjust indices.
            gap_mask = diploid_chars == "-"
            cum_gaps = np.cumsum(gap_mask)
            adjusted_ins: Dict[int, List[Tuple[int, str]]] = {}
            for arr_idx, syms in ins_by_idx.items():
                new_idx = arr_idx - int(cum_gaps[arr_idx])
                adjusted_ins[new_idx] = syms
            ins_by_idx = adjusted_ins
        else:
            main_chars = diploid_chars

        if not ins_by_idx:
            result = "".join(main_chars)
            if _p:
                _p._record_elapsed("string_building", _sb_t0, _rs)
            return result

        main_str = "".join(main_chars)

        # Collect insertion points (sorted by position) and build result
        # by concatenating segments between insertion points.
        sorted_idxs = sorted(ins_by_idx.keys())
        parts = []
        prev_end = 0
        for arr_idx in sorted_idxs:
            # Segment from previous end up to and including the anchor
            seg_end = arr_idx + 1
            if seg_end > prev_end:
                parts.append(main_str[prev_end:seg_end])
            # Splice in insertion symbols
            for _, ins_sym in ins_by_idx[arr_idx]:
                parts.append(ins_sym)
            prev_end = seg_end
        # Remaining segment after last insertion
        if prev_end < len(main_str):
            parts.append(main_str[prev_end:])

        result = "".join(parts)
        if _p:
            _p._record_elapsed("string_building", _sb_t0, _rs)
        return result

    def _sequence_haploid_str_from_variants(
        self,
        variants: Dict[int, "Variant"],
        chrom: str,
        start: int,
        end: int,
        sample_name: str,
        skip_gaps: bool = True,
        *,
        return_ref_idx: bool = False,
    ):
        """Build the haploid sequence string from an already-parsed variants dict.

        Reuses the per-position decomposition produced by
        :func:`_build_haplotype_arrays` and collapses each cell with the
        ALT-wins-leftmost rule (see :func:`_collapse_haploid`). Always emits
        with SNP/INDEL boundary markers; the diploid 256x256 lookup is not
        used. Direction markers are never emitted (phase is destroyed by
        the collapse, by design).

        The output alphabet is ``{A,C,G,T,N,^,{,}}`` (plus lowercase
        variants). ``skip_gaps`` must be True; haploid output never contains
        '-'. The internal per-cell uint8 array may carry '-' at deletion
        content cells — those cells are dropped from the emitted string.
        """
        if not skip_gaps:
            raise ValueError(
                "Haploid mode requires skip_gaps=True; the output alphabet "
                "is {A,C,G,T,N,^,{,}} and does not allow '-'."
            )

        _p = self._perf
        _rs = end - start

        if _p:
            with _p.time_block("prepare_haplotype_data", _rs, len(variants)):
                (
                    ref_arr,
                    hap0,
                    hap1,
                    _source_is_vcf,
                    _diploid_chars,
                    insertion_rows,
                    is_snp,
                    is_indel_non_anchor,
                    _phased_mask,
                ) = self._prepare_haplotype_data(
                    variants, chrom, start, end, sample_name,
                )
        else:
            (
                ref_arr,
                hap0,
                hap1,
                _source_is_vcf,
                _diploid_chars,
                insertion_rows,
                is_snp,
                is_indel_non_anchor,
                _phased_mask,
            ) = self._prepare_haplotype_data(
                variants, chrom, start, end, sample_name,
            )

        _sb_t0 = time.perf_counter() if _p else 0

        collapsed_arr = _collapse_haploid(hap0, hap1, ref_arr)
        collapsed_chars = collapsed_arr.view("S1").astype("U1")

        ins_by_idx = _haploid_ins_by_idx(insertion_rows, start)

        if return_ref_idx:
            result = self._build_marked_string_haploid_with_ref_idx(
                collapsed_chars,
                is_snp,
                is_indel_non_anchor,
                ins_by_idx,
            )
        else:
            result = self._build_marked_string_haploid(
                collapsed_chars,
                is_snp,
                is_indel_non_anchor,
                ins_by_idx,
            )
        if _p:
            _p._record_elapsed("string_building", _sb_t0, _rs)
        return result

    def _dispatch_parallel_str(
        self,
        regions: List[Tuple[str, int, int]],
        vcf_path: Optional[Path],
        bcf_dir: Optional[Path],
        sample_name: Optional[str],
        max_workers: Optional[int],
        region_fn,
        region_batch_size: Optional[int] = None,
        ref_fn=None,
    ) -> List[any]:
        """Shared dispatch for string-returning parallel methods.

        Groups regions by chromosome and dispatches them to a thread pool.
        When ``region_batch_size`` is set, large chromosomes are split into
        sub-batches so that the thread pool can balance load across more tasks
        than the number of chromosomes (e.g., preventing chr1 from becoming
        the bottleneck).  Each sub-batch opens its own VCF reader.

        Without sub-batching (region_batch_size=None), one task is submitted
        per chromosome — the original behaviour.

        Args:
            regions: List of (chrom, start, end) tuples (0-based BED coords).
            vcf_path: Path to single-sample VCF/BCF (mutually exclusive with bcf_dir).
            bcf_dir: Directory of per-chromosome BCFs (mutually exclusive with vcf_path).
            sample_name: Sample to extract from multi-sample BCF.
            max_workers: Max parallel threads.  Default: ``min(48, cpu_count)`` when
                region_batch_size is set, ``min(25, cpu_count)`` otherwise.
            region_fn: Callable(variants, chrom, start, end) -> result_value.
                Called for each region with its parsed variants.
            region_batch_size: When set, split each chromosome's regions into
                sub-batches of at most this many regions.  Each sub-batch runs
                as an independent thread-pool task with its own VCF reader.
                Recommended value: 500.  Default: None (one task per chromosome).
            ref_fn: Optional Callable(chrom, start, end) -> result_value.
                Fast path for regions with no variants.  When provided,
                regions that have no parsed variants call ref_fn instead of
                region_fn, skipping unnecessary variant processing overhead.

        Returns:
            List of region_fn results in the same order as the input ``regions``.
        """
        import os
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not regions:
            return []

        if vcf_path is not None and (bcf_dir is not None or sample_name is not None):
            raise ValueError("vcf_path is mutually exclusive with bcf_dir/sample_name.")
        if vcf_path is None and (bcf_dir is None or sample_name is None):
            raise ValueError(
                "Must provide either vcf_path or both bcf_dir and sample_name."
            )

        if max_workers is None:
            # Use a higher default when sub-batching so the larger task pool can
            # actually run concurrently.
            if region_batch_size is not None:
                max_workers = min(48, os.cpu_count() or 1)
            else:
                max_workers = min(25, os.cpu_count() or 1)

        vcf_path = Path(vcf_path) if vcf_path is not None else None
        bcf_dir = Path(bcf_dir) if bcf_dir is not None else None

        # Tag each region with its original index before grouping so we can
        # reassemble results in input order without tuple-keyed dict lookups.
        # Group and sort by chromosome for sequential tabix access.
        chrom_groups: Dict[str, List[Tuple[int, Tuple[str, int, int]]]] = defaultdict(
            list
        )
        for orig_idx, region in enumerate(regions):
            chrom_groups[region[0]].append((orig_idx, region))
        for chrom in chrom_groups:
            chrom_groups[chrom].sort(key=lambda x: x[1][1])

        n_chromosomes = len(chrom_groups)

        # Build BCF path cache once (avoids repeated directory listing +
        # regex match per chromosome per sub-batch).
        bcf_file_cache: Dict[str, Path] = {}
        if bcf_dir is not None:
            for _chrom in chrom_groups:
                try:
                    bcf_file_cache[_chrom] = _resolve_sample_bcf(
                        bcf_dir, _chrom, sample_name
                    )
                except FileNotFoundError:
                    logger.warning(
                        "No BCF file for '%s' in '%s'; skipping %d regions.",
                        _chrom,
                        bcf_dir,
                        len(chrom_groups[_chrom]),
                    )

        def process_chrom_regions(indexed_chrom_regions):
            """Process all regions for one chromosome with a single VCF reader.

            Returns a dict keyed by original index (int) — far cheaper to hash
            than 3-tuple region keys.
            """
            chrom = indexed_chrom_regions[0][1][0]
            _n_batch = len(indexed_chrom_regions)
            _t0_batch = time.perf_counter()
            _counters: Counter = Counter()  # merged into self.integrity below
            if bcf_dir is not None:
                bcf_file = bcf_file_cache.get(chrom)
                if bcf_file is None:
                    return {}
                _t0_open = time.perf_counter()
                thread_reader = VCF(str(bcf_file))
                _t_open = time.perf_counter() - _t0_open
                try:
                    if sample_name not in thread_reader.samples:
                        raise ValueError(
                            f"Sample '{sample_name}' not found in '{bcf_file}'"
                        )
                    thread_reader.set_samples([sample_name])
                    _p = self._perf
                    # Batch query: single CSI seek for all regions on this chromosome.
                    # After set_samples, target sample is at index 0.
                    _t0_bcf = time.perf_counter()
                    all_variants, _n_recs, _n_vars = _batch_parse_variants_for_chrom(
                        indexed_chrom_regions,
                        thread_reader,
                        chrom,
                        sample_idx=0,
                        _perf=_p,
                        filter_pass_only=self._filter_pass_only,
                        counters=_counters,
                    )
                    _t_bcf = time.perf_counter() - _t0_bcf
                    chunk_results = {}
                    _t0_fn = time.perf_counter()
                    for orig_idx, rgn in indexed_chrom_regions:
                        rgn_chrom, start, end = rgn
                        variants = all_variants.get(orig_idx)
                        if not variants and ref_fn is not None:
                            chunk_results[orig_idx] = ref_fn(rgn_chrom, start, end)
                            continue
                        variants = variants or {}
                        _rs = end - start
                        _nv = len(variants)
                        if _p:
                            with _p.time_block("region_fn_total", _rs, _nv):
                                chunk_results[orig_idx] = region_fn(
                                    variants, rgn_chrom, start, end
                                )
                        else:
                            chunk_results[orig_idx] = region_fn(
                                variants, rgn_chrom, start, end
                            )
                    _t_fn = time.perf_counter() - _t0_fn
                    _t_wall = time.perf_counter() - _t0_batch
                    # Guard the format args: time.strftime() (and the %-format
                    # itself) run eagerly before logger.info decides to drop the
                    # record. Default level is ERROR, so skip the work entirely.
                    if logger.isEnabledFor(logging.INFO):
                        _sn = sample_name or "?"
                        logger.info(
                            "  [%s] %s[%dr] open=%.2fs bcf=%.2fs(%dscn/%dvar) fn=%.2fs wall=%.2fs [%s]",
                            _sn, chrom, _n_batch,
                            _t_open, _t_bcf, _n_recs, _n_vars, _t_fn, _t_wall,
                            time.strftime('%H:%M:%S'),
                        )
                    with self._integrity_lock:
                        self.integrity.update(_counters)
                    return chunk_results
                finally:
                    thread_reader.close()
            else:
                _t0_open = time.perf_counter()
                thread_reader = VCF(str(vcf_path))
                _t_open = time.perf_counter() - _t0_open
                try:
                    _p = self._perf
                    # Batch query: single CSI seek for all regions on this chromosome.
                    _t0_bcf = time.perf_counter()
                    all_variants, _n_recs, _n_vars = _batch_parse_variants_for_chrom(
                        indexed_chrom_regions,
                        thread_reader,
                        chrom,
                        sample_idx=0,
                        _perf=_p,
                        filter_pass_only=self._filter_pass_only,
                        counters=_counters,
                    )
                    _t_bcf = time.perf_counter() - _t0_bcf
                    chunk_results = {}
                    _t0_fn = time.perf_counter()
                    for orig_idx, rgn in indexed_chrom_regions:
                        rgn_chrom, start, end = rgn
                        variants = all_variants.get(orig_idx)
                        if not variants and ref_fn is not None:
                            chunk_results[orig_idx] = ref_fn(rgn_chrom, start, end)
                            continue
                        variants = variants or {}
                        _rs = end - start
                        _nv = len(variants)
                        if _p:
                            with _p.time_block("region_fn_total", _rs, _nv):
                                chunk_results[orig_idx] = region_fn(
                                    variants, rgn_chrom, start, end
                                )
                        else:
                            chunk_results[orig_idx] = region_fn(
                                variants, rgn_chrom, start, end
                            )
                    _t_fn = time.perf_counter() - _t0_fn
                    _t_wall = time.perf_counter() - _t0_batch
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(
                            "  [vcf] %s[%dr] open=%.2fs bcf=%.2fs(%dscn/%dvar) fn=%.2fs wall=%.2fs [%s]",
                            chrom, _n_batch,
                            _t_open, _t_bcf, _n_recs, _n_vars, _t_fn, _t_wall,
                            time.strftime('%H:%M:%S'),
                        )
                    with self._integrity_lock:
                        self.integrity.update(_counters)
                    return chunk_results
                finally:
                    thread_reader.close()

        results: List[any] = [None] * len(regions)

        if region_batch_size is not None:
            # Sub-chromosome batching: split large chromosomes into sub-batches so
            # that the thread pool balances work beyond the chromosome count.
            # Each sub-batch is processed by process_chrom_regions with its own
            # VCF reader and a CSI seek spanning only its sub-range.
            all_tasks: List[List[Tuple[int, Tuple[str, int, int]]]] = []
            for chrom_regs in chrom_groups.values():
                for i in range(0, len(chrom_regs), region_batch_size):
                    all_tasks.append(chrom_regs[i : i + region_batch_size])
            n_tasks = len(all_tasks)
            if n_tasks > 1:
                effective_workers = min(max_workers, n_tasks)
                if effective_workers <= 1:
                    # Sequential — no thread pool overhead (GIL makes
                    # threading useless for cyvcf2 BGZF decompression).
                    for task in all_tasks:
                        for orig_idx, value in process_chrom_regions(task).items():
                            results[orig_idx] = value
                else:
                    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                        futures = {
                            executor.submit(process_chrom_regions, task): task
                            for task in all_tasks
                        }
                        for future in as_completed(futures):
                            task = futures[future]
                            try:
                                for orig_idx, value in future.result().items():
                                    results[orig_idx] = value
                            except Exception as e:
                                chrom = task[0][1][0]
                                logger.error(
                                    "Error processing sub-batch on %s (sample=%s): %s",
                                    chrom, sample_name, e,
                                )
                                raise
            else:
                for orig_idx, value in process_chrom_regions(all_tasks[0]).items():
                    results[orig_idx] = value
        else:
            # Original: one task per chromosome.
            effective_workers = min(max_workers, n_chromosomes)
            use_parallel = n_chromosomes > 1 and effective_workers > 1
            if use_parallel:
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    future_to_chrom = {
                        executor.submit(process_chrom_regions, chrom_regs): chrom
                        for chrom, chrom_regs in chrom_groups.items()
                    }
                    for future in as_completed(future_to_chrom):
                        chrom = future_to_chrom[future]
                        try:
                            for orig_idx, value in future.result().items():
                                results[orig_idx] = value
                        except Exception as e:
                            logger.error(
                                "Error processing chromosome %s: %s", chrom, e
                            )
                            raise
            else:
                # Single chromosome — process sequentially
                for chrom_regs in chrom_groups.values():
                    for orig_idx, value in process_chrom_regions(chrom_regs).items():
                        results[orig_idx] = value

        return results

    def get_sequence_parallel(
        self,
        regions: List[Tuple[str, int, int]],
        vcf_path: Optional[Path] = None,
        skip_gaps: bool = True,
        bcf_dir: Optional[Path] = None,
        sample_name: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> List[str]:
        """Process multiple regions, returning one sequence string per region.

        Groups regions by chromosome — each chromosome is processed with a
        single reused VCF reader. Multiple chromosomes are processed in
        parallel (one worker per chromosome); single-chromosome region lists
        are processed sequentially to avoid unnecessary threading overhead.

        Args:
            regions: List of ``(chrom, start, end)`` tuples (0-based BED coords).
            vcf_path: Path to single-sample VCF/BCF.
                      Mutually exclusive with bcf_dir/sample_name.
            skip_gaps: When True, gap characters ('-') are omitted from results.
            bcf_dir: Directory containing per-chromosome BCF files.
                     Mutually exclusive with vcf_path. Requires sample_name.
            sample_name: Sample to extract from multi-sample BCF.
            max_workers: Number of parallel threads.  Defaults to
                ``min(25, os.cpu_count())``.

        Returns:
            List of diploid sequence strings in the same order as the input
            ``regions`` list.
        """

        def region_fn(variants, chrom, start, end):
            return self._sequence_str_from_variants(
                variants,
                chrom,
                start,
                end,
                sample_name or "",
                skip_gaps,
                False,
            )

        return self._dispatch_parallel_str(
            regions,
            vcf_path,
            bcf_dir,
            sample_name,
            max_workers,
            region_fn,
        )

    def get_marked_sequence_parallel_uint32(
        self,
        regions: List[Tuple[str, int, int]],
        vcf_path: Optional[Path] = None,
        skip_gaps: bool = True,
        bcf_dir: Optional[Path] = None,
        sample_name: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Process multiple regions, returning marked uint32 codepoint arrays.

        Like :meth:`get_sequence_parallel` but returns numpy uint32 arrays
        with SNP/INDEL delimiter markers (``^``, ``{``, ``}``) baked in.
        Intended for dataset generation where codepoints are stored directly.

        Args:
            regions: List of ``(chrom, start, end)`` tuples (0-based BED coords).
            vcf_path: Path to single-sample VCF/BCF.
                      Mutually exclusive with bcf_dir/sample_name.
            skip_gaps: When True, gap characters ('-') are omitted from results.
            bcf_dir: Directory containing per-chromosome BCF files.
                     Mutually exclusive with vcf_path. Requires sample_name.
            sample_name: Sample to extract from multi-sample BCF.
            max_workers: Number of parallel threads.  Defaults to
                ``min(25, os.cpu_count())``.

        Returns:
            List of uint32 numpy arrays in the same order as the input
            ``regions`` list. Each array contains diploid symbol codepoints
            with variant boundary markers.
        """

        def region_fn(variants, chrom, start, end):
            return self._sequence_marked_uint32_from_variants(
                variants,
                chrom,
                start,
                end,
                sample_name or "",
                skip_gaps,
            )

        return self._dispatch_parallel_str(
            regions,
            vcf_path,
            bcf_dir,
            sample_name,
            max_workers,
            region_fn,
        )

    def get_marked_sequence_parallel_str(
        self,
        regions: List[Tuple[str, int, int]],
        vcf_path: Optional[Path] = None,
        skip_gaps: bool = True,
        bcf_dir: Optional[Path] = None,
        sample_name: Optional[str] = None,
        max_workers: Optional[int] = None,
        region_batch_size: Optional[int] = 500,
    ) -> List[str]:
        """Process multiple regions, returning marked diploid sequence strings.

        Like :meth:`get_sequence_parallel` but with SNP/INDEL delimiter
        markers (``^``, ``{``, ``}``) baked into each string.

        Args:
            regions: List of ``(chrom, start, end)`` tuples (0-based BED coords).
            vcf_path: Path to single-sample VCF/BCF.
                      Mutually exclusive with bcf_dir/sample_name.
            skip_gaps: When True, gap characters ('-') are omitted from results.
            bcf_dir: Directory containing per-chromosome BCF files.
                     Mutually exclusive with vcf_path. Requires sample_name.
            sample_name: Sample to extract from multi-sample BCF.
            max_workers: Number of parallel threads.  Defaults to
                ``min(48, os.cpu_count())`` when region_batch_size is set,
                ``min(25, os.cpu_count())`` otherwise.
            region_batch_size: Max regions per thread-pool task.  Defaults to
                500, which splits large chromosomes (e.g. chr1 with ~15k
                regions) into ~30 sub-batches for better load balancing.
                Pass ``None`` to restore one-task-per-chromosome behaviour.

        Returns:
            List of marked diploid sequence strings in the same order as
            the input ``regions`` list.
        """

        def region_fn(variants, chrom, start, end):
            return self._sequence_str_from_variants(
                variants,
                chrom,
                start,
                end,
                sample_name or "",
                skip_gaps,
                True,  # mark_variant_boundaries
            )

        def ref_fn(chrom, start, end):
            return self._fast_reference_str(chrom, start, end, skip_gaps)

        return self._dispatch_parallel_str(
            regions,
            vcf_path,
            bcf_dir,
            sample_name,
            max_workers,
            region_fn,
            region_batch_size=region_batch_size,
            ref_fn=ref_fn,
        )

    def _fast_reference_str(
        self, chrom: str, start: int, end: int, skip_gaps: bool = True
    ) -> str:
        """Reference-only diploid string — no variant processing overhead.

        Bypasses _prepare_haplotype_data and _build_marked_string entirely.
        Used as a fast path for regions known to have no variants.
        """
        ref_arr = self._ref_file.fetch(chrom, start, end, as_array=True)
        diploid_codes = self._unphased_table[ref_arr, ref_arr]
        diploid_chars = diploid_codes.view("<U1")
        if skip_gaps:
            return _u1_to_str(diploid_chars[diploid_chars != "-"])
        return _u1_to_str(diploid_chars)

    def _fast_reference_str_with_ref_idx(
        self, chrom: str, start: int, end: int, skip_gaps: bool = True
    ) -> Tuple[str, np.ndarray]:
        """Reference-only diploid (string, seq_ref_idx) — no variants.

        Every kept char comes from a ref position, so ``seq_ref_idx`` is the
        kept index (after optional gap removal). No markers, no insertions.
        """
        ref_arr = self._ref_file.fetch(chrom, start, end, as_array=True)
        diploid_codes = self._unphased_table[ref_arr, ref_arr]
        diploid_chars = diploid_codes.view("<U1")
        if skip_gaps:
            mask = diploid_chars != "-"
            return _u1_to_str(diploid_chars[mask]), np.flatnonzero(mask).astype(np.int32)
        return _u1_to_str(diploid_chars), np.arange(len(diploid_chars), dtype=np.int32)

    def _fast_reference_str_with_ref_idx_and_var_idx(
        self, chrom: str, start: int, end: int, skip_gaps: bool = True,
    ) -> Tuple[str, np.ndarray, np.ndarray, List["VariantCall"]]:
        """Reference-only counterpart of
        :meth:`_build_marked_string_with_ref_idx_and_var_idx`.

        No variants in this region, so ``seq_var_idx`` is uniformly ``-1``
        and ``variant_calls`` is empty. Kept here as a fast path to skip
        ``_prepare_haplotype_data`` when ``ref_fn`` is invoked by
        :meth:`_dispatch_parallel_str`.
        """
        seq, seq_ref_idx = self._fast_reference_str_with_ref_idx(
            chrom, start, end, skip_gaps,
        )
        seq_var_idx = np.full(len(seq_ref_idx), -1, dtype=np.int32)
        return seq, seq_ref_idx, seq_var_idx, []

    def get_marked_sequence_with_ref_idx_parallel_str(
        self,
        regions: List[Tuple[str, int, int]],
        vcf_path: Optional[Path] = None,
        skip_gaps: bool = True,
        bcf_dir: Optional[Path] = None,
        sample_name: Optional[str] = None,
        max_workers: Optional[int] = None,
        region_batch_size: Optional[int] = 500,
    ) -> List[Tuple[str, np.ndarray]]:
        """Like :meth:`get_marked_sequence_parallel_str` but also emits ``seq_ref_idx``.

        Returns one ``(seq, seq_ref_idx)`` tuple per input region, in input
        order. ``seq_ref_idx`` is an ``int32`` array of length
        ``len(seq)``: nonneg = ref offset within the region; ``-1`` =
        inserted base; ``-2`` = marker codepoint. See
        :meth:`_build_marked_string_with_ref_idx`.

        Used by the v7 annotation pipeline (e.g. PhyloP) to align per-ref
        scores with codepoints of the marked sequence string.
        """

        def region_fn(variants, chrom, start, end):
            return self._sequence_str_from_variants(
                variants,
                chrom,
                start,
                end,
                sample_name or "",
                skip_gaps,
                True,  # mark_variant_boundaries
                return_ref_idx=True,
            )

        def ref_fn(chrom, start, end):
            return self._fast_reference_str_with_ref_idx(chrom, start, end, skip_gaps)

        return self._dispatch_parallel_str(
            regions,
            vcf_path,
            bcf_dir,
            sample_name,
            max_workers,
            region_fn,
            region_batch_size=region_batch_size,
            ref_fn=ref_fn,
        )

    def get_marked_sequence_with_ref_idx_and_variants_parallel_str(
        self,
        regions: List[Tuple[str, int, int]],
        vcf_path: Optional[Path] = None,
        skip_gaps: bool = True,
        bcf_dir: Optional[Path] = None,
        sample_name: Optional[str] = None,
        max_workers: Optional[int] = None,
        region_batch_size: Optional[int] = 500,
    ) -> List[Tuple[str, np.ndarray, np.ndarray, List["VariantCall"]]]:
        """Like :meth:`get_marked_sequence_with_ref_idx_parallel_str` but also
        emits ``seq_var_idx`` and the per-region resolved variant list.

        Returns one ``(seq, seq_ref_idx, seq_var_idx, variants)`` tuple per
        input region, in input order. ``seq_var_idx`` is documented on
        :meth:`_build_marked_string_with_ref_idx_and_var_idx`; ``variants``
        is a list of :class:`VariantCall` in ascending ``pos`` order.

        Used by variant-aware annotations (e.g. CADD) that need per-haplotype
        allele info to pick the right score and the codepoint-to-variant
        mapping to place it on the correct anchor codepoint.
        """

        def region_fn(variants, chrom, start, end):
            return self._sequence_str_from_variants(
                variants,
                chrom,
                start,
                end,
                sample_name or "",
                skip_gaps,
                True,  # mark_variant_boundaries
                return_var_idx=True,
            )

        def ref_fn(chrom, start, end):
            return self._fast_reference_str_with_ref_idx_and_var_idx(
                chrom, start, end, skip_gaps,
            )

        return self._dispatch_parallel_str(
            regions,
            vcf_path,
            bcf_dir,
            sample_name,
            max_workers,
            region_fn,
            region_batch_size=region_batch_size,
            ref_fn=ref_fn,
        )

    def get_reference_sequences_str(
        self,
        regions: List[Tuple[str, int, int]],
        skip_gaps: bool = True,
    ) -> List[str]:
        """Return diploid-encoded reference sequences (no variants) for regions.

        Encodes each reference position as a homozygous diploid symbol.
        No VCF/BCF is needed — only the preloaded reference genome is used.

        Args:
            regions: List of ``(chrom, start, end)`` tuples (0-based BED coords).
            skip_gaps: When True, gap characters ('-') are omitted from results.

        Returns:
            List of diploid sequence strings in the same order as input.
        """
        results: List[str] = []
        for chrom, start, end in regions:
            seq = self._sequence_str_from_variants(
                {},  # empty variants → pure reference
                chrom,
                start,
                end,
                "",  # no sample
                skip_gaps,
                True,  # mark_variant_boundaries — harmless with empty variants;
                       # avoids ValueError when use_direction_markers=True
            )
            results.append(seq)
        return results

    def get_reference_sequences_with_ref_idx_str(
        self,
        regions: List[Tuple[str, int, int]],
        skip_gaps: bool = True,
    ) -> List[Tuple[str, np.ndarray]]:
        """Reference-only counterpart of :meth:`get_marked_sequence_with_ref_idx_parallel_str`.

        Returns one ``(seq, seq_ref_idx)`` tuple per region; the reference
        has no variants, no insertions and no markers, so ``seq_ref_idx``
        is simply the kept-indices (``np.flatnonzero(non-gap mask)`` with
        ``skip_gaps=True``).
        """
        return [
            self._fast_reference_str_with_ref_idx(chrom, start, end, skip_gaps)
            for chrom, start, end in regions
        ]

    def get_marked_haploid_sequence_parallel_str(
        self,
        regions: List[Tuple[str, int, int]],
        vcf_path: Optional[Path] = None,
        skip_gaps: bool = True,
        bcf_dir: Optional[Path] = None,
        sample_name: Optional[str] = None,
        max_workers: Optional[int] = None,
        region_batch_size: Optional[int] = 500,
    ) -> List[str]:
        """Process multiple regions, returning marked haploid sequence strings.

        Haploid counterpart of :meth:`get_marked_sequence_parallel_str`.
        Emits one ALT-biased sequence per region over the alphabet
        ``{A,C,G,T,N,^,{,}}`` (see :func:`_collapse_haploid` for the
        per-cell rule). ``skip_gaps`` must be True; haploid output never
        contains '-'. Direction markers and the diploid 256x256 lookup are
        not used.

        Args:
            regions: List of ``(chrom, start, end)`` tuples (0-based BED coords).
            vcf_path: Path to single-sample VCF/BCF.
                      Mutually exclusive with bcf_dir/sample_name.
            skip_gaps: Must be True. Present for signature symmetry with the
                diploid method; passing False raises ``ValueError``.
            bcf_dir: Directory containing per-chromosome BCF files.
                     Mutually exclusive with vcf_path. Requires sample_name.
            sample_name: Sample to extract from multi-sample BCF.
            max_workers: Number of parallel threads. Defaults to
                ``min(48, os.cpu_count())`` when region_batch_size is set,
                ``min(25, os.cpu_count())`` otherwise.
            region_batch_size: Max regions per thread-pool task. Defaults to
                500. Pass ``None`` to restore one-task-per-chromosome behaviour.

        Returns:
            List of marked haploid sequence strings in the same order as
            the input ``regions`` list.
        """
        if not skip_gaps:
            raise ValueError(
                "Haploid mode requires skip_gaps=True; the output alphabet "
                "is {A,C,G,T,N,^,{,}} and does not allow '-'."
            )
        if self._use_direction_markers:
            raise ValueError(
                "use_direction_markers=True is incompatible with haploid "
                "output (phase is destroyed by the per-cell ALT-wins "
                "collapse). Construct the processor without "
                "use_direction_markers when calling the haploid methods."
            )

        def region_fn(variants, chrom, start, end):
            return self._sequence_haploid_str_from_variants(
                variants,
                chrom,
                start,
                end,
                sample_name or "",
                skip_gaps=skip_gaps,
            )

        def ref_fn(chrom, start, end):
            return self._fast_reference_haploid_str(chrom, start, end, skip_gaps)

        return self._dispatch_parallel_str(
            regions,
            vcf_path,
            bcf_dir,
            sample_name,
            max_workers,
            region_fn,
            region_batch_size=region_batch_size,
            ref_fn=ref_fn,
        )

    def _fast_reference_haploid_str(
        self, chrom: str, start: int, end: int, skip_gaps: bool = True
    ) -> str:
        """Reference-only haploid string — no variant processing overhead.

        Used as a fast path inside :meth:`get_marked_haploid_sequence_parallel_str`
        for regions with no parsed variants. The output is simply the
        reference bases joined as ACGTN (no markers, no insertions).
        """
        if not skip_gaps:
            raise ValueError(
                "Haploid mode requires skip_gaps=True; the output alphabet "
                "is {A,C,G,T,N,^,{,}} and does not allow '-'."
            )
        ref_arr = self._ref_file.fetch(chrom, start, end, as_array=True)
        ref_chars = ref_arr.view("S1").astype("U1")
        result = "".join(ref_chars[ref_chars != "-"])

        unexpected = set(result) - _HAPLOID_ALLOWED
        if unexpected:
            raise RuntimeError(
                f"Reference fetch for {chrom}:{start}-{end} produced "
                f"disallowed character(s) {sorted(unexpected)!r}. "
                f"Allowed haploid alphabet: {sorted(_HAPLOID_ALLOWED)}."
            )
        return result

    def _fast_reference_haploid_str_with_ref_idx(
        self, chrom: str, start: int, end: int, skip_gaps: bool = True
    ) -> Tuple[str, np.ndarray]:
        """Reference-only haploid (string, seq_ref_idx) — no variants."""
        if not skip_gaps:
            raise ValueError(
                "Haploid mode requires skip_gaps=True; the output alphabet "
                "is {A,C,G,T,N,^,{,}} and does not allow '-'."
            )
        ref_arr = self._ref_file.fetch(chrom, start, end, as_array=True)
        ref_chars = ref_arr.view("S1").astype("U1")
        mask = ref_chars != "-"
        result = "".join(ref_chars[mask])

        unexpected = set(result) - _HAPLOID_ALLOWED
        if unexpected:
            raise RuntimeError(
                f"Reference fetch for {chrom}:{start}-{end} produced "
                f"disallowed character(s) {sorted(unexpected)!r}. "
                f"Allowed haploid alphabet: {sorted(_HAPLOID_ALLOWED)}."
            )
        return result, np.flatnonzero(mask).astype(np.int32)

    def get_marked_haploid_sequence_with_ref_idx_parallel_str(
        self,
        regions: List[Tuple[str, int, int]],
        vcf_path: Optional[Path] = None,
        skip_gaps: bool = True,
        bcf_dir: Optional[Path] = None,
        sample_name: Optional[str] = None,
        max_workers: Optional[int] = None,
        region_batch_size: Optional[int] = 500,
    ) -> List[Tuple[str, np.ndarray]]:
        """Haploid counterpart of :meth:`get_marked_sequence_with_ref_idx_parallel_str`.

        Returns one ``(seq, seq_ref_idx)`` tuple per region. ``seq_ref_idx``
        semantics match the diploid version: nonneg = ref offset; ``-1`` =
        inserted base; ``-2`` = marker codepoint.
        """
        if not skip_gaps:
            raise ValueError(
                "Haploid mode requires skip_gaps=True; the output alphabet "
                "is {A,C,G,T,N,^,{,}} and does not allow '-'."
            )
        if self._use_direction_markers:
            raise ValueError(
                "use_direction_markers=True is incompatible with haploid "
                "output (phase is destroyed by the per-cell ALT-wins "
                "collapse). Construct the processor without "
                "use_direction_markers when calling the haploid methods."
            )

        def region_fn(variants, chrom, start, end):
            return self._sequence_haploid_str_from_variants(
                variants,
                chrom,
                start,
                end,
                sample_name or "",
                skip_gaps=skip_gaps,
                return_ref_idx=True,
            )

        def ref_fn(chrom, start, end):
            return self._fast_reference_haploid_str_with_ref_idx(
                chrom, start, end, skip_gaps
            )

        return self._dispatch_parallel_str(
            regions,
            vcf_path,
            bcf_dir,
            sample_name,
            max_workers,
            region_fn,
            region_batch_size=region_batch_size,
            ref_fn=ref_fn,
        )

    def get_haplotype_pair_parallel_str(
        self,
        regions: List[Tuple[str, int, int]],
        bcf_dir: Optional[Path] = None,
        vcf_path: Optional[Path] = None,
        sample_name: Optional[str] = None,
        max_workers: Optional[int] = None,
        region_batch_size: Optional[int] = 500,
        table_mode: Optional[str] = None,
    ) -> List[Optional[tuple]]:
        """Per-region **phased haplotype pairs** — the v10 encoder entry point.

        For every region returns ``(hap0, hap1, n_variants, n_het)`` — or,
        with ``table_mode`` set, ``(hap0, hap1, n_variants, n_het, table)`` —
        or ``None``:

        * ``hap0`` / ``hap1`` — the left / right GT allele's haploid string
          over the alphabet ``{A,C,G,T,N,^,{,}}``. Markers are **per
          haplotype**: ``^`` precedes a base only when it is a 1-bp
          substitution *on that haplotype*; ``{...}`` encloses INDEL content
          *on that haplotype*, and a carried deletion is rendered as an empty
          span (``REF=ACG ALT=A`` → carrier ``A{}T``, non-carrier ``ACGT``,
          no braces). So each string is independently interpretable:
          ``^``/``{}`` mean "non-reference content on this haplotype".
          Hom-alt sites are marked on both; chrX hemizygous calls arrive
          mirrored (encoded as hom) and render identically on both. The
          indel anchor base stays outside the braces (inherited haploid
          grammar: MNV ``REF=AC ALT=GT`` renders ``G{T}`` with an unmarked
          anchor). A deletion straddling the region start renders as a
          left-truncated ``{}`` / ``{C}`` block, unlike the v4–v8 haploid
          builder where it vanished.
        * ``n_variants`` — positions that **change at least one base inside
          the region**: records whose REF span overlaps the region, after
          same-POS merge, after :func:`_resolve_overlapping_variants` (a
          record trimmed to reference on both haplotypes is not counted) and
          after :func:`_variant_touches_region` (a deletion anchored on the
          region's last base deletes nothing inside it and is not counted);
          ``n_het`` — those with ``gt[0] != gt[1]``, so ``n_het > 0`` implies
          ``hap0 != hap1``.
        * ``None`` — no counted variant (variant-free, or every record's
          change lies outside the region: the caller writes no row), **or**
          its chromosome has no BCF in ``bcf_dir`` (the CLI must pre-filter
          regions to BCF-header chromosomes, as ``create_dataset.py`` does,
          so the two cases are not confused).

        Raises :class:`UnphasedHeterozygoteError` for any heterozygous call
        that is not phased (``0/1``), checked on the raw variant dict before
        overlap resolution (see :func:`_require_fully_phased`). Unphased
        homozygous calls (``1/1``) are accepted.

        Must be called on a processor constructed with
        ``force_unphased=False`` and ``use_direction_markers=False``; either
        contradicts a format that promises true phased pairs and raises
        ``ValueError``. The diploid 256×256 lookup inside
        :meth:`_prepare_haplotype_data` is skipped (``want_diploid=False``);
        the tokens file remains required by ``__init__``.

        Args:
            regions: ``(chrom, start, end)`` tuples (0-based BED coords).
            bcf_dir: Directory of per-chromosome BCFs; requires ``sample_name``.
                Mutually exclusive with ``vcf_path``.
            vcf_path: Single-sample VCF/BCF. Mutually exclusive with ``bcf_dir``.
            sample_name: Sample to extract when using ``bcf_dir``.
            max_workers: Parallel threads (see :meth:`_dispatch_parallel_str`).
            region_batch_size: Regions per thread-pool task; default 500.
            table_mode: ``None`` (default), ``"variants"`` or ``"full"``. When
                set, each result tuple carries a 5th element: a dict of numpy
                arrays describing the two haplotypes **position-synced** — the
                flat dump of the encoder's aligned state that the v10
                ``--format tsv`` output writes (``create_haplotype_dataset.py``).
                Keys: ``pos`` (int32, 1-based reference position; for an
                inserted base the anchor's position), ``offset`` (int32, 0 for
                a reference position, ``1..k`` for the k bases inserted right
                after the anchor — the same ``(pos, offset)`` convention as
                :meth:`get_sequence_df`), ``ref``/``hap0``/``hap1`` (uint8
                ASCII; ``-`` where that haplotype deletes the base, or on an
                insertion row for ``ref`` and for a haplotype lacking the
                insertion), and the bool flags ``snp0``/``snp1``/``indel0``/
                ``indel1`` (the per-hap masks; on insertion rows ``indelX`` is
                True for the haplotype carrying the base). Rows are sorted by
                ``(pos, offset)``. ``"full"`` keeps every row; ``"variants"``
                keeps only rows where ``hap0 != ref or hap1 != ref or offset >
                0`` (deleted cells count, since their hap byte is ``-``), so
                the table is proportional to the variant count. Built from the
                arrays already in hand — no second pass over the BCF.

        Returns:
            List aligned with ``regions``: ``(hap0, hap1, n_variants, n_het)``
            per region (plus the table when ``table_mode`` is set), or ``None``
            as described above.
        """
        if table_mode not in (None, "variants", "full"):
            raise ValueError(
                f"table_mode must be None, 'variants' or 'full'; got {table_mode!r}"
            )
        if self._use_direction_markers:
            raise ValueError(
                "use_direction_markers=True is incompatible with haplotype-pair "
                "output (each haplotype is a plain ACGTN^{} string; phase is "
                "carried by having two of them). Construct the processor "
                "without use_direction_markers."
            )
        if self._force_unphased:
            raise ValueError(
                "force_unphased=True is incompatible with haplotype-pair "
                "output, which requires the BCF phasing to be honoured. "
                "Construct the processor with force_unphased=False."
            )

        _sample = sample_name or ""

        def region_fn(variants, chrom, start, end):
            # Phase check on the RAW dict — before _resolve_overlapping_variants
            # can turn an unphased hom into a one-hap (0,1) with phased=False.
            _require_fully_phased(variants, chrom, start, end, _sample)
            # Counts are taken AFTER overlap resolution so they describe what
            # the strings actually carry: a het SNP sitting inside a hom
            # deletion is trimmed to nothing and must not count as a het
            # (it would make `n_het > 0` with `hap0 == hap1`). Resolution is
            # idempotent, so handing the resolved dict to
            # _prepare_haplotype_data (which resolves again) changes nothing.
            _counters: Counter = Counter()
            resolved = _resolve_overlapping_variants(
                variants, chrom, _sample, counters=_counters
            )
            with self._integrity_lock:
                self.integrity.update(_counters)
            # Count only what the strings can show: a record handed to the
            # region by REF-span overlap may change nothing inside it (a
            # deletion anchored on the region's last base). No such variant →
            # the region is reference for this sample → no row.
            counted = [
                v for p, v in resolved.items() if _variant_touches_region(p, v, start, end)
            ]
            if not counted:
                return None
            n_variants = len(counted)
            n_het = sum(1 for v in counted if v.gt[0] != v.gt[1])

            masks: Dict[str, np.ndarray] = {}
            (
                ref_arr,
                hap0,
                hap1,
                _source_is_vcf,
                _diploid_chars,
                insertion_rows,
                _is_snp,
                _is_indel_non_anchor,
                _phased_mask,
            ) = self._prepare_haplotype_data(
                resolved, chrom, start, end, _sample,
                per_hap_masks_out=masks, want_diploid=False,
            )
            ins_h0, ins_h1 = _per_hap_ins_by_idx(insertion_rows, start)
            s0 = self._build_marked_string_single_hap(
                hap0, masks["is_snp_h0"], masks["is_indel_h0"], ins_h0,
            )
            s1 = self._build_marked_string_single_hap(
                hap1, masks["is_snp_h1"], masks["is_indel_h1"], ins_h1,
            )
            if table_mode is None:
                return (s0, s1, n_variants, n_het)
            table = _haplotype_pair_table(
                ref_arr, hap0, hap1, insertion_rows, masks, start, table_mode,
            )
            return (s0, s1, n_variants, n_het, table)

        def ref_fn(chrom, start, end):
            # Variant-free region → no row. The results list is pre-filled
            # with None in region order by _dispatch_parallel_str.
            return None

        return self._dispatch_parallel_str(
            regions,
            vcf_path,
            bcf_dir,
            sample_name,
            max_workers,
            region_fn,
            region_batch_size=region_batch_size,
            ref_fn=ref_fn,
        )

    def get_reference_haploid_sequences_str(
        self,
        regions: List[Tuple[str, int, int]],
        skip_gaps: bool = True,
    ) -> List[str]:
        """Return haploid-encoded reference sequences (no variants) for regions.

        Each output is the reference ACGTN string for the region — no
        markers, no insertions. Used to generate the HG38_REF pseudo-sample
        in haploid dataset generation.
        """
        if not skip_gaps:
            raise ValueError(
                "Haploid mode requires skip_gaps=True; the output alphabet "
                "is {A,C,G,T,N,^,{,}} and does not allow '-'."
            )
        return [
            self._fast_reference_haploid_str(chrom, start, end, skip_gaps)
            for chrom, start, end in regions
        ]

    def get_reference_haploid_sequences_with_ref_idx_str(
        self,
        regions: List[Tuple[str, int, int]],
        skip_gaps: bool = True,
    ) -> List[Tuple[str, np.ndarray]]:
        """Reference-only counterpart of :meth:`get_marked_haploid_sequence_with_ref_idx_parallel_str`."""
        if not skip_gaps:
            raise ValueError(
                "Haploid mode requires skip_gaps=True; the output alphabet "
                "is {A,C,G,T,N,^,{,}} and does not allow '-'."
            )
        return [
            self._fast_reference_haploid_str_with_ref_idx(chrom, start, end, skip_gaps)
            for chrom, start, end in regions
        ]

    def _build_marked_no_ins_core(
        self,
        diploid_chars: np.ndarray,
        is_snp: np.ndarray,
        is_indel_non_anchor: np.ndarray,
        skip_gaps: bool,
        snp_marker: str,
        indel_open_sym: str,
        indel_close_sym: str,
        direction: Optional[np.ndarray],
        dir_fwd: str,
        dir_rev: str,
        *,
        want_ref_idx: bool = False,
        pos_to_var_idx: Optional[np.ndarray] = None,
    ) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
        """Segment-join build for the common no-insertion case.

        Applies the gap mask once with numpy, then iterates only over the
        sparse event positions (INDEL open/close, SNPs, direction markers)
        rather than every position in the region.  Plain segments between
        events are joined in bulk via string slicing.

        Event ordering per position matches _build_marked_string:
          } (INDEL close) → { (INDEL open) → ^ (SNP) → direction → char

        When ``want_ref_idx`` is True, also returns the ``int32``
        ``seq_ref_idx`` array that :meth:`_build_marked_string_with_ref_idx`
        would produce (marker codepoints → -2; each kept char → its ORIGINAL,
        pre-gap-mask ref offset).  When ``pos_to_var_idx`` is given (implies
        ``want_ref_idx``), additionally returns ``seq_var_idx`` matching
        :meth:`_build_marked_string_with_ref_idx_and_var_idx` (markers → -1;
        each kept char → its ``pos_to_var_idx`` value).  The string output is
        byte-identical to the original loops; the index arrays are assembled
        in lockstep with the same segment/event layout, so every token in
        ``pre_tokens`` is a marker and the per-segment ref offsets are simply
        slices of ``kept`` (recovered before masking).  Insertions are NOT
        handled here — callers fall back to the per-position loop when
        ``ins_by_idx`` is non-empty (the only source of ``ref_idx == -1``).
        """
        if pos_to_var_idx is not None:
            want_ref_idx = True
        want_var_idx = pos_to_var_idx is not None
        _EMPTY = np.empty(0, dtype=np.int32)

        # kept[p] = ORIGINAL ref offset of the p-th retained char. The
        # per-position builders store the ORIGINAL (pre-gap-mask) index in
        # seq_ref_idx, so we recover it here after masking.
        kept = None
        if skip_gaps:
            mask = diploid_chars != "-"
            if not mask.all():
                if want_ref_idx:
                    kept = np.flatnonzero(mask).astype(np.int32)
                diploid_chars = diploid_chars[mask]
                is_snp = is_snp[mask]
                is_indel_non_anchor = is_indel_non_anchor[mask]
                if direction is not None:
                    direction = direction[mask]
                if want_var_idx:
                    pos_to_var_idx = pos_to_var_idx[mask]
        if want_ref_idx and kept is None:
            kept = np.arange(len(diploid_chars), dtype=np.int32)

        n = len(diploid_chars)
        if n == 0:
            return "", (_EMPTY if want_ref_idx else None), (_EMPTY if want_var_idx else None)

        if not is_snp.any() and not is_indel_non_anchor.any():
            seq = "".join(diploid_chars.tolist())
            ref = kept if want_ref_idx else None
            var = (
                pos_to_var_idx.astype(np.int32, copy=False) if want_var_idx else None
            )
            return seq, ref, var

        # Pre-build flat string once; slice it O(1) per plain segment.
        full_str = "".join(diploid_chars.tolist())

        # pre_tokens[pos] = ordered list of tokens emitted before full_str[pos]
        pre_tokens: Dict[int, List[str]] = {}
        trailing_close = False

        if is_indel_non_anchor.any():
            padded = np.empty(n + 2, dtype=np.int8)
            padded[0] = 0
            padded[1 : n + 1] = is_indel_non_anchor.view(np.uint8)
            padded[n + 1] = 0
            diff = np.diff(padded)  # length n+1

            run_starts_arr = np.where(diff[:n] == 1)[0]
            run_closes_arr = np.where(diff == -1)[0]

            for rs, rc in zip(run_starts_arr.tolist(), run_closes_arr.tolist()):
                # Find first non-SNP in [rs, rc) — that position gets {.
                # If the entire run is covered by SNPs, no bracket is emitted
                # (SNP takes priority over INDEL open in the elif chain).
                run_is_snp = is_snp[rs:rc]
                if run_is_snp.all():
                    continue
                open_pos = rs + int(np.argmax(~run_is_snp))
                pre_tokens.setdefault(open_pos, []).insert(0, indel_open_sym)

                if rc < n:
                    pre_tokens.setdefault(rc, []).insert(0, indel_close_sym)
                else:
                    trailing_close = True

        for pos in np.where(is_snp)[0].tolist():
            tokens = pre_tokens.get(pos)
            if tokens is None:
                pre_tokens[pos] = [snp_marker]
            else:
                # } must stay first; insert ^ right after it (or at 0 if none)
                insert_at = 1 if tokens[0] == indel_close_sym else 0
                tokens.insert(insert_at, snp_marker)

        if direction is not None:
            for pos in np.where(direction != 0)[0].tolist():
                dval = int(direction[pos])
                token = dir_fwd if dval == 1 else dir_rev
                tokens = pre_tokens.get(pos)
                if tokens is not None:
                    tokens.append(token)
                elif is_indel_non_anchor[pos]:
                    # Direction within INDEL content without a prior { bracket
                    # at this specific position.
                    pre_tokens[pos] = [token]
                elif pos + 1 < n and is_indel_non_anchor[pos + 1]:
                    # Deletion anchor: the { bracket opens at the next position.
                    # The direction marker here precedes the char without { prefix.
                    pre_tokens[pos] = [token]
                else:
                    raise RuntimeError(
                        f"Bare direction marker at position {pos}: direction set "
                        "but no SNP or INDEL context present. Encoder bug."
                    )

        event_positions = sorted(pre_tokens)
        parts: List[str] = []
        # Index pieces accumulate in lockstep with `parts`; every token in
        # pre_tokens is a marker (ref=-2 / var=-1) and each char gets its
        # ref offset (kept[pos]) / variant index (pos_to_var_idx[pos]).
        ref_parts: List[np.ndarray] = [] if want_ref_idx else None
        var_parts: List[np.ndarray] = [] if want_var_idx else None
        prev = 0

        for pos in event_positions:
            if pos > prev:
                parts.append(full_str[prev:pos])
                if want_ref_idx:
                    ref_parts.append(kept[prev:pos])
                    if want_var_idx:
                        var_parts.append(pos_to_var_idx[prev:pos])
            toks = pre_tokens[pos]
            parts.extend(toks)
            if want_ref_idx:
                ref_parts.append(np.full(len(toks), -2, dtype=np.int32))
                if want_var_idx:
                    var_parts.append(np.full(len(toks), -1, dtype=np.int32))
            parts.append(full_str[pos])
            if want_ref_idx:
                ref_parts.append(kept[pos : pos + 1])
                if want_var_idx:
                    var_parts.append(pos_to_var_idx[pos : pos + 1])
            prev = pos + 1

        if prev < n:
            parts.append(full_str[prev:])
            if want_ref_idx:
                ref_parts.append(kept[prev:])
                if want_var_idx:
                    var_parts.append(pos_to_var_idx[prev:])

        if trailing_close:
            parts.append(indel_close_sym)
            if want_ref_idx:
                ref_parts.append(np.full(1, -2, dtype=np.int32))
                if want_var_idx:
                    var_parts.append(np.full(1, -1, dtype=np.int32))

        seq = "".join(parts)
        ref = (
            (np.concatenate(ref_parts).astype(np.int32, copy=False) if ref_parts else _EMPTY)
            if want_ref_idx else None
        )
        var = (
            (np.concatenate(var_parts).astype(np.int32, copy=False) if var_parts else _EMPTY)
            if want_var_idx else None
        )
        return seq, ref, var

    def _build_marked_string_no_ins(
        self,
        diploid_chars: np.ndarray,
        is_snp: np.ndarray,
        is_indel_non_anchor: np.ndarray,
        skip_gaps: bool,
        snp_marker: str,
        indel_open_sym: str,
        indel_close_sym: str,
        direction: Optional[np.ndarray],
        dir_fwd: str,
        dir_rev: str,
    ) -> str:
        """String-only no-insertion build (thin wrapper over
        :meth:`_build_marked_no_ins_core`)."""
        seq, _, _ = self._build_marked_no_ins_core(
            diploid_chars,
            is_snp,
            is_indel_non_anchor,
            skip_gaps,
            snp_marker,
            indel_open_sym,
            indel_close_sym,
            direction,
            dir_fwd,
            dir_rev,
        )
        return seq

    def _build_marked_string(
        self,
        diploid_chars: np.ndarray,
        is_snp: np.ndarray,
        is_indel_non_anchor: np.ndarray,
        ins_by_idx: Dict[int, List[Tuple]],
        skip_gaps: bool,
        direction: Optional[np.ndarray] = None,
    ) -> str:
        """Build output string with variant boundary markers.

        Iterates character-by-character over the original arrays to
        insert the SNP marker before SNPs and INDEL span markers around
        the non-anchor content of INDELs (including insertions).
        Marker symbols are read from the ``meta`` entries in the diploid
        tokens file (``<SNP>``, ``<INDEL_span_start>``, ``<INDEL_span_end>``).

        When *direction* is provided (three-level encoding), direction
        indicator characters (``‹``/``›``) are emitted before each
        phased heterozygous diploid symbol.
        """
        snp_marker = self._snp_marker
        indel_open = self._indel_open
        indel_close = self._indel_close
        dir_fwd = self._dir_fwd
        dir_rev = self._dir_rev

        # Fast path: no variant events — just join the chars (no marker insertion).
        if not is_snp.any() and not is_indel_non_anchor.any() and not ins_by_idx:
            if skip_gaps:
                return "".join(diploid_chars[diploid_chars != "-"])
            return "".join(diploid_chars)

        # Fast path for no insertions: segment-join visits only sparse event
        # positions instead of iterating all ~7 kbp per region.
        if not ins_by_idx:
            return self._build_marked_string_no_ins(
                diploid_chars,
                is_snp,
                is_indel_non_anchor,
                skip_gaps,
                snp_marker,
                indel_open,
                indel_close,
                direction,
                dir_fwd,
                dir_rev,
            )

        # Convert numpy arrays to Python lists before the inner loop.
        # Python list element access is ~3-5x faster than numpy scalar boxing
        # in a tight for-loop, giving a meaningful speedup for large regions.
        chars_list: List[str] = diploid_chars.tolist()
        is_snp_list: List[bool] = is_snp.tolist()
        is_indel_list: List[bool] = is_indel_non_anchor.tolist()
        dir_list: Optional[List[int]] = direction.tolist() if direction is not None else None

        parts: List[str] = []
        in_indel = False
        n = len(chars_list)

        if ins_by_idx:
            for i in range(n):
                ch = chars_list[i]
                is_gap = ch == "-"

                if skip_gaps and is_gap:
                    # Close INDEL bracket if this gap exits the INDEL span.
                    if in_indel and not is_indel_list[i]:
                        parts.append(indel_close)
                        in_indel = False
                    continue

                # Transitioning out of INDEL region.
                if in_indel and not is_indel_list[i]:
                    parts.append(indel_close)
                    in_indel = False

                # Emit the main character with appropriate marking.
                if is_snp_list[i]:
                    parts.append(snp_marker)
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                    parts.append(ch)
                elif is_indel_list[i]:
                    if not in_indel:
                        parts.append(indel_open)
                        in_indel = True
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                    parts.append(ch)
                else:
                    if dir_list and dir_list[i]:
                        if (i + 1 < n and is_indel_list[i + 1]) or (i in ins_by_idx):
                            # Deletion anchor (next pos is INDEL content) or
                            # insertion anchor (ins_by_idx fires at this pos).
                            parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        else:
                            raise RuntimeError(
                                f"Bare direction marker at position {i}: direction "
                                "set but no SNP or INDEL context present. Encoder bug."
                            )
                    parts.append(ch)

                # Splice in insertion symbols (always INDEL content).
                if i in ins_by_idx:
                    if not in_indel:
                        parts.append(indel_open)
                        in_indel = True
                    for ins_tup in ins_by_idx[i]:
                        ins_sym = ins_tup[1]
                        if skip_gaps and ins_sym == "-":
                            continue
                        if dir_list and len(ins_tup) > 2 and ins_tup[2]:
                            parts.append(dir_fwd if ins_tup[2] == 1 else dir_rev)
                        parts.append(ins_sym)
        else:
            # No insertions: skip the `i in ins_by_idx` dict lookup on every iteration.
            for i in range(n):
                ch = chars_list[i]
                is_gap = ch == "-"

                if skip_gaps and is_gap:
                    if in_indel and not is_indel_list[i]:
                        parts.append(indel_close)
                        in_indel = False
                    continue

                if in_indel and not is_indel_list[i]:
                    parts.append(indel_close)
                    in_indel = False

                if is_snp_list[i]:
                    parts.append(snp_marker)
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                    parts.append(ch)
                elif is_indel_list[i]:
                    if not in_indel:
                        parts.append(indel_open)
                        in_indel = True
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                    parts.append(ch)
                else:
                    if dir_list and dir_list[i]:
                        if i + 1 < n and is_indel_list[i + 1]:
                            # Deletion anchor: { opens at the next position.
                            parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        else:
                            raise RuntimeError(
                                f"Bare direction marker at position {i}: direction "
                                "set but no SNP or INDEL context present. Encoder bug."
                            )
                    parts.append(ch)

        if in_indel:
            parts.append(indel_close)

        return "".join(parts)

    def _build_marked_string_with_ref_idx(
        self,
        diploid_chars: np.ndarray,
        is_snp: np.ndarray,
        is_indel_non_anchor: np.ndarray,
        ins_by_idx: Dict[int, List[Tuple]],
        skip_gaps: bool,
        direction: Optional[np.ndarray] = None,
    ) -> Tuple[str, np.ndarray]:
        """Same as :meth:`_build_marked_string` but also returns ``seq_ref_idx``.

        ``seq_ref_idx`` is an ``int32`` array of length == len(returned string):
            ``seq_ref_idx[k] = i`` (``i >= 0``) -- codepoint from diploid_chars[i]
                                                   (ref position ``start + i``)
            ``seq_ref_idx[k] = -1``             -- inserted base (no ref pos)
            ``seq_ref_idx[k] = -2``             -- marker (^, {, }, '‹', '›')
        Used by the v7 annotation pipeline to project per-ref-position scores
        (e.g. PhyloP) onto the codepoints of ``seq``.
        """
        snp_marker = self._snp_marker
        indel_open = self._indel_open
        indel_close = self._indel_close
        dir_fwd = self._dir_fwd
        dir_rev = self._dir_rev

        # Fast path: no variant events — every kept char is a ref position;
        # no markers, no insertions.
        if not is_snp.any() and not is_indel_non_anchor.any() and not ins_by_idx:
            if skip_gaps:
                mask = diploid_chars != "-"
                s = _u1_to_str(diploid_chars[mask])
                idx = np.flatnonzero(mask).astype(np.int32)
                return s, idx
            s = _u1_to_str(diploid_chars)
            idx = np.arange(len(diploid_chars), dtype=np.int32)
            return s, idx

        # Sparse fast path: no insertions and no direction markers — iterate
        # only the sparse event positions instead of every base in the region.
        # Direction-marker runs keep the per-position loop below (proven path;
        # the encoder's three-level mode is rarer and the no-event/direction
        # interaction is subtle, so it is intentionally not fast-pathed here).
        if not ins_by_idx and direction is None:
            seq, sri, _ = self._build_marked_no_ins_core(
                diploid_chars,
                is_snp,
                is_indel_non_anchor,
                skip_gaps,
                snp_marker,
                indel_open,
                indel_close,
                None,
                dir_fwd,
                dir_rev,
                want_ref_idx=True,
            )
            return seq, sri

        chars_list: List[str] = diploid_chars.tolist()
        is_snp_list: List[bool] = is_snp.tolist()
        is_indel_list: List[bool] = is_indel_non_anchor.tolist()
        dir_list: Optional[List[int]] = (
            direction.tolist() if direction is not None else None
        )

        parts: List[str] = []
        ref_idx: List[int] = []  # parallel to parts
        in_indel = False
        n = len(chars_list)

        if ins_by_idx:
            for i in range(n):
                ch = chars_list[i]
                is_gap = ch == "-"

                if skip_gaps and is_gap:
                    if in_indel and not is_indel_list[i]:
                        parts.append(indel_close)
                        ref_idx.append(-2)
                        in_indel = False
                    continue

                if in_indel and not is_indel_list[i]:
                    parts.append(indel_close)
                    ref_idx.append(-2)
                    in_indel = False

                if is_snp_list[i]:
                    parts.append(snp_marker)
                    ref_idx.append(-2)
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                    parts.append(ch)
                    ref_idx.append(i)
                elif is_indel_list[i]:
                    if not in_indel:
                        parts.append(indel_open)
                        ref_idx.append(-2)
                        in_indel = True
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                    parts.append(ch)
                    ref_idx.append(i)
                else:
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                    parts.append(ch)
                    ref_idx.append(i)

                if i in ins_by_idx:
                    if not in_indel:
                        parts.append(indel_open)
                        ref_idx.append(-2)
                        in_indel = True
                    for ins_tup in ins_by_idx[i]:
                        ins_sym = ins_tup[1]
                        if skip_gaps and ins_sym == "-":
                            continue
                        if dir_list and len(ins_tup) > 2 and ins_tup[2]:
                            parts.append(dir_fwd if ins_tup[2] == 1 else dir_rev)
                            ref_idx.append(-2)
                        parts.append(ins_sym)
                        ref_idx.append(-1)
        else:
            for i in range(n):
                ch = chars_list[i]
                is_gap = ch == "-"

                if skip_gaps and is_gap:
                    if in_indel and not is_indel_list[i]:
                        parts.append(indel_close)
                        ref_idx.append(-2)
                        in_indel = False
                    continue

                if in_indel and not is_indel_list[i]:
                    parts.append(indel_close)
                    ref_idx.append(-2)
                    in_indel = False

                if is_snp_list[i]:
                    parts.append(snp_marker)
                    ref_idx.append(-2)
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                    parts.append(ch)
                    ref_idx.append(i)
                elif is_indel_list[i]:
                    if not in_indel:
                        parts.append(indel_open)
                        ref_idx.append(-2)
                        in_indel = True
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                    parts.append(ch)
                    ref_idx.append(i)
                else:
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                    parts.append(ch)
                    ref_idx.append(i)

        if in_indel:
            parts.append(indel_close)
            ref_idx.append(-2)

        return "".join(parts), np.asarray(ref_idx, dtype=np.int32)

    def _build_marked_string_with_ref_idx_and_var_idx(
        self,
        diploid_chars: np.ndarray,
        is_snp: np.ndarray,
        is_indel_non_anchor: np.ndarray,
        ins_by_idx: Dict[int, List[Tuple]],
        skip_gaps: bool,
        direction: Optional[np.ndarray],
        pos_to_var_idx: np.ndarray,
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """Like :meth:`_build_marked_string_with_ref_idx` but also returns
        ``seq_var_idx``.

        ``pos_to_var_idx`` is an ``int32`` array of length ``end - start``
        mapping each ref offset within the region to a variant index in the
        per-region ``variants`` list (or ``-1`` for ref positions not
        covered by any variant).

        Returns ``(seq, seq_ref_idx, seq_var_idx)``. ``seq_var_idx`` is an
        ``int32`` array of length ``len(seq)`` with the same shape semantics
        as ``seq_ref_idx``:
            ``seq_var_idx[k] = j`` (``j >= 0``) -- codepoint k belongs to
                                                   ``variants[j]``
            ``seq_var_idx[k] = -1``             -- codepoint is pure ref or
                                                   a marker (no variant)
        Inserted-base slots inherit ``pos_to_var_idx`` from the anchor ref
        offset they were spliced after, so the CADD anchor lookup
        (``seq_var_idx == j AND seq_ref_idx == anchor_offset``) reliably
        picks the single anchor codepoint per variant.
        """
        snp_marker = self._snp_marker
        indel_open = self._indel_open
        indel_close = self._indel_close
        dir_fwd = self._dir_fwd
        dir_rev = self._dir_rev

        # Sparse fast path: no insertions and no direction markers — iterate
        # only the sparse event positions instead of every base. Handles the
        # no-event case too, so it sits above the pos_to_var_idx.tolist()
        # materialization (only the per-position fallback below needs it).
        # Direction-marker runs keep the per-position loop (see the sibling
        # _build_marked_string_with_ref_idx for the rationale).
        if not ins_by_idx and direction is None:
            return self._build_marked_no_ins_core(
                diploid_chars,
                is_snp,
                is_indel_non_anchor,
                skip_gaps,
                snp_marker,
                indel_open,
                indel_close,
                None,
                dir_fwd,
                dir_rev,
                want_ref_idx=True,
                pos_to_var_idx=pos_to_var_idx,
            )

        pos_to_var_list: List[int] = pos_to_var_idx.tolist()

        # Fast path: no variant events — every kept char is a ref position;
        # no markers, no insertions. (pos_to_var_idx is all -1 in this case.)
        if not is_snp.any() and not is_indel_non_anchor.any() and not ins_by_idx:
            if skip_gaps:
                mask = diploid_chars != "-"
                s = _u1_to_str(diploid_chars[mask])
                kept = np.flatnonzero(mask).astype(np.int32)
                # var_idx defaults to -1; with no variants, pos_to_var is all -1.
                var = np.asarray(
                    [pos_to_var_list[i] for i in kept.tolist()], dtype=np.int32,
                )
                return s, kept, var
            s = _u1_to_str(diploid_chars)
            ref = np.arange(len(diploid_chars), dtype=np.int32)
            var = np.asarray(pos_to_var_list, dtype=np.int32)
            return s, ref, var

        chars_list: List[str] = diploid_chars.tolist()
        is_snp_list: List[bool] = is_snp.tolist()
        is_indel_list: List[bool] = is_indel_non_anchor.tolist()
        dir_list: Optional[List[int]] = (
            direction.tolist() if direction is not None else None
        )

        parts: List[str] = []
        ref_idx: List[int] = []
        var_idx: List[int] = []
        in_indel = False
        n = len(chars_list)

        if ins_by_idx:
            for i in range(n):
                ch = chars_list[i]
                is_gap = ch == "-"
                vi = pos_to_var_list[i]

                if skip_gaps and is_gap:
                    if in_indel and not is_indel_list[i]:
                        parts.append(indel_close)
                        ref_idx.append(-2)
                        var_idx.append(-1)
                        in_indel = False
                    continue

                if in_indel and not is_indel_list[i]:
                    parts.append(indel_close)
                    ref_idx.append(-2)
                    var_idx.append(-1)
                    in_indel = False

                if is_snp_list[i]:
                    parts.append(snp_marker)
                    ref_idx.append(-2)
                    var_idx.append(-1)
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                        var_idx.append(-1)
                    parts.append(ch)
                    ref_idx.append(i)
                    var_idx.append(vi)
                elif is_indel_list[i]:
                    if not in_indel:
                        parts.append(indel_open)
                        ref_idx.append(-2)
                        var_idx.append(-1)
                        in_indel = True
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                        var_idx.append(-1)
                    parts.append(ch)
                    ref_idx.append(i)
                    var_idx.append(vi)
                else:
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                        var_idx.append(-1)
                    parts.append(ch)
                    ref_idx.append(i)
                    var_idx.append(vi)

                if i in ins_by_idx:
                    if not in_indel:
                        parts.append(indel_open)
                        ref_idx.append(-2)
                        var_idx.append(-1)
                        in_indel = True
                    for ins_tup in ins_by_idx[i]:
                        ins_sym = ins_tup[1]
                        if skip_gaps and ins_sym == "-":
                            continue
                        if dir_list and len(ins_tup) > 2 and ins_tup[2]:
                            parts.append(dir_fwd if ins_tup[2] == 1 else dir_rev)
                            ref_idx.append(-2)
                            var_idx.append(-1)
                        parts.append(ins_sym)
                        ref_idx.append(-1)
                        # Inserted bases belong to the variant whose anchor
                        # they were attached to (= the current ref slot i).
                        var_idx.append(vi)
        else:
            for i in range(n):
                ch = chars_list[i]
                is_gap = ch == "-"
                vi = pos_to_var_list[i]

                if skip_gaps and is_gap:
                    if in_indel and not is_indel_list[i]:
                        parts.append(indel_close)
                        ref_idx.append(-2)
                        var_idx.append(-1)
                        in_indel = False
                    continue

                if in_indel and not is_indel_list[i]:
                    parts.append(indel_close)
                    ref_idx.append(-2)
                    var_idx.append(-1)
                    in_indel = False

                if is_snp_list[i]:
                    parts.append(snp_marker)
                    ref_idx.append(-2)
                    var_idx.append(-1)
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                        var_idx.append(-1)
                    parts.append(ch)
                    ref_idx.append(i)
                    var_idx.append(vi)
                elif is_indel_list[i]:
                    if not in_indel:
                        parts.append(indel_open)
                        ref_idx.append(-2)
                        var_idx.append(-1)
                        in_indel = True
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                        var_idx.append(-1)
                    parts.append(ch)
                    ref_idx.append(i)
                    var_idx.append(vi)
                else:
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd if dir_list[i] == 1 else dir_rev)
                        ref_idx.append(-2)
                        var_idx.append(-1)
                    parts.append(ch)
                    ref_idx.append(i)
                    var_idx.append(vi)

        if in_indel:
            parts.append(indel_close)
            ref_idx.append(-2)
            var_idx.append(-1)

        return (
            "".join(parts),
            np.asarray(ref_idx, dtype=np.int32),
            np.asarray(var_idx, dtype=np.int32),
        )

    def _build_marked_string_haploid(
        self,
        collapsed_chars: np.ndarray,
        is_snp: np.ndarray,
        is_indel_non_anchor: np.ndarray,
        ins_by_idx: Dict[int, List[Tuple[int, str]]],
        *,
        block_left_truncated_at_zero: bool = False,
        block_right_truncated_at_end: bool = False,
    ) -> str:
        """Build the marked haploid output string.

        Identical bracketing/marker semantics to :meth:`_build_marked_string`
        (the ``^`` SNP marker, the ``{ ... }`` INDEL span markers, the P5
        left/right truncation handling), but operates on the per-cell ACGTN
        ``collapsed_chars`` array produced by :func:`_collapse_haploid`
        instead of the 256x256 diploid codepoints. No direction markers are
        emitted (haploid output destroys phase by design).

        ``skip_gaps`` is implicitly True: any '-' in ``collapsed_chars`` is
        dropped from the output. Every emitted character is asserted to be
        in the haploid alphabet ``{A,C,G,T,N,^,{,}}``; an unexpected byte
        raises ``RuntimeError`` rather than landing silently in the dataset.
        """
        snp_marker = self._snp_marker
        indel_open = self._indel_open
        indel_close = self._indel_close

        # Defensive: the marker symbols must be in the allowed alphabet so the
        # post-hoc check passes. The diploid token file is free to redefine
        # them; haploid requires single ACGTN-or-marker chars.
        for label, sym in (
            ("<SNP>", snp_marker),
            ("<INDEL_span_start>", indel_open),
            ("<INDEL_span_end>", indel_close),
        ):
            if sym not in _HAPLOID_ALLOWED:
                raise RuntimeError(
                    f"Meta token {label}={sym!r} is not in the haploid "
                    f"alphabet {sorted(_HAPLOID_ALLOWED)}. Use the default "
                    f"markers or supply a tokens file with single-char ASCII "
                    f"meta tokens."
                )

        chars_list: List[str] = collapsed_chars.tolist()
        is_snp_list: List[bool] = is_snp.tolist()
        is_indel_list: List[bool] = is_indel_non_anchor.tolist()

        parts: List[str] = []
        in_indel = False
        n = len(chars_list)

        for i in range(n):
            ch = chars_list[i]
            is_gap = ch == "-"

            if is_gap:
                if in_indel and not is_indel_list[i]:
                    parts.append(indel_close)
                    in_indel = False
                continue

            if in_indel and not is_indel_list[i]:
                parts.append(indel_close)
                in_indel = False

            if not in_indel and is_indel_list[i]:
                if not (i == 0 and block_left_truncated_at_zero):
                    parts.append(indel_open)
                in_indel = True

            if is_snp_list[i]:
                parts.append(snp_marker)

            parts.append(ch)

            if i in ins_by_idx:
                if not in_indel:
                    parts.append(indel_open)
                    in_indel = True
                for _offset, ins_sym in ins_by_idx[i]:
                    if ins_sym == "-":
                        continue
                    parts.append(ins_sym)

        if in_indel and not block_right_truncated_at_end:
            parts.append(indel_close)

        result = "".join(parts)

        # Alphabet guard: no silent garbage.
        for ch in result:
            if ch not in _HAPLOID_ALLOWED:
                raise RuntimeError(
                    f"Haploid output contains disallowed character {ch!r} "
                    f"(code {ord(ch)}). Allowed alphabet: "
                    f"{sorted(_HAPLOID_ALLOWED)}."
                )

        return result

    def _build_marked_string_haploid_with_ref_idx(
        self,
        collapsed_chars: np.ndarray,
        is_snp: np.ndarray,
        is_indel_non_anchor: np.ndarray,
        ins_by_idx: Dict[int, List[Tuple[int, str]]],
        *,
        block_left_truncated_at_zero: bool = False,
        block_right_truncated_at_end: bool = False,
    ) -> Tuple[str, np.ndarray]:
        """Same as :meth:`_build_marked_string_haploid` but also returns ``seq_ref_idx``.

        ``seq_ref_idx`` is parallel to the returned string codepoints:
            >= 0 -- ref offset within the region
            -1   -- inserted base
            -2   -- marker codepoint (^, {, })
        """
        snp_marker = self._snp_marker
        indel_open = self._indel_open
        indel_close = self._indel_close

        for label, sym in (
            ("<SNP>", snp_marker),
            ("<INDEL_span_start>", indel_open),
            ("<INDEL_span_end>", indel_close),
        ):
            if sym not in _HAPLOID_ALLOWED:
                raise RuntimeError(
                    f"Meta token {label}={sym!r} is not in the haploid "
                    f"alphabet {sorted(_HAPLOID_ALLOWED)}. Use the default "
                    f"markers or supply a tokens file with single-char ASCII "
                    f"meta tokens."
                )

        chars_list: List[str] = collapsed_chars.tolist()
        is_snp_list: List[bool] = is_snp.tolist()
        is_indel_list: List[bool] = is_indel_non_anchor.tolist()

        parts: List[str] = []
        ref_idx: List[int] = []
        in_indel = False
        n = len(chars_list)

        for i in range(n):
            ch = chars_list[i]
            is_gap = ch == "-"

            if is_gap:
                if in_indel and not is_indel_list[i]:
                    parts.append(indel_close)
                    ref_idx.append(-2)
                    in_indel = False
                continue

            if in_indel and not is_indel_list[i]:
                parts.append(indel_close)
                ref_idx.append(-2)
                in_indel = False

            if not in_indel and is_indel_list[i]:
                if not (i == 0 and block_left_truncated_at_zero):
                    parts.append(indel_open)
                    ref_idx.append(-2)
                in_indel = True

            if is_snp_list[i]:
                parts.append(snp_marker)
                ref_idx.append(-2)

            parts.append(ch)
            ref_idx.append(i)

            if i in ins_by_idx:
                if not in_indel:
                    parts.append(indel_open)
                    ref_idx.append(-2)
                    in_indel = True
                for _offset, ins_sym in ins_by_idx[i]:
                    if ins_sym == "-":
                        continue
                    parts.append(ins_sym)
                    ref_idx.append(-1)

        if in_indel and not block_right_truncated_at_end:
            parts.append(indel_close)
            ref_idx.append(-2)

        result = "".join(parts)

        for ch in result:
            if ch not in _HAPLOID_ALLOWED:
                raise RuntimeError(
                    f"Haploid output contains disallowed character {ch!r} "
                    f"(code {ord(ch)}). Allowed alphabet: "
                    f"{sorted(_HAPLOID_ALLOWED)}."
                )

        return result, np.asarray(ref_idx, dtype=np.int32)

    def _build_marked_string_single_hap(
        self,
        hap_chars: np.ndarray,
        is_snp_h: np.ndarray,
        is_indel_h: np.ndarray,
        ins_by_idx_h: Dict[int, List[Tuple[int, str]]],
    ) -> str:
        """Build one haplotype's marked string for the v10 haplotype-pair output.

        Grammar (the v4–v8 haploid grammar with **one change**: a gap cell —
        ``'-'``, a base deleted on this haplotype — *opens* an INDEL span when
        ``is_indel_h`` is set there, so a carried deletion renders as ``A{}T``
        instead of silently dropping its bases):

        * the string is "in a span" after cell ``i`` iff ``is_indel_h[i]`` or
          cell ``i`` carries an insertion (``i in ins_by_idx_h``);
        * before cell ``i`` a ``}`` closes a span the cell is not in, a ``{``
          opens one it is in (gap cells included);
        * a kept cell emits ``^`` when ``is_snp_h[i]``, then its base, then —
          if it carries an insertion — ``{`` unless already in a span and the
          inserted bases;
        * a trailing ``}`` closes a span still open after the last cell.

        The per-base loop that used to implement this (92 % of v10 encode
        time) survives as the oracle in
        ``test_get_genome_sequence.TestBuildMarkedStringSingleHap``; this
        version computes the event cells with numpy and only loops over them.

        ``hap_chars`` may be the uint8 byte array straight from
        :meth:`_prepare_haplotype_data` or an already-converted ``U1`` array.
        Gaps are always dropped from the output. Every emitted character is
        asserted to be in ``_HAPLOID_ALLOWED``; anything else raises
        ``RuntimeError`` rather than landing silently in a dataset.
        """
        snp_marker = self._snp_marker
        indel_open = self._indel_open
        indel_close = self._indel_close

        for label, sym in (
            ("<SNP>", snp_marker),
            ("<INDEL_span_start>", indel_open),
            ("<INDEL_span_end>", indel_close),
        ):
            if sym not in _HAPLOID_ALLOWED:
                raise RuntimeError(
                    f"Meta token {label}={sym!r} is not in the haploid "
                    f"alphabet {sorted(_HAPLOID_ALLOWED)}. Use the default "
                    f"markers or supply a tokens file with single-char ASCII "
                    f"meta tokens."
                )

        hap = np.asarray(hap_chars)
        if hap.dtype != np.uint8:
            hap = hap.astype("S1").view(np.uint8)
        is_snp_h = np.asarray(is_snp_h, dtype=bool)
        is_indel_h = np.asarray(is_indel_h, dtype=bool)
        n = len(hap)
        if len(is_snp_h) != n or len(is_indel_h) != n:
            raise ValueError(
                f"Mask length mismatch: hap={n}, is_snp_h={len(is_snp_h)}, "
                f"is_indel_h={len(is_indel_h)}."
            )
        if n == 0:
            return ""

        keep = hap != ord("-")
        has_ins = np.zeros(n, dtype=bool)
        if ins_by_idx_h:
            anchors = np.fromiter(ins_by_idx_h, dtype=np.intp, count=len(ins_by_idx_h))
            has_ins[anchors[keep[anchors]]] = True  # a gap cell emits nothing
        in_span_after = is_indel_h | has_ins
        in_span_before = np.empty(n, dtype=bool)
        in_span_before[0] = False
        in_span_before[1:] = in_span_after[:-1]
        close_at = in_span_before & ~is_indel_h
        open_at = ~in_span_before & is_indel_h
        snp_at = is_snp_h & keep

        full = hap[keep].tobytes().decode("ascii")
        before = np.cumsum(keep) - keep  # kept cells strictly before cell i

        events = np.flatnonzero(close_at | open_at | snp_at | has_ins)
        parts: List[str] = []
        cur = 0
        for i, p, cl, op, sn, kp, ins, ind in zip(
            events.tolist(), before[events].tolist(), close_at[events].tolist(),
            open_at[events].tolist(), snp_at[events].tolist(), keep[events].tolist(),
            has_ins[events].tolist(), is_indel_h[events].tolist(),
        ):
            parts.append(full[cur:p])
            cur = p
            if cl:
                parts.append(indel_close)
            if op:
                parts.append(indel_open)
            if kp:
                if sn:
                    parts.append(snp_marker)
                parts.append(full[p])
                cur = p + 1
                if ins:
                    if not ind:
                        parts.append(indel_open)
                    parts.extend(sym for _offset, sym in ins_by_idx_h[i] if sym != "-")
        parts.append(full[cur:])
        if in_span_after[n - 1]:
            parts.append(indel_close)

        result = "".join(parts)
        if not set(result) <= _HAPLOID_ALLOWED:
            bad = sorted(set(result) - _HAPLOID_ALLOWED)
            raise RuntimeError(
                f"Haplotype output contains disallowed character(s) {bad!r}. "
                f"Allowed alphabet: {sorted(_HAPLOID_ALLOWED)}."
            )
        return result

    def _build_marked_uint32(
        self,
        diploid_chars: np.ndarray,
        is_snp: np.ndarray,
        is_indel_non_anchor: np.ndarray,
        ins_by_idx: Dict[int, List[Tuple]],
        skip_gaps: bool,
        direction: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build uint32 codepoint array with variant boundary markers.

        Same logic as :meth:`_build_marked_string` but produces a numpy
        uint32 array directly, avoiding a str→uint32 round-trip when
        codepoints are the desired output (e.g. for dataset generation).
        """
        snp_marker = ord(self._snp_marker)
        indel_open = ord(self._indel_open)
        indel_close = ord(self._indel_close)
        dir_fwd_cp = ord(self._dir_fwd)
        dir_rev_cp = ord(self._dir_rev)

        # Fast path: no variant events — just emit codepoints directly.
        if not is_snp.any() and not is_indel_non_anchor.any() and not ins_by_idx:
            if skip_gaps:
                return np.array(
                    [ord(c) for c in diploid_chars if c != "-"], dtype=np.uint32
                )
            return np.array([ord(c) for c in diploid_chars], dtype=np.uint32)

        # Convert numpy arrays to Python lists for faster element access
        # in the tight loop (avoids numpy scalar boxing overhead).
        chars_list: List[str] = diploid_chars.tolist()
        is_snp_list: List[bool] = is_snp.tolist()
        is_indel_list: List[bool] = is_indel_non_anchor.tolist()
        dir_list: Optional[List[int]] = direction.tolist() if direction is not None else None

        parts: List[int] = []
        in_indel = False
        n = len(chars_list)

        if ins_by_idx:
            for i in range(n):
                ch = chars_list[i]
                is_gap = ch == "-"

                if skip_gaps and is_gap:
                    if in_indel and not is_indel_list[i]:
                        parts.append(indel_close)
                        in_indel = False
                    continue

                if in_indel and not is_indel_list[i]:
                    parts.append(indel_close)
                    in_indel = False

                if is_snp_list[i]:
                    parts.append(snp_marker)
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd_cp if dir_list[i] == 1 else dir_rev_cp)
                    parts.append(ord(ch))
                elif is_indel_list[i]:
                    if not in_indel:
                        parts.append(indel_open)
                        in_indel = True
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd_cp if dir_list[i] == 1 else dir_rev_cp)
                    parts.append(ord(ch))
                else:
                    if dir_list and dir_list[i]:
                        if (i + 1 < n and is_indel_list[i + 1]) or (i in ins_by_idx):
                            parts.append(dir_fwd_cp if dir_list[i] == 1 else dir_rev_cp)
                        else:
                            raise RuntimeError(
                                f"Bare direction marker at position {i}: direction "
                                "set but no SNP or INDEL context present. Encoder bug."
                            )
                    parts.append(ord(ch))

                if i in ins_by_idx:
                    if not in_indel:
                        parts.append(indel_open)
                        in_indel = True
                    for ins_tup in ins_by_idx[i]:
                        ins_sym = ins_tup[1]
                        if skip_gaps and ins_sym == "-":
                            continue
                        if dir_list and len(ins_tup) > 2 and ins_tup[2]:
                            parts.append(dir_fwd_cp if ins_tup[2] == 1 else dir_rev_cp)
                        parts.append(ord(ins_sym))
        else:
            # No insertions: skip the `i in ins_by_idx` dict lookup on every iteration.
            for i in range(n):
                ch = chars_list[i]
                is_gap = ch == "-"

                if skip_gaps and is_gap:
                    if in_indel and not is_indel_list[i]:
                        parts.append(indel_close)
                        in_indel = False
                    continue

                if in_indel and not is_indel_list[i]:
                    parts.append(indel_close)
                    in_indel = False

                if is_snp_list[i]:
                    parts.append(snp_marker)
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd_cp if dir_list[i] == 1 else dir_rev_cp)
                    parts.append(ord(ch))
                elif is_indel_list[i]:
                    if not in_indel:
                        parts.append(indel_open)
                        in_indel = True
                    if dir_list and dir_list[i]:
                        parts.append(dir_fwd_cp if dir_list[i] == 1 else dir_rev_cp)
                    parts.append(ord(ch))
                else:
                    if dir_list and dir_list[i]:
                        if i + 1 < n and is_indel_list[i + 1]:
                            parts.append(dir_fwd_cp if dir_list[i] == 1 else dir_rev_cp)
                        else:
                            raise RuntimeError(
                                f"Bare direction marker at position {i}: direction "
                                "set but no SNP or INDEL context present. Encoder bug."
                            )
                    parts.append(ord(ch))

        if in_indel:
            parts.append(indel_close)

        return np.array(parts, dtype=np.uint32)

    def _sequence_marked_uint32_from_variants(
        self,
        variants: Dict[int, "Variant"],
        chrom: str,
        start: int,
        end: int,
        sample_name: str,
        skip_gaps: bool,
    ) -> np.ndarray:
        """Build marked uint32 codepoint array from parsed variants.

        Like :meth:`_sequence_str_from_variants` with
        ``mark_variant_boundaries=True``, but returns a uint32 numpy
        array instead of a string.
        """
        (
            _ref_seq,
            hap0,
            hap1,
            _source_is_vcf,
            diploid_chars,
            insertion_rows,
            is_snp,
            is_indel_non_anchor,
            phased_mask,
        ) = self._prepare_haplotype_data(
            variants,
            chrom,
            start,
            end,
            sample_name,
        )

        _use_dir = self._use_direction_markers and not self._force_unphased

        # Resolve insertion symbols and group by 0-based array index.
        ins_by_idx: Dict[int, List[Tuple]] = {}
        for pos_1based, offset, h0_byte, h1_byte, is_phased in insertion_rows:
            pair = chr(h0_byte) + chr(h1_byte)
            if _use_dir:
                lookup = self._unphased_lookup
            elif self._force_unphased:
                lookup = self._unphased_lookup
            else:
                lookup = self._phased_lookup if is_phased else self._unphased_lookup
            sym = lookup.get(pair, UNKNOWN_DIPLOID_SYMBOL)
            if skip_gaps and sym == "-":
                continue
            arr_idx = pos_1based - start - 1
            if _use_dir:
                dir_val = _compute_insertion_direction(h0_byte, h1_byte, is_phased)
                ins_by_idx.setdefault(arr_idx, []).append((offset, sym, dir_val))
            else:
                ins_by_idx.setdefault(arr_idx, []).append((offset, sym))

        for syms in ins_by_idx.values():
            syms.sort()

        direction = None
        if _use_dir:
            direction = _compute_direction_array(hap0, hap1, phased_mask)

        return self._build_marked_uint32(
            diploid_chars,
            is_snp,
            is_indel_non_anchor,
            ins_by_idx,
            skip_gaps,
            direction,
        )

    def _load_bed_regions(
        self,
        bed_path: Path,
    ) -> List[Tuple[str, Optional[int], Optional[int]]]:
        """Parse a BED file into a list of (chrom, start, end) tuples.

        Args:
            bed_path: Path to BED file (0-based, half-open coordinates).

        Returns:
            List of (chrom, start, end) tuples.

        Raises:
            FileNotFoundError: If BED file not found.
            ValueError: If BED file is empty or malformed.
        """
        bed_path = Path(bed_path)
        if not bed_path.exists():
            raise FileNotFoundError(f"BED file not found: '{bed_path}'")

        regions: List[Tuple[str, Optional[int], Optional[int]]] = []

        for line_num, _line, fields in bedio.iter_bed_fields(bed_path, split="tab"):
            if len(fields) < 3:
                raise ValueError(
                    f"BED file '{bed_path}' line {line_num}: expected at least 3 tab-separated "
                    f"columns (chrom, start, end), got {len(fields)}"
                )

            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as e:
                raise ValueError(
                    f"BED file '{bed_path}' line {line_num}: invalid coordinates - {e}"
                ) from e

            regions.append((chrom, start, end))

        if not regions:
            raise ValueError(f"BED file '{bed_path}' contains no valid regions")

        logger.info("Loaded %d regions from BED file '%s'", len(regions), bed_path)
        return regions

    def get_sequences_df_from_bed(
        self,
        vcf_path: Optional[Path] = None,
        bed_path: Optional[Path] = None,
        max_workers: Optional[int] = None,
        parallel: bool = True,
        debug: bool = False,
        bcf_dir: Optional[Path] = None,
        sample_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Build per-position haplotype DataFrame for regions specified in a BED file.

        Supports two modes:
        1. Single-sample VCF: Provide vcf_path (must contain exactly one sample)
        2. Multi-sample BCF: Provide bcf_dir and sample_name

        Args:
            vcf_path: Path to VCF/BCF file (single-sample). Mutually exclusive
                      with bcf_dir/sample_name.
            bed_path: Path to BED file (0-based, half-open coordinates).
            max_workers: Maximum number of parallel threads (default: min(25, cpu_count)).
            parallel: If True (default), process regions concurrently.
            debug: If True, include 'source' column in output.
            bcf_dir: Directory containing per-chromosome BCF files.
                     Mutually exclusive with vcf_path. Requires sample_name.
            sample_name: Sample name to extract from multi-sample BCF.
                         Required when using bcf_dir.

        Returns:
            DataFrame with columns: chrom, pos, offset, hap0, hap1, diploid_symbol,
            and optionally source.

        Raises:
            FileNotFoundError: If BED file or VCF/BCF not found.
            ValueError: If BED file is empty/malformed or parameter validation fails.
            RuntimeError: If any region processing fails.
        """
        if vcf_path is not None and (bcf_dir is not None or sample_name is not None):
            raise ValueError("vcf_path is mutually exclusive with bcf_dir/sample_name.")
        if vcf_path is None and (bcf_dir is None or sample_name is None):
            raise ValueError(
                "Must provide either vcf_path or both bcf_dir and sample_name."
            )

        regions = self._load_bed_regions(bed_path)

        if bcf_dir is not None:
            return self.get_sequences_df_multisample(
                bcf_dir=bcf_dir,
                sample_name=sample_name,
                regions=regions,
                max_workers=max_workers,
                parallel=parallel,
                debug=debug,
            )

        return self.get_sequences_df(
            vcf_path=vcf_path,
            regions=regions,
            max_workers=max_workers,
            parallel=parallel,
            debug=debug,
        )

    def _get_or_open_vcf(
        self,
        vcf_path: Path,
        sample_name: Optional[str] = None,
    ) -> VCF:
        """Return a cached VCF/BCF reader, opening and validating if needed.

        The reader is replaced (and the previous one closed) whenever
        *vcf_path* or *sample_name* changes.  Passing ``sample_name`` puts
        the reader into multi-sample BCF mode (calls ``set_samples``).

        Args:
            vcf_path: Path to indexed VCF/BCF file.
            sample_name: When provided, assume a multi-sample BCF and restrict
                decoding to this sample via ``set_samples([sample_name])``.
                When ``None``, assume a single-sample VCF and validate that
                the file has exactly one sample.

        Returns:
            An open ``cyvcf2.VCF`` reader ready for region queries.
        """
        vcf_path = Path(vcf_path)
        if (
            self._cached_vcf_reader is not None
            and self._cached_vcf_path == vcf_path
            and self._cached_vcf_sample == sample_name
        ):
            return self._cached_vcf_reader

        # Close previous reader before opening a new one
        if self._cached_vcf_reader is not None:
            self._cached_vcf_reader.close()
            self._cached_vcf_reader = None
            self._cached_vcf_path = None
            self._cached_vcf_sample = None

        vcf = VCF(str(vcf_path))
        try:
            if sample_name is None:
                # Single-sample VCF: must contain exactly one sample
                if len(vcf.samples) == 0:
                    raise ValueError(
                        f"VCF file '{vcf_path}' has no samples. "
                        f"A VCF with exactly one sample is required."
                    )
                elif len(vcf.samples) > 1:
                    raise ValueError(
                        f"VCF file '{vcf_path}' contains {len(vcf.samples)} samples: "
                        f"{vcf.samples}. A VCF with exactly one sample is required."
                    )
            else:
                # Multi-sample BCF: validate sample exists, restrict decoding
                if sample_name not in vcf.samples:
                    raise ValueError(
                        f"Sample '{sample_name}' not found in BCF '{vcf_path}'; "
                        f"samples={vcf.samples}"
                    )
                vcf.set_samples([sample_name])

            # Validate chromosome naming consistency between VCF and reference.
            if self._ref_file is not None:
                vcf_contigs = set(vcf.seqnames) if vcf.seqnames else set()
                ref_contigs = set(self._ref_file.chromosome_names())
                if vcf_contigs and ref_contigs and not vcf_contigs & ref_contigs:
                    # Check for systematic chr prefix mismatch
                    stripped_vcf = {c.replace("chr", "") for c in vcf_contigs}
                    stripped_ref = {c.replace("chr", "") for c in ref_contigs}
                    if stripped_vcf & stripped_ref:
                        raise ValueError(
                            f"Chromosome name mismatch between VCF and reference: "
                            f"VCF uses {sorted(vcf_contigs)[:3]}... "
                            f"but reference uses {sorted(ref_contigs)[:3]}... "
                            f"(chr prefix difference). Normalize names before "
                            f"processing."
                        )
        except Exception:
            vcf.close()
            raise

        self._cached_vcf_reader = vcf
        self._cached_vcf_path = vcf_path
        self._cached_vcf_sample = sample_name
        return vcf

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close resources."""
        with self._vcf_lock:
            if self._cached_vcf_reader is not None:
                self._cached_vcf_reader.close()
                self._cached_vcf_reader = None
                self._cached_vcf_path = None
                self._cached_vcf_sample = None
        if self._ref_file:
            self._ref_file.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        """Explicitly close resources."""
        self.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Binary save / load for uint32 sequence + variant-prefix arrays
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Command-line interface for GenomeSequenceProcessor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract haplotype sequences from reference + VCF/BCF"
    )
    parser.add_argument(
        "--reference",
        "-r",
        help="Reference genome file (FASTA or 2bit, default: ./hg38.2bit)",
    )

    # VCF/BCF specification: either --vcf or --vcf-dir (mutually exclusive, one required)
    vcf_group = parser.add_mutually_exclusive_group(required=True)
    vcf_group.add_argument(
        "--vcf",
        "-v",
        help="VCF/BCF file (must be indexed and contain exactly one sample). "
        "Mutually exclusive with --vcf-dir.",
    )
    vcf_group.add_argument(
        "--vcf-dir",
        help="Directory containing per-chromosome BCF files (multi-sample). "
        "Requires --sample. Mutually exclusive with --vcf.",
    )

    parser.add_argument(
        "--sample",
        "-s",
        help="Sample name to extract from multi-sample BCF. Required with --vcf-dir.",
    )

    # Region specification: either --region or --bed (mutually exclusive, one required)
    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument(
        "--region",
        help="Region as chrom[:start-end] (BED coordinates: 0-based, half-open). "
        "If start-end omitted, entire chromosome is used.",
    )
    region_group.add_argument(
        "--bed",
        "-b",
        help="BED file with regions to process (0-based, half-open coordinates). "
        "Mutually exclusive with --region.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Include source column in output",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing for BED regions.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: min(25, cpu_count)). Only applies with --bed.",
    )
    parser.add_argument("--output", "-o", help="Output TSV file (default: stdout)")
    parser.add_argument(
        "--diploid-tokens",
        "-t",
        help="Path to diploid tokens TSV file (default: ./diploid_tokens.tsv)",
    )

    args = parser.parse_args()

    # Validate --sample is provided when using --vcf-dir
    if args.vcf_dir and not args.sample:
        parser.error("--sample is required when using --vcf-dir")
    if args.sample and not args.vcf_dir:
        parser.error("--sample can only be used with --vcf-dir")

    # Determine parallel setting
    parallel = not args.no_parallel

    # Use new class-based API
    ref_path = Path(args.reference) if args.reference else None
    tokens_path = Path(args.diploid_tokens) if args.diploid_tokens else None

    with GenomeSequenceProcessor(
        ref_path=ref_path,
        diploid_tokens_path=tokens_path,
    ) as processor:
        if args.bed:
            # Process regions from BED file
            if args.vcf_dir:
                df = processor.get_sequences_df_from_bed(
                    bed_path=Path(args.bed),
                    bcf_dir=Path(args.vcf_dir),
                    sample_name=args.sample,
                    max_workers=args.max_workers,
                    parallel=parallel,
                    debug=args.debug,
                )
            else:
                df = processor.get_sequences_df_from_bed(
                    vcf_path=Path(args.vcf),
                    bed_path=Path(args.bed),
                    max_workers=args.max_workers,
                    parallel=parallel,
                    debug=args.debug,
                )
        else:
            # Parse single region
            start = None
            end = None
            try:
                if ":" in args.region:
                    chrom, coords = args.region.split(":")
                    if coords:  # coords provided
                        start, end = map(int, coords.split("-"))
                else:
                    chrom = args.region
            except ValueError:
                parser.error(
                    "Region must be in format chrom[:start-end] (e.g., chr1:1000-2000 or chr1)"
                )

            # Call appropriate method based on VCF mode
            if args.vcf_dir:
                df = processor.get_sequence_df(
                    bcf_dir=Path(args.vcf_dir),
                    sample_name=args.sample,
                    chrom=chrom,
                    start=start,
                    end=end,
                    debug=args.debug,
                )
            else:
                df = processor.get_sequence_df(
                    vcf_path=Path(args.vcf),
                    chrom=chrom,
                    start=start,
                    end=end,
                    debug=args.debug,
                )

    if args.output:
        df.to_csv(args.output, sep="\t", index=False)
        logger.info("Wrote %d rows to %s", len(df), args.output)
    else:
        print(df.to_csv(sep="\t", index=False), end="")


if __name__ == "__main__":
    main()
