#!/usr/bin/env python3
"""Position-level verification of diploid-encoded V5 parquet output vs source BCF.

The encoder (preprocess.huggingface_dataset.create_dataset) writes per-region
parquet rows with columns ``sample_id, chrom, start, end, seq`` where ``seq``
is a diploid-encoded Unicode string with SNP (``^``), INDEL (``{...}``) and
optional direction (``‹``/``›``) markers, derived by applying the sample's
BCF variants to the reference genome.

This verifier walks the seq + BCF in lockstep and asserts, per BCF variant,
that the variant lands at its expected genomic coordinate in the parquet
seq, with the expected per-haplotype alleles and phasing. A position-mapping
bug in the encoder's `_build_haplotype_arrays` would produce variants at
wrong ref_offsets in the seq — the walker catches that as a token mismatch.

Per region the walker emits a list of `TokenDiff`s (empty on PASS); per
sample those are aggregated into a single sample-level diff log capped at
50 entries. The sample passes iff zero diffs, zero seq parse errors, and
at least one region was processed.

Out of scope (documented unrecoverable cases):
  - FILTER column round-trip (V5 schema does not store FILTER per variant)
  - INFO/FORMAT fields beyond GT (QUAL, ID, etc. are not round-trippable)
  - Multi-allelic ALT-list ordering
  - Hemizygous originals (encoder mirrors hemizygous to both haps)
  - Half-call genotypes (`./0`, `1/.`): encoder mirrors on chrX/Y, raises on
    autosomes — verifier inherits that behavior and never sees the raw call
  - Missing genotypes (`./.`): silently skipped by `_collect_variants`
  - Symbolic alleles (`<DEL>`, `<INS>`, ...): filtered before the verifier
  - Breakend notation (`]` / `[`): filtered before the verifier
  - Non-ACGT alleles: encoder raises before the verifier sees them
  - Bare-anchor variants the BCF doesn't know about: a homozygous full-span
    deletion looks identical to a plain ref char in the seq, so an
    encoder-injected spurious deletion at such a position is not detectable

Exit codes:
  0 — PASS (zero diffs, zero parse errors, ≥1 region processed)
  1 — FAIL (diffs and/or parse errors)
  2 — exception during verification

Usage:
  python -m preprocess.tools.verify_encoding \\
      --parquet <data>/diploid_hf_v5/train/HG00096.parquet \\
      --bcf-dir <data>/raw_bcf_bed_intersected_hg38_pretrain_canonical_expanded
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from preprocess import paths, tokens
from preprocess.encoder.get_genome_sequence import GenomeSequenceProcessor
from preprocess.markers import (
    DIR_FWD,
    DIR_REV,
    INDEL_CLOSE,
    INDEL_OPEN,
    SNP_MARKER,
    decode_direction,
    decode_indel_spans,
    decode_snp,
    rehydrate_markers,
)


def _rehydrate_marked_table(table):
    """Reconstruct the inline-marked ``seq`` from v8 sparse marker columns.

    The verifier walks the inline-marked string (``^``/``{ }``/``‹ ›``). A v8
    dataset stores those markers in the ``snp`` / ``indel_span`` / ``direction``
    columns with a marker-free ``seq``; this rebuilds the inline form per row so
    the entire downstream parser/comparator is reused unchanged. A pre-v8
    (inline) table is returned as-is. The result keeps only the five columns the
    per-region verifier reads.
    """
    names = set(table.schema.names)
    if "snp" not in names:
        return table
    seqs = table.column("seq").to_pylist()
    snp_col = table.column("snp").to_pylist()
    ind_col = table.column("indel_span").to_pylist()
    dir_col = (
        table.column("direction").to_pylist()
        if "direction" in names else [None] * len(seqs)
    )
    marked = [
        rehydrate_markers(
            seqs[i],
            decode_snp(snp_col[i]),
            decode_indel_spans(ind_col[i]),
            decode_direction(dir_col[i]),
        )
        for i in range(len(seqs))
    ]
    idx = table.schema.get_field_index("seq")
    table = table.set_column(idx, "seq", pa.array(marked, pa.large_utf8()))
    return table.select(["sample_id", "chrom", "start", "end", "seq"])

# ---------------------------------------------------------------------------
# Constants (marker glyphs come from preprocess.markers)
# ---------------------------------------------------------------------------

GAP = "-"

DEFAULT_REF_PATH = paths.HG38_2BIT
DEFAULT_TOKENS_PATH = paths.DIPLOID_TOKENS

logger = logging.getLogger("verify_encoding")


# ---------------------------------------------------------------------------
# Token decoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenInfo:
    """Decoded info about a diploid symbol."""
    alleles: Tuple[str, ...]   # (a0, a1) for diploid; (a,) for hemizygous; meta tokens have ()
    order: str                 # 'symmetric' | 'unphased' | 'phased' | 'single' | 'meta'


class TokenDecoder:
    """Loads diploid_tokens.tsv and resolves each Unicode symbol to (alleles, order).

    Stores ALL classes (symmetric, unphased, phased, single, meta). When a single
    symbol appears under multiple orders (e.g. an unphased symbol that also serves
    as a homozygous ('-' for '--')), the *first* entry wins; this matches encoder
    semantics, which loads tokens in TSV order and does not overwrite.
    """

    def __init__(self, tokens_tsv_path: Path):
        self.path = Path(tokens_tsv_path)
        # symbol -> TokenInfo
        self.symbol_to_info: Dict[str, TokenInfo] = {}
        # meta name (e.g. '<SNP>') -> symbol
        self.meta_name_to_symbol: Dict[str, str] = {}
        # (a0, a1) -> symbol; first occurrence wins (matches encoder semantics)
        self.alleles_to_symbol: Dict[Tuple[str, str], str] = {}

        for pair, order, symbol, _unicode in tokens.iter_token_rows(self.path):
            if order == "meta":
                self.meta_name_to_symbol[pair] = symbol
                if symbol not in self.symbol_to_info:
                    self.symbol_to_info[symbol] = TokenInfo(alleles=(), order="meta")
                continue

            if order == "symmetric":
                alleles: Tuple[str, ...] = (pair[0], pair[1])
            elif order in ("unphased", "phased"):
                if len(pair) != 2:
                    continue
                alleles = (pair[0], pair[1])
            elif order == "single":
                alleles = (pair[0],)
            else:
                continue

            if symbol not in self.symbol_to_info:
                self.symbol_to_info[symbol] = TokenInfo(alleles=alleles, order=order)
            if order in ("symmetric", "unphased", "phased") and len(alleles) == 2:
                key = (alleles[0], alleles[1])
                if key not in self.alleles_to_symbol:
                    self.alleles_to_symbol[key] = symbol

        if not self.symbol_to_info:
            raise ValueError(f"No tokens loaded from {self.path}")

        for required in ("<SNP>", "<INDEL_span_start>", "<INDEL_span_end>"):
            if required not in self.meta_name_to_symbol:
                logger.warning("meta token %s missing from %s", required, self.path)

    def decode(self, symbol: str) -> TokenInfo:
        info = self.symbol_to_info.get(symbol)
        if info is None:
            raise KeyError(
                f"Unknown diploid symbol {symbol!r} (U+{ord(symbol):04X}) in {self.path}"
            )
        return info


# ---------------------------------------------------------------------------


@dataclass
class RegionVerdict:
    """Per-region intermediate result for the position-level design.

    No PASS/FAIL is set per region — comparison happens once at the sample
    level after aggregating all regions' parquet/expected token lists. This
    struct just holds the per-region counts and any seq-parse errors so that
    sample-level diagnostics know where to point.
    """
    sample_id: str
    chrom: str
    start: int
    end: int
    parquet_seq_len: int
    n_parquet_tokens: int = 0
    n_expected_tokens: int = 0
    parse_errors: List[str] = field(default_factory=list)


@dataclass
class Report:
    sample_id: str
    parquet: Path
    bcf_dir: Path
    schema: str = "per_region"
    direction_markers_detected: Optional[bool] = None
    runtime_s: float = 0.0

    # Per-region counts
    n_regions_in_parquet: int = 0
    n_regions_processed: int = 0
    n_regions_skipped_no_bcf_chrom: int = 0
    n_regions_with_parse_errors: int = 0

    # Sample-level token counts
    n_parquet_tokens_total: int = 0
    n_expected_tokens_total: int = 0

    # Sample-level comparison results
    n_token_diffs: int = 0
    diffs: List[Dict[str, Any]] = field(default_factory=list)  # first 50, JSON-friendly
    parse_errors: List[str] = field(default_factory=list)  # first 50

    # Optional error capture (used by batch driver)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# BCF chromosome resolution
# ---------------------------------------------------------------------------


def find_bcf_for_chrom(bcf_dir: Path, chrom: str, pattern: Optional[str]) -> Optional[Path]:
    """Locate the BCF file for *chrom* under *bcf_dir*.

    If *pattern* is given (with a '{chrom}' placeholder), use it directly.
    Otherwise glob for *chrom* in filenames.
    """
    if pattern:
        candidate = bcf_dir / pattern.format(chrom=chrom)
        return candidate if candidate.exists() else None
    chrom_token = re.escape(chrom)
    rx = re.compile(rf"(^|[^A-Za-z0-9]){chrom_token}([^A-Za-z0-9]|$)")
    matches = [p for p in bcf_dir.glob("*.bcf") if rx.search(p.name)]
    if not matches:
        return None
    matches.sort(key=lambda p: len(p.name))
    return matches[0]


# ---------------------------------------------------------------------------
# Position-level VariantToken: shared shape between seq parser and BCF predictor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantToken:
    """One variant as it appears (or is expected to appear) at a genomic position.

    Both the parquet-seq parser and the BCF expected-stream emit tokens in the
    encoder's emission order: anchor first, then any insertions, then any
    ref-span continuations. ``h0_seq``/``h1_seq`` capture the per-haplotype
    bytes in that same order. ``-`` denotes a deletion gap on that haplotype.

    Equality on (chrom, region_start, pos_1based, h0_seq, h1_seq, phased) is
    sufficient for position-level verification: a position-mapping bug shifts
    pos_1based; a content bug perturbs h0_seq / h1_seq; a phasing bug flips
    ``phased`` or swaps h0/h1.
    """
    chrom: str
    region_start: int   # 0-based BED start; keys cross-region clip fragments
    pos_1based: int     # 1-based genomic anchor position (post-clip if cross-region)
    h0_seq: str
    h1_seq: str
    phased: bool

    def sort_key(self) -> Tuple[str, int, int, str, str]:
        return (self.chrom, self.region_start, self.pos_1based, self.h0_seq, self.h1_seq)


# ---------------------------------------------------------------------------
# Seq → variant tokens (parquet side)
# ---------------------------------------------------------------------------


def _decode_token_alleles(
    sym: str, direction: Optional[str], decoder: TokenDecoder
) -> Tuple[str, str]:
    """Decode one diploid symbol with optional direction marker → (h0_byte, h1_byte) chars.

    ``direction`` ∈ {None, DIR_FWD, DIR_REV}. SINGLE (hemizygous) tokens mirror
    the lone allele to both haps. Direction reverses canonical order on diploid
    tokens; on SINGLE tokens or homozygous tokens it is meaningless and raises.
    """
    info = decoder.decode(sym)
    if info.order == "meta":
        raise ValueError(f"meta token {sym!r} appeared in variant content position")
    if info.order == "single":
        a = info.alleles[0]
        if direction is not None:
            raise ValueError(
                f"direction marker {direction!r} on single (hemizygous) token {sym!r}"
            )
        return a, a
    a0, a1 = info.alleles[0], info.alleles[1]
    if direction == DIR_REV:
        a0, a1 = a1, a0
    return a0, a1


def _parse_indel_block(
    seq: str, i: int, decoder: TokenDecoder
) -> Tuple[int, str, str, bool]:
    """Parse a ``{...}`` block starting at ``seq[i] == '{'``.

    Token grammar inside the block:
      token := [^][<dir>?]<sym>

    The leading ``^`` (SNP marker) appears when a position is flagged as
    BOTH ``is_snp[i]`` and ``is_indel_non_anchor[i]`` (a SNP whose anchor
    falls inside an active indel span). The encoder's SNP branch in
    ``_build_marked_string`` emits ``^[dir]ch`` without first closing the
    open indel bracket, so a literal ``^`` ends up nested inside the
    ``{...}`` block. The walker treats the ``^`` as a no-op sub-marker
    and reads the following ``[dir?]+sym`` as the actual content.

    Returns ``(new_i, h0_chars, h1_chars, any_phased)``.
    """
    n = len(seq)
    if i >= n or seq[i] != INDEL_OPEN:
        raise ValueError(f"_parse_indel_block called at non-'{{' position seq_idx={i}")
    i += 1
    h0_parts: List[str] = []
    h1_parts: List[str] = []
    any_phased = False
    while i < n and seq[i] != INDEL_CLOSE:
        # Optional nested '^' marker for overlapping SNP+INDEL positions.
        if seq[i] == SNP_MARKER:
            i += 1
            if i >= n:
                raise ValueError("truncated nested '^' inside indel block")
        direction: Optional[str] = None
        if seq[i] in (DIR_FWD, DIR_REV):
            direction = seq[i]
            any_phased = True
            i += 1
            if i >= n:
                raise ValueError("truncated direction marker at end of indel block")
        sym = seq[i]
        i += 1
        a0, a1 = _decode_token_alleles(sym, direction, decoder)
        h0_parts.append(a0)
        h1_parts.append(a1)
    if i >= n:
        raise ValueError("unterminated '{' indel block")
    i += 1  # consume '}'
    return i, "".join(h0_parts), "".join(h1_parts), any_phased




# ---------------------------------------------------------------------------
# BCF → expected variant tokens (predictor)
# ---------------------------------------------------------------------------


def _predict_overlap_group_token(
    group_vars: List[Any],
    group_anchor_positions: List[int],
    region_start: int,
    region_end: int,
    processor: "GenomeSequenceProcessor",
    chrom: str,
) -> Tuple[Optional[VariantToken], int]:
    """Apply all variants in an overlapping group to a fresh ref-span byte
    array and emit one combined VariantToken in encoder-emission order.

    This is an independent reimplementation of the variant-application logic
    in get_genome_sequence._build_haplotype_arrays — independent so that a
    position bug in that function surfaces as a token mismatch here. Mirrors
    the encoder's emission order:
      for each ref offset i in [group_start_offset, group_end_offset):
        emit hap0[i]/hap1[i] (skip iff both are GAP)
        emit any insertions whose anchor is at offset i, sorted by allele offset
            (skip iff both insertion bytes are GAP)
    """
    from preprocess.encoder.get_genome_sequence import _get_alleles_for_variant

    earliest_anchor = min(group_anchor_positions)
    latest_end_excl = max(p + len(v.ref) for p, v in zip(group_anchor_positions, group_vars))

    # Cross-region clip on the group's earliest anchor: when the group
    # starts before the region, the per-variant clip below realigns each
    # variant's allele bytes; the group-level work here is just to set
    # ``group_start_offset`` and the reported ``effective_pos_1based``.
    if earliest_anchor <= region_start:
        group_start_offset = 0
        effective_pos_1based = region_start + 1
    else:
        group_start_offset = earliest_anchor - region_start - 1
        effective_pos_1based = earliest_anchor

    # Clamp to region end.
    n_region = region_end - region_start
    group_end_offset = min(latest_end_excl - region_start - 1, n_region)
    span_len = group_end_offset - group_start_offset
    if span_len <= 0:
        return None, 0

    ref_arr = processor._ref_file.fetch(chrom, region_start, region_end, as_array=True)
    hap0 = bytearray(ref_arr[group_start_offset:group_end_offset].tobytes())
    hap1 = bytearray(hap0)

    deletion_h0 = [False] * span_len
    deletion_h1 = [False] * span_len
    ins_by_offset: Dict[int, List[Tuple[int, str, str]]] = defaultdict(list)
    any_phased = False
    GAP_BYTE = ord(GAP)

    for p_1based, v in zip(group_anchor_positions, group_vars):
        any_phased = any_phased or v.phased
        h0_allele, h1_allele = _get_alleles_for_variant(v)
        ref_len = len(v.ref)
        anchor_offset_in_region = p_1based - region_start - 1

        # Per-variant cross-region clip on anchor before region start.
        if anchor_offset_in_region < 0:
            clip_v = -anchor_offset_in_region
            if ref_len - clip_v <= 0:
                continue
            h0_allele = h0_allele[clip_v:]
            h1_allele = h1_allele[clip_v:]
            ref_len -= clip_v
            anchor_offset_in_region = 0

        local_idx = anchor_offset_in_region - group_start_offset
        if local_idx < 0 or local_idx >= span_len:
            continue

        mod_h0 = v.gt[0] != 0
        mod_h1 = v.gt[1] != 0

        # Apply anchor.
        if mod_h0 and h0_allele:
            hap0[local_idx] = ord(h0_allele[0])
        if mod_h1 and h1_allele:
            hap1[local_idx] = ord(h1_allele[0])

        # Apply ref-span continuations.
        if mod_h0:
            for k in range(1, min(len(h0_allele), ref_len)):
                if local_idx + k < span_len:
                    hap0[local_idx + k] = ord(h0_allele[k])
        if mod_h1:
            for k in range(1, min(len(h1_allele), ref_len)):
                if local_idx + k < span_len:
                    hap1[local_idx + k] = ord(h1_allele[k])

        # Mark deletion bytes (allele shorter than REF span).
        if mod_h0 and len(h0_allele) < ref_len:
            for k in range(len(h0_allele), ref_len):
                if local_idx + k < span_len:
                    deletion_h0[local_idx + k] = True
        if mod_h1 and len(h1_allele) < ref_len:
            for k in range(len(h1_allele), ref_len):
                if local_idx + k < span_len:
                    deletion_h1[local_idx + k] = True

        # Collect insertions.
        max_allele_len = max(len(h0_allele), len(h1_allele))
        for offset in range(ref_len, max_allele_len):
            h0_is_ins = offset < len(h0_allele) and len(h0_allele) > ref_len
            h1_is_ins = offset < len(h1_allele) and len(h1_allele) > ref_len
            if not h0_is_ins and not h1_is_ins:
                continue
            ch0 = h0_allele[offset] if offset < len(h0_allele) else GAP
            ch1 = h1_allele[offset] if offset < len(h1_allele) else GAP
            ins_by_offset[local_idx].append((offset, ch0, ch1))

    # Apply deletion masks.
    for i in range(span_len):
        if deletion_h0[i]:
            hap0[i] = GAP_BYTE
        if deletion_h1[i]:
            hap1[i] = GAP_BYTE

    # Sort insertions per anchor offset. Encoder side uses bare ``.sort()``
    # on ``(offset, sym[, dir_val])``; here we sort on ``(offset, ch0, ch1)``.
    # Both agree on the primary ``offset`` key; ties at the same offset are
    # prevented upstream by ``_resolve_overlapping_variants`` (overlapping
    # insertions on the same haplotype are trimmed before this point).
    for syms in ins_by_offset.values():
        syms.sort()

    # Emit chars in encoder order: at each ref offset, emit hap0[i]/hap1[i]
    # then any insertions, applying skip_gaps semantics.
    h0_chars: List[str] = []
    h1_chars: List[str] = []
    for i in range(span_len):
        c0 = chr(hap0[i])
        c1 = chr(hap1[i])
        if not (c0 == GAP and c1 == GAP):
            h0_chars.append(c0)
            h1_chars.append(c1)
        for _, ic0, ic1 in ins_by_offset.get(i, ()):
            if not (ic0 == GAP and ic1 == GAP):
                h0_chars.append(ic0)
                h1_chars.append(ic1)

    h0_seq = "".join(h0_chars)
    h1_seq = "".join(h1_chars)

    # Note: empty h0_seq/h1_seq is valid (e.g. fully-clipped homozygous
    # deletion that produces no chars in the seq). The walker still needs
    # the ref_len so it can advance ref_offset past the deleted span.

    is_het = h0_seq != h1_seq
    return VariantToken(
        chrom=chrom,
        region_start=region_start,
        pos_1based=effective_pos_1based,
        h0_seq=h0_seq,
        h1_seq=h1_seq,
        phased=any_phased and is_het,
    ), span_len


def bcf_to_expected_variant_tokens(
    records: List[Any],
    chrom: str,
    region_start: int,
    region_end: int,
    sample_id: str,
    processor: "GenomeSequenceProcessor",
) -> Tuple[List[VariantToken], List[int]]:
    """Run encoder filter/clip/resolve logic over BCF records and produce the
    expected list of VariantTokens plus the parallel list of ``ref_len`` per
    token (for cursor advancement in the seq parser).

    Returns ``(tokens, ref_lens)`` where ``ref_lens[i]`` is the number of
    reference positions the i-th variant's encoded representation consumes
    (1 for SNPs; ref_len for complex; post-clip; clamped to remaining region
    bytes when the variant extends past region_end).

    Reuses ``_collect_variants`` and ``_resolve_overlapping_variants`` so that
    the verifier and the encoder agree on which records survive filtering and
    overlap resolution.
    """
    from preprocess.encoder.get_genome_sequence import (
        _collect_variants,
        _resolve_overlapping_variants,
    )

    if not records:
        return [], []

    variants_dict = _collect_variants(
        records,
        sample_idx=0,
        chrom=chrom,
        context_label=f"{sample_id}@{chrom}:{region_start}-{region_end}",
        filter_pass_only=processor._filter_pass_only,
    )
    variants_dict = _resolve_overlapping_variants(variants_dict, chrom, sample_id)

    n = region_end - region_start  # ref span (0-based)

    # Detect overlapping-variant groups: when V_b's anchor falls within V_a's
    # REF span, the encoder fuses both modifications into a single output
    # stream. We predict the combined effect independently (no reuse of
    # _build_haplotype_arrays) so that a position bug in the encoder's
    # variant-application logic surfaces as a token mismatch.
    sorted_pos = sorted(variants_dict.keys())
    groups: List[List[int]] = []  # list of groups; each is a list of 1-based anchor positions
    grp_end = None
    cur_group: List[int] = []
    for p in sorted_pos:
        v = variants_dict[p]
        rlen = len(v.ref)
        if not cur_group:
            cur_group = [p]
            grp_end = p + rlen - 1
            continue
        if p <= grp_end:
            cur_group.append(p)
            grp_end = max(grp_end, p + rlen - 1)
        else:
            groups.append(cur_group)
            cur_group = [p]
            grp_end = p + rlen - 1
    if cur_group:
        groups.append(cur_group)

    tokens: List[VariantToken] = []
    ref_lens: List[int] = []

    for grp in groups:
        # Always use the unified multi-variant predictor — it handles the
        # single-variant case correctly (group of 1) and properly applies
        # deletion masks for cross-region-clipped variants whose entire
        # allele was clipped away.
        tok, rl = _predict_overlap_group_token(
            [variants_dict[p] for p in grp], grp, region_start, region_end,
            processor, chrom,
        )
        if tok is not None:
            tokens.append(tok)
            ref_lens.append(rl)

    return tokens, ref_lens


# ---------------------------------------------------------------------------
# Per-sample comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenDiff:
    """One disagreement between reconstructed and expected token streams."""
    kind: str  # 'missing_in_parquet' | 'extra_in_parquet' | 'allele_mismatch'
    chrom: str
    pos_1based: int
    region_start: int
    expected: Optional[VariantToken] = None
    parquet: Optional[VariantToken] = None

    def short(self) -> str:
        loc = f"{self.chrom}:{self.pos_1based} (region_start={self.region_start})"
        if self.kind == "missing_in_parquet":
            v = self.expected
            return f"{loc}: BCF has {v.h0_seq}/{v.h1_seq} phased={v.phased}, parquet has nothing"
        if self.kind == "extra_in_parquet":
            v = self.parquet
            return f"{loc}: parquet has {v.h0_seq}/{v.h1_seq} phased={v.phased}, BCF has nothing"
        e, p = self.expected, self.parquet
        return (
            f"{loc}: BCF={e.h0_seq}/{e.h1_seq} phased={e.phased} "
            f"vs parquet={p.h0_seq}/{p.h1_seq} phased={p.phased}"
        )




class BcfReaderCache:
    """One cyvcf2 reader per (bcf_path, sample_id), reused across regions.

    Per-region open of cyvcf2.VCF dominated wall time when verifying ~125k regions
    per sample; this cache turns that into one open per chromosome.
    """

    def __init__(self, bcf_dir: Path, sample_id: str):
        self._bcf_dir = bcf_dir
        self._sample_id = sample_id
        # chrom -> Optional[Path]; None means "no BCF for this chrom" (negative cache)
        self._path_for_chrom: Dict[str, Optional[Path]] = {}
        # chrom -> open cyvcf2.VCF (or None when no BCF available)
        self._readers: Dict[str, Any] = {}

    def _path(self, chrom: str) -> Optional[Path]:
        if chrom not in self._path_for_chrom:
            self._path_for_chrom[chrom] = find_bcf_for_chrom(self._bcf_dir, chrom, None)
        return self._path_for_chrom[chrom]

    def query(self, chrom: str, start: int, end: int) -> List[Any]:
        """Return the list of cyvcf2 records overlapping [start, end) on *chrom*.
        Empty list if no BCF exists for the chromosome."""
        if chrom not in self._readers:
            from cyvcf2 import VCF

            path = self._path(chrom)
            if path is None:
                self._readers[chrom] = None
            else:
                self._readers[chrom] = VCF(str(path), samples=[self._sample_id])
        vcf = self._readers[chrom]
        if vcf is None:
            return []
        return list(vcf(f"{chrom}:{start + 1}-{end}"))

    def close(self) -> None:
        for vcf in self._readers.values():
            if vcf is not None:
                try:
                    vcf.close()
                except Exception:
                    pass
        self._readers.clear()




def _detect_direction_markers(seqs: Iterable[str], probe_limit: int = 256) -> bool:
    """Scan up to *probe_limit* sequences containing variant markers; return
    True iff any contain a direction marker (`‹` U+2039 or `›` U+203A).

    For variant-free sequences (no `^` or `{`), direction marker presence is
    inherently undetectable — so we focus on rows where direction *would*
    appear. If no probed sequence contains a variant marker, return False
    (direction is irrelevant for re-encoding regardless of setting).
    """
    n_probed = 0
    for s in seqs:
        if not s:
            continue
        if SNP_MARKER not in s and INDEL_OPEN not in s:
            continue
        if DIR_FWD in s or DIR_REV in s:
            return True
        n_probed += 1
        if n_probed >= probe_limit:
            break
    return False


def _consume_variant_emission_at(
    seq: str, i: int, anchor_pos_1based: int, chrom: str, region_start: int,
    decoder: TokenDecoder,
) -> Tuple[int, Optional[VariantToken], Optional[str]]:
    """Consume one full variant emission from seq[i].

    Encoder emission grammar (single uniform form, regardless of variant kind):
        emission := [^]?[<dir>?]<sym> ( {...} )?

    The optional leading ``^`` is the SNP marker; appears when the underlying
    variant is a SNP OR when an overlapping SNP shares the anchor position
    of a complex variant. The optional direction marker (``‹``/``›``)
    appears for phased het positions. The anchor diploid symbol is always
    present. The optional ``{...}`` block carries insertions and ref-span
    continuations of a complex variant.

    Returns ``(new_i, token, parse_error)``.
    """
    n = len(seq)
    if i >= n:
        return i, None, "EOF at variant emission start"
    if seq[i] == SNP_MARKER:
        i += 1
        if i >= n:
            return i, None, "truncated SNP marker"
    direction: Optional[str] = None
    if seq[i] in (DIR_FWD, DIR_REV):
        direction = seq[i]
        i += 1
        if i >= n:
            return i, None, "truncated direction marker"
    anchor_sym = seq[i]
    i += 1
    try:
        anchor_a0, anchor_a1 = _decode_token_alleles(anchor_sym, direction, decoder)
    except ValueError as e:
        return i, None, str(e)
    if i < n and seq[i] == INDEL_OPEN:
        try:
            i, blk_h0, blk_h1, blk_phased = _parse_indel_block(seq, i, decoder)
        except ValueError as e:
            return i, None, str(e)
        return i, VariantToken(
            chrom=chrom, region_start=region_start,
            pos_1based=anchor_pos_1based,
            h0_seq=anchor_a0 + blk_h0,
            h1_seq=anchor_a1 + blk_h1,
            phased=(direction is not None) or blk_phased,
        ), None
    return i, VariantToken(
        chrom=chrom, region_start=region_start,
        pos_1based=anchor_pos_1based,
        h0_seq=anchor_a0,
        h1_seq=anchor_a1,
        phased=direction is not None,
    ), None


def _scan_for_seq_variant(
    seq: str, i: int, ref_offset: int, chrom: str, region_start: int,
    decoder: TokenDecoder,
) -> Tuple[int, int, Optional[VariantToken], Optional[str]]:
    """Try to consume one variant emission at seq[i] (assumes seq[i] is or
    precedes a variant marker).

    Used by ``_scan_extras_until`` when consuming unexpected variants in the
    parquet that the BCF didn't predict. Best-guess ref advancement = 1
    (correct for SNPs and pure insertions; off for deletions/MNVs that
    weren't in the BCF — but those would already be flagged by the BCF
    being silent on them, so cascading desync is acceptable).
    """
    n = len(seq)
    if i >= n:
        return i, 0, None, None
    ch = seq[i]
    looks_like_variant = (
        ch == SNP_MARKER
        or ch in (DIR_FWD, DIR_REV)
        or (i + 1 < n and seq[i + 1] == INDEL_OPEN)
    )
    if not looks_like_variant:
        return i, 0, None, None
    pos = region_start + ref_offset + 1
    new_i, tok, err = _consume_variant_emission_at(seq, i, pos, chrom, region_start, decoder)
    return new_i, 1, tok, err


def _tokens_agree(parquet_tok: VariantToken, expected_tok: VariantToken) -> bool:
    """Compare a walker-decoded token to a BCF-predicted token.

    For unphased het positions the diploid symbol is order-insensitive
    (e.g. `Ą` covers both (A,C) and (C,A)) and the decoder always returns
    the canonical TSV order, while the predictor preserves the GT-driven
    order. So when both sides agree the position is unphased het, we
    compare the unordered haplotype pair.
    """
    if parquet_tok.phased != expected_tok.phased:
        return False
    p_het = parquet_tok.h0_seq != parquet_tok.h1_seq
    e_het = expected_tok.h0_seq != expected_tok.h1_seq
    if not parquet_tok.phased and p_het and e_het:
        return sorted((parquet_tok.h0_seq, parquet_tok.h1_seq)) == sorted(
            (expected_tok.h0_seq, expected_tok.h1_seq)
        )
    return (
        parquet_tok.h0_seq == expected_tok.h0_seq
        and parquet_tok.h1_seq == expected_tok.h1_seq
    )


def verify_one_region(
    processor: GenomeSequenceProcessor,
    decoder: TokenDecoder,
    sample_id: str,
    chrom: str,
    start: int,
    end: int,
    parquet_seq: str,
    is_ref_sample: bool,
    bcf_cache: Optional[BcfReaderCache] = None,
) -> Tuple[RegionVerdict, List[VariantToken], List[VariantToken], List["TokenDiff"]]:
    """BCF-driven lockstep walker: for each expected variant from the BCF,
    advance the seq cursor to its expected ref position and validate the
    variant token at that position. Plain ref symbols between variants are
    consumed silently. Unexpected markers in the parquet are flagged as
    'extra_in_parquet'. Expected variants whose anchor position the seq
    can't reach (because seq is shorter or has unexpected ordering) are
    flagged as 'missing_in_parquet'.

    Special case — *marker-less anchor*: complex variants whose `{...}`
    block would be entirely homozygous-gap (e.g. a homozygous deletion
    REF=AGT ALT=A with both haplotypes deleting positions idx+1, idx+2)
    emit only the anchor char with no `{` marker. The walker uses BCF
    info to recognize these: when the BCF expects a complex variant at
    the next ref position and the seq has a plain char (no marker),
    we treat that char as the anchor and advance the cursor by ref_len.

    Returns ``(verdict, parquet_tokens, expected_tokens, diffs)`` —
    parquet/expected tokens are aggregated across regions for sample-level
    statistics; diffs are aggregated for sample-level diagnosis.
    """
    expected_tokens: List[VariantToken] = []
    expected_ref_lens: List[int] = []
    if not is_ref_sample and bcf_cache is not None:
        records = bcf_cache.query(chrom, start, end)
        expected_tokens, expected_ref_lens = bcf_to_expected_variant_tokens(
            records, chrom, start, end, sample_id, processor,
        )

    # For HG38_REF rows, also load the actual reference bytes so each plain
    # symbol in the parquet seq can be checked against the homozygous-ref
    # token we expect at that position (B3 in the verify_encoding review).
    expected_ref_symbols: Optional[List[str]] = None
    if is_ref_sample:
        ref_arr = processor._ref_file.fetch(chrom, start, end, as_array=True)
        expected_ref_symbols = []
        for b in ref_arr.tolist():
            base = chr(b)
            sym = decoder.alleles_to_symbol.get((base, base))
            # Fall back to the base char itself when the token table has no
            # entry (e.g. soft-masked lowercase or unusual bases). This
            # preserves the prior pass-through behavior for unknown bases
            # while flagging real corruption on canonical bases.
            expected_ref_symbols.append(sym if sym is not None else base)

    diffs: List[TokenDiff] = []
    parse_errors: List[str] = []
    parquet_tokens: List[VariantToken] = []

    n = len(parquet_seq)
    seq_idx = 0
    ref_offset = 0

    def _scan_extras_until(target_offset: int) -> None:
        """Scan from the current seq cursor up to ref_offset == target_offset,
        consuming plain ref chars and flagging any markers as extras.

        For HG38_REF rows (``expected_ref_symbols`` set), each plain char is
        also compared to the expected homozygous-ref symbol at that ref
        offset; mismatches are recorded as parse errors.
        """
        nonlocal seq_idx, ref_offset
        while ref_offset < target_offset and seq_idx < n:
            ch = parquet_seq[seq_idx]
            is_variant = (
                ch == SNP_MARKER or ch in (DIR_FWD, DIR_REV) or
                (seq_idx + 1 < n and parquet_seq[seq_idx + 1] == INDEL_OPEN)
            )
            if not is_variant:
                if expected_ref_symbols is not None:
                    exp = expected_ref_symbols[ref_offset]
                    if ch != exp:
                        parse_errors.append(
                            f"{chrom}:{start + ref_offset + 1} (seq_idx={seq_idx}): "
                            f"HG38_REF mismatch: expected {exp!r}, got {ch!r}"
                        )
                ref_offset += 1
                seq_idx += 1
                continue
            new_i, ref_used, tok, err = _scan_for_seq_variant(
                parquet_seq, seq_idx, ref_offset, chrom, start, decoder,
            )
            if err is not None:
                parse_errors.append(
                    f"{chrom}:{start + ref_offset + 1} (seq_idx={seq_idx}): {err}"
                )
            if tok is not None:
                parquet_tokens.append(tok)
                diffs.append(TokenDiff(
                    kind="extra_in_parquet",
                    chrom=tok.chrom,
                    pos_1based=tok.pos_1based,
                    region_start=tok.region_start,
                    parquet=tok,
                ))
            seq_idx = new_i
            ref_offset += ref_used

    for V_exp, V_ref_len in zip(expected_tokens, expected_ref_lens):
        target_offset = V_exp.pos_1based - start - 1
        _scan_extras_until(target_offset)

        if ref_offset != target_offset or seq_idx >= n:
            diffs.append(TokenDiff(
                kind="missing_in_parquet",
                chrom=V_exp.chrom,
                pos_1based=V_exp.pos_1based,
                region_start=V_exp.region_start,
                expected=V_exp,
            ))
            # Keep cursor consistent for subsequent variants.
            ref_offset = target_offset + V_ref_len
            continue

        # Consume one or more atoms until total len(h0_seq) matches the
        # expected token's h0_seq length. Most variants are a single atom;
        # overlap groups whose member variants have *different* anchor
        # positions produce multiple atoms back-to-back (a same-anchor
        # group can't occur because `_merge_same_position_variants` collapses
        # those into one multi-allelic variant). Termination on
        # ``expected_h0_len`` is correct because the predictor and encoder
        # apply identical per-position gap-skipping, so non-skipped chars
        # are 1:1 between the two streams.
        anchor_pos = start + ref_offset + 1
        expected_h0_len = len(V_exp.h0_seq)
        coll_h0: List[str] = []
        coll_h1: List[str] = []
        coll_phased = False
        atom_err: Optional[str] = None
        while sum(len(s) for s in coll_h0) < expected_h0_len and seq_idx < n:
            atom_anchor_pos = start + ref_offset + 1  # only first matters for the token
            new_i, atom_tok, err = _consume_variant_emission_at(
                parquet_seq, seq_idx, atom_anchor_pos, chrom, start, decoder,
            )
            if err is not None:
                atom_err = err
                seq_idx = new_i
                break
            if atom_tok is None:
                atom_err = "no token returned from emission consumer"
                seq_idx = new_i
                break
            coll_h0.append(atom_tok.h0_seq)
            coll_h1.append(atom_tok.h1_seq)
            if atom_tok.phased:
                coll_phased = True
            seq_idx = new_i
        ref_offset += V_ref_len
        if atom_err is not None:
            parse_errors.append(
                f"{chrom}:{anchor_pos} (seq_idx={seq_idx}): {atom_err}"
            )
        if coll_h0:
            tok = VariantToken(
                chrom=chrom, region_start=start,
                pos_1based=anchor_pos,
                h0_seq="".join(coll_h0),
                h1_seq="".join(coll_h1),
                phased=coll_phased,
            )
            parquet_tokens.append(tok)
            if not _tokens_agree(tok, V_exp):
                diffs.append(TokenDiff(
                    kind="allele_mismatch",
                    chrom=tok.chrom,
                    pos_1based=tok.pos_1based,
                    region_start=tok.region_start,
                    expected=V_exp,
                    parquet=tok,
                ))

    # Trailing seq: should be plain ref symbols only. Any markers are extras.
    _scan_extras_until(end - start)

    if ref_offset != end - start:
        parse_errors.append(
            f"{chrom}:{start}-{end}: ref length walked={ref_offset}, "
            f"expected={end - start}"
        )

    # B2: any seq bytes left over after the walk reached end-start (or
    # exhausted ref positions) are trailing extras. Scan them for markers
    # and flag as extras-in-parquet; remaining plain bytes become a
    # length-mismatch parse error so the diff log points at the offending
    # region.
    if seq_idx < n:
        trailing_start = seq_idx
        while seq_idx < n:
            ch = parquet_seq[seq_idx]
            is_variant = (
                ch == SNP_MARKER or ch in (DIR_FWD, DIR_REV)
                or (seq_idx + 1 < n and parquet_seq[seq_idx + 1] == INDEL_OPEN)
            )
            if not is_variant:
                seq_idx += 1
                continue
            new_i, _ref_used, tok, err = _scan_for_seq_variant(
                parquet_seq, seq_idx, ref_offset, chrom, start, decoder,
            )
            if err is not None:
                parse_errors.append(
                    f"{chrom}:trailing@seq_idx={seq_idx}: {err}"
                )
            if tok is not None:
                parquet_tokens.append(tok)
                diffs.append(TokenDiff(
                    kind="extra_in_parquet",
                    chrom=tok.chrom, pos_1based=tok.pos_1based,
                    region_start=tok.region_start, parquet=tok,
                ))
            seq_idx = new_i if new_i > seq_idx else seq_idx + 1
        parse_errors.append(
            f"{chrom}:{start}-{end}: {n - trailing_start} trailing chars "
            f"in seq past expected end (n={n}, ref_walked={ref_offset})"
        )

    verdict = RegionVerdict(
        sample_id=sample_id,
        chrom=chrom,
        start=start,
        end=end,
        parquet_seq_len=len(parquet_seq),
        n_parquet_tokens=len(parquet_tokens),
        n_expected_tokens=len(expected_tokens),
        parse_errors=parse_errors,
    )
    return verdict, parquet_tokens, expected_tokens, diffs




# ---------------------------------------------------------------------------
# Main verification driver
# ---------------------------------------------------------------------------


def _resolve_sample_id(distinct_samples: List[str], sample_id: Optional[str]) -> str:
    if sample_id is None:
        if len(distinct_samples) != 1:
            raise ValueError(
                f"--sample-id required: parquet contains {len(distinct_samples)} samples"
            )
        return distinct_samples[0]
    if sample_id not in distinct_samples:
        raise ValueError(
            f"--sample-id {sample_id} not found in parquet. Have: {distinct_samples}"
        )
    return sample_id


def _available_chroms(processor: GenomeSequenceProcessor, bcf_dir: Path) -> set:
    available: set = set()
    for p in bcf_dir.glob("*.bcf"):
        for chrom in processor._ref_file.chromosome_names():
            if re.search(rf"(^|[^A-Za-z0-9]){re.escape(chrom)}([^A-Za-z0-9]|$)", p.name):
                available.add(chrom)
    return available


def verify_sample(
    parquet_path: Path,
    bcf_dir: Path,
    ref_path: Path,
    tokens_path: Path,
    sample_id: Optional[str],
    max_regions: Optional[int],
    *,
    processor: Optional[GenomeSequenceProcessor] = None,
    decoder: Optional[TokenDecoder] = None,
    filter_pass_only: bool = True,
    max_diffs: int = 50,
) -> Report:
    """Dispatch position-level verification for the V5 per-region schema.

    The parquet must have columns ``seq + chrom + start + end + sample_id``.
    When *processor*/*decoder* are passed in, they are reused; otherwise a
    fresh `GenomeSequenceProcessor` is built (this preloads ~3GB of
    reference).

    The ``direction_markers_detected`` field on the report is a diagnostic
    only — the walker decodes ``‹``/``›`` markers regardless of the
    processor's ``use_direction_markers`` setting, and the verifier never
    re-encodes seqs, so the autodetect does not affect verification
    correctness. It is only useful as a sanity check that the dataset's
    encoding flag matches what is observed in the seqs.
    """
    t_start = time.perf_counter()

    schema = pq.read_schema(str(parquet_path))
    cols = set(schema.names)
    is_per_region = (
        "seq" in cols and "chrom" in cols and "start" in cols
        and "end" in cols and "sample_id" in cols
    )
    if not is_per_region:
        raise ValueError(
            f"Parquet at {parquet_path} does not match the per-region schema "
            f"(seq + chrom + start + end + sample_id). Columns: {sorted(cols)}"
        )

    read_cols = ["sample_id", "chrom", "start", "end", "seq"]
    if "snp" in cols:  # v8: markers externalized into sparse columns
        read_cols += ["snp", "indel_span"]
        if "direction" in cols:
            read_cols.append("direction")
    table = pq.read_table(str(parquet_path), columns=read_cols)
    # v8 → reconstruct the inline-marked seq so the existing walker applies.
    table = _rehydrate_marked_table(table)

    return _verify_sample_per_region(
        parquet_path=parquet_path,
        bcf_dir=bcf_dir,
        ref_path=ref_path,
        tokens_path=tokens_path,
        sample_id=sample_id,
        max_regions=max_regions,
        table=table,
        t_start=t_start,
        processor=processor,
        decoder=decoder,
        filter_pass_only=filter_pass_only,
        max_diffs=max_diffs,
    )




def _verify_sample_per_region(
    *,
    parquet_path: Path,
    bcf_dir: Path,
    ref_path: Path,
    tokens_path: Path,
    sample_id: Optional[str],
    max_regions: Optional[int],
    table,
    t_start: float,
    processor: Optional[GenomeSequenceProcessor],
    decoder: Optional[TokenDecoder],
    filter_pass_only: bool = True,
    max_diffs: int = 50,
) -> Report:
    """Per-region V4/V5 schema verification (one parquet row per region)."""
    import pyarrow.compute as pc

    sample_id_col = table.column("sample_id")
    distinct_samples = sorted(pc.unique(sample_id_col).to_pylist())
    sample_id = _resolve_sample_id(distinct_samples, sample_id)
    is_ref_sample = sample_id == "HG38_REF"
    logger.info("sample_id=%s (HG38_REF=%s)", sample_id, is_ref_sample)

    # Filter to this sample at the Arrow layer; avoids materializing every column
    # to a Python list when the parquet contains multiple samples.
    if len(distinct_samples) > 1:
        mask = pc.equal(sample_id_col, sample_id)
        sub = table.filter(mask)
    else:
        sub = table

    chrom_list = sub.column("chrom").to_pylist()
    start_list = sub.column("start").to_pylist()
    end_list = sub.column("end").to_pylist()
    seq_list = sub.column("seq").to_pylist()

    rows: List[Tuple[str, int, int, str]] = [
        (chrom_list[i], int(start_list[i]), int(end_list[i]), seq_list[i])
        for i in range(len(chrom_list))
    ]
    if not rows:
        raise ValueError(f"No rows for sample_id={sample_id} in parquet")

    # Sort for deterministic ordering and locality.
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    if max_regions is not None:
        rows = rows[:max_regions]

    # Direction-marker autodetect: scan the sample's seqs (variant rows only).
    # When a processor is passed in (batch driver), skip the autodetect — caller
    # has already chosen `use_direction_markers` for the whole run.
    if processor is None:
        direction_detected = _detect_direction_markers((r[3] for r in rows))
        logger.info("direction_markers_detected=%s for sample %s",
                    direction_detected, sample_id)
    else:
        direction_detected = None

    report = Report(
        sample_id=sample_id,
        parquet=parquet_path,
        bcf_dir=bcf_dir,
        schema="per_region",
        direction_markers_detected=direction_detected,
    )
    report.n_regions_in_parquet = len(rows)

    owns_processor = processor is None
    if processor is None:
        logger.info("loading reference genome from %s (use_direction_markers=%s "
                    "filter_pass_only=%s)",
                    ref_path, direction_detected, filter_pass_only)
        processor = GenomeSequenceProcessor(
            ref_path=ref_path,
            diploid_tokens_path=tokens_path,
            use_direction_markers=direction_detected,
            filter_pass_only=filter_pass_only,
        )
    if decoder is None:
        decoder = TokenDecoder(tokens_path)

    available_chroms: set = set() if is_ref_sample else _available_chroms(processor, bcf_dir)
    if not is_ref_sample and not available_chroms:
        logger.warning("No BCF files matched chromosomes under %s", bcf_dir)

    # Filter rows to those whose chrom has a BCF (skipped rows are tallied below).
    if is_ref_sample:
        encodable_rows = rows
    else:
        encodable_rows = [r for r in rows if r[0] in available_chroms]
        report.n_regions_skipped_no_bcf_chrom = len(rows) - len(encodable_rows)

    # One cyvcf2 reader per chromosome, reused across all regions of this sample.
    bcf_cache: Optional[BcfReaderCache] = (
        None if is_ref_sample else BcfReaderCache(bcf_dir, sample_id)
    )

    # ---- Per-region BCF-driven walker: aggregate diffs across regions ----
    # Each verify_one_region call performs a BCF-driven lockstep walk and emits
    # per-region diffs. Sample-level result = aggregate diffs (capped at 50).
    diffs_total: List[TokenDiff] = []
    parse_errors_total: List[str] = []
    cap = max_diffs

    try:
        n_total = len(encodable_rows)
        for n, (chrom, start, end, parquet_seq) in enumerate(encodable_rows, 1):
            verdict, parquet_tokens, expected_tokens, region_diffs = verify_one_region(
                processor=processor,
                decoder=decoder,
                sample_id=sample_id,
                chrom=chrom,
                start=start,
                end=end,
                parquet_seq=parquet_seq,
                is_ref_sample=is_ref_sample,
                bcf_cache=bcf_cache,
            )
            report.n_regions_processed += 1
            report.n_parquet_tokens_total += len(parquet_tokens)
            report.n_expected_tokens_total += len(expected_tokens)
            if verdict.parse_errors:
                report.n_regions_with_parse_errors += 1
                for err in verdict.parse_errors:
                    if len(parse_errors_total) < cap:
                        parse_errors_total.append(err)
            report.n_token_diffs += len(region_diffs)
            for d in region_diffs:
                if len(diffs_total) < cap:
                    diffs_total.append(d)

            if n % 1000 == 0 or n == n_total:
                logger.info(
                    "[%s] processed %d/%d regions (parquet_toks=%d expected_toks=%d "
                    "diffs=%d parse_err_regions=%d)",
                    datetime.now().strftime("%H:%M:%S"),
                    n, n_total,
                    report.n_parquet_tokens_total,
                    report.n_expected_tokens_total,
                    report.n_token_diffs,
                    report.n_regions_with_parse_errors,
                )
    finally:
        if bcf_cache is not None:
            bcf_cache.close()
        if owns_processor:
            processor.close()

    report.diffs = [
        {
            "kind": d.kind,
            "chrom": d.chrom,
            "pos_1based": d.pos_1based,
            "region_start": d.region_start,
            "expected_h0": d.expected.h0_seq if d.expected else None,
            "expected_h1": d.expected.h1_seq if d.expected else None,
            "expected_phased": d.expected.phased if d.expected else None,
            "parquet_h0": d.parquet.h0_seq if d.parquet else None,
            "parquet_h1": d.parquet.h1_seq if d.parquet else None,
            "parquet_phased": d.parquet.phased if d.parquet else None,
        }
        for d in diffs_total
    ]
    report.parse_errors = parse_errors_total

    report.runtime_s = time.perf_counter() - t_start
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--parquet", type=Path, required=True, help="Per-sample parquet shard.")
    p.add_argument("--bcf-dir", type=Path, required=True, help="Directory of per-chrom BCFs.")
    p.add_argument("--reference", type=Path, default=DEFAULT_REF_PATH, help="hg38 2bit/FASTA.")
    p.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS_PATH, help="diploid_tokens.tsv.")
    p.add_argument("--sample-id", type=str, default=None,
                   help="Sample to verify (required if parquet has multiple).")
    p.add_argument("--max-regions", type=int, default=None,
                   help="Limit number of regions (per-region smoke test).")
    p.add_argument(
        "--filter-pass-only",
        dest="filter_pass_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only process PASS variants (must match the encoder setting "
             "used when generating the dataset). Use --no-filter-pass-only "
             "to also include non-PASS variants.",
    )
    p.add_argument(
        "--max-diffs", type=int, default=50,
        help="Cap the number of recorded diffs and parse errors per "
             "sample (default 50). The full counts are still reported.",
    )
    p.add_argument("--report", type=Path, default=None, help="Write text report to file (default: stdout).")
    p.add_argument("--report-json", type=Path, default=None,
                   help="Write structured Report as JSON (machine-readable; for batch aggregation).")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING")
    p.add_argument(
        "--representation",
        choices=["diploid", "haploid"],
        default="diploid",
        help="Which encoding to verify. 'diploid' (default) runs the "
             "position-level token verifier. 'haploid' regenerates each "
             "region's haploid string locally via get_haploid_sequence_str "
             "and asserts exact byte equality with the parquet 'seq' field.",
    )
    return p.parse_args()


def report_passed(report: Report) -> bool:
    """Single source of truth for overall PASS/FAIL judgment.

    PASS iff: at least one region was processed, zero token-level diffs were
    found in the sample-level comparison, and zero seq parse errors occurred.
    """
    if report.schema != "per_region":
        return False
    return (
        report.n_token_diffs == 0
        and report.n_regions_with_parse_errors == 0
        and report.n_regions_processed > 0
    )


def render_report(report: Report) -> str:
    lines: List[str] = []
    lines.append("=== verify_encoding.py report (position-level) ===")
    lines.append(f"schema:                       {report.schema}")
    lines.append(f"sample_id:                    {report.sample_id}")
    lines.append(f"parquet:                      {report.parquet}")
    lines.append(f"bcf-dir:                      {report.bcf_dir}")
    lines.append(f"direction_markers_detected:   {report.direction_markers_detected}")
    lines.append(f"regions_in_parquet:           {report.n_regions_in_parquet}")
    lines.append(f"regions_processed:            {report.n_regions_processed}")
    lines.append(f"regions_skipped_no_bcf:       {report.n_regions_skipped_no_bcf_chrom}")
    lines.append(f"regions_with_parse_errors:    {report.n_regions_with_parse_errors}")
    lines.append(f"parquet_variant_tokens:       {report.n_parquet_tokens_total}")
    lines.append(f"expected_variant_tokens:      {report.n_expected_tokens_total}")
    lines.append(f"token_diffs:                  {report.n_token_diffs}")
    lines.append("")

    if report.diffs:
        lines.append(f"--- first {len(report.diffs)} token diffs ---")
        for d in report.diffs[:20]:
            loc = f"{d['chrom']}:{d['pos_1based']} (region_start={d['region_start']})"
            kind = d["kind"]
            if kind == "missing_in_parquet":
                lines.append(
                    f"  [{kind}] {loc}: BCF={d['expected_h0']}/{d['expected_h1']} "
                    f"phased={d['expected_phased']}"
                )
            elif kind == "extra_in_parquet":
                lines.append(
                    f"  [{kind}] {loc}: parquet={d['parquet_h0']}/{d['parquet_h1']} "
                    f"phased={d['parquet_phased']}"
                )
            else:
                lines.append(
                    f"  [{kind}] {loc}: BCF={d['expected_h0']}/{d['expected_h1']} "
                    f"phased={d['expected_phased']} vs "
                    f"parquet={d['parquet_h0']}/{d['parquet_h1']} "
                    f"phased={d['parquet_phased']}"
                )
        lines.append("")

    if report.parse_errors:
        lines.append(f"--- first {len(report.parse_errors)} parse errors ---")
        for err in report.parse_errors[:20]:
            lines.append(f"  {err}")
        lines.append("")

    lines.append("result: " + ("PASS" if report_passed(report) else "FAIL"))
    lines.append(f"runtime: {report.runtime_s:.1f}s")
    return "\n".join(lines)


def report_to_json_dict(report: Report) -> Dict[str, Any]:
    """Convert a Report to a JSON-serializable dict (Path → str)."""
    d = asdict(report)
    d["parquet"] = str(report.parquet)
    d["bcf_dir"] = str(report.bcf_dir)
    d["passed"] = report_passed(report)
    return d


def verify_sample_haploid(
    parquet_path: Path,
    bcf_dir: Path,
    ref_path: Path,
    sample_id: Optional[str] = None,
    max_regions: Optional[int] = None,
    filter_pass_only: bool = True,
    max_diffs: int = 50,
) -> Dict[str, Any]:
    """Regenerate-and-compare haploid verification.

    For each (chrom, start, end) row in the parquet, computes the haploid
    string from BCF+reference via :meth:`GenomeSequenceProcessor.get_haploid_sequence_str`
    and asserts byte-equality with the stored ``seq``. Mismatches are
    accumulated up to ``max_diffs``; the total mismatch count is always
    reported.
    """
    t0 = time.perf_counter()
    _names = set(pq.read_schema(str(parquet_path)).names)
    _read_cols = ["sample_id", "chrom", "start", "end", "seq"]
    if "snp" in _names:  # v8 haploid: markers externalized (snp + indel_span)
        _read_cols += ["snp", "indel_span"]
    table = pq.read_table(parquet_path, columns=_read_cols)
    table = _rehydrate_marked_table(table)  # v8 → inline-marked seq
    sample_ids = table.column("sample_id").to_pylist()
    chroms = table.column("chrom").to_pylist()
    starts = table.column("start").to_pylist()
    ends = table.column("end").to_pylist()
    seqs = table.column("seq").to_pylist()
    distinct_samples = sorted(set(sample_ids))

    target_sample = _resolve_sample_id(distinct_samples, sample_id)
    row_indices = [i for i, s in enumerate(sample_ids) if s == target_sample]
    if max_regions is not None:
        row_indices = row_indices[:max_regions]

    processor = GenomeSequenceProcessor(
        ref_path=ref_path,
        diploid_tokens_path=None,  # haploid path doesn't need user tokens
        filter_pass_only=filter_pass_only,
    )

    n_total = len(row_indices)
    n_match = 0
    n_skipped_no_bcf = 0
    diffs: List[Dict[str, Any]] = []
    parse_errors: List[str] = []

    avail = _available_chroms(processor, bcf_dir)

    try:
        for i in row_indices:
            chrom = chroms[i]
            start = starts[i]
            end = ends[i]
            parquet_seq = seqs[i]

            if chrom not in avail:
                n_skipped_no_bcf += 1
                continue

            try:
                expected_seq = processor.get_haploid_sequence_str(
                    bcf_dir=bcf_dir,
                    sample_name=target_sample,
                    chrom=chrom,
                    start=start,
                    end=end,
                    skip_gaps=True,
                )
            except Exception as exc:
                msg = (
                    f"{chrom}:{start}-{end}: regeneration raised "
                    f"{type(exc).__name__}: {exc}"
                )
                if len(parse_errors) < max_diffs:
                    parse_errors.append(msg)
                continue

            if expected_seq == parquet_seq:
                n_match += 1
            else:
                if len(diffs) < max_diffs:
                    # Find first divergence index for diagnostic
                    div_idx = next(
                        (k for k, (a, b) in enumerate(zip(expected_seq, parquet_seq))
                         if a != b),
                        min(len(expected_seq), len(parquet_seq)),
                    )
                    diffs.append({
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "diverge_idx": div_idx,
                        "expected_len": len(expected_seq),
                        "parquet_len": len(parquet_seq),
                        "expected_prefix": expected_seq[max(0, div_idx - 20):div_idx + 20],
                        "parquet_prefix": parquet_seq[max(0, div_idx - 20):div_idx + 20],
                    })
    finally:
        processor.close()

    n_mismatch = n_total - n_match - n_skipped_no_bcf - len(parse_errors)

    return {
        "schema": "per_region_haploid",
        "sample_id": target_sample,
        "parquet": str(parquet_path),
        "bcf_dir": str(bcf_dir),
        "n_regions_total": n_total,
        "n_regions_skipped_no_bcf_chrom": n_skipped_no_bcf,
        "n_regions_match": n_match,
        "n_regions_mismatch": n_mismatch,
        "n_regions_with_regeneration_errors": len(parse_errors),
        "diffs": diffs,
        "parse_errors": parse_errors,
        "runtime_s": time.perf_counter() - t0,
        "passed": (
            n_mismatch == 0
            and len(parse_errors) == 0
            and n_match > 0
        ),
    }


def render_haploid_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=== verify_encoding.py report (haploid, regenerate-and-compare) ===")
    lines.append(f"sample_id:                    {report['sample_id']}")
    lines.append(f"parquet:                      {report['parquet']}")
    lines.append(f"bcf-dir:                      {report['bcf_dir']}")
    lines.append(f"regions_total:                {report['n_regions_total']}")
    lines.append(f"regions_skipped_no_bcf:       {report['n_regions_skipped_no_bcf_chrom']}")
    lines.append(f"regions_match:                {report['n_regions_match']}")
    lines.append(f"regions_mismatch:             {report['n_regions_mismatch']}")
    lines.append(f"regions_with_regen_errors:    {report['n_regions_with_regeneration_errors']}")
    lines.append("")

    if report["diffs"]:
        lines.append(f"--- first {len(report['diffs'])} mismatches ---")
        for d in report["diffs"][:20]:
            lines.append(
                f"  {d['chrom']}:{d['start']}-{d['end']}  "
                f"diverge_idx={d['diverge_idx']} "
                f"(expected_len={d['expected_len']} parquet_len={d['parquet_len']})"
            )
            lines.append(f"    expected[~div]: {d['expected_prefix']!r}")
            lines.append(f"    parquet[~div]:  {d['parquet_prefix']!r}")
        lines.append("")

    if report["parse_errors"]:
        lines.append(f"--- first {len(report['parse_errors'])} regeneration errors ---")
        for err in report["parse_errors"][:20]:
            lines.append(f"  {err}")
        lines.append("")

    lines.append("result: " + ("PASS" if report["passed"] else "FAIL"))
    lines.append(f"runtime: {report['runtime_s']:.1f}s")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # The encoder emits per-variant WARNINGs (e.g. "Skipping homozygous-ref")
    # that flood the verify output without adding diagnostic value here —
    # those code paths are well-trodden and hom-ref skip is the encoder's
    # documented behaviour. Silence them but keep the verify-script's own
    # WARNINGs (parse-error notices, FAIL summaries) at the user's chosen level.
    logging.getLogger("preprocess.get_genome_sequence").setLevel(logging.ERROR)
    t_wall = datetime.now()
    logger.info("start: %s", t_wall.isoformat(timespec="seconds"))
    logger.info("cmd: %s", " ".join(sys.argv))

    if args.representation == "haploid":
        try:
            report_dict = verify_sample_haploid(
                parquet_path=args.parquet,
                bcf_dir=args.bcf_dir,
                ref_path=args.reference,
                sample_id=args.sample_id,
                max_regions=args.max_regions,
                filter_pass_only=args.filter_pass_only,
                max_diffs=args.max_diffs,
            )
        except Exception as e:
            logger.exception("verify_encoding (haploid) failed: %s", e)
            end = datetime.now()
            logger.info("end: %s (runtime: %.1fs)", end.isoformat(timespec="seconds"),
                        (end - t_wall).total_seconds())
            return 2

        rendered = render_haploid_report(report_dict)
        if args.report:
            args.report.write_text(rendered + "\n")
            logger.info("wrote report to %s", args.report)
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report_dict, indent=2) + "\n")
            logger.info("wrote JSON report to %s", args.report_json)
        print(rendered)

        end = datetime.now()
        logger.info("end: %s (runtime: %.1fs)", end.isoformat(timespec="seconds"),
                    (end - t_wall).total_seconds())
        return 0 if report_dict["passed"] else 1

    try:
        report = verify_sample(
            parquet_path=args.parquet,
            bcf_dir=args.bcf_dir,
            ref_path=args.reference,
            tokens_path=args.tokens,
            sample_id=args.sample_id,
            max_regions=args.max_regions,
            filter_pass_only=args.filter_pass_only,
            max_diffs=args.max_diffs,
        )
    except Exception as e:
        logger.exception("verify_encoding failed: %s", e)
        end = datetime.now()
        logger.info("end: %s (runtime: %.1fs)", end.isoformat(timespec="seconds"),
                    (end - t_wall).total_seconds())
        return 2

    rendered = render_report(report)
    if args.report:
        args.report.write_text(rendered + "\n")
        logger.info("wrote report to %s", args.report)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report_to_json_dict(report), indent=2) + "\n")
        logger.info("wrote JSON report to %s", args.report_json)
    print(rendered)

    end = datetime.now()
    logger.info("end: %s (runtime: %.1fs)", end.isoformat(timespec="seconds"),
                (end - t_wall).total_seconds())

    return 0 if report_passed(report) else 1


if __name__ == "__main__":
    sys.exit(main())
