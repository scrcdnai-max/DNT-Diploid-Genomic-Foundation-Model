# SPDX-FileCopyrightText: Copyright (c) 2026 Sheba Medical Center. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module does not embed third-party data download URLs.
from dataclasses import dataclass
import hashlib
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal, Union
import os

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pyfaidx import Fasta

from gfmbench_api.utils.preprocutils import extract_snv_sequences_centered


_DEPRECATED_ORDER_TERMS = {"ordered": "phased", "unordered": "unphased"}


def _normalize_order_label(order: str, *, source: str) -> str:
    """Accept canonical 'phased'/'unphased'; warn on legacy 'ordered'/'unordered'."""
    replacement = _DEPRECATED_ORDER_TERMS.get(order)
    if replacement is not None:
        warnings.warn(
            f"Diploid order label '{order}' is deprecated "
            f"(in {source}); use '{replacement}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return replacement
    return order

# =============================================================================
# Diploid Tokens Handling Utilities
# =============================================================================
@dataclass(frozen=True)
class DiploidEncodingMap:
    singles: Dict[str, str]
    phased_tok: Dict[str, str]
    unphased_tok: Dict[str, str]
    canonical_fallback: Dict[str, str]
    snp_token: str
    indel_start_token: str
    indel_end_token: str
    gap_token: str

# Controls allele ordering in diploid mode
DIPLOID_HET_HAPLOTYPE_ORDER_DEFAULT = "var_ref" # or "ref_var"
# Diploid-seq order policy: "use_phased" uses phased tokens first, then the canonical fallback; "use_unphased" tries unphased and fails if not found
DIPLOID_PAIR_POLICY = os.environ.get("DIPLOID_PAIR_POLICY", "use_unphased")
# Controls fallback behavior for missing diploid mappings
DIPLOID_CANONICALIZE_FALLBACK = True
# Controls whether to use special <SNP> token
USE_SNP_TOKEN = True
# Controls whether to use special INDEL span tokens (e.g. <INDEL_span_start> and <INDEL_span_end>)
USE_SPECIAL_INDEL_TOKENS = True

INDEL_GAP_TOKEN = "-"

@dataclass(frozen=True)
class DiploidEncodingConfig:
    encoding_map: DiploidEncodingMap
    use_snp_token: bool = USE_SNP_TOKEN
    use_special_indel_tokens: bool = USE_SPECIAL_INDEL_TOKENS
    diploid_pair_policy: str = DIPLOID_PAIR_POLICY
    canonicalize_fallback: bool = DIPLOID_CANONICALIZE_FALLBACK
    het_haplotype_order: Literal["var_ref", "ref_var"] = DIPLOID_HET_HAPLOTYPE_ORDER_DEFAULT


def _canon_pair(a: str, b: str) -> str:
    return a + b if a <= b else b + a

def get_snp_marker_overhead(config: DiploidEncodingConfig) -> int:
    """Number of extra characters added by the optional SNP marker."""
    return 1 if config.use_snp_token else 0


def get_indel_marker_overhead(config: DiploidEncodingConfig) -> int:
    """Number of extra characters added by optional indel span markers."""
    return 2 if config.use_special_indel_tokens else 0


def get_decorated_indel_allele_length(
    allele: str,
    config: DiploidEncodingConfig,
) -> int:
    """Length of an indel allele after optional span-marker decoration."""
    return len(allele) + get_indel_marker_overhead(config)


def insert_snp_token_into_haploid(
    seq: str,
    config: DiploidEncodingConfig,
) -> str:
    """
    Insert the configured SNP marker at the locus position.

    This is a no-op when config.use_snp_token is False.
    """
    if not config.use_snp_token:
        return seq

    center = (len(seq) - 1) // 2
    return seq[:center] + config.encoding_map.snp_token + seq[center:]


def wrap_allele_with_indel_tokens(
    allele: str,
    config: DiploidEncodingConfig,
) -> str:
    """
    Wrap an indel allele payload with configured span markers.

    This is a no-op when config.use_special_indel_tokens is False.
    """
    if not config.use_special_indel_tokens:
        return allele

    return (
        f"{config.encoding_map.indel_start_token}"
        f"{allele}"
        f"{config.encoding_map.indel_end_token}"
    )

def load_diploid_seq_map(tsv_path: str) -> DiploidEncodingMap:
    SNP_TOKEN_ID = "<SNP>"
    INDEL_START_TOKEN_ID = "<INDEL_span_start>"
    INDEL_END_TOKEN_ID = "<INDEL_span_end>"

    # The canonical table (preprocess/diploid_tokens.tsv, and the copy published with
    # the HF datasets) is 10 comment lines then data -- and the header row is ITSELF a
    # comment ("# Pair\tOrder\tSymbol\tUnicode"). Without comment="#" pandas infers one
    # column from line 1 and dies on line 10 with "Expected 1 fields, saw 4", so this
    # loader could not read the very file its own error message calls canonical. Column
    # names are supplied explicitly because comment="#" also eats that header; the
    # layout is pinned by the sha256 check above. No data row contains "#".
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, comment="#",
                     names=["Pair", "Order", "Symbol", "Unicode"])
    # a variant of the file with an UNcommented header would now arrive as a data row
    df = df[df["Pair"] != "Pair"]
    required = {"Pair", "Order", "Symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"diploid TSV missing columns: {sorted(missing)}")

    phased_tok: Dict[str, str] = {}
    unphased_tok: Dict[str, str] = {}
    phased_any_dir: Dict[str, str] = {}
    singles: Dict[str, str] = {}

    snp_token = None
    indel_start_token = None
    indel_end_token = None

    for _, row in df.iterrows():
        pair = str(row["Pair"])
        order = str(row["Order"]).strip()
        sym = str(row["Symbol"])

        if pair == SNP_TOKEN_ID:
            snp_token = sym
            continue

        if pair == INDEL_START_TOKEN_ID:
            indel_start_token = sym
            continue

        if pair == INDEL_END_TOKEN_ID:
            indel_end_token = sym
            continue

        if order == "single":
            if len(pair) == 1:
                singles[pair] = sym
            continue

        if len(pair) != 2:
            continue

        a, b = pair[0], pair[1]
        canon = _canon_pair(a, b)

        order = _normalize_order_label(order, source=str(tsv_path))

        if order == "phased":
            phased_tok[pair] = sym
            phased_any_dir.setdefault(canon, sym)

        elif order in {"unphased", "symmetric"}:
            prev = unphased_tok.get(canon)
            if prev is not None and prev != sym:
                raise ValueError(
                    f"Conflicting unphased tokens for {canon}: {prev} vs {sym}"
                )
            unphased_tok[canon] = sym

    for b in ["A", "C", "G", "T", "-", "N", "*"]:
        if b not in singles:
            raise ValueError(f"diploid TSV missing single mapping for base '{b}'")

    if snp_token is None:
        raise ValueError("diploid TSV missing <SNP> mapping row")
    if indel_start_token is None:
        raise ValueError("diploid TSV missing <INDEL_span_start> mapping row")
    if indel_end_token is None:
        raise ValueError("diploid TSV missing <INDEL_span_end> mapping row")

    return DiploidEncodingMap(
        singles=singles,
        phased_tok=phased_tok,
        unphased_tok=unphased_tok,
        canonical_fallback=phased_any_dir,
        snp_token=snp_token,
        indel_start_token=indel_start_token,
        indel_end_token=indel_end_token,
        gap_token=INDEL_GAP_TOKEN,
    )


def get_default_diploid_encoding_config(
    source: Union[str, DiploidEncodingMap]
) -> DiploidEncodingConfig:
    """
    Build a DiploidEncodingConfig from either:
      - a TSV file path
      - an existing DiploidEncodingMap
    """

    if isinstance(source, DiploidEncodingMap):
        encoding_map = source
    elif isinstance(source, str):
        encoding_map = load_diploid_seq_map(source)
    else:
        raise TypeError(
            "source must be either a TSV path (str) or DiploidEncodingMap"
        )

    return DiploidEncodingConfig(
        encoding_map=encoding_map,
        use_snp_token=USE_SNP_TOKEN,
        use_special_indel_tokens=USE_SPECIAL_INDEL_TOKENS,
        diploid_pair_policy=DIPLOID_PAIR_POLICY,
        canonicalize_fallback=DIPLOID_CANONICALIZE_FALLBACK,
        het_haplotype_order=DIPLOID_HET_HAPLOTYPE_ORDER_DEFAULT,
    )


def _read_indel_content(s: str, pos: int, indel_start: str, indel_end: str) -> Tuple[str, int]:
    if pos >= len(s) or s[pos] != indel_start:
        raise ValueError(f"Expected indel start at position {pos}")
    end = s.find(indel_end, pos + 1)
    if end == -1:
        raise ValueError(f"Unclosed indel starting at position {pos}")
    return s[pos + 1:end], end + 1


def _find_next_indel_start(s: str, start: int, indel_start: str, indel_end: str) -> int:
    """
    Find the next top-level indel_start token in s[start:].
    Assumes no nested indels.
    Returns -1 if none found.
    """
    for i in range(start, len(s)):
        if s[i] == indel_start:
            return i
    return -1


def _align_fixed_window_alleles(
    allele1: str,
    allele2: str,
    *,
    indel_start: str,
    indel_end: str,
    gap_token: str,
    use_special_indel_tokens: bool,
) -> List[Tuple[str, str]]:
    """
    Align two fixed-window allele strings where one or both may contain inline indels.

    Important semantics:
    - Raw allele strings are NOT assumed to be globally aligned.
    - When an indel block is encountered in one allele, the other allele contributes
      ordinary sequence characters to match the expanded block length.
    - The shorter block/segment inside the indel is padded with gap_token.
    - Output follows the expanded representation induced by the indel event(s).
    - Extra trailing raw characters in one input are allowed and ignored once the
      representation is complete.

    Example:
        allele1 = 'C{T}C'
        allele2 = 'C{TCCAG}C'

    use_special_indel_tokens=True =>
        [('C','C'),
         ('{','{'),
         ('T','T'),
         ('-','C'),
         ('-','C'),
         ('-','A'),
         ('-','G'),
         ('}','}'),
         ('C','C')]

    use_special_indel_tokens=False =>
        [('C','C'),
         ('T','T'),
         ('-','C'),
         ('-','C'),
         ('-','A'),
         ('-','G'),
         ('C','C')]
    """
    i = j = 0
    out: List[Tuple[str, str]] = []

    while i < len(allele1) and j < len(allele2):
        a = allele1[i]
        b = allele2[j]

        # Case 1: both alleles have an indel block starting here
        if a == indel_start and b == indel_start:
            content1, next_i = _read_indel_content(allele1, i, indel_start, indel_end)
            content2, next_j = _read_indel_content(allele2, j, indel_start, indel_end)

            if use_special_indel_tokens:
                out.append((indel_start, indel_start))

            max_len = max(len(content1), len(content2))
            content1 = content1.ljust(max_len, gap_token)
            content2 = content2.ljust(max_len, gap_token)
            out.extend(zip(content1, content2))

            if use_special_indel_tokens:
                out.append((indel_end, indel_end))

            i = next_i
            j = next_j
            continue

        # Case 2: only allele1 has an indel block here
        if a == indel_start:
            content1, next_i = _read_indel_content(allele1, i, indel_start, indel_end)

            # Consume ordinary sequence from allele2 to match the expanded indel block.
            next_block_j = _find_next_indel_start(allele2, j, indel_start, indel_end)
            if next_block_j == -1:
                segment2 = allele2[j:]
                next_j = len(allele2)
            else:
                segment2 = allele2[j:next_block_j]
                next_j = next_block_j

            max_len = max(len(content1), len(segment2))
            content1 = content1.ljust(max_len, gap_token)
            segment2 = segment2.ljust(max_len, gap_token)

            if use_special_indel_tokens:
                out.append((indel_start, indel_start))

            out.extend(zip(content1, segment2))

            if use_special_indel_tokens:
                out.append((indel_end, indel_end))

            i = next_i
            j = next_j
            continue

        # Case 3: only allele2 has an indel block here
        if b == indel_start:
            content2, next_j = _read_indel_content(allele2, j, indel_start, indel_end)

            # Consume ordinary sequence from allele1 to match the expanded indel block.
            next_block_i = _find_next_indel_start(allele1, i, indel_start, indel_end)
            if next_block_i == -1:
                segment1 = allele1[i:]
                next_i = len(allele1)
            else:
                segment1 = allele1[i:next_block_i]
                next_i = next_block_i

            max_len = max(len(segment1), len(content2))
            segment1 = segment1.ljust(max_len, gap_token)
            content2 = content2.ljust(max_len, gap_token)

            if use_special_indel_tokens:
                out.append((indel_start, indel_start))

            out.extend(zip(segment1, content2))

            if use_special_indel_tokens:
                out.append((indel_end, indel_end))

            i = next_i
            j = next_j
            continue

        # Case 4: ordinary base-vs-base
        if a == indel_end or b == indel_end:
            raise ValueError("Unexpected indel end token outside indel block")

        out.append((a, b))
        i += 1
        j += 1

    # Do NOT require full raw consumption.
    # Fixed-window inputs may intentionally leave trailing sequence on one side.
    return out

def _align_fixed_window_alleles_by_spans(
    allele1: str,
    allele2: str,
    *,
    span1: Tuple[int, int],
    span2: Tuple[int, int],
    gap_token: str,
) -> List[Tuple[str, str]]:
    """
    Align two fixed-length haplotypes using known allele payload spans.

    This is required for indels when special span tokens are disabled. Without
    explicit span markers in the sequence, the encoder cannot infer where the
    indel event starts and ends from sequence content alone.

    span1/span2 are half-open intervals [start, end) pointing to the allele
    payload in each haplotype.
    """
    start1, end1 = span1
    start2, end2 = span2

    if not (0 <= start1 <= end1 <= len(allele1)):
        raise ValueError(f"Invalid span1={span1} for allele1 length {len(allele1)}")
    if not (0 <= start2 <= end2 <= len(allele2)):
        raise ValueError(f"Invalid span2={span2} for allele2 length {len(allele2)}")

    if start1 != start2:
        raise ValueError(
            f"Expected equal allele starts after haplotype construction, "
            f"got span1={span1}, span2={span2}"
        )

    out: List[Tuple[str, str]] = []

    # Shared left context.
    out.extend(zip(allele1[:start1], allele2[:start2]))

    # Allele payloads, padded to represent insertion/deletion.
    payload1 = allele1[start1:end1]
    payload2 = allele2[start2:end2]
    max_payload_len = max(len(payload1), len(payload2))

    payload1 = payload1.ljust(max_payload_len, gap_token)
    payload2 = payload2.ljust(max_payload_len, gap_token)
    out.extend(zip(payload1, payload2))

    # Shared right context. One haplotype may have more right context because
    # the other spent more length budget on a longer allele.
    suffix1 = allele1[end1:]
    suffix2 = allele2[end2:]
    shared_suffix_len = min(len(suffix1), len(suffix2))
    out.extend(zip(suffix1[:shared_suffix_len], suffix2[:shared_suffix_len]))

    return out

def encode_diploid_seq(
    allele1: str,
    allele2: str,
    config: DiploidEncodingConfig,
    *,
    locus_spans: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> str:
    """
    Encode two haplotypes into a diploid sequence.

    Parameters
    ----------
    allele1, allele2:
        Fixed-window haplotype strings.
    config:
        Diploid encoding configuration.
    locus_spans:
        Optional pair of half-open allele payload spans:
            ((allele1_start, allele1_end), (allele2_start, allele2_end))

        This should be provided for heterozygous indels when special indel
        tokens are disabled. In that mode the sequence itself does not mark
        the indel span, so span-aware alignment is required.
    """
    out = []

    if locus_spans is not None:
        pairs = _align_fixed_window_alleles_by_spans(
            allele1,
            allele2,
            span1=locus_spans[0],
            span2=locus_spans[1],
            gap_token=config.encoding_map.gap_token,
        )
    else:
        pairs = _align_fixed_window_alleles(
            allele1,
            allele2,
            indel_start=config.encoding_map.indel_start_token,
            indel_end=config.encoding_map.indel_end_token,
            gap_token=config.encoding_map.gap_token,
            use_special_indel_tokens=config.use_special_indel_tokens,
        )

    for a, b in pairs:
        if a == config.encoding_map.snp_token and b == config.encoding_map.snp_token:
            out.append(a)
            continue

        if (
            a == config.encoding_map.indel_start_token
            and b == config.encoding_map.indel_start_token
        ):
            out.append(a)
            continue

        if (
            a == config.encoding_map.indel_end_token
            and b == config.encoding_map.indel_end_token
        ):
            out.append(a)
            continue

        if a == b:
            if a.upper() in ("A", "C", "G", "T", "N"):
                out.append(a.upper())
                continue
            tok = config.encoding_map.singles.get(a)
            if tok is not None:
                out.append(tok)
                continue

        canon = a + b if a <= b else b + a

        if config.diploid_pair_policy == "use_phased":
            tok = config.encoding_map.phased_tok.get(a + b)
            if tok is not None:
                out.append(tok)
                continue

            tok = config.encoding_map.canonical_fallback.get(canon)
            if tok is not None and config.canonicalize_fallback:
                out.append(tok)
                continue

            raise KeyError(
                f"No token for pair {a}{b} (canon {canon}) "
                f"with pair policy {config.diploid_pair_policy}"
            )

        if config.diploid_pair_policy == "use_unphased":
            tok = config.encoding_map.unphased_tok.get(canon)
            if tok is not None:
                out.append(tok)
                continue

            raise KeyError(
                f"No token for pair {a}{b} (canon {canon}) "
                f"with pair policy {config.diploid_pair_policy}"
            )

        raise ValueError("pair_policy must be 'use_phased' or 'use_unphased'")

    return "".join(out)

# =============================================================================
# Row selection for centered SNV extraction
# =============================================================================
#
# `extract_snv_sequences_centered` reports how many rows it skipped but not which ones,
# so its output cannot be paired back with the frame it was given -- and the diploid
# encoders need that pairing to read each row's zygosity. Instead of forking the upstream
# extractor to hand back the kept indices, drop every row it would skip *before* calling
# it: the output is then 1:1 with the frame that went in, positionally.
#
# The filter also enforces one rule of its own. Upstream pads a window that crosses a
# chromosome edge with 'P' and keeps the variant; there is no diploid token for a 'P'/'P'
# pair, so encode_diploid_seq() would raise KeyError on such a row. Those rows are dropped
# here, which is also what the previous forked extractor did.

_PAD_CHAR = "P"


def _resolve_chromosome(genome: Any, chrom: Any) -> Optional[Any]:
    """
    Return the reference record for `chrom`, or None if the genome has no such sequence.

    Tries the 'chr'-prefixed name first and then the bare one, the order
    extract_snv_sequences_centered() uses.
    """
    chrom_str = str(chrom)
    keys = [chrom_str] if chrom_str.startswith("chr") else [f"chr{chrom_str}", chrom_str]
    for key in keys:
        try:
            return genome[key]
        except KeyError:
            continue
    return None


def select_extractable_snv_rows(
    df: pd.DataFrame,
    genome: Any,
    seq_len: int,
    *,
    chrom_col: str = "chrom",
    pos_col: str = "pos",
    ref_col: str = "ref",
    alt_col: str = "alt",
    split_name: str = "",
) -> pd.DataFrame:
    """
    Keep only the rows extract_snv_sequences_centered() will emit, in the frame's own order.

    Dropped rows are the ones the upstream extractor skips -- chromosome absent from the
    reference, non-SNV alleles, unparseable position, reference allele disagreeing with the
    genome -- plus windows that would need edge padding, which the diploid encoding cannot
    represent.

    Returns:
        A copy of `df` (index reset) holding the extractable rows.
    """
    frame = df.reset_index(drop=True)
    if frame.empty:
        return frame

    flank = (seq_len - 1) // 2
    keep = np.zeros(len(frame), dtype=bool)
    dropped: Dict[str, int] = {}

    def _drop(reason: str, n: int = 1) -> None:
        if n:
            dropped[reason] = dropped.get(reason, 0) + n

    positions = pd.to_numeric(frame[pos_col], errors="coerce")
    ref_alleles = frame[ref_col].astype(str).str.upper()
    alt_alleles = frame[alt_col].astype(str).str.upper()
    is_snv = (ref_alleles.str.len() == 1) & (alt_alleles.str.len() == 1)

    for chrom, group in frame.groupby(chrom_col, sort=False):
        record = _resolve_chromosome(genome, chrom)
        if record is None:
            _drop("chromosome not in reference genome", len(group))
            continue

        chrom_len = len(record)
        for i in group.index:
            if pd.isna(positions.iat[i]):
                _drop("unparseable position")
                continue
            if not is_snv.iat[i]:
                _drop("not an SNV (multi-base ref/alt)")
                continue

            variant_pos_0based = int(positions.iat[i]) - 1
            window_start = variant_pos_0based - flank
            if window_start < 0 or window_start + seq_len > chrom_len:
                _drop("window crosses a chromosome edge (would need padding)")
                continue

            actual_ref = str(record[variant_pos_0based:variant_pos_0based + 1]).upper()
            if actual_ref != ref_alleles.iat[i]:
                _drop("reference allele does not match the genome")
                continue

            keep[i] = True

    n_dropped = len(frame) - int(keep.sum())
    if n_dropped:
        detail = ", ".join(f"{count} {reason}" for reason, count in sorted(dropped.items()))
        logging.warning(
            "[%s] Dropping %d/%d variants before extraction: %s",
            split_name, n_dropped, len(frame), detail,
        )

    return frame[keep].reset_index(drop=True)


def _verify_window_alignment(
    kept_df: pd.DataFrame,
    variant_sequences: List[str],
    reference_sequences: List[str],
    seq_len: int,
    split_name: str,
    ref_col: str,
    alt_col: str,
) -> None:
    """
    Fail loudly if extracted windows stopped lining up with the rows they came from.

    The centre base of each window is the row's ref (reference window) and alt (variant
    window) allele, so a shifted or reordered result is caught here rather than silently
    pairing sequences with another row's zygosity.
    """
    flank = (seq_len - 1) // 2
    observed_ref = np.array([seq[flank] for seq in reference_sequences])
    observed_alt = np.array([seq[flank] for seq in variant_sequences])
    expected_ref = kept_df[ref_col].astype(str).str.upper().to_numpy()
    expected_alt = kept_df[alt_col].astype(str).str.upper().to_numpy()

    mismatched = np.flatnonzero((observed_ref != expected_ref) | (observed_alt != expected_alt))
    if mismatched.size:
        first = int(mismatched[0])
        raise RuntimeError(
            f"[{split_name}] Extracted windows are not aligned with their rows: "
            f"{mismatched.size} of {len(kept_df)} mismatch, first at row {first} "
            f"(expected ref/alt {expected_ref[first]}/{expected_alt[first]}, "
            f"window centre {observed_ref[first]}/{observed_alt[first]})"
        )


def extract_centered_snv_windows(
    df: pd.DataFrame,
    genome: Any,
    seq_len: int,
    split_name: str,
    *,
    chrom_col: str = "chrom",
    pos_col: str = "pos",
    ref_col: str = "ref",
    alt_col: str = "alt",
    label_col: str = "label",
) -> Tuple[List[str], List[str], List[int], pd.DataFrame]:
    """
    Extract centered SNV windows, keeping the results aligned with their source rows.

    Pre-filters with select_extractable_snv_rows() and then calls the upstream extractor
    once per chromosome, walking chromosomes in the sorted order it groups by, so the
    concatenated output keeps the row order a single call would have produced. Output row
    i belongs to kept_df.iloc[i].

    Returns:
        (variant_sequences, reference_sequences, labels, kept_df)
    """
    kept_df = select_extractable_snv_rows(
        df,
        genome,
        seq_len,
        chrom_col=chrom_col,
        pos_col=pos_col,
        ref_col=ref_col,
        alt_col=alt_col,
        split_name=split_name,
    )
    if kept_df.empty:
        return [], [], [], kept_df

    variant_sequences: List[str] = []
    reference_sequences: List[str] = []
    labels: List[int] = []
    groups: List[pd.DataFrame] = []

    chrom_keys = kept_df[chrom_col].astype(str)
    for chrom in sorted(chrom_keys.unique()):
        group = kept_df[chrom_keys == chrom]
        var_seqs, ref_seqs, group_labels, skipped = extract_snv_sequences_centered(
            df=group,
            genome=genome,
            max_sequence_length=seq_len,
            chrom_col=chrom_col,
            pos_col=pos_col,
            ref_col=ref_col,
            alt_col=alt_col,
            label_col=label_col,
            use_logging=True,
        )
        if skipped or len(group_labels) != len(group):
            raise RuntimeError(
                f"[{split_name}] extract_snv_sequences_centered() skipped {skipped} of "
                f"{len(group)} pre-validated rows on chromosome {chrom} "
                f"({len(group_labels)} returned): select_extractable_snv_rows() is out of "
                "sync with the upstream extractor and must be updated to match it"
            )
        variant_sequences.extend(var_seqs)
        reference_sequences.extend(ref_seqs)
        labels.extend(group_labels)
        groups.append(group)

    kept_df = pd.concat(groups).reset_index(drop=True)
    _verify_window_alignment(
        kept_df, variant_sequences, reference_sequences, seq_len, split_name, ref_col, alt_col
    )
    return variant_sequences, reference_sequences, labels, kept_df


def extract_variant_ref_diploid_sequences(
    df: pd.DataFrame,
    genome: Any,
    seq_len: int,
    split_name: str,
    config: DiploidEncodingConfig,
) -> List[Tuple[str, str, torch.Tensor, np.ndarray]]:
    """
    Extract zygosity-aware diploid SNV variant/reference sequence pairs.

    Final encoded variant and reference sequences are exactly seq_len characters.
    If the SNP marker is enabled, raw haploid windows are extracted with length
    seq_len - 1 and then decorated back to seq_len.
    """
    if df.empty:
        logging.warning(f"Empty dataframe for split '{split_name}'")
        return []

    marker_overhead = get_snp_marker_overhead(config)
    raw_seq_len = seq_len - marker_overhead
    if raw_seq_len <= 0:
        raise ValueError(
            f"seq_len={seq_len} is too short for SNP marker overhead "
            f"{marker_overhead}"
        )

    variant_sequences, reference_sequences, labels, kept_df = extract_centered_snv_windows(
        df, genome, raw_seq_len, split_name
    )

    if len(labels) == 0:
        logging.warning(f"No valid sequences extracted for split '{split_name}'")
        return []

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    dataset: List[Tuple[str, str, torch.Tensor, np.ndarray]] = []
    empty_cond = np.array([], dtype=np.float32)

    for i, row in kept_df.iterrows():
        var_seq = insert_snp_token_into_haploid(variant_sequences[i], config)
        ref_seq = insert_snp_token_into_haploid(reference_sequences[i], config)
        label = labels_tensor[i]

        if len(var_seq) != seq_len or len(ref_seq) != seq_len:
            raise ValueError(
                f"[{split_name}] Decorated SNV length mismatch at row {i}: "
                f"var={len(var_seq)}, ref={len(ref_seq)}, expected={seq_len}"
            )

        # If no zygosity info is available, fallback to homozygous variants.
        z = str(row.get("zygosity", "")).lower()
        is_het = "het" in z and "hom" not in z

        ref_dip = encode_diploid_seq(ref_seq, ref_seq, config)

        if is_het:
            if config.het_haplotype_order == "var_ref":
                allele1, allele2 = var_seq, ref_seq
            else:
                allele1, allele2 = ref_seq, var_seq

            var_dip = encode_diploid_seq(allele1, allele2, config)
        else:
            var_dip = encode_diploid_seq(var_seq, var_seq, config)

        if len(var_dip) != seq_len or len(ref_dip) != seq_len:
            raise ValueError(
                f"[{split_name}] Encoded SNV length mismatch at row {i}: "
                f"var={len(var_dip)}, ref={len(ref_dip)}, expected={seq_len}"
            )

        dataset.append((var_dip, ref_dip, label, empty_cond))

    logging.info(
        f"[{split_name}] Built {len(dataset)} samples from {len(kept_df)} kept variants "
        f"(avg length: {np.mean([len(d[0]) for d in dataset]):.0f}bp)"
    )

    return dataset

def extract_variant_ref_homozygous_diploid_sequences(
    df: pd.DataFrame,
    genome: Any,
    seq_len: int,
    split_name: str,
    config: DiploidEncodingConfig,
) -> List[Tuple[str, str, torch.Tensor, np.ndarray]]:
    """
    Extract homozygous diploid SNV variant/reference sequence pairs.

    Final encoded variant and reference sequences are exactly seq_len characters.
    If the SNP marker is enabled, raw haploid windows are extracted with length
    seq_len - 1 and then decorated back to seq_len.

    For every SNV:
        reference sequence = REF / REF
        variant sequence   = ALT / ALT
    """
    if df.empty:
        logging.warning(f"Empty dataframe for split '{split_name}'")
        return []

    marker_overhead = get_snp_marker_overhead(config)
    raw_seq_len = seq_len - marker_overhead
    if raw_seq_len <= 0:
        raise ValueError(
            f"seq_len={seq_len} is too short for SNP marker overhead "
            f"{marker_overhead}"
        )

    variant_sequences, reference_sequences, labels, kept_df = extract_centered_snv_windows(
        df, genome, raw_seq_len, split_name
    )

    if len(labels) == 0:
        logging.warning(f"No valid sequences extracted for split '{split_name}'")
        return []

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    dataset: List[Tuple[str, str, torch.Tensor, np.ndarray]] = []
    empty_cond = np.array([], dtype=np.float32)

    for i in range(len(labels_tensor)):
        var_seq = insert_snp_token_into_haploid(variant_sequences[i], config)
        ref_seq = insert_snp_token_into_haploid(reference_sequences[i], config)
        label = labels_tensor[i]

        if len(var_seq) != seq_len or len(ref_seq) != seq_len:
            raise ValueError(
                f"[{split_name}] Decorated SNV length mismatch at row {i}: "
                f"var={len(var_seq)}, ref={len(ref_seq)}, expected={seq_len}"
            )

        ref_dip = encode_diploid_seq(ref_seq, ref_seq, config)
        var_dip = encode_diploid_seq(var_seq, var_seq, config)

        if len(var_dip) != seq_len or len(ref_dip) != seq_len:
            raise ValueError(
                f"[{split_name}] Encoded SNV length mismatch at row {i}: "
                f"var={len(var_dip)}, ref={len(ref_dip)}, expected={seq_len}"
            )

        dataset.append((var_dip, ref_dip, label, empty_cond))

    logging.info(
        f"[{split_name}] Built {len(dataset)} homozygous diploid samples "
        f"from {len(kept_df)} kept variants "
        f"(avg length: {np.mean([len(d[0]) for d in dataset]):.0f}bp)"
    )

    return dataset

def build_fixed_length_haplotype(
    chrom_seq: str,
    pos_0: int,
    ref_allele: str,
    allele_for_haplotype: str,
    left_context_len: int,
    diploid_encoding_config: DiploidEncodingConfig,
    max_sequence_length: int,
) -> Tuple[str, int]:
    """
    Build a single fixed-length haplotype window.

    Returns
    -------
    hap:
        Final haplotype of exactly max_sequence_length.
    allele_start:
        Zero-based start index of the allele payload in the final haplotype.
        If special indel tokens are enabled, this points to the first base
        inside the tokenized span, not to the start-token.
    """
    allele_block = wrap_allele_with_indel_tokens(
        allele_for_haplotype,
        diploid_encoding_config,
    )
    right_context_len = max_sequence_length - left_context_len - len(allele_block)

    if right_context_len < 0:
        raise ValueError(
            f"Allele block length {len(allele_block)} exceeds available sequence "
            f"budget for max_sequence_length={max_sequence_length} "
            f"with left_context_len={left_context_len}"
        )

    # Track the actual left boundary currently represented by left_seq.
    current_left_start = max(0, pos_0 - left_context_len)
    left_seq = chrom_seq[current_left_start:pos_0]

    # Compensate on the right if near chromosome start.
    if len(left_seq) < left_context_len:
        missing_left = left_context_len - len(left_seq)
        right_context_len += missing_left

    ref_end = pos_0 + len(ref_allele)
    right_seq = chrom_seq[ref_end:ref_end + right_context_len]

    # Compensate on the left if near chromosome end.
    if len(right_seq) < right_context_len:
        missing_right = right_context_len - len(right_seq)
        current_left_start = max(0, current_left_start - missing_right)
        left_seq = chrom_seq[current_left_start:pos_0]

    hap = left_seq + allele_block + right_seq

    # Final repair to exact length: first extend to the right if possible.
    if len(hap) < max_sequence_length:
        need = max_sequence_length - len(hap)
        extra_right_start = ref_end + len(right_seq)
        hap += chrom_seq[extra_right_start:extra_right_start + need]

    # Then extend to the left using the true current left boundary.
    prepended_left = 0
    if len(hap) < max_sequence_length:
        need = max_sequence_length - len(hap)
        extra_left_end = current_left_start
        extra_left_start = max(0, extra_left_end - need)
        prefix = chrom_seq[extra_left_start:extra_left_end]
        hap = prefix + hap
        prepended_left = len(prefix)

    if len(hap) != max_sequence_length:
        raise ValueError(
            f"Built haplotype length {len(hap)} != expected {max_sequence_length}"
        )

    allele_start = prepended_left + len(left_seq)
    if diploid_encoding_config.use_special_indel_tokens:
        allele_start += 1

    return hap, allele_start

def extract_diploid_indel_sequences(
    df: pd.DataFrame,
    genome: Fasta,
    diploid_encoding_config: DiploidEncodingConfig,
    max_sequence_length: int = 1048576,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Build diploid-encoded indel variant/reference sequence pairs.

    Final encoded variant and reference sequences are exactly max_sequence_length.
    When special indel tokens are disabled, heterozygous ref/alt alignment uses
    explicit allele payload spans instead of token scanning.
    """
    variant_sequences: List[str] = []
    reference_sequences: List[str] = []
    labels: List[int] = []
    variant_positions: List[int] = []

    skip_counts = {
        "missing_chromosome": 0,
        "reference_mismatch": 0,
        "allele_too_long": 0,
    }

    df = df.reset_index(drop=True)

    for chrom, group in df.groupby("chrom", sort=False):
        try:
            chrom_str = f"chr{chrom}" if not str(chrom).startswith("chr") else str(chrom)
            chrom_seq = str(genome[chrom_str][:]).upper()
        except KeyError:
            logging.warning(
                f"Chromosome {chrom} missing from reference genome; "
                f"skipping {len(group)} rows"
            )
            skip_counts["missing_chromosome"] += len(group)
            continue

        for _, row in group.iterrows():
            pos = int(row["pos"])
            pos_0 = pos - 1
            ref_allele = str(row["ref"]).upper()
            alt_allele = str(row["alt"]).upper()
            zygosity = str(row.get("zygosity", "hom")).lower()
            label = int(row["label"])

            genome_ref = chrom_seq[pos_0:pos_0 + len(ref_allele)]
            if genome_ref != ref_allele:
                logging.debug(
                    f"Reference mismatch at {chrom}:{pos} "
                    f"expected={ref_allele} got={genome_ref}"
                )
                skip_counts["reference_mismatch"] += 1
                continue

            longer_allele_block_len = max(
                get_decorated_indel_allele_length(ref_allele, diploid_encoding_config),
                get_decorated_indel_allele_length(alt_allele, diploid_encoding_config),
            )
            center = max_sequence_length // 2

            if longer_allele_block_len > max_sequence_length:
                logging.debug(
                    f"Variant at {chrom}:{pos} skipped: allele block length "
                    f"{longer_allele_block_len} exceeds max_sequence_length="
                    f"{max_sequence_length}"
                )
                skip_counts["allele_too_long"] += 1
                continue

            shared_left_context_len = min(
                center,
                max_sequence_length - longer_allele_block_len,
            )

            ref_h, allele_start_ref = build_fixed_length_haplotype(
                chrom_seq=chrom_seq,
                pos_0=pos_0,
                ref_allele=ref_allele,
                allele_for_haplotype=ref_allele,
                left_context_len=shared_left_context_len,
                diploid_encoding_config=diploid_encoding_config,
                max_sequence_length=max_sequence_length,
            )
            alt_h, allele_start_alt = build_fixed_length_haplotype(
                chrom_seq=chrom_seq,
                pos_0=pos_0,
                ref_allele=ref_allele,
                allele_for_haplotype=alt_allele,
                left_context_len=shared_left_context_len,
                diploid_encoding_config=diploid_encoding_config,
                max_sequence_length=max_sequence_length,
            )

            if allele_start_ref != allele_start_alt:
                raise ValueError(
                    f"Inconsistent allele start positions at {chrom}:{pos}: "
                    f"ref={allele_start_ref}, alt={allele_start_alt}"
                )

            allele_start = allele_start_ref

            ref_payload_span = (
                allele_start_ref,
                allele_start_ref + len(ref_allele),
            )
            alt_payload_span = (
                allele_start_alt,
                allele_start_alt + len(alt_allele),
            )

            ref_dip = encode_diploid_seq(
                ref_h,
                ref_h,
                diploid_encoding_config,
            )

            is_het = "het" in zygosity and "hom" not in zygosity

            if is_het:
                if diploid_encoding_config.het_haplotype_order == "var_ref":
                    var_h1, var_h2 = alt_h, ref_h
                    span1, span2 = alt_payload_span, ref_payload_span
                else:
                    var_h1, var_h2 = ref_h, alt_h
                    span1, span2 = ref_payload_span, alt_payload_span

                if diploid_encoding_config.use_special_indel_tokens:
                    var_dip = encode_diploid_seq(
                        var_h1,
                        var_h2,
                        diploid_encoding_config,
                    )
                else:
                    var_dip = encode_diploid_seq(
                        var_h1,
                        var_h2,
                        diploid_encoding_config,
                        locus_spans=(span1, span2),
                    )
            else:
                var_dip = encode_diploid_seq(
                    alt_h,
                    alt_h,
                    diploid_encoding_config,
                )

            if len(ref_dip) != max_sequence_length or len(var_dip) != max_sequence_length:
                raise ValueError(
                    f"Encoded indel length mismatch at {chrom}:{pos}: "
                    f"var={len(var_dip)}, ref={len(ref_dip)}, "
                    f"expected={max_sequence_length}"
                )

            variant_sequences.append(var_dip)
            reference_sequences.append(ref_dip)
            labels.append(label)
            variant_positions.append(allele_start)

    total_skipped = sum(skip_counts.values())
    if total_skipped > 0:
        logging.info(
            "Skipped %d indels during sequence extraction: %s",
            total_skipped,
            ", ".join(f"{k}={v}" for k, v in skip_counts.items() if v > 0),
        )

    return variant_sequences, reference_sequences, labels, variant_positions
# =============================================================================
# Deterministic but fast flips
# =============================================================================


def _seed64_from_str(s: str) -> int:
    d = hashlib.sha1(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little", signed=False)


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def _det_bit(seed64: int, pos: int, sym_ord: int) -> int:
    x = seed64 ^ (pos * 0xD1342543DE82EF95) ^ (sym_ord * 0x9E3779B185EBCA87)
    return _splitmix64(x) & 1


def default_seed64_from_variant(variant_seq: str, sample_id: Optional[str] = None) -> int:
    """Index-free deterministic seed."""
    if sample_id is not None:
        return _seed64_from_str(str(sample_id))
    return _seed64_from_str(variant_seq)


# =============================================================================
# Tokenization helpers (needed to handle <SNP> and INDEL span tokens)
# =============================================================================

def _build_decode_maps_from_seq_map(
    singles: Dict[str, str],
    phased_tok: Dict[str, str],
    unphased_tok: Dict[str, str],
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]], set]:
    """
    Build decoder maps compatible with the old decode API, but derived from load_diploid_seq_map() output.

    Returns:
      var_map: symbol/token -> (a,b)
        - unphased symbols map to canonical sorted pair (a<=b)
        - phased symbols map to (a,b) as specified
        - single symbols map to (base,base)
      ref_map: symbol/token -> (a,a) for homozygous symbols (plus singles)
      unphased_syms: set(symbol/token) that are unphased
    """
    var_map: Dict[str, Tuple[str, str]] = {}
    unphased_syms: set = set()

    def _assign(sym: str, pair: Tuple[str, str], *, source: str) -> None:
        prev = var_map.get(sym)
        if prev is not None and prev != pair:
            prev_is_gap = INDEL_GAP_TOKEN in prev
            new_is_gap = INDEL_GAP_TOKEN in pair
            if prev_is_gap and not new_is_gap:
                return  # keep the gap spelling the encoder emits
            if not prev_is_gap and not new_is_gap:
                logging.warning(
                    "diploid symbol %r is shared by unrelated %s pairs %s and %s; "
                    "decoding will use %s",
                    sym, source, prev, pair, pair,
                )
        var_map[sym] = pair

    # Singles: base -> sym  ==> reverse to sym -> (base,base)
    for base, sym in singles.items():
        _assign(sym, (base, base), source="single")

    # Phased: "AB" -> sym  ==> reverse to sym -> (A,B)
    for pair, sym in phased_tok.items():
        if len(pair) == 2:
            _assign(sym, (pair[0], pair[1]), source="phased")

    # Unphased: canon("AB") -> sym  ==> reverse to sym -> (canon[0], canon[1])
    for canon, sym in unphased_tok.items():
        if len(canon) == 2:
            _assign(sym, (canon[0], canon[1]), source="unphased")
            unphased_syms.add(sym)

    # ref_map = homozygous only
    ref_map: Dict[str, Tuple[str, str]] = {}
    for sym, (a, b) in var_map.items():
        if a == b:
            ref_map[sym] = (a, b)

    return var_map, ref_map, unphased_syms


def _greedy_tokenize(seq: str, multi_tokens: List[str]) -> List[str]:
    """
    Tokenize a sequence string into tokens, supporting multi-character tokens via greedy matching.
    - If seq contains whitespace, we assume it's already tokenized: split().
    - Otherwise, greedily match any known multi_tokens (sorted desc by length); fallback to single char.
    """
    if not seq:
        return []
    if any(c.isspace() for c in seq):
        return seq.split()

    if not multi_tokens:
        return list(seq)

    toks: List[str] = []
    i = 0
    n = len(seq)

    while i < n:
        matched = None
        for t in multi_tokens:
            if seq.startswith(t, i):
                matched = t
                break
        if matched is not None:
            toks.append(matched)
            i += len(matched)
        else:
            toks.append(seq[i])
            i += 1
    return toks


def _render_tokens_like_input(original: str, tokens: List[str]) -> str:
    """
    Try to preserve the input representation:
      - if original had whitespace -> join with single spaces
      - else -> concatenate
    """
    if any(c.isspace() for c in original):
        return " ".join(tokens)
    return "".join(tokens)


# =============================================================================
# Decoding (token-aware)
# =============================================================================


GAP_POLICIES = ("drop", "keep")


def _het_blocks(
    encoded_tokens: List[str],
    var_map: Dict[str, Tuple[str, str]],
    unphased_syms: set,
    *,
    indel_start_token: str,
    indel_end_token: str,
) -> List[List[int]]:
    """
    Group heterozygous token positions into *phase blocks*.

    A block is the set of positions that must be assigned to the same haplotype. Phase is
    a property of an allele, not of a base: an indel is one event spanning many
    consecutive positions, so all of its het positions belong to a single block.

    Blocking rules:
      - every het position inside an <INDEL_span_start>..<INDEL_span_end> span joins the
        one block for that span (even when homozygous positions are interleaved, which
        happens whenever ref and alt share bases inside the padded payload);
      - outside spans, maximal runs of consecutive het positions form one block, so an
        MNV-like run is phased as a unit.
    """
    blocks: List[List[int]] = []
    span_block: Optional[int] = None
    in_span = False
    prev_het = False

    for i, tok in enumerate(encoded_tokens):
        if tok == indel_start_token:
            in_span, span_block, prev_het = True, None, False
            continue
        if tok == indel_end_token:
            in_span, span_block, prev_het = False, None, False
            continue

        pair = var_map.get(tok)
        is_het = tok in unphased_syms and pair is not None and pair[0] != pair[1]

        if not is_het:
            if not in_span:
                prev_het = False
            continue

        if in_span:
            if span_block is None:
                blocks.append([])
                span_block = len(blocks) - 1
            blocks[span_block].append(i)
        else:
            if not prev_het:
                blocks.append([])
            blocks[-1].append(i)
            prev_het = True

    return blocks


def _span_bounds(
    tokens: List[str], indel_start_token: str, indel_end_token: str
) -> Optional[Tuple[int, int]]:
    """(open, close) token indices of the first indel span, or None if unmarked."""
    try:
        start = tokens.index(indel_start_token)
        end = tokens.index(indel_end_token, start + 1)
    except ValueError:
        return None
    return start, end


def _align_reference_chars(
    variant_tokens: List[str],
    ref_tokens: List[str],
    ref_map: Dict[str, Tuple[str, str]],
    *,
    snp_token: str,
    indel_start_token: str,
    indel_end_token: str,
) -> Tuple[List[Optional[str]], Optional[bool]]:
    """
    Reference base per VARIANT token index, plus which haplotype the alignment gaps belong to.

    The two streams are not column-aligned in general. Both are exactly
    max_sequence_length long and share their left context, but the variant payload occupies
    max(len(ref), len(alt)) columns while the reference payload occupies len(ref): for an
    insertion the variant span is wider, and everything after it is shifted. Comparing the
    streams index-by-index therefore reads the wrong reference base past the span. Align on
    the indel span markers instead.

    The second return value is True when a gap belongs to haplotype 1, False when it
    belongs to haplotype 2, None when undetermined:
      - equal payload widths  -> the variant allele was padded, so gaps are the alt side;
      - wider variant payload -> the reference allele was padded, so gaps are the ref side.
    This is what orients a heterozygous insertion, where the padded columns have no
    reference base to compare against at all.
    """
    ref_chars_raw: List[Optional[str]] = []
    for tok in ref_tokens:
        if tok in (snp_token, indel_start_token, indel_end_token):
            ref_chars_raw.append(None)
            continue
        pair = ref_map.get(tok)
        ref_chars_raw.append(pair[0] if pair is not None else None)

    v_span = _span_bounds(variant_tokens, indel_start_token, indel_end_token)
    r_span = _span_bounds(ref_tokens, indel_start_token, indel_end_token)

    # No indel span (e.g. the SNV path): the streams are column-aligned by construction.
    if v_span is None or r_span is None:
        if len(ref_tokens) == len(variant_tokens):
            return ref_chars_raw, None
        return [None] * len(variant_tokens), None

    v_start, v_end = v_span
    r_start, r_end = r_span
    if v_start != r_start:
        logging.warning(
            "indel span starts disagree (variant %d vs reference %d); decoding phase "
            "without reference orientation", v_start, r_start,
        )
        return [None] * len(variant_tokens), None

    out: List[Optional[str]] = [None] * len(variant_tokens)

    # Shared left context.
    for i in range(v_start):
        out[i] = ref_chars_raw[i]

    # Allele payloads, aligned from the left edge of the span.
    v_payload = v_end - v_start - 1
    r_payload = r_end - r_start - 1
    for k in range(min(v_payload, r_payload)):
        out[v_start + 1 + k] = ref_chars_raw[r_start + 1 + k]

    # Right context, shifted by the payload width difference.
    for k in range(len(variant_tokens) - (v_end + 1)):
        j = r_end + 1 + k
        if j >= len(ref_chars_raw):
            break
        out[v_end + 1 + k] = ref_chars_raw[j]

    gap_on_hap1: Optional[bool] = None
    if v_payload == r_payload:
        gap_on_hap1 = True
    elif v_payload > r_payload:
        gap_on_hap1 = False

    return out, gap_on_hap1


def _orient_block(
    block: List[int],
    encoded_tokens: List[str],
    var_map: Dict[str, Tuple[str, str]],
    ref_chars: Optional[List[Optional[str]]],
    seed64: int,
    gap_on_hap1: Optional[bool] = None,
) -> bool:
    """
    Decide whether a phase block's (x, y) must be swapped.

    Returns True when haplotype 1 should take y -- i.e. when x is the side that matches the
    reference. Orientation is evidence-based, so the common het case needs no randomness at
    all: haplotype 1 becomes the variant-bearing allele and haplotype 2 the reference one.
    Two independent sources vote:

      - reference bases, wherever the aligned reference has a plain base (SNVs, and the
        overlapping part of an indel payload);
      - the gap side, for padded columns that have no reference base to compare against --
        the padded allele is the shorter one, which `_align_reference_chars` identifies.

    The deterministic bit is the last resort: compound hets where neither side is the
    reference, and callers that pass no reference at all.
    """
    keep_votes = swap_votes = 0

    for i in block:
        pair = var_map.get(encoded_tokens[i])
        if pair is None:
            continue
        x, y = pair

        r = ref_chars[i] if ref_chars is not None and i < len(ref_chars) else None
        if r is not None:
            if y == r and x != r:
                keep_votes += 1
                continue
            if x == r and y != r:
                swap_votes += 1
                continue

        if gap_on_hap1 is None:
            continue
        x_is_gap = x == INDEL_GAP_TOKEN
        y_is_gap = y == INDEL_GAP_TOKEN
        if x_is_gap == y_is_gap:
            continue
        # haplotype 1 receives x when the block is not swapped
        gap_lands_on_hap1 = x_is_gap
        if gap_lands_on_hap1 == gap_on_hap1:
            keep_votes += 1
        else:
            swap_votes += 1

    if keep_votes != swap_votes:
        return swap_votes > keep_votes

    i0 = block[0]
    tok = encoded_tokens[i0]
    return bool(_det_bit(seed64, i0, ord(tok[0]) if tok else 0))


def decode_variant_tokens_to_two_haploids(
    encoded_tokens: List[str],
    var_map: Dict[str, Tuple[str, str]],
    unphased_syms: set,
    seed64: int,
    *,
    snp_token: str,
    indel_start_token: str,
    indel_end_token: str,
    snp_haploid_char: Optional[str] = None,
    ref_tokens: Optional[List[str]] = None,
    ref_map: Optional[Dict[str, Tuple[str, str]]] = None,
    gap_policy: str = "drop",
) -> Tuple[List[str], List[str]]:
    """
    Split a diploid token stream into its two haplotypes.

    Phase is resolved per block (see _het_blocks) and, when ref_tokens/ref_map are
    supplied, oriented against the reference so haplotype 1 carries the variant allele and
    haplotype 2 the reference allele.

    Special tokens:
      - snp_token: passed through unchanged by default. Set snp_haploid_char to substitute
        a base instead; the old default of "N" injected an uninformative base into every
        haploid SNV sequence while the diploid sequence kept a trained token.
      - indel span start/end tokens: passed through to both haploids unchanged.

    gap_policy:
      - "drop" (default): alignment gaps are removed, so each haplotype is a real,
        ungapped allele. Haplotypes then differ in length by the indel size, which is what
        a haploid representation of an indel genuinely looks like.
      - "keep": gaps are kept as INDEL_GAP_TOKEN, preserving column alignment and equal
        lengths. Use for length-sensitive scores, but note a gapped row is not a haplotype.

    Residual limitation: the diploid encoding pads the shorter allele instead of extending
    its flank, so the flanking bases build_fixed_length_haplotype() pushed past the window
    edge cannot be recovered here. A "drop" haplotype is a correct allele but is shorter
    than the same allele extracted directly from the genome. Tasks needing exact windows
    should hand their real haplotypes to the evaluator instead of re-deriving them.
    """
    if gap_policy not in GAP_POLICIES:
        raise ValueError(f"gap_policy must be one of {GAP_POLICIES}, got {gap_policy!r}")

    ref_chars: Optional[List[Optional[str]]] = None
    gap_on_hap1: Optional[bool] = None
    if ref_tokens is not None and ref_map is not None:
        ref_chars, gap_on_hap1 = _align_reference_chars(
            encoded_tokens,
            ref_tokens,
            ref_map,
            snp_token=snp_token,
            indel_start_token=indel_start_token,
            indel_end_token=indel_end_token,
        )

    swap_at: Dict[int, bool] = {}
    for block in _het_blocks(
        encoded_tokens,
        var_map,
        unphased_syms,
        indel_start_token=indel_start_token,
        indel_end_token=indel_end_token,
    ):
        swap = _orient_block(
            block, encoded_tokens, var_map, ref_chars, seed64, gap_on_hap1
        )
        for i in block:
            swap_at[i] = swap

    a1: List[str] = []
    a2: List[str] = []
    vget = var_map.get

    for i, tok in enumerate(encoded_tokens):
        if tok == snp_token:
            ch = tok if snp_haploid_char is None else snp_haploid_char
            a1.append(ch)
            a2.append(ch)
            continue

        if tok == indel_start_token or tok == indel_end_token:
            a1.append(tok)
            a2.append(tok)
            continue

        pair = vget(tok)
        if pair is None:
            raise KeyError(f"Unknown diploid token {tok!r} at token-index {i}")

        x, y = pair
        if swap_at.get(i, False):
            x, y = y, x

        a1.append(x)
        a2.append(y)

    if gap_policy == "drop":
        a1 = [t for t in a1 if t != INDEL_GAP_TOKEN]
        a2 = [t for t in a2 if t != INDEL_GAP_TOKEN]

    return a1, a2


def collapse_homozygous_ref_tokens(
    ref_tokens: List[str],
    ref_map: Dict[str, Tuple[str, str]],
    *,
    snp_token: str,
    indel_start_token: str,
    indel_end_token: str,
    snp_haploid_char: Optional[str] = None,
    gap_policy: str = "drop",
) -> List[str]:
    """
    Collapse a homozygous diploid ref token stream to haploid tokens.
    Special tokens:
      - snp_token -> passed through unchanged by default (see decode_variant_tokens_to_two_haploids)
      - indel span tokens -> pass through unchanged
    """
    if gap_policy not in GAP_POLICIES:
        raise ValueError(f"gap_policy must be one of {GAP_POLICIES}, got {gap_policy!r}")

    out: List[str] = []
    rget = ref_map.get

    for i, tok in enumerate(ref_tokens):
        if tok == snp_token:
            out.append(tok if snp_haploid_char is None else snp_haploid_char)
            continue

        if tok == indel_start_token or tok == indel_end_token:
            out.append(tok)
            continue

        pair = rget(tok)
        if pair is None:
            raise ValueError(
                f"ref_seq must be homozygous diploid (ref/ref), but saw heterozygous/unknown token "
                f"{tok!r} at token-index {i}"
            )
        out.append(pair[0])

    if gap_policy == "drop":
        out = [t for t in out if t != INDEL_GAP_TOKEN]

    return out


# =============================================================================
# Dataset (precomputed)
# =============================================================================


class PrecomputedDiploidDataset(Dataset):
    """
    Base dataset must return:
        (variant_seq, ref_seq, label, conditional_input)

    Modes:
      - "haploid":
          returns (variant_haplotype, ref_haplotype, label, conditional_input)
          Exactly ONE sample per input row: the variant-bearing haplotype scored against
          the reference haplotype. ref_seq is assumed homozygous diploid (ref/ref) and is
          only collapsed.
      - "biallelic":
          returns (allele1, allele2, label, conditional_input)
          ONLY decodes variant_seq. allele1 is the variant-bearing allele, allele2 the
          other allele. For a homozygous row the two are identical, so any score computed
          from this pair is a zygosity contrast, NOT a variant-effect score -- read the
          biallelic columns accordingly.

    Because a haploid representation cannot express zygosity, the het and hom rows of the
    same variant collapse to the same input here. Where a benchmark labels those rows
    differently (e.g. recessive pathogenic: het=0, hom=1) that is an irreducible ceiling,
    not a model failure; the constructor counts and reports such collisions.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        diploid_tsv_path: str,
        mode: str = "haploid",
        get_sample_id: Optional[Callable[[Tuple[Any, Any, Any, Any]], Optional[str]]] = None,
        *,
        snp_haploid_char: Optional[str] = None,
        gap_policy: str = "drop",
    ):
        if mode not in ("haploid", "biallelic"):
            raise ValueError("mode must be 'haploid' or 'biallelic'")
        if gap_policy not in GAP_POLICIES:
            raise ValueError(f"gap_policy must be one of {GAP_POLICIES}, got {gap_policy!r}")

        if gap_policy == "keep":
            logging.warning(
                "PrecomputedDiploidDataset: gap_policy='keep' writes the bare gap token %r "
                "into haploid rows, where it tokenizes as the homozygous-gap symbol. That "
                "symbol has zero occurrences in the training corpus (token_counts.tsv: "
                "'--' = 0, while heterozygous gap pairs number ~1e8), so 'keep' buys equal "
                "sequence lengths at the price of an untrained token. Prefer 'drop'.",
                INDEL_GAP_TOKEN,
            )

        self.mode = mode
        self.gap_policy = gap_policy
        self.samples: List[Tuple[Any, ...]] = []
        self.labels: List[int] = []

        self.encoding_map = load_diploid_seq_map(diploid_tsv_path)

        var_map, ref_map, unphased_syms = _build_decode_maps_from_seq_map(
            singles=self.encoding_map.singles,
            phased_tok=self.encoding_map.phased_tok,
            unphased_tok=self.encoding_map.unphased_tok,
        )

        # Build a multi-token vocabulary for greedy tokenization.
        # Include special tokens and any mapping symbols that are multi-character.
        multi_tokens = {
            self.encoding_map.snp_token,
            self.encoding_map.indel_start_token,
            self.encoding_map.indel_end_token,
        }
        for sym in list(var_map.keys()) + list(ref_map.keys()):
            if isinstance(sym, str) and len(sym) > 1:
                multi_tokens.add(sym)
        multi_tokens_sorted = sorted(multi_tokens, key=len, reverse=True)

        n_identical_pairs = 0
        seen_inputs: Dict[bytes, set] = {}

        for i in range(len(base_dataset)):
            item = base_dataset[i]
            if not isinstance(item, (tuple, list)) or len(item) != 4:
                raise ValueError(
                    "base_dataset must return 4 items: (variant_seq, ref_seq, label, conditional_input)"
                )

            variant_seq, ref_seq, label, cond = item
            variant_seq_str = self._to_str(variant_seq)
            ref_seq_str = self._to_str(ref_seq)

            variant_tokens = _greedy_tokenize(variant_seq_str, multi_tokens_sorted)
            ref_tokens = _greedy_tokenize(ref_seq_str, multi_tokens_sorted)

            sample_id = get_sample_id(item) if get_sample_id else None
            seed64 = default_seed64_from_variant(variant_seq_str, sample_id)

            v1_toks, v2_toks = decode_variant_tokens_to_two_haploids(
                variant_tokens,
                var_map,
                unphased_syms,
                seed64,
                snp_token=self.encoding_map.snp_token,
                indel_start_token=self.encoding_map.indel_start_token,
                indel_end_token=self.encoding_map.indel_end_token,
                snp_haploid_char=snp_haploid_char,
                ref_tokens=ref_tokens,
                ref_map=ref_map,
                gap_policy=gap_policy,
            )

            v1 = _render_tokens_like_input(variant_seq_str, v1_toks)

            if mode == "biallelic":
                first = v1
                second = _render_tokens_like_input(variant_seq_str, v2_toks)
            else:
                ref_hap_toks = collapse_homozygous_ref_tokens(
                    ref_tokens,
                    ref_map,
                    snp_token=self.encoding_map.snp_token,
                    indel_start_token=self.encoding_map.indel_start_token,
                    indel_end_token=self.encoding_map.indel_end_token,
                    snp_haploid_char=snp_haploid_char,
                    gap_policy=gap_policy,
                )
                first = v1
                second = _render_tokens_like_input(ref_seq_str, ref_hap_toks)

            if first == second:
                n_identical_pairs += 1

            label_int = self._to_label_int(label)
            key = hashlib.blake2b(
                f"{len(first)}:{first}\x00{second}".encode("utf-8"), digest_size=16
            ).digest()
            seen_inputs.setdefault(key, set()).add(label_int)

            self.samples.append((first, second, label, cond))
            self.labels.append(label_int)

        self._log_composition(n_identical_pairs, seen_inputs)

    def _log_composition(self, n_identical_pairs: int, seen_inputs: Dict[bytes, set]) -> None:
        n = len(self.samples)
        if n == 0:
            logging.warning("PrecomputedDiploidDataset[%s]: empty dataset", self.mode)
            return

        pos = sum(1 for y in self.labels if y == 1)
        contradictory = sum(1 for labels in seen_inputs.values() if len(labels) > 1)
        duplicated = n - len(seen_inputs)

        logging.info(
            "PrecomputedDiploidDataset[%s]: %d samples, positive rate %.4f, "
            "%d distinct inputs (%d duplicated), gap_policy=%s",
            self.mode, n, pos / n, len(seen_inputs), duplicated, self.gap_policy,
        )

        if self.mode == "biallelic" and n_identical_pairs:
            logging.info(
                "PrecomputedDiploidDataset[biallelic]: %d/%d rows are homozygous, so their two "
                "alleles are identical and any distance/LLR between them is exactly 0. This "
                "mode scores zygosity, not variant effect.",
                n_identical_pairs, n,
            )
        elif n_identical_pairs:
            logging.warning(
                "PrecomputedDiploidDataset[%s]: %d/%d samples have identical variant and "
                "reference sequences (the score is degenerate for these rows)",
                self.mode, n_identical_pairs, n,
            )

        if contradictory:
            logging.warning(
                "PrecomputedDiploidDataset[%s]: %d distinct inputs carry BOTH labels -- a "
                "haploid representation cannot express zygosity, so zygosity-dependent labels "
                "(e.g. recessive het=0 / hom=1) collapse here and cap the achievable AUROC. "
                "Evaluate stratified by zygosity to interpret this mode.",
                self.mode, contradictory,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    @staticmethod
    def _to_label_int(label: Any) -> int:
        if torch.is_tensor(label):
            return int(label.reshape(-1)[0].item())
        try:
            return int(label)
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _to_str(x: Any) -> str:
        """
        Converts common dataset outputs to string.
        If your dataset returns token IDs (not strings), decode IDs->token-strings before using this wrapper.
        """
        if isinstance(x, str):
            return x
        if isinstance(x, (list, tuple)):
            # If it's already a list of tokens, keep a space-separated representation to avoid ambiguity.
            # This is safer when tokens can be multi-character (e.g. "<SNP>").
            if all(isinstance(t, str) for t in x):
                return " ".join(x)
            return "".join(str(t) for t in x)
        if torch.is_tensor(x):
            try:
                return "".join(chr(int(v)) for v in x.view(-1))
            except Exception:
                pass
        raise TypeError(f"Cannot convert {type(x)} to string")
