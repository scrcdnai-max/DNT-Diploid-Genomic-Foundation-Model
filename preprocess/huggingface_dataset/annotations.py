"""Per-base annotation framework for v7+ datasets.

The encoder ([preprocess/encoder/get_genome_sequence.py](preprocess/encoder/get_genome_sequence.py))
emits a per-codepoint ``seq_ref_idx`` array parallel to the marked ``seq``
string:

    seq_ref_idx[k] = i  (0 <= i < end-start)  -> ref position start+i
    seq_ref_idx[k] = -1                       -> inserted base (no ref pos)
    seq_ref_idx[k] = -2                       -> marker codepoint
                                                 (^, {, }, '<', '>')

For variant-aware annotations (e.g. CADD), the encoder also emits a
parallel ``seq_var_idx`` array and the per-region ``variants`` list of
:class:`get_genome_sequence.VariantCall` records (see
``get_marked_sequence_with_ref_idx_and_variants_parallel_str``).

Annotations come in two flavours:

* **Allele-agnostic** (PhyloP): override :meth:`Annotation.get_values` to
  fetch one float per ref position in ``[start, end)``. The base class's
  default :meth:`Annotation.compute` calls ``align_to_seq`` on the result.
* **Variant-aware** (CADD): override :meth:`Annotation.compute` directly,
  set ``requires_variants = True``, and use ``seq_var_idx`` + ``variants``
  to place per-variant scores on their anchor codepoints.

Adding a new annotation is a one-class change here + appending it to
``ANNOTATIONS``; nothing else in the pipeline needs to know.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pyarrow as pa

# ---------------------------------------------------------------------------
# Encoding constants
# ---------------------------------------------------------------------------

# PhyloP: linear mapping [−20.0, +10.003] → uint8 [0, 254]; 255 = NaN sentinel.
# Resolution ≈ 0.118 phyloP units.  Decode: v/254*30.003 − 20.
_PHYLOP_OFFSET: float = 20.0
_PHYLOP_RANGE: float = 30.003
_PHYLOP_NAN_SENTINEL: int = 255

# CADD: sparse list<struct<pos: int32, score: float32>> (uint8 in datasets built before 2026-10-14).
# pos = codepoint index within the aligned seq array; absence = NaN.
_CADD_STRUCT_TYPE = pa.struct([pa.field("pos", pa.int32()), pa.field("score", pa.float32())])

# ---------------------------------------------------------------------------
# Marker policy
# ---------------------------------------------------------------------------


class MarkerPolicy(str, Enum):
    """How per-base annotation arrays handle marker codepoints in ``seq``.

    SKIP: drop marker slots; len(out) == count of non-marker codepoints.
    NAN:  keep marker slots, fill with NaN; len(out) == len(seq).
    """

    SKIP = "skip"
    NAN = "nan"


# ---------------------------------------------------------------------------
# Annotation base + helpers
# ---------------------------------------------------------------------------


class Annotation(ABC):
    """Per-base score over genomic positions.

    Subclasses set ``name`` (the parquet column name) and ``arrow_type``
    (the element type inside the ``list_`` column). They implement
    :meth:`open`, :meth:`close`, :meth:`provenance`, and either
    :meth:`get_values` (allele-agnostic, e.g. PhyloP) or :meth:`compute`
    (variant-aware, e.g. CADD).

    Variant-aware subclasses must also set ``requires_variants = True``
    so :mod:`create_dataset` knows to switch the encoder to the variant-
    emitting parallel method (``get_marked_sequence_with_ref_idx_and_variants_parallel_str``).
    """

    name: str = ""
    arrow_type: pa.DataType = pa.float32()
    requires_variants: bool = False

    @property
    def field(self) -> pa.Field:
        return pa.field(self.name, pa.list_(self.arrow_type))

    def encode(self, arr: np.ndarray) -> np.ndarray:
        """Convert compute()'s float32 output to the on-disk storage format.

        Default: return arr unchanged (float32 stored as-is).
        Subclasses override to apply quantisation or sparse encoding.
        Called once per region, after compute() and before Parquet write.
        """
        return arr

    @abstractmethod
    def open(self) -> None:
        """Acquire any per-worker resources (file handles etc.)."""

    @abstractmethod
    def close(self) -> None:
        """Release per-worker resources."""

    def get_values(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Return one float per ref position in ``[start, end)``.

        Length must equal ``end - start``. Missing values must be NaN.
        Implement this for allele-agnostic annotations (PhyloP). Variant-
        aware annotations override :meth:`compute` instead and need not
        implement this hook.
        """
        raise NotImplementedError(
            f"{type(self).__name__} did not implement get_values; "
            "variant-aware subclasses should override compute() instead."
        )

    def compute(
        self,
        chrom: str,
        start: int,
        end: int,
        seq_ref_idx: np.ndarray,
        seq_var_idx: np.ndarray,
        variants: Sequence[Any],
        marker_policy: "MarkerPolicy",
    ) -> np.ndarray:
        """Per-codepoint float32 array aligned to ``seq`` under ``marker_policy``.

        Default implementation: call :meth:`get_values` then
        :func:`align_to_seq` (ignoring ``seq_var_idx`` and ``variants``).
        Override for variant-aware scoring.
        """
        vals = self.get_values(chrom, start, end)
        return align_to_seq(vals, seq_ref_idx, marker_policy)

    @abstractmethod
    def provenance(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict identifying the data source.

        Used by the generation provenance sidecar so reruns can't silently
        mix datasets built from different annotation files.
        """


class PhyloPAnnotation(Annotation):
    """100-way vertebrate PhyloP conservation score from a bigWig file.

    The caller MUST pass an explicit ``source_path``. The previous default
    (a site-specific ``hg38.phyloP100way.bw`` path) was removed because
    silently substituting a default invited the kind of wrong-shard
    misconfiguration that produced ~96% silent NaN in the v7 pipeline.

    Stored as ``list<uint8>`` with linear quantisation over [−20, +10.003]:
    ``encoded = round((v + 20) / 30.003 * 254).clip(0, 254).astype(uint8)``
    Sentinel 255 = NaN (~0.37% of bases).  Decode: ``v/254*30.003 − 20``.
    """

    name = "phylop"
    arrow_type = pa.uint8()  # stored as list<uint8>; decode via dataset.to_phylop()

    def __init__(self, source_path: str):
        self._path = source_path
        self._bw = None

    def open(self) -> None:
        import pyBigWig  # local import: workers fork before opening
        self._bw = pyBigWig.open(self._path)

    def close(self) -> None:
        if self._bw is not None:
            self._bw.close()
            self._bw = None

    def get_values(self, chrom: str, start: int, end: int) -> np.ndarray:
        assert self._bw is not None, "PhyloPAnnotation.open() not called"
        # pyBigWig returns a numpy array with NaN at uncovered positions.
        vals = self._bw.values(chrom, start, end, numpy=True)
        return vals.astype(np.float32, copy=False)

    def encode(self, arr: np.ndarray) -> np.ndarray:
        """float32 → uint8 with linear quantisation; 255 = NaN sentinel."""
        out = np.full(len(arr), _PHYLOP_NAN_SENTINEL, dtype=np.uint8)
        valid = ~np.isnan(arr)
        out[valid] = (
            np.round((arr[valid] + _PHYLOP_OFFSET) / _PHYLOP_RANGE * 254)
            .clip(0, 254)
            .astype(np.uint8)
        )
        return out

    def provenance(self) -> Dict[str, Any]:
        st = os.stat(self._path)
        return {
            "name": self.name,
            "source_path": str(self._path),
            "source_size": st.st_size,
            "source_mtime": int(st.st_mtime),
            "encoding": "uint8_linear",
            "phylop_offset": _PHYLOP_OFFSET,
            "phylop_range": _PHYLOP_RANGE,
            "nan_sentinel": _PHYLOP_NAN_SENTINEL,
        }


# ---------------------------------------------------------------------------
# CADD annotation (variant-aware)
# ---------------------------------------------------------------------------


# ACGT -> column index in the per-chrom SNV phred table (cf. extract_cadd_for_bed.py).
_CADD_BASE_TO_COL = {ord(b): i for i, b in enumerate("ACGT")}


class CADDAnnotation(Annotation):
    """CADD-v1.7 GRCh38 per-variant deleteriousness score (PHRED-scaled).

    Variant-aware: needs the sample's per-haplotype alleles to pick the right
    score from CADD's allele-keyed tables. The single output column
    ``cadd_phred`` holds the higher of the two haplotypes' phreds
    (each NaN if that haplotype is reference or absent from the indel CADD
    file). Anchor-only placement: exactly one supervised slot per variant —
    the codepoint where ``seq_ref_idx == anchor_offset AND seq_var_idx == j``.

    Source artifacts (whole-genome Arrow under ``arrow_dir``, produced by
    ``poc_a2_convert_raw_cadd.py``):

    Both shard kinds are **memory-mapped** (``pa.memory_map`` +
    ``ipc.open_file``), and every lookup reads through **zero-copy numpy
    views** into the mmap. The OS page cache therefore holds exactly one
    physical copy per file, shared across all worker processes via the file
    inode; reads never trigger copy-on-write (the mmap'd data pages carry no
    Python refcounts), so per-worker private RAM for CADD is ~0 regardless of
    worker count. Pages fault in lazily, so coverage auto-restricts to the
    positions actually looked up (the BED footprint).

    This replaces the earlier per-worker private-heap design (a Python dict
    ``{(pos, ref, alt): phred}`` for indels, an ``np.stack`` copy for SNVs).
    That design avoided fork-COW by allocating post-fork, but in doing so it
    *guaranteed* a full private copy of the genome's CADD in every worker:
    each worker walks all chromosomes for its sample and cached every shard,
    so it materialised ~58 GB (SNV) + ~42 GB (105 M-row indel dict) privately
    — the reason >20 workers OOM'd a 1 TB box.

    * Per-chrom SNV shards ``cadd_snv_<chrom>.arrow``, columns
      ``pos:int32, ref:uint8, phred_A/C/G/T:float32``. SNV lookup is a
      ``searchsorted`` on the zero-copy ``pos`` view, then an index into the
      matching zero-copy ``phred_<base>`` column view.
    * Per-chrom indel shards ``cadd_indel_<chrom>.arrow``, columns
      ``chrom:large_utf8, pos:int32, ref:large_utf8, alt:large_utf8,
      phred:float32``. Indel lookup brackets the rows at ``pos`` with
      ``searchsorted`` (left/right) and scans that small block, comparing the
      mmap-backed ``ref``/``alt`` Arrow string columns.

    Misses (indels absent from CADD's gnomAD-only file, ~3 % expected) are
    silently skipped. Not a hard fail — "no silent fails" applies to bugs /
    coord-mismatches / schema problems, not to legitimate data absence.
    """

    name = "cadd_phred"
    # Stored as list<struct<pos: int32, score: float32>>: sparse, anchor-only.
    # pos = codepoint index within the aligned seq array; absent entries = NaN.
    # No sentinel needed — absence IS the NaN encoding.
    arrow_type = pa.float32()   # unused; field() is overridden below
    requires_variants = True

    @property
    def field(self) -> pa.Field:
        return pa.field(self.name, pa.list_(_CADD_STRUCT_TYPE))
    # Raw source paths are kept as class constants for provenance reporting
    # only — never used as a lookup fallback. Caller MUST pass an explicit
    # ``arrow_dir`` to the constructor.
    RAW_SNV_SOURCE = "whole_genome_SNVs.tsv.gz"
    RAW_INDEL_SOURCE = "gnomad.genomes.r4.0.indel.tsv.gz"
    CADD_VERSION = "CADD-v1.7-GRCh38"

    def __init__(self, arrow_dir: str):
        self._arrow_dir = Path(arrow_dir)
        # Per-chrom memmapped SNV cache: chrom → (table, pos, ref, phred_cols).
        # The Table reference keeps the mmap alive; pos/ref/phred_cols are
        # zero-copy views into it (no private copy).
        self._snv_cache: Dict[str, "tuple"] = {}
        # Per-chrom memmapped indel cache: chrom → (table, pos, ref_col,
        # alt_col, phred). Also zero-copy views into the mmap.
        self._indel_cache: Dict[str, "tuple"] = {}
        self._sample_id: Optional[str] = None

    def set_sample(self, sample_id: str) -> None:
        """Set the current sample's identifier.

        Called by the worker before its per-region loop. One worker handles
        one sample at a time, so the attribute is worker-local."""
        self._sample_id = sample_id

    def open(self) -> None:
        if not self._arrow_dir.is_dir():
            raise FileNotFoundError(
                f"CADD arrow dir not found: {self._arrow_dir}. "
                f"Run poc_a2_convert_raw_cadd.py / convert_all_cadd_to_arrow.sh first."
            )
        # Per-chrom SNV and indel shards are loaded lazily on first lookup
        # by _load_snv_shard() / _load_indel_shard().  No data is loaded here.

    def close(self) -> None:
        self._snv_cache.clear()
        self._indel_cache.clear()

    def _load_snv_shard(self, chrom_with_chr: str) -> tuple:
        """Return ``(table, pos[n], ref[n], (phred_A, phred_C, phred_G, phred_T))``.

        ``pos`` (int32), ``ref`` (uint8) and each ``phred_<base>`` (float32) are
        zero-copy numpy views into a memory-mapped shard; the OS page cache
        holds one physical copy shared across all workers. The ``table`` is
        returned (and cached) only to keep the mmap alive. Cached per worker;
        first access for an unseen chrom maps the file, later accesses are
        O(1)."""
        cached = self._snv_cache.get(chrom_with_chr)
        if cached is not None:
            return cached
        import pyarrow.ipc as ipc
        path = self._arrow_dir / f"cadd_snv_{chrom_with_chr}.arrow"
        if not path.is_file():
            raise FileNotFoundError(
                f"CADD SNV shard missing for {chrom_with_chr}: {path}"
            )
        # memory_map -> Arrow buffers are slices of the shared page cache; the
        # views below never copy and never dirty a page (no COW).
        mm = pa.memory_map(str(path), "r")
        t = ipc.open_file(mm).read_all()
        pos = t.column("pos").chunk(0).to_numpy(zero_copy_only=True)
        ref = t.column("ref").chunk(0).to_numpy(zero_copy_only=True)
        phred_cols = tuple(
            t.column(f"phred_{b}").chunk(0).to_numpy(zero_copy_only=True)
            for b in "ACGT"
        )
        # Sanity: rows must be sorted by pos (extract script guarantees this).
        if pos.size > 1 and not np.all(pos[:-1] < pos[1:]):
            raise ValueError(
                f"CADD SNV shard not sorted by pos: {path}"
            )
        entry = (t, pos, ref, phred_cols)
        self._snv_cache[chrom_with_chr] = entry
        return entry

    def _load_indel_shard(self, chrom_with_chr: str) -> tuple:
        """Return ``(table, pos[n], ref_col, alt_col, phred[n])`` for chrom.

        ``pos`` (int32) and ``phred`` (float32) are zero-copy views into a
        memory-mapped shard; ``ref_col``/``alt_col`` are the mmap-backed Arrow
        string arrays (scanned per-block in :meth:`_lookup`). One physical copy
        lives in the shared page cache — no per-worker private heap. The
        ``table`` is returned only to keep the mmap alive. Cached per worker."""
        cached = self._indel_cache.get(chrom_with_chr)
        if cached is not None:
            return cached
        import pyarrow.ipc as ipc
        path = self._arrow_dir / f"cadd_indel_{chrom_with_chr}.arrow"
        if not path.is_file():
            raise FileNotFoundError(
                f"CADD indel shard missing for chrom '{chrom_with_chr}': {path}. "
                f"Run poc_a2_convert_raw_cadd.py / convert_all_cadd_to_arrow.sh first."
            )
        mm = pa.memory_map(str(path), "r")
        t = ipc.open_file(mm).read_all()
        pos = t.column("pos").chunk(0).to_numpy(zero_copy_only=True)
        ref_col = t.column("ref").chunk(0)
        alt_col = t.column("alt").chunk(0)
        phred = t.column("phred").chunk(0).to_numpy(zero_copy_only=True)
        # Sanity: rows must be sorted by pos (non-decreasing — multiple indels
        # may share a position) so searchsorted can bracket each position.
        if pos.size > 1 and not np.all(pos[:-1] <= pos[1:]):
            raise ValueError(
                f"CADD indel shard not sorted by pos: {path}"
            )
        entry = (t, pos, ref_col, alt_col, phred)
        self._indel_cache[chrom_with_chr] = entry
        return entry

    def _lookup(
        self,
        chrom_with_chr: str,
        pos: int,
        ref: str,
        alt: str,
        last_pos: Optional[int] = None,
    ) -> Optional[float]:
        """Return PHRED for one (pos, ref, alt) on one haplotype, or None.

        ``None`` means "this haplotype contributes nothing to the per-variant
        score" — either because it carries the reference allele (alt == ref)
        or because the indel is absent from CADD's gnomAD-only indel file
        (the latter is logged to the sidecar). ``last_pos`` is the region's
        last 1-based position: an SNV found past it is not scored."""
        if alt == ref or "-" in alt:  # '-' = spanning deletion (VCF '*'): no CADD score
            return None
        # CADD keys are trimmed alleles; the encoder hands over padded ones. A same-POS
        # merge right-pads an SNV or a shorter indel to the longest REF at that position
        # (AC>A next to ACG>A arrives as ACG>AG): strip the shared suffix.
        while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
            ref, alt = ref[:-1], alt[:-1]
        if len(ref) == len(alt) > 1:
            # Same length, so no indel: if one base differs (CG>CT) it is an SNV further
            # along. Anything else keeps its key and misses the indel table.
            r, a, p = ref, alt, pos
            while len(r) > 1 and r[0] == a[0]:
                r, a, p = r[1:], a[1:], p + 1
            if len(r) == 1:
                if last_pos is not None and p > last_pos:
                    return None  # the changed base is outside this row
                ref, alt, pos = r, a, p
        if len(ref) == 1 and len(alt) == 1:
            # SNV lookup via per-chrom memmapped shard.
            _, pos_arr, ref_arr, phred_cols = self._load_snv_shard(chrom_with_chr)
            # Cast to np.int32: passing a Python int to searchsorted on a
            # 13M-row int32 array is ~44ms/call on numpy 2.x (int-promotion
            # regression); np scalar matches dtype and runs in ~2us.
            idx = int(np.searchsorted(pos_arr, np.int32(pos)))
            if idx == pos_arr.size or int(pos_arr[idx]) != pos:
                raise ValueError(
                    f"SNV position absent from CADD shard "
                    f"(BED drift?): {chrom_with_chr}:{pos}"
                )
            stored_ref = int(ref_arr[idx])
            if stored_ref != ord(ref):
                raise ValueError(
                    f"CADD ref mismatch at {chrom_with_chr}:{pos} — "
                    f"shard says {chr(stored_ref)}, VCF says {ref}"
                )
            col = _CADD_BASE_TO_COL.get(ord(alt))
            if col is None:
                raise ValueError(
                    f"non-ACGT alt {alt!r} at {chrom_with_chr}:{pos}"
                )
            return float(phred_cols[col][idx])
        # INDEL lookup via per-chrom memmapped shard: bracket the rows at this
        # position with one searchsorted for the left edge, then walk forward
        # over the (mean ~1.7 rows/pos) equal block comparing the mmap-backed
        # ref/alt string columns.  ipos is sorted non-decreasing, so the equal
        # values are contiguous starting at lo — the forward scan stops at the
        # same index a second searchsorted(side="right") would return, but
        # avoids a second full binary search over the 105M-row pos array.
        _, ipos, ref_col, alt_col, iphred = self._load_indel_shard(chrom_with_chr)
        pos32 = np.int32(pos)
        lo = int(np.searchsorted(ipos, pos32, side="left"))
        n_ipos = ipos.size
        hi = lo
        while hi < n_ipos and ipos[hi] == pos32:
            hi += 1
        for k in range(lo, hi):
            if ref_col[k].as_py() == ref and alt_col[k].as_py() == alt:
                return float(iphred[k])
        return None

    def compute(
        self,
        chrom: str,
        start: int,
        end: int,
        seq_ref_idx: np.ndarray,
        seq_var_idx: np.ndarray,
        variants: Sequence[Any],
        marker_policy: "MarkerPolicy",
    ) -> np.ndarray:
        out = np.full(len(seq_ref_idx), np.nan, dtype=np.float32)
        region_len = end - start
        # Map each variant index -> the codepoint slots that carry it, in one
        # O(len(seq)) pass. Avoids allocating two full-length bool arrays per
        # variant (the old `(seq_var_idx == j) & (seq_ref_idx == anchor)` was
        # O(n_variants * len(seq))). Variant slots are sparse, so the Python
        # loop only visits variant codepoints.
        slots_by_var: Dict[int, List[int]] = {}
        for k in np.flatnonzero(seq_var_idx >= 0):
            slots_by_var.setdefault(int(seq_var_idx[k]), []).append(int(k))
        for j, v in enumerate(variants):
            anchor_offset = v.pos - 1 - start
            # Cross-region clipped anchor: the variant overlaps the region
            # only via its non-anchor content (e.g. deletion). There is no
            # anchor codepoint to attach the score to; log and continue
            # WITHOUT calling _lookup (the lookup itself would still succeed
            # for an indel hit, but the anchor-mask trick below would
            # spuriously match inserted-base slots whose sri happens to
            # equal the negative anchor_offset). Skip cleanly.
            if not (0 <= anchor_offset < region_len):
                continue
            # Homozygous variants (~40% of calls) carry identical alleles on
            # both haplotypes; _lookup is pure, so the second call repeats the
            # same searchsorted over the (13M-row SNV / 105M-row indel) pos
            # array and returns the same value.  Look up once and reuse.
            # max(p, p) == p, so the reduction below is unchanged.
            if v.hap0 == v.hap1:
                p0 = p1 = self._lookup(chrom, v.pos, v.ref, v.hap0, end)
            else:
                p0 = self._lookup(chrom, v.pos, v.ref, v.hap0, end)
                p1 = self._lookup(chrom, v.pos, v.ref, v.hap1, end)
            vals = [p for p in (p0, p1) if p is not None]
            if not vals:
                continue
            # Two different alternates (VCF 1|2): the more damaging one, not their mean.
            score = float(max(vals))
            # The anchor is the slot for this variant whose ref offset matches.
            anchor_idx = -1
            n_anchor = 0
            for k in slots_by_var.get(j, ()):
                if seq_ref_idx[k] == anchor_offset:
                    anchor_idx = k
                    n_anchor += 1
            if n_anchor == 1:
                out[anchor_idx] = score
            elif n_anchor == 0:
                pass  # in-region anchor but no codepoint carries it; skip
            else:
                raise ValueError(
                    f"unexpected multi-anchor for variant j={j} at "
                    f"{chrom}:{v.pos} ref={v.ref} (n_anchor={n_anchor})"
                )
        if marker_policy is MarkerPolicy.SKIP:
            out = out[seq_ref_idx != -2]
        elif marker_policy is MarkerPolicy.NAN:
            # Marker slots are already NaN: ``out`` is NaN-initialised and a
            # marker codepoint (seq_ref_idx == -2) is never an anchor, so it
            # never receives a score. Keep them as-is, 1:1 with seq.
            pass
        else:
            raise NotImplementedError(
                f"CADDAnnotation.compute does not handle marker_policy="
                f"{marker_policy!r}; only SKIP and NAN are implemented."
            )
        return out

    def encode(self, arr: np.ndarray) -> list:
        """float32 dense → sparse list of {pos, score} dicts for list<struct> storage.

        pos is the codepoint index within arr; score is the CADD phred as float32. Until 2026-10-14 it was
        rounded to a whole number (uint8), which stored about 28% of scores as 0; datasets built before then
        still hold uint8 scores, and every reader takes both. Absent entries decode as NaN.
        """
        nz = np.flatnonzero(~np.isnan(arr))
        return [{"pos": int(i), "score": float(arr[i])} for i in nz]

    def provenance(self) -> Dict[str, Any]:
        st_dir = os.stat(self._arrow_dir)
        return {
            "name": self.name,
            "arrow_dir": str(self._arrow_dir),
            "arrow_dir_mtime": int(st_dir.st_mtime),
            "raw_snv_source": self.RAW_SNV_SOURCE,
            "raw_indel_source": self.RAW_INDEL_SOURCE,
            "version": self.CADD_VERSION,
            "reduction": "max_hap0_hap1",
            "placement": "anchor_only",
        }


# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------


def align_to_seq(
    values: np.ndarray,
    seq_ref_idx: np.ndarray,
    marker_policy: MarkerPolicy,
) -> np.ndarray:
    """Project per-ref-position ``values`` onto codepoints of ``seq``.

    ``seq_ref_idx`` is the encoder's parallel index array (length == len(seq)):
    nonneg = ref offset within the region; -1 = inserted base; -2 = marker.

    SKIP -> drop -2 entries, NaN at -1, ``values[idx]`` elsewhere.
    NAN  -> keep -2 entries (NaN), NaN at -1, ``values[idx]`` elsewhere.
    """
    if marker_policy is MarkerPolicy.SKIP:
        keep = seq_ref_idx != -2
        idx = seq_ref_idx[keep]
        out = np.full(idx.shape, np.nan, dtype=np.float32)
        ref_mask = idx >= 0
        out[ref_mask] = values[idx[ref_mask]]
        return out

    # MarkerPolicy.NAN: full length parallel to seq
    out = np.full(seq_ref_idx.shape, np.nan, dtype=np.float32)
    ref_mask = seq_ref_idx >= 0
    out[ref_mask] = values[seq_ref_idx[ref_mask]]
    return out


# ---------------------------------------------------------------------------
# Per-worker performance accumulator
# ---------------------------------------------------------------------------


@dataclass
class AnnotationPerf:
    """Per-worker timing counters for annotation fetch + align."""

    regions: int = 0
    fetch_s: float = 0.0
    align_s: float = 0.0
    per_annotation_fetch_s: Dict[str, float] = field(default_factory=dict)

    def add(self, name: str, fetch_s: float, align_s: float) -> None:
        self.fetch_s += fetch_s
        self.align_s += align_s
        self.per_annotation_fetch_s[name] = (
            self.per_annotation_fetch_s.get(name, 0.0) + fetch_s
        )

    def tick_region(self) -> None:
        self.regions += 1

    def merge(self, other: "AnnotationPerf") -> None:
        self.regions += other.regions
        self.fetch_s += other.fetch_s
        self.align_s += other.align_s
        for k, v in other.per_annotation_fetch_s.items():
            self.per_annotation_fetch_s[k] = (
                self.per_annotation_fetch_s.get(k, 0.0) + v
            )

    def format(self) -> str:
        if self.regions == 0:
            return "annotations: 0 regions"
        avg_ms = (self.fetch_s + self.align_s) * 1000.0 / self.regions
        parts = [
            f"fetch={self.fetch_s:.2f}s",
            f"align={self.align_s:.2f}s",
            f"{self.regions} regions",
            f"avg={avg_ms:.2f}ms/region",
        ]
        if self.per_annotation_fetch_s:
            per = ", ".join(
                f"{k}={v:.2f}s" for k, v in self.per_annotation_fetch_s.items()
            )
            parts.append(f"[{per}]")
        return "annotations: " + " ".join(parts)


# ---------------------------------------------------------------------------
# Worker-side convenience: fetch+align all annotations for one region
# ---------------------------------------------------------------------------


def fetch_and_align(
    annotations: List[Annotation],
    chrom: str,
    start: int,
    end: int,
    seq_ref_idx: np.ndarray,
    seq_var_idx: Optional[np.ndarray],
    variants: Optional[Sequence[Any]],
    marker_policy: MarkerPolicy,
    perf: Optional[AnnotationPerf] = None,
) -> Dict[str, np.ndarray]:
    """Compute each annotation for ``[start, end)`` and align onto codepoints.

    ``seq_var_idx`` and ``variants`` are required when any registered
    annotation has ``requires_variants = True``; pass ``None`` otherwise
    (allele-agnostic-only registry).

    Returns a dict mapping annotation name -> per-codepoint float32 array.
    """
    needs_variants = any(a.requires_variants for a in annotations)
    if needs_variants and (seq_var_idx is None or variants is None):
        raise ValueError(
            "fetch_and_align: at least one registered annotation requires "
            "variants but seq_var_idx/variants were not provided."
        )
    out: Dict[str, np.ndarray] = {}
    for annot in annotations:
        t0 = time.perf_counter()
        aligned = annot.compute(
            chrom, start, end,
            seq_ref_idx, seq_var_idx, variants,
            marker_policy,
        )
        t1 = time.perf_counter()
        out[annot.name] = aligned
        # compute() fuses fetch + align; report combined as fetch_s and
        # leave align_s at 0 so the per-annotation timing field is still
        # populated.
        if perf is not None:
            perf.add(annot.name, t1 - t0, 0.0)
    if perf is not None:
        perf.tick_region()
    return out


# ---------------------------------------------------------------------------
# Registry — the *single* place to enable annotations
# ---------------------------------------------------------------------------

# Empty by default. ``create_dataset.py`` appends ``PhyloPAnnotation`` and/or
# ``CADDAnnotation`` instances based on CLI args (``--phylop-bw``,
# ``--cadd-shards-dir`` + ``--cadd-indel-path``). Defaults were removed
# because silent path defaults invited the wrong-shard misconfiguration that
# produced ~96% silent NaN in the v7 pipeline; callers must opt in
# explicitly with explicit paths.
ANNOTATIONS: List[Annotation] = []

# v7 default: NaN at marker positions so the annotation array stays 1:1
# with codepoints of ``seq`` (downstream code can index by codepoint
# without a marker-aware mapping). Switch to MarkerPolicy.SKIP to drop the
# marker slots entirely. See feedback_v7_annotation_marker_policy memory.
ACTIVE_MARKER_POLICY: MarkerPolicy = MarkerPolicy.NAN


def annotation_schema_fields() -> List[pa.Field]:
    """Pyarrow fields contributed by the active annotation list."""
    return [a.field for a in ANNOTATIONS]


def annotation_provenance() -> List[Dict[str, Any]]:
    """Provenance entries for the active annotations (sidecar/contamination guard)."""
    return [a.provenance() for a in ANNOTATIONS]
