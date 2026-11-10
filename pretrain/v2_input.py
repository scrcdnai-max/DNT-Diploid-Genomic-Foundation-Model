"""v2 input: one token per position, read from the stored v7 rows without converting them.

A stored row writes a variant as markers plus a glyph: ``^‹Ⱥ`` (heterozygous SNV), ``^G`` (homozygous
SNV), ``{‹Ģ‹Č}`` (indel). This module rewrites the glyph that FOLLOWS a marker and leaves the markers
where they are, so the row keeps its length and the per-position phyloP / CADD targets stay aligned;
the collator's marker drop (``--marker-drop-target all --marker-drop-prob 1``) then removes them.

  direction token + unphased pair glyph  ->  the phased glyph for (copy 0 allele, copy 1 allele)
  ``^`` + plain base                     ->  that base's homozygous-alternate glyph (``homalt`` rows)
  unphased pair glyph with no direction token before it (a window cut, or an unphased call)  ->  ``N``

Direction rule, as the encoder writes it (``preprocess/encoder/get_genome_sequence.py``
``_compute_direction_array``): alleles are ordered A < C < G < T < N < gap; ``‹`` means copy 0 carries
the first, ``›`` the second.

Reference-aware prototype (``preprocess/diploid_tokens_v3.tsv``): given the HG38 base under each character, a
heterozygous SNV token also says which copy carries the reference. The ``phased`` tokens then mean reference on
copy 0 (VCF 0|1) and the ``phased_ref1`` tokens reference on copy 1 (1|0); a site where neither allele is the
reference becomes ``N``. ``row_reference`` gives the HG38 base under each character by aligning the row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)  # train.py runs as a script from pretrain/
from embedding.cohort import rows as cohort_rows

MARKERS = cohort_rows.MARKERS

_ALLELE_ORDER = "ACGTN-*"
SNP_MARKER, DIR_FWD, DIR_REV = "^", "‹", "›"


class V2Rewrite:
    """Callable on a window's uint32 codepoints; returns a rewritten copy of the same length."""

    def __init__(self, tokens_tsv: Path | str):
        rows = [
            ln.split("\t")[:3]
            for ln in Path(tokens_tsv).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        phased = {pair: sym for pair, order, sym in rows if order == "phased"}
        ref1 = {pair: sym for pair, order, sym in rows if order == "phased_ref1"}
        maps: dict[str, dict[str, str]] = {DIR_FWD: {}, DIR_REV: {}, SNP_MARKER: {}}
        for pair, order, sym in rows:
            if order == "unphased":
                first, second = sorted(pair, key=_ALLELE_ORDER.index)
                for marker, glyph in (
                    (DIR_FWD, phased[first + second]),
                    (DIR_REV, phased[second + first]),
                ):
                    # `*` rows share the gap glyphs, so a glyph is assigned twice: it must agree.
                    if maps[marker].setdefault(sym, glyph) != glyph:
                        raise ValueError(
                            f"{tokens_tsv}: {sym!r} maps to two phased glyphs after {marker!r}"
                        )
            elif order == "homalt":
                maps[SNP_MARKER][pair[0]] = sym
        if not maps[DIR_FWD] or not maps[SNP_MARKER]:
            raise ValueError(
                f"{tokens_tsv}: needs unphased, phased and homalt rows (is this the v2 token file?)"
            )
        #: (copy 0 allele + copy 1 allele) -> phased glyph, and base -> homozygous-alternate glyph
        self.phased, self.homalt = phased, maps[SNP_MARKER]
        #: reference-aware file only: (copy 0 allele + copy 1 allele) -> glyph with the reference on copy 1
        self.phased_ref1 = ref1
        #: every glyph this rewrite can write, for the token-counts check
        self.emitted = frozenset(
            g for m in maps.values() for g in m.values()
        ) | frozenset(ref1.values())
        # One slot past the largest codepoint: codepoints outside the file clip there and stay unchanged.
        size = (
            max(
                [ord(c) for m in maps.values() for kv in m.items() for c in kv]
                + [ord(g) for g in ref1.values()]
            )
            + 2
        )
        self._luts = []
        for marker, mapping in maps.items():
            lut = np.arange(size, dtype=np.uint32)
            for src, dst in mapping.items():
                lut[ord(src)] = ord(dst)
            self._luts.append((np.uint32(ord(marker)), lut))
        self._is_unphased = np.zeros(size, dtype=bool)
        self._is_unphased[[ord(s) for s in maps[DIR_FWD]]] = True
        # phased SNV glyph -> its copy-0 base, its copy-1 base, and its twin with the reference on copy 1
        self.reference_aware = bool(ref1)
        self._ref1 = np.zeros((3, size), dtype=np.uint32) if ref1 else None
        if self._ref1 is not None:
            for pair, glyph in phased.items():
                if pair in ref1:
                    self._ref1[:, ord(glyph)] = (
                        ord(pair[0]),
                        ord(pair[1]),
                        ord(ref1[pair]),
                    )

    def __call__(
        self, codepoints: np.ndarray, reference: np.ndarray | None = None
    ) -> np.ndarray:
        """``reference``: the HG38 base under each codepoint (same length; ignored under markers)."""
        out = codepoints.copy()
        for marker, lut in self._luts:
            # Markers are read from the INPUT: a rewritten glyph is never itself a marker.
            idx = np.flatnonzero(codepoints[:-1] == marker) + 1
            idx = idx[out[idx] < len(lut)]
            out[idx] = lut[out[idx]]
        out[self._is_unphased.take(out, mode="clip")] = ord("N")
        if reference is not None:
            if self._ref1 is None:
                raise ValueError(
                    "a reference needs the reference-aware token file (preprocess/diploid_tokens_v3.tsv)"
                )
            copy0, copy1, twin = self._ref1.take(out, axis=1, mode="clip")
            het = copy0 > 0
            ref_on_1 = het & (reference == copy1)
            out[het & (reference != copy0) & ~ref_on_1] = ord("N")
            out[ref_on_1] = twin[ref_on_1]
        return out

    def het_glyph(self, ref: str, alt: str, alt_copy: int) -> str:
        """The token of a heterozygous SNV with the alternate on copy ``alt_copy`` (VCF 0|1 is ``alt_copy=1``)."""
        if alt_copy == 1:
            return self.phased[ref + alt]
        return (self.phased_ref1 if self.reference_aware else self.phased)[alt + ref]

    def twin(self, glyph: str) -> str:
        """The same heterozygous SNV token with the two copies swapped."""
        for table, other in (
            (self.phased, self.phased_ref1),
            (self.phased_ref1, self.phased),
        ):
            pair = next((p for p, g in table.items() if g == glyph), None)
            if pair is not None:
                return (other if self.reference_aware else table)[pair[::-1]]
        raise KeyError(glyph)

    def rewrite_str(self, seq: str, reference: np.ndarray | None = None) -> str:
        return (
            self(np.frombuffer(seq.encode("utf-32-le"), dtype=np.uint32), reference)
            .tobytes()
            .decode("utf-32-le")
        )


def row_reference(seq: str, chrom: str, start: int, end: int, fasta) -> np.ndarray:
    """The HG38 base under every character of a stored row (0 under markers and inserted letters).

    The row is aligned to HG38 by the cohort extraction's resync (``embedding/cohort/rows.py``), so a change there
    changes what reference-aware training reads. An indel-free row maps one to one (identical to the direct map on
    300 of 300 rows); on 1kG test rows with indels the token matched the per-sample BCF at 15,593 of 15,649 phased
    heterozygous SNVs, never with the copies swapped (the rest overlap an indel or another allele at that
    position). ``fasta`` is anything with ``fetch(chrom, start, end)`` (pysam.FastaFile).
    """
    hg38 = fasta.fetch(chrom, start, end).upper()
    return reference_under(
        np.frombuffer(hg38.encode("utf-32-le"), dtype=np.uint32),
        cohort_rows.source_offsets(hg38, seq)[1],
    )


def reference_under(hg38: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """HG38 codepoints and each row character's HG38 offset (-2 marker, -1 inserted) -> the base under each character."""
    out = np.zeros(len(offsets), dtype=np.uint32)
    hit = offsets >= 0
    out[hit] = np.asarray(hg38, dtype=np.uint32)[offsets[hit]]
    return out


def swap_copies(codepoints: np.ndarray) -> np.ndarray:
    """A stored row with copy 0 and copy 1 exchanged (``--v2-copy-swap``): every direction token flips."""
    fwd, rev = np.uint32(ord(DIR_FWD)), np.uint32(ord(DIR_REV))
    return np.where(
        codepoints == fwd, rev, np.where(codepoints == rev, fwd, codepoints)
    )


def filtered_tokens_text(tokens: Path | str, dropped: set[str]) -> str:
    """The token file a checkpoint records (``diploid_tokens_used.tsv``): every line but the dropped symbols' rows."""
    return "".join(
        ln
        for ln in Path(tokens).read_text(encoding="utf-8").splitlines(keepends=True)
        if ln.startswith("#")
        or len(ln.split("\t")) < 3
        or ln.split("\t")[2] not in dropped
    )
