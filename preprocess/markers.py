"""Structural marker glyphs and the v8 marker-column round trip, in one place.

The inline-marked ``seq`` (v4–v7) carries ``^`` before a SNP, ``{ … }`` around an
INDEL span and ``‹``/``›`` before a phased-het base (three-level encoding). v8
moves those markers out of ``seq`` into the sparse ``snp`` / ``indel_span`` /
``direction`` columns (``get_genome_sequence.split_markers`` does the split).
This module holds the glyphs, the column decoders and the single emission walk
that puts the markers back — used by the encoder (``rehydrate_markers``), the
dataset reader (``dataset._rehydrate_marked``) and the v8→v7 annotation
realignment (``compat`` via :func:`emission_layout`).

Leaf module, stdlib only: ``dataset.py`` and ``compat.py`` import it without
loading the encoder, cyvcf2 or numpy.
"""

from typing import Dict, Iterable, List, Optional, Tuple

# Default marker glyphs. The token table's <SNP>/<INDEL_span_start>/
# <INDEL_span_end>/<DIR_FWD>/<DIR_REV> meta tokens may override them for the
# encoder; a v8 ``seq`` stores no glyphs, so readers assume this set (production
# never overrides them).
SNP_MARKER = "^"
INDEL_OPEN = "{"
INDEL_CLOSE = "}"
# ‹ (U+2039) = hap0 holds the canonical-first allele; › (U+203A) = the second.
DIR_FWD = "‹"  # ‹
DIR_REV = "›"  # ›


# ---------------------------------------------------------------------------
# v8 marker-column decoders (Arrow ``to_pylist`` values -> plain ints)
# ---------------------------------------------------------------------------


def decode_snp(raw) -> List[int]:
    """Decode the ``snp`` column (list<int32>) to a list of ints."""
    return [int(p) for p in (raw or [])]


def decode_indel_spans(raw) -> List[Tuple[int, int]]:
    """Decode the ``indel_span`` column (list<struct<start,end>>) to ``(start, end)``
    tuples; a ``-1`` in either field marks an edge-truncated block (see
    ``split_markers``)."""
    return [(int(e["start"]), int(e["end"])) for e in (raw or [])]


def decode_direction(raw) -> List[Tuple[int, int]]:
    """Decode the ``direction`` column (list<struct<pos,dir>>) to ``(pos, dir)``
    tuples (dir 1=fwd ``‹``, 2=rev ``›``)."""
    return [(int(e["pos"]), int(e["dir"])) for e in (raw or [])]


# ---------------------------------------------------------------------------
# The emission walk
# ---------------------------------------------------------------------------


def _marker_runs(
    clean_len: int,
    snp_pos: Iterable,
    indel_spans: Iterable,
    dir_events: Iterable,
    snp_marker: str,
    indel_open: str,
    indel_close: str,
    dir_fwd: str,
    dir_rev: str,
) -> List[Tuple[int, str]]:
    """The marker text emitted immediately before each clean base, ascending.

    Returns ``(k, markers)`` pairs, ``0 <= k <= clean_len``, only where ``markers``
    is non-empty. Emission order at a base mirrors the builders: close ``}`` (spans
    ending here) → open ``{`` (spans starting here) → ``^`` → direction → the base.
    ``k == clean_len`` carries only the trailing ``}`` of spans ending at the end.
    A ``-1`` span start/end (edge-truncated block) suppresses that bracket; positions
    outside ``[0, clean_len]``, and direction values other than 1/2, emit nothing;
    a repeated SNP position emits one ``^``; the last direction event at a position wins.
    """
    opens: Dict[int, int] = {}
    closes: Dict[int, int] = {}
    for start, end in indel_spans:
        if start != -1:
            opens[int(start)] = opens.get(int(start), 0) + 1
        if end != -1:
            closes[int(end)] = closes.get(int(end), 0) + 1
    snp_set = {int(p) for p in snp_pos}
    dir_map = {int(p): int(d) for p, d in dir_events}
    dir_glyph = {1: dir_fwd, 2: dir_rev}

    runs: List[Tuple[int, str]] = []
    for k in sorted(closes.keys() | opens.keys() | snp_set | dir_map.keys()):
        if k < 0 or k > clean_len:
            continue
        text = indel_close * closes.get(k, 0)
        if k < clean_len:
            text += (
                indel_open * opens.get(k, 0)
                + (snp_marker if k in snp_set else "")
                + dir_glyph.get(dir_map.get(k), "")
            )
        if text:
            runs.append((k, text))
    return runs


def rehydrate_markers(
    clean_seq: str,
    snp_pos,
    indel_spans,
    dir_events,
    *,
    snp_marker: str = SNP_MARKER,
    indel_open: str = INDEL_OPEN,
    indel_close: str = INDEL_CLOSE,
    dir_fwd: str = DIR_FWD,
    dir_rev: str = DIR_REV,
) -> str:
    """Reconstruct the inline-marked sequence from ``split_markers`` output.

    Exact inverse of ``get_genome_sequence.split_markers``. Emission order at each
    base index mirrors the builders: close ``}`` (spans ending here) → open ``{``
    (spans starting here) → ``^`` (SNP) → direction → the base codepoint; a
    trailing ``}`` is emitted for any span whose ``end == len(clean_seq)``. A
    ``-1`` start or end sentinel marks an edge-truncated block (close-only /
    open-only) and suppresses the corresponding bracket.
    """
    parts: List[str] = []
    prev = 0
    for k, text in _marker_runs(
        len(clean_seq), snp_pos, indel_spans, dir_events,
        snp_marker, indel_open, indel_close, dir_fwd, dir_rev,
    ):
        parts.append(clean_seq[prev:k])
        parts.append(text)
        prev = k
    parts.append(clean_seq[prev:])
    return "".join(parts)


def emission_layout(
    clean_len: int,
    snp: Optional[Iterable],
    indel_spans: Iterable,
    dir_events: Iterable,
) -> Tuple[int, List[int]]:
    """Marked-sequence layout of a v8 record, from the same walk as :func:`rehydrate_markers`.

    Returns ``(marked_len, clean_to_marked)``: ``clean_to_marked[k]`` is the index,
    in the re-inlined marked sequence, of the base at clean index ``k``. Sharing
    the walk keeps annotation realignment (``compat``) in lockstep with the
    ``seq`` rebuild.
    """
    clean_to_marked: List[int] = []
    shift = 0
    prev = 0
    for k, text in _marker_runs(
        clean_len, [] if snp is None else snp, indel_spans, dir_events,
        SNP_MARKER, INDEL_OPEN, INDEL_CLOSE, DIR_FWD, DIR_REV,
    ):
        clean_to_marked.extend(range(prev + shift, k + shift))
        shift += len(text)
        prev = k
    clean_to_marked.extend(range(prev + shift, clean_len + shift))
    return clean_len + shift, clean_to_marked
