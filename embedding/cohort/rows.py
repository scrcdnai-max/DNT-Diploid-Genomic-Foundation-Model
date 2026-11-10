"""Dataset rows, marker handling, and alignment to reference token positions."""

from __future__ import annotations

import csv
import os

import numpy as np

MARKERS = frozenset("^{}‹›")
_IS_MARKER = np.zeros(max(ord(c) for c in MARKERS) + 1, bool)
_IS_MARKER[[ord(c) for c in MARKERS]] = True
BUCKET = 8192          # the training bucket: rows are padded to a multiple of this
# The input rules (`resync_offsets`, `collapse_mapped_haploid`, `onto_hg38_frame`, `extract.content_offsets`), stamped on
# every extraction root so a root is never resumed under other rules. Bump it whenever a change alters what the model
# reads or how its states pair with HG38's.
# 2026-10-10: a hom deletion next to a variant letter or inside a duplicated stretch is placed on its own letters.
INPUT_ENCODING = "2026-10-10"


def is_marker(cps: np.ndarray) -> np.ndarray:
    """Which codepoints are structural markers; anything above the table is not one."""
    return (cps < _IS_MARKER.size) & _IS_MARKER.take(cps, mode="clip")


def strip_markers(seq: str) -> tuple[str, np.ndarray]:
    """-> (content without markers, sorted int32 indices of variant positions in the content).

    Variant positions: the glyph after `^`, every glyph inside `{`...`}`.
    """
    cps = np.frombuffer(seq.encode("utf-32-le"), dtype=np.uint32)
    marker = is_marker(cps)
    mpos = np.flatnonzero(marker)
    mcp = cps[mpos]
    after = mpos - np.arange(len(mpos))             # content index of the glyph after each marker
    clean = cps[~marker].tobytes().decode("utf-32-le")
    snp = after[mcp == ord("^")]
    opens, closes = after[mcp == ord("{")], after[mcp == ord("}")]
    spans = [np.arange(o, c) for o, c in zip(opens, closes)]
    var = np.unique(np.concatenate([snp, *spans]).astype(np.int32)) if len(snp) or spans \
        else np.zeros(0, np.int32)
    return clean, var.astype(np.int32)


def indel_runs(seq: str) -> list[tuple[int, int]]:
    """Content index ranges [start, stop) of the glyphs inside each `{`...`}` of a marked row (markers removed)."""
    cps = np.frombuffer(seq.encode("utf-32-le"), dtype=np.uint32)
    mpos = np.flatnonzero(is_marker(cps))
    after = mpos - np.arange(len(mpos))
    mcp = cps[mpos]
    return [(int(o), int(c)) for o, c in zip(after[mcp == ord("{")], after[mcp == ord("}")]) if c > o]


def reference_token_indices(
    seq: str, ref_idx: list[int] | np.ndarray, n_ref: int, tok: int, wins: list[tuple] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Map HG38 token centers to states from the original marker-bearing sequence.

    Args:
        seq: Exact sequence sent to the diploid checkpoint.
        ref_idx: HG38 offset per codepoint; -2 marks structural tokens, -1 insertions.
        n_ref: Number of HG38 bases.
        tok: Bases per pooled model token.
        wins: `letter_windows` of the row: index the windows' owned token spans laid end to end (what
            `extract.layer_rows` returns), each HG38 token read from the window that owns its centre letter.

    Returns:
        Paired indices into model states and reference states. Fully deleted
        reference tokens are omitted.

    Raises:
        ValueError: If the map is missing, malformed, or inconsistent with the sequence.
    """
    pos = np.asarray(ref_idx, dtype=np.int64)
    if len(seq) != len(pos):
        raise ValueError("offset map length differs from marked sequence length")
    cps = np.frombuffer(seq.encode("utf-32-le"), dtype=np.uint32)
    if np.any(is_marker(cps) != (pos == -2)):
        raise ValueError("offset map marker positions disagree with sequence")
    valid = pos >= 0
    positions = pos[valid]
    if n_ref <= 0 or tok <= 0 or not len(positions):
        raise ValueError("offset map needs mapped bases and positive reference/token lengths")
    if np.any(pos < -2):
        raise ValueError("offset map is invalid")
    if positions.max() >= n_ref:
        raise ValueError("offset map has offsets absent from the reference row")
    if np.any(np.diff(positions) <= 0):
        raise ValueError("offset map mapped offsets must be strictly sorted")
    ref_tokens = np.arange(-(-n_ref // tok))
    starts = ref_tokens * tok
    left_edges = np.searchsorted(positions, starts)
    right_edges = np.searchsorted(positions, np.minimum(starts + tok, n_ref))
    present = left_edges < right_edges
    ref_tokens = ref_tokens[present]
    if not len(ref_tokens):
        raise ValueError("offset map has no reference token with mapped bases")
    centers = np.minimum(ref_tokens * tok + tok // 2, n_ref - 1)
    right = np.minimum(np.searchsorted(positions, centers), right_edges[present] - 1)
    left = np.maximum(right - 1, left_edges[present])
    nearest = np.where(centers - positions[left] <= positions[right] - centers, left, right)
    c = np.flatnonzero(valid)[nearest]                      # each HG38 token's centre letter in the row
    if wins is None:
        return (c // tok).astype(np.int64), ref_tokens.astype(np.int64)
    s, _e, a, b = np.asarray(wins, np.int64).T
    w = np.searchsorted(s + a, c, "right") - 1              # the window owning letter c
    first = a // tok
    base = np.concatenate([[0], np.cumsum(-(-b // tok) - first)])
    return base[w] + (c - s[w]) // tok - first[w], ref_tokens.astype(np.int64)


def letter_windows(offsets: np.ndarray, n_ref: int, max_len: int, stride: int, tok: int) -> list[tuple]:
    """A row's windows cut at the HG38 letters where HG38's own row is cut (`features.windows` over n_ref letters), so
    a letter sits in the same window, with the same context edges, in the person's pass and in HG38's (markers and
    indels shift a row's characters against HG38). -> [(start, end, own_from, own_to)] in the row's characters, the
    owned span relative to start. A cut falls right after an HG38 letter: markers and inserted letters go with the
    letter after them. A window may hold more than max_len characters (its markers and inserted letters).
    (Ticket 6, 1e9b09b.)"""
    from embedding.cohort.features import windows
    if -(-n_ref // tok) <= max_len // tok:        # one window (most rows): the whole row, owned
        return [(0, len(offsets), 0, len(offsets))]
    at = np.flatnonzero(np.asarray(offsets) >= 0)
    hg = np.asarray(offsets)[at]

    def cut(p: int) -> int:
        if p >= n_ref:
            return len(offsets)
        k = int(np.searchsorted(hg, p))           # the first HG38 letter at or after p; cut after the one before it
        return int(at[k - 1]) + 1 if k else 0
    out = []
    for s, e, a, b in windows(n_ref, max_len, stride, tok):
        start = cut(s)
        out.append((start, cut(e), cut(s + a * tok) - start, cut(min(s + b * tok, e)) - start))
    return out


def onto_hg38_frame(content: str, offsets: np.ndarray, n_ref: int, tok: int) -> str:
    """A marker-free haploid row -> HG38's length, so input token t covers exactly HG38 token t.

    Each token holds its own bases in row order: a deleted base is removed and the token's tail filled with N, an
    inserted base (offset -1) sits where it was inserted, in the token of the HG38 base after it, and the bases it
    pushes past the token's end are cut (the grid resyncs at every token, as the pre-9f878a1 `reframe` rule did for
    deletions; no untrained `-` reaches the model). If the tail cut would give back the token's own bases, the head is
    cut instead."""
    cp = np.frombuffer(content.encode("utf-32-le"), np.uint32)
    off = np.asarray(offsets)
    out = np.full(n_ref, _GAP, np.uint32)
    out[off[off >= 0]] = cp[off >= 0]
    ins = np.flatnonzero(off == -1)
    blocks = set((np.flatnonzero(out == _GAP) // tok).tolist())
    if len(ins):
        mapped = np.flatnonzero(off >= 0)
        nxt = np.minimum(np.searchsorted(mapped, ins), len(mapped) - 1)   # the next HG38 base, else the last one
        bins = np.where(off >= 0, off, 0) // tok
        bins[ins] = off[mapped[nxt]] // tok
        blocks |= set(bins[ins].tolist())
        sel = off >= -1
        bases, base_bins = cp[sel], bins[sel]                             # once, not per edited token
    for blk in sorted(blocks):
        s = out[blk * tok:(blk + 1) * tok]              # a view: edits land in `out`
        keep = bases[base_bins == blk] if len(ins) else s[s != _GAP]
        cut = keep[:len(s)]
        if len(keep) > len(s) and (cut == s).all():   # an insertion repeating to the token's end: the tail cut would
            cut = keep[-len(s):]                       # hide it, so cut the head (a variant on that base is lost)
        s[:len(cut)], s[len(cut):] = cut, _N
    return out.tobytes().decode("utf-32-le")


def onto_hg38_inplace(seq: str, offsets: np.ndarray, n_ref: int) -> str:
    """A diploid row -> HG38's length with each site's own glyph (a base, a pair glyph, a het-deletion glyph) and no
    markers: a deleted base (no character of its own) reads N in its slot, an inserted base (offset -1) has no slot."""
    cp = np.frombuffer(seq.encode("utf-32-le"), np.uint32)
    off = np.asarray(offsets)
    out = np.full(n_ref, _N, np.uint32)
    out[off[off >= 0]] = cp[off >= 0]
    return out.tobytes().decode("utf-32-le")


def hg38_bins(offsets: np.ndarray, tok: int) -> np.ndarray:
    """HG38 bin (offset // tok) of every input character: markers (-2) and inserted bases (-1) join the bin of
    the last HG38 base before them, leading ones the first base's bin. Pooling model states by these bins
    compares a person with HG38 bin for bin, however markers and indels move the model's own token grid."""
    off = np.asarray(offsets, np.int64)
    mapped = np.flatnonzero(off >= 0)
    last = np.maximum.accumulate(np.where(off >= 0, np.arange(len(off)), -1))
    return off[np.where(last >= 0, last, mapped[0])] // tok


def _deletes(a: np.ndarray, b: np.ndarray, i: int, e: int, j: int, wild: np.ndarray, stop: np.ndarray,
             probe: int, widest: int) -> bool:
    """Does the indel span a[i:e] consume HG38 bases b[j:j + (e - i)] (a deletion) rather than none (an insertion)?
    Whichever reading lines the next `probe` row bases (up to the next span) up with HG38 better, in this order:
    1. the reading that matches more of them, unless neither fits them all (another edit sits among them): then the
       reading under which fewer HG38 letters must be lost, after the letters that do fit, for the rest to fit
       (`_lost`; 10-10: a hom deletion soon after a hom insertion of its length; het deletions on opposite copies,
       whose shared letters are gone on both);
    2. on a tie cut short by another span: the `probe` letters after that span, once under exactly one reading of this
       span they fit HG38 under either reading of that one (10-10: two het insertions one letter apart in a run);
    3. on a tie over the whole window (a repeat longer than the probe: a het insertion and a het deletion of its unit
       carry the same glyphs): wider windows up to `widest`, once exactly one reading fits one exactly;
    4. else a deletion only if every glyph is a gap or a base-and-gap pair on that HG38 base.
    A span is read whole: one that holds inserted glyphs and then het-deleted ones (an insertion and a deletion called
    at the same letter; 2 of 79,995 BioMe spans checked) gets one reading for all its glyphs, and a deletion's glyphs
    take consecutive HG38 letters (letters gone on both copies between two of them are found after the span)."""
    def window(p: int, at: int = e) -> int:
        cut = np.flatnonzero(stop[at:at + p])
        return int(cut[0]) if len(cut) else min(p, len(a) - at)

    def score(jj: int, m: int, at: int = e) -> int:
        ref = b[jj:jj + m]
        return int(((a[at:at + len(ref)] == ref) | wild[at:at + len(ref)]).sum())

    def loss(jj: int) -> float:
        """HG38 letters to lose after the row letters that fit b[jj:], for the next `probe` to fit; inf if none does."""
        bad = np.flatnonzero((a[e:e + m] != b[jj:jj + m]) & ~wild[e:e + m]) if jj + m <= len(b) else a[:0]
        return (len(bad) and _lost(a, b, e + int(bad[0]), jj + int(bad[0]), wild, stop, probe, 2 * probe)) or np.inf
    jd = j + e - i                                          # where HG38 resumes if the span is a deletion
    m = window(probe)
    fit_ins, fit_del = score(j, m), score(jd, m)
    if m == probe and fit_ins < m > fit_del:                # a whole window: a few letters before a span say little
        need_ins, need_del = loss(j), loss(jd)
        if need_ins != need_del:
            return need_del < need_ins
    if fit_ins != fit_del:
        return fit_del > fit_ins
    after = np.flatnonzero(~stop[e + m:e + m + widest]) if m < probe else a[:0]   # a span cut the window: its length
    if len(after):
        n, at = int(after[0]), e + m + int(after[0])
        w = window(probe, at)
        ins, dele = (any(score(jj + m + c, w, at) == w for c in (0, n)) for jj in (j, jd))
        if ins != dele:
            return dele
    while m == probe < widest:              # a fit, not a count; a short HG38 slice (the row end) cannot fit
        probe = min(4 * probe, widest)
        m = window(probe)
        ins, dele = score(j, m) == m, score(jd, m) == m
        if ins != dele:
            return dele
    glyph = a[i:e]
    lookup = np.minimum(glyph, len(_ALLELE_A) - 1)
    x = np.where(glyph < len(_ALLELE_A), _ALLELE_A[lookup], glyph)
    y = np.where(glyph < len(_ALLELE_B), _ALLELE_B[lookup], glyph)
    gap_x, gap_y = np.isin(x, [_GAP, _STAR]), np.isin(y, [_GAP, _STAR])
    base = b[j:j + len(glyph)]
    return len(base) == len(glyph) and bool(((gap_x | gap_y) & np.where(gap_x, gap_y | (y == base),
                                                                         x == base)).all())


def _lost(a: np.ndarray, b: np.ndarray, i: int, j: int, wild: np.ndarray, stop: np.ndarray, probe: int,
          max_shift: int) -> int:
    """HG38 letters lost before row letter i, outside any marked span: the smallest s that puts the next `probe` row
    letters (up to the next span) on b[j + s:], a variant glyph matching any letter; 0 if none within `max_shift`.
    A fit on n letters is chance once in 4^n placements. Near shifts (up to 64) are tried while they number at most
    4^n / 16, further ones only up to 4^n / 65,536 (all 4,096 from 14 letters on): a window cut to a few letters by a
    span or the row end still finds a loss of a few letters (a hom deletion three letters before a het one in a GA
    repeat, or five before the row end) and cannot send the row thousands of letters on (review 10-10: with any
    length believed, 2 of 3 six-letter windows fit somewhere).
    Small shifts are tried first: of 6,521 hom deletions in 3,960 BioMe rows the longest lost 31 letters.
    ponytail: a stretch with no fit pays the whole search at every letter (~0.1 ms); cache the miss if such rows show."""
    cut = np.flatnonzero(stop[i:i + probe])
    m = int(cut[0]) if len(cut) else min(probe, len(a) - i)
    keep = ~wild[i:i + m]
    want = a[i:i + m][keep]
    max_shift = min(max_shift, max(min(64, 4 ** len(want) >> 4), 4 ** len(want) >> 16))
    if not max_shift:
        return 0
    for lo, hi in ((1, min(64, max_shift)), (65, max_shift)):
        ref = b[j + lo:j + hi + m]
        if len(ref) < m:
            break
        fit = np.flatnonzero((np.lib.stride_tricks.sliding_window_view(ref, m)[:, keep] == want).all(1))
        if len(fit):
            return lo + int(fit[0])
    return 0


def _after_loss(a: np.ndarray, b: np.ndarray, i: int, j: int, lost: int, wild: np.ndarray, first: int) -> int:
    """How many row letters before i (back to `first`, where the matching run began) belong after the `lost` HG38
    letters found at (i, j). A variant glyph matches any HG38 letter, so the scan walks past a loss that a variant
    letter follows and finds it late. The call set left-aligns a deletion: the loss goes before the furthest variant
    letter from which every letter up to i also fits `lost` letters on (a plain letter is HG38's there; a het glyph
    holds HG38's letter; a hom one is another letter). Plain letters alone are never moved (a repeat: same pairing)."""
    out = 0
    for k in range(1, i - first + 1):
        g, r = int(a[i - k]), b[j - k + lost]
        if not wild[i - k]:
            if g != r:
                break
            continue
        x, y = (int(_ALLELE_A[g]), int(_ALLELE_B[g])) if g < len(_ALLELE_A) else (g, g)
        if (x == r or y == r) != (x != y):
            break
        out = k
    return out


def resync_offsets(content: str, ref, var: np.ndarray, probe: int = 32, max_shift: int = 4096,
                   runs: list[tuple[int, int]] | None = None) -> np.ndarray:
    """HG38 offset of each content base (-1 = inserted), recovered from the sequence alone.

    Rows carry indels by changing length, which shifts every downstream token off the pooling
    grid; the offsets pair each HG38 token with the model state over its bases instead. Matching
    runs are compared in bulk (a variant glyph matches any HG38 letter). At a mismatch outside a marked span HG38
    letters are lost, the fewest that re-synchronise (`_lost`: the row's next `probe` letters found further on in
    HG38, variant glyphs among them matching anything), and the loss is placed before the variant letters it precedes
    (`_after_loss`): a deletion on both copies is the only edit rows carry unmarked (no unmarked inserted letter in
    11,823 encoder maps). Only where no loss fits are row letters read as inserted (HG38's next `probe` letters found
    further on in the row: a way back into step after a span read wrong; a row without `runs` may hold unmarked
    insertions and takes the smaller of the two edits); before 10-10 that search ran first and the loss search was
    exact, so a loss inside a duplicated stretch or next to a variant was misread. An unresolved mismatch is read as
    a substitution, so every row keeps its full alignment.
    10-10, against the encoder's own map on 11,823 BioMe rows (1.18 billion letters;
    benchmarks_results/readout_fix_20261007/resync_1010): 8 letters off their HG38 letter by more than a shift inside
    a repeat (80,246 before; a hom deletion is placed at the right end of its repeat, the call set writes the left
    end). `ref` is the gap-filtered reference row, str or codepoints.
    `runs` (`indel_runs`): the glyphs inside `{}`. Each span is read whole, as a deletion that consumes HG38 bases or
    an insertion that consumes none (`_deletes`); without it an inserted glyph took the next HG38 base's offset and
    that base was read as inserted, so a het insertion changed both copies of a phased row.
    """
    b_str = ref if isinstance(ref, str) else ref.tobytes().decode()
    a = np.frombuffer(content.encode("utf-32-le"), np.uint32)
    b = (np.frombuffer(b_str.encode("utf-32-le"), np.uint32) if isinstance(ref, str)
         else ref.astype(np.uint32))
    isvar = np.zeros(len(a), bool)
    isvar[var[(var >= 0) & (var < len(a))]] = True
    inspan = np.zeros(len(a), bool)
    ends = {}
    for s, e in runs or ():
        inspan[s:e] = True
        ends[s] = e
    wild = isvar & ~inspan                                   # an SNV glyph matches whatever HG38 base it sits on
    if len(a) == len(b) and not ((a != b) & ~isvar).any() and not inspan.any():
        return np.arange(len(a), dtype=np.int32)            # already on the frame
    off = np.full(len(a), -1, np.int32)
    i = j = 0
    while i < len(a) and j < len(b):
        n = min(len(a) - i, len(b) - j)
        k, w = 0, 512                                       # bases already on the frame: scan a
        while k < n:                                        # growing window, not the whole rest
            m = min(w, n - k)                               # of the row on every iteration
            bad = np.flatnonzero(((a[i + k:i + k + m] != b[j + k:j + k + m]) & ~wild[i + k:i + k + m])
                                 | inspan[i + k:i + k + m])
            if len(bad):
                k += int(bad[0])
                break
            k, w = k + m, w * 4
        off[i:i + k] = np.arange(j, j + k)
        i, j = i + k, j + k
        if k == n:
            break
        if i in ends:                                       # an indel span: consumes HG38 bases or none
            e = ends[i]
            if _deletes(a, b, i, e, j, wild, inspan, probe, max_shift):   # tie windows reach as far as the search
                take = min(e - i, len(b) - j)
                off[i:i + take] = np.arange(j, j + take)
                j += take
            i = e
            continue
        lost = _lost(a, b, i, j, wild, inspan, probe, max_shift)
        # rows with marked spans carry no unmarked insertion: a loss that fits is the reading. Without `runs` one may
        hit = -1 if lost and runs is not None else content.find(b_str[j:j + probe], i + 1, i + max_shift + probe)
        if hit >= 0 and inspan[hit]:                        # a span is read from its start, never entered midway
            hit = -1
        if lost and (hit < 0 or lost <= hit - i):           # ref[j:j + lost] lost (without `runs`: the smaller edit)
            off[i - _after_loss(a, b, i, j, lost, wild, i - k):i] += lost
            j += lost
            continue
        if hit >= 0:                                        # content[i:hit] inserted
            i = hit
            continue
        off[i] = j                                          # substitution: step both
        i, j = i + 1, j + 1
    return off


def source_offsets(ref_seq, seq: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (variant positions, HG38 offset of every source character: -2 marker, -1 inserted), resynced to the HG38 row."""
    cps = np.frombuffer(seq.encode("utf-32-le"), np.uint32)
    bases, var = strip_markers(seq)
    offsets = np.full(len(cps), -2, np.int32)
    offsets[~is_marker(cps)] = resync_offsets(bases, ref_seq, var, runs=indel_runs(seq))
    return var, offsets


def letters(seq: str, hg_cps: np.ndarray, source: tuple | None = None) -> dict:
    """A marked diploid row against its HG38 row -> offsets (HG38 offset per character: -2 marker, -1 inserted), het
    and hom variant letters (a mapped letter whose glyph is not HG38's base: a pair glyph is het, a plain glyph hom)
    and the counts block (het letters, hom letters, inserted letters, deleted HG38 letters, markers, variant
    positions). `source`: `source_offsets(hg_cps, seq)` if already computed. (onepass437, 10-07; its `letters` adds
    the L1 letters.)"""
    var, off = source or source_offsets(hg_cps, seq)
    cps = np.frombuffer(seq.encode("utf-32-le"), np.uint32)
    mapped = off >= 0
    base = np.zeros(len(cps), np.uint32)
    base[mapped] = hg_cps[off[mapped]]
    diff = mapped & (cps != base)
    look = np.minimum(cps, len(_ALLELE_A) - 1)
    pair = (cps < len(_ALLELE_A)) & (_ALLELE_A[look] != _ALLELE_B[look])
    counts = [int((diff & pair).sum()), int((diff & ~pair).sum()), int((off == -1).sum()),
              len(hg_cps) - int(mapped.sum()), int((off == -2).sum()), len(var)]
    return {"off": off, "het": diff & pair, "hom": diff & ~pair, "counts": counts}


def variant_letters(a: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`letters` of a row -> its variant letters only: (character index, HG38 offset, het flag) of each."""
    iv = np.flatnonzero(a["het"] | a["hom"])
    return iv, a["off"][iv], a["het"][iv]


def bucket_len(n: int) -> int:
    return -(-n // BUCKET) * BUCKET



# --- haploid collapse: diploid glyphs -> a vocab of A C G T N (vanilla NTv3) --------------- #
def _allele_luts(tsv: str = os.path.join(os.path.dirname(__file__), "..", "..", "preprocess",
                                         "diploid_tokens.tsv")) -> tuple[np.ndarray, np.ndarray]:
    """Codepoint -> first / second allele codepoint, identity for anything the TSV does not pair.
    Pair rows give both alleles (first occurrence wins, as `pretrain.train._parse_diploid_tsv`);
    single rows (hemizygous `a`, `.`, `·`) give their one allele twice."""
    pairs: dict[str, str] = {}
    with open(tsv) as f:
        for row in csv.reader((l for l in f if not l.startswith("#")), delimiter="\t"):
            if len(row) >= 3 and len(row[2]) == 1 and len(row[0]) in (1, 2):
                pairs.setdefault(row[2], row[0] * (3 - len(row[0])))
    size = max(ord(s) for s in pairs) + 1
    a, b = np.arange(size, dtype=np.uint32), np.arange(size, dtype=np.uint32)
    for s, p in pairs.items():
        a[ord(s)], b[ord(s)] = ord(p[0]), ord(p[1])
    return a, b


_ALLELE_A, _ALLELE_B = _allele_luts()


def _unphased_glyphs(tsv: str = os.path.join(os.path.dirname(__file__), "..", "..", "preprocess",
                                             "diploid_tokens.tsv")) -> frozenset[str]:
    """Het glyphs a marked row carries at an HG38 base: an unphased pair of two alleles from A C G T and the gap
    (a het SNV, or a het deletion), never both gaps."""
    out = set()
    with open(tsv) as f:
        for row in csv.reader((l for l in f if not l.startswith("#")), delimiter="\t"):
            if (len(row) >= 3 and row[1] in ("unphased", "unordered") and len(row[0]) == 2 and row[0][0] != row[0][1]
                    and set(row[0]) <= set("ACGT-")):
                out.add(row[2])
    return frozenset(out)


UNPHASED_GLYPHS = _unphased_glyphs()
DIPLOID_GLYPHS = frozenset(
    int(cp) for cp in np.flatnonzero(_ALLELE_A != np.arange(len(_ALLELE_A)))
    if cp > 127
)
_GAP, _N, _STAR = ord("-"), ord("N"), ord("*")
_ACGTN = np.array([ord(c) for c in "ACGTN"], np.uint32)


def _drop_orphan_markers(cps: np.ndarray, marker: np.ndarray, keep: np.ndarray) -> None:
    """In `keep`: drop a variant marker whose base was dropped, and a brace pair (or a lone brace) that encloses none."""
    kept_bases = keep & ~marker
    base_positions = np.flatnonzero(~marker)
    for marker_pos in np.flatnonzero(np.isin(cps, [ord("^"), ord("‹"), ord("›")])):
        at = np.searchsorted(base_positions, marker_pos)
        if at == len(base_positions) or not kept_bases[base_positions[at]]:
            keep[marker_pos] = False
    base_prefix = np.concatenate(([0], np.cumsum(kept_bases)))
    openings: list[int] = []
    for at in np.flatnonzero(np.isin(cps, [ord("{"), ord("}")])):
        if cps[at] == ord("{"):
            openings.append(int(at))
        elif openings:
            opening = openings.pop()
            if base_prefix[at] == base_prefix[opening + 1]:
                keep[[opening, int(at)]] = False
        elif base_prefix[at] == 0:
            keep[at] = False
    for opening in openings:
        if base_prefix[-1] == base_prefix[opening + 1]:
            keep[opening] = False


def collapse_mapped_haploid(
    seq: str, source_offsets: np.ndarray, ref_seq: np.ndarray, markers: frozenset[str],
    collapse: str = "alt",
) -> tuple[str, np.ndarray]:
    """Convert diploid glyphs to one allele per site while retaining supported markers and offsets.

    Args:
        seq: Inline-marked diploid or haploid source.
        source_offsets: Reference row positions, with -1 insertions and -2 markers.
        ref_seq: Reference base codepoints in the gap-filtered sequence.
        markers: Structural codepoints supported by the checkpoint.
        collapse: `alt` writes a site's alt allele if either copy carries it; `hom_alt` only if both do: any pair
            glyph (het, including het indels and two different alts) shows the reference base. One pass of each keeps
            zygosity marker-free. `hap0` / `hap1` write one phased copy each: a pair glyph's direction marker (the
            character before it) says which copy holds which allele, `‹` copy 0 the first (lower-priority) allele and
            `›` the second (hf_cards/diploid_encode.py); the two passes keep zygosity and phase.
    Returns:
        Converted input and its parallel source offsets.
    Raises:
        ValueError: If the source map is inconsistent with the input.
    """
    pos = np.asarray(source_offsets, np.int64)
    if len(seq) != len(pos):
        raise ValueError("offset map length differs from sequence length")
    cps = np.frombuffer(seq.encode("utf-32-le"), np.uint32)
    marker = np.isin(cps, [ord(c) for c in MARKERS])
    if np.any(marker != (pos == -2)):
        raise ValueError("offset map marker positions disagree with sequence")
    reference = np.full(len(cps), _N, np.uint32)
    mapped = (pos >= 0) & (pos < len(ref_seq))
    reference[mapped] = ref_seq[pos[mapped]]
    lookup = np.minimum(cps, len(_ALLELE_A) - 1)
    first = np.where(cps < len(_ALLELE_A), _ALLELE_A[lookup], cps)
    second = np.where(cps < len(_ALLELE_B), _ALLELE_B[lookup], cps)
    if collapse == "alt":
        converted = np.where(first == reference, second, first)
        inserted_gap = (pos == -1) & (first == _GAP)
        converted[inserted_gap] = second[inserted_gap]
    elif collapse == "hom_alt":   # a pair glyph is on one copy each: show the reference (an unmapped insertion: none)
        converted = np.where(first == second, first, np.where(pos == -1, _GAP, reference))
    elif collapse in ("hap0", "hap1"):
        before = np.concatenate(([0], cps[:-1])).astype(np.uint32)
        fwd, rev = before == ord("‹"), before == ord("›")
        pair = (first != second) & ~marker
        if np.any(pair & ~(fwd | rev)):
            raise ValueError("a het glyph without its direction marker: the haplotype views need phased rows")
        converted = np.where(pair & (rev if collapse == "hap0" else fwd), second, first)
    else:
        raise ValueError(f"unknown haploid collapse {collapse!r}; use alt, hom_alt, hap0 or hap1")
    converted[converted == _STAR] = _GAP
    keep = ((~marker) & (converted != _GAP)) | np.isin(cps, [ord(c) for c in markers])
    converted[marker] = cps[marker]
    if markers:          # drop kept markers that no longer mark a base (a marker-free model keeps none: 14% of host prep)
        _drop_orphan_markers(cps, marker, keep)
    bad = np.unique(converted[keep & ~marker & ~np.isin(converted, _ACGTN)])
    if len(bad):
        raise ValueError(f"not collapsible to A C G T N: {[chr(c) for c in bad]}")
    return converted[keep].astype(np.uint32).tobytes().decode("utf-32-le"), pos[keep].astype(np.int32)
