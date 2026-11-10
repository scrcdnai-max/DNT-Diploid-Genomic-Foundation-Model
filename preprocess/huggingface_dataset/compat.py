"""Cross-version compatibility + on-the-fly "downhill" conversion of dataset shards.

The schema versions (see :mod:`versions`) form a strict lattice: a higher version
is a superset of a lower one — it adds columns, or moves the structural markers
OUT of ``seq`` into sparse columns. Going *downhill* therefore needs NO source
data (no BCF re-parse, no PhyloP/CADD lookup): you only drop columns and/or
re-inline markers that the shards already carry. This module is the single source
of truth for which conversions are feasible and the per-shard ``pyarrow.Table``
transform that performs each one.

Feasible (downhill) routes — the only ones offered:

  * v7 → v6  (DROP_ANNOTATIONS)   — v7 ``seq`` is byte-identical to v6; drop the
                                    optional ``phylop`` / ``cadd_phred`` columns.
  * v8 → v6  (REHYDRATE_DROP)     — v8 stores pure-base ``seq`` + sparse
                                    ``snp``/``indel_span``/``direction``; rebuild
                                    the inline-marked ``seq`` via
                                    ``rehydrate_markers`` and drop every extra
                                    column (v6 carries no annotations).
  * v8 → v7  (REHYDRATE_REALIGN)  — as above, but KEEP annotations: splice the
                                    PhyloP NaN sentinel into ``phylop`` at the
                                    re-inserted marker codepoints and remap the
                                    sparse ``cadd_phred`` ``pos`` from clean-seq
                                    to marked-seq coordinates.
  * v6 → v6nd (STRIP_DIRECTION)   — v6 uses the three-level encoding: the diploid
                                    *symbol* in ``seq`` is already the unphased
                                    token; only the ``‹``/``›`` direction markers
                                    carry phase.  Deleting those two codepoints
                                    from every ``seq`` string produces a valid
                                    v6nd shard with no token-table lookup needed.
                                    **Requires pre-dedup normalisation** (see
                                    below): two v6 records that differ only in
                                    direction markers are distinct v6 strings but
                                    produce identical v6nd strings, so dedup must
                                    hash the *normalised* seq (via
                                    :func:`route_seq_normalizer`) — not the raw
                                    source seq — to merge them correctly before
                                    the Step-4 shard transform runs.

Everything else is INCOMPATIBLE (uphill — needs regeneration from source):
adding annotation capability (v6 → v7/v8), changing the seq content axes
(v5 ↔ v6 anchor-SNP), splitting markers out (v6/v7 → v8), or — for
an annotation-preserving route — differing on ``marker_policy`` (the realign only
emits the NAN-policy annotation layout; no such divergent preset ships today).
v6nd → anything is always INCOMPATIBLE because direction/phase cannot be recovered.

Dedup correctness for the REHYDRATE routes: assemble hashes the *source* v8
shards with the compound key (:func:`dedup._record_key_bytes`), which is an
injective serialization of ``(seq, snp, indel_span, direction)``.
:func:`preprocess.markers.rehydrate_markers` is a deterministic bijection on
that same 4-tuple, so deduping on the compound key yields exactly the unique
groups deduping on the rehydrated ``seq`` would — the hash step needs no change
and the transform runs only on surviving rows. (A property test in
``test_compat.py`` codifies this.)

Dedup correctness for STRIP_DIRECTION: unlike the REHYDRATE routes, the source
v6 key is NOT a bijection of the target v6nd key — two v6 records with opposite
phasings hash differently in v6 but map to the same v6nd ``seq``.  Callers MUST
pass :func:`route_seq_normalizer` to the Step-2 hasher so dedup hashes the
normalised seq; the :attr:`RouteDecision.pre_dedup` flag documents this
requirement.

Leaf module: imports only stdlib + ``pyarrow`` + :mod:`versions` and
:mod:`preprocess.markers` — so ``create_dataset`` imports it without a cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

import pyarrow as pa
import pyarrow.compute as pc

from preprocess.huggingface_dataset import versions
from preprocess.markers import (
    DIR_FWD as _DIR_FWD,
    DIR_REV as _DIR_REV,
    decode_direction,
    decode_indel_spans,
    emission_layout,
    rehydrate_markers,
)

# Column names (must match create_dataset._BASE_FIELDS + the v7/v8 extras).
_SEQ = "seq"
_SNP = "snp"
_INDEL_SPAN = "indel_span"
_DIRECTION = "direction"
_MARKER_COLS = (_SNP, _INDEL_SPAN, _DIRECTION)
_PHYLOP = "phylop"
_CADD = "cadd_phred"
_ANNOTATION_COLS = (_PHYLOP, _CADD)

# Re-built column types — must match create_dataset / annotations exactly so a
# converted shard is byte-schema-identical to a natively generated one.
_SEQ_TYPE = pa.large_utf8()
_PHYLOP_TYPE = pa.list_(pa.uint8())

# PhyloP NaN sentinel — kept in lockstep with annotations._PHYLOP_NAN_SENTINEL
# (a test asserts they agree). Marker codepoints carry NaN under v7's NAN policy.
_PHYLOP_NAN_SENTINEL = 255


# str.translate table that deletes both direction markers — used by
# strip_direction_from_seq and the Step-2 dedup normaliser.
_DIR_STRIP_TABLE = str.maketrans("", "", _DIR_FWD + _DIR_REV)


def strip_direction_from_seq(s: str) -> str:
    """Delete ``‹``/``›`` direction markers from a v6 ``seq`` string.

    Public so ``create_dataset`` can pass it as the Step-2 dedup normaliser
    without importing a private symbol.  Idempotent: calling it twice is safe.
    """
    return s.translate(_DIR_STRIP_TABLE)


class Route(Enum):
    """How (or whether) one schema version can be converted into another."""

    IDENTITY = "identity"                    # source == target; no transform
    DROP_ANNOTATIONS = "drop_annotations"    # v7 → v6: drop phylop/cadd columns
    REHYDRATE_DROP = "rehydrate_drop"        # v8 → v6: re-inline markers, drop extras
    REHYDRATE_REALIGN = "rehydrate_realign"  # v8 → v7: re-inline + realign annotations
    STRIP_DIRECTION = "strip_direction"      # v6 → v6nd: strip ‹/› from seq (pre-dedup)
    INCOMPATIBLE = "incompatible"            # uphill — regenerate from source


@dataclass(frozen=True)
class RouteDecision:
    """A route plus a human-readable reason (always set for INCOMPATIBLE).

    ``pre_dedup`` is True when the route's seq normaliser MUST be applied to
    the dedup hash key (Step 2) rather than only to surviving rows (Step 4).
    Callers should use :func:`route_seq_normalizer` to retrieve the callable.
    """

    route: Route
    reason: str
    pre_dedup: bool = False


def conversion_route(source_version: str, target_version: str) -> RouteDecision:
    """Decide how to convert *source_version* shards into *target_version*.

    Single source of truth for the compatibility matrix. Decides from the
    resolved :class:`versions.DatasetVersion` *fields* (not the version strings)
    so the matrix stays correct if the presets evolve.
    """
    try:
        src = versions.resolve(source_version)
        tgt = versions.resolve(target_version)
    except ValueError as exc:
        return RouteDecision(Route.INCOMPATIBLE, str(exc))

    if source_version == target_version:
        return RouteDecision(Route.IDENTITY, "source and target are the same version")

    # 0. Region set must match. Every route below is a per-row rewrite of `seq`
    #    and its companion columns — none of them can add, drop, or re-window
    #    ROWS. Two versions over different BEDs hold different (chrom,start,end)
    #    records, so relabelling one as the other would be a false provenance
    #    claim even though every encoder field agrees. This check must come
    #    first: with identical encoder fields (e.g. v7-ldb vs v7) the routes
    #    below would otherwise happily return DROP_ANNOTATIONS with the reason
    #    "seq is identical" — true of the encoding, false of the dataset.
    if src.region_set != tgt.region_set:
        return RouteDecision(
            Route.INCOMPATIBLE,
            f"{source_version}→{target_version}: different region sets "
            f"({src.region_set} vs {tgt.region_set}) — the datasets cover "
            f"different BED regions, so their rows are not the same records. "
            f"Regenerate over the target's BED.",
        )

    # 1. Seq-content axes must match — these determine the actual symbols in
    #    ``seq`` and cannot be reconstructed without re-encoding from the variant
    #    calls. (anchor-SNP '^' detection, the ‹/› direction tokens, and the
    #    phased-vs-unphased symbol table.)
    if src.emit_anchor_snp_marker != tgt.emit_anchor_snp_marker:
        return RouteDecision(
            Route.INCOMPATIBLE,
            f"{source_version}→{target_version}: differ on anchor-SNP '^' "
            f"detection, which needs the variant calls — regenerate.",
        )

    # v6 → v6nd: the three-level encoding stores the *unphased* diploid symbol
    # at each position and uses ‹/› only to carry direction.  Deleting those two
    # codepoints from every seq string is sufficient — no re-encode needed.
    # Guard on src.allow_annotations=False (i.e. not v7/v8 sources) because
    # transform_strip_direction does not drop annotation columns; extend to
    # v7→v6nd with a combined DROP+STRIP route if that becomes needed.
    # This is a downhill, lossy route: phase cannot be recovered, so the reverse
    # (v6nd → v6/v7/v8) falls through to the INCOMPATIBLE guard below.
    # The target must also force unphased: v4 likewise has no ‹/› but keeps phase
    # in PHASED symbols (Ã vs Ğ), so stripping v5 would leave unphased symbols
    # stamped "v4". Without this condition v5→v4 routed here (fixed 2026-10-06).
    if (
        src.use_direction_markers
        and not tgt.use_direction_markers
        and tgt.force_unphased
        and not src.allow_annotations
        and not src.externalize_markers
    ):
        return RouteDecision(
            Route.STRIP_DIRECTION,
            f"{source_version}→{target_version}: v6 embeds unphased symbols "
            f"with ‹/› direction markers; strip those two codepoints from seq.",
            pre_dedup=True,
        )

    if src.use_direction_markers != tgt.use_direction_markers or (
        src.force_unphased != tgt.force_unphased
    ):
        return RouteDecision(
            Route.INCOMPATIBLE,
            f"{source_version}→{target_version}: differ on phase/direction "
            f"encoding (dropped phase cannot be reconstructed) — regenerate.",
        )

    # 2. Target adds per-base annotation capability the source lacks: the
    #    PhyloP/CADD values are not in the shards and cannot be synthesized.
    if tgt.allow_annotations and not src.allow_annotations:
        return RouteDecision(
            Route.INCOMPATIBLE,
            f"{source_version}→{target_version}: target adds per-base annotation "
            f"columns; the source shards carry no PhyloP/CADD data — regenerate "
            f"with --phylop-bw / --cadd-arrow-dir.",
        )

    # 3. Target externalizes markers the source has inline (v6/v7 → v8): the
    #    marker-split is an uphill step we deliberately do not offer here.
    if tgt.externalize_markers and not src.externalize_markers:
        return RouteDecision(
            Route.INCOMPATIBLE,
            f"{source_version}→{target_version}: marker externalization (split "
            f"into snp/indel_span/direction columns) is not offered as on-the-fly "
            f"assemble conversion — regenerate with --version {target_version}.",
        )

    # 4. Same seq-content axes + downhill on annotations/externalization → feasible.
    if src.externalize_markers and not tgt.externalize_markers:
        # v8 → v6/v7: re-inline markers. Keep+realign annotations iff the target
        # allows them (v7), else drop everything extra (v6).
        if tgt.allow_annotations:
            # The realign emits the PhyloP NaN sentinel at every re-inserted
            # marker codepoint — correct only under the NAN marker policy. If a
            # future preset pairs annotations with a different policy (e.g. SKIP,
            # where annotation arrays carry no marker slots at all), the scatter
            # would silently misalign. Guard on the field so the matrix honours
            # its own "stays correct if the presets evolve" invariant.
            if src.marker_policy != tgt.marker_policy:
                return RouteDecision(
                    Route.INCOMPATIBLE,
                    f"{source_version}→{target_version}: differ on marker_policy "
                    f"({src.marker_policy.value}→{tgt.marker_policy.value}); the "
                    f"annotation realign only emits the NAN-policy layout — "
                    f"regenerate.",
                )
            return RouteDecision(
                Route.REHYDRATE_REALIGN,
                f"{source_version}→{target_version}: re-inline markers into seq "
                f"and realign annotation columns to marked coordinates.",
            )
        return RouteDecision(
            Route.REHYDRATE_DROP,
            f"{source_version}→{target_version}: re-inline markers into seq and "
            f"drop the marker + annotation columns.",
        )

    if not src.externalize_markers and not tgt.externalize_markers:
        # Same inline seq content; only the annotation columns differ. The only
        # distinct feasible pair here is v7 → v6 (target forbids annotations).
        return RouteDecision(
            Route.DROP_ANNOTATIONS,
            f"{source_version}→{target_version}: seq is identical; drop the "
            f"optional annotation columns.",
        )

    # Defensive: any combination not handled above (e.g. v8 → v8 with differing
    # presets, which cannot occur) is treated as incompatible rather than guessed.
    return RouteDecision(
        Route.INCOMPATIBLE,
        f"{source_version}→{target_version}: no straightforward downhill route "
        f"— regenerate.",
    )


# ---------------------------------------------------------------------------
# Per-shard transforms (pyarrow.Table -> pyarrow.Table)
# ---------------------------------------------------------------------------


def _present_columns(table: pa.Table, names) -> List[str]:
    """Return the subset of *names* that exist in *table* (order preserved)."""
    have = set(table.schema.names)
    return [n for n in names if n in have]


def transform_drop_annotations(table: pa.Table) -> pa.Table:
    """v7 → v6: drop the optional annotation columns; ``seq`` is untouched.

    ``drop_columns`` keeps the surviving columns' types (and the dictionary
    encoding of sample_id/chrom), so the result is byte-schema-identical to a
    natively generated v6 shard.
    """
    drop = _present_columns(table, _ANNOTATION_COLS)
    return table.drop_columns(drop) if drop else table


def _rebuild_seq_column(table: pa.Table) -> List[str]:
    """Re-inline markers for every row → list of marked ``seq`` strings."""
    seqs = table.column(_SEQ).to_pylist()
    snp_col = table.column(_SNP).to_pylist()
    ind_col = table.column(_INDEL_SPAN).to_pylist()
    has_dir = _DIRECTION in table.schema.names
    dir_col = table.column(_DIRECTION).to_pylist() if has_dir else [None] * len(seqs)

    out: List[str] = []
    for i, clean in enumerate(seqs):
        out.append(
            rehydrate_markers(
                clean,
                snp_col[i] or [],
                decode_indel_spans(ind_col[i]),
                decode_direction(dir_col[i]),
            )
        )
    return out


def transform_rehydrate_drop(table: pa.Table) -> pa.Table:
    """v8 → v6: re-inline markers into ``seq``; drop marker + annotation columns.

    Drops ``snp``/``indel_span``/``direction`` (and any annotation columns), then
    replaces ``seq`` in place so sample_id/chrom keep their dictionary encoding.
    The surviving 6 columns match a native v6 shard.
    """
    new_seqs = _rebuild_seq_column(table)
    drop = _present_columns(table, _MARKER_COLS + _ANNOTATION_COLS)
    out = table.drop_columns(drop)
    seq_idx = out.schema.get_field_index(_SEQ)
    return out.set_column(seq_idx, _SEQ, pa.array(new_seqs, type=_SEQ_TYPE))


def _realign_phylop(values, clean_to_marked: List[int], marked_len: int) -> list:
    """Project a v8 (marker-free) PhyloP row onto v7's marked coordinates.

    v8 ``phylop`` has one slot per clean base; v7 has one slot per marked
    codepoint with the NaN sentinel at every marker. Scatter the clean values to
    their marked base positions; marker slots keep the sentinel.
    """
    if values is None:
        return None
    if len(values) != len(clean_to_marked):
        raise ValueError(
            f"phylop length {len(values)} != clean seq length "
            f"{len(clean_to_marked)} — corrupt v8 record."
        )
    out = [_PHYLOP_NAN_SENTINEL] * marked_len
    for k, v in enumerate(values):
        out[clean_to_marked[k]] = int(v)
    return out


def _realign_cadd(values, clean_to_marked: List[int]) -> list:
    """Remap a sparse v8 CADD row's ``pos`` from clean-seq to marked-seq coords."""
    if values is None:
        return None
    out = []
    for e in values:
        pos = int(e["pos"])
        if not 0 <= pos < len(clean_to_marked):
            raise ValueError(
                f"cadd pos {pos} out of range for clean seq length "
                f"{len(clean_to_marked)} — corrupt v8 record."
            )
        out.append({"pos": int(clean_to_marked[pos]), "score": e["score"]})
    return out


def transform_rehydrate_realign(table: pa.Table) -> pa.Table:
    """v8 → v7: re-inline markers into ``seq`` and realign annotations.

    Rebuilds ``seq``, drops the marker columns, and rewrites ``phylop`` /
    ``cadd_phred`` from clean-seq to marked-seq coordinates so the result matches
    a natively generated v7 shard. Degenerates to a plain rehydrate-and-drop when
    the v8 shard carries no annotation columns.
    """
    seqs = table.column(_SEQ).to_pylist()
    snp_col = table.column(_SNP).to_pylist()
    ind_col = table.column(_INDEL_SPAN).to_pylist()
    has_dir = _DIRECTION in table.schema.names
    dir_col = table.column(_DIRECTION).to_pylist() if has_dir else [None] * len(seqs)

    has_phylop = _PHYLOP in table.schema.names
    has_cadd = _CADD in table.schema.names
    phylop_col = table.column(_PHYLOP).to_pylist() if has_phylop else None
    cadd_col = table.column(_CADD).to_pylist() if has_cadd else None

    new_seqs: List[str] = []
    new_phylop: List[Optional[list]] = []
    new_cadd: List[Optional[list]] = []
    for i, clean in enumerate(seqs):
        snp = snp_col[i] or []
        spans = decode_indel_spans(ind_col[i])
        dirs = decode_direction(dir_col[i])
        new_seqs.append(rehydrate_markers(clean, snp, spans, dirs))
        marked_len, clean_to_marked = emission_layout(len(clean), snp, spans, dirs)
        if has_phylop:
            new_phylop.append(_realign_phylop(phylop_col[i], clean_to_marked, marked_len))
        if has_cadd:
            new_cadd.append(_realign_cadd(cadd_col[i], clean_to_marked))

    out = table.drop_columns(_present_columns(table, _MARKER_COLS))
    seq_idx = out.schema.get_field_index(_SEQ)
    out = out.set_column(seq_idx, _SEQ, pa.array(new_seqs, type=_SEQ_TYPE))
    if has_phylop:
        out = out.set_column(
            out.schema.get_field_index(_PHYLOP), _PHYLOP,
            pa.array(new_phylop, type=_PHYLOP_TYPE),
        )
    if has_cadd:
        out = out.set_column(
            out.schema.get_field_index(_CADD), _CADD,
            pa.array(new_cadd, type=table.schema.field(_CADD).type),  # score uint8 before 2026-10-14, then float32
        )
    return out


def transform_strip_direction(table: pa.Table) -> pa.Table:
    """v6 → v6nd: delete ``‹``/``›`` direction markers from every ``seq`` value.

    The diploid symbol at each position in a v6 ``seq`` is already the unphased
    token; only the direction markers carry phase.  Removing them produces a
    byte-schema-identical v6nd shard.  Null ``seq`` values are preserved as-is.
    """
    seq_idx = table.schema.get_field_index(_SEQ)
    stripped = pc.replace_substring_regex(
        table.column(_SEQ), pattern=f"[{_DIR_FWD}{_DIR_REV}]", replacement=""
    ).cast(_SEQ_TYPE)
    return table.set_column(seq_idx, _SEQ, stripped)


_TRANSFORMS: Dict[Route, Optional[Callable[[pa.Table], pa.Table]]] = {
    Route.IDENTITY: None,
    Route.DROP_ANNOTATIONS: transform_drop_annotations,
    Route.REHYDRATE_DROP: transform_rehydrate_drop,
    Route.REHYDRATE_REALIGN: transform_rehydrate_realign,
    Route.STRIP_DIRECTION: transform_strip_direction,
}


def route_transform(route: Route) -> Optional[Callable[[pa.Table], pa.Table]]:
    """Return the per-shard transform for *route* (``None`` for IDENTITY).

    Raises ``ValueError`` for INCOMPATIBLE (which has no transform) so a caller
    that skipped the feasibility check fails loudly.
    """
    try:
        return _TRANSFORMS[route]
    except KeyError:
        raise ValueError(
            f"Route {route} has no per-shard transform "
            f"(INCOMPATIBLE conversions must be rejected before this point)."
        ) from None


def route_seq_normalizer(route: Route) -> Optional[Callable[[str], str]]:
    """Return the seq-level normaliser to apply during dedup hashing, or ``None``.

    For routes where :attr:`RouteDecision.pre_dedup` is True, the dedup hasher
    must normalise each ``seq`` string with this function before computing the
    hash — so that two source records that differ only in the stripped characters
    land in the same dedup group and are merged before the Step-4 transform runs.
    For all other routes the dedup key is the raw source ``seq`` (or the v8
    compound key), and this returns ``None``.
    """
    if route is Route.STRIP_DIRECTION:
        return strip_direction_from_seq
    return None
