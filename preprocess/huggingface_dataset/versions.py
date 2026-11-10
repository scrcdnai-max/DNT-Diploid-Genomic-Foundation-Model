"""Dataset version presets — the single source of truth for what ``v4``/``v5``/
``v6``/``v6nd``/``v7``/``v8`` and the ``-ldb`` region-set variants
(``v4-ldb``/``v6-ldb``/``v7-ldb``) *mean* behaviorally.

Historically the dataset "version" was an emergent property of a scattered set
of knobs (the unconditional anchor-SNP ``^`` in the encoder, the presence of
``--phylop-bw``/``--cadd-arrow-dir``, a hardcoded ``SCHEMA_VERSION`` label).
This module collapses that into a single selectable input: ``create_dataset.py
generate --version {v4,v5,v6,v6nd,v7,v8,v4-ldb,v6-ldb,v7-ldb}`` resolves to a
:class:`DatasetVersion` preset that drives encoder behaviour, whether annotation
columns are permitted, and the ``schema_version`` stamped into provenance.

What a preset controls (the axes that distinguish the shipped lineage):
  * ``emit_anchor_snp_marker`` — whether compound-het INDEL anchors get a ``^``
    prefix. This is the *only* byte difference between v5 and v6 output (the
    pre-v6 form left those anchors bare). See ``GenomeSequenceProcessor`` and
    ``_build_haplotype_arrays`` in ``get_genome_sequence.py``.
  * ``use_direction_markers`` — the three-level ``‹``/``›`` direction-marker
    encoding. Every shipped diploid version EXCEPT ``v6nd`` uses it (verified
    against the real datasets), so it is preset-owned, not a separate flag:
    requiring ``--use-direction-markers`` alongside ``--version`` would let
    ``--version v5`` silently produce a non-real-v5 dataset. ``v6nd`` is the lone
    preset that turns it off (v6 without the direction tokens). Forced OFF for
    ``--representation haploid`` (haploid output cannot carry direction markers).
  * ``force_unphased`` — drop haplotype phase, routing every het through the
    unordered diploid lookup. Normally an orthogonal CLI ablation knob, but
    ``v6nd`` bundles it ON so its het symbols stay v6's unordered symbols once the
    direction tokens are removed (the encoder would otherwise switch to the
    *ordered* phased lookup). The CLI ``--force-unphased`` flag is OR-ed with this
    preset field, so either source enables it.
  * ``allow_annotations`` — whether per-base annotation columns (phylop,
    cadd_phred) may be attached. v7 and v8.
  * ``externalize_markers`` — whether the structural markers (``^`` SNP, ``{``/
    ``}`` INDEL span, ``‹``/``›`` direction) are emitted INLINE in ``seq`` (v5/
    v6/v7) or moved out into separate sparse columns (``snp``/``indel_span``/
    ``direction``) leaving ``seq`` as pure base content (v8). See
    ``GenomeSequenceProcessor(externalize_markers=...)``.
  * ``schema_version`` — the label written to provenance.json /
    generation_provenance.json.
  * ``marker_policy`` — how annotation arrays treat marker codepoints. Identical
    (``NAN``) across all presets today, but preset-owned so a future version can
    diverge without another scattered constant. Moot under ``externalize_markers``
    (``seq`` has no marker codepoints, so SKIP and NAN coincide).
  * ``region_set`` / ``region_bed_sha256`` / ``region_bed_hint`` — WHICH REGIONS
    the dataset covers, i.e. the BED that ``generate`` was run over. Unlike the
    fields above this says nothing about the encoder; it is preset-owned because
    the region universe determines *record identity* — two datasets over
    different BEDs hold different rows and are not inter-convertible, so
    ``compat.py`` must refuse to route between them (a v7→v6 "drop the annotation
    columns" conversion is a lie if the two also disagree on which regions
    exist). Presets that leave ``region_set`` at :data:`UNBOUND_REGION_SET`
    accept any ``--bed``; a preset that names a region set pins the BED's sha256,
    and ``create_dataset.py generate`` rejects a mismatching ``--bed``.

There is deliberately NO default version: ``create_dataset.py generate`` requires
``--version`` to be stated explicitly, and readers adopt the version recorded in
the dataset's provenance sidecar (validated against :data:`SUPPORTED_VERSIONS`).
This removes the previously-hardcoded ``DEFAULT_VERSION``/``SCHEMA_VERSION``
globals so a version can never be silently assumed.

Orthogonal axes deliberately NOT owned by the preset: ``--representation``
(diploid/haploid) is an independent output knob. ``--force-unphased`` is likewise
an independent ablation flag, with one exception — ``v6nd`` bundles it (the only
preset that does), because dropping phase is intrinsic to what v6nd *is*; the CLI
flag and the preset field are OR-ed.

The region set used to belong on that "orthogonal" list too — the BED was purely
a CLI argument, and a run of ``--version v8 --bed ld_blocks_eur_grch38.bed``
produced a dataset whose ``schema_version`` was indistinguishable from a
canonical-BED v8. It was promoted to a preset field (2026-07-26) precisely
because that ambiguity is unresolvable after the fact. It is NOT analogous to
``--representation``/``--force-unphased``: those change how each record is
encoded and are recoverable from the data, whereas the region set changes which
records exist at all. Only region-BOUND presets constrain ``--bed``; the stock
presets stay unbound so "stock version, custom BED" (README_allgenes.md) keeps
working.

A reserved future variant (NOT yet shipped) would instead keep phase by replacing
the direction tokens with the *ordered* phased diploid symbols
(``use_direction_markers=False``, ``force_unphased=False``). It needs no new
preset field — only its own entry and a distinct name (e.g. ``v6o``/``v6p``; not a
``*nd`` name, since it too carries no direction markers).

This is a leaf module: it imports only :class:`MarkerPolicy` from
``annotations`` (itself a leaf), so ``dataset`` and ``create_dataset`` can both
import from here without an import cycle.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from preprocess.huggingface_dataset.annotations import MarkerPolicy  # noqa: E402

# Sentinel region set: the preset places no constraint on ``--bed``. Every
# preset shipped before 2026-07-26 is unbound, which is what keeps
# "stock version + custom BED" runs (README_allgenes.md) working.
UNBOUND_REGION_SET = "unbound"

# The deCODE EUR LD-block region set — 1,361 gap-free blocks over chr1-22
# (2.806 Gbp), normalized from ld_blocks/raw/deCODE_EUR_LD_blocks.bed by
# preprocess/regions/build_ld_block_bed.py. See preprocess/regions/ld_blocks/PROVENANCE.md.
LDB_REGION_SET = "ldb_decode_eur"
LDB_BED_HINT = "preprocess/regions/ld_blocks_eur_grch38.bed"
LDB_BED_SHA256 = "d9f271cae20cbc0251e3ccf1912daaf8e7280b58fa571c1a57fb4d4c31cfb0ea"

# The nested-genome-division region set, built by
# preprocess/regions/nested_genome_division/build_division.py.
#
# tss_tes: 59,572 per-gene intervals over chr1-22 + chrX from GENCODE v50,
# covering 1.948 Gbp of distinct sequence. One interval per gene: transcripts of
# 11 biotypes (protein_coding plus the ten functional non-coding classes) are
# merged per gene into a TSS->TES envelope, extended 1 kb upstream as a promoter
# margin and then downstream to a multiple of 128 bp. So unlike every other
# region set in this file its intervals OVERLAP, wherever the genes do, and its
# rows sum to 2.313 Gbp over 1.948 Gbp of distinct bases. Column 4 is the bare
# gene symbol (usable with --genes-file); columns 5-7 are score/strand/biotype.
#
# Every interval length is a multiple of 128 bp, and that is the *only* size
# constraint — there is no floor and no ceiling, so rows run from 1,152 bp to
# 2.47 Mbp. A consumer relying on a minimum size must check the region set.
#
# The "_tx128" suffix marks the 2026-09-01 rebuild onto transcript records and
# the 128-bp grid. Its two predecessors were "tss_tes_gencode50_pergene"
# (67,123 gene-record intervals, 256-bp grid, 10-150 kbp) and before that
# "tss_tes_gencode50" (29,296 merged-locus intervals); renaming rather than
# re-pinning is what makes compat.py refuse to treat them as interchangeable.
TSSTES_REGION_SET = "tss_tes_gencode50_tx128"
TSSTES_BED_HINT = "preprocess/regions/nested_genome_division/hg38_tss_tes.bed"
TSSTES_BED_SHA256 = "360014cc451951c0a118f4af7bbc0a9c0c3143bb2064983324c33a5bd954a70f"

# The companion tad_tss_tes region set (40,543 nested TAD->gene intervals from
# insulation-score TADs) was retired on 2026-09-01 together with the TAD
# division itself; no dataset was ever built on it. See
# frozen at tag `pre-reorg` (was nested_genome_division/legacy/).

@dataclass(frozen=True)
class DatasetVersion:
    """Behavioral preset for one dataset schema version.

    Frozen so a resolved preset cannot be mutated mid-run.
    """

    schema_version: str
    emit_anchor_snp_marker: bool
    use_direction_markers: bool
    allow_annotations: bool
    marker_policy: MarkerPolicy
    externalize_markers: bool = False
    force_unphased: bool = False
    # Region set — see the "What a preset controls" docstring bullet. These three
    # move together: a preset either names a region set AND pins its BED hash and
    # path hint, or leaves all three at their unbound defaults.
    region_set: str = UNBOUND_REGION_SET
    region_bed_sha256: Optional[str] = None
    region_bed_hint: Optional[str] = None


# The shipped lineage. v4 uses ordered (phased) diploid symbols with inline
# ^/{} markers but NO ‹/› direction tokens — the original pre-direction-marker
# encoding. v5/v6/v7 all use the three-level direction-marker encoding
# (use_direction_markers) with markers INLINE in ``seq``. v5 differs from v6
# only by the absence of the anchor-SNP ``^``; v6 differs from v7 only by
# forbidding the per-base annotation columns. v8 keeps v7's content but moves
# the structural markers OUT of ``seq`` into sparse columns (externalize_markers).
# v6nd is a side variant of v6 that removes the ‹/› direction tokens and drops
# phase (force_unphased) — its seq equals v6's seq minus the ‹/› codepoints.
# All entries in this literal are region-UNBOUND (any --bed accepted); the
# region-bound -ldb variants are derived from them further below.
# Every preset, including the derived ones, is documented in
# schemas/<name>_schema.{md,json}; test_schema_docs.py fails if one is missing
# or its "preset" block disagrees with the entry here.
PRESETS: dict[str, DatasetVersion] = {
    # v4: original encoding — ordered (phased) diploid symbols, ^/{} boundary
    # markers baked in, no direction tokens. Predates the ‹/› three-level scheme
    # introduced in v5. force_unphased=False keeps het positions on the ordered
    # phased lookup table (same encoder path as before direction markers were added).
    "v4": DatasetVersion(
        schema_version="v4",
        emit_anchor_snp_marker=False,
        use_direction_markers=False,
        allow_annotations=False,
        marker_policy=MarkerPolicy.NAN,
        externalize_markers=False,
        force_unphased=False,
    ),
    "v5": DatasetVersion(
        schema_version="v5",
        emit_anchor_snp_marker=False,
        use_direction_markers=True,
        allow_annotations=False,
        marker_policy=MarkerPolicy.NAN,
        externalize_markers=False,
    ),
    "v6": DatasetVersion(
        schema_version="v6",
        emit_anchor_snp_marker=True,
        use_direction_markers=True,
        allow_annotations=False,
        marker_policy=MarkerPolicy.NAN,
        externalize_markers=False,
    ),
    "v7": DatasetVersion(
        schema_version="v7",
        emit_anchor_snp_marker=True,
        use_direction_markers=True,
        allow_annotations=True,
        marker_policy=MarkerPolicy.NAN,
        externalize_markers=False,
    ),
    # v6 with the ‹/› direction tokens removed and phase DROPPED: same compact
    # unordered vocabulary and same ^/{} structural markers as v6, but het
    # positions carry no order. Equivalent to ``--version v6 --force-unphased``
    # except it stamps its own schema_version so provenance/readers can tell it
    # apart from real v6. ``force_unphased`` makes the encoder pick the unordered
    # lookup (see _sequence_str_from_variants in get_genome_sequence.py), so the
    # seq is byte-for-byte v6's seq with the ‹/› codepoints deleted. v6nd is the
    # only shipped preset that turns use_direction_markers off (diploid-only).
    "v6nd": DatasetVersion(
        schema_version="v6nd",
        emit_anchor_snp_marker=True,
        use_direction_markers=False,
        allow_annotations=False,
        marker_policy=MarkerPolicy.NAN,
        externalize_markers=False,
        force_unphased=True,
    ),
    "v8": DatasetVersion(
        schema_version="v8",
        emit_anchor_snp_marker=True,
        use_direction_markers=True,   # still computed — now stored in the column
        allow_annotations=True,
        marker_policy=MarkerPolicy.NAN,  # moot: seq has no marker codepoints
        externalize_markers=True,
    ),
}

# ── LD-block region-set variants ─────────────────────────────────────────────
# v4-ldb / v6-ldb / v7-ldb are the v4 / v6 / v7 ENCODINGS applied to a different
# REGION UNIVERSE: the deCODE EUR LD blocks (1,361 blocks, 2.806 Gbp, gap-free)
# instead of the canonical hg38_pretrain_canonical_expanded.bed (140,099 regions,
# 0.943 Gbp). Nothing about the encoder changes — the seq of a region present in
# both datasets is byte-identical between v7 and v7-ldb — so these are derived
# with ``dataclasses.replace`` rather than copy-pasted literals: "same encoding,
# different regions" then holds structurally and cannot drift when a base preset
# is edited.
#
# They exist because the region set is otherwise invisible after generation: a
# dataset built as ``--version v7 --bed ld_blocks_eur_grch38.bed`` records
# schema_version "v7" and is indistinguishable from the canonical-BED v7. The
# suffixed name makes the region universe part of the dataset's identity, and
# ``region_bed_sha256`` makes it enforceable at the CLI.
#
# Deliberately NOT mirrored: v5 (superseded by v6), v6nd and v8 (no LD-block use
# case yet). Add them only when a run actually needs them — each new region set ×
# encoding pair is another preset, which is the known cost of suffixed naming.
_LDB_BINDING = dict(
    region_set=LDB_REGION_SET,
    region_bed_sha256=LDB_BED_SHA256,
    region_bed_hint=LDB_BED_HINT,
)

PRESETS["v4-ldb"] = replace(PRESETS["v4"], schema_version="v4-ldb", **_LDB_BINDING)
PRESETS["v6-ldb"] = replace(PRESETS["v6"], schema_version="v6-ldb", **_LDB_BINDING)
PRESETS["v7-ldb"] = replace(PRESETS["v7"], schema_version="v7-ldb", **_LDB_BINDING)

# ── Nested-genome-division region-set variants ───────────────────────────────
# Same "same encoding, different regions" construction as the -ldb block above,
# for the two divisions built by preprocess/regions/nested_genome_division/. They are
# mirrored onto v7 and v8 only — the encodings actually in use — following the
# -ldb rule that each region set x encoding pair is a preset with a real cost,
# added when a run needs it rather than pre-emptively.
#
# What makes this region set different from every other one in this file: every
# record's span is a multiple of 128 bp (with NO minimum or maximum: 1,152 bp to
# 2.47 Mbp), and its records overlap, which the canonical and LD-block BEDs do
# not. Any consumer that relies on that (fixed-stride tokenisation) must check the
# region set, not just the encoding — which is exactly what pinning it here makes
# possible. (Only one division remains: the TAD one was retired 2026-09-01.)
_TSSTES_BINDING = dict(
    region_set=TSSTES_REGION_SET,
    region_bed_sha256=TSSTES_BED_SHA256,
    region_bed_hint=TSSTES_BED_HINT,
)
PRESETS["v7-tsstes"] = replace(
    PRESETS["v7"], schema_version="v7-tsstes", **_TSSTES_BINDING
)
PRESETS["v8-tsstes"] = replace(
    PRESETS["v8"], schema_version="v8-tsstes", **_TSSTES_BINDING
)

# The set of known versions. There is deliberately no default: ``--version`` is
# required at the CLI and readers adopt the dataset's recorded version. Callers
# validate an externally-supplied version against this set. Derived AFTER the
# -ldb entries are added, so it covers them.
SUPPORTED_VERSIONS: frozenset[str] = frozenset(PRESETS)

# v10 — the per-patient phased haplotype-pair format written by
# ``create_haplotype_dataset.py`` (two haploid ``ACGTN^{}`` strings per row,
# ``sample_id`` preserved, no dedup/shuffle/assemble, HG38_REF weighted by the
# number of variant-free samples). It is deliberately a standalone constant and
# NOT a ``DatasetVersion`` in ``PRESETS``: adding it there would make
# ``create_dataset.py generate --version v10`` legal (``SUPPORTED_VERSIONS`` is
# derived from ``PRESETS``) and emit a diploid shard stamped "v10". The diploid
# /haploid readers fail closed on its representation via ``_verify_representation``.
HAPLOTYPE_PAIR_SCHEMA_VERSION = "v10"
HAPLOTYPE_PAIR_REPRESENTATION = "haplotype_pair"
assert HAPLOTYPE_PAIR_SCHEMA_VERSION not in PRESETS


def resolve(name: str) -> DatasetVersion:
    """Return the :class:`DatasetVersion` preset for *name* (e.g. ``"v7"``).

    Raises ``ValueError`` with the list of known versions on an unknown name,
    so a typo at the CLI fails loudly rather than silently picking a default.
    """
    try:
        return PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"Unknown dataset version {name!r}. Known versions: {known}."
        ) from None
