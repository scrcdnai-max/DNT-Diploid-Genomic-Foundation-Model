"""Canonical codepoint->token-id lookup table + the tokenizer/LUT parity guard.

Single source of truth for the char-level LUT that the training data path uses instead of
``tokenizer.encode`` (the LUT is normalizer-immune and ~26x faster than batch encode). This
module exists to stop the LUT from being copy-pasted across training entrypoints — that
duplication is what let the added-token-shadowing bug (encode("ACGT") -> rows 15-19 instead
of trained 6-9) fork between files. Import from here; do not re-copy.

``assert_lut_tokenizer_parity`` is the structural guard: it fails LOUD at train start if a
saved tokenizer's encode() ever disagrees with this LUT again (e.g. a future
``normalized=True`` slip), so the bug class cannot silently ship.
"""
from __future__ import annotations

import logging

import numpy as np
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def merged_vocab(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    """``get_vocab()`` plus the ADDED tokens, which it does not include.

    On the slow NTv3 tokenizer every diploid glyph and marker is an *added*
    token, so ``get_vocab()`` alone silently returns a base-only vocabulary.
    Resolving anything against it yields an empty or partial result with no
    error -- the failure mode that left ``loss/marker_mlm`` unlogged in every
    run to 2026-08-19, and before that left markers in the MLM replacement pool.
    Same reason this module exists at all: import it, do not re-copy the merge.
    """
    added = getattr(tokenizer, "added_tokens_encoder", None) or {}
    return {**tokenizer.get_vocab(), **{str(t): i for t, i in added.items()}}


def build_char_to_id_table(tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
    """Build numpy lookup: code_point -> token_id (~26x faster than tokenizer.encode).

    OOV codepoints (>= table length) must be remapped by the caller to ``unk_id``
    (table[-1] is reserved as the unk sentinel for ``np.minimum``-based clipping).
    """
    unk_id = tokenizer.convert_tokens_to_ids("<unk>")
    unk_id = unk_id if isinstance(unk_id, int) else 0

    full_vocab = merged_vocab(tokenizer)
    char_map = {ord(tok): tid for tok, tid in full_vocab.items() if len(tok) == 1}

    # Reserve a final sentinel slot holding unk_id so callers can clip with
    # np.minimum(cp, len(table)-1) without silently aliasing OOV to a real token.
    max_cp = max(char_map) if char_map else 0
    table = np.full(max_cp + 2, unk_id, dtype=np.int64)
    for cp, tid in char_map.items():
        table[cp] = tid
    return table


def assert_lut_tokenizer_parity(
    tokenizer: PreTrainedTokenizerBase,
    char_to_id: np.ndarray,
    expect_mapped: set[str] | None = None,
) -> None:
    """Fail LOUD if the codepoint LUT (what training uses) disagrees with what the saved
    tokenizer's encode() returns — structural guard against added-token-shadowing
    (encode("A") drifting off the trained row). Source of truth = tokenizer vocab; LUT derived
    and asserted equal. Cheap (single chars only).

    ``expect_mapped`` (the diploid TSV's symbols) is warn-only: any symbol the TSV
    declares but the vocabulary lacks resolves to ``<unk>``, so if it ever appears
    in data it is destroyed silently. That is EXPECTED under
    ``--drop-unseen-tokens`` and is why the warning names the symbols — it is the
    running record of which vocabulary a run actually had.
    """
    full_vocab = merged_vocab(tokenizer)
    if expect_mapped:
        unmapped = sorted(s for s in expect_mapped if len(s) == 1 and s not in full_vocab)
        if unmapped:
            logger.warning(
                "%d diploid TSV symbols are NOT in the vocabulary and will "
                "tokenize to <unk> if they ever occur: %s",
                len(unmapped),
                " ".join(unmapped),
            )
    bad = []
    for tok, tid in full_vocab.items():
        if len(tok) != 1:
            continue
        lut = int(char_to_id[ord(tok)]) if ord(tok) < len(char_to_id) else None
        conv = tokenizer.convert_tokens_to_ids(tok)
        enc = tokenizer.encode(tok, add_special_tokens=False)
        enc0 = enc[0] if len(enc) == 1 else None
        if not (lut == tid == conv == enc0):
            bad.append((tok, hex(ord(tok)), {"vocab": tid, "lut": lut, "conv": conv, "encode": enc}))
    if bad:
        raise AssertionError(
            "Tokenizer/LUT parity FAILED — encode() disagrees with the training codepoint table "
            "(added-token-shadowing class bug; check normalizer / AddedToken(normalized=...)). "
            f"{len(bad)} mismatched single-char tokens, e.g. {bad[:8]}"
        )
    logger.info("Tokenizer/LUT parity OK: encode()==convert==LUT for all single-char tokens")
