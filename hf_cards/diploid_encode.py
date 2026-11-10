"""Reference-aligned diploid encoder for the DNT (Diploid Nucleotide Transformer) models.

Turns two aligned haplotype strings into the single token stream these models were
pretrained on (the "v6 directional" scheme): unphased allele-pair symbols carrying
zygosity, plus explicit `<DIR_FWD>`/`<DIR_REV>` markers carrying haplotype orientation.

Requires `diploid_tokens.tsv` (shipped in this repo). Verified byte-identical to the
production preprocessing pipeline on both substitutions and indels.

    from diploid_encode import encode_diploid
    encode_diploid("ACGT", "AGGT", tokens_tsv="diploid_tokens.tsv")
    # -> 'A^‹ȺGT'   (het A/G at position 1, hap0 carries the lower-priority allele)
"""

from __future__ import annotations

import csv
from typing import Dict, Optional, Sequence, Tuple

__all__ = ["load_token_table", "encode_diploid", "symbol_for", "ALLELE_PRIORITY"]

# Canonical allele ordering used to orient a phased heterozygous pair.
# `<DIR_FWD>` means hap0 holds the lower-priority allele; `<DIR_REV>` the higher.
ALLELE_PRIORITY: Dict[str, int] = {c: i for i, c in enumerate("ACGTN-*")}


def load_token_table(tokens_tsv: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Read diploid_tokens.tsv.

    Returns (pair_to_symbol, meta) where pair_to_symbol covers the `symmetric`
    (homozygous) and `unphased` (heterozygous) rows -- the two classes these
    checkpoints were trained on. The `phased` rows define an alternative
    ordered-symbol vocabulary that this corpus did not exercise, and the `single`
    rows a proposed hemizygous vocabulary; both are skipped here on purpose.
    """
    pair_to_symbol: Dict[str, str] = {}
    meta: Dict[str, str] = {}
    with open(tokens_tsv, encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row or row[0].startswith("#"):
                continue
            pair, order, symbol = row[0], row[1], row[2]
            if order in ("symmetric", "unphased"):
                pair_to_symbol[pair] = symbol
            elif order == "meta":
                meta[pair] = symbol
    return pair_to_symbol, meta


def symbol_for(a: str, b: str, tokens_tsv: str) -> str:
    """Return the single diploid symbol for one allele pair, e.g. ('C','T') -> 'Ȼ'.

    Useful for scoring a specific genotype with the MLM head:
    `tok.convert_tokens_to_ids(symbol_for('C', 'T', tokens_tsv))`.
    Order-insensitive: the symbol carries zygosity, not orientation.
    """
    pair_to_symbol, _ = load_token_table(tokens_tsv)
    try:
        return pair_to_symbol[a + b]
    except KeyError:
        raise KeyError(f"no diploid symbol for allele pair {a!r}/{b!r}") from None


def encode_diploid(
    hap0: str,
    hap1: str,
    *,
    tokens_tsv: str,
    snp_mask: Optional[Sequence[bool]] = None,
    indel_mask: Optional[Sequence[bool]] = None,
    phased: bool = True,
    skip_gaps: bool = True,
    return_offsets: bool = False,
):
    """Encode two reference-aligned haplotypes into one diploid token stream.

    Args:
        hap0, hap1: equal-length haplotype strings over A/C/G/T/N/-/*. `-` is a
            gap (a base deleted on that haplotype, or a position that exists only
            because the other haplotype has an insertion there).
        tokens_tsv: path to `diploid_tokens.tsv`.
        snp_mask: per-position flags for substitution sites, which get a `<SNP>`
            (`^`) marker. Must be one entry per position. Defaults to every
            heterozygous position -- correct when the only variants present are
            heterozygous SNVs. It **replaces** that default rather than adding to
            it, so a supplied mask must be True at every site you want marked,
            heterozygous ones included. Homozygous-ALT substitutions are only
            markable this way; they are invisible in the haplotype pair itself.
        indel_mask: per-position flags for indel content, i.e. deleted reference
            bases and inserted bases. Each contiguous run is wrapped in
            `<INDEL_span_start>`/`<INDEL_span_end>` (`{`/`}`). The indel's anchor
            base sits outside the braces and is not flagged.
        phased: emit direction markers. Set False to encode zygosity only, which
            is what you want when the input genotypes are unphased -- these
            models never infer phase, they only read what the input encodes.
        skip_gaps: drop gap/gap positions, matching how the pretraining corpus
            was built. Set False to keep them as explicit `-` tokens, which is the
            only way to make a homozygous deletion visible to the model.
        return_offsets: also return a list mapping each input position `i` to the
            index of its allele-pair symbol in the output string (-1 if the
            position was dropped). Markers shift everything downstream, so use
            this rather than assuming output index == input position.

    Returns:
        The encoded string, or `(string, offsets)` when `return_offsets=True`.
        Every character is exactly one token, so output index == token index.
    """
    if len(hap0) != len(hap1):
        raise ValueError(
            f"haplotypes must be reference-aligned to equal length, got {len(hap0)} and {len(hap1)}"
        )
    for mask_name, mask in (("snp_mask", snp_mask), ("indel_mask", indel_mask)):
        if mask is not None and len(mask) != len(hap0):
            raise ValueError(
                f"{mask_name} must have one entry per haplotype position: "
                f"got {len(mask)}, expected {len(hap0)}"
            )
    pair_to_symbol, meta = load_token_table(tokens_tsv)
    snp = meta["<SNP>"]
    open_b, close_b = meta["<INDEL_span_start>"], meta["<INDEL_span_end>"]
    fwd, rev = meta["<DIR_FWD>"], meta["<DIR_REV>"]

    out: list[str] = []
    offsets: list[int] = []
    in_indel = False
    for i, (a, b) in enumerate(zip(hap0, hap1)):
        symbol = pair_to_symbol.get(a + b)
        if symbol is None:
            extra = ""
            if "N" in (a, b) and a != b:
                extra = (
                    " -- the token table has no heterozygous N state (only 'NN'); "
                    "resolve or drop ambiguous positions before encoding"
                )
            raise KeyError(
                f"no diploid symbol for allele pair {a!r}/{b!r} at position {i}{extra}"
            )

        is_indel = bool(indel_mask[i]) if indel_mask is not None else False
        if skip_gaps and symbol == "-":
            # A gap on both haplotypes carries no information; it can still close
            # an open indel span.
            if in_indel and not is_indel:
                out.append(close_b)
                in_indel = False
            offsets.append(-1)
            continue

        if in_indel and not is_indel:
            out.append(close_b)
            in_indel = False
        elif is_indel and not in_indel:
            out.append(open_b)
            in_indel = True

        het = a != b
        if (bool(snp_mask[i]) if snp_mask is not None else het) and not is_indel:
            out.append(snp)
        if het and phased:
            out.append(fwd if ALLELE_PRIORITY[a] <= ALLELE_PRIORITY[b] else rev)
        offsets.append(len(out))
        out.append(symbol)

    if in_indel:
        out.append(close_b)
    seq = "".join(out)
    return (seq, offsets) if return_offsets else seq


def to_ordered(seq: str, tokens_tsv: str, offsets: Optional[Sequence[int]] = None):
    """Rewrite a directional-marker stream into the ordered-symbol scheme.

    Only for checkpoints trained on ordered (``phased``) symbols, e.g.
    ``scrc-dnai/Phased_Nucleotide_Tokenization_8M``. Each ``‹X`` / ``›X`` pair
    (direction marker + unphased symbol) becomes the single ordered symbol for
    that allele order, e.g. ``^‹Ⱦ`` -> ``^Ă`` (A on hap0, T on hap1). The ``^``
    and ``{`` ``}`` markers are unchanged. Pass the ``offsets`` from ``encode_diploid``
    to get them re-indexed for the shorter string: returns ``(seq, offsets)``.
    """
    phased: Dict[str, str] = {}
    unphased: Dict[str, str] = {}
    meta: Dict[str, str] = {}
    with open(tokens_tsv, encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row or row[0].startswith("#"):
                continue
            pair, order, symbol = row[0], row[1], row[2]
            {"phased": phased, "unphased": unphased, "meta": meta}.get(order, {})[pair] = symbol
    # unphased symbol -> canonical (low-priority-first) pair
    canon: Dict[str, str] = {}
    for pair, sym in unphased.items():
        a, b = pair
        if ALLELE_PRIORITY[a] <= ALLELE_PRIORITY[b]:
            canon.setdefault(sym, pair)
    fwd, rev = meta["<DIR_FWD>"], meta["<DIR_REV>"]
    out: list[str] = []
    new_index: Dict[int, int] = {}  # old index of a kept character -> new index
    i = 0
    while i < len(seq):
        c = seq[i]
        if c in (fwd, rev) and i + 1 < len(seq) and seq[i + 1] in canon:
            a, b = canon[seq[i + 1]]
            new_index[i + 1] = len(out)
            out.append(phased[a + b] if c == fwd else phased[b + a])
            i += 2
        else:
            new_index[i] = len(out)
            out.append(c)
            i += 1
    if offsets is None:
        return "".join(out)
    return "".join(out), [new_index.get(o, -1) if o >= 0 else -1 for o in offsets]
