"""The one reader of ``diploid_tokens.tsv`` (and of any token table in its format).

Each non-comment line is ``pair<TAB>order<TAB>symbol[<TAB>U+XXXX]``. ``order`` is one of
``symmetric`` / ``unphased`` / ``phased`` / ``single`` / ``meta``; the legacy labels
``ordered`` / ``unordered`` still parse but emit a ``DeprecationWarning``
(:func:`normalize_order_label`). Many pairs share a symbol (``AC`` and ``CA`` both map to
the unphased ``Ą``), and one pair may appear under several orders (``AC`` is phased ``Á``
*and* unphased ``Ą``).

Stdlib only, so tools that must not load the encoder stack (``gather_dataset_stats``) can
use it. Callers adapt :class:`TokenTable` to the shape they need:
``GenomeSequenceProcessor._load_tokens``, ``gather_diploid_stats_bcf.load_diploid_lookups``,
``gather_dataset_stats.load_token_table`` and ``verify_encoding.TokenDecoder`` (which walks
:func:`iter_token_rows` itself).
"""

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Union

_DEPRECATED_ORDER_TERMS = {"ordered": "phased", "unordered": "unphased"}


def normalize_order_label(order: str, *, source: str) -> str:
    """Normalize a diploid-order label and warn on deprecated terms.

    Accepts both the legacy labels ``"ordered"``/``"unordered"`` and the new
    ``"phased"``/``"unphased"``. Legacy labels are still supported but emit a
    ``DeprecationWarning`` identifying ``source`` so callers can locate them.
    Other labels (``"symmetric"``, ``"meta"``, ``"single"``, ...) pass through
    unchanged.
    """
    replacement = _DEPRECATED_ORDER_TERMS.get(order)
    if replacement is not None:
        # Credit the warning to the first caller outside this module (the adapter
        # in the encoder / a tool), as when each reader called this directly: the
        # default filters only show DeprecationWarnings credited to __main__.
        frame, level = sys._getframe(1), 2
        while frame is not None and frame.f_code.co_filename == __file__:
            frame, level = frame.f_back, level + 1
        warnings.warn(
            f"Diploid order label '{order}' is deprecated "
            f"(in {source}); use '{replacement}' instead.",
            DeprecationWarning,
            stacklevel=level,
        )
        return replacement
    return order


class TokenRow(NamedTuple):
    pair: str
    order: str              # normalised (legacy labels already mapped)
    symbol: str
    unicode: Optional[str]  # the 4th column, None when the row has only three


def iter_token_rows(
    path: Union[str, Path],
    *,
    on_malformed: Optional[Callable[[int, str], None]] = None,
    strip: bool = True,
) -> Iterator[TokenRow]:
    """Yield one :class:`TokenRow` per data line, in file order.

    Lines are stripped; blank lines and ``#`` comments are skipped. A line with fewer
    than three tab-separated fields is skipped after calling ``on_malformed(lineno,
    line)`` (if given). Deprecation warnings name ``str(path)``.

    ``strip=False`` keeps ``gather_dataset_stats``' pre-reorg rule until its follow-up
    fix: only the newline is removed, so padding stays in the fields and an indented
    ``#`` line is data.
    """
    source = str(path)
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip() if strip else raw.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                if on_malformed is not None:
                    on_malformed(lineno, line)
                continue
            yield TokenRow(
                parts[0],
                normalize_order_label(parts[1], source=source),
                parts[2],
                parts[3] if len(parts) > 3 else None,
            )


@dataclass
class TokenTable:
    """Every view of a token file the preprocess tools use, built in one pass by
    :func:`load_token_table`; callers treat it as read-only."""

    symbols: List[str]            # every distinct symbol, in order of first appearance (all rows)
    symbol_class: Dict[str, str]  # symbol -> order of its first row
    symbol_unicode: Dict[str, str]  # symbol -> 4th column of its first row, else computed "U+XXXX"
    phased: Dict[str, str]        # pair -> symbol over phased + symmetric rows; a later row wins
    unphased: Dict[str, str]      # pair -> symbol over unphased + symmetric rows; a later row wins
    meta: Dict[str, str]          # meta name ("<SNP>") -> symbol; a later row wins


def load_token_table(
    path: Union[str, Path],
    *,
    on_malformed: Optional[Callable[[int, str], None]] = None,
) -> TokenTable:
    """Parse a token file into a :class:`TokenTable` (empty tables if it has no rows)."""
    table = TokenTable([], {}, {}, {}, {}, {})
    for pair, order, symbol, uni in iter_token_rows(path, on_malformed=on_malformed):
        if symbol not in table.symbol_class:
            table.symbols.append(symbol)
            table.symbol_class[symbol] = order
            table.symbol_unicode[symbol] = (
                uni if uni is not None else " ".join(f"U+{ord(c):04X}" for c in symbol)
            )
        if order in ("phased", "symmetric"):
            table.phased[pair] = symbol
        if order in ("unphased", "symmetric"):
            table.unphased[pair] = symbol
        if order == "meta":
            table.meta[pair] = symbol
    return table
