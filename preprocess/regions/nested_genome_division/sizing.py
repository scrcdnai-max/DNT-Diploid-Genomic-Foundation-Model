#!/usr/bin/env python3
"""The length algebra for the TSS-TES genome division.

One constraint, and it is structural rather than a post-hoc fixup:

    length % 128 == 0

There is **no minimum and no maximum**. An interval is exactly the gene it came
from, plus a fixed upstream promoter margin, plus however few bases it takes to
reach the next multiple of ``GRID`` — so a 94 bp vault_RNA yields a 1,152 bp row
and the longest lncRNA yields a 2,474,624 bp one, and both are correct.

Two primitives, both pure and I/O-free:

``pad_downstream``  apply the upstream margin and round the far end up to the
                    grid, respecting strand.
``place``           re-seat a window of a fixed length between two hard walls,
                    spilling whatever one wall blocks onto the other side.

The previous version of this module also owned a ``10240 <= length <= 149760``
contract, with ``plan_units`` / ``harmonize_tiling`` / ``coalesce_short`` to
enforce it. That contract existed to let the TAD division tile exactly; with the
TAD level gone it only served to inflate every small RNA to a 10 kbp window. See
the copy frozen at tag `pre-reorg` (was nested_genome_division/legacy/) for it.
"""

from __future__ import annotations

GRID = 128
"""Base-pair granularity: every emitted length is a whole multiple of this."""

DEFAULT_UPSTREAM = 1_000
"""Thumb-rule promoter margin added on the 5' side of every gene, in bp."""


def snap_down(pos: int) -> int:
    """Largest multiple of ``GRID`` at or below *pos*."""
    return (pos // GRID) * GRID


def snap_up(pos: int) -> int:
    """Smallest multiple of ``GRID`` at or above *pos*."""
    return -(-pos // GRID) * GRID


def snap_nearest(pos: int) -> int:
    """Multiple of ``GRID`` closest to *pos* (ties round up)."""
    return ((pos + GRID // 2) // GRID) * GRID


def pad_downstream(
    start: int,
    end: int,
    strand: str,
    *,
    upstream: int = DEFAULT_UPSTREAM,
) -> tuple[int, int]:
    """Grow ``[start, end)`` to a grid-multiple length, 5' margin first.

    Two steps, in this order, both **strand-aware**:

    1. extend the 5' end by *upstream* bases — ``start -= upstream`` on ``+``,
       ``end += upstream`` on ``-``;
    2. extend the 3' end by ``(-length) % GRID`` bases, which is whatever it
       takes to reach the next multiple of ``GRID``.

    Order matters: the promoter margin is a fixed biological quantity and the
    grid remainder is slack, so the slack goes downstream where it costs
    nothing. The result is at most ``upstream + GRID - 1`` bases longer than the
    input and its length is always ``% GRID == 0``.

    Coordinates are not clamped to the chromosome here — :func:`place` and the
    caller's free-region bounds do that.
    """
    if end <= start:
        raise ValueError(f"span must be positive, got [{start}, {end})")
    if upstream < 0:
        raise ValueError(f"upstream margin must be non-negative, got {upstream}")
    if strand not in ("+", "-"):
        raise ValueError(f"strand must be '+' or '-', got {strand!r}")

    if strand == "+":
        start -= upstream
    else:
        end += upstream

    extra = -(end - start) % GRID
    if strand == "+":
        end += extra
    else:
        start -= extra

    return start, end


def place(
    start: int,
    end: int,
    total_bp: int,
    *,
    lo_bound: int,
    hi_bound: int,
) -> tuple[int, int] | None:
    """Seat a window of *total_bp* bases over ``[start, end)`` within the bounds.

    ``lo_bound``/``hi_bound`` are hard walls — the ends of the free region
    between two unreliably-sequenced stretches, clipped to the chromosome. The
    window keeps the position :func:`pad_downstream` gave it unless a wall is in
    the way; then it slides just far enough to clear the wall, so the bases the
    wall took back are added on the opposite side and the window keeps its
    **exact** requested length. That is what preserves divisibility through the
    compensation: the length never changes, only the offset.

    Returns ``None`` when the free region cannot hold *total_bp* at all, which
    is the caller's signal to drop the gene.
    """
    if total_bp <= 0:
        raise ValueError(f"total_bp must be positive, got {total_bp}")
    if hi_bound - lo_bound < total_bp:
        return None

    # Centre only matters when the caller hands us a window that is not already
    # the right length; for the normal path start/end already span total_bp and
    # `new_start` is simply `start`.
    pad = total_bp - (end - start)
    new_start = start - pad // 2

    if new_start < lo_bound:
        new_start = lo_bound
    elif new_start + total_bp > hi_bound:
        new_start = hi_bound - total_bp

    return new_start, new_start + total_bp
