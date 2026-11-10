"""Windowing helpers shared across pretraining scripts."""

from __future__ import annotations


def stride_from_overlap(window_len: int, overlap_frac: float) -> int:
    """Return stride from window length and overlap fraction.

    Example: ``window_len=2048, overlap_frac=0.2`` -> ``stride=1638``.
    """
    if window_len <= 0:
        raise ValueError(f"window_len must be > 0, got {window_len}")
    if not 0 <= overlap_frac < 1:
        raise ValueError(
            f"overlap_frac must be in [0, 1), got {overlap_frac}"
        )

    overlap_bp = int(round(window_len * overlap_frac))
    overlap_bp = min(window_len - 1, max(0, overlap_bp))
    return max(1, window_len - overlap_bp)


def resolve_stride(window_len: int, stride: int | None, overlap_frac: float) -> int:
    """Resolve explicit stride or derive one from overlap fraction."""
    if stride is not None:
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        return stride
    return stride_from_overlap(window_len, overlap_frac)
