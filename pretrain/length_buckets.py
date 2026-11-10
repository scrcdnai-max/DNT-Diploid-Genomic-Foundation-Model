"""Length-bucketed batching for the variable-length diploid corpus.

v7 was fixed tiles, so padding every window to ``--window-len`` cost nothing.
v7-tsstes is transcript-anchored gene spans (1 kb - 2.47 Mb, median 33 kb): at
``--window-len 131072`` 84.5% of rows are shorter than one window and carry only
41.75% of the bases, so ~57% of the compute would be spent on padding.

The fix is the one already proven on this corpus by ``semantic_dedup.embed_batched``
(measured 0.0000% padding waste): quantize each window's real length to a coarse
bucket, batch only same-bucket windows together, and size the batch by a BASE
budget rather than a row count.

Two deliberate departures from ``embed_batched``:

- the bucket step is ``BUCKET_STEP`` (8192), not the model stride (128). 128 would
  give ~3,062 distinct shapes on this corpus, which is a recompile storm under
  ``--compile-backbone``; 8192 gives 16 shapes up to 131072. 8192 is a multiple of
  both ``PAD_MULTIPLE`` (128) and every NTv3 OG2 replay bin, which is what lets the
  token budget stay exactly divisible for ``og2_batch_sizes``.
- ``embed_batched`` must batch strictly-equal padded lengths because the NTv3 tower
  takes no attention mask. Training runs with ``--pad-mask``, so the quantization to
  8192 is safe: padded positions are masked out rather than attended as sequence.

Base-weighted efficiency of the 8192 ladder on v7-tsstes is ~93% (waste per row is
~step/2 / L, and 96.7% of bases live in rows >= 16 kb), against 43% for fixed 128K
padding. Power-of-two buckets were considered and rejected: 8 shapes instead of 16,
but a flat ~25% waste (~78% efficiency).

At ``--window-len <= BUCKET_STEP`` every window buckets to the full window length,
so this module is a no-op and the fixed-length path is reproduced exactly, which is
what makes a v7 run with and without bucketing comparable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import numpy as np

# Bucket granularity in bases -- see the module docstring for why 8192.
BUCKET_STEP = 8192


def bucket_of(n: int, total_len: int, step: int = BUCKET_STEP) -> int:
    """Padded length for a window holding ``n`` real bases.

    ``total_len`` is the fixed-length path's padded window length (``window_len``
    rounded up to ``PAD_MULTIPLE``) and is the cap: a full-length window buckets to
    exactly what the fixed-length path would have produced, so the two paths agree
    on every full window and differ only by not over-padding the short ones.
    """
    n = max(1, min(int(n), int(total_len)))
    return min(int(total_len), -(-n // int(step)) * int(step))


def bucket_lengths(
    win_len: np.ndarray, total_len: int, step: int = BUCKET_STEP
) -> np.ndarray:
    """Vectorized ``bucket_of`` over a per-window real-length array."""
    n = np.clip(win_len.astype(np.int64), 1, int(total_len))
    return np.minimum(
        int(total_len), -(-n // int(step)) * int(step)
    ).astype(np.int32)


class LengthBucketBatchSampler:
    """Group a sampler's index stream into single-bucket, base-budgeted batches.

    Wraps whatever sampler the Trainer would otherwise have used (in this repo that
    is ``_BinaryWeightSampler``, the REF over-sampler), so bucketing composes with
    sampling instead of replacing it -- the draw distribution is untouched, only the
    order in which drawn windows are packed into batches changes.

    Every emitted batch holds ``budget // bucket_len`` windows of ONE bucket, so
    bases-per-batch is constant to within one row regardless of which bucket it came
    from. That constancy is the point: it is what makes a fixed
    ``bases-per-optimizer-step`` target (and therefore a pinned ``--genome-steps``)
    meaningful once rows-per-batch varies by 16x across buckets.

    ``drop_last`` defaults True to match ``dataloader_drop_last``; the leftover
    partial queues at the end of an epoch are dropped rather than emitted, which
    keeps every batch at the full budget.
    """

    def __init__(
        self,
        sampler,
        bucket_len: np.ndarray,
        token_budget: int,
        drop_last: bool = True,
    ) -> None:
        budget = int(token_budget)
        if budget <= 0:
            raise ValueError(f"token_budget must be > 0, got {token_budget}")
        smallest = int(np.min(bucket_len)) if len(bucket_len) else BUCKET_STEP
        if budget < smallest:
            raise ValueError(
                f"token_budget {budget} is smaller than the smallest bucket "
                f"({smallest} bases); every batch would hold a single window"
            )
        self.sampler = sampler
        self.bucket_len = np.asarray(bucket_len)
        self.token_budget = budget
        self.drop_last = bool(drop_last)
        # Rows per batch and window count, per bucket. Precomputed so __iter__
        # does no division and __len__ / mean_bases_per_batch do no array scan --
        # on the real corpus bucket_len is 80M int32, so counting per bucket on
        # demand would rescan ~5 GB per call.
        buckets, counts = np.unique(self.bucket_len, return_counts=True)
        self._cap = {int(b): max(1, budget // int(b)) for b in buckets}
        self._count = {int(b): int(c) for b, c in zip(buckets, counts)}

    def _batches(self, bucket: int) -> int:
        n, cap = self._count[bucket], self._cap[bucket]
        return n // cap if self.drop_last else -(-n // cap)

    def __iter__(self) -> Iterator[list[int]]:
        pending: dict[int, list[int]] = defaultdict(list)
        cap = self._cap
        blen = self.bucket_len
        for i in self.sampler:
            i = int(i)
            b = int(blen[i])
            q = pending[b]
            q.append(i)
            if len(q) >= cap[b]:
                yield q
                pending[b] = []
        if not self.drop_last:
            for q in pending.values():
                if q:
                    yield q

    def mean_bases_per_batch(self) -> float:
        """Realized bases per batch, averaged over the epoch's batches.

        NOT equal to ``token_budget``: a bucket that does not divide the budget
        leaves ``budget % bucket`` bases unused (bucket 106496 against a 524288
        budget fits 4 rows = 425984, 81% of budget). The shortfall is bounded by
        one bucket per batch and averages out over the ~16 micro-batches in an
        optimizer step, but it is systematic, so ``--genome-steps`` -- which pins
        genome-N to an absolute BASE budget -- must be derived from this number
        rather than from the nominal one.
        """
        num = sum(self._batches(b) * cap * b for b, cap in self._cap.items())
        den = sum(self._batches(b) for b in self._cap)
        return num / den if den else 0.0

    def __len__(self) -> int:
        """Batches per epoch, exact under ``drop_last``.

        Exact because the sampler draws each bucket a known number of times: a
        bucket holding ``n_b`` windows fills ``n_b // cap_b`` full batches. The
        remainder is what ``drop_last`` discards.
        """
        return max(1, sum(self._batches(b) for b in self._cap))


def padding_efficiency(win_len: np.ndarray, total_len: int, step: int = BUCKET_STEP):
    """(real bases, padded bases, ratio) for a window-length array. Diagnostic only."""
    real = int(np.sum(np.clip(win_len.astype(np.int64), 1, int(total_len))))
    padded = int(np.sum(bucket_lengths(win_len, total_len, step).astype(np.int64)))
    return real, padded, (real / padded if padded else 0.0)
