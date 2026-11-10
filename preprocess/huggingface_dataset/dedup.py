"""Deduplication machinery for ``create_dataset.py`` — extracted verbatim so
the orchestration module stays focused on the generate/assemble flow.

Three concerns live here:

1. **Hashing** — every sequence is keyed by a ``(h1, h2)`` pair (two independent
   xxh64 hashes). Two sequences are treated as identical only when BOTH match, so
   a false merge needs a simultaneous collision on two independent 64-bit hashes
   (~2^-128); distinct sequences are never silently merged.
2. **Assemble dedup** — ``_dedup_by_hash_pair`` / ``_primary_hash_collisions``
   collapse duplicate rows and report (never merge) primary-hash collisions.
3. **Inline dedup** — a live ``(h1, h2)`` hash set in POSIX shared memory
   (``_LiveDedupTable``), shared across all generate workers so a sequence written
   by any worker is skipped by every other for the rest of the run. Skipped
   weights are recorded in per-sample sidecars (``_write_skip_weight_sidecar`` /
   ``_load_skip_weights``) and folded back during assemble.

Module state shared across the worker fork: ``_GEN_DEDUP`` (per-split shared
table descriptors) is set by ``cmd_generate`` BEFORE the Pool forks and inherited
by workers; ``_WORKER_DEDUP_TABLES`` caches one attach per worker process. The
orchestrator must assign these as ``dedup._GEN_DEDUP = …`` (module attribute) so
the value the forked children inherit is the one this module exposes.

Leaf module: imports only stdlib + numpy/pyarrow/xxhash, so ``create_dataset``
imports it without a cycle.
"""
from __future__ import annotations

import logging
import struct
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

logger = logging.getLogger(__name__)


def _record_key_bytes(seq: str, snp, indel_span, direction) -> bytes:
    """Canonical, lossless byte key for a v8 record (markers externalized).

    For v8 the ``seq`` string alone is NOT a unique key: under three-level
    encoding two opposite phasings emit the same canonical symbol and differ
    only by the (now externalized) direction, and a homozygous-ALT SNP collapses
    to a plain base distinguished only by the ``snp`` column. So the key folds in
    the marker columns: the UTF-8 ``seq`` followed by tagged, length-prefixed
    serializations of ``snp`` (ints), ``indel_span`` ((start, end) pairs) and
    ``direction`` ((pos, dir) pairs). All three are ascending by construction, so
    the encoding is deterministic. Generation (should_skip) and assemble
    (_hash_shard_pair) must build this identically so the (h1, h2) keys — and the
    skip-weight fold — agree.
    """
    parts = [seq.encode("utf-8"), b"\x00S", struct.pack("<I", len(snp))]
    for p in snp:
        parts.append(struct.pack("<i", int(p)))
    parts.append(b"\x00I")
    parts.append(struct.pack("<I", len(indel_span)))
    for s, e in indel_span:
        parts.append(struct.pack("<ii", int(s), int(e)))
    parts.append(b"\x00D")
    parts.append(struct.pack("<I", len(direction)))
    for p, d in direction:
        parts.append(struct.pack("<iB", int(p), int(d)))
    return b"".join(parts)


# Independent second-hash seed.  Two sequences are treated as identical only
# when BOTH the primary (seed-0) hash and this verify hash match — a false merge
# then needs a simultaneous collision on two independent 64-bit hashes (~2^-128),
# so distinct sequences are never silently merged even though each hash is 64-bit.
# Used by both the assemble dedup and the live inline-dedup table.
_VERIFY_HASH_SEED = 0x9E3779B97F4A7C15


def _hash_shard_pair(
    parquet_path: str,
    seq_normalizer: Optional[Callable[[str], str]] = None,
) -> tuple:
    """Read one Parquet shard; return (h1, h2, weights).

    h1 is the primary xxh64 (identical to _hash_seq_column and the live
    inline-dedup table's primary key); h2 is an independent xxh64 (different
    seed) used to keep sequences that collide on h1 apart (see
    _dedup_by_hash_pair and the live dedup table).

    PERFORMANCE: this computes a SECOND hash per row.  Hashing is not the
    assemble bottleneck (HF shuffle/save dominates) and xxh64 runs at ~GB/s, so
    the extra pass should be minor — but it is the cost of the collision
    safeguard; profile here if assemble Step 2 ever shows up hot.

    v8 (self-describing): a shard whose schema has the ``snp`` column carries
    externalized markers, so the key must fold them in via _record_key_bytes —
    exactly as generation's should_skip did — or distinct phasings/hom-alt SNPs
    that share a ``seq`` would be wrongly merged. v5/v6/v7 shards keep the
    byte-identical seq-only hash.

    ``seq_normalizer``: when provided, each v5/v6/v7 ``seq`` string is passed
    through this callable before hashing.  Used for STRIP_DIRECTION (v6→v6nd)
    so that two v6 records differing only in ``‹``/``›`` markers hash to the
    same key and are correctly merged by dedup.  Must be a module-level function
    (picklable) when called via ``multiprocessing.Pool``.  Ignored for v8 shards
    (their compound key already captures the full record identity).
    """
    names = set(pq.read_schema(parquet_path).names)
    externalize = "snp" in names
    cols = ["seq", "sample_weight"]
    if externalize:
        cols += ["snp", "indel_span"]
        if "direction" in names:
            cols.append("direction")
    table = pq.read_table(parquet_path, columns=cols)
    seqs = table.column("seq").to_pylist()
    n = len(seqs)
    h1 = np.empty(n, dtype=np.uint64)
    h2 = np.empty(n, dtype=np.uint64)
    if externalize:
        snp_col = table.column("snp").to_pylist()
        ind_col = table.column("indel_span").to_pylist()
        dir_col = (
            table.column("direction").to_pylist()
            if "direction" in names else [None] * n
        )
        for i in range(n):
            kb = _record_key_bytes(
                seqs[i],
                snp_col[i] or [],
                [(e["start"], e["end"]) for e in (ind_col[i] or [])],
                [(e["pos"], e["dir"]) for e in (dir_col[i] or [])],
            )
            h1[i] = xxhash.xxh64_intdigest(kb)
            h2[i] = xxhash.xxh64_intdigest(kb, seed=_VERIFY_HASH_SEED)
    else:
        for i, s in enumerate(seqs):
            if seq_normalizer is not None:
                s = seq_normalizer(s)
            sb = s.encode("utf-8")
            h1[i] = xxhash.xxh64_intdigest(sb)
            h2[i] = xxhash.xxh64_intdigest(sb, seed=_VERIFY_HASH_SEED)
    weights = table.column("sample_weight").to_numpy().astype(np.int64)
    del table, seqs
    return h1, h2, weights


def _dedup_by_hash_pair(
    h1: np.ndarray, h2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dedup rows by the (primary, verify) hash pair.

    Returns (unique_pairs, first_indices, inverse) from np.unique over a
    structured (h1, h2) key.  unique_pairs is sorted by h1 then h2, so
    unique_pairs["h1"] is non-decreasing (used by the skip-weight fold and the
    collision report).  Distinct sequences that collide on h1 land in separate
    groups because their h2 differs.

    PERFORMANCE / MEMORY: the structured key is 16 bytes/row vs 8 for a single
    hash, so np.unique's sort and the transient arrays roughly double the peak
    hash memory of Step 3.  This is the price of collision-safety; revisit if
    assemble memory becomes a constraint.
    """
    pair = np.empty(len(h1), dtype=np.dtype([("h1", "<u8"), ("h2", "<u8")]))
    pair["h1"] = h1
    pair["h2"] = h2
    unique_pairs, first_indices, inverse = np.unique(
        pair, return_index=True, return_inverse=True
    )
    return unique_pairs, first_indices, inverse.reshape(-1)


def _primary_hash_collisions(unique_pairs: np.ndarray) -> Tuple[int, int, list]:
    """Detect primary-hash (h1) values shared by ≥2 distinct (h1, h2) groups.

    Returns (n_colliding_hashes, n_extra_sequences, example_h1).  n_extra_sequences
    is the number of distinct sequences a hash-only dedup would have silently
    merged/dropped.  unique_pairs must be sorted by h1 (np.unique guarantees it).
    """
    h1_uniq = unique_pairs["h1"]
    if len(h1_uniq) < 2:
        return 0, 0, []
    dup = np.zeros(len(h1_uniq), dtype=bool)
    dup[1:] = h1_uniq[1:] == h1_uniq[:-1]
    n_extra = int(dup.sum())
    if n_extra == 0:
        return 0, 0, []
    colliding = np.unique(h1_uniq[dup])
    return len(colliding), n_extra, colliding[:5].tolist()


# ---------------------------------------------------------------------------
# Inline deduplication: a live (h1, h2) hash set SHARED across all generate
# workers, so a sequence written by any worker is skipped by every other worker
# for the rest of the run — cross-sample dedup that works on a FRESH start, not
# only on resume.  Open addressing in POSIX shared memory, guarded by one Lock.
# A sample's kept hashes are published to the table only AFTER its shard is
# atomically finalized (crash-safe: the table never references unwritten rows).
# Keyed on the (h1, h2) pair so an h1 collision never skips a distinct sequence.
# ---------------------------------------------------------------------------

# Per-split shared dedup state, set by cmd_generate BEFORE the worker Pool forks
# and inherited by workers (fork start method).  None when --no-dedup-inline.
# Maps split -> {"shm_name", "capacity", "lock", "size", "full"}.
_GEN_DEDUP: Optional[Dict[str, dict]] = None
# Per-worker cache of attached tables (one attach per worker process per split).
_WORKER_DEDUP_TABLES: Dict[str, "_LiveDedupTable"] = {}


def _dedup_table_capacity(n_pending: int, n_regions: int,
                          override: Optional[int]) -> int:
    """Power-of-two slot count for the live dedup table (16 bytes/slot).

    Default assumes a high duplicate rate (unique << rows) but stays
    conservative; a table that fills just degrades to assemble-only dedup, never
    an error."""
    if override:
        return 1 << max(0, int(override) - 1).bit_length()
    rows = max(1, n_pending) * max(1, n_regions)
    target = max(1 << 12, int(0.5 * rows))
    cap = 1 << (target - 1).bit_length()
    return min(cap, 1 << 30)


class _LiveDedupTable:
    """Concurrent fixed-capacity (h1, h2) hash set in shared memory.

    Open addressing, linear probing, power-of-two capacity (index is a
    bit-mask).  Empty slot == (0, 0); a sequence hashing to exactly (0, 0)
    (~2^-128) is simply never deduped.  Every operation holds the external lock.
    Inserts stop at 70% load (logged once); remaining duplicates then fall
    through to the authoritative assemble dedup — never an error or data loss.
    """

    def __init__(self, arr, capacity, lock, size, full):
        self._arr = arr            # uint64 view, length 2*capacity
        self._cap = capacity
        self._mask = capacity - 1
        self._lock = lock
        self._size = size          # RawValue('q'): live element count
        self._full = full          # RawValue('b'): table-full flag
        self._max_load = int(capacity * 0.7)
        self._shm = None           # set by attach(), kept alive

    @classmethod
    def attach(cls, shm_name, capacity, lock, size, full):
        shm = SharedMemory(name=shm_name)
        arr = np.ndarray((capacity * 2,), dtype=np.uint64, buffer=shm.buf)
        table = cls(arr, capacity, lock, size, full)
        table._shm = shm
        return table

    def contains(self, h1: int, h2: int) -> bool:
        if h1 == 0 and h2 == 0:
            return False
        with self._lock:
            arr, mask = self._arr, self._mask
            idx = h1 & mask
            while True:
                j = idx << 1
                s1 = arr[j]; s2 = arr[j + 1]
                if s1 == 0 and s2 == 0:
                    return False
                if s1 == h1 and s2 == h2:
                    return True
                idx = (idx + 1) & mask

    def add_batch(self, pairs) -> None:
        with self._lock:
            if self._full.value:
                return
            arr, mask = self._arr, self._mask
            for h1, h2 in pairs:
                if h1 == 0 and h2 == 0:
                    continue
                if self._size.value >= self._max_load:
                    self._full.value = 1
                    logger.warning(
                        "Live dedup table full (%d/%d slots); remaining "
                        "duplicates deferred to assemble dedup.",
                        self._size.value, self._cap,
                    )
                    return
                idx = h1 & mask
                while True:
                    j = idx << 1
                    s1 = arr[j]; s2 = arr[j + 1]
                    if s1 == 0 and s2 == 0:
                        arr[j] = h1; arr[j + 1] = h2
                        self._size.value += 1
                        break
                    if s1 == h1 and s2 == h2:
                        break
                    idx = (idx + 1) & mask


class _LiveSkipFilter:
    """Per-sample skip filter over the shared live dedup table.

    should_skip(seq[, snp, indel_span, direction]) -> (skip, h1, h2): skip is
    True when the record was already written this run (present in the shared
    table) or earlier in THIS sample (local set).  Novel records are remembered
    locally and published to the shared table by publish(), called once the
    sample's shard is finalized. The (h1, h2) pair returned keys the skip-weight
    sidecar (exact fold).

    ``externalize`` selects the key: v5/v6/v7 hash the ``seq`` string alone
    (byte-identical to the legacy behaviour); v8 hashes the lossless
    :func:`_record_key_bytes` over seq + the marker columns, since seq alone is
    not unique once markers are externalized.
    """

    def __init__(self, table: "_LiveDedupTable", externalize: bool = False):
        self._table = table
        self._externalize = externalize
        self._local: Set[Tuple[int, int]] = set()

    def should_skip(
        self, seq: str, snp=None, indel_span=None, direction=None,
    ) -> Tuple[bool, int, int]:
        if self._externalize:
            kb = _record_key_bytes(
                seq, snp or [], indel_span or [], direction or [])
            h1 = xxhash.xxh64_intdigest(kb)
            h2 = xxhash.xxh64_intdigest(kb, seed=_VERIFY_HASH_SEED)
        else:
            sb = seq.encode("utf-8")
            h1 = xxhash.xxh64_intdigest(sb)
            h2 = xxhash.xxh64_intdigest(sb, seed=_VERIFY_HASH_SEED)
        key = (h1, h2)
        if key in self._local or self._table.contains(h1, h2):
            return True, h1, h2
        self._local.add(key)
        return False, h1, h2

    def publish(self) -> None:
        if self._local:
            self._table.add_batch(self._local)


def _get_worker_dedup_table(split: str) -> "_LiveDedupTable":
    """Attach (once per worker process) to the shared dedup table for *split*."""
    table = _WORKER_DEDUP_TABLES.get(split)
    if table is None:
        info = _GEN_DEDUP[split]
        table = _LiveDedupTable.attach(
            info["shm_name"], info["capacity"],
            info["lock"], info["size"], info["full"],
        )
        _WORKER_DEDUP_TABLES[split] = table
    return table


# ---------------------------------------------------------------------------
# Skip-weight sidecars: per-sample {seq_hash: Σ skipped weight}.  Inline dedup
# drops duplicate occurrences instead of writing them, so their weight must be
# recorded here and folded back into the surviving row during assemble.  They
# live under a skip_weights/ subdir so the (non-recursive) *.parquet shard/seed
# globs never pick them up.
# ---------------------------------------------------------------------------

_SKIP_WEIGHTS_SUBDIR = "skip_weights"


def _write_skip_weight_sidecar(skip_weight: Dict[Tuple[int, int], int],
                               sidecar_path: Path) -> None:
    """Atomically write {(h1, h2): weight} to a Parquet sidecar (.tmp -> rename).

    Storing both hashes lets assemble fold the weight onto the exact (h1, h2)
    group, so attribution is correct even under a primary-hash collision."""
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(skip_weight)
    h1 = np.empty(n, dtype=np.uint64)
    h2 = np.empty(n, dtype=np.uint64)
    w = np.empty(n, dtype=np.int64)
    for i, ((a, b), wt) in enumerate(skip_weight.items()):
        h1[i] = a
        h2[i] = b
        w[i] = wt
    table = pa.table({
        "seq_hash": pa.array(h1, pa.uint64()),
        "seq_hash2": pa.array(h2, pa.uint64()),
        "skip_weight": pa.array(w, pa.int64()),
    })
    tmp = sidecar_path.parent / (sidecar_path.name + ".tmp")
    pq.write_table(table, tmp, compression="none")
    tmp.rename(sidecar_path)


def _load_skip_weights(
    shard_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load all skip-weight sidecars under shard_dir/skip_weights/.

    Returns (h1, h2, skip_weights, n_sidecar_files).  Pairs are NOT pre-aggregated
    across files — the caller folds them via np.add.at.  A missing skip_weights/
    dir (e.g. shards generated before this feature) yields empty arrays, i.e.
    zero skip weight.
    """
    d = shard_dir / _SKIP_WEIGHTS_SUBDIR
    files = sorted(d.glob("*.parquet")) if d.is_dir() else []
    if not files:
        return (np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64),
                np.empty(0, dtype=np.int64), 0)
    h1c, h2c, wc = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["seq_hash", "seq_hash2", "skip_weight"])
        h1c.append(np.asarray(
            t.column("seq_hash").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.uint64))
        h2c.append(np.asarray(
            t.column("seq_hash2").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.uint64))
        wc.append(np.asarray(
            t.column("skip_weight").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64))
    return np.concatenate(h1c), np.concatenate(h2c), np.concatenate(wc), len(files)
