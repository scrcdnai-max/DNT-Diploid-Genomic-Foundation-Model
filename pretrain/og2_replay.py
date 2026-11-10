"""OpenGenome2 replay pool: parsing, NTv3 length bins, tiling, and the streaming reader.

Single source of truth for how an OG2 ``.jsonl.gz`` record becomes training windows. The
download tool counts "usable bp" with the same functions the reader yields windows from, so
the manifest's quota accounting cannot drift from what training actually sees.

NTv3's short-context pretraining binned eligible sequences by their ORIGINAL length and
trained each bin at its own window length:

    1024 <= L < 2048 -> 1024      4096 <= L < 8192 -> 4096
    2048 <= L < 4096 -> 2048      L >= 8192        -> 8192

with token shares 7.9 / 13.0 / 13.2 / 66.0 across those bins. Reproducing that length
diversity is part of the replay experiment, not an implementation detail.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import random
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

# NTv3 short-context bins: (lower bound on original length, training window length).
# Descending so the first match wins.
_BINS: tuple[tuple[int, int], ...] = ((8192, 8192), (4096, 4096), (2048, 2048), (1024, 1024))

MIN_LEN = 1024

_NON_ACGTN = re.compile(r"[^ACGTN]+")


def segment_record(text: str, min_len: int = MIN_LEN) -> list[str]:
    """Split one OG2 ``text`` field into windowable uppercase-ACGTN segments.

    Uppercase first, then split on every non-ACGTN character. The order matters:
    eukaryotic_genic_windows is heavily soft-masked (~28% lowercase in the records
    sampled), and that is real sequence -- but the lowercase vocabulary rows are diploid
    *added* tokens that ``--drop-unseen-tokens`` deletes, so left as-is they would tokenise
    to ``<unk>``, which is a special token and therefore silently inert rather than
    loudly wrong. Uppercasing rescues them; splitting on the rest (never joining across it)
    keeps the ``@`` segment separator used by the mRNA-stitch and genic-window sources, and
    the occasional IUPAC ambiguity code, from splicing unrelated sequence into one window.
    ``min_len`` then drops the debris without needing to parse any of it.
    """
    return [s for s in _NON_ACGTN.split(text.upper()) if len(s) >= min_len]


def bin_for_length(n: int) -> int | None:
    """Return the NTv3 training-window length for a record of ``n`` bases, or None if ineligible."""
    for lower, win in _BINS:
        if n >= lower:
            return win
    return None


def tile_offsets(n: int, win: int) -> list[int]:
    """Start offsets of the full windows in NTv3's two complementary tilings.

    Tile from the left and from the right; keep only complete windows (fragments are
    discarded, never padded); return the union sorted. When ``n % win == 0`` the two
    tilings coincide and the union collapses -- deduping here is what stops those records
    from contributing every window twice.
    """
    if n < win:
        return []
    left = range(0, n - win + 1, win)
    right = range(n - win, -1, -win)
    return sorted(set(left) | set(right))


def shard_seed(base: int, source: str, shard: str = "") -> int:
    """Derive a stable child seed from ``(base, source, shard)``.

    sha256 rather than ``hash()``: str hashing is salted by PYTHONHASHSEED, so a
    ``hash()``-derived seed would silently pick different records in a different process
    and make the pool unreproducible from the manifest.
    """
    digest = hashlib.sha256(f"{base}:{source}:{shard}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def iter_texts(path: str | Path) -> Iterator[str]:
    """Yield the ``text`` field of every record in a gzipped OG2 JSONL shard."""
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)["text"]


def usable_bp(text: str) -> int:
    """Bases in ``text`` that survive segmentation -- what the reader can actually window."""
    return sum(len(s) for s in segment_record(text))


def trim_shard(src: str | Path, dst: str | Path, accept_p: float, seed: int) -> tuple[int, int]:
    """Copy a seeded Bernoulli(``accept_p``) subset of ``src``'s records into ``dst``.

    Used when a source's bp quota lands mid-shard. Accepting across the whole shard rather
    than stopping at the head avoids the taxon ordering inside a shard (records are
    concatenated per source genome/assembly). Returns ``(kept_records, kept_usable_bp)``.

    ``compresslevel=1``: at Python's default of 9 this write runs at a few MB/s and takes
    HOURS per multi-GB shard, dwarfing the download it follows, for ~10% on a file we only
    ever stream. ``mtime=0`` and an empty stored filename drop the two header fields that
    would otherwise make an identical re-run hash differently, so the manifest's sha256
    stays a real provenance check.
    """
    if not 0.0 <= accept_p <= 1.0:
        raise ValueError(f"accept_p must be in [0, 1], got {accept_p}")
    rng = random.Random(seed)
    tmp = Path(f"{dst}.tmp")
    kept = kept_bp = 0
    with gzip.open(src, "rt") as fin, open(tmp, "wb") as fh, \
            gzip.GzipFile(filename="", mode="wb", fileobj=fh, compresslevel=1, mtime=0) as gz:
        for line in fin:
            if not line.strip() or rng.random() >= accept_p:
                continue
            gz.write(line.encode() if line.endswith("\n") else (line + "\n").encode())
            kept += 1
            kept_bp += usable_bp(json.loads(line)["text"])
    os.replace(tmp, dst)
    return kept, kept_bp


def shards_for(manifest_tsv: str | Path, sources: Sequence[str] | None = None) -> list[str]:
    """Local paths of the pool's shards, from the download manifest."""
    with open(manifest_tsv, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [r["local_path"] for r in rows if sources is None or r["source"] in sources]


def bin_weights(
    manifest_tsv: str | Path,
    bin_len: int,
    sources: Sequence[str] | None = None,
) -> tuple[list[str], list[int]]:
    """Pool shards and how many ``bin_len`` windows each holds, per its download-time scan.

    The pool is length-stratified BY SHARD -- a metagenome shard can hold 17M 1024-bin
    windows and nothing else, while its neighbours hold none -- so a uniform pass spends
    nearly all its decompression on shards that yield nothing for the bin being streamed.
    Feed these weights to ``iter_windows`` to visit each shard in proportion to what it
    actually holds.

    Sparse shards keep a small weight rather than being dropped: the thin tail of the 1024
    bin is where ncRNA, GTDB and organelle live, and replaying biological diversity is the
    point of the experiment. A shard with no scan sidecar gets the median weight, so an
    unscanned shard is visited like a typical one instead of being silently excluded.
    """
    paths = shards_for(manifest_tsv, sources)
    scans = Path(manifest_tsv).parent.parent / "stats" / "shards"
    known: dict[str, int] = {}
    for p in paths:
        sidecar = scans / f"{Path(p).name}.json"
        if sidecar.exists():
            bins = json.loads(sidecar.read_text())["bins"]
            known[p] = int(bins.get(str(bin_len), {}).get("windows", 0))
    fallback = int(np.median(list(known.values()))) if known else 1
    return paths, [known.get(p, fallback) for p in paths]


def scan_shard(path: str | Path) -> dict:
    """One pass over a shard, producing everything both the quota loop and the stats need.

    ``usable_bp`` here is what the reader would actually window (post-segmentation), which
    is what a bp quota has to mean -- raw record bases would over-count every tagged or
    ambiguity-code-bearing source. The character histogram is the tripwire for a source
    that turns out not to be plain ACGTN.
    """
    stats = {
        "records": 0,
        "raw_bp": 0,
        "usable_bp": 0,
        "chars": {c: 0 for c in "ACGTN"} | {"other": 0},
        "bins": {str(w): {"segments": 0, "bp": 0, "windows": 0} for _, w in _BINS},
    }
    for text in iter_texts(path):
        up = text.upper()
        stats["records"] += 1
        stats["raw_bp"] += len(up)
        known = 0
        for c in "ACGTN":
            n = up.count(c)
            stats["chars"][c] += n
            known += n
        stats["chars"]["other"] += len(up) - known
        for seg in segment_record(up):
            win = bin_for_length(len(seg))
            if win is None:
                continue
            b = stats["bins"][str(win)]
            b["segments"] += 1
            b["bp"] += len(seg)
            b["windows"] += len(tile_offsets(len(seg), win))
            stats["usable_bp"] += len(seg)
    return stats


def consumer_slot(rank: int, world_size: int, worker_id: int, num_workers: int) -> tuple[int, int]:
    """Flatten (DDP rank, DataLoader worker) into ``(index, n_consumers)``."""
    return rank * num_workers + worker_id, world_size * num_workers


def iter_windows(
    shard_paths: Sequence[str | Path],
    bin_len: int,
    seed: int = 42,
    flat: int = 0,
    total: int = 1,
    shuffle_buffer: int = 10_000,
    max_cycles: int | None = None,
    weights: Sequence[float] | None = None,
) -> Iterator[str]:
    """Stream replay windows of exactly ``bin_len`` bases, as strings.

    Records are assigned to consumers by position (``counter % total == flat``), not by
    shard: five of the nine OG2 sources end up as a single shard, and shard-level
    assignment would leave most consumers with nothing to read from them. Records outside
    the requested bin are skipped, so one instance per bin gives homogeneous-length batches
    with no bucketing logic downstream.

    With ``weights`` (see ``bin_weights``) each pass draws ONE shard with probability
    proportional to its window count for this bin, instead of walking all of them. The pool
    is length-stratified by shard, so a uniform walk spends nearly all its decompression on
    shards that hold nothing for the current bin; weighting makes scan time track yield
    while leaving the expected composition equal to uniform sampling over the bin's windows.

    Runs forever by default so no rank can exhaust while others are still training;
    ``max_cycles`` bounds it for tests. A pass that produces nothing raises instead of
    spinning: a consumer whose slice of the pool holds no record in this bin would
    otherwise burn a core forever while the rank it feeds hangs at the next all-reduce,
    with no error anywhere.
    """
    paths = list(shard_paths)
    if not paths:
        raise ValueError("iter_windows: no shard paths")
    if weights is not None and not any(weights):
        raise RuntimeError(f"iter_windows: no shard holds a {bin_len}bp window")
    rng = random.Random(seed + flat)
    buf: list[str] = []
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        if weights is None:
            order = sorted(paths)
            random.Random(seed + cycle).shuffle(order)
        else:
            order = random.Random(seed + flat + cycle).choices(paths, weights, k=1)
        counter = 0
        n_windows = 0
        for path in order:
            for text in iter_texts(path):
                take = counter % total == flat
                counter += 1
                if not take:
                    continue
                for seg in segment_record(text):
                    if bin_for_length(len(seg)) != bin_len:
                        continue
                    for off in tile_offsets(len(seg), bin_len):
                        buf.append(seg[off:off + bin_len])
                        n_windows += 1
                        if len(buf) >= shuffle_buffer:
                            j = rng.randrange(len(buf))
                            buf[j], buf[-1] = buf[-1], buf[j]
                            yield buf.pop()
        if not n_windows and weights is None:
            raise RuntimeError(
                f"iter_windows: consumer {flat}/{total} found no {bin_len}bp windows in "
                f"{len(paths)} shard(s). Either this bin is unsupplied (check the pool "
                f"stats) or the pool has fewer records than consumers."
            )
        # Drain what the buffer never filled up to: on a pool smaller than the buffer the
        # windows would otherwise sit there while the consumer spins through more cycles.
        rng.shuffle(buf)
        while buf:
            yield buf.pop()
        cycle += 1


def encode_window(window: str, char_to_id: "np.ndarray") -> "np.ndarray":
    """Codepoint-LUT gather, the same three lines the human path uses in ``train.py``.

    ``mode="clip"`` sends any codepoint past the end of the table to the reserved unk
    sentinel rather than aliasing it onto a real token. Segmentation has already
    guaranteed uppercase ACGTN, so nothing should reach that sentinel here.
    """
    codepoints = np.frombuffer(window.encode("utf-32-le"), dtype=np.uint32)
    return char_to_id.take(codepoints, mode="clip")


class Og2WindowIterable(torch.utils.data.IterableDataset):
    """Infinite stream of fixed-length OG2 replay windows for one NTv3 length bin.

    One instance per bin: batches are then homogeneous in length by construction, with no
    bucketing and no padding, and the caller controls the 7.9/13.0/13.2/66.0 token mix by
    how often it draws from each bin's loader. The human loader is untouched -- these are
    separate batches, never mixed into one tensor.

    Emits only ``input_ids`` and ``attention_mask`` (all ones): MLM masking happens in the
    collator, and the diploid auxiliary objectives (CPL/SMP) fall back to their DDP dummies
    on a marker-free batch, so no fabricated diploid targets are needed.
    """

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        bin_len: int,
        char_to_id: "np.ndarray",
        seed: int = 42,
        shuffle_buffer: int = 10_000,
        rank: int = 0,
        world_size: int = 1,
        weights: Sequence[float] | None = None,
    ) -> None:
        self.shard_paths = list(shard_paths)
        self.bin_len = bin_len
        self.char_to_id = char_to_id
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.rank = rank
        self.world_size = world_size
        self.weights = list(weights) if weights is not None else None

    def __iter__(self) -> Iterator[dict]:
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        flat, total = consumer_slot(self.rank, self.world_size, worker_id, num_workers)
        attn = torch.ones(self.bin_len, dtype=torch.int64)
        for window in iter_windows(
            self.shard_paths,
            self.bin_len,
            seed=self.seed,
            flat=flat,
            total=total,
            shuffle_buffer=self.shuffle_buffer,
            weights=self.weights,
        ):
            ids = torch.from_numpy(encode_window(window, self.char_to_id).copy())
            yield {"input_ids": ids, "attention_mask": attn}
