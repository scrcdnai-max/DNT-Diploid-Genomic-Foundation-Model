#!/usr/bin/env python
"""Pre-flight for ``--drop-unseen-tokens``: prove the symbols it would drop really
are absent from the corpus.

The filter trusts ``preprocess/token_counts.tsv``. If that table is stale for the
dataset actually being trained on, a live symbol gets marked dead, its embedding
row is deleted, and from then on it silently tokenizes to ``<unk>`` -- corrupting
exactly the variant positions the model exists to read. Nothing downstream would
notice, so the check has to happen before the run starts.

    uv run --no-sync python -m pretrain.tools.verify_dead_tokens \
        --data /storage/data/1kgenomes/diploid_hf_v7/assembled/train

Exit 1 if any symbol the filter would drop is actually present. Sampling is by
whole shards, which is sound here because shards are not locus-partitioned
(2,561 samples and 127,779 distinct loci appear in shard 0 alone) -- see
SEMANTIC_DEDUP.md.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pretrain"))

from train import (
    DEFAULT_TOKEN_COUNTS_PATH,
    _parse_diploid_tsv,
    load_token_unigram_freqs,
)

DEFAULT_TOKENS_TSV = REPO_ROOT / "preprocess" / "diploid_tokens.tsv"
DEFAULT_DATA = Path("/storage/data/1kgenomes/diploid_hf_v7/assembled/train")


def _count_symbols(shard: Path, wanted: set[str], max_rows: int) -> Counter[str]:
    """Count occurrences of *wanted* symbols in one arrow shard.

    Decodes to UTF-32 and works on the codepoint array rather than doing a
    per-character Python loop: these shards are ~5 Gbase each.
    """
    codes = {ord(sym): sym for sym in wanted}
    found: Counter[str] = Counter()
    with pa.memory_map(str(shard), "rb") as src:
        table = pa.ipc.open_stream(src).read_all()
        seqs = table.column("seq")
        seen = 0
        for chunk in seqs.chunks:
            for s in chunk.to_pylist():
                if s is None:
                    continue
                arr = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)
                for cp, sym in codes.items():
                    n = int(np.count_nonzero(arr == cp))
                    if n:
                        found[sym] += n
                seen += 1
                if seen >= max_rows:
                    return found
    return found


def _select_shards(all_shards: list[Path], n: int) -> list[Path]:
    """Pick ``n`` shards spread across the corpus, not the first ``n``.

    This is a PRESENCE test -- absence in the sample is what green-lights dropping
    29 embedding rows -- and the assembly is ordered by sample_id, so the head of
    the list is one slice of the panel rather than a sample of the corpus. v7 had
    204 shards (the default 4 = 2% of them); v7-tsstes has 1112 (4 = 0.36%), so the
    head-of-list sample got much weaker exactly where the corpus grew.
    """
    if not all_shards:
        return []
    n = max(1, int(n))
    step = max(1, len(all_shards) // n)
    return all_shards[::step][:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS_TSV)
    ap.add_argument("--token-counts", type=Path, default=DEFAULT_TOKEN_COUNTS_PATH)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--rows-per-shard", type=int, default=20000)
    a = ap.parse_args()

    symbols, _, _ = _parse_diploid_tsv(a.tokens)
    freqs = load_token_unigram_freqs(a.token_counts)
    if freqs is None:
        print(f"FATAL: no token counts at {a.token_counts}", file=sys.stderr)
        return 2

    # Same predicate as _filter_unseen_tokens: a symbol survives if the counts
    # table records any occurrence. Symbols missing from the table count as zero,
    # which is the conservative direction -- they land in the drop set, so this
    # check catches it if they actually occur.
    would_drop = {s for s in symbols if freqs.get(s, 0.0) <= 0.0}
    if not would_drop:
        print("filter would drop nothing -- vacuously safe")
        return 0

    shards = _select_shards(sorted(a.data.glob("data-*.arrow")), a.shards)
    if not shards:
        print(f"FATAL: no arrow shards under {a.data}", file=sys.stderr)
        return 2

    print(f"checking {len(would_drop)} would-be-dropped symbols against "
          f"{len(shards)} shards x {a.rows_per_shard} rows")
    print("dropped set: " + " ".join(sorted(would_drop)))

    total: Counter[str] = Counter()
    for sh in shards:
        got = _count_symbols(sh, would_drop, a.rows_per_shard)
        total.update(got)
        print(f"  {sh.name}: {'clean' if not got else dict(got)}", flush=True)

    if total:
        print()
        print("FAILED: these symbols are marked count-0 but DO occur:")
        for sym, n in sorted(total.items(), key=lambda kv: -kv[1]):
            print(f"  {sym!r} (U+{ord(sym):04X}): {n} occurrences")
        print("token_counts.tsv is stale for this dataset. Do NOT enable "
              "--drop-unseen-tokens until it is regenerated.")
        return 1

    print()
    print(f"OK: none of the {len(would_drop)} dropped symbols occur in the sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
