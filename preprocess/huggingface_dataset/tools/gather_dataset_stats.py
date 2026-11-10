#!/usr/bin/env python3
"""Token-frequency statistics over an **assembled** diploid/haploid dataset.

This measures what the model actually sees: the post-dedup, post-shuffle Arrow /
Parquet corpus produced by ``create_dataset.py assemble``. It is the dataset-side
counterpart to ``preprocess/tools/gather_diploid_stats_bcf.py``, which derives the same kind of
counts from the *source* BCFs plus the reference genome. The two answer different
questions and will not agree — dedup drops duplicate regions, ``sample_weight``
re-weights what survives, and only BED-covered regions are stored.

Reported:

* corpus-wide symbol counts, classed via ``diploid_tokens.tsv``
  (symmetric / unphased / phased / single / meta);
* per-``sample_id`` counts (incl. the ``HG38_REF`` pseudo-sample), with record
  count and summed ``sample_weight``;
* structural markers (``^`` ``{`` ``}`` ``‹`` ``›``) counted separately, and
  **comparably across schema versions**: v5/v6/v7 carry them inline in ``seq``,
  v8 stores them in the sparse ``snp`` / ``indel_span`` / ``direction`` columns,
  and both paths produce the same marker totals for the same logical content.

Usage:
    python -m preprocess.huggingface_dataset.tools.gather_dataset_stats /path/to/assembled_ds --output-dir ./ds_stats

    # single split, quick smoke test on the first 20k rows
    python -m preprocess.huggingface_dataset.tools.gather_dataset_stats /path/to/assembled_ds --output-dir ./ds_stats \\
        --split train --limit 20000

Parallelism mirrors ``compare_assembled.py fingerprint``: the dataset is opened
once in the parent and sharded logically (so it works for both Arrow dirs and
Parquet dirs), workers inherit it via fork copy-on-write, and each returns its
own counters to be merged. Workers default to the physical-core count — the scan
is Arrow-decode/memory-bandwidth bound, so oversubscribing regresses.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from preprocess import paths, tokens
from preprocess.huggingface_dataset.dataset import (  # noqa: E402
    DiploidDataset,
    HaploidDataset,
    read_representation,
)

DEFAULT_TOKENS_PATH = paths.DIPLOID_TOKENS

# Meta-token *pair* names in diploid_tokens.tsv → the marker they denote. Used to
# map v8's sparse marker columns onto the same symbols v7 stores inline.
_PAIR_SNP = "<SNP>"
_PAIR_INDEL_OPEN = "<INDEL_span_start>"
_PAIR_INDEL_CLOSE = "<INDEL_span_end>"
_PAIR_DIR_FWD = "<DIR_FWD>"
_PAIR_DIR_REV = "<DIR_REV>"

_UNKNOWN_CLASS = "unknown"


# ---------------------------------------------------------------------------
# Token table
# ---------------------------------------------------------------------------


class TokenTable:
    """Symbol → class / unicode lookup parsed from ``diploid_tokens.tsv``.

    Rows come from :func:`preprocess.tokens.iter_token_rows` (stdlib only, so this
    script still runs without cyvcf2/pysam or the encoder on the path). It keeps the
    file order and the Unicode column, both needed for stable output columns.
    """

    def __init__(self, symbols, symbol_class, symbol_unicode, meta_by_pair):
        self.symbols = symbols                  # file order, deduplicated
        self.symbol_class = symbol_class        # symbol -> class
        self.symbol_unicode = symbol_unicode    # symbol -> "U+XXXX"
        self.meta_by_pair = meta_by_pair        # "<SNP>" -> "^"

    def class_of(self, symbol: str) -> str:
        return self.symbol_class.get(symbol, _UNKNOWN_CLASS)

    def unicode_of(self, symbol: str) -> str:
        return self.symbol_unicode.get(symbol, f"U+{ord(symbol):04X}")

    def is_meta(self, symbol: str) -> bool:
        return self.symbol_class.get(symbol) == "meta"


def load_token_table(tokens_path: Path) -> TokenTable:
    """Parse ``diploid_tokens.tsv`` into a :class:`TokenTable`.

    Many pairs map to one symbol (``AC`` and ``CA`` both → ``Ą``), so symbols are
    deduplicated on first appearance; the class of the first row wins.
    """
    symbols: list = []
    symbol_class: dict = {}
    symbol_unicode: dict = {}
    meta_by_pair: dict = {}

    # strip=False and the eager ord() keep this reader's pre-reorg behaviour on padded
    # lines and multi-character symbols (post-reorg follow-up in preprocess/docs/REORG_PLAN_REV1.md);
    # after that fix this becomes an adapter over tokens.load_token_table.
    for pair, order, symbol, uni in tokens.iter_token_rows(tokens_path, strip=False):
        if uni is None:
            uni = f"U+{ord(symbol):04X}"
        if symbol not in symbol_class:
            symbols.append(symbol)
            symbol_class[symbol] = order
            symbol_unicode[symbol] = uni
        if order == "meta":
            meta_by_pair[pair] = symbol

    if not symbols:
        raise ValueError(f"No token rows parsed from {tokens_path}.")
    return TokenTable(symbols, symbol_class, symbol_unicode, meta_by_pair)


# ---------------------------------------------------------------------------
# Parallel counting workers (fork-inherited globals; no per-worker reload)
# ---------------------------------------------------------------------------

# Set in the parent before the pool forks; children inherit via copy-on-write.
_HF = None                # the (possibly --limit-sliced) HuggingFace Dataset
_BATCH_SIZE = 2000
_MARKER_SYMBOLS = None    # (snp, indel_open, indel_close, dir_fwd, dir_rev) or None
_HAS_DIRECTION = False    # haploid v8 has no `direction` column


def _count_shard(task) -> dict:
    """Count symbols over one contiguous shard, bucketed by ``sample_id``.

    Returns a picklable telemetry dict whose ``samples`` entry holds one
    ``Counter`` per sample — small (≤ a few hundred keys) regardless of row
    count, so merging in the parent is cheap.

    Stages are timed per *batch* (two ``perf_counter`` calls per ~2000 rows) so
    the instrumentation cost stays negligible while still separating Arrow
    decode from counting.
    """
    shard_index, num_shards = task
    sub = _HF.shard(num_shards=num_shards, index=shard_index, contiguous=True)

    counters: dict = {}      # sample_id -> Counter(symbol -> count)
    records: Counter = Counter()   # sample_id -> n records
    weights: Counter = Counter()   # sample_id -> summed sample_weight

    iter_s = count_s = marker_s = 0.0
    n = 0
    seq_chars = 0

    it = sub.iter(batch_size=_BATCH_SIZE)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        iter_s += t1 - t0

        sample_ids = batch["sample_id"]
        seqs = batch["seq"]
        sample_weights = batch["sample_weight"]

        for sid, seq, w in zip(sample_ids, seqs, sample_weights):
            c = counters.get(sid)
            if c is None:
                c = counters[sid] = Counter()
            # Counter.update over a str is the C-level _count_elements path;
            # do NOT go through to_numpy(), which is a per-char Python ord() loop.
            c.update(seq)
            records[sid] += 1
            weights[sid] += w
            seq_chars += len(seq)
        n += len(seqs)
        t2 = time.perf_counter()
        count_s += t2 - t1

        # v8: markers live in sparse columns, not in `seq`. Fold them into the
        # same per-sample counters so every table is version-comparable. Cost is
        # proportional to marker count, not sequence length.
        if _MARKER_SYMBOLS is not None:
            snp_sym, open_sym, close_sym, fwd_sym, rev_sym = _MARKER_SYMBOLS
            snps = batch["snp"]
            spans = batch["indel_span"]
            dirs = batch["direction"] if _HAS_DIRECTION else None
            for k, sid in enumerate(sample_ids):
                c = counters[sid]
                n_snp = len(snps[k] or ())
                if n_snp:
                    c[snp_sym] += n_snp
                for span in (spans[k] or ()):
                    # A -1 sentinel marks an edge-truncated block: that side's
                    # glyph was never emitted inline either (see
                    # dataset._rehydrate_marked), so it must not be counted.
                    if span["start"] != -1:
                        c[open_sym] += 1
                    if span["end"] != -1:
                        c[close_sym] += 1
                if dirs is not None:
                    for entry in (dirs[k] or ()):
                        c[fwd_sym if entry["dir"] == 1 else rev_sym] += 1
        marker_s += time.perf_counter() - t2

    return {
        "n": n,
        "seq_chars": seq_chars,
        "samples": {
            sid: {"n": records[sid], "w": weights[sid], "c": c}
            for sid, c in counters.items()
        },
        "iter_s": iter_s, "count_s": count_s, "marker_s": marker_s,
        "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


# ---------------------------------------------------------------------------
# Dataset opening
# ---------------------------------------------------------------------------


def _open_dataset(path: str, split):
    """Open ``path`` with the wrapper matching its recorded representation.

    Auto-detecting avoids a RuntimeError on a haploid dataset (DiploidDataset
    refuses it). Returns ``(wrapper, representation, schema_version)``."""
    rep = read_representation(path)
    if rep not in ("diploid", "haploid"):
        # e.g. "haplotype_pair" (v10, create_haplotype_dataset.py): two `hap0`/
        # `hap1` columns and no `seq`, so neither wrapper applies. Fail here
        # with the reason rather than deep inside with a missing-`seq` KeyError.
        raise ValueError(
            f"{path}: representation {rep!r} is not supported by "
            f"gather_dataset_stats.py (expected 'diploid' or 'haploid'). "
            f"v10 haplotype-pair datasets are read with "
            f"create_haplotype_dataset.HaplotypePairDataset."
        )
    cls = DiploidDataset if rep == "diploid" else HaploidDataset
    ds = cls(path, split=split)
    return ds, rep, (ds.schema_version or "unknown")


def _available_splits(path: str) -> list:
    """Return the split names of a saved ``DatasetDict``, or ``[None]``.

    ``save_to_disk`` writes ``dataset_dict.json`` listing the splits, so this is
    a cheap file probe rather than a speculative load."""
    sidecar = Path(path) / "dataset_dict.json"
    if sidecar.is_file():
        with open(sidecar, "r", encoding="utf-8") as fh:
            splits = json.load(fh).get("splits") or []
        if splits:
            return list(splits)
    return [None]


def _physical_cores():
    """Physical (non-hyperthread) core count, or None if it can't be determined."""
    try:
        import psutil  # optional; present in the project venv
        phys = psutil.cpu_count(logical=False)
        if phys:
            return phys
    except Exception:
        pass
    return None


def _default_workers() -> int:
    """Physical-core count, falling back to half the logical count.

    Same sizing rationale as ``compare_assembled._default_workers``: this scan is
    Arrow-decode/memory-bandwidth bound, and oversubscribing hyperthreads
    regresses on the 112-thread box."""
    phys = _physical_cores()
    if phys:
        return phys
    logical = os.cpu_count() or 2
    return max(1, logical // 2)


# ---------------------------------------------------------------------------
# Scan one split
# ---------------------------------------------------------------------------


def scan_split(path: str, split, table: TokenTable, workers: int,
               batch_size: int, limit) -> dict:
    """Count one split and return its merged per-sample statistics."""
    global _HF, _BATCH_SIZE, _MARKER_SYMBOLS, _HAS_DIRECTION
    _BATCH_SIZE = batch_size

    t_load0 = time.perf_counter()
    ds, rep, schema = _open_dataset(path, split)
    hf = ds.hf_dataset
    n_full = len(hf)
    if limit is not None and limit < n_full:
        hf = hf.select(range(limit))
    rows = len(hf)
    _HF = hf

    # v8 externalizes the markers out of `seq`; earlier versions keep them inline
    # (where the plain character count already covers them).
    externalized = "snp" in hf.column_names
    _HAS_DIRECTION = "direction" in hf.column_names
    if externalized:
        missing = [p for p in (_PAIR_SNP, _PAIR_INDEL_OPEN, _PAIR_INDEL_CLOSE,
                               _PAIR_DIR_FWD, _PAIR_DIR_REV)
                   if p not in table.meta_by_pair]
        if missing:
            raise ValueError(
                f"Token file lacks meta rows {missing}, needed to count the "
                f"externalized marker columns of this {schema} dataset."
            )
        _MARKER_SYMBOLS = (
            table.meta_by_pair[_PAIR_SNP],
            table.meta_by_pair[_PAIR_INDEL_OPEN],
            table.meta_by_pair[_PAIR_INDEL_CLOSE],
            table.meta_by_pair[_PAIR_DIR_FWD],
            table.meta_by_pair[_PAIR_DIR_REV],
        )
    else:
        _MARKER_SYMBOLS = None
    wall_load = time.perf_counter() - t_load0

    label = split if split is not None else "-"
    print(f"  [{datetime.now():%H:%M:%S}] {path}[{label}] schema={schema} "
          f"repr={rep} markers={'external' if externalized else 'inline'} "
          f"rows={rows} (of {n_full}{', --limit' if limit is not None else ''})",
          flush=True)

    eff_workers = max(1, min(workers, rows)) if rows else 1
    # Serial for tiny inputs: a pool would cost more than it saves.
    if rows <= batch_size * 2:
        eff_workers = 1

    t_c0 = time.perf_counter()
    try:
        if rows == 0:
            parts = []
        elif eff_workers == 1:
            parts = [_count_shard((0, 1))]
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(eff_workers) as pool:
                parts = pool.map(
                    _count_shard,
                    [(i, eff_workers) for i in range(eff_workers)],
                )
    finally:
        _HF = None
    wall_compute = time.perf_counter() - t_c0

    # Merge per-worker results.
    samples: dict = {}
    for part in parts:
        for sid, s in part["samples"].items():
            agg = samples.get(sid)
            if agg is None:
                agg = samples[sid] = {"n": 0, "w": 0, "c": Counter()}
            agg["n"] += s["n"]
            agg["w"] += s["w"]
            agg["c"].update(s["c"])

    return {
        "split": label,
        "schema_version": schema,
        "representation": rep,
        "externalized_markers": externalized,
        "n_records": sum(p["n"] for p in parts),
        "n_records_total_in_split": n_full,
        "seq_codepoints": sum(p["seq_chars"] for p in parts),
        "samples": samples,
        "workers": eff_workers,
        "wall_load_s": wall_load,
        "wall_compute_s": wall_compute,
        "parts": parts,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _ordered_symbols(table: TokenTable, observed) -> list:
    """Token-file order first, then any unrecognised symbols (sorted).

    Every known symbol is listed even at zero count, so the column set of the
    by-sample TSV is stable across datasets and directly diffable."""
    extra = sorted(s for s in observed if s not in table.symbol_class)
    return list(table.symbols) + extra


def _write_token_counts(out_dir: Path, table: TokenTable, totals: Counter,
                        symbols: list) -> None:
    """Corpus-wide non-meta symbol counts (markers get their own file)."""
    base_total = sum(c for s, c in totals.items() if not table.is_meta(s))
    path = out_dir / "dataset_token_counts.tsv"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("symbol\tunicode\ttoken_class\tcount\tfraction\n")
        for sym in symbols:
            if table.is_meta(sym):
                continue
            count = totals.get(sym, 0)
            frac = count / base_total if base_total else 0.0
            fh.write(f"{sym}\t{table.unicode_of(sym)}\t{table.class_of(sym)}\t"
                     f"{count}\t{frac:.10f}\n")


def _write_marker_counts(out_dir: Path, table: TokenTable, totals: Counter,
                         symbols: list) -> None:
    base_total = sum(c for s, c in totals.items() if not table.is_meta(s))
    path = out_dir / "dataset_marker_counts.tsv"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("marker\tunicode\tcount\tper_1k_tokens\n")
        for sym in symbols:
            if not table.is_meta(sym):
                continue
            count = totals.get(sym, 0)
            per_1k = 1000.0 * count / base_total if base_total else 0.0
            fh.write(f"{sym}\t{table.unicode_of(sym)}\t{count}\t{per_1k:.6f}\n")


def _write_by_sample(out_dir: Path, split_results: list, symbols: list) -> None:
    path = out_dir / "dataset_token_counts_by_sample.tsv"
    header = ["split", "sample_id", "n_records", "sum_sample_weight",
              "total_tokens"] + [f"token_{s}" for s in symbols]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for res in split_results:
            for sid in sorted(res["samples"]):
                s = res["samples"][sid]
                c = s["c"]
                row = [res["split"], str(sid), str(s["n"]), str(s["w"]),
                       str(sum(c.values()))]
                row.extend(str(c.get(sym, 0)) for sym in symbols)
                fh.write("\t".join(row) + "\n")


def _print_telemetry(split_results, batch_size, wall_total, parent_rss_kb):
    """Per-stage timing breakdown so optimization targets the measured hot path."""
    parts = [p for res in split_results for p in res["parts"]]
    rows = sum(res["n_records"] for res in split_results)
    iter_s = sum(p["iter_s"] for p in parts)
    count_s = sum(p["count_s"] for p in parts)
    marker_s = sum(p["marker_s"] for p in parts)
    max_child_kb = max((p["maxrss_kb"] for p in parts), default=0)
    load_s = sum(res["wall_load_s"] for res in split_results)
    compute_s = sum(res["wall_compute_s"] for res in split_results)
    rps = rows / wall_total if wall_total > 0 else 0.0
    print("==== TELEMETRY (gather_dataset_stats) ====", flush=True)
    print(f"rows={rows}  batch_size={batch_size}  "
          f"workers per split={[r['workers'] for r in split_results]}", flush=True)
    print(f"wall: total={wall_total:.2f}s  load={load_s:.2f}s  "
          f"compute={compute_s:.2f}s", flush=True)
    print("CPU-seconds across workers (overlap in wall time):", flush=True)
    print(f"  iterate/decode={iter_s:.1f}s  count={count_s:.1f}s  "
          f"markers={marker_s:.1f}s", flush=True)
    print(f"throughput: {rps:,.0f} rows/s", flush=True)
    print(f"peak RSS: parent={parent_rss_kb/1024:.0f} MB  "
          f"max worker={max_child_kb/1024:.0f} MB", flush=True)
    print("==========================================", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(args) -> int:
    t_start = time.perf_counter()
    table = load_token_table(Path(args.tokens))

    splits = [args.split] if args.split is not None else _available_splits(args.dataset)
    split_results = []
    for split in splits:
        split_results.append(
            scan_split(args.dataset, split, table, args.workers,
                       args.batch_size, args.limit)
        )

    totals: Counter = Counter()
    for res in split_results:
        for s in res["samples"].values():
            totals.update(s["c"])

    observed = set(totals)
    unknown = sorted(s for s in observed if s not in table.symbol_class)
    if unknown:
        print(f"\n  [WARN] {len(unknown)} symbol(s) present in the data but "
              f"absent from {args.tokens}: "
              f"{', '.join(repr(s) for s in unknown[:20])}"
              f"{' ...' if len(unknown) > 20 else ''}\n"
              f"         They are reported with token_class="
              f"{_UNKNOWN_CLASS!r}. This usually means the dataset was written "
              f"with a different token table.", flush=True)

    symbols = _ordered_symbols(table, observed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_token_counts(out_dir, table, totals, symbols)
    _write_marker_counts(out_dir, table, totals, symbols)
    _write_by_sample(out_dir, split_results, symbols)

    base_total = sum(c for s, c in totals.items() if not table.is_meta(s))
    marker_total = sum(c for s, c in totals.items() if table.is_meta(s))
    by_class: Counter = Counter()
    for sym, count in totals.items():
        by_class[table.class_of(sym)] += count

    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "tokens_file": str(Path(args.tokens).resolve()),
        "schema_version": split_results[0]["schema_version"] if split_results else None,
        "representation": split_results[0]["representation"] if split_results else None,
        "externalized_markers": (
            split_results[0]["externalized_markers"] if split_results else None),
        "limit": args.limit,
        "splits": [
            {
                "split": r["split"],
                "n_records": r["n_records"],
                "n_records_total_in_split": r["n_records_total_in_split"],
                "seq_codepoints": r["seq_codepoints"],
                "n_samples": len(r["samples"]),
            }
            for r in split_results
        ],
        "n_records": sum(r["n_records"] for r in split_results),
        "seq_codepoints": sum(r["seq_codepoints"] for r in split_results),
        "total_base_tokens": base_total,
        "total_markers": marker_total,
        "n_samples": len({sid for r in split_results for sid in r["samples"]}),
        "n_distinct_symbols": len(observed),
        "unknown_symbols": unknown,
        "counts_by_token_class": dict(by_class),
        "token_counts": {sym: totals[sym] for sym in symbols if totals.get(sym)},
        "wall_seconds": round(time.perf_counter() - t_start, 3),
    }
    with (out_dir / "dataset_stats_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    wall_total = time.perf_counter() - t_start
    if not args.no_telemetry:
        _print_telemetry(split_results, args.batch_size, wall_total,
                         resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    print(f"\nrecords={summary['n_records']}  samples={summary['n_samples']}  "
          f"base tokens={base_total:,}  markers={marker_total:,}  "
          f"distinct symbols={len(observed)}", flush=True)
    print(f"wrote {out_dir}/dataset_token_counts.tsv, dataset_marker_counts.tsv, "
          f"dataset_token_counts_by_sample.tsv, dataset_stats_summary.json",
          flush=True)
    return 0


def main() -> int:
    t0 = datetime.now()
    print(f"[{t0:%Y-%m-%d %H:%M:%S}] cmd: {' '.join(sys.argv)}", flush=True)

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dataset", help="path to the assembled dataset dir "
                                    "(the top-level dir holding provenance.json)")
    ap.add_argument("--output-dir", required=True,
                    help="directory for the TSV/JSON outputs (created if absent)")
    ap.add_argument("--split", default=None,
                    help="split name (train/test); default: every split present, "
                         "each reported as its own row block and aggregated in "
                         "the corpus tables")
    ap.add_argument("--tokens", default=str(DEFAULT_TOKENS_PATH),
                    help=f"path to diploid_tokens.tsv (default: {DEFAULT_TOKENS_PATH})")
    ap.add_argument("--workers", type=int, default=_default_workers(),
                    help="parallel counting workers (default: physical-core count)")
    ap.add_argument("--batch-size", type=int, default=2000,
                    help="Arrow iteration batch size (rows)")
    ap.add_argument("--limit", type=int, default=None,
                    help="count only the first N rows of each split (smoke test)")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="suppress the per-stage timing breakdown")

    args = ap.parse_args()
    try:
        rc = run(args)
    except (ValueError, RuntimeError, KeyError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", flush=True)
        rc = 2

    dt = datetime.now()
    print(f"[{dt:%Y-%m-%d %H:%M:%S}] done in {dt - t0}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
