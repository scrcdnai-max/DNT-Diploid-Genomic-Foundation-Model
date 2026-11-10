#!/usr/bin/env python3
"""Order-independent content comparison of two assembled diploid datasets.

Two `assemble` runs differ at the byte level (sha256) because of shuffle order,
shard boundaries, and Arrow/dict-encoding framing -- even when the *content* is
identical. This compares them as multisets of canonical per-row hashes instead.

The datasets typically live on separate machines and are multi-TB, so the design
is: compute a small *fingerprint* locally on each machine, transfer only those,
then `compare` them. Do NOT copy arrow shards: `assemble` shuffles before
sharding, so shard k on machine A holds different rows than shard k on machine B
-- a single-shard sample lines up with nothing and proves nothing.

This tool only compares datasets of the **same schema version** (e.g. both v6,
both v8). It hashes the raw stored columns directly, and the on-disk column set /
encoding differ across versions, so a cross-version compare is meaningless --
`compare` refuses it as its very first check (the schema/representation/hash
recorded in each fingerprint header must match).

Workflow:
    # on machine A:
    python -m preprocess.huggingface_dataset.tools.compare_assembled fingerprint /path/to/ds_a --split train -o a.fp
    # on machine B:
    python -m preprocess.huggingface_dataset.tools.compare_assembled fingerprint /path/to/ds_b --split train -o b.fp
    # copy the small .fp files to one host, then:
    python -m preprocess.huggingface_dataset.tools.compare_assembled compare a.fp b.fp

`fingerprint` is parallel (one worker per contiguous Arrow shard) and reports a
per-stage telemetry breakdown so the cost of each phase is measured, not guessed.
Use `--limit N` for a quick smoke test on the first N rows, and `--workers` to
size the pool (default: physical-core count).

A .fp file is a header line
    #diploid-fp v2 rows=<N> schema=<v8> repr=<diploid> hash=xxh128 combined=<sha256>
followed by the SORTED per-row xxh128 hashes, one per line. The combined digest
alone settles equality; the per-row hashes let `compare` localize differences.
`--digest-only` writes just the header (no body) for a minimal transfer.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import multiprocessing as mp
import os
import resource
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import xxhash

from preprocess.huggingface_dataset.dataset import (  # noqa: E402
    DiploidDataset,
    HaploidDataset,
    read_representation,
)

# .fp format/version markers. Bump HEADER_VERSION if the hashing scheme changes;
# `compare` refuses to mix versions (an older v1 file used sha256 per row).
_HEADER_PREFIX = "#diploid-fp"
_HEADER_VERSION = "v2"
_HASH_NAME = "xxh128"

# Field separators folded into the per-row payload. Fixed bytes that cannot occur
# inside a column name and are vanishingly unlikely inside a value; they keep the
# concatenation of (name, value) pairs unambiguous.
_KV_SEP = b"\x00"
_FIELD_SEP = b"\x1e"

# ---------------------------------------------------------------------------
# Canonical per-row hashing (raw stored columns, same-schema only)
# ---------------------------------------------------------------------------

def _canon_value(v) -> bytes:
    """Deterministic UTF-8 encoding of one stored column value.

    The assembled schema carries only strings (sample_id, chrom, seq), ints
    (start, end, sample_weight) and, on v7/v8, list / list-of-struct columns
    (snp, indel_span, direction, phylop, cadd_phred). ``repr`` of a Python list
    / list-of-dict is deterministic across machines (Arrow decodes to lists and
    dicts with a stable field order), so it is a safe canonical form. There are
    no float columns, so there is no float-formatting nondeterminism.
    """
    if v is None:
        return b""
    if type(v) is str:
        return v.encode("utf-8")
    if type(v) is bool:  # bool is an int subclass; pin it explicitly
        return b"1" if v else b"0"
    if type(v) is int:
        return str(v).encode("ascii")
    return repr(v).encode("utf-8")


def _row_hash(cols_b, colvals, j) -> str:
    """xxh128 hex of row ``j``, folding every column in fixed (sorted) order.

    ``cols_b`` are the pre-encoded column-name bytes (sorted); ``colvals`` the
    parallel list of per-column value lists for the current batch. Streaming the
    hasher avoids materializing a big per-row payload string (``seq`` is ~7 kbp).
    """
    h = xxhash.xxh3_128()
    for k in range(len(cols_b)):
        h.update(cols_b[k])
        h.update(_KV_SEP)
        h.update(_canon_value(colvals[k][j]))
        h.update(_FIELD_SEP)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Parallel fingerprint workers (fork-inherited globals; no per-worker reload)
# ---------------------------------------------------------------------------

# Set in the parent before the pool forks; children inherit them via copy-on-write.
_HF = None              # the (possibly --limit-sliced) HuggingFace Dataset
_TMPDIR = None          # scratch dir for per-worker part files
_BATCH_SIZE = 2000


def _hash_shard(task) -> dict:
    """Hash one contiguous shard; write its sorted hashes to a part file.

    Returns a telemetry dict (row count, per-stage seconds, peak RSS, part path).
    Stages are timed per *batch* (two perf_counter calls per ~2000 rows) so the
    instrumentation cost is negligible while still separating Arrow decode from
    hashing.
    """
    shard_index, num_shards = task
    sub = _HF.shard(num_shards=num_shards, index=shard_index, contiguous=True)

    # Fixed, sorted column order (identical on both machines for a given schema).
    cols = sorted(sub.column_names)
    cols_b = [c.encode("utf-8") for c in cols]

    iter_s = hash_s = 0.0
    n = 0
    hashes: list = []
    it = sub.iter(batch_size=_BATCH_SIZE)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        iter_s += t1 - t0

        colvals = [batch[c] for c in cols]
        bn = len(colvals[0])
        for j in range(bn):
            hashes.append(_row_hash(cols_b, colvals, j))
        hash_s += time.perf_counter() - t1
        n += bn

    t = time.perf_counter()
    hashes.sort()
    sort_s = time.perf_counter() - t

    t = time.perf_counter()
    part = os.path.join(_TMPDIR, f"part-{shard_index:05d}.txt")
    with open(part, "w") as fh:
        fh.write("\n".join(hashes))
        if hashes:
            fh.write("\n")
    write_s = time.perf_counter() - t

    return {
        "part": part, "n": n,
        "iter_s": iter_s, "hash_s": hash_s, "sort_s": sort_s, "write_s": write_s,
        "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------

def _open_dataset(path: str, split):
    """Open ``path`` with the wrapper matching its recorded representation.

    Auto-detecting avoids a RuntimeError when fingerprinting a haploid dataset
    (DiploidDataset refuses it). Returns (wrapper, representation, schema)."""
    rep = read_representation(path)
    if rep not in ("diploid", "haploid"):
        # e.g. "haplotype_pair" (v10, create_haplotype_dataset.py): two `hap0`/
        # `hap1` columns and no `seq`, so neither wrapper applies. Fail here
        # with the reason rather than deep inside with a missing-`seq` KeyError.
        raise ValueError(
            f"{path}: representation {rep!r} is not supported by "
            f"compare_assembled.py (expected 'diploid' or 'haploid'). "
            f"v10 haplotype-pair datasets are read with "
            f"create_haplotype_dataset.HaplotypePairDataset."
        )
    cls = DiploidDataset if rep == "diploid" else HaploidDataset
    ds = cls(path, split=split)
    return ds, rep, (ds.schema_version or "unknown")


def _print_telemetry(rows, workers, batch_size, wall_total, wall_load,
                     wall_compute, wall_merge, parts_stats, parent_rss_kb):
    """Per-stage timing breakdown so optimization targets the measured hot path."""
    iter_s = sum(p["iter_s"] for p in parts_stats)
    hash_s = sum(p["hash_s"] for p in parts_stats)
    sort_s = sum(p["sort_s"] for p in parts_stats)
    write_s = sum(p["write_s"] for p in parts_stats)
    max_child_kb = max((p["maxrss_kb"] for p in parts_stats), default=0)
    rps = rows / wall_total if wall_total > 0 else 0.0
    print("==== TELEMETRY (fingerprint) ====", flush=True)
    print(f"rows={rows}  workers={workers}  batch_size={batch_size}", flush=True)
    print(f"wall: total={wall_total:.2f}s  load={wall_load:.2f}s  "
          f"compute={wall_compute:.2f}s  merge+write={wall_merge:.2f}s", flush=True)
    print("CPU-seconds across workers (overlap in wall time):", flush=True)
    print(f"  iterate/decode={iter_s:.1f}s  hash={hash_s:.1f}s  "
          f"sort={sort_s:.1f}s  write={write_s:.1f}s", flush=True)
    print(f"throughput: {rps:,.0f} rows/s", flush=True)
    print(f"peak RSS: parent={parent_rss_kb/1024:.0f} MB  "
          f"max worker={max_child_kb/1024:.0f} MB", flush=True)
    print("=================================", flush=True)


def cmd_fingerprint(args) -> int:
    global _HF, _TMPDIR, _BATCH_SIZE
    _BATCH_SIZE = args.batch_size

    t_load0 = time.perf_counter()
    ds, rep, schema = _open_dataset(args.dataset, args.split)
    hf = ds.hf_dataset
    n_full = len(hf)
    if args.limit is not None and args.limit < n_full:
        hf = hf.select(range(args.limit))
    rows = len(hf)
    _HF = hf
    wall_load = time.perf_counter() - t_load0

    print(f"  [{datetime.now():%H:%M:%S}] {args.dataset}[{args.split}] "
          f"schema={schema} repr={rep} rows={rows} "
          f"(of {n_full}{' , --limit' if args.limit is not None else ''})", flush=True)

    phys = _physical_cores()
    if phys and args.workers > phys:
        print(f"  [warn] --workers={args.workers} exceeds physical cores ({phys}); "
              f"hashing is memory-bandwidth bound, so oversubscribing tends to "
              f"REGRESS (measured: 112 workers slower than 56 on this box).",
              flush=True)

    workers = max(1, min(args.workers, rows)) if rows else 1
    # Serial for tiny inputs: a pool would cost more than it saves.
    if rows <= _BATCH_SIZE * 2:
        workers = 1
    num_shards = workers

    out = Path(args.output)
    _TMPDIR = tempfile.mkdtemp(prefix=".fp_parts_", dir=str(out.resolve().parent))
    try:
        t_c0 = time.perf_counter()
        if rows == 0:
            parts_stats = []
        elif workers == 1:
            parts_stats = [_hash_shard((0, 1))]
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(workers) as pool:
                parts_stats = pool.map(_hash_shard, [(i, num_shards) for i in range(num_shards)])
        wall_compute = time.perf_counter() - t_c0

        n_total = sum(p["n"] for p in parts_stats)

        # Merge the per-worker sorted part files into one globally sorted stream,
        # feeding the combined digest and (unless --digest-only) the body, in a
        # single memory-light pass.
        t_m0 = time.perf_counter()
        combined = hashlib.sha256()
        body_tmp = os.path.join(_TMPDIR, "body.txt")
        readers = [open(p["part"]) for p in parts_stats]
        try:
            merged = heapq.merge(*(( _l.rstrip("\n") for _l in r) for r in readers))
            with open(body_tmp, "w") as bf:
                for hexd in merged:
                    combined.update(hexd.encode("ascii"))
                    combined.update(b"\n")
                    if not args.digest_only:
                        bf.write(hexd)
                        bf.write("\n")
        finally:
            for r in readers:
                r.close()
        combined_hex = combined.hexdigest()

        with out.open("w") as fh:
            fh.write(f"{_HEADER_PREFIX} {_HEADER_VERSION} rows={n_total} "
                     f"schema={schema} repr={rep} hash={_HASH_NAME} "
                     f"combined={combined_hex}\n")
            if not args.digest_only:
                with open(body_tmp) as bf:
                    shutil.copyfileobj(bf, fh)
        wall_merge = time.perf_counter() - t_m0
    finally:
        shutil.rmtree(_TMPDIR, ignore_errors=True)
        _TMPDIR = None
        _HF = None

    wall_total = time.perf_counter() - t_load0
    parent_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not args.no_telemetry:
        _print_telemetry(n_total, workers, _BATCH_SIZE, wall_total, wall_load,
                         wall_compute, wall_merge, parts_stats, parent_rss)
    print(f"\nwrote {out}  rows={n_total}  schema={schema}  repr={rep}  "
          f"combined={combined_hex}", flush=True)
    print("transfer this file to the comparison host.", flush=True)
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def _parse_header(line: str, path: str) -> dict:
    """Parse a .fp header line into a dict; raise on a non-v2 / malformed file."""
    line = line.rstrip("\n")
    if not line.startswith(_HEADER_PREFIX):
        raise ValueError(f"{path}: not a fingerprint file (bad header).")
    toks = line.split()
    # toks[0] == _HEADER_PREFIX, toks[1] == version
    version = toks[1] if len(toks) > 1 else "?"
    if version != _HEADER_VERSION:
        raise ValueError(
            f"{path}: fingerprint format {version!r} != {_HEADER_VERSION!r}. "
            f"Regenerate it with the current compare_assembled.py "
            f"(v1 used sha256 per row and is not comparable)."
        )
    hdr = {"version": version, "schema": None, "repr": None,
           "hash": None, "rows": None, "combined": None}
    for tok in toks[2:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            if k == "rows":
                hdr["rows"] = int(v)
            elif k in hdr:
                hdr[k] = v
    return hdr


def _read_header(path: str):
    """Return (header_dict, has_body): peek the first two lines only."""
    with open(path) as fh:
        first = fh.readline()
        if not first:
            raise ValueError(f"{path}: empty file (not a fingerprint).")
        hdr = _parse_header(first, path)
        has_body = bool(fh.readline().strip())
    return hdr, has_body


def _iter_body(fh):
    """Yield stripped non-empty hash lines from an already-header-skipped file."""
    for line in fh:
        line = line.strip()
        if line:
            yield line


def _check_compatible(ha: dict, hb: dict, pa: str, pb: str) -> None:
    """Raise ValueError unless the two fingerprints are same-schema comparable.

    This is the FIRST thing compare does: a cross-schema (or cross-representation,
    or cross-hash) comparison is meaningless because the raw column set/encoding
    differs, so we refuse it before reading any content."""
    for field, human in (("schema", "schema version"),
                         ("repr", "representation"),
                         ("hash", "hash algorithm")):
        va, vb = ha.get(field), hb.get(field)
        if va != vb:
            raise ValueError(
                f"{human} mismatch: A({pa})={va!r} vs B({pb})={vb!r}. "
                f"compare_assembled only compares datasets of the same {human}; "
                f"re-generate both fingerprints from same-{field} datasets."
            )


def cmd_compare(args) -> int:
    try:
        ha, body_a = _read_header(args.fp_a)
        hb, body_b = _read_header(args.fp_b)
    except (ValueError, FileNotFoundError) as exc:
        print("==== RESULT ====")
        print(f"\nERROR: {exc}")
        return 2

    # --- GUARD FIRST: refuse incompatible schemas before any content work. ---
    try:
        _check_compatible(ha, hb, args.fp_a, args.fp_b)
    except ValueError as exc:
        print("==== RESULT ====")
        print(f"A: {args.fp_a}  schema={ha.get('schema')} repr={ha.get('repr')}")
        print(f"B: {args.fp_b}  schema={hb.get('schema')} repr={hb.get('repr')}")
        print(f"\nERROR: {exc}")
        return 2

    ca, cb = ha["combined"], hb["combined"]
    ra, rb = ha["rows"], hb["rows"]
    print("==== RESULT ====")
    print(f"A: rows={ra} schema={ha['schema']} combined={ca}")
    print(f"B: rows={rb} schema={hb['schema']} combined={cb}")

    # Fast path: combined digests settle equality without the per-row bodies.
    if ca == cb and ra == rb:
        print("\nIDENTICAL CONTENT (combined digests match).")
        return 0
    if not body_a or not body_b:
        print("\nCONTENT DIFFERS (combined digests differ; re-run fingerprint "
              "without --digest-only on both sides to localize).")
        return 1

    # Streaming merge-join over the two SORTED bodies: O(total) time, O(1) memory.
    only_a = only_b = matched = 0
    ex_a: list = []
    ex_b: list = []
    with open(args.fp_a) as fa, open(args.fp_b) as fb:
        fa.readline()  # skip header
        fb.readline()
        ia, ib = _iter_body(fa), _iter_body(fb)
        x = next(ia, None)
        y = next(ib, None)
        while x is not None and y is not None:
            if x == y:
                matched += 1
                x = next(ia, None)
                y = next(ib, None)
            elif x < y:
                only_a += 1
                if len(ex_a) < 5:
                    ex_a.append(x)
                x = next(ia, None)
            else:
                only_b += 1
                if len(ex_b) < 5:
                    ex_b.append(y)
                y = next(ib, None)
        while x is not None:
            only_a += 1
            if len(ex_a) < 5:
                ex_a.append(x)
            x = next(ia, None)
        while y is not None:
            only_b += 1
            if len(ex_b) < 5:
                ex_b.append(y)
            y = next(ib, None)

    print(f"matched rows: {matched}")
    print(f"rows only in A: {only_a}")
    print(f"rows only in B: {only_b}")

    if only_a == 0 and only_b == 0:
        print("\nIDENTICAL CONTENT (multisets match; byte differences are "
              "seed/framing only).")
        return 0

    print("\nCONTENT DIFFERS.")
    for label, ex in (("A-only", ex_a), ("B-only", ex_b)):
        for hh in ex:
            print(f"  {label}: {hh}")
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = datetime.now()
    print(f"[{t0:%Y-%m-%d %H:%M:%S}] cmd: {' '.join(sys.argv)}", flush=True)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    fp = sub.add_parser("fingerprint", help="hash one dataset locally -> .fp file")
    fp.add_argument("dataset", help="path to the assembled dataset dir "
                                    "(the top-level dir holding provenance.json)")
    fp.add_argument("-o", "--output", required=True)
    fp.add_argument("--split", default=None,
                    help="split name (train/test) if the dir is a DatasetDict")
    fp.add_argument("--digest-only", action="store_true",
                    help="write only the combined digest (KB), not per-row hashes")
    fp.add_argument("--workers", type=int, default=_default_workers(),
                    help="parallel hashing workers (default: physical-core count)")
    fp.add_argument("--batch-size", type=int, default=2000,
                    help="Arrow iteration batch size (rows)")
    fp.add_argument("--limit", type=int, default=None,
                    help="hash only the first N rows (smoke test)")
    fp.add_argument("--no-telemetry", action="store_true",
                    help="suppress the per-stage timing breakdown")
    fp.set_defaults(func=cmd_fingerprint)

    cmp = sub.add_parser("compare", help="diff two .fp files (same schema only)")
    cmp.add_argument("fp_a")
    cmp.add_argument("fp_b")
    cmp.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    try:
        rc = args.func(args)
    except (ValueError, RuntimeError, KeyError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", flush=True)
        rc = 2

    dt = datetime.now()
    print(f"[{dt:%Y-%m-%d %H:%M:%S}] done in {dt - t0}", flush=True)
    return rc


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
    """Physical-core count (the measured sweet spot), falling back to half the
    logical count when the physical count is unknown.

    Fingerprinting is memory-bandwidth bound, not CPU bound: a worker sweep on
    the 112-thread box plateaued by ~28-56 workers and *regressed* at 112
    (hyperthread + cross-NUMA contention)."""
    phys = _physical_cores()
    if phys:
        return phys
    logical = os.cpu_count() or 2
    return max(1, logical // 2)


if __name__ == "__main__":
    sys.exit(main())
