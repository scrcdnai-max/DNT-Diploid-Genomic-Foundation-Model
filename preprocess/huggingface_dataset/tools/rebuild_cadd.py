"""Rebuild the ``cadd_phred`` column of an assembled split with the current CADD annotation code.

Datasets built before 2026-10-14 store CADD rounded to whole numbers (uint8) and averaged where the two copies
carry different alternates. This tool reruns the encoder for every stored row (its ``sample_id``, its region),
checks that the rebuilt sequence equals the stored one, and writes the current scores to ``<out>/<shard name>``:
an Arrow IPC file with one column, ``cadd_phred``, row for row with the shard; its schema metadata
(``rebuild_cadd``) names the shard and holds the shard's counts. A row whose rebuilt sequence differs (the encoder
changed since the dataset was built), or whose rebuild raises, keeps its stored scores: they index the stored
``seq``. The run stops if more than 5% of the rows rebuilt so far differ (a wrong ``--version``), and fails loudly if a
worker dies. Reference rows (``HG38_REF``) carry no variant and get an empty
list. With ``--keep-rows`` (a training keep-list: global row numbers in sorted-shard order) only those rows are
rebuilt and every other row keeps what it stores. Shards already written are skipped, so a stopped run resumes.

Inline-marker presets only (v7 family): under v8 the scores index the marker-free sequence and the markers live in
separate columns, which this tool does not rebuild.

    python -m preprocess.huggingface_dataset.tools.rebuild_cadd --split-dir <assembled>/train --out <dir> \\
        --bcf-dir <per-sample BCFs> --ref <hg38.2bit> --tokens preprocess/diploid_tokens.tsv \\
        --cadd-arrow-dir /storage/data/CADD/full_arrow --version v7-tsstes --workers 64 [--keep-rows keep.npy]
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa

from preprocess.encoder.get_genome_sequence import GenomeSequenceProcessor
from preprocess.huggingface_dataset.annotations import CADDAnnotation
from preprocess.huggingface_dataset.versions import PRESETS

logger = logging.getLogger(__name__)
REF_SAMPLE_ID = "HG38_REF"
COUNTS = ("rows", "rebuilt", "sequence_differs", "failed", "rows_with_scores")
MAX_DIFFERS, CHECK_AFTER_ROWS = 0.05, 1000      # real builds: 1.2-1.6% differ; a wrong preset: all of them
_JOB: dict = {}  # set in the parent before the pool forks: processor, CADD tables, paths, marker policy, keep mask


def _rebuild_shard(job: tuple[str, int]) -> Counter:
    shard, first_row = job
    out = _JOB["out"] / Path(shard).name
    if out.exists():
        return Counter()
    with pa.memory_map(shard) as src:
        tbl = pa.ipc.open_stream(src).read_all()
    keep = _JOB["keep"]
    rows = range(tbl.num_rows) if keep is None else np.flatnonzero(keep[first_row:first_row + tbl.num_rows])
    samples, chroms = tbl.column("sample_id"), tbl.column("chrom")
    starts, ends, seqs, stored = tbl.column("start"), tbl.column("end"), tbl.column("seq"), tbl.column("cadd_phred")
    new: list = [None] * tbl.num_rows  # None: keep what is stored
    tally = Counter(rows=tbl.num_rows, rebuilt=len(rows))
    rows_of = defaultdict(list)
    for i in rows:
        rows_of[samples[int(i)].as_py()].append(int(i))
    cadd = _JOB["cadd"]
    for sample, idx in rows_of.items():
        if sample == REF_SAMPLE_ID:
            for i in idx:
                new[i] = []
            continue
        regions = [(chroms[i].as_py(), starts[i].as_py(), ends[i].as_py()) for i in idx]
        try:
            results = _JOB["processor"].get_marked_sequence_with_ref_idx_and_variants_parallel_str(
                regions=regions, bcf_dir=str(_JOB["bcf_dir"]), sample_name=sample, skip_gaps=True, max_workers=1)
        except Exception as exc:  # noqa: BLE001 - one sample's failure must not stop the other rows
            logger.warning("%s: encoder failed for %s (%d rows): %s", Path(shard).name, sample, len(idx), exc)
            results = [None] * len(idx)
        for i, region, res in zip(idx, regions, results, strict=True):
            if res is None:
                tally["failed"] += 1
                continue
            seq, ref_idx, var_idx, variants = res
            if seq != seqs[i].as_py():
                tally["sequence_differs"] += 1
                continue
            try:
                new[i] = cadd.encode(cadd.compute(*region, ref_idx, var_idx, variants, _JOB["marker_policy"]))
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: CADD failed for %s %s: %s", Path(shard).name, sample, region, exc)
                tally["failed"] += 1
                continue
            tally["rows_with_scores"] += bool(new[i])
    col = pa.array([stored[i].as_py() if v is None else v for i, v in enumerate(new)], type=cadd.field.type)
    meta = {"shard": Path(shard).name, **{k: tally[k] for k in COUNTS}, **_JOB["provenance"]}
    schema = pa.schema([pa.field("cadd_phred", col.type)], metadata={"rebuild_cadd": json.dumps(meta)})
    tmp = out.with_suffix(".tmp")
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, schema) as w:
        w.write_table(pa.table({"cadd_phred": col}, schema=schema))
    tmp.rename(out)
    return tally


def rebuild(split_dir, out, *, bcf_dir, ref, tokens, cadd_arrow_dir, version, workers, keep_rows=None) -> dict:
    """Rebuild every shard of ``split_dir`` into ``out``; returns the counts over the shards rebuilt in this call
    (each shard's own counts are in its output's schema metadata)."""
    preset = PRESETS[version]
    if preset.externalize_markers:
        raise ValueError(f"{version}: externalized markers (v8) are not supported; rebuild the v7 dataset")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shards = [str(p) for p in sorted(Path(split_dir).glob("*.arrow"))  # the trainer's order (load_parquet_dataset)
              if not p.name.endswith(".tmp") and not p.name.startswith("cache-")]
    sizes = []
    for s in shards:
        with pa.memory_map(s) as src:
            sizes.append(sum(b.num_rows for b in pa.ipc.open_stream(src)))
    firsts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int).tolist()
    keep = None
    if keep_rows is not None:
        keep = np.zeros(sum(sizes), dtype=bool)
        keep[np.asarray(keep_rows, dtype=np.int64)] = True
    cadd = CADDAnnotation(str(cadd_arrow_dir))
    cadd.open()  # the tables load lazily in each worker; the object is shared by fork, like the processor
    _JOB.update(
        out=out, bcf_dir=Path(bcf_dir), cadd=cadd, marker_policy=preset.marker_policy,
        keep=keep,
        provenance={"split_dir": str(split_dir), "version": version,
                    "keep_rows": "all" if keep_rows is None else len(keep_rows), **cadd.provenance()},
        processor=GenomeSequenceProcessor(  # built once; the workers share it by fork
            ref_path=str(ref), diploid_tokens_path=str(tokens), force_unphased=preset.force_unphased,
            use_direction_markers=preset.use_direction_markers, emit_anchor_snp_marker=preset.emit_anchor_snp_marker),
    )
    total = Counter()
    jobs = list(zip(shards, firsts))
    # _JOB reaches the workers only by fork; a killed worker raises BrokenProcessPool (Pool would wait forever)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for n, f in enumerate(as_completed([pool.submit(_rebuild_shard, j) for j in jobs]), 1):
            total += f.result()
            print(f"{n}/{len(jobs)} shards  {dict(total)}", flush=True)
            if total["rebuilt"] >= CHECK_AFTER_ROWS and total["sequence_differs"] > MAX_DIFFERS * total["rebuilt"]:
                pool.shutdown(cancel_futures=True)
                raise SystemExit(f"{total['sequence_differs']} of {total['rebuilt']} rebuilt rows differ from the "
                                 f"stored seq: is {version} the preset {split_dir} was built with?")
    return {k: total[k] for k in COUNTS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bcf-dir", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--cadd-arrow-dir", type=Path, required=True)
    p.add_argument("--version", required=True, choices=sorted(PRESETS))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--keep-rows", type=Path, default=None, help="rebuild only these global rows (.npy)")
    a = p.parse_args()
    report = rebuild(a.split_dir, a.out, bcf_dir=a.bcf_dir, ref=a.ref, tokens=a.tokens,
                     cadd_arrow_dir=a.cadd_arrow_dir, version=a.version, workers=a.workers,
                     keep_rows=None if a.keep_rows is None else np.load(a.keep_rows))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
