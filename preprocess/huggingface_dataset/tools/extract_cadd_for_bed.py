#!/usr/bin/env python3
"""
extract_cadd_for_bed.py — Pre-extract CADD scores restricted to a BED.

Produces three artifacts in ``--out``:

  cadd_snv_<chrom>.arrow   per-chrom Arrow IPC table with columns
                           pos:int32, ref:uint8 (ascii), phred_A/C/G/T:float32
                           (ref-slot phred is NaN). Restricted to positions
                           inside the BED regions.

  cadd_indel.arrow         entire CADD indel file in Arrow IPC. Columns
                           chrom:large_utf8, pos:int32, ref:large_utf8,
                           alt:large_utf8, phred:float32. Not BED-filtered
                           (it's small enough to keep whole and lookup is
                           by (chrom,pos,ref,alt)).

  provenance.json          sha256 of the BED, size/mtime of CADD sources,
                           commit-line + timestamp + CADD version. The
                           output is regenerated from scratch if anything
                           here changes.

Chromosome naming: the BED uses the reference convention (e.g. ``chr1``);
CADD uses the unprefixed convention (``1``). We strip ``chr`` for tabix and
add it back for reference lookups.

No silent failures: any reference-base mismatch, unknown CADD chromosome,
or unexpected TSV row raises with the offending data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from preprocess.encoder.reference import ReferenceFile  # noqa: E402
from preprocess.regions.bedio import parse_bed_regions  # noqa: E402


logger = logging.getLogger("extract_cadd_for_bed")


CADD_VERSION = "CADD-v1.7-GRCh38"

BASE_TO_COL = {ord(b): i for i, b in enumerate("ACGT")}
COL_TO_BASE = "ACGT"

SNV_HEADER_FIELDS = ("Chrom", "Pos", "Ref", "Alt", "RawScore", "PHRED")


# ---------------------------------------------------------------------------
# Provenance + idempotence
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stat_meta(path: Path) -> Dict[str, object]:
    st = os.stat(path)
    return {"path": str(path), "size": st.st_size, "mtime": int(st.st_mtime)}


def _build_provenance(
    *,
    bed_path: Path,
    snv_source: Path,
    indel_source: Path,
    reference: Path,
    out_dir: Path,
    chroms: List[str],
    cmdline: List[str],
) -> Dict[str, object]:
    return {
        "cadd_version": CADD_VERSION,
        "bed": {"path": str(bed_path), "sha256": _sha256_file(bed_path)},
        "snv_source": _stat_meta(snv_source),
        "indel_source": _stat_meta(indel_source),
        "reference": {"path": str(reference)},
        "chroms": chroms,
        "out_dir": str(out_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cmdline": cmdline,
    }


def _provenance_matches(existing: Dict[str, object], current: Dict[str, object]) -> bool:
    """Return True iff the inputs in ``existing`` match ``current`` for the
    fields that determine output validity (BED hash + source size+mtime +
    chrom list + CADD version)."""
    keys_must_match = ("cadd_version", "chroms")
    for k in keys_must_match:
        if existing.get(k) != current.get(k):
            return False
    if existing.get("bed", {}).get("sha256") != current["bed"]["sha256"]:
        return False
    for src in ("snv_source", "indel_source"):
        e = existing.get(src, {})
        c = current[src]
        if e.get("size") != c["size"] or e.get("mtime") != c["mtime"]:
            return False
    return True


# ---------------------------------------------------------------------------
# Per-chromosome SNV extraction
# ---------------------------------------------------------------------------


def _extract_snv_for_chrom(args: Tuple[str, List[Tuple[int, int]], Path, Path, Path]) -> Tuple[str, int]:
    """Worker: extract SNV CADD for one chromosome's BED regions.

    ``args`` is ``(chrom_with_chr, regions, snv_source, reference, out_dir)``.
    Returns ``(chrom_with_chr, n_rows_written)``.
    """
    chrom_with_chr, regions, snv_source, reference, out_dir = args
    chrom_cadd = chrom_with_chr[3:] if chrom_with_chr.startswith("chr") else chrom_with_chr

    # Build a per-chrom BED file CADD can consume directly (one tabix call
    # for the whole chrom instead of one per region).
    with tempfile.NamedTemporaryFile(
        "w", prefix=f"cadd_bed_{chrom_cadd}_", suffix=".bed", delete=False
    ) as tmp:
        tmp_bed = Path(tmp.name)
        for start, end in regions:
            tmp.write(f"{chrom_cadd}\t{start}\t{end}\n")

    # Preload the chrom's reference into RAM once (one .fetch per pos is too
    # slow when there are millions of positions).
    with ReferenceFile(reference) as ref:
        if chrom_with_chr not in ref.chromosome_names():
            raise ValueError(
                f"Reference {reference} does not contain chromosome "
                f"{chrom_with_chr!r}; available: {ref.chromosome_names()}"
            )
        ref_chrom_len = ref.get_reference_length(chrom_with_chr)
        ref_array = ref.fetch(chrom_with_chr, 0, ref_chrom_len, as_array=True)
    # ref_array is uint8 ascii uppercase.

    # tabix -R <bed> <snv>  → streams all rows in BED order.
    cmd = ["tabix", "-R", str(tmp_bed), str(snv_source)]
    logger.info("[%s] tabix start (%d regions)", chrom_with_chr, len(regions))
    t0 = time.time()

    # We accumulate per-position rows in growing python lists, then convert
    # to numpy at the end. Faster than per-position numpy growth.
    pos_list: List[int] = []
    ref_list: List[int] = []
    phred_rows: List[List[float]] = []  # one [pA,pC,pG,pT] row per unique pos

    current_pos: Optional[int] = None
    current_ref: Optional[int] = None
    current_row: List[float] = [float("nan")] * 4
    n_alts_seen = 0

    def flush_current():
        nonlocal n_alts_seen
        if current_pos is None:
            return
        # We expect exactly 3 ALT rows per pos in the SNV file (one per
        # non-ref base). Anything else is a data anomaly — fail loudly.
        if n_alts_seen != 3:
            raise ValueError(
                f"{chrom_cadd}:{current_pos} expected 3 ALT rows, "
                f"saw {n_alts_seen} (row {current_row})"
            )
        pos_list.append(current_pos)
        ref_list.append(current_ref)
        phred_rows.append(current_row)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for line in proc.stdout:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            # Chrom Pos Ref Alt RawScore PHRED
            if len(parts) != 6:
                raise ValueError(
                    f"unexpected CADD SNV row for {chrom_cadd}: {parts!r}"
                )
            row_chrom, pos_s, ref_s, alt_s, _raw_s, phred_s = parts
            if row_chrom != chrom_cadd:
                raise ValueError(
                    f"tabix returned wrong chrom: expected {chrom_cadd}, "
                    f"got {row_chrom} (row {parts!r})"
                )
            pos = int(pos_s)
            if len(ref_s) != 1 or len(alt_s) != 1:
                raise ValueError(
                    f"{chrom_cadd}:{pos} non-SNV row in SNV file: "
                    f"ref={ref_s} alt={alt_s}"
                )
            ref_b = ord(ref_s)
            alt_b = ord(alt_s)
            if ref_b not in BASE_TO_COL or alt_b not in BASE_TO_COL:
                raise ValueError(
                    f"{chrom_cadd}:{pos} non-ACGT base ref={ref_s} alt={alt_s}"
                )

            if pos != current_pos:
                flush_current()
                current_pos = pos
                current_ref = ref_b
                current_row = [float("nan")] * 4
                n_alts_seen = 0
                # Cross-check reference base (1-based pos → 0-based index).
                expected = ref_array[pos - 1]
                if expected != ref_b:
                    raise ValueError(
                        f"reference mismatch at {chrom_with_chr}:{pos} — "
                        f"CADD ref={chr(ref_b)} reference={chr(expected)}"
                    )
            else:
                if current_ref != ref_b:
                    raise ValueError(
                        f"{chrom_cadd}:{pos} ref base flipped within position: "
                        f"{chr(current_ref)} → {chr(ref_b)}"
                    )

            col = BASE_TO_COL[alt_b]
            if not np.isnan(current_row[col]):
                raise ValueError(
                    f"{chrom_cadd}:{pos} duplicate alt {alt_s} (already "
                    f"seen phred={current_row[col]})"
                )
            current_row[col] = float(phred_s)
            n_alts_seen += 1
        flush_current()
    finally:
        stderr_text = proc.stderr.read() if proc.stderr else ""
        proc.wait()
        try:
            tmp_bed.unlink()
        except FileNotFoundError:
            pass
    if proc.returncode != 0:
        raise RuntimeError(
            f"tabix failed for {chrom_with_chr} (rc={proc.returncode}): {stderr_text!r}"
        )

    n_rows = len(pos_list)
    if n_rows == 0:
        raise ValueError(
            f"no SNV rows extracted for {chrom_with_chr} — BED regions probably "
            f"do not overlap CADD coverage on this chrom"
        )

    pos_arr = np.asarray(pos_list, dtype=np.int32)
    ref_arr = np.asarray(ref_list, dtype=np.uint8)
    phred_arr = np.asarray(phred_rows, dtype=np.float32)  # shape (n, 4)

    # Defensive: ensure sorted by pos (tabix returns sorted, but BED regions
    # may overlap, producing duplicate-but-equal rows). De-dup on (pos).
    order = np.argsort(pos_arr, kind="stable")
    pos_arr = pos_arr[order]
    ref_arr = ref_arr[order]
    phred_arr = phred_arr[order]
    keep_mask = np.empty(n_rows, dtype=bool)
    keep_mask[0] = True
    keep_mask[1:] = pos_arr[1:] != pos_arr[:-1]
    if not keep_mask.all():
        # Dedupe — but assert the rows match (same ref, same phreds).
        dup_idx = np.where(~keep_mask)[0]
        for k in dup_idx:
            if ref_arr[k] != ref_arr[k - 1] or not np.array_equal(
                phred_arr[k], phred_arr[k - 1], equal_nan=True
            ):
                raise ValueError(
                    f"{chrom_with_chr}:{int(pos_arr[k])} duplicate position "
                    f"with conflicting CADD rows"
                )
        pos_arr = pos_arr[keep_mask]
        ref_arr = ref_arr[keep_mask]
        phred_arr = phred_arr[keep_mask]

    table = pa.table(
        {
            "pos": pa.array(pos_arr, type=pa.int32()),
            "ref": pa.array(ref_arr, type=pa.uint8()),
            "phred_A": pa.array(phred_arr[:, 0], type=pa.float32()),
            "phred_C": pa.array(phred_arr[:, 1], type=pa.float32()),
            "phred_G": pa.array(phred_arr[:, 2], type=pa.float32()),
            "phred_T": pa.array(phred_arr[:, 3], type=pa.float32()),
        }
    )

    out_path = out_dir / f"cadd_snv_{chrom_with_chr}.arrow"
    tmp_out = out_path.with_suffix(".arrow.tmp")
    with pa.OSFile(str(tmp_out), "wb") as sink:
        with ipc.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    os.replace(tmp_out, out_path)

    logger.info(
        "[%s] wrote %s (%d positions) in %.1fs",
        chrom_with_chr, out_path.name, len(pos_arr), time.time() - t0,
    )
    return chrom_with_chr, len(pos_arr)


# ---------------------------------------------------------------------------
# Indel extraction (single pass through the whole file)
# ---------------------------------------------------------------------------


def _extract_indels(
    indel_source: Path,
    out_path: Path,
    bed_intervals_cadd: Dict[str, List[Tuple[int, int]]],
) -> int:
    """Stream the gzipped indel TSV, keep only rows whose anchor (chrom, pos)
    falls inside a BED interval, write to Arrow IPC. Returns the number of
    rows written.

    ``bed_intervals_cadd`` maps CADD-chromosome name (no 'chr' prefix) to a
    list of ``(start, end)`` BED intervals (0-based half-open). Rows on
    chroms not present in this dict are dropped (chrom filter); rows whose
    anchor position is not contained in any interval on the matching chrom
    are also dropped (position filter). The BED filter cuts the gnomAD CADD
    indel file (~105 M rows) by ~20x for typical pretrain BEDs (which cover
    ~5 % of the genome), making the per-worker dict CADDAnnotation builds
    tractable at 60+ workers.
    """
    from bisect import bisect_right

    # Pre-sort intervals per chrom and pre-extract ``starts`` for bisect.
    # (BED is usually sorted but we don't rely on that.)
    sorted_intervals: Dict[str, List[Tuple[int, int]]] = {}
    sorted_starts: Dict[str, List[int]] = {}
    for chrom_cadd, ivs in bed_intervals_cadd.items():
        ivs_sorted = sorted(ivs)
        sorted_intervals[chrom_cadd] = ivs_sorted
        sorted_starts[chrom_cadd] = [s for s, _e in ivs_sorted]

    def pos_in_bed(chrom_cadd: str, pos_1based: int) -> bool:
        starts = sorted_starts.get(chrom_cadd)
        if starts is None:
            return False
        ivs = sorted_intervals[chrom_cadd]
        # 0-based pos = pos_1based - 1; contained in [start, end) iff
        # start <= pos0 < end.
        pos0 = pos_1based - 1
        idx = bisect_right(starts, pos0) - 1
        if idx < 0:
            return False
        s, e = ivs[idx]
        return s <= pos0 < e

    cmd = ["zcat", str(indel_source)]
    logger.info("indel: streaming %s (anchor must fall in BED)", indel_source)
    t0 = time.time()

    chroms: List[str] = []
    positions: List[int] = []
    refs: List[str] = []
    alts: List[str] = []
    phreds: List[float] = []

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    n_in = 0
    n_chrom_dropped = 0
    n_bed_dropped = 0
    try:
        for line in proc.stdout:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                raise ValueError(f"unexpected CADD indel row: {parts!r}")
            chrom, pos_s, ref_s, alt_s, _raw_s, phred_s = parts
            n_in += 1
            if chrom not in sorted_intervals:
                n_chrom_dropped += 1
                continue
            pos_i = int(pos_s)
            if not pos_in_bed(chrom, pos_i):
                n_bed_dropped += 1
                continue
            # Indel: ref and alt differ in length; both ACGT only.
            for s in (ref_s, alt_s):
                for ch in s:
                    if ch not in "ACGT":
                        raise ValueError(
                            f"{chrom}:{pos_s} non-ACGT in indel: "
                            f"ref={ref_s} alt={alt_s}"
                        )
            chroms.append(chrom)
            positions.append(pos_i)
            refs.append(ref_s)
            alts.append(alt_s)
            phreds.append(float(phred_s))
    finally:
        stderr_text = proc.stderr.read() if proc.stderr else ""
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"zcat failed (rc={proc.returncode}): {stderr_text!r}"
        )

    n_out = len(positions)
    table = pa.table(
        {
            "chrom": pa.array(chroms, type=pa.large_utf8()),
            "pos": pa.array(positions, type=pa.int32()),
            "ref": pa.array(refs, type=pa.large_utf8()),
            "alt": pa.array(alts, type=pa.large_utf8()),
            "phred": pa.array(phreds, type=pa.float32()),
        }
    )
    tmp_out = out_path.with_suffix(".arrow.tmp")
    with pa.OSFile(str(tmp_out), "wb") as sink:
        with ipc.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    os.replace(tmp_out, out_path)
    logger.info(
        "indel: read %d rows; dropped %d (off-chrom) + %d (off-BED); "
        "kept %d; wrote %s in %.1fs",
        n_in, n_chrom_dropped, n_bed_dropped, n_out,
        out_path.name, time.time() - t0,
    )
    return n_out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _expected_outputs(out_dir: Path, chroms: List[str]) -> List[Path]:
    return [out_dir / f"cadd_snv_{c}.arrow" for c in chroms] + [out_dir / "cadd_indel.arrow"]


def _validate_chroms(bed_chroms: List[str], cadd_chroms_cadd: set) -> List[str]:
    """For each BED chrom (chr-prefixed), confirm CADD covers the unprefixed
    form. Hard-fail with the offending list otherwise."""
    missing = []
    for c in bed_chroms:
        cadd_c = c[3:] if c.startswith("chr") else c
        if cadd_c not in cadd_chroms_cadd:
            missing.append(c)
    if missing:
        raise ValueError(
            f"BED chromosomes absent from CADD (chrM is the typical "
            f"offender; filter the BED): {missing}"
        )
    return [c for c in bed_chroms]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bed", required=True, type=Path)
    parser.add_argument("--snv-source", required=True, type=Path,
                        help="CADD whole_genome_SNVs.tsv.gz (tabix-indexed)")
    parser.add_argument("--indel-source", required=True, type=Path,
                        help="CADD gnomad.genomes.r4.0.indel.tsv.gz")
    parser.add_argument("--reference", required=True, type=Path,
                        help="Reference genome (.2bit or .fa). Must match CADD's GRCh38.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1),
                        help="Parallel chromosome workers for SNV extraction.")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if provenance.json matches the inputs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    t_start = time.time()
    logger.info("started: %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("cmdline: %s", " ".join(sys.argv))

    for p in (args.bed, args.snv_source, args.indel_source, args.reference):
        if not p.exists():
            raise FileNotFoundError(p)

    args.out.mkdir(parents=True, exist_ok=True)

    # Parse BED → ordered list of chroms in BED order.
    regions_by_chrom: Dict[str, List[Tuple[int, int]]] = parse_bed_regions(str(args.bed))
    bed_chroms_in_order: List[str] = list(regions_by_chrom.keys())

    cadd_snv_chroms = set(
        subprocess.check_output(["tabix", "-l", str(args.snv_source)], text=True).split()
    )
    cadd_indel_chroms = set(
        subprocess.check_output(["tabix", "-l", str(args.indel_source)], text=True).split()
    )
    chroms = _validate_chroms(bed_chroms_in_order, cadd_snv_chroms & cadd_indel_chroms)
    logger.info("BED has %d regions across %d chroms: %s",
                sum(len(v) for v in regions_by_chrom.values()),
                len(chroms), chroms)

    current_prov = _build_provenance(
        bed_path=args.bed,
        snv_source=args.snv_source,
        indel_source=args.indel_source,
        reference=args.reference,
        out_dir=args.out,
        chroms=chroms,
        cmdline=sys.argv,
    )
    prov_path = args.out / "provenance.json"
    expected = _expected_outputs(args.out, chroms)
    all_exist = all(p.exists() for p in expected)

    if not args.force and prov_path.exists() and all_exist:
        with open(prov_path) as f:
            existing_prov = json.load(f)
        if _provenance_matches(existing_prov, current_prov):
            logger.info("provenance matches and all outputs exist — nothing to do.")
            logger.info("total time: %.1fs", time.time() - t_start)
            return 0
        else:
            logger.info("provenance mismatch — regenerating from scratch.")

    # Wipe stale outputs so a partial previous run can't masquerade as
    # complete.
    for p in args.out.glob("cadd_snv_*.arrow"):
        p.unlink()
    for p in args.out.glob("cadd_snv_*.arrow.tmp"):
        p.unlink()
    indel_out = args.out / "cadd_indel.arrow"
    if indel_out.exists():
        indel_out.unlink()

    # Parallel per-chrom SNV extraction.
    snv_tasks = [
        (chrom, regions_by_chrom[chrom], args.snv_source, args.reference, args.out)
        for chrom in chroms
    ]
    n_workers = max(1, min(args.workers, len(snv_tasks)))
    logger.info("SNV: %d chroms × workers=%d", len(snv_tasks), n_workers)
    if n_workers == 1:
        for task in snv_tasks:
            _extract_snv_for_chrom(task)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for chrom, n in pool.imap_unordered(_extract_snv_for_chrom, snv_tasks):
                logger.info("[%s] done (%d positions)", chrom, n)

    # Indel pass (single-threaded). BED-filtered on anchor pos so the
    # per-worker dict CADDAnnotation builds stays small enough to scale
    # to many workers.
    bed_intervals_cadd: Dict[str, List[Tuple[int, int]]] = {
        (c[3:] if c.startswith("chr") else c): list(regions_by_chrom[c])
        for c in chroms
    }
    _extract_indels(args.indel_source, indel_out, bed_intervals_cadd)

    with open(prov_path, "w") as f:
        json.dump(current_prov, f, indent=2)

    elapsed = time.time() - t_start
    logger.info("done: wrote %d SNV shards + cadd_indel.arrow to %s",
                len(chroms), args.out)
    logger.info("ended: %s (total %.1fs)",
                datetime.now().isoformat(timespec="seconds"), elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
