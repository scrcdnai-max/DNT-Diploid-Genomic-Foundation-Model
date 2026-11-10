"""Convert raw CADD TSVs (whole-genome) to per-chrom Arrow files (no BED filter).

Emits two files under ``--out-dir``:
  cadd_snv_<chrom>.arrow    pos:int32, ref:uint8, phred_A/C/G/T:float32
  cadd_indel_<chrom>.arrow  chrom:large_utf8, pos:int32, ref:large_utf8,
                            alt:large_utf8, phred:float32

The SNV file collapses three raw TSV rows per position (one per non-ref ALT)
into a single row with a 4-wide phred vector (NaN at the ref column and at
any missing alt). The ``Chrom`` and ``RawScore`` columns are dropped — chrom
is implicit in the filename, RawScore is redundant given PHRED.

The indel file rewrites the chrom column from CADD's raw ``22`` to the BCF
convention ``chr22`` (the value passed via ``--chrom``), so downstream
consumers can match against BCF chrom names without a strip step.

Run:
  uv run python poc_a2_convert_raw_cadd.py \\
    --snv-tsv   /opt/hpc/data/CADD/whole_genome_SNVs.tsv.gz \\
    --indel-tsv /opt/hpc/data/CADD/gnomad.genomes.r4.0.indel.tsv.gz \\
    --chrom chr22 \\
    --out-dir /opt/hpc/talz/diploid_glm_poc_annotate/s4_cadd_arrow/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pysam


_BASE_TO_COL = {ord(b): i for i, b in enumerate("ACGT")}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _strip_chr(chrom: str) -> str:
    """Raw CADD files use chrom names without the ``chr`` prefix."""
    return chrom[3:] if chrom.startswith("chr") else chrom


def convert_snv(snv_tsv: Path, chrom: str, out_path: Path) -> None:
    """Fetch chrom rows from a tabix-indexed SNV TSV and write the collapsed arrow shard."""
    chrom_no_chr = _strip_chr(chrom)
    log(f"streaming SNV tabix: {snv_tsv} chrom={chrom_no_chr}")
    t = time.time()
    # Group by (pos, ref) → list of (alt, phred).
    by_pos: Dict[Tuple[int, str], list] = {}
    n_rows = 0
    with pysam.TabixFile(str(snv_tsv)) as tbx:
        for line in tbx.fetch(chrom_no_chr):
            cols = line.split("\t")
            # Chrom Pos Ref Alt RawScore PHRED — drop cols[0] (chrom) and cols[4] (raw).
            pos = int(cols[1])
            ref = cols[2]
            alt = cols[3]
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                # Non-SNV rows shouldn't appear in whole_genome_SNVs.tsv.gz; skip
                # defensively (matches the removed POC loader poc_a2_build.py:load_snv_tabix).
                continue
            by_pos.setdefault((pos, ref), []).append((alt, float(cols[5])))
            n_rows += 1
    log(f"  parsed {n_rows:,} raw SNV rows → {len(by_pos):,} unique (pos, ref) in {time.time() - t:.1f}s")

    # Build sorted arrays.
    t = time.time()
    keys = sorted(by_pos.keys())
    n = len(keys)
    pos_arr = np.empty(n, dtype=np.int32)
    ref_arr = np.empty(n, dtype=np.uint8)
    phred_arr = np.full((n, 4), np.nan, dtype=np.float32)
    for i, (pos, ref) in enumerate(keys):
        pos_arr[i] = pos
        ref_arr[i] = ord(ref)
        for alt, phred in by_pos[(pos, ref)]:
            col = _BASE_TO_COL.get(ord(alt))
            if col is not None:
                phred_arr[i, col] = phred
    log(f"  built numpy arrays in {time.time() - t:.1f}s")

    # Write arrow IPC file.
    t = time.time()
    table = pa.table(
        {
            "pos": pa.array(pos_arr),
            "ref": pa.array(ref_arr),
            "phred_A": pa.array(phred_arr[:, 0]),
            "phred_C": pa.array(phred_arr[:, 1]),
            "phred_G": pa.array(phred_arr[:, 2]),
            "phred_T": pa.array(phred_arr[:, 3]),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(out_path), "wb") as f:
        with ipc.new_file(f, table.schema) as w:
            w.write_table(table)
    log(f"  wrote {out_path} ({os.path.getsize(out_path):,} bytes) in {time.time() - t:.1f}s")


def convert_indel(indel_tsv: Path, chrom: str, out_path: Path) -> None:
    """Fetch chrom rows from a tabix-indexed indel TSV and write a per-chrom indel arrow.

    Rewrites the chrom column from the raw CADD value (``22``) to the value
    of ``--chrom`` (``chr22``), so consumers can match BCF chrom names
    directly with no strip step."""
    chrom_no_chr = _strip_chr(chrom)
    log(f"streaming indel tabix: {indel_tsv} chrom={chrom_no_chr}")
    t = time.time()
    positions: list = []
    refs: list = []
    alts: list = []
    phreds: list = []
    with pysam.TabixFile(str(indel_tsv)) as tbx:
        for line in tbx.fetch(chrom_no_chr):
            cols = line.split("\t")
            # Chrom Pos Ref Alt RawScore PHRED — drop cols[0] (replaced) and cols[4].
            positions.append(int(cols[1]))
            refs.append(cols[2])
            alts.append(cols[3])
            phreds.append(float(cols[5]))
    n = len(positions)
    log(f"  parsed {n:,} indel rows in {time.time() - t:.1f}s")

    t = time.time()
    table = pa.table(
        {
            "chrom": pa.array([chrom] * n, type=pa.large_utf8()),
            "pos": pa.array(positions, type=pa.int32()),
            "ref": pa.array(refs, type=pa.large_utf8()),
            "alt": pa.array(alts, type=pa.large_utf8()),
            "phred": pa.array(phreds, type=pa.float32()),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(out_path), "wb") as f:
        with ipc.new_file(f, table.schema) as w:
            w.write_table(table)
    log(f"  wrote {out_path} ({os.path.getsize(out_path):,} bytes) in {time.time() - t:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snv-tsv", type=Path, required=True)
    ap.add_argument("--indel-tsv", type=Path, required=True)
    ap.add_argument("--chrom", type=str, required=True,
                    help="BCF-style name (e.g. chr22); also written to the indel arrow's chrom column.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if not args.snv_tsv.exists():
        raise FileNotFoundError(args.snv_tsv)
    if not args.indel_tsv.exists():
        raise FileNotFoundError(args.indel_tsv)

    log(f"START — cmd: {' '.join(sys.argv)}")
    t0 = time.time()

    snv_out = args.out_dir / f"cadd_snv_{args.chrom}.arrow"
    indel_out = args.out_dir / f"cadd_indel_{args.chrom}.arrow"

    convert_snv(args.snv_tsv, args.chrom, snv_out)
    convert_indel(args.indel_tsv, args.chrom, indel_out)

    log(f"END — TOTAL_s={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
