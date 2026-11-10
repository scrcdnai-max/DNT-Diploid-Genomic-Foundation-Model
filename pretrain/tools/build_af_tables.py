#!/usr/bin/env python3
"""Build per-chrom (pos, ref, alt) -> 1000G AF lookup tables for Track C.

Reads the phased-panel BCFs, keeps biallelic-decomposed SNV records, writes one
compact .npz per chrom with sorted-by-pos arrays: pos(int32), ref(uint8 0-3),
alt(uint8 0-3), af(float32). The collator memmaps these and does searchsorted(pos)
+ ref/alt match for on-the-fly AF targets at variant token positions.

ponytail: SNV-only v1 (indel windows are masked out of Track C anyway). Indel AF
would need the indel-arrow path; add only if Track C proves worth it.

Usage:
  python tools/build_af_tables.py            # all autosomes
  CHROMS=chr22 python tools/build_af_tables.py   # one chrom (test)
"""
import os, subprocess
import numpy as np

BCF_DIR = "/data/1kgenomes/raw_bcf"
OUT_DIR = os.environ.get("AF_DIR", "/data/1kgenomes/af_tables")
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


def bcf_path(chrom):
    import glob as _glob
    std = f"{BCF_DIR}/1kGP_high_coverage_Illumina.{chrom}.filtered.SNV_INDEL_SV_phased_panel.bcf"
    if os.path.exists(std):
        return std
    # chrX (and any re-released chrom) carry a .v2 suffix etc. — match by chrom token.
    hits = [p for p in _glob.glob(f"{BCF_DIR}/*.{chrom}.filtered.*.bcf")]
    return hits[0] if hits else std


def build_one(chrom):
    bcf = bcf_path(chrom)
    if not os.path.exists(bcf):
        print(f"  SKIP {chrom}: no BCF", flush=True)
        return
    out = f"{OUT_DIR}/af_{chrom}.npz"
    # bcftools: split multiallelics, keep SNVs, emit pos/ref/alt/AF.
    cmd = ["bcftools", "norm", "-m-", "-f", "/data/diploid_glm_new/preprocess/hg38.fa", bcf] \
        if os.path.exists("/data/diploid_glm_new/preprocess/hg38.fa") else ["bcftools", "norm", "-m-", bcf]
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(
        ["bcftools", "view", "-v", "snps", "-"],
        stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(
        ["bcftools", "query", "-f", "%POS\t%REF\t%ALT\t%INFO/AF\n", "-"],
        stdin=p2.stdout, stdout=subprocess.PIPE, text=True)
    p1.stdout.close(); p2.stdout.close()
    pos, ref, alt, af = [], [], [], []
    for ln in p3.stdout:
        f = ln.rstrip("\n").split("\t")
        if len(f) != 4:
            continue
        r, a = f[1].upper(), f[2].upper()
        if r not in BASE or a not in BASE:
            continue
        try:
            v = float(f[3])
        except ValueError:
            continue
        pos.append(int(f[0])); ref.append(BASE[r]); alt.append(BASE[a]); af.append(v)
    p3.wait()
    if not pos:
        print(f"  {chrom}: 0 SNVs (?)", flush=True); return
    pos = np.asarray(pos, np.int32); ref = np.asarray(ref, np.uint8)
    alt = np.asarray(alt, np.uint8); af = np.asarray(af, np.float32)
    order = np.argsort(pos, kind="stable")
    pos, ref, alt, af = pos[order], ref[order], alt[order], af[order]
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(out, pos=pos, ref=ref, alt=alt, af=af)
    print(f"  {chrom}: {len(pos):,} SNV-AF rows -> {out}", flush=True)


def main():
    chroms = os.environ.get("CHROMS")
    chroms = chroms.split(",") if chroms else [f"chr{i}" for i in range(1, 23)]
    print(f"build_af_tables -> {OUT_DIR} for {len(chroms)} chrom(s)", flush=True)
    for c in chroms:
        build_one(c)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
