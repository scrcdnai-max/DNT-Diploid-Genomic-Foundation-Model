#!/usr/bin/env python
"""Write the token-counts table for the v2 input (``preprocess/token_counts_v2.tsv``).

Counts what the model READS: stored rows of the training set, through ``V2Rewrite``, markers
removed. The assembled dataset is shuffled, so a few thousand rows from any shards are a random
sample; only relative frequencies matter (head-bias prior, random-fill pool, and which tokens
``--drop-unseen-tokens`` keeps). Markers get count 1: the collator drops them by id, so they
must stay in the vocabulary, but they never reach the model.

    python pretrain/tools/make_v2_token_counts.py

Reference-aware tokens: ``--reference`` (HG38 FASTA) with ``--tokens preprocess/diploid_tokens_v3.tsv`` reads each
row's HG38 bases (``v2_input.row_reference``, which aligns rows with indels) and writes which copy carries the
reference into each heterozygous token, as training does with ``--v2-reference``.
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
from v2_input import MARKERS, V2Rewrite, row_reference  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", type=Path, default=Path("data/1kgenomes/diploid_hf_v7-tsstes/assembled/train"))
    p.add_argument("--tokens", type=Path, default=REPO_ROOT / "preprocess" / "diploid_tokens_v2.tsv")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "preprocess" / "token_counts_v2.tsv")
    p.add_argument("--shards", type=int, nargs="+", default=[17, 641, 1090])
    p.add_argument("--rows-per-shard", type=int, default=1700)
    p.add_argument("--reference", default=None, help="HG38 FASTA: reference-aware tokens")
    args = p.parse_args()
    rewrite = V2Rewrite(args.tokens)
    if bool(args.reference) != rewrite.reference_aware:
        raise SystemExit("--reference and a reference-aware token file (preprocess/diploid_tokens_v3.tsv) go together: "
                         f"--tokens {args.tokens} is {'' if rewrite.reference_aware else 'not '}reference-aware, "
                         f"--reference was {'' if args.reference else 'not '}given")
    fasta = None
    if args.reference:
        import pysam

        fasta = pysam.FastaFile(args.reference)
    marker_cp = np.array([ord(m) for m in MARKERS], dtype=np.uint32)
    counts: Counter[int] = Counter()
    n_rows = 0
    shards = sorted(args.train_dir.glob("*.arrow"))
    for k in args.shards:
        left = args.rows_per_shard
        for batch in pa.ipc.open_stream(pa.memory_map(str(shards[k]))):
            for i, seq in enumerate(batch.column(batch.schema.get_field_index("seq"))[:left]):
                ref = None
                if fasta is not None:
                    ref = row_reference(seq.as_py(), *(batch.column(c)[i].as_py() for c in ("chrom", "start", "end")), fasta)
                out = rewrite(np.frombuffer(seq.as_py().encode("utf-32-le"), dtype=np.uint32), ref)
                cp, n = np.unique(out[~np.isin(out, marker_cp)], return_counts=True)
                counts.update(dict(zip(cp.tolist(), n.tolist())))
                n_rows += 1
            left -= len(batch)
            if left <= 0:
                break

    lines = [
        f"# v2 input token counts: {n_rows} stored rows ({sum(counts.values())} tokens) of {args.train_dir},",
        f"# shards {args.shards}, read through pretrain/v2_input.py with markers removed"
        + (f", with the reference {args.reference}" if fasta is not None else "") + ". Markers count 1 (kept in the",
        "# vocabulary for the collator's drop). Written by pretrain/tools/make_v2_token_counts.py.",
        "pair\torder\tsymbol\tcount",
    ]
    in_file: set[int] = set()
    for ln in args.tokens.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        pair, order, sym = ln.split("\t")[:3]
        in_file.add(ord(sym))
        # every glyph the rewrite can write counts at least 1: a rare one the sample missed (C/* outside braces)
        # would otherwise be dropped from the vocabulary and stop the run when it turns up
        count = 1 if order == "meta" else max(int(sym in rewrite.emitted), counts.get(ord(sym), 0))
        lines.append(f"{pair}\t{order}\t{sym}\t{count}")
    stray = sorted(set(counts) - in_file)
    if stray:
        raise SystemExit(f"characters in the data that the token file lacks: {[chr(c) for c in stray]}")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {n_rows} rows")


if __name__ == "__main__":
    main()
