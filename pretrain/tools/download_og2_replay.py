#!/usr/bin/env python
"""Build the bounded OpenGenome2 replay pool: download shards to a bp quota, then report.

The replay experiment mixes ~20% of NTv3's original OpenGenome2 distribution back into
diploid continued pretraining. This tool assembles a ~100B-bp local subset of that
distribution -- generous on purpose: even a long run at 20% replay consumes only a few
billion OG2 tokens, so the pool exists to provide diversity, not volume.

Source quotas approximate NTv3's *eligible* corpus composition (its <=8kb phase excluded
whole eukaryotic genomes and sub-1kb sequences), NOT Evo2's dataloader reweighting -- those
are a different, deliberate reweighting and using them here would change the variable this
experiment is trying to isolate.

Two subcommands:

    download   pull shards until each source hits its bp quota (resumable, idempotent)
    stats      merge the per-shard scans into stats/summary.json + stats/README.md

Usage:
    python pretrain/tools/download_og2_replay.py download --dry-run
    python pretrain/tools/download_og2_replay.py download
    python pretrain/tools/download_og2_replay.py stats
    python pretrain/tools/download_og2_replay.py download --source metagenomes --extra-bp 10e9
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from og2_replay import scan_shard, shard_seed, trim_shard  # noqa: E402

logger = logging.getLogger("og2_download")

REPO_ID = "arcinstitute/opengenome2"
# Pinned: the pool must stay re-derivable from the manifest alone.
REVISION = "9a24315bdc54a9949f2d8e0eca6c718134815a11"
ROOT = "json/pretraining_or_both_phases"
SEED = 42
DEFAULT_OUT = Path("data/opengenome2")

# source -> (usable-bp quota, bp per compressed byte).
# Quotas are the dataset card's phase-1 composition (854B metagenomic, 405B eukaryotic
# genic, 351B GTDB/IMG-PR, 99B stitched mRNA, 89B other mRNA, 34B IMG/VR, 19B ncRNA, 3B
# organellar) scaled to ~100B. The ratio is card-tokens/listed-bytes per source and is used
# only to print a --dry-run estimate; every real quota decision uses a measured count.
# EPDnew promoters (0.11B bp) are skipped -- too small to matter, and NTv3's promoter
# handling is a phase-2 concern.
SOURCES: dict[str, tuple[float, float]] = {
    "metagenomes": (46e9, 3.45),
    "eukaryotic_genic_windows": (22e9, 3.15),
    "gtdb_v220_imgpr": (19e9, 3.43),
    "mrna_splice_promoter": (5e9, 1.72),
    "mrna": (5e9, 2.65),
    "imgvr_untagged": (2e9, 3.26),
    "ncrna": (1e9, 2.05),
    "organelle": (0.3e9, 1.85),
}

# NTv3's token shares across its four short-context length bins.
TARGET_BIN_SHARE = {"1024": 7.9, "2048": 13.0, "4096": 13.2, "8192": 66.0}

MANIFEST_FIELDS = [
    "hf_revision", "source", "hf_path", "local_path",
    "compressed_bytes", "usable_bp", "trim_p", "sha256",
]


def list_train_shards(source: str) -> list[tuple[str, int]]:
    """``(repo path, size)`` of a source's train shards. Filenames carry the split, and the
    prefix is not derivable from the directory name, so enumerate rather than template."""
    from huggingface_hub import HfApi

    tree = HfApi().list_repo_tree(
        REPO_ID, path_in_repo=f"{ROOT}/{source}", repo_type="dataset",
        revision=REVISION, recursive=True,
    )
    shards = [
        (e.path, e.size) for e in tree
        if getattr(e, "size", None) is not None
        and e.path.endswith(".jsonl.gz") and "_train_" in Path(e.path).name
    ]
    return sorted(shards)


def shuffled_shards(source: str) -> list[tuple[str, int]]:
    """Source's train shards in seeded random order -- never the first N, which would be a
    handful of assemblies rather than a sample of the source."""
    shards = list_train_shards(source)
    random.Random(shard_seed(SEED, source)).shuffle(shards)
    return shards


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        if new:
            w.writeheader()
        w.writerow(row)


def write_scan(out: Path, source: str, local_path: Path, stats: dict) -> None:
    """Per-shard scan sidecar, so `stats` merges cheap JSON instead of re-reading the pool."""
    d = out / "stats" / "shards"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{local_path.name}.json").write_text(
        json.dumps({"source": source, "local_path": str(local_path), **stats}, indent=1)
    )


def cmd_download(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    out = Path(args.out)
    manifest = out / "manifests" / "shards.tsv"
    raw = out / "raw"
    rows = read_manifest(manifest)
    sources = [args.source] if args.source else list(SOURCES)

    for source in sources:
        quota, bp_per_byte = SOURCES[source]
        quota += args.extra_bp
        done = {r["hf_path"] for r in rows if r["source"] == source}
        have = sum(int(r["usable_bp"]) for r in rows if r["source"] == source)
        shards = shuffled_shards(source)

        if args.dry_run:
            need = max(0.0, quota - have)
            est = sum(s for _, s in shards[:1]) or 1
            logger.info(
                "%-26s quota %5.1fGbp  have %5.1fGbp  %d train shards (%d already in "
                "manifest); ~%.1f more shards at ~%.1f bp/byte",
                source, quota / 1e9, have / 1e9, len(shards), len(done),
                need / (est * bp_per_byte), bp_per_byte,
            )
            continue

        # A trimmed shard lands the source a hair under quota (Bernoulli variance), and a
        # plain `have >= quota` would then spend another whole 3GB shard -- ~10 Gbp, a 20%+
        # overshoot on the source and a skewed mixture -- to close a 0.01% gap. Once a
        # source has been trimmed it is done, on this run and on any resume.
        if any(float(r["trim_p"]) < 1.0 for r in rows if r["source"] == source):
            logger.info("[%s] already complete (%.2f Gbp, trimmed)", source, have / 1e9)
            continue

        for hf_path, _size in shards:
            if have >= quota:
                break
            if hf_path in done:
                continue
            logger.info("[%s] %.1f/%.1f Gbp -- fetching %s", source, have / 1e9,
                        quota / 1e9, hf_path)
            local = Path(hf_hub_download(REPO_ID, hf_path, repo_type="dataset",
                                         revision=REVISION, local_dir=str(raw)))
            stats = scan_shard(local)
            bp, trim_p = stats["usable_bp"], 1.0

            if have + bp > quota:
                # The quota lands inside this shard. Accept a seeded Bernoulli subset of
                # the WHOLE shard rather than its head: records are concatenated per source
                # genome, so a head slice is a few taxa, not a sample of the source.
                trim_p = (quota - have) / bp
                trimmed = Path(str(local).replace(".jsonl.gz", ".trimmed.jsonl.gz"))
                kept, _ = trim_shard(local, trimmed, trim_p, shard_seed(SEED, source, local.name))
                os.remove(local)
                local = trimmed
                stats = scan_shard(local)
                bp = stats["usable_bp"]
                logger.info("[%s] trimmed to p=%.4f -> %d records, %.2f Gbp",
                            source, trim_p, kept, bp / 1e9)

            write_scan(out, source, local, stats)
            append_manifest(manifest, {
                "hf_revision": REVISION, "source": source, "hf_path": hf_path,
                "local_path": str(local), "compressed_bytes": local.stat().st_size,
                "usable_bp": bp, "trim_p": f"{trim_p:.6f}", "sha256": sha256_file(local),
            })
            have += bp
            if trim_p < 1.0:
                break  # the trim was sized to fill the quota exactly; nothing left to fetch

        if not args.dry_run:
            if have < quota * 0.97:
                logger.warning("[%s] SHORT: %.2f of %.2f Gbp -- source ran out of shards",
                               source, have / 1e9, quota / 1e9)
            else:
                logger.info("[%s] done: %.2f Gbp", source, have / 1e9)


def cmd_stats(args: argparse.Namespace) -> None:
    out = Path(args.out)
    scans = sorted((out / "stats" / "shards").glob("*.json"))
    if not scans:
        raise SystemExit(f"no shard scans under {out/'stats'/'shards'} -- run `download` first")

    by_source: dict[str, dict] = {}
    bins = {b: {"segments": 0, "bp": 0, "windows": 0} for b in TARGET_BIN_SHARE}
    chars = {c: 0 for c in "ACGTN"} | {"other": 0}
    total = {"records": 0, "raw_bp": 0, "usable_bp": 0, "compressed_bytes": 0, "shards": 0}

    sizes = {r["local_path"]: int(r["compressed_bytes"]) for r in
             read_manifest(out / "manifests" / "shards.tsv")}

    for p in scans:
        s = json.loads(p.read_text())
        src = by_source.setdefault(s["source"], {"shards": 0, "records": 0, "usable_bp": 0,
                                                 "raw_bp": 0, "compressed_bytes": 0})
        cb = sizes.get(s["local_path"], 0)
        for k, v in (("shards", 1), ("records", s["records"]), ("usable_bp", s["usable_bp"]),
                     ("raw_bp", s["raw_bp"]), ("compressed_bytes", cb)):
            src[k] += v
            if k in total:
                total[k] += v
        for c, n in s["chars"].items():
            chars[c] += n
        for b, v in s["bins"].items():
            for k in v:
                bins[b][k] += v[k]

    tok_target = {b: v / sum(TARGET_BIN_SHARE.values()) for b, v in TARGET_BIN_SHARE.items()}
    avail = {b: bins[b]["windows"] * int(b) for b in bins}
    tok_total = sum(avail.values()) or 1
    share = {b: avail[b] / tok_total for b in bins}
    # Binding constraint: how many OG2 replay tokens the pool serves at NTv3's bin mix
    # before any bin has to repeat, and which bin binds.
    servable = {b: (avail[b] / tok_target[b] if tok_target[b] else float("inf")) for b in bins}
    binding = min(servable, key=servable.get)

    total_bp = total["usable_bp"] or 1
    summary = {
        "hf_repo": REPO_ID, "hf_revision": REVISION, "seed": SEED,
        "totals": total | {"acgtn_fraction": 1 - chars["other"] / (total["raw_bp"] or 1)},
        "chars": chars,
        "by_source": {k: v | {"pct_usable_bp": 100 * v["usable_bp"] / total_bp,
                              "quota_bp": SOURCES[k][0],
                              "pct_of_quota": 100 * v["usable_bp"] / SOURCES[k][0]}
                      for k, v in sorted(by_source.items())},
        "bins": {b: bins[b] | {"tokens": avail[b], "token_share_pct": 100 * share[b],
                               "target_share_pct": 100 * tok_target[b],
                               "supply_ratio": share[b] / tok_target[b] if tok_target[b] else 0}
                 for b in sorted(bins, key=int)},
        "og2_tokens_one_pass": servable[binding],
        "binding_bin": binding,
        "bin_supply_ratio": min(share[b] / tok_target[b] for b in bins if tok_target[b]),
    }
    (out / "stats").mkdir(parents=True, exist_ok=True)
    (out / "stats" / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "stats" / "README.md").write_text(_render(summary))
    logger.info("usable %.1f Gbp over %d shards; bin_supply_ratio %.2f (binding: %s); "
                "one pass serves %.1f G replay tokens at the NTv3 mix",
                total["usable_bp"] / 1e9, total["shards"], summary["bin_supply_ratio"],
                binding, summary["og2_tokens_one_pass"] / 1e9)


def _render(s: dict) -> str:
    lines = [
        "# OpenGenome2 replay pool", "",
        f"- repo: `{s['hf_repo']}` @ `{s['hf_revision']}`",
        f"- selection seed: {s['seed']}",
        f"- shards: {s['totals']['shards']}, "
        f"compressed: {s['totals']['compressed_bytes'] / 1e9:.1f} GB",
        f"- records: {s['totals']['records']:,}, "
        f"usable: {s['totals']['usable_bp'] / 1e9:.2f} Gbp "
        f"of {s['totals']['raw_bp'] / 1e9:.2f} Gbp raw",
        f"- ACGTN fraction of raw text: {s['totals']['acgtn_fraction']:.4f}",
        "", "Usable bp = what the reader can actually window: uppercased, split on any "
        "non-ACGTN character, segments >= 1024bp.", "",
        "## By source", "",
        "| source | shards | records | usable Gbp | % of pool | % of quota |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, v in s["by_source"].items():
        lines.append(f"| {k} | {v['shards']} | {v['records']:,} | "
                     f"{v['usable_bp'] / 1e9:.2f} | {v['pct_usable_bp']:.1f} | "
                     f"{v['pct_of_quota']:.0f} |")
    lines += [
        "", "## By NTv3 length bin", "",
        "| bin | segments | windows | tokens (G) | share % | NTv3 target % | supply ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b, v in s["bins"].items():
        lines.append(f"| {b} | {v['segments']:,} | {v['windows']:,} | "
                     f"{v['tokens'] / 1e9:.2f} | {v['token_share_pct']:.1f} | "
                     f"{v['target_share_pct']:.1f} | {v['supply_ratio']:.2f} |")
    lines += [
        "", f"Binding bin: **{s['binding_bin']}** -- one pass over the pool serves "
        f"**{s['og2_tokens_one_pass'] / 1e9:.1f} G** replay tokens at NTv3's 7.9/13.0/13.2/66.0 "
        "mix before that bin repeats.", "",
        "Character histogram: "
        + ", ".join(f"{k}={v:,}" for k, v in s["chars"].items()), "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(DEFAULT_OUT), help="pool root (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch shards up to each source's bp quota")
    d.add_argument("--dry-run", action="store_true", help="list the plan, download nothing")
    d.add_argument("--source", choices=list(SOURCES), help="restrict to one source")
    d.add_argument("--extra-bp", type=float, default=0.0,
                   help="raise the quota by this many bp (top-up an undersupplied bin)")
    d.set_defaults(func=cmd_download)

    st = sub.add_parser("stats", help="merge shard scans into stats/summary.json + README.md")
    st.set_defaults(func=cmd_stats)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
