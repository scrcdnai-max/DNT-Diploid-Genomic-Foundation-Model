"""Build a small, persisted eval set for fast in-training validation.

Unlike `prepare_eval_data.py` (which filters to variant-only windows), this
script preserves the natural distribution INCLUDING non-variant windows so
the model's eval-loss tracks real-world inference distribution where most
positions are reference.

The produced dataset is a HuggingFace `Dataset` saved with `save_to_disk`
so all weekend gating arms can load the SAME eval set via `--test-data`,
ensuring cross-arm comparability.

Sampling strategy:
1. Read the source test parquet shards in streaming chunks.
2. Slice each record into non-overlapping windows of `--window-len`.
3. Dedup by sequence hash.
4. Optionally stratify: keep X% variant-containing + Y% non-variant.
5. Random-sample down to `--target-windows`.
6. Save as HF Dataset.

At default `--target-windows 10000` and `--window-len 4096`, the produced
set is ~40 MB on disk and runs in <2 min on an H100 at batch=512 (so the
training loop's eval cycle fits well under the user's 9-min cap with
headroom).

Usage:
    uv run python pretrain/build_eval_small.py \\
        --source /data/diploid_hf_v6/assembled/test \\
        --output /data/diploid_hf_v6/assembled/test_small_v6 \\
        --window-len 4096 \\
        --target-windows 10000 \\
        --variant-fraction 0.5 \\
        --seed 42 \\
        --max-source-shards 4
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import random
from collections import deque
from pathlib import Path

import datasets

logger = logging.getLogger(__name__)

SNP_MARKER = "^"
INDEL_OPEN = "{"


def _has_variant(seq: str) -> bool:
    return SNP_MARKER in seq or INDEL_OPEN in seq


def _slice_record(record: dict, window_len: int) -> list[dict]:
    """Non-overlapping window slicing for one source record."""
    seq = record["seq"]
    n = len(seq)
    if n <= window_len:
        return [{
            "sample_id": record["sample_id"],
            "chrom": record["chrom"],
            "start": int(record["start"]),
            "end": int(record["start"]) + n,
            "seq": seq,
            "sample_weight": int(record.get("sample_weight", 1)),
        }]
    out = []
    for off in range(0, n - window_len + 1, window_len):
        chunk = seq[off : off + window_len]
        out.append({
            "sample_id": record["sample_id"],
            "chrom": record["chrom"],
            "start": int(record["start"]) + off,
            "end": int(record["start"]) + off + window_len,
            "seq": chunk,
            "sample_weight": int(record.get("sample_weight", 1)),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--source", type=Path, required=True,
                   help="Source HF dataset dir (e.g., /data/diploid_hf_v6/assembled/test)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output HF dataset dir (overwritten if exists)")
    p.add_argument("--window-len", type=int, default=4096)
    p.add_argument("--target-windows", type=int, default=10000)
    p.add_argument(
        "--variant-fraction", type=float, default=0.5,
        help="Target fraction of OUTPUT windows that contain a variant marker "
        "(IGNORED when --natural-distribution is set). 0.5 = half variant, half "
        "non-variant. Set 1.0 for variant-only.",
    )
    p.add_argument(
        "--natural-distribution", action="store_true",
        help="Skip stratified sampling; draw windows uniformly at random from "
        "the dedup'd pool. Result reflects the natural source distribution "
        "(typically ~75%% variant-containing post-dedup at 4 kb windows on v6 "
        "test). Recommended for production eval to track real-world inference "
        "distribution; the stratified 50/50 mode is mainly for ablation.",
    )
    p.add_argument("--max-source-shards", type=int, default=None,
                   help="Cap source-shard count (each shard ~150k records).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.variant_fraction < 0 or args.variant_fraction > 1:
        raise SystemExit("--variant-fraction must be in [0, 1]")

    rng = random.Random(args.seed)

    if args.natural_distribution:
        # Single uniform reservoir; oversample 5x for dedup + final subsample headroom.
        target_variant = args.target_windows
        target_nonvariant = 0
        reservoir_uniform: deque[dict] = deque(maxlen=args.target_windows * 5)
        reservoir_var: deque[dict] = deque()
        reservoir_nonvar: deque[dict] = deque()
    else:
        target_variant = int(args.target_windows * args.variant_fraction)
        target_nonvariant = args.target_windows - target_variant
        reservoir_uniform = deque()
        reservoir_var = deque(maxlen=target_variant * 5)
        reservoir_nonvar = deque(maxlen=target_nonvariant * 5)
    seen_hashes: set[bytes] = set()

    logger.info("Loading source from %s ...", args.source)
    src = datasets.load_from_disk(str(args.source))
    n_src = len(src)
    logger.info("Source size: %d records", n_src)

    # Stream records in random order (HF Dataset shuffle + take).
    shuffled = src.shuffle(seed=args.seed)
    if args.max_source_shards is not None:
        # Approximate: cap to first N "shards worth" of records.
        approx_per_shard = max(1, n_src // 256)
        max_records = args.max_source_shards * approx_per_shard
        shuffled = shuffled.take(min(max_records, n_src))

    n_seen = 0
    n_windows_made = 0
    saturation_target_uniform = args.target_windows * 3
    saturation_target_var = max(target_variant * 3, 1) if target_variant else 0
    saturation_target_nonvar = max(target_nonvariant * 3, 1) if target_nonvariant else 0

    for record in shuffled:
        n_seen += 1
        for win in _slice_record(record, args.window_len):
            h = hashlib.md5(win["seq"].encode("utf-8")).digest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            n_windows_made += 1
            if args.natural_distribution:
                reservoir_uniform.append(win)
            elif _has_variant(win["seq"]):
                reservoir_var.append(win)
            else:
                reservoir_nonvar.append(win)
        # Stop early once reservoir is saturated.
        if args.natural_distribution:
            if len(reservoir_uniform) >= saturation_target_uniform:
                break
        elif (
            len(reservoir_var) >= saturation_target_var
            and len(reservoir_nonvar) >= saturation_target_nonvar
        ):
            break
        if n_seen % 100_000 == 0:
            logger.info(
                "Seen %d records, %d unique windows, uniform=%d, var=%d, nonvar=%d",
                n_seen, n_windows_made,
                len(reservoir_uniform),
                len(reservoir_var),
                len(reservoir_nonvar),
            )

    logger.info(
        "Done streaming: seen %d records, %d unique windows; "
        "uniform=%d, variant=%d, nonvariant=%d",
        n_seen, n_windows_made,
        len(reservoir_uniform), len(reservoir_var), len(reservoir_nonvar),
    )

    # Final subsample.
    if args.natural_distribution:
        pool = list(reservoir_uniform)
        rng.shuffle(pool)
        final_records = pool[:args.target_windows]
        n_var = sum(1 for r in final_records if _has_variant(r["seq"]))
        n_nonvar = len(final_records) - n_var
        logger.info(
            "Final eval set (natural distribution): %d windows "
            "(var=%d / %.1f%%, nonvar=%d / %.1f%%)",
            len(final_records),
            n_var, 100.0 * n_var / max(len(final_records), 1),
            n_nonvar, 100.0 * n_nonvar / max(len(final_records), 1),
        )
    else:
        var_pool = list(reservoir_var)
        nonvar_pool = list(reservoir_nonvar)
        rng.shuffle(var_pool)
        rng.shuffle(nonvar_pool)
        selected_var = var_pool[:target_variant]
        selected_nonvar = nonvar_pool[:target_nonvariant]

        actual_var = len(selected_var)
        actual_nonvar = len(selected_nonvar)
        if actual_var < target_variant:
            logger.warning(
                "Only %d variant windows available (wanted %d); "
                "topping up from non-variant pool.",
                actual_var, target_variant,
            )
            extra = target_variant - actual_var
            selected_nonvar += nonvar_pool[target_nonvariant:target_nonvariant + extra]
        if actual_nonvar < target_nonvariant:
            logger.warning(
                "Only %d non-variant windows available (wanted %d); "
                "topping up from variant pool.",
                actual_nonvar, target_nonvariant,
            )
            extra = target_nonvariant - actual_nonvar
            selected_var += var_pool[target_variant:target_variant + extra]

        final_records = selected_var + selected_nonvar
        rng.shuffle(final_records)
        logger.info(
            "Final eval set (stratified, target_variant_frac=%.2f): %d windows "
            "(var=%d, nonvar=%d)",
            args.variant_fraction,
            len(final_records),
            sum(1 for r in final_records if _has_variant(r["seq"])),
            sum(1 for r in final_records if not _has_variant(r["seq"])),
        )

    if not final_records:
        raise SystemExit("No records survived; check source and window-len.")

    # Save as HF Dataset (parquet-backed).
    out_ds = datasets.Dataset.from_list(final_records)
    if args.output.exists():
        import shutil
        shutil.rmtree(args.output)
    out_ds.save_to_disk(str(args.output))
    logger.info("Saved %d records to %s", len(out_ds), args.output)


if __name__ == "__main__":
    main()
