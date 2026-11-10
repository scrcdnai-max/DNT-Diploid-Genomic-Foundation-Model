#!/usr/bin/env python3
"""Prepare a deduplicated test dataset for faster MLM evaluation.

Default mode (``--mode window``): slice each region into non-overlapping 4 K
windows, keep only variant-containing windows, exact-dedup by window hash, and
cap at K (default:4) unique windows per genomic slot (chrom, window_start).

Processes one shard at a time to bound memory usage.  Uses vectorized Polars
operations for window extraction instead of Python loops.

Output schema is compatible with train.py's ``load_parquet_dataset`` /
``build_window_index`` path.  Each output row carries one window as ``seq``,
with recalculated ``start``/``end`` coordinates for that window.

This script never modifies the source test dataset.

Usage (window mode, default):
    python prepare_eval_data.py \\
        --test-data /data/diploid_hf_v5/assembled/test \\
        --output-dir /data/diploid_hf_v5/assembled/test_4k_windowed

    # First 100 shards only (faster):
    python prepare_eval_data.py \\
        --test-data /data/diploid_hf_v5/assembled/test \\
        --output-dir /data/diploid_hf_v5/assembled/test_4k_windowed \\
        --max-shards 100

Usage (legacy region mode):
    python prepare_eval_data.py \\
        --mode region \\
        --test-data /data/diploid_hf_v5/assembled/test \\
        --output-dir /data/diploid_hf_v5/assembled/test_deduped
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
import time
from pathlib import Path

import polars as pl
import pyarrow.ipc as ipc

logger = logging.getLogger(__name__)

SNP_MARKER = "^"
INDEL_OPEN = "{"
VALID_SHARD_SUFFIXES = {".arrow", ".parquet"}
REQUIRED_COLUMNS = ("sample_id", "chrom", "start", "end", "seq", "sample_weight")
_OUTPUT_SCHEMA = {
    "sample_id": pl.Utf8,
    "chrom": pl.Utf8,
    "start": pl.Int64,
    "end": pl.Int64,
    "seq": pl.Utf8,
    "sample_weight": pl.Float64,
}


def _list_input_files(data_dir: Path, max_shards: int | None = None) -> list[Path]:
    """Return sorted Arrow/Parquet shard paths from *data_dir*."""
    files = [
        p
        for p in sorted(data_dir.iterdir())
        if p.is_file()
        and p.suffix in VALID_SHARD_SUFFIXES
        and ".tmp" not in p.suffixes
        and not p.name.startswith("cache-")
    ]
    if not files:
        raise FileNotFoundError(
            f"No valid .arrow or .parquet shards found in {data_dir}"
        )
    if max_shards is not None:
        files = files[:max_shards]
    return files


def _read_input_shard(
    shard_path: Path, columns: list[str] | None = None
) -> pl.DataFrame:
    """Read one input shard regardless of whether it is Parquet or Arrow IPC."""
    if shard_path.suffix == ".parquet":
        return pl.read_parquet(shard_path, columns=columns)
    if shard_path.suffix == ".arrow":
        return pl.read_ipc_stream(shard_path, columns=columns)
    raise ValueError(f"Unsupported shard format: {shard_path}")


def _read_schema_names(shard_path: Path) -> tuple[str, ...]:
    """Return schema column names for a Parquet or Arrow IPC shard."""
    if shard_path.suffix == ".parquet":
        return tuple(pl.read_parquet_schema(str(shard_path)).keys())
    if shard_path.suffix == ".arrow":
        with ipc.open_stream(shard_path) as reader:
            return tuple(reader.schema.names)
    raise ValueError(f"Unsupported shard format: {shard_path}")


def _validate_paths(input_dir: Path, output_dir: Path, out_file: Path) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input test dir not found: {input_dir}")

    if input_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "Output directory must be different from input test directory "
            "(to avoid overwriting source data)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_parquets = list(output_dir.glob("*.parquet"))
    if existing_parquets:
        raise FileExistsError(
            f"Output directory already contains parquet files: {output_dir}. "
            "Choose a new directory or delete existing files."
        )

    if out_file.exists():
        raise FileExistsError(f"Output file already exists: {out_file}")


def _count_rows(lf: pl.LazyFrame) -> int:
    return int(lf.select(pl.len().alias("n")).collect()["n"][0])


def _extract_windows_vectorized(
    df: pl.DataFrame, window_len: int, stride: int
) -> pl.DataFrame:
    """Vectorized window extraction using Polars expressions.

    - Keeps short seqs and short tail windows.
    - Generates offsets across the full sequence via ``int_ranges`` + ``explode``.
    - Extracts window substrings with ``str.slice`` (last chunk may be shorter).
    - Filters to variant-containing windows.
    - Recalculates genomic coordinates per window.
    """
    # Empty input short-circuit
    if len(df) == 0:
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    # Generate offsets over full sequence length, then explode to one row per window.
    # This keeps short sequences and short trailing windows.
    df = (
        df.with_columns(
            pl.int_ranges(
                0,
                pl.col("seq").str.len_chars(),
                stride,
            ).alias("_offset")
        )
        .explode("_offset")
        .with_columns(
            pl.col("seq").str.slice(pl.col("_offset"), window_len).alias("_win_seq"),
            (pl.col("start") + pl.col("_offset")).alias("_win_start"),
        )
        # Keep only variant-containing windows
        .filter(pl.col("_win_seq").str.contains(r"[\^\{]"))
        .select(
            pl.col("sample_id"),
            pl.col("chrom"),
            pl.col("_win_start").alias("start"),
            (pl.col("_win_start") + pl.col("_win_seq").str.len_chars()).alias("end"),
            pl.col("_win_seq").alias("seq"),
            pl.col("sample_weight"),
        )
    )
    return df


def _process_shard(
    shard_path: Path,
    window_len: int,
    stride: int,
    seed: int,
) -> pl.DataFrame:
    """Read one shard, extract variant windows, per-shard dedup, return DataFrame."""
    df = _read_input_shard(shard_path, columns=list(REQUIRED_COLUMNS))
    # Pre-filter: only variant regions. Short sequences are kept.
    df = df.filter(pl.col("seq").str.contains(r"[\^\{]"))
    if len(df) == 0:
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    wins = _extract_windows_vectorized(df, window_len, stride)
    del df

    if len(wins) == 0:
        return wins

    # Per-shard exact dedup (reduces downstream global dedup work)
    wins = (
        wins.with_columns(pl.col("seq").hash(seed=seed).alias("_h"))
        .unique(subset=["_h"], keep="first")
        .drop("_h")
    )
    return wins


def _run_window_mode(
    files: list[Path],
    window_len: int,
    stride: int,
    max_per_slot: int,
    seed: int,
    compression: str,
) -> pl.DataFrame:
    """Streaming shard-by-shard window-mode pipeline.

    1. For each shard: read → variant filter → vectorized windowing → per-shard dedup → write temp parquet
    2. Lazy scan of all temp parquets → global exact dedup by hash → per-slot cap → return DataFrame
    """
    n_shards = len(files)
    tmpdir = Path(tempfile.mkdtemp(prefix="eval_windows_"))
    t0 = time.monotonic()

    try:
        shard_temps: list[Path] = []
        cum_windows = 0

        for i, shard_path in enumerate(files):
            wins = _process_shard(shard_path, window_len, stride, seed)
            n_wins = len(wins)
            cum_windows += n_wins

            if n_wins > 0:
                tmp = tmpdir / f"shard_{i:04d}.parquet"
                wins.write_parquet(str(tmp), compression=compression)
                shard_temps.append(tmp)

            del wins
            if (i + 1) % 10 == 0 or i == n_shards - 1:
                logger.info(
                    "[%d/%d] %s — %d wins (cum %d, %.1fs)",
                    i + 1,
                    n_shards,
                    shard_path.name,
                    n_wins,
                    cum_windows,
                    time.monotonic() - t0,
                )

        if not shard_temps:
            logger.warning("No variant windows produced from %d shards.", n_shards)
            return pl.DataFrame(schema=_OUTPUT_SCHEMA)

        logger.info(
            "Shard pass complete: %d temp files, %d cumulative windows in %.1fs",
            len(shard_temps),
            cum_windows,
            time.monotonic() - t0,
        )

        # Global dedup pass — scan temp parquets lazily
        lf = pl.scan_parquet([str(p) for p in shard_temps])
        df = (
            lf.with_columns(pl.col("seq").hash(seed=seed).alias("_h"))
            .unique(subset=["_h"], keep="first")
            .drop("_h")
            .collect()
        )
        logger.info("After global exact window dedup: %d windows", len(df))

        # Deterministic per-slot cap
        df = (
            df.with_columns(pl.col("seq").hash(seed=seed).alias("_sk"))
            .sort(["chrom", "start", "_sk"])
            .group_by(["chrom", "start"], maintain_order=True)
            .head(max_per_slot)
            .drop("_sk")
        )
        logger.info(
            "After per-slot cap (max %d): %d windows (%.1fs total)",
            max_per_slot,
            len(df),
            time.monotonic() - t0,
        )
        return df
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_region_mode(
    lf: pl.LazyFrame,
    max_per_region: int,
    seed: int,
) -> pl.LazyFrame:
    """Legacy region-mode pipeline (V1 behaviour, kept for debugging)."""
    lf = lf.filter(pl.col("seq").str.contains(r"[\^\{]"))
    logger.info("After variant-marker filter: %d records", _count_rows(lf))

    lf = lf.unique(subset=["seq"], keep="first")
    logger.info("After exact sequence dedup: %d records", _count_rows(lf))

    lf = (
        lf.with_columns(pl.col("seq").hash(seed=seed).alias("_seq_hash"))
        .sort(["chrom", "start", "end", "_seq_hash"])
        .group_by(["chrom", "start", "end"], maintain_order=True)
        .head(max_per_region)
        .drop("_seq_hash")
    )
    logger.info(
        "After per-region cap (max %d): %d records",
        max_per_region,
        _count_rows(lf),
    )
    return lf


def _prepare_region_scan_files(
    files: list[Path], tmpdir: Path, compression: str
) -> list[Path]:
    """Return Parquet scan inputs for region mode, converting Arrow shards if needed."""
    scan_files: list[Path] = []
    for idx, shard_path in enumerate(files):
        if shard_path.suffix == ".parquet":
            scan_files.append(shard_path)
            continue

        tmp_path = tmpdir / f"input_{idx:04d}.parquet"
        _read_input_shard(shard_path, columns=list(REQUIRED_COLUMNS)).write_parquet(
            str(tmp_path),
            compression=compression,
        )
        scan_files.append(tmp_path)
        if (idx + 1) % 10 == 0 or idx == len(files) - 1:
            logger.info(
                "Prepared %d/%d input shard(s) for region-mode scan",
                idx + 1,
                len(files),
            )
    return scan_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build deduplicated eval dataset from test Arrow/Parquet shards.",
    )
    p.add_argument(
        "--test-data",
        type=Path,
        required=True,
        help="Input test directory with per-sample Arrow/Parquet shards.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for deduplicated eval dataset.",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="eval_windows.parquet",
        help="Output parquet filename under --output-dir.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="window",
        choices=["window", "region"],
        help="'window': 4K non-overlapping window dedup (default). 'region': legacy.",
    )
    p.add_argument(
        "--window-len",
        type=int,
        default=4096,
        help="Window length in characters (window mode only).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride. Defaults to --window-len (non-overlapping).",
    )
    p.add_argument(
        "--max-per-slot",
        type=int,
        default=4,
        help="Max unique windows kept per (chrom, window_start) slot (window mode).",
    )
    p.add_argument(
        "--max-per-region",
        type=int,
        default=4,
        help="Max sequences per (chrom, start, end) region (region mode only).",
    )
    p.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Cap on input shard count (smoke tests).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic ordering within slots/regions.",
    )
    p.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "zstd", "snappy", "uncompressed"],
        help="Parquet compression codec for output.",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle output rows before writing.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args()
    if args.stride is None:
        args.stride = args.window_len
    if args.max_per_slot < 1:
        raise ValueError("--max-per-slot must be >= 1")
    if args.max_per_region < 1:
        raise ValueError("--max-per-region must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M",
    )

    output_file = args.output_dir / args.output_name
    _validate_paths(args.test_data, args.output_dir, output_file)

    files = _list_input_files(args.test_data, args.max_shards)
    logger.info(
        "Reading %d shard(s) from %s [mode=%s]", len(files), args.test_data, args.mode
    )

    # Validate schema against first shard
    schema_names = _read_schema_names(files[0])
    missing = [c for c in REQUIRED_COLUMNS if c not in schema_names]
    if missing:
        raise KeyError(f"Input dataset is missing required columns: {missing}")

    if args.mode == "window":
        final_df = _run_window_mode(
            files,
            window_len=args.window_len,
            stride=args.stride,
            max_per_slot=args.max_per_slot,
            seed=args.seed,
            compression=args.compression,
        )
        if len(final_df) == 0:
            logger.warning("No windows produced — output file not written.")
            return
        if args.shuffle:
            logger.info("Shuffling output rows with seed=%d", args.seed)
            final_df = final_df.sample(fraction=1.0, shuffle=True, seed=args.seed)
        logger.info("Writing %d windows to %s", len(final_df), output_file)
        final_df.write_parquet(str(output_file), compression=args.compression)
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="eval_regions_"))
        try:
            scan_files = _prepare_region_scan_files(files, tmpdir, args.compression)
            lf = pl.scan_parquet([str(p) for p in scan_files])
            final_lf = _run_region_mode(lf, args.max_per_region, args.seed)
            if args.shuffle:
                logger.info("Shuffling output rows with seed=%d", args.seed)
                region_df = final_lf.collect().sample(
                    fraction=1.0, shuffle=True, seed=args.seed
                )
                logger.info("Writing region-deduped records to %s", output_file)
                region_df.write_parquet(str(output_file), compression=args.compression)
            else:
                logger.info("Writing region-deduped records to %s", output_file)
                final_lf.sink_parquet(str(output_file), compression=args.compression)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    out_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info("Done. Output: %s (%.1f MB)", output_file, out_size_mb)


if __name__ == "__main__":
    main()
