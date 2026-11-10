"""Gather diploid token usage statistics from single-sample BCF files.

Statistics are derived from the *source* BCFs plus the reference genome —
this script never reads a generated HuggingFace dataset.  For token
frequencies over an assembled dataset (what the model actually sees, post
dedup/shuffle/marker-externalization) use
``huggingface_dataset/tools/gather_dataset_stats.py`` instead.

Iterates through single-sample BCF files, classifying each variant to
produce per-genome statistics including:

- Diploid token usage counts (genome-wide or variant-only)
- Indel statistics (count, min/mean/max length, insertions vs deletions)
- Zygosity counts (heterozygous vs homozygous-alt, SNP vs indel)
- Phasing statistics

Designed for batch processing thousands of genomes efficiently using
multiprocessing.  Workers only need the BCF file and lightweight lookup
dicts — the reference genome is loaded only once in the main process to
precompute base composition.

Usage:
    # Single genome
    python -m preprocess.tools.gather_diploid_stats_bcf \\
        --bcf /path/to/sample.bcf \\
        --output-dir ./stats_output

    # Batch (all BCFs in a directory)
    python -m preprocess.tools.gather_diploid_stats_bcf \\
        --bcf-dir /path/to/demultiplexed/ \\
        --output-dir ./stats_output \\
        --workers 8
"""

import argparse
import json
import logging
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from cyvcf2 import VCF

from preprocess.encoder.get_genome_sequence import (
    _collect_variants,
    _get_alleles_for_variant,
    _resolve_overlapping_variants,
)
from preprocess import paths, tokens
from preprocess.encoder.reference import ReferenceFile

logger = logging.getLogger(__name__)

STANDARD_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


# ---------------------------------------------------------------------------
# Token loading (standalone, no GenomeSequenceProcessor needed)
# ---------------------------------------------------------------------------


def load_diploid_lookups(
    tokens_path: Path,
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """Load diploid token lookup dicts from TSV file.

    Returns:
        (ordered_lookup, unordered_lookup, all_symbols) where the lookups
        map pair strings (e.g. "AC") to single-character symbols, and
        all_symbols is the deduplicated list of unique symbols in file
        order (used for consistent column ordering in output).
    """
    # Canonical labels are "phased"/"unphased"; legacy labels still work but
    # emit DeprecationWarning (preprocess.tokens.normalize_order_label).
    table = tokens.load_token_table(tokens_path)
    return table.phased, table.unphased, table.symbols


# ---------------------------------------------------------------------------
# Reference base composition (precomputed once)
# ---------------------------------------------------------------------------


def compute_reference_base_counts(
    ref_path: Path,
    chromosomes: List[str],
) -> Dict[str, Dict[str, int]]:
    """Count each base in the reference genome per chromosome.

    Opens the reference file, iterates over chromosomes, and uses
    ``np.bincount`` for fast counting.  The result is a lightweight dict
    suitable for passing to worker processes.

    Returns:
        ``{chrom: {"A": n, "C": n, "G": n, "T": n, "N": n, "length": n}}``
    """
    counts: Dict[str, Dict[str, int]] = {}
    with ReferenceFile(ref_path) as ref:
        available = set(ref.references)
        for chrom in chromosomes:
            if chrom not in available:
                logger.warning("Chromosome %s not in reference, skipping", chrom)
                continue
            length = ref.get_reference_length(chrom)
            arr = ref.fetch(chrom, 0, length, as_array=True)
            bc = np.bincount(arr, minlength=256)
            counts[chrom] = {
                "A": int(bc[ord("A")]),
                "C": int(bc[ord("C")]),
                "G": int(bc[ord("G")]),
                "T": int(bc[ord("T")]),
                "N": int(bc[ord("N")]),
                "length": length,
            }
    return counts


# ---------------------------------------------------------------------------
# Core per-genome statistics
# ---------------------------------------------------------------------------

def gather_stats_for_genome(
    bcf_path: Path,
    ordered_lookup: Dict[str, str],
    unordered_lookup: Dict[str, str],
    ref_base_counts: Optional[Dict[str, Dict[str, int]]],
    chromosomes: List[str],
) -> dict:
    """Gather diploid token and variant statistics for a single genome.

    Args:
        bcf_path: Path to single-sample BCF file.
        ordered_lookup: Phased pair → symbol mapping.
        unordered_lookup: Unphased pair → symbol mapping.
        ref_base_counts: Precomputed reference base composition per
            chromosome (from ``compute_reference_base_counts``).  If
            ``None``, only variant-site token counts are reported.
        chromosomes: List of chromosome names to process.

    Returns:
        Dict with all statistics for this genome.
    """
    sample_name = Path(bcf_path).stem

    # Accumulators
    token_counts: Counter = Counter()
    ref_bases_replaced: Counter = Counter()  # which ref bases were overwritten
    total_inserted_positions = 0

    het_snp = 0
    hom_alt_snp = 0
    het_indel = 0
    hom_alt_indel = 0
    total_phased = 0
    total_unphased = 0
    insertion_count = 0
    deletion_count = 0
    indel_lengths: list = []
    total_variant_sites = 0

    try:
        vcf = VCF(str(bcf_path))
    except Exception as e:
        logger.error("Failed to open %s: %s", bcf_path, e)
        return {"sample_name": sample_name, "error": str(e)}

    try:
        for chrom in chromosomes:
            # Fetch all variants on this chromosome
            region = chrom
            try:
                records = vcf(region)
            except Exception:
                # Chromosome not in BCF — skip
                continue

            variants = _collect_variants(records, 0, chrom, sample_name)
            if not variants:
                continue

            variants = _resolve_overlapping_variants(
                variants, chrom, sample_name,
            )

            for pos, var in variants.items():
                hap0_allele, hap1_allele = _get_alleles_for_variant(
                    var, pos=pos,
                )
                ref_len = len(var.ref)
                h0_len = len(hap0_allele)
                h1_len = len(hap1_allele)
                max_allele_len = max(h0_len, h1_len)
                is_het = hap0_allele != hap1_allele

                # Classify SNP vs indel
                is_snp = (
                    ref_len == 1 and h0_len == 1 and h1_len == 1
                )

                if is_snp:
                    if is_het:
                        het_snp += 1
                    else:
                        hom_alt_snp += 1
                else:
                    if is_het:
                        het_indel += 1
                    else:
                        hom_alt_indel += 1

                    # Indel length = abs(len(allele) - len(ref)) for
                    # each non-ref allele (event length).
                    for allele_len in (h0_len, h1_len):
                        if allele_len != ref_len:
                            indel_lengths.append(abs(allele_len - ref_len))

                    # Classify insertion vs deletion (by the longer
                    # non-ref allele).
                    has_insertion = h0_len > ref_len or h1_len > ref_len
                    has_deletion = h0_len < ref_len or h1_len < ref_len
                    if has_insertion:
                        insertion_count += 1
                    if has_deletion:
                        deletion_count += 1

                # Phasing
                if var.phased:
                    total_phased += 1
                else:
                    total_unphased += 1

                total_variant_sites += 1

                # --- Token accounting ---
                lookup = ordered_lookup if var.phased else unordered_lookup

                # Positions within the REF span
                for k in range(ref_len):
                    ref_base = var.ref[k].upper()
                    h0_base = hap0_allele[k] if k < h0_len else "-"
                    h1_base = hap1_allele[k] if k < h1_len else "-"
                    pair = h0_base + h1_base
                    token = lookup.get(pair, "N")
                    token_counts[token] += 1
                    ref_bases_replaced[ref_base] += 1

                # Insertion positions (beyond REF span)
                for k in range(ref_len, max_allele_len):
                    h0_base = hap0_allele[k] if k < h0_len else "-"
                    h1_base = hap1_allele[k] if k < h1_len else "-"
                    pair = h0_base + h1_base
                    token = lookup.get(pair, "N")
                    token_counts[token] += 1
                    total_inserted_positions += 1
    finally:
        vcf.close()

    # Indel summary
    if indel_lengths:
        indel_min = min(indel_lengths)
        indel_max = max(indel_lengths)
        indel_mean = sum(indel_lengths) / len(indel_lengths)
    else:
        indel_min = 0
        indel_max = 0
        indel_mean = 0.0

    # Build genome-wide token counts if reference composition provided
    genome_token_counts: Counter = Counter()
    total_genome_positions = 0

    if ref_base_counts is not None:
        # Start with reference homozygous tokens
        for chrom in chromosomes:
            if chrom not in ref_base_counts:
                continue
            bc = ref_base_counts[chrom]
            total_genome_positions += bc["length"]
            for base in ("A", "C", "G", "T", "N"):
                # Homozygous ref: "AA"→"A", "CC"→"C", etc.
                genome_token_counts[base] += bc[base]

        # Subtract replaced reference positions
        for base, count in ref_bases_replaced.items():
            if base in genome_token_counts:
                genome_token_counts[base] -= count

        # Add variant tokens
        for token, count in token_counts.items():
            genome_token_counts[token] += count

        # Add insertion positions to total
        total_genome_positions += total_inserted_positions

    # Assemble result dict
    result = {
        "sample_name": sample_name,
        "total_variant_sites": total_variant_sites,
        "het_snp_count": het_snp,
        "hom_alt_snp_count": hom_alt_snp,
        "het_indel_count": het_indel,
        "hom_alt_indel_count": hom_alt_indel,
        "total_het_count": het_snp + het_indel,
        "total_hom_alt_count": hom_alt_snp + hom_alt_indel,
        "indel_count": het_indel + hom_alt_indel,
        "insertion_count": insertion_count,
        "deletion_count": deletion_count,
        "indel_min_len": indel_min,
        "indel_max_len": indel_max,
        "indel_mean_len": round(indel_mean, 2),
        "total_phased_variants": total_phased,
        "total_unphased_variants": total_unphased,
        "total_inserted_positions": total_inserted_positions,
    }

    if ref_base_counts is not None:
        result["total_genome_positions"] = total_genome_positions
        result["genome_token_counts"] = dict(genome_token_counts)

    result["variant_token_counts"] = dict(token_counts)

    return result


# ---------------------------------------------------------------------------
# Worker wrapper for multiprocessing
# ---------------------------------------------------------------------------

# Module-level state set by pool initializer
_worker_ordered_lookup: Dict[str, str] = {}
_worker_unordered_lookup: Dict[str, str] = {}
_worker_ref_base_counts: Optional[Dict[str, Dict[str, int]]] = None
_worker_chromosomes: List[str] = []


def _init_worker(
    ordered_lookup: Dict[str, str],
    unordered_lookup: Dict[str, str],
    ref_base_counts: Optional[Dict[str, Dict[str, int]]],
    chromosomes: List[str],
) -> None:
    """Pool initializer — store shared read-only data in each worker."""
    global _worker_ordered_lookup, _worker_unordered_lookup
    global _worker_ref_base_counts, _worker_chromosomes
    _worker_ordered_lookup = ordered_lookup
    _worker_unordered_lookup = unordered_lookup
    _worker_ref_base_counts = ref_base_counts
    _worker_chromosomes = chromosomes


def _worker_fn(bcf_path: str) -> dict:
    """Worker function for multiprocessing.Pool.map."""
    return gather_stats_for_genome(
        Path(bcf_path),
        _worker_ordered_lookup,
        _worker_unordered_lookup,
        _worker_ref_base_counts,
        _worker_chromosomes,
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def _build_tsv_header(all_symbols: List[str], has_genome_counts: bool) -> str:
    """Build TSV header line."""
    base_cols = [
        "sample_name",
        "total_variant_sites",
        "het_snp_count",
        "hom_alt_snp_count",
        "het_indel_count",
        "hom_alt_indel_count",
        "total_het_count",
        "total_hom_alt_count",
        "indel_count",
        "insertion_count",
        "deletion_count",
        "indel_min_len",
        "indel_max_len",
        "indel_mean_len",
        "total_phased_variants",
        "total_unphased_variants",
        "total_inserted_positions",
    ]
    if has_genome_counts:
        base_cols.append("total_genome_positions")

    # Token columns — prefixed with "token_" for clarity
    token_cols = [f"token_{s}" for s in all_symbols]

    return "\t".join(base_cols + token_cols)


def _result_to_tsv_row(
    result: dict,
    all_symbols: List[str],
    has_genome_counts: bool,
) -> str:
    """Convert a result dict to a TSV row string."""
    if "error" in result:
        return ""  # skip errored genomes in TSV

    token_source = (
        result.get("genome_token_counts", {})
        if has_genome_counts
        else result.get("variant_token_counts", {})
    )

    vals = [
        result["sample_name"],
        str(result["total_variant_sites"]),
        str(result["het_snp_count"]),
        str(result["hom_alt_snp_count"]),
        str(result["het_indel_count"]),
        str(result["hom_alt_indel_count"]),
        str(result["total_het_count"]),
        str(result["total_hom_alt_count"]),
        str(result["indel_count"]),
        str(result["insertion_count"]),
        str(result["deletion_count"]),
        str(result["indel_min_len"]),
        str(result["indel_max_len"]),
        str(result["indel_mean_len"]),
        str(result["total_phased_variants"]),
        str(result["total_unphased_variants"]),
        str(result["total_inserted_positions"]),
    ]
    if has_genome_counts:
        vals.append(str(result.get("total_genome_positions", 0)))

    for sym in all_symbols:
        vals.append(str(token_source.get(sym, 0)))

    return "\t".join(vals)


def process_batch(
    bcf_paths: List[Path],
    ref_path: Optional[Path],
    tokens_path: Path,
    output_dir: Path,
    n_workers: int = 8,
    chromosomes: Optional[List[str]] = None,
    variant_only: bool = False,
) -> None:
    """Process a batch of genomes and write statistics.

    Args:
        bcf_paths: List of BCF file paths to process.
        ref_path: Path to reference genome (2bit or FASTA).  Required
            unless *variant_only* is True.
        tokens_path: Path to diploid_tokens.tsv.
        output_dir: Directory for output files.
        n_workers: Number of parallel workers.
        chromosomes: Chromosomes to process (default: autosomes + sex).
        variant_only: If True, skip reference composition and only
            report variant-site token counts.
    """
    if chromosomes is None:
        chromosomes = STANDARD_CHROMS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load token lookups
    ordered_lookup, unordered_lookup, all_symbols = load_diploid_lookups(
        tokens_path,
    )
    logger.info(
        "Loaded %d ordered + %d unordered token mappings, %d unique symbols",
        len(ordered_lookup), len(unordered_lookup), len(all_symbols),
    )

    # Precompute reference base counts (unless variant-only mode)
    ref_base_counts: Optional[Dict[str, Dict[str, int]]] = None
    if not variant_only:
        if ref_path is None:
            raise ValueError(
                "Reference genome path is required for genome-wide token "
                "counts.  Use --variant-only to skip reference composition."
            )
        logger.info("Computing reference base counts from %s ...", ref_path)
        t0 = time.time()
        ref_base_counts = compute_reference_base_counts(ref_path, chromosomes)
        logger.info(
            "Reference base counts computed in %.1fs", time.time() - t0,
        )

    has_genome_counts = ref_base_counts is not None

    # Prepare output files
    tsv_path = output_dir / "diploid_stats_summary.tsv"
    json_dir = output_dir / "per_genome"
    json_dir.mkdir(exist_ok=True)

    header = _build_tsv_header(all_symbols, has_genome_counts)

    n_total = len(bcf_paths)
    n_done = 0
    n_errors = 0
    t_start = time.time()

    bcf_path_strs = [str(p) for p in bcf_paths]

    with open(tsv_path, "w") as tsv_f:
        tsv_f.write(header + "\n")

        if n_workers <= 1 or n_total == 1:
            # Sequential mode
            _init_worker(
                ordered_lookup, unordered_lookup,
                ref_base_counts, chromosomes,
            )
            for bcf_str in bcf_path_strs:
                result = _worker_fn(bcf_str)
                n_done += 1
                _handle_result(
                    result, tsv_f, json_dir, all_symbols,
                    has_genome_counts, n_done, n_total, t_start,
                )
                if "error" in result:
                    n_errors += 1
        else:
            with Pool(
                n_workers,
                initializer=_init_worker,
                initargs=(
                    ordered_lookup, unordered_lookup,
                    ref_base_counts, chromosomes,
                ),
            ) as pool:
                for result in pool.imap_unordered(
                    _worker_fn, bcf_path_strs,
                ):
                    n_done += 1
                    _handle_result(
                        result, tsv_f, json_dir, all_symbols,
                        has_genome_counts, n_done, n_total, t_start,
                    )
                    if "error" in result:
                        n_errors += 1

    elapsed = time.time() - t_start
    print(
        f"\nDone. Processed {n_done} genomes in {elapsed:.1f}s "
        f"({n_errors} errors). Output: {tsv_path}"
    )


def _handle_result(
    result: dict,
    tsv_f,
    json_dir: Path,
    all_symbols: List[str],
    has_genome_counts: bool,
    n_done: int,
    n_total: int,
    t_start: float,
) -> None:
    """Write a single result to TSV and JSON, print progress."""
    sample = result.get("sample_name", "unknown")

    # Write per-genome JSON
    json_path = json_dir / f"{sample}.json"
    with open(json_path, "w") as jf:
        json.dump(result, jf, indent=2)

    # Write TSV row
    row = _result_to_tsv_row(result, all_symbols, has_genome_counts)
    if row:
        tsv_f.write(row + "\n")
        tsv_f.flush()

    # Progress
    elapsed = time.time() - t_start
    rate = n_done / elapsed if elapsed > 0 else 0
    eta = (n_total - n_done) / rate if rate > 0 else 0
    status = "ERROR" if "error" in result else "ok"
    print(
        f"[{n_done}/{n_total}] {sample} ({status}) "
        f"- {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather diploid token usage statistics from single-sample "
                    "BCFs (for an assembled dataset, use "
                    "huggingface_dataset/tools/gather_dataset_stats.py).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--bcf-dir",
        type=Path,
        help="Directory containing single-sample BCF files.",
    )
    group.add_argument(
        "--bcf",
        type=Path,
        help="Path to a single BCF file.",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=None,
        help="Path to reference genome (2bit or FASTA). "
             f"Default: {paths.HG38_2BIT}",
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=None,
        help="Path to diploid_tokens.tsv. "
             f"Default: {paths.DIPLOID_TOKENS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    parser.add_argument(
        "--variant-only",
        action="store_true",
        help="Only count tokens at variant sites (skip reference "
             "composition — faster, no reference genome needed).",
    )

    args = parser.parse_args()

    ref_path = args.ref or paths.HG38_2BIT
    tokens_path = args.tokens or paths.DIPLOID_TOKENS

    if not args.variant_only and not ref_path.exists():
        parser.error(
            f"Reference genome not found at '{ref_path}'. "
            f"Provide --ref or use --variant-only."
        )
    if not tokens_path.exists():
        parser.error(f"Diploid tokens file not found at '{tokens_path}'.")

    # Collect BCF files
    if args.bcf:
        if not args.bcf.exists():
            parser.error(f"BCF file not found: {args.bcf}")
        bcf_paths = [args.bcf]
    else:
        if not args.bcf_dir.exists() or not args.bcf_dir.is_dir():
            parser.error(f"BCF directory not found: {args.bcf_dir}")
        bcf_paths = sorted(args.bcf_dir.glob("*.bcf"))
        # Exclude index files
        bcf_paths = [p for p in bcf_paths if not p.name.endswith(".csi")]
        if not bcf_paths:
            parser.error(f"No BCF files found in {args.bcf_dir}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Processing {len(bcf_paths)} genome(s) with {args.workers} workers")
    print(f"Reference: {ref_path}")
    print(f"Tokens: {tokens_path}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {'variant-only' if args.variant_only else 'genome-wide'}")
    print()

    process_batch(
        bcf_paths=bcf_paths,
        ref_path=ref_path if not args.variant_only else None,
        tokens_path=tokens_path,
        output_dir=args.output_dir,
        n_workers=args.workers,
        variant_only=args.variant_only,
    )


if __name__ == "__main__":
    main()
