"""One BED line reader for preprocess/ (reorg step 8c).

:func:`iter_bed_fields` owns the part every BED reader shares: opening (optionally gzip),
stripping, skipping blank / ``#`` / ``track`` / ``browser`` lines, and splitting into
fields. Validation stays with each caller, because their policies differ on purpose —
the encoder and :func:`parse_bed_regions` raise, ``create_dataset.load_bed_regions``
warns and skips, ``nested_genome_division.sources.read_bed`` skips silently and
``validate_division.read_division_bed`` raises with its own message. Pre-reorg quirks are
kept until the post-reorg follow-ups (``preprocess/docs/REORG_PLAN_REV1.md`` §Follow-up list): each caller
decides gzip itself (``compressed``), and :func:`parse_bed_regions` still treats only a
``#`` in column 1 as a comment (``comment_at_col0_only``).

Two readers keep their own loop because the shared predicate would change what they accept:
``create_haplotype_dataset.load_bed6_regions`` (``#region_set=`` / ``#chrom`` header lines,
unstripped rows) and ``chunk_union_bed.read_union_bed`` (splits the raw, unstripped line;
stdlib-only so it runs by path); ``build_artifact_data.read_committed_bed`` skips nothing.
"""

import gzip
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

logger = logging.getLogger(__name__)

Split = Literal["tab", "ws", "tab-then-ws"]


def iter_bed_fields(
    path: Union[str, Path],
    *,
    split: Split = "tab",
    skip_track_browser: bool = True,
    compressed: bool = False,
    comment_at_col0_only: bool = False,
) -> Iterator[Tuple[int, str, List[str]]]:
    """Yield ``(lineno, line, fields)`` for every data line of a BED file.

    ``line`` is stripped. Blank lines and ``#`` comments are skipped, and so are
    ``track`` / ``browser`` lines unless ``skip_track_browser`` is False. With
    ``comment_at_col0_only`` an indented ``#`` line is data, not a comment (the
    pre-reorg rule of :func:`parse_bed_regions`). ``split``: ``"tab"`` splits on tabs,
    ``"ws"`` on any whitespace, ``"tab-then-ws"`` on tabs and falls back to whitespace
    when that gives fewer than three fields (space-delimited BEDs). ``compressed``
    opens the file with gzip. Field counts and coordinates are not checked here.
    """
    if split not in ("tab", "ws", "tab-then-ws"):
        raise ValueError(f"unknown split mode {split!r}")
    return _iter_bed_fields(path, split, skip_track_browser, compressed, comment_at_col0_only)


def _iter_bed_fields(path, split, skip_track_browser, compressed, comment_at_col0_only):
    opener = gzip.open if compressed else open
    with opener(path, "rt") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if (
                not line
                or (raw if comment_at_col0_only else line).startswith("#")
                or (skip_track_browser and line.startswith(("track", "browser")))
            ):
                continue
            if split == "ws":
                fields = line.split()
            else:
                fields = line.split("\t")
                if split == "tab-then-ws" and len(fields) < 3:
                    fields = line.split()
            yield lineno, line, fields


def parse_bed_regions(bed_file_path: str, target_chrom: Optional[str] = None) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parses a BED file (plain or gzipped) and returns a dictionary mapping
    chromosome names to lists of (start, end) tuples.
    If target_chrom is specified, only regions for that chromosome are returned.
    If bed_file_path is None, returns an empty dictionary.

    Raises:
        FileNotFoundError: If the BED file does not exist.
        ValueError: If a BED line has fewer than 3 fields or invalid coordinates.
        RuntimeError: If the file cannot be read.
    """
    if not bed_file_path:
        return {}

    all_regions: Dict[str, List[Tuple[int, int]]] = {}
    compressed = bed_file_path.endswith('.gz')

    try:
        for line_num, line, fields in iter_bed_fields(
            bed_file_path, split="ws", skip_track_browser=False,
            compressed=compressed, comment_at_col0_only=True,
        ):
            if len(fields) < 3:
                raise ValueError(
                    f"BED file '{bed_file_path}' line {line_num}: "
                    f"expected at least 3 fields (chrom, start, end), "
                    f"got {len(fields)}: {line!r}"
                )

            chrom = fields[0]
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as e:
                raise ValueError(
                    f"BED file '{bed_file_path}' line {line_num}: "
                    f"invalid coordinates — {e}"
                ) from e

            if target_chrom and chrom != target_chrom:
                continue  # Skip regions not on the target chromosome

            all_regions.setdefault(chrom, []).append((start, end))

    except (ValueError, FileNotFoundError):
        raise
    except Exception as e:
        raise RuntimeError(
            f"Error reading BED file '{bed_file_path}': {e}"
        ) from e

    # If target_chrom was specified, but not found, log a warning
    if target_chrom and target_chrom not in all_regions:
        logger.warning(
            "Chromosome '%s' not found in BED file '%s'.",
            target_chrom, bed_file_path,
        )

    return all_regions
