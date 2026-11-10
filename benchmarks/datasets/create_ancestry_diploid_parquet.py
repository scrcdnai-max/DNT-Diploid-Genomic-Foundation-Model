import argparse
import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

try:
    import pysam
except ImportError:
    pysam = None

try:
    from pyfaidx import Fasta
except ImportError:
    Fasta = None


# =============================================================================
# Configuration
# =============================================================================

ROOT_DATA_DIR = "../data"
DATASET_NAME = "diploid_ancestry_mhc"
DATASET_DIR = os.path.join(ROOT_DATA_DIR, DATASET_NAME)

# Base shards will be named:
#
#   diploid_ancestry_mhc_0.parquet
#   diploid_ancestry_mhc_1.parquet
#   diploid_ancestry_mhc_2.parquet
#   ...
#
DEFAULT_BASE_PREFIX = os.path.join(
    DATASET_DIR,
    "diploid_ancestry_mhc",
)

# Compatibility with the single-file version.
LEGACY_SINGLE_PARQUET = os.path.join(
    DATASET_DIR,
    "diploid_ancestry_mhc_windows.parquet",
)

DOWNLOAD_DIR = os.path.join(
    DATASET_DIR,
    "downloads",
)

REFERENCE_DIR = os.path.join(
    DOWNLOAD_DIR,
    "reference",
)

VCF_DIR = os.path.join(
    DOWNLOAD_DIR,
    "vcf",
)

METADATA_DIR = os.path.join(
    DOWNLOAD_DIR,
    "metadata",
)


# =============================================================================
# Chunking
# =============================================================================

# Maximum number of ROWS in each generated Parquet shard.
#
# This is now a real row-based chunk size.
#
# Can be overridden with:
#
#   --chunk-size 2000
#
CHUNK_SIZE = 200_000

# Number of samples reconstructed in memory at once while generating sequences.
# This is independent of the number of rows written per Parquet shard.
SAMPLE_BATCH_SIZE = 100

# Number of Parquet rows read at once while splitting existing shards.
PARQUET_READ_BATCH_SIZE = 128


# =============================================================================
# Stable public IGSR / 1000 Genomes resources
# =============================================================================

REFERENCE_FASTA_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/"
    "GRCh38_reference_genome/"
    "GRCh38_full_analysis_set_plus_decoy_hla.fa"
)

REFERENCE_FASTA_FAI_URL = (
    REFERENCE_FASTA_URL + ".fai"
)

CHR6_VCF_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/working/"
    "20201028_3202_phased/"
    "CCDG_14151_B01_GRM_WGS_2020-08-05_chr6.filtered."
    "shapeit2-duohmm-phased.vcf.gz"
)

CHR6_VCF_TBI_URL = (
    CHR6_VCF_URL + ".tbi"
)

SAMPLE_METADATA_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/"
    "20130606_g1k_3202_samples_ped_population.txt"
)


REFERENCE_FASTA_PATH = os.path.join(
    REFERENCE_DIR,
    "GRCh38_full_analysis_set_plus_decoy_hla.fa",
)

REFERENCE_FASTA_FAI_PATH = (
    REFERENCE_FASTA_PATH + ".fai"
)

VCF_PATH = os.path.join(
    VCF_DIR,
    "CCDG_14151_B01_GRM_WGS_2020-08-05_chr6.filtered."
    "shapeit2-duohmm-phased.vcf.gz",
)

VCF_TBI_PATH = (
    VCF_PATH + ".tbi"
)

SAMPLE_METADATA_PATH = os.path.join(
    METADATA_DIR,
    "20130606_g1k_3202_samples_ped_population.txt",
)


# =============================================================================
# Benchmark configuration
# =============================================================================

MHC_CHROM = "6"
MHC_START_1BASED = 28_510_120
MHC_END_1BASED = 33_480_577

WINDOW_SIZE = 450_000
N_WINDOWS = 11

VALID_BASES = set("ACGT")

SUPERPOP_TO_LABEL = {
    "AFR": 0,
    "EUR": 1,
    "AMR": 2,
    "SAS": 3,
    "EAS": 4,
}

LABEL_TO_SUPERPOP = {
    value: key
    for key, value in SUPERPOP_TO_LABEL.items()
}

SNVS_ONLY = True
DROP_NON_ACGT_WINDOWS = True

# Leave False to retain the full phased sample collection.
DROP_RELATED_INDIVIDUALS = False


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class GenomicWindow:
    window_id: int
    chrom: str
    start_1based: int
    end_1based: int

    @property
    def length(self) -> int:
        return (
            self.end_1based
            - self.start_1based
            + 1
        )

    @property
    def name(self) -> str:
        return (
            f"{self.chrom}:"
            f"{self.start_1based}-"
            f"{self.end_1based}"
        )


@dataclass
class SampleInfo:
    sample_id: str
    superpopulation: str
    label: int


# =============================================================================
# Dependency helpers
# =============================================================================

def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required.\n"
            "Install with:\n"
            "    pip install pyarrow"
        ) from exc

    return pa, pc, pq


def _require_generation_dependencies() -> None:
    """
    pysam and pyfaidx are only needed if the base dataset must be generated.

    If existing Parquet shards are found, they are not required.
    """

    if pysam is None:
        raise ImportError(
            "pysam is required to generate the base dataset.\n"
            "Install with:\n"
            "    pip install pysam"
        )

    if Fasta is None:
        raise ImportError(
            "pyfaidx is required to generate the base dataset.\n"
            "Install with:\n"
            "    pip install pyfaidx"
        )


# =============================================================================
# Basic file utilities
# =============================================================================

def _is_nonempty_file(
    path: str,
) -> bool:

    return (
        os.path.isfile(path)
        and os.path.getsize(path) > 0
    )


def _download_file(
    url: str,
    dest: str,
) -> None:

    if _is_nonempty_file(dest):
        print(
            f"Already exists: {dest}"
        )
        return

    os.makedirs(
        os.path.dirname(dest),
        exist_ok=True,
    )

    print(
        f"Downloading: {url}"
    )

    result = subprocess.run(
        [
            "wget",
            "-c",
            "-O",
            dest,
            url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Download failed for {url}\n"
            f"{result.stderr}"
        )

    print(
        f"Saved: {dest}"
    )


def _download_required_resources() -> None:

    _download_file(
        REFERENCE_FASTA_URL,
        REFERENCE_FASTA_PATH,
    )

    _download_file(
        REFERENCE_FASTA_FAI_URL,
        REFERENCE_FASTA_FAI_PATH,
    )

    _download_file(
        CHR6_VCF_URL,
        VCF_PATH,
    )

    _download_file(
        CHR6_VCF_TBI_URL,
        VCF_TBI_PATH,
    )

    _download_file(
        SAMPLE_METADATA_URL,
        SAMPLE_METADATA_PATH,
    )


# =============================================================================
# Parquet shard utilities
# =============================================================================

def _shard_path(
    prefix: str,
    shard_index: int,
) -> str:

    return (
        f"{prefix}_{shard_index}.parquet"
    )


def _shard_index(
    path: str,
    prefix: str,
) -> Optional[int]:
    """
    Return N for:

        <prefix>_N.parquet

    Otherwise return None.
    """

    pattern = (
        "^"
        + re.escape(
            os.path.abspath(prefix)
        )
        + r"_(\d+)\.parquet$"
    )

    match = re.match(
        pattern,
        os.path.abspath(path),
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


def discover_numbered_shards(
    prefix: str,
) -> List[str]:
    """
    Find exactly:

        <prefix>_0.parquet
        <prefix>_1.parquet
        ...

    Files are numerically sorted.
    """

    candidates = glob.glob(
        f"{prefix}_*.parquet"
    )

    indexed = []

    for path in candidates:

        index = _shard_index(
            path,
            prefix,
        )

        if index is None:
            continue

        if not _is_nonempty_file(
            path
        ):
            continue

        indexed.append(
            (
                index,
                path,
            )
        )

    indexed.sort(
        key=lambda x: x[0]
    )

    return [
        path
        for _, path in indexed
    ]


def remove_numbered_shards(
    prefix: str,
) -> None:

    shards = discover_numbered_shards(
        prefix
    )

    for path in shards:

        print(
            f"Removing old shard: {path}"
        )

        os.remove(
            path
        )

    # Also remove incomplete files left by a failed run.
    for path in glob.glob(
        f"{prefix}_*.parquet.incomplete"
    ):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def expand_parquet_arguments(
    values: Sequence[str],
) -> List[str]:
    """
    Allow either explicit paths:

        --base-parquets a.parquet b.parquet

    or quoted glob expressions:

        --base-parquets "/data/diploid_ancestry_mhc_*.parquet"
    """

    paths = []

    for value in values:

        expanded = glob.glob(
            value
        )

        if expanded:
            paths.extend(
                expanded
            )
        else:
            paths.append(
                value
            )

    # Remove duplicates.
    paths = list(
        dict.fromkeys(
            os.path.abspath(path)
            for path in paths
        )
    )

    missing = [
        path
        for path in paths
        if not _is_nonempty_file(path)
    ]

    if missing:

        raise FileNotFoundError(
            "The following Parquet file(s) "
            "do not exist or are empty:\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )

    return paths


# =============================================================================
# Generic sharded Parquet writer
# =============================================================================

class ShardedParquetWriter:
    """
    Write Arrow tables to numbered Parquet files:

        prefix_0.parquet
        prefix_1.parquet
        ...

    Each shard contains at most chunk_size rows.

    Files are first written as:

        prefix_0.parquet.incomplete

    They are renamed to final .parquet names only after the complete operation
    succeeds.
    """

    def __init__(
        self,
        prefix: str,
        schema,
        chunk_size: int,
        compression: str = "zstd",
    ):

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be > 0"
            )

        _, _, pq = _require_pyarrow()

        self.pq = pq
        self.prefix = prefix
        self.schema = schema
        self.chunk_size = chunk_size
        self.compression = compression

        self.shard_index = 0

        self.current_writer = None
        self.current_rows = 0

        self.pending_files: List[
            Tuple[str, str]
        ] = []

        os.makedirs(
            os.path.dirname(
                os.path.abspath(prefix)
            ),
            exist_ok=True,
        )


    def _open_writer(
        self,
    ) -> None:

        if self.current_writer is not None:
            return

        final_path = _shard_path(
            self.prefix,
            self.shard_index,
        )

        tmp_path = (
            final_path
            + ".incomplete"
        )

        if os.path.exists(
            tmp_path
        ):
            os.remove(
                tmp_path
            )

        self.current_writer = (
            self.pq.ParquetWriter(
                tmp_path,
                schema=self.schema,
                compression=self.compression,
            )
        )

        self.current_rows = 0

        self.pending_files.append(
            (
                tmp_path,
                final_path,
            )
        )


    def _finish_current_writer(
        self,
    ) -> None:

        if self.current_writer is None:
            return

        self.current_writer.close()

        self.current_writer = None
        self.current_rows = 0

        self.shard_index += 1


    def write_table(
        self,
        table,
    ) -> None:

        if table.num_rows == 0:
            return

        # Force metadata-compatible schema.
        if table.schema != self.schema:
            table = table.cast(
                self.schema
            )

        offset = 0

        while offset < table.num_rows:

            if self.current_writer is None:
                self._open_writer()

            remaining_capacity = (
                self.chunk_size
                - self.current_rows
            )

            rows_remaining = (
                table.num_rows
                - offset
            )

            n_rows = min(
                remaining_capacity,
                rows_remaining,
            )

            piece = table.slice(
                offset,
                n_rows,
            )

            self.current_writer.write_table(
                piece
            )

            self.current_rows += (
                n_rows
            )

            offset += (
                n_rows
            )

            if (
                self.current_rows
                >= self.chunk_size
            ):
                self._finish_current_writer()


    def close(
        self,
    ) -> None:

        self._finish_current_writer()


    def commit(
        self,
    ) -> List[str]:
        """
        Rename all .incomplete files to final .parquet names.
        """

        self.close()

        final_paths = []

        for (
            tmp_path,
            final_path,
        ) in self.pending_files:

            if not os.path.exists(
                tmp_path
            ):
                continue

            os.replace(
                tmp_path,
                final_path,
            )

            final_paths.append(
                final_path
            )

        return final_paths


    def abort(
        self,
    ) -> None:

        try:
            if self.current_writer is not None:
                self.current_writer.close()
        except Exception:
            pass

        self.current_writer = None

        for (
            tmp_path,
            _,
        ) in self.pending_files:

            try:
                if os.path.exists(
                    tmp_path
                ):
                    os.remove(
                        tmp_path
                    )
            except Exception:
                pass


# =============================================================================
# Benchmark generation
# =============================================================================

def _build_windows() -> List[GenomicWindow]:

    windows: List[
        GenomicWindow
    ] = []

    for i in range(
        N_WINDOWS
    ):

        start = (
            MHC_START_1BASED
            + i * WINDOW_SIZE
        )

        end = (
            start
            + WINDOW_SIZE
            - 1
        )

        windows.append(
            GenomicWindow(
                window_id=i + 1,
                chrom=MHC_CHROM,
                start_1based=start,
                end_1based=end,
            )
        )

    return windows


def _read_sample_metadata(
    path: str,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        comment="#",
        header=None,
    )

    if df.shape[1] < 7:
        raise ValueError(
            "Unexpected metadata format: "
            f"expected at least 7 columns, "
            f"got {df.shape[1]}"
        )

    df = df.iloc[
        :,
        :7,
    ].copy()

    df.columns = [
        "family_id",
        "sample_id",
        "paternal_id",
        "maternal_id",
        "population",
        "superpopulation",
    ]

    df["sample_id"] = (
        df["sample_id"]
        .astype(str)
    )

    df["population"] = (
        df["population"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["superpopulation"] = (
        df["superpopulation"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df[
        df["superpopulation"].isin(
            SUPERPOP_TO_LABEL
        )
    ].copy()

    if DROP_RELATED_INDIVIDUALS:

        founder_mask = (
            df["paternal_id"]
            .astype(str)
            .isin(
                {
                    "0",
                    "0.0",
                }
            )
        ) & (
            df["maternal_id"]
            .astype(str)
            .isin(
                {
                    "0",
                    "0.0",
                }
            )
        )

        df = df[
            founder_mask
        ].copy()

    df["label"] = (
        df["superpopulation"]
        .map(
            SUPERPOP_TO_LABEL
        )
        .astype(int)
    )

    df = (
        df
        .drop_duplicates(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "Loaded sample metadata"
    )

    print(
        df["superpopulation"]
        .value_counts()
        .sort_index()
    )

    print(
        "Samples after metadata filtering: "
        f"{len(df)}"
    )

    return df


def _load_reference_sequence(
    fasta: Any,
    chrom: str,
    start_1based: int,
    end_1based: int,
) -> str:

    value = fasta[
        chrom
    ][
        start_1based - 1:
        end_1based
    ]

    if isinstance(
        value,
        str,
    ):
        seq = value
    else:
        seq = value.seq

    return str(
        seq
    ).upper()


def _sample_names_from_vcf(
    vcf: Any,
) -> List[str]:

    return list(
        vcf.header.samples
    )


def _intersect_samples(
    vcf_samples: Sequence[str],
    meta_df: pd.DataFrame,
) -> List[SampleInfo]:

    vcf_set = set(
        vcf_samples
    )

    rows = meta_df[
        meta_df["sample_id"].isin(
            vcf_set
        )
    ].copy()

    samples = [
        SampleInfo(
            sample_id=row.sample_id,
            superpopulation=row.superpopulation,
            label=int(row.label),
        )
        for row
        in rows.itertuples(
            index=False
        )
    ]

    samples.sort(
        key=lambda x: x.sample_id
    )

    print(
        f"Samples in metadata: "
        f"{len(meta_df)}"
    )

    print(
        f"Samples in VCF: "
        f"{len(vcf_samples)}"
    )

    print(
        f"Intersected samples: "
        f"{len(samples)}"
    )

    return samples


def _gt_to_haplotype_allele_indices(
    gt,
) -> Optional[
    Tuple[int, int]
]:

    if gt is None:
        return None

    if len(gt) != 2:
        return None

    if (
        gt[0] is None
        or gt[1] is None
    ):
        return None

    return (
        int(gt[0]),
        int(gt[1]),
    )


def _fetch_records_for_window(
    vcf: Any,
    window: GenomicWindow,
):

    chrom_candidates = [
        window.chrom,
        f"chr{window.chrom}",
    ]

    for chrom in chrom_candidates:

        try:

            return list(
                vcf.fetch(
                    chrom,
                    window.start_1based - 1,
                    window.end_1based,
                )
            )

        except ValueError:
            continue

    raise ValueError(
        "Could not fetch chromosome "
        f"{window.chrom} or "
        f"chr{window.chrom} from VCF"
    )


def _apply_variants_to_window(
    ref_seq: str,
    window: GenomicWindow,
    records,
    sample_name: str,
) -> Tuple[
    Optional[str],
    Optional[str],
    List[str],
    List[str],
]:

    hap_a = list(
        ref_seq
    )

    hap_b = list(
        ref_seq
    )

    hap_a_vars: List[str] = []
    hap_b_vars: List[str] = []

    for rec in records:

        sample_call = (
            rec.samples[
                sample_name
            ]
        )

        gt = sample_call.get(
            "GT"
        )

        if not sample_call.phased:
            continue

        gt_pair = (
            _gt_to_haplotype_allele_indices(
                gt
            )
        )

        if gt_pair is None:
            continue

        pos = int(
            rec.pos
        )

        ref = str(
            rec.ref
        ).upper()

        alts = tuple(
            str(a).upper()
            for a
            in (
                rec.alts
                or ()
            )
        )

        if len(ref) != 1:
            continue

        if (
            SNVS_ONLY
            and any(
                len(a) != 1
                for a in alts
            )
        ):
            continue

        if ref not in VALID_BASES:
            continue

        if any(
            len(a) != 1
            or a not in VALID_BASES
            for a in alts
        ):
            continue

        offset = (
            pos
            - window.start_1based
        )

        if (
            offset < 0
            or offset >= len(ref_seq)
        ):
            continue

        if (
            ref_seq[offset]
            != ref
        ):
            continue

        allele_strings = [
            ref
        ] + list(
            alts
        )

        a_idx, b_idx = (
            gt_pair
        )

        if (
            a_idx < 0
            or b_idx < 0
        ):
            continue

        if (
            a_idx >= len(
                allele_strings
            )
            or b_idx >= len(
                allele_strings
            )
        ):
            continue

        a_allele = (
            allele_strings[
                a_idx
            ]
        )

        b_allele = (
            allele_strings[
                b_idx
            ]
        )

        if (
            a_allele
            not in VALID_BASES
            or b_allele
            not in VALID_BASES
        ):
            continue

        hap_a[offset] = (
            a_allele
        )

        hap_b[offset] = (
            b_allele
        )

        if a_idx != 0:

            hap_a_vars.append(
                f"{window.chrom}:"
                f"{pos}:"
                f"{ref}:"
                f"{a_allele}"
            )

        if b_idx != 0:

            hap_b_vars.append(
                f"{window.chrom}:"
                f"{pos}:"
                f"{ref}:"
                f"{b_allele}"
            )

    hap_a_seq = "".join(
        hap_a
    )

    hap_b_seq = "".join(
        hap_b
    )

    if DROP_NON_ACGT_WINDOWS:

        if not set(
            hap_a_seq
        ).issubset(
            VALID_BASES
        ):

            return (
                None,
                None,
                [],
                [],
            )

        if not set(
            hap_b_seq
        ).issubset(
            VALID_BASES
        ):

            return (
                None,
                None,
                [],
                [],
            )

    return (
        hap_a_seq,
        hap_b_seq,
        hap_a_vars,
        hap_b_vars,
    )


def _create_rows_for_sample_batch(
    ref_seq: str,
    window: GenomicWindow,
    records,
    samples: Sequence[SampleInfo],
) -> pd.DataFrame:

    rows: List[
        Dict[str, Any]
    ] = []

    for sample in samples:

        (
            hap_a_seq,
            hap_b_seq,
            hap_a_vars,
            hap_b_vars,
        ) = _apply_variants_to_window(
            ref_seq=ref_seq,
            window=window,
            records=records,
            sample_name=sample.sample_id,
        )

        if (
            hap_a_seq is None
            or hap_b_seq is None
        ):
            continue

        n_het_sites = 0
        n_hom_alt_sites = 0

        for (
            a,
            b,
            ref,
        ) in zip(
            hap_a_seq,
            hap_b_seq,
            ref_seq,
        ):

            if (
                a != ref
                and b != ref
                and a == b
            ):

                n_hom_alt_sites += 1

            elif (
                (a != ref)
                != (b != ref)
            ):

                n_het_sites += 1

        rows.append(
            {
                "sample_id":
                    sample.sample_id,

                "superpopulation":
                    sample.superpopulation,

                "label":
                    sample.label,

                "chrom":
                    window.chrom,

                "window_id":
                    window.window_id,

                "window_start":
                    window.start_1based,

                "window_end":
                    window.end_1based,

                "window_length":
                    window.length,

                "sequence_length":
                    len(hap_a_seq),

                "reference_seq":
                    ref_seq,

                "haplotype_a_seq":
                    hap_a_seq,

                "haplotype_b_seq":
                    hap_b_seq,

                "haplotype_a_variants":
                    hap_a_vars,

                "haplotype_b_variants":
                    hap_b_vars,

                "n_alt_hap_a":
                    len(hap_a_vars),

                "n_alt_hap_b":
                    len(hap_b_vars),

                "n_het_sites":
                    n_het_sites,

                "n_hom_alt_sites":
                    n_hom_alt_sites,
            }
        )

    return pd.DataFrame(
        rows
    )


def _validate_batch(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:
        return df

    bad_len = (
        (
            df["window_length"]
            != WINDOW_SIZE
        )
        |
        (
            df["sequence_length"]
            != WINDOW_SIZE
        )
        |
        (
            df["haplotype_a_seq"]
            .str.len()
            != WINDOW_SIZE
        )
        |
        (
            df["haplotype_b_seq"]
            .str.len()
            != WINDOW_SIZE
        )
    )

    if bad_len.any():

        raise ValueError(
            f"Found "
            f"{int(bad_len.sum())} "
            "rows with invalid "
            "sequence length"
        )

    return df


def _base_arrow_schema(
    pa,
):

    return pa.schema(
        [
            pa.field(
                "sample_id",
                pa.string(),
            ),

            pa.field(
                "superpopulation",
                pa.string(),
            ),

            pa.field(
                "label",
                pa.int64(),
            ),

            pa.field(
                "chrom",
                pa.string(),
            ),

            pa.field(
                "window_id",
                pa.int64(),
            ),

            pa.field(
                "window_start",
                pa.int64(),
            ),

            pa.field(
                "window_end",
                pa.int64(),
            ),

            pa.field(
                "window_length",
                pa.int64(),
            ),

            pa.field(
                "sequence_length",
                pa.int64(),
            ),

            pa.field(
                "reference_seq",
                pa.string(),
            ),

            pa.field(
                "haplotype_a_seq",
                pa.string(),
            ),

            pa.field(
                "haplotype_b_seq",
                pa.string(),
            ),

            pa.field(
                "haplotype_a_variants",
                pa.list_(
                    pa.string()
                ),
            ),

            pa.field(
                "haplotype_b_variants",
                pa.list_(
                    pa.string()
                ),
            ),

            pa.field(
                "n_alt_hap_a",
                pa.int64(),
            ),

            pa.field(
                "n_alt_hap_b",
                pa.int64(),
            ),

            pa.field(
                "n_het_sites",
                pa.int64(),
            ),

            pa.field(
                "n_hom_alt_sites",
                pa.int64(),
            ),
        ]
    )


# =============================================================================
# Create numbered base Parquet shards
# =============================================================================

def create_base_dataset(
    output_prefix: str,
    chunk_size: int,
    sample_batch_size: int,
) -> List[str]:
    """
    Create:

        <output_prefix>_0.parquet
        <output_prefix>_1.parquet
        ...

    Each Parquet contains at most `chunk_size` rows.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be > 0"
        )

    if sample_batch_size <= 0:
        raise ValueError(
            "sample_batch_size must be > 0"
        )

    _require_generation_dependencies()

    pa, _, _ = (
        _require_pyarrow()
    )

    _download_required_resources()

    # Prevent old extra shards from surviving a rebuild.
    remove_numbered_shards(
        output_prefix
    )

    windows = (
        _build_windows()
    )

    meta_df = (
        _read_sample_metadata(
            SAMPLE_METADATA_PATH
        )
    )

    fasta = Fasta(
        REFERENCE_FASTA_PATH,
        as_raw=True,
        sequence_always_upper=True,
    )

    vcf = pysam.VariantFile(
        VCF_PATH
    )

    samples = (
        _intersect_samples(
            _sample_names_from_vcf(
                vcf
            ),
            meta_df,
        )
    )

    if not samples:
        raise ValueError(
            "No overlapping samples "
            "between metadata and VCF"
        )

    schema = (
        _base_arrow_schema(
            pa
        )
    )

    writer = (
        ShardedParquetWriter(
            prefix=output_prefix,
            schema=schema,
            chunk_size=chunk_size,
            compression="zstd",
        )
    )

    rows_written = 0

    try:

        for window in windows:

            print(
                f"\nProcessing window "
                f"{window.window_id}/"
                f"{len(windows)}: "
                f"{window.name}"
            )

            ref_seq = (
                _load_reference_sequence(
                    fasta=fasta,
                    chrom=window.chrom,
                    start_1based=
                        window.start_1based,
                    end_1based=
                        window.end_1based,
                )
            )

            if (
                len(ref_seq)
                != WINDOW_SIZE
            ):

                raise ValueError(
                    "Reference length "
                    f"mismatch in "
                    f"{window.name}: "
                    f"{len(ref_seq)} != "
                    f"{WINDOW_SIZE}"
                )

            records = (
                _fetch_records_for_window(
                    vcf,
                    window,
                )
            )

            print(
                f"  VCF records in window: "
                f"{len(records)}"
            )

            for start in range(
                0,
                len(samples),
                sample_batch_size,
            ):

                sample_batch = (
                    samples[
                        start:
                        start
                        + sample_batch_size
                    ]
                )

                end = (
                    start
                    + len(sample_batch)
                )

                print(
                    f"  samples "
                    f"{start + 1}-"
                    f"{end}/"
                    f"{len(samples)}"
                )

                df = (
                    _create_rows_for_sample_batch(
                        ref_seq=ref_seq,
                        window=window,
                        records=records,
                        samples=sample_batch,
                    )
                )

                df = (
                    _validate_batch(
                        df
                    )
                )

                if df.empty:
                    continue

                table = (
                    pa.Table.from_pandas(
                        df,
                        schema=schema,
                        preserve_index=False,
                        safe=True,
                    )
                )

                writer.write_table(
                    table
                )

                rows_written += (
                    table.num_rows
                )

        if rows_written == 0:

            writer.abort()

            raise ValueError(
                "Generated dataset "
                "contains zero rows"
            )

        shard_paths = (
            writer.commit()
        )

    except Exception:

        writer.abort()
        raise

    finally:

        try:
            vcf.close()
        except Exception:
            pass

    print(
        "\nBase dataset complete"
    )

    print(
        "---------------------"
    )

    print(
        f"Rows: {rows_written}"
    )

    print(
        f"Chunk size: "
        f"{chunk_size}"
    )

    print(
        f"Parquet shards: "
        f"{len(shard_paths)}"
    )

    for path in shard_paths:
        print(
            f"  {path}"
        )

    return shard_paths


# =============================================================================
# Reuse existing base Parquet shard(s)
# =============================================================================

def ensure_base_dataset(
    base_prefix: str,
    explicit_parquets: Optional[
        Sequence[str]
    ],
    chunk_size: int,
    sample_batch_size: int,
    force_rebuild: bool,
) -> List[str]:
    """
    Priority:

    1. Explicit --base-parquets
    2. Existing numbered shards matching --base-prefix
    3. Legacy single diploid_ancestry_mhc_windows.parquet
    4. Generate numbered shards
    """

    # -------------------------------------------------------------------------
    # Explicit Parquet file list
    # -------------------------------------------------------------------------

    if explicit_parquets:

        if force_rebuild:
            raise ValueError(
                "--base-parquets and "
                "--force-rebuild-base "
                "cannot be used together.\n"
                "For rebuilding, specify "
                "--base-prefix instead."
            )

        paths = (
            expand_parquet_arguments(
                explicit_parquets
            )
        )

        print(
            "Using explicitly supplied "
            "base Parquet file(s):"
        )

        for path in paths:
            print(
                f"  {path}"
            )

        print(
            "Skipping downloads and "
            "base reconstruction."
        )

        return paths


    # -------------------------------------------------------------------------
    # Force regeneration
    # -------------------------------------------------------------------------

    if force_rebuild:

        print(
            "Force rebuild requested."
        )

        return create_base_dataset(
            output_prefix=
                base_prefix,
            chunk_size=
                chunk_size,
            sample_batch_size=
                sample_batch_size,
        )


    # -------------------------------------------------------------------------
    # Automatically discover numbered shards
    # -------------------------------------------------------------------------

    shards = (
        discover_numbered_shards(
            base_prefix
        )
    )

    if shards:

        print(
            "Found existing numbered "
            "base Parquet shards:"
        )

        for path in shards:
            print(
                f"  {path}"
            )

        print(
            "Skipping downloads and "
            "base reconstruction."
        )

        return shards


    # -------------------------------------------------------------------------
    # Compatibility with previous single-Parquet version
    # -------------------------------------------------------------------------

    if (
        os.path.abspath(
            base_prefix
        )
        == os.path.abspath(
            DEFAULT_BASE_PREFIX
        )
        and _is_nonempty_file(
            LEGACY_SINGLE_PARQUET
        )
    ):

        print(
            "Found existing legacy "
            "single Parquet:"
        )

        print(
            f"  {LEGACY_SINGLE_PARQUET}"
        )

        print(
            "Skipping downloads and "
            "base reconstruction."
        )

        return [
            LEGACY_SINGLE_PARQUET
        ]


    # -------------------------------------------------------------------------
    # No data found
    # -------------------------------------------------------------------------

    print(
        "No existing base Parquet "
        "shards found."
    )

    print(
        "Generating dataset with "
        f"CHUNK_SIZE={chunk_size} rows."
    )

    return create_base_dataset(
        output_prefix=
            base_prefix,
        chunk_size=
            chunk_size,
        sample_batch_size=
            sample_batch_size,
    )


# =============================================================================
# ID file parsing
# =============================================================================

def read_sample_ids(
    path: str,
) -> List[str]:
    """
    Read one sample ID per line.

    Empty lines and # comments are ignored.
    """

    if not os.path.isfile(
        path
    ):

        raise FileNotFoundError(
            f"ID file not found: "
            f"{path}"
        )

    ids: List[str] = []

    seen: Set[str] = set()
    duplicates: Set[str] = set()

    # utf-8-sig safely handles BOM-containing text files.
    with open(
        path,
        "r",
        encoding="utf-8-sig",
    ) as handle:

        for line in handle:

            sample_id = (
                line.strip()
            )

            if not sample_id:
                continue

            if sample_id.startswith(
                "#"
            ):
                continue

            if sample_id in seen:

                duplicates.add(
                    sample_id
                )

                continue

            seen.add(
                sample_id
            )

            ids.append(
                sample_id
            )

    if not ids:

        raise ValueError(
            "No sample IDs found in: "
            f"{path}"
        )

    if duplicates:

        print(
            f"Warning: ignored "
            f"{len(duplicates)} "
            f"duplicate IDs in "
            f"{path}"
        )

    return ids


# =============================================================================
# Open and validate multiple Parquets
# =============================================================================

def _open_parquet_files(
    paths: Sequence[str],
):

    _, _, pq = (
        _require_pyarrow()
    )

    parquet_files = []

    for path in paths:

        if not _is_nonempty_file(
            path
        ):
            raise FileNotFoundError(
                f"Missing/empty Parquet: "
                f"{path}"
            )

        parquet_files.append(
            pq.ParquetFile(
                path
            )
        )

    return parquet_files


def _validate_parquet_schemas(
    paths: Sequence[str],
    parquet_files,
):
    """
    Ensure all shards have the same logical Arrow schema.

    Arrow metadata is ignored in this comparison.
    """

    if not parquet_files:
        raise ValueError(
            "No Parquet files supplied"
        )

    schema = (
        parquet_files[0]
        .schema_arrow
        .remove_metadata()
    )

    if (
        "sample_id"
        not in schema.names
    ):

        raise ValueError(
            "Base Parquet has no "
            "'sample_id' column.\n"
            f"Columns: {schema.names}"
        )

    for (
        path,
        parquet_file,
    ) in zip(
        paths[1:],
        parquet_files[1:],
    ):

        other_schema = (
            parquet_file
            .schema_arrow
            .remove_metadata()
        )

        if other_schema != schema:

            raise ValueError(
                "Schema mismatch between "
                "base Parquet shards.\n"
                f"Problematic file:\n"
                f"  {path}"
            )

    return schema


# =============================================================================
# Split multiple base shards by sample ID
# =============================================================================

def split_parquets_by_ids(
    base_parquets: Sequence[str],
    train_ids_path: str,
    test_ids_path: str,
    output_dir: str,
    split_chunk_size: int,
    parquet_read_batch_size: int,
    strict_ids: bool = False,
) -> Tuple[
    List[str],
    List[str],
]:
    """
    Stream ALL base Parquet shards and create:

        train_0.parquet
        train_1.parquet
        ...

        test_0.parquet
        test_1.parquet
        ...

    Each output shard contains at most split_chunk_size rows.
    """

    if split_chunk_size <= 0:
        raise ValueError(
            "split_chunk_size "
            "must be > 0"
        )

    if parquet_read_batch_size <= 0:
        raise ValueError(
            "parquet_read_batch_size "
            "must be > 0"
        )

    pa, pc, _ = (
        _require_pyarrow()
    )

    train_ids = (
        read_sample_ids(
            train_ids_path
        )
    )

    test_ids = (
        read_sample_ids(
            test_ids_path
        )
    )

    train_set = set(
        train_ids
    )

    test_set = set(
        test_ids
    )

    overlap = (
        train_set
        & test_set
    )

    if overlap:

        preview = ", ".join(
            sorted(
                overlap
            )[:20]
        )

        raise ValueError(
            "Train and test ID files "
            "overlap for "
            f"{len(overlap)} samples.\n"
            f"Examples: {preview}"
        )

    parquet_files = (
        _open_parquet_files(
            base_parquets
        )
    )

    schema = (
        _validate_parquet_schemas(
            base_parquets,
            parquet_files,
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    train_prefix = (
        os.path.join(
            output_dir,
            "train",
        )
    )

    test_prefix = (
        os.path.join(
            output_dir,
            "test",
        )
    )

    # Remove stale output shards from a previous split.
    remove_numbered_shards(
        train_prefix
    )

    remove_numbered_shards(
        test_prefix
    )

    train_writer = (
        ShardedParquetWriter(
            prefix=train_prefix,
            schema=schema,
            chunk_size=
                split_chunk_size,
            compression="zstd",
        )
    )

    test_writer = (
        ShardedParquetWriter(
            prefix=test_prefix,
            schema=schema,
            chunk_size=
                split_chunk_size,
            compression="zstd",
        )
    )

    train_values = pa.array(
        sorted(
            train_set
        ),
        type=pa.string(),
    )

    test_values = pa.array(
        sorted(
            test_set
        ),
        type=pa.string(),
    )

    found_train: Set[str] = set()
    found_test: Set[str] = set()
    all_seen: Set[str] = set()

    total_rows = 0
    train_rows = 0
    test_rows = 0

    try:

        for (
            file_index,
            (
                path,
                parquet_file,
            ),
        ) in enumerate(
            zip(
                base_parquets,
                parquet_files,
            ),
            start=1,
        ):

            print(
                f"\nReading base shard "
                f"{file_index}/"
                f"{len(base_parquets)}:"
            )

            print(
                f"  {path}"
            )

            for batch in (
                parquet_file
                .iter_batches(
                    batch_size=
                        parquet_read_batch_size
                )
            ):

                table = (
                    pa.Table.from_batches(
                        [batch]
                    )
                )

                if (
                    table.schema
                    .remove_metadata()
                    != schema
                ):

                    table = (
                        table.cast(
                            schema
                        )
                    )

                sample_col = (
                    table[
                        "sample_id"
                    ]
                )

                batch_ids = set(
                    sample_col
                    .to_pylist()
                )

                all_seen.update(
                    batch_ids
                )

                total_rows += (
                    table.num_rows
                )

                # -------------------------------------------------------------
                # Train rows
                # -------------------------------------------------------------

                train_mask = (
                    pc.is_in(
                        sample_col,
                        value_set=
                            train_values,
                    )
                )

                train_table = (
                    table.filter(
                        train_mask
                    )
                )

                if (
                    train_table.num_rows
                    > 0
                ):

                    train_writer.write_table(
                        train_table
                    )

                    train_rows += (
                        train_table.num_rows
                    )

                    found_train.update(
                        train_table[
                            "sample_id"
                        ].to_pylist()
                    )


                # -------------------------------------------------------------
                # Test rows
                # -------------------------------------------------------------

                test_mask = (
                    pc.is_in(
                        sample_col,
                        value_set=
                            test_values,
                    )
                )

                test_table = (
                    table.filter(
                        test_mask
                    )
                )

                if (
                    test_table.num_rows
                    > 0
                ):

                    test_writer.write_table(
                        test_table
                    )

                    test_rows += (
                        test_table.num_rows
                    )

                    found_test.update(
                        test_table[
                            "sample_id"
                        ].to_pylist()
                    )


        if train_rows == 0:
            raise ValueError(
                "Train split contains "
                "zero rows"
            )

        if test_rows == 0:
            raise ValueError(
                "Test split contains "
                "zero rows"
            )

        missing_train = (
            train_set
            - found_train
        )

        missing_test = (
            test_set
            - found_test
        )

        if (
            strict_ids
            and (
                missing_train
                or missing_test
            )
        ):

            raise ValueError(
                "Some requested IDs "
                "were absent from the "
                "base dataset.\n"
                f"Missing train: "
                f"{len(missing_train)}\n"
                f"Missing test: "
                f"{len(missing_test)}"
            )

        train_paths = (
            train_writer.commit()
        )

        test_paths = (
            test_writer.commit()
        )

    except Exception:

        train_writer.abort()
        test_writer.abort()

        raise


    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    missing_train = (
        train_set
        - found_train
    )

    missing_test = (
        test_set
        - found_test
    )

    excluded = (
        all_seen
        - train_set
        - test_set
    )

    print(
        "\nSplit summary"
    )

    print(
        "-------------"
    )

    print(
        f"Base Parquet shards:    "
        f"{len(base_parquets)}"
    )

    print(
        f"Base rows scanned:      "
        f"{total_rows}"
    )

    print(
        f"Unique base IDs seen:   "
        f"{len(all_seen)}"
    )

    print(
        f"Train IDs requested:    "
        f"{len(train_set)}"
    )

    print(
        f"Train IDs found:        "
        f"{len(found_train)}"
    )

    print(
        f"Train rows written:     "
        f"{train_rows}"
    )

    print(
        f"Train Parquet shards:   "
        f"{len(train_paths)}"
    )

    print(
        f"Test IDs requested:     "
        f"{len(test_set)}"
    )

    print(
        f"Test IDs found:         "
        f"{len(found_test)}"
    )

    print(
        f"Test rows written:      "
        f"{test_rows}"
    )

    print(
        f"Test Parquet shards:    "
        f"{len(test_paths)}"
    )

    print(
        f"Base IDs excluded:      "
        f"{len(excluded)}"
    )

    if missing_train:

        print(
            "\nWarning: "
            f"{len(missing_train)} "
            "train IDs were not found."
        )

        print(
            "Examples: "
            + ", ".join(
                sorted(
                    missing_train
                )[:20]
            )
        )

    if missing_test:

        print(
            "\nWarning: "
            f"{len(missing_test)} "
            "test IDs were not found."
        )

        print(
            "Examples: "
            + ", ".join(
                sorted(
                    missing_test
                )[:20]
            )
        )

    print(
        "\nTrain shards:"
    )

    for path in train_paths:
        print(
            f"  {path}"
        )

    print(
        "\nTest shards:"
    )

    for path in test_paths:
        print(
            f"  {path}"
        )

    return (
        train_paths,
        test_paths,
    )


# =============================================================================
# Hugging Face conversion
# =============================================================================

def convert_to_huggingface(
    train_parquets: Sequence[str],
    test_parquets: Sequence[str],
    hf_output_dir: str,
) -> str:
    """
    Create a Hugging Face DatasetDict from multiple Parquet shards.

    The Parquet files themselves are also directly Hugging Face-ready.
    """

    try:

        from datasets import (
            ClassLabel,
            DatasetDict,
            load_dataset,
        )

    except ImportError as exc:

        raise ImportError(
            "Hugging Face datasets "
            "is required.\n"
            "Install with:\n"
            "    pip install datasets"
        ) from exc


    print(
        "\nCreating Hugging Face "
        "DatasetDict..."
    )

    print(
        f"Train shards: "
        f"{len(train_parquets)}"
    )

    print(
        f"Test shards: "
        f"{len(test_parquets)}"
    )


    # Hugging Face natively accepts multiple Parquet files for each split.
    dataset = load_dataset(
        "parquet",
        data_files={
            "train":
                list(
                    train_parquets
                ),

            "test":
                list(
                    test_parquets
                ),
        },
    )


    # -------------------------------------------------------------------------
    # Attach meaningful class names to label
    # -------------------------------------------------------------------------

    label_names = [
        LABEL_TO_SUPERPOP[i]
        for i in range(
            len(
                LABEL_TO_SUPERPOP
            )
        )
    ]

    class_label = (
        ClassLabel(
            names=label_names
        )
    )

    converted = {}

    for (
        split_name,
        split_dataset,
    ) in dataset.items():

        converted[
            split_name
        ] = (
            split_dataset
            .cast_column(
                "label",
                class_label,
            )
        )

    dataset = (
        DatasetDict(
            converted
        )
    )


    # -------------------------------------------------------------------------
    # Save DatasetDict
    # -------------------------------------------------------------------------

    if os.path.exists(
        hf_output_dir
    ):

        shutil.rmtree(
            hf_output_dir
        )

    os.makedirs(
        os.path.dirname(
            os.path.abspath(
                hf_output_dir
            )
        ),
        exist_ok=True,
    )

    dataset.save_to_disk(
        hf_output_dir
    )

    print(
        "\nHugging Face dataset:"
    )

    print(
        dataset
    )

    print(
        "\nSaved to:"
    )

    print(
        f"  {hf_output_dir}"
    )

    return hf_output_dir


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Create or reuse the "
            "1000 Genomes diploid MHC "
            "ancestry benchmark, split it "
            "by individual ID, and optionally "
            "create a Hugging Face DatasetDict."
        )
    )


    # -------------------------------------------------------------------------
    # Existing base files
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--base-parquets",
        nargs="+",
        default=None,
        help=(
            "Optional explicit existing "
            "Parquet files or glob(s). "
            "Examples:\n"
            "  --base-parquets "
            "data_0.parquet data_1.parquet\n"
            "or\n"
            "  --base-parquets "
            "'diploid_ancestry_mhc_*.parquet'\n"
            "If omitted, numbered shards "
            "are discovered automatically."
        ),
    )


    parser.add_argument(
        "--base-prefix",
        default=
            DEFAULT_BASE_PREFIX,
        help=(
            "Prefix used for automatic "
            "Parquet discovery and generation. "
            "For example prefix "
            "'diploid_ancestry_mhc' produces "
            "diploid_ancestry_mhc_0.parquet, "
            "diploid_ancestry_mhc_1.parquet, ..."
        ),
    )


    # -------------------------------------------------------------------------
    # Split files
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--train-ids",
        required=True,
        help=(
            "Text file containing "
            "one training individual ID "
            "per line."
        ),
    )


    parser.add_argument(
        "--test-ids",
        required=True,
        help=(
            "Text file containing "
            "one test individual ID "
            "per line."
        ),
    )


    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            DATASET_DIR,
            "splits",
        ),
        help=(
            "Directory for "
            "train_*.parquet and "
            "test_*.parquet."
        ),
    )


    parser.add_argument(
        "--hf-output-dir",
        default=os.path.join(
            DATASET_DIR,
            "huggingface",
        ),
        help=(
            "Directory for the "
            "Hugging Face DatasetDict."
        ),
    )


    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=(
            "Maximum number of ROWS "
            "per generated base "
            "Parquet shard. "
            f"Default: {CHUNK_SIZE}"
        ),
    )


    parser.add_argument(
        "--split-chunk-size",
        type=int,
        default=None,
        help=(
            "Maximum rows per train/test "
            "Parquet shard. "
            "If omitted, --chunk-size "
            "is used."
        ),
    )


    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=
            SAMPLE_BATCH_SIZE,
        help=(
            "Number of samples reconstructed "
            "in memory at once during "
            "base generation. "
            f"Default: "
            f"{SAMPLE_BATCH_SIZE}"
        ),
    )


    parser.add_argument(
        "--parquet-read-batch-size",
        type=int,
        default=
            PARQUET_READ_BATCH_SIZE,
        help=(
            "Rows read at once while "
            "streaming existing Parquet "
            "files during splitting. "
            f"Default: "
            f"{PARQUET_READ_BATCH_SIZE}"
        ),
    )


    # -------------------------------------------------------------------------
    # Behavior
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--force-rebuild-base",
        action="store_true",
        help=(
            "Ignore existing automatically "
            "discovered base shards and "
            "recreate them using "
            "--chunk-size."
        ),
    )


    parser.add_argument(
        "--strict-ids",
        action="store_true",
        help=(
            "Fail if any requested "
            "train/test ID is absent "
            "from the base dataset."
        ),
    )


    parser.add_argument(
        "--skip-hf-conversion",
        action="store_true",
        help=(
            "Create the train/test "
            "Parquet shards only."
        ),
    )


    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    args = (
        parse_args()
    )

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be > 0"
        )

    split_chunk_size = (
        args.split_chunk_size
        if args.split_chunk_size
        is not None
        else args.chunk_size
    )


    # =========================================================================
    # Step 1:
    # Reuse or generate all base Parquet shards
    # =========================================================================

    base_parquets = (
        ensure_base_dataset(
            base_prefix=
                args.base_prefix,

            explicit_parquets=
                args.base_parquets,

            chunk_size=
                args.chunk_size,

            sample_batch_size=
                args.sample_batch_size,

            force_rebuild=
                args.force_rebuild_base,
        )
    )

    print(
        "\nBase dataset consists of "
        f"{len(base_parquets)} "
        "Parquet shard(s)."
    )


    # =========================================================================
    # Step 2:
    # Subject-level train/test split across ALL base shards
    # =========================================================================

    (
        train_parquets,
        test_parquets,
    ) = split_parquets_by_ids(
        base_parquets=
            base_parquets,

        train_ids_path=
            args.train_ids,

        test_ids_path=
            args.test_ids,

        output_dir=
            args.output_dir,

        split_chunk_size=
            split_chunk_size,

        parquet_read_batch_size=
            args.parquet_read_batch_size,

        strict_ids=
            args.strict_ids,
    )


    # =========================================================================
    # Step 3:
    # Hugging Face conversion
    # =========================================================================

    if not args.skip_hf_conversion:

        convert_to_huggingface(
            train_parquets=
                train_parquets,

            test_parquets=
                test_parquets,

            hf_output_dir=
                args.hf_output_dir,
        )


if __name__ == "__main__":
    main()