"""ReferenceFile: random-access FASTA / 2bit reference reader (moved from preprocess_lib, reorg step 8-0)."""

import logging
import time
from pathlib import Path
from typing import List, Union

import numpy as np
from pyfaidx import Fasta as PyfaidxFasta  # For FASTA parsing in ReferenceFile

logger = logging.getLogger(__name__)

# Try to import py2bit for 2bit file support
try:
    import py2bit
    HAS_PY2BIT = True
except ImportError:
    HAS_PY2BIT = False


class ReferenceFile:
    """Unified interface for reference genome files (FASTA and 2bit formats).

    Provides random-access sequence fetching. For 2bit files, all chromosomes
    are preloaded into memory at initialization (~3GB for hg38, takes ~7s).
    This provides instant access for all subsequent fetches.

    Args:
        file_path: Path to reference genome file (.fa, .fasta, .2bit).

    Attributes:
        file_path: Resolved path to the reference file.
        file_type: Detected file type ('fasta' or '2bit').

    Example:
        >>> with ReferenceFile('hg38.2bit') as ref:
        ...     seq = ref.fetch('chr1', 1000000, 1000100)
        ...     print(f"Fetched {len(seq)} bases")
        Fetched 100 bases
    """
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.file_type = self._detect_file_type()
        self._file_handle = None
        self._chromosome_cache = {}

    def _detect_file_type(self) -> str:
        """Detect if file is FASTA or 2bit based on extension."""
        suffix = self.file_path.suffix.lower()
        if suffix == '.2bit':
            return '2bit'
        elif suffix in ['.fa', '.fasta', '.fna'] or self.file_path.name.endswith('.fa.gz') or self.file_path.name.endswith('.fasta.gz'):
            return 'fasta'
        else:
            # Try to guess from content or default to fasta
            return 'fasta'

    def __enter__(self):
        if self.file_type == '2bit':
            if not HAS_PY2BIT:
                raise ImportError(
                    "py2bit library is required for 2bit files. "
                    "Install it with: pip install py2bit"
                )
            try:
                self._file_handle = py2bit.open(str(self.file_path))
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to open 2bit file '{self.file_path}': {e}\n"
                    f"Please ensure the file exists and is a valid 2bit format file."
                )

            # Preload all chromosomes into memory as numpy uint8 arrays
            # for zero-copy slicing in fetch() and direct use in numpy
            # operations (avoids repeated str.encode + np.frombuffer).
            logger.info("Pre-loading all chromosomes into memory...")
            start_time = time.time()
            chrom_names = list(self._file_handle.chroms().keys())
            for chrom in chrom_names:
                seq = self._file_handle.sequence(chrom).upper()
                self._chromosome_cache[chrom] = np.frombuffer(
                    seq.encode("ascii"), dtype=np.uint8,
                ).copy()  # .copy() to own the memory (decouple from str)
            elapsed = time.time() - start_time
            total_mb = sum(arr.nbytes for arr in self._chromosome_cache.values()) / (1024**2)
            logger.info(
                f"Loaded {len(self._chromosome_cache)} chromosomes "
                f"({total_mb:.1f} MB) in {elapsed:.1f}s"
            )
        else:  # fasta
            self._file_handle = PyfaidxFasta(
                str(self.file_path), as_raw=True, build_index=False,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file_handle is not None:
            if self.file_type == 'fasta':
                self._file_handle.close()
            elif self.file_type == '2bit':
                self._file_handle.close()
        self._file_handle = None
        self._chromosome_cache.clear()

    def chromosome_names(self) -> List[str]:
        """Return the list of chromosome/contig names in the reference."""
        if self.file_type == '2bit':
            return list(self._chromosome_cache.keys())
        else:  # fasta (pyfaidx)
            return list(self._file_handle.keys())

    def get_reference_length(self, chrom_name: str) -> int:
        """Get the length of a chromosome."""
        if self.file_type == '2bit':
            if chrom_name in self._chromosome_cache:
                return len(self._chromosome_cache[chrom_name])
            return self._file_handle.chroms(chrom_name)
        else:  # fasta (pyfaidx)
            try:
                return len(self._file_handle[chrom_name])
            except KeyError:
                raise KeyError(chrom_name) from None

    def fetch(self, chrom: str, start: int, end: int, as_array: bool = False):
        """Fetch sequence from reference (preloaded in memory for 2bit files).

        Args:
            chrom: Chromosome name.
            start: 0-based start position.
            end: 0-based end position (exclusive).
            as_array: If True, return a numpy uint8 array (zero-copy view
                for 2bit files). If False (default), return a string for
                backward compatibility.

        Returns:
            Uppercase sequence string, or numpy uint8 array if as_array=True.
        """
        if self.file_type == '2bit':
            arr = self._chromosome_cache[chrom][start:end]
            if as_array:
                return arr  # zero-copy numpy view
            return arr.tobytes().decode("ascii")
        else:  # fasta (pyfaidx)
            seq_str = str(self._file_handle[chrom][start:end]).upper()
            if as_array:
                return np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8).copy()
            return seq_str

    @property
    def references(self):
        """Get list of chromosome/contig names in the reference file.

        Returns:
            List of chromosome names as strings.
        """
        if self.file_type == '2bit':
            return list(self._file_handle.chroms().keys())
        else:  # fasta (pyfaidx)
            return list(self._file_handle.keys())
