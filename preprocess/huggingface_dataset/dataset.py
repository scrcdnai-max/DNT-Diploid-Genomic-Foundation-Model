"""Dataset wrappers for diploid and haploid genomic sequences.

This module exposes two HuggingFace-dataset wrappers:

* :class:`DiploidDataset` — the primary class, unchanged in shape: returns the
  256x256-encoded diploid sequences with ``^``/``{ }`` markers.
* :class:`HaploidDataset` — companion class for haploid datasets produced by
  ``create_dataset.py generate --representation haploid``. The ``seq`` field
  is a literal ALT-biased ACGTN string with ``^``/``{ }`` markers (no
  diploid codepoints, no direction markers).

Each class enforces (via ``provenance.json``) that the on-disk dataset
matches the expected representation: instantiating the wrong wrapper raises
``RuntimeError`` so users never silently decode haploid bytes as diploid
codepoints or vice versa.

Schema versions (the resolved version is read from ``provenance.json``; there is
no hardcoded default — see :mod:`versions`):
  v8:           same content as v7, but the structural markers (``^``/``{ }``/
                ``‹ ›``) are externalized out of ``seq`` into the sparse ``snp`` /
                ``indel_span`` / ``direction`` columns (``seq`` is pure base
                content). :meth:`DiploidDataset.to_marked_string` reconstructs the
                v7-equivalent inline string from those columns.
  v7:           adds per-base annotation columns (e.g. ``phylop``) populated
                from external sources (bigWig files). Drops the v6 backward
                compatibility gate (``dup_ratio > 0.10``) in ``assemble`` —
                weights are aggregated whenever duplicates exist. Initial
                annotation: PhyloP 100-way. Marker codepoints carry NaN by
                default (annotation arrays are 1:1 with ``seq`` codepoints).
  v6:           bare direction markers replaced with ^-prefixed form
                via anchor-SNP detection in GenomeSequenceProcessor.
                Byte-identical to v5 elsewhere.
  v5:           three-level direction-marker encoding.
  v4:           string sequences with SNP/INDEL delimiters; no direction.

Records can be retrieved as:
  1. string      — the stored ``seq`` field directly (primary form)
  2. numpy array — uint32 codepoints decoded from the string
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional, Union

import datasets
import numpy as np

# Absolute package imports only (no relative imports): this file must stay
# loadable by file path (pretrain/tests/test_aux_decode_constants.py).
from preprocess.huggingface_dataset import versions  # noqa: E402
from preprocess.huggingface_dataset.versions import SUPPORTED_VERSIONS  # noqa: E402
from preprocess.huggingface_dataset.provenance import (  # noqa: E402
    ASSEMBLED_PROVENANCE_FILENAME as _ASSEMBLED_PROVENANCE_FILENAME,
    GENERATION_PROVENANCE_FILENAME as _GENERATION_PROVENANCE_FILENAME,
    load_json_strict as _load_json_strict,
    verify_schema_version as _verify_schema_version_impl,
)
from preprocess.markers import (  # noqa: E402
    decode_direction,
    decode_indel_spans,
    decode_snp,
    rehydrate_markers,
)

# ---------------------------------------------------------------------------
# Annotation decode helpers (inverse of annotations.py encode())
# ---------------------------------------------------------------------------

# PhyloP: uint8 [0,254] → float32 [−20, +10.003]; 255 = NaN sentinel.
_PHYLOP_OFFSET: float = 20.0
_PHYLOP_RANGE: float = 30.003
_PHYLOP_NAN_SENTINEL: int = 255


def _decode_phylop(raw) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.uint8)
    out = np.full(len(arr), np.nan, dtype=np.float32)
    valid = arr != _PHYLOP_NAN_SENTINEL
    out[valid] = arr[valid].astype(np.float32) / 254.0 * _PHYLOP_RANGE - _PHYLOP_OFFSET
    return out


def _decode_cadd(raw, seq_len: int) -> np.ndarray:
    """Expand sparse list<struct<pos,score>> to dense float32 aligned to seq."""
    dense = np.full(seq_len, np.nan, dtype=np.float32)
    for entry in raw:
        dense[entry["pos"]] = float(entry["score"])
    return dense


def _rehydrate_marked(seq: str, snp, indel_span, direction) -> str:
    """Reconstruct the inline-marked sequence from marker-free ``seq`` + columns.

    Thin wrapper over :func:`preprocess.markers.rehydrate_markers` (stdlib only,
    so consumers need no cyvcf2 or encoder import). Used by ``to_marked_string``
    of both dataset classes below, and imported under this name by
    ``embedding/cohort/extract.py`` and ``test_gather_dataset_stats.py``.
    """
    return rehydrate_markers(seq, snp, indel_span, direction)


def read_schema_version(path: Union[str, Path]):
    """Return the schema_version recorded in *path*'s provenance sidecar(s).

    Checks ``provenance.json`` (assemble) then ``generation_provenance.json``
    (generate). Returns the recorded version string, or ``None`` if neither
    sidecar records one (legacy dataset). Raises ``RuntimeError`` if both
    sidecars are present and disagree.
    """
    path = Path(path)
    found: dict = {}
    for name in (_ASSEMBLED_PROVENANCE_FILENAME, _GENERATION_PROVENANCE_FILENAME):
        sidecar = path / name
        if sidecar.is_file():
            sv = _load_json_strict(sidecar).get("schema_version")
            if sv is not None:
                found[name] = sv
    if not found:
        return None
    values = set(found.values())
    if len(values) > 1:
        details = ", ".join(f"{k}={v!r}" for k, v in found.items())
        raise RuntimeError(
            f"Provenance sidecars in {path} disagree on schema_version: {details}."
        )
    return values.pop()


def _require_column(record: dict, name: str):
    """Return ``record[name]`` or raise a clear error naming the missing column.

    A dataset generated without an annotation simply lacks that column. A bare
    ``KeyError`` from a decode helper is opaque, so name the column and the
    likely cause (the source flag was omitted at generation time)."""
    try:
        return record[name]
    except KeyError:
        raise KeyError(
            f"Record has no {name!r} column — this dataset was generated "
            f"without it (no --phylop-bw / --cadd-arrow-dir passed to "
            f"`create_dataset.py generate`). Available columns: "
            f"{sorted(record.keys())}."
        ) from None

def read_representation(path: Union[str, Path]) -> str:
    """Return the representation recorded in *path*'s provenance sidecar.

    Inspects both filenames written by the pipeline:
      * ``provenance.json`` — written by ``create_dataset.py assemble``
        next to assembled HF datasets.
      * ``generation_provenance.json`` — written by
        ``create_dataset.py generate`` next to each shard directory.

    Returns ``"diploid"`` or ``"haploid"``. If both sidecars are present they
    must agree; mismatch raises ``RuntimeError``. If neither sidecar is
    present, defaults to ``"diploid"`` for backward compatibility with
    legacy shards/datasets written before this field existed.
    """
    path = Path(path)
    candidates = [
        path / _ASSEMBLED_PROVENANCE_FILENAME,
        path / _GENERATION_PROVENANCE_FILENAME,
    ]
    found: dict = {}
    for sidecar in candidates:
        if sidecar.is_file():
            found[sidecar.name] = _load_json_strict(sidecar).get(
                "representation", "diploid")

    if not found:
        return "diploid"

    values = set(found.values())
    if len(values) > 1:
        details = ", ".join(f"{k}={v!r}" for k, v in found.items())
        raise RuntimeError(
            f"Provenance sidecars in {path} disagree on representation: {details}."
        )
    return values.pop()


class DiploidDataset:
    """Wrapper around a saved HuggingFace dataset of diploid genomic sequences.

    Parameters
    ----------
    path:
        Path to a directory of Parquet shards (loaded with ``load_dataset("parquet")``)
        or a HuggingFace Arrow dataset (loaded with ``load_from_disk``).
    split:
        If the dataset was saved with train/test splits (``DatasetDict``), specify
        which split to use (``"train"`` or ``"test"``). If ``None`` and the data
        is a ``DatasetDict``, defaults to ``"train"``.
    expected_schema_version:
        The expected schema version. Defaults to ``None``, meaning "adopt the
        version recorded in the dataset's provenance sidecar" (validated against
        :data:`versions.SUPPORTED_VERSIONS`). Pass a concrete version (e.g.
        ``"v8"``) to enforce it: a mismatch with the recorded version raises
        ``RuntimeError``. There is deliberately no hardcoded default version.

    Notes
    -----
    The on-disk ``seq`` format depends on the version. v5/v6/v7 carry the
    structural markers (``^``/``{ }``/``‹ ›``) inline in ``seq``; v8 stores pure
    base content in ``seq`` and the markers in the sparse ``snp`` / ``indel_span``
    / ``direction`` columns. :meth:`to_marked_string` reconstructs the
    v7-equivalent inline-marked string regardless of version.
    """

    def __init__(
        self,
        path: Union[str, Path],
        split: Optional[str] = None,
        expected_schema_version: Optional[str] = None,
    ) -> None:
        path = Path(path)

        self._schema_version = self._resolve_version(path, expected_schema_version)
        self._externalize_markers = (
            self._schema_version in SUPPORTED_VERSIONS
            and versions.resolve(self._schema_version).externalize_markers
        )
        self._verify_representation(path, expected="diploid")

        parquet_files = list(path.glob("*.parquet"))
        if parquet_files:
            self._hf: datasets.Dataset = datasets.load_dataset(
                "parquet",
                data_files=[str(p) for p in sorted(parquet_files)],
                split="train",
            )
        else:
            loaded = datasets.load_from_disk(str(path))
            if isinstance(loaded, datasets.DatasetDict):
                split = split or "train"
                if split not in loaded:
                    raise KeyError(
                        f"Split {split!r} not found. Available: {list(loaded.keys())}"
                    )
                self._hf = loaded[split]
            else:
                self._hf = loaded

    @staticmethod
    def _resolve_version(path: Path, expected: Optional[str]) -> Optional[str]:
        """Resolve the schema version to use for this dataset.

        * ``expected`` given — enforce it (raises ``RuntimeError`` on mismatch
          with the recorded version; warns if no provenance) and return it.
        * ``expected`` None — adopt the version recorded in provenance, validated
          against :data:`versions.SUPPORTED_VERSIONS`. Returns ``None`` for a
          legacy dataset with no recorded version (treated as inline markers).
        """
        if expected is not None:
            _verify_schema_version_impl(path, expected)
            return expected
        declared = read_schema_version(path)
        if declared is None:
            warnings.warn(
                f"No schema_version recorded in provenance at {path}; assuming "
                f"a legacy inline-marker dataset. Pass expected_schema_version "
                f"to enforce a specific version.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        if declared not in SUPPORTED_VERSIONS:
            raise RuntimeError(
                f"Dataset at {path} reports schema_version={declared!r}, which "
                f"this code does not support (known: "
                f"{', '.join(sorted(SUPPORTED_VERSIONS))}). Upgrade the code or "
                f"regenerate the dataset."
            )
        return declared

    @staticmethod
    def _verify_representation(path: Path, expected: str) -> None:
        """Refuse to load a dataset whose representation differs from ``expected``.

        Reads both sidecars via :func:`read_representation` — ``provenance.json``
        (assembled) and ``generation_provenance.json`` (a shard directory).
        Reading only the former (until 2026-10-06) let a haploid SHARD dir load
        as diploid, since shard dirs carry only the generation sidecar. No
        sidecar, or no ``representation`` field, means ``"diploid"`` (legacy
        datasets written before the field existed).
        """
        actual = read_representation(path)
        if actual != expected:
            sibling = {
                "diploid": "HaploidDataset",
                "haploid": "DiploidDataset",
            }.get(actual, "<unknown>")
            raise RuntimeError(
                f"Dataset at {path} reports representation={actual!r} but "
                f"this loader expects {expected!r}. Use {sibling} to load "
                f"this dataset, or regenerate the shards with "
                f"--representation {expected}."
            )

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int) -> dict:
        return self._hf[idx]

    def __iter__(self):
        yield from self._hf

    # ------------------------------------------------------------------
    # Output forms
    # ------------------------------------------------------------------

    def to_numpy(self, record: dict) -> np.ndarray:
        """Return uint32 codepoints array from a dataset record."""
        seq = record["seq"]
        return np.array([ord(c) for c in seq], dtype=np.uint32)

    def to_string(self, record: dict) -> str:
        """Return the diploid symbol sequence as a Unicode string."""
        return record["seq"]

    def to_annotation(self, record: dict, name: str) -> np.ndarray:
        """Return a per-base annotation column as a ``float32`` ``np.ndarray``.

        Decodes the on-disk storage format back to float32:
          ``phylop``     — uint8 linear quantisation → float32 [−20, +10.003]
          ``cadd_phred`` — sparse list<struct<pos,score>> → dense float32 aligned to seq
        """
        if name == "phylop":
            return _decode_phylop(_require_column(record, "phylop"))
        if name == "cadd_phred":
            return _decode_cadd(
                _require_column(record, "cadd_phred"), len(record["seq"]))
        return np.asarray(_require_column(record, name), dtype=np.float32)

    def to_phylop(self, record: dict) -> np.ndarray:
        """Return PhyloP scores as float32, decoded from uint8 storage."""
        return _decode_phylop(_require_column(record, "phylop"))

    def to_cadd(self, record: dict) -> np.ndarray:
        """Return CADD phred as dense float32, decoded from sparse struct storage.

        Per-variant PHRED with anchor-only placement and
        ``max(hap0, hap1)`` reduction (see
        :class:`huggingface_dataset.annotations.CADDAnnotation` for the full
        rule table). NaN at ref-matching positions, inserted bases, marker
        codepoints, indels absent from the CADD indel file, and cross-region
        anchor-clipped variants.
        """
        return _decode_cadd(_require_column(record, "cadd_phred"), len(record["seq"]))

    # ------------------------------------------------------------------
    # v8 structural markers (sparse columns)
    # ------------------------------------------------------------------

    def to_markers(self, record: dict) -> dict:
        """Return the decoded structural marker columns for a v8 record.

        ``{"snp": [pos...], "indel_span": [(start, end)...], "direction":
        [(pos, dir)...]}``. For pre-v8 (inline-marker) datasets the columns are
        absent; this returns empty lists (the markers live inside ``seq``).
        """
        return {
            "snp": decode_snp(record.get("snp")),
            "indel_span": decode_indel_spans(record.get("indel_span")),
            "direction": decode_direction(record.get("direction")),
        }

    def to_marked_string(self, record: dict) -> str:
        """Return the v7-equivalent inline-marked sequence string.

        For pre-v8 datasets ``seq`` already carries the inline markers, so this
        returns it unchanged. For v8 datasets it reconstructs the inline form by
        re-inserting ``^``/``{ }``/``‹ ›`` from the sparse columns (the exact
        inverse of generation's marker externalization).
        """
        if not self._externalize_markers:
            return record["seq"]
        m = self.to_markers(record)
        return _rehydrate_marked(
            record["seq"], m["snp"], m["indel_span"], m["direction"])

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def schema_version(self) -> Optional[str]:
        """The resolved schema version of this dataset (or ``None`` if legacy)."""
        return self._schema_version

    @property
    def hf_dataset(self) -> datasets.Dataset:
        """Direct access to the underlying HuggingFace Dataset object."""
        return self._hf

    def __repr__(self) -> str:
        return f"DiploidDataset(num_records={len(self)})"


# Haploid alphabet (used by HaploidDataset.to_numpy for a defensive
# alphabet check at decode time). ACGTN + lowercase variants + markers.
_HAPLOID_ALPHABET = frozenset("ACGTNacgtn^{}")


class HaploidDataset:
    """Wrapper around a saved HuggingFace dataset of haploid genomic sequences.

    Sibling of :class:`DiploidDataset` for datasets produced by
    ``create_dataset.py generate --representation haploid``. The ``seq``
    field is a literal ACGTN string with ``^`` SNP markers and ``{ }``
    INDEL span markers — no diploid codepoints, no direction markers.

    Parameters
    ----------
    path:
        Path to a directory of Parquet shards or a HuggingFace Arrow dataset.
    split:
        If the dataset was saved with train/test splits (``DatasetDict``),
        which split to load. Defaults to ``"train"`` for ``DatasetDict``.
    expected_schema_version:
        The expected schema version. Defaults to ``None`` (adopt the version
        recorded in provenance, validated against
        :data:`versions.SUPPORTED_VERSIONS`). Pass a concrete version to enforce
        it. There is no hardcoded default version.
    """

    def __init__(
        self,
        path: Union[str, Path],
        split: Optional[str] = None,
        expected_schema_version: Optional[str] = None,
    ) -> None:
        path = Path(path)

        self._schema_version = DiploidDataset._resolve_version(
            path, expected_schema_version)
        self._externalize_markers = (
            self._schema_version in SUPPORTED_VERSIONS
            and versions.resolve(self._schema_version).externalize_markers
        )
        DiploidDataset._verify_representation(path, expected="haploid")

        parquet_files = list(path.glob("*.parquet"))
        if parquet_files:
            self._hf: datasets.Dataset = datasets.load_dataset(
                "parquet",
                data_files=[str(p) for p in sorted(parquet_files)],
                split="train",
            )
        else:
            loaded = datasets.load_from_disk(str(path))
            if isinstance(loaded, datasets.DatasetDict):
                split = split or "train"
                if split not in loaded:
                    raise KeyError(
                        f"Split {split!r} not found. Available: {list(loaded.keys())}"
                    )
                self._hf = loaded[split]
            else:
                self._hf = loaded

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int) -> dict:
        return self._hf[idx]

    def __iter__(self):
        yield from self._hf

    # ------------------------------------------------------------------
    # Output forms
    # ------------------------------------------------------------------

    def to_string(self, record: dict) -> str:
        """Return the haploid sequence as a Unicode string (ACGTN + markers)."""
        return record["seq"]

    def to_numpy(self, record: dict) -> np.ndarray:
        """Return uint32 codepoints array decoded from the haploid string.

        Defensive alphabet check: raises ``ValueError`` if the record's
        ``seq`` field contains a character outside the haploid alphabet
        ``{A,C,G,T,N,^,{,}}`` (plus lowercase variants). This protects
        against accidental ingestion of diploid bytes through this wrapper.
        """
        seq = record["seq"]
        for ch in seq:
            if ch not in _HAPLOID_ALPHABET:
                raise ValueError(
                    f"HaploidDataset record contains disallowed character "
                    f"{ch!r} (code {ord(ch)}). Allowed alphabet: "
                    f"{sorted(_HAPLOID_ALPHABET)}."
                )
        return np.array([ord(c) for c in seq], dtype=np.uint32)

    def to_annotation(self, record: dict, name: str) -> np.ndarray:
        """Return a per-base annotation column as a ``float32`` ``np.ndarray``.

        See :meth:`DiploidDataset.to_annotation` for decode semantics.
        """
        if name == "phylop":
            return _decode_phylop(_require_column(record, "phylop"))
        if name == "cadd_phred":
            return _decode_cadd(
                _require_column(record, "cadd_phred"), len(record["seq"]))
        return np.asarray(_require_column(record, name), dtype=np.float32)

    def to_phylop(self, record: dict) -> np.ndarray:
        """Return PhyloP scores as float32, decoded from uint8 storage."""
        return _decode_phylop(_require_column(record, "phylop"))

    def to_cadd(self, record: dict) -> np.ndarray:
        """Return CADD phred as dense float32, decoded from sparse struct storage."""
        return _decode_cadd(_require_column(record, "cadd_phred"), len(record["seq"]))

    # ------------------------------------------------------------------
    # v8 structural markers (sparse columns; haploid has no direction)
    # ------------------------------------------------------------------

    def to_markers(self, record: dict) -> dict:
        """Return the decoded structural marker columns for a v8 haploid record.

        ``{"snp": [pos...], "indel_span": [(start, end)...]}``. Haploid datasets
        carry no ``direction`` column. Empty for pre-v8 (inline) datasets.
        """
        return {
            "snp": decode_snp(record.get("snp")),
            "indel_span": decode_indel_spans(record.get("indel_span")),
        }

    def to_marked_string(self, record: dict) -> str:
        """Return the inline-marked haploid string (``^`` / ``{ }``, no direction).

        Returns ``seq`` unchanged for pre-v8 datasets; reconstructs the inline
        form from the sparse columns for v8 datasets.
        """
        if not self._externalize_markers:
            return record["seq"]
        m = self.to_markers(record)  # haploid has no direction column
        return _rehydrate_marked(record["seq"], m["snp"], m["indel_span"], [])

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def schema_version(self) -> Optional[str]:
        """The resolved schema version of this dataset (or ``None`` if legacy)."""
        return self._schema_version

    @property
    def hf_dataset(self) -> datasets.Dataset:
        """Direct access to the underlying HuggingFace Dataset object."""
        return self._hf

    def __repr__(self) -> str:
        return f"HaploidDataset(num_records={len(self)})"
