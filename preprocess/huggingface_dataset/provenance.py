"""Provenance sidecar I/O — shared by ``create_dataset.py`` (writer) and
``dataset.py`` (reader).

Two JSON sidecars travel with a dataset:

* ``generation_provenance.json`` — written next to the per-sample Parquet shards
  by ``create_dataset.py generate``. Records the encoder settings (schema
  version, representation, marker policy, annotation sources) so a resumed run
  cannot silently mix incompatible shards in one output directory.
* ``provenance.json`` — written next to the assembled HuggingFace dataset by
  ``create_dataset.py assemble``. Read by ``DiploidDataset`` / ``HaploidDataset``
  to validate that the on-disk schema matches the loader's expectation.

This module is intentionally a pure leaf (only ``json`` / ``pathlib`` /
``warnings``): both ``create_dataset`` and ``dataset`` import from it without an
import cycle, and the previously-duplicated ``_load_json_strict`` now lives in
one place.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Optional

# Canonical sidecar filenames (the two modules alias these).
GENERATION_PROVENANCE_FILENAME = "generation_provenance.json"
ASSEMBLED_PROVENANCE_FILENAME = "provenance.json"


def load_json_strict(path: Path) -> dict:
    """Load a provenance sidecar, raising a clear error on a corrupt file.

    A truncated/hand-edited sidecar would otherwise surface as a bare
    ``JSONDecodeError`` deep in a run; this names the offending file and the
    fix (delete it to force a clean rewrite) instead.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Corrupt JSON in {path}: {exc}. This is a provenance sidecar — "
            f"remove or repair it (deleting it forces a clean rewrite on the "
            f"next run)."
        ) from exc


def verify_schema_version(path: Path, expected: Optional[str]) -> None:
    """Check the dataset's ``provenance.json`` schema_version against *expected*.

    Raises ``RuntimeError`` if the recorded schema_version doesn't match.
    Emits a warning if ``provenance.json`` is absent (legacy dataset).
    Skips the check entirely if *expected* is None.
    """
    if expected is None:
        return
    provenance_path = path / ASSEMBLED_PROVENANCE_FILENAME
    if not provenance_path.is_file():
        warnings.warn(
            f"No {ASSEMBLED_PROVENANCE_FILENAME} at {path}. Legacy dataset "
            f"assumed; cannot verify schema_version against expected "
            f"{expected!r}. Pass expected_schema_version=None to suppress this "
            f"warning.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    provenance = load_json_strict(provenance_path)
    actual = provenance.get("schema_version")
    if actual != expected:
        raise RuntimeError(
            f"Dataset at {path} reports schema_version={actual!r} "
            f"but this code expects {expected!r}. Schema changes are "
            f"not backward-compatible; datasets must be regenerated. "
            f"Pass expected_schema_version=None or "
            f"expected_schema_version={actual!r} to override."
        )


def _annotation_key(a: dict) -> str:
    """Canonical comparison key for one annotation provenance dict.

    Excludes volatile ``*_mtime`` fields so a touch-only mtime change does not
    invalidate a resume, while a real source swap does. ``json.dumps(sort_keys)``
    gives a stable key regardless of which fields a given annotation reports
    (phylop reports ``source_path``/``source_size``; CADD reports ``arrow_dir``).
    """
    return json.dumps(
        {k: v for k, v in a.items() if not k.endswith("_mtime")},
        sort_keys=True,
    )


def check_or_write_generation_provenance(
    output_dir: Path,
    *,
    schema_version: str,
    representation: str,
    force_unphased: bool,
    use_direction_markers: bool,
    marker_policy: str,
    externalize_markers: bool = False,
    annotations: List[dict],
    inputs: Optional[dict] = None,
) -> None:
    """Read or write the per-shard-dir generation provenance sidecar.

    Prevents silently mixing diploid and haploid shards (or differing encoder
    settings — including the v5/v6/v7 schema version) in the same output
    directory across resumed runs.

    On first run for an output directory: writes the sidecar with the current
    settings. On subsequent runs: refuses to start if any recorded field
    mismatches the current invocation.

    ``schema_version`` and ``marker_policy`` are supplied by the caller (from
    the resolved version preset) rather than read from a global, so this module
    stays a pure leaf.

    ``inputs`` (optional) is a caller-built fingerprint dict of the
    content-determining inputs (BED/ref/BCF/tokens/selection files, truncation
    knobs). It is verified sub-field by sub-field on resume so a re-run with a
    different BED, reference, variant source, etc. is rejected rather than
    silently mixed. Hashing is the caller's job — this module stays a pure leaf.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_dir / GENERATION_PROVENANCE_FILENAME

    current = {
        "schema_version": schema_version,
        "representation": representation,
        "force_unphased": bool(force_unphased),
        "use_direction_markers": bool(use_direction_markers),
        "marker_policy": marker_policy,
        "externalize_markers": bool(externalize_markers),
        "annotations": annotations,
        "inputs": inputs or {},
    }

    if sidecar_path.exists():
        recorded = load_json_strict(sidecar_path)
        mismatched = []
        for k in current:
            if k == "annotations":
                cur_keys = sorted(_annotation_key(a) for a in current[k])
                rec_keys = sorted(_annotation_key(a) for a in recorded.get(k, []))
                if cur_keys != rec_keys:
                    mismatched.append(k)
            elif k == "inputs":
                # Compare only sub-fields present (non-None) on BOTH sides:
                # a legacy sidecar without `inputs` is grandfathered, and an
                # optional input absent from one run (None) doesn't force a
                # false mismatch. A genuine swap (two differing non-None values)
                # is reported with its own `inputs.<field>` name.
                rec = recorded.get("inputs", {})
                for sub, cur_val in current[k].items():
                    rec_val = rec.get(sub)
                    if rec_val is not None and cur_val is not None and rec_val != cur_val:
                        mismatched.append(f"inputs.{sub}")
            elif recorded.get(k) != current[k]:
                mismatched.append(k)
        if mismatched:
            def _lookup(src: dict, key: str):
                # `inputs.<sub>` keys live one level down; plain keys are flat.
                if key.startswith("inputs."):
                    return src.get("inputs", {}).get(key.split(".", 1)[1])
                return src.get(key)
            details = ", ".join(
                f"{k}: recorded={_lookup(recorded, k)!r} vs "
                f"current={_lookup(current, k)!r}"
                for k in mismatched
            )
            raise RuntimeError(
                f"Generation provenance mismatch in {output_dir} "
                f"({sidecar_path.name}). Conflicting fields: {details}. "
                f"Use a different --output directory or remove the existing "
                f"shards if you intend to overwrite."
            )
        return

    # Atomic write (.tmp -> rename): a hard crash mid-write must not leave a
    # truncated JSON. load_json_strict turns any corrupt sidecar into a clear,
    # actionable error rather than wedging every future resume of this dir.
    tmp = sidecar_path.parent / (sidecar_path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(current, f, indent=2)
    tmp.rename(sidecar_path)


def append_run_record(sidecar_path: Path, run: dict) -> None:
    """Append one run record to the sidecar's top-level ``runs`` array.

    Run history (command, params, timestamps, milestones — one entry per
    invocation, including every resume) lives inside the SAME sidecar the schema
    guard / loaders already use, so provenance is documented in one place rather
    than scattered. This never touches the schema keys, and the guard never
    reads ``runs`` (it compares only its own ``current`` keys), so the two are
    independent: appending history can't affect a resume or a load.

    The caller builds the ``run`` dict (with pre-formatted timestamp strings) so
    this module stays a pure leaf. Atomic .tmp->rename: a crash mid-write cannot
    truncate the sidecar.
    """
    doc = load_json_strict(sidecar_path) if sidecar_path.exists() else {}
    runs = doc.get("runs")
    if not isinstance(runs, list):
        runs = []
    runs.append(run)
    doc["runs"] = runs
    tmp = sidecar_path.parent / (sidecar_path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    tmp.rename(sidecar_path)
