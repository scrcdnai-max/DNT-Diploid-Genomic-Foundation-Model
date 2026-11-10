"""The diploid encoder: ``get_genome_sequence`` (one module, kept whole), ``reference``, ``perf_timer``.

Public names are re-exported lazily, so ``import preprocess.encoder.reference`` (as
``extract_cadd_for_bed`` does) never loads the 6,800-line encoder or cyvcf2. Private helpers are
imported from ``preprocess.encoder.get_genome_sequence`` directly.
"""

_EXPORTS = {
    "GenomeSequenceProcessor": "get_genome_sequence",
    "INTEGRITY_COUNTER_KEYS": "get_genome_sequence",
    "UnphasedHeterozygoteError": "get_genome_sequence",
    "Variant": "get_genome_sequence",
    "VariantCall": "get_genome_sequence",
    "rehydrate_markers": "get_genome_sequence",
    "split_markers": "get_genome_sequence",
    "ReferenceFile": "reference",
    "PerfAccumulator": "perf_timer",
}
__all__ = sorted(_EXPORTS)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
