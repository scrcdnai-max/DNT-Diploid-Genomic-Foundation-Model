"""Thread-safe performance timing accumulator for pipeline profiling.

Usage:
    perf = PerfAccumulator()
    processor._perf = perf  # inject into GenomeSequenceProcessor

    # ... run processing ...

    report = perf.report()
    for fn, stats in sorted(report.items(), key=lambda x: -x[1]["total_s"]):
        print(f"{fn}: {stats['mean_ms']:.2f}ms mean, {stats['total_s']:.3f}s total")
"""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(slots=True)
class TimingSample:
    """One timing measurement."""
    function: str
    thread_name: str
    elapsed_s: float
    region_size: int
    n_variants: int


class PerfAccumulator:
    """Thread-safe accumulator for per-function timing samples."""

    def __init__(self):
        self._lock = threading.Lock()
        self._samples: List[TimingSample] = []

    @contextmanager
    def time_block(self, function_name: str, region_size: int = 0, n_variants: int = 0):
        """Context manager that records wall-clock time for a code block."""
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        sample = TimingSample(
            function=function_name,
            thread_name=threading.current_thread().name,
            elapsed_s=elapsed,
            region_size=region_size,
            n_variants=n_variants,
        )
        with self._lock:
            self._samples.append(sample)

    def _record_elapsed(self, function_name: str, t0: float, region_size: int = 0, n_variants: int = 0):
        """Record elapsed time from a start timestamp (for multi-exit-point functions)."""
        elapsed = time.perf_counter() - t0
        sample = TimingSample(
            function=function_name,
            thread_name=threading.current_thread().name,
            elapsed_s=elapsed,
            region_size=region_size,
            n_variants=n_variants,
        )
        with self._lock:
            self._samples.append(sample)

    def report(self) -> Dict[str, dict]:
        """Compute aggregate stats grouped by function name.

        Returns dict keyed by function name, each value a dict with:
            count, total_s, mean_ms, median_ms, p95_ms, max_ms, min_ms
        """
        with self._lock:
            samples = list(self._samples)

        grouped: Dict[str, List[float]] = defaultdict(list)
        for s in samples:
            grouped[s.function].append(s.elapsed_s)

        result = {}
        for fn, times in sorted(grouped.items()):
            arr = np.array(times)
            result[fn] = {
                "count": len(arr),
                "total_s": float(arr.sum()),
                "mean_ms": float(arr.mean() * 1000),
                "median_ms": float(np.median(arr) * 1000),
                "p95_ms": float(np.percentile(arr, 95) * 1000),
                "max_ms": float(arr.max() * 1000),
                "min_ms": float(arr.min() * 1000),
            }
        return result

    def clear(self):
        """Remove all collected samples."""
        with self._lock:
            self._samples.clear()
