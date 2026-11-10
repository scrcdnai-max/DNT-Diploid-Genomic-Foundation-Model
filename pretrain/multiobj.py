"""Pure, GPU-free building blocks for the multi-objective CPT run (Phase-2 B/C + CPL).

Three new objectives layered on the existing MLM + phyloP/CADD aux heads:

  * CPL (re-enabled): phase-swap contrastive on direction markers (the model code
    lives in train.py; this module only holds shared helpers).
  * Track B: effect-ranked WT-vs-mutant contrastive. At each variant token the
    embedding displacement between the real (mutant) window and a wild-type twin
    (variant token -> homozygous-reference base) must rank-order by CADD effect.
  * Track C: 1000G population allele-frequency regression at variant tokens
    (handled as an ordinary aux task; only the AF-target lookup is here).

The token<->genome alignment is the load-bearing piece: outside indel ``{}`` spans
every base/het token consumes exactly one reference base and the markers
``^ ‹ › { }`` consume none, so a variant token's genomic position is
``window_genomic_start + (#non-marker tokens strictly before it)`` (0-based).
Verified empirically: het tokens' reference base matches hg38 at 93% with this rule
and ~50% (chance) at off-by-one. See build_af_tables.py for the AF source tables.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


def token_genomic_offsets(input_ids: torch.Tensor, marker_mask: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix count of ref-consuming (non-marker) tokens.

    input_ids: (B, L) long. marker_mask: (B, L) bool, True where the token is a
    structural marker (^ ‹ › { }) that consumes no reference base. Returns (B, L)
    long where out[b, i] = number of non-marker tokens strictly before position i
    in row b. A variant token's genomic position = window_genomic_start + out[b,i].
    """
    consuming = (~marker_mask).long()
    return consuming.cumsum(dim=1) - consuming  # inclusive cumsum minus self = exclusive


def effect_ranked_margin_loss(
    displacement: torch.Tensor,
    effect: torch.Tensor,
    margin: float = 0.1,
    min_gap: float = 1.0,
    max_pairs: int = 4096,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Margin ranking: higher-effect variants should move farther WT->mut.

    displacement: (N,) >=0 embedding-displacement magnitude at N variant positions
    (already gathered, valid only). effect: (N,) CADD (or any effect score).
    For ordered pairs (i, j) with effect_i - effect_j >= min_gap, penalise
    relu(margin - (disp_i - disp_j)). Pairs are subsampled to max_pairs for cost.
    Returns a scalar; 0 (with grad) if N < 2 or no qualifying pair.
    """
    n = displacement.shape[0]
    if n < 2:
        return displacement.sum() * 0.0  # keep in graph, zero value
    # Candidate pairs: all-pairs if small, else random sample.
    if n * n <= max_pairs:
        ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        ii, jj = ii.reshape(-1), jj.reshape(-1)
    else:
        ii = torch.randint(0, n, (max_pairs,), generator=generator)
        jj = torch.randint(0, n, (max_pairs,), generator=generator)
    ii = ii.to(displacement.device); jj = jj.to(displacement.device)
    gap = effect[ii] - effect[jj]
    qualifies = gap >= min_gap
    if not bool(qualifies.any()):
        return displacement.sum() * 0.0
    di, dj = displacement[ii[qualifies]], displacement[jj[qualifies]]
    return F.relu(margin - (di - dj)).mean()


# --- AF lookup tables (Track C) -------------------------------------------------
class AFTables:
    """Lazy per-chrom (pos, ref, alt) -> AF lookup from build_af_tables.py .npz shards.

    Sorted-by-pos arrays per chrom; lookup is searchsorted(pos) + a short scan over
    ties matching (ref, alt). Returns NaN when absent. One instance per dataloader
    worker (numpy arrays are memmap-friendly; load on first touch per chrom).
    """

    def __init__(self, af_dir: str):
        self.af_dir = af_dir
        self._cache: dict[str, tuple] = {}

    def _chrom(self, chrom: str):
        if chrom not in self._cache:
            import os

            path = os.path.join(self.af_dir, f"af_{chrom}.npz")
            if not os.path.exists(path):
                self._cache[chrom] = None
            else:
                z = np.load(path, mmap_mode="r")
                self._cache[chrom] = (
                    np.asarray(z["pos"]), np.asarray(z["ref"]),
                    np.asarray(z["alt"]), np.asarray(z["af"]),
                )
        return self._cache[chrom]

    def lookup(self, chrom: str, pos1: int, ref: int, alt: int) -> float:
        """pos1 = 1-based genomic position (BCF convention); ref/alt in 0..3."""
        tab = self._chrom(chrom)
        if tab is None:
            return float("nan")
        parr, rarr, aarr, afarr = tab
        i = int(np.searchsorted(parr, pos1))
        while i < len(parr) and parr[i] == pos1:
            if rarr[i] == ref and aarr[i] == alt:
                return float(afarr[i])
            i += 1
        return float("nan")


if __name__ == "__main__":  # ponytail: one runnable self-check, no framework
    # token_genomic_offsets: markers consume nothing.
    ids = torch.tensor([[5, 9, 9, 5, 9]])           # 5 = marker, 9 = base
    mm = ids == 5
    off = token_genomic_offsets(ids, mm)
    assert off.tolist() == [[0, 0, 1, 2, 2]], off.tolist()

    # effect_ranked_margin_loss: perfectly ordered disp -> low loss; reversed -> high.
    eff = torch.tensor([10.0, 5.0, 1.0])
    good = effect_ranked_margin_loss(torch.tensor([3.0, 2.0, 1.0]), eff, margin=0.1)
    bad = effect_ranked_margin_loss(torch.tensor([1.0, 2.0, 3.0]), eff, margin=0.1)
    assert good < bad, (good.item(), bad.item())
    assert good.item() == 0.0, good.item()  # gaps all >=1 and disp gap>margin
    # grad flows
    d = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    effect_ranked_margin_loss(d, eff).backward()
    assert d.grad is not None and bool((d.grad != 0).any())
    print("multiobj self-check OK")
