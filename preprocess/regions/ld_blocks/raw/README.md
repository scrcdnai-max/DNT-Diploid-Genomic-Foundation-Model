# `ld_blocks/raw/` — source files and curation notes

Five raw LD-block BED files downloaded verbatim from **jmacdon/LDblocks_GRCh38**
(https://github.com/jmacdon/LDblocks_GRCh38; paper: "Ancestry-specific maps of GRCh38
linkage disequilibrium blocks", bioRxiv 2022.03.04.483057). Downloaded 2026-07-06 from
`https://raw.githubusercontent.com/jmacdon/LDblocks_GRCh38/master/data/<file>`. This file covers
what's *in* the raw data and how it's normalized.

Nothing in this directory is hand-edited. If a file here needs to change, re-download
from the source URL above and re-run the curation script described
below; do not edit these BEDs directly.

## Files

| file | population | method | blocks | mean block size | max block size |
|---|---|---|---:|---:|---:|
| `deCODE_EUR_LD_blocks.bed` | European | deCODE pedigree recombination map | 1,361 | 2.06 Mb | 28.45 Mb |
| `pyrho_EUR_LD_blocks.bed` | European | pyrho population-LD map | 1,336 | 1.96 Mb | 6.50 Mb |
| `pyrho_AFR_LD_blocks.bed` | African | pyrho population-LD map | 1,580 | 1.66 Mb | 8.38 Mb *(not included)* |
| `pyrho_EAS_LD_blocks.bed` | East Asian | pyrho population-LD map | 1,121 | 2.33 Mb | 7.94 Mb *(not included)* |
| `pyrho_SAS_LD_blocks.bed` | South Asian | pyrho population-LD map | 1,267 | 2.07 Mb | 7.04 Mb *(not included)* |

All five are GRCh38-native, autosomes only (`chr1`–`chr22`, no liftOver needed).

## Raw format quirk

Each file is nominally tab-separated `chr / start / stop` with a header row, but
**not uniformly**:

- `pyrho_*`: clean, `chr\tstart\tend`, no padding.
- `deCODE_EUR_LD_blocks.bed`: padded with a literal space on each side of the tab
  (`chr1 \t 16103 \t 2888443`), including in the `chr` field itself
  (`"chr1 "`, trailing space). Any consumer must `.strip()` each field, not just
  split on `\t`.

`preprocess/regions/build_ld_block_bed.py` handles both: it strips whitespace per field,
drops the header row, validates ascending/non-overlapping order per chromosome, and
writes a clean unpadded 3-column BED (0-based half-open). It does **not** require
gap-free tiling — see below for why.

## Contiguity check (per-chromosome, `stop[i] == start[i+1]`)

Checked every file by sorting each chromosome's blocks by `start` and confirming each
block's `stop` equals the next block's `start`, with no overlaps.

- **`deCODE_EUR_LD_blocks.bed` is fully contiguous** — 0 gaps across all 22
  chromosomes. Every chromosome is tiled edge-to-edge from its first block's start to
  its last block's stop.
- **All four `pyrho_*` files have real gaps** — 17–18 gaps each, one per
  metacentric/submetacentric autosome, sized 2.9–33.5 Mb. These line up with each
  chromosome's centromeric/pericentromeric heterochromatin: pyrho's recombination-rate
  estimator has no signal there (highly repetitive, sparse informative variants), so
  the map simply omits the region rather than emitting a block for it. The five
  acrocentric autosomes (`chr13`, `chr14`, `chr15`, `chr21`, `chr22`) mostly show no
  internal gap — not because they're exempt, but because their unmappable p-arm sits
  before the map's start coordinate rather than in the middle (one exception:
  `pyrho_EAS` also has a gap on `chr14`).

  This is why `build_ld_block_bed.py`'s validator only checks ascending order and
  non-overlap, not `stop[i] == start[i+1]` — that invariant holds for deCODE but is
  false by construction for pyrho.

- All five files use the **same per-chromosome start/end anchors** regardless of
  method or population (e.g. `chr1` always starts at `16103` and ends at
  `248945661`; `chr2` always starts at `10560` and ends at `242171545`) — only the
  *internal* block boundaries differ between files.

## deCODE vs. pyrho: what's actually different

Both describe LD block partitions of the EUR population on GRCh38, but they're built
by unrelated methods and are not interchangeable:

- **`deCODE_EUR_LD_blocks.bed`** uses the deCODE (Icelandic) pedigree-based genetic
  recombination map — recombination events observed directly in tens of thousands of
  parent-offspring transmissions — to place block boundaries at recombination
  hotspots. Because the underlying genetic map assigns *some* recombination distance
  to every base pair (including through centromeres), the resulting blocks tile each
  chromosome with no gaps; the centromere/heterochromatin just ends up folded into one
  unusually large block (up to 28.45 Mb, vs. a 2.06 Mb mean elsewhere).
- **`pyrho_EUR_LD_blocks.bed`** (and the AFR/EAS/SAS siblings) is built from the
  `pyrho` composite-likelihood recombination-rate estimator (Spence & Song) run on
  population genetic diversity/LD patterns from 1000 Genomes superpopulations —
  a statistical LD-based estimate, not a pedigree observation. It has no method for
  estimating recombination rate where there isn't enough informative variation
  (centromeric/pericentromeric heterochromatin), so those spans are dropped outright
  rather than merged into a block — hence the real gaps documented above.
- Consequently even the two EUR maps disagree on block count (1,361 vs. 1,336) and
  boundaries do not align block-for-block, despite sharing the same chromosome-level
  start/end anchors.

`deCODE_EUR_LD_blocks.bed` is no longer merely "the one used downstream" — as of
2026-07-26 it **defines a named dataset region set**. Normalized to
`preprocess/regions/ld_blocks_eur_grch38.bed`, it is the `ldb_decode_eur` region set
pinned by the `v4-ldb` / `v6-ldb` / `v7-ldb` dataset versions in
`preprocess/huggingface_dataset/versions.py`, which record its sha256
(`d9f271ca…b0ea`) and reject any other `--bed`. Two consequences for this
directory:

- **Editing or re-downloading `deCODE_EUR_LD_blocks.bed` breaks those presets.**
  The normalized BED's hash is pinned in `versions.py` and asserted by
  `test_versions.py::test_ldb_pinned_hash_matches_the_bed_on_disk`. If the file
  must change, re-run `preprocess/regions/build_ld_block_bed.py`, update the pin, and
  treat existing `-ldb` datasets as a different region set.
- **The `pyrho_*` files are still unpinned** and kept for a future multi-ancestry
  comparison. Promoting one would mean adding its own presets (`-ldbpy-eur`, …) —
  at which point the suffixed-version scheme should probably be replaced by a
  separate region-set axis; see the "Region sets" section of
  `preprocess/huggingface_dataset/README.md`.

Any pipeline that assumes gap-free genome coverage (e.g. "every position is
assigned to exactly one LD block") must not swap in a `pyrho_*` file without
accounting for the centromeric gaps.
