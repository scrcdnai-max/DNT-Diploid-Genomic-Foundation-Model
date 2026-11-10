# nested_genome_division

One biologically-motivated division of hg38 into per-gene chunks, usable directly as
`create_dataset.py generate --bed` (plain BED, `hg38_tss_tes.bed`) and, in its annotated form
(`hg38_tss_tes_v10.bed`, same coordinates), as the input of the v10 per-patient haplotype-pair
dataset (`../../huggingface_dataset/create_haplotype_dataset.py`).

| mode | scope | intervals | coverage | disjoint? | what it is |
|---|---|---:|---:|---|---|
| `tss-tes` | chr1-22 + **chrX** | 59,572 | 1.948 Gbp (69.4% of mappable) | **no** | **one interval per gene**, TSS→TES + promoter margin |

Rows overlap wherever the genes do: this is a per-gene division, so a snoRNA in an intron gets its
own row inside its host gene's row. `validate_division.py` says so out loud — `total` (sum of row
lengths, 2.313 Gbp) and `covered` (distinct bases, 1.948 Gbp) are reported separately, a 1.19x
redundancy.

> **Previously there were two modes.** `tad-tss-tes` — TADs called from a 15.8 GB 4DN HFFc6 `.mcool`,
> subdivided by genes — was retired on 2026-09-01 along with the 256-bp / 10–150 kbp size contract
> that existed to make it tile. Both are frozen at tag `pre-reorg` (was `legacy/`), which nothing
> imports. No dataset was ever built on the TAD region set.

## The size contract

Every interval satisfies, without exception:

* `length % 128 == 0`

That is the whole contract. **There is no minimum and no maximum** — lengths run from 1,152 bp (a
94 bp `vault_RNA` plus its promoter margin) to 2,474,624 bp (the longest coding gene). A consumer
that assumed the old 10–150 kbp band must check the region set name, not just the encoding: it is
`tss_tes_gencode50_tx128`, and the `_tx128` suffix is there precisely so a stale assumption fails
loudly.

[`sizing.py`](sizing.py) owns the length algebra; [`validate_division.py`](validate_division.py)
re-checks every invariant, and the builder runs it before writing, so a BED that reaches disk has
already passed.

## How a gene becomes an interval

**Input is GENCODE v50 `transcript` records, not `gene` records.** Transcripts of the eleven
biotypes below are merged per `gene_id` into a TSS→TES envelope — `min(start)..max(end)` over the
gene's *eligible* transcripts. Then, for each gene:

1. **Exclusion gate.** If the envelope touches an unreliably-sequenced region — a UCSC assembly gap
   or the ENCODE blacklist v2 — the gene is **dropped whole**. No fragment-splitting: a gene that
   straddles a centromere contributes nothing rather than two half-genes.
2. **Promoter margin.** 1,000 bp is added on the 5' side — `start -= 1000` on `+`, `end += 1000`
   on `-`. A thumb rule, not a measured promoter; `--upstream` changes it.
3. **Grid.** The 3' end is pushed out by `(-length) % 128` bases, whatever it takes to reach the
   next multiple of 128. The slack goes downstream because the promoter margin is the part with a
   biological claim on its position and the grid remainder is not.
4. **Walls.** If that window runs past the end of the free region the gene sits in (the stretch
   between two excluded regions, clipped to the chromosome), it slides just far enough to clear the
   wall. The bases the wall takes back are added on the opposite end, so the interval keeps its
   **exact** length and stays 128-divisible. 23 intervals genome-wide need this.

Genes are placed **independently** — overlapping a neighbour is allowed, so no gene ever has to give
way to another.

```
473,038  transcript records in the 11 biotypes (chr1-22 + chrX)
 61,374  genes after the per-gene merge
 -1,802  envelopes touch a gap or the blacklist  ->  dropped whole
 59,572  intervals
```

### What is in

chr1-chr22 **and chrX**. Eleven `transcript_type` values:

| biotype | intervals | | biotype | intervals |
|---|---:|---|---|---:|
| `lncRNA` | 33,576 | | `snoRNA` | 962 |
| `protein_coding` | 19,131 | | `scaRNA` | 49 |
| `misc_RNA` | 2,141 | | `rRNA` | 38 |
| `snRNA` | 1,835 | | `ribozyme` | 8 |
| `miRNA` | 1,823 | | `sRNA` | 5 |
| | | | `vault_RNA` | 4 |

### What is out

* **Chromosomes:** chrY (present in only half a cohort, and its ampliconic palindromes are largely
  blacklisted anyway) and chrM (a 16.6 kb circle whose 37 genes would each wrap most of it), plus
  all scaffolds. `Mt_rRNA`/`Mt_tRNA` are consequently not in the biotype list at all — under this
  chromosome scope they could only ever be a no-op.
* **Biotypes:** every pseudogene class (`processed_pseudogene` 9,348 · `unprocessed_pseudogene`
  1,789 · `transcribed_*_pseudogene` 2,852 · `rRNA_pseudogene` 489 · `unitary_pseudogene` 90 —
  ~14,800 genes, by far the largest excluded group), `TEC` 1,089, the rearranging `IG_*`/`TR_*` gene
  segments (the reference locus is not what any given B or T cell actually transcribes), and
  `artifact`.
* **Isoform classes of coding genes:** `nonsense_mediated_decay`, `retained_intron`,
  `protein_coding_CDS_not_defined`, `non_stop_decay`, `protein_coding_LoF`, `processed_transcript`.
* **Sequence:** UCSC assembly gaps and the ENCODE blacklist v2 — 224.5 Mbp over the in-scope
  chromosomes.

### Selecting on `transcript_type` is not selecting on `gene_type`

For these eleven biotypes GENCODE v50 has **zero** rows where `transcript_type != gene_type`, so no
biotype disagreement has to be resolved. But the two selections still differ, in one direction that
matters:

**19,676 genes carry at least one `protein_coding` transcript, against 20,033 `protein_coding` gene
records. The 357-gene difference is genes annotated *only* through `retained_intron` /
`nonsense_mediated_decay` / `protein_coding_CDS_not_defined` transcripts.** They have no eligible
transcript, so they are absent from the division entirely. 36 `lncRNA` genes go the same way, for
393 in total. Likewise, a kept gene's envelope is its
*eligible-transcript* extent, which can be narrower than the gene record when a retained-intron
isoform runs further. Both follow directly from "take only transcripts from these categories" and
are intended; `test_build_division.py` pins them so they cannot be re-discovered as a bug.

## Interval sizes

Coding against non-coding, because they are two different distributions wearing one histogram.
`total length` is the sum of row lengths, not distinct coverage — these rows overlap.

| | # intervals | min (bp) | median (bp) | **mean (bp)** | max (bp) | total length (Mb) |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 59,572 | 1,152 | 10,368 | **38,833** | 2,474,624 | 2,313.4 |
| **coding** (`protein_coding`) | 19,131 | 1,152 | 28,672 | **67,664** | 2,474,624 | 1,294.5 |
| **non-coding** (10 biotypes) | 40,441 | 1,152 | 5,120 | **25,194** | 1,376,384 | 1,018.9 |
| &nbsp;&nbsp;· `lncRNA` | 33,576 | 1,152 | 8,064 | **30,101** | 1,376,384 | 1,010.7 |
| &nbsp;&nbsp;· small + structural RNA | 6,865 | 1,152 | 1,152 | **1,196** | 1,792 | 8.2 |

Three things the split makes visible that the pooled row hides:

* **The pooled median (10,368) describes neither group.** Coding sits at 28,672, non-coding at
  5,120. The pooled number is just the mixing ratio.
* **The nine small/structural-RNA classes are now 8.2 Mb, not 70.3 Mb.** Under the old 10,240 bp
  floor every one of those 6,865 intervals was padded to exactly 10,240 bp — 63.8 Mb of context
  wrapped around 0.91 Mb of annotation. With no floor they are their own span plus 1 kb: a median of
  1,152 bp and a maximum of 1,792. That is the single largest behavioural change of the rebuild.
* **`lncRNA` straddles both.** Median 8,064, mean 30,101 — half of them are short, and the other
  half include some of the longest genes in the file.

Rows sum to **2,313.4 Mb** but cover **1,948.1 Mb** of distinct sequence: 365.3 Mb is
double-counted, 1.19x redundancy. That is the cost of the per-gene view, and it is reported on every
build (`total` vs `covered` in the validator summary).

<details><summary>Per biotype</summary>

| biotype | intervals | min | median | mean | max | sum |
|---|---:|---:|---:|---:|---:|---:|
| `lncRNA` | 33,576 | 1,152 | 8,064 | 30,101 | 1,376,384 | 1,010.70 Mb |
| `protein_coding` | 19,131 | 1,152 | 28,672 | 67,664 | 2,474,624 | 1,294.48 Mb |
| `misc_RNA` | 2,141 | 1,152 | 1,280 | 1,275 | 1,536 | 2.73 Mb |
| `snRNA` | 1,835 | 1,152 | 1,152 | 1,163 | 1,408 | 2.14 Mb |
| `miRNA` | 1,823 | 1,152 | 1,152 | 1,152 | 1,280 | 2.10 Mb |
| `snoRNA` | 962 | 1,152 | 1,152 | 1,166 | 1,792 | 1.12 Mb |
| `scaRNA` | 49 | 1,152 | 1,152 | 1,201 | 1,536 | 0.06 Mb |
| `rRNA` | 38 | 1,152 | 1,152 | 1,152 | 1,152 | 0.04 Mb |
| `ribozyme` | 8 | 1,152 | 1,280 | 1,296 | 1,408 | 0.01 Mb |
| `sRNA` | 5 | 1,152 | 1,152 | 1,177 | 1,280 | 0.01 Mb |
| `vault_RNA` | 4 | 1,152 | 1,152 | 1,152 | 1,152 | 0.00 Mb |

</details>

## Why 59,572 and not ~76,000

The previous build of this file had 67,123 rows. It got there from 61,767 genes by **splitting every
gene over 149,760 bp into abutting pieces (+6,279 rows)** — a consequence of the size ceiling, not of
scope. With no ceiling, that source of extra rows is gone.

Reaching ~76,000 would take a different biotype policy, not a different algorithm:

| scope | genes | intervals |
|---|---:|---:|
| **this division** — 11 biotypes, drop a gene whose body hits a gap | 61,374 | **59,572** |
| + all pseudogene classes | 76,181 | 73,564 |
| + IG/TR segments + TEC + the coding-adjacent isoform classes + artifact | 78,024 | 75,301 |
| + split a gene at a gap instead of dropping it whole | 78,024 | 76,281 |
| *(alternative)* re-add a 149,760 / 131,072 bp ceiling to the 11 biotypes | 61,374 | 65,311 / 66,761 |

Only "every transcript_type, split at gaps" reaches ~76,000, and it does so by admitting ~14,800
pseudogenes. Both levers were considered and declined; 59,572 is the intended count.

## Output format

BED6+1, so column 4 stays a bare gene symbol and `create_dataset.py --genes-file` keeps working:

```
chrom  start    end      gene_symbol      0  strand  gene_type
chr1   816362   819946   FAM87B           0  +       lncRNA
chr1   824137   868937   LINC01128        0  +       lncRNA
chr1   833034   836618   ENSG00000297178  0  -       lncRNA
```

The old column 8 (`piece`, `1of3` etc.) is gone: nothing splits any more. Any reader that takes only
columns 1-3, or 1-4, is unaffected.

### Annotated BED (`--annotated-out`)

`build_division.py --annotated-out <bed>` writes the **same intervals** a second time, as the
input the v10 per-patient haplotype-pair dataset needs — a label that is unique and stable (the
symbol in column 4 of the plain BED is neither: symbols repeat, and `create_dataset.py
--genes-file` matches that column exactly, so it cannot be decorated) plus every annotation that
should reach every dataset row:

```
#region_set=tss_tes_gencode50_tx128_v10
#chrom  start   end     name               score  strand  gene_name  gene_type  n_transcripts
chr1    816362  819946  ENSG00000177757.3  0      +       FAM87B     lncRNA     1
chr1    824137  868937  ENSG00000228794.13 0      +       LINC01128  lncRNA     94
```

* Headered BED6+3. Column 4 is the GENCODE `gene_id` **verbatim, version suffix included**;
  `gene_name`, `gene_type` and `n_transcripts` (the eligible-transcript count) follow the strand.
  The `#chrom…` header names the columns; the v10 builder propagates **every** header-named column
  beyond the six BED fields as a `region_<col>` Parquet column, so adding an annotation later is a
  change to this builder only.
* `<stem>.genes.tsv` beside it, one row per placed gene: `gene_id gene_name gene_type strand
  chrom tss_tes_start tss_tes_end transcript_ids transcript_names transcript_types hgnc_id level
  gencode_version` (`;`-joined lists; `tss_tes_start/end` is the TSS→TES envelope *before* the
  promoter margin and grid padding, so it differs from the BED interval). Dropped genes are not
  listed. `create_haplotype_dataset.py generate` copies it to `<out>/region_annotations.tsv`.
* **Columns 1-3 + strand are byte-identical to `hg38_tss_tes.bed`**, row for row (same inputs,
  same algorithm; a test diffs the two outputs and the real build is checked with an md5 over
  those columns). That is what lets the BioMe `bcf_tss_tes_per_sample` dir (165 GB, 11,586
  samples), cut on the plain BED's union, serve as v10 input without a new BCF build.
* `hg38_tss_tes.bed` is untouched: its sha256 stays pinned in `versions.TSSTES_BED_SHA256`, and the
  annotated file has its own region-set name (`…_v10`) so nothing mistakes one for the other.
  `--out` becomes optional when `--annotated-out` is given (`/dev/null` is accepted).

```bash
# from the repo root; D=preprocess/regions/nested_genome_division
uv run --no-sync python -m preprocess.regions.nested_genome_division.build_division --out /dev/null --annotated-out $D/hg38_tss_tes_v10.bed --quiet
cut -f1-3,6 $D/hg38_tss_tes.bed | md5sum; grep -v '^#' $D/hg38_tss_tes_v10.bed | cut -f1-3,6 | md5sum   # must match
uv run --no-sync python -m preprocess.regions.nested_genome_division.validate_division $D/hg38_tss_tes_v10.bed                     # skips the header
```

## Regenerate

```bash
# from the repo root (the modules import `preprocess.…` absolutely, so run them with -m)
D=preprocess/regions/nested_genome_division
uv run --no-sync python -m preprocess.regions.nested_genome_division.build_division --out $D/out/hg38_tss_tes.bed
uv run --no-sync python -m preprocess.regions.nested_genome_division.validate_division $D/out/hg38_tss_tes.bed
cp $D/out/hg38_tss_tes.bed $D/hg38_tss_tes.bed
sha256sum $D/hg38_tss_tes.bed     # re-pin as TSSTES_BED_SHA256 in preprocess/huggingface_dataset/versions.py
```

The builder runs the validator before writing, so a BED that lands in `out/` has already passed.
After re-pinning the hash, run `uv run --no-sync pytest preprocess/tests/test_versions.py`,
which checks the pin against the file on disk.

`build_artifact_data.py --verify-only` is the strongest single check in this directory: it
re-derives every row through an independent re-implementation of the placement loop and refuses to
proceed unless it reproduces the committed BED row for row.

```bash
uv run --no-sync python -m preprocess.regions.nested_genome_division.build_artifact_data --verify-only
```

## Files

| file | what |
|---|---|
| [`build_division.py`](build_division.py) | the builder; one flat CLI, no subcommands |
| [`sizing.py`](sizing.py) | `GRID = 128`, `pad_downstream`, `place` |
| [`sources.py`](sources.py) | GENCODE transcript reader, gap/blacklist/chrom.sizes parsers, interval algebra |
| [`validate_division.py`](validate_division.py) | invariant checker; importable and a CLI |
| [`build_artifact_data.py`](build_artifact_data.py) | the JS data blob for the published browser artifact, plus `--verify-only` |
| [`hg38_tss_tes.bed`](hg38_tss_tes.bed) | the committed division, 59,572 rows |
| [`hg38_tss_tes_v10.bed`](hg38_tss_tes_v10.bed) | the same 59,572 intervals as a headered BED6+3 (`#region_set=tss_tes_gencode50_tx128_v10`; GENCODE `gene_id` in col 4, `gene_name gene_type n_transcripts`) — the v10 haplotype-pair dataset input; built with `--annotated-out` |
| `hg38_tss_tes_v10.genes.tsv` (not committed: 23 MB, written by `--annotated-out` next to the BED) | per-gene sidecar of the annotated BED (transcript ids/names/types, HGNC id, level, GENCODE version); copied into each v10 dataset as `region_annotations.tsv` |
| `raw/` | gitignored inputs: `gap.txt.gz`, `hg38-blacklist.v2.bed.gz`, `hg38.chrom.sizes` |

Data inputs (nothing is fetched; defaults are in `sources.py`, override with `--gencode`, `--gaps`, `--blacklist`, `--chrom-sizes`):

* GENCODE v50 — `gencode.v50.annotation.gtf.gz` (`--gencode`)
* UCSC assembly gaps — `raw/gap.txt.gz`
* ENCODE blacklist v2 — `raw/hg38-blacklist.v2.bed.gz`
* hg38 chromosome sizes — `raw/hg38.chrom.sizes`

## Building a dataset over it

Use [`../../huggingface_dataset/runners/run_tsstes_dataset.sh`](../../huggingface_dataset/runners/run_tsstes_dataset.sh)
for v7/v8-tsstes, or [`../../huggingface_dataset/runners/run_v10_dataset.sh`](../../huggingface_dataset/runners/run_v10_dataset.sh)
for the v10 per-patient haplotype pairs over the annotated BED — never bare `create_dataset.py`. The runner handles the three things this region set does that no
other one in the repo does: deriving the merged **union** BED for `bcftools -R` (23,165 regions /
1.948 Gbp — `-R` only needs the union, and feeding it 59,572 overlapping regions is wasted work),
requiring the full-panel BCFs, and **asserting chrX carries real variants** before generate. That
last guard is load-bearing: `create_dataset.py` reads available chromosomes from the BCF *header*,
so a dir built against a chr1-22 BED still advertises chrX and all 1,875 chrX regions would silently
yield zero variant rows.

> **Datasets built over this BED before 2026-10-05 (commit `875faec`) need regenerating** before they
> are used for training: the encoder's batch variant parser assumed regions never nest, which on this
> per-gene BED leaked a host gene's variants into nested regions and left 43 host-gene tails per
> sample unqueried. Fixed in the encoder; the regeneration is planned, not started. See
> [`../../huggingface_dataset/README.md`](../../huggingface_dataset/README.md) §"Planned: regenerate the
> `-tsstes` datasets".
