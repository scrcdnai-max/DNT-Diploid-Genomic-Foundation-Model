# Third-party notices

This repository commits a few files derived from third-party data, and its benchmark tasks and
dataset builders download or read others. Review the terms of each source before use.

## Datasets

| Source | Used by | License |
|---|---|---|
| https://huggingface.co/datasets/scrc-dnai/clinvar-diploid-snv | `benchmarks/gfmbench/gfmbench_api/tasks/concrete/clinvar_diploid_task.py` | `<DATASET-LICENSE>` |
| https://huggingface.co/datasets/scrc-dnai/clinvar-diploid-indel | `benchmarks/gfmbench/gfmbench_api/tasks/concrete/clinvar_diploid_indel_task.py` | `<DATASET-LICENSE>` |

Both datasets are derived from NCBI ClinVar releases (built by `benchmarks/datasets/`):

| Source | License |
|---|---|
| https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-01.txt.gz | LicenseRef-NCBI-Data |
| https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarVCVRelease_2026-01.xml.gz | LicenseRef-NCBI-Data |

## Reference genome

| Source | Used by | License |
|---|---|---|
| https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | both tasks, fetched via `gfmbench_api.utils.fileutils.ensure_reference_genome` | LicenseRef-UCSC-Genome-Browser — https://genome.ucsc.edu/license/ |

## Files committed in this repository

| File | Derived from | Terms |
|---|---|---|
| `preprocess/regions/ld_blocks/raw/*.bed`, `preprocess/regions/ld_blocks_eur_grch38.bed` | LD blocks from https://github.com/jmacdon/LDblocks_GRCh38 (bioRxiv 10.1101/2022.03.04.483057) | see the source repository |
| `preprocess/regions/hg38_pretrain_canonical_expanded.bed` | GENCODE Basic coding exons and UCSC 100-way phastCons elements on GRCh38 | https://www.gencodegenes.org/pages/data_access.html, https://genome.ucsc.edu/license/ |
| `preprocess/regions/nested_genome_division/hg38_tss_tes*.bed` | GENCODE v50 annotation, UCSC hg38 gap table, ENCODE blacklist v2 | as above; ENCODE blacklist: https://github.com/Boyle-Lab/Blacklist |
| `preprocess/regions/sample_splits/sample_split_20251209_v5/` | 1000 Genomes Project sample and pedigree identifiers | https://www.internationalgenome.org/IGSR_disclaimer |

## Data read by the preprocessing pipeline (not committed)

| Source | Used by |
|---|---|
| 1000 Genomes high-coverage phased panel (https://www.internationalgenome.org/data-portal/data-collection/30x-grch38) | `preprocess/huggingface_dataset/` |
| UCSC hg38 2bit, phyloP 100-way | `preprocess/` |
| CADD v1.7 (https://cadd.gs.washington.edu/) | `preprocess/huggingface_dataset/cadd_arrow/`, `--cadd-arrow-dir` |
| InstaDeep Nucleotide Transformer v3 base models (https://huggingface.co/InstaDeepAI) | `pretrain/train.py` |

## Python packages

Python dependency attributions are covered by GFMBench-API's own
[THIRD_PARTY_NOTICES.md](https://github.com/NVIDIA/GFMBench-api/blob/main/THIRD_PARTY_NOTICES.md);
the benchmark tasks add no dependency beyond the ones it already lists. The preprocessing and
pretraining code depend on the packages in `requirements.txt`.
