# Third-party notices

The tasks in this repository download third-party data at runtime. Review the terms of each
source before use.

## Datasets

| Source | Used by | License |
|---|---|---|
| https://huggingface.co/datasets/scrc-dnai/clinvar-diploid-snv | `gfmbench_api/tasks/concrete/clinvar_diploid_task.py` | `<DATASET-LICENSE>` |
| https://huggingface.co/datasets/scrc-dnai/clinvar-diploid-indel | `gfmbench_api/tasks/concrete/clinvar_diploid_indel_task.py` | `<DATASET-LICENSE>` |

Both datasets are derived from NCBI ClinVar releases:

| Source | License |
|---|---|
| https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-01.txt.gz | LicenseRef-NCBI-Data |
| https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarVCVRelease_2026-01.xml.gz | LicenseRef-NCBI-Data |

## Reference genome

| Source | Used by | License |
|---|---|---|
| https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | both tasks, fetched via `gfmbench_api.utils.fileutils.ensure_reference_genome` | LicenseRef-UCSC-Genome-Browser — https://genome.ucsc.edu/license/ |

## Python packages

Python dependency attributions are covered by GFMBench-API's own
[THIRD_PARTY_NOTICES.md](https://github.com/NVIDIA/GFMBench-api/blob/main/THIRD_PARTY_NOTICES.md);
this repository adds no dependency beyond the ones it already lists.
