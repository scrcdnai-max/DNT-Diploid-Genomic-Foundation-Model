# Benchmark dataset builders

Scripts that build the diploid benchmark datasets. The ClinVar SNV and indel outputs are published as
[`scrc-dnai/clinvar-diploid-snv`](https://huggingface.co/datasets/scrc-dnai/clinvar-diploid-snv) and
[`scrc-dnai/clinvar-diploid-indel`](https://huggingface.co/datasets/scrc-dnai/clinvar-diploid-indel),
which the tasks in [`../gfmbench/`](../gfmbench/README.md) download. Rebuilding them is only
needed to change or audit them.

| Script | Output |
|---|---|
| `create_diploid_snv_parquet.py` | ClinVar SNVs with zygosity labels (`label_het` / `label_hom`: benign 0/0, dominant pathogenic 1/1, recessive pathogenic 0/1), mode of inheritance parsed from the ClinVar XML release, and the diploid token for each genotype |
| `create_diploid_indels_parquet.py` | ClinVar insertions and deletions, one het and one hom row per variant (`label_variant`, plus `label_genotype` with the same zygosity rule) |
| `create_ancestry_diploid_parquet.py` | 1000 Genomes diploid MHC-region sequences with super-population labels, split by individual; optionally a Hugging Face `DatasetDict` (`--help` for options) |

The ClinVar scripts download the 2026-01 `variant_summary` table and VCV XML release from NCBI
and write under `../data/` (`diploid_clean/`, `diploid_clean_indel/`), relative to the working
directory, matching a GFMBench-API `root_data_path` of `../data`. Sequence windows are cut at
task time from GRCh38, not here. The SNV script also adds the diploid token of each genotype,
reading the table from `../data/diploid_clean/diploid_tokens.tsv` (copy
`preprocess/diploid_tokens.tsv` there; it must be the table the model was trained with).

Requires `pandas`, `numpy`, `pyfaidx`, `pysam`, `pyarrow`, `datasets` and, for faster XML
parsing, `lxml`.
