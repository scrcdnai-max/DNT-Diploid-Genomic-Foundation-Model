# preprocess — diploid encoding and the pretraining corpus

Turns phased genotypes into the diploid token strings DNT reads, and builds the pretraining
dataset from them.

## The encoding

`diploid_tokens.tsv` is the vocabulary: one Unicode symbol per allele pair (homozygous,
heterozygous unphased, heterozygous phased), plus the structural markers written into the
sequence:

| Marker | Meaning |
|---|---|
| `^` | the next symbol is a SNV site |
| `{` … `}` | an indel span (both haplotypes aligned inside it, gaps as `-`) |
| `‹` / `›` | phase direction of the next heterozygous symbol: which copy carries which allele |

`tokens.py` is the one reader of the table; `markers.py` holds the marker constants.
`diploid_tokens_v2.tsv` / `_v3.tsv` and the matching `token_counts_*.tsv` are the tables for the
one-token-per-position input of `pretrain/v2_input.py` (v3 also records which copy carries the
reference allele). `token_counts*.tsv` give corpus unigram frequencies; `pretrain/train.py` uses
them to drop unseen symbols and to set the new LM-head bias prior.

`encoder/get_genome_sequence.py` (`GenomeSequenceProcessor`) renders one sample over one region
against GRCh38 (`encoder/reference.py` reads `hg38.2bit`).

## Building the dataset

`huggingface_dataset/create_dataset.py` has three subcommands:

```bash
# 1. one Parquet shard per sample (resumable, parallel)
python -m preprocess.huggingface_dataset.create_dataset generate \
    --bcf-dir <1kGP per-chromosome BCFs> \
    --bed preprocess/regions/hg38_pretrain_canonical_expanded.bed \
    --ref preprocess/hg38.2bit --tokens preprocess/diploid_tokens.tsv \
    --test-samples-file preprocess/regions/sample_splits/sample_split_20251209_v5/1kg_test_samples_v5.txt \
    --version v7 --output <out> --workers 64

# 2. assemble shards into train/test Hugging Face datasets
python -m preprocess.huggingface_dataset.create_dataset assemble --help

# 3. check whether one dataset version can be converted into another
python -m preprocess.huggingface_dataset.create_dataset check-compat --help
```

`--version` is required and selects a preset in `huggingface_dataset/versions.py`, the single
source of truth for the schema:

| Version | Encoding | Region set |
|---|---|---|
| `v4` | phased symbols, inline `^`/`{}`, no direction markers | any `--bed` |
| `v5` | adds `‹`/`›` direction markers | any |
| `v6` | adds the compound-het anchor `^` | any |
| `v6nd` | `v6` with direction markers removed and phase dropped | any |
| `v7` | `v6` + optional per-base phyloP / CADD columns | any |
| `v8` | `v7` with markers moved out of `seq` into sparse columns | any |
| `v4-ldb`, `v6-ldb`, `v7-ldb` | as above | deCODE EUR LD blocks (sha256-pinned) |
| `v7-tsstes`, `v8-tsstes` | as above | per-gene TSS–TES spans (sha256-pinned) |

`--representation haploid` writes plain ACGTN strings instead (ablation). `create_haplotype_dataset.py`
builds the experimental `v10` format, which keeps the two haplotypes as separate columns.
`dataset.py` (`DiploidDataset` / `HaploidDataset`) reads an assembled dataset and refuses the
wrong representation; `provenance.json` next to each dataset records how it was built.

`huggingface_dataset/runners/` are the shell wrappers used to build the published corpora
(`run_canonical_dataset.sh --version v7`, `run_tsstes_dataset.sh`, `run_ldb_dataset.sh`,
`run_v10_dataset.sh`, plus the per-sample BCF helpers). `run_canonical_dataset.sh`,
`run_ldb_dataset.sh`, `run_tsstes_dataset.sh` and `run_v10_dataset.sh` take `--dry-run`, which
prints the resolved commands without running anything. Each script's header lists its options.

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `<repo>/data/1kgenomes` | input BCFs and output datasets (layout below) |
| `PYTHON` | `python3` on `PATH` | interpreter of the environment `requirements.txt` is installed in |
| `PHYLOP_BW`, `CADD_ARROW_DIR` | none | per-base annotations; `--phylop-bw` / `--cadd-arrow-dir` for `run_canonical_dataset.sh` |
| `DOCKER_IMAGE` | `glm_biotools:1.2.0` | bcftools image for the per-sample BCF steps, built from [`docker_images/glm_biotools/`](../docker_images/glm_biotools/README.md) |
| `DOCKER_EXTRA_MOUNTS` | none | directories outside `REPO` and `DATA_ROOT` the container must see |
| `BCF_DIR`, `RAW_BCF_DIR`, `BED`, `REF`, `TOKENS`, `TEST_SAMPLES`, `OUT_DIR`, `WORKERS` | see script | per-run overrides |

Layout under `DATA_ROOT` the runners expect (directories they create are marked *out*):

| Path | Contents |
|---|---|
| `raw_bcf/` | 1000 Genomes high-coverage phased panel, one multi-sample BCF per chromosome (chr1–22, X) |
| `raw_bcf_ld_blocks_eur_intersected/` | *out* of `preprocess/regions/intersect_1kg_ld_blocks.sh` |
| `raw_bcf_ld_blocks_eur_per_sample/`, `bcf_tss_tes_per_sample/` | *out*: per-sample BCFs restricted to a region set (`make_persample_bcf.sh`, or `--build-bcf`) |
| `raw_bcf_bed_intersected_hg38_pretrain_canonical_expanded_per_sample/` | per-sample BCFs restricted to the canonical BED, for `run_canonical_dataset.sh` |
| `<representation>_hf_<version>/`, `diploid_hf_<version>/`, `haplotype_pairs_v10/` | *out*: shards, `assembled/{train,test}`, logs |

Per-base annotations: `cadd_arrow/` converts CADD v1.7 to the Arrow layout `--cadd-arrow-dir`
expects; `--phylop-bw` takes the UCSC phyloP 100-way bigWig directly.

## Region sets and sample split

| File | What |
|---|---|
| `regions/hg38_pretrain_canonical_expanded.bed` | canonical pretraining regions: 140,099 regions, ~0.94 Gbp |
| `regions/ld_blocks_eur_grch38.bed` | 1,361 deCODE EUR LD blocks (2.8 Gbp), built from `regions/ld_blocks/raw/` by `regions/build_ld_block_bed.py` |
| `regions/nested_genome_division/hg38_tss_tes.bed`, `…_v10.bed` | 59,572 per-gene TSS–TES spans on a 128-bp grid; rebuilt by `build_division.py` ([README](regions/nested_genome_division/README.md)) |
| `regions/nested_genome_division/hg38_tss_tes_v10.genes.tsv` | per-gene annotation of the v10 BED (GENCODE ids, names, types, transcripts, HGNC id); copied into each v10 dataset as `region_annotations.tsv` |
| `regions/sample_splits/sample_split_20251209_v5/` | the pedigree-aware, population-stratified 1000 Genomes train/test split ([README](regions/sample_splits/sample_split_20251209_v5/README.md)) |

## Tools

| Tool | What |
|---|---|
| `tools/view_data.py` | print records of an assembled dataset in readable form ([README](tools/VIEW_DATA_README.md)) |
| `tools/verify_encoding.py` | walk encoded records and the source BCF in lockstep and check every variant |
| `tools/gather_diploid_stats_bcf.py`, `huggingface_dataset/tools/gather_dataset_stats.py` | variant / token statistics of the input BCFs and of a built dataset |
| `huggingface_dataset/tools/compare_assembled.py` | fingerprint and compare two assembled datasets across machines |
| `huggingface_dataset/tools/chunk_union_bed.py` | cut a merged region BED into per-chromosome chunks for a distributed `bcftools +split` |
| `huggingface_dataset/tools/extract_cadd_for_bed.py`, `rebuild_cadd.py` | CADD sidecars for an existing dataset |
