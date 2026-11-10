# Diploid ClinVar tasks for GFMBench-API

Zygosity-aware ClinVar benchmark tasks for
[GFMBench-API](https://github.com/NVIDIA/GFMBench-api).

Standard variant-effect benchmarks score a haploid alt sequence against a haploid reference,
so a heterozygous and a homozygous carrier of the same variant are indistinguishable. These
tasks encode both haplotypes into a single diploid sequence — one glyph per position, drawn
from a diploid token table — so zygosity is part of the model input, and a recessive
pathogenic variant can be labelled `het = 0` (carrier) / `hom = 1` (affected).

## Contents

| Path | What |
|---|---|
| `gfmbench_api/tasks/concrete/clinvar_diploid_task.py` | `DiploidClinvarSupervisedTask` — supervised variant-effect classification |
| `gfmbench_api/tasks/concrete/clinvar_diploid_indel_task.py` | `DiploidIndelClinvarTask` — zero-shot indel variant effect |
| `gfmbench_api/utils/diploidutils.py` | diploid encoding/decoding, SNV and indel window extraction, run-mode dataset wrappers |
| `usage_examples/run_diploid_benchmark.py` | runner with the diploid / haploid / biallelic mode fan-out |
| `usage_examples/trainers/linear_probe.py` | embedding probe fitters (`SNVEmbeddingProbeFitter`, `SequenceEmbeddingProbeFitter`) |
| `usage_examples/trainers/__init__.py` | upstream's file with the probe-fitter exports added |

Paths in the table are relative to `benchmarks/gfmbench/`. The layout mirrors the GFMBench-API tree so the files drop straight into a checkout. Package
`__init__.py` files are deliberately absent: nothing here replaces an upstream module except
`usage_examples/trainers/__init__.py`, which adds two exports.

## Tasks

**`diploid_clinvar_supervised_variant_effect`** — `BaseGFMSupervisedVariantEffectTask`.
Binary pathogenicity (single label, two classes) over variant/reference diploid sequence
pairs, fine-tuned or linear-probed. Train/test splits come from the published dataset, whose
test split is chromosomes 19–22. Metrics: accuracy, MCC, AUROC, AUPRC.

Zygosity logic behind the labels:

| Variant | het | hom |
|---|---|---|
| Benign | 0 | 0 |
| Pathogenic, dominant | 1 | 1 |
| Pathogenic, recessive | 0 (carrier) | 1 (affected) |

**`clinvar_diploid_indel`** — `BaseGFMZeroShotGeneralIndelTask`. Zero-shot pathogenicity for
insertions and deletions, scored by the zero-shot metrics of the base class (sum-probability
LLR, embedding cosine/L2 AUROC and AUPRC).

## Data

Both tasks download their own data; nothing needs staging.

| Source | Used for |
|---|---|
| [`scrc-dnai/clinvar-diploid-snv`](https://huggingface.co/datasets/scrc-dnai/clinvar-diploid-snv) | SNV task, `train` + `test` splits |
| [`scrc-dnai/clinvar-diploid-indel`](https://huggingface.co/datasets/scrc-dnai/clinvar-diploid-indel) | indel task, `test` split |
| hg38 (UCSC) | reference windows, fetched on first use into `<root_data_path>/reference_genome/` |

Each task reads a locally staged `diploid_tokens.tsv` when one exists
(`<root_data_path>/clinvar_zygosity/` for SNV, `<root_data_path>/diploid_clean/` for indel)
and otherwise downloads the table published alongside the data. Local-first is deliberate:
the table defines the vocabulary the sequences are encoded with and must match the one the
model was trained on, so a machine that already has a table keeps producing results
comparable with its earlier runs.

## Install

```bash
benchmarks/gfmbench/install.sh /path/to/GFMBench-api
```

Copies `gfmbench_api/` and `usage_examples/` into the checkout. A `PYTHONPATH` overlay cannot
work here — `gfmbench_api` is a regular package, so Python resolves it to one directory and
will not merge two.

## Run

```bash
cd /path/to/GFMBench-api
python usage_examples/run_diploid_benchmark.py --help
```

Diploid-capable tasks run in three modes, selected with `--modes`:

- `diploid` — the task's native encoding, both haplotypes in one sequence;
- `haploid` — one sample per row: the variant-bearing haplotype against the reference haplotype;
- `biallelic` — the two alleles of the variant sequence against each other. Homozygous rows
  give two identical alleles, so scores from this pair measure zygosity, not variant effect.

`--gap_policy` controls how indel alignment gaps are handled when decoding to haploid or
biallelic: `drop` (default) emits real ungapped haplotypes of differing length, `keep`
preserves column alignment at the cost of an untrained gap token.

## Requirements

Python ≥ 3.10 (GFMBench-API's floor), plus `datasets`, `huggingface_hub`, `pyfaidx`, `torch`,
`pandas`, `numpy`, `scikit-learn`.

## Tests

```bash
cd /path/to/GFMBench-api      # after install.sh
PYTHONPATH=. pytest /path/to/this/repo/benchmarks/gfmbench/tests
```

The tests skip themselves when `gfmbench_api` is not importable.

The ClinVar datasets themselves are built by [`../datasets/`](../datasets/README.md).
