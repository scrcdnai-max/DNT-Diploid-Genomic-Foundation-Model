# DNT: Diploid Genomic Foundation Model

Code for DNT, a genomic language model that reads a person's **two haplotypes as one sequence**.
Each position is a single token drawn from a diploid vocabulary: homozygous and heterozygous
genotypes, phase direction and indel spans are all part of the input. Zygosity is visible to the
model, so a heterozygous and a homozygous carrier of the same variant are no longer the same input.

DNT is continually pretrained from InstaDeep's
[Nucleotide Transformer v3](https://huggingface.co/InstaDeepAI) on phased genomes from the 1000
Genomes high-coverage release, with the vocabulary expanded from 11 to the diploid token set.
The released models are on the Hugging Face Hub under
[`scrc-dnai`](https://huggingface.co/scrc-dnai).

## Repository layout

| Path | What |
|---|---|
| [`preprocess/`](preprocess/README.md) | the diploid encoder (two haplotypes → one token string) and the pretraining-corpus builder: per-sample BCF × region BED → Hugging Face Parquet dataset |
| [`pretrain/`](pretrain/README.md) | continued pretraining on NTv3 with vocabulary expansion (`train.py`) and its modules |
| `embedding/cohort/rows.py` | row ↔ HG38 coordinate helpers used by `pretrain/v2_input.py` |
| [`hf_cards/`](hf_cards/README.md) | the shipped `diploid_encode.py`, the model cards and the Hub publishing tools |
| [`benchmarks/gfmbench/`](benchmarks/gfmbench/README.md) | zygosity-aware ClinVar tasks for [GFMBench-API](https://github.com/NVIDIA/GFMBench-api) |
| [`benchmarks/datasets/`](benchmarks/datasets/README.md) | builders for the benchmark datasets (ClinVar diploid SNV / indel, 1000 Genomes MHC ancestry) |

## Pipeline

1. **Encode and build the corpus** — `preprocess/`: render each sample's phased genotypes over the
   pretraining regions against GRCh38, one diploid token per position, and assemble a sharded
   Parquet dataset with a pedigree-aware train/test split.
2. **Pretrain** — `pretrain/train.py`: expand the NTv3 vocabulary, warm up the new embedding rows
   with the backbone frozen, then continue pretraining with masked language modelling and the
   optional phase objectives (CPL, SMP).
3. **Evaluate** — `benchmarks/`: build the diploid ClinVar datasets and run them through
   GFMBench-API in diploid, haploid and biallelic modes.
4. **Use a released model** — `hf_cards/diploid_encode.py` turns two haplotypes into model input;
   every model card has a runnable quickstart.

## Install

Python 3.13 is what the code was developed and run on.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch` must match your CUDA. `flash-attn` / `kernels` are optional and only used when
`pretrain/ntv3_attention_patch.py` is asked to patch in FlashAttention.

Run everything from the repository root: modules import each other as `preprocess.…` and
`pretrain.…`, and default data paths are resolved from the repository root.

## Reference data you need to supply

Nothing large is committed. You need:

| Input | Used by |
|---|---|
| GRCh38 as `preprocess/hg38.2bit` (UCSC; checksum in `preprocess/hg38.2bit.md5`) | encoder, dataset builder, `train.py --v2-input` |
| 1000 Genomes high-coverage phased panel, per-chromosome BCFs | dataset builder |
| UCSC phyloP 100-way bigWig and CADD v1.7 (optional per-base annotations) | dataset builder, `--version v7`/`v8` with annotations |
| GENCODE v50 GTF, UCSC gap table, ENCODE blacklist v2, hg38 chrom sizes | only to rebuild the TSS–TES region BED |

Shell runners and a few CLI defaults point at the authors' cluster paths; every one of them can
be overridden by a flag or an environment variable (see each script's header).

## Licence

Code: Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Model weights are released separately on the Hub
under a research-only licence inherited from NTv3 ([`hf_cards/LICENSE`](hf_cards/LICENSE)).
Third-party data: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@article{leib2026dnt,
  title   = {DNT: Diploid Genomic Foundation Model},
  author  = {Leib, Guy and Zinger, Tal and Ofer, Dan and Kellerman, Raizy and Nayshool, Omri and others},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.05.749576}
}
```

The benchmark tasks build on GFMBench-API:

```bibtex
@article{larey2026gfmbench,
  title={GFMBench-API: A Standardized Interface for Benchmarking Genomic Foundation Models},
  author={Ariel Larey and Elay Dahan and Amit Bleiweiss and Raizy Kellerman and Guy Leib and
          Omri Nayshool and Dan Ofer and Tal Zinger and Dan Dominissini and Gideon Rechavi and
          Nicole Bussola and Simon Lee and Shane O'Connell and Dung Hoang and Marissa Wirth and
          Alexander W. Charney and Yoli Shavit and Nati Daniel},
  journal={bioRxiv},
  pages={2026--02},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```
