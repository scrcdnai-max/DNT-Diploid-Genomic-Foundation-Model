# pretrain — continued pretraining with vocabulary expansion

`train.py` continues pretraining an NTv3 checkpoint on the diploid corpus built by
`../preprocess/`. It runs in two stages:

1. **Embedding warm-up** (`--embedding-warmup-steps`): the embedding layer and LM head are widened
   to the diploid vocabulary, each new row is seeded from its constituent nucleotides, and only the
   new rows train while the backbone stays frozen.
2. **Main training**: masked language modelling over windows of the corpus, with optional phase
   objectives — CPL, Contrastive Phase Loss (`--cpl-lambda`), and SMP, Supervised Masked Phase
   (`--smp-lambda`) — and optional per-base phyloP / CADD auxiliary heads.

Checkpoints are written at milestones counted in genomes of data seen (`--save-at-genomes`).

```bash
python -m torch.distributed.run --nproc_per_node <N> pretrain/train.py \
  --model-name InstaDeepAI/NTv3_100M_post \
  --tokens preprocess/diploid_tokens.tsv \
  --train-data <dataset>/assembled/train --test-data <dataset>/assembled/test \
  --window-len 4096 --mlm-prob 0.15 \
  --cpl-lambda 0.5 --smp-lambda 0.2 \
  --embedding-warmup-steps 5000 --lr 1e-4 \
  --save-at-genomes 5 10 20 50 100 200
python pretrain/train.py --help        # every option
```

The settings for each released model (base model, objectives and weights, window, learning rate,
batch, steps) are listed in the *Provenance* section of its model card, in `../hf_cards/cards/`.

## Modules

| File | What |
|---|---|
| `train.py` | the trainer: data loading and windowing, the collator (masking, marker dropout), vocabulary expansion, losses, milestones |
| `tokenizer_lut.py` | codepoint → token-id table for the diploid alphabet, and its parity check against the HF tokenizer |
| `ntv3_attention_patch.py` | SDPA / FlashAttention patch for NTv3 attention with a padding mask |
| `ntv3_pool_surgery.py` | optional bypass of NTv3's down/up-sampling towers (`NTV3_POOL_BYPASS`) |
| `length_buckets.py` | length-bucketed batching under a token budget (`--token-budget`) |
| `windowing.py` | window stride resolution |
| `multiobj.py` | allele-frequency tables and the effect-ranked margin loss for the auxiliary objectives |
| `v2_input.py` | the one-token-per-position input (`--v2-input`), read from stored rows without re-encoding |
| `og2_mix.py`, `og2_replay.py` | optional OpenGenome2 replay mixing (`--og2-*`; data from `tools/download_og2_replay.py`, default `data/opengenome2/`) |
| `prepare_eval_data.py` | deduplicated, variant-window test set for fast MLM evaluation |
| `build_eval_small.py` | small persisted eval set with the natural variant / reference mix |

## Tools

| Tool | What |
|---|---|
| `tools/verify_dead_tokens.py` | pre-flight for `--drop-unseen-tokens`: checks the symbols it would drop really are absent from the corpus |
| `tools/prune_dead_vocab_rows.py` | slice a finished checkpoint down to the vocabulary rows its corpus uses |
| `tools/make_v2_token_counts.py` | token-count table for the `--v2-input` vocabulary |
| `tools/build_af_tables.py` | per-chromosome 1000 Genomes allele-frequency tables for the AF objective |
| `tools/epoch_budget.py` | steps per epoch and wall-clock from a run's bucketing log line |
| `tools/verify_freeze.py` | checks a frozen-parameter run left the frozen weights bitwise unchanged |
