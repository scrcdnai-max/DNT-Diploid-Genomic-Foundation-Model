---
license: other
license_name: research-only
license_link: LICENSE
library_name: transformers
pipeline_tag: fill-mask
tags:
- genomics
- dna
- diploid
- zygosity
- phase
- nucleotide-transformer
- ntv3
- continued-pretraining
base_model:
- InstaDeepAI/NTv3_8M_pre
---

# Directional_Tokens_plus_Unordered_8M

An **8M-parameter** [Nucleotide Transformer v3](https://huggingface.co/InstaDeepAI/NTv3_8M_pre) backbone continually pretrained
on reference-aligned **diploid token streams** from the phased 1000 Genomes Project. Both homologues are written into one
sequence, so zygosity, allele dosage and cis/trans phase reach the encoder directly instead of being reconstructed from
two separately encoded haplotypes. Objectives: **MLM**. Paper: [DNT: Diploid Genomic Foundation Model](https://doi.org/10.64898/2026.09.05.749576)
(bioRxiv 2026).

> Early two-stage MLM-only run with the marker-phase tokenizer used by every DNT-* checkpoint (corpus build `v5`, which differs from `v6` only in that a heterozygous SNV sitting on an indel anchor base carries no `^`). Twelve checkpoints in one repo: eleven stage-1 genome milestones and `stage2/final`, which is also the repo root.

## At a glance

| | |
|---|---|
| Base checkpoint | [`InstaDeepAI/NTv3_8M_pre`](https://huggingface.co/InstaDeepAI/NTv3_8M_pre) |
| Parameters | 7.70M (31 MB float32 weights) |
| Embedding dim / transformer layers / heads | 256 / 2 / 8 |
| Objectives | MLM |
| Training context | 4,096 (stage 1), 8,192 (stage 2) tokens |
| Input length | multiple of 128 tokens, at least 128 |
| Vocabulary | 55 tokens: 11 NTv3 + 44 diploid (`diploid_tokens.tsv`) |
| Corpus | 1000 Genomes high-coverage phased release, diploid build `v5` (see [Pretraining data](#pretraining-data)) |
| Schedule | two stages; stage-1 checkpoints at genome milestones 2 … 2000 (after N genomes' worth of windows), stage-2 `final`; peak LR 4e-4 (stage 1), 2e-4 (stage 2) |

## Model family

All repos share the token table and `diploid_encode.py`, so code written for one runs on the others. Two exceptions, each
spelled out in its own Quickstart: POST-backbone models load differently, and the ordered-phase repo needs `to_ordered()`.

| Repo | Scale | Base | Objectives | Peak LR | Context |
|---|---|---|---|---|---|
| [`DNT-8M-MLM`](https://huggingface.co/scrc-dnai/DNT-8M-MLM) | 8M | NTv3_8M_pre | MLM | 2e-4 | 4,096 |
| [`DNT-8M-CPL`](https://huggingface.co/scrc-dnai/DNT-8M-CPL) | 8M | NTv3_8M_pre | MLM + CPL | 5e-5 | 4,096 |
| [`DNT-8M-CPL-2e4`](https://huggingface.co/scrc-dnai/DNT-8M-CPL-2e4) | 8M | NTv3_8M_pre | MLM + CPL | 2e-4 | 4,096 |
| [`DNT-8M-CPL-SMP`](https://huggingface.co/scrc-dnai/DNT-8M-CPL-SMP) | 8M | NTv3_8M_pre | MLM + CPL + SMP | 2e-4 | 4,096 |
| [`DNT-100M_PRE-CPL`](https://huggingface.co/scrc-dnai/DNT-100M_PRE-CPL) | 100M | NTv3_100M_pre | MLM + CPL | 5e-5 | 4,096 |
| [`DNT-100M_PRE-CPL-2e4`](https://huggingface.co/scrc-dnai/DNT-100M_PRE-CPL-2e4) | 100M | NTv3_100M_pre | MLM + CPL | 2e-4 | 4,096 |
| [`DNT-100M_PRE-CPL-SMP`](https://huggingface.co/scrc-dnai/DNT-100M_PRE-CPL-SMP) | 100M | NTv3_100M_pre | MLM + CPL + SMP | 2e-4 | 4,096 |
| [`DNT-100M_POST-CPL-SMP`](https://huggingface.co/scrc-dnai/DNT-100M_POST-CPL-SMP) | 100M | NTv3_100M_post | MLM + CPL + SMP | 8e-5 | 4,096 |
| [`DNT-650M_POST-CPL-SMP`](https://huggingface.co/scrc-dnai/DNT-650M_POST-CPL-SMP) | 650M | NTv3_650M_post | MLM + CPL + SMP | 2e-4 | 8,192 |
| `Directional_Tokens_plus_Unordered_8M` (this repo) | 8M | NTv3_8M_pre | MLM | 4e-4 (stage 1), 2e-4 (stage 2) | 4,096 (stage 1), 8,192 (stage 2) |
| [`Phased_Nucleotide_Tokenization_8M`](https://huggingface.co/scrc-dnai/Phased_Nucleotide_Tokenization_8M) | 8M | NTv3_8M_pre | MLM (ordered-phase tokenizer) | 4e-4 (stage 1), 2e-4 (stage 2) | 4,096 (stage 1), 8,192 (stage 2) |

## Quickstart

```python
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os, sys, torch

REPO = "scrc-dnai/Directional_Tokens_plus_Unordered_8M"
tokens_tsv = hf_hub_download(REPO, "diploid_tokens.tsv")
sys.path.insert(0, os.path.dirname(hf_hub_download(REPO, "diploid_encode.py")))
import diploid_encode as de

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(REPO, trust_remote_code=True).eval().to(device)  # subfolder="checkpoints/stage1/genome-200" for a milestone

# Two reference-aligned haplotypes of one individual, 510 bp, heterozygous C/G at position 5.
hap0 = "ACGTACGTAC" * 51
hap1 = hap0[:5] + "G" + hap0[6:]
seq, offsets = de.encode_diploid(hap0, hap1, tokens_tsv=tokens_tsv, return_offsets=True)
print(seq[:12], len(seq))               # ACGTA^‹ĈGTAC 512   — 510 bp + 2 marker characters; 512 % 128 == 0

batch = tok(seq, return_tensors="pt").to(device)   # one character = one token; no CLS/EOS is added
with torch.no_grad():
    out = model(**batch, output_hidden_states=True)
per_token = out.hidden_states[-1][0]    # (512, 256): one vector per token, input resolution
i = offsets[5]                          # 7 — markers shift indices, so always index through `offsets`
print(seq[i], per_token[i].shape)       # Ĉ torch.Size([256])
```

## Usage notes

**Loading.** `AutoModelForMaskedLM` is the registered class; `AutoModel` raises `Unrecognized configuration class`. Weights are 31 MB in float32. Loading prints a `LOAD REPORT` flagging
`rotary_embedding.sin_cached` / `cos_cached` as unexpected; that is benign, the caches are rebuilt at runtime.

**Length.** The encoded string must be a multiple of 128 tokens long (seven 2x conv downsamplings) and at least 128.
Anything else raises `RuntimeError: The size of tensor a (N) must match the size of tensor b (N+1) ...`. Markers add
tokens (+2 per phased heterozygous SNV, +1 unphased, +2 per indel span, +1 per phased position inside a span), so encode first, then trim a flank until the length fits:

```python
def encode_to_multiple(hap0, hap1, tokens_tsv, block=128, **kw):
    """Trim the right flank until the encoded length is a multiple of `block`. Returns (seq, offsets, hi)."""
    hi = len(hap0)
    while hi:
        seq, offsets = de.encode_diploid(hap0[:hi], hap1[:hi], tokens_tsv=tokens_tsv, return_offsets=True,
                                         **{k: (v[:hi] if k.endswith("_mask") and v is not None else v) for k, v in kw.items()})
        if seq and len(seq) % block == 0:
            return seq, offsets, hi
        hi -= 1
    raise ValueError("no prefix encodes to a multiple of %d" % block)
```

`kw` passes through `snp_mask` / `indel_mask` (see [Encoding genotypes](#encoding-genotypes)). Right-flank trimming silently
drops a variant near the right edge; to trim the left flank instead, slice `hap0`, `hap1` and the masks from the left.

**Do not pad.** This architecture ignores `attention_mask` (output is bit-identical with or without it), so padding is
not masked out: padding a 512-token window to 1,024 changes every position's embedding. For the same reason, **batch only
windows of identical length** — `tok([...], padding=True)` does not raise, it returns quietly wrong rows.

**Unphased input.** Pass `phased=False` for the whole window when genotypes are unphased (`/` in the VCF). Do not mix marked and unmarked heterozygous positions in one window: every phased heterozygous site carried a direction marker in training, so a mixed window is unlike anything the model saw.

**float32 only.** `config.json` pins the compute dtypes; a bfloat16 or float16 model loads but the forward pass raises.

**Context.** Inputs longer than the training context (4,096 tokens for the stage-1 milestones, 8,192 for `stage2/final`) run but were not evaluated.

**Pooling.** `hidden_states[-1]` is one vector per token at input resolution; index it through `offsets`. The transformer
itself runs at one position per 128 tokens; the last `hidden_states` entry of that length is its output. For a window
vector, mean-pool `hidden_states[-1]` over genomic positions only: marker tokens are not positions, and their share of a
window grows with variant density.

```python
keep = [o for o in offsets if o >= 0]                                             # genomic positions only
pooled = per_token[keep].mean(0)                                                  # excludes ^ ‹ › { } marker tokens
coarse = [h for h in out.hidden_states if h.shape[1] == len(seq) // 128][-1][0]   # (L/128, 256) transformer output
```

## Encoding genotypes

```python
encode_diploid(hap0, hap1, *, tokens_tsv, snp_mask=None, indel_mask=None,
               phased=True, skip_gaps=True, return_offsets=False) -> str | (str, list[int])
```

`hap0` and `hap1` are equal-length, **reference-aligned, uppercase** strings over `A C G T N - *`; the encoder does not align.
A deleted base is `-` on the haplotype that lost it; an inserted base is `-` on the haplotype that lacks it. A plain
haploid sequence is valid input as `encode_diploid(seq, seq, ...)`: homozygous positions encode to the ordinary base letters,
so reference DNA is just the NTv3 alphabet.

**From a phased VCF.** For SNV-only windows, `bcftools consensus -f ref.fa -r chr:start-end -s SAMPLE -i 'TYPE="snp"' -H 1`
and `-H 2` give `hap0` and `hap1` directly (equal length, reference-aligned; uppercase them). With indels, build the strings
from the VCF records: a deletion becomes `-` on the haplotype that carries it, an insertion adds columns that are `-` on the
other haplotype, and `indel_mask` flags those columns.

| Argument | Effect |
|---|---|
| `snp_mask` | Which positions get the `^` (`<SNP>`) marker. Default: every heterozygous position. A supplied mask **replaces** that default, so include the heterozygous sites too. The only way to mark a homozygous-ALT substitution, which is otherwise indistinguishable from reference. |
| `indel_mask` | Which positions are indel content (deleted reference bases, inserted bases). Each run is wrapped in `{`…`}`; the anchor base stays outside. Without it, indel positions come out as bare `^`-marked heterozygous states. |
| `phased` | Emit direction markers; `False` for unphased genotypes. |
| `skip_gaps` | Drop gap/gap positions, as in the pretraining corpus. `False` keeps them as `-` tokens, the only way to make a homozygous deletion visible. |
| `return_offsets` | Also return, for every input position, the output index of its allele-pair symbol (`-1` if dropped). |

Each position becomes one allele-pair symbol, preceded at variant sites by a variant marker (`^`, or `{`…`}` around indel
content) and, at phased heterozygous sites, by a direction marker: `‹` when hap0 holds the lower-priority allele in the
order `A < C < G < T < N < - < *`, `›` otherwise. The pair symbol itself is unordered; homozygous pairs collapse to the plain base.

| Genotype | Call | Output |
|---|---|---|
| hom REF `A`/`A` | `encode_diploid("A", "A", ...)` | `A` |
| het `A`/`G`, hap0 = A | `encode_diploid("A", "G", ...)` | `^‹Ⱥ` |
| het `G`/`A`, hap0 = G | `encode_diploid("G", "A", ...)` | `^›Ⱥ` |
| het `A`/`G`, unphased | `encode_diploid("A", "G", ..., phased=False)` | `^Ⱥ` |
| hom ALT `A`/`A` (ref `G`) | `encode_diploid("A", "A", ..., snp_mask=[True])` | `^A` |
| het 2 bp deletion, hap1 loses `GC` | `encode_diploid("AGCT", "A--T", ..., indel_mask=[0,1,1,0])` | `A{‹Ģ‹Č}T` |
| het 3 bp insertion `TTA` on hap0 | `encode_diploid("GTTAC", "G---C", ..., indel_mask=[0,1,1,1,0])` | `G{‹Ţ‹Ţ‹Å}C` |

`de.symbol_for("C", "T", tokens_tsv)` returns the symbol for a pair (`Ȼ`), for use with the MLM head below.

## Scoring genotypes with the MLM head

```python
ids = batch["input_ids"].clone()
ids[0, i] = tok.mask_token_id                      # mask the allele-pair symbol at output index i
with torch.no_grad():
    logprobs = model(input_ids=ids).logits[0, i].log_softmax(-1)
score = logprobs[tok.convert_tokens_to_ids(de.symbol_for("C", "G", tokens_tsv))]   # log P(het C/G here)
```

Masking hides the whole allele-pair symbol, so the head scores zygosity (and, where a direction marker precedes the site, orientation). The `^` and direction markers stay visible, so one masked position compares genotypes that share them (heterozygous against homozygous-ALT, or the two orientations). For a reference-versus-alternative score, encode the window in both genotypes — their lengths differ by the
marker tokens, so trim each to a multiple of 128 — mask the site in each and compare the log-probabilities of the true
symbol. Use real genomic context; on the synthetic repeat above the ranking means nothing.

## Objectives

**MLM.** 15% of tokens selected; 80% replaced by `<mask>`, 10% by a random token, 10% left unchanged. Masking is
per allele-pair symbol, so recovering a heterozygous position means predicting the joint allelic state.

## Pretraining data

Phased genotypes from the 1000 Genomes high-coverage release rendered against GRCh38 (corpus build `v5`, the marker-phase
encoding); stage 1 windowed at 4,096 tokens, stage 2 at 8,192.

## Limitations

- **Only the marker-phase rows of the token table were trained.** The 20 ordered-phase symbols and the 7 hemizygous
  (lowercase) symbols in `diploid_tokens.tsv` have zero count in the corpus; their embeddings are at initialisation.
  Never hand raw strings to the tokenizer: lowercase `acgt` are the untrained hemizygous tokens, and anything else becomes
  `<unk>` silently. Go through `encode_diploid`, which rejects everything outside `A C G T N - *`.
- **No heterozygous `N`.** `A`/`N` raises `KeyError`; resolve or drop ambiguous positions first.
- **`-` and `*` share symbols** at heterozygous positions (`A`/`-` and `A`/`*` are both `Å`); `-`/`*` and `.` have none.
- **Homozygous deletions leave no trace** under `skip_gaps=True`: the position is dropped with no marker. About 3% of
  variant calls, ~0.014% of bases. Heterozygous deletions are represented.
- **Hemizygous sites are encoded as homozygous** (male X and Y).
- **Phase is inherited, not inferred.** Markers carry the errors of whatever phasing produced the input.
- **Structural variation is out of scope.** CNVs, repeat expansions and inversions are not represented; windows are
  local, not gene-scale.

## Results

No published evaluation; superseded by the DNT-* checkpoints in the family table.

## Files

| File | Purpose |
|---|---|
| `model.safetensors`, `config.json` | weights and architecture |
| `checkpoints/stage1/genome-N/`, `checkpoints/stage2/final/` | eleven stage-1 milestones and stage-2 `final`, each a complete model directory; load with `subfolder=` |
| `modeling_ntv3_pretrained.py`, `configuration_ntv3_pretrained.py` | NTv3 implementation; needs `trust_remote_code=True` |
| `vocab.json`, `added_tokens.json`, `tokenization_ntv3.py`, `tokenizer_config.json` | 55-token character tokenizer (11 NTv3 + 44 diploid) |
| `diploid_tokens.tsv` | token table; sha256 prefix `9664edd161cc156d` |
| `diploid_encode.py` | haplotype pair -> token stream, byte-identical to the training pipeline |

## Provenance

Stage 1: 4,096-token windows, LR 4e-4, batch 256, checkpoints saved at genome milestones 2, 10, 20, 50, 100, 200, 400, 800, 1000, 1600 and 2000. Stage 2: continued at 8,192-token windows, LR 2e-4, batch 256, saved as `final`.

Embedding layer and LM head resized from 11 to 55 rows; each new heterozygous row seeded from the mean of its constituent
nucleotide embeddings.

## Licence

Derived from InstaDeep's Nucleotide Transformer v3, which is released for non-commercial use only under the
[InstaDeep Open Model Licence v1.0](https://huggingface.co/InstaDeepAI/NTv3_8M_pre/blob/main/MODEL-LICENSE.md).
These weights inherit those terms: **research use only, no commercial use, no serving outputs as a service.** See `LICENSE`.
Not for clinical use.

## Citation

```
Leib G, Zinger T, Ofer D, Kellerman R, Nayshool O, Dominissini D, Larey A, Levy J, Nahshan Y, Dahan E, Bleiweiss A,
Bussola N, Lee S, O'Connell S, Hoang D, Wirth M, Beckmann ND, Charney AW, Shavit Y, Daniel N, Rechavi G.
DNT: Diploid Genomic Foundation Model. bioRxiv 2026.09.05.749576 (2026). https://doi.org/10.64898/2026.09.05.749576
```

Sheba Medical Center · NVIDIA · Icahn School of Medicine at Mount Sinai.
