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
- nucleotide-transformer
- ntv3
base_model:
- InstaDeepAI/NTv3_650M_post
---

# DNT-650M_POST-CPL-SMP

A 650M-parameter diploid model based on [NTv3 650M POST](https://huggingface.co/InstaDeepAI/NTv3_650M_post). It was continually pretrained on 1,000 phased 1000 Genomes samples using 8,192-token windows for 212,000 steps, with masked language modeling (MLM), Contrastive Phase Loss (CPL) and Supervised Masked Phase (SMP).

## Quickstart

```python
import os
import sys
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

repo = "scrc-dnai/DNT-650M_POST-CPL-SMP"
tokens_tsv = hf_hub_download(repo, "diploid_tokens.tsv")
encoder_dir = os.path.dirname(hf_hub_download(repo, "diploid_encode.py"))
sys.path.insert(0, encoder_dir)
import diploid_encode as de

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval().to(device)

# Two reference-aligned haplotypes; one phased C/G variant.
hap0 = "ACGTACGTAC" * 51
hap1 = hap0[:5] + "G" + hap0[6:]
sequence, offsets = de.encode_diploid(
    hap0, hap1, tokens_tsv=tokens_tsv, return_offsets=True
)
batch = tokenizer(sequence, return_tensors="pt").to(device)
species_ids = model.encode_species("human").to(device)
with torch.no_grad():
    output = model(**batch, species_ids=species_ids, output_hidden_states=True)

# Mean embedding over genomic positions; excludes inserted marker tokens.
token_embeddings = output.hidden_states[-1][0]
position_indices = [index for index in offsets if index >= 0]
sequence_embedding = token_embeddings[position_indices].mean(dim=0)
print(sequence_embedding.shape)  # (1536,)
```

## Mean-pooling results

| Benchmark | Readout | Result |
|---|---|---|
| ClinVar indels | Zero-shot L2 distance between mean embeddings; diploid / haploid / biallelic | AUROC **0.662 / 0.545 / 0.217** |
| GUE promoter / splice / TF | Mean embeddings with train/dev/test ridge probes | MCC **0.829 / 0.519 / 0.606** |

## Input and licensing

Encode reference-aligned haplotype pairs with the included `diploid_encode.py`; the model expects token lengths divisible by 128. For batched POST-model inference, pass one `species_ids` value per sequence (for example, `model.encode_species(["human"] * batch_size)`).

These weights inherit the non-commercial terms of the [InstaDeep Open Model Licence v1.0](https://huggingface.co/InstaDeepAI/NTv3_650M_post/blob/main/MODEL-LICENSE.md). Research use only; no commercial use or serving outputs as a service. See [`LICENSE`](LICENSE).

## Citation

Leib G et al. DNT: Diploid Genomic Foundation Model. bioRxiv (2026). [doi:10.64898/2026.09.05.749576](https://doi.org/10.64898/2026.09.05.749576).
