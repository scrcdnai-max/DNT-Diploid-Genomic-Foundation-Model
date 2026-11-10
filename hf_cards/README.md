# Hugging Face model cards for the DNT models

Tooling used to write the model cards for the public DNT model repos on the Hugging Face Hub
([`scrc-dnai`](https://huggingface.co/scrc-dnai)) and to publish them.

| File | What |
|---|---|
| `diploid_encode.py` | the encoder shipped with every model: two haplotypes → one diploid sequence (+ `to_ordered()` for the ordered-symbol model) |
| `cards.py` | one template plus per-model facts (`MODELS`) → `cards/<repo>.md`; refuses to render internal terms or paths |
| `cards/` | the rendered cards, as published |
| `stage.py` | builds `staging/<repo>/`: the full model directory for a new repo; card, LICENSE, encoder and token table for an existing one. `NEW` maps each repo to its source checkpoint under `$DNT_MODEL_ROOT` (default `models/`) |
| `publish.py {new,existing,aux,collection,verify}` | Hub side: upload the staged repos and check what is live |
| `test_cards.py <repo> <model_dir>` | runs every Python block of a card against local weights and checks the printed values |
| `smoke_model.py <model_dir> [pre\|post]` | prints the facts a card must state (hidden-states layout, attention mask, bf16, species ids) |
| `LICENSE` | the research-only licence that ships **with the model weights** (they derive from InstaDeep's NTv3 and inherit its non-commercial terms). The code in this repository is Apache-2.0 (see the top-level `LICENSE`) |

`stage.py` and `smoke_model.py` read the token table from `../preprocess/diploid_tokens.tsv`.
