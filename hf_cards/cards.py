"""Generate the README.md (model card) for every public scrc-dnai DNT repo.

    python cards.py            # writes cards/<repo>.md
Shared prose lives once, per-model facts live in MODELS.
"""
from pathlib import Path

ORG = "scrc-dnai"
PAPER = "https://doi.org/10.64898/2026.09.05.749576"
TSV_SHA = "9664edd161cc156d"


def md(p):  # marker dropout, defined once, same words everywhere
    return f"marker dropout {p} (the `^` marker deleted from {int(round(p * 100))}% of variant sites in the training input)"


def rare_fix(sibling):
    return (f"Two later training fixes are in this run and `{sibling}` only: MLM random-replacement tokens are drawn from "
            "the corpus unigram distribution, and new-token LM-head biases start from a log-frequency prior.")
V6_FULL = ("peak LR {lr}, 10,000 warm-up steps, linear decay, weight decay 0.005, AdamW, bf16 autocast, "
           "4,096-token windows; all 922,206 steps of one epoch; " + md(0.15) + ".")

# ---------------------------------------------------------------- per-model facts
MODELS = [
    dict(repo="DNT-8M-MLM", scale="8M", base="InstaDeepAI/NTv3_8M_pre", kind="pre",
         params="7.70M", weights="31 MB", dim=256, layers=2, heads=8, obj=["MLM"], ctx=4096, corpus="v6", lr="2e-4",
         schedule="360,580 steps at effective batch 240 (39% of one epoch; milestone `genome-1000`, i.e. after 1,000 genomes' worth of windows)",
         provenance="6 x B300 (per-device batch 40, effective 240), peak LR 2e-4, 10,000 warm-up steps, linear decay, weight decay "
                    "0.005, AdamW, bf16 autocast, 4,096-token windows; stopped at the `genome-1000` milestone, 360,580 optimizer steps; "
                    + md(0.15) + ". " + rare_fix("DNT-100M_POST-CPL-SMP"),
         results=None,
         note=f"MLM-only counterpart of [`DNT-8M-CPL-2e4`](https://huggingface.co/{ORG}/DNT-8M-CPL-2e4): same base, corpus, tokenizer, "
              "LR schedule and effective batch, no phase objective; stopped at 39% of one epoch."),
    dict(repo="DNT-8M-CPL", scale="8M", base="InstaDeepAI/NTv3_8M_pre", kind="pre",
         params="7.70M", weights="31 MB", dim=256, layers=2, heads=8, obj=["MLM", "CPL"], ctx=4096, corpus="v6", lr="5e-5",
         schedule="one epoch = 922,206 steps at effective batch 240",
         provenance="3 x H100 (per-device batch 80, effective 240), " + V6_FULL.format(lr="5e-5"),
         results=["Compound-heterozygous cis/trans, 1,811-example test split: frozen linear probe **0.551 ± 0.009** AUROC, "
                  "full fine-tune **0.627 ± 0.010** (preprint Supplementary Table 3, learning-rate 5e-5)."],
         note="Peak learning rate 5e-5. The preprint's primary configuration is this run at 2e-4: "
              f"[`DNT-8M-CPL-2e4`](https://huggingface.co/{ORG}/DNT-8M-CPL-2e4)."),
    dict(repo="DNT-8M-CPL-2e4", scale="8M", base="InstaDeepAI/NTv3_8M_pre", kind="pre",
         params="7.70M", weights="31 MB", dim=256, layers=2, heads=8, obj=["MLM", "CPL"], ctx=4096, corpus="v6", lr="2e-4",
         schedule="one epoch = 922,206 steps at effective batch 240",
         provenance="Effective batch 240 throughout (steps 0-200,000 on 2 GPUs at per-device batch 120, then 4 GPUs at per-device "
                    "batch 60), " + V6_FULL.format(lr="2e-4"),
         results=["Compound-heterozygous cis/trans, 1,811-example test split: frozen linear probe **0.613 ± 0.009** AUROC, "
                  "full fine-tune **0.649 ± 0.005** (preprint Table 1).",
                  "ClinVar mode-of-inheritance-aware SNV genotypes: **0.852** AUROC (Table 2). "
                  "ClinVar indel genotypes, zero-shot embedding distance: **0.658** AUROC (Table 3)."],
         note="The preprint's **primary configuration** at 8M."),
    dict(repo="DNT-8M-CPL-SMP", scale="8M", base="InstaDeepAI/NTv3_8M_pre", kind="pre",
         params="7.70M", weights="31 MB", dim=256, layers=2, heads=8, obj=["MLM", "CPL", "SMP"], ctx=4096, corpus="v6", lr="2e-4",
         schedule="one epoch = 922,206 steps at effective batch 240",
         provenance="8 x H100 (per-device batch 30, effective 240), " + V6_FULL.format(lr="2e-4"),
         results=None, note=None),
    dict(repo="DNT-100M_PRE-CPL", scale="100M", base="InstaDeepAI/NTv3_100M_pre", kind="pre",
         params="106.50M", weights="430 MB", dim=768, layers=6, heads=12, obj=["MLM", "CPL"], ctx=4096, corpus="v6", lr="5e-5",
         schedule="one epoch = 922,206 steps at effective batch 240",
         provenance="5 x H100 (per-device batch 48, effective 240), " + V6_FULL.format(lr="5e-5"),
         results=["Compound-heterozygous cis/trans, 1,811-example test split: frozen linear probe **0.610 ± 0.011** AUROC, "
                  "full fine-tune **0.625 ± 0.006** (preprint Supplementary Table 3, learning-rate 5e-5).",
                  "This is the checkpoint behind the CFTR case study: zero-shot reference/alternative log-likelihood ratio "
                  "separates pathogenic from benign ClinVar CFTR SNVs at **0.780** AUROC (Fig. 4)."],
         note="Peak learning rate 5e-5. The preprint's primary configuration is this run at 2e-4: "
              f"[`DNT-100M_PRE-CPL-2e4`](https://huggingface.co/{ORG}/DNT-100M_PRE-CPL-2e4)."),
    dict(repo="DNT-100M_PRE-CPL-2e4", scale="100M", base="InstaDeepAI/NTv3_100M_pre", kind="pre",
         params="106.50M", weights="430 MB", dim=768, layers=6, heads=12, obj=["MLM", "CPL"], ctx=4096, corpus="v6", lr="2e-4",
         schedule="one epoch = 922,206 steps at effective batch 240",
         provenance="Effective batch 240 throughout (steps 0-570,000 on 8 x H100 at per-device batch 30, then 3 x B300 at "
                    "per-device batch 80), " + V6_FULL.format(lr="2e-4"),
         results=["Compound-heterozygous cis/trans, 1,811-example test split: frozen linear probe **0.624 ± 0.005** AUROC, "
                  "full fine-tune **0.639 ± 0.005** (preprint Table 1).",
                  "ClinVar mode-of-inheritance-aware SNV genotypes: **0.755** AUROC (Table 2). "
                  "ClinVar indel genotypes, zero-shot embedding distance: **0.717** AUROC (Table 3)."],
         note="The preprint's **primary configuration** at 100M."),
    dict(repo="DNT-100M_PRE-CPL-SMP", scale="100M", base="InstaDeepAI/NTv3_100M_pre", kind="pre",
         params="106.50M", weights="430 MB", dim=768, layers=6, heads=12, obj=["MLM", "CPL", "SMP"], ctx=4096, corpus="v6", lr="2e-4",
         schedule="one epoch = 922,206 steps at effective batch 240",
         provenance="8 x H100 (per-device batch 30, effective 240), " + V6_FULL.format(lr="2e-4"),
         results=None, note=None),
    dict(repo="DNT-100M_POST-CPL-SMP", scale="100M", base="InstaDeepAI/NTv3_100M_post", kind="post",
         params="120.09M", weights="480 MB", dim=768, layers=6, heads=12, obj=["MLM", "CPL", "SMP"], ctx=4096, corpus="v7", lr="8e-5",
         schedule="27,044 steps at effective batch 640 (~8% of one epoch; milestone `genome-200`, i.e. after 200 genomes' worth of windows)",
         provenance="4 x B300 (per-device batch 160, effective 640), LR 8e-5 constant after 500 warm-up steps, weight decay 0.005, "
                    "AdamW, bf16 autocast, 4,096-token windows; stopped at the `genome-200` milestone, 27,044 optimizer steps. A separate "
                    "5,000-step warm-up (new embedding rows only, backbone frozen, 1,024-token windows, LR 1.6e-4) preceded the main "
                    "run and is not counted. CPL every 2nd step; "
                    "SMP mask fraction 0.5; " + md(0.10) + ". " + rare_fix("DNT-8M-MLM"),
         results=None,
         note="Built on the **post-trained** NTv3 backbone (species-conditioned, annotation-track fine-tuned by InstaDeep). "
              "Loading and the forward call differ from the PRE models; see Quickstart."),
    dict(repo="DNT-650M_POST-CPL-SMP", scale="650M", base="InstaDeepAI/NTv3_650M_post", kind="post",
         params="679.86M", weights="2.7 GB", dim=1536, layers=12, heads=24, obj=["MLM", "CPL", "SMP"], ctx=8192, corpus="v7",
         data_key="v7 (1,000-genome subset)", lr="2e-4",
         schedule="212,000 steps at effective batch 256 (~1 epoch of a 1,000-genome subset)",
         provenance="4 x B300 (per-device batch 32, gradient accumulation 2, effective 256), peak LR 2e-4, 1,000 warm-up steps, "
                    "linear decay, weight decay 0.005, AdamW, bf16 autocast, 8,192-token windows, 212,000 optimizer steps; base "
                    "revision `ad622051`. A separate 5,000-step warm-up (new embedding rows only, backbone frozen, 1,024-token windows, "
                    "LR 4e-4) preceded the main run and is not counted. CPL every 3rd step; SMP mask fraction 0.4; " + md(0.40) + ".",
         results=None,
         note="Largest model in the family. Built on the **post-trained** NTv3 backbone (species-conditioned, annotation-track "
              "fine-tuned by InstaDeep). Loading and the forward call differ from the PRE models; see Quickstart."),
    dict(repo="Directional_Tokens_plus_Unordered_8M", scale="8M", base="InstaDeepAI/NTv3_8M_pre", kind="pre", family_ckpts=True,
         params="7.70M", weights="31 MB", dim=256, layers=2, heads=8, obj=["MLM"], ctx="4,096 (stage 1), 8,192 (stage 2)", corpus="v5",
         lr="4e-4 (stage 1), 2e-4 (stage 2)",
         schedule="two stages; stage-1 checkpoints at genome milestones 2 … 2000 (after N genomes' worth of windows), stage-2 `final`",
         provenance="Stage 1: 4,096-token windows, LR 4e-4, batch 256, checkpoints saved at genome milestones 2, 10, 20, 50, 100, "
                    "200, 400, 800, 1000, 1600 and 2000. Stage 2: continued at 8,192-token windows, LR 2e-4, batch 256, saved as `final`.",
         results="early",
         note="Early two-stage MLM-only run with the marker-phase tokenizer used by every DNT-* checkpoint (corpus build `v5`, "
              "which differs from `v6` only in that a heterozygous SNV sitting on an indel anchor base carries no `^`). Twelve "
              "checkpoints in one repo: eleven stage-1 genome milestones and `stage2/final`, which is also the repo root."),
    dict(repo="Phased_Nucleotide_Tokenization_8M", scale="8M", base="InstaDeepAI/NTv3_8M_pre", kind="pre", family_ckpts=True, ordered=True,
         params="7.70M", weights="31 MB", dim=256, layers=2, heads=8, obj=["MLM"], ctx="4,096 (stage 1), 8,192 (stage 2)", corpus="v4",
         data_key="v4 (ordered symbols)", lr="4e-4 (stage 1), 2e-4 (stage 2)",
         schedule="two stages; stage-1 checkpoints at genome milestones 2 … 2000 (after N genomes' worth of windows), stage-2 `final`",
         provenance="Stage 1: 4,096-token windows, LR 4e-4, batch 256, checkpoints saved at genome milestones 2, 10, 20, 50, 100, "
                    "200, 400, 800, 1000, 1600 and 2000. Stage 2: continued at 8,192-token windows, LR 2e-4, batch 256, saved as `final`.",
         results="early",
         note="Early two-stage MLM-only run with the **ordered-phase tokenizer**: one symbol per *ordered* allele pair (`^Ă` = A on "
              "hap0, T on hap1; `^Ť` the reverse) instead of a direction marker plus an unordered pair symbol. Convert encoder output "
              "with `to_ordered()` before tokenizing (see Quickstart). Twelve checkpoints in one repo: eleven stage-1 genome "
              "milestones and `stage2/final`, which is also the repo root."),
]

# ---------------------------------------------------------------- shared prose
FAMILY_HEADER = "| Repo | Scale | Base | Objectives | Peak LR | Context |\n|---|---|---|---|---|---|\n"


def ctx_str(m):
    return m["ctx"] if isinstance(m["ctx"], str) else f"{m['ctx']:,}"


def family_table(current):
    rows = []
    for m in MODELS:
        name = f"`{m['repo']}` (this repo)" if m["repo"] == current else f"[`{m['repo']}`](https://huggingface.co/{ORG}/{m['repo']})"
        obj = " + ".join(m["obj"]) + (" (ordered-phase tokenizer)" if m.get("ordered") else "")
        rows.append(f"| {name} | {m['scale']} | {m['base'].split('/')[1]} | {obj} | {m['lr']} | {ctx_str(m)} |")
    return FAMILY_HEADER + "\n".join(rows)


def quickstart(m):
    repo = f"{ORG}/{m['repo']}"
    post = m["kind"] == "post"
    load = "AutoModel" if post else "AutoModelForMaskedLM"
    species = ('species_ids = model.encode_species("human").to(device)   # POST models: required, one entry per batch row\n'
               if post else "")
    call = "model(**batch, species_ids=species_ids, output_hidden_states=True)" if post else "model(**batch, output_hidden_states=True)"
    d = m["dim"]
    sub = '  # subfolder="checkpoints/stage1/genome-200" for a milestone' if m.get("family_ckpts") else ""
    if m.get("ordered"):
        hap = ('# Two reference-aligned haplotypes of one individual, 511 bp, heterozygous C/G at position 5.\n'
               'hap0 = ("ACGTACGTAC" * 52)[:511]\n'
               'hap1 = hap0[:5] + "G" + hap0[6:]\n'
               'seq, offsets = de.encode_diploid(hap0, hap1, tokens_tsv=tokens_tsv, return_offsets=True)\n'
               'seq, offsets = de.to_ordered(seq, tokens_tsv, offsets)   # this repo uses ordered-phase symbols: ^‹Ĉ -> ^Ƈ\n'
               'print(seq[:12], len(seq))               # ACGTA^ƇGTACA 512   — 511 bp + 1 marker character; 512 % 128 == 0')
        i_comment, tail = "6", "Ƈ"
    else:
        hap = ('# Two reference-aligned haplotypes of one individual, 510 bp, heterozygous C/G at position 5.\n'
               'hap0 = "ACGTACGTAC" * 51\n'
               'hap1 = hap0[:5] + "G" + hap0[6:]\n'
               'seq, offsets = de.encode_diploid(hap0, hap1, tokens_tsv=tokens_tsv, return_offsets=True)\n'
               'print(seq[:12], len(seq))               # ACGTA^‹ĈGTAC 512   — 510 bp + 2 marker characters; 512 % 128 == 0')
        i_comment, tail = "7", "Ĉ"
    return f"""```python
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, {load}
import os, sys, torch

REPO = "{repo}"
tokens_tsv = hf_hub_download(REPO, "diploid_tokens.tsv")
sys.path.insert(0, os.path.dirname(hf_hub_download(REPO, "diploid_encode.py")))
import diploid_encode as de

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = {load}.from_pretrained(REPO, trust_remote_code=True).eval().to(device){sub}
{species}
{hap}

batch = tok(seq, return_tensors="pt").to(device)   # one character = one token; no CLS/EOS is added
with torch.no_grad():
    out = {call}
per_token = out.hidden_states[-1][0]    # (512, {d}): one vector per token, input resolution
i = offsets[5]                          # {i_comment} — markers shift indices, so always index through `offsets`
print(seq[i], per_token[i].shape)       # {tail} torch.Size([{d}])
```"""


def rules(m):
    post = m["kind"] == "post"
    if post:
        loading = ("`AutoModel` is the registered class; it returns `logits` and `hidden_states`. `species_ids` is required, one "
                   "entry per batch row, on the model's device: `model.encode_species([\"human\"] * B).to(device)` (human is token 27).")
    else:
        loading = "`AutoModelForMaskedLM` is the registered class; `AutoModel` raises `Unrecognized configuration class`."
    if m.get("ordered"):
        unphased = ("**Unphased input is not supported** by this checkpoint: `phased=False` produces the unordered pair symbols, "
                    "which it never trained on.")
    else:
        unphased = ("**Unphased input.** Pass `phased=False` for the whole window when genotypes are unphased (`/` in the VCF). Do "
                    "not mix marked and unmarked heterozygous positions in one window: every phased heterozygous site carried a "
                    "direction marker in training, so a mixed window is unlike anything the model saw.")
    markers = ("+1 per heterozygous SNV, +2 per indel span; after `to_ordered()` there are no direction markers" if m.get("ordered")
               else "+2 per phased heterozygous SNV, +1 unphased, +2 per indel span, +1 per phased position inside a span")
    context = ("Inputs longer than the training context (4,096 tokens for the stage-1 milestones, 8,192 for `stage2/final`) run "
               "but were not evaluated." if m.get("family_ckpts")
               else f"Inputs longer than the {ctx_str(m)}-token training context run but were not evaluated.")
    return f"""**Loading.** {loading} Weights are {m['weights']} in float32. Loading prints a `LOAD REPORT` flagging
`rotary_embedding.sin_cached` / `cos_cached` as unexpected; that is benign, the caches are rebuilt at runtime.

**Length.** The encoded string must be a multiple of 128 tokens long (seven 2x conv downsamplings) and at least 128.
Anything else raises `RuntimeError: The size of tensor a (N) must match the size of tensor b (N+1) ...`. Markers add
tokens ({markers}), so encode first, then trim a flank until the length fits:

```python
def encode_to_multiple(hap0, hap1, tokens_tsv, block=128, **kw):
    \"\"\"Trim the right flank until the encoded length is a multiple of `block`. Returns (seq, offsets, hi).\"\"\"
    hi = len(hap0)
    while hi:
        seq, offsets = de.encode_diploid(hap0[:hi], hap1[:hi], tokens_tsv=tokens_tsv, return_offsets=True,
                                         **{{k: (v[:hi] if k.endswith("_mask") and v is not None else v) for k, v in kw.items()}})
{"        seq, offsets = de.to_ordered(seq, tokens_tsv, offsets)" + chr(10) if m.get("ordered") else ""}        if seq and len(seq) % block == 0:
            return seq, offsets, hi
        hi -= 1
    raise ValueError("no prefix encodes to a multiple of %d" % block)
```

`kw` passes through `snp_mask` / `indel_mask` (see [Encoding genotypes](#encoding-genotypes)). Right-flank trimming silently
drops a variant near the right edge; to trim the left flank instead, slice `hap0`, `hap1` and the masks from the left.

**Do not pad.** This architecture ignores `attention_mask` (output is bit-identical with or without it), so padding is
not masked out: padding a 512-token window to 1,024 changes every position's embedding. For the same reason, **batch only
windows of identical length** — `tok([...], padding=True)` does not raise, it returns quietly wrong rows.

{unphased}

**float32 only.** `config.json` pins the compute dtypes; a bfloat16 or float16 model loads but the forward pass raises.

**Context.** {context}

**Pooling.** `hidden_states[-1]` is one vector per token at input resolution; index it through `offsets`. The transformer
itself runs at one position per 128 tokens; the last `hidden_states` entry of that length is its output. For a window
vector, mean-pool `hidden_states[-1]` over genomic positions only: marker tokens are not positions, and their share of a
window grows with variant density.

```python
keep = [o for o in offsets if o >= 0]                                             # genomic positions only
pooled = per_token[keep].mean(0)                                                  # excludes ^ ‹ › {{ }} marker tokens
coarse = [h for h in out.hidden_states if h.shape[1] == len(seq) // 128][-1][0]   # (L/128, {m['dim']}) transformer output
```"""


ENCODING = """```python
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
| het 3 bp insertion `TTA` on hap0 | `encode_diploid("GTTAC", "G---C", ..., indel_mask=[0,1,1,1,0])` | `G{‹Ţ‹Ţ‹Å}C` |"""

SYMBOL_FOR = "\n\n`de.symbol_for(\"C\", \"T\", tokens_tsv)` returns the symbol for a pair (`Ȼ`), for use with the MLM head below."

ORDERED_ENCODING = """

**This repo's tokenizer is the ordered-phase scheme.** Run the encoder as above, then convert:
`seq, offsets = de.to_ordered(seq, tokens_tsv, offsets)`. Each `‹X` / `›X` pair collapses to one ordered symbol (`^‹Ⱦ` -> `^Ă`,
`^›Ⱦ` -> `^Ť`; `{‹Ģ‹Č}` -> `{ƓĊ}`), so the string is one token shorter per phased heterozygous position — apply the
multiple-of-128 rule to the converted string. The `phased` rows of `diploid_tokens.tsv` list the 20 ordered symbols; use
those, not `symbol_for`, with the MLM head."""

SCORING = """```python
ids = batch["input_ids"].clone()
ids[0, i] = tok.mask_token_id                      # mask the allele-pair symbol at output index i
with torch.no_grad():
    logprobs = model(input_ids=ids{species}).logits[0, i].log_softmax(-1)
score = logprobs[tok.convert_tokens_to_ids({symbol})]   # {what}
```

Masking hides the whole allele-pair symbol, so the head scores zygosity{orient}. {visible} For a reference-versus-alternative score, encode the window in both genotypes — their lengths differ by the
marker tokens, so trim each to a multiple of 128 — mask the site in each and compare the log-probabilities of the true
symbol. Use real genomic context; on the synthetic repeat above the ranking means nothing."""

OBJ = {
    "MLM": """**MLM.** 15% of tokens selected; 80% replaced by `<mask>`, 10% by a random token, 10% left unchanged. Masking is
per allele-pair symbol, so recovering a heterozygous position means predicting the joint allelic state.""",
    "CPL": """**CPL, Contrastive Phase Loss (weight 0.5).** MLM alone has little reason to read the direction markers. CPL builds a
twin of each window with every direction marker flipped, holds everything else fixed, and minimises the cosine similarity
between the two views' hidden states at heterozygous positions after a 64-d projection head, so flipping phase must move
the representation. The projection head is training-time only and is not shipped; flipping markers and measuring cosine on
`hidden_states[-1]` will not reproduce the training value.""",
    "SMP": """**SMP, Supervised Masked Phase (weight 0.2).** A fraction of direction markers is replaced by `<mask>` — a contiguous
block plus scattered singletons, always leaving at least one visible — and a small head must recover each hidden
orientation from the representation of the allele-pair symbol that follows it. Positions where copying the nearest visible
marker would be wrong (phase switches) are up-weighted 3x.""",
}

V6_BASE = """Phased genotypes from the 1000 Genomes high-coverage release (3,202 individuals, 602 trios) rendered against GRCh38 over
**140,099 regions, ~943 Mb (~30% of the genome)**: GENCODE v47 Basic coding exons and UCSC 100-way phastCons elements
(log-odds > 500), each inflated to at least 4,094 bp and merged within 300 bp. Records are (individual x region) pairs
deduplicated across individuals — 110,887,674 for training, corpus build `{build}` — with a pedigree-aware split so no
relative crosses it."""
V6_WINDOWS = """
Records are windowed at 4,096 tokens (stride 3,686), keeping every variant-bearing window, every reference window and 40%
of the rest: **221,329,576 training windows**."""
DATA = {
    "v6": V6_BASE.format(build="v6") + V6_WINDOWS,
    "v7": V6_BASE.format(build="v7") + V6_WINDOWS + " Build `v7` is a re-render of the same regions and encoding as `v6`.",
    "v7 (1,000-genome subset)": V6_BASE.format(build="v7") + """
This checkpoint saw a seeded subset of **1,000 of the 2,560 training individuals** plus GRCh38 (43.1M records), windowed at
8,192 tokens: 53.4M training windows, ~1 epoch.""",
    "v5": """Phased genotypes from the 1000 Genomes high-coverage release rendered against GRCh38 (corpus build `v5`, the marker-phase
encoding); stage 1 windowed at 4,096 tokens, stage 2 at 8,192.""",
    "v4 (ordered symbols)": """Phased genotypes from the 1000 Genomes high-coverage release rendered against GRCh38 (corpus build `v4`, ordered-phase
symbols); stage 1 windowed at 4,096 tokens, stage 2 at 8,192.""",
}

LIMITS = """- **Only the marker-phase rows of the token table were trained.** The 20 ordered-phase symbols and the 7 hemizygous
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
  local, not gene-scale."""

LIMITS_ORDERED = LIMITS.replace(
    "- **Only the marker-phase rows of the token table were trained.** The 20 ordered-phase symbols and the 7 hemizygous\n"
    "  (lowercase) symbols in `diploid_tokens.tsv` have zero count in the corpus; their embeddings are at initialisation.",
    "- **Only the ordered-phase and homozygous rows of the token table were trained.** The direction markers `‹` `›`, the\n"
    "  unordered pair symbols and the 7 hemizygous (lowercase) symbols have zero count in this corpus; their embeddings are at\n"
    "  initialisation.")
assert LIMITS_ORDERED != LIMITS


def files_table(m):
    code = ("`modeling_ntv3_posttrained.py`, `configuration_ntv3_posttrained.py` (+ the `_pretrained` pair they import)"
            if m["kind"] == "post" else "`modeling_ntv3_pretrained.py`, `configuration_ntv3_pretrained.py`")
    tokf = ("`vocab.json`, `added_tokens.json`, `tokenization_ntv3.py`, `tokenizer_config.json`"
            if m.get("family_ckpts") else "`tokenizer.json`, `tokenizer_config.json`")
    rows = [("`model.safetensors`, `config.json`", "weights and architecture"),
            (code, "NTv3 implementation; needs `trust_remote_code=True`"),
            (tokf, "55-token character tokenizer (11 NTv3 + 44 diploid)"),
            ("`diploid_tokens.tsv`", f"token table; sha256 prefix `{TSV_SHA}`"),
            ("`diploid_encode.py`", "haplotype pair -> token stream, byte-identical to the training pipeline")]
    if m.get("family_ckpts"):
        rows.insert(1, ("`checkpoints/stage1/genome-N/`, `checkpoints/stage2/final/`",
                        "eleven stage-1 milestones and stage-2 `final`, each a complete model directory; load with `subfolder=`"))
    return "| File | Purpose |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def results_block(m):
    snv = f"[`{ORG}/clinvar-diploid-snv`](https://huggingface.co/datasets/{ORG}/clinvar-diploid-snv)"
    indel = f"[`{ORG}/clinvar-diploid-indel`](https://huggingface.co/datasets/{ORG}/clinvar-diploid-indel)"
    if m["results"] == "early":
        return "No published evaluation; superseded by the DNT-* checkpoints in the family table."
    if m["results"] is None:
        return f"No published evaluation. Evaluate on {snv} and {indel}."
    return "\n".join(f"- {r}" for r in m["results"]) + f"\n\nBenchmark datasets: {snv} and {indel}."


def render(m):
    repo = m["repo"]
    objs = " + ".join(m["obj"])
    post = m["kind"] == "post"
    species_kw = ", species_ids=species_ids" if post else ""
    article = "An" if m["scale"].startswith("8") else "A"
    evaluated = "; this checkpoint is evaluated there" if isinstance(m["results"], list) else ""
    note = f"\n> {m['note']}\n" if m.get("note") else ""
    if m.get("ordered"):
        scoring = SCORING.format(species=species_kw, symbol='"Ƈ"', what="log P(C on hap0, G on hap1); ordered symbols are the `phased` rows of the token table",
                                 orient=" and orientation, which the ordered symbol carries",
                                 visible="Only the `^` marker stays visible, so one masked position compares every genotype at the site.")
    else:
        scoring = SCORING.format(species=species_kw, symbol='de.symbol_for("C", "G", tokens_tsv)', what="log P(het C/G here)",
                                 orient=" (and, where a direction marker precedes the site, orientation)",
                                 visible="The `^` and direction markers stay visible, so one masked position compares genotypes that share them "
                                         "(heterozygous against homozygous-ALT, or the two orientations).")
    encoding = ENCODING + (ORDERED_ENCODING if m.get("ordered") else SYMBOL_FOR)
    tags = ["genomics", "dna", "diploid", "zygosity", "phase", "nucleotide-transformer", "ntv3", "continued-pretraining"]
    return f"""---
license: other
license_name: research-only
license_link: LICENSE
library_name: transformers
pipeline_tag: fill-mask
tags:
{chr(10).join('- ' + t for t in tags)}
base_model:
- {m['base']}
---

# {repo}

{article} **{m['scale']}-parameter** [Nucleotide Transformer v3](https://huggingface.co/{m['base']}) backbone continually pretrained
on reference-aligned **diploid token streams** from the phased 1000 Genomes Project. Both homologues are written into one
sequence, so zygosity, allele dosage and cis/trans phase reach the encoder directly instead of being reconstructed from
two separately encoded haplotypes. Objectives: **{objs}**. Paper: [DNT: Diploid Genomic Foundation Model]({PAPER})
(bioRxiv 2026{evaluated}).
{note}
## At a glance

| | |
|---|---|
| Base checkpoint | [`{m['base']}`](https://huggingface.co/{m['base']}) |
| Parameters | {m['params']} ({m['weights']} float32 weights) |
| Embedding dim / transformer layers / heads | {m['dim']} / {m['layers']} / {m['heads']} |
| Objectives | {objs} |
| Training context | {ctx_str(m)} tokens |
| Input length | multiple of 128 tokens, at least 128 |
| Vocabulary | 55 tokens: 11 NTv3 + 44 diploid (`diploid_tokens.tsv`) |
| Corpus | 1000 Genomes high-coverage phased release, diploid build `{m['corpus']}` (see [Pretraining data](#pretraining-data)) |
| Schedule | {m['schedule']}; peak LR {m['lr']} |

## Model family

All repos share the token table and `diploid_encode.py`, so code written for one runs on the others. Two exceptions, each
spelled out in its own Quickstart: POST-backbone models load differently, and the ordered-phase repo needs `to_ordered()`.

{family_table(repo)}

## Quickstart

{quickstart(m)}

## Usage notes

{rules(m)}

## Encoding genotypes

{encoding}

## Scoring genotypes with the MLM head

{scoring}

## Objectives

{(chr(10) * 2).join(OBJ[o] for o in m['obj'])}

## Pretraining data

{DATA[m.get("data_key", m['corpus'])]}

## Limitations

{LIMITS_ORDERED if m.get("ordered") else LIMITS}

## Results

{results_block(m)}

## Files

{files_table(m)}

## Provenance

{m['provenance']}

Embedding layer and LM head resized from 11 to 55 rows; each new heterozygous row seeded from the mean of its constituent
nucleotide embeddings.

## Licence

Derived from InstaDeep's Nucleotide Transformer v3, which is released for non-commercial use only under the
[InstaDeep Open Model Licence v1.0](https://huggingface.co/{m['base']}/blob/main/MODEL-LICENSE.md).
These weights inherit those terms: **research use only, no commercial use, no serving outputs as a service.** See `LICENSE`.
Not for clinical use.

## Citation

```
Leib G, Zinger T, Ofer D, Kellerman R, Nayshool O, Dominissini D, Larey A, Levy J, Nahshan Y, Dahan E, Bleiweiss A,
Bussola N, Lee S, O'Connell S, Hoang D, Wirth M, Beckmann ND, Charney AW, Shavit Y, Daniel N, Rechavi G.
DNT: Diploid Genomic Foundation Model. bioRxiv 2026.09.05.749576 (2026). {PAPER}
```

Sheba Medical Center · NVIDIA · Icahn School of Medicine at Mount Sinai.
"""


BANNED = ("covariance", "covpool", "recipe", "phylop", "cadd", "allele freq", "allele-freq", "aux", "leak", "/storage", "/data/", "wandb")

if __name__ == "__main__":
    out = Path(__file__).with_name("cards"); out.mkdir(exist_ok=True)
    for m in MODELS:
        text = render(m)
        low = text.lower()
        hits = [b for b in BANNED if b in low]
        assert not hits, (m["repo"], hits)
        assert "{species}" not in text and "{symbol}" not in text
        (out / f"{m['repo']}.md").write_text(text, encoding="utf-8")
        print(m["repo"], len(text), "bytes")
