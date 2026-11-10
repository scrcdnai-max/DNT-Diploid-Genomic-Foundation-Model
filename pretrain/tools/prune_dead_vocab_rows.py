#!/usr/bin/env python
"""Rewrite a checkpoint with only the vocabulary rows its corpus actually uses.

``--drop-unseen-tokens`` applies at vocab-expansion time. An arm that already has
a completed embedding warmup would have to redo it -- and, worse, would lose the
shared anchor that makes it comparable to the rest of the ladder. This tool gets
the same vocabulary by slicing the finished checkpoint instead.

That is exact, not approximate. The dropped symbols never occur, so their input
rows are never gathered; the surviving rows keep bitwise values; and the
embedding benchmarks read hidden states, not LM-head logits. So a pruned arm may
keep its unpruned anchor -- see test_prune_dead_vocab_rows.py, which asserts the
hidden states are bitwise equal rather than arguing it.

    uv run --no-sync python -m pretrain.tools.prune_dead_vocab_rows \
        --src  /storage/users/ddofer/models/100m_post_bundle_w2048/_emb_warmup/final \
        --dst  /storage/users/ddofer/models/100m_post_loraconv/_emb_warmup/final

RUN pretrain/tools/verify_dead_tokens.py FIRST. If the counts table is stale for
the corpus, this deletes a row for a symbol that does occur and it silently
becomes <unk> from then on.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pretrain"))

from tokenizer_lut import merged_vocab
from train import (
    DEFAULT_TOKEN_COUNTS_PATH,
    _convert_tokenizer_to_fast,
    _filter_unseen_tokens,
    _parse_diploid_tsv,
    _write_filtered_tokens_tsv,
    load_token_unigram_freqs,
    prune_vocab_rows,
)

DEFAULT_TOKENS_TSV = REPO_ROOT / "preprocess" / "diploid_tokens.tsv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="checkpoint to prune")
    ap.add_argument("--dst", type=Path, required=True, help="where to write it")
    ap.add_argument(
        "--base-model",
        type=Path,
        default=Path("/storage/models/NTv3_100M_post"),
        help="Unexpanded NTv3 dir, used to rebuild the tokenizer from its base "
        "vocabulary. Adding the kept symbols to a fresh base assigns exactly the "
        "ids a --drop-unseen-tokens run would; editing the expanded "
        "tokenizer.json in place does not (its added tokens cannot be renumbered).",
    )
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS_TSV)
    ap.add_argument("--token-counts", type=Path, default=DEFAULT_TOKEN_COUNTS_PATH)
    a = ap.parse_args()

    if not (a.src / "model.safetensors").exists():
        print(f"FATAL: no model.safetensors under {a.src}", file=sys.stderr)
        return 2
    if a.dst.exists() and any(a.dst.iterdir()):
        print(f"FATAL: {a.dst} exists and is not empty", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(a.src, trust_remote_code=True)
    config = json.loads((a.src / "config.json").read_text())
    is_post = config.get("model_type", "").endswith("posttrained")
    model = (AutoModel if is_post else AutoModelForMaskedLM).from_pretrained(
        a.src, trust_remote_code=True
    )

    base = int(getattr(model.config, "base_vocab_size", 11))
    vocab = merged_vocab(tok)
    symbols, _, _ = _parse_diploid_tsv(a.tokens)
    freqs = load_token_unigram_freqs(a.token_counts)
    if freqs is None:
        print(f"FATAL: no token counts at {a.token_counts}", file=sys.stderr)
        return 2

    # Only consider symbols this checkpoint actually carries as ADDED rows;
    # ACGTN live in the base block and are never candidates.
    added = sorted(s for s in symbols if vocab.get(s, -1) >= base)
    kept, dropped = _filter_unseen_tokens(added, freqs)
    if not dropped:
        print("nothing to prune -- every symbol occurs in the corpus")
        return 0

    keep_rows = list(range(base)) + [vocab[s] for s in kept]
    print(f"vocab {len(vocab)} -> {len(keep_rows)}  (base {base} + {len(kept)} added)")
    print(f"dropping {len(dropped)}: {' '.join(dropped)}")

    prune_vocab_rows(model, keep_rows)

    # Rebuild the tokenizer from the ORIGINAL base vocabulary plus the kept
    # symbols, so ids are assigned exactly as a fresh --drop-unseen-tokens run
    # would assign them -- rather than surgically renumbering tokenizer.json.
    new_tok = AutoTokenizer.from_pretrained(a.base_model, trust_remote_code=True)
    if len(merged_vocab(new_tok)) != base:
        print(
            f"FATAL: --base-model {a.base_model} has "
            f"{len(merged_vocab(new_tok))} tokens, expected the unexpanded {base}",
            file=sys.stderr,
        )
        return 2
    new_tok.add_tokens(kept)

    a.dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(a.dst))
    _convert_tokenizer_to_fast(new_tok).save_pretrained(str(a.dst))
    _write_filtered_tokens_tsv(a.tokens, a.dst, set(dropped))

    # Carry over the remote-code modules so the pruned dir loads standalone.
    for py in a.src.glob("*.py"):
        shutil.copy2(py, a.dst / py.name)

    check = merged_vocab(AutoTokenizer.from_pretrained(a.dst, trust_remote_code=True))
    bad = [s for i, s in enumerate(kept) if check.get(s) != base + i]
    if bad:
        print(f"FATAL: tokenizer ids do not match pruned rows for {bad}", file=sys.stderr)
        return 1
    print(f"OK: wrote {a.dst} (vocab {len(check)}); tokenizer ids match pruned rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
