#!/usr/bin/env python
"""Verify a freeze arm actually froze what it claims, and measure whether its
trainable rows moved enough to interpret a null result.

The freeze arms' headline claim is "ACGT-task damage is ZERO BY CONSTRUCTION --
the base model is bitwise frozen". Nothing in the training loop checks that the
construction held after main training (the embedding-warmup phase has such a
check, main training does not). This script is that check, run against saved
milestones so it needs no restart and no GPU.

It also answers the question that decides whether a null result from `embfreeze`
means anything. That arm's entire representational degree of freedom is 44 rows
x token_embed_dim=16 = 704 numbers (its other ~34k trainable params are LM-head
rows, which move the MLM softmax and nothing a downstream embedding probe reads).
If those 704 barely moved, "no gain" is an LR artifact, not a result. The
reference is the matched full-fine-tune bundle arm at the same milestone.

    python -m pretrain.tools.verify_freeze \
        --arm <models>/100m_post_embfreeze \
        --reference <models>/100m_post_bundle_w2048

Exit code 1 if any frozen tensor moved -- that is a failed experiment, not a
warning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open

BASE_VOCAB = 11  # rows [0:11) are the original NTv3 ACGTN vocab; 11..55 are diploid
EMB = "core.embed_layer.weight"
HEAD_W = "core.lm_head.head.weight"
HEAD_B = "core.lm_head.head.bias"


KEYS = (EMB, HEAD_W, HEAD_B)


def _load(path: Path) -> dict[str, torch.Tensor]:
    """Only the per-vocab-row tensors: reading whole checkpoints here would pull
    gigabytes per milestone to compare a few thousand rows."""
    with safe_open(str(path / "model.safetensors"), "pt") as f:
        available = set(f.keys())
        return {k: f.get_tensor(k) for k in KEYS if k in available}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative Frobenius change, the same metric the audit used."""
    denom = a.norm().item()
    return (a - b).norm().item() / denom if denom else 0.0


def _milestones(arm: Path) -> list[Path]:
    return sorted(
        (d for d in arm.glob("genome-*") if (d / "model.safetensors").exists()),
        key=lambda d: int(d.name.split("-")[1]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Full-fine-tune arm to compare new-row movement against (e.g. the bundle arm).",
    )
    ap.add_argument(
        "--anchor",
        type=Path,
        default=None,
        help="Baseline to diff against (default: <arm>/_emb_warmup/final, which is "
        "what the arm actually resumed from).",
    )
    a = ap.parse_args()

    anchor_dir = a.anchor or (a.arm / "_emb_warmup" / "final")
    if not (anchor_dir / "model.safetensors").exists():
        print(f"FATAL: anchor {anchor_dir} missing", file=sys.stderr)
        return 2
    anchor = _load(anchor_dir)

    ms = _milestones(a.arm)
    if not ms:
        print(f"no milestones saved yet under {a.arm}")
        return 0

    ref_by_genome: dict[int, Path] = {}
    ref_anchor = None
    if a.reference:
        ref_anchor_dir = a.reference / "_emb_warmup" / "final"
        if (ref_anchor_dir / "model.safetensors").exists():
            ref_anchor = _load(ref_anchor_dir)
            # path only; loaded on demand below, so milestones this arm never
            # reached are never read
            ref_by_genome = {
                int(d.name.split("-")[1]): d for d in _milestones(a.reference)
            }

    print(f"arm    : {a.arm}")
    print(f"anchor : {anchor_dir}")
    print()
    hdr = f"{'milestone':<12} {'FROZEN base rows':<20} {'new emb rows':>14} {'ref new emb':>12}  verdict"
    print(hdr)
    print("-" * len(hdr))

    failed = False
    for d in ms:
        g = int(d.name.split("-")[1])
        cur = _load(d)

        # 1. The load-bearing claim: base vocab rows must be BITWISE identical.
        frozen_ok = True
        worst = 0.0
        for key in (EMB, HEAD_W, HEAD_B):
            if key not in cur or key not in anchor:
                continue
            c, b = cur[key][:BASE_VOCAB], anchor[key][:BASE_VOCAB]
            if not torch.equal(c, b):
                frozen_ok = False
                worst = max(worst, (c - b).abs().max().item())

        # 2. Did the rows that CAN change the representation actually move?
        new_emb = _rel(cur[EMB][BASE_VOCAB:], anchor[EMB][BASE_VOCAB:])
        ref_txt = "-"
        ref_val = None
        if g in ref_by_genome and ref_anchor is not None:
            ref_cur = _load(ref_by_genome[g])
            ref_val = _rel(ref_cur[EMB][BASE_VOCAB:], ref_anchor[EMB][BASE_VOCAB:])
            ref_txt = f"{ref_val:.4f}"

        base_txt = "bitwise OK" if frozen_ok else f"MOVED max={worst:.2e}"
        verdict = ""
        if not frozen_ok:
            verdict = "<== FREEZE LEAK"
            failed = True
        elif ref_val:
            ratio = new_emb / ref_val
            verdict = (
                f"moved {ratio:.2f}x the full-FT arm"
                + ("  <== under-trained, a null here is an LR artifact" if ratio < 0.4 else "")
            )
        print(f"{d.name:<12} {base_txt:<20} {new_emb:>14.4f} {ref_txt:>12}  {verdict}")

    print()
    if failed:
        print("FAILED: frozen rows moved. The arm's central claim does not hold.")
        return 1
    print("All frozen tensors bitwise-identical to the anchor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
