"""Diploid-aware MLM pretraining on NTv3 with vocabulary expansion.

Reads pre-built HuggingFace Parquet shards (one per sample) containing diploid
genomic sequences with ^/{/} variant markers.  Slides windows over records,
filters for variant-containing windows, and trains masked language modeling.

Examples:
    python train.py --quick-test --train-data /path/to/train/
    python train.py --model-name InstaDeepAI/NTv3_100M_post \
        --train-data /data/diploid_hf_v7/assembled/train --window-len 2048 --lr 1e-4
    bash pretrain/run_100m_tsstes.sh      # production launchers set the env and call this file
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import logging
import math
import os
import shutil
import sys
import tempfile
import time
import types
import warnings
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import datasets
import numpy as np
import orjson
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.modeling_outputs import MaskedLMOutput
from transformers.utils import logging as hf_logging

try:
    from .multiobj import (
        BASE_TO_INT,
        AFTables,
        effect_ranked_margin_loss,
        token_genomic_offsets,
    )
    from .ntv3_attention_patch import (
        disable_cudnn_sdp,
        apply_pad_mask_from_env,
        patch_ntv3_attention_sdpa,
    )
    from .ntv3_pool_surgery import apply_pool_bypass_from_env
    from .windowing import resolve_stride
    from .length_buckets import (
        BUCKET_STEP,
        LengthBucketBatchSampler,
        bucket_lengths,
        bucket_of,
        padding_efficiency,
    )
except ImportError:
    from multiobj import (
        BASE_TO_INT,
        AFTables,
        effect_ranked_margin_loss,
        token_genomic_offsets,
    )
    from ntv3_attention_patch import (
        disable_cudnn_sdp,
        apply_pad_mask_from_env,
        patch_ntv3_attention_sdpa,
    )
    from ntv3_pool_surgery import apply_pool_bypass_from_env
    from windowing import resolve_stride
    from length_buckets import (
        BUCKET_STEP,
        LengthBucketBatchSampler,
        bucket_lengths,
        bucket_of,
        padding_efficiency,
    )

logger = logging.getLogger(__name__)

# -- Defaults -----------------------------------------------------------------
MODEL_NAME = "InstaDeepAI/NTv3_8M_pre"
PAD_MULTIPLE = (
    128  # NTv3 U-Net requires sequence length divisible by 128 (= 2**num_downsamples)
)
SCRIPT_DIR = Path(__file__).resolve().parent


def _assert_pad_multiple_matches_model(model: Any) -> None:
    """Fail LOUD if PAD_MULTIPLE (hardcoded U-Net length divisor) disagrees with the model's
    actual conv-tower stride (2**num_downsamples). Aux-target pooling + the bottleneck head
    assume windows are a multiple of PAD_MULTIPLE; a future NTv3 variant with a different
    num_downsamples would otherwise silently desync them. Mirrors the tokenizer/LUT parity
    guard. Missing field (older configs) -> warn, no raise."""
    nd = getattr(getattr(model, "config", None), "num_downsamples", None)
    if nd is None:
        logger.warning(
            "num_downsamples absent from model config; cannot verify PAD_MULTIPLE=%d",
            PAD_MULTIPLE,
        )
        return
    derived = 2 ** int(nd)
    if derived != PAD_MULTIPLE:
        raise AssertionError(
            f"PAD_MULTIPLE={PAD_MULTIPLE} != 2**num_downsamples={derived} (num_downsamples={nd}). "
            "The conv tower stride changed (new model variant?); aux-target pooling + the "
            "bottleneck head would silently corrupt. Update PAD_MULTIPLE to match the model."
        )
    logger.info(
        "PAD_MULTIPLE parity OK: %d == 2**num_downsamples (%d)", PAD_MULTIPLE, nd
    )


def _maybe_bypass_pools(model: Any) -> None:
    """Apply the pool-bypass surgery (env, else the incoming stamp) and RECORD it.

    The surgery deliberately leaves `state_dict` untouched, so a checkpoint trained with it
    is byte-identical in shape to a vanilla one and nothing on disk would otherwise say the
    Transformer ran at 2x resolution. Stamping the config means a later loader can tell --
    and, critically, that benchmarking the checkpoint WITHOUT re-applying the bypass is a
    detectable mistake rather than a silently wrong number. `_resolve_bypass` makes RESUMING
    it without the env the same kind of non-mistake: the stamp is what a relaunch follows.

    PAD_MULTIPLE stays at 2**num_downsamples: windows are still a multiple of the (now
    halved) effective stride, so padding is unaffected.

    Aux heads: FULL-RESOLUTION phyloP/CADD heads are unaffected -- they read the post-deconv
    hidden state, which is still base resolution. Only `--aux-bottleneck-tasks` changes meaning,
    and only semantically: its pooling stride is derived (`tgt.shape[1] // pred.shape[1]`), so it
    still runs, but each bin covers 2**(downsamples - n) bp instead of 128 bp.
    """
    _, n, scale = apply_pool_bypass_from_env(model)
    # Stamp unconditionally: the stamp must equal what THIS process applied. The
    # resolution honours the incoming stamp when the env is unset, so a crash-guard
    # relaunch that lost NTV3_POOL_BYPASS keeps training at the resolution it
    # started at -- and an explicit env that overrides the stamp is recorded here
    # rather than leaving a stale one behind for scorers and the rotary prewarm.
    model.config.pool_bypass = n
    model.config.pool_bypass_pos_scale = scale
    if n or scale != 1.0:
        logger.warning(
            "pool bypass n=%d pos_scale=%g stamped on model.config", n, scale
        )


DEFAULT_TRAIN_DATA = Path("/data/diploid_hf_v7/assembled/train")
DEFAULT_RAW_TEST_DATA = Path("/data/diploid_hf_v7/assembled/test")
# v6 has no _4k_windowed split; keep the v5 fallback so legacy callers asking
# for pre-computed windowed eval shards still resolve to something on disk.
DEFAULT_WINDOWED_TEST_DATA = Path("/data/diploid_hf_v7/assembled/test_4k_windowed")
DEFAULT_WINDOWED_TEST_DATA_FALLBACK = SCRIPT_DIR / "data" / "test_4k_windowed"

# Variant markers baked into sequences by create_dataset.py
SNP_MARKER = "^"
INDEL_OPEN = "{"
INDEL_CLOSE = "}"
DIR_FWD = "\u2039"  # ‹
DIR_REV = "\u203a"  # ›
# All marker tokens (used for special-token detection / has_variant indexing).
MARKER_TOKENS = (SNP_MARKER, INDEL_OPEN, INDEL_CLOSE, DIR_FWD, DIR_REV)

def _marker_ids(tokenizer: Any) -> set[int]:
    """Resolve the structural marker tokens against the FULL vocabulary.

    Markers are added tokens, so `get_vocab()` alone returns an empty set here --
    which fails silently: callers then treat every marker as an ordinary diploid
    token. That is what left `loss/marker_mlm` unlogged in every run to
    2026-08-19 and blended bracket grammar into `loss/diploid_mlm`.
    """
    vocab = merged_vocab(tokenizer)
    return {vocab[m] for m in MARKER_TOKENS if m in vocab}


# Markers that may be randomly DELETED from the input during MLM training to
# prevent leakage of variant existence/phase. INDEL span braces are kept so the
# bracketed structure stays intact.
MARKER_DROP_TOKENS = (DIR_FWD, DIR_REV)
REF_SAMPLE_ID = "HG38_REF"

# -- Auxiliary per-position regression (v7) -----------------------------------
# Two per-position targets ride alongside MLM, each with a Linear(hidden, 1) head
# and a NaN-masked, sample-weighted Huber loss (see AuxRegressionLossMixin):
#   * phylop      — conservation, DENSE uint8 aligned 1:1 with seq codepoints;
#                   255 = NaN sentinel, [0,254] linearly encodes [-20, +10.003]
#                   (~99% finite; NaN at markers/inserts/no-coverage).
#   * cadd_phred  — deleteriousness, SPARSE list<struct<pos,score>> on variant
#                   anchors only (~0.1% density); expanded to a dense NaN target.
# Standardization constants are fixed dataset-level values computed offline over
# v7 train shards; min −20 is the phyloP bigWig floor, clipped by the Huber tail.
# They are stamped on model.config and overridable via CLI so eval/inference
# match. The mechanism is generic over AUX_POSITION_TASKS; the only per-task
# specialization is the dense-vs-sparse decode (AUX_SPARSE_TASKS), isolated to
# DiploidWindowDataset.__getitem__.
PHYLOP_MEAN = 0.25  # decoded-scale mean over v7 train shards (measured 0.285)
PHYLOP_STD = 1.45  # decoded-scale std over v7 train shards (measured 1.458)
CADD_MEAN = 3.02  # PHRED mean over 10.0M anchors across 12 v7 train shards
CADD_STD = 4.00  # PHRED std over 10.0M anchors across 12 v7 train shards

# PhyloP uint8 → float32 decode. MUST match preprocess/huggingface_dataset/
# dataset.py (_PHYLOP_OFFSET/_RANGE/_NAN_SENTINEL); a drift-guard test asserts
# equality (tests/test_aux_decode_constants.py).
_PHYLOP_OFFSET = 20.0
_PHYLOP_RANGE = 30.003
_PHYLOP_NAN_SENTINEL = 255


def _decode_phylop_window(raw) -> np.ndarray:
    """Decode a uint8 phyloP slice to float32: 255→NaN, else linear [-20,+10.003]."""
    arr = np.asarray(raw, dtype=np.uint8)
    out = np.full(len(arr), np.nan, dtype=np.float32)
    valid = arr != _PHYLOP_NAN_SENTINEL
    out[valid] = arr[valid].astype(np.float32) / 254.0 * _PHYLOP_RANGE - _PHYLOP_OFFSET
    return out


# Precomputed uint8→float16 decode table. `_PHYLOP_LUT[v]` is bit-identical to
# `_decode_phylop_window([v]).astype(float16)` for every byte value (255→NaN), but
# turns the per-window decode into a single gather — measurably cheaper on the hot
# __getitem__ path. Derived from `_decode_phylop_window` so it can never drift from
# the constants (the decode-drift test still guards those).
_PHYLOP_LUT = _decode_phylop_window(np.arange(256, dtype=np.uint8)).astype(np.float16)


# Per-position aux columns to read/window/gather and attach heads for. The HF
# dataset column name doubles as the batch key and the head name. Tasks listed in
# AUX_SPARSE_TASKS are stored sparsely (list<struct<pos,score>>) and expanded to a
# dense NaN target in the dataset; the rest are dense per-position arrays.
AUX_POSITION_TASKS = ("phylop", "cadd_phred")
AUX_SPARSE_TASKS = frozenset({"cadd_phred"})

# Per-task standardization (mean, std) + Huber transition beta. Module-level,
# data-derived constants — the single source of truth (also stamped on
# model.config so eval/inference match). NOT CLI knobs; only the loss WEIGHT is
# tunable (--aux-*-weight). CADD uses a larger beta so its deleterious right tail
# stays closer to MSE (beta=1 SD would over-clip the high-PHRED signal of interest).
# Track-C: 1000G allele frequency in [0,1], heavily right-skewed toward rare. A
# carried-variant's AF spans the whole range; (mean,std) here just z-scores it for
# the Huber head (collator-produced target, not a dataset column).
AF_MEAN, AF_STD = 0.15, 0.22
AUX_STANDARDIZE: dict[str, tuple[float, float]] = {
    "phylop": (PHYLOP_MEAN, PHYLOP_STD),
    "cadd_phred": (CADD_MEAN, CADD_STD),
    "af": (AF_MEAN, AF_STD),
}
AUX_HUBER_BETA: dict[str, float] = {"phylop": 1.0, "cadd_phred": 3.0, "af": 1.0}


os.environ.setdefault("WANDB_PROJECT", "diploid_glm_pretrain")

_CHECKPOINT_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _is_post_model(name: str) -> bool:
    """True when `name` refers to a post-trained (multi-head) NTv3 variant."""
    return "_post" in name.lower().replace("-", "_")


def _patch_post_model_embeddings(model: Any) -> None:
    """Add embedding accessors to NTv3PostTrained (present on _pre, missing on _post).

    Once patched, the HF base ``resize_token_embeddings`` works out of the box.
    """
    import types

    def _set_out(self, v):
        self.core.lm_head["head"] = v

    model.get_input_embeddings = types.MethodType(lambda s: s.core.embed_layer, model)
    model.set_input_embeddings = types.MethodType(
        lambda s, v: setattr(s.core, "embed_layer", v), model
    )
    model.get_output_embeddings = types.MethodType(
        lambda s: s.core.lm_head["head"], model
    )
    model.set_output_embeddings = types.MethodType(_set_out, model)
    model.tie_weights = types.MethodType(lambda s, **kw: None, model)


os.environ.setdefault("WANDB_LOG_MODEL", "end")
os.environ.setdefault("WANDB_WATCH", "false")


def _ensure_hub_repo_private_gated(repo_id: str) -> bool:
    """Create the repo (if needed) as private and set gated="manual".

    Idempotent and rank-0 only. HF Hub writes are best-effort so transient
    network issues do not abort an expensive training job.
    """
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return True
    for attempt in range(3):
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
            api.update_repo_settings(repo_id=repo_id, gated="manual", private=True)
            logger.info("HF Hub repo ready (private+gated): %s", repo_id)
            return True
        except Exception as exc:  # noqa: BLE001 -- best-effort external I/O
            if attempt == 2:
                logger.warning("Could not configure HF Hub repo %s: %s", repo_id, exc)
                return False
            time.sleep(2**attempt)
    return False


def _wrapped_resume_seed(ckpt: Path | None, output_dir: Path) -> Path | None:
    """Where to build the ARCHITECTURE from when resuming a wrapped run.

    A raw Trainer checkpoint of an AuxHeadsModel is a bare state_dict (no config.json
    -- the wrapper is not a PreTrainedModel), so from_pretrained cannot build the model
    from it. Seed the architecture from a loadable sibling instead; the exact weights
    and optimizer are restored afterwards by trainer.train(resume_from_checkpoint=...).
    Keyed on the missing config.json, NOT on --aux-heads: the wrapper is also active for
    cpl/smp/ctp/trackb runs.

    Order: the checkpoint itself if loadable; else this run's baseline_expanded_vocab;
    else the (possibly inherited) _emb_warmup/final -- arms seeded from another arm's
    warmup have no baseline of their own, which is the case that used to fail. Returns
    the original path when nothing better exists so the caller fails loudly on it.
    """
    if ckpt is None or (ckpt / "config.json").exists():
        return ckpt
    for cand in (output_dir / "baseline_expanded_vocab", output_dir / "_emb_warmup" / "final"):
        if (cand / "config.json").exists():
            logger.info(
                "Wrapped resume: %s has no config.json; seeding architecture from %s "
                "(exact weights restored from the checkpoint by trainer.train).",
                ckpt, cand,
            )
            return cand
    return ckpt


def _is_valid_checkpoint_dir(checkpoint_dir: Path) -> bool:
    """Return True when *checkpoint_dir* contains serialized model weights."""
    return checkpoint_dir.is_dir() and any(
        (checkpoint_dir / filename).exists() for filename in _CHECKPOINT_WEIGHT_FILES
    )


def _save_model_artifacts(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path,
    *,
    allow_peft_mutation: bool = False,
) -> None:
    """Persist model and tokenizer assets into *save_dir*.

    PEFT-aware: when *model* is a ``PeftModel`` (LoRA-wrapped), the default
    ``save_pretrained`` writes only the adapter weights — downstream
    ``AutoModelForMaskedLM.from_pretrained(save_dir)`` would then fail or
    silently load the un-fine-tuned base. We produce a standard HF
    artifact (loadable by all existing tooling: benchmarks,
    ``validate_diploid_embeddings``, Hub consumers) by merging LoRA deltas
    into the base weights before saving. The original adapter is also
    preserved under ``save_dir/adapter/`` for audit.

    Two save modes for PEFT models:

    - ``allow_peft_mutation=True`` (end-of-training): mutate the live
      trainer model via ``merge_and_unload``. Cheap, no extra memory.
    - ``allow_peft_mutation=False`` (mid-training; e.g., genome
      milestones): deepcopy the model to CPU, merge the COPY, save the
      merged copy, discard the copy. The live trainer model is untouched.
      Memory overhead: 2× model weight transiently (CPU side), released
      via ``del`` + ``gc.collect`` after save. For 8M=~16MB, 100M=~200MB,
      650M=~1.3GB — acceptable on any modern host.

    Either way, the produced ``save_dir`` is a standard, benchmark-loadable
    HF artifact (the cross-arm comparison protocol depends on this). The
    adapter is always preserved alongside for audit.

    Aux-aware: when *model* is an ``AuxHeadsModel``, the backbone is saved as the
    clean HF artifact (so benchmarks load a plain ``AutoModelForMaskedLM``) and
    the regression heads are written to ``aux_heads.safetensors`` alongside.
    """
    aux_model: AuxHeadsModel | None = None
    if isinstance(model, AuxHeadsModel):
        aux_model = model
        model = model.backbone  # save the clean (optionally PEFT) backbone

    try:
        from peft import PeftModel  # local import; only when PEFT runs
    except ImportError:
        PeftModel = None  # type: ignore[assignment]

    if PeftModel is not None and isinstance(model, PeftModel):
        adapter_dir = save_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        # save_embedding_layers="auto" (the default) probes the base config's `vocab_size` --
        # NTv3PostTrainedConfig has no such attribute (it uses `alphabet_size`), which crashes
        # peft's auto-detection. Embeddings here are excluded from LoRA targeting and saved via
        # `modules_to_save` instead (a separate, unaffected code path), so the embedding-layer
        # probe this flag guards is moot -- pin it off rather than let auto-detection run.
        model.save_pretrained(str(adapter_dir), save_embedding_layers=False)

        if allow_peft_mutation:
            logger.info(
                "PEFT model (end-of-training): saved adapter to %s; merging "
                "adapters into base IN-PLACE for standard HF save at %s.",
                adapter_dir,
                save_dir,
            )
            merged = model.merge_and_unload(progressbar=False)
            merged.save_pretrained(str(save_dir))
        else:
            import copy as _copy

            logger.info(
                "PEFT model (mid-training): saved adapter to %s; deepcopying "
                "model to CPU and merging the COPY for standard HF save at "
                "%s (live trainer model untouched).",
                adapter_dir,
                save_dir,
            )
            model_copy = _copy.deepcopy(model).cpu()
            merged_copy = model_copy.merge_and_unload(progressbar=False)
            merged_copy.save_pretrained(str(save_dir))
            del model_copy, merged_copy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        model.save_pretrained(str(save_dir))
    fast_tok = _convert_tokenizer_to_fast(tokenizer)
    fast_tok.save_pretrained(str(save_dir))
    if aux_model is not None:
        aux_model.save_aux_heads(save_dir)


def _atomic_save_dir(target_dir: Path, write_fn: Callable[[Path], None]) -> None:
    """Write *target_dir* via a sibling temp dir, then swap it into place."""
    parent_dir = target_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{target_dir.name}.tmp-", dir=str(parent_dir))
    )
    backup_dir = parent_dir / f".{target_dir.name}.bak"

    try:
        write_fn(tmp_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if target_dir.exists():
            target_dir.rename(backup_dir)
        try:
            tmp_dir.rename(target_dir)
        except Exception:
            if backup_dir.exists() and not target_dir.exists():
                backup_dir.rename(target_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def _resolve_marker_drop_ids(
    tokenizer: PreTrainedTokenizerBase,
    marker_drop_prob: float,
    target: str = "direction",
) -> tuple[int, ...]:
    """Resolve marker-drop token ids, failing fast when required tokens are missing.

    ``target='direction'`` (default) drops the ‹/› direction markers (MARKER_DROP_TOKENS);
    ``target='snp'`` drops only the ``^`` SNP marker, leaving ‹/› intact — the augmentation
    the published CPL recipe used, and the one SMP/CTP can share (they read ‹/› + the het
    base, which ``^``-drop never touches); ``target='all'`` drops every marker (^ { } ‹ ›), so
    at p=1 the model reads the diploid glyphs in place, as the cohort's in-place input does."""
    drop_tokens = {"snp": (SNP_MARKER,), "all": MARKER_TOKENS}.get(target, MARKER_DROP_TOKENS)
    vocab = {
        **tokenizer.get_vocab(),
        **{
            str(token): token_id
            for token, token_id in getattr(
                tokenizer, "added_tokens_encoder", {}
            ).items()
        },
    }
    missing_marker_tokens = [token for token in drop_tokens if token not in vocab]
    if marker_drop_prob > 0 and missing_marker_tokens:
        raise SystemExit(
            "Marker dropout requires tokenizer entries for all marker tokens; "
            f"missing={missing_marker_tokens}"
        )
    return tuple(vocab[token] for token in drop_tokens if token in vocab)


def _push_folder_as_branch(repo_id: str, folder: Path, branch: str) -> None:
    """Push *folder* contents to *repo_id* on a dedicated branch (rank-0 only)."""
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    marker_path = folder / ".push_failed"
    for attempt in range(3):
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_branch(repo_id=repo_id, branch=branch, exist_ok=True)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(folder),
                revision=branch,
                commit_message=f"Add {branch} checkpoint",
            )
            marker_path.unlink(missing_ok=True)
            logger.info("Pushed %s -> %s@%s", folder, repo_id, branch)
            return
        except Exception as exc:  # noqa: BLE001 -- best-effort external I/O
            if attempt == 2:
                marker_path.write_text(
                    "\n".join(
                        (
                            f"repo_id={repo_id}",
                            f"branch={branch}",
                            f"error={exc!r}",
                            f"timestamp_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )
                logger.warning("HF Hub push failed for %s@%s: %s", repo_id, branch, exc)
                return
            time.sleep(2**attempt)


def _hub_branch_prefix(output_dir: Path) -> str:
    """Stable branch prefix derived from run dir + stage dir."""
    parent_name = output_dir.parent.name.strip()
    if parent_name:
        return f"{parent_name}-{output_dir.name}"
    return output_dir.name


def _has_data_shards(data_dir: Path) -> bool:
    return data_dir.exists() and any(
        p.suffix in {".parquet", ".arrow"}
        and not p.name.endswith(".tmp")
        and not p.name.startswith("cache-")
        for p in data_dir.iterdir()
        if p.is_file()
    )


def _default_eval_data_dir() -> Path:
    if _has_data_shards(DEFAULT_RAW_TEST_DATA):
        return DEFAULT_RAW_TEST_DATA
    for candidate in (
        DEFAULT_WINDOWED_TEST_DATA,
        DEFAULT_WINDOWED_TEST_DATA_FALLBACK,
    ):
        if _has_data_shards(candidate):
            return candidate
    return DEFAULT_RAW_TEST_DATA


DEFAULT_TEST_DATA = _default_eval_data_dir()


# -- Helpers ------------------------------------------------------------------


def _parse_diploid_tsv(
    tokens_path: Path,
) -> tuple[set[str], dict[str, tuple[str, str]], dict[str, str]]:
    """Parse diploid tokens TSV.

    Returns
    -------
    symbols   : set of all unique single-char symbols.
    sym2pair  : symbol -> (parent_a, parent_b) for the 2-char pair rows.
                Preserves the TSV's stated ordering — for phased rows this
                distinguishes the canonical-ordered symbol (e.g. ``AC -> Á``)
                from its reciprocal (``CA -> Ç``).
    sym2order : symbol -> order column value ("symmetric" | "unphased" |
                "phased" | "single" | "meta"). Used by the collision-aware
                embedding initializer to distinguish phased forward/reverse
                from unphased symbols that share the same parent pair.
    """
    df = pl.read_csv(
        tokens_path,
        separator="\t",
        comment_prefix="#",
        has_header=False,
        new_columns=["pair", "order", "symbol", "unicode"],
    )
    symbols = set(df["symbol"].to_list())
    # First occurrence of each symbol where pair is a 2-char nucleotide combo.
    pairs_df = df.filter(pl.col("pair").str.len_chars() == 2).unique(
        subset=["symbol"], keep="first"
    )
    sym2pair = {
        row["symbol"]: (row["pair"][0], row["pair"][1])
        for row in pairs_df.iter_rows(named=True)
    }
    # First occurrence of each symbol's order tag (covers single-char rows too).
    order_df = df.unique(subset=["symbol"], keep="first")
    sym2order = {
        row["symbol"]: str(row["order"]) for row in order_df.iter_rows(named=True)
    }
    return symbols, sym2pair, sym2order


def load_diploid_symbols(
    tokens_path: Path, tokenizer_vocab: dict[str, int]
) -> list[str]:
    """Return new diploid symbols not already in token vocabulary."""
    symbols, _, _ = _parse_diploid_tsv(tokens_path)
    return sorted(symbols - set(tokenizer_vocab))


# Whole-train-set symbol counts produced by preprocess/gather_diploid_stats.py.
DEFAULT_TOKEN_COUNTS_PATH = SCRIPT_DIR.parent / "preprocess" / "token_counts.tsv"

# Bumped whenever the rare-token MLM recipe changes in a way that makes losses
# non-comparable across checkpoints. Stamped on model.config.rare_token_recipe;
# ABSENT on any checkpoint older than v1 (uniform-over-vocab random fills, no
# head-bias rarity prior).
RARE_TOKEN_RECIPE_VERSION = 1


def load_token_unigram_freqs(counts_path: Path | str) -> dict[str, float] | None:
    """Parse ``preprocess/token_counts.tsv`` into ``{symbol: unigram_prob}``.

    The TSV repeats each symbol once per (pair, order) row with the same count
    (e.g. Ą appears under both AC and CA); dedupe keeps the max so a drifted
    row can only over-count, never silently zero a symbol. Symbols absent from
    the data are kept as explicit 0.0 (known-zero, distinct from unknown).
    Returns ``None`` when the file is missing/empty so callers can fall back.
    """
    path = Path(counts_path)
    if not path.is_file():
        logger.warning("Token counts table not found at %s", path)
        return None
    df = (
        pl.read_csv(path, separator="\t", comment_prefix="#")
        .group_by("symbol")
        .agg(pl.col("count").max())
    )
    total = int(df["count"].sum())
    if total <= 0:
        logger.warning("Token counts table %s has no positive counts", path)
        return None
    return {row["symbol"]: row["count"] / total for row in df.iter_rows(named=True)}


def prune_vocab_rows(model: Any, keep_rows: list[int]) -> Any:
    """Drop vocabulary rows from a trained model in place; returns *model*.

    Lets an arm adopt the ``--drop-unseen-tokens`` vocabulary without redoing its
    embedding warmup -- and, more importantly, without giving up a shared anchor:
    pruning rows whose symbols never occur is a forward-pass NO-OP, not an
    approximation.

    - the dropped input rows are never gathered (their symbols are absent), so
      no activation anywhere depends on them;
    - surviving rows keep their bitwise values, so the id shift is invisible as
      long as the tokenizer travels with the checkpoint (it does);
    - the embedding benchmarks read hidden states, never LM-head logits, so
      removing output classes cannot move a score.

    What DOES change is training: the removed head rows were permanently-wrong
    MLM classes, so the softmax denominator loses their (tiny) mass.

    ``keep_rows`` must include every base ACGT row -- those are held frozen by
    construction everywhere else, and dropping one would change what every ACGT
    benchmark reads.
    """
    vocab = int(model.config.alphabet_size)
    if any(r < 0 or r >= vocab for r in keep_rows):
        raise ValueError(f"keep_rows has ids out of range for vocab {vocab}")
    base = int(getattr(model.config, "base_vocab_size", 11))
    missing_base = [r for r in range(min(base, vocab)) if r not in set(keep_rows)]
    if missing_base:
        raise ValueError(
            f"keep_rows drops base vocab rows {missing_base}; base rows [0:{base}) "
            "are frozen by construction in every other path and must be kept"
        )
    if len(keep_rows) == vocab and list(keep_rows) == list(range(vocab)):
        return model

    idx = torch.tensor(keep_rows, dtype=torch.long)
    emb, head = model.core.embed_layer, model.core.lm_head.head

    new_emb = torch.nn.Embedding(
        len(keep_rows),
        emb.embedding_dim,
        device=emb.weight.device,
        dtype=emb.weight.dtype,
    )
    new_emb.weight.data.copy_(emb.weight.data[idx])

    new_head = torch.nn.Linear(
        head.in_features,
        len(keep_rows),
        bias=head.bias is not None,
        device=head.weight.device,
        dtype=head.weight.dtype,
    )
    new_head.weight.data.copy_(head.weight.data[idx])
    if head.bias is not None:
        new_head.bias.data.copy_(head.bias.data[idx])

    model.core.embed_layer = new_emb
    model.core.lm_head.head = new_head
    model.config.alphabet_size = len(keep_rows)
    if hasattr(model.config, "vocab_size"):
        model.config.vocab_size = len(keep_rows)
    logger.info(
        "Pruned vocabulary %d -> %d rows (kept base [0:%d) + %d added).",
        vocab,
        len(keep_rows),
        base,
        len(keep_rows) - base,
    )
    return model


def _widen_token_embed_dim(
    model: Any, new_dim: int, zero_base_new_dims: bool = False
) -> int:
    """Widen NTv3's token bottleneck in place, function-preserving at init.

    ``token_embed_dim`` has exactly two consumers -- ``core.embed_layer``
    (V x dim) and ``core.stem.conv`` (Conv1d(dim, conv_init_embed_dim, 15)) --
    so widening is a two-tensor Net2Net expansion plus a config field. No code
    fork: the modeling files travel with each checkpoint and read the dim from
    the config.

    Why the halves are initialized differently: zeroing BOTH the new embedding
    dims and the new conv input channels is function-preserving but a dead
    symmetry. d(loss)/d(conv new channels) is proportional to the embedding's
    new dims and vice versa, so both would stay at zero for the whole run and
    the arm would silently train as if never widened. Zeroing exactly one side
    -- the conv, here -- keeps the output identical at init while leaving the
    embedding side with signal for the conv to start learning from.

    ``zero_base_new_dims`` zeroes the new dimensions of the BASE vocab rows only,
    which is what makes ``--freeze-acgt-path`` able to promise a bitwise-identical
    stem output for pure-ACGT input: those rows then contribute nothing through
    the new conv channels no matter what the new channels learn. The new (diploid)
    rows keep their random new dims, so the conv's new channels still receive
    gradient and the dead symmetry is still avoided.

    Returns the OLD dim, which is the boundary ``_freeze_acgt_input_path`` needs.
    """
    old_dim = int(model.config.token_embed_dim)
    if new_dim == old_dim:
        return old_dim
    if new_dim < old_dim:
        raise ValueError(
            f"--token-embed-dim {new_dim} is narrower than the model's {old_dim}; "
            "narrowing would discard trained dimensions"
        )

    emb = model.core.embed_layer
    conv = model.core.stem.conv
    w_emb, w_conv = emb.weight.data, conv.weight.data
    dev, dtype = w_emb.device, w_emb.dtype

    new_emb = torch.nn.Embedding(emb.num_embeddings, new_dim, device=dev, dtype=dtype)
    new_emb.weight.data.normal_(0.0, float(w_emb.std()))
    new_emb.weight.data[:, :old_dim] = w_emb
    if zero_base_new_dims:
        base_vocab = int(getattr(model.config, "base_vocab_size", 11))
        new_emb.weight.data[:base_vocab, old_dim:] = 0.0

    new_conv = torch.nn.Conv1d(
        new_dim,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        padding=conv.padding,
        bias=conv.bias is not None,
        device=dev,
        dtype=dtype,
    )
    new_conv.weight.data.zero_()
    new_conv.weight.data[:, :old_dim, :] = w_conv
    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data)

    model.core.embed_layer = new_emb
    model.core.stem.conv = new_conv

    # token_embed_dim is NOT a two-consumer field on a _post model. It is also the
    # SPECIES-CONDITIONING width (modeling_ntv3_posttrained.py:823-829): every
    # cond_tables.* embedding and every FiLM modulation_layers.* Linear takes it as
    # their input dim. Leaving them at the old width trains fine -- they are built
    # before this call and stay self-consistent in memory -- but the SAVED config
    # then claims the new width, reconstruction builds them wide, and every load
    # fails on a size mismatch. That is how the first acgtsafe milestones became
    # unreadable; pretrain/tools/repair_widened_ckpt.py fixes existing ones.
    #
    # Pad with ZEROS on both sides, which is exactly function-preserving:
    # FiLM out = W_old @ c_old + W_new @ 0. The new block is DELIBERATELY dead --
    # this arm widens the token bottleneck, not the conditioning capacity.
    n_cond = 0
    for name, mod in model.named_modules():
        if "cond_tables" in name and isinstance(mod, torch.nn.Embedding):
            if mod.embedding_dim < new_dim:
                w = mod.weight.data
                mod.weight.data = torch.cat(
                    [w, w.new_zeros(w.shape[0], new_dim - w.shape[1])], dim=1
                )
                mod.embedding_dim = new_dim
                n_cond += 1
        elif "modulation_layers" in name and isinstance(mod, torch.nn.Linear):
            if mod.in_features < new_dim:
                w = mod.weight.data
                mod.weight.data = torch.cat(
                    [w, w.new_zeros(w.shape[0], new_dim - w.shape[1])], dim=1
                )
                mod.in_features = new_dim
                n_cond += 1
    if n_cond:
        logger.info(
            "Widened %d conditioning tensors (cond_tables + modulation_layers) "
            "%d -> %d with zeros. Function-preserving and deliberately dead: the "
            "conditioning capacity is unchanged, this only keeps the saved config "
            "consistent so the checkpoint can be loaded.",
            n_cond,
            old_dim,
            new_dim,
        )
    model.config.token_embed_dim = new_dim
    logger.info(
        "Widened token_embed_dim %d -> %d (embed %s, stem conv %s). Output is "
        "unchanged at init: new conv input channels are zero, new embedding "
        "dims carry signal so they can be recruited.",
        old_dim,
        new_dim,
        tuple(new_emb.weight.shape),
        tuple(new_conv.weight.shape),
    )
    return old_dim


def _filter_unseen_tokens(
    new_tokens: list[str],
    token_freqs: dict[str, float] | None,
) -> tuple[list[str], list[str]]:
    """Split *new_tokens* into ``(kept, dropped)`` by corpus occurrence.

    ``diploid_tokens.tsv`` enumerates every symbol the encoder *can* emit, not
    the ones a given corpus contains: v7 is unphased, so its 28 phased glyphs,
    7 hemizygous forms and the `-`/`*` gap symbols never appear in any of its
    110,887,674 rows. Allocating rows for them costs a permanently-wrong class
    in every MLM softmax and leaves `*`/`-` at cosine 1.0000 forever, since
    nothing ever separates two rows that start equal and never take gradient.

    BUG SURFACE -- read before reusing a filtered checkpoint. The kept set is a
    property of the CORPUS, not of the tokenizer. Run a filtered model on a
    corpus that does contain the dropped symbols (a phased dataset, or v8) and
    they resolve to <unk>, silently destroying exactly the variant positions the
    model exists to read. Two things guard that: `build_char_to_id_table` warns
    for any TSV symbol that lands on unk_id, and the run stamps its own filtered
    TSV so the benchmark-side drift check fails loudly rather than mis-tokenizing.
    `pretrain/tools/verify_dead_tokens.py` is the pre-flight that proves the
    dropped set really is absent before a run starts.

    ``token_freqs is None`` (no counts table) keeps everything: an unreadable
    table means "unknown", never "nothing occurs".
    """
    if token_freqs is None:
        return list(new_tokens), []
    kept = [t for t in new_tokens if token_freqs.get(t, 0.0) > 0.0]
    dropped = [t for t in new_tokens if token_freqs.get(t, 0.0) <= 0.0]
    return kept, dropped


def _dead_tokens_in_vocab(
    vocab: dict[str, int],
    token_freqs: dict[str, float] | None,
    symbols: Iterable[str],
) -> list[str]:
    """Symbols that are IN this tokenizer but occur zero times in the corpus.

    ``--drop-unseen-tokens`` only runs on the fresh vocab-expansion path, so a
    checkpoint built before the filter (or under --no-drop-unseen-tokens) keeps
    its dead rows and ``setup_model_and_tokenizer(resume_from=...)`` inherits
    them silently. This is the resume-side detector; it only WARNS. Pruning on
    resume would change the embedding/LM-head shapes and remap token ids, which
    invalidates the optimizer moments and every cached tokenization -- that is a
    new run, not a resume.

    ``token_freqs is None`` reports nothing: an unreadable table means
    "unknown", never "everything is dead" (same contract as
    ``_filter_unseen_tokens``).
    """
    if not token_freqs:
        return []
    return sorted(
        t for t in symbols if t in vocab and token_freqs.get(t, 0.0) <= 0.0
    )


def _write_filtered_tokens_tsv(
    tokens_path: Path, output_dir: Path, dropped: set[str]
) -> Path:
    """Write the tokens TSV this run actually used, minus *dropped* symbols.

    Kept line-for-line rather than regenerated so a human can diff it against
    the repo TSV, and so its hash is a genuine identifier of the vocabulary in
    the checkpoint.
    """
    out = Path(output_dir) / "diploid_tokens_used.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Comment and short lines pass through untouched so the file stays diffable against the repo TSV.
    out.write_text(filtered_tokens_text(tokens_path, dropped), encoding="utf-8")
    logger.info("Wrote filtered tokens TSV (%d symbols dropped) to %s", len(dropped), out)
    return out


def _convert_tokenizer_to_fast(
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedTokenizerBase:
    """Convert slow tokenizer to fast tokenizer for efficient serialization.

    Creates a PreTrainedTokenizerFast wrapper around the tokenizers.Tokenizer backend,
    preserving all vocabulary (including added/special tokens) and metadata from
    the slow tokenizer. Bit-exact correctness: same token IDs for identical inputs.

    Failure is non-fatal: if the conversion raises (e.g. due to a tokenizers /
    transformers API change), the slow tokenizer is returned with a warning so
    the surrounding save still succeeds — at the cost of slower inference-time
    tokenization downstream.
    """
    try:
        return _build_fast_tokenizer(tokenizer)
    except Exception as exc:  # noqa: BLE001 — best-effort optimization
        logger.warning(
            "Fast-tokenizer conversion failed (%s: %s); falling back to slow "
            "tokenizer. Downstream inference will use the slower path.",
            type(exc).__name__,
            exc,
        )
        return tokenizer


def _build_fast_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedTokenizerFast:
    """Inner builder: see _convert_tokenizer_to_fast for the public contract."""
    from tokenizers import AddedToken, Regex, Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel

    vocab = tokenizer.get_vocab()
    if not vocab:
        raise ValueError("Tokenizer vocabulary is empty")

    unk_token = tokenizer.unk_token or "<unk>"
    backend = Tokenizer(WordLevel(vocab, unk_token=unk_token))
    # NO normalizer. Lowercase a/c/g/t/n are DISTINCT trained tokens (ids 15-19, own
    # embedding rows); uppercase A/C/G/T/N are 6-10. Training tokenized via a codepoint
    # table that never folded case (train._build_char_to_id_table), so case MUST be
    # preserved. An uppercasing normalizer + the normalized=True added-tokens below caused
    # encode("A")->15 (added-token shadowing). Dropping the normalizer AND adding tokens
    # with normalized=False (below) makes encode match the training LUT (A->6, a->15).
    # Char-level split via Rust-native Split. [\s\S] matches any char incl.
    # newlines (Oniguruma doesn't accept (?s)). Unknown chars fall through
    # WordLevel to <unk>, matching slow-tokenizer behavior.
    backend.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(r"[\s\S]"), behavior="isolated"
    )

    # Build kwargs preserving all special token types exactly
    init_kwargs = {
        "tokenizer_object": backend,
        "unk_token": tokenizer.unk_token,
        "pad_token": tokenizer.pad_token,
        "mask_token": tokenizer.mask_token,
        "additional_special_tokens": [
            t
            for t in tokenizer.all_special_tokens
            if t
            not in {
                tokenizer.unk_token,
                tokenizer.pad_token,
                tokenizer.mask_token,
                getattr(tokenizer, "cls_token", None),
                getattr(tokenizer, "eos_token", None),
                getattr(tokenizer, "bos_token", None),
            }
        ]
        or [],
    }

    # Add optional special tokens if they exist and differ from defaults
    cls_tok = getattr(tokenizer, "cls_token", None)
    eos_tok = getattr(tokenizer, "eos_token", None)
    bos_tok = getattr(tokenizer, "bos_token", None)

    if cls_tok is not None:
        init_kwargs["cls_token"] = cls_tok
    if eos_tok is not None:
        init_kwargs["eos_token"] = eos_tok
    if bos_tok is not None:
        init_kwargs["bos_token"] = bos_tok

    fast_tokenizer = PreTrainedTokenizerFast(**init_kwargs)

    # CRITICAL: Re-add all diploid/non-special added tokens to the fast tokenizer.
    # These need to be in added_tokens_encoder for proper save/load in added_tokens.json.
    non_special_added = [
        token
        for token in tokenizer.added_tokens_encoder.keys()
        if str(token) not in tokenizer.all_special_tokens
    ]
    if non_special_added:
        # normalized=False: match raw token content (no case-fold) so encode() agrees
        # with the training codepoint LUT (A->6, a->15).
        fast_tokenizer.add_tokens(
            [AddedToken(str(t), normalized=False) for t in non_special_added]
        )
        logger.debug(
            "Re-added %d non-special tokens to fast tokenizer: %s",
            len(non_special_added),
            non_special_added[:10],  # Log first 10 for debugging
        )

    # `add_tokens()` above already populates `added_tokens_encoder` and the
    # backend metadata. Older code also tried to copy private fields
    # (`_added_tokens_decoder`, `unique_no_split_tokens`) directly; modern
    # transformers makes those read-only properties synced from the backend,
    # so we defensively try and ignore if the attribute is owned.
    for attr in ("_added_tokens_decoder", "unique_no_split_tokens"):
        try:
            setattr(fast_tokenizer, attr, getattr(tokenizer, attr, None))
        except AttributeError:
            pass

    logger.debug(
        "Converted tokenizer: %d base tokens, %d added tokens",
        len(vocab) - len(tokenizer.added_tokens_encoder),
        len(tokenizer.added_tokens_encoder),
    )

    # Self-validate the ACTUAL fast artifact before anyone saves/uses it: its encode() must
    # match the codepoint LUT (built from this same tokenizer). Catches any re-introduction of
    # the added-token-shadowing bug at the source, so a buggy tokenizer.json can never be saved.
    _assert_lut_tokenizer_parity(
        fast_tokenizer, _build_char_to_id_table(fast_tokenizer)
    )
    return fast_tokenizer


# Canonical codepoint LUT + parity guard live in pretrain/tokenizer_lut.py (single source of
# truth, so the bug class can't fork across entrypoints again). Re-exported here under the
# legacy private names so existing callers/tests keep working. Dual import: works both when
# train.py runs as a script (sys.path[0]=pretrain/) and when imported as ``pretrain.train``.
try:
    from pretrain.tokenizer_lut import (
        assert_lut_tokenizer_parity as _assert_lut_tokenizer_parity,
    )
    from pretrain.tokenizer_lut import (
        build_char_to_id_table as _build_char_to_id_table,
    )
    from pretrain.tokenizer_lut import merged_vocab
    from pretrain.v2_input import (
        V2Rewrite,
        filtered_tokens_text,
        row_reference,
        swap_copies,
    )
except ImportError:  # pragma: no cover - script-style invocation from pretrain/
    from tokenizer_lut import (
        assert_lut_tokenizer_parity as _assert_lut_tokenizer_parity,
    )
    from tokenizer_lut import (
        build_char_to_id_table as _build_char_to_id_table,
    )
    from tokenizer_lut import merged_vocab
    from v2_input import (
        V2Rewrite,
        filtered_tokens_text,
        row_reference,
        swap_copies,
    )


@functools.cache
def _fasta(path: str, pid: int) -> Any:
    """One pysam.FastaFile per process (keyed by pid): a handle must not cross into forked dataloader workers."""
    import pysam

    return pysam.FastaFile(path)


# Canonical allele priority (matches preprocess/CLAUDE.md):
# A(0) < C(1) < G(2) < T(3) < N(4) < -(5) < *(6).
_ALLELE_PRIORITY = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "-": 5, "*": 6}

# Default noise scale for new-row init (HF default scale order, was 0.005 —
# collisions had cosine ≳ 0.97 between reciprocal phased symbols; 0.02 brings
# baseline cosine down to ~0.85 for D≈512).
_NEW_ROW_NOISE_SIGMA = 0.03
# Magnitude of the deterministic phase-axis bias as a fraction of ‖E[a]−E[b]‖.
# Guarantees the three colliding symbols at one parent pair (unphased, +phase,
# −phase) are linearly separable regardless of noise draw.
_PHASE_AXIS_DELTA_FRAC = 0.05


def _orthogonal_noise(D: int, parent: torch.Tensor, scale: float) -> torch.Tensor:
    """Gaussian noise of magnitude ``scale * ‖parent‖`` projected off the
    parent direction so the new row doesn't lie on the parent's ray (avoids
    cosine ≈ 1 collisions in single-parent clusters like {Å, Æ, Д} all
    cloned from E[A]).
    """
    noise = torch.randn_like(parent)
    if parent.norm() > 0:
        # project off parent
        coef = (noise @ parent) / (parent @ parent)
        noise = noise - coef * parent
    return noise * (scale * (parent.norm().item() + 1e-8))


def _phased_kind(symbol: str, a: str, b: str, sym2order: dict[str, str]) -> str:
    """Classify a symbol as one of:
    'phased_fwd' : phased het, canonical-ordered parent pair (a < b in priority)
    'phased_rev' : phased het, reversed-ordered parent pair
    'unphased'   : unphased het (or symmetric homozygous)
    'single'     : single-parent (e.g. indel '-' or star '*' against one allele)
    'meta'       : marker / control token (e.g. ^, {, ‹, ›)
    'unknown'    : symbol had no order info in the TSV
    """
    kind = sym2order.get(symbol, "unknown")
    if kind != "phased":
        return kind
    pa = _ALLELE_PRIORITY.get(a.upper(), 99)
    pb = _ALLELE_PRIORITY.get(b.upper(), 99)
    return "phased_fwd" if pa < pb else "phased_rev"


@torch.no_grad()
def _init_new_rows_collision_aware(
    weight: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    sym2pair: dict[str, tuple[str, str]],
    sym2order: dict[str, str],
    base_vocab_size: int,
    label: str,
) -> int:
    """Initialize rows ``weight[base_vocab_size:]`` from parent nucleotide
    statistics, with deterministic separation of reciprocal phased symbols.

    Used for BOTH input embeddings and untied LM-head weights — anything that
    derives a row from a parent-pair mean must apply this same hardening or
    the structural mean-collisions (e.g. Á / Ç at (A,C); Ã / Ğ at (A,G);
    etc.) defeat the diploid representation.

    Returns the number of rows initialised.
    """
    nuc_ids = {
        nuc: tokenizer.get_vocab()[nuc]
        for nuc in "ACGTN"
        if nuc in tokenizer.get_vocab()
    }
    initialised = 0
    skipped: list[str] = []

    for tok_obj, tid in tokenizer.added_tokens_encoder.items():
        tok = str(tok_obj)
        if tid < base_vocab_size:
            continue

        D = weight.shape[1]
        pair = sym2pair.get(tok)
        if pair is not None:
            a, b = pair
            aid, bid = nuc_ids.get(a.upper()), nuc_ids.get(b.upper())
            if aid is not None and bid is not None:
                mean = (weight[aid] + weight[bid]) / 2.0
                axis = weight[aid] - weight[bid]
                kind = _phased_kind(tok, a, b, sym2order)
                # Deterministic phase-axis bias: reciprocal phased symbols
                # get +/-δ along (E[a]-E[b]); unphased stays at the mean.
                if kind == "phased_fwd":
                    bias = _PHASE_AXIS_DELTA_FRAC * axis
                elif kind == "phased_rev":
                    bias = -_PHASE_AXIS_DELTA_FRAC * axis
                else:
                    bias = torch.zeros_like(axis)
                # Fixed-sigma noise — keeps the noise small relative to the
                # phase-axis bias even when ‖E[a]−E[b]‖ varies.
                noise = (
                    torch.randn(D, device=weight.device, dtype=weight.dtype)
                    * _NEW_ROW_NOISE_SIGMA
                )
                weight[tid] = mean + bias + noise
                initialised += 1
                continue
            known = aid if aid is not None else bid
            if known is not None:
                # Single-parent (indel/star) cluster — multiple symbols clone
                # the same parent (Å, Æ, Д all clone E[A]). Use orthogonal
                # noise so they don't lie on the parent's ray.
                weight[tid] = weight[known] + _orthogonal_noise(
                    D, weight[known], scale=_NEW_ROW_NOISE_SIGMA
                )
                initialised += 1
                continue
        else:
            upper_id = nuc_ids.get(tok.upper())
            if upper_id is not None:
                # Lowercase single-allele tokens (a/c/g/t) clone uppercase.
                weight[tid] = weight[upper_id] + _orthogonal_noise(
                    D, weight[upper_id], scale=_NEW_ROW_NOISE_SIGMA
                )
                initialised += 1
                continue

        skipped.append(f"{tok!r} (U+{ord(tok) if len(tok) == 1 else 0:04X})")

    if skipped:
        logger.warning(
            "%s: %d new tokens had no parent match and kept HF random init: %s",
            label,
            len(skipped),
            ", ".join(skipped[:10]) + ("..." if len(skipped) > 10 else ""),
        )
    return initialised


@torch.no_grad()
def _init_new_input_rows_v2(
    weight: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    sym2pair: dict[str, tuple[str, str]],
    base_vocab_size: int,
    mode: str,
) -> None:
    """Start of the new INPUT rows for the v2 input (``--new-row-init``).

    ``random``  : every new row is a random direction at the mean norm of the ACGT rows.
    ``between`` : a row for (copy 0 allele, copy 1 allele) starts at 2/3 of the copy-0 base row
                  plus 1/3 of the copy-1 base row, plus small noise. So A/G and G/A start on
                  opposite sides of the midpoint, and a homozygous-alternate row starts at its
                  base. A gap allele has no base row; it stands at the mean of ACGT. Tokens
                  with no allele pair (markers) get a random row.

    Not ``_init_new_rows_collision_aware``: that puts both phased glyphs of a pair on the SAME
    side of the midpoint (its axis flips with the pair, and so does its sign) and both gap
    glyphs of a base at that base. Draws from its own generator, so every DDP rank builds
    identical rows whatever the global RNG has consumed.
    """
    if mode not in ("random", "between"):
        raise ValueError(f"--new-row-init must be random or between here, got {mode!r}")
    acgt = weight[tokenizer.convert_tokens_to_ids(list("ACGT"))]  # indexed: a copy
    base = dict(zip("ACGT", acgt))
    gap, size = acgt.mean(dim=0), acgt.norm(dim=1).mean()
    gen = torch.Generator().manual_seed(0)
    for tok, tid in tokenizer.added_tokens_encoder.items():
        if tid < base_vocab_size:
            continue
        noise = torch.randn(weight.shape[1], generator=gen).to(weight)
        pair = sym2pair.get(str(tok))
        if mode == "random" or pair is None:
            weight[tid] = noise * (size / noise.norm())
        else:
            copy0, copy1 = (base.get(allele, gap) for allele in pair)
            weight[tid] = (2.0 * copy0 + copy1) / 3.0 + _NEW_ROW_NOISE_SIGMA * noise


def _install_new_row_grad_masks(params: list[torch.Tensor], boundary: int) -> list[Any]:
    """Zero the gradient of rows ``[:boundary]`` on each param (PEFT-style
    warmup: only newly-added vocab rows train, base rows stay bitwise-identical).

    Applies to any per-vocab-row tensor — embedding weight, head weight, AND
    head bias (the bias was previously left unmasked, letting base-vocab
    priors drift through warmup).
    """

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        grad[:boundary].zero_()
        return grad

    frozen = [i for i, p in enumerate(params) if not p.requires_grad]
    if frozen:
        # Silently skipping would leave base rows trainable while the caller
        # logs a mask count that says otherwise — fail loud instead.
        raise RuntimeError(
            f"_install_new_row_grad_masks: params {frozen} have requires_grad=False; "
            "they cannot be masked and would train unmasked or not at all."
        )
    return [p.register_hook(_hook) for p in params]


# A run shorter than this cannot amortize a large dataloader-worker pool's fork +
# prefetch cost. 5000 steps is ~18 genome-equivalents here -- comfortably above any
# diagnostic probe and far below any real arm.
WORKER_AMORTIZE_STEPS = 5000
AMORTIZE_MIN_WORKERS = 4


def _amortized_worker_count(requested: int, max_steps: int) -> int:
    """Cap dataloader workers on runs too short to amortize their startup.

    Workers are forked AFTER the window index is materialised, so each one
    copy-on-write faults a multi-GB array and prefetches before step 1. Measured
    2026-08-19: 16 workers x 2 ranks cost ~18 min of startup on a 200-step probe
    whose training then took ~4 min; the same 16 workers are free on a 55k-step
    arm. This is a startup/amortization question, not CPU oversubscription (256
    cores, load ~60 at the time).

    The knob stays a CEILING: an explicit low request is never raised, and runs
    with no step limit (max_steps <= 0, i.e. every real arm) are never touched.

    ponytail: caps workers instead of fixing the fork. The deeper fix is to
    mmap-back the window index in `_window_index_from_table` so worker pages are
    file-backed and shared rather than COW -- that would remove the cost at every
    run length, but it touches the hot data path for a cost measured at zero on
    every arm actually trained here. Do it only if short probes become routine.
    """
    if requested <= AMORTIZE_MIN_WORKERS or max_steps <= 0:
        return requested
    if max_steps >= WORKER_AMORTIZE_STEPS:
        return requested
    return min(requested, AMORTIZE_MIN_WORKERS)


def _vocab_embedding_modules(model: Any) -> list[Any]:
    """The modules holding per-vocab-row parameters (input embedding, LM head).

    One definition, because two things must agree about this set or the
    base-row freeze silently breaks: `_rowmaskable_vocab_params` installs the
    gradient masks here, and `EmbeddingsNoWDOptimizerMixin.create_optimizer`
    must put the same tensors in the weight-decay-FREE group -- AdamW decays a
    parameter whose gradient was masked to zero, so a masked row in a decaying
    group still drifts. If a future model exposes vocab rows through a third
    accessor, both sites gain it at once.
    """
    out = []
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        module = getattr(model, getter, lambda: None)()
        if module is not None:
            out.append(module)
    return out


def _mark_nodecay(model: Any, tensors: list[torch.Tensor]) -> None:
    """Exempt these tensors from weight decay.

    A GRADIENT MASK IS NOT A FREEZE under AdamW. Decoupled weight decay is applied
    straight to the parameter in the optimizer step and never passes through the
    gradient, so a `register_hook` that zeroes part of a gradient still lets that
    part shrink by `(1 - lr*wd)` every step. Measured on the acgtsafe arm, whose
    stem-conv channel mask was supposed to be bitwise: ||W[:, :16, :]|| ratio
    0.993849 at genome-50 vs a (1-lr*wd)^13879 prediction of 0.993085 -- a
    four-decimal match, and the arm's "bitwise invariant" guarantee was false by
    4e-2 in the ACGT stem output.

    `_vocab_embedding_modules` already handles the vocab-row case; this is the
    same rule for any other partially-frozen tensor. Register here whenever you
    mask part of a tensor's gradient.
    """
    ids = getattr(model, "_extra_nodecay_param_ids", None)
    if ids is None:
        ids = set()
        model._extra_nodecay_param_ids = ids
    ids.update(id(t) for t in tensors)


def _rowmaskable_vocab_params(model: Any, base_vocab_size: int) -> list[torch.Tensor]:
    """Trainable per-vocab-row tensors reachable from the model's embeddings.

    Used to re-protect base rows after PEFT replaces the embedding / LM head with
    its own trainable deep-copies. Selects by SHAPE (leading dim >= vocab size)
    rather than by parameter name, because the name depends on which wrapper PEFT
    installed and that has changed across versions.
    """
    out: list[torch.Tensor] = []
    seen: set[int] = set()
    for module in _vocab_embedding_modules(model):
        # recurse=True on purpose: under PEFT the live tensor sits at
        # `modules_to_save.default.weight`, not on the wrapper itself. The
        # requires_grad filter below keeps the frozen `original_module.weight`
        # copy out.
        for tensor in module.parameters():
            if (
                tensor.requires_grad
                and tensor.shape
                and tensor.shape[0] >= base_vocab_size
                and id(tensor) not in seen
            ):
                seen.add(id(tensor))
                out.append(tensor)
    return out


# The RoPE-facing projections. NTv3 has NO learnable positional parameter --
# `sa_layer.rotary_embedding.inv_freq` is a frequency basis of shape 32 (head_dim/2)
# and is bitwise identical across every checkpoint measured (19 of them, 4 arms,
# out to genome-200 = 55,516 steps). Phase is computed on the fly as
# `position * inv_freq`, so there is no position-indexed table to freeze or decay.
# The learnable half of positional-attention behaviour is therefore the projections
# the rotation is applied to: query and key. Freezing them pins the attention
# geometry while value/output/FFN/conv/embedding stay free.
_QK_NEEDLES = ("query_head", "key_head")



def _freeze_acgt_input_path(model: Any, base_vocab_size: int, old_dim: int) -> None:
    """Pin everything a pure-ACGT sequence touches on the way into the trunk.

    Five arms established that ACGT-task damage happens iff the INPUT
    representation of pure-ACGT sequence changes -- base embedding rows or stem
    conv -- while downstream movement is harmless (txfreeze moved its transformer
    MORE than bundle, 0.01717 vs 0.01672, and scored -0.0011 on gue_splice).
    Freezing that path outright also blocks the diploid learning, because the new
    tokens enter through it: txfreeze ended at diploid_mlm 2.454 against bundle's
    0.935.

    Widening breaks the tie. After ``--token-embed-dim 32`` with
    ``zero_base_new_dims``, the base rows are zero in dims [16:32), so freezing

      * embedding rows [0:base_vocab)  (all dims), and
      * stem.conv.weight[:, :old_dim, :], and
      * stem.conv.bias  (it is added regardless of input)

    leaves the pure-ACGT stem output BITWISE identical for the whole run, while
    the diploid rows and the new conv channels [old_dim:] stay free. The
    guarantee is structural, not measured.

    Gradient masks rather than requires_grad because these are slices of tensors
    whose other half must keep training -- the same technique
    ``_install_new_row_grad_masks`` uses for the vocab rows.
    """
    if int(model.config.token_embed_dim) <= old_dim:
        raise SystemExit(
            "--freeze-acgt-path needs a WIDENED token embedding: pass "
            f"--token-embed-dim > {old_dim}. Without a free half this would "
            "freeze the entire input path, which is the txfreeze arm (no damage, "
            "but no diploid learning either)."
        )

    emb_w = model.core.embed_layer.weight
    conv = model.core.stem.conv

    def _mask_rows(grad: torch.Tensor) -> torch.Tensor:
        grad[:base_vocab_size].zero_()
        return grad

    def _mask_old_channels(grad: torch.Tensor) -> torch.Tensor:
        grad[:, :old_dim, :].zero_()
        return grad

    emb_w.register_hook(_mask_rows)
    conv.weight.register_hook(_mask_old_channels)
    if conv.bias is not None:
        conv.bias.requires_grad = False
    # Without this the masked halves still decay -- the hook only stops gradients.
    _mark_nodecay(model, [emb_w, conv.weight])

    logger.info(
        "--freeze-acgt-path: embedding rows [0:%d) pinned, stem conv input "
        "channels [0:%d) pinned, stem conv bias frozen. Pure-ACGT stem output is "
        "now bitwise-invariant for the run; diploid rows and conv channels "
        "[%d:%d) remain trainable.",
        base_vocab_size,
        old_dim,
        old_dim,
        int(model.config.token_embed_dim),
    )


def _freeze_qk(model) -> int:
    """Freeze every query/key projection parameter. Returns how many it froze.

    Caveat worth stating on any result: Q/K carry CONTENT-based attention too, so
    this is "attention pattern frozen", not "positional frozen" in isolation. It
    is the closest available intervention on a RoPE model, not a clean positional
    knob.

    Under LoRA the adapter tensors also match these names and are frozen too,
    which is the intended reading ("Q/K do not adapt"), but it makes the flag
    redundant with simply dropping them from --lora-target.
    """
    frozen = 0
    names: list[str] = []
    for name, p_ in model.named_parameters():
        if any(n in name for n in _QK_NEEDLES):
            p_.requires_grad = False
            frozen += 1
            names.append(name)
    if frozen:
        logger.info(
            "--freeze-qk: froze %d query/key projection tensors (%s ... %s); "
            "RoPE itself has no parameters, so this pins the attention geometry.",
            frozen,
            names[0],
            names[-1],
        )
    else:
        logger.warning(
            "--freeze-qk matched NOTHING (%s). The model's attention modules are "
            "not named as expected -- the flag is silently doing nothing.",
            "/".join(_QK_NEEDLES),
        )
    return frozen


def _apply_main_trainable(model, spec: str, base_vocab_size: int) -> int:
    """Freeze everything except the components named in ``spec`` (comma list).

    Generalises the embedding warmup's freeze to main training: 'new_rows'
    re-enables the input embedding + LM head and installs the same gradient-row
    masks so base-vocab rows stay bitwise-identical; 'transformer' / 'conv' /
    'deconv' re-enable the corresponding tower. The Trainer builds its optimizer
    from requires_grad only and DDP syncs only trainable grads, so frozen arms
    also train faster and resume cleanly (same flag -> same param set).

    Returns the trainable-parameter count (also logged, for provenance).
    """
    # A component may carry a block RANGE: "transformer:3-5" trains only blocks
    # 3,4,5 and freezes 0,1,2. Without this the flag is all-or-nothing per tower,
    # so "freeze the lower half" -- the one point between txfreeze (whole path
    # frozen: no damage, no learning) and bundle (all free: damage + learning) --
    # cannot be expressed.
    known = {"new_rows", "transformer", "conv", "deconv"}
    parts: set[str] = set()
    ranges: dict[str, tuple[int, int]] = {}
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        name, _, span = tok.partition(":")
        parts.add(name)
        if not span:
            continue
        if name == "new_rows":
            raise SystemExit(
                "--main-trainable: 'new_rows' takes no block range (it is vocab "
                "rows, not a tower)"
            )
        try:
            lo, hi = (int(x) for x in span.split("-", 1))
        except ValueError:
            raise SystemExit(
                f"--main-trainable: bad block range {tok!r}; expected e.g. "
                "'transformer:3-5'"
            ) from None
        ranges[name] = (lo, hi)
    if parts - known:
        raise SystemExit(
            f"--main-trainable: unknown component(s) {sorted(parts - known)}; "
            f"known: {sorted(known)}"
        )
    for p_ in model.parameters():
        p_.requires_grad = False
    def _span(name: str, blocks: Any) -> list[Any]:
        """The blocks of this tower that `spec` selects."""
        if name not in ranges:
            return list(blocks)
        lo, hi = ranges[name]
        if not (0 <= lo <= hi < len(blocks)):
            raise SystemExit(
                f"--main-trainable: block range {lo}-{hi} out of range for "
                f"{name!r}, which has {len(blocks)} blocks (valid 0-{len(blocks) - 1})"
            )
        return [blocks[i] for i in range(lo, hi + 1)]

    selected: list[torch.nn.Module] = []
    if "transformer" in parts:
        selected += _span("transformer", model.core.transformer_blocks)
    if "conv" in parts:
        # The stem is the INPUT path, not a tower block. A bare `conv` includes it
        # (unchanged behaviour); a range means "these tower blocks", and the stem
        # stays frozen -- it has its own flag (--freeze-acgt-path).
        if "conv" not in ranges:
            selected.append(model.core.stem)
        selected += _span("conv", model.core.conv_tower_blocks)
    if "deconv" in parts:
        selected += _span("deconv", model.core.deconv_tower_blocks)
    for mod in selected:
        for p_ in mod.parameters():
            p_.requires_grad = True
    if "new_rows" in parts:
        in_w = model.get_input_embeddings().weight
        out_embs = model.get_output_embeddings()
        out_w = out_embs.weight
        in_w.requires_grad = True
        out_w.requires_grad = True
        masked = [in_w] + ([out_w] if out_w is not in_w else [])
        out_b = getattr(out_embs, "bias", None)
        if out_b is not None and out_b.shape[0] == out_w.shape[0]:
            out_b.requires_grad = True
            masked.append(out_b)
        _install_new_row_grad_masks(masked, base_vocab_size)
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    n_total = sum(p_.numel() for p_ in model.parameters())
    logger.info(
        "--main-trainable=%s: %d / %d params trainable (%.2f%%)%s.",
        spec,
        n_train,
        n_total,
        100.0 * n_train / n_total,
        " with new-row grad masks (base rows bitwise-frozen)"
        if "new_rows" in parts
        else "",
    )
    return n_train


@torch.no_grad()
def _apply_new_token_bias_prior(
    bias: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    sym2pair: dict[str, tuple[str, str]],
    base_vocab_size: int,
    token_freqs: dict[str, float] | None,
    fallback_offset: float,
) -> None:
    """Add a rarity prior ``ln(freq_tok / freq_anchor)`` to every new head-bias
    entry (Karpathy "init the final bias to the base rate").

    Without this, new rows keep parent-scale (~25%-frequency) priors — or 0
    for markers — while their true unigram frequency is 1e-4..1e-6, leaking
    order-1 softmax mass to the new block at step 0 (hockey-stick loss; the
    embedding warmup then burns its budget suppressing the artifact). The
    anchor is the parent-pair mean frequency when known, else the mean ACGT
    frequency (markers). Tokens the table can't resolve (or no table) get the
    constant ``fallback_offset`` — the sibling repo's measured basin is flat
    over [-6, -12], so precision beyond order-of-magnitude doesn't matter.
    Offsets are clamped to [-14, 0]: never boost a new token, never underflow
    bf16 softmax.
    """
    acgt_freqs = (
        [token_freqs[t] for t in "ACGT" if token_freqs.get(t, 0.0) > 0.0]
        if token_freqs
        else []
    )
    acgt_mean = sum(acgt_freqs) / len(acgt_freqs) if acgt_freqs else None
    applied: list[float] = []
    for tok_obj, tid in tokenizer.added_tokens_encoder.items():
        tok = str(tok_obj)
        if tid < base_vocab_size or tid >= bias.shape[0]:
            continue
        offset = fallback_offset
        if token_freqs and token_freqs.get(tok, 0.0) > 0.0:
            pair = sym2pair.get(tok)
            parent_freqs = (
                [
                    token_freqs[p.upper()]
                    for p in pair
                    if token_freqs.get(p.upper(), 0.0) > 0.0
                ]
                if pair
                else []
            )
            anchor = (
                sum(parent_freqs) / len(parent_freqs) if parent_freqs else acgt_mean
            )
            if anchor:
                offset = math.log(token_freqs[tok] / anchor)
        offset = max(-14.0, min(0.0, offset))
        bias[tid] += offset
        applied.append(offset)
    if applied:
        logger.info(
            "Applied ln-frequency bias priors to %d new head rows "
            "(offset min/max = %.2f / %.2f; fallback=%.1f for tokens absent "
            "from the counts table).",
            len(applied),
            min(applied),
            max(applied),
            fallback_offset,
        )


@torch.no_grad()
def _init_diploid_embeddings(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    sym2pair: dict[str, tuple[str, str]],
    sym2order: dict[str, str],
    base_vocab_size: int,
    token_freqs: dict[str, float] | None = None,
    new_token_bias_offset: float = -10.0,
    new_row_init: str = "legacy",
) -> None:
    """Initialize new input-token AND LM-head rows from parent nucleotides.

    Applies the same collision-aware initializer to both matrices when the
    output head is untied (NTv3-pre case). The head must be hardened too —
    leaving it at fresh-resize random init was the cause of the
    "head-init-noise spills into all head rows during warmup" pathology.
    """
    emb_weight = model.get_input_embeddings().weight.data
    if new_row_init != "legacy":
        # v2 input rows; the new LM-head rows start at the ACGT mean (below).
        _init_new_input_rows_v2(emb_weight, tokenizer, sym2pair, base_vocab_size, new_row_init)
        logger.info("Initialised new INPUT embedding rows with --new-row-init %s.", new_row_init)
    else:
        in_count = _init_new_rows_collision_aware(
            emb_weight,
            tokenizer,
            sym2pair,
            sym2order,
            base_vocab_size,
            label="input_embeddings",
        )
        logger.info(
            "Initialised %d/%d new INPUT embedding rows from parent pairs.",
            in_count,
            emb_weight.shape[0] - base_vocab_size,
        )

    out_embs = model.get_output_embeddings()
    if out_embs is None:
        logger.info("No output embeddings module; skipping LM-head init.")
        return

    if out_embs is model.get_input_embeddings():
        logger.info(
            "Output embeddings are tied to input embeddings; LM head shares the init."
        )
        return

    out_weight = getattr(out_embs, "weight", None)
    if out_weight is None:
        logger.warning("Output embeddings has no .weight; skipping LM-head init.")
        return

    # Untied head: shapes can differ (NTv3 has a head with a width != input_dim
    # in some configs). For NTv3, this is the LIVE path on every run:
    # token_embed_dim=16, conv_init_embed_dim ∈ {256,768,1024+}. Parent-pair
    # averaging across dimensions is impossible — but we MUST still match the
    # trained-row scale so new tokens aren't 7-13× under-magnitude vs existing
    # rows (HF resize default uses normal_(0, initializer_range≈0.02); NTv3
    # has no initializer_range so falls back to 0.02; NTv3 trained head std is
    # ~0.07-0.27 per measurement).
    out_bias = getattr(out_embs, "bias", None)
    if out_weight.shape != emb_weight.shape:
        new_count = out_weight.shape[0] - base_vocab_size
        if new_count > 0 and new_row_init != "legacy":
            # v2: the mean of the ACGT head rows, no noise. The legacy mean below runs over
            # all 11 base rows, six of them special-token rows the pretrained head holds
            # ~250 logits under the bases, so new tokens started ~140 nats under ACGT
            # (measured) and the ln-frequency bias prior (-7 to -14) never mattered. Head
            # rows need no tie-break: each takes its own gradient when its token is the label.
            acgt = tokenizer.convert_tokens_to_ids(list("ACGT"))
            out_weight.data[base_vocab_size:] = out_weight.data[acgt].mean(dim=0)
            logger.info("Initialised %d new LM-head rows at the mean of the ACGT head rows.", new_count)
        elif new_count > 0:
            with torch.no_grad():
                existing = out_weight.data[:base_vocab_size]
                # HF/LLaMA-style vocab-expansion init: anchor new rows at the
                # per-column mean of existing head rows (the "average head
                # direction") plus a small Gaussian perturbation to break ties.
                # This keeps new tokens from systematically dominating softmax
                # at init while giving them a learnable starting point.
                # Naive N(0, sigma_base) overshoots: random high-magnitude
                # rows in random directions happen to align with the hidden
                # state for specific inputs and explode the logits.
                mu = existing.mean(dim=0)  # shape: (out_dim,)
                sigma = float(existing.std().item())
                noise = torch.randn(
                    new_count,
                    mu.shape[0],
                    device=out_weight.device,
                    dtype=out_weight.dtype,
                )
                out_weight.data[base_vocab_size:] = (
                    mu.unsqueeze(0) + 0.03 * sigma * noise
                )
            logger.info(
                "LM head shape %s != input embeddings shape %s; "
                "re-initialized %d new head rows as mu_existing + N(0, 0.03·%.4f) "
                "(HF/LLaMA-style vocab-expansion init; replaces HF resize default ~0.02 "
                "which is dimensionally fine but loses the per-column mean structure of the "
                "trained head). Parent-pair averaging skipped (dimension mismatch).",
                tuple(out_weight.shape),
                tuple(emb_weight.shape),
                new_count,
                sigma,
            )
        # Bias init still works — bias dim matches out_weight ROWS (vocab), not cols.
        if out_bias is not None and out_bias.shape[0] == out_weight.shape[0]:
            bias_2d = out_bias.data.unsqueeze(1)  # treat as (V, 1)
            _init_new_rows_collision_aware(
                bias_2d,
                tokenizer,
                sym2pair,
                sym2order,
                base_vocab_size,
                label="lm_head_bias",
            )
            _apply_new_token_bias_prior(
                out_bias.data,
                tokenizer,
                sym2pair,
                base_vocab_size,
                token_freqs,
                new_token_bias_offset,
            )
        return

    out_count = _init_new_rows_collision_aware(
        out_weight.data,
        tokenizer,
        sym2pair,
        sym2order,
        base_vocab_size,
        label="lm_head",
    )
    logger.info(
        "Initialised %d/%d new LM-HEAD rows from parent pairs.",
        out_count,
        out_weight.shape[0] - base_vocab_size,
    )

    # If the head has a bias term, init new bias entries from parent-pair
    # bias means too — same collision-avoidance rationale at lower dimension.
    if out_bias is not None and out_bias.shape[0] == out_weight.shape[0]:
        bias_2d = out_bias.data.unsqueeze(1)  # treat as (V, 1)
        _init_new_rows_collision_aware(
            bias_2d,
            tokenizer,
            sym2pair,
            sym2order,
            base_vocab_size,
            label="lm_head_bias",
        )
        _apply_new_token_bias_prior(
            out_bias.data,
            tokenizer,
            sym2pair,
            base_vocab_size,
            token_freqs,
            new_token_bias_offset,
        )


@torch.no_grad()
def _log_init_scale_check(
    model: Any,
    base_vocab_size: int,
    token_freqs: dict[str, float] | None = None,
) -> None:
    """Karpathy step-0 sanity check: new-token rows must match trained-row scale.

    HF's resize_token_embeddings uses N(0, config.initializer_range≈0.03) for new
    rows, but NTv3 trained rows can have std up to ~0.27 — a 13x scale mismatch
    that effectively zeros new tokens at start. Logs a ratio and warns if off.
    """

    def _row_norms(
        weight: torch.Tensor, start: int, end: int | None = None
    ) -> torch.Tensor:
        sl = weight[start:end] if end is not None else weight[start:]
        return sl.detach().float().norm(dim=1)

    in_w = model.get_input_embeddings().weight
    base_in = _row_norms(in_w, 0, base_vocab_size)
    new_in = _row_norms(in_w, base_vocab_size)
    base_in_mean = float(base_in.mean()) if base_in.numel() else float("nan")
    new_in_mean = float(new_in.mean()) if new_in.numel() else float("nan")
    ratio_in = new_in_mean / max(base_in_mean, 1e-12)

    out_embs = model.get_output_embeddings()
    out_w = getattr(out_embs, "weight", None) if out_embs is not None else None
    ratio_out: float | None = None
    if out_w is not None and out_w is not in_w:
        base_out = _row_norms(out_w, 0, base_vocab_size)
        new_out = _row_norms(out_w, base_vocab_size)
        ratio_out = (
            float(new_out.mean()) / max(float(base_out.mean()), 1e-12)
            if new_out.numel() and base_out.numel()
            else None
        )

    vocab_size = in_w.shape[0]
    expected_init_loss = float(np.log(max(vocab_size, 1)))
    if ratio_out is not None:
        logger.info(
            "[step-0 sanity] input new/base row-norm ratio=%.3f, "
            "head new/base row-norm ratio=%.3f, expected init loss ~ %.2f (-log(1/%d))",
            ratio_in,
            ratio_out,
            expected_init_loss,
            vocab_size,
        )
    else:
        logger.info(
            "[step-0 sanity] input new/base row-norm ratio=%.3f, "
            "head tied to input, expected init loss ~ %.2f (-log(1/%d))",
            ratio_in,
            expected_init_loss,
            vocab_size,
        )
    if token_freqs:
        # ln(V) assumes a uniform predictor; the honest reference for a
        # frequency-aware init is the unigram entropy (Karpathy "verify loss @
        # init": a correctly-primed head should start near H, not ln(V), and
        # anything below H early is the model reading context, not priors).
        ps = [p for p in token_freqs.values() if p > 0]
        entropy = float(-sum(p * np.log(p) for p in ps))
        logger.info(
            "[step-0 sanity] unigram entropy H(freqs) = %.3f nats — the honest "
            "loss floor for a prior-only predictor (vs ln(V) = %.3f).",
            entropy,
            expected_init_loss,
        )

    for name, ratio in (("input", ratio_in), ("head", ratio_out)):
        if ratio is not None and not (0.5 <= ratio <= 2.0):
            logger.warning(
                "[step-0 sanity] %s embedding new/base row-norm ratio=%.3f "
                "is OUTSIDE [0.5, 2.0]; new tokens may dominate (high) or "
                "vanish (low) until gradient steps rescale them.",
                name,
                ratio,
            )


def _maybe_wrap_lora(
    model: Any, args: argparse.Namespace, base_vocab_size: int | None = None
) -> Any:
    """Wrap *model* with LoRA adapters when --peft-main=lora.

    Coverage: Linear, Embedding, Conv1d, Conv2d, Conv3d, MultiheadAttention
    (PEFT 0.18+). The 7 modules PEFT cannot wrap are ConvTranspose1d, and they
    are the deconv blocks' RESIDUAL/SKIP branches (`res_conv.conv_block.conv`),
    NOT the deconv tower itself: this config is `deconv_upsample_type:
    repeat+conv`, so each block's MAIN path is a plain Conv1d and IS adapted
    (verified in saved weights — main path moved 0.037 at genome-20, res_conv
    exactly 0.000). So the tower is adapted with a frozen skip branch, not
    "frozen"; do not build analysis on an encoder-adapted/decoder-frozen story.
    encoder Conv1d modules ARE LoRA-wrapped so the model can adapt the
    genomic "tokenizer" stage to human-specific motif statistics during
    continued pretraining.

    Embedding rows and the LM head are kept FULL-RANK trainable via
    ``modules_to_save`` so new-vocab learning continues at full capacity
    (standard vocab-expansion recipe, Cui 2023 LLaMA-Chinese).

    Suffix resolution is type-aware: --lora-target accepts simple module
    suffixes ("conv", "fc1", "linear"); we pre-filter to wrappable types
    before passing to LoraConfig as a full-path list. This prevents PEFT
    from crashing on a ConvTranspose1d/ModuleDict that happens to suffix-
    match.
    """
    if getattr(args, "peft_main", "none") != "lora":
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise SystemExit(
            "--peft-main=lora requires the `peft` package. "
            "Install with `uv pip install peft` and re-run."
        ) from exc

    target_suffixes = [t.strip() for t in args.lora_target.split(",") if t.strip()]
    modules_to_save = ["embed_layer", "lm_head.head"]

    # PEFT 0.18+ supported wrappable layer types for LoRA. ConvTranspose1d is
    # NOT here, which in this config skips only the deconv blocks' residual/skip
    # branches; their main Conv1d path is wrapped. See _maybe_wrap_lora.
    from torch import nn as _nn

    SUPPORTED = (_nn.Linear, _nn.Embedding, _nn.Conv1d, _nn.Conv2d, _nn.Conv3d)

    def _matches_suffix(name: str) -> bool:
        return name in target_suffixes or any(
            name.endswith(f".{t}") or name == t for t in target_suffixes
        )

    # Resolve to FULL module paths so PEFT does exact matching (avoids the
    # ModuleDict / LinearProjectionHeInit suffix collisions seen before).
    resolved: list[str] = []
    skipped_unsupported: list[tuple[str, str]] = []
    skipped_modules_to_save: list[str] = []
    for name, mod in model.named_modules():
        if not _matches_suffix(name):
            continue
        if any(name.endswith(s) or name == s for s in modules_to_save):
            skipped_modules_to_save.append(name)
            continue
        if isinstance(mod, SUPPORTED):
            resolved.append(name)
        else:
            skipped_unsupported.append((name, type(mod).__name__))

    if not resolved:
        listing = [
            f"{type(m).__name__:18s} {n}"
            for n, m in model.named_modules()
            if isinstance(m, SUPPORTED)
        ][:30]
        raise SystemExit(
            f"--peft-main=lora but target_modules={target_suffixes} resolved to 0 "
            f"wrappable modules. PEFT-supported types in this model (first 30):\n  "
            + "\n  ".join(listing)
        )

    # Group resolved counts by type for the log.
    type_counts: dict[str, int] = {}
    type_params: dict[str, int] = {}
    for name in resolved:
        mod = model.get_submodule(name)
        t = type(mod).__name__
        type_counts[t] = type_counts.get(t, 0) + 1
        type_params[t] = type_params.get(t, 0) + sum(
            p.numel() for p in mod.parameters()
        )

    lora_alpha = int(args.lora_alpha) if args.lora_alpha is not None else int(args.lora_r)
    config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=lora_alpha,
        target_modules=resolved,
        lora_dropout=float(args.lora_dropout),
        bias="none",
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, config)

    # PEFT's ModulesToSaveWrapper deep-copies each `modules_to_save` module and
    # calls requires_grad_(True) on the WHOLE tensor, so without this the base
    # ACGT vocab rows train -- at the LoRA arm's higher LR -- while every other
    # path in this file holds them bitwise frozen. The copy is a NEW tensor, so
    # any hook installed before get_peft_model is gone; re-install here.
    if base_vocab_size and not getattr(args, "lora_protect_base_rows", True):
        # Deliberate: an arm asking whether letting the base embedding geometry
        # reorganize is itself the damage mechanism needs those rows free. Loud,
        # because it voids the invariant every other path in this file upholds.
        logger.warning(
            "LoRA: base vocab rows [0:%d) are TRAINABLE (--no-lora-protect-base-rows). "
            "This arm is NOT comparable to any arm that pins them; verify with "
            "pretrain/tools/verify_freeze.py, which should report them MOVED.",
            base_vocab_size,
        )
    elif base_vocab_size:
        masked = _rowmaskable_vocab_params(model, base_vocab_size)
        if masked:
            _install_new_row_grad_masks(masked, base_vocab_size)
            logger.info(
                "LoRA: re-installed new-row gradient masks on %d modules_to_save "
                "tensor(s); base vocab rows [0:%d) stay bitwise frozen.",
                len(masked),
                base_vocab_size,
            )
        else:
            logger.warning(
                "LoRA: found no per-vocab-row tensors to mask; base vocab rows "
                "may train. Check modules_to_save=%s.",
                modules_to_save,
            )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    logger.info(
        # `skipped_modules_to_save` counts modules excluded from LoRA TARGETING
        # because they are already full-rank in modules_to_save -- it is not the
        # number kept full-rank, which is len(modules_to_save) and is always on.
        # Reporting the former as the latter read as "embeddings are frozen".
        "LoRA wrap: r=%d alpha=%d dropout=%.2f. Resolved %d modules from %d "
        "suffixes; %d unsupported types skipped (ConvTranspose1d — the deconv "
        "blocks' residual/skip branches; their main Conv1d path IS adapted); "
        "%d kept full-rank (%s); %d excluded from LoRA targeting "
        "for already being full-rank.",
        config.r,
        config.lora_alpha,
        config.lora_dropout,
        len(resolved),
        len(target_suffixes),
        len(skipped_unsupported),
        len(modules_to_save),
        ",".join(modules_to_save),
        len(skipped_modules_to_save),
    )
    for t, cnt in sorted(type_counts.items()):
        logger.info(
            "  LoRA-wrapped %-18s : %d modules, %d underlying params",
            t,
            cnt,
            type_params[t],
        )
    if skipped_unsupported:
        kinds = sorted({k for _, k in skipped_unsupported})
        logger.info(
            "  Skipped (unsupported types): %d modules of kinds %s",
            len(skipped_unsupported),
            kinds,
        )
    logger.info(
        "  Trainable params (LoRA + modules_to_save): %d / %d (%.2f%%).",
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )
    return model


def _bottleneck_len(config: Any, window_len: int) -> int:
    """Transformer sequence length for a `window_len` input, per the model's own config.

    Args:
        config: The backbone's config (or None for a stock 7-downsample tower).
        window_len: Input length in base pairs.

    Returns:
        ``window_len // 2 ** (num_downsamples - pool_bypass)``, at least 1.

    Pool-bypass surgery removes `pool_bypass` pooling rounds, so the transformer runs
    at window/(128>>n) rather than the hardcoded window/PAD_MULTIPLE.
    """
    downsamples = int(getattr(config, "num_downsamples", 7)) - int(
        getattr(config, "pool_bypass", 0) or 0
    )
    return max(1, int(window_len) // (1 << max(0, downsamples)))


def _prewarm_rotary_caches_for_compile(backbone: Any, args: argparse.Namespace) -> int:
    """Pre-initialize NTv3 rotary caches on the target CUDA device before compile.

    Warming to the unbypassed PAD_MULTIPLE stride leaves `_ensure_cache` re-registering
    a buffer inside the compiled graph on step 1, hence `_bottleneck_len`.
    """
    parameters = getattr(backbone, "parameters", None)
    if parameters is None:
        return 0
    try:
        first_param = next(parameters())
    except StopIteration:
        return 0
    if first_param.device.type != "cuda":
        return 0
    seq_len = _bottleneck_len(
        getattr(backbone, "config", None), getattr(args, "window_len", PAD_MULTIPLE)
    )
    warmed = 0
    for module in backbone.modules():
        ensure_cache = getattr(module, "_ensure_cache", None)
        if (
            callable(ensure_cache)
            and hasattr(module, "cos_cached")
            and hasattr(module, "sin_cached")
        ):
            ensure_cache(seq_len, first_param.device, torch.float32)
            warmed += 1
    return warmed


def _maybe_compile_backbone_forward(model: Any, args: argparse.Namespace) -> Any:
    """Optionally compile only the expensive NTv3 backbone forward.

    Compiling the whole AuxHeadsModel/HF Trainer surface captures data-dependent
    aux/CPL/SMP glue and can produce graph breaks or checkpoint serialization
    wrinkles. Replacing only ``backbone.forward`` keeps the module object and
    state_dict unchanged while allowing Inductor to optimize the conv/U-Net +
    transformer tower used by both the main and CPL twin passes.
    """
    if (
        args.no_compile
        or args.quick_test
        or not getattr(args, "compile_backbone", False)
    ):
        return model
    backbone = getattr(model, "backbone", model)
    if getattr(backbone, "_diploid_glm_forward_compiled", False):
        return model
    warmed_rotary = _prewarm_rotary_caches_for_compile(backbone, args)
    if warmed_rotary:
        logger.info("Prewarmed %d rotary caches before torch.compile.", warmed_rotary)
    backbone.forward = torch.compile(
        backbone.forward,
        mode=args.torch_compile_mode,
        dynamic=bool(getattr(args, "compile_dynamic", True)),
    )
    backbone._diploid_glm_forward_compiled = True
    logger.info(
        "torch.compile backbone.forward enabled: mode=%s dynamic=%s target=%s",
        args.torch_compile_mode,
        bool(getattr(args, "compile_dynamic", True)),
        backbone.__class__.__name__,
    )
    return model


def _patch_post_track_heads_for_pretraining(model: Any) -> bool:
    """Skip unused NTv3 POST BigWig/BED heads without dropping their weights."""
    core = getattr(model, "core", None)
    if core is None or getattr(core, "_diploid_glm_skip_post_track_heads", False):
        return False
    if not (hasattr(core, "bigwig_head") or hasattr(core, "bed_head")):
        return False

    parent_forward = None
    for cls in core.__class__.mro()[1:]:
        if cls.__name__ == "DiscreteConditionedNTv3PreTrainedCore":
            parent_forward = cls.forward
            break
    if parent_forward is None:
        logger.warning(
            "Could not find DiscreteConditionedNTv3PreTrainedCore; POST track heads remain active."
        )
        return False

    def _forward_without_tracks(
        self,
        input_ids: torch.LongTensor | None = None,
        species_ids: torch.Tensor | None = None,
        output_track: bool = False,
        inputs_embeds: torch.FloatTensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        del output_track
        assert (input_ids is None) != (inputs_embeds is None), (
            "You must specify exactly one of input_ids or inputs_embeds"
        )
        batch_size = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )  # type: ignore[union-attr]
        assert species_ids is not None, "species_ids must be provided"
        assert species_ids.shape == (batch_size,), (
            f"species_ids must be shape (B,), got {species_ids.shape}"
        )
        return parent_forward(
            self,
            input_ids=input_ids,
            condition_ids=[species_ids],
            conditions_masks=None,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

    core.forward = types.MethodType(_forward_without_tracks, core)
    core._diploid_glm_skip_post_track_heads = True
    logger.info(
        "Patched NTv3 POST core to skip BigWig/BED heads during MLM pretraining."
    )
    return True


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return latest checkpoint-* directory from *output_dir*, or None."""
    if not output_dir.exists() or not output_dir.is_dir():
        return None
    latest: tuple[int, Path] | None = None
    for ckpt in output_dir.glob("checkpoint-*"):
        if not (ckpt / "trainer_state.json").exists():
            continue
        try:
            step = int(ckpt.name.split("-")[-1])
            if latest is None or step > latest[0]:
                latest = (step, ckpt)
        except ValueError:
            continue
    return latest[1] if latest is not None else None


def _configured_world_size() -> int:
    """Return the effective world size even before torch.distributed.init_process_group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    env_world_size = os.environ.get("WORLD_SIZE")
    if env_world_size and env_world_size.isdigit():
        return max(1, int(env_world_size))
    return 1


# -- Data Loading -------------------------------------------------------------


def _with_cadd_sidecar(shard: datasets.Dataset, sidecar: Path) -> datasets.Dataset:
    """``shard`` with its ``cadd_phred`` column replaced by the rebuilt one in ``sidecar``
    (preprocess/huggingface_dataset/tools/rebuild_cadd.py writes one per shard, row for row)."""
    if not sidecar.exists():
        raise FileNotFoundError(f"--cadd-sidecar: no rebuilt CADD column {sidecar}")
    with pa.memory_map(str(sidecar)) as src:  # the column stays memory-mapped after the block closes
        tbl = pa.ipc.open_file(src).read_all()
    made_for = orjson.loads((tbl.schema.metadata or {}).get(b"rebuild_cadd", b"{}")).get("shard")
    if made_for != sidecar.name or tbl.num_rows != len(shard):
        raise ValueError(f"--cadd-sidecar: {sidecar} ({tbl.num_rows} rows, made for {made_for}) does not match its "
                         f"shard ({len(shard)} rows)")
    return shard.remove_columns("cadd_phred").add_column(
        "cadd_phred", tbl.column("cadd_phred"), new_fingerprint=f"{shard._fingerprint}_cadd_sidecar")


def load_parquet_dataset(
    data_dir: Path,
    max_shards: int | None = None,
    cache_dir: Path | None = None,
    cadd_dir: Path | None = None,
) -> datasets.Dataset:
    parquet_files = [
        p for p in sorted(data_dir.glob("*.parquet")) if not p.name.endswith(".tmp")
    ]
    arrow_files = [
        p
        for p in sorted(data_dir.glob("*.arrow"))
        if not p.name.endswith(".tmp") and not p.name.startswith("cache-")
    ]

    if parquet_files:
        files = parquet_files
        dataset_format = "parquet"
    elif arrow_files:
        files = arrow_files
        dataset_format = "arrow"
    else:
        raise FileNotFoundError(
            f"No valid .parquet or .arrow shards found in {data_dir}"
        )

    if max_shards is not None:
        files = files[:max_shards]

    if dataset_format == "arrow":
        # Read Arrow IPC files directly — bypasses HF cache entirely, no disk writes.
        # datasets.load_dataset("arrow", ...) would re-serialize 83M+ records to
        # ~/.cache/huggingface/datasets/ and easily exhaust home-partition space.
        shards = [datasets.Dataset.from_file(str(f)) for f in files]
        if cadd_dir is not None:
            shards = [_with_cadd_sidecar(sh, Path(cadd_dir) / f.name) for sh, f in zip(shards, files)]
            logger.info("cadd_phred read from the rebuilt columns in %s (%d shards)", cadd_dir, len(shards))
        ds = datasets.concatenate_datasets(shards) if len(shards) > 1 else shards[0]
    else:
        if cadd_dir is not None:
            raise ValueError("--cadd-sidecar needs an arrow dataset (one rebuilt column per .arrow shard)")
        ds = datasets.load_dataset(
            dataset_format,
            data_files=[str(p) for p in files],
            split="train",
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )

    stable_cache_fingerprint = hashlib.md5(
        "|".join(
            [
                dataset_format,
                str(data_dir.resolve()),
                *(f"{p.name}:{p.stat().st_size}" for p in files),
            ]
        ).encode()
    ).hexdigest()[:16]
    ds._diploid_cache_fingerprint = stable_cache_fingerprint

    logger.info(
        "Loaded %d records (%d %s shards) from %s",
        len(ds),
        len(files),
        dataset_format,
        data_dir,
    )
    return ds


def check_cadd_sidecar_keep_list(train_hf: datasets.Dataset, cadd_dir: Path, keep_rows: Path | None) -> None:
    """A rebuilt CADD column covers the rows of the keep-list it was built for; every other row keeps its old
    (rounded, averaged) score. Refuse a run whose keep-list keeps a different number of rows."""
    first = sorted(Path(cadd_dir).glob("*.arrow"))[0]
    with pa.memory_map(str(first)) as src:
        built_for = orjson.loads(pa.ipc.open_file(src).schema.metadata[b"rebuild_cadd"])["keep_rows"]
    mask, _ = row_keep_mask(train_hf, keep_rows)
    kept = len(train_hf) if mask is None else int(mask.sum())
    if built_for != "all" and built_for != kept:
        raise SystemExit(f"--cadd-sidecar {cadd_dir} was rebuilt for a keep-list of {built_for} rows; this run's "
                         f"keep-list ({keep_rows}) keeps {kept}: rows outside the sidecar's keep-list hold old scores")


def row_keep_mask(
    hf_dataset: datasets.Dataset, keep_rows_path: Path | str | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Per-record ``(mask, weight)`` from a saved row keeplist (semantic dedup).

    The dedup products are index lists over the concatenated train split in
    sorted-shard order -- the exact order ``load_parquet_dataset`` builds -- so
    they are positional, not content-addressed. Applied to a corpus with a
    different row count they would select the WRONG rows with no error, which is
    why an out-of-range index is fatal rather than clipped.

    Returns a per-record mask, not a subset: callers project it onto the window
    index (``mask[w_idx.record_idx]``) exactly as ``--max-genomes`` does. Going
    through ``Dataset.select`` instead would set ``_indices`` and drop
    ``_load_hf_dataset`` off its fast path.

    A ``.npy`` index list is a mask only (``weight`` None: survivors keep their own
    ``sample_weight``). The ``semantic_dedup.py keeplist`` ``.parquet`` also yields
    per-record ``cluster_weight`` -- a survivor's own weight plus everything its
    cluster's dropped rows stood for -- which replaces ``sample_weight`` so dedup
    removes rows, not individuals (the corpus's own "summed on dedup" rule).
    """
    if keep_rows_path is None:
        return None, None
    path = Path(keep_rows_path)
    weight = None
    if path.suffix == ".parquet":
        kept = pq.read_table(
            path,
            columns=["global_row", "cluster_weight"],
            filters=[("keep", "==", True)],
        )
        idx = kept.column("global_row").to_numpy()
        weight = kept.column("cluster_weight").to_numpy()
    else:
        idx = np.load(path, mmap_mode="r")
    if idx.size == 0:
        raise ValueError(f"--keep-rows {path} is empty; that would train on nothing")
    lo, hi = int(idx.min()), int(idx.max())
    n = len(hf_dataset)
    if lo < 0 or hi >= n:
        raise ValueError(
            f"keeplist {path} has indices out of range for this dataset: "
            f"[{lo}, {hi}] vs {n} records. A keeplist is positional into the "
            f"corpus it was built on -- rebuild it for this one."
        )
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(idx)] = True
    logger.info(
        "Row keeplist %s: kept %d/%d records (%.1f%%).",
        path.name,
        int(mask.sum()),
        n,
        100.0 * mask.sum() / n,
    )
    if weight is not None:
        rec_weight = np.zeros(n, dtype=np.float32)
        rec_weight[idx] = weight
        logger.info(
            "Row keeplist %s: survivors carry cluster_weight, %d individuals in all.",
            path.name,
            int(weight.sum(dtype=np.float64)),  # float32 sums drift past 2**24
        )
        weight = rec_weight
    return mask, weight


def _sid_isin(hf_dataset: datasets.Dataset, ids) -> np.ndarray:
    """Per-record bool: is the record's sample_id in `ids`? Pyarrow-side, no
    Python materialization of 100M+ strings; `zero_copy_only=False` because a
    ChunkedArray of bools has no zero-copy numpy view."""
    return (
        pc.is_in(hf_dataset.data.column("sample_id"), value_set=pa.array(sorted(ids)))
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )


def genome_keep_mask(
    hf_dataset: datasets.Dataset,
    max_genomes: int,
    seed: int,
    output_dir: Path | None = None,
) -> np.ndarray | None:
    """Per-record bool mask keeping a seeded subset of non-REF genomes + HG38_REF.

    Operates on the pyarrow column directly (no Python materialization of 100M+
    strings). Returns None when the subset would keep every genome.

    Args:
        hf_dataset: Corpus with a ``sample_id`` column.
        max_genomes: How many non-REF genomes to keep.
        seed: Subset seed, decoupled from the training seed.
        output_dir: When given, rank 0 records the kept ids there as
            ``genome_subset_seed{seed}.json`` -- the exact input a later phase
            passes to ``--exclude-genomes`` to train on the complement.
    """
    sid_col = hf_dataset.data.column("sample_id")
    uniq = sorted(
        s for s in pc.unique(sid_col.combine_chunks()).to_pylist() if s != REF_SAMPLE_ID
    )
    if max_genomes >= len(uniq):
        logger.info(
            "--max-genomes %d >= %d available non-REF genomes; keeping all.",
            max_genomes,
            len(uniq),
        )
        return None
    pick = np.random.default_rng(seed).choice(
        len(uniq), size=max_genomes, replace=False
    )
    keep_ids = [uniq[i] for i in np.sort(pick)] + [REF_SAMPLE_ID]
    mask = _sid_isin(hf_dataset, keep_ids)
    if output_dir is not None and os.environ.get("RANK", "0") == "0":
        output_dir.mkdir(parents=True, exist_ok=True)
        subset_json = output_dir / f"genome_subset_seed{seed}.json"
        subset_json.write_bytes(
            orjson.dumps(
                {"seed": seed, "n": len(keep_ids), "sample_ids": keep_ids},
                option=orjson.OPT_INDENT_2,
            )
        )
        logger.info("Wrote %s (--exclude-genomes input for a later phase)", subset_json)
    logger.info(
        "Genome subset (seed=%d): kept %d/%d non-REF genomes (+%s), %d/%d records.",
        seed,
        max_genomes,
        len(uniq),
        REF_SAMPLE_ID,
        int(mask.sum()),
        len(mask),
    )
    return mask


def genome_exclude_mask(hf_dataset: datasets.Dataset, exclude_json: Path) -> np.ndarray:
    """Per-record bool mask DROPPING the sample_ids listed in a subset JSON.

    The JSON is a prior run's ``genome_subset_seed*.json`` (``{"sample_ids":
    [...]}``, written by `genome_keep_mask`) -- phase-2 trains on the complement
    of what phase-1 saw. HG38_REF is never excluded even when listed: dropping it
    would leave ``is_ref`` all-False and silently degrade the REF oversampler to
    uniform.

    Raises:
        ValueError: The JSON is not a genome subset, lists nothing but HG38_REF,
            names no genome present in this corpus, or would leave no non-REF
            genome at all. Every check is at GENOME level: HG38_REF survives by
            construction, so a record count can never reach zero and a REF-only
            residue would otherwise train on one genome without complaint.
    """
    data = orjson.loads(Path(exclude_json).read_bytes())
    listed = data.get("sample_ids") if isinstance(data, dict) else None
    if listed is None:
        found = f"keys {sorted(data)}" if isinstance(data, dict) else type(data).__name__
        raise ValueError(
            f'--exclude-genomes {exclude_json}: expected {{"sample_ids": [...]}} '
            f"(a prior run's genome_subset_seed*.json), got {found}"
        )
    exclude_ids = set(listed) - {REF_SAMPLE_ID}
    if not exclude_ids:
        raise ValueError(
            f"--exclude-genomes {exclude_json}: no non-REF sample_ids to exclude"
        )
    present = set(
        pc.unique(hf_dataset.data.column("sample_id").combine_chunks()).to_pylist()
    )
    dropped = exclude_ids & present
    if not dropped:
        raise ValueError(
            f"--exclude-genomes {exclude_json}: none of the {len(exclude_ids)} "
            "listed sample_ids occur in this corpus -- wrong corpus or wrong "
            "subset JSON? (excluding nothing would silently retrain on the "
            "phase-1 genomes)"
        )
    if not present - dropped - {REF_SAMPLE_ID}:
        raise ValueError(
            f"--exclude-genomes {exclude_json} would leave no non-{REF_SAMPLE_ID} "
            f"genomes: all {len(dropped)} genomes in this corpus are listed."
        )
    mask = ~_sid_isin(hf_dataset, dropped)
    logger.info(
        "--exclude-genomes: dropped %d genomes (%s always retained), kept %d/%d records.",
        len(dropped),
        REF_SAMPLE_ID,
        int(mask.sum()),
        len(mask),
    )
    return mask


@dataclass
class WindowIndex:
    record_idx: np.ndarray  # int32: which record
    window_start: np.ndarray  # int32: character offset within seq
    has_variant: np.ndarray  # bool: window contains ^ or {
    is_ref: np.ndarray | None = None  # bool: True if window comes from HG38_REF
    sample_weight: np.ndarray | None = None  # float32: per-record weight, per window
    win_len: np.ndarray | None = None  # int32: real (unpadded) bases in the window


def _window_index_from_table(table: pa.Table) -> WindowIndex:
    """Build a WindowIndex from a mapped/ cached pyarrow table (zero-copy).

    ``is_ref`` and ``sample_weight`` are optional for backward-compat: caches
    built before those columns existed simply omit them (→ None, treated as
    non-ref / weight 1.0 downstream).
    """

    def col(name: str) -> np.ndarray:
        return table.column(name).combine_chunks().to_numpy(zero_copy_only=False)

    def opt_col(name: str) -> np.ndarray | None:
        return col(name) if name in table.column_names else None

    return WindowIndex(
        record_idx=col("record_idx"),
        window_start=col("window_start"),
        has_variant=col("has_variant"),
        is_ref=opt_col("is_ref"),
        sample_weight=opt_col("sample_weight"),
        win_len=opt_col("win_len"),
    )


@dataclass(frozen=True)
class GenomeMilestonePlan:
    n_genomes: int
    total_nonref_windows: int
    windows_per_genome: float
    steps_per_genome: float
    milestone_steps: dict[int, list[int]]


def _map_windows(
    batch: dict[str, list[Any]],
    indices: list[int],
    window_len: int,
    stride: int,
) -> dict[str, np.ndarray]:
    """Map a batch of records to flat window index arrays."""
    snp_ord, indel_ord = ord(SNP_MARKER), ord(INDEL_OPEN)
    recs, wins, vars_, refs, weights, lens = [], [], [], [], [], []
    sample_ids = batch.get("sample_id") or [None] * len(indices)
    # Per-record sample_weight (=1 normally, =10 for HG38_REF, summed on dedup);
    # absent on v5/v6 → default 1.0. Broadcast per record to all its windows.
    sample_weights = batch.get("sample_weight") or [1] * len(indices)
    for global_i, seq, sid, sw in zip(
        indices, batch["seq"], sample_ids, sample_weights
    ):
        seq_len = len(seq)

        starts = np.arange(0, max(1, seq_len - window_len + 1), stride, dtype=np.int32)
        # End-aligned final window. Without it the tail of any record whose length
        # is not (k*stride + window_len) is never seen: at window_len 131072 a 200 kb
        # record yields arange(0, 68929, 117965) == [0] -- ONE window, 69 kb (34.5% of
        # the record) never trained on. The 114k-467k band is 41.37% of all v7-tsstes
        # bases, so this is not marginal. No-op for records shorter than a window (they
        # take the max(1, ...) single-window branch) and for exactly-tiled records.
        if seq_len > window_len and int(starts[-1]) + window_len < seq_len:
            starts = np.append(starts, seq_len - window_len).astype(np.int32)
        # Real (unpadded) base count per window, for length-bucketed batching. Jitter
        # in __getitem__ cannot change it: jitter is bounded by the window's own slack,
        # which is > 0 only for windows that are already full length.
        win_lens = np.minimum(window_len, seq_len - starts).astype(np.int32)
        cp = np.frombuffer(seq.encode("utf-32-le"), dtype=np.uint32)
        marker_pos = np.where((cp == snp_ord) | (cp == indel_ord))[0]

        if len(marker_pos) == 0:
            has_var = np.zeros(len(starts), dtype=bool)
        else:
            left = np.searchsorted(marker_pos, starts, side="left")
            has_var = (left < len(marker_pos)) & (
                marker_pos[np.minimum(left, len(marker_pos) - 1)] < starts + window_len
            )

        recs.append(np.full(len(starts), global_i, dtype=np.int32))
        wins.append(starts)
        vars_.append(has_var)
        refs.append(np.full(len(starts), sid == REF_SAMPLE_ID, dtype=bool))
        weights.append(np.full(len(starts), float(sw), dtype=np.float32))
        lens.append(win_lens)

    return {
        "record_idx": np.concatenate(recs) if recs else np.array([], dtype=np.int32),
        "window_start": np.concatenate(wins) if wins else np.array([], dtype=np.int32),
        "has_variant": np.concatenate(vars_) if vars_ else np.array([], dtype=bool),
        "is_ref": np.concatenate(refs) if refs else np.array([], dtype=bool),
        "sample_weight": (
            np.concatenate(weights) if weights else np.array([], dtype=np.float32)
        ),
        "win_len": np.concatenate(lens) if lens else np.array([], dtype=np.int32),
    }


def build_window_index(
    hf_dataset: datasets.Dataset,
    window_len: int,
    stride: int,
    cache_dir: Path | None = None,
    max_workers: int | None = None,
) -> WindowIndex:
    """Build a flat index of (record, window_start).

    Uses vectorized marker detection and HuggingFace dataset mapping for speed.
    """
    n_rec = len(hf_dataset)
    t0 = time.monotonic()

    # Generate a unique cache name derived from dataset fingerprint to allow relocating
    # the cache to a writable directory (e.g. if the original data dir is read-only).
    fingerprint = getattr(
        hf_dataset,
        "_diploid_cache_fingerprint",
        getattr(hf_dataset, "_fingerprint", "nofingerprint"),
    )
    # Schema token marks the index layout; bumping it forces a rebuild so a stale
    # cache can't silently feed a newer reader. "sw" added sample_weight; "wl" adds
    # win_len AND the end-aligned tail window, which changes the window COUNT, not
    # just the columns -- a stale "sw" cache would be short by one window on every
    # record with an uncovered tail.
    cache_key = hashlib.md5(
        f"{fingerprint}_{n_rec}_{window_len}_{stride}_sw_wl".encode()
    ).hexdigest()[:12]

    if max_workers is not None:
        n_workers = max(1, int(max_workers))
    else:
        cpu_count = os.cpu_count() or 1
        n_workers = min(max(1, cpu_count - 2), 64)
    n_workers = min(n_workers, max(1, n_rec))

    map_kwargs = {}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_stem = f"index_cache_{cache_key}"
        expected_shards = [
            cache_dir / f"{cache_stem}_{rank:05d}_of_{n_workers:05d}.arrow"
            for rank in range(n_workers)
        ]
        if all(path.exists() for path in expected_shards):
            logger.info(
                "Loading cached window index for %d records from %s (%d shards)",
                n_rec,
                cache_dir,
                n_workers,
            )
            mapped_ds = datasets.concatenate_datasets(
                [datasets.Dataset.from_file(str(path)) for path in expected_shards]
            )
            table = mapped_ds.data
            idx = _window_index_from_table(table)
            logger.info(
                "Index loaded from cache in %.1fs: %d variant / %d total windows",
                time.monotonic() - t0,
                idx.has_variant.sum(),
                len(idx.record_idx),
            )
            return idx
        # map() will append _{rank}_of_{num_proc} to this base name automatically
        map_kwargs["cache_file_name"] = str(
            cache_dir / f"index_cache_{cache_key}.arrow"
        )

    logger.info("Building window index for %d records using HF map...", n_rec)

    # Only the columns _map_windows reads: map pickles each worker's dataset, and an in-memory column (a
    # --cadd-sidecar) would otherwise be pickled whole into every worker.
    slim = hf_dataset.select_columns([c for c in ("seq", "sample_id", "sample_weight") if c in hf_dataset.column_names])
    mapped_ds = slim.map(
        _map_windows,
        batched=True,
        with_indices=True,
        num_proc=n_workers,
        remove_columns=slim.column_names,
        fn_kwargs={"window_len": window_len, "stride": stride},
        desc="Building Window Index",
        **map_kwargs,
    )

    # Extract arrays instantly using PyArrow zero-copy (avoiding slow python dict/list loop)
    table = mapped_ds.data
    idx = _window_index_from_table(table)

    logger.info(
        "Index built in %.1fs: %d variant / %d total windows",
        time.monotonic() - t0,
        idx.has_variant.sum(),
        len(idx.record_idx),
    )

    return idx


def filter_window_index(
    idx: WindowIndex,
    mode: str = "variant_enriched",
    nonvariant_keep_frac: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    n = len(idx.record_idx)
    if mode == "all":
        return np.arange(n, dtype=np.int64)
    elif mode == "variant_only":
        return np.where(idx.has_variant)[0].astype(np.int64)
    elif mode == "variant_only_no_ref":
        # Backward-compatible alias for canonical datasets where HG38_REF has no variants.
        return np.where(idx.has_variant)[0].astype(np.int64)
    elif mode == "variant_enriched":
        v_idx = np.where(idx.has_variant)[0]
        # HG38_REF windows have has_variant=False by design but are essential
        # reference signal; keep all of them. Subsample only the non-REF
        # non-variant windows.
        if idx.is_ref is not None:
            ref_idx = np.where(idx.is_ref)[0]
            nv_nonref_idx = np.where((~idx.has_variant) & (~idx.is_ref))[0]
        else:
            ref_idx = np.array([], dtype=np.int64)
            nv_nonref_idx = np.where(~idx.has_variant)[0]
        rng = np.random.default_rng(seed)
        n_keep = max(1, int(len(nv_nonref_idx) * nonvariant_keep_frac))
        kept_nv = (
            rng.choice(nv_nonref_idx, size=n_keep, replace=False)
            if n_keep < len(nv_nonref_idx)
            else nv_nonref_idx
        )
        combined = np.concatenate([v_idx, ref_idx, kept_nv])
        combined.sort()
        return combined.astype(np.int64)
    raise ValueError(f"Unknown mode: {mode}")


def plan_genome_milestones(
    hf_dataset: datasets.Dataset,
    idx: WindowIndex,
    filter_indices: np.ndarray,
    milestone_genomes: list[int],
    batch_size: int,
    grad_accum: int,
    estimated_total_steps: int,
    ref_sampler_weight: float = 1.0,
    genome_steps: float | None = None,
    og2_fraction: float = 0.0,
    windows_per_step: float | None = None,
) -> GenomeMilestonePlan:
    """Approximate genome-equivalent checkpoints from actual train-window counts.

    ``windows_per_step``: under ``--token-budget`` rows/batch is NOT ``batch_size``
    (that default, 128, gave the tsstes run "this corpus would have given 6.24"
    against a true ~142). The caller passes bases-per-step / mean bucket length.

    ``og2_fraction``: OG2 replay REPLACES that fraction of optimizer steps with
    non-human batches, so N human-genome-equivalents take N*steps/(1-og2) steps.
    Without this term genome-N under 1/8 replay held 7/8 N genomes of human data
    (phase-2 650M documented that as a convention; the tsstes launcher claimed the
    opposite). Scaling here makes the label mean the same thing at every og2.

    ``genome_steps`` pins optimizer-steps-per-genome instead of deriving it from
    this dataset. Without it, genome-N is a fraction of whatever corpus is
    mounted: a 20%-smaller (deduplicated) corpus over the same 2,560 genomes
    shrinks ``windows_per_genome`` by 20%, so genome-50 fires 20% early and the
    row silently stops being comparable to every other row in RESULTS.md. Pin it
    to an arm's measured value (the scored ladder ran at 277.58) to make genome-N
    an absolute optimization budget across datasets and batch sizes.

    When ``ref_sampler_weight > 1.0``, the ``WeightedRandomSampler`` over-draws
    HG38_REF windows by this factor. At the same training step the trainer has
    seen fewer non-REF windows than the uniform-sampling estimate would
    suggest, so the milestone-step calculation must be adjusted upward.

    Concretely: with base REF share ``p0 = n_ref / n_total``, sampler share
    becomes ``p_samp = p0·w / (1 + p0·(w-1))`` and the non-REF fraction of
    each batch is ``(1 - p_samp)``. We scale ``windows_per_step`` by that
    factor so genome-N milestones fire at the step where ~N non-REF genomes
    have actually been sampled, making cross-arm comparison honest.
    """
    if len(filter_indices) == 0:
        return GenomeMilestonePlan(0, 0, 0.0, 0.0, {})

    selected_records = idx.record_idx[filter_indices]
    record_window_counts = np.bincount(selected_records, minlength=len(hf_dataset))

    # Use PyArrow compute to aggregate without materializing 329M strings into Python
    sample_id_col = hf_dataset.data.column("sample_id")
    wc_arr = pa.array(record_window_counts, type=pa.int64())
    mask = pc.and_(
        pc.greater(wc_arr, 0),
        pc.not_equal(sample_id_col, REF_SAMPLE_ID),
    )
    tbl = pa.table({"sid": sample_id_col, "wc": wc_arr}).filter(mask)
    agg = tbl.group_by("sid").aggregate([("wc", "sum")])

    n_genomes = agg.num_rows
    if n_genomes == 0:
        return GenomeMilestonePlan(0, 0, 0.0, 0.0, {})

    total_nonref_windows = int(pc.sum(agg.column("wc_sum")).as_py())
    windows_per_genome = total_nonref_windows / n_genomes
    world_size = _configured_world_size()
    raw_windows_per_step = max(
        1.0,
        float(windows_per_step)
        if windows_per_step
        else float(batch_size * grad_accum * world_size),
    )

    # Correct for sampler-induced REF over-draw.
    nonref_windows_per_step = raw_windows_per_step
    if ref_sampler_weight > 1.0 and idx.is_ref is not None:
        n_ref = int(idx.is_ref[filter_indices].sum())
        n_total = len(filter_indices)
        if n_total > 0:
            p0 = n_ref / n_total
            p_samp = (
                p0
                * ref_sampler_weight
                / max(1.0 + p0 * (ref_sampler_weight - 1.0), 1e-12)
            )
            nonref_windows_per_step = max(1, raw_windows_per_step * (1.0 - p_samp))
            logger.info(
                "Genome-milestone REF correction: base REF share=%.4f, "
                "sampler weight=%.1f -> effective REF share=%.4f, "
                "non-REF windows/step %.1f -> %.1f.",
                p0,
                ref_sampler_weight,
                p_samp,
                raw_windows_per_step,
                nonref_windows_per_step,
            )

    steps_per_genome = windows_per_genome / nonref_windows_per_step
    if genome_steps is not None and genome_steps > 0:
        logger.info(
            "Genome pacing PINNED to %.2f steps/genome-equivalent (this corpus "
            "would have given %.2f). genome-N is an absolute budget in v7-individual "
            "units: 1 pinned genome = %.2f of THIS corpus's %d genomes, so genome-N = "
            "N x %.2f of them.",
            genome_steps,
            steps_per_genome,
            float(genome_steps) / steps_per_genome,
            n_genomes,
            float(genome_steps) / steps_per_genome,
        )
        steps_per_genome = float(genome_steps)
    if 0.0 < og2_fraction < 1.0:
        steps_per_genome /= 1.0 - og2_fraction
        logger.info(
            "OG2 replay %.3f: genome-N milestones stretched to %.2f steps/genome so "
            "N counts HUMAN data only.",
            og2_fraction,
            steps_per_genome,
        )

    milestone_steps: dict[int, list[int]] = {}
    for g in sorted(set(milestone_genomes)):
        if g < 1 or g > n_genomes:
            continue
        step = max(1, int(np.ceil(g * steps_per_genome)))
        if step > estimated_total_steps:
            continue
        milestone_steps.setdefault(step, []).append(g)

    return GenomeMilestonePlan(
        n_genomes=n_genomes,
        total_nonref_windows=total_nonref_windows,
        windows_per_genome=windows_per_genome,
        steps_per_genome=steps_per_genome,
        milestone_steps=milestone_steps,
    )


class DiploidWindowDataset(Dataset):
    """Tokenized windows from HF Parquet records via pre-computed WindowIndex."""

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        window_index: WindowIndex,
        filter_indices: np.ndarray,
        char_to_id: np.ndarray,
        pad_id: int,
        window_len: int,
        aux_position_cols: tuple[str, ...] = (),
        emit_genomic: bool = False,
        window_jitter: int = 0,
        bucket_step: int = 0,
        record_weight: np.ndarray | None = None,
        v2_rewrite: V2Rewrite | None = None,
        reference_fasta: str | Path | None = None,
        copy_swap: bool = False,
    ):
        self.hf_dataset = hf_dataset
        # v2 input (--v2-input): rewrite the window's characters before tokenising.
        self._v2_rewrite = v2_rewrite
        # Reference-aware tokens (--v2-reference): each row's HG38 bases come from v2_input.row_reference.
        self._reference_fasta = reference_fasta
        if reference_fasta is not None and v2_rewrite is None:
            raise ValueError("a reference FASTA needs the v2 rewrite")
        self._copy_swap = copy_swap   # --v2-copy-swap
        # Track-B/C need each window's genomic start (chrom + 0-based coord of the
        # window's first token). Off by default so plain runs pay nothing.
        self.emit_genomic = bool(emit_genomic)
        self._marker_ords = np.array([ord(m) for m in MARKER_TOKENS], dtype=np.uint32)
        # Pre-slice to only the windows we need (saves ~60% memory vs storing full index)
        self._rec_idx = window_index.record_idx[filter_indices]
        self._win_start = window_index.window_start[filter_indices]
        # Pre-sliced is_ref flag per surviving window (None when the upstream
        # index didn't track it — backwards-compatible with cached pre-is_ref
        # window indices).
        if window_index.is_ref is not None:
            self._is_ref: np.ndarray | None = window_index.is_ref[filter_indices]
        else:
            self._is_ref = None
        # Per-window sample_weight (None on legacy indices → treated as 1.0). A
        # per-record record_weight (a dedup keep-list's cluster_weight) replaces it.
        self._sample_weight: np.ndarray | None
        if record_weight is not None:
            self._sample_weight = record_weight[self._rec_idx]
        elif window_index.sample_weight is not None:
            self._sample_weight = window_index.sample_weight[filter_indices]
        else:
            self._sample_weight = None
        self.filter_indices = filter_indices  # kept for plan_genome_milestones
        self.window_len = window_len
        self.pad_id = pad_id
        self._char_to_id = char_to_id
        # Per-__getitem__ random shift of the window start, bounded by the record's
        # slack (len(seq) - window_len). Window starts are otherwise deterministic and
        # epoch-invariant (_map_windows), so with ~57% of BED regions sitting at the
        # 4094bp inflation floor the model sees an identical position->content map on
        # every pass. NOTE this is a no-op for records with no slack: at window_len
        # 4096 only 43% of records have any, so jitter needs window_len < 4094 to bite.
        self.window_jitter = int(window_jitter)

        # Per-position aux columns to slice alongside the sequence (e.g. phylop).
        # Only keep those actually present in the dataset (graceful v5/v6 fallback).
        self._aux_position_cols = tuple(
            c for c in aux_position_cols if c in hf_dataset.column_names
        )

        # Fast Arrow window-slicing path: read seq + aux columns straight from the
        # backing Arrow table and slice the window BEFORE materializing to Python.
        # HF's default formatting boxes the *entire* per-record phyloP list (a ~7k
        # uint8 array on real v7 records, up to 138k) into Python ints on every
        # __getitem__, even though only a `window_len` slice is used — measured as
        # the dominant data-loading cost (~70x slower than the Arrow slice). This
        # path requires logical row order == Arrow row order, i.e. NO `_indices`
        # remap (no .shuffle/.select/.train_test_split on the dataset). It falls
        # back to the formatted-record path otherwise, and on any Arrow API change.
        self._arrow_seq = None
        self._arrow_aux: dict[str, Any] = {}
        self._arrow_chrom = None
        self._arrow_start = None
        self._fast_arrow = False
        try:
            if getattr(hf_dataset, "_indices", None) is None:
                tbl = hf_dataset.data
                self._arrow_seq = tbl.column("seq")
                self._arrow_aux = {c: tbl.column(c) for c in self._aux_position_cols}
                if self.emit_genomic or reference_fasta is not None:
                    self._arrow_chrom = tbl.column("chrom")
                    self._arrow_start = tbl.column("start")
                self._arrow_end = tbl.column("end") if reference_fasta is not None else None
                self._fast_arrow = True
        except Exception:  # pragma: no cover - defensive; formatted path still works
            self._arrow_seq = None
            self._arrow_aux = {}
            self._arrow_chrom = None
            self._arrow_start = None
            self._fast_arrow = False

        pad_len = (-window_len) % PAD_MULTIPLE
        self._total_len = window_len + pad_len
        self._pad_template = np.full(self._total_len, pad_id, dtype=np.int64)
        # NaN-filled template for per-position regression targets; the loss masks
        # on isfinite so padded tail positions never contribute.
        self._nan_template = np.full(self._total_len, np.nan, dtype=np.float16)
        self._attn_template = torch.cat(
            [
                torch.ones(window_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )

        # -- Length bucketing (off unless bucket_step > 0) ---------------------
        # With it on, a window is padded to its BUCKET, not to the full window
        # length, and LengthBucketBatchSampler only ever batches one bucket at a
        # time. `_bucket_len` is what the sampler groups on; __getitem__ derives
        # the bucket from the window's ACTUAL length instead of trusting that
        # array, so a disagreement surfaces as a loud torch.stack shape error in
        # the collator rather than as silent truncation.
        self.bucket_step = int(bucket_step)
        self._templates: dict[int, tuple[np.ndarray, np.ndarray, torch.Tensor]] = {}
        self._bucket_len: np.ndarray | None = None
        if self.bucket_step:
            if self.bucket_step % PAD_MULTIPLE:
                raise ValueError(
                    f"bucket_step {self.bucket_step} must be a multiple of "
                    f"PAD_MULTIPLE {PAD_MULTIPLE}"
                )
            if window_index.win_len is None:
                raise ValueError(
                    "length bucketing needs the window index's win_len column; this "
                    "index predates it -- delete the cached index so it rebuilds"
                )
            wl = window_index.win_len[filter_indices]
            self._bucket_len = bucket_lengths(wl, self._total_len, self.bucket_step)
            real, _, ratio = padding_efficiency(wl, self._total_len, self.bucket_step)
            fixed = real / (len(wl) * self._total_len) if len(wl) else 0.0
            logger.info(
                "Length bucketing: step %d, %d windows, %d buckets, "
                "%.4f padded-base efficiency (fixed-length path would be %.4f)",
                self.bucket_step,
                len(wl),
                len(np.unique(self._bucket_len)),
                ratio,
                fixed,
            )

    def __len__(self) -> int:
        return len(self._rec_idx)

    def _bucket_templates(self, n: int):
        """(pad, nan, attn, total_len) templates for a window of ``n`` real bases.

        Returns the full-window templates when bucketing is off, so the
        fixed-length path is byte-identical and pays no lookup.
        """
        if not self.bucket_step:
            return (
                self._pad_template,
                self._nan_template,
                self._attn_template,
                self._total_len,
            )
        b = bucket_of(n, self._total_len, self.bucket_step)
        tpl = self._templates.get(b)
        if tpl is None:
            tpl = (
                np.full(b, self.pad_id, dtype=np.int64),
                np.full(b, np.nan, dtype=np.float16),
                torch.ones(b, dtype=torch.long),
            )
            self._templates[b] = tpl
        return tpl[0], tpl[1], tpl[2], b

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec_idx = int(self._rec_idx[idx])
        ws = int(self._win_start[idx])

        # Fetch the record's seq (+ aux columns below). The fast path reads the
        # raw Arrow value and avoids boxing the full per-record phyloP list; the
        # fallback materializes the formatted record (used when _indices remaps).
        if self._fast_arrow:
            seq = self._arrow_seq[rec_idx].as_py()
            rec = None
        else:
            rec = self.hf_dataset[rec_idx]
            seq = rec["seq"]
        if self.window_jitter:
            # Global RNG on purpose: dataloader workers are re-seeded per epoch, so the
            # offset differs on every pass over the same window (same reasoning as the
            # unseeded MLM masking). Clamped to the record's slack, so this is a no-op
            # for records with none. has_variant in the index was computed for the
            # un-jittered start and may go stale by <= jitter bp; it only drives
            # index-level filtering, never the loss.
            # Slack is measured from THIS window's start, not the record's: a
            # non-first window of a multi-window record can only shift by what is
            # left after it. Measured 2026-08-23: with `len(seq) - window_len` the
            # end-aligned second window of a 4094-bp record (57% of the corpus at
            # window 2048) drew up to 2046 bp past the end and came back ~50%
            # padding on average -- attended as sequence, since NTv3 drops the mask.
            slack = len(seq) - self.window_len - ws
            if slack > 0:
                # Bound the DRAW by the slack, do not clamp the result: with
                # `--window-jitter == --window-len` (what every current arm uses)
                # and a record whose slack is small, drawing from [0, jitter] and
                # then clamping sends almost every draw to exactly `slack` -- e.g.
                # slack=104 with jitter=2048 lands on 104 in 95% of draws. That is
                # a new deterministic grid, which is precisely the failure jitter
                # exists to remove.
                ws += int(np.random.randint(0, min(self.window_jitter, slack) + 1))
        # v2 input: one character of left context, because the glyph at the window's first
        # position is rewritten from the marker just before it.
        lead = min(ws, 1) if self._v2_rewrite is not None else 0
        codepoints = np.frombuffer(
            seq[ws - lead : ws + self.window_len].encode("utf-32-le"), dtype=np.uint32
        )
        if self._v2_rewrite is not None:
            reference = None
            if self._reference_fasta is not None:
                if rec is None:
                    region = (self._arrow_chrom[rec_idx].as_py(), self._arrow_start[rec_idx].as_py(),
                              self._arrow_end[rec_idx].as_py())
                else:
                    region = (rec["chrom"], rec["start"], rec["end"])
                reference = row_reference(seq, *region, _fasta(str(self._reference_fasta), os.getpid()))[ws - lead : ws + self.window_len]
            if self._copy_swap and np.random.random() < 0.5:
                codepoints = swap_copies(codepoints)
            codepoints = self._v2_rewrite(codepoints, reference)[lead:]
        n = len(codepoints)
        pad_tpl, nan_tpl, attn_tpl, total_len = self._bucket_templates(n)

        out_ids = pad_tpl.copy()
        # `mode="clip"` maps any out-of-range codepoint to the table's last row
        # (the reserved unk sentinel) — identical to `np.minimum(cp, len-1)` then
        # index, but fuses the clamp and gather into one pass.
        out_ids[:n] = self._char_to_id.take(codepoints, mode="clip")
        if self._v2_rewrite is not None:
            # The v2 vocabulary holds exactly what the rewrite writes. <unk> here means the
            # token or counts file does not match the data: stop, do not train on it.
            unk = np.flatnonzero(out_ids[:n] == self._char_to_id[-1])
            if unk.size:
                raise ValueError(
                    f"--v2-input: {unk.size} character(s) outside the vocabulary in record "
                    f"{rec_idx} window start {ws}, first {chr(int(codepoints[unk[0]]))!r} "
                    f"(U+{int(codepoints[unk[0]]):04X}) at offset {int(unk[0])}"
                )

        if n < total_len:
            attn = torch.zeros(total_len, dtype=torch.long)
            attn[:n] = 1
        else:
            attn = attn_tpl

        item = {"input_ids": torch.from_numpy(out_ids), "attention_mask": attn}
        if self.emit_genomic:
            # Genomic coord (0-based) of the window's first token = record.start +
            # (# non-marker tokens in seq[:ws]); markers consume no reference base.
            # This 1-token=1-ref rule is EXACT only when no indel ({}) appears in the
            # window or its prefix (inserted/deleted bases inside a span break it and
            # the span encoding can't be decoded on the fly). So gate Track-B/C on an
            # indel-free prefix+window; otherwise emit genomic_start=-1 and the
            # collator skips the row. (~28% of 4k windows qualify; alignment ~95%+.)
            if self._fast_arrow:
                chrom = self._arrow_chrom[rec_idx].as_py()
                rstart = int(self._arrow_start[rec_idx].as_py())
            else:
                chrom = rec["chrom"]
                rstart = int(rec["start"])
            item["chrom"] = chrom
            if "{" in seq[: ws + n]:
                item["genomic_start"] = -1  # indel in prefix/window -> not alignable
            elif ws:
                pre = np.frombuffer(seq[:ws].encode("utf-32-le"), dtype=np.uint32)
                item["genomic_start"] = rstart + (
                    ws - int(np.isin(pre, self._marker_ords).sum())
                )
            else:
                item["genomic_start"] = rstart
        if self._is_ref is not None:
            item["is_ref"] = torch.tensor(bool(self._is_ref[idx]), dtype=torch.bool)
        if self._sample_weight is not None:
            item["sample_weight"] = torch.tensor(
                float(self._sample_weight[idx]), dtype=torch.float32
            )
        # Per-position regression targets: produce a dense float16 array aligned
        # 1:1 with the seq window (tokens are single codepoints), NaN-padded to the
        # padded length. These ride through the collator's marker-drop gather in
        # lockstep with tokens. phyloP is stored dense (uint8, decoded here); CADD
        # is stored sparse (list<struct<pos,score>>, expanded onto its anchors).
        for col in self._aux_position_cols:
            out_aux = nan_tpl.copy()
            if col in AUX_SPARSE_TASKS:
                # Sparse anchors: place each in-window entry at its window-relative
                # index. The `< ws + n` bound also guards a pos == seq_len overflow.
                if self._fast_arrow:
                    # Read the struct child's pos/score as numpy and place in one
                    # vectorized scatter — avoids materializing a list-of-dicts.
                    sv = self._arrow_aux[col][rec_idx].values
                    if sv is not None and len(sv):
                        pos = sv.field("pos").to_numpy(zero_copy_only=False)
                        score = sv.field("score").to_numpy(zero_copy_only=False)
                        m = (pos >= ws) & (pos < ws + n)
                        if m.any():
                            out_aux[(pos[m] - ws).astype(np.intp)] = score[m].astype(
                                np.float16
                            )
                else:
                    for entry in rec[col]:
                        pos = entry["pos"]
                        if ws <= pos < ws + n:
                            out_aux[pos - ws] = entry["score"]
            else:
                # Dense uint8 (phyloP): decode 255→NaN + linear [-20,+10.003] via the
                # precomputed LUT. The fast path slices the window from the Arrow
                # list child (zero-copy) so the full per-record list is never boxed.
                if self._fast_arrow:
                    child = self._arrow_aux[col][rec_idx].values
                    raw = (
                        child[ws : ws + n].to_numpy(zero_copy_only=False)
                        if child is not None
                        else np.empty(0, dtype=np.uint8)
                    )
                else:
                    raw = np.asarray(rec[col][ws : ws + n], dtype=np.uint8)
                out_aux[: raw.shape[0]] = _PHYLOP_LUT[raw]
            item[col] = torch.from_numpy(out_aux)
        return item


class DiploidMLMCollator(DataCollatorForLanguageModeling):
    """MLM collator for the diploid vocab (uniform mask rate, see __init__).

    N tokens are never masked. The 10%-random replacements are drawn from the
    data's unigram distribution (``token_freqs``), never specials/markers.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        base_vocab_size: int,
        mlm_prob: float = 0.15,
        token_freqs: dict[str, float] | None = None,
    ):
        super().__init__(tokenizer=tokenizer, mlm_probability=mlm_prob)
        self.base_vocab_size = base_vocab_size
        # Mask rate is uniform across base and diploid tokens. The historical
        # asymmetric path made per-class gradient comparisons harder to
        # interpret and never showed a downstream gain.

        vocab = tokenizer.get_vocab()
        n_id = tokenizer.convert_tokens_to_ids("N") if "N" in vocab else None
        self._ignore_token_ids: set[int] = {n_id} if isinstance(n_id, int) else set()
        self._mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        full_vocab = merged_vocab(tokenizer)
        # Marker tokens (^, {, }) are maskable (model should learn to predict
        # them) but excluded from the random-replacement pool so they don't
        # appear as noise where a nucleotide/diploid token belongs. Markers
        # are ADDED tokens, so they must be resolved against the merged vocab
        # — ``get_vocab()`` alone misses them on the slow NTv3 tokenizer,
        # which silently left them in the old replacement pool.
        marker_ids: set[int] = set()
        for marker in MARKER_TOKENS:
            mid = full_vocab.get(marker)
            if mid is not None:
                marker_ids.add(mid)

        all_ids = set(full_vocab.values())
        self._special_ids = frozenset(tokenizer.all_special_ids)

        table_size = (max(all_ids) + 1) if all_ids else len(tokenizer)
        self._is_special = torch.zeros(table_size, dtype=torch.bool)
        for sid in self._special_ids:
            if sid < table_size:
                self._is_special[sid] = True

        # Random-replacement pool: specials and markers are always excluded
        # (control / structural tokens — a stray ^/{ is trivially detectable
        # corruption and breaks span parsing). The fill DISTRIBUTION follows
        # the data's unigram frequencies (token_freqs): a fill is then
        # a-priori indistinguishable from real data, so only context reveals
        # corruption. Rare diploid tokens stay possible fills at their honest
        # rarity (~1e-3 of fills) instead of ~88% of them (the old uniform
        # pool injected noise diploids at the same rate as REAL variants,
        # ~1.5e-3/position, teaching "diploid token = probably noise").
        # Fallback (no table, or a tokenizer sharing no symbol with it):
        # uniform over canonical nucleotides — within a few points of the true
        # marginal, and rare-token mass ~ 0.
        excluded = self._special_ids | marker_ids
        pool = [
            (tid, (token_freqs or {})[tok])
            for tok, tid in full_vocab.items()
            if tid not in excluded and (token_freqs or {}).get(tok, 0.0) > 0.0
        ] or [(full_vocab[t], 1.0) for t in "ACGT" if t in full_vocab]
        w = torch.tensor([wt for _, wt in pool], dtype=torch.float)
        self._random_token_ids_tensor = torch.tensor(
            [tid for tid, _ in pool], dtype=torch.long
        )
        self._random_token_weights = w / w.sum()

    def torch_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Any | None = None,
        offset_mapping: Any | None = None,
    ) -> tuple[Any, Any]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Vectorized O(1) special-token detection via lookup table
        if special_tokens_mask is None:
            special_tokens_mask = self._is_special[inputs]
        special_tokens_mask = (
            special_tokens_mask.bool()
            if isinstance(special_tokens_mask, torch.Tensor)
            else torch.tensor(special_tokens_mask, dtype=torch.bool)
        )
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)

        for tid in self._ignore_token_ids:
            probability_matrix.masked_fill_(inputs == tid, 0.0)

        # Uniform mask rate for base and diploid tokens (see __init__ note).
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% -> [MASK], 10% -> random, 10% unchanged (single rand call)
        rand = torch.rand(labels.shape, device=inputs.device)
        indices_replaced = (rand < 0.8) & masked_indices
        indices_random = (rand >= 0.8) & (rand < 0.9) & masked_indices
        inputs[indices_replaced] = self._mask_token_id

        # Cache random token tensors on the right device
        if self._random_token_ids_tensor.device != inputs.device:
            self._random_token_ids_tensor = self._random_token_ids_tensor.to(
                device=inputs.device
            )
            self._random_token_weights = self._random_token_weights.to(
                device=inputs.device
            )
        n_random = int(indices_random.sum())
        if n_random:
            # Unigram-frequency-weighted draw (uniform when no freq table).
            draw = torch.multinomial(
                self._random_token_weights, n_random, replacement=True
            )
            inputs[indices_random] = self._random_token_ids_tensor[draw]

        return inputs, labels


class MarkerDropMLMCollator(DiploidMLMCollator):
    """``DiploidMLMCollator`` + random deletion of selected marker tokens.

    Direction markers (‹/›) deterministically reveal variant phase, letting the
    MLM head "cheat" when predicting an adjacent phased base. Randomly deleting
    a fraction of them forces the model to learn ungapped reconstructions —
    especially important since downstream inference may lack phasing info even
    when training does. Deletion happens BEFORE masking so a deleted position
    is never a label, and adjacent-base labels are KEPT (predicting them
    without the marker is the whole point).

    The deletion is vectorized across the batch: a single ``cumsum``-based gather
    rewrites every row, then re-pads to the original sequence length so the model
    still sees a fixed shape (NTv3 needs length divisible by ``PAD_MULTIPLE``).
    Which markers are deleted is ``marker_drop_token_ids`` (``--marker-drop-target``): the
    default keeps the INDEL braces and ``^``; ``all`` at p=1 deletes every marker (the
    in-place diploid input).
    """

    def __init__(
        self,
        *args: Any,
        marker_drop_token_ids: tuple[int, ...] = (),
        marker_drop_prob: float = 0.0,
        pad_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Back-compat: ignore the obsolete dir_marker_ids kwarg if passed.
        kwargs.pop("dir_marker_ids", None)
        super().__init__(*args, **kwargs)
        self.marker_drop_prob = float(marker_drop_prob)
        ids = sorted({int(i) for i in marker_drop_token_ids})
        self._marker_drop_ids = torch.tensor(ids, dtype=torch.long)
        self._pad_id_for_drop = (
            int(pad_id) if pad_id is not None else int(self.tokenizer.pad_token_id or 0)
        )
        if ids and self.marker_drop_prob > 0:
            table = torch.zeros(max(ids) + 1, dtype=torch.bool)
            table[self._marker_drop_ids] = True
            self._is_marker_drop = table
            logger.info(
                "MarkerDropMLMCollator: dropping marker ids %s with p=%.2f",
                ids,
                self.marker_drop_prob,
            )
        else:
            self._is_marker_drop = None

    # Per-example scalar aux carried alongside tokens. Popped before super (which
    # only knows input_ids/attention_mask/labels), re-attached after.
    _SCALAR_AUX_KEYS = ("is_ref", "sample_weight")

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            return super().torch_call(examples)

        # Pop per-example scalar aux (is_ref, sample_weight) → stacks.
        scalar_stacks: dict[str, torch.Tensor] = {}
        for key in self._SCALAR_AUX_KEYS:
            if key in examples[0]:
                scalar_stacks[key] = torch.stack(
                    [torch.as_tensor(e[key]) for e in examples]
                )

        # Pop per-position aux (e.g. phylop) → (B, L) stacks that MUST ride
        # through the marker-drop gather in lockstep with input_ids so each
        # target stays aligned to its token after direction markers are deleted.
        pos_arrays: dict[str, torch.Tensor] = {}
        for key in AUX_POSITION_TASKS:
            if key in examples[0]:
                pos_arrays[key] = torch.stack(
                    [torch.as_tensor(e[key]) for e in examples]
                )

        drop_keys = set(scalar_stacks) | set(pos_arrays)
        if drop_keys:
            examples = [
                {k: v for k, v in ex.items() if k not in drop_keys} for ex in examples
            ]

        if self._is_marker_drop is not None:
            examples, pos_arrays = self._batch_drop_markers(examples, pos_arrays)

        batch = super().torch_call(examples)
        for key, value in scalar_stacks.items():
            batch[key] = value
        for key, value in pos_arrays.items():
            batch[key] = value
        return batch

    def _batch_drop_markers(
        self,
        examples: list[dict[str, Any]],
        pos_arrays: dict[str, torch.Tensor] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
        pos_arrays = pos_arrays or {}
        ids = torch.stack(
            [
                e["input_ids"]
                if isinstance(e["input_ids"], torch.Tensor)
                else torch.as_tensor(e["input_ids"], dtype=torch.long)
                for e in examples
            ]
        )
        attn = torch.stack(
            [
                e["attention_mask"]
                if isinstance(e["attention_mask"], torch.Tensor)
                else torch.as_tensor(e["attention_mask"], dtype=torch.long)
                for e in examples
            ]
        )

        b, length = ids.shape
        in_range = ids < self._is_marker_drop.numel()
        is_marker = torch.zeros_like(ids, dtype=torch.bool)
        is_marker[in_range] = self._is_marker_drop[ids[in_range]]
        if not is_marker.any():
            return examples, pos_arrays

        drop = is_marker & (torch.rand(b, length) < self.marker_drop_prob)
        if not drop.any():
            return examples, pos_arrays

        keep = ~drop
        new_pos = keep.cumsum(dim=1) - 1  # new index of each kept position

        # Compact every kept position to its new index via a single flat scatter
        # (cheaper than boolean advanced-index assignment, which is the hot spot
        # of this every-batch path). Each kept position maps to a unique flat
        # destination `row*length + new_pos`; dropped positions route to a shared
        # dump slot at index `b*length` that is sliced off. Unwritten trailing
        # positions keep the per-tensor default (pad_id / 0 / NaN).
        rows = torch.arange(b, device=ids.device).unsqueeze(1).expand(b, length)
        dump = b * length
        flat_dest = torch.where(
            keep, rows * length + new_pos, torch.full_like(new_pos, dump)
        ).reshape(-1)

        def _compact(src: torch.Tensor, fill: float) -> torch.Tensor:
            flat = src.new_full((dump + 1,), fill)
            flat.scatter_(0, flat_dest, src.reshape(-1))
            return flat[:dump].reshape(b, length)

        new_ids = _compact(ids, self._pad_id_for_drop)
        new_attn = _compact(attn, 0)

        # Gather per-position aux with the SAME keep/new_pos; NaN-pad dropped and
        # trailing positions. Dropped marker positions are NaN in the source, so
        # this only removes NaNs and shifts the remaining (real) targets to track
        # their tokens.
        new_pos_arrays: dict[str, torch.Tensor] = {
            key: _compact(arr, float("nan")) for key, arr in pos_arrays.items()
        }

        new_examples = [
            {"input_ids": new_ids[i], "attention_mask": new_attn[i]} for i in range(b)
        ]
        return new_examples, new_pos_arrays


class MultiObjectiveCollator(MarkerDropMLMCollator):
    """MarkerDrop collator + Phase-2 fields for CPL / Track-B / Track-C.

    On top of the base MLM batch it emits (each gated by an enable flag):
      * CPL  → ``swapped_input_ids`` (‹↔› flipped, same MLM mask) + ``het_position_mask``.
      * Track-B → ``wt_input_ids`` (variant token → hg38 reference base), ``variant_pos_mask``,
        ``variant_effect`` (= CADD at variant anchors, for effect-ranking).
      * Track-C → ``af`` (B, L): 1000G AF at variant anchors, NaN elsewhere (an aux task).

    Track-B/C need each variant token's genomic coordinate. Per window we carry the
    window's genomic start (``genomic_start``, computed in the dataset) and recover a
    token's position as ``genomic_start + #non-marker-tokens-before`` (the verified
    SNV alignment). Indel windows ({}/}) are skipped (alignment shifts there). Requires
    ``marker_drop_prob == 0`` so direction markers survive (CPL) and offsets stay exact.
    """

    def __init__(
        self,
        *args: Any,
        tokens_path: str | None = None,
        twobit_path: str | None = None,
        af_dir: str | None = None,
        cpl_enable: bool = False,
        cpl_swap_mode: str = "directional",
        trackb_enable: bool = False,
        trackc_enable: bool = False,
        smp_enable: bool = False,
        smp_mask_frac: float = 0.4,
        smp_loo_frac: float = 0.15,
        smp_block_max: int = 7,
        smp_w_hard: float = 3.0,
        smp_target: str = "base",
        ctp_enable: bool = False,
        ctp_pairs_per_row: int = 8,
        ctp_dist_weight: bool = True,
        ctp_min_dist: int = 0,
        ctp_mask_markers: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cpl_enable = cpl_enable
        self.trackb_enable = trackb_enable
        self.trackc_enable = trackc_enable
        # SMP (supervised masked-phase) knobs. mask_frac = fraction of markers hidden
        # per row (span for linkage + scattered for coverage, ≥1 anchor kept); loo_frac
        # = P(hide exactly one marker instead); block_max bounds the contiguous span;
        # w_hard up-weights positions where copy-nearest is wrong (phase switches).
        self.smp_enable = smp_enable
        self.smp_mask_frac = float(smp_mask_frac)
        self.smp_loo_frac = float(smp_loo_frac)
        self.smp_block_max = int(smp_block_max)
        self.smp_w_hard = float(smp_w_hard)
        self.smp_target = str(
            smp_target
        )  # 'marker' (supervise the masked ‹/›) or 'base' (the c+1 het base)
        # CTP (supervised cis/trans pairwise) knobs. pairs_per_row = max het pairs sampled
        # per row; dist_weight biases pair sampling + loss toward longer inter-marker
        # distance (counter the copy-nearest shortcut); min_dist drops pairs closer than
        # that; mask_markers hides BOTH markers of a pair (de-novo regime) vs the default
        # marker-present regime that matches the frozen probe.
        self.ctp_enable = ctp_enable
        self.ctp_pairs_per_row = int(ctp_pairs_per_row)
        self.ctp_dist_weight = bool(ctp_dist_weight)
        self.ctp_min_dist = int(ctp_min_dist)
        self.ctp_mask_markers = bool(ctp_mask_markers)
        self._smp_mask_id = self.tokenizer.mask_token_id
        self._twobit_path = twobit_path
        self._af_dir = af_dir
        self._tb = None  # lazy py2bit handle (per dataloader worker)
        self._aft = None  # lazy AFTables
        c2i = self.tokenizer.convert_tokens_to_ids
        self._dir_fwd_id = c2i(DIR_FWD)
        self._dir_rev_id = c2i(DIR_REV)
        self._open_id = c2i(INDEL_OPEN)
        self._close_id = c2i(INDEL_CLOSE)
        self._marker_ids = [c2i(m) for m in MARKER_TOKENS]
        self._base_id = {i: c2i(b) for b, i in BASE_TO_INT.items()}  # 0..3 -> token id
        # CPL twin-swap mode. 'directional' (default) swaps ‹/› markers; 'phased' swaps the
        # ordered symbol to its reverse-order counterpart (Á<->Ç) for v4/phased tokenization.
        self.cpl_swap_mode = cpl_swap_mode
        self._cpl_remap = None
        self._cpl_is_phased = None
        if cpl_enable and cpl_swap_mode == "phased":
            if not tokens_path:
                raise ValueError(
                    "CPL swap-mode 'phased' requires tokens_path (--tokens)."
                )
            # NTv3Tokenizer.__len__ == base vocab (11); the diploid symbols (ids ~11-54) live in
            # added_tokens_decoder. Size the remap to cover all token ids that can appear in input_ids.
            _added = getattr(self.tokenizer, "added_tokens_decoder", None) or {}
            vocab_n = (max(_added) + 1) if _added else len(self.tokenizer)
            remap = torch.arange(vocab_n, dtype=torch.long)
            isph = torch.zeros(vocab_n, dtype=torch.bool)
            phased: dict[str, str] = {}
            for line in open(tokens_path, encoding="utf-8"):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[1] == "phased":
                    phased[parts[0]] = parts[2]
            for pair, sym in phased.items():
                rev = pair[::-1]
                if rev not in phased:
                    continue
                sid = c2i(sym)
                rid = c2i(phased[rev])
                if (
                    sid is None
                    or rid is None
                    or not (0 <= sid < vocab_n and 0 <= rid < vocab_n)
                    or sid == rid
                ):
                    continue
                remap[sid] = rid
                isph[sid] = True
            if not bool(isph.any()):
                raise ValueError(
                    "CPL swap-mode 'phased' but no phased reverse-pairs found in tokens; the phase "
                    "swap would be a no-op (wrong tokenization for --cpl-swap-mode phased?)."
                )
            self._cpl_remap = remap
            self._cpl_is_phased = isph
        # Track-B/C recover a variant's genomic coordinate as genomic_start + #tokens-before;
        # marker-drop COMPACTS the sequence, so ANY drop breaks that alignment → forbid it.
        if (trackb_enable or trackc_enable) and self.marker_drop_prob > 0:
            raise ValueError(
                "MultiObjectiveCollator (Track-B/C) needs marker_drop_prob=0 "
                "(compaction breaks genomic-offset alignment)."
            )
        # SMP/CTP only need the ‹/› direction markers present (to derive orientation/cis-trans
        # labels + read the het base at c+1); they run on the already-compacted ids, so dropping
        # the ^ SNP marker (--marker-drop-target snp) is fine. Forbid only a drop set that would
        # remove a direction marker.
        _drop_ids = set(int(i) for i in self._marker_drop_ids.tolist())
        _drops_dir = (self._dir_fwd_id in _drop_ids) or (self._dir_rev_id in _drop_ids)
        if (smp_enable or ctp_enable) and self.marker_drop_prob > 0 and _drops_dir:
            raise ValueError(
                "SMP/CTP need the ‹/› direction markers present (labels) — use "
                "--marker-drop-target snp to drop only ^, or set marker_drop_prob=0."
            )
        # token id -> set of ACGT alleles it encodes (for Track-C alt detection).
        self._tok_alleles: dict[int, frozenset] = {}
        if trackc_enable and tokens_path:
            acc: dict[int, set] = {}
            for ln in open(tokens_path):
                if ln.startswith("#") or not ln.strip():
                    continue
                parts = ln.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                pair, _order, symbol = parts[0], parts[1], parts[2]
                if len(symbol) != 1:
                    continue
                tid = c2i(symbol)
                if tid is None:
                    continue
                acc.setdefault(tid, set()).update(c for c in pair if c in BASE_TO_INT)
            self._tok_alleles = {k: frozenset(v) for k, v in acc.items() if v}

    def _twobit(self):
        if self._tb is None and self._twobit_path:
            import py2bit

            self._tb = py2bit.open(self._twobit_path)
        return self._tb

    def _aftables(self):
        if self._aft is None and self._af_dir:
            self._aft = AFTables(self._af_dir)
        return self._aft

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        chroms = gstart = None
        if examples and "chrom" in examples[0]:
            chroms = [e.get("chrom") for e in examples]
            gstart = [int(e["genomic_start"]) for e in examples]
            examples = [
                {k: v for k, v in e.items() if k not in ("chrom", "genomic_start")}
                for e in examples
            ]
        batch = super().torch_call(examples)
        if not (
            self.cpl_enable
            or self.trackb_enable
            or self.trackc_enable
            or self.smp_enable
            or self.ctp_enable
        ):
            return batch
        ids = batch["input_ids"]
        labels = batch.get("labels")
        # SMP runs FIRST: it hides a subset of direction markers in ``ids`` (→ [MASK]).
        # CPL (below) then builds its twin from the already-SMP-masked ``ids``, so the
        # two views stay consistent at SMP-hidden positions (the combined SMP+CPL arm).
        if self.smp_enable and labels is not None:
            self._add_smp_fields(batch, ids, labels)
        # CTP runs after SMP (so in the combined arm it pairs the markers SMP left
        # visible) and before CPL. In the default (visible) regime it does not touch
        # ``ids``; with --ctp-mask-markers it hides both markers of each pair.
        if self.ctp_enable and labels is not None:
            self._add_cistrans_fields(batch, ids, labels)
        if self.cpl_enable and labels is not None:
            orig = ids.clone()
            sel = labels != -100
            orig[sel] = labels[sel]
            if self.cpl_swap_mode == "phased":
                # v4/phased: swap each ordered symbol to its reverse-order counterpart (Á<->Ç).
                remap = self._cpl_remap.to(orig.device)
                sw = remap[orig]
                het = self._cpl_is_phased.to(orig.device)[orig] & (~sel)
            else:
                # directional: flip the ‹/› markers; het = the base right after a direction marker.
                sw = orig.clone()
                fwd = sw == self._dir_fwd_id
                rev = sw == self._dir_rev_id
                sw[fwd] = self._dir_rev_id
                sw[rev] = self._dir_fwd_id
                is_dir = (orig == self._dir_fwd_id) | (orig == self._dir_rev_id)
                after = torch.zeros_like(is_dir)
                after[:, 1:] = is_dir[:, :-1]
                het = after & (~sel) & (~is_dir)
            swapped = sw.clone()
            swapped[sel] = ids[sel]
            batch["swapped_input_ids"] = swapped
            batch["het_position_mask"] = het
            # Guard the C2-class silent footgun: CPL on but the swap is inert -> phase-blind model, no error.
            if not bool(het.any()) and not getattr(self, "_cpl_inert_warned", False):
                logger.warning(
                    "CPL enabled but 0 het swap positions this batch — wrong --cpl-swap-mode for this "
                    "tokenization? ('directional' needs ‹/› markers; 'phased' needs ordered symbols)."
                )
                self._cpl_inert_warned = True
        if (self.trackb_enable or self.trackc_enable) and chroms is not None:
            self._add_variant_fields(batch, ids, chroms, gstart)
        return batch

    def _add_smp_fields(self, batch, ids, labels):
        """SMP (supervised masked-phase): hide a fraction of direction markers and make
        the model predict their orientation FROM THE HET-BASE token's representation.
        For each chosen marker at position ``c`` we overwrite ``ids[c]`` with [MASK]
        (hiding the phase) but emit the supervised target at ``c+1`` — the unphased
        het-base symbol the downstream frozen probe and CPL actually read — forcing
        phase into THAT representation, not the marker slot. Per row ~``smp_mask_frac``
        of the markers are hidden (a contiguous span for long-range linkage + scattered
        for representation coverage), leaving ≥1 visible anchor; positions where
        copy-nearest is wrong get weight ``smp_w_hard`` (the phase-switch / hard
        positions). A small ``smp_loo_frac`` of rows hide exactly one marker (to keep
        the train distribution covering the eval probe). Needs ≥2 candidate markers in
        a row (≥1 anchor) for orientation to be identifiable; a candidate also needs a
        VISIBLE het-base at ``c+1`` (in range, not MLM-masked, not itself a marker)."""
        mask_id = self._smp_mask_id
        if mask_id is None:
            return
        fwd_id, rev_id = self._dir_fwd_id, self._dir_rev_id
        B, L = ids.shape
        orig = ids.clone()
        sel = labels != -100
        orig[sel] = labels[sel]
        is_rev = orig == rev_id
        is_marker = (orig == fwd_id) | is_rev
        target_base = self.smp_target == "base"
        cand = is_marker & (~sel)
        if target_base:
            # the het base at c+1 must be visible (in range, not MLM-masked, not a marker)
            base_ok = torch.zeros((B, L), dtype=torch.bool)
            base_ok[:, :-1] = (~sel[:, 1:]) & (~is_marker[:, 1:])
            cand = cand & base_ok
        label_mask = torch.zeros((B, L), dtype=torch.bool)
        smp_labels = torch.zeros((B, L), dtype=torch.long)
        smp_weight = torch.zeros((B, L), dtype=torch.float)
        frac = float(self.smp_mask_frac)
        block_max = max(1, int(self.smp_block_max))
        w_hard = float(self.smp_w_hard)
        any_target = False
        for b in range(B):
            pos = torch.nonzero(cand[b], as_tuple=False).flatten()
            nd = int(pos.numel())
            if nd < 2:  # need ≥1 anchor → orientation is unidentifiable otherwise
                continue
            if torch.rand(()).item() < self.smp_loo_frac:
                idx = [int(torch.randint(0, nd, ()).item())]  # LOO: hide one marker
            else:
                n_mask = min(max(1, round(frac * nd)), nd - 1)  # ~frac, leave ≥1 anchor
                ks = min(int(torch.randint(1, block_max + 1, ()).item()), n_mask)
                start = int(torch.randint(0, nd - ks + 1, ()).item())
                idx = list(range(start, start + ks))  # contiguous span (linkage)
                extra = n_mask - ks
                if extra > 0:  # + scattered (coverage)
                    rest = [i for i in range(nd) if i < start or i >= start + ks]
                    take = min(extra, max(0, len(rest) - 1))  # keep ≥1 anchor
                    if take > 0:
                        for j in torch.randperm(len(rest))[:take].tolist():
                            idx.append(rest[j])
            chosen = pos[idx]
            visible = pos[[i for i in range(nd) if i not in set(idx)]]
            if visible.numel() == 0:
                continue
            for c in chosen.tolist():
                lp = (
                    c + 1 if target_base else c
                )  # supervise at the het base or the masked marker
                ids[b, c] = mask_id
                label_mask[b, lp] = True
                truth = bool(is_rev[b, c].item())
                smp_labels[b, lp] = 1 if truth else 0  # 0=‹ (fwd), 1=› (rev)
                nn = int(visible[torch.argmin((visible - c).abs())].item())
                copy_pred = bool(is_rev[b, nn].item())
                smp_weight[b, lp] = w_hard if copy_pred != truth else 1.0
                any_target = True
        if any_target:
            batch["smp_label_mask"] = label_mask
            batch["smp_labels"] = smp_labels
            batch["smp_weight"] = smp_weight
        elif not getattr(self, "_smp_inert_warned", False):
            logger.warning(
                "SMP enabled but 0 maskable direction markers this batch — wrong "
                "tokenization (needs directional ‹/› markers) or marker_drop_prob>0?"
            )
            self._smp_inert_warned = True

    def _add_cistrans_fields(self, batch, ids, labels):
        """CTP (supervised cis/trans pairwise): sample het-marker PAIRS per row and emit,
        per pair, the two HET-BASE positions (c+1 — the SYM tokens the frozen
        compound-het probe reads via ``pos_A/pos_B``) plus a GAUGE-INVARIANT label
        (1=cis/same-haplotype if the two markers share orientation, 0=trans/different).
        Pairs are sampled with a bias toward LONGER inter-marker distance (counters the
        copy-nearest autocorrelation shortcut the naive control already captures). By
        default markers stay VISIBLE — the marker-present regime the benchmark probe
        runs in (it rewards routing the visible marker into the het-base rep, not
        inferring it); ``ctp_mask_markers`` instead hides BOTH markers of a pair (de-novo
        inference). Needs ≥2 candidate het markers (each with a visible het-base at c+1)
        in a row. The label is derived for free from the folded-back ``orig`` markers and
        is invariant to a within-window hap0/hap1 relabel — matching the comphet task."""
        mask_id = self._smp_mask_id
        if mask_id is None:
            return
        fwd_id, rev_id = self._dir_fwd_id, self._dir_rev_id
        B, L = ids.shape
        orig = ids.clone()
        sel = labels != -100
        orig[sel] = labels[sel]
        is_rev = orig == rev_id
        is_marker = (orig == fwd_id) | is_rev
        # a usable marker has a VISIBLE het-base at c+1 (in range, not MLM-masked, not a marker)
        base_ok = torch.zeros((B, L), dtype=torch.bool)
        base_ok[:, :-1] = (~sel[:, 1:]) & (~is_marker[:, 1:])
        cand = is_marker & (~sel) & base_ok
        K = max(1, int(self.ctp_pairs_per_row))
        pair_pos = torch.zeros((B, K, 2), dtype=torch.long)
        pair_label = torch.zeros((B, K), dtype=torch.long)
        pair_mask = torch.zeros((B, K), dtype=torch.bool)
        pair_weight = torch.zeros((B, K), dtype=torch.float)
        min_dist = max(0, int(self.ctp_min_dist))
        dist_w = self.ctp_dist_weight
        any_pair = False
        for b in range(B):
            posl = torch.nonzero(cand[b], as_tuple=False).flatten().tolist()
            nd = len(posl)
            if nd < 2:
                continue
            cand_pairs, wts = [], []
            for ii in range(nd):
                for jj in range(ii + 1, nd):
                    d = posl[jj] - posl[ii]
                    if d < min_dist:
                        continue
                    cand_pairs.append((posl[ii], posl[jj]))
                    wts.append(float(d) if dist_w else 1.0)
            if not cand_pairs:
                continue
            n_take = min(K, len(cand_pairs))
            idxs = torch.multinomial(
                torch.tensor(wts), n_take, replacement=False
            ).tolist()
            for slot, pi in enumerate(idxs):
                c1, c2 = cand_pairs[pi]
                pair_pos[b, slot, 0] = (
                    c1 + 1
                )  # het-base (SYM) of marker 1 — what the probe reads
                pair_pos[b, slot, 1] = c2 + 1  # het-base (SYM) of marker 2
                same = bool(is_rev[b, c1].item()) == bool(is_rev[b, c2].item())
                pair_label[b, slot] = 1 if same else 0  # 1=cis (same hap), 0=trans
                pair_mask[b, slot] = True
                pair_weight[b, slot] = (1.0 + (c2 - c1) / L) if dist_w else 1.0
                if self.ctp_mask_markers:
                    ids[b, c1] = mask_id
                    ids[b, c2] = mask_id
                any_pair = True
        if any_pair:
            batch["ctp_pair_pos"] = pair_pos
            batch["ctp_pair_label"] = pair_label
            batch["ctp_pair_mask"] = pair_mask
            batch["ctp_pair_weight"] = pair_weight
        elif not getattr(self, "_ctp_inert_warned", False):
            logger.warning(
                "CTP enabled but 0 het marker pairs this batch — wrong tokenization "
                "(needs directional ‹/› markers) or marker_drop_prob>0?"
            )
            self._ctp_inert_warned = True

    def _add_variant_fields(self, batch, ids, chroms, gstart):
        cadd = batch.get("cadd_phred")
        if cadd is None:
            return
        B, L = ids.shape
        marker_mask = torch.zeros_like(ids, dtype=torch.bool)
        for mid in self._marker_ids:
            marker_mask |= ids == mid
        offsets = token_genomic_offsets(ids, marker_mask)
        finite = torch.isfinite(cadd)
        has_indel = ((ids == self._open_id) | (ids == self._close_id)).any(dim=1)
        ids_np = ids.numpy()
        off_np = offsets.numpy()
        wt = ids.clone()
        vmask = torch.zeros(B, L, dtype=torch.bool)
        af = torch.full((B, L), float("nan"), dtype=torch.float32)
        tb = self._twobit() if self.trackb_enable or self.trackc_enable else None
        aft = self._aftables() if self.trackc_enable else None
        for b in range(B):
            # genomic_start<0 (indel in prefix/window -> not alignable) or any indel
            # token in the window: skip so Track-B/C only see exactly-aligned rows.
            if gstart[b] < 0 or bool(has_indel[b]):
                continue
            rows = torch.nonzero(finite[b], as_tuple=False).flatten().tolist()
            if not rows or tb is None:
                continue
            span = int(off_np[b, max(rows)]) + 1
            try:
                refseq = tb.sequence(chroms[b], gstart[b], gstart[b] + span).upper()
            except Exception:
                continue
            for p in rows:
                o = int(off_np[b, p])
                if o >= len(refseq):
                    continue
                ri = BASE_TO_INT.get(refseq[o])
                if ri is None:
                    continue
                if self.trackb_enable:
                    wt[b, p] = self._base_id[ri]
                    vmask[b, p] = True
                if self.trackc_enable and aft is not None:
                    alleles = self._tok_alleles.get(int(ids_np[b, p]))
                    if not alleles:
                        continue
                    alts = [a for a in alleles if a != refseq[o]]
                    if len(alts) == 1:
                        v = aft.lookup(
                            chroms[b], gstart[b] + o + 1, ri, BASE_TO_INT[alts[0]]
                        )
                        if v == v:  # not NaN
                            af[b, p] = v
        if self.trackb_enable:
            batch["wt_input_ids"] = wt
            batch["variant_pos_mask"] = vmask
            batch["variant_effect"] = cadd
        if self.trackc_enable:
            batch["af"] = af


class EvalCollatorMixin:
    """Trainer mixin: route eval dataloaders through ``self._eval_data_collator``.

    Set the attribute after construction. ``None`` falls back to the train collator.
    """

    _eval_data_collator: Any = None

    def get_eval_dataloader(self, eval_dataset: Any = None) -> Any:  # type: ignore[override]
        if self._eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)  # type: ignore[misc]
        train_collator = self.data_collator  # type: ignore[attr-defined]
        self.data_collator = self._eval_data_collator  # type: ignore[attr-defined]
        try:
            return super().get_eval_dataloader(eval_dataset)  # type: ignore[misc]
        finally:
            self.data_collator = train_collator  # type: ignore[attr-defined]


class _BinaryWeightSampler(torch.utils.data.Sampler):
    """O(1)/draw sampler for binary REF vs non-REF weighting.

    Replaces ``torch.utils.data.WeightedRandomSampler`` which (a) crashes
    with ``RuntimeError: number of categories cannot exceed 2^24`` on weight
    tensors > 16.7M entries (our index is ~217M), and (b) is slow at our
    scale because each per-epoch draw runs an O(N) multinomial. Our weight
    distribution is binary (REF rows: weight w; non-REF rows: weight 1), so
    we can do a per-draw Bernoulli(p_ref) → uniform-within-bucket in O(1).

    Per-epoch construction: O(num_samples). Per-draw: O(1).
    """

    def __init__(
        self,
        is_ref: np.ndarray,
        ref_weight: float,
        num_samples: int,
        seed: int = 42,
    ) -> None:
        # torch.utils.data.Sampler in current torch has a no-arg __init__
        # that delegates to object; passing None breaks. Call without args.
        super().__init__()
        self._ref_indices = np.where(is_ref)[0].astype(np.int64)
        self._nonref_indices = np.where(~is_ref)[0].astype(np.int64)
        self._num_samples = int(num_samples)
        self._seed = int(seed)
        self._epoch = 0
        n_ref = len(self._ref_indices)
        n_nonref = len(self._nonref_indices)
        total = n_ref * ref_weight + n_nonref
        self._p_ref = (n_ref * ref_weight) / max(total, 1.0)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed + 1000 * self._epoch)
        # Decide REF vs non-REF for each of the num_samples draws (vectorized Bernoulli).
        rand = torch.rand(self._num_samples, generator=g).numpy()
        is_ref_choice = rand < self._p_ref
        n_ref_draws = int(is_ref_choice.sum())
        n_nonref_draws = self._num_samples - n_ref_draws
        # Uniform within each bucket.
        np_rng = np.random.default_rng(self._seed + 1000 * self._epoch + 1)
        ref_picks = np_rng.integers(0, len(self._ref_indices), size=n_ref_draws)
        nonref_picks = np_rng.integers(
            0, len(self._nonref_indices), size=n_nonref_draws
        )
        out = np.empty(self._num_samples, dtype=np.int64)
        out[is_ref_choice] = self._ref_indices[ref_picks]
        out[~is_ref_choice] = self._nonref_indices[nonref_picks]
        return iter(out.tolist())


class WeightedSamplerMixin:
    """Trainer mixin: REF over-sampling via ``_BinaryWeightSampler``.

    Reads ``train_dataset._is_ref`` (numpy bool array). Falls back to the
    default Trainer sampler when the dataset doesn't expose ``_is_ref`` or
    when the weight is 1.0.
    """

    _ref_sampler_weight: float = 1.0

    def _get_train_sampler(self, train_dataset: Any = None):  # type: ignore[override]
        ds = train_dataset if train_dataset is not None else self.train_dataset  # type: ignore[attr-defined]
        is_ref = getattr(ds, "_is_ref", None)
        if self._ref_sampler_weight == 1.0 or is_ref is None:
            try:
                return super()._get_train_sampler(train_dataset)  # type: ignore[misc]
            except TypeError:
                return super()._get_train_sampler()  # type: ignore[misc]

        n_total = int(is_ref.size)
        n_ref = int(is_ref.sum())
        seed = int(getattr(self.args, "seed", 42))  # type: ignore[attr-defined]
        sampler = _BinaryWeightSampler(
            is_ref=is_ref,
            ref_weight=float(self._ref_sampler_weight),
            num_samples=n_total,
            seed=seed,
        )
        logger.info(
            "WeightedSamplerMixin (_BinaryWeightSampler): %d total / %d REF "
            "windows, ref weight=%.2f -> per-draw P(REF)=%.4f "
            "(baseline %.4f). O(1)/draw; bypasses WeightedRandomSampler's "
            "2^24 limit.",
            n_total,
            n_ref,
            self._ref_sampler_weight,
            sampler._p_ref,
            n_ref / max(n_total, 1),
        )
        return sampler


def _transform_sample_weight(
    sample_weight: torch.Tensor | None, transform: str, clip: float
) -> torch.Tensor | None:
    """Tame the heavy-tailed dedup-summed sample_weight before use in the loss.

    ``sqrt`` (default) compresses the tail while preserving order; ``log1p`` is
    stronger; ``clip``/``none`` leave it linear. All are capped at ``clip`` (>0)
    so a few ultra-common windows can't dominate the gradient.
    """
    if sample_weight is None:
        return None
    w = sample_weight.float().clamp_min(0.0)
    if transform == "sqrt":
        w = torch.sqrt(w)
    elif transform == "log1p":
        w = torch.log1p(w)
    if clip and clip > 0:
        w = w.clamp_max(float(clip))
    return w


def _per_example_weight(
    is_ref: torch.Tensor | None,
    sample_weight: torch.Tensor | None,
    *,
    use_sample_weight: bool,
    transform: str,
    clip: float,
    ref_loss_weight: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-example loss weight = transform(sample_weight) × ref multiplier.

    Shared by the MLM (``SampleWeightedLossMixin``) and aux-regression
    (``AuxRegressionLossMixin``) paths so both weight every example identically.
    """
    w = torch.ones(batch_size, device=device)
    if use_sample_weight:
        sw = _transform_sample_weight(sample_weight, transform, clip)
        if sw is not None:
            w = sw.to(device)
    if ref_loss_weight != 1.0 and is_ref is not None:
        w = w * torch.where(
            is_ref.to(device).bool(),
            torch.tensor(float(ref_loss_weight), device=device),
            torch.tensor(1.0, device=device),
        )
    return w


def _masked_weighted_regression(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    w_per_example: torch.Tensor,
    mean: float,
    std: float,
    *,
    loss_type: str = "huber",
    beta: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """NaN-safe, per-example-weighted regression loss over finite targets.

    ``pred``/``tgt`` are ``(B, L)``; ``w_per_example`` is ``(B,)``. Targets are
    standardized with FIXED ``(mean, std)`` (NOT batch stats). NaN targets are
    masked out *before* any arithmetic. When no finite target exists in the
    batch, returns ``pred.sum()*0`` (a graph-connected zero) so every head
    parameter still receives a gradient — required for DDP not to hang.
    """
    pred = pred.float()
    tgt = tgt.float()
    mask = torch.isfinite(tgt)
    n = int(mask.sum().item())
    if n == 0:
        return pred.sum() * 0.0, 0
    std = max(float(std), 1e-3)
    # Replace NaN with 0 before the subtraction so NaN never enters arithmetic;
    # the mask zeroes those positions out of the reduction anyway.
    tgt_std = (torch.where(mask, tgt, torch.zeros_like(tgt)) - mean) / std
    if loss_type == "mse":
        per_pos = F.mse_loss(pred, tgt_std, reduction="none")
    else:
        per_pos = F.smooth_l1_loss(pred, tgt_std, beta=float(beta), reduction="none")
    w2d = w_per_example.to(pred.device).unsqueeze(1).expand_as(per_pos) * mask.float()
    loss = (per_pos * w2d).sum() / w2d.sum().clamp_min(1e-8)
    return loss, n


def _pool_aux_targets(
    tgt: torch.Tensor, stride: int, mode: str = "mean"
) -> torch.Tensor:
    """Pool per-position targets ``(B, L)`` → ``(B, L//stride)``, NaN-aware.

    Used to bring full-resolution phyloP/CADD targets down to the transformer-
    bottleneck resolution (``stride`` = the model's downsample factor, 128) for
    the Track-A bottleneck heads. Each output bin reduces over only the FINITE
    inputs among its ``stride`` positions; an all-NaN bin stays NaN so the
    regression loss masks it (matching ``_masked_weighted_regression``).

    ``mode``: ``"mean"`` (average conservation), ``"max"`` (the peak — keeps the
    most-conserved/most-deleterious base per bin, the only sane reduction for the
    sparse CADD track), ``"l2"`` (RMS magnitude — preserves peaks plain mean would
    wash out, ignoring sign).
    """
    b, length = tgt.shape
    if length % stride != 0:
        raise ValueError(f"length {length} not divisible by stride {stride}")
    x = tgt.float().reshape(b, length // stride, stride)
    mask = torch.isfinite(x)
    count = mask.sum(dim=-1)  # finite count per bin, (B, n_bins)
    empty = count == 0
    safe = torch.where(mask, x, torch.zeros_like(x))
    denom = count.clamp_min(1).to(x.dtype)
    if mode == "mean":
        out = safe.sum(dim=-1) / denom
    elif mode == "l2":
        out = torch.sqrt((safe * safe).sum(dim=-1) / denom)
    elif mode == "max":
        out = torch.where(mask, x, torch.full_like(x, float("-inf"))).amax(dim=-1)
    else:
        raise ValueError(f"unknown pool mode {mode!r}")
    return torch.where(empty, torch.full_like(out, float("nan")), out)


class SampleWeightedLossMixin:
    """Trainer mixin: reduce the MLM loss as a per-example weighted mean
    ``Σ(w·l)/Σw``, where ``w = transform(sample_weight) × (ref_loss_weight if
    is_ref)``. Active when ``--use-sample-weight`` or ``ref_loss_weight != 1.0``.

    ``is_ref`` and ``sample_weight`` are popped unconditionally so they never
    reach the model. Cooperates with ``PerClassLossLoggingMixin`` /
    ``PostMLMTrainer.compute_loss`` via Python MRO + super() chains.
    """

    _ref_loss_weight: float = 1.0
    _use_sample_weight: bool = False
    _sample_weight_transform: str = "sqrt"
    _sample_weight_clip: float = 8.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        # HF passes the collated batch as a BatchEncoding (a MutableMapping, NOT a
        # dict subclass), so guard on MutableMapping — `isinstance(inputs, dict)`
        # would be False here and silently skip the per-example weight extraction.
        is_mapping = isinstance(inputs, MutableMapping)
        is_ref = inputs.pop("is_ref", None) if is_mapping else None
        sample_weight = inputs.pop("sample_weight", None) if is_mapping else None
        weighting_active = self._use_sample_weight or self._ref_loss_weight != 1.0
        if (
            not weighting_active
            or (is_ref is None and sample_weight is None)
            or inputs.get("labels") is None
        ):
            return super().compute_loss(  # type: ignore[misc]
                model, inputs, return_outputs=return_outputs, **kwargs
            )

        labels = inputs.get("labels")
        kwargs.pop("return_outputs", None)
        result = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, **kwargs
        )
        if isinstance(result, tuple):
            _unweighted, outputs = result
        else:
            _unweighted, outputs = result, None

        if outputs is None or getattr(outputs, "logits", None) is None:
            return (_unweighted, outputs) if return_outputs else _unweighted

        logits = outputs.logits
        labels_dev = labels.to(logits.device)
        per_pos = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels_dev.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels_dev.shape)
        valid = (labels_dev != -100).float()
        per_sample_loss = (per_pos * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        w = _per_example_weight(
            is_ref,
            sample_weight,
            use_sample_weight=self._use_sample_weight,
            transform=self._sample_weight_transform,
            clip=self._sample_weight_clip,
            ref_loss_weight=self._ref_loss_weight,
            batch_size=per_sample_loss.shape[0],
            device=logits.device,
        )
        loss = (per_sample_loss * w).sum() / w.sum().clamp_min(1e-8)
        return (loss, outputs) if return_outputs else loss


class AuxRegressionLossMixin:
    """Trainer mixin (top of MRO): add per-position regression loss term(s) to
    the (already sample-weighted) MLM loss.

    For each active task: ``total += lambda · masked_weighted_huber(pred, tgt)``,
    where ``pred`` comes from ``outputs.aux_preds[task]`` (set by
    ``AuxHeadsModel``) and ``tgt`` from ``inputs[task]`` (NaN-padded targets).
    Reported ``eval_loss`` stays MLM-only (aux added only during training) so
    milestone / load-best comparisons are unaffected; aux components are logged
    separately. Inert when the model is unwrapped (no ``aux_preds``) or no
    targets are present (v5/v6 fallback).
    """

    _aux_tasks: tuple[str, ...] = ()
    _aux_weights: dict[str, float] = {}  # task -> lambda
    _aux_standardize: dict[str, tuple[float, float]] = {}  # task -> (mean, std)
    _aux_loss_type: str = "huber"
    _aux_huber_beta: float = 1.0  # fallback when a task has no entry in _aux_betas
    _aux_betas: dict[str, float] = {}  # task -> per-task Huber beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        # Pop per-position targets so they never reach the model; read (without
        # popping) the per-example weighting fields that SampleWeightedLossMixin
        # pops downstream — the tensor refs stay valid after the pop.
        targets: dict[str, torch.Tensor] = {}
        sample_weight = is_ref = None
        # HF passes the collated batch as a BatchEncoding (a MutableMapping, NOT a
        # dict subclass); guarding on `dict` here would skip every aux target.
        if isinstance(inputs, MutableMapping):
            for name in self._aux_tasks:
                if name in inputs:
                    targets[name] = inputs.pop(name)
            sample_weight = inputs.get("sample_weight")
            is_ref = inputs.get("is_ref")
            # Step-gate the contrastive twins: on non-firing steps drop the twin
            # input so the model skips the extra backbone pass (the "1/3-step"
            # compute saving). The projector still gets a zero-dummy touch so DDP
            # marks it used. variant_effect/masks ride along harmlessly.
            cpl_fire, trackb_fire = self._contrastive_fire()
            if not cpl_fire:
                inputs.pop("swapped_input_ids", None)
            if not trackb_fire:
                inputs.pop("wt_input_ids", None)
            # SMP/CTP are head-only (no extra backbone pass) so _smp_every/_ctp_every
            # gate the objective as step-dropout regularization, not compute; drop the
            # label mask so the model's DDP-dummy branch fires (0 contribution, head
            # stays in-graph) on non-firing steps.
            smp_fire, ctp_fire = self._smp_ctp_fire()
            if not smp_fire:
                inputs.pop("smp_label_mask", None)
            if not ctp_fire:
                inputs.pop("ctp_pair_mask", None)

        # An active regression or contrastive objective can arrive with no
        # per-position targets: OG2 has no phyloP/CADD fields, and CPL/Track-B/SMP
        # may be the only configured objective. Keep the full path in either case
        # so contrastive terms are added and regression heads get their DDP dummies.
        _contrastive_on = (
            float(getattr(self, "_lambda_cpl", 0.0)) > 0
            or float(getattr(self, "_lambda_trackb", 0.0)) > 0
            or float(getattr(self, "_lambda_smp", 0.0)) > 0
            or float(getattr(self, "_lambda_ctp", 0.0)) > 0
        )
        if not targets and not _contrastive_on and not self._aux_tasks:
            return super().compute_loss(  # type: ignore[misc]
                model, inputs, return_outputs=return_outputs, **kwargs
            )

        kwargs.pop("return_outputs", None)
        result = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, **kwargs
        )
        mlm_loss, outputs = result if isinstance(result, tuple) else (result, None)

        aux_preds = getattr(outputs, "aux_preds", None)
        # Only add aux terms during training, so reported eval_loss stays MLM-only.
        if aux_preds is None or not getattr(model, "training", True):
            return (mlm_loss, outputs) if return_outputs else mlm_loss

        device = mlm_loss.device
        bsz = (
            next(iter(targets.values())).shape[0]
            if targets
            else int(inputs["input_ids"].shape[0])
        )
        w = _per_example_weight(
            is_ref,
            sample_weight,
            use_sample_weight=getattr(self, "_use_sample_weight", False),
            transform=getattr(self, "_sample_weight_transform", "sqrt"),
            clip=getattr(self, "_sample_weight_clip", 8.0),
            ref_loss_weight=getattr(self, "_ref_loss_weight", 1.0),
            batch_size=bsz,
            device=device,
        )

        total = mlm_loss
        logs: dict[str, float] = {}
        # Compute the predict-the-mean floor (pred≡standardized mean) only on log
        # steps — it's logged beside each aux loss so a plateau reads as "beats
        # baseline" rather than "stuck" (the Phase-1 misread).
        do_log = self._aux_should_log()
        for name in self._aux_tasks:
            pred = aux_preds.get(name)
            tgt = targets.get(name)
            if pred is None:
                continue
            if tgt is None:
                # OG2 batches have no diploid regression targets. Preserve their
                # MLM-only loss while keeping this head in the graph; otherwise
                # DDP with find_unused_parameters=False aborts on the next step.
                total = total + pred.sum() * 0.0
                continue
            lam = float(self._aux_weights.get(name, 0.0))
            mean, std = self._aux_standardize.get(name, (0.0, 1.0))
            aux_loss, n = _masked_weighted_regression(
                pred,
                tgt.to(pred.device),
                w,
                mean,
                std,
                loss_type=self._aux_loss_type,
                beta=self._aux_betas.get(name, self._aux_huber_beta),
            )
            total = total + lam * aux_loss
            logs[f"loss/{name}"] = float(aux_loss.detach())
            logs[f"count/{name}_targets"] = float(n)
            if do_log:
                floor, _ = _masked_weighted_regression(
                    torch.zeros_like(pred),
                    tgt.to(pred.device),
                    w,
                    mean,
                    std,
                    loss_type=self._aux_loss_type,
                    beta=self._aux_betas.get(name, self._aux_huber_beta),
                )
                logs[f"loss/{name}_floor"] = float(floor.detach())

        # Track-A bottleneck heads: reuse the (already-popped) full-res target,
        # pool it down to the bottleneck-head's resolution (peak-preserving per
        # task), and score it the same way. Same standardize/beta as the full-res
        # task; a separate weight so the two depths are tunable independently.
        bneck_pool = getattr(self, "_aux_bottleneck_pool", {})
        bneck_w = getattr(self, "_aux_bottleneck_weights", {})
        for name in getattr(self, "_aux_bottleneck", ()):
            pred = aux_preds.get(f"{name}@bneck")
            tgt = targets.get(name)
            if pred is None or tgt is None:
                continue
            stride = tgt.shape[1] // pred.shape[1]
            tgt_pooled = _pool_aux_targets(
                tgt.to(pred.device), stride, mode=bneck_pool.get(name, "mean")
            )
            lam = float(bneck_w.get(name, 0.0))
            mean, std = self._aux_standardize.get(name, (0.0, 1.0))
            aux_loss, n = _masked_weighted_regression(
                pred,
                tgt_pooled,
                w,
                mean,
                std,
                loss_type=self._aux_loss_type,
                beta=self._aux_betas.get(name, self._aux_huber_beta),
            )
            total = total + lam * aux_loss
            logs[f"loss/{name}@bneck"] = float(aux_loss.detach())
            logs[f"count/{name}@bneck_targets"] = float(n)
            if do_log:
                floor, _ = _masked_weighted_regression(
                    torch.zeros_like(pred),
                    tgt_pooled,
                    w,
                    mean,
                    std,
                    loss_type=self._aux_loss_type,
                    beta=self._aux_betas.get(name, self._aux_huber_beta),
                )
                logs[f"loss/{name}@bneck_floor"] = float(floor.detach())

        # Phase-2 contrastive terms (raw scalars from the model; 0-valued dummy on
        # non-firing steps). Constant lambda → a real term 1/N steps == step-dropout.
        cpl = getattr(outputs, "cpl_loss", None)
        if cpl is not None:
            lam = float(getattr(self, "_lambda_cpl", 0.0))
            total = total + lam * cpl
            logs["loss/cpl"] = float(cpl.detach())
        trackb = getattr(outputs, "trackb_loss", None)
        if trackb is not None:
            lam = float(getattr(self, "_lambda_trackb", 0.0))
            total = total + lam * trackb
            logs["loss/trackb"] = float(trackb.detach())
        # SMP supervised masked-phase term (a head only — no extra backbone pass;
        # _smp_every gates it as step-dropout, via the smp_label_mask pop above).
        smp = getattr(outputs, "smp_loss", None)
        if smp is not None:
            lam = float(getattr(self, "_lambda_smp", 0.0))
            total = total + lam * smp
            logs["loss/smp"] = float(smp.detach())
        # CTP supervised cis/trans pairwise term (a head only — no extra backbone pass;
        # _ctp_every gates it as step-dropout, via the ctp_pair_mask pop above).
        ctp = getattr(outputs, "ctp_loss", None)
        if ctp is not None:
            lam = float(getattr(self, "_lambda_ctp", 0.0))
            total = total + lam * ctp
            logs["loss/ctp"] = float(ctp.detach())

        if logs and do_log:
            logs["loss/mlm"] = float(mlm_loss.detach())
            self.log(logs)  # type: ignore[attr-defined]
        return (total, outputs) if return_outputs else total

    def _contrastive_fire(self) -> tuple[bool, bool]:
        """(cpl_fire, trackb_fire) for this step. Each objective runs every Nth
        step (N=_cpl_every/_trackb_every); Track-B is offset so the two extra
        backbone passes never stack on one step (≤1 extra pass/step)."""
        step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        cpl_every = max(1, int(getattr(self, "_cpl_every", 3)))
        tb_every = max(1, int(getattr(self, "_trackb_every", 3)))
        tb_off = int(getattr(self, "_trackb_offset", 1)) % tb_every
        cpl_fire = (
            float(getattr(self, "_lambda_cpl", 0.0)) > 0 and step % cpl_every == 0
        )
        tb_fire = (
            float(getattr(self, "_lambda_trackb", 0.0)) > 0
            and step % tb_every == tb_off
        )
        return cpl_fire, tb_fire

    def _smp_ctp_fire(self) -> tuple[bool, bool]:
        """(smp_fire, ctp_fire) for this step (N=_smp_every/_ctp_every). Both heads
        are head-only (no extra backbone pass), so this is step-dropout on the
        objective itself, not a compute saving."""
        step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        smp_every = max(1, int(getattr(self, "_smp_every", 1)))
        ctp_every = max(1, int(getattr(self, "_ctp_every", 1)))
        smp_fire = (
            float(getattr(self, "_lambda_smp", 0.0)) > 0 and step % smp_every == 0
        )
        ctp_fire = (
            float(getattr(self, "_lambda_ctp", 0.0)) > 0 and step % ctp_every == 0
        )
        return smp_fire, ctp_fire

    def _aux_should_log(self) -> bool:
        state = getattr(self, "state", None)
        if state is None or not getattr(state, "is_world_process_zero", False):
            return False
        step = state.global_step
        if step <= 0:
            return False
        logging_steps = max(int(self.args.logging_steps or 1), 1)  # type: ignore[attr-defined]
        return step % logging_steps == 0


class EmbeddingsNoWDOptimizerMixin:
    """Trainer mixin: exclude embeddings + LayerNorm + biases from weight decay.

    Standard practice for continued pretraining (LLaMA, BERT recipes). Weight
    decay pulls all parameters toward zero, which directly opposes the learning
    goal for new vocab rows that need to MOVE substantially during continued
    pretraining. Excluding LayerNorms and biases is also standard.
    """

    def create_optimizer(self):  # type: ignore[override]
        if getattr(self, "optimizer", None) is not None:
            return self.optimizer  # type: ignore[attr-defined]

        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names

        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)  # type: ignore[attr-defined]
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        # Additional exclusions: input embeddings + output LM head (any param
        # reachable from the model's get_input/output_embeddings()).
        embedding_param_ids: set[int] = {
            id(p)
            for module in _vocab_embedding_modules(self.model)  # type: ignore[attr-defined]
            for p in module.parameters()
        }

        # Aux regression heads (AuxHeadsModel.heads) get a dedicated, higher-LR
        # group — they start from random init while the backbone is near-
        # converged. Head weights get weight decay; head biases do not.
        aux_head_lr = getattr(self, "_aux_head_lr", None)
        aux_param_ids: set[int] = set()
        if aux_head_lr is not None:
            for mod_name in (
                "heads",
                "bottleneck_heads",
                "cpl_projector",
                "trackb_projector",
            ):
                mod = getattr(self.model, mod_name, None)  # type: ignore[attr-defined]
                if isinstance(mod, nn.Module):
                    for p in mod.parameters():
                        aux_param_ids.add(id(p))

        decay_named, nodecay_named = [], []
        aux_decay, aux_nodecay = [], []
        # Tensors whose gradient is only PARTLY masked (see _mark_nodecay): decay
        # would move the masked half regardless of the hook.
        extra_nodecay_ids: set[int] = getattr(
            self.model, "_extra_nodecay_param_ids", set()
        )  # type: ignore[attr-defined]
        for n, p in self.model.named_parameters():  # type: ignore[attr-defined]
            if not p.requires_grad:
                continue
            if id(p) in aux_param_ids:
                if "bias" in n or n not in decay_parameters:
                    aux_nodecay.append((n, p))
                else:
                    aux_decay.append((n, p))
            elif (
                id(p) in embedding_param_ids
                or id(p) in extra_nodecay_ids
                or n not in decay_parameters
            ):
                nodecay_named.append((n, p))
            else:
                decay_named.append((n, p))

        optimizer_grouped_parameters = [
            {
                "params": [p for _, p in decay_named],
                "weight_decay": self.args.weight_decay,  # type: ignore[attr-defined]
            },
            {
                "params": [p for _, p in nodecay_named],
                "weight_decay": 0.0,
            },
        ]
        if aux_decay:
            optimizer_grouped_parameters.append(
                {
                    "params": [p for _, p in aux_decay],
                    "weight_decay": self.args.weight_decay,  # type: ignore[attr-defined]
                    "lr": float(aux_head_lr),
                }
            )
        if aux_nodecay:
            optimizer_grouped_parameters.append(
                {
                    "params": [p for _, p in aux_nodecay],
                    "weight_decay": 0.0,
                    "lr": float(aux_head_lr),
                }
            )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )  # type: ignore[attr-defined]
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)  # type: ignore[attr-defined]

        logger.info(
            "Optimizer: %d params w/ WD=%.3g, %d params w/o WD (embeddings+LN+bias)%s.",
            len(decay_named),
            self.args.weight_decay,  # type: ignore[attr-defined]
            len(nodecay_named),
            (
                f"; {len(aux_decay)}+{len(aux_nodecay)} aux-head params @ lr={aux_head_lr:.1e}"
                if aux_param_ids
                else ""
            ),
        )
        return self.optimizer  # type: ignore[attr-defined]


class PerClassLossLoggingMixin:
    """Trainer mixin: every ``logging_steps`` log per-token-class MLM loss
    (``loss/acgt_mlm``, ``loss/diploid_mlm``, ``loss/marker_mlm``) to wandb.

    Diagnoses whether the new-token loss is dominating, plateauing, or rising
    while base ACGT loss drifts during continued pretraining.
    """

    _cached_marker_ids: set[int] | None = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        labels = inputs.get("labels")
        kwargs.pop("return_outputs", None)
        result = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, **kwargs
        )
        if isinstance(result, tuple):
            loss, outputs = result
        else:
            loss, outputs = result, None

        if (
            outputs is not None
            and labels is not None
            and getattr(outputs, "logits", None) is not None
            and self._should_log_per_class()
        ):
            self._log_per_class_loss(outputs.logits, labels)

        return (loss, outputs) if return_outputs else loss

    def _should_log_per_class(self) -> bool:
        if not self.state.is_world_process_zero:  # type: ignore[attr-defined]
            return False
        step = self.state.global_step  # type: ignore[attr-defined]
        if step <= 0:
            return False
        logging_steps = max(int(self.args.logging_steps or 1), 1)  # type: ignore[attr-defined]
        return step % logging_steps == 0

    def _get_marker_ids(self) -> set[int]:
        if self._cached_marker_ids is not None:
            return self._cached_marker_ids
        tokenizer = getattr(self, "processing_class", None) or getattr(
            self, "tokenizer", None
        )
        if tokenizer is None:
            self._cached_marker_ids = set()
            return self._cached_marker_ids
        self._cached_marker_ids = _marker_ids(tokenizer)
        if not self._cached_marker_ids:
            logger.warning(
                "Per-class loss logging: resolved 0 marker ids from a %d-token vocab; "
                "loss/marker_mlm will not be logged and markers will be counted as "
                "diploid.",
                len(merged_vocab(tokenizer)),
            )
        return self._cached_marker_ids

    def _log_per_class_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        with torch.no_grad():
            base_vocab_size = getattr(self.model.config, "base_vocab_size", None)  # type: ignore[attr-defined]
            marker_ids = self._get_marker_ids()

            logits_flat = logits.view(-1, logits.size(-1)).float()
            labels_flat = labels.view(-1)
            losses = F.cross_entropy(
                logits_flat, labels_flat, reduction="none", ignore_index=-100
            )

            valid = labels_flat != -100
            is_marker = torch.zeros_like(labels_flat, dtype=torch.bool)
            for mid in marker_ids:
                is_marker |= labels_flat == mid
            is_marker &= valid

            if base_vocab_size is not None:
                is_diploid = (labels_flat >= base_vocab_size) & valid & ~is_marker
            else:
                is_diploid = torch.zeros_like(is_marker)

            is_acgt = valid & ~is_marker & ~is_diploid

            log_dict: dict[str, float] = {}
            for name, mask in (
                ("acgt", is_acgt),
                ("diploid", is_diploid),
                ("marker", is_marker),
            ):
                cnt = int(mask.sum().item())
                if cnt > 0:
                    log_dict[f"loss/{name}_mlm"] = float(
                        (losses * mask.float()).sum().item() / cnt
                    )
            if log_dict:
                self.log(log_dict)  # type: ignore[attr-defined]


class PostMLMTrainer(Trainer):
    """Trainer for _post models: injects constant human species_ids, MLM-only loss."""

    def __init__(self, human_species_id: int, **kwargs: Any):
        super().__init__(**kwargs)
        self._human_species_id = human_species_id

    def _inject_species(self, inputs: dict) -> dict:
        """Add species_ids to inputs dict (in-place)."""
        ids = inputs["input_ids"]
        inputs["species_ids"] = ids.new_full((ids.shape[0],), self._human_species_id)
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        self._inject_species(inputs)
        out = model(**inputs)
        loss = F.cross_entropy(
            out.logits.view(-1, out.logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        return (loss, out) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        self._inject_species(inputs)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def _make_trainer(is_post: bool, human_species_id: int | None, **kwargs) -> Trainer:
    eval_collator = kwargs.pop("eval_data_collator", None)
    og2_cfg = kwargs.pop("og2_cfg", None)
    ref_sampler_weight = float(kwargs.pop("ref_sampler_weight", 1.0))
    ref_loss_weight = float(kwargs.pop("ref_loss_weight", 1.0))
    use_sample_weight = bool(kwargs.pop("use_sample_weight", False))
    sample_weight_transform = str(kwargs.pop("sample_weight_transform", "sqrt"))
    sample_weight_clip = float(kwargs.pop("sample_weight_clip", 8.0))
    aux_tasks = tuple(kwargs.pop("aux_tasks", ()))
    aux_weights = dict(kwargs.pop("aux_weights", {}))
    aux_standardize = dict(kwargs.pop("aux_standardize", {}))
    aux_loss_type = str(kwargs.pop("aux_loss_type", "huber"))
    aux_huber_beta = float(kwargs.pop("aux_huber_beta", 1.0))
    aux_betas = dict(kwargs.pop("aux_betas", {}))
    aux_bottleneck = tuple(kwargs.pop("aux_bottleneck", ()))
    aux_bottleneck_weights = dict(kwargs.pop("aux_bottleneck_weights", {}))
    aux_bottleneck_pool = dict(kwargs.pop("aux_bottleneck_pool", {}))
    aux_head_lr = kwargs.pop("aux_head_lr", None)
    lambda_cpl = float(kwargs.pop("lambda_cpl", 0.0))
    cpl_every = int(kwargs.pop("cpl_every", 3))
    lambda_trackb = float(kwargs.pop("lambda_trackb", 0.0))
    trackb_every = int(kwargs.pop("trackb_every", 3))
    lambda_smp = float(kwargs.pop("lambda_smp", 0.0))
    smp_every = int(kwargs.pop("smp_every", 1))
    lambda_ctp = float(kwargs.pop("lambda_ctp", 0.0))
    ctp_every = int(kwargs.pop("ctp_every", 1))
    token_budget = int(kwargs.pop("token_budget", 0) or 0)
    if is_post:
        trainer: Trainer = _PostMLMTrainerWithEval(
            human_species_id=human_species_id, **kwargs
        )
    else:
        trainer = _TrainerWithEval(**kwargs)
    trainer._eval_data_collator = eval_collator  # type: ignore[attr-defined]
    trainer._og2_cfg = og2_cfg  # type: ignore[attr-defined]
    trainer._ref_sampler_weight = ref_sampler_weight  # type: ignore[attr-defined]
    trainer._ref_loss_weight = ref_loss_weight  # type: ignore[attr-defined]
    trainer._use_sample_weight = use_sample_weight  # type: ignore[attr-defined]
    trainer._sample_weight_transform = sample_weight_transform  # type: ignore[attr-defined]
    trainer._sample_weight_clip = sample_weight_clip  # type: ignore[attr-defined]
    trainer._aux_tasks = aux_tasks  # type: ignore[attr-defined]
    trainer._aux_weights = aux_weights  # type: ignore[attr-defined]
    trainer._aux_standardize = aux_standardize  # type: ignore[attr-defined]
    trainer._aux_loss_type = aux_loss_type  # type: ignore[attr-defined]
    trainer._aux_huber_beta = aux_huber_beta  # type: ignore[attr-defined]
    trainer._aux_betas = aux_betas  # type: ignore[attr-defined]
    trainer._aux_bottleneck = aux_bottleneck  # type: ignore[attr-defined]
    trainer._aux_bottleneck_weights = aux_bottleneck_weights  # type: ignore[attr-defined]
    trainer._aux_bottleneck_pool = aux_bottleneck_pool  # type: ignore[attr-defined]
    trainer._aux_head_lr = (  # type: ignore[attr-defined]
        float(aux_head_lr) if aux_head_lr is not None else None
    )
    trainer._lambda_cpl = lambda_cpl  # type: ignore[attr-defined]
    trainer._cpl_every = cpl_every  # type: ignore[attr-defined]
    trainer._lambda_trackb = lambda_trackb  # type: ignore[attr-defined]
    trainer._trackb_every = trackb_every  # type: ignore[attr-defined]
    trainer._trackb_offset = 1  # type: ignore[attr-defined]
    trainer._lambda_smp = lambda_smp  # type: ignore[attr-defined]
    trainer._smp_every = smp_every  # type: ignore[attr-defined]
    trainer._lambda_ctp = lambda_ctp  # type: ignore[attr-defined]
    trainer._ctp_every = ctp_every  # type: ignore[attr-defined]
    trainer._token_budget = token_budget  # type: ignore[attr-defined]
    trainer.model_accepts_loss_kwargs = False
    return trainer



class L2SPAnchor:
    """Decay toward the pretrained weights (theta0) instead of toward zero.

    Standard weight decay is wrong by construction for continued pretraining. It
    pulls every parameter toward 0, but the solution worth preserving is theta0 --
    and on this project the damage tracks drift FROM theta0: at 5552 steps `bundle`
    (lr 1e-4, wd 5e-3) drifts 0.0827 and loses gue_splice 0.9387 -> 0.8510, while
    `dipgentle` (lr 2e-5, wd 0) drifts 0.0162 and holds 0.9284. AdamW's decoupled
    decay is the one force in the optimizer guaranteed to point AWAY from what we
    are trying not to forget.

    L2-SP replaces ||theta||^2 with ||theta - theta0||^2. Applied DECOUPLED, exactly
    mirroring AdamW's decoupled weight decay, so it composes with any optimizer and
    costs one fused op per parameter per step rather than a second backward:

        theta <- theta + lr * lambda * (theta0 - theta)

    Frozen parameters are never anchored: a slice-freeze elsewhere would otherwise be
    silently nudged, the same class of bug as decaying a gradient-masked tensor.

    theta0 is held on-device (~0.5 GB at 100M params, fp32).
    """

    def __init__(self, model: Any, lam: float) -> None:
        self.lam = float(lam)
        self.params = [p for p in model.parameters() if p.requires_grad]
        # theta0 is cloned by capture(), NOT here. The Trainer moves the model to the
        # accelerator after this object is built, and `nn.Module.to()` rebinds each
        # Parameter's .data in place -- so self.params stays valid, but a clone taken
        # now would still be on CPU and `p.add_(p0 - p)` would raise a device mismatch.
        self.theta0: list[torch.Tensor] | None = None

    def capture(self) -> None:
        """Snapshot theta0 from the parameters as they currently live.

        Called from ``L2SPCallback.on_train_begin``, i.e. after device placement and
        any DDP wrap, and after a resume has restored weights.
        """
        self.theta0 = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def step(self, lr: float) -> None:
        if self.lam <= 0.0:
            return
        if self.theta0 is None:
            # Safety net for a caller that skipped on_train_begin: one step of drift
            # in the anchor beats silently training with no anchor at all.
            self.capture()
        for p, p0 in zip(self.params, self.theta0):
            p.add_(p0 - p, alpha=lr * self.lam)


class LengthBucketMixin:
    """Batch by a BASE budget within one length bucket, instead of by row count.

    Off unless ``_token_budget`` is set, in which case the fixed-length path is
    reached through ``super()`` untouched and every existing run script reproduces
    byte-identically.

    This overrides ``_get_dataloader`` rather than ``get_train_dataloader`` because
    HF funnels both through the former, and the only thing that has to change is
    ``batch_size`` + ``sampler`` -> ``batch_sampler``; everything else (collator,
    workers, pin_memory, prefetch, worker seeding, ``accelerator.prepare``) is
    copied from ``transformers.Trainer._get_dataloader`` verbatim so the two paths
    cannot drift on anything but the batching rule.

    Ordering note: this must sit AFTER ``Og2ReplayMixin`` in the MRO. OG2 replay
    wraps whatever ``super().get_train_dataloader()`` returns, so the bucketed human
    loader is what it wraps.
    """

    _token_budget: int = 0

    def _get_dataloader(  # type: ignore[override]
        self,
        dataset,
        description: str,
        batch_size: int,
        sampler_fn=None,
        is_training: bool = False,
        dataloader_key: str | None = None,
    ):
        budget = int(getattr(self, "_token_budget", 0) or 0)
        bucket_len = getattr(dataset, "_bucket_len", None)
        if not (is_training and budget and bucket_len is not None):
            return super()._get_dataloader(  # type: ignore[misc]
                dataset,
                description,
                batch_size,
                sampler_fn=sampler_fn,
                is_training=is_training,
                dataloader_key=dataloader_key,
            )

        from functools import partial

        from torch.utils.data import DataLoader, RandomSampler
        from transformers.trainer_utils import seed_worker

        base_sampler = (
            sampler_fn(dataset) if sampler_fn is not None else RandomSampler(dataset)
        )
        batch_sampler = LengthBucketBatchSampler(
            base_sampler,
            bucket_len,
            budget,
            drop_last=bool(self.args.dataloader_drop_last),  # type: ignore[attr-defined]
        )
        realized = batch_sampler.mean_bases_per_batch()
        logger.info(
            "Length bucketing ON: budget %d bases/micro-batch, %d buckets (%s), "
            "%d batches/epoch, rows/batch %d..%d, REALIZED %.0f bases/batch "
            "(%.3f of budget -- derive --genome-steps from the realized number)",
            budget,
            len(batch_sampler._cap),
            ",".join(str(b) for b in sorted(batch_sampler._cap)),
            len(batch_sampler),
            min(batch_sampler._cap.values()),
            max(batch_sampler._cap.values()),
            realized,
            realized / budget,
        )
        # len(batch_sampler) is GLOBAL (accelerate shards it below), so this is the
        # true epoch. --epochs is decorative under --max-steps: HF keeps going past
        # the epoch boundary, and every budgeted run in this repo is a FRACTION of one
        # (the 1e-1 tsstes run: 13.1%). Say so where the horizon is set, so nobody
        # reads "--epochs 1" as "one pass over the data" again.
        # getattr: unit tests drive this with a bare SimpleNamespace for args.
        ga = int(getattr(self.args, "gradient_accumulation_steps", 1))  # type: ignore[attr-defined]
        ws = int(getattr(self.args, "world_size", 1))  # type: ignore[attr-defined]
        steps_per_epoch = max(1, len(batch_sampler) // (ga * ws))
        max_steps = int(getattr(self.args, "max_steps", 0))  # type: ignore[attr-defined]
        if max_steps > 0:
            logger.info(
                "EPOCH BUDGET: max_steps %d = %.1f%% of one epoch (%d optimizer steps "
                "at accum %d x %d ranks, %.3f Tbase/epoch). --epochs is NOT the horizon.",
                max_steps,
                100.0 * max_steps / steps_per_epoch,
                steps_per_epoch,
                ga,
                ws,
                len(batch_sampler) * realized / 1e12,
            )

        collator = self._get_collator_with_removed_columns(  # type: ignore[attr-defined]
            self.data_collator, description=description  # type: ignore[attr-defined]
        )
        args = self.args  # type: ignore[attr-defined]
        params = {
            "batch_sampler": batch_sampler,
            "collate_fn": collator,
            "num_workers": args.dataloader_num_workers,
            "pin_memory": args.dataloader_pin_memory,
            "persistent_workers": args.dataloader_persistent_workers,
            "multiprocessing_context": args.dataloader_multiprocessing_context,
            "worker_init_fn": partial(
                seed_worker,
                num_workers=args.dataloader_num_workers,
                rank=args.process_index,
            ),
        }
        if args.dataloader_num_workers > 0:
            params["prefetch_factor"] = args.dataloader_prefetch_factor
        return self.accelerator.prepare(DataLoader(dataset, **params))  # type: ignore[attr-defined]


class Og2ReplayMixin:
    """Substitute a whole OpenGenome2 batch on every Nth optimizer step.

    Set up by ``_make_trainer`` from ``og2_cfg``; a falsy/zero-fraction config makes
    every method here a pass-through, so the control arm is byte-identical.

    The human dataloader is built by ``super().get_train_dataloader()`` and never
    touched -- replay REPLACES whole optimizer steps rather than mixing sources
    inside a tensor or inside an accumulation window.
    """

    _og2_cfg: dict | None = None
    _og2_mixed = None
    _cur_og2_bin: int | None = None

    def get_train_dataloader(self):  # type: ignore[override]
        dl = super().get_train_dataloader()
        cfg = self._og2_cfg
        if not cfg or cfg.get("fraction", 0.0) <= 0.0:
            return dl
        from og2_mix import MixedReplayLoader, build_og2_loaders

        loaders = build_og2_loaders(
            cfg["manifest"],
            cfg["char_to_id"],
            self.data_collator,
            cfg["human_batch"],
            cfg["human_len"],
            rank=int(self.args.process_index),
            world_size=int(self.args.world_size),
            num_workers=int(cfg.get("workers", 2)),
            seed=int(cfg.get("seed", 42)),
        )
        self._og2_mixed = MixedReplayLoader(
            dl,
            loaders,
            step_fn=lambda: self.state.global_step,
            seed=int(cfg.get("seed", 42)),
            fraction=float(cfg["fraction"]),
        )
        self._og2_loss = {"human": [0.0, 0], "og2": [0.0, 0]}
        logger.info(
            "OG2 replay ON: fraction=%.3f (every %d-th optimizer step), bins %s, "
            "token-matched to the human batch (%d x %d = %d tokens), species_id=%s, "
            "rank %d/%d, %d workers per bin.",
            cfg["fraction"],
            round(1.0 / cfg["fraction"]),
            sorted(loaders),
            cfg["human_batch"],
            cfg["human_len"],
            cfg["human_batch"] * cfg["human_len"],
            cfg.get("species_id"),
            int(self.args.process_index),
            int(self.args.world_size),
            int(cfg.get("workers", 2)),
        )
        return self._og2_mixed

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        # remove_unused_columns=False, so a stray key would reach the model and raise.
        _bin = inputs.pop("og2_bin", None)
        self._cur_og2_bin = _bin
        try:
            out = super().compute_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )
        finally:
            # MUST clear: prediction_step also calls _inject_species, and a stale
            # value there would silently score eval batches under the OG2 species id
            # instead of human. Only --no-eval hides this today.
            self._cur_og2_bin = None
        if self._og2_cfg and getattr(self, "_og2_loss", None) is not None:
            loss = out[0] if isinstance(out, tuple) else out
            slot = self._og2_loss["og2" if _bin is not None else "human"]
            slot[0] += float(loss.detach())
            slot[1] += 1
        return out

    def _inject_species(self, inputs: dict) -> dict:  # type: ignore[override]
        """_post ONLY. `_pre` models take no species_ids and never call this.

        Defined unconditionally because the mixin sits ahead of PostMLMTrainer in
        the MRO; the super() call is what would fail on a `_pre` trainer, and it is
        unreachable there because nothing in the `_pre` path calls _inject_species.
        The getattr guard makes that explicit rather than relying on it.

        OG2 records carry no usable taxon after segmentation, and <unk>=0 is not
        available (the core asserts species_ids >= num_species_special_tokens = 6),
        so the replay id is a recorded experimental variable (--og2-species-id),
        not a detail. See pretrain/tools/og2_species_probe.py.
        """
        parent = getattr(super(), "_inject_species", None)
        if parent is None:  # _pre model: no species conditioning at all
            return inputs
        inputs = parent(inputs)
        sp = (self._og2_cfg or {}).get("species_id")
        if self._cur_og2_bin is not None and sp is not None:
            inputs["species_ids"].fill_(int(sp))
        return inputs

    def log(self, logs: dict, *a, **kw):  # type: ignore[override]
        m = self._og2_mixed
        if m is not None:
            logs.update(m.token_report())
            for k, (tot, n) in getattr(self, "_og2_loss", {}).items():
                if n:
                    logs[f"loss/mlm_{k}"] = tot / n
            self._og2_loss = {"human": [0.0, 0], "og2": [0.0, 0]}
        return super().log(logs, *a, **kw)


class _TrainerWithEval(
    Og2ReplayMixin,
    LengthBucketMixin,
    AuxRegressionLossMixin,
    PerClassLossLoggingMixin,
    EvalCollatorMixin,
    WeightedSamplerMixin,
    SampleWeightedLossMixin,
    EmbeddingsNoWDOptimizerMixin,
    Trainer,
):
    pass


class _PostMLMTrainerWithEval(
    Og2ReplayMixin,
    LengthBucketMixin,
    AuxRegressionLossMixin,
    PerClassLossLoggingMixin,
    EvalCollatorMixin,
    WeightedSamplerMixin,
    SampleWeightedLossMixin,
    EmbeddingsNoWDOptimizerMixin,
    PostMLMTrainer,
):
    pass


class L2SPCallback(TrainerCallback):
    """Apply the L2-SP pull after every optimizer step, at the live learning rate.

    on_step_end fires after `optimizer.step()`, which is where a DECOUPLED penalty
    belongs -- the same place AdamW applies its own weight decay. Reading the LR from
    the scheduler rather than args matters: under `constant_with_warmup` the first
    3000 steps run at a fraction of the nominal rate, and an anchor applied at the
    nominal rate there would over-pull exactly when the model is most fragile.
    """

    def __init__(self, anchor: "L2SPAnchor") -> None:
        self.anchor = anchor

    def on_train_begin(self, args, state, control, **kw):
        self.anchor.capture()
        return control

    def on_step_end(self, args, state, control, optimizer=None, lr_scheduler=None, **kw):
        lr = None
        if lr_scheduler is not None:
            try:
                lr = float(lr_scheduler.get_last_lr()[0])
            except (AttributeError, IndexError):
                lr = None
        if lr is None and optimizer is not None:
            lr = float(optimizer.param_groups[0].get("lr", args.learning_rate))
        self.anchor.step(lr if lr is not None else float(args.learning_rate))
        return control


class GenomeMilestoneCallback(TrainerCallback):
    """Save model weights at approximate genome-equivalent milestones.

    When ``hub_model_id`` is set, also push each milestone as a separate branch
    on the same private+gated repo.

    When ``benchmark_script`` is set, lazily spawn a background ``mp.Process``
    that runs the external benchmark CLI against each saved milestone (queued,
    non-blocking — the trainer continues stepping). See ``benchmark_bg_runner``
    in ``phase_aware_pretrain/pretrain/``.
    """

    def __init__(
        self,
        milestone_steps: dict[int, list[int]],
        output_dir: Path,
        tokenizer: PreTrainedTokenizerBase,
        hub_model_id: str | None = None,
        branch_prefix: str | None = None,
        benchmark_script: Path | None = None,
        benchmark_cuda_visible_devices: str = "0",
        benchmark_pythonpath: str | None = None,
        wait_for_benchmarks_on_train_end: bool = True,
        stop_after_genome: int | None = None,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.hub_model_id = hub_model_id
        self.branch_prefix = branch_prefix or _hub_branch_prefix(self.output_dir)
        self.milestone_steps = {
            step: sorted(genomes) for step, genomes in sorted(milestone_steps.items())
        }
        if self.milestone_steps:
            logger.info(
                "Milestones configuration: %s",
                {
                    ",".join(str(genome) for genome in genomes): f"~step {step}"
                    for step, genomes in self.milestone_steps.items()
                },
            )

        self.benchmark_script = (
            Path(benchmark_script) if benchmark_script is not None else None
        )
        self.benchmark_cuda_visible_devices = benchmark_cuda_visible_devices
        self.benchmark_pythonpath = benchmark_pythonpath
        self.wait_for_benchmarks_on_train_end = wait_for_benchmarks_on_train_end
        # Stop the run once this milestone is saved. The callback already knows the
        # exact step each genome lands on, so bounding a run in-process is strictly
        # better than the external "poll for the checkpoint file, then SIGTERM"
        # loop the run scripts used to carry: a clean trainer stop still runs
        # on_train_end (benchmark-worker poison pill, wandb flush, dataloader
        # shutdown) and cannot land mid-`_save_checkpoint`, which left a partial
        # checkpoint-<step>/ that the resume logic would then pick up as newest.
        self.stop_after_genome = stop_after_genome
        self._benchmark_queue: Any | None = None
        self._benchmark_process: Any | None = None
        self._benchmark_worker_fn: Any | None = None

    @staticmethod
    def _is_rank_zero() -> bool:
        return (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

    def _load_benchmark_worker(self):
        """Lazily import ``benchmark_worker`` from the directory of benchmark_script."""
        if self._benchmark_worker_fn is not None:
            return self._benchmark_worker_fn
        import importlib.util

        worker_path = self.benchmark_script.parent / "benchmark_bg_runner.py"
        if not worker_path.exists():
            raise FileNotFoundError(
                f"Expected benchmark_bg_runner.py next to {self.benchmark_script}, "
                f"got {worker_path}"
            )
        spec = importlib.util.spec_from_file_location(
            "benchmark_bg_runner_loaded", str(worker_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {worker_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._benchmark_worker_fn = module.benchmark_worker
        return self._benchmark_worker_fn

    def _start_benchmark_worker_if_needed(self):
        if self.benchmark_script is None or self._benchmark_process is not None:
            return
        import multiprocessing as mp

        worker_fn = self._load_benchmark_worker()
        log_dir = self.output_dir / "benchmark-logs"
        self._benchmark_queue = mp.Queue()
        self._benchmark_process = mp.Process(
            target=worker_fn,
            args=(
                self._benchmark_queue,
                str(self.benchmark_script),
                self.benchmark_cuda_visible_devices,
                str(log_dir),
                self.benchmark_pythonpath,
            ),
            daemon=not self.wait_for_benchmarks_on_train_end,
        )
        self._benchmark_process.start()
        logger.info(
            "Started benchmark worker (pid=%s, script=%s, cuda=%s)",
            self._benchmark_process.pid,
            self.benchmark_script,
            self.benchmark_cuda_visible_devices,
        )

    def _queue_benchmark(self, checkpoint_dir: Path, genome: int, step: int):
        if self.benchmark_script is None:
            return
        self._start_benchmark_worker_if_needed()
        self._benchmark_queue.put((str(checkpoint_dir), genome, step))
        logger.info(
            "Queued benchmark for genome-%d at step %d: %s",
            genome,
            step,
            checkpoint_dir,
        )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        genomes = self.milestone_steps.pop(state.global_step, None)
        if genomes is None or model is None:
            return
        # BEFORE the rank-zero return, on purpose: every rank pops the same step,
        # so every rank must flip the flag and leave the loop together. Stopping
        # on rank 0 only hangs the others on the next collective.
        if self.stop_after_genome is not None and max(genomes) >= self.stop_after_genome:
            control.should_training_stop = True
            logger.info(
                "Reached genome-%d (step %d); stopping training.",
                self.stop_after_genome,
                state.global_step,
            )
        if not self._is_rank_zero():
            return
        for genome in genomes:
            save_dir = self.output_dir / f"genome-{genome}"
            _atomic_save_dir(
                save_dir,
                lambda target_dir: _save_model_artifacts(
                    model, self.tokenizer, target_dir
                ),
            )
            logger.info(
                "Saved genome-%d checkpoint -> %s (step %d)",
                genome,
                save_dir,
                state.global_step,
            )
            if self.hub_model_id:
                _push_folder_as_branch(
                    self.hub_model_id,
                    save_dir,
                    branch=f"{self.branch_prefix}-genome-{genome}",
                )
            self._queue_benchmark(save_dir, genome, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if not self._is_rank_zero():
            return
        if self._benchmark_queue is None or self._benchmark_process is None:
            return
        self._benchmark_queue.put(None)  # poison pill
        if self.wait_for_benchmarks_on_train_end:
            logger.info("Waiting for queued benchmarks to finish.")
            self._benchmark_process.join()
            logger.info("Benchmark worker finished.")
        else:
            logger.info("Training ended; benchmark worker may continue (daemon=True).")


class ModeCollapseCallback(TrainerCallback):
    """Probe whether new-token embeddings have collapsed onto each other.

    Runs every N steps (default 20_000) on rank 0 only. Computes the maximum
    pairwise cosine similarity over the *new* embedding rows
    (``[base_vocab_size:]``). Logs ``embed/max_new_token_cosine`` to wandb.

    **Off by default** (``--mode-collapse-check``). Decoupled from save
    cadence so it doesn't fire on every checkpoint with frequent saves on
    small models. Never aborts training — collapse is logged as a metric.
    """

    def __init__(
        self,
        enabled: bool = False,
        interval_steps: int = 20_000,
        base_vocab_size: int | None = None,
    ) -> None:
        self.enabled = enabled
        self.interval_steps = max(1, interval_steps)
        self.base_vocab_size = base_vocab_size
        self._last_logged_step = -1

    @staticmethod
    def _is_rank_zero() -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self.enabled or not self._is_rank_zero() or model is None:
            return
        step = state.global_step
        if step <= 0 or step == self._last_logged_step:
            return
        if step % self.interval_steps != 0:
            return
        bvs = self.base_vocab_size or getattr(model.config, "base_vocab_size", None)
        if bvs is None:
            return

        with torch.no_grad():
            in_w = model.get_input_embeddings().weight
            new_rows = in_w.data[bvs:].float()
            if new_rows.shape[0] < 2:
                return
            normed = F.normalize(new_rows, dim=-1)
            cos = normed @ normed.T  # (N, N)
            # Zero the diagonal; take the max of the upper triangle.
            cos.fill_diagonal_(0.0)
            max_cos = float(cos.abs().max().item())

        self._last_logged_step = step
        try:
            import wandb  # local import; only when callback is enabled

            if wandb.run is not None:
                wandb.log({"embed/max_new_token_cosine": max_cos}, step=step)
        except ImportError:
            pass
        if max_cos > 0.99:
            logger.warning(
                "[mode-collapse] step=%d max pairwise cosine on new rows = %.4f "
                "(>0.99 suggests collapse).",
                step,
                max_cos,
            )


def _build_phased_pair_diagnostic_ids(
    tokenizer: PreTrainedTokenizerBase,
    tokens_path: Path,
) -> list[tuple[int, int, int, int]]:
    """Return ``[(fwd_id, rev_id, parent_a_id, parent_b_id), ...]`` for reciprocal
    phased pairs (AC↔CA, AG↔GA, ...). Used by the cosine-drift diagnostic.
    """
    try:
        _symbols, sym2pair, sym2order = _parse_diploid_tsv(tokens_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Phased-pair diagnostic: failed to parse %s: %s", tokens_path, exc
        )
        return []

    vocab = {
        **tokenizer.get_vocab(),
        **{
            str(token): token_id
            for token, token_id in getattr(
                tokenizer, "added_tokens_encoder", {}
            ).items()
        },
    }

    # Group phased symbols by parent-pair (unordered), one ordering per pair.
    pair_to_symbols: dict[frozenset, dict[tuple[str, str], str]] = {}
    for symbol, order in sym2order.items():
        if order != "phased" or symbol not in vocab:
            continue
        pair = sym2pair.get(symbol)
        if pair is None or len(pair) != 2:
            continue
        key = frozenset(pair)
        pair_to_symbols.setdefault(key, {})[pair] = symbol

    result: list[tuple[int, int, int, int]] = []
    for orderings in pair_to_symbols.values():
        if len(orderings) != 2:
            continue
        ordered_keys = sorted(orderings.keys())
        fwd_pair, rev_pair = ordered_keys[0], ordered_keys[1]
        parent_a, parent_b = fwd_pair
        if parent_a not in vocab or parent_b not in vocab:
            continue
        result.append(
            (
                vocab[orderings[fwd_pair]],
                vocab[orderings[rev_pair]],
                vocab[parent_a],
                vocab[parent_b],
            )
        )
    return result


class TimedStopCallback(TrainerCallback):
    """Stop training after a wall-clock budget; save a final checkpoint first.

    Used for time-budgeted production runs (e.g., 27h58m for weekend gating
    arms that must finish within a fixed window). Sets ``control.should_save
    = True`` and ``control.should_training_stop = True`` simultaneously on
    the step where the elapsed time exceeds ``max_seconds``. HF Trainer
    processes ``should_save`` before ``should_training_stop`` in the same
    step's tail logic, so the final pre-stop checkpoint is written.

    A final ``final/`` save happens in ``main()`` after ``trainer.train()``
    returns; both checkpoints will be available for downstream comparison.

    PEFT-arm asymmetry: the mid-run ``checkpoint-<step>/`` written by HF
    Trainer's ``_save_checkpoint`` is ADAPTER-ONLY for PeftModel (just
    ``adapter_model.safetensors`` + ``adapter_config.json``). It is NOT a
    standalone HF artifact and ``AutoModelForMaskedLM.from_pretrained()``
    on it will silently load the un-fine-tuned base. Only ``final/`` and
    ``genome-<N>/`` are standard-HF (merged) for LoRA arms — those go
    through ``_save_model_artifacts`` which handles the merge/deepcopy.
    """

    def __init__(self, max_seconds: float):
        super().__init__()
        self.max_seconds = float(max_seconds)
        self.start_time: float | None = None
        self._fired = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.monotonic()
        logger.info(
            "TimedStopCallback armed: budget = %.1f hours.",
            self.max_seconds / 3600.0,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None or self._fired:
            return control
        elapsed = time.monotonic() - self.start_time
        if elapsed > self.max_seconds:
            self._fired = True
            control.should_save = True
            control.should_training_stop = True
            logger.warning(
                "TimedStopCallback: elapsed %.1f h > budget %.1f h at step %d. "
                "Triggering checkpoint save + training stop.",
                elapsed / 3600.0,
                self.max_seconds / 3600.0,
                state.global_step,
            )
        return control


class EmbeddingDiagnosticsCallback(TrainerCallback):
    """Log new-token embedding health to detect collapse / drift regressions.

    Metrics (rank-0 only):
      * ``embed/new_norm_mean``, ``embed/old_norm_mean``: mean L2 row norm.
      * ``embed/new_offdiag_cos``: mean off-diagonal cosine within the new-token
        block (a high value signals collapse, the bug seen in earlier runs).
      * ``embed/new_vs_init_l2``: mean L2 drift since callback init.
      * ``embed/phased_pair_cosine_mean`` and ``..._max``: mean / max cosine
        similarity between reciprocal phased pairs (Á↔Ç, Ã↔Ğ, ...). Should
        stay well below 1.0; values >0.98 mean the model lost the phase
        distinction. Emits a logger.warning at >0.98.
      * ``embed/phased_axis_separation_mean``: mean |proj_fwd - proj_rev|
        along the parent-axis E[parent_a] - E[parent_b]. Initialised at
        ±_PHASE_AXIS_DELTA_FRAC; warns if it drifts to ~0.
    """

    def __init__(
        self,
        base_vocab_size: int,
        log_every_n_steps: int = 0,
        phased_pair_ids: list[tuple[int, int, int, int]] | None = None,
    ):
        super().__init__()
        self.base_vocab_size = int(base_vocab_size)
        self.log_every_n_steps = int(log_every_n_steps)
        self.phased_pair_ids = phased_pair_ids or []
        self._init_snapshot: torch.Tensor | None = None

    @staticmethod
    def _resolve_wandb_step(run: Any, global_step: int, tag: str) -> int:
        run_step = getattr(run, "step", None)
        if run_step is None:
            run_step = getattr(run, "_step", None)
        try:
            resolved_step = max(int(global_step), int(run_step))
        except (TypeError, ValueError):
            resolved_step = int(global_step)
        if tag == "train_end" and resolved_step == int(global_step):
            resolved_step += 1
        return resolved_step

    @staticmethod
    def _is_rank_zero() -> bool:
        return (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )

    @torch.no_grad()
    def _compute(self, model: Any) -> dict[str, float]:
        emb = model.get_input_embeddings().weight.detach().float().cpu()
        old, new = emb[: self.base_vocab_size], emb[self.base_vocab_size :]
        if new.numel() == 0:
            return {}
        new_norms = new.norm(dim=1)
        unit = new / new_norms.clamp_min(1e-8).unsqueeze(1)
        cos = unit @ unit.T
        n = cos.size(0)
        off_mean = (cos.sum() - cos.diag().sum()) / max(1, n * (n - 1))
        out = {
            "embed/new_norm_mean": float(new_norms.mean()),
            "embed/old_norm_mean": float(old.norm(dim=1).mean()),
            "embed/new_offdiag_cos": float(off_mean),
        }
        if self._init_snapshot is not None and self._init_snapshot.shape == new.shape:
            out["embed/new_vs_init_l2"] = float(
                (new - self._init_snapshot).norm(dim=1).mean()
            )

        # LM head row norms (when untied from input embeddings). Adds a signal
        # for whether WD or LR is pulling new head rows off the trained manifold.
        out_embs = model.get_output_embeddings()
        out_w = getattr(out_embs, "weight", None) if out_embs is not None else None
        if (
            out_w is not None
            and out_w is not model.get_input_embeddings().weight
            and out_w.shape[0] > self.base_vocab_size
        ):
            head = out_w.detach().float().cpu()
            head_old = head[: self.base_vocab_size]
            head_new = head[self.base_vocab_size :]
            if head_new.numel() > 0:
                out["head/new_norm_mean"] = float(head_new.norm(dim=1).mean())
                out["head/old_norm_mean"] = float(head_old.norm(dim=1).mean())
                # New head rows start near a common mean direction (mu +
                # 0.03·sigma·noise on the live NTv3 path). This tracks whether
                # they actually SEPARATE during warmup/training; a flat ~1.0
                # means the rows never differentiated.
                if head_new.shape[0] > 1:
                    hu = head_new / head_new.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    hcos = hu @ hu.T
                    m = hcos.shape[0]
                    out["head/new_offdiag_cos"] = float(
                        (hcos.sum() - hcos.diag().sum()) / max(1, m * (m - 1))
                    )

        # Phased-pair erosion diagnostic. Cheap (microseconds) — operates on
        # ~30 reciprocal pairs.
        if self.phased_pair_ids:
            fwd_ids = torch.tensor([p[0] for p in self.phased_pair_ids])
            rev_ids = torch.tensor([p[1] for p in self.phased_pair_ids])
            pa_ids = torch.tensor([p[2] for p in self.phased_pair_ids])
            pb_ids = torch.tensor([p[3] for p in self.phased_pair_ids])
            fwd = emb[fwd_ids]
            rev = emb[rev_ids]
            pa = emb[pa_ids]
            pb = emb[pb_ids]

            cosines = F.cosine_similarity(fwd, rev, dim=-1)
            mid = 0.5 * (pa + pb)
            axis = pa - pb
            axis_norm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            axis_unit = axis / axis_norm
            proj_fwd = ((fwd - mid) * axis_unit).sum(dim=-1)
            proj_rev = ((rev - mid) * axis_unit).sum(dim=-1)
            separation = (proj_fwd - proj_rev).abs() / axis_norm.squeeze(-1)

            cos_mean = float(cosines.mean())
            cos_max = float(cosines.max())
            out["embed/phased_pair_cosine_mean"] = cos_mean
            out["embed/phased_pair_cosine_max"] = cos_max
            out["embed/phased_axis_separation_mean"] = float(separation.mean())

            if cos_max > 0.98:
                logger.warning(
                    "[phased-pair-diag] max reciprocal cosine = %.4f (>0.98) — "
                    "phase distinction may be collapsing.",
                    cos_max,
                )

        return out

    def _log(self, model: Any, state: Any, tag: str) -> None:
        if not self._is_rank_zero() or model is None:
            return
        metrics = self._compute(model)
        if not metrics:
            return
        logger.info("[embed-diag/%s step=%d] %s", tag, state.global_step, metrics)
        try:
            import wandb

            if wandb.run is not None:
                wandb_step = self._resolve_wandb_step(wandb.run, state.global_step, tag)
                wandb.log(metrics, step=wandb_step)
        except ImportError:
            pass

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            with torch.no_grad():
                self._init_snapshot = (
                    model.get_input_embeddings()
                    .weight[self.base_vocab_size :]
                    .detach()
                    .float()
                    .cpu()
                    .clone()
                )
        self._log(model, state, "train_begin")

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        self._log(model, state, "eval")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._log(model, state, "train_end")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if (
            self.log_every_n_steps > 0
            and state.global_step > 0
            and state.global_step % self.log_every_n_steps == 0
        ):
            self._log(model, state, "step")


class EvalSplitDiffCallback(TrainerCallback):
    """Log ``eval/wt_minus_primary_loss`` after each evaluation.

    Single-scalar forgetting/learning balance: positive means the WT
    (HG38_REF, no variants) split is harder than the primary eval split
    (the --eval-window-filter mode, "all" by default), which is the
    catastrophic-forgetting signature on a vocab-expanded backbone.
    Watches for the Gururangan 2020 / Cui 2023 anchor-distribution failure
    mode where new-vocab CPT erodes the base ACGT manifold.

    Cheap: one float subtraction per eval, one wandb scalar.
    """

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        wt = metrics.get("eval_wt_loss")
        primary = metrics.get("eval_primary_loss")
        if wt is None or primary is None:
            return
        diff = float(wt) - float(primary)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"eval/wt_minus_primary_loss": diff}, step=state.global_step)
        except ImportError:
            pass
        # Always log to the HF metrics dict too so it appears in trainer logs.
        metrics["eval_wt_minus_primary_loss"] = diff


class EvalSubsetAuxCallback(TrainerCallback):
    """Log held-out per-token-class MLM + auxiliary-regression losses each eval.

    The trainer's ``eval_loss`` stays MLM-only (``AuxRegressionLossMixin`` skips
    aux at eval, and per-class logging is train-step gated), so the per-subset and
    phyloP/CADD curves the user wants are not produced inline. This callback runs
    ONE forward-only pass over a capped sample of the *primary* eval split on rank
    0 and logs:

      * ``eval/acgt_mlm`` / ``eval/diploid_mlm`` / ``eval/marker_mlm`` — the same
        token-class split as ``PerClassLossLoggingMixin._log_per_class_loss``.
      * ``eval/phylop`` / ``eval/cadd_phred`` — the standardized Huber from
        ``_masked_weighted_regression`` (comparable to the train ``loss/*`` curves).

    Isolated from the loss path; reuses the frozen eval collator so masking/marker
    policy matches the trainer's own eval. Dedups per ``global_step`` because a dict
    eval set fires ``on_evaluate`` once per split.
    """

    def __init__(
        self,
        eval_dataset,
        data_collator,
        base_vocab_size: int,
        marker_ids: set[int],
        aux_tasks: tuple[str, ...],
        batch_size: int,
        max_windows: int = 8000,
        num_workers: int = 4,
        seed: int = 42,
    ):
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.base_vocab_size = base_vocab_size
        self.marker_ids = {int(m) for m in marker_ids}
        self.aux_tasks = tuple(aux_tasks)
        self.batch_size = int(batch_size)
        self.max_windows = int(max_windows)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self._last_step = -1

    @staticmethod
    def _unwrap(model):
        # Peel DDP (.module) and torch.compile (._orig_mod) wrappers in any order
        # so we run the eager AuxHeadsModel (avoids recompiling for eval shapes).
        for _ in range(4):
            if hasattr(model, "module"):
                model = model.module
            elif hasattr(model, "_orig_mod"):
                model = model._orig_mod
            else:
                break
        return model

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        if model is None or self.eval_dataset is None or self.max_windows <= 0:
            return
        if state.global_step == self._last_step:  # one pass per eval cycle
            return
        self._last_step = state.global_step

        net = self._unwrap(model)
        was_training = net.training
        net.eval()
        device = next(net.parameters()).device

        n = len(self.eval_dataset)
        if n > self.max_windows:
            g = torch.Generator().manual_seed(self.seed)
            idx = torch.randperm(n, generator=g)[: self.max_windows].tolist()
            ds: Any = torch.utils.data.Subset(self.eval_dataset, idx)
        else:
            ds = self.eval_dataset
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

        cls_sums = {"acgt": 0.0, "diploid": 0.0, "marker": 0.0}
        cls_cnts = {k: 0 for k in cls_sums}
        aux_sums = {t: 0.0 for t in self.aux_tasks}
        aux_cnts = {t: 0 for t in self.aux_tasks}

        with torch.inference_mode():
            for batch in loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                attn = batch.get("attention_mask")
                attn = attn.to(device, non_blocking=True) if attn is not None else None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = net(input_ids=input_ids, attention_mask=attn)
                logits = out.logits.float()

                lf = logits.view(-1, logits.size(-1))
                lbf = labels.view(-1)
                losses = F.cross_entropy(lf, lbf, reduction="none", ignore_index=-100)
                valid = lbf != -100
                is_marker = torch.zeros_like(lbf, dtype=torch.bool)
                for mid in self.marker_ids:
                    is_marker |= lbf == mid
                is_marker &= valid
                is_dip = (lbf >= self.base_vocab_size) & valid & ~is_marker
                is_acgt = valid & ~is_marker & ~is_dip
                for name, m in (
                    ("acgt", is_acgt),
                    ("diploid", is_dip),
                    ("marker", is_marker),
                ):
                    cls_sums[name] += float((losses * m.float()).sum().item())
                    cls_cnts[name] += int(m.sum().item())

                aux_preds = getattr(out, "aux_preds", None) or {}
                w = torch.ones(input_ids.size(0), device=device)
                for t in self.aux_tasks:
                    pred = aux_preds.get(t)
                    tgt = batch.get(t)
                    if pred is None or tgt is None:
                        continue
                    mean, std = AUX_STANDARDIZE[t]
                    loss_t, n_t = _masked_weighted_regression(
                        pred,
                        tgt.to(device),
                        w,
                        mean,
                        std,
                        loss_type="huber",
                        beta=AUX_HUBER_BETA[t],
                    )
                    if n_t > 0:
                        aux_sums[t] += float(loss_t.item()) * n_t
                        aux_cnts[t] += n_t

        if was_training:
            net.train()

        logs: dict[str, float] = {}
        for k in cls_sums:
            if cls_cnts[k] > 0:
                logs[f"eval/{k}_mlm"] = cls_sums[k] / cls_cnts[k]
        for t in self.aux_tasks:
            if aux_cnts[t] > 0:
                logs[f"eval/{t}"] = aux_sums[t] / aux_cnts[t]
        if not logs:
            return
        if metrics is not None:
            metrics.update(logs)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(logs, step=state.global_step)
        except ImportError:
            pass


class ThroughputCallback(TrainerCallback):
    """Log ``train/tokens_per_second`` on every logging interval.

    HF Trainer auto-logs ``learning_rate`` and ``grad_norm`` but not
    throughput. This callback computes effective throughput as
    ``global_step * tokens_per_step / elapsed_seconds``, where
    ``tokens_per_step = batch_size * grad_accum * world_size * window_len``.

    Useful for run-to-run comparison across LR sweeps, batch-size
    ablations, and stage 1 vs stage 2 (with different context lengths).
    """

    def __init__(self, window_len: int, token_budget: int = 0):
        self._window_len = int(window_len)
        # Under length bucketing rows/batch varies but BASES/batch does not, so the
        # budget is the exact per-micro-batch token count and batch_size is not.
        self._token_budget = int(token_budget or 0)
        self._t0: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self._t0 is None or state.global_step <= 0:
            return
        elapsed = time.monotonic() - self._t0
        if elapsed <= 0:
            return
        per_micro = (
            self._token_budget
            if self._token_budget
            else args.per_device_train_batch_size * self._window_len
        )
        tokens_per_step = (
            per_micro * args.gradient_accumulation_steps * _configured_world_size()
        )
        logs["train/tokens_per_second"] = state.global_step * tokens_per_step / elapsed


@dataclass
class AuxMaskedLMOutput(MaskedLMOutput):
    """``MaskedLMOutput`` + per-task per-position regression predictions, plus the
    optional Phase-2B/C contrastive terms (CPL phase-swap repulsion, Track-B
    effect-ranked WT-vs-mut displacement) and the SMP supervised masked-phase term
    (predict the hidden ‹/› orientation from context). The extra terms are raw
    scalars; ``AuxRegressionLossMixin`` scales them by their lambdas and adds them."""

    aux_preds: dict[str, torch.Tensor] | None = None
    cpl_loss: torch.Tensor | None = None
    trackb_loss: torch.Tensor | None = None
    smp_loss: torch.Tensor | None = None
    ctp_loss: torch.Tensor | None = None


class AuxHeadsModel(nn.Module):
    """Wrap an NTv3 MLM backbone with per-position regression head(s) (phyloP).

    The backbone still produces MLM logits exactly as before. We additionally
    run a small ``Linear(hidden_dim, 1)`` head per active task on the model's
    full-resolution hidden state — the same ``(B, L, conv_init_embed_dim)``
    tensor the LM head consumes (``hidden_states[-1]``) — and expose the
    predictions on the output for ``AuxRegressionLossMixin``.

    Capture uses ``output_hidden_states=True`` rather than a forward hook: it is
    purely functional (compile-safe; the hook's attribute side-effect is fragile
    under ``torch.compile``, the default for real runs). The extra hidden-state
    tuple is dropped from the returned output immediately after the head reads it.

    The wrapper delegates the HF surface (config, embeddings, resize, tie,
    gradient checkpointing, plus a catch-all ``__getattr__``) to the backbone so
    existing setup/optimizer code is unchanged. It deliberately does NOT expose
    ``save_pretrained``: HF Trainer then checkpoints the full wrapper state_dict
    (``backbone.*`` + ``heads.*``), so mid-run resume restores both. Clean
    benchmark artifacts (backbone-only HF save + ``aux_heads.safetensors``
    sidecar) are produced at explicit save points via ``_save_model_artifacts``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        tasks: tuple[str, ...],
        bottleneck_tasks: tuple[str, ...] = (),
        bottleneck_layer: int = 12,
        cpl_proj_dim: int = 0,
        trackb_proj_dim: int = 0,
        smp_proj: bool | None = None,
        ctp_proj: bool | None = None,
        cpl_margin: float = -1.0,
        cpl_attract_weight: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        hidden_dim = int(backbone.config.conv_init_embed_dim)
        self.aux_hidden_dim = hidden_dim
        self.aux_tasks = tuple(tasks)
        # Phase-2 contrastive projectors (0 = objective off). CPL: phase-swap
        # repulsion on direction-marker het positions (re-enabled objective).
        # Track-B: effect-ranked WT-vs-mut displacement. Both read the full-res
        # hidden state and run an EXTRA backbone pass on a perturbed twin; the
        # trainer step-gates those passes (compute every Nth step) to bound cost.
        self.cpl_proj_dim = int(cpl_proj_dim)
        # CPL repulsive term is a hinge, not a raw minimize-to--1 (see _cpl_term).
        # cpl_margin=-1.0 reproduces the original objective exactly.
        self.cpl_margin = float(cpl_margin)
        self.cpl_attract_weight = float(cpl_attract_weight)
        self.trackb_proj_dim = int(trackb_proj_dim)
        self.cpl_projector = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.cpl_proj_dim),
            )
            if self.cpl_proj_dim > 0
            else None
        )
        self.trackb_projector = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.trackb_proj_dim),
            )
            if self.trackb_proj_dim > 0
            else None
        )
        # SMP: supervised masked-phase prediction. A per-position binary classifier
        # that reads the full-res hidden state at MASKED direction-marker positions
        # and predicts the true orientation (‹ vs ›). NO extra backbone pass (reads
        # the already-computed hidden state, unlike CPL/Track-B). ``smp_proj=None``
        # = off; True = projector trunk (LayerNorm→Linear→GELU→Linear, pushes phase
        # into the backbone); False = shallow head (LayerNorm→Linear) straight on the
        # backbone rep (the proj-vs-generality ablation arm).
        self.smp_proj = smp_proj
        if smp_proj is None:
            self.smp_head = None
        elif smp_proj:
            self.smp_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.smp_head = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2)
            )
        if self.smp_head is not None:
            nn.init.zeros_(self.smp_head[-1].bias)
        # CTP: supervised cis/trans pairwise head. Reads the full-res hidden state at
        # TWO het-base positions (c1+1, c2+1 — exactly the SYM tokens the frozen
        # compound-het probe reads), builds the SAME relational feature the probe uses
        # — cat([z1, z2, z1-z2, z1*z2]) (4H) — and predicts the gauge-invariant
        # same-haplotype (cis) vs different-haplotype (trans) label. Deliberately
        # mirrors hgmd .../ntv3_variant_aware._extract_pairwise_features so the
        # pretraining gradient shapes the difference/product subspace the frozen
        # LINEAR probe consumes. NO extra backbone pass (reads hidden_states[-1]).
        # ``ctp_proj=None`` = off; True = projector trunk (protects ancestry); False =
        # shallow LayerNorm→Linear(4H,2) (max linear-readability — the probe is linear).
        self.ctp_proj = ctp_proj
        feat_dim = 4 * hidden_dim
        if ctp_proj is None:
            self.ctp_head = None
        elif ctp_proj:
            self.ctp_head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.ctp_head = nn.Sequential(
                nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 2)
            )
        if self.ctp_head is not None:
            nn.init.zeros_(self.ctp_head[-1].bias)
        # Track-A: optional heads on an intermediate (downsampled) hidden state —
        # the transformer bottleneck (default index 12 = last of the 6 transformer
        # blocks, L/128). These push the aux gradient into the transformer, not just
        # the post-deconv final layer the full-res heads read. Their per-bin targets
        # are pooled to the bottleneck resolution in AuxRegressionLossMixin.
        self.bottleneck_tasks = tuple(bottleneck_tasks)
        self.bottleneck_layer = int(bottleneck_layer)
        # LayerNorm -> Linear per task (mirrors NTv3's own LinearHead). The LN
        # normalizes the un-normalized final hidden state before the fresh linear,
        # stabilizing init: a bare Linear on a large-magnitude bf16 hidden state
        # yields a huge initial regression loss. Under bf16 autocast the LayerNorm
        # runs in fp32 automatically, and its params auto-route to the no-weight-
        # decay optimizer group via ALL_LAYERNORM_LAYERS.
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
                for name in self.aux_tasks
            }
        )
        # Mean-init (Karpathy trick): start each head predicting its target's global
        # mean so it doesn't spend the first steps learning the bias. The loss
        # z-scores targets with fixed (mean, std) constants, so the standardized
        # target mean is ~0 by construction — the mean-bias is simply 0, with no
        # per-shard scan to compute/store. (If a head ever regressed RAW targets,
        # this is where the dataset mean would go instead.)
        for head in self.heads.values():
            nn.init.zeros_(head[-1].bias)
        # Bottleneck heads: same LayerNorm→Linear shape, read hidden_states[layer].
        self.bottleneck_heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
                for name in self.bottleneck_tasks
            }
        )
        for head in self.bottleneck_heads.values():
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        swapped_input_ids: torch.Tensor | None = None,
        het_position_mask: torch.Tensor | None = None,
        wt_input_ids: torch.Tensor | None = None,
        variant_pos_mask: torch.Tensor | None = None,
        variant_effect: torch.Tensor | None = None,
        smp_label_mask: torch.Tensor | None = None,
        smp_labels: torch.Tensor | None = None,
        smp_weight: torch.Tensor | None = None,
        ctp_pair_pos: torch.Tensor | None = None,
        ctp_pair_label: torch.Tensor | None = None,
        ctp_pair_mask: torch.Tensor | None = None,
        ctp_pair_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> AuxMaskedLMOutput:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        # Last hidden state == full-resolution (B, L, hidden_dim) the LM head reads.
        hidden = out.hidden_states[-1]
        preds = {name: head(hidden).squeeze(-1) for name, head in self.heads.items()}
        cpl_loss = self._cpl_term(
            hidden,
            attention_mask,
            input_ids,
            swapped_input_ids,
            het_position_mask,
            kwargs,
        )
        trackb_loss = self._trackb_term(
            hidden,
            attention_mask,
            wt_input_ids,
            variant_pos_mask,
            variant_effect,
            kwargs,
        )
        smp_loss = self._smp_term(hidden, smp_label_mask, smp_labels, smp_weight)
        ctp_loss = self._ctp_term(
            hidden, ctp_pair_pos, ctp_pair_label, ctp_pair_mask, ctp_pair_weight
        )
        if self.bottleneck_heads:
            hb = out.hidden_states[self.bottleneck_layer]
            # Layout guard: a bottleneck head must read a DOWNSAMPLED state of the
            # expected width — fail loudly if the index points at a full-res layer
            # or the architecture differs from what was mapped.
            if hb.shape[-1] != self.aux_hidden_dim or hb.shape[1] >= hidden.shape[1]:
                raise ValueError(
                    f"bottleneck_layer={self.bottleneck_layer} gave shape {tuple(hb.shape)}; "
                    f"expected a downsampled (<{hidden.shape[1]}) state of width {self.aux_hidden_dim}"
                )
            for name, head in self.bottleneck_heads.items():
                preds[f"{name}@bneck"] = head(hb).squeeze(-1)
        return AuxMaskedLMOutput(
            # _post outputs (NTv3PostTrainedOutput) have no `loss` field — that
            # trainer computes MLM loss from logits itself; only _pre backbones
            # populate out.loss. getattr keeps the wrapper backbone-agnostic.
            loss=getattr(out, "loss", None),
            logits=out.logits,
            hidden_states=None,  # drop the heavy tuple; preds already extracted
            attentions=getattr(out, "attentions", None),
            aux_preds=preds,
            cpl_loss=cpl_loss,
            trackb_loss=trackb_loss,
            smp_loss=smp_loss,
            ctp_loss=ctp_loss,
        )

    # -- Phase-2 contrastive objectives (extra backbone passes) ----------------
    def _cpl_term(
        self, hidden, attention_mask, input_ids, swapped_input_ids, het_position_mask, kwargs
    ):
        """CPL: push cos(proj(h_orig), proj(h_swap)) BELOW ``cpl_margin`` at het
        positions, so a phase swap (‹↔›) must change the representation.

        The DEFAULT ``cpl_margin = -1.0`` IS the original raw ``mean(cos)``: cos >= -1
        always, so ``relu(cos + 1) == cos + 1`` -- same gradient, constant offset.
        Raising it (e.g. -0.3) bounds the repulsion, which the raw term does not: its
        minimizer is cos=-1 on exactly the c+1 het-base positions ``--smp-target base``
        supervises, and every logged run got there (8M -0.976, 100M -0.885, 650M
        -0.974). A 1500-step 8M A/B moved cos as designed but left MLM/SMP within
        noise, so the margin is an UNPROVEN lever, not an established fix.

        ``cpl_attract_weight > 0`` (default 0 = off) penalises (1-cos) at positions the
        swap left untouched. Off by default on both grounds: with global attention a
        marker swap legitimately changes distant states, so this fights the long-range
        phase propagation CTP needs -- and it costs 2.1x the time and 2x the peak
        memory of the default, since its mask is the complement of het.

        Selected rows are gathered BEFORE projecting. Measured, that is a memory win
        (~1.7x lower peak), not a speed one: boolean-mask indexing reads the whole
        tensor anyway, so these shapes are bandwidth-bound and the projector FLOPs
        saved do not show up in wall-clock.

        Returns None if the projector is disabled; a 0-valued (in-graph) dummy on
        non-firing steps so DDP always marks the projector used."""
        if self.cpl_projector is None:
            return None
        het = het_position_mask
        if het is not None and attention_mask is not None:
            het = het & attention_mask.bool()  # never score padding (either term)
        if (
            # Guard on the SAME tensor that gets scored: guarding on the un-masked
            # het while scoring the padding-intersected one would run the twin pass
            # and then .mean() an empty select -> NaN.
            self.training
            and swapped_input_ids is not None
            and het is not None
            and bool(het.any())
        ):
            out_swap = self.backbone(
                input_ids=swapped_input_ids,
                attention_mask=attention_mask,
                # `labels=None`, not omitted: torch.compile guards on the kwarg KEY SET, so a
                # twin call missing a key the main call had hits a graph traced for a
                # different signature -> `KeyError: 'labels'` out of the attention-patch
                # wrapper. None means "compute no loss", which is what the twin wants anyway
                # (only its hidden states are read). Invisible until now because phase-2
                # compiled with CPL off and phase-1 ran CPL with --no-compile.
                labels=None,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            h_swap = out_swap.hidden_states[-1]
            proj = self.cpl_projector

            def _cos(mask):  # project ONLY the selected positions (see note above)
                return F.cosine_similarity(proj(hidden[mask]), proj(h_swap[mask]), dim=-1)

            loss = F.relu(_cos(het) - self.cpl_margin).mean()
            if self.cpl_attract_weight > 0:
                attract_mask = (input_ids == swapped_input_ids) & (~het)
                if attention_mask is not None:
                    attract_mask = attract_mask & attention_mask.bool()
                cos_a = _cos(attract_mask)  # .numel(), not a second .any() host sync
                if cos_a.numel():
                    loss = loss + self.cpl_attract_weight * (1.0 - cos_a).mean()
            return loss
        if self.training:  # DDP dummy: keep projector in graph, zero contribution
            return (
                self.cpl_projector(hidden[:, :1, :].reshape(-1, hidden.shape[-1])).sum()
                * 0.0
            )
        return None

    def _trackb_term(
        self,
        hidden,
        attention_mask,
        wt_input_ids,
        variant_pos_mask,
        variant_effect,
        kwargs,
    ):
        """Track B: at variant positions, the projected displacement between the
        real (mut) window and a wild-type twin should rank-order by CADD effect.
        Extra backbone pass on the WT twin; margin-ranking loss on displacement."""
        if self.trackb_projector is None:
            return None
        if (
            self.training
            and wt_input_ids is not None
            and variant_pos_mask is not None
            and variant_effect is not None
            and bool(variant_pos_mask.any())
        ):
            out_wt = self.backbone(
                input_ids=wt_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            h_mut = self.trackb_projector(hidden[variant_pos_mask])
            h_wt = self.trackb_projector(out_wt.hidden_states[-1][variant_pos_mask])
            disp = (h_mut - h_wt).norm(dim=-1)
            eff = variant_effect[variant_pos_mask].to(disp.device).float()
            return effect_ranked_margin_loss(disp, eff)
        if self.training:  # DDP dummy
            return (
                self.trackb_projector(
                    hidden[:, :1, :].reshape(-1, hidden.shape[-1])
                ).sum()
                * 0.0
            )
        return None

    def _smp_term(self, hidden, smp_label_mask, smp_labels, smp_weight):
        """SMP: supervised masked-phase prediction. At MASKED direction-marker
        positions (the ‹/› token replaced with [MASK] by the collator, the unphased
        symbol kept), predict the true orientation (0=‹ / 1=›) from context with a
        weighted 2-class cross-entropy. Reads the already-computed full-res hidden
        state — NO extra backbone pass. Returns None if the head is disabled; a
        0-valued in-graph dummy on empty/all-hom batches so DDP always marks the
        head used."""
        if self.smp_head is None:
            return None
        if (
            # No .any() here: nothing expensive hangs on it (head-only, no twin
            # pass), and the z.shape[0] check below already covers the empty case
            # for free -- one GPU->CPU sync per step instead of two.
            self.training
            and smp_label_mask is not None
            and smp_labels is not None
        ):
            z = hidden[smp_label_mask]  # (N, hidden_dim)
            if z.shape[0] > 0:
                logits = self.smp_head(z)  # (N, 2)
                tgt = smp_labels[smp_label_mask].to(logits.device).long()
                per = F.cross_entropy(logits, tgt, reduction="none")  # (N,)
                if smp_weight is not None:
                    w = smp_weight[smp_label_mask].to(per.device).float()
                    return (per * w).sum() / w.sum().clamp_min(1e-8)
                return per.mean()
        if self.training:  # DDP dummy: keep head in graph, zero contribution
            return (
                self.smp_head(hidden[:, :1, :].reshape(-1, hidden.shape[-1])).sum()
                * 0.0
            )
        return None

    def _ctp_term(
        self, hidden, ctp_pair_pos, ctp_pair_label, ctp_pair_mask, ctp_pair_weight
    ):
        """CTP: supervised cis/trans pairwise prediction. For each sampled het pair the
        collator emits the two HET-BASE positions (c+1 — the SYM tokens the frozen
        compound-het probe reads) and a gauge-invariant label (1=cis/same-hap,
        0=trans/diff-hap). We gather the two hidden states, build the SAME relational
        feature the probe uses — cat([z1, z2, z1-z2, z1*z2]) (4H) — and apply a weighted
        2-class cross-entropy. Reads the already-computed full-res hidden state — NO
        extra backbone pass. Returns None if the head is disabled; a 0-valued in-graph
        dummy on empty/all-hom batches so DDP always marks the head used."""
        if self.ctp_head is None:
            return None
        if (
            self.training
            and ctp_pair_pos is not None
            and ctp_pair_label is not None
            and ctp_pair_mask is not None
            and bool(ctp_pair_mask.any())
        ):
            B, K, _ = ctp_pair_pos.shape
            bidx = torch.arange(B, device=hidden.device).unsqueeze(1).expand(B, K)
            p1 = ctp_pair_pos[..., 0].clamp_min(0)  # pad (-1) → 0; masked out below
            p2 = ctp_pair_pos[..., 1].clamp_min(0)
            z1 = hidden[bidx, p1]  # (B, K, H)
            z2 = hidden[bidx, p2]  # (B, K, H)
            feat = torch.cat(
                [z1, z2, z1 - z2, z1 * z2], dim=-1
            )  # (B, K, 4H) — probe geometry
            m = ctp_pair_mask
            logits = self.ctp_head(feat[m])  # (N, 2)
            tgt = ctp_pair_label[m].to(logits.device).long()
            per = F.cross_entropy(logits, tgt, reduction="none")  # (N,)
            if ctp_pair_weight is not None:
                w = ctp_pair_weight[m].to(per.device).float()
                return (per * w).sum() / w.sum().clamp_min(1e-8)
            return per.mean()
        if self.training:  # DDP dummy: keep head in graph, zero contribution
            h = hidden[:1, :1, :].reshape(1, hidden.shape[-1])
            return self.ctp_head(torch.cat([h, h, h * 0, h * 0], dim=-1)).sum() * 0.0
        return None

    # -- HF surface delegation -------------------------------------------------
    @property
    def config(self):
        return self.backbone.config

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.backbone.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.backbone.get_output_embeddings()

    def set_output_embeddings(self, value):
        return self.backbone.set_output_embeddings(value)

    def resize_token_embeddings(self, *a: Any, **k: Any):
        return self.backbone.resize_token_embeddings(*a, **k)

    def tie_weights(self):
        return self.backbone.tie_weights()

    def gradient_checkpointing_enable(self, *a: Any, **k: Any):
        return self.backbone.gradient_checkpointing_enable(*a, **k)

    def gradient_checkpointing_disable(self, *a: Any, **k: Any):
        return self.backbone.gradient_checkpointing_disable(*a, **k)

    def __getattr__(self, name: str) -> Any:
        # nn.Module.__getattr__ handles registered params/buffers/submodules
        # (backbone, heads). Anything else falls through to the backbone so the
        # wrapper transparently exposes the HF model surface (.dtype, .device,
        # .can_generate, .warnings_issued, ...).
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "backbone":
                raise
            return getattr(super().__getattr__("backbone"), name)

    # -- aux head (de)serialization -------------------------------------------
    def save_aux_heads(self, save_dir: Path | str) -> None:
        import json

        from safetensors.torch import save_file

        save_dir = Path(save_dir)
        state = {
            f"{name}.{pname}": tensor.detach().cpu()
            for name, head in self.heads.items()
            for pname, tensor in head.state_dict().items()
        }
        # Bottleneck heads under a distinct "{name}@bneck." prefix (no collision
        # with the full-res "{name}." keys — '@' != '.').
        state.update(
            {
                f"{name}@bneck.{pname}": tensor.detach().cpu()
                for name, head in self.bottleneck_heads.items()
                for pname, tensor in head.state_dict().items()
            }
        )
        # Phase-2 contrastive projectors (distinct "@cpl."/"@trackb." prefixes).
        for tag, proj in (
            ("cpl", self.cpl_projector),
            ("trackb", self.trackb_projector),
            ("smp", self.smp_head),
            ("ctp", self.ctp_head),
        ):
            if proj is not None:
                state.update(
                    {
                        f"@{tag}.{pname}": tensor.detach().cpu()
                        for pname, tensor in proj.state_dict().items()
                    }
                )
        save_file(state, str(save_dir / "aux_heads.safetensors"))
        (save_dir / "aux_heads_config.json").write_text(
            json.dumps(
                {
                    "hidden_dim": self.aux_hidden_dim,
                    "tasks": list(self.aux_tasks),
                    "bottleneck_tasks": list(self.bottleneck_tasks),
                    "bottleneck_layer": self.bottleneck_layer,
                    "smp_proj": self.smp_proj,
                    "ctp_proj": self.ctp_proj,
                },
                indent=2,
            )
        )

    def load_aux_heads(self, load_dir: Path | str) -> bool:
        from safetensors.torch import load_file

        path = Path(load_dir) / "aux_heads.safetensors"
        if not path.exists():
            return False
        flat = load_file(str(path))
        for name, head in self.heads.items():
            sub = {
                pname.split(".", 1)[1]: tensor
                for pname, tensor in flat.items()
                if pname.startswith(name + ".")
            }
            if sub:
                head.load_state_dict(sub)
        # Bottleneck heads (if any, and if present in the sidecar). A sidecar from
        # a full-res-only checkpoint simply has no "@bneck." keys → those heads
        # stay at fresh init (intended for resume-from-genome-1000).
        for name, head in self.bottleneck_heads.items():
            prefix = f"{name}@bneck."
            sub = {
                pname[len(prefix) :]: tensor
                for pname, tensor in flat.items()
                if pname.startswith(prefix)
            }
            if sub:
                head.load_state_dict(sub)
        # Contrastive projectors: absent in a checkpoint that predates Phase-2B/C
        # → they stay at fresh init (intended on resume from control/2b).
        for tag, proj in (
            ("cpl", self.cpl_projector),
            ("trackb", self.trackb_projector),
            ("smp", self.smp_head),
            ("ctp", self.ctp_head),
        ):
            if proj is None:
                continue
            prefix = f"@{tag}."
            sub = {
                pname[len(prefix) :]: t
                for pname, t in flat.items()
                if pname.startswith(prefix)
            }
            if sub:
                proj.load_state_dict(sub)
        logger.info("Loaded aux heads from %s", path)
        return True


def setup_model_and_tokenizer(
    tokens_path: Path,
    output_dir: Path,
    resume_from: Path | None = None,
    model_name: str = MODEL_NAME,
    save_baseline: bool = True,
    sdpa_attention: bool = True,
    token_freqs: dict[str, float] | None = None,
    new_token_bias_offset: float = -10.0,
    drop_unseen_tokens: bool = False,
    new_row_init: str = "legacy",
):
    """Load NTv3, expand vocab with diploid tokens, init embeddings, save baseline.

    Returns ``(model, tokenizer, base_vocab_size, human_species_id)``.
    ``human_species_id`` is an int for _post models, ``None`` for _pre.
    """
    is_post = _is_post_model(model_name)
    _load_cls = AutoModel if is_post else AutoModelForMaskedLM

    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)
        tokenizer = AutoTokenizer.from_pretrained(resume_from, trust_remote_code=True)
        model = _load_cls.from_pretrained(resume_from, trust_remote_code=True)
        stamped = getattr(model.config, "new_row_init", None)
        if new_row_init != "legacy" and stamped != new_row_init:
            # A resume never re-initialises rows: without this, an arm relaunched into
            # another arm's output directory silently continues that arm's rows.
            raise SystemExit(
                f"--new-row-init {new_row_init} has no effect on a resume: {resume_from} "
                f"was initialised with {stamped}. Use a fresh --output-dir, or remove its "
                "_emb_warmup and baseline_expanded_vocab to re-initialise."
            )
        _assert_pad_multiple_matches_model(
            model
        )  # guard: PAD_MULTIPLE == 2**num_downsamples
        _maybe_bypass_pools(model)
        if sdpa_attention:
            n_patched = patch_ntv3_attention_sdpa(model, fast=True)
            logger.info("Patched %d NTv3 MHA modules to use SDPA", n_patched)
        # Independent of SDPA: the mask must be threaded either way, and nesting
        # it under sdpa_attention would silently no-op --no-sdpa-attention.
        apply_pad_mask_from_env(model)
        # --drop-unseen-tokens never runs on this path (we adopt the checkpoint's
        # tokenizer wholesale), so a checkpoint built without it keeps its dead
        # rows and nothing would otherwise say so. WARN only -- pruning here would
        # reshape the embedding/LM-head and remap ids, invalidating the optimizer
        # moments and the cached tokenization. Fix it at the next clean start.
        try:
            dead = _dead_tokens_in_vocab(
                merged_vocab(tokenizer), token_freqs, _parse_diploid_tsv(tokens_path)[0]
            )
        except Exception:  # noqa: BLE001 - a diagnostic must never break a resume
            dead = []
        if dead:
            logger.warning(
                "Resumed checkpoint carries %d diploid tokens with ZERO corpus "
                "occurrences: %s. Their LM-head rows are permanently-wrong MLM "
                "classes. --drop-unseen-tokens does not apply on resume; prune at "
                "the next fresh vocab expansion.",
                len(dead),
                " ".join(dead),
            )
        human_species_id = None
        if is_post:
            _patch_post_model_embeddings(model)
            human_species_id = model.encode_species(["human"])[0].item()
            logger.info("Post-trained model: human species_id=%d", human_species_id)
        return (
            model,
            cast(PreTrainedTokenizerBase, tokenizer),
            getattr(model.config, "base_vocab_size", 11),
            human_species_id,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    b_size = tokenizer.vocab_size

    symbols, sym2pair, sym2order = _parse_diploid_tsv(tokens_path)
    new_tokens = sorted(symbols - set(tokenizer.get_vocab()))
    dropped_tokens: list[str] = []
    if drop_unseen_tokens:
        new_tokens, dropped_tokens = _filter_unseen_tokens(new_tokens, token_freqs)
        if dropped_tokens:
            logger.info(
                "Dropping %d diploid tokens with zero corpus occurrences: %s",
                len(dropped_tokens),
                " ".join(dropped_tokens),
            )
        else:
            logger.info("--drop-unseen-tokens: every token occurs; nothing dropped")
    n_added = tokenizer.add_tokens(new_tokens)
    logger.info("Added %d diploid tokens (base vocab=%d)", n_added, b_size)

    model = _load_cls.from_pretrained(model_name, trust_remote_code=True)
    _assert_pad_multiple_matches_model(
        model
    )  # guard: PAD_MULTIPLE == 2**num_downsamples
    _maybe_bypass_pools(model)
    if sdpa_attention:
        n_patched = patch_ntv3_attention_sdpa(model, fast=True)
        logger.info("Patched %d NTv3 MHA modules to use SDPA", n_patched)
    # Independent of SDPA: the mask must be threaded either way, and nesting
    # it under sdpa_attention would silently no-op --no-sdpa-attention.
    apply_pad_mask_from_env(model)

    human_species_id: int | None = None
    if is_post:
        _patch_post_model_embeddings(model)
        human_species_id = model.encode_species(["human"])[0].item()
        logger.info("Post-trained model: human species_id=%d", human_species_id)

    model.resize_token_embeddings(b_size + n_added)
    model.config.base_vocab_size = b_size
    if is_post:
        model.config.alphabet_size = b_size + n_added
    logger.info("Resized embeddings: %d -> %d", b_size, b_size + n_added)

    _init_diploid_embeddings(
        model,
        tokenizer,
        sym2pair,
        sym2order,
        b_size,
        token_freqs=token_freqs,
        new_token_bias_offset=new_token_bias_offset,
        new_row_init=new_row_init,
    )
    model.config.new_row_init = new_row_init

    # Stamp the tokens-TSV hash into the model config so downstream consumers
    # (ClinVar / HGMD benchmark tasks) can verify their benchmark-side
    # diploid_tokens.tsv matches the one this model was trained against.
    # Drift between the two TSVs (e.g., a new symbol added on one side)
    # would cause silent <unk>-tokenization of new tokens at inference.
    #
    # A filtered run hashes the FILTERED TSV it writes, not the repo one: the
    # two disagree on which symbols exist, which is precisely the drift this
    # guard is for. Consumers must point at <output_dir>/diploid_tokens_used.tsv
    # (or read the tokenizer from the checkpoint, which also travels correctly).
    try:
        hashed_path = tokens_path
        if dropped_tokens:
            hashed_path = _write_filtered_tokens_tsv(
                tokens_path, output_dir, set(dropped_tokens)
            )
        tokens_hash = hashlib.sha256(hashed_path.read_bytes()).hexdigest()[:16]
        model.config.diploid_tokens_sha256_16 = tokens_hash
        model.config.diploid_added_tokens = list(new_tokens)
        model.config.diploid_dropped_tokens = list(dropped_tokens)
        logger.info(
            "Stamped diploid_tokens.tsv hash on model.config.diploid_tokens_sha256_16 = %s",
            tokens_hash,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not stamp tokens-TSV hash on config: %s", exc)

    # Stamp what the rare-token recipe ACTUALLY did (not what was requested):
    # whether the unigram counts table resolved, and the bias-prior fallback.
    # A checkpoint with rare_token_recipe absent predates the fix and therefore
    # trained with the uniform-over-vocab MLM fill pool and no head-bias rarity
    # prior — that is how historical runs are told apart (see TRAINING_GUIDE).
    model.config.rare_token_recipe = {
        "version": RARE_TOKEN_RECIPE_VERSION,
        "unigram_counts_loaded": bool(token_freqs),
        "new_token_bias_offset": float(new_token_bias_offset),
    }
    if not token_freqs:
        logger.warning(
            "No unigram counts table: MLM random fills fall back to uniform "
            "ACGT and new head biases use the constant %.1f offset. Stamped "
            "unigram_counts_loaded=False on model.config.rare_token_recipe.",
            new_token_bias_offset,
        )

    _log_init_scale_check(model, b_size, token_freqs=token_freqs)

    if save_baseline and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        baseline_dir = output_dir / "baseline_expanded_vocab"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(baseline_dir)
        fast_tok = _convert_tokenizer_to_fast(tokenizer)
        fast_tok.save_pretrained(baseline_dir)

    return model, cast(PreTrainedTokenizerBase, tokenizer), b_size, human_species_id


# -- Validation ---------------------------------------------------------------


def validate_diploid_embeddings(
    baseline_dir: Path,
    trained_dir: Path,
    base_vocab_size: int,
) -> None:
    """Log embedding L2 drift between baseline and trained checkpoints.

    Validation is best-effort: if required files are missing, log a clear warning
    and continue so successful training runs are not marked as failed.
    """
    from safetensors.torch import load_file

    def _load_state_dict(weights_dir: Path) -> dict[str, torch.Tensor] | None:
        st_path = weights_dir / "model.safetensors"
        pt_path = weights_dir / "pytorch_model.bin"
        st_index = weights_dir / "model.safetensors.index.json"
        pt_index = weights_dir / "pytorch_model.bin.index.json"

        if st_path.exists():
            try:
                return load_file(st_path)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", st_path, exc)
                return None

        if pt_path.exists():
            try:
                state = torch.load(pt_path, map_location="cpu")
                if isinstance(state, dict):
                    return state
                logger.warning(
                    "Unexpected checkpoint format in %s (type=%s)",
                    pt_path,
                    type(state),
                )
                return None
            except Exception as exc:
                logger.warning("Failed to load %s: %s", pt_path, exc)
                return None

        # Sharded checkpoints need index-driven loading that this quick validator
        # does not implement; skip explicitly with a clear message.
        if st_index.exists() or pt_index.exists():
            logger.warning(
                "Skipping embedding validation: sharded weights detected in %s "
                "(index file present).",
                weights_dir,
            )
            return None

        logger.warning(
            "Skipping embedding validation: no weight file found in %s "
            "(expected model.safetensors or pytorch_model.bin).",
            weights_dir,
        )
        return None

    b_w = _load_state_dict(baseline_dir)
    t_w = _load_state_dict(trained_dir)
    if b_w is None or t_w is None:
        logger.warning(
            "Post-train embedding validation skipped; training checkpoint save "
            "status is unaffected."
        )
        return

    emb_key = next(
        (k for k in b_w if "embed" in k.lower() and "weight" in k.lower()), None
    )
    if emb_key is None:
        logger.warning("Could not find embedding weight for validation")
        return

    if emb_key not in t_w:
        logger.warning(
            "Could not find matching embedding key '%s' in trained checkpoint; "
            "skipping validation.",
            emb_key,
        )
        return

    n_dip = min(b_w[emb_key].shape[0], t_w[emb_key].shape[0]) - base_vocab_size
    if n_dip <= 0:
        return

    # --token-embed-dim widens this tensor mid-run, so the trained embedding can be
    # WIDER than the baseline it is compared against. Only the shared column span is
    # comparable; the new dims have no baseline to differ from. Without this the
    # subtraction below raises and the process exits non-zero AFTER a successful
    # save -- which the arm runner reads as a crash and retries up to MAX_RETRIES.
    n_dim = min(b_w[emb_key].shape[1], t_w[emb_key].shape[1])
    if b_w[emb_key].shape[1] != t_w[emb_key].shape[1]:
        logger.info(
            "validate_diploid_embeddings: embedding was widened %d -> %d during "
            "this run; comparing the shared first %d dims only.",
            b_w[emb_key].shape[1],
            t_w[emb_key].shape[1],
            n_dim,
        )
    b_w = {emb_key: b_w[emb_key][:, :n_dim], **{k: v for k, v in b_w.items() if k != emb_key}}
    t_w = {emb_key: t_w[emb_key][:, :n_dim], **{k: v for k, v in t_w.items() if k != emb_key}}

    s = base_vocab_size
    dip_l2 = (
        (t_w[emb_key][s : s + n_dip].float() - b_w[emb_key][s : s + n_dip].float())
        .norm(dim=1)
        .mean()
        .item()
    )
    base_l2 = (
        (t_w[emb_key][:s].float() - b_w[emb_key][:s].float()).norm(dim=1).mean().item()
    )
    logger.info(
        "Embedding drift -- diploid: L2=%.4f (%d tokens), base: L2=%.4f (%d tokens)",
        dip_l2,
        n_dip,
        base_l2,
        base_vocab_size,
    )
    if dip_l2 < 1e-6:
        logger.warning(
            "Diploid embeddings barely moved! Training may not be effective."
        )
    else:
        logger.info("Diploid token embeddings updated successfully.")


# -- CLI / Main ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    p.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    p.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        help="Pre-windowed eval dataset (from prepare_eval_data.py). "
        "Overrides --test-data for evaluation only.",
    )
    p.add_argument(
        "--tokens",
        type=Path,
        default=SCRIPT_DIR.parent / "preprocess" / "diploid_tokens.tsv",
    )
    p.add_argument(
        "--token-counts",
        type=Path,
        default=DEFAULT_TOKEN_COUNTS_PATH,
        help="Whole-train-set symbol counts TSV (preprocess/token_counts.tsv). "
        "Drives (a) the unigram-weighted 10%% MLM random-replacement pool and "
        "(b) the ln-frequency rarity prior on new LM-head biases. A nonexistent "
        "path degrades to ACGT-uniform fills and the constant "
        "--new-token-bias-offset prior — NOT the pre-fix behavior (that was "
        "uniform over the whole non-special vocab, markers included, with no "
        "bias prior at all); it is stamped as unigram_counts_loaded=False.",
    )
    p.add_argument(
        "--v2-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Read the stored v7 rows as one token per position (pretrain/v2_input.py): a "
        "direction token + pair glyph becomes its phased glyph, `^` + base its "
        "homozygous-alternate glyph. Needs the v2 token and counts files (--tokens "
        "preprocess/diploid_tokens_v2.tsv --token-counts preprocess/token_counts_v2.tsv), "
        "every marker dropped (--marker-drop-target all --marker-drop-prob 1) and both "
        "phase losses off; a character outside the vocabulary then fails the run.",
    )
    p.add_argument(
        "--cadd-sidecar",
        type=Path,
        default=None,
        help="Directory of rebuilt CADD columns, one per training shard "
        "(preprocess/huggingface_dataset/tools/rebuild_cadd.py): read in place of each shard's cadd_phred, which "
        "datasets built before 2026-10-14 store rounded to whole numbers and averaged at 1|2 sites.",
    )
    p.add_argument(
        "--v2-reference",
        type=Path,
        default=None,
        help="Reference-aware tokens: the HG38 FASTA. With --v2-input and the reference-aware token "
        "file (preprocess/diploid_tokens_v3.tsv), a heterozygous SNV token also says which copy carries "
        "the reference. Rows with indels are aligned to HG38 (v2_input.row_reference). Needs --no-eval.",
    )
    p.add_argument(
        "--v2-copy-swap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --v2-input: read each training window with its two copies exchanged half the time "
        "(data augmentation), so the arbitrary copy-1 / copy-2 labels carry no signal. Training windows only.",
    )
    p.add_argument(
        "--new-row-init",
        choices=("legacy", "random", "between"),
        default="legacy",
        help="Start of the new INPUT embedding rows on a fresh vocabulary expansion. "
        "legacy: parent-pair mean (every run before v2). random: random direction at the "
        "size of a base row. between: 2/3 of the copy-0 base row + 1/3 of the copy-1 base "
        "row + small noise. With random or between, the new LM-head rows start at the "
        "mean of the ACGT head rows (legacy: the mean of all 11 base rows).",
    )
    p.add_argument(
        "--drop-unseen-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ON BY DEFAULT (pass --no-drop-unseen-tokens to keep the dead rows). "
        "Only add diploid tokens that actually occur in --token-counts. On "
        "v7 this drops 29 of 44 (all phased glyphs, the lowercase set, '-' and "
        "'*'), leaving a 26-token vocabulary. Their input rows are inert, but "
        "their LM-head rows are 29 permanently-wrong MLM classes, and '*'/'-' "
        "sit at cosine 1.0000 because nothing ever separates two rows that "
        "start equal and never take gradient. WARNING: the kept set is a "
        "property of the CORPUS. A filtered checkpoint run against data that "
        "does contain the dropped symbols (phased, or v8) resolves them to "
        "<unk>. Verify first with pretrain/tools/verify_dead_tokens.py; the run "
        "writes the filtered TSV it used to <output-dir>/diploid_tokens_used.tsv "
        "and stamps THAT hash, so the benchmark-side drift check fails loudly "
        "instead of mis-tokenizing.",
    )
    p.add_argument(
        "--new-token-bias-offset",
        type=float,
        default=-10.0,
        help="Logit-prior offset applied to new LM-head bias rows when the "
        "counts table can't resolve a token (default -10.0 ~ ln(1e-5/0.25); "
        "the loss basin is flat over [-6,-12]). 0.0 disables the fallback.",
    )
    p.add_argument("--model-name", type=str, default=MODEL_NAME)
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("output/diploid_pretrain"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--token-budget",
        type=int,
        default=0,
        help="Bases per micro-batch per GPU. 0 (default) keeps fixed-length "
        "row batching (--batch-size windows, each padded to --window-len) and is "
        "byte-identical to every run before this flag existed. Non-zero switches "
        "to length-bucketed batching: a window is padded to its %d-base bucket "
        "instead of to the full window, and one batch holds budget//bucket "
        "windows of a SINGLE bucket. Needed on a variable-length corpus -- on "
        "v7-tsstes at --window-len 131072, 84%%%% of rows are shorter than one "
        "window and fixed-length padding spends ~57%%%% of the run on pad tokens. "
        "Bases per optimizer step = budget * --grad-accum * world_size; the NTv3 "
        "protocol measured >8M bases/update as WORSE, so keep it at or under "
        "that. Must be a multiple of 8192 when --og2-fraction > 0 (the replay "
        "bins have to divide it exactly). Requires --genome-steps: milestone "
        "pacing cannot be derived from a row count once rows/batch varies."
        % BUCKET_STEP,
    )
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.005,
        help="AdamW weight decay. Kept low for continued pretraining so the "
        "already-trained weights are not pulled toward zero over the epoch. "
        "(Power Lines / Bergsma 2025 suggests lambda scales linearly with batch "
        "size for from-scratch runs, but light regularization is preferred for CPT.)",
    )
    p.add_argument("--window-len", type=int, default=4096)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument(
        "--overlap-frac",
        type=float,
        default=0.10,
        help="Window overlap fraction used when --stride is unset (0.1 = 10%% overlap).",
    )
    p.add_argument(
        "--window-jitter",
        type=int,
        default=0,
        help="Max random shift (bp) applied to each TRAIN window's start on every "
        "__getitem__, clamped to the record's slack. 0 (default) = off, reproducing "
        "the historical fixed, epoch-invariant window grid. Eval datasets never "
        "jitter. Only bites where a record is longer than --window-len: at "
        "--window-len 4096 just 43%% of BED regions have any slack (57%% sit at the "
        "4094bp inflation floor), so pair this with --window-len < 4094 (e.g. 3968 or "
        "2048, both 128-divisible) to break the fixed position->content prior.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-steps", type=int, default=10000)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--dataloader-workers", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument(
        "--nonvariant-keep-frac",
        type=float,
        default=0.4,
        help="Fraction of non-REF non-variant windows kept in train. 0.05 was "
        "the prior default; 0.4 is a hedge between catastrophic-forgetting "
        "anchor needs (Gururangan 2020, Cui 2023) and variant-signal dilution. "
        "Verify against the `eval_wt_loss - eval_primary_loss` panel in wandb; "
        "raise if WT loss drifts up, lower if primary loss plateaus early.",
    )
    p.add_argument("--embedding-warmup-steps", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=10000)
    p.add_argument(
        "--torch-compile-mode",
        type=str,
        default="default",
        help="torch.compile mode: default|reduce-overhead|max-autotune. "
        "'default' uses Inductor without CUDAGraph capture and is the safest "
        "choice for NTv3 (rotary cache changes shape between steps).",
    )
    p.add_argument(
        "--compile-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile only the NTv3 backbone forward with torch.compile, leaving "
        "the Trainer wrapper and aux/CPL/SMP loss glue eager. Disabled by --no-compile.",
    )
    p.add_argument(
        "--compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass dynamic=True to the manual backbone torch.compile path.",
    )
    p.add_argument(
        "--skip-post-track-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For NTv3 _post MLM pretraining, preserve BigWig/BED modules in the "
        "checkpoint but skip computing their logits. Disable only when training "
        "the original POST track objectives.",
    )
    p.add_argument(
        "--eval-max-windows",
        type=int,
        default=50_000,
        help="Cap primary-eval set size for faster eval cycles (0=unlimited). "
        "The primary eval applies --eval-window-filter (default 'all'). "
        "Combined with --eval-wt-windows the total default eval is ~100K "
        "windows (50K primary + 50K WT).",
    )
    p.add_argument(
        "--eval-subset-windows",
        type=int,
        default=8000,
        help="Windows sampled for the held-out per-subset + aux eval logged by "
        "EvalSubsetAuxCallback (eval/{acgt,diploid,marker}_mlm, eval/{phylop,"
        "cadd_phred}). One forward-only pass per eval on rank 0; 0 disables it.",
    )
    p.add_argument(
        "--eval-window-filter",
        type=str,
        default="all",
        choices=["variant_only", "variant_only_no_ref", "variant_enriched", "all"],
        help="Filter mode for eval window selection. Default 'all' keeps all "
        "windows — matches real-world inference distribution where most "
        "positions are reference. Use 'variant_only' to focus the primary eval "
        "on variant-containing windows only. Pair with a pre-built small eval "
        "set (see build_eval_small.py).",
    )
    p.add_argument(
        "--eval-wt-windows",
        type=int,
        default=50_000,
        help="If >0, build a second eval split of HG38_REF (WT) windows with no "
        "variants. Reported as `eval_wt_loss` alongside the primary split's "
        "`eval_primary_loss` so reference-manifold drift is visible during "
        "training. 0 disables.",
    )
    p.add_argument(
        "--eval-nonvariant-windows",
        type=int,
        default=0,
        help="If >0, build a third eval split of patient-sourced non-variant "
        "windows (`~is_ref & ~has_variant`). Reported as `eval_nonvariant_loss`. "
        "OFF by default — the primary + WT pair already covers the diagnostic "
        "axes (full-distribution loss + reference-manifold drift). Set to a "
        "non-zero value to add patient non-variant drift as a third signal.",
    )
    p.add_argument(
        "--max-wall-seconds",
        type=int,
        default=0,
        help="If > 0, stop training after this many seconds of wall-clock "
        "(checkpoint first via control.should_save=True). Used for time-budgeted "
        "production runs (e.g., 27h58m = 100680). 0 disables.",
    )
    p.add_argument(
        "--stop-at-genome",
        type=int,
        default=None,
        help="Stop training once this genome-equivalent milestone is saved. Must "
        "also appear in --save-at-genomes. Bounds an arm in-process instead of "
        "externally killing it, so the run still shuts down cleanly (final save, "
        "benchmark flush, dataloader teardown) rather than being SIGTERMed at an "
        "arbitrary point, possibly mid-checkpoint.",
    )
    p.add_argument(
        "--save-at-genomes",
        nargs="*",
        type=int,
        default=[2, 10, 50, 100, 200, 400, 800, 1600, 2000],
    )
    p.add_argument(
        "--token-embed-dim",
        type=int,
        default=None,
        help="Widen NTv3's token bottleneck (default 16) to N before main "
        "training, function-preserving at init. All 55 tokens -- 11 base + 44 "
        "added -- currently share 16 dims feeding a 15-tap conv fitted when only "
        "11 token identities existed, so added tokens must reuse directions ACGT "
        "already occupies. Applied AFTER the embedding warmup, so a widened arm "
        "and its narrow sibling share a bitwise-identical warmup. Forks the "
        "architecture: the arm needs its own anchor.",
    )
    p.add_argument(
        "--embedding-warmup-lr-mult",
        type=float,
        default=None,
        help="Multiplier on --lr for the embedding-warmup sub-run. Default "
        "(None) keeps the historical values: 2.0 with --peft-style-warmup, else "
        "8.0. Raising it is comparatively safe -- warmup freezes the backbone "
        "and gradient-masks the base rows, and a 5000-step warmup at 8x was "
        "measured to do ZERO damage to ACGT benchmarks despite 15.6x genome-5's "
        "LR-integral -- but watch the post-warmup new-row scale check.",
    )
    p.add_argument(
        "--keep-rows",
        type=Path,
        default=None,
        help="Row-index .npy keeplist (semantic dedup) restricting MAIN training "
        "to those corpus rows. Pass the keeplist's .parquet instead (same rows) "
        "and each survivor's sample_weight becomes its cluster_weight, so rows "
        "are removed but individuals are not. Indices are positional into the "
        "concatenated train split in sorted-shard order, so a list built on a different "
        "corpus is a hard error, not a silent mis-selection. Filtering happens "
        "at the window-index level, leaving the shared window-index cache valid. "
        "NOTE: a smaller corpus shifts genome-equivalent pacing -- pair this with "
        "--genome-steps or genome-N stops meaning the same thing across arms.",
    )
    p.add_argument(
        "--warmup-keep-rows",
        type=Path,
        default=None,
        help="Same, for the embedding-warmup stage only, and mask only (a "
        ".parquet's cluster_weight is not applied here). Kept separate because "
        "the vocab-warmup subset is capped per locus: that flattens locus "
        "frequencies, which is right for warming new token rows and wrong for "
        "anything that should see the corpus distribution.",
    )
    p.add_argument(
        "--genome-steps",
        type=float,
        default=None,
        help="Pin optimizer-steps-per-genome-equivalent instead of deriving it "
        "from this dataset. Default (None) keeps the historical behavior: "
        "genome-N is a fraction of whatever corpus is mounted, so a "
        "20%%-smaller deduplicated corpus over the same genomes fires genome-50 "
        "20%% early and its row stops being comparable to the rest of "
        "RESULTS.md. Every scored arm ran at 277.58 (eff-batch 640); pass that "
        "to keep genome-N an absolute budget across datasets and batch sizes.",
    )
    p.add_argument("--no-eval", action="store_true")
    p.add_argument(
        "--eval-on-start",
        action="store_true",
        help="Run evaluation at step 0 before any training steps.",
    )
    p.add_argument("--eval-steps", type=int, default=10_000)
    p.add_argument("--report-to", nargs="*", default=["wandb"])
    p.add_argument("--quick-test", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="HuggingFace datasets cache directory for parquet->arrow conversion. "
        "Arrow files bypass caching entirely; parquet files use this path "
        "(defaults to ~/.cache/huggingface/datasets/). "
        "Set to a large-capacity partition to avoid ENOSPC errors.",
    )
    p.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Limit number of parquet shards loaded (train & test). For fast ablations.",
    )
    p.add_argument(
        "--max-genomes",
        type=int,
        default=None,
        help="Train on a seeded random subset of N non-REF genomes (HG38_REF always "
        "kept). Applied as a post-hoc window-index filter on the train split only; "
        "the full-corpus window-index cache is reused. Eval is unaffected.",
    )
    p.add_argument(
        "--max-genomes-seed",
        type=int,
        default=42,
        help="Seed for the --max-genomes subset, decoupled from --seed so the "
        "genome subset stays stable if the training seed changes.",
    )
    p.add_argument(
        "--exclude-genomes",
        type=Path,
        default=None,
        help="Train on the COMPLEMENT of a prior run's genome subset: path to its "
        "genome_subset_seed*.json; every listed non-REF sample_id is dropped "
        "(HG38_REF always kept). Same post-index window filter as --max-genomes; "
        "composes with it and with --keep-rows. Pair with --genome-steps.",
    )
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    p.add_argument(
        "--sdpa-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Patch NTv3 MultiHeadAttention to use F.scaled_dot_product_attention "
        "(routes to FlashAttention-2 on Ampere/Hopper bf16). NTv3 ignores HF's "
        "attn_implementation kwarg and runs hand-rolled fp32 attention by default.",
    )
    p.add_argument(
        "--pad-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Thread attention_mask into NTv3's transformer tower. NTv3 hardcodes "
        "attention_mask=None on every block, so padding is self-attended as real "
        "sequence: 0%% of tokens at window 2048/4096 but 30.4%% at 8192 and 41.2%% at "
        "10240. Off by default because turning it on moves every padded score, so a "
        "run with it on is NOT comparable to an already-scored checkpoint without it "
        "-- including its own anchor. The choice is stamped on config.pad_mask and "
        "replayed by every scorer unless NTV3_PAD_MASK overrides it, so set that "
        "variable on both sides of a comparison to hold the mode fixed. Defaults to "
        "the NTV3_PAD_MASK env var when unset; passing it either way records the "
        "choice in run_args.json, which the env var alone does not.",
    )
    p.add_argument(
        "--pool-bypass",
        type=int,
        default=None,
        help="Drop the LAST n conv-tower mean-pools and their deconv twins, so the "
        "transformer runs at window/(128>>n) instead of window/128. Weight-free "
        "runtime surgery (pretrain/ntv3_pool_surgery.py); the state_dict is "
        "byte-identical either way, which is exactly why it has to be recorded. "
        "Defaults to the NTV3_POOL_BYPASS env var when unset, and passing it "
        "either way ALSO lands it in run_args.json -- the env var alone does not, "
        "and a result CSV has no column for it, so two cells can match on every "
        "recorded field and still be different architectures (that is what made "
        "phase-2's 'delta vs base' bundle four changes at once). Every "
        "scorer/extractor must re-apply the same value: layer_sweep hard-errors on "
        "a mismatch against the config stamp.",
    )
    p.add_argument(
        "--peft-style-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="During embedding warmup, train ONLY the newly-added token rows "
        "via gradient-row masking. Original NTv3 ACGTN rows remain "
        "bitwise-identical. Legacy behavior (--no-peft-style-warmup) trains "
        "the entire input and output embedding matrices.",
    )
    p.add_argument(
        "--mode-collapse-check",
        action="store_true",
        help="Periodically check max pairwise cosine of new-token rows and log "
        "embed/max_new_token_cosine to wandb. OFF by default; warns when "
        "max > 0.99 but never aborts training.",
    )
    p.add_argument(
        "--diag-every-n-steps",
        type=int,
        default=4000,
        help="Cadence (steps) for EmbeddingDiagnosticsCallback periodic logging "
        "(0 disables periodic logging — train_begin/eval/train_end still fire). "
        "Default 4000 ≈ once per 2 genomes at typical batch sizes.",
    )
    p.add_argument(
        "--ref-sampler-weight",
        type=float,
        default=2.0,
        help="Per-window over-sampling weight for HG38_REF windows in the "
        "training sampler. Default 2.0 (modest oversampling). Multiplies with "
        "--ref-loss-weight: effective REF gradient mass ≈ "
        "ref_sampler_weight × ref_loss_weight × base_REF_share.",
    )
    p.add_argument(
        "--ref-loss-weight",
        type=float,
        default=None,
        help="Per-instance loss multiplier on HG38_REF windows. Effective default "
        "is 1.0 because --use-sample-weight is ON by default and the baked "
        "sample_weight=10 on HG38_REF already supplies loss-side ref emphasis; "
        "stacking a multiplier on top would over-focus on ref (sampler-only "
        "extra). Under --no-use-sample-weight it auto-defaults to 4.0 (make REF "
        "count more per gradient step without overpowering batch composition). "
        "Multiplicative with --ref-sampler-weight; see its help. Pass an explicit "
        "value to override the auto-default.",
    )
    # -- Auxiliary per-position regression + per-example sample weighting (v7) --
    p.add_argument(
        "--aux-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach per-position regression head(s) (phyloP + CADD) on the "
        "model's full-resolution hidden state and train them jointly with MLM. "
        "ON by default for v7; pass --no-aux-heads for an MLM-only ablation. "
        "Gracefully falls back to MLM-only if the columns are absent (v5/v6).",
    )
    p.add_argument(
        "--aux-phylop-weight",
        type=float,
        default=0.1,
        help="Loss weight lambda for the phyloP regression term: "
        "total = mlm + lambda * phylop_huber. Default 0.1 (keeps MLM primary).",
    )
    p.add_argument(
        "--aux-cadd-weight",
        type=float,
        default=0.5,
        help="Loss weight lambda for the CADD (cadd_phred) regression term: "
        "total += lambda * cadd_huber. Default 0.5 (raised vs phyloP's 0.1: CADD "
        "is ~1000x sparser, so a higher weight compensates for its rarer gradient).",
    )
    p.add_argument(
        "--aux-bottleneck-tasks",
        nargs="*",
        default=[],
        choices=list(AUX_POSITION_TASKS),
        help="Track-A: also attach aux head(s) on the transformer BOTTLENECK "
        "(downsampled hidden state, default index 12 = L/128) for these tasks, in "
        "addition to the full-resolution heads. Pushes aux gradient into the "
        "transformer, not just the post-deconv final layer. Targets are pooled to "
        "the bottleneck resolution (phyloP via --aux-bottleneck-phylop-pool; the "
        "sparse CADD via max). Empty (default) = full-res heads only.",
    )
    p.add_argument(
        "--aux-bottleneck-layer",
        type=int,
        default=12,
        help="hidden_states index for the bottleneck heads. 12 = last of the 6 "
        "transformer blocks (L/128) for NTv3_100M (7 conv + 6 transformer + 7 "
        "deconv = 20 states). A runtime guard rejects a full-resolution index.",
    )
    p.add_argument(
        "--aux-bottleneck-phylop-weight",
        type=float,
        default=0.3,
        help="Loss weight for the bottleneck phyloP term (separate from the "
        "full-res --aux-phylop-weight so the two depths are tunable apart).",
    )
    p.add_argument(
        "--aux-bottleneck-cadd-weight",
        type=float,
        default=3.0,
        help="Loss weight for the bottleneck CADD term.",
    )
    p.add_argument(
        "--aux-bottleneck-phylop-pool",
        type=str,
        default="l2",
        choices=["mean", "max", "l2"],
        help="Reduction pooling per-nucleotide phyloP to the bottleneck bins. "
        "Default 'l2' (RMS — preserves conserved peaks plain mean washes out). "
        "CADD always uses 'max' (sparse: the most-deleterious base per bin).",
    )
    p.add_argument(
        "--aux-loss-type",
        type=str,
        default="huber",
        choices=["huber", "mse"],
        help="Regression loss for aux heads. Default 'huber' (SmoothL1; robust "
        "to phyloP's heavy tails / the −20 bigWig floor).",
    )
    # NOTE: phyloP/CADD standardization (mean/std) and per-task Huber beta are
    # module-level data constants (AUX_STANDARDIZE / AUX_HUBER_BETA), not CLI
    # flags — they are fixed properties of the v7 data, stamped on model.config.
    p.add_argument(
        "--aux-head-lr",
        type=float,
        default=None,
        help="Learning rate for the freshly-initialized aux regression head(s), "
        "set via a dedicated optimizer param group. Default None keeps aux heads "
        "in the normal optimizer groups, so they use the same LR as the backbone "
        "and MLM head. Head weights get weight decay; head biases do not.",
    )
    # --- Phase-2 B/C + CPL objectives ----------------------------------------
    p.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        help="HF LR scheduler ('linear', 'constant', 'constant_with_warmup', "
        "'cosine', 'cosine_with_min_lr', 'warmup_stable_decay'). Use "
        "'constant_with_warmup' for a fixed LR after warmup; 'cosine_with_min_lr' "
        "with --min-lr-rate for a one-cycle schedule that does not decay to zero.",
    )
    p.add_argument(
        "--min-lr-rate",
        type=float,
        default=None,
        help="Floor for 'cosine_with_min_lr', as a FRACTION of --lr (0.1 = decay to "
        "10%% of peak). Passed through as lr_scheduler_kwargs={'min_lr_rate': x}. "
        "Ignored by other schedulers. A one-cycle run that decays to exactly 0 "
        "spends its last stretch learning nothing; the schedule length is "
        "--max-steps, so set that to the intended full-run horizon even when "
        "--stop-at-genome ends the job early (a pilot then runs the production "
        "schedule and stops, instead of a compressed cycle).",
    )
    p.add_argument(
        "--cpl-lambda",
        type=float,
        default=0.0,
        help="CPL phase-swap contrastive weight (0=off, the default). Not recommended: a matched 100M ablation showed CPL costs every non-noise benchmark.",
    )
    p.add_argument(
        "--cpl-proj-dim", type=int, default=64, help="CPL projector output dim."
    )
    p.add_argument(
        "--cpl-swap-mode",
        choices=["directional", "phased"],
        default="directional",
        help="CPL twin construction. 'directional': swap the ‹/› direction markers "
        "(v6/directional data). 'phased': swap the ordered symbol to its reverse-order "
        "counterpart, Á<->Ç (v4/phased data, which has no ‹/› markers).",
    )
    p.add_argument(
        "--cpl-every",
        type=int,
        default=3,
        help="Run the CPL extra backbone pass every Nth step (compute saving; "
        "default 3 = ~66%% step-dropout).",
    )
    p.add_argument(
        "--cpl-margin",
        type=float,
        default=-1.0,
        help="Hinge margin for the CPL repulsive term: a het position stops "
        "contributing gradient once cos(orig,swap) falls below this. DEFAULT -1.0 == "
        "the original objective (relu(cos+1) == cos+1, same gradient), so the default "
        "recipe is unchanged. Raising it (e.g. -0.3) bounds the repulsion, which the "
        "raw term does not -- it saturated at cos=-1 in every logged run. A 1500-step "
        "8M A/B found no MLM/SMP difference outside noise: an unproven lever, so sweep "
        "it at a horizon where the saturation bites before adopting.",
    )
    p.add_argument(
        "--cpl-attract-weight",
        type=float,
        default=0.0,
        help="Weight of the CPL attractive/invariance term (0 = off, the default): "
        "positions whose token is IDENTICAL in orig and the phase-swapped twin get a "
        "(1-cos) penalty. EXPERIMENTAL: with global attention a marker swap "
        "legitimately changes distant states, so this can fight the long-range phase "
        "propagation CTP needs, and it costs 2.1x time / 2x peak memory.",
    )
    p.add_argument(
        "--smp-lambda",
        type=float,
        default=0.0,
        help="SMP supervised masked-phase weight (0=off, the default). Predicts "
        "the hidden ‹/› orientation at masked direction markers. DIRECTIONAL "
        "tokenization only (v4/phased bakes phase into the symbol → would leak).",
    )
    p.add_argument(
        "--smp-projector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SMP head shape: --smp-projector = LayerNorm→Linear→GELU→Linear(2) "
        "(pushes phase into the backbone); --no-smp-projector = shallow "
        "LayerNorm→Linear(2) on the backbone rep (the proj-vs-generality arm).",
    )
    p.add_argument(
        "--smp-mask-frac",
        type=float,
        default=0.4,
        help="Fraction of direction markers hidden per window (span+scattered mix, "
        "≥1 anchor kept). Higher = more positions forced to encode inferred phase; "
        "replaces the old marker-drop.",
    )
    p.add_argument(
        "--smp-target",
        choices=["marker", "base"],
        default="base",
        help="Where SMP reads/supervises: 'base' (default) = the c+1 het-base token, the "
        "SYM position the downstream frozen probe actually reads; 'marker' = the masked "
        "‹/› slot (the rep whose job is to reconstruct the missing orientation). The "
        "isolation grid showed 'base' is the probe-aligned position. Ablation lever.",
    )
    p.add_argument(
        "--smp-loo-frac",
        type=float,
        default=0.15,
        help="Per-row P(hide exactly one marker instead of a frac mix) — keeps the "
        "train distribution covering the eval probe.",
    )
    p.add_argument(
        "--smp-block-max",
        type=int,
        default=7,
        help="Max contiguous SMP span length in markers (the linkage component).",
    )
    p.add_argument(
        "--smp-w-hard",
        type=float,
        default=3.0,
        help="Loss up-weight at masked positions where copy-nearest is wrong "
        "(phase-switch boundaries) — directly optimizes lift-over-copy-nearest.",
    )
    p.add_argument(
        "--smp-every",
        type=int,
        default=1,
        help="SMP fires every Nth step (default 1; head-only, no extra backbone pass).",
    )
    p.add_argument(
        "--ctp-lambda",
        type=float,
        default=0.0,
        help="CTP supervised cis/trans pairwise weight (0=off; recommended ~0.2). Reads the "
        "two het-base SYM tokens the frozen compound-het probe reads, builds the SAME "
        "relational feature cat([z1,z2,z1-z2,z1*z2]), predicts the gauge-invariant "
        "cis/trans label. DIRECTIONAL tokenization only. Head-only, no extra backbone pass.",
    )
    p.add_argument(
        "--ctp-projector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CTP head shape: --ctp-projector = LayerNorm→Linear(4H,H)→GELU→Linear(2) "
        "(protects ancestry); --no-ctp-projector = shallow LayerNorm→Linear(4H,2) "
        "(max linear-readability — the frozen probe is linear; the proj-vs-generality arm).",
    )
    p.add_argument(
        "--ctp-pairs-per-row",
        type=int,
        default=8,
        help="Max het pairs sampled per row for CTP (distance-weighted sampling).",
    )
    p.add_argument(
        "--ctp-dist-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bias CTP pair sampling + loss toward LONGER inter-marker distance "
        "(counters the copy-nearest autocorrelation shortcut). --no-ctp-dist-weight = "
        "uniform (the Goodhart control arm).",
    )
    p.add_argument(
        "--ctp-min-dist",
        type=int,
        default=0,
        help="Drop CTP pairs whose two markers are closer than this many tokens.",
    )
    p.add_argument(
        "--ctp-mask-markers",
        action="store_true",
        help="CTP regime: hide BOTH markers of a pair (de-novo inference). Default off = "
        "markers VISIBLE, matching the benchmark probe's marker-present regime.",
    )
    p.add_argument(
        "--ctp-every",
        type=int,
        default=1,
        help="CTP fires every Nth step (default 1; head-only, no extra backbone pass).",
    )
    p.add_argument(
        "--trackb-lambda",
        type=float,
        default=0.0,
        help="Track-B (effect-ranked WT-vs-mut contrastive) weight (0=off).",
    )
    p.add_argument(
        "--trackb-proj-dim",
        type=int,
        default=128,
        help="Track-B displacement projector output dim.",
    )
    p.add_argument(
        "--trackb-every",
        type=int,
        default=3,
        help="Run the Track-B WT-twin pass every Nth step (offset from CPL so "
        "the two extra passes never stack).",
    )
    p.add_argument(
        "--trackc-af-weight",
        type=float,
        default=0.0,
        help="Track-C 1000G allele-frequency regression weight (0=off).",
    )
    p.add_argument(
        "--af-dir",
        type=str,
        default="/data/1kgenomes/af_tables",
        help="Dir of per-chrom af_<chrom>.npz (build_af_tables.py) for Track-C.",
    )
    p.add_argument(
        "--twobit",
        type=str,
        default="/data/diploid_glm_new/preprocess/hg38.2bit",
        help="hg38 .2bit for the Track-B wild-type reference base.",
    )
    p.add_argument(
        "--use-sample-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Weight EVERY example's loss (MLM + aux) by its per-record "
        "sample_weight (=1 normally, =10 for HG38_REF, summed over dedup "
        "duplicates → heavy-tailed for common regions), via the normalized "
        "weighted mean Σ(w·l)/Σw. ON by default for v7 (the baked sample_weight "
        "carries dedup-frequency + HG38_REF emphasis); pass --no-use-sample-weight "
        "for the legacy is_ref-only path. When ON, --ref-loss-weight auto-defaults "
        "to 1.0 so ref emphasis isn't double-counted (see its help).",
    )
    p.add_argument(
        "--sample-weight-transform",
        type=str,
        default="none",
        choices=["none", "sqrt", "log1p", "clip"],
        help="Transform applied to sample_weight before use, to tame the "
        "heavy-tailed dedup-summed weights. Default 'none' "
        "(sqrt(10)≈3.2x for ref). 'clip' = identity then cap; others also "
        "capped by --sample-weight-clip.",
    )
    p.add_argument(
        "--sample-weight-clip",
        type=float,
        default=20.0,
        help="Cap on the (transformed) sample_weight so a few ultra-common "
        "windows can't dominate the gradient. Default 20.0. Set 0 to disable.",
    )
    p.add_argument(
        "--main-trainable",
        type=str,
        default="all",
        help="Comma list of what stays TRAINABLE during main training; everything "
        "else is frozen. 'all' (default) = full fine-tune, unchanged behavior. "
        "Components: 'new_rows' (only the newly-added vocab rows of the input "
        "embedding + LM head, enforced by the same gradient-row masks the "
        "embedding warmup uses -- base ACGT rows stay bitwise-identical), "
        "'transformer' (core.transformer_blocks), 'conv' (core.stem + "
        "core.conv_tower_blocks), 'deconv' (core.deconv_tower_blocks). "
        "Example: 'new_rows,transformer'. Mutually exclusive with "
        "--peft-main lora (which brings its own freezing).",
    )
    p.add_argument(
        "--og2-manifest",
        type=str,
        default="/storage/users/ddofer/datasets/opengenome2/manifests/shards.tsv",
        help="OpenGenome2 replay pool manifest (shards.tsv). Only read when "
        "--og2-fraction > 0.",
    )
    p.add_argument(
        "--og2-fraction",
        type=float,
        default=0.0,
        help="Fraction of TRAINING TOKENS drawn from the OpenGenome2 replay pool "
        "instead of the human diploid corpus. 0 (default) disables replay and is "
        "byte-identical to the control. Must be 1/N for an integer N: replay "
        "substitutes WHOLE optimizer steps (0.20 -> every 5th), and every replay "
        "batch is token-matched to the human batch, so the step share IS the token "
        "share. Within replay, bins are drawn at NTv3's own 7.9/13.0/13.2/66.0 "
        "token mix over 1024/2048/4096/8192.",
    )
    p.add_argument(
        "--og2-species-id",
        type=int,
        default=10,
        help="Condition id stamped on OG2 replay batches. NOT a detail -- a real "
        "experimental variable. <unk>=0 is UNAVAILABLE: modeling_ntv3_posttrained "
        "asserts species_ids >= num_species_special_tokens (6), so the six special "
        "rows are rejected and no semantically-neutral option exists. Measured MLM "
        "loss on real OG2 windows (pretrain/tools/og2_species_probe.py, 1024+2048 "
        "bins): danio_rerio=10 1.339 < c_elegans=21 1.356 < arabidopsis=19 1.381 < "
        "human=27 1.413 < mouse=29 1.445. Default 10: lowest surprise on a pool "
        "that is 46%% metagenomes + 19%% GTDB, AND it keeps replay gradients off "
        "the human condition row this project depends on.",
    )
    p.add_argument(
        "--og2-workers",
        type=int,
        default=2,
        help="Dataloader workers PER BIN for replay (4 bins). Records are split "
        "across consumers by position, so workers do not duplicate windows.",
    )
    p.add_argument(
        "--l2sp",
        type=float,
        default=0.0,
        help="Decay toward the PRETRAINED weights instead of toward zero, with this "
        "lambda (0 = off). Standard --weight-decay pulls parameters to 0, which is "
        "the wrong target for continued pretraining: the solution worth keeping is "
        "theta0. Damage on this project tracks drift from theta0 (bundle drifts "
        "0.0827 at 5552 steps and loses splice 0.9387->0.8510; dipgentle drifts "
        "0.0162 and holds 0.9284). Applied decoupled, mirroring AdamW's decoupled "
        "weight decay. Costs ~0.5 GB for the theta0 copy. Typically pair with "
        "--weight-decay 0.",
    )
    p.add_argument(
        "--freeze-acgt-path",
        action="store_true",
        help="Pin everything a pure-ACGT sequence touches on the way into the "
        "trunk: base-vocab embedding rows, the OLD stem-conv input channels, and "
        "the stem-conv bias. Requires --token-embed-dim > the model's current "
        "width; the widening then zero-inits the base rows' new dims so the "
        "pure-ACGT stem output is BITWISE invariant for the whole run while the "
        "diploid rows and the new conv channels train freely. Motivation: damage "
        "on ACGT tasks tracks changes to the input path, and downstream movement "
        "is harmless -- but freezing the path outright also blocks diploid "
        "learning (txfreeze: diploid_mlm 2.454 vs bundle 0.935).",
    )
    p.add_argument(
        "--freeze-qk",
        action="store_true",
        help="Freeze every query/key projection (sa_layer.query_head / key_head) "
        "during main training. NTv3 is RoPE and has NO learnable positional "
        "parameter -- rotary_embedding.inv_freq is a frequency basis, measured "
        "bitwise-identical across 19 checkpoints -- so Q/K are the only learnable "
        "surface the rotation touches. Composes with --main-trainable. NOTE: Q/K "
        "also carry content-based attention, so this is 'attention geometry "
        "frozen', not a clean positional knob.",
    )
    # LoRA (used by gating-experiment arms C and D)
    p.add_argument(
        "--peft-main",
        type=str,
        default="none",
        choices=["none", "lora"],
        help="Main-training PEFT mode (NOT the embedding warmup, which is "
        "controlled by --peft-style-warmup). 'none' = full fine-tune (default); "
        "'lora' = wrap model with LoRA adapters after warmup, freezing the "
        "backbone except for the LoRA matrices.",
    )
    p.add_argument(
        "--lora-protect-base-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep base ACGT vocab rows [0:base_vocab_size) bitwise frozen under "
        "LoRA (default). PEFT's modules_to_save deep-copies the embedding and "
        "unfreezes the WHOLE tensor, so without this the base rows train -- at "
        "the LoRA arm's higher LR -- which is how the lora16 arm moved them "
        "0.20 relative Frobenius by genome-20 while claiming to be a gentler "
        "control. Pass --no-lora-protect-base-rows only for an arm whose "
        "QUESTION is whether base-embedding reorganization causes the damage.",
    )
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA scaling numerator; the update is scaled by alpha/r. Defaults to "
        "r (alpha/r = 1, the usual convention). The old fixed default of 32 made "
        "the scale depend on r behind your back: alpha/r = 0.5 at the default "
        "r=64 but 2.0 at r=16, so re-running 'the LoRA arm' without repeating "
        "--lora-r silently changed the effective update size 4x.",
    )
    p.add_argument(
        "--lora-target",
        type=str,
        default=(
            "query_head.linear,key_head.linear,value_head.linear,mha_output,"
            "fc1,fc2,"
            "stem.conv,conv.conv,conv_block.conv"
        ),
        help="Comma-separated list of module name SUFFIXES to LoRA-wrap. "
        "_maybe_wrap_lora resolves each suffix against the model and TYPE-FILTERS "
        "to PEFT-supported types (Linear, Embedding, Conv1d/2d/3d) so suffixes "
        "that collide with ConvTranspose1d or ModuleDict are silently skipped. "
        "Default covers: (a) all 6 Linear modules per NTv3 transformer block "
        "(Q/K/V/O attention + fc1/fc2 MLP); (b) all Conv1d in the encoder/conv "
        "tower (stem.conv + conv_tower_blocks.* main + residual convs) — this "
        "is the majority of NTv3's weight (~65%% in 8M) and lets the model "
        "adapt its genomic 'tokenizer' stage to human-specific motif statistics. "
        "The deconv tower (ConvTranspose1d) is NOT wrapped — PEFT does not "
        "support it natively; those modules stay frozen under LoRA mode. "
        "Embeddings and LM head are kept FULL-RANK trainable via modules_to_save "
        "(see _maybe_wrap_lora) — standard vocab-expansion recipe (Cui 2023).",
    )
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="OFF by default: NTv3 contains no nn.Dropout modules at all, so a "
        "non-zero value here makes the LoRA arm the only one carrying a "
        "regularizer -- an uncontrolled difference from every arm it is compared "
        "against.",
    )
    p.add_argument(
        "--mode-collapse-check-interval",
        type=int,
        default=20_000,
        help="Steps between mode-collapse checks (default 20_000; decoupled "
        "from save cadence to avoid hourly probes on small models).",
    )
    p.add_argument(
        "--benchmark-script",
        type=Path,
        default=None,
        help="Path to run_diploid_benchmarks.py. When set, GenomeMilestoneCallback "
        "spawns a background mp.Process that benchmarks each saved milestone "
        "(non-blocking — trainer keeps stepping). Expects benchmark_bg_runner.py "
        "alongside the script. Default None disables benchmarking. Recommended: "
        "/data/diploid_glm_new/phase_aware_pretrain/pretrain/run_diploid_benchmarks.py",
    )
    p.add_argument(
        "--benchmark-cuda",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES for the benchmark worker subprocess. Use a "
        "different device from training to avoid contention (e.g. train on 0, "
        "benchmark on 1).",
    )
    p.add_argument(
        "--benchmark-pythonpath",
        type=str,
        default=None,
        help="PYTHONPATH override for the benchmark worker (so its `benchmarks.*` "
        "imports resolve). Default None inherits parent env.",
    )
    p.add_argument(
        "--wait-benchmarks-on-train-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If on, the trainer waits for queued benchmarks to finish at the "
        "end of training. If --no-wait-benchmarks-on-train-end, the worker is "
        "a daemon and is killed when the parent exits.",
    )
    p.add_argument(
        "--mlm-prob",
        type=float,
        default=0.15,
        help="Base MLM masking probability (default 0.15).",
    )
    p.add_argument(
        "--marker-drop-prob",
        type=float,
        default=0.15,
        help="Probability of randomly DELETING a marker token (^, "
        "directionality arrows) from each input position. Forces the MLM "
        "head to also reconstruct sequences without leakage from variant "
        "markers. INDEL braces { } are always preserved. Set 0 to disable. "
        "Default 0.15 follows BERT/RoBERTa masking rate (Liu 2019); "
        "rates >0.30 may cause monotone degradation (Sennrich 2016).",
    )
    p.add_argument(
        "--marker-drop-target",
        choices=["direction", "snp", "all"],
        default="direction",
        help="Which marker(s) marker-drop deletes: 'direction' (default) = the ‹/› direction "
        "markers; 'snp' = only the ^ SNP marker (leaves ‹/› intact). Use 'snp' to give SMP/CTP "
        "the same ^-drop augmentation the published CPL recipe used (SMP/CTP need ‹/› present). "
        "'all' = every marker (^ { } ‹ ›); with --marker-drop-prob 1 the model trains on the "
        "marker-free in-place diploid input (SMP/CTP need the ‹/› markers, Track-B/C no drop at all).",
    )
    p.add_argument(
        "--eval-mlm-prob",
        type=float,
        default=0.15,
        help="Base MLM probability used at EVAL time (frozen across runs so "
        "eval losses stay comparable when --mlm-prob is swept).",
    )
    p.add_argument(
        "--load-best-model",
        action="store_true",
        help="Load best checkpoint (by eval_loss) at end of training.",
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push checkpoints to the HF Hub (private+gated). Requires "
        "HF_TOKEN env var. Each checkpoint becomes a branch on the repo.",
    )
    p.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="HF Hub repo id, e.g. GrimSqueaker/diploid-glm-NTv3-8M-pre. "
        "Required when --push-to-hub is set.",
    )

    args = p.parse_args()
    if args.cadd_sidecar is not None and not args.no_eval:
        raise SystemExit("--cadd-sidecar needs --no-eval: only the training split is rebuilt, so an evaluation "
                         "would score the CADD head on the old (rounded, averaged) targets")
    args.stride = resolve_stride(args.window_len, args.stride, args.overlap_frac)

    # Resolve ref loss-weight default based on whether per-example sample_weight
    # is in the loss. The baked sample_weight=10 on HG38_REF already supplies
    # loss-side ref emphasis, so stacking the historical 4.0 multiplier on top
    # would over-focus on ref (the v7 decision is "sampler-only extra").
    if args.og2_fraction > 0:
        # replay_schedule() also validates, but only at the FIRST TRAINING STEP --
        # measured 1098 s into a run before a typo was reported. Fail here instead.
        _period = 1.0 / args.og2_fraction
        if args.og2_fraction > 1.0 or abs(_period - round(_period)) > 1e-9:
            raise SystemExit(
                f"--og2-fraction {args.og2_fraction} must be 1/N for an integer N "
                "(1.0 = every step, 0.20 = every 5th, 0.10 = every 10th). Replay "
                "substitutes WHOLE optimizer steps, so anything else cannot be an "
                "exact period and would silently miss its token target."
            )
        # MixedReplayLoader.__len__ is the HUMAN loader's (og2_mix.py): an epoch horizon under
        # replay substitutes OG2 batches for ~og2 of the human windows, and genome-N milestones
        # (stretched by 1/(1-og2)) can land after the last step. run_100m_bundle_arm.sh passed
        # --max-steps -1 with og2 0.2 (run_100m_replay20.sh).
        if args.max_steps is None or args.max_steps <= 0:
            raise SystemExit(
                f"--og2-fraction {args.og2_fraction} needs an explicit --max-steps > 0: an epoch "
                "under replay covers only ~(1 - og2) of the human windows."
            )

    if args.token_budget:
        # Every one of these is a silent-wrong-result failure if left to runtime.
        if args.token_budget % BUCKET_STEP:
            raise SystemExit(
                f"--token-budget {args.token_budget} must be a multiple of "
                f"{BUCKET_STEP} so every bucket divides it exactly (and so the OG2 "
                "replay bins, which are all powers of two up to 8192, do too)."
            )
        if args.genome_steps is None:
            raise SystemExit(
                "--token-budget requires --genome-steps. Milestone pacing is "
                "derived from batch_size * grad_accum * world_size WINDOWS, which "
                "is meaningless once rows-per-batch varies 16x across buckets; "
                "genome-N would silently stop meaning what it means in every other "
                "ladder. Pin it in BASES: genome_steps = 352.4e6 / (token_budget * "
                "grad_accum * world_size), which reproduces phase-2's 44.83 at its "
                "7.86 Mbase/step."
            )
        if args.window_len <= BUCKET_STEP:
            logger.warning(
                "--token-budget with --window-len %d <= bucket step %d: every "
                "window buckets to the full window length, so this is exactly the "
                "fixed-length path with a base-sized batch. Intended for the "
                "equivalence check; on a real long-context run raise --window-len.",
                args.window_len,
                BUCKET_STEP,
            )

    if args.ref_loss_weight is None:
        args.ref_loss_weight = 1.0 if args.use_sample_weight else 4.0
    elif args.use_sample_weight and args.ref_loss_weight > 1.0:
        logger.warning(
            "Stacking --ref-loss-weight=%.2f on top of --use-sample-weight: "
            "HG38_REF already carries sample_weight=10 in the loss, so ref "
            "gets ~%.0fx loss emphasis (×%.1f sampler) — likely over-focusing "
            "on reference. Consider --ref-loss-weight 1.0 (sampler-only extra).",
            args.ref_loss_weight,
            args.ref_loss_weight * 10.0,
            args.ref_sampler_weight,
        )
    return args


def _cap_filter(t_filter: np.ndarray, cap: int, seed: int = 42) -> np.ndarray:
    if cap and len(t_filter) > cap:
        rng = np.random.default_rng(seed)
        t_filter = rng.choice(t_filter, size=cap, replace=False)
        t_filter.sort()
    return t_filter


def _v2_rewrite_from_args(args) -> V2Rewrite | None:
    """The ``--v2-input`` row rewrite, or None. Refuses the settings that cannot go with it."""
    if args.v2_reference is not None:
        if not args.v2_input:
            raise SystemExit("--v2-reference needs --v2-input")
        if not args.no_eval:
            raise SystemExit("--v2-reference needs --no-eval: the evaluation dataset is built without the reference")
    if args.v2_copy_swap and not args.v2_input:
        raise SystemExit("--v2-copy-swap needs --v2-input")
    if not args.v2_input:
        return None
    if args.marker_drop_target != "all" or args.marker_drop_prob != 1.0:
        raise SystemExit("--v2-input needs --marker-drop-target all --marker-drop-prob 1: no marker may reach the model")
    if args.smp_lambda or args.cpl_lambda:
        raise SystemExit("--v2-input needs --smp-lambda 0 --cpl-lambda 0: both phase losses act on direction markers")
    rewrite = V2Rewrite(args.tokens)
    if args.v2_reference is not None and not rewrite.reference_aware:
        raise SystemExit("--v2-reference needs the reference-aware token file (preprocess/diploid_tokens_v3.tsv)")
    return rewrite


def _fetch_eval_dataset(
    args, output_dir, char_to_id: np.ndarray, pad_id: int, v2_rewrite: V2Rewrite | None = None
) -> DiploidWindowDataset | dict[str, DiploidWindowDataset] | None:
    """Build eval dataset(s), failing hard unless eval is explicitly disabled.

    Returns:
      - None if eval disabled.
      - Single DiploidWindowDataset if no extra splits requested (back-compat).
      - dict with any of {"primary", "wt", "nonvariant"} if --eval-wt-windows
        or --eval-nonvariant-windows > 0. HF Trainer accepts a dict and emits
        per-split metrics (eval_primary_loss, eval_wt_loss,
        eval_nonvariant_loss). The "primary" split applies --eval-window-filter
        (default "all") and is the objective tracked by --load-best-model.
    """
    if args.no_eval or args.quick_test:
        return None

    # --eval-data takes priority over --test-data for evaluation
    eval_dir = args.eval_data if args.eval_data is not None else args.test_data

    if eval_dir in {
        DEFAULT_WINDOWED_TEST_DATA,
        DEFAULT_WINDOWED_TEST_DATA_FALLBACK,
    }:
        logger.info("Using canonical windowed eval dataset: %s", eval_dir)
    elif eval_dir == DEFAULT_RAW_TEST_DATA and any(
        _has_data_shards(candidate)
        for candidate in (
            DEFAULT_WINDOWED_TEST_DATA,
            DEFAULT_WINDOWED_TEST_DATA_FALLBACK,
        )
    ):
        logger.info(
            "Canonical windowed eval dataset exists but raw test data was requested explicitly: %s",
            eval_dir,
        )
    if not eval_dir.exists():
        raise SystemExit(
            f"Eval data dir not found: {eval_dir}. Pass --no-eval to disable evaluation intentionally."
        )
    files = list(eval_dir.glob("*.parquet")) + list(eval_dir.glob("*.arrow"))
    if not files:
        raise SystemExit(
            f"Eval data dir has no parquet/arrow shards: {eval_dir}. Pass --no-eval to disable evaluation intentionally."
        )

    test_hf = load_parquet_dataset(
        eval_dir,
        max_shards=getattr(args, "max_shards", None),
        cache_dir=getattr(args, "hf_cache_dir", None),
    )
    # Load + index once; reuse across variant and WT splits.
    t_idx = build_window_index(
        test_hf,
        args.window_len,
        args.stride,
        cache_dir=getattr(args, "hf_cache_dir", None),
    )

    # Variant split (primary objective).
    eval_mode = getattr(args, "eval_window_filter", "variant_only")
    variant_filter = filter_window_index(t_idx, mode=eval_mode)
    if len(variant_filter) == 0:
        raise SystemExit(
            f"Eval dataset contains 0 variant windows: {eval_dir}. Pass --no-eval to disable evaluation intentionally."
        )
    variant_filter = _cap_filter(variant_filter, args.eval_max_windows, seed=42)
    # Eval carries the same aux targets so phyloP eval metrics can be logged
    # (eval_loss itself stays MLM-only — see AuxRegressionLossMixin).
    aux_cols = AUX_POSITION_TASKS if getattr(args, "aux_heads", False) else ()
    variant_ds = DiploidWindowDataset(
        test_hf,
        t_idx,
        variant_filter,
        char_to_id,
        pad_id,
        args.window_len,
        aux_position_cols=aux_cols,
        v2_rewrite=v2_rewrite,
    )
    logger.info(
        "Eval[variant]: %d windows (filter=%s) from %d test records",
        len(variant_ds),
        eval_mode,
        len(test_hf),
    )

    splits: dict[str, DiploidWindowDataset] = {"primary": variant_ds}

    # WT split (HG38_REF, no variants).
    wt_cap = getattr(args, "eval_wt_windows", 0)
    if wt_cap and wt_cap > 0:
        if t_idx.is_ref is None:
            logger.warning(
                "--eval-wt-windows requested but window index has no is_ref "
                "column (legacy cache?). Skipping WT split. Rebuild the index "
                "to enable."
            )
        else:
            wt_filter = np.where(t_idx.is_ref & ~t_idx.has_variant)[0].astype(np.int64)
            if len(wt_filter) == 0:
                logger.warning(
                    "WT split: 0 windows match `is_ref & ~has_variant`. "
                    "Skipping WT split."
                )
            else:
                wt_filter = _cap_filter(wt_filter, wt_cap, seed=4242)
                splits["wt"] = DiploidWindowDataset(
                    test_hf,
                    t_idx,
                    wt_filter,
                    char_to_id,
                    pad_id,
                    args.window_len,
                    aux_position_cols=aux_cols,
                    v2_rewrite=v2_rewrite,
                )
                logger.info(
                    "Eval[wt]: %d windows (HG38_REF, no variants) from %d test records",
                    len(splits["wt"]),
                    len(test_hf),
                )

    # Non-variant patient split (`~is_ref & ~has_variant`). Catches drift on
    # boring patient regions that should still look like reference.
    nv_cap = getattr(args, "eval_nonvariant_windows", 0)
    if nv_cap and nv_cap > 0:
        if t_idx.is_ref is None:
            logger.warning(
                "--eval-nonvariant-windows requested but window index has no "
                "is_ref column (legacy cache?). Skipping nonvariant split. "
                "Rebuild the index to enable."
            )
        else:
            nv_filter = np.where(~t_idx.is_ref & ~t_idx.has_variant)[0].astype(np.int64)
            if len(nv_filter) == 0:
                logger.warning(
                    "Nonvariant split: 0 windows match `~is_ref & ~has_variant`. "
                    "Skipping nonvariant split."
                )
            else:
                nv_filter = _cap_filter(nv_filter, nv_cap, seed=4343)
                splits["nonvariant"] = DiploidWindowDataset(
                    test_hf,
                    t_idx,
                    nv_filter,
                    char_to_id,
                    pad_id,
                    args.window_len,
                    aux_position_cols=aux_cols,
                    v2_rewrite=v2_rewrite,
                )
                logger.info(
                    "Eval[nonvariant]: %d windows (non-REF, no variants) from %d test records",
                    len(splits["nonvariant"]),
                    len(test_hf),
                )

    # Back-compat: return a bare dataset when only the variant split exists,
    # so callers that don't handle dicts keep working.
    if len(splits) == 1:
        return variant_ds
    return splits


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M",
    )
    for noisy in ("urllib3", "requests", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="warmup_ratio is deprecated.*")
    warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0.*")
    hf_logging.set_verbosity_warning()

    args = parse_args()
    torch.set_float32_matmul_precision("high")

    # NTv3 is a ConvNet with fixed-size padded inputs; auto-tuning is safe
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    # torch 2.10's default SDPA dispatcher tries cuDNN attention first and
    # segfaults on NTv3's small bottleneck shapes (L_bot=16 or 64). Must be
    # disabled before any forward pass when --sdpa-attention is on.
    # The flag wins over the env var when given, and is written back into it so
    # apply_pad_mask_from_env stays the single place the contract is read. Unlike
    # the bare env var, args.pad_mask lands in run_args.json, so a checkpoint
    # records which arm it belongs to.
    if args.pad_mask is not None:
        os.environ["NTV3_PAD_MASK"] = "1" if args.pad_mask else "0"

    # Same contract as --pad-mask: the flag wins over the env var and is written
    # back into it, so _resolve_bypass stays the single place the value is read.
    if args.pool_bypass is not None:
        os.environ["NTV3_POOL_BYPASS"] = str(int(args.pool_bypass))

    if args.sdpa_attention:
        disable_cudnn_sdp()

    output_dir = args.output_dir
    temp_output = (
        tempfile.TemporaryDirectory(prefix="diploid_pretrain_")
        if args.no_save
        else None
    )
    if temp_output:
        output_dir = Path(temp_output.name)

    main_resume_from = args.resume_from
    if main_resume_from is None and not args.no_save:
        main_resume_from = find_latest_checkpoint(output_dir)

    warmup_dir = output_dir / "_emb_warmup"
    warmup_final_dir = warmup_dir / "final"
    warmup_resume_from: Path | None = None
    model_seed_checkpoint = main_resume_from
    if main_resume_from is None and not args.no_save:
        if _is_valid_checkpoint_dir(warmup_final_dir):
            model_seed_checkpoint = warmup_final_dir
            logger.info("Found completed embedding warmup at %s", warmup_final_dir)
        else:
            if warmup_final_dir.exists():
                logger.warning(
                    "Ignoring incomplete embedding warmup checkpoint at %s",
                    warmup_final_dir,
                )
            warmup_resume_from = find_latest_checkpoint(warmup_dir)
            if warmup_resume_from is not None:
                model_seed_checkpoint = warmup_resume_from
                logger.info("Resuming embedding warmup from %s", warmup_resume_from)

    model_seed_checkpoint = _wrapped_resume_seed(model_seed_checkpoint, output_dir)

    # Unigram frequencies from the whole-train-set counts table: used for the
    # new-token bias prior (init) and the MLM random-fill distribution.
    token_freqs = load_token_unigram_freqs(args.token_counts)

    # Unigram frequencies from the whole-train-set counts table: used for the
    # new-token bias prior (init) and the MLM random-fill distribution.
    token_freqs = load_token_unigram_freqs(args.token_counts)

    try:
        model, tokenizer, base_vocab_size, human_species_id = setup_model_and_tokenizer(
            args.tokens,
            output_dir,
            model_seed_checkpoint,
            args.model_name,
            not args.no_save,
            sdpa_attention=args.sdpa_attention,
            token_freqs=token_freqs,
            new_token_bias_offset=args.new_token_bias_offset,
            drop_unseen_tokens=args.drop_unseen_tokens,
            new_row_init=args.new_row_init,
        )
        is_post = human_species_id is not None
        if is_post and args.skip_post_track_heads:
            _patch_post_track_heads_for_pretraining(model)

        # Build char->token_id lookup once, share across all datasets
        char_to_id = _build_char_to_id_table(tokenizer)
        _assert_lut_tokenizer_parity(
            tokenizer, char_to_id, expect_mapped=_parse_diploid_tsv(args.tokens)[0]
        )  # guard: encode()==LUT (catches shadowing); warns on unmapped TSV symbols
        pad_id = tokenizer.pad_token_id or 1

        train_hf = load_parquet_dataset(
            args.train_data,
            max_shards=args.max_shards or (2 if args.quick_test else None),
            cache_dir=args.hf_cache_dir,
            cadd_dir=args.cadd_sidecar,
        )
        if args.cadd_sidecar is not None and not (args.max_shards or args.quick_test):  # a shard subset cannot hold the whole keep-list
            check_cadd_sidecar_keep_list(train_hf, args.cadd_sidecar, args.keep_rows)

        w_idx = build_window_index(
            train_hf, args.window_len, args.stride, cache_dir=args.hf_cache_dir
        )

        # Optional genome subset: filter windows post-index so the full-corpus
        # window-index cache stays shared across subset and full runs.
        keep_win = None
        if args.max_genomes:
            rec_mask = genome_keep_mask(
                train_hf, args.max_genomes, args.max_genomes_seed, output_dir
            )
            if rec_mask is not None:
                keep_win = rec_mask[w_idx.record_idx]
        if args.exclude_genomes:
            excl_win = genome_exclude_mask(train_hf, args.exclude_genomes)[
                w_idx.record_idx
            ]
            keep_win = excl_win if keep_win is None else (keep_win & excl_win)

        # Row keeplists (semantic dedup). Separate lists for the two stages: the
        # vocab-warmup subset is capped per locus, which flattens locus
        # frequencies -- correct for warming new token rows, wrong for anything
        # that should see the corpus distribution.
        def _and_keeplist(
            base: np.ndarray | None, path: Path | None
        ) -> tuple[np.ndarray | None, np.ndarray | None]:
            rec_mask, rec_weight = row_keep_mask(train_hf, path)
            if rec_mask is None:
                return base, None
            win_mask = rec_mask[w_idx.record_idx]
            return (win_mask if base is None else (base & win_mask)), rec_weight

        # Only MAIN training takes a dedup .parquet's cluster_weight; the warmup
        # subset is capped per locus, so its weights mean nothing -- mask only.
        keep_win_main, keep_weight_main = _and_keeplist(keep_win, args.keep_rows)
        keep_win_warmup, _ = _and_keeplist(keep_win, args.warmup_keep_rows)

        def _subset(fi: np.ndarray, keep: np.ndarray | None) -> np.ndarray:
            return fi[keep[fi]] if keep is not None else fi

        def _genome_subset(fi: np.ndarray) -> np.ndarray:
            return _subset(fi, keep_win_main)

        warmup_window_len = min(
            1024, args.window_len
        )  # shorter window during embedding warmup for faster iterations
        v2_rewrite = _v2_rewrite_from_args(args)
        warmup_ds = DiploidWindowDataset(
            train_hf,
            w_idx,
            _subset(filter_window_index(w_idx, "variant_only_no_ref"), keep_win_warmup),
            char_to_id,
            pad_id,
            warmup_window_len,  # shorter window, same index start positions
            v2_rewrite=v2_rewrite,
            reference_fasta=args.v2_reference,
            copy_swap=args.v2_copy_swap,
        )
        train_ds = DiploidWindowDataset(
            train_hf,
            w_idx,
            _genome_subset(
                filter_window_index(
                    w_idx, "variant_enriched", args.nonvariant_keep_frac, args.seed
                )
            ),
            char_to_id,
            pad_id,
            args.window_len,
            aux_position_cols=AUX_POSITION_TASKS if args.aux_heads else (),
            # Track-B/C need each window's genomic start for the 2bit/AF lookups.
            emit_genomic=(args.trackb_lambda > 0 or args.trackc_af_weight > 0),
            window_jitter=args.window_jitter,
            # Only the main dataset buckets. warmup_ds runs at min(1024, window_len),
            # below one bucket, so bucketing it would be a no-op; eval stays
            # fixed-length so eval_loss remains comparable across runs.
            bucket_step=BUCKET_STEP if args.token_budget else 0,
            # Survivors carry their dedup cluster's individuals (None: own weight).
            record_weight=keep_weight_main,
            v2_rewrite=v2_rewrite,
            reference_fasta=args.v2_reference,
            copy_swap=args.v2_copy_swap,
        )
        eval_ds = _fetch_eval_dataset(args, output_dir, char_to_id, pad_id, v2_rewrite)

        # Patient-level overlap audit (sample_id intersection between train/eval).
        # Affects internal eval_loss interpretation only; external benchmarks
        # (ClinVar/HGMD) run elsewhere and are unaffected. HG38_REF is shared
        # by design and excluded.
        eval_ds_iter = (
            eval_ds.items()
            if isinstance(eval_ds, dict)
            else ([("eval", eval_ds)] if eval_ds is not None else [])
        )
        for split_name, split_ds in eval_ds_iter:
            try:
                train_ids = set(train_hf.unique("sample_id"))
                eval_ids = set(split_ds.hf_dataset.unique("sample_id"))
                overlap = (train_ids & eval_ids) - {REF_SAMPLE_ID}
                if overlap:
                    sample = sorted(overlap)[:5]
                    logger.warning(
                        "Patient-level overlap [%s]: %d samples in both train "
                        "and eval: %s%s. Internal eval_loss inflated; external "
                        "benchmarks unaffected.",
                        split_name,
                        len(overlap),
                        sample,
                        " ..." if len(overlap) > 5 else "",
                    )
            except (KeyError, AttributeError) as exc:
                logger.debug(
                    "Skipping sample_id overlap check [%s]: %s", split_name, exc
                )

        _world_size = _configured_world_size()
        if args.max_steps > 0:
            est_steps = args.max_steps
        elif args.token_budget and train_ds._bucket_len is not None:
            # Rows/step varies under bucketing; padded BASES/step does not.
            _total_padded = int(train_ds._bucket_len.astype(np.int64).sum())
            est_steps = max(
                1,
                (_total_padded // (args.token_budget * args.grad_accum * _world_size))
                * args.epochs,
            )
        else:
            est_steps = max(
                1,
                (len(train_ds) // args.batch_size // args.grad_accum // _world_size)
                * args.epochs,
            )
        genome_plan = plan_genome_milestones(
            train_hf,
            w_idx,
            train_ds.filter_indices,
            args.save_at_genomes,
            args.batch_size,
            args.grad_accum,
            est_steps,
            ref_sampler_weight=args.ref_sampler_weight,
            genome_steps=args.genome_steps,
            og2_fraction=float(args.og2_fraction or 0.0),
            # _bucket_len is already per KEPT window (len == len(filter_indices));
            # indexing it with filter_indices overruns -- it did, on 2026-09-08.
            windows_per_step=(
                args.token_budget * args.grad_accum * _world_size
                / float(np.mean(train_ds._bucket_len))
                if args.token_budget and train_ds._bucket_len is not None
                else None
            ),
        )
        if genome_plan.n_genomes > 0:
            logger.info(
                "Genome-equivalent pacing: %d genomes, %d non-ref train windows, "
                "%.1f windows/genome, %.2f optimizer steps/genome-equivalent",
                genome_plan.n_genomes,
                genome_plan.total_nonref_windows,
                genome_plan.windows_per_genome,
                genome_plan.steps_per_genome,
            )
            if args.stop_at_genome is not None and not any(
                args.stop_at_genome in g for g in genome_plan.milestone_steps.values()
            ):
                raise SystemExit(
                    f"--stop-at-genome {args.stop_at_genome} is not a scheduled "
                    f"milestone (--save-at-genomes {args.save_at_genomes}, effective "
                    f"genomes={genome_plan.n_genomes}). The run would never stop and "
                    "would train the full epoch instead -- a guarantee the old "
                    "external kill-watcher could not make."
                )
            if args.save_at_genomes and not genome_plan.milestone_steps:
                logger.warning(
                    "No genome checkpoints were scheduled. Requested milestones=%s; "
                    "effective genomes=%d; estimated_total_steps=%d.",
                    args.save_at_genomes,
                    genome_plan.n_genomes,
                    est_steps,
                )

        # Resolve marker-drop token ids (‹, ›) — INDEL braces are kept so the
        # bracketed span structure remains parseable. SNP marker (^) is NOT
        # in the drop pool: variants are always known at inference time, so
        # simulating missing ^ markers introduces unrealistic noise.
        marker_drop_ids = _resolve_marker_drop_ids(
            tokenizer, args.marker_drop_prob, args.marker_drop_target
        )
        _cpl_on = args.cpl_lambda > 0
        if _cpl_on:
            # Loud, because CPL's cost is easy to miss: it does not show up in the
            # MLM loss, only in downstream benchmarks. A matched 100M three-way
            # ablation put it at gue_splice -0.028, traitgym_mendelian -0.042,
            # traitgym_complex -0.026, songlab_clinvar -0.045, brca1 -0.129 --
            # i.e. it costs every task that carries signal, including the ClinVar
            # gain CPT is supposed to deliver. Keep it on only for a run that is
            # ABOUT CPL.
            logger.warning(
                "CPL IS ON (--cpl-lambda %.3g). A matched 100M ablation found it "
                "costs every signal-carrying benchmark (splice -0.028, mendelian "
                "-0.042, clinvar -0.045, brca1 -0.129). Unless this run is about "
                "CPL, pass --cpl-lambda 0.",
                args.cpl_lambda,
            )
        _tb_on = args.trackb_lambda > 0
        _tc_on = args.trackc_af_weight > 0
        _smp_on = args.smp_lambda > 0
        _ctp_on = args.ctp_lambda > 0
        if _cpl_on or _tb_on or _tc_on or _smp_on or _ctp_on:
            collator = MultiObjectiveCollator(
                tokenizer,
                base_vocab_size,
                mlm_prob=args.mlm_prob,
                token_freqs=token_freqs,
                marker_drop_token_ids=marker_drop_ids,
                marker_drop_prob=args.marker_drop_prob,
                pad_id=pad_id,
                tokens_path=str(args.tokens),
                twobit_path=args.twobit,
                af_dir=args.af_dir,
                cpl_enable=_cpl_on,
                cpl_swap_mode=args.cpl_swap_mode,
                trackb_enable=_tb_on,
                trackc_enable=_tc_on,
                smp_enable=_smp_on,
                smp_mask_frac=args.smp_mask_frac,
                smp_loo_frac=args.smp_loo_frac,
                smp_block_max=args.smp_block_max,
                smp_w_hard=args.smp_w_hard,
                smp_target=args.smp_target,
                ctp_enable=_ctp_on,
                ctp_pairs_per_row=args.ctp_pairs_per_row,
                ctp_dist_weight=args.ctp_dist_weight,
                ctp_min_dist=args.ctp_min_dist,
                ctp_mask_markers=args.ctp_mask_markers,
            )
        else:
            collator = MarkerDropMLMCollator(
                tokenizer,
                base_vocab_size,
                mlm_prob=args.mlm_prob,
                token_freqs=token_freqs,
                marker_drop_token_ids=marker_drop_ids,
                marker_drop_prob=args.marker_drop_prob,
                pad_id=pad_id,
            )
        # Eval collator: fixed canonical rates and NO marker dropout so eval
        # losses stay comparable across runs that change masking hyperparams.
        # Except under --v2-input, where no marker may reach the model: every
        # marker is dropped (p=1 is deterministic, so eval stays comparable).
        eval_collator = MarkerDropMLMCollator(
            tokenizer,
            base_vocab_size,
            mlm_prob=args.eval_mlm_prob,
            token_freqs=token_freqs,
            marker_drop_token_ids=marker_drop_ids if v2_rewrite is not None else (),
            marker_drop_prob=1.0 if v2_rewrite is not None else 0.0,
            pad_id=pad_id,
        )
        phased_pair_ids = _build_phased_pair_diagnostic_ids(tokenizer, args.tokens)
        callbacks: list[TrainerCallback] = [
            EmbeddingDiagnosticsCallback(
                base_vocab_size,
                log_every_n_steps=args.diag_every_n_steps,
                phased_pair_ids=phased_pair_ids,
            ),
            EvalSplitDiffCallback(),
            ThroughputCallback(
                window_len=args.window_len, token_budget=args.token_budget
            ),
        ]

        if args.mode_collapse_check:
            callbacks.append(
                ModeCollapseCallback(
                    enabled=True,
                    interval_steps=args.mode_collapse_check_interval,
                    base_vocab_size=base_vocab_size,
                )
            )

        if args.max_wall_seconds > 0:
            callbacks.append(
                TimedStopCallback(max_seconds=float(args.max_wall_seconds))
            )

        if (
            args.save_at_genomes
            and genome_plan.n_genomes > 1
            and genome_plan.milestone_steps
            and not args.no_save
        ):
            if args.benchmark_script is not None and not args.benchmark_script.exists():
                raise SystemExit(
                    f"--benchmark-script {args.benchmark_script} does not exist. "
                    "Pass a valid path or omit the flag to disable benchmarking."
                )
            callbacks.append(
                GenomeMilestoneCallback(
                    genome_plan.milestone_steps,
                    output_dir,
                    tokenizer,
                    stop_after_genome=args.stop_at_genome,
                    hub_model_id=args.hub_model_id if args.push_to_hub else None,
                    branch_prefix=_hub_branch_prefix(output_dir),
                    benchmark_script=args.benchmark_script,
                    benchmark_cuda_visible_devices=args.benchmark_cuda,
                    benchmark_pythonpath=args.benchmark_pythonpath,
                    wait_for_benchmarks_on_train_end=args.wait_benchmarks_on_train_end,
                )
            )

        _effective_workers = (
            0
            if args.quick_test
            else _amortized_worker_count(args.dataloader_workers, args.max_steps)
        )
        if _effective_workers != args.dataloader_workers:
            logger.info(
                "Dataloader workers %d -> %d: --max-steps %d is below the %d-step "
                "amortization threshold, and worker fork/prefetch is a fixed "
                "startup cost paid before step 1.",
                args.dataloader_workers,
                _effective_workers,
                args.max_steps,
                WORKER_AMORTIZE_STEPS,
            )

        base_kws = dict(
            per_device_train_batch_size=args.batch_size,
            dataloader_num_workers=_effective_workers,
            dataloader_pin_memory=True,
            dataloader_drop_last=True,
            bf16=True,
            gradient_accumulation_steps=args.grad_accum,
            seed=args.seed,
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_persistent_workers=_effective_workers > 0,
        )
        if _effective_workers > 1:
            base_kws["dataloader_prefetch_factor"] = 3

        push_to_hub_enabled = bool(args.push_to_hub) and not args.no_save
        if push_to_hub_enabled and not args.hub_model_id:
            raise SystemExit("--push-to-hub requires --hub-model-id")
        if push_to_hub_enabled:
            if not _ensure_hub_repo_private_gated(args.hub_model_id):
                raise SystemExit(
                    "HF Hub preflight failed for --hub-model-id. "
                    "Check authentication/permissions, or disable with --no-push-to-hub."
                )

        # Stage-2 honors --warmup-steps. The original concern was a 5x LR
        # spike (S1 terminal ~4e-5 → S2 peak 2e-4 over 5k warmup) that took
        # ~250k steps to recover. With S2 peak now at 5e-5 (similar to S1
        # terminal scale) and warmup at 1k, the ramp is gentle.
        resuming_for_continuation = main_resume_from is not None
        effective_warmup_steps = args.warmup_steps
        if resuming_for_continuation:
            # Warmup is always honored on resume. The old --no-restart-lr-schedule
            # flag claimed to suppress it but had stopped doing so; removed 2026-08-25.
            logger.info(
                "Resuming for continuation: warmup_steps=%d, lr=%.2e.",
                args.warmup_steps,
                args.lr,
            )
        use_hf_torch_compile = (
            not args.quick_test and not args.no_compile and not args.compile_backbone
        )
        train_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=1.0,
            adam_beta1=0.9,
            adam_beta2=0.98,
            warmup_steps=effective_warmup_steps,
            max_steps=args.max_steps,
            optim="adamw_torch_fused",
            lr_scheduler_type=args.lr_scheduler_type,
            lr_scheduler_kwargs=(
                {"min_lr_rate": args.min_lr_rate}
                if args.min_lr_rate is not None
                and args.lr_scheduler_type in ("cosine_with_min_lr", "cosine_warmup_with_min_lr")
                else {}
            ),
            logging_steps=20,
            save_strategy="no" if args.no_save else "steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=args.eval_steps if eval_ds is not None else None,
            eval_on_start=args.eval_on_start and eval_ds is not None,
            load_best_model_at_end=args.load_best_model and eval_ds is not None,
            metric_for_best_model=(
                ("eval_primary_loss" if isinstance(eval_ds, dict) else "eval_loss")
                if (args.load_best_model and eval_ds is not None)
                else None
            ),
            greater_is_better=(
                False if (args.load_best_model and eval_ds is not None) else None
            ),
            report_to=args.report_to,
            ddp_find_unused_parameters=is_post,  # _post: bigwig/BED heads unused
            # NTv3's rotary cos/sin caches are buffers that grow to the longest sequence each rank has seen.
            # With length buckets, ranks see different lengths, the per-step buffer broadcast then has a
            # different shape on each rank and NCCL hangs (found on the short indel-free rows, 2026-10-14).
            # Every NTv3 buffer is deterministic (inv_freq, the caches, zero bias_k/v), so nothing needs it.
            ddp_broadcast_buffers=False,
            torch_compile=use_hf_torch_compile,
            torch_compile_mode=args.torch_compile_mode
            if use_hf_torch_compile
            else None,
            # OG2 replay wraps the dataloader in MixedReplayLoader, which cannot
            # support accelerate's skip_first_batches (it reads .batch_sampler, and
            # rebuilding a plain DataLoader from it would discard both the mixer and
            # the MLM collator). Skipping batches is an optimisation, not a
            # correctness requirement: the replay schedule is keyed on
            # state.global_step, so a resume reproduces it exactly either way.
            # Also true under --token-budget: accelerate's skip_first_batches reads
            # .batch_sampler and rebuilds a plain DataLoader from it, which would drop
            # the length-bucket grouping and hand the collator a mixed-length batch.
            ignore_data_skip=bool(args.og2_fraction > 0 or args.token_budget),
            push_to_hub=False,
            hub_model_id=None,
            hub_strategy="end",
            hub_private_repo=None,
            **base_kws,
        )

        if (
            args.embedding_warmup_steps > 0
            and main_resume_from is None
            and not _is_valid_checkpoint_dir(warmup_final_dir)
        ):
            logger.info(
                "Embedding warmup: %d steps (variant-only, %s)",
                args.embedding_warmup_steps,
                "PEFT-style: only new-token rows train"
                if args.peft_style_warmup
                else "all input + LM-head rows train (legacy)",
            )
            # In all warmup modes the backbone is frozen.
            for param in model.parameters():
                param.requires_grad = False
            for param in model.get_input_embeddings().parameters():
                param.requires_grad = True
            for param in model.get_output_embeddings().parameters():
                param.requires_grad = True

            # PEFT-style: install gradient-row hooks so only the NEW token rows
            # (>= base_vocab_size) receive updates during warmup. The original
            # ACGTN rows remain bitwise-identical after warmup, preventing the
            # high-LR warmup phase from drifting base-vocab representations
            # away from NTv3's pretraining manifold.
            peft_hook_handles: list[Any] = []
            if args.peft_style_warmup:
                in_w = model.get_input_embeddings().weight
                out_embs = model.get_output_embeddings()
                out_w = out_embs.weight
                bvs = base_vocab_size
                masked_params = [in_w]
                if out_w is not in_w:
                    masked_params.append(out_w)
                # The head BIAS is per-vocab-row regardless of weight tying, so
                # it needs the mask either way — unmasked, base-vocab priors
                # drift during the high-LR warmup.
                out_b = getattr(out_embs, "bias", None)
                if out_b is not None and out_b.shape[0] == out_w.shape[0]:
                    masked_params.append(out_b)
                peft_hook_handles.extend(
                    _install_new_row_grad_masks(masked_params, bvs)
                )
                logger.info(
                    "PEFT-style warmup: installed gradient-row mask on rows "
                    "[0:%d) of %d parameter tensors: %s.",
                    bvs,
                    len(masked_params),
                    ", ".join(
                        ["input embeddings"]
                        + (["LM head weight"] if out_w is not in_w else ["(head tied)"])
                        + (["LM head bias"] if out_b is not None else [])
                    ),
                )

            # Snapshot base rows before warmup so we can verify they survive.
            base_in_snapshot = (
                model.get_input_embeddings()
                .weight.data[:base_vocab_size]
                .detach()
                .clone()
                if args.peft_style_warmup
                else None
            )

            warmup_kws = dict(base_kws)
            warmup_kws.pop("dataloader_prefetch_factor", None)
            # Lower the warmup LR multiplier when only new rows train. The
            # legacy lr*8 was calibrated for warming up all embeddings; on
            # just ~70 new rows it's overkill (~30x base-pretrain peak).
            warmup_lr_mult = args.embedding_warmup_lr_mult
            if warmup_lr_mult is None:
                warmup_lr_mult = 2.0 if args.peft_style_warmup else 8.0
            warmup_args = TrainingArguments(
                output_dir=str(output_dir / "_emb_warmup"),
                num_train_epochs=1,
                max_steps=args.embedding_warmup_steps,
                learning_rate=args.lr * warmup_lr_mult,
                weight_decay=0.001,
                warmup_steps=max(1, min(100, args.embedding_warmup_steps // 10)),
                logging_steps=max(1, args.embedding_warmup_steps // 10),
                save_strategy="no" if args.no_save else "steps",
                save_steps=max(1, min(500, max(1, args.embedding_warmup_steps // 4))),
                save_total_limit=1,
                # Mirror main run's reporting so warmup grad_norm/lr/loss are
                # visible in wandb (they were silently hidden before, masking
                # whether the new embeddings even moved).
                report_to=args.report_to,
                run_name=(
                    f"{os.environ.get('WANDB_RUN_NAME')}_warmup"
                    if os.environ.get("WANDB_RUN_NAME")
                    else None
                ),
                ddp_find_unused_parameters=False,
                ddp_broadcast_buffers=False,  # see the main stage's TrainingArguments
                ignore_data_skip=True,
                **warmup_kws,
            )
            warmup_trainer = _make_trainer(
                is_post,
                human_species_id,
                model=model,
                args=warmup_args,
                train_dataset=warmup_ds,
                data_collator=collator,
            )
            warmup_trainer.train(
                resume_from_checkpoint=(
                    str(warmup_resume_from) if warmup_resume_from is not None else None
                )
            )
            if not args.no_save and warmup_trainer.is_world_process_zero():
                _atomic_save_dir(
                    warmup_final_dir,
                    lambda save_dir: _save_model_artifacts(
                        warmup_trainer.model, tokenizer, save_dir
                    ),
                )
            logger.info("Embedding warmup complete.")

            # Remove the gradient-row hooks before main training resumes
            # full-parameter updates.
            for handle in peft_hook_handles:
                handle.remove()

            # Verify base rows survived if PEFT-style warmup was used. This
            # catches accidental regressions (e.g. someone bypasses the hook).
            if base_in_snapshot is not None:
                in_w_after = model.get_input_embeddings().weight.data[:base_vocab_size]
                if not torch.equal(in_w_after, base_in_snapshot.to(in_w_after.device)):
                    drift = (
                        (in_w_after - base_in_snapshot.to(in_w_after.device))
                        .norm()
                        .item()
                    )
                    logger.warning(
                        "PEFT-style warmup expected zero drift on base rows; "
                        "observed L2 drift = %.6f (this should be exactly 0).",
                        drift,
                    )
                else:
                    logger.info(
                        "PEFT-style warmup verified: %d base-vocab rows bitwise-identical.",
                        base_vocab_size,
                    )

                # The base-row check above proves what did NOT move; this reports
                # whether what DID move came out healthy. New rows initialised far
                # off the base manifold (or collapsed toward zero) are the failure
                # this phase exists to prevent, and nothing was reporting it.
                # NB: read the FULL weight, not `in_w_after` -- that is already
                # sliced to the base rows, so slicing it again for the new rows
                # yields an empty tensor and a nan mean.
                _w = model.get_input_embeddings().weight.data
                _base_n = _w[:base_vocab_size].norm(dim=1).mean().item()
                _new_n = _w[base_vocab_size:].norm(dim=1).mean().item()
                logger.info(
                    "Post-warmup new-row scale: mean L2 %.4f vs base %.4f "
                    "(ratio %.2f; healthy is ~1, and far from it means the new "
                    "embeddings are off the base manifold the frozen tower expects).",
                    _new_n,
                    _base_n,
                    (_new_n / _base_n) if _base_n else float("nan"),
                )

            for param in model.parameters():
                param.requires_grad = True

            # Force destruction of the heavy optimizer & trainer states
            del warmup_trainer.optimizer
            del warmup_trainer.lr_scheduler
            del warmup_trainer

            # Force-close the warmup wandb run so the main trainer's
            # wandb.init() creates a NEW run with the unsuffixed name
            # (otherwise wandb sees a still-active run and appends main
            # metrics to the `_warmup`-named run — confirmed bug in the
            # weekend_20260521_161422 run; all post-warmup metrics ended
            # up in `..._warmup` runs because no finish() was called here).
            try:
                import wandb as _wandb

                if _wandb.run is not None:
                    _wandb.finish()
            except ImportError:
                pass

            # Free warmup optimizer/scheduler memory before main training.
            gc.collect()
            torch.cuda.empty_cache()

            # Post-warmup sanity log: re-check the new/base row-norm ratio
            # AFTER the warmup has updated the new rows. The pre-warmup log
            # in setup_model_and_tokenizer only captured init state; this
            # captures what main training actually starts from.
            logger.info("[post-warmup sanity] re-running init-scale check...")
            _log_init_scale_check(model, base_vocab_size)

        # Widening also happens AFTER the warmup, so a widened arm and its
        # 16-dim sibling share a bitwise-identical warmup and differ in exactly
        # one thing. Before the LoRA wrap, which would bury core.* under
        # base_model.model.*.
        _widen_old_dim = int(model.config.token_embed_dim)
        if args.token_embed_dim is not None:
            _widen_old_dim = _widen_token_embed_dim(
                model, args.token_embed_dim, zero_base_new_dims=args.freeze_acgt_path
            )
            logger.info("[post-widen sanity] re-running init-scale check...")
            _log_init_scale_check(model, base_vocab_size)

        # LoRA wrap happens AFTER the embedding warmup so the warmup gradient
        # hooks aren't interfered with. For --peft-main=none this is a no-op.
        model = _maybe_wrap_lora(model, args, base_vocab_size=base_vocab_size)

        # Selective freezing for main training (--main-trainable). Reuses the
        # embedding warmup's machinery; see _apply_main_trainable.
        if args.main_trainable.strip().lower() != "all":
            if args.peft_main != "none":
                raise SystemExit(
                    "--main-trainable and --peft-main lora are mutually exclusive: "
                    "LoRA brings its own freezing."
                )
            _apply_main_trainable(model, args.main_trainable, base_vocab_size)

        if args.freeze_qk:
            _freeze_qk(model)

        if args.freeze_acgt_path:
            _freeze_acgt_input_path(model, base_vocab_size, _widen_old_dim)

        # AFTER every freeze/LoRA/widen step, so theta0 is captured from the final
        # parameter set and frozen tensors are never anchored.
        _l2sp_anchor = None
        if args.l2sp > 0:
            _l2sp_anchor = L2SPAnchor(model, args.l2sp)
            logger.info(
                "--l2sp %.4g: anchoring %d trainable tensors to their pretrained "
                "values (decay toward theta0, NOT toward zero). weight_decay=%.4g "
                "still pulls toward zero -- set it to 0 unless you mean both.",
                args.l2sp,
                len(_l2sp_anchor.params),
                args.weight_decay,
            )

        # Aux-head wrap happens AFTER warmup + LoRA so warmup trains the bare
        # backbone and the heads ride on top of the (optionally LoRA-wrapped)
        # model. Applies to both _pre and _post backbones (post models share the
        # NTv3 conv/transformer/deconv tower; hidden_states[-1] is full-resolution
        # (B, L, conv_init_embed_dim) and species_ids condition via modulation /
        # output-head selection, NOT a prepended token — so per-position aux
        # targets stay 1:1 with the sequence). Requires the aux column(s) present.
        aux_standardize: dict[str, tuple[float, float]] = {}
        aux_weights: dict[str, float] = {}
        aux_betas: dict[str, float] = {}
        aux_bottleneck: tuple[str, ...] = ()
        aux_bottleneck_weights: dict[str, float] = {}
        aux_bottleneck_pool: dict[str, str] = {}
        if args.aux_heads:
            active_tasks = tuple(
                t for t in AUX_POSITION_TASKS if t in train_hf.column_names
            )
            if not active_tasks:
                logger.warning(
                    "--aux-heads set but none of %s present in the dataset "
                    "(%s); training MLM-only.",
                    AUX_POSITION_TASKS,
                    train_hf.column_names,
                )
            else:
                # Track-A: bottleneck heads only for requested tasks that are
                # actually active (present in the data).
                aux_bottleneck = tuple(
                    t for t in args.aux_bottleneck_tasks if t in active_tasks
                )
                # Track-C 'af' is a collator-produced target (no dataset column);
                # add it as a regression task so a head predicts it and the loss
                # mixin pops it. CPL / Track-B add contrastive projectors.
                enable_trackc = args.trackc_af_weight > 0
                model_tasks = active_tasks + (("af",) if enable_trackc else ())
                model = AuxHeadsModel(
                    model,
                    model_tasks,
                    bottleneck_tasks=aux_bottleneck,
                    bottleneck_layer=args.aux_bottleneck_layer,
                    cpl_proj_dim=args.cpl_proj_dim if args.cpl_lambda > 0 else 0,
                    trackb_proj_dim=args.trackb_proj_dim
                    if args.trackb_lambda > 0
                    else 0,
                    smp_proj=(args.smp_projector if args.smp_lambda > 0 else None),
                    ctp_proj=(args.ctp_projector if args.ctp_lambda > 0 else None),
                    cpl_margin=args.cpl_margin,
                    cpl_attract_weight=args.cpl_attract_weight,
                )
                # Per-task (weight, mean, std) registry; only the active tasks are
                # wired, so a dataset missing a column simply omits it. Constants
                # are stamped on config (per-task fields keyed by task name, plus
                # aux_tasks) so eval/inference can reconstruct standardization.
                # Loss WEIGHT is the tunable knob (CLI); standardization (mean/std)
                # and per-task Huber beta come from the module-level data constants
                # (AUX_STANDARDIZE / AUX_HUBER_BETA), stamped on config so
                # eval/inference reconstruct the same transform.
                cli_weight = {
                    "phylop": args.aux_phylop_weight,
                    "cadd_phred": args.aux_cadd_weight,
                    "af": args.trackc_af_weight,
                }
                for task in model_tasks:
                    mean, std = AUX_STANDARDIZE[task]
                    aux_weights[task] = cli_weight[task]
                    aux_standardize[task] = (mean, std)
                    aux_betas[task] = AUX_HUBER_BETA[task]
                    setattr(model.config, f"{task}_mean", mean)
                    setattr(model.config, f"{task}_std", std)
                model.config.aux_tasks = list(model_tasks)
                # Track-A bottleneck weights + pooling (CADD sparse -> always max).
                if aux_bottleneck:
                    bneck_cli_weight = {
                        "phylop": args.aux_bottleneck_phylop_weight,
                        "cadd_phred": args.aux_bottleneck_cadd_weight,
                    }
                    for task in aux_bottleneck:
                        aux_bottleneck_weights[task] = bneck_cli_weight[task]
                        aux_bottleneck_pool[task] = (
                            "max"
                            if task in AUX_SPARSE_TASKS
                            else args.aux_bottleneck_phylop_pool
                        )
                    model.config.aux_bottleneck_tasks = list(aux_bottleneck)
                    model.config.aux_bottleneck_layer = int(args.aux_bottleneck_layer)
                # Resume head weights from a clean-dir sidecar if present (Trainer
                # checkpoints restore heads via the full state_dict instead).
                if model_seed_checkpoint is not None:
                    model.load_aux_heads(model_seed_checkpoint)
                logger.info(
                    "Aux heads enabled: tasks=%s, lambdas=%s, std=%s, head_lr=%s,"
                    " backbone_lr=%.1e; bottleneck=%s w=%s pool=%s layer=%s"
                    " (capture=output_hidden_states).",
                    active_tasks,
                    aux_weights,
                    aux_standardize,
                    f"{args.aux_head_lr:.1e}"
                    if args.aux_head_lr is not None
                    else "backbone",
                    args.lr,
                    aux_bottleneck or "off",
                    aux_bottleneck_weights,
                    aux_bottleneck_pool,
                    args.aux_bottleneck_layer if aux_bottleneck else "-",
                )

            # Held-out per-subset + aux eval logging (eval_loss stays MLM-only).
            if active_tasks and eval_ds is not None and args.eval_subset_windows > 0:
                marker_ids = _marker_ids(tokenizer)
                primary_eval = (
                    eval_ds["primary"] if isinstance(eval_ds, dict) else eval_ds
                )
                callbacks.append(
                    EvalSubsetAuxCallback(
                        eval_dataset=primary_eval,
                        data_collator=eval_collator,
                        base_vocab_size=base_vocab_size,
                        marker_ids=marker_ids,
                        aux_tasks=active_tasks,
                        batch_size=args.batch_size,
                        max_windows=args.eval_subset_windows,
                        num_workers=min(args.dataloader_workers, 4),
                        seed=args.seed,
                    )
                )

        # Wrap-gating fix: a contrastive / supervised-phase objective (CPL / SMP /
        # Track-B / Track-C) needs the AuxHeadsModel wrapper even with NO phylop/cadd
        # regression columns (and even under --no-aux-heads). If the aux-heads block
        # above did not already wrap (no active regression tasks), wrap now with empty
        # regression heads so the contrastive/SMP terms are actually trained —
        # previously such a run silently trained MLM-only.
        if (
            args.cpl_lambda > 0
            or args.smp_lambda > 0
            or args.ctp_lambda > 0
            or args.trackb_lambda > 0
            or args.trackc_af_weight > 0
        ) and not isinstance(model, AuxHeadsModel):
            _enable_trackc = args.trackc_af_weight > 0
            _model_tasks = ("af",) if _enable_trackc else ()
            model = AuxHeadsModel(
                model,
                _model_tasks,
                bottleneck_tasks=(),
                bottleneck_layer=args.aux_bottleneck_layer,
                cpl_proj_dim=args.cpl_proj_dim if args.cpl_lambda > 0 else 0,
                trackb_proj_dim=args.trackb_proj_dim if args.trackb_lambda > 0 else 0,
                smp_proj=(args.smp_projector if args.smp_lambda > 0 else None),
                ctp_proj=(args.ctp_projector if args.ctp_lambda > 0 else None),
                cpl_margin=args.cpl_margin,
                cpl_attract_weight=args.cpl_attract_weight,
            )
            if _enable_trackc:
                _mean, _std = AUX_STANDARDIZE["af"]
                aux_weights["af"] = args.trackc_af_weight
                aux_standardize["af"] = (_mean, _std)
                aux_betas["af"] = AUX_HUBER_BETA["af"]
                model.config.af_mean = _mean
                model.config.af_std = _std
            model.config.aux_tasks = list(_model_tasks)
            logger.info(
                "Contrastive/SMP wrapper enabled with no regression columns "
                "(cpl_lambda=%.3g smp_lambda=%.3g trackb_lambda=%.3g trackc=%.3g).",
                args.cpl_lambda,
                args.smp_lambda,
                args.trackb_lambda,
                args.trackc_af_weight,
            )

        if _l2sp_anchor is not None:
            callbacks.append(L2SPCallback(_l2sp_anchor))

        trainer = _make_trainer(
            is_post,
            human_species_id,
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            eval_data_collator=eval_collator,
            og2_cfg=(
                {
                    "manifest": args.og2_manifest,
                    "fraction": args.og2_fraction,
                    "char_to_id": char_to_id,
                    # Under length bucketing a "human batch" has no single length,
                    # but it does have a fixed base count -- and og2_batch_sizes only
                    # ever uses the product. Passing (1, budget) token-matches replay
                    # to the real per-micro-batch base count instead of to a nominal
                    # batch_size x window_len that no longer describes any batch.
                    "human_batch": 1 if args.token_budget else args.batch_size,
                    "human_len": args.token_budget or args.window_len,
                    # _post only: `_pre` models take no species_ids at all, so
                    # None means "do not stamp" rather than "stamp id 0".
                    "species_id": args.og2_species_id if is_post else None,
                    "workers": args.og2_workers,
                    "seed": args.seed,
                }
                if args.og2_fraction > 0
                else None
            ),
            callbacks=callbacks,
            token_budget=args.token_budget,
            ref_sampler_weight=args.ref_sampler_weight,
            ref_loss_weight=args.ref_loss_weight,
            use_sample_weight=args.use_sample_weight,
            sample_weight_transform=args.sample_weight_transform,
            sample_weight_clip=args.sample_weight_clip,
            aux_tasks=tuple(aux_weights.keys()),
            aux_weights=aux_weights,
            aux_standardize=aux_standardize,
            aux_loss_type=args.aux_loss_type,
            aux_betas=aux_betas,
            aux_bottleneck=aux_bottleneck,
            aux_bottleneck_weights=aux_bottleneck_weights,
            aux_bottleneck_pool=aux_bottleneck_pool,
            aux_head_lr=args.aux_head_lr if args.aux_heads else None,
            lambda_cpl=args.cpl_lambda,
            cpl_every=args.cpl_every,
            lambda_trackb=args.trackb_lambda,
            trackb_every=args.trackb_every,
            lambda_smp=args.smp_lambda,
            smp_every=args.smp_every,
            lambda_ctp=args.ctp_lambda,
            ctp_every=args.ctp_every,
        )

        trainer.model = _maybe_compile_backbone_forward(trainer.model, args)
        model = trainer.model

        if args.load_best_model and eval_ds is None:
            logger.warning(
                "--load-best-model requested but evaluation is disabled; final model will be the last checkpoint."
            )

        resume_ckpt = None
        if main_resume_from:
            trainer_state = main_resume_from / "trainer_state.json"
            if trainer_state.exists():
                resume_ckpt = str(main_resume_from)
                logger.info("Resuming optimizer/scheduler from %s", resume_ckpt)
            else:
                has_model_weights = any(
                    (main_resume_from / fname).exists()
                    for fname in (
                        "model.safetensors",
                        "pytorch_model.bin",
                        "model.safetensors.index.json",
                        "pytorch_model.bin.index.json",
                    )
                )
                logger.warning(
                    "No trainer_state.json in %s; loading weights only%s.",
                    main_resume_from,
                    " (optimizer/scheduler state will be reset)"
                    if has_model_weights
                    else "",
                )

        if train_args.torch_compile:
            logger.info("torch.compile: first pass will JIT-compile (1-2 min).")

        # Provenance sidecar: HF saves TrainingArguments + log history per checkpoint,
        # but not the objective/data flags (cpl/smp lambdas, window, --max-genomes,
        # corpus path). Dump the parsed namespace so a checkpoint stays reproducible
        # and describable (model card, paper methods) without the shell history.
        if not args.no_save and trainer.is_world_process_zero():
            import json as _json
            import subprocess as _sp

            (output_dir / "run_args.json").write_text(
                _json.dumps(
                    {
                        "argv": sys.argv,
                        "args": {k: str(v) for k, v in sorted(vars(args).items())},
                        "git_commit": _sp.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=str(SCRIPT_DIR),
                            capture_output=True,
                            text=True,
                        ).stdout.strip(),
                        "train_windows": len(train_ds),
                        "world_size": _world_size,
                    },
                    indent=2,
                )
            )

        logger.info(
            "Starting training: %d windows, batch=%d, lr=%.1e, window=%d ...",
            len(train_ds),
            args.batch_size,
            args.lr,
            args.window_len,
        )
        trainer.train(resume_from_checkpoint=resume_ckpt)

        if not args.no_save and trainer.is_world_process_zero():
            final_dir = output_dir / "final"
            # End-of-training: safe to mutate the trainer model via
            # PeftModel.merge_and_unload (training is done).
            _atomic_save_dir(
                final_dir,
                lambda save_dir: _save_model_artifacts(
                    trainer.model, tokenizer, save_dir, allow_peft_mutation=True
                ),
            )
            logger.info("Saved final checkpoint -> %s", final_dir)

            baseline_dir = output_dir / "baseline_expanded_vocab"
            validate_diploid_embeddings(baseline_dir, final_dir, base_vocab_size)

            if push_to_hub_enabled:
                _push_folder_as_branch(
                    args.hub_model_id,
                    final_dir,
                    branch=f"{_hub_branch_prefix(output_dir)}-final",
                )

        logger.info("Done.")

    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if temp_output is not None:
            temp_output.cleanup()


if __name__ == "__main__":
    main()
