"""SDPA attention patch for NTv3 (`MultiHeadAttention`).

NTv3's hand-rolled MHA forward casts Q/K/V to fp32 and runs matmul+softmax
manually, ignoring HF's `attn_implementation` kwarg. Under bf16 autocast this
wastes both memory bandwidth (cast round-trip) and FLOPs (fp32 matmul on
tensor cores that prefer bf16/TF32). This module replaces the forward with
`F.scaled_dot_product_attention`, which dispatches to FlashAttention-2 on
A100/H100 with bf16.

Used by both the embedding pipeline (inference) and the pretraining loop.
"""

from __future__ import annotations

import functools
import logging
import os
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_SDPA_PATCH_FLAG = "_diploid_glm_sdpa_patched"
_MASK_PATCH_FLAG = "_diploid_glm_mask_patched"          # on each MultiHeadAttention
_MASK_MODEL_FLAG = "_diploid_glm_mask_model_wrapped"    # on the model itself
_STASHED_MASK = "_diploid_glm_pad_mask"

# --- FlashAttention fast path (optional) ------------------------------------
# NTv3 attention is mask-free bidirectional (Core always passes attention_mask=None), and
# flash_attn_func wants (B, L, H, D) -- exactly the post-rotary layout, no permute needed.
# The kernels need a half dtype, so Q/K/V are cast to bf16 (the training autocast dtype).
# Chosen per GPU on first use: FA3 on Hopper (sm_90); FA4 from the HF kernels hub on
# Blackwell (sm_100/103), because every FA3 wheel is Hopper-only and aborts a Blackwell
# process uncatchably. Measured on B300 (2026-09-27): 1.02-1.04x end to end (attention is a
# few % of an NTv3 forward) and it moves person-minus-HG38 deltas by 5-12% relative (pooled
# cosine 0.992-0.999), so it is opt-in per call: `patch_ntv3_attention_sdpa(model, fast=True)`
# (train.py; layer_sweep --use-fa3). Every other caller keeps fp32 SDPA. In eager mode a kernel
# error falls back to SDPA for the rest of the run; under torch.compile the choice is traced
# once and an error propagates. Disable everywhere: NTV3_USE_FA3=0.
FA4_REPO, FA4_REVISION = "kernels-community/flash-attn4", "ed382ded331d0521a2aaba90959a0fcb1d803413"
_FAST_FLAG = "_diploid_glm_fast_attention"   # on each MultiHeadAttention whose caller opted in
_FA_WARNED = False


@functools.lru_cache(maxsize=1)
def _fast_kernel() -> Any:
    """-> flash_attn_func(q, k, v, causal) for this GPU, or None (use SDPA)."""
    if os.environ.get("NTV3_USE_FA3", "1").lower() in ("0", "false", "no") or not torch.cuda.is_available():
        return None
    major = torch.cuda.get_device_capability()[0]
    try:
        if major == 9:
            from flash_attn_interface import flash_attn_func

            return flash_attn_func
        if major >= 10:   # needs `kernels` + nvidia-cutlass-dsl 4.4 (4.8 breaks this revision)
            from kernels import get_kernel, get_local_kernel

            try:
                return get_kernel(FA4_REPO, revision=FA4_REVISION).flash_attn_func
            except Exception as hub:  # noqa: BLE001 - offline, kernels wants a complete hub snapshot; load the cached one
                from huggingface_hub.constants import HF_HUB_CACHE

                cached = Path(HF_HUB_CACHE) / f"kernels--{FA4_REPO.replace('/', '--')}" / "snapshots" / FA4_REVISION
                try:
                    return get_local_kernel(cached).flash_attn_func
                except Exception:  # noqa: BLE001 - report why the hub load failed, not the cache miss
                    raise hub from None
    except Exception as exc:  # noqa: BLE001 - the fast path is optional
        logging.getLogger(__name__).warning("no FlashAttention kernel (%s); using SDPA", exc)
    return None


def disable_cudnn_sdp() -> None:
    """torch 2.10's default SDPA dispatcher tries cuDNN attention first and
    segfaults on NTv3's small bottleneck shapes (L_bot=16 or 64). Call this
    once at process startup before any forward pass.
    """
    torch.backends.cuda.enable_cudnn_sdp(False)


def _sdpa_mha_forward(
    self: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    attention_weight_bias: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Drop-in replacement for NTv3 `MultiHeadAttention.forward` that uses
    `F.scaled_dot_product_attention` (FlashAttention-2 backend on A100/H100
    with bf16).

    The original NTv3 attention always casts Q/K/V to fp32 and does the
    matmul/softmax manually, so the HF `attn_implementation="sdpa"` argument
    is silently ignored by this model family. This forward replaces that with
    SDPA so the efficient kernel is actually used.

    Falls back to the original `MultiHeadAttention.forward` (stashed as
    `_diploid_glm_orig_forward` during patching) for the rare paths the
    embedding pipeline never hits: `add_bias_kv=True` or a non-None
    `attention_weight_bias`. NTv3's `Core.forward` always calls MHA with both
    of these unset, so the fallback is dead code in practice.
    """
    if getattr(self, "add_bias_kv", False) or attention_weight_bias is not None:
        return self._diploid_glm_orig_forward(
            query, key, value, attention_mask, attention_weight_bias
        )

    q = self.query_head(query)
    k = self.key_head(key)
    v = self.value_head(value)

    if self.rotary_embedding is not None:
        q, k = self.rotary_embedding(q, k)

    # FlashAttention fast path (FA3 on Hopper, FA4 on Blackwell; see `_fast_kernel`): mask-free
    # bidirectional attention in the native (B, L, H, D) layout, bf16 inside the kernel.
    fast = _fast_kernel() if getattr(self, _FAST_FLAG, False) and q.is_cuda and attention_mask is None else None
    if fast is not None:
        global _FA_WARNED
        try:
            o = fast(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                causal=False,
            )
            attn = o[0] if isinstance(o, tuple) else o  # (B, L, H, D)
            proj_in = attn.reshape((*attn.shape[:-2], -1)).to(self.mha_output.weight.dtype)
            return {"embeddings": self.mha_output(proj_in), "attention_weights": None}
        except Exception as exc:  # noqa: BLE001 - fall back to SDPA
            if not _FA_WARNED:
                logging.getLogger(__name__).warning(
                    "FlashAttention failed (%s); using SDPA for the rest of the run.",
                    exc,
                )
                _FA_WARNED = True
                _fast_kernel.cache_clear()
                os.environ["NTV3_USE_FA3"] = "0"

    # (B, L, H, D) -> (B, H, L, D)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    # NTv3 mask convention is bool with True=keep. SDPA accepts a bool mask
    # broadcastable to (B, H, Lq, Lk). NTv3 Core passes attention_mask=None
    # in every code path we use, so this branch is defensive only.
    attn_mask: torch.Tensor | None
    if attention_mask is None:
        attn_mask = None
    elif attention_mask.dim() == 2:
        attn_mask = attention_mask.unsqueeze(1).unsqueeze(1)
    elif attention_mask.dim() == 3:
        attn_mask = attention_mask.unsqueeze(1)
    else:
        attn_mask = attention_mask

    attn = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
    )
    attn = attn.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
    proj_in = attn.reshape((*attn.shape[:-2], -1))
    proj_in = proj_in.to(self.mha_output.weight.dtype)
    out = self.mha_output(proj_in)
    # SDPA does not materialise weights. The NTv3 core only reads this field
    # when `output_attentions=True`, which neither pipeline requests.
    return {"embeddings": out, "attention_weights": None}


def patch_ntv3_attention_sdpa(model: Any, fast: bool = False) -> int:
    """Walk `model` and replace `MultiHeadAttention.forward` with the SDPA
    variant on every instance found.

    Identifies the target class by class name (`MultiHeadAttention`), which is
    stable across all NTv3 variants (pre/post-trained, 8M/100M/etc.). Skips
    modules already patched. Returns the number of instances patched.
    `fast` opts every instance into the bf16 FlashAttention kernel (see `_fast_kernel`).
    """
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "MultiHeadAttention":
            continue
        if fast:
            setattr(module, _FAST_FLAG, True)
        if getattr(module, _SDPA_PATCH_FLAG, False):
            continue
        # Stash original forward on the instance for fallback paths.
        module._diploid_glm_orig_forward = module.forward
        # Bind the new forward to the instance (not the class, to avoid
        # touching cached/shared HF module classes).
        module.forward = types.MethodType(_sdpa_mha_forward, module)
        setattr(module, _SDPA_PATCH_FLAG, True)
        patched += 1
    return patched


# --- padding mask threading -------------------------------------------------
# NTv3's `Core` hardcodes `attention_mask=None` on every transformer-block call
# (`modeling_ntv3_pretrained.py:683` and `:785`), so `forward`'s `attention_mask`
# argument is accepted and then dropped: padding is globally self-attended as if
# it were real sequence. That is 0% of tokens at window 2048/4096, but 30.4% at
# 8192 (the 650M run), 41.2% at 10240 (the pre-fix 100M run) and ~22% on GUE
# benchmarks (seq_len 300 rounded up to the 128 multiple).
#
# The vendored model file cannot be edited: every checkpoint directory ships its
# own trust_remote_code copy. So the mask is threaded at runtime instead, the
# same way the SDPA forward is swapped in above.


def _bottleneck_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    """OR-pool a base-resolution mask down to the transformer bottleneck, as (B, Q, K).

    A bottleneck token survives if ANY base position it covers is real, matching
    how the avg_pool stem treats padding. `length` is always an exact multiple of
    `target_len` -- both are the sequence length divided by powers of two, and a
    length that did not divide would already have failed in the conv tower's skip
    connections -- so a non-divisible ratio raises here rather than being papered
    over into a mask that silences the wrong positions.
    """
    b, length = mask.shape
    pooled = mask.reshape(b, target_len, length // target_len).amax(-1).bool()
    # An all-pad row would softmax over an empty set and return NaN, poisoning
    # every downstream gradient. Such a row carries no information either way, so
    # let it attend to itself rather than emit NaN.
    pooled = pooled | ~pooled.any(dim=1, keepdim=True)
    # (B, K) keys -> (B, 1, K). Three dims are required: both consumers do
    # `attention_mask.unsqueeze(1)` against (B, H, Q, K) logits (NTv3's own fp32
    # path at modeling_ntv3_pretrained.py:292, and `_sdpa_mha_forward` above), and
    # a 2-D mask only broadcasts by accident at batch size 1 -- it raises for any
    # real batch. But the query dim stays 1 rather than being expanded to
    # `target_len`: every query attends to the same keys, so (B, 1, 1, K)
    # broadcasts against the logits for free, whereas materialising (B, Q, K)
    # costs O(B * T_bottleneck^2) and makes the SDPA kernel realise it contiguous.
    # Negligible at window 2048 (T_bot=16) but quadratic, so it is the difference
    # that matters if this model is ever run at long context.
    return pooled.unsqueeze(1)


class _PadMask:
    """The current forward's mask, shared by every patched MHA in one model.

    One object rather than an attribute per module, so `model.forward` writes
    once instead of once per layer. `for_length` memoises the pooled mask: NTv3
    runs every transformer block at a single bottleneck resolution, so without
    this the identical reshape+amax would run once per layer (6x at 100M).
    """

    __slots__ = ("_mask", "_pooled")

    def __init__(self) -> None:
        self._mask: torch.Tensor | None = None
        self._pooled: dict[int, torch.Tensor] = {}

    def set(self, mask: torch.Tensor) -> None:
        self._mask = mask
        self._pooled.clear()

    def clear(self) -> None:
        self._mask = None
        self._pooled.clear()

    def for_length(self, target_len: int) -> torch.Tensor | None:
        if self._mask is None:
            return None
        if target_len not in self._pooled:
            self._pooled[target_len] = _bottleneck_mask(self._mask, target_len)
        return self._pooled[target_len]


def pad_mask_enabled() -> bool:
    """What `NTV3_PAD_MASK` alone says, ignoring any checkpoint stamp.

    Only meaningful when the variable is set; `_pad_mask_active` is the decision
    every call site should ask for.
    """
    return os.environ.get("NTV3_PAD_MASK", "0").lower() in ("1", "true", "yes")


def _pad_mask_active(model: Any) -> bool:
    """Should the padding mask be threaded for THIS model?

    Precedence: NTV3_PAD_MASK env when set (explicit override, either way) >
    the checkpoint's own ``config.pad_mask`` stamp (written by train.py, so a
    model trained with the mask is scored with it) > off. No env mutation, so
    scoring several checkpoints with different stamps in one process stays
    correct.

    Note what this does NOT give you: turning the mask on changes the
    representation of every padded sequence, so an unstamped checkpoint and a
    stamped one are each scored the way they were trained but are NOT comparable
    to each other. A campaign that mixes them -- an unstamped anchor beside
    stamped milestones -- measures the recipe change, not the training delta.
    Set NTV3_PAD_MASK explicitly for both sides to isolate one or the other.
    """
    if "NTV3_PAD_MASK" in os.environ:
        return pad_mask_enabled()
    return bool(getattr(getattr(model, "config", None), "pad_mask", False))


def patch_ntv3_attention_mask(model: Any) -> int:
    """Make `forward(attention_mask=...)` actually reach the transformer tower.

    Wraps each `MultiHeadAttention.forward` so that, when `Core` calls it with
    `attention_mask=None`, it falls back to the mask published by the wrapped
    `model.forward`. Apply AFTER `patch_ntv3_attention_sdpa` so the lookup sits in
    front of the SDPA forward. Returns the number of instances patched; calling
    twice is a no-op.

    A batch with nothing masked out is passed straight through with no mask at
    all, which keeps the FlashAttention-3 fast path (gated on `attention_mask is
    None`) alive for the 0%-pad windows.

    Padding still reaches real positions through the conv tower, whose receptive
    field straddles the boundary (+/-254 base positions at 128x). That leak is
    bounded and, since real runs always fill with `pad_id`, constant -- unlike the
    attention leak fixed here, which is global and grows with the pad fraction.
    """
    mhas = [m for m in model.modules() if m.__class__.__name__ == "MultiHeadAttention"]
    holder: _PadMask = getattr(model, _STASHED_MASK, None) or _PadMask()
    patched = 0
    for mha in mhas:
        if getattr(mha, _MASK_PATCH_FLAG, False):
            continue

        def forward(
            self: Any,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            attention_weight_bias: torch.Tensor | None = None,
            _inner: Any = mha.forward,
            _holder: _PadMask = holder,
        ) -> dict[str, torch.Tensor]:
            if attention_mask is None:
                attention_mask = _holder.for_length(query.shape[1])
            return _inner(query, key, value, attention_mask, attention_weight_bias)

        mha.forward = types.MethodType(forward, mha)
        setattr(mha, _MASK_PATCH_FLAG, True)
        patched += 1

    if patched and not getattr(model, _MASK_MODEL_FLAG, False):
        orig_forward = model.forward
        setattr(model, _STASHED_MASK, holder)

        def model_forward(*args: Any, **kwargs: Any) -> Any:
            # Pop rather than bind a named parameter: a mask passed POSITIONALLY
            # lands in *args, and re-forwarding attention_mask= alongside it
            # raised "got multiple values for argument". A positional mask now
            # flows through to the original forward untouched (holder unset).
            # Known limitation: gradient-checkpointing recomputes run after the
            # finally-clear below, so they see an empty holder.
            attention_mask = kwargs.pop("attention_mask", None)
            # Nothing masked out: leave the mask off so FA3 stays reachable.
            if attention_mask is None or bool(attention_mask.all()):
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                return orig_forward(*args, **kwargs)
            holder.set(attention_mask)
            try:
                return orig_forward(*args, attention_mask=attention_mask, **kwargs)
            finally:
                holder.clear()

        model.forward = model_forward
        setattr(model, _MASK_MODEL_FLAG, True)

    return patched


def apply_pad_mask_from_env(model: Any) -> int:
    """Thread the padding mask iff `_pad_mask_active` says to, and log the decision.

    The contract lives here, next to the patch it configures, because four call
    sites read it -- `train.py` twice (resume and fresh load), the embedding
    pipeline, and `layer_sweep.load_backbone`. Spelling the check and its log line
    out separately in each is how `apply_pool_bypass_from_env` says its own
    siblings drifted.

    The resolved mode is logged either way, on and off. Two checkpoints scored in
    different modes are not comparable to each other (see `_pad_mask_active`), and
    logging only the mask-on side left that difference invisible in the very logs
    a mixed campaign would be audited from.

    Returns:
        Number of MHA modules patched; 0 when the mask stays off.
    """
    active = _pad_mask_active(model)
    from_env = "NTV3_PAD_MASK" in os.environ
    src = "NTV3_PAD_MASK env" if from_env else "config.pad_mask stamp"
    log = logging.getLogger(__name__)
    cfg = getattr(model, "config", None)
    if cfg is not None and from_env:
        # Stamp what THIS process actually does: config.json rides along in every
        # later save (baseline_expanded_vocab included), so the checkpoint records
        # its own training mode and scorers read it back via _pad_mask_active.
        if not active and getattr(cfg, "pad_mask", False):
            log.warning(
                "NTV3_PAD_MASK=0 overrides this checkpoint's pad_mask=True stamp "
                "-- running mask-free by explicit request"
            )
        cfg.pad_mask = active
    if not active:
        log.info("pad mask OFF (%s)", src)
        return 0
    n = patch_ntv3_attention_mask(model)
    log.info(
        "pad mask ON (%s): threaded the padding mask into %d MHA module(s)", src, n
    )
    return n
