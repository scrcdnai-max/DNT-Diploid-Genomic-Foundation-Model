"""Final-pool bypass surgery for NTv3 — run the Transformer at 2x spatial resolution.

NTv3 is a conv/Transformer/deconv U-Net: seven encoder stages each halve the sequence, so
the Transformer sees one token per 128 bp. This module removes the LAST encoder pool (and,
necessarily, the matching FIRST decoder upsample) so the Transformer sees one token per
64 bp instead, while the decoder still returns a base-resolution output.

Nothing here builds, removes or renames a parameter. `AvgPool1d` has no weights, and the
decoder's `repeat+conv` upsample is `repeat_interleave` followed by a stride-1 SAME
convolution — dropping the repeat leaves the pretrained convolution untouched and simply
feeds it a sequence that is already at the target length. So the surgery is a runtime
patch on module *attributes*, applied after `from_pretrained`, and checkpoint loading is
never involved. Same class, same `state_dict`, strict loading, one flag to undo.

Why the skip connections still line up: `residuals[j]` is the input to encoder block j, at
length L / 2**j. With seven blocks and the last pool bypassed, the Transformer runs at
L / 2**6 -- exactly `residuals[6]`, which is what the first decoder block adds to. Every
later decoder block upsamples as before and meets its own residual. No crop, no
interpolation, no padding.

Positional encoding is an explicit choice, not an accident (`pos_scale`):

  1.0  keep ordinary position increments. The model is treated as genuinely higher
       resolution and adjacent tokens are one position apart, as in pretraining.
  0.5  halve the increments, so a given genomic distance keeps the rotary phase it had
       during pretraining. Exact, because rotary phase is `t * inv_freq`: scaling the
       positions is the same as scaling `inv_freq`, which is what this does.

Usage:
    from pretrain.ntv3_pool_surgery import apply_pool_bypass
    undo = apply_pool_bypass(model, n=1, pos_scale=0.5)   # n=0 is a no-op
    ...
    undo()                                                # exact original behaviour
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

_ENUM_REPEAT_CONV = "repeat+conv"
_PATCH_FLAG = "_diploid_glm_pool_bypassed"


def apply_pool_bypass(model: nn.Module, n: int = 1, pos_scale: float = 1.0):
    """Bypass the last `n` encoder pools and the matching first `n` decoder upsamples.

    `n=0` is a no-op and returns a no-op undo, so the flag is safe to thread through call
    sites unconditionally. Returns a callable that restores the original behaviour exactly.

    Note that `n` is a number of POOLING OPERATIONS, not of blocks. Every encoder and
    decoder block still exists and still holds its pretrained weights; only how many of
    them pool changes. Lowering `config.num_downsamples` to get the same resolution would
    instead build a smaller model and break strict checkpoint loading.
    """
    if n == 0 and pos_scale == 1.0:
        return lambda: None
    # Same re-entrancy guard as `patch_ntv3_attention_sdpa` (ntv3_attention_patch.py:157).
    # A second application would read the ALREADY-patched state -- `avg_pool` is Identity by
    # then -- so it would stack a second bypass and `undo()` would restore the Identity
    # rather than the pool.
    if getattr(model, _PATCH_FLAG, False):
        raise RuntimeError("pool bypass already applied to this model; undo() it first")

    core = next((m for m in model.modules() if hasattr(m, "conv_tower_blocks")), None)
    if core is None:
        raise TypeError(f"{type(model).__name__} has no NTv3 conv/deconv towers")
    conv_blocks, deconv_blocks = core.conv_tower_blocks, core.deconv_tower_blocks
    if not 0 <= n <= len(conv_blocks):
        raise ValueError(f"n={n} outside [0, {len(conv_blocks)}]")
    restore: list = []

    # Encoder: the pool is a parameter-free module, so swapping in Identity changes the
    # forward and leaves the state_dict alone.
    for block in list(conv_blocks)[len(conv_blocks) - n:]:
        restore.append((block, "avg_pool", block.avg_pool))
        block.avg_pool = nn.Identity()

    # Decoder: `upsample` is only read to decide whether to repeat_interleave before the
    # convolution (see UpsamplingDeconvBlock.forward and its Adaptive subclass). The
    # conv_transpose variant upsamples inside the convolution itself and cannot be
    # bypassed without changing which weights are used, so it is refused rather than
    # silently approximated. Every NTv3 checkpoint released so far is repeat+conv.
    for block in list(deconv_blocks)[:n]:
        up = block.conv.upsample
        if str(getattr(up, "value", up)) != _ENUM_REPEAT_CONV:
            raise NotImplementedError(
                f"pool bypass supports deconv_upsample_type='repeat+conv', got {up!r}"
            )
        restore.append((block.conv, "upsample", up))
        block.conv.upsample = None

    if pos_scale != 1.0:
        for rot in model.modules():
            if rot.__class__.__name__ == "RotaryEmbedding":
                restore.append((rot, "forward", rot.forward))
                rot.forward = _scaled_rotary_forward(rot, pos_scale)

    setattr(model, _PATCH_FLAG, True)

    def undo() -> None:
        for obj, name, value in restore:
            setattr(obj, name, value)
        setattr(model, _PATCH_FLAG, False)

    return undo


def _resolve_bypass(model: nn.Module) -> tuple[int, float]:
    """How many pools to bypass for THIS model, and at what position scale.

    Precedence: `NTV3_POOL_BYPASS` / `NTV3_POS_SCALE` when set (explicit override,
    either way) > the checkpoint's own `config.pool_bypass` /
    `config.pool_bypass_pos_scale` stamp (written by `pretrain.train`) > vanilla.
    The same rule `ntv3_attention_patch._pad_mask_active` and
    `layer_sweep.model.resolve_pool_bypass` follow.

    Reading only the environment is unsafe precisely because the surgery leaves
    `state_dict` untouched: a resume or a benchmark launched without the variable
    then runs at half the resolution it was trained at and reports a plausible
    wrong number instead of failing.
    """
    cfg = getattr(model, "config", None)
    env_n = os.environ.get("NTV3_POOL_BYPASS")
    env_scale = os.environ.get("NTV3_POS_SCALE")
    n = int(env_n) if env_n is not None else int(getattr(cfg, "pool_bypass", 0) or 0)
    scale = (
        float(env_scale)
        if env_scale is not None
        else float(getattr(cfg, "pool_bypass_pos_scale", 1.0) or 1.0)
    )
    return n, scale


def apply_pool_bypass_from_env(model: nn.Module):
    """Apply the surgery as configured by NTV3_POOL_BYPASS / NTV3_POS_SCALE or the stamp.

    The contract lives here, next to the surgery it configures, because it is read from
    two repos: `layer_sweep.model.load_backbone` and the GFMBench NTv3 adapter. Spelling the
    two variable names and their defaults out separately in each was how they drifted.
    """
    n, scale = _resolve_bypass(model)
    stamped = int(getattr(getattr(model, "config", None), "pool_bypass", 0) or 0)
    log = logging.getLogger(__name__)
    if stamped != n:
        log.warning(
            "NTV3_POOL_BYPASS=%d overrides this checkpoint's pool_bypass=%d stamp",
            n, stamped)
    if n or scale != 1.0:
        log.warning("NTv3 POOL BYPASS ACTIVE: n=%d pos_scale=%g", n, scale)
    return apply_pool_bypass(model, n=n, pos_scale=scale), n, scale


def _scaled_rotary_forward(rot: nn.Module, pos_scale: float):
    """Replacement `RotaryEmbedding.forward` whose positions advance by `pos_scale`.

    Rotary phase is `t * inv_freq`, so scaling the positions is the same as scaling the
    frequencies -- but `inv_freq` and the cos/sin caches are all persistent buffers, and
    writing to them would change what `state_dict()` reports and be silently reverted by
    the next strict checkpoint load. The scaled frequencies and their cache therefore live
    in this closure, entirely outside the module's buffer registry.
    """
    inv = rot.inv_freq * pos_scale
    # `_apply_rotary` is a @staticmethod, so binding it here rather than capturing `rot`
    # keeps the closure from holding a reference back to the module it is attached to --
    # that would be a cycle (rot.__dict__['forward'] -> cell -> rot) freed only by gc.
    apply = rot._apply_rotary
    cache: dict = {}

    def forward(q: torch.Tensor, k: torch.Tensor):
        # Keyed on length/device/dtype and never evicted: a process sees a handful of
        # distinct padded lengths at most, so this stays bounded without a clear().
        key = (q.shape[1], q.device, q.dtype)
        if key not in cache:
            t = torch.arange(q.shape[1], device=q.device, dtype=inv.dtype)
            freqs = torch.outer(t, inv.to(q.device))
            cache[key] = (torch.cos(freqs).to(q.dtype)[None, :, None, :],
                          torch.sin(freqs).to(q.dtype)[None, :, None, :])
        cos, sin = cache[key]
        return apply(q, cos, sin), apply(k, cos, sin)

    return forward


__all__ = ["apply_pool_bypass"]
