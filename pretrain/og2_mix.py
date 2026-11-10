"""Interleave OpenGenome2 replay batches with the human diploid stream.

Continued diploid pretraining improves the variant/pathogenicity tasks and degrades
the broad genomic ones (gue_splice -0.071, traitgym_mendelian -0.108 at genome-5),
and five freeze arms showed the two move together: every configuration that learns
the diploid vocabulary pays the broad-task cost. Only 0.19% of the CPT corpus is
pure reference, so the broad regime is simply absent from training. Replaying ~20%
of NTv3's own OpenGenome2 distribution -- including its length diversity -- tests
whether that loss is preventable.

The scheme is whole-batch interleaving, 4 human : 1 OG2, NOT mixing sources inside
a tensor. Every OG2 batch is token-matched to the human batch, so "every 5th
optimizer step is OG2" is 20% by TOKENS, and the per-bin draw probabilities are
NTv3's own 7.9/13.0/13.2/66.0 token mix. Both are properties of tokens, not of
examples or batches -- confusing the two is the single most common way this
experiment gets silently mis-run, so `og2_batch_sizes` and `replay_schedule` are
separated out here and tested directly.
"""

from __future__ import annotations

import numpy as np

# NTv3's pretraining token mix across its four sequence lengths, in percent.
NTV3_BIN_MIX: dict[int, float] = {1024: 7.9, 2048: 13.0, 4096: 13.2, 8192: 66.0}


def og2_batch_sizes(human_batch: int, human_len: int) -> dict[int, int]:
    """Per-bin batch size carrying the same token count as one human batch.

    A token-matched 8192 batch is NOT a token-matched 2048 batch -- it holds a
    quarter as many sequences. Letting an 8192 loader inherit the 2048 sequence
    count silently processes 4x the tokens per step and breaks the 20%.
    """
    tokens = human_batch * human_len
    sizes = {}
    for b in NTV3_BIN_MIX:
        n, rem = divmod(tokens, b)
        if rem or n < 1:
            raise ValueError(
                f"human batch {human_batch}x{human_len} = {tokens} tokens is not "
                f"divisible into {b}-length windows; pick a human batch/length "
                "whose token count is a multiple of 8192"
            )
        sizes[b] = n
    return sizes


def replay_schedule(
    opt_step: int, seed: int = 42, fraction: float = 0.20
) -> int | None:
    """Which source this OPTIMIZER step draws from: a bin length, or None for human.

    A pure function of the step -- no RNG state, no rank, no worker. Every rank
    computes the same answer independently, which is what keeps them from
    diverging in step count and hanging at the next all-reduce; and because it is
    keyed off ``state.global_step`` rather than a private counter, a resume
    reproduces the schedule exactly instead of approximately.

    Under gradient accumulation the caller passes the OPTIMIZER step (microbatch
    index // accum), so a source is never mixed inside one accumulation window --
    that would change the effective loss weighting.
    """
    if fraction <= 0.0:
        return None
    period = 1.0 / fraction
    if abs(period - round(period)) > 1e-9:
        raise ValueError(
            f"--og2-fraction {fraction} must be 1/N for an integer N (0.20 -> every "
            "5th step, 0.10 -> every 10th); anything else cannot be an exact "
            "whole-batch period and would silently miss its token target"
        )
    period = int(round(period))
    if opt_step % period != period - 1:
        return None
    bins = list(NTV3_BIN_MIX)
    p = np.array([NTV3_BIN_MIX[b] for b in bins], dtype=np.float64)
    rng = np.random.default_rng((seed * 1_000_003 + opt_step) & 0xFFFFFFFF)
    return int(rng.choice(bins, p=p / p.sum()))


class MixedReplayLoader:
    """Yields human batches, substituting a whole OG2 batch on scheduled steps.

    ``__len__`` is the human loader's length so max-steps, the LR schedule and step
    accounting stay byte-identical to the control arm.

    ``step_fn`` should return the Trainer's ``state.global_step``. Using the real
    step rather than a private counter is what makes the schedule survive a resume
    (Trainer skips batches on resume, which would desynchronise a local counter)
    and what keeps every rank on the same source: global_step is identical across
    ranks by construction, so no communication is needed to agree.
    """

    def __init__(
        self,
        human_loader,
        og2_loaders: dict[int, object],
        step_fn,
        seed: int = 42,
        fraction: float = 0.20,
    ) -> None:
        self.human_loader = human_loader
        self.og2_loaders = og2_loaders
        self.step_fn = step_fn
        self.seed = seed
        self.fraction = fraction
        self.tokens: dict[str, int] = {"human": 0, **{f"og2_{b}": 0 for b in og2_loaders}}

    def __len__(self) -> int:
        return len(self.human_loader)

    @property
    def dataset(self):
        # Trainer reaches through for length/sampler introspection.
        return self.human_loader.dataset

    @property
    def sampler(self):
        # HF calls accelerate.skip_first_batches() on a mid-epoch resume
        # (ignore_data_skip=False), and that reads .sampler. Without this a resume
        # dies with AttributeError -- and the arm runner retries up to
        # MAX_RETRIES=3, turning one crash into a loop.
        return self.human_loader.sampler

    @property
    def batch_size(self):
        return self.human_loader.batch_size

    def __iter__(self):
        # At fraction 1.0 the human loader is never read, so do NOT start it --
        # iter() forks its workers (16/rank by default, persistent, prefetch 3)
        # and they sit filling a queue nobody drains.
        human = iter(self.human_loader) if self.fraction < 1.0 else None
        og2 = {b: iter(dl) for b, dl in self.og2_loaders.items()}
        for _ in range(len(self.human_loader)):
            b = replay_schedule(int(self.step_fn()), self.seed, self.fraction)
            if b is None:
                batch = next(human)
                self.tokens["human"] += int(batch["attention_mask"].sum())
            else:
                try:
                    batch = next(og2[b])
                except StopIteration:  # streams are infinite; a restart means a shard cycle
                    og2[b] = iter(self.og2_loaders[b])
                    batch = next(og2[b])
                self.tokens[f"og2_{b}"] += int(batch["attention_mask"].sum())
                batch["og2_bin"] = b
            yield batch

    def token_report(self) -> dict[str, float]:
        total = sum(self.tokens.values())
        if not total:
            return {}
        og2 = total - self.tokens["human"]
        out = {f"data/{k}_tokens": float(v) for k, v in self.tokens.items()}
        out["data/og2_fraction"] = og2 / total
        return out


def build_og2_loaders(
    manifest,
    char_to_id,
    collator,
    human_batch: int,
    human_len: int,
    rank: int,
    world_size: int,
    num_workers: int,
    seed: int = 42,
):
    """One DataLoader per bin, token-matched, sharing the HUMAN collator instance.

    Sharing the collator is what keeps MLM masking, label construction and -100
    handling identical between the two streams; a second collator with the same
    settings would still draw from a different RNG.

    Deliberately NOT ``accelerator.prepare``d: ``Og2WindowIterable`` already shards
    by rank, and ``Trainer._prepare_inputs`` does the device move.
    """
    import torch

    from og2_replay import Og2WindowIterable, bin_weights

    sizes = og2_batch_sizes(human_batch, human_len)
    loaders = {}
    for b, bs in sizes.items():
        paths, weights = bin_weights(manifest, b)
        ds = Og2WindowIterable(
            paths, b, char_to_id, weights=weights, seed=seed,
            rank=rank, world_size=world_size,
        )
        loaders[b] = torch.utils.data.DataLoader(
            ds, batch_size=bs, collate_fn=collator,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    return loaders
