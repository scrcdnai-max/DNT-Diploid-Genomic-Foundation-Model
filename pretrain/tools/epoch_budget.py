#!/usr/bin/env python
"""Turn a run's "Length bucketing ON" log line into steps-per-epoch and hours.

Under --token-budget the trainer logs GLOBAL micro-batches per epoch (the sampler is
built before accelerate shards it across ranks); optimizer steps per epoch are that
divided by grad-accum x world size. Every MAX_STEPS in this repo is a fraction of
that number, never an epoch -- the finished 1e-1 tsstes run was 13.1% of one.

    python -m pretrain.tools.epoch_budget \\
        --log output/dedup_ab/smoke_dedup/smoke.log --accum 2 --nproc 4 \\
        [--max-steps 84564] [--sec-per-step 1.17]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_RE = re.compile(r"(\d+) batches/epoch.*?REALIZED (\d+(?:\.\d+)?) bases/batch")


def parse_bucketing_line(line: str) -> tuple[int, float]:
    m = _RE.search(line)
    if not m:
        raise ValueError(f"no batches/epoch + REALIZED in: {line[:120]}")
    return int(m.group(1)), float(m.group(2))


def epoch_steps(batches_per_epoch: int, accum: int, nproc: int) -> int:
    # at least 1: a short smoke can have fewer micro-batches than accum*nproc, and this value is
    # a DIVISOR below ("% of an epoch"), so flooring to 0 turned a small log into a crash
    return max(1, batches_per_epoch // (accum * nproc))


_TQDM = re.compile(r"\| (\d+)/(\d+) \[(\d+):(\d+)(?::(\d+))?<")


def steady_state_rate(text: str, n_steps: int, frac: float = 0.6) -> float | None:
    """Seconds per optimizer step over the last (1-frac) of a run's tqdm bars.

    The tqdm mean is poisoned by the cold start (5-7 s/it for the first ~10 steps);
    over a 250-step smoke that over-estimated wall clock 6x. Uses the slowest rank's
    elapsed time at step ``frac*n`` and at ``n``; None if either is missing.
    """
    lo, hi = int(n_steps * frac), int(n_steps)
    seen: dict[int, float] = {}
    for m in _TQDM.finditer(text):
        step, total = int(m.group(1)), int(m.group(2))
        if total != hi:
            continue
        h, mnt, sec = m.group(3), m.group(4), m.group(5)
        elapsed = (int(h) * 3600 + int(mnt) * 60 + int(sec)) if sec else (int(h) * 60 + int(mnt))
        seen[step] = max(seen.get(step, 0.0), float(elapsed))
    if lo not in seen or hi not in seen:
        return None
    return (seen[hi] - seen[lo]) / (hi - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--accum", type=int, required=True)
    ap.add_argument("--nproc", type=int, required=True)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--sec-per-step", type=float, help="override; default: measured from the log's tqdm bars")
    ap.add_argument("--smoke-steps", type=int, help="the smoke's --max-steps, to read steady-state s/step from its bars")
    a = ap.parse_args()

    line = next(
        (l for l in a.log.read_text(errors="replace").splitlines() if "batches/epoch" in l),
        None,
    )
    if line is None:
        raise SystemExit(f"{a.log}: no 'batches/epoch' line (not a --token-budget run?)")
    batches, realized = parse_bucketing_line(line)
    steps = epoch_steps(batches, a.accum, a.nproc)
    if a.sec_per_step is None and a.smoke_steps:
        a.sec_per_step = steady_state_rate(a.log.read_text(errors="replace"), a.smoke_steps)
        if a.sec_per_step:
            print(f"steady-state {a.sec_per_step:.2f} s/step (last 40% of the {a.smoke_steps}-step smoke)")
    per_step = realized * a.accum * a.nproc
    print(f"{batches:,} micro-batches/epoch x {realized:,.0f} bases = {batches*realized/1e12:.3f} Tbase/epoch")
    print(f"{per_step/1e6:.2f} Mbase/optimizer step  ->  {steps:,} steps = 1 epoch")
    if a.max_steps:
        print(f"max_steps {a.max_steps:,} = {100*a.max_steps/steps:.1f}% of an epoch, {a.max_steps*per_step/1e12:.3f} Tbase")
    if a.sec_per_step:
        n = a.max_steps or steps
        print(f"{n:,} steps x {a.sec_per_step} s = {n*a.sec_per_step/3600:.1f} h")


if __name__ == "__main__":
    main()
