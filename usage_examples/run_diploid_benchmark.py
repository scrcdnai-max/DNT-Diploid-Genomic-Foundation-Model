# SPDX-FileCopyrightText: Copyright (c) 2026 Sheba Medical Center. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module does not embed third-party data download URLs.
import argparse
import copy
import logging
import csv
from collections import Counter
from typing import Any, Dict, List, Optional
import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from gfmbench_api.utils import logutils

from gfmbench_api.benchmark_report import BenchmarkReport
from gfmbench_api.tasks.concrete.clinvar_diploid_task import DiploidClinvarSupervisedTask
from gfmbench_api.tasks.concrete.clinvar_diploid_zeroshot_task import DiploidClinvarZeroShotTask
from gfmbench_api.tasks.concrete.clinvar_diploid_indel_task import DiploidIndelClinvarTask
from gfmbench_api.tasks.concrete.traitgym_complex_task import TraitGymComplexTask
from gfmbench_api.tasks.concrete.traitgym_mendelian_task import TraitGymMendelianTask
from gfmbench_api.tasks.concrete.brca1_task import BRCA1Task
from usage_examples.trainers import GFMFinetuner, SNVEmbeddingProbeFitter
from usage_examples.sanity_models.ntv3 import NucleotideTransformerV3Model

from gfmbench_api.utils.diploidutils import PrecomputedDiploidDataset


# ============================================================
# Diploid token mapping TSV
# ============================================================
# The decoder behind the haploid/biallelic wrappers MUST read the same table the task used
# to encode; otherwise het symbols are reinterpreted against a different vocabulary (the
DEFAULT_DIPLOID_TSV_RELPATH = os.path.join("diploid_clean", "diploid_tokens.tsv")

RUN_MODES = ("diploid", "haploid", "biallelic")


def resolve_diploid_tsv_path(
    task: Any,
    root_data_dir_path: str,
    override: Optional[str] = None,
) -> str:
    """Locate the diploid token table for `task`, preferring the one the task encodes with."""
    if override:
        path = override
    else:
        path = getattr(task, "diploid_token_tsv_path", None) or os.path.join(
            root_data_dir_path, DEFAULT_DIPLOID_TSV_RELPATH
        )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Diploid tokens TSV not found: {path}\n"
            f"The haploid/biallelic wrappers cannot decode without the same table the task "
            f"encoded with. Pass --diploid_tsv explicitly (canonical copy: "
            f"preprocess/diploid_tokens.tsv)."
        )
    return path


def is_diploid_task(task: Any) -> bool:
    """
    Whether `task` produces diploid-encoded sequences that can be decoded to other modes.

    The GFMBench task API carries no diploid flag (get_task_attributes() does not report
    one), so ask the task object directly: the diploid tasks in this repo expose
    is_diploid_task(). Anything else -- an upstream task, say -- is treated as haploid.
    """
    marker = getattr(task, "is_diploid_task", None)
    return bool(marker()) if callable(marker) else False


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark tasks on a model")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="diplod_results.csv",
        help="CSV path for the benchmark report (loads existing if present, saves here)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="DNABERT2_jepa_cls_context_encoder_continual_training_50k",
        help="Name for this model (or experiment) in the report"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint to load (set to None to use default HuggingFace weights). "
             "Ignored if --checkpoints_file is provided."
    )
    parser.add_argument(
        "--checkpoints_file",
        type=str,
        default=None,
        help="Text file with one full checkpoint directory path per line. "
             "Empty lines and lines starting with # are ignored."
    )
    parser.add_argument(
        "--mlm_head_path",
        type=str,
        default=None,
        help="Path to checkpoint containing MLM head weights. If provided along with "
             "--checkpoint_path, first loads full checkpoint, then overwrites MLM head "
             "with weights from this path. Only mlm_head_state_dict is loaded; other weights are ignored."
    )
    parser.add_argument(
        "--root_data_path",
        type=str,
        default="/efs/alarey/projects/sense/data/",
        help="Path to the root data directory"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use (default: 0)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use for each task (default: None, use all samples)"
    )

    # ------------------------------------------------------------
    # Imbalance handling
    # ------------------------------------------------------------
    parser.add_argument(
        "--balance_train_loader",
        action="store_true",
        help="Use a WeightedRandomSampler for training datasets to handle class imbalance."
    )
    parser.add_argument(
        "--sampler_power",
        type=float,
        default=1.0,
        help=(
            "Strength of rebalancing for WeightedRandomSampler. "
            "sample_weight = 1 / (class_count ** sampler_power). "
            "1.0 = full inverse-frequency balancing, 0.5 = softer balancing, 0.0 = no balancing."
        )
    )
    parser.add_argument(
        "--training_batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes for task evaluation and training (default: 0)"
    )
    parser.add_argument(
        "--cache_size",
        type=float,
        default=None,
        help="Inference cache RAM limit in GB, shared by the zero-shot reference cache "
             "and the fine-tune forward cache. Default: unlimited; pass 0 to disable."
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override existing benchmark results in the CSV and rerun tasks even if results exist",
    )
    parser.add_argument(
        "--use_linear_probe",
        action="store_true",
        help="Use linear probe fitting instead of full fine-tuning for tasks with fine-tuning data. Only applicable to variant effect prediction tasks. Overrides --only_proj_layer."
    )

    # ------------------------------------------------------------
    # Diploid encoding / mode handling
    # ------------------------------------------------------------
    parser.add_argument(
        "--diploid_tsv",
        type=str,
        default=None,
        help="Diploid token mapping TSV used to DECODE haploid/biallelic modes. Defaults to "
             "the table the task itself encodes with (task.diploid_token_tsv_path), else "
             "<root_data_path>/diploid_clean/diploid_tokens.tsv. Must match the table the "
             "model was trained on."
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=None,
        choices=list(RUN_MODES),
        help="Run modes for diploid-capable tasks (default: all of diploid haploid biallelic). "
             "Non-diploid tasks always run as 'diploid'."
    )
    parser.add_argument(
        "--gap_policy",
        type=str,
        default="drop",
        choices=["drop", "keep"],
        help="How haploid/biallelic decoding treats indel alignment gaps. 'drop' (default) "
             "emits real ungapped haplotypes, so the two haplotypes of an indel differ in "
             "length. 'keep' preserves column alignment and equal lengths for "
             "length-sensitive scores, at the cost of emitting gapped (non-haplotype) rows."
    )

    return parser.parse_args()


def read_checkpoints_list(path: str):
    """
    Read checkpoints file (one checkpoint dir per line).
    Ignores empty lines and comments starting with '#'.
    """
    ckpts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or not os.path.isdir(s):
                continue
            ckpts.append(s)
    return ckpts


def load_mlm_head_only(model, mlm_head_path: str):
    """
    Load only the MLM head weights from a checkpoint.

    Handles checkpoints that contain:
    - mlm_head_state_dict: Load MLM head weights directly
    - model_state_dict + mlm_head_state_dict: Load only mlm_head_state_dict, ignore model weights
    - mlm_model_state_dict (legacy): Extract and load only cls.* weights

    Args:
        model: DNABERT2Model instance
        mlm_head_path: Path to checkpoint file containing MLM head weights
    """
    logging.info(f"Loading MLM head only from: {mlm_head_path}")
    state = torch.load(mlm_head_path, map_location=model.device)

    if "mlm_head_state_dict" in state:
        model.mlm_head.load_state_dict(state["mlm_head_state_dict"])
        model.mlm_head_loaded = True
        logging.info("Loaded MLM head weights from mlm_head_state_dict")

    elif "mlm_model_state_dict" in state:
        mlm_state = state["mlm_model_state_dict"]
        mlm_head_state = {}
        for key, value in mlm_state.items():
            if key.startswith("cls."):
                mlm_head_state[key[4:]] = value  # Remove 'cls.' prefix

        if mlm_head_state:
            model.mlm_head.load_state_dict(mlm_head_state, strict=False)
            model.mlm_head_loaded = True
            logging.info("Loaded MLM head weights from legacy mlm_model_state_dict")
        else:
            logging.warning("No MLM head weights found in legacy checkpoint")
    else:
        logging.error(f"No MLM head weights found in {mlm_head_path}")
        logging.error(f"  Available keys: {list(state.keys())}")
        raise ValueError("No MLM head weights found in MLM head checkpoint")


def _wrap_dataset(dataset, mode: str, diploid_tsv_path: str, gap_policy: str = "drop"):
    """
    mode:
      - "diploid": leave dataset as-is, with diploid tokenization
      - "haploid": PrecomputedDiploidDataset(..., mode="haploid")
            one sample per row: the variant-bearing haplotype vs the reference haplotype
      - "biallelic": PrecomputedDiploidDataset(..., mode="biallelic")  # allele1/allele2
            NOTE: homozygous rows give two identical alleles, so scores derived from this
            pair measure zygosity, not variant effect.
    """
    if mode == "diploid":
        return dataset
    if mode in ("haploid", "biallelic"):
        return PrecomputedDiploidDataset(
            base_dataset=dataset,
            diploid_tsv_path=diploid_tsv_path,
            mode=mode,
            gap_policy=gap_policy,
        )
    raise ValueError(f"Unknown mode: {mode}")


def diploid_mode_view(task, mode: str, diploid_tsv_path: str, gap_policy: str = "drop"):
    """
    Return a view of `task` whose datasets are decoded to `mode`.

    A run mode changes the *samples*, not the metrics: every mode must be scored by the
    task's own evaluation code. Rather than reaching into the task's protected
    `_eval_dataset`, the view is a shallow copy of the task with its three dataset
    attributes replaced by the decoded ones, so the ordinary public entry points
    (`get_finetune_dataset()`, `eval_test_set()`, `eval_validation_set()`) already operate
    on the right samples: `eval_test_set()` is defined as
    `self._eval_dataset(model, self.test_dataset)`, so it reaches the task's evaluator with
    exactly the (model, wrapped dataset) pair the direct call used to pass.

    The copy is shallow on purpose: the expensive state (extracted sequences, the open
    reference genome, the encoding map) is shared, so a mode costs one decode pass and no
    re-extraction. Nothing in the task is mutated, so modes and checkpoints stay
    order-independent. Decoding is deterministic (the per-row seed is hashed from the
    variant sequence) and touches no global RNG, so building the views up front leaves the
    fine-tuning RNG stream untouched.

    "diploid" is the task's native encoding and returns the task itself, unwrapped.
    """
    if mode == "diploid":
        return task

    view = copy.copy(task)
    for attr in ("train_dataset", "validation_dataset", "test_dataset"):
        dataset = getattr(task, attr, None)
        if dataset is None:
            continue
        setattr(view, attr, _wrap_dataset(dataset, mode, diploid_tsv_path, gap_policy))
    return view


def _log_eval_composition(dataset, task_name: str, mode: str) -> None:
    """
    Log rows and positive rate of the evaluated dataset.

    Modes used to disagree on row count and class balance (heterozygous rows were expanded
    into two samples in haploid mode), which silently made columns of the report
    incomparable across modes. Log it so that is checkable rather than assumed.
    """
    try:
        labels = _get_labels_from_dataset(dataset)
    except Exception as e:  # diagnostics must never break a run
        logging.warning("Could not summarise eval composition for %s [%s]: %s", task_name, mode, e)
        return

    n = len(labels)
    if n == 0:
        logging.warning("Eval dataset for %s [%s] is EMPTY", task_name, mode)
        return

    pos = sum(1 for y in labels if y == 1)
    logging.info(
        "Eval composition %s [%s]: n=%d, positives=%d (%.4f)",
        task_name, mode, n, pos, pos / n,
    )


def _to_python_scalar(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()
    return x

def _get_labels_from_dataset(dataset) -> List[int]:
    """
    Extract all labels from a dataset.

    Fast paths:
    - dataset.labels
    - dataset.targets
    - dataset.y

    Fallback:
    - iterate through dataset[i] and infer the label
    """
    if isinstance(dataset, Subset):
        parent_labels = _get_labels_from_dataset(dataset.dataset)
        return [int(parent_labels[i]) for i in dataset.indices]

    for attr_name in ["labels", "targets", "y"]:
        if hasattr(dataset, attr_name):
            values = getattr(dataset, attr_name)
            values = _to_python_scalar(values)
            if isinstance(values, list):
                return [int(v) for v in values]

    labels: List[int] = []
    for i in range(len(dataset)):
        item = dataset[i]
        label = item[2] # item is var_seq, ref_seq, label, conditional_input
        if label is None:
            raise ValueError(
                f"Could not infer label for sample {i}. "
                "Please expose labels via dataset.labels / dataset.targets / dataset.y, "
                "or return them as the last element of each sample."
            )
        # Coerce out of torch.Tensor: Counter() in _build_weighted_sampler hashes tensors by
        # identity, so leaving them as tensors silently gives every sample its own class.
        labels.append(int(_to_python_scalar(label)))

    return labels


def _build_weighted_sampler(dataset, sampler_power: float = 1.0) -> WeightedRandomSampler:
    """
    Builds a WeightedRandomSampler that oversamples minority classes.

    sample_weight[label_i] = 1 / (count(label_i) ** sampler_power)

    - sampler_power = 1.0: full inverse-frequency balancing
    - sampler_power = 0.5: softer balancing
    - sampler_power = 0.0: uniform sampling
    """
    labels = _get_labels_from_dataset(dataset)
    class_counts = Counter(labels)

    sample_weights = [
        1.0 / (class_counts[label] ** sampler_power)
        for label in labels
    ]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def _make_train_loader(dataset, batch_size: int, use_balanced_sampler: bool, sampler_power: float,
                       num_workers: int = 0):
    """
    Create a training DataLoader.

    If use_balanced_sampler=True, uses WeightedRandomSampler and disables shuffle.
    Otherwise uses standard shuffle=True.
    """
    if use_balanced_sampler:
        logging.info(
            f"Using WeightedRandomSampler for class imbalance handling "
            f"(sampler_power={sampler_power})"
        )
        sampler = _build_weighted_sampler(dataset, sampler_power=sampler_power)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

def checkpoint_display_name(ckpt_path: str) -> str:
    """
    Produce a stable short identifier for the report.
    Uses last path component; falls back to full path if needed.
    """
    p = ckpt_path.rstrip("/").rstrip("\\")
    name = os.path.basename(p)
    return name if name else p


def checkpoint_model_name(ckpt_path: str) -> str:
    """
    Derive a descriptive model name from a checkpoint path, using the parent
    directory (usually the run/experiment name) plus the checkpoint directory.

        /data/exp/my_run/checkpoint-1000  ->  my_run_checkpoint-1000

    This keeps names distinct across experiments that share checkpoint step
    names (e.g. two runs both with a checkpoint-1000). Falls back to the
    basename, then the full path, when there is no usable parent component.
    """
    p = ckpt_path.rstrip("/").rstrip("\\").replace("\\", "/")
    base = os.path.basename(p)
    parent = os.path.basename(os.path.dirname(p))
    if parent and base:
        return f"{parent}_{base}"
    return base or p


def main():
    logutils.init_logger()
    args = parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    sanity_check_mode = False  # Set to True for quick testing with 100 samples

    root_data_dir_path = args.root_data_path
    csv_path = args.csv_path

    report = BenchmarkReport(csv_path=csv_path)

    base_model_name = args.model_name
    max_length = 1024 #512

    checkpoint_path = args.checkpoint_path
    mlm_head_path = args.mlm_head_path

    task_config = {
        "max_sequence_length": max_length,
        "batch_size": 128,
        "num_workers": args.num_workers,
        "max_num_samples": None,
        "cache_size": args.cache_size,
    }

    brca1_task_config = {
        "max_sequence_length": 8192,  # 8kb sequences centered on BRCA1 variants
        "batch_size": 128,
        "num_workers": args.num_workers,
        "max_num_samples": None,
        "cache_size": args.cache_size,
    }

    if args.num_samples is not None:
        task_config["max_num_samples"] = args.num_samples
        logging.info(f"SANITY CHECK MODE: Limiting to {args.num_samples} samples per dataset")

    tasks = [
        DiploidClinvarSupervisedTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        #DiploidClinvarZeroShotTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        DiploidIndelClinvarTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        TraitGymComplexTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        TraitGymMendelianTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        BRCA1Task(root_data_dir_path=root_data_dir_path, task_config=brca1_task_config),
    ]

    training_params = {
        "num_epochs": 3,
        "lr": 3e-5,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "only_proj_layer": False,
        "batch_size": args.training_batch_size,
    }

    if args.checkpoints_file:
        checkpoints = read_checkpoints_list(args.checkpoints_file)
        if not checkpoints:
            raise ValueError(f"No checkpoints found in --checkpoints_file: {args.checkpoints_file}")
        logging.info(f"Loaded {len(checkpoints)} checkpoints from {args.checkpoints_file}")
    else:
        # single checkpoint (or None -> default weights)
        checkpoints = [args.checkpoint_path]  # can be [None]
        logging.info("Running single-checkpoint mode")

    for ckpt_path in checkpoints:
        # Derive the report model name from the checkpoint path (parent run dir
        # + checkpoint dir). --model_name is used only for the baseline /
        # default-weights case where there is no path to derive from.
        if ckpt_path:
            model_base = checkpoint_model_name(ckpt_path)
            ckpt_tag = checkpoint_display_name(ckpt_path)
        else:
            model_base = f"{base_model_name}_baseline"
            ckpt_tag = "baseline"
        logging.info(f"{'='*70}")
        logging.info(f"Benchmarking checkpoint: {ckpt_path if ckpt_path else '[DEFAULT HF WEIGHTS]'}")
        logging.info(f"Checkpoint tag: {ckpt_tag}")
        logging.info(f"Model name base: {model_base}")
        logging.info(f"{'='*70}")

        for task in tasks:
            task_name = task.get_task_name()
            logging.info(f"{'=' * 50}")
            logging.info(f"Running task: {task_name}")
            logging.info(f"{'=' * 50}")

            task_attrs = task.get_task_attributes()
            has_finetuning_data = task_attrs.get("has_finetuning_data", False)
            is_variant_effect = task_attrs.get("is_variant_effect_prediction", False)
            task_is_diploid = is_diploid_task(task)

            # Decode with the table the task encodes with, not a hardcoded relative path.
            diploid_tsv_path = resolve_diploid_tsv_path(
                task, root_data_dir_path, args.diploid_tsv
            )

            requested_modes = list(args.modes) if args.modes else list(RUN_MODES)
            # Non-diploid tasks have no diploid encoding to decode, so they only run as-is.
            run_modes = requested_modes if task_is_diploid else ["diploid"]
            logging.info(
                "Task %s: diploid=%s, modes=%s, tokens TSV=%s",
                task_name, task_is_diploid, run_modes, diploid_tsv_path,
            )
            for run_mode in run_modes:
                logging.info(f"{'-' * 50}")
                logging.info(f"Mode: {run_mode}")
                logging.info(f"{'-' * 50}")

                model_to_use = NucleotideTransformerV3Model
                if ckpt_path:
                    model = model_to_use(device=device, model_name=ckpt_path)
                else:
                    model = model_to_use(device=device)

                if mlm_head_path:
                    logging.info(f"Loading MLM head from: {mlm_head_path}")
                    load_mlm_head_only(model, mlm_head_path)

                # Precision policy: fp16 backbone forwards only for true fine-tune
                # tasks (has finetuning data AND not a linear-probe variant-effect
                # task). Zero-shot eval and linear probe both run in FP32. ntv3's
                # _amp_context() reads these attributes live on every forward, so
                # toggling them per task is all that's needed. The representative is
                # upcast to fp32 on return (see ntv3.py), so the fp16 backbone stays
                # compatible with the fp32 projection and the numpy-backed
                # SequenceInferenceCache.
                if has_finetuning_data and not args.use_linear_probe:
                    model.use_amp = True
                    model.amp_dtype = torch.float16
                else:
                    model.use_amp = False

                model_name = f"{model_base}_{run_mode}"

                # Strict CSV check: only skip when the CSV header contains the exact
                # model column name and there is a row with matching `task` and a
                # non-empty cell for that model. This avoids false positives from
                # loose value scanning.
                if not args.override and os.path.exists(csv_path):
                    exists = False
                    try:
                        with open(csv_path, "r", encoding="utf-8") as f:
                            reader = csv.reader(f)
                            header = next(reader, None)
                            if header is None:
                                exists = False
                            else:
                                # Identify model column by exact header match
                                try:
                                    model_col = header.index(model_name)
                                except ValueError:
                                    model_col = None

                                # Identify task column (prefer explicit 'task' header)
                                if "task" in header:
                                    task_col = header.index("task")
                                else:
                                    task_col = 0

                                if model_col is not None:
                                    for row in reader:
                                        # be defensive about short rows
                                        if len(row) <= max(task_col, model_col):
                                            continue
                                        if row[task_col] == task_name and str(row[model_col]).strip() != "":
                                            exists = True
                                            break
                    except Exception as e:
                        logging.warning(f"Could not read CSV {csv_path}: {e}")
                        exists = False

                    if exists:
                        logging.info(
                            "Result exists for task %s and model %s — skipping (use --override to force)",
                            task_name,
                            model_name,
                        )
                        continue

                # One view of the task per run mode: same task, same metrics, samples
                # decoded to this mode. Built after the skip check so a cached result
                # costs no decoding.
                task_view = diploid_mode_view(
                    task, run_mode, diploid_tsv_path, args.gap_policy
                )

                if has_finetuning_data:
                    train_dataset = task_view.get_finetune_dataset()
                    _log_eval_composition(train_dataset, f"{task_name}/train", run_mode)

                    train_loader = _make_train_loader(
                        dataset=train_dataset,
                        batch_size=training_params["batch_size"],
                        use_balanced_sampler=args.balance_train_loader,
                        sampler_power=args.sampler_power,
                        num_workers=args.num_workers,
                    )

                    if args.use_linear_probe:
                        linear_probe = SNVEmbeddingProbeFitter(
                            model=model,
                            train_loader=train_loader,
                            device=device,
                        )
                        test_model = linear_probe.fit()
                    else:
                        hidden_dim = model.get_hidden_dim()
                        # num_labels counts independent targets and num_classes the
                        # classes per target; the head width follows from that pair the
                        # same way the upstream runner derives it.
                        classification_mode = task_attrs["classification_mode"]
                        num_labels = task_attrs["num_labels"]
                        num_classes = task_attrs["num_classes"]
                        num_outputs = (
                            num_classes
                            if num_labels == 1
                            else num_labels if num_classes == 2 else num_labels * num_classes
                        )

                        finetuner = GFMFinetuner(
                            model=model,
                            train_loader=train_loader,
                            hidden_dim=hidden_dim,
                            num_outputs=num_outputs,
                            num_epochs=training_params["num_epochs"],
                            lr=training_params["lr"],
                            optimizer_name=training_params["optimizer"],
                            weight_decay=training_params["weight_decay"],
                            only_proj_layer=training_params["only_proj_layer"],
                            classification_mode=classification_mode,
                            num_labels=num_labels,
                            num_classes=num_classes,
                            is_variant_effect_prediction=is_variant_effect,
                            cache_size=args.cache_size,
                            device=device,
                        )
                        logging.info("Fine-tuning...")
                        test_model = finetuner.fine_tune()
                else:
                    logging.info("Zero-shot task detected. Skipping fine-tuning.")
                    test_model = model

                logging.info("Evaluating...")
                test_model.eval()
                # The view's test_dataset is already decoded to run_mode, so the task's
                # own public evaluation path scores the right samples.
                _log_eval_composition(task_view.test_dataset, task_name, run_mode)
                scores = task_view.eval_test_set(test_model)
                logging.info(f"Test scores for {task_name} [{run_mode}]: {scores}")

                # The projection wrapper keeps a reference-inference cache; free it before
                # the next mode/checkpoint builds its own.
                if hasattr(test_model, "clear_ref_cache"):
                    test_model.clear_ref_cache()

                report.add_scores(task_name, model_name, scores)
                report.save_csv()
                logging.info(f"Saved progress to: {csv_path}")

    logging.info(f"{'=' * 50}")
    logging.info(f"Benchmark complete! Report saved to: {csv_path}")
    logging.info(f"{'=' * 50}")
    logging.info(report)


if __name__ == "__main__":
    main()
    logging.info("Benchmark completed!")