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

# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://huggingface.co/datasets/scrc-dnai/clinvar-diploid-snv — <DATASET-LICENSE>
# - https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-01.txt.gz — LicenseRef-NCBI-Data
# - https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarVCVRelease_2026-01.xml.gz — LicenseRef-NCBI-Data
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz — LicenseRef-UCSC-Genome-Browser
"""
ClinVar Zygosity-Aware Heritable Disease Supervised Task.

This task provides a supervised, linear-probe-friendly variant of the ClinVar
zygosity-aware benchmark.

Task Overview
-------------
- **Format**: Alternative / reference sequence pair → binary label (0: benign/carrier, 1: pathogenic)
- **Data source**: `scrc-dnai/clinvar-diploid-snv` on the HuggingFace Hub — the preprocessed
  ClinVar zygosity-aware benchmark with mode of inheritance
- **Split strategy**: the `train` / `test` splits published with the dataset (they carry the
  chromosome-based partition, chr19-22 being the test split), so every run splits the data
  identically without re-deriving it locally
- **Sequence construction**: <variant_seq>, <reference_seq>

The variant and reference sequences are extracted from hg38 with total
max_sequence_length.

Zygosity Logic
--------------
- Benign variants: het → 0, hom → 0
- Pathogenic dominant: het → 1, hom → 1
- Pathogenic recessive: het → 0 (carrier), hom → 1 (affected)

Supported Metrics
-----------------
- classification_accuracy: Standard accuracy score
- classification_mcc: Matthews Correlation Coefficient
- classification_auroc: Area Under ROC Curve
- classification_auprc: Area Under Precision-Recall Curve

Configuration (task_config)
---------------------------
- max_sequence_length: Total length (default: 1048576)
- batch_size: Batch size for dataloaders (default: 32)
- max_num_samples: Limit samples per split for testing (default: None)

Example
-------
>>> task_config = {
...     "max_sequence_length": 2048,
...     "batch_size": 64,
...     "max_num_samples": 100  # For sanity check mode
... }
>>> task = DiploidClinvarSupervisedTask(root_data_dir_path, task_config)
"""
import logging
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pyfaidx import Fasta

from gfmbench_api.tasks.base.base_gfm_supervised_variant_effect_task import BaseGFMSupervisedVariantEffectTask
from gfmbench_api.utils.fileutils import ensure_reference_genome
from gfmbench_api.utils.diploidutils import get_default_diploid_encoding_config, extract_variant_ref_diploid_sequences

# ============================================================================
# Module-level constants
# ============================================================================

# Hub dataset carrying the benchmark
_HF_DATASET_ID = "scrc-dnai/clinvar-diploid-snv"
_TOKENS_MAPPING_FILENAME = "diploid_tokens.tsv"

# Columns the sequence extraction needs from every split.
_REQUIRED_COLUMNS = {"chrom", "pos", "ref", "alt", "label", "zygosity"}

# ClinVar data source URLs (for reference; the Hub dataset is built from these releases)
_CLINVAR_SUMMARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/"
    "variant_summary_2026-01.txt.gz"
)
_CLINVAR_VCV_XML_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarVCVRelease_2026-01.xml.gz"
)

# ============================================================================


class DiploidClinvarSupervisedTask(BaseGFMSupervisedVariantEffectTask):
    """
    Supervised zygosity-aware ClinVar task using variant/reference sequence pairs.

    This is a variant effect prediction supervised classification task for fine-tuning
    genomic language models on zygosity-aware ClinVar pathogenicity prediction. It
    provides separate variant and reference sequences with binary classification labels
    based on mode of inheritance.

    Data comes from the `scrc-dnai/clinvar-diploid-snv` Hub dataset: its `train` and `test`
    splits are used as published, and there is no validation split.

    Inherits from BaseGFMSupervisedVariantEffectTask, which provides:
    - Standard multi-class classification metrics (accuracy, MCC, AUROC, AUPRC)
    - Fine-tuning dataset (training set)
    - Evaluation on validation/test sets
    - Variant effect prediction inference
    """

    def __init__(
        self,
        root_data_dir_path: str,
        task_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the supervised zygosity-aware ClinVar task.

        Args:
            root_data_dir_path: Path to root data directory
            task_config: Optional configuration dictionary with keys:
                - max_sequence_length: Maximum sequence length for variant/ref (default: 1048576)
                - batch_size: Batch size (default: 32)
                - max_num_samples: Limit samples per split (default: None)
        """
        task_config = task_config or {}

        # Hub dataset holding both the benchmark splits and the token table
        self.hf_repo_id = _HF_DATASET_ID

        # Directory where an already-staged token table would live
        self.benchmark_dir = os.path.join(root_data_dir_path, "clinvar_zygosity")
        self.diploid_token_tsv_path = self._resolve_diploid_token_tsv_path() #TODO

        # Load diploid sequence mappings and create encoding config
        self.diploid_encoding_config = get_default_diploid_encoding_config(self.diploid_token_tsv_path)
        self.encoding_map = self.diploid_encoding_config.encoding_map

        # Reference genome path
        self.reference_genome_path = os.path.join(
            root_data_dir_path, "reference_genome", "hg38.fa"
        )

        # Call parent initialization (computes max_sequence_length and calls _create_datasets)
        super().__init__(root_data_dir_path, task_config)

    def _resolve_diploid_token_tsv_path(self) -> str:
        """
        Locate the diploid token table, preferring a locally staged copy over the Hub.

        The table defines the vocabulary the sequences are encoded with and must match the
        one the model was trained on, so a node that already staged a table keeps using it
        (its results stay comparable with earlier runs); a fresh node falls back to the copy
        published alongside the data.

        Returns:
            Filesystem path to the token TSV (also read by the haploid/biallelic decoders)

        Raises:
            ImportError: If huggingface_hub is missing and there is no local copy
        """
        local_path = os.path.join(self.benchmark_dir, _TOKENS_MAPPING_FILENAME)
        if os.path.exists(local_path):
            logging.info(f"Using locally staged diploid token table: {local_path}")
            return local_path

        logging.info(
            f"No token table at {local_path}; fetching {_TOKENS_MAPPING_FILENAME} "
            f"from {self.hf_repo_id}"
        )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to fetch the diploid token table from "
                f"{self.hf_repo_id}. Install it, or stage the table at {local_path}."
            ) from exc

        return hf_hub_download(
            repo_id=self.hf_repo_id,
            filename=_TOKENS_MAPPING_FILENAME,
            repo_type="dataset",
        )

    def _load_split(self, split: str) -> pd.DataFrame:
        """
        Load one published split of the benchmark from the Hub.

        Args:
            split: Split name as published ("train" or "test")

        Returns:
            DataFrame with columns: chrom, pos, ref, alt, gene, zygosity, label, etc.

        Raises:
            ImportError: If the `datasets` package is not installed
            ValueError: If the split is missing columns the extraction needs
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                f"The `datasets` package is required to load {self.hf_repo_id}. "
                "Install it with `pip install datasets`."
            ) from exc

        logging.info(f"Loading split '{split}' from {self.hf_repo_id} ...")
        df = load_dataset(self.hf_repo_id, split=split).to_pandas()

        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"{self.hf_repo_id} split '{split}' is missing required columns: "
                f"{sorted(missing)}"
            )

        # Chromosomes are identifiers, not numbers: keep them as strings so '1' and 'X'
        # are handled the same way downstream.
        df["chrom"] = df["chrom"].astype(str)
        return df

    def _get_sequence_length(self) -> int:
        """
        Return the maximum sequence length for each variant/reference sequence.

        Returns:
            Integer representing length of each sequence (variant and reference)
        """
        return self.max_sequence_length

    def _create_datasets(self) -> Tuple[Any, Optional[Any], Any]:
        """
        Create train/test datasets for the supervised variant effect task.

        Workflow:
        1. Load the published train/test splits from the Hub
        2. Limit samples if max_num_samples is configured
        3. Ensure reference genome exists
        4. Extract variant and reference sequences for each split

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            Each dataset is a list of (variant_sequence, reference_sequence, label, conditional_input) tuples.
            val_dataset is always None: the published dataset has no validation split.

        Raises:
            ImportError: If pyfaidx or datasets is not installed
            FileNotFoundError: If the reference genome is not found
            ValueError: If no valid test samples after processing
        """
        # Load the published splits
        train_df = self._load_split("train")
        test_df = self._load_split("test")

        # Limit samples if configured
        if self.max_num_samples is not None:
            train_df = train_df.head(min(self.max_num_samples, len(train_df)))
            test_df = test_df.head(min(self.max_num_samples, len(test_df)))

        logging.info(f"Split sizes: train={len(train_df)}, test={len(test_df)}")
        logging.info(
            f"Sample distribution (label counts):\n Train:{train_df['label'].value_counts()}\n"
            f"Test:{test_df['label'].value_counts()}\n"
        )

        # Ensure reference genome exists
        if not os.path.exists(self.reference_genome_path):
            logging.info("Reference genome not found. Downloading hg38.fa...")
            ensure_reference_genome(self.reference_genome_path)

        logging.info(f"Loading reference genome: {self.reference_genome_path}")
        genome = Fasta(self.reference_genome_path)

        # Get sequence length for variant and reference sequences
        seq_len = self._get_sequence_length()
        logging.info(
            f"Sequence length: {seq_len}bp per sequence "
            f"(variant and reference sequences kept separate)"
        )

        # Extract variant and reference sequences for each split
        train_dataset = extract_variant_ref_diploid_sequences(
            train_df, genome, seq_len, "train", self.diploid_encoding_config,
        )
        test_dataset = extract_variant_ref_diploid_sequences(
            test_df, genome, seq_len, "test", self.diploid_encoding_config
        )

        # Validation: ensure test set is not empty
        if len(test_dataset) == 0:
            raise ValueError(
                "No valid samples in test split after sequence extraction. "
                f"Check the '{self.hf_repo_id}' test split and the reference genome."
            )

        # Log warnings for empty training
        if len(train_dataset) == 0:
            logging.warning(
                "WARNING: No valid samples in training split. "
                "This task will not be able to fine-tune models."
            )

        return train_dataset, None, test_dataset

    def _get_default_max_seq_len(self) -> int:
        """Return task's default maximum sequence length."""
        return 1048576

    def _get_num_labels(self) -> int:
        """
        Return 1: a single classification target (pathogenic vs. benign/carrier).

        `num_labels` counts independent targets, not classes -- the two outcomes are
        reported by `_get_num_classes()`.
        """
        return 1

    def _get_num_classes(self) -> int:
        """Return 2 (binary: 0 = benign/carrier, 1 = pathogenic)."""
        return 2

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None

    def get_task_attributes(self) -> Dict[str, Any]:
        """
        Return task attributes with emphasis on supervised variant effect prediction.

        This task is designed for supervised fine-tuning on binary classification
        using variant effect prediction (variant vs. reference sequences).

        Returns:
            Dictionary with task attributes including:
            - has_finetuning_data: True when a training split was built
            - has_validation_data: False (no validation split is published)
            - task_type: "classification"
            - classification_mode: "single_label" (one target, two classes)
            - num_labels: 1, num_classes: 2
            - is_variant_effect_prediction: True (variant-reference sequence pairs)
            - conditional_input_metadata: None (no metadata inputs)
        """
        attrs = super().get_task_attributes()
        # The base class hardcodes True; report what the split actually produced.
        attrs["has_finetuning_data"] = self.train_dataset is not None
        return attrs

    def is_diploid_task(self) -> bool:
        """
        Diploid-encoded task marker.

        Not part of the GFMBench task API -- the diploid runner calls it to decide which
        run modes (diploid / haploid / biallelic) a task can be evaluated in.
        """
        return True

    def get_task_name(self) -> str:
        """Return task name (identical to data directory name for conventions)."""
        return "diploid_clinvar_supervised_variant_effect"
