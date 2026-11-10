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
# - https://huggingface.co/datasets/scrc-dnai/clinvar-diploid-indel — <DATASET-LICENSE>
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz — LicenseRef-UCSC-Genome-Browser
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from pyfaidx import Fasta

from gfmbench_api.tasks.base.base_gfm_zeroshot_general_indel_task import (
    BaseGFMZeroShotGeneralIndelTask,
)
from gfmbench_api.utils.fileutils import ensure_reference_genome
from gfmbench_api.utils.diploidutils import get_default_diploid_encoding_config, extract_diploid_indel_sequences

# ============================================================================
# Module-level constants
# ============================================================================

# Hub dataset carrying the benchmark
_HF_DATASET_ID = "scrc-dnai/clinvar-diploid-indel"
_TOKENS_MAPPING_FILENAME = "diploid_tokens.tsv"

# Columns the sequence extraction needs. 'zygosity' is optional and defaults to 'hom'.
_REQUIRED_COLUMNS = {"chrom", "pos", "ref", "alt", "label"}

# ============================================================================


class DiploidIndelClinvarTask(BaseGFMZeroShotGeneralIndelTask):
    """
    Diploid zero-shot ClinVar indel task backed by a HuggingFace dataset.

    Data source:
        `scrc-dnai/clinvar-diploid-indel` on the HuggingFace Hub. Being a zero-shot task,
        it uses the dataset's single published `test` split; there is nothing to fine-tune
        on and no train/validation split is published.

    Expected columns:
        chrom, pos, ref, alt, label

    Optional columns:
        zygosity (defaults to 'hom'), gene, indel_type, ref_len, alt_len, delta_len,
        dominant_flag, quality_stars

    Notes
    -----
    - Diploid encoding is applied after building two fixed-length haplotypes.
    - Optional special indel tokens:
          <INDEL_span_start> = "{"
          <INDEL_span_end>   = "}"
    """

    def __init__(
        self,
        root_data_dir_path: str,
        task_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        task_config = task_config or {}

        self.reference_genome_path = os.path.join(
            root_data_dir_path, "reference_genome", "hg38.fa"
        )

        # Hub dataset holding both the benchmark split and the token table
        self.hf_repo_id = _HF_DATASET_ID

        # Directory where an already-staged token table would live
        self.benchmark_dir = os.path.join(root_data_dir_path, "diploid_clean")
        self.diploid_token_tsv_path = self._resolve_diploid_token_tsv_path()

        # Load diploid sequence mappings and create encoding config
        self.diploid_encoding_config = get_default_diploid_encoding_config(self.diploid_token_tsv_path)
        self.encoding_map = self.diploid_encoding_config.encoding_map

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
            split: Split name as published ("test")

        Returns:
            DataFrame with columns: chrom, pos, ref, alt, zygosity, label, etc.

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

        if "zygosity" not in df.columns:
            logging.warning(
                f"{self.hf_repo_id} split '{split}' has no 'zygosity'; "
                "defaulting all rows to 'hom'"
            )
            df["zygosity"] = "hom"

        # Chromosomes are identifiers, not numbers: keep them as strings so '1' and 'X'
        # are handled the same way downstream.
        df["chrom"] = df["chrom"].astype(str)
        return df

    def _create_test_dataset(self):
        df = self._load_split("test")
        logging.info(f"Loaded benchmark with {len(df)} total samples")

        if self.max_num_samples is not None:
            df = df.head(min(self.max_num_samples, len(df)))

        if not os.path.exists(self.reference_genome_path):
            logging.info("Reference genome not found. Downloading hg38.fa...")
            ensure_reference_genome(self.reference_genome_path)

        logging.info(f"Loading reference genome: {self.reference_genome_path}")
        genome = Fasta(self.reference_genome_path)

        logging.info(
            f"Sequence length: {self.max_sequence_length}bp per haplotype "
            f"(diploid encoded, use_special_indel_tokens={self.diploid_encoding_config.use_special_indel_tokens})"
        )

        variant_sequences, reference_sequences, labels, variant_positions = (
            extract_diploid_indel_sequences(df, genome, self.diploid_encoding_config, self.max_sequence_length)
        )

        if len(labels) == 0:
            raise ValueError(
                "No valid samples in test split after sequence extraction. "
                f"Check the '{self.hf_repo_id}' test split and the reference genome."
            )

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        self.labels = labels_tensor
        self._variant_positions = variant_positions

        test_dataset = [
            (var_seq, ref_seq, label, np.array([], dtype=np.float32))
            for var_seq, ref_seq, label in zip(
                variant_sequences,
                reference_sequences,
                labels_tensor,
            )
        ]

        logging.info(
            f"Built {len(test_dataset)} samples "
            f"(avg encoded length: {np.mean([len(x[0]) for x in test_dataset]):.0f})"
        )

        return test_dataset

    def _get_default_max_seq_len(self) -> int:
        return 1048576

    def use_reference_cache(self) -> bool:
        """
        Cache reference inferences during eval.

        The benchmark carries one het and one hom row per variant, and both rows share the
        same ref/ref reference sequence, so every reference is inferred twice without a
        cache.
        """
        return True

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        return None

    def is_diploid_task(self) -> bool:
        """
        Diploid-encoded task marker.

        Not part of the GFMBench task API -- the diploid runner calls it to decide which
        run modes (diploid / haploid / biallelic) a task can be evaluated in.
        """
        return True

    def get_task_name(self) -> str:
        return "clinvar_diploid_indel"

    def _get_variant_position_in_sequence(self) -> int:
        """
        Return the nominal center position used by base classes that require a single
        variant position. Actual indel allele starts are sample-specific and stored in
        self._variant_positions.

        Examples:
            >>> # Typical even length
            >>> task = ClinvarIndelTask.__new__(ClinvarIndelTask)
            >>> task.max_sequence_length = 1000
            >>> task._get_variant_position_in_sequence()
            500

            >>> # Odd length (floor division)
            >>> task.max_sequence_length = 1001
            >>> task._get_variant_position_in_sequence()
            500
        """
        return self.max_sequence_length // 2
