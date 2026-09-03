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
"""
Offline tests for the SNV window extraction used by the diploid tasks.

`extract_snv_sequences_centered` reports how many rows it skipped but not which ones, so the
diploid path pre-filters with `select_extractable_snv_rows` and relies on the result being
1:1 with the frame that goes in. These tests pin that contract: the filter drops exactly the
rows the extractor would skip, and every window comes back attached to its own row.

Run from a GFMBench-API checkout that has the diploid files installed:

    PYTHONPATH=. pytest /path/to/this/repo/tests
"""
import random

import pytest

pytest.importorskip("pyfaidx", reason="pyfaidx is required to read the fixture genome")
diploidutils = pytest.importorskip(
    "gfmbench_api.utils.diploidutils",
    reason="run from a GFMBench-API checkout with the diploid files installed",
)

import pandas as pd
from pyfaidx import Fasta

CHROM_LEN = 4000
SEQ_LEN = 100
FLANK = (SEQ_LEN - 1) // 2

_COMPLEMENTARY = {"A": "C", "C": "G", "G": "T", "T": "A"}


def _other_base(base: str) -> str:
    return _COMPLEMENTARY[base]


@pytest.fixture(scope="module")
def chromosomes():
    rng = random.Random(0)
    return {
        "chr1": "".join(rng.choice("ACGT") for _ in range(CHROM_LEN)),
        "chr2": "".join(rng.choice("ACGT") for _ in range(CHROM_LEN)),
    }


@pytest.fixture(scope="module")
def genome(chromosomes, tmp_path_factory):
    path = tmp_path_factory.mktemp("genome") / "mini.fa"
    with open(path, "w") as handle:
        for name, seq in chromosomes.items():
            handle.write(f">{name}\n")
            for start in range(0, len(seq), 60):
                handle.write(seq[start:start + 60] + "\n")
    return Fasta(str(path))


@pytest.fixture(scope="module")
def variants(chromosomes):
    """One keepable row per chromosome plus one row per drop rule."""
    rows = []

    def add(chrom, pos, zygosity, label, ref=None, alt=None, note=""):
        genome_ref = chromosomes[chrom][pos - 1] if chrom in chromosomes else "A"
        rows.append({
            "chrom": chrom.replace("chr", ""),
            "pos": pos,
            "ref": ref if ref is not None else genome_ref,
            "alt": alt if alt is not None else _other_base(genome_ref),
            "zygosity": zygosity,
            "label": label,
            "note": note,
        })

    # interleaved chromosomes, so grouping inside the extractor reorders the rows
    add("chr1", 1000, "het", 0, note="keep chr1:1000 het")
    add("chr2", 1500, "hom", 1, note="keep chr2:1500 hom")
    add("chr1", 1000, "hom", 1, note="keep chr1:1000 hom")
    add("chr2", 2500, "het", 0, note="keep chr2:2500 het")
    # one row per drop rule
    add("chr1", 5, "het", 1, note="drop: window crosses the chromosome start")
    add("chr1", CHROM_LEN - 3, "hom", 1, note="drop: window crosses the chromosome end")
    add("chr1", 2000, "het", 0, ref="AC", alt="A", note="drop: not an SNV")
    add("chr1", 2100, "hom", 1, ref=_other_base(chromosomes["chr1"][2099]),
        note="drop: reference allele mismatch")
    add("chr9", 1200, "het", 0, ref="A", alt="C", note="drop: chromosome not in reference")
    add("chr1", 3000, "het", 0, note="keep chr1:3000 het")

    return pd.DataFrame(rows)


def test_filter_keeps_only_extractable_rows(variants, genome):
    kept = diploidutils.select_extractable_snv_rows(
        variants, genome, SEQ_LEN, split_name="test"
    )
    expected = list(variants.loc[variants["note"].str.startswith("keep"), "note"])
    assert list(kept["note"]) == expected


def test_extractor_skips_nothing_after_filtering(variants, genome):
    _, _, labels, kept_df = diploidutils.extract_centered_snv_windows(
        variants, genome, SEQ_LEN, "test"
    )
    assert len(labels) == len(kept_df)
    assert len(kept_df) == int(variants["note"].str.startswith("keep").sum())


def test_windows_stay_attached_to_their_row(variants, genome):
    var_seqs, ref_seqs, labels, kept_df = diploidutils.extract_centered_snv_windows(
        variants, genome, SEQ_LEN, "test"
    )
    for i, row in kept_df.iterrows():
        assert ref_seqs[i][FLANK] == row["ref"].upper()
        assert var_seqs[i][FLANK] == row["alt"].upper()
        assert labels[i] == row["label"]
    # het and hom rows of one variant share ref/alt, so their order is what a base check
    # cannot see: it must follow the input order within a chromosome.
    chr1_1000 = kept_df[(kept_df["chrom"] == "1") & (kept_df["pos"] == 1000)]
    assert list(chr1_1000["zygosity"]) == ["het", "hom"]


def test_windows_are_unpadded_genome_slices(variants, genome, chromosomes):
    _, ref_seqs, _, kept_df = diploidutils.extract_centered_snv_windows(
        variants, genome, SEQ_LEN, "test"
    )
    for i, row in kept_df.iterrows():
        start = int(row["pos"]) - 1 - FLANK
        expected = chromosomes[f"chr{row['chrom']}"][start:start + SEQ_LEN]
        assert ref_seqs[i] == expected
        assert "P" not in ref_seqs[i]  # no edge padding reaches the diploid encoder
