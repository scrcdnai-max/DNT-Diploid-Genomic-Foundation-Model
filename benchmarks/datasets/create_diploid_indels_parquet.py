import os
import subprocess
import gzip
from typing import Optional

import numpy as np
import pandas as pd

# =============================================================================
# ClinVar Annotation Mappings
# =============================================================================

REVIEW_STATUS_TO_GOLD_STARS = {
    "criteria provided, single submitter": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": np.nan,
    "reviewed by expert panel": 3,
    "no assertion provided": np.nan,
    "no interpretation for the single variant": np.nan,
    "practice guideline": 4,
}

CLINICAL_SIGNIFICANCE_TO_LABEL = {
    "Benign": 0,
    "Likely benign": 0,
    "Benign/Likely benign": 0,
    "Pathogenic": 1,
    "Likely pathogenic": 1,
    "Pathogenic/Likely pathogenic": 1,
}

root_data_dir_path = "../data"
dataset_dir = os.path.join(root_data_dir_path, "diploid_clean_indel")
dataset_path = os.path.join(dataset_dir, "clinvar_diploid_indel.parquet")

clinvar_summary_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-01.txt.gz"
clinvar_vcv_xml_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarVCVRelease_2026-01.xml.gz"

# ClinVar can use several names depending on release / row
INDEL_TYPES = {
    "Deletion",
    "Insertion",
    "Indel",
}

STANDARD_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
VALID_BASES = set("ACGT")


def _download_file(url: str, dest: str) -> None:
    print(f"Downloading {url} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    result = subprocess.run(
        ["wget", "-c", "-O", dest, url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")
    print(f"Downloaded to: {dest}")


def _normalise_moi(raw: str) -> str:
    if not raw:
        return "unknown"

    s = raw.strip().lower()

    if "autosomal dominant" in s:
        return "AD"
    if "autosomal recessive" in s or "biallelic" in s:
        return "AR"
    if "x-linked dominant" in s:
        return "XL_dom"
    if "x-linked recessive" in s or "x-linked inheritance" in s:
        return "XL_rec"
    if "mitochondrial" in s:
        return "MT"
    if "somatic" in s:
        return "somatic"
    if "multifactorial" in s or "polygenic" in s:
        return "multifactorial"

    return "other"


def _dominant_flag(moi_class: str) -> Optional[int]:
    if moi_class in {"AD", "XL_dom"}:
        return 1
    if moi_class in {"AR", "XL_rec"}:
        return 0
    return None


def _load_clinvar_summary() -> pd.DataFrame:
    summary_path = os.path.join(dataset_dir, "variant_summary.txt.gz")
    intermediate_path = os.path.join(dataset_dir, "IndelClinVarBenchmark.csv.gz")

    if os.path.exists(intermediate_path):
        print(f"Loading preprocessed data from {intermediate_path} ...")
        return pd.read_csv(intermediate_path, compression="gzip", low_memory=False)

    if not os.path.exists(summary_path):
        _download_file(clinvar_summary_url, summary_path)

    print("Loading and filtering ClinVar variant summary ...")
    df = pd.read_csv(
        summary_path,
        sep="\t",
        dtype={"Chromosome": str},
        low_memory=False,
        compression="gzip",
    )

    # Assembly
    df = df[df["Assembly"] == "GRCh38"].copy()
    print(f"After GRCh38 filter: {len(df)}")

    # Germline only
    df = df[df["OriginSimple"] != "somatic"].copy()
    print(f"After excluding somatic: {len(df)}")

    # Standard chromosomes
    df = df[df["Chromosome"].isin(STANDARD_CHROMS)].copy()
    print(f"After chromosome filter: {len(df)}")

    # Indel types only
    df = df[df["Type"].isin(INDEL_TYPES)].copy()
    print(f"After indel type filter: {len(df)}")

    # Clinical mappings
    df["ClinVar_annotation"] = df["ClinicalSignificance"].map(CLINICAL_SIGNIFICANCE_TO_LABEL)
    df["gold_stars"] = df["ReviewStatus"].map(REVIEW_STATUS_TO_GOLD_STARS)

    # Keep only clear benign/pathogenic
    df = df[df["ClinVar_annotation"].isin({0, 1})].copy()
    print(f"After annotation filter: {len(df)}")

    columns_to_keep = [
        "Chromosome",
        "PositionVCF",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "Type",
        "GeneSymbol",
        "VariationID",
        "ClinVar_annotation",
        "gold_stars",
    ]
    df = df[[c for c in columns_to_keep if c in df.columns]].copy()

    df.to_csv(intermediate_path, index=False, compression="gzip")
    print(f"Saved intermediate file: {intermediate_path}")
    return df


def _parse_moi_from_xml(xml_path: str) -> pd.DataFrame:
    try:
        from lxml import etree as ET
        print("Using lxml for optimized XML parsing")
    except ImportError:
        import xml.etree.ElementTree as ET
        print("lxml not found, using standard ElementTree (slower)")

    rows = []
    n_variation = 0
    n_with_moi = 0

    def open_maybe_gzip(path: str):
        if path.endswith(".gz"):
            return gzip.open(path, "rb")
        return open(path, "rb")

    with open_maybe_gzip(xml_path) as fh:
        context = ET.iterparse(fh, events=("end",))

        for _, elem in context:
            tag = elem.tag.rsplit("}", 1)[-1]

            if tag == "VariationArchive":
                n_variation += 1
                var_id = elem.get("VariationID") or elem.get("VariationId")
                if not var_id:
                    elem.clear()
                    continue

                moi_terms = set()
                for sub in elem.iter():
                    sub_tag = sub.tag.rsplit("}", 1)[-1]
                    if sub_tag == "Attribute" and sub.get("Type") == "ModeOfInheritance":
                        if sub.text:
                            term = sub.text.strip()
                            if term:
                                moi_terms.add(term)

                if moi_terms:
                    n_with_moi += 1
                    for term in moi_terms:
                        mc = _normalise_moi(term)
                        dom = _dominant_flag(mc)
                        rows.append(
                            {
                                "VariationID": int(var_id),
                                "moi_raw": term,
                                "moi_class": mc,
                                "dominant_flag": dom,
                            }
                        )

                elem.clear()

            if n_variation > 0 and n_variation % 100000 == 0:
                print(f"Processed {n_variation} records ...")

    print(f"Parsed {n_with_moi}/{n_variation} records with MOI annotations")

    if not rows:
        return pd.DataFrame(columns=["VariationID", "moi_raw", "moi_class", "dominant_flag"])

    df = pd.DataFrame(rows).drop_duplicates(["VariationID", "moi_raw"])
    df = df.sort_values("VariationID").reset_index(drop=True)

    # Prefer dominant when multiple MOIs exist, same as your SNP script
    prio = df["dominant_flag"].map({1: 2, 0: 1}).fillna(0).astype(int)
    df = (
        df.assign(_prio=prio)
        .sort_values(["VariationID", "_prio"], ascending=[True, False])
        .drop_duplicates("VariationID")
        .drop(columns="_prio")
    )

    return df


def _get_moi_annotations() -> pd.DataFrame:
    cache_dir = os.path.join(dataset_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    moi_cache_path = os.path.join(cache_dir, "clinvar_moi.pkl")
    if os.path.exists(moi_cache_path):
        print(f"Loading cached MOI annotations from {moi_cache_path} ...")
        return pd.read_pickle(moi_cache_path)

    xml_path = os.path.join(cache_dir, "ClinVarVCVRelease.xml.gz")
    if not os.path.exists(xml_path):
        _download_file(clinvar_vcv_xml_url, xml_path)

    moi_df = _parse_moi_from_xml(xml_path)
    pd.to_pickle(moi_df, moi_cache_path)
    print(f"Saved MOI cache to: {moi_cache_path}")
    return moi_df


def _is_valid_indel_allele(s: str) -> bool:
    """
    Allow only explicit A/C/G/T strings.
    Reject symbolic, missing, star, breakend-like, and ambiguous alleles.
    """
    if pd.isna(s):
        return False

    s = str(s).strip().upper()
    if not s:
        return False

    # Reject symbolic / malformed alleles
    forbidden_chars = set("<>[]*:/|")
    if any(ch in forbidden_chars for ch in s):
        return False

    return set(s).issubset(VALID_BASES)


def _classify_indel_type(ref: str, alt: str) -> str:
    if len(ref) < len(alt):
        return "insertion"
    if len(ref) > len(alt):
        return "deletion"
    return "complex_indel"


def _merge_and_filter(df: pd.DataFrame, moi_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(moi_df, on="VariationID", how="left", validate="many_to_one")
    print(f"After MOI merge: {len(df)}")

    df = df.rename(
        columns={
            "Chromosome": "chrom",
            "PositionVCF": "pos",
            "ReferenceAlleleVCF": "ref",
            "AlternateAlleleVCF": "alt",
            "GeneSymbol": "gene",
            "ClinVar_annotation": "label",
            "gold_stars": "quality_stars",
            "Type": "clinvar_type",
        }
    )

    # Quality filter
    df = df[df["quality_stars"] > 1].copy()
    print(f"After quality filter (2+ stars): {len(df)}")

    # Keep only variants whose genotype label can actually be assigned.
    #
    # Zygosity changes the label only for PATHOGENIC variants (dominant: het=hom=1;
    # recessive: het=0, hom=1). A benign variant is 0 in either zygosity and never consults
    # dominant_flag. Dropping every row without an inheritance call therefore discarded
    # benign indels for lacking an annotation they do not use, leaving only a few percent
    # negatives -- which is precisely what made zygosity, rather than sequence, the dominant
    # signal in this benchmark: with almost no benign variants, "positive" is nearly
    # synonymous with "homozygous" and any zygosity-sensitive score scores ~0.8 (or ~0.2
    # inverted) while learning nothing about variant effect.
    before = len(df)
    df = df[(df["label"] == 0) | df["dominant_flag"].notna()].copy()
    print(
        f"After MOI filter (inheritance required for pathogenic only): {len(df)} "
        f"(dropped {before - len(df)}; benign rows are kept without an MOI call)"
    )

    # Valid explicit alleles only
    df = df[
        df["ref"].map(_is_valid_indel_allele) &
        df["alt"].map(_is_valid_indel_allele)
    ].copy()
    print(f"After allele validation: {len(df)}")

    # Ensure these are actually indels at allele level
    df = df[df["ref"].str.len() != df["alt"].str.len()].copy()
    print(f"After strict indel-length filter: {len(df)}")

    # Optional: keep left-anchored VCF-style indels only
    # This removes odd rows where ref/alt share no anchor at first base.
    df = df[df["ref"].str[0] == df["alt"].str[0]].copy()
    print(f"After left-anchor filter: {len(df)}")

    df["ref_len"] = df["ref"].str.len().astype(int)
    df["alt_len"] = df["alt"].str.len().astype(int)
    df["delta_len"] = df["alt_len"] - df["ref_len"]
    df["indel_type"] = [
        _classify_indel_type(r, a) for r, a in zip(df["ref"], df["alt"])
    ]

    return df


def _augment_with_zygosity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate each variant into het and hom rows.

    Label logic (genotype-level, i.e. "is this genotype disease-causing"):
    - Benign: het=0, hom=0
    - Pathogenic Dominant: het=1, hom=1
    - Pathogenic Recessive: het=0, hom=1

    Both zygosities are emitted for EVERY variant on purpose. Dropping the redundant rows
    (benign/dominant, where het and hom carry the same label) would leave hom rows enriched
    for positives and make zygosity an even better predictor than it already is. The cure
    for the zygosity confound is a real benign population plus stratified reporting, not
    pruning rows.

    Two extra columns are emitted so downstream evaluation can see the confound instead of
    scoring through it:
      - variant_id:    groups the het/hom pair of one variant, for grouped splits and for
                       per-variant (rather than per-genotype) aggregation.
      - label_variant: zygosity-INDEPENDENT pathogenicity of the variant itself. This is
                       the target a zygosity-blind representation (haploid/biallelic mode)
                       can actually be asked to predict; `label` is not, because the het and
                       hom rows of a recessive variant are the same sequence with opposite
                       labels.
    """
    df = df.copy()

    df["variant_id"] = (
        df["chrom"].astype(str) + ":" + df["pos"].astype(str)
        + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str)
    )
    df["label_variant"] = df["label"].astype(int)

    het = df.copy()
    het["zygosity"] = "het"

    hom = df.copy()
    hom["zygosity"] = "hom"

    aug = pd.concat([het, hom], ignore_index=True)
    aug["label_genotype"] = np.nan

    benign = aug["label"] == 0
    path = aug["label"] == 1
    dom = aug["dominant_flag"] == 1
    rec = aug["dominant_flag"] == 0

    aug.loc[benign, "label_genotype"] = 0
    aug.loc[path & dom, "label_genotype"] = 1
    aug.loc[path & rec & (aug["zygosity"] == "het"), "label_genotype"] = 0
    aug.loc[path & rec & (aug["zygosity"] == "hom"), "label_genotype"] = 1

    aug = aug.dropna(subset=["label_genotype"]).copy()
    aug["label_genotype"] = aug["label_genotype"].astype(int)

    print(f"After zygosity augmentation: {len(aug)} instances")
    return aug


def _auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Tie-aware AUROC (Mann-Whitney U). Local so this builder keeps its numpy/pandas deps."""
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    positions = np.arange(1, len(s) + 1, dtype=float)
    ranks = np.empty(len(s), dtype=float)

    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = positions[i:j + 1].mean()
        i = j + 1

    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _report_confounds(df: pd.DataFrame) -> None:
    """
    Print what a model would have to beat to demonstrate variant-effect signal.

    The headline number is the zygosity-only baseline: the AUROC obtained by predicting
    "homozygous" with no access to the sequence at all. Any score that correlates with
    zygosity -- an embedding distance (a homozygous indel shifts the whole downstream
    window, a heterozygous one does not), or a likelihood ratio (heterozygous rows carry
    rare paired/gap tokens) -- inherits this number or its inversion (1 - AUROC) for free.
    """
    y = df["label"].to_numpy()
    is_hom = (df["zygosity"] == "hom").to_numpy().astype(float)

    hom_auroc = _auroc(y, is_hom)
    n = len(df)

    print("\n--- Confound report ---")
    print(f"  Rows: {n}  |  prevalence (label=1): {y.mean():.4f}")

    per_variant = df.drop_duplicates("variant_id")
    n_benign = int((per_variant["label_variant"] == 0).sum())
    n_dom = int(((per_variant["label_variant"] == 1) & (per_variant["dominant_flag"] == 1)).sum())
    n_rec = int(((per_variant["label_variant"] == 1) & (per_variant["dominant_flag"] == 0)).sum())
    n_var = len(per_variant)
    print(
        f"  Variants: {n_var}  |  benign {n_benign} ({n_benign / max(n_var, 1):.1%}), "
        f"pathogenic-dominant {n_dom} ({n_dom / max(n_var, 1):.1%}), "
        f"pathogenic-recessive {n_rec} ({n_rec / max(n_var, 1):.1%})"
    )

    print(
        f"  ZYGOSITY-ONLY BASELINE AUROC on `label`: {hom_auroc:.4f} "
        f"(a het-leaning score scores {1 - hom_auroc:.4f})"
    )
    if not np.isnan(hom_auroc) and abs(hom_auroc - 0.5) > 0.15:
        print(
            "  WARNING: zygosity alone separates the classes. Any headline AUROC near this\n"
            "           value (or its inversion) is evidence of a zygosity detector, not of\n"
            "           variant-effect signal. Report stratified by zygosity, and/or use\n"
            "           `label_variant` for zygosity-blind (haploid/biallelic) modes."
        )

    contradictory = (
        df.groupby("variant_id")["label"].nunique().gt(1).sum()
    )
    print(
        f"  Variants whose het/hom rows carry different labels: {contradictory} "
        f"({2 * contradictory} rows). A zygosity-blind representation sees these as one\n"
        f"           input with two labels -- an irreducible ceiling for haploid/biallelic modes."
    )
    print(
        f"           That is {2 * contradictory / max(n, 1):.1%} of all rows. "
        f"`label_variant` has no such collision."
    )
    print("--- end confound report ---\n")


def _finalize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_keep = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "gene",
        "variant_id",
        "zygosity",
        "label_genotype",
        "label_variant",
        "indel_type",
        "ref_len",
        "alt_len",
        "delta_len",
        "dominant_flag",
        "quality_stars",
    ]
    cols_present = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_present].copy()

    df = df.rename(columns={"label_genotype": "label"})

    df["chrom"] = pd.Categorical(df["chrom"], categories=STANDARD_CHROMS, ordered=True)
    df = df.sort_values(["chrom", "pos", "ref", "alt", "zygosity"]).reset_index(drop=True)

    print(f"Final benchmark: {len(df)} instances")
    print(f"  Label 0: {(df['label'] == 0).sum()}")
    print(f"  Label 1: {(df['label'] == 1).sum()}")
    print(f"  Insertions: {(df['indel_type'] == 'insertion').sum()}")
    print(f"  Deletions: {(df['indel_type'] == 'deletion').sum()}")
    print(f"  het rows: {(df['zygosity'] == 'het').sum()}, hom rows: {(df['zygosity'] == 'hom').sum()}")

    _report_confounds(df)

    return df


def create_benchmark() -> pd.DataFrame:
    df = _load_clinvar_summary()
    moi_df = _get_moi_annotations()
    df = _merge_and_filter(df, moi_df)
    df = _augment_with_zygosity_labels(df)
    df = _finalize_benchmark(df)

    os.makedirs(dataset_dir, exist_ok=True)
    df.to_parquet(dataset_path, index=False)
    print(f"Saved ClinVar diploid indel benchmark to: {dataset_path}")
    return df


if __name__ == "__main__":
    create_benchmark()