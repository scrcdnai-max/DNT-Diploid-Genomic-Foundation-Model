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
    'criteria provided, single submitter': 1,
    'criteria provided, multiple submitters, no conflicts': 2,
    'criteria provided, conflicting interpretations': 1,
    'no assertion criteria provided': np.nan,
    'reviewed by expert panel': 3,
    'no assertion provided': np.nan,
    'no interpretation for the single variant': np.nan,
    'practice guideline': 4,
}

CLINICAL_SIGNIFICANCE_TO_LABEL = {
    'Benign': 0,
    'Likely benign': 0,  # Collapsed to binary
    'Benign/Likely benign': 0,
    'Pathogenic': 1,
    'Likely pathogenic': 1,  # Collapsed to binary
    'Pathogenic/Likely pathogenic': 1,
}

root_data_dir_path = "../data"
reference_genome_path = os.path.join(
            root_data_dir_path, "reference_genome", "hg38.fa"
        )
dataset_dir = os.path.join(root_data_dir_path, "diploid_clean")
dataset_path = os.path.join(dataset_dir, "clinvar_diploid.parquet")


clinvar_summary_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-01.txt.gz"
clinvar_vcv_xml_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarVCVRelease_2026-01.xml.gz"

def _download_file(url: str, dest: str) -> None:
    """Download a file using wget."""
    print(f"Downloading {url}...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    result = subprocess.run(
        ["wget", "-c", "-O", dest, url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")
    print(f"Downloaded to: {dest}")

def _load_clinvar_summary():
    summary_path = os.path.join(dataset_dir, "variant_summary.txt.gz")
    intermediate_path = os.path.join(dataset_dir, "ZygosityClinVarBenchmark.csv.gz")
    if os.path.exists(intermediate_path):
        print(f"Loading preprocessed data from {intermediate_path}...")
        return pd.read_csv(intermediate_path, compression='gzip', low_memory=False)
    if not os.path.exists(summary_path):
        print("Downloading ClinVar variant_summary.txt.gz...")
        _download_file(clinvar_summary_url, summary_path)
        
    print("Loading and filtering ClinVar variant summary...")
    df = pd.read_csv(summary_path, sep='\t', dtype={'Chromosome': str}, low_memory=False, compression='gzip')
    # Filter to GRCh38
    df = df[df['Assembly'] == 'GRCh38'].copy()
    print(f"After GRCh38 filter: {len(df)} variants")
    # Exclude somatic (focus on germline)
    df = df[df['OriginSimple'] != 'somatic'].copy()
    print(f"After excluding somatic: {len(df)} variants")
    # Keep only standard chromosomes
    basic_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    df = df[df['Chromosome'].isin(basic_chromosomes)].copy()
    print(f"After chromosome filter: {len(df)} variants")
    # Filter to SNVs only
    df = df[df['Type'] == 'single nucleotide variant'].copy()
    print(f"After SNV filter: {len(df)} variants")
    # Map annotations
    df['ClinVar_annotation'] = df['ClinicalSignificance'].map(CLINICAL_SIGNIFICANCE_TO_LABEL)
    df['gold_stars'] = df['ReviewStatus'].map(REVIEW_STATUS_TO_GOLD_STARS)
    # Keep only clear benign (0) or pathogenic (1)
    df = df[df['ClinVar_annotation'].isin({0, 1})].copy()
    print(f"After annotation filter: {len(df)} variants")
    # Select and rename columns
    columns_to_keep = ['Chromosome', 'PositionVCF', 'ReferenceAlleleVCF', 'AlternateAlleleVCF',
        'Type', 'GeneSymbol', 'VariationID', 'ClinVar_annotation', 'gold_stars']
    df = df[[c for c in columns_to_keep if c in df.columns]].copy()
    # Save intermediate for reuse
    df.to_csv(intermediate_path, index=False, compression='gzip')
    print(f"Saved intermediate file: {intermediate_path}")
    return df
    
def _normalise_moi(raw: str) -> str:
    """Normalize Mode of Inheritance string to standard category."""
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
    """Map MOI class to dominance flag: 1=dominant, 0=recessive, None=unknown."""
    if moi_class in {"AD", "XL_dom"}:
        return 1
    if moi_class in {"AR", "XL_rec"}:
        return 0
    return None

def _parse_moi_from_xml(xml_path: str):
    try:
        from lxml import etree as ET
        print("Using lxml for optimized XML parsing")
    except ImportError:
        import xml.etree.ElementTree as ET
        print("lxml not found, using standard ElementTree (slower)")
    rows = []
    n_variation = 0
    n_with_moi = 0
    
    def open_maybe_gzip(path):
        if path.endswith(".gz"):
            return gzip.open(path, "rb")
        return open(path, "rb")
    
    try:
        with open_maybe_gzip(xml_path) as fh:
            context = ET.iterparse(fh, events=("end",))
        
            for event, elem in context:
                tag = elem.tag.rsplit("}", 1)[-1]  # strip namespace
            
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
                            rows.append({
                                "VariationID": int(var_id),
                                "moi_raw": term,
                                "moi_class": mc,
                                "dominant_flag": dom,
                            })
                    
                    elem.clear()
            
                if n_variation > 0 and n_variation % 100000 == 0:
                    print(f"Processed {n_variation} records...")
    except EOFError:
        print("Encountered EOF earlier than expected")
    
    print(f"Parsed {n_with_moi}/{n_variation} records with MOI annotations")
    
    if not rows:
        return pd.DataFrame(columns=["VariationID", "moi_raw", "moi_class", "dominant_flag"])
    
    df = pd.DataFrame(rows).drop_duplicates(["VariationID", "moi_raw"])
    df = df.sort_values("VariationID").reset_index(drop=True)
    
    # Keep only one MOI per VariationID (prefer dominant)
    prio = df["dominant_flag"].map({1: 2, 0: 1}).fillna(0).astype(int)
    df = (df.assign(_prio=prio)
        .sort_values(["VariationID","_prio"], ascending=[True, False])
        .drop_duplicates("VariationID")
        .drop(columns="_prio"))
    
    return df
    
def _get_moi_annotations():
    cache_dir = os.path.join(dataset_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    moi_cache_path = os.path.join(cache_dir, "clinvar_moi.pkl")
    if os.path.exists(moi_cache_path):
        print(f"Loading cached MOI annotations from {moi_cache_path}...")
        return pd.read_pickle(moi_cache_path)
    xml_path = os.path.join(cache_dir, "ClinVarVCVRelease.xml.gz")
    if not os.path.exists(xml_path):
        print("Downloading ClinVar VCV XML...")
        _download_file(clinvar_vcv_xml_url, xml_path)
    moi_df = _parse_moi_from_xml(xml_path)
    pd.to_pickle(moi_df, moi_cache_path)
    print(f"Saved MOI cache to: {moi_cache_path}")
    return moi_df
    
def _merge_and_filter(df: pd.DataFrame, moi_df: pd.DataFrame) -> pd.DataFrame:
    """Merge variant summary with MOI annotations and filter."""
    # Merge on VariationID
    df = df.merge(moi_df, on="VariationID", how="left", validate="many_to_one")
    print(f"After MOI merge: {len(df)} variants")
    
    # Rename columns for consistency
    df = df.rename(columns={
        'Chromosome': 'chrom',
        'PositionVCF': 'pos',
        'ReferenceAlleleVCF': 'ref',
        'AlternateAlleleVCF': 'alt',
        'GeneSymbol': 'gene',
        'ClinVar_annotation': 'label',
        'gold_stars': 'quality_stars',
    })
    
    # Filter to 2+ gold stars
    df = df[df['quality_stars'] > 1].copy()
    print(f"After quality filter (2+ stars): {len(df)} variants")
    
    # Keep only variants whose genotype label can actually be assigned.
    #
    # Zygosity changes the label only for PATHOGENIC variants (dominant: het=hom=1;
    # recessive: het=0, hom=1). A benign variant is 0 in either zygosity and never consults
    # dominant_flag. Dropping every row without an inheritance call therefore discarded
    # benign SNVs for lacking an annotation they do not use, leaving only a few percent
    # negatives -- which is what makes zygosity, rather than sequence, the dominant signal:
    # with almost no benign variants, "positive" is nearly synonymous with "homozygous", and
    # any zygosity-sensitive score inherits that separation for free.
    before = len(df)
    df = df[(df['label'] == 0) | df['dominant_flag'].notna()].copy()
    print(
        f"After MOI filter (inheritance required for pathogenic only): {len(df)} variants "
        f"(dropped {before - len(df)}; benign variants are kept without an MOI call)"
    )
    
    # Ensure valid alleles
    valid_alleles = {'A', 'C', 'G', 'T'}
    df = df[
        df['ref'].isin(valid_alleles) & df['alt'].isin(valid_alleles)
    ].copy()
    print(f"After allele validation: {len(df)} variants")
    
    return df

    
def _augment_with_zygosity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment dataframe by creating het and hom rows for each variant.

    Logic (genotype-level, i.e. "is this genotype disease-causing"):
    - Benign (label=0): het=0, hom=0
    - Pathogenic Dominant (flag=1): het=1, hom=1
    - Pathogenic Recessive (flag=0): het=0 (carrier), hom=1 (affected)

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
                       hom rows of a recessive variant collapse to the same haploid sequence
                       with opposite labels.
    """
    df = df.copy()

    df['variant_id'] = (
        df['chrom'].astype(str) + ':' + df['pos'].astype(str)
        + ':' + df['ref'].astype(str) + ':' + df['alt'].astype(str)
    )
    df['label_variant'] = df['label'].astype(int)

    # Create het and hom copies
    het = df.copy()
    het['zygosity'] = 'het'

    hom = df.copy()
    hom['zygosity'] = 'hom'

    aug = pd.concat([het, hom], ignore_index=True)
    
    # Initialize label_genotype
    aug['label_genotype'] = np.nan
    
    # Benign: Always 0
    mask_benign = aug['label'] == 0
    aug.loc[mask_benign, 'label_genotype'] = 0
    
    # Pathogenic cases
    mask_path = aug['label'] == 1  # mask_pathogenic
    
    # Dominant: het=1, hom=1
    mask_dominant = mask_path & (aug['dominant_flag'] == 1)
    aug.loc[mask_dominant, 'label_genotype'] = 1
    
    # Recessive: het=0 (carrier), hom=1 (affected)
    mask_recessive = mask_path & (aug['dominant_flag'] == 0)
    aug.loc[mask_recessive & (aug['zygosity'] == 'het'), 'label_genotype'] = 0
    aug.loc[mask_recessive & (aug['zygosity'] == 'hom'), 'label_genotype'] = 1
    
    # Drop rows with unknown genotype label
    aug = aug.dropna(subset=['label_genotype'])
    aug['label_genotype'] = aug['label_genotype'].astype(int)
    
    print(f"After zygosity augmentation: {len(aug)} instances")
    
    return aug
    
    
def _auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Tie-aware AUROC (Mann-Whitney U). Local so this builder keeps its numpy/pandas deps."""
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    order = np.argsort(s, kind='mergesort')
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
    zygosity -- an embedding distance, or a likelihood ratio (heterozygous rows carry rare
    paired tokens the homozygous rows do not) -- inherits this number, or its inversion
    (1 - AUROC), for free.
    """
    y = df['label'].to_numpy()
    is_hom = (df['zygosity'] == 'hom').to_numpy().astype(float)

    hom_auroc = _auroc(y, is_hom)
    n = len(df)

    print("\n--- Confound report ---")
    print(f"  Rows: {n}  |  prevalence (label=1): {y.mean():.4f}")

    if 'variant_id' in df.columns:
        per_variant = df.drop_duplicates('variant_id')
        n_var = len(per_variant)
        if 'label_variant' in per_variant.columns and 'dominant_flag' in per_variant.columns:
            n_benign = int((per_variant['label_variant'] == 0).sum())
            n_dom = int(((per_variant['label_variant'] == 1) & (per_variant['dominant_flag'] == 1)).sum())
            n_rec = int(((per_variant['label_variant'] == 1) & (per_variant['dominant_flag'] == 0)).sum())
            print(
                f"  Variants: {n_var}  |  benign {n_benign} ({n_benign / max(n_var, 1):.1%}), "
                f"pathogenic-dominant {n_dom} ({n_dom / max(n_var, 1):.1%}), "
                f"pathogenic-recessive {n_rec} ({n_rec / max(n_var, 1):.1%})"
            )
        contradictory = int(df.groupby('variant_id')['label'].nunique().gt(1).sum())
    else:
        n_var, contradictory = 0, 0

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

    if n_var:
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


def _add_diploid_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Add diploid token columns for future diploid model support."""
    # Load token map
    #token_tsv_path = os.path.join(
    #    os.path.dirname(__file__),
    #    "..", "clinvar_dan_benchmarks", "diploid_tokens.tsv"
    #)
    # Look next to the parquet this script writes, not at a stale sibling directory: the
    # tasks load "<root>/diploid_clean/diploid_tokens.tsv", and pointing elsewhere silently
    # produced all-None token columns.
    token_tsv_candidates = [
        os.path.join(dataset_dir, "diploid_tokens.tsv"),
        os.path.join(root_data_dir_path, "clinvar_zygosity", "diploid_tokens.tsv"),
    ]
    token_tsv_path = next(
        (p for p in token_tsv_candidates if os.path.exists(p)), token_tsv_candidates[0]
    )

    if not os.path.exists(token_tsv_path):
        print(
            f"Diploid token file not found; tried: {token_tsv_candidates}. "
            f"genotype_pair/diploid_token columns will be empty "
            f"(canonical copy: preprocess/diploid_tokens.tsv)"
        )
        df['genotype_pair'] = None
        df['diploid_token'] = None
        return df

    print(f"Using diploid token table: {token_tsv_path}")
    
    token_df = pd.read_csv(token_tsv_path, sep='\t', comment='#',
                           names=['Pair', 'Order', 'Symbol', 'Unicode'])
    token_map = dict(zip(token_df['Pair'], token_df['Symbol']))
    
    def get_diploid_info(row):
        ref = str(row.get('ref', '')).strip()
        alt = str(row.get('alt', '')).strip()
        zyg = row.get('zygosity', '')
        
        if zyg == 'hom':
            pair = alt + alt
        elif zyg == 'het':
            pair = ref + alt
        else:
            return None, None
        
        token = token_map.get(pair, None)
        return pair, token
    
    results = df.apply(get_diploid_info, axis=1, result_type='expand')
    df['genotype_pair'] = results[0]
    df['diploid_token'] = results[1]
    
    n_mapped = df['diploid_token'].notna().sum()
    print(f"Mapped diploid tokens: {n_mapped}/{len(df)}")
    
    return df

def _finalize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Finalize the benchmark dataframe."""
    # Select and rename columns
    cols_to_keep = [
        'chrom', 'pos', 'ref', 'alt', 'gene',
        'variant_id',
        'zygosity', 'label_genotype', 'label_variant',
        # dominant_flag and quality_stars were dropped here, which left the parquet with no
        # way to stratify by inheritance or filter by review quality downstream -- exactly
        # the columns needed to tell a variant-effect result from a zygosity artifact.
        'dominant_flag', 'quality_stars',
        'genotype_pair', 'diploid_token'
    ]
    cols_present = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_present].copy()

    # Rename label_genotype to label:
    df = df.rename(columns={'label_genotype': 'label'})

    chrom_order = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
    df["chrom"] = pd.Categorical(df["chrom"], categories=chrom_order, ordered=True)
    df = df.sort_values(["chrom", "pos"]).reset_index(drop=True)

    print(f"Final benchmark: {len(df)} instances")
    print(f"  Label 0: {(df['label'] == 0).sum()}")
    print(f"  Label 1: {(df['label'] == 1).sum()}")
    print(f"  het rows: {(df['zygosity'] == 'het').sum()}, hom rows: {(df['zygosity'] == 'hom').sum()}")

    _report_confounds(df)

    return df
    
def _finalize_benchmark_no_dup(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "chrom", "pos", "ref", "alt", "gene",
        "dominant_flag", "quality_stars",
        "label",          # variant-level benign/pathogenic
        "label_het", "label_hom",
        "genotype_pair_het", "diploid_token_het",
        "genotype_pair_hom", "diploid_token_hom",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    # proper chrom ordering
    chrom_order = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
    df["chrom"] = pd.Categorical(df["chrom"], categories=chrom_order, ordered=True)
    df = df.sort_values(["chrom", "pos"]).reset_index(drop=True)

    print(f"Final benchmark: {len(df)} variants (no duplication)")
    print(f"  Label 0: {(df['label'] == 0).sum()}")
    print(f"  Label 1: {(df['label'] == 1).sum()}")
    return df
    
def _add_diploid_fields_no_dup(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per variant. Add diploid label/token fields for het and hom.

    label rules:
      - benign (label=0): het=0, hom=0
      - pathogenic + dominant_flag=1: het=1, hom=1
      - pathogenic + dominant_flag=0: het=0, hom=1
    """
    df = df.copy()

    # --- labels ---
    df["label_het"] = np.nan
    df["label_hom"] = np.nan

    benign = df["label"] == 0
    path = df["label"] == 1
    dom = df["dominant_flag"] == 1
    rec = df["dominant_flag"] == 0

    # benign
    df.loc[benign, ["label_het", "label_hom"]] = 0

    # pathogenic dominant
    df.loc[path & dom, ["label_het", "label_hom"]] = 1

    # pathogenic recessive
    df.loc[path & rec, "label_het"] = 0
    df.loc[path & rec, "label_hom"] = 1

    # drop anything we couldn't label
    df = df.dropna(subset=["label_het", "label_hom"]).copy()
    df["label_het"] = df["label_het"].astype(int)
    df["label_hom"] = df["label_hom"].astype(int)

    # --- tokens ---
    token_tsv_path = os.path.join(
        os.path.dirname(__file__),
        "..", "clinvar_dan_benchmarks", "diploid_tokens.tsv"
    )

    if not os.path.exists(token_tsv_path):
        print(f"Diploid token file not found: {token_tsv_path}")
        df["genotype_pair_het"] = None
        df["diploid_token_het"] = None
        df["genotype_pair_hom"] = None
        df["diploid_token_hom"] = None
        return df

    token_df = pd.read_csv(
        token_tsv_path, sep="\t", comment="#",
        names=["Pair", "Order", "Symbol", "Unicode"]
    )
    token_map = dict(zip(token_df["Pair"], token_df["Symbol"]))

    # genotype pairs
    df["genotype_pair_het"] = df["ref"].astype(str).str.strip() + df["alt"].astype(str).str.strip()
    df["genotype_pair_hom"] = df["alt"].astype(str).str.strip() + df["alt"].astype(str).str.strip()

    # map tokens
    df["diploid_token_het"] = df["genotype_pair_het"].map(token_map)
    df["diploid_token_hom"] = df["genotype_pair_hom"].map(token_map)

    n_het = df["diploid_token_het"].notna().sum()
    n_hom = df["diploid_token_hom"].notna().sum()
    print(f"Mapped diploid tokens (het): {n_het}/{len(df)}")
    print(f"Mapped diploid tokens (hom): {n_hom}/{len(df)}")

    return df

def create_benchmark():
    df = _load_clinvar_summary()
    moi_df = _get_moi_annotations()
    df = _merge_and_filter(df, moi_df)
    dup_diploids = True
    if dup_diploids:
        df = _augment_with_zygosity_labels(df)
        df = _add_diploid_tokens(df)
        df = _finalize_benchmark(df)
    else:
        df = _add_diploid_fields_no_dup(df)
        df = _finalize_benchmark_no_dup(df)
    os.makedirs(dataset_dir, exist_ok=True)
    df.to_parquet(dataset_path, index=False)
    print(f"Saved ClinVar Zygosity benchmark to: {dataset_path}")
    return df


# Guarded so the module can be imported (for tests, or to reuse _report_confounds) without
# kicking off a multi-GB ClinVar download and a full rebuild as an import side effect.
if __name__ == "__main__":
    create_benchmark()
