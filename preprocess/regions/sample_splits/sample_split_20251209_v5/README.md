# 1000 Genomes Project: Stratified Train/Test Split Report

**Report Generated:** 2025-01-09  
**Algorithm Version:** Component-Aware Stratified Split v5  
**Input Dataset:** 20130606_g1k_3202_samples_ped_population.csv  

---

## Executive Summary

This report documents the methodology and results of splitting the 1000 Genomes Project sample set (3,202 samples) into training (80%) and testing (20%) subsets for machine learning applications. The split maintains **strict family integrity** while achieving **balanced representation** across demographic dimensions.

| Metric | Value |
|--------|-------|
| Total Samples | 3,202 |
| Training Set | 2,560 (79.95%) |
| Test Set | 642 (20.05%) |
| Related Individuals Separated | **0** |

---

## 1. Dataset Overview

### 1.1 Sample Composition

The 1000 Genomes Project Phase 3 dataset contains 3,202 samples from 26 populations grouped into 5 superpopulations:

| Superpopulation | Code | Populations | Samples | Percentage |
|-----------------|------|-------------|---------|------------|
| African | AFR | ACB, ASW, ESN, GWD, LWK, MSL, YRI | 893 | 27.9% |
| Ad Mixed American | AMR | CLM, MXL, PEL, PUR | 490 | 15.3% |
| East Asian | EAS | CDX, CHB, CHS, JPT, KHV | 585 | 18.3% |
| European | EUR | CEU, FIN, GBR, IBS, TSI | 633 | 19.8% |
| South Asian | SAS | BEB, GIH, ITU, PJL, STU | 601 | 18.8% |

### 1.2 Sex Distribution

| Sex | Samples | Percentage |
|-----|---------|------------|
| Male | 1,599 | 49.9% |
| Female | 1,603 | 50.1% |

### 1.3 Family Structure

Samples include trios (father-mother-child) and extended pedigrees:

| Component Size | Description | Components | Samples | Percentage |
|----------------|-------------|------------|---------|------------|
| 1 | Unrelated individuals | 1,407 | 1,407 | 43.9% |
| 2 | Parent-child pair | 1 | 2 | 0.1% |
| 3 | Standard trios | 587 | 1,761 | 55.0% |
| 4 | Nuclear family (2 children) | 2 | 8 | 0.2% |
| 6 | 3-generation pedigrees | 4 | 24 | 0.7% |
| **Total** | | **2,001** | **3,202** | **100%** |

### 1.4 Trio Role Distribution

| Role | Definition | Samples | Percentage |
|------|------------|---------|------------|
| Unrelated | No parents or children in dataset | 1,402 | 43.8% |
| Parent | Has at least one child in dataset | 1,195 | 37.3% |
| Child | Has at least one parent in dataset | 605 | 18.9% |

---

## 2. Splitting Considerations

### 2.1 Family Integrity (Critical)

**Problem:** Family members share genetic variants due to Mendelian inheritance. If relatives appear in both training and test sets, models can exploit familial patterns rather than learning generalizable features, leading to inflated performance metrics.

**Solution:** All members of a **connected pedigree** must be assigned to the same set. This includes:
- Direct parent-child relationships
- Siblings (individuals sharing at least one parent)
- Extended multi-generational relationships (e.g., grandparent-grandchild)

**Implementation:** A union-find algorithm identifies true connected components by:
1. Adding edges between each individual and their parents (if present in dataset)
2. Adding edges between siblings (individuals sharing at least one parent)
3. Computing connected components

### 2.2 FamilyID vs. Connected Components

**Discovery:** The `FamilyID` field in the PED file does not always capture the complete pedigree structure. Some biologically connected families span multiple FamilyIDs.

| Metric | FamilyID-based | Component-based |
|--------|----------------|-----------------|
| Number of groups | 1,879 | 2,001 |
| Multi-family pedigrees identified | 0 | 6 |

**Six pedigrees spanning multiple FamilyIDs:**

| Component | FamilyIDs | Population | Size | Structure |
|-----------|-----------|------------|------|-----------|
| NA18912 | Y028, Y117 | YRI | 6 | 3-generation |
| NA19660 | m008, m011 | MXL | 6 | 3-generation |
| NA19679 | m004, m009 | MXL | 6 | 3-generation |
| HG00701 | SH074, SH089 | CHS | 6 | 3-generation |
| HG03469 | SL56, SL56i | MSL | 3 | Trio |
| NA20282 | 2467, 2469 | ASW | 3 | Trio |

**Impact of ignoring this:** Using FamilyID directly caused 2 violations where related individuals were separated across train/test sets.

### 2.3 Variable Family Sizes

**Problem:** Families range from 1 to 6 members. Naïve splitting by family count (rather than sample count) leads to imbalanced sample ratios. Additionally, larger families contain proportionally more parents and children, which can skew trio role distribution.

**Solution:** Stratify by **Superpopulation × Component Size**, ensuring that within each stratum, components are split to achieve ~80% of *samples* (not families) in training.

### 2.4 Demographic Stratification

**Objective:** Ensure both training and test sets are representative of the full dataset across multiple demographic dimensions.

**Dimensions stratified:**

| Dimension | Categories | Rationale |
|-----------|------------|-----------|
| Superpopulation | 5 | Population-specific allele frequencies and LD patterns |
| Component Size | 5 (1,2,3,4,6) | Preserve family structure distribution |
| Sex | 2 (indirect) | Balanced through component assignment |
| Trio Role | 3 (indirect) | Balanced through component size stratification |
| Population | 26 (indirect) | Inherits balance from superpopulation stratification |

### 2.5 Reproducibility

**Requirement:** The split must be deterministic and reproducible.

**Implementation:** 
- Fixed random seed (42) for all shuffling operations
- Documented algorithm version (v5)
- MD5 checksums provided for all output files

---

## 3. Algorithm Description

### 3.1 Pseudocode

```
ALGORITHM: Component-Aware Stratified Train/Test Split

INPUT: 
  - samples: DataFrame with SampleID, FatherID, MotherID, Superpopulation
  - train_ratio: target proportion for training (0.8)
  - seed: random seed for reproducibility (42)

OUTPUT:
  - samples with 'split' column ('train' or 'test')

STEP 1: Build Connected Components
  edges ← ∅
  FOR each sample s:
    IF s.father IN samples: edges ← edges ∪ {(s, s.father)}
    IF s.mother IN samples: edges ← edges ∪ {(s, s.mother)}
  
  FOR each parent p with children C:
    FOR each pair (c1, c2) in C:
      edges ← edges ∪ {(c1, c2)}  # sibling edge
  
  components ← UnionFind(samples, edges)

STEP 2: Stratified Assignment
  train_components ← ∅
  
  FOR each superpopulation sp:
    FOR each component_size sz:
      stratum ← components WHERE superpop=sp AND size=sz
      SHUFFLE(stratum, seed)
      
      target_samples ← |stratum| × sz × train_ratio
      n_train ← ROUND(target_samples / sz)
      
      # Ensure at least 1 in each set if possible
      IF |stratum| ≥ 2:
        n_train ← CLAMP(n_train, 1, |stratum| - 1)
      
      train_components ← train_components ∪ stratum[0:n_train]

STEP 3: Assign Splits
  FOR each sample s:
    IF component(s) IN train_components:
      s.split ← 'train'
    ELSE:
      s.split ← 'test'

RETURN samples
```

### 3.2 Complexity

- **Time:** O(n × α(n)) where n = number of samples, α = inverse Ackermann function (effectively constant)
- **Space:** O(n) for union-find data structure

---

## 4. Results

### 4.1 Overall Split

| Set | Samples | Percentage | Target |
|-----|---------|------------|--------|
| Training | 2,560 | 79.95% | 80% |
| Test | 642 | 20.05% | 20% |

### 4.2 Component Integrity Verification

| Metric | Value |
|--------|-------|
| Total connected components | 2,001 |
| Components split across train/test | **0** |
| Verification status | ✓ **PASSED** |

### 4.3 Stratification by Superpopulation

| Superpopulation | Train | Test | Train % | Deviation |
|-----------------|-------|------|---------|-----------|
| AFR | 715 | 178 | 80.1% | +0.1% |
| AMR | 389 | 101 | 79.4% | -0.6% |
| EAS | 469 | 116 | 80.2% | +0.2% |
| EUR | 508 | 125 | 80.3% | +0.3% |
| SAS | 479 | 122 | 79.7% | -0.3% |

**Maximum deviation: 0.6%**

### 4.4 Stratification by Sex

| Sex | Train | Test | Train % | Deviation |
|-----|-------|------|---------|-----------|
| Male | 1,279 | 320 | 80.0% | 0.0% |
| Female | 1,281 | 322 | 79.9% | -0.1% |

**Maximum deviation: 0.1%**

### 4.5 Stratification by Trio Role

| Role | Train | Test | Train % | Deviation |
|------|-------|------|---------|-----------|
| Unrelated | 1,122 | 280 | 80.0% | 0.0% |
| Parent | 955 | 240 | 79.9% | -0.1% |
| Child | 483 | 122 | 79.8% | -0.2% |

**Maximum deviation: 0.2%**

### 4.6 Stratification by Component Size

| Size | Description | Train | Test | Train % |
|------|-------------|-------|------|---------|
| 1 | Unrelated individuals | 1,126 | 281 | 80.0% |
| 2 | Parent-child pair | 2 | 0 | 100.0%* |
| 3 | Standard trios | 1,410 | 351 | 80.1% |
| 4 | Nuclear family (2 children) | 4 | 4 | 50.0%* |
| 6 | 3-generation pedigrees | 18 | 6 | 75.0%* |

*\* Higher deviation due to very small component counts (1, 2, and 4 components respectively)*

### 4.7 Stratification by Population

| Population | Superpop | Train | Test | Total | Train % | Deviation |
|------------|----------|-------|------|-------|---------|-----------|
| ACB | AFR | 82 | 34 | 116 | 70.7% | 9.3% |
| ASW | AFR | 62 | 12 | 74 | 83.8% | 3.8% |
| BEB | SAS | 100 | 31 | 131 | 76.3% | 3.7% |
| CDX | EAS | 76 | 17 | 93 | 81.7% | 1.7% |
| CEU | EUR | 151 | 28 | 179 | 84.4% | 4.4% |
| CHB | EAS | 79 | 24 | 103 | 76.7% | 3.3% |
| CHS | EAS | 129 | 34 | 163 | 79.1% | 0.9% |
| CLM | AMR | 116 | 16 | 132 | 87.9% | 7.9% |
| ESN | AFR | 118 | 31 | 149 | 79.2% | 0.8% |
| FIN | EUR | 85 | 14 | 99 | 85.9% | 5.9% |
| GBR | EUR | 71 | 20 | 91 | 78.0% | 2.0% |
| GIH | SAS | 77 | 26 | 103 | 74.8% | 5.2% |
| GWD | AFR | 151 | 27 | 178 | 84.8% | 4.8% |
| IBS | EUR | 116 | 41 | 157 | 73.9% | 6.1% |
| ITU | SAS | 89 | 18 | 107 | 83.2% | 3.2% |
| JPT | EAS | 87 | 17 | 104 | 83.7% | 3.7% |
| KHV | EAS | 98 | 24 | 122 | 80.3% | 0.3% |
| LWK | AFR | 80 | 19 | 99 | 80.8% | 0.8% |
| MSL | AFR | 79 | 20 | 99 | 79.8% | 0.2% |
| MXL | AMR | 75 | 22 | 97 | 77.3% | 2.7% |
| PEL | AMR | 102 | 20 | 122 | 83.6% | 3.6% |
| PJL | SAS | 126 | 20 | 146 | 86.3% | 6.3% |
| PUR | AMR | 96 | 43 | 139 | 69.1% | 10.9% |
| STU | SAS | 87 | 27 | 114 | 76.3% | 3.7% |
| TSI | EUR | 85 | 22 | 107 | 79.4% | 0.6% |
| YRI | AFR | 143 | 35 | 178 | 80.3% | 0.3% |

**Population deviation statistics:**
- Minimum: 0.2% (MSL)
- Average: 3.7%
- Maximum: 10.9% (PUR)

*Note: Higher deviation in smaller populations is expected due to discrete family units and limited sample sizes.*

---

## 5. Output Files

### 5.1 File Manifest

| File | Description | Records | Size |
|------|-------------|---------|------|
| `1kg_train_test_split.csv` | Full split with metadata | 3,203 (header + 3,202 samples) | 173,240 bytes |
| `1kg_train_samples.txt` | Training sample IDs (one per line) | 2,560 | 20,480 bytes |
| `1kg_test_samples.txt` | Test sample IDs (one per line) | 642 | 5,136 bytes |
| `1kg_stratified_split.py` | Splitting algorithm script | 249 lines | 9,112 bytes |
| `ped_to_json.py` | PED to JSON conversion utility | 281 lines | 8,267 bytes |

### 5.2 File Checksums (MD5)

```
1kg_train_test_split.csv    ca401d4044937f0d492dfaab51443328
1kg_train_samples.txt       e89e5c90e9b269ea6280b62995a25974
1kg_test_samples.txt        0b1d749f7a0906199e847bf10e92a19b
1kg_stratified_split.py     3e66f38ea785a99d23bec6bf82aaf0ee
ped_to_json.py              fa6763af06edee26838376d1ae30657a
```

### 5.3 CSV Schema

The `1kg_train_test_split.csv` file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| SampleID | string | Unique sample identifier |
| FamilyID | string | Original family ID from PED file |
| ComponentID | string | Connected component identifier |
| Sex | integer | 1 = Male, 2 = Female |
| Sex_label | string | "Male" or "Female" |
| Population | string | 3-letter population code |
| Superpopulation | string | 3-letter superpopulation code |
| trio_role | string | "unrelated", "parent", or "child" |
| ComponentSize | integer | Number of individuals in connected component |
| split | string | "train" or "test" |

### 5.4 Script Version Information

| Script | INTERNAL_VERSION | Purpose |
|--------|------------------|---------|
| `1kg_stratified_split.py` | 5 | Main splitting algorithm |
| `ped_to_json.py` | 2 | Convert PED samples to individual JSON files |

---

## 6. Usage

### 6.1 Regenerating the Split

```bash
python3 1kg_stratified_split.py
```

The script reads from the original PED file and generates all output files with identical results (deterministic due to fixed seed).

### 6.2 Loading the Split in Python

```python
import pandas as pd

# Load full split data
df = pd.read_csv('1kg_train_test_split.csv')
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

# Or load sample lists
train_samples = open('1kg_train_samples.txt').read().splitlines()
test_samples = open('1kg_test_samples.txt').read().splitlines()
```

### 6.3 Generating Sample JSON Files

```bash
# Flat output (all JSONs in one directory)
python3 ped_to_json.py -i samples.ped -o /output_dir/

# Subdirectory per sample
python3 ped_to_json.py -i samples.ped -o /output_dir/ --dir-per-sample --create-dirs
```

---

## 7. Limitations and Caveats

1. **Population-level variance:** Smaller populations (e.g., ASW with 74 samples, PUR with 139 samples) exhibit higher deviation from 80% due to the discrete nature of family units.

2. **Rare family sizes:** Components of size 2, 4, and 6 are rare (1, 2, and 4 components respectively), making exact 80/20 splits impossible for these strata.

3. **Indirect stratification:** Sex and trio role are balanced indirectly through component-level assignment. While results show excellent balance, these dimensions are not explicitly constrained in the optimization.

4. **Single split:** This report documents a single train/test split. For cross-validation, multiple splits with different seeds should be generated.

---

## 8. References

- The 1000 Genomes Project Consortium. "A global reference for human genetic variation." *Nature* 526, 68–74 (2015).
- Input file: `20130606_g1k_3202_samples_ped_population.csv` (Phase 3 sample manifest)

---

*Report generated by automated analysis pipeline. Algorithm version: Component-Aware Stratified Split v5.*
