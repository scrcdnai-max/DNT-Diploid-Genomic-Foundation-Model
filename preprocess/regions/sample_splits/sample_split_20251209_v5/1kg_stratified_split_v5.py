#!/usr/bin/env python3
"""
1000 Genomes Stratified Train/Test Split (80/20)
Uses true connected components (not FamilyID) to ensure related individuals
are never split across train/test sets.

INTERNAL_VERSION = 5
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# Read data
df = pd.read_csv('/mnt/user-data/uploads/20130606_g1k_3202_samples_ped_population.csv', 
                 sep=r'\s+', header=0)

print("=" * 60)
print("BUILDING TRUE CONNECTED COMPONENTS")
print("=" * 60)

# Build connected components using union-find
all_samples = set(df['SampleID'].tolist())
edges = set()

# Add parent-child edges
for _, row in df.iterrows():
    sample = row['SampleID']
    father = row['FatherID'] if row['FatherID'] != '0' else None
    mother = row['MotherID'] if row['MotherID'] != '0' else None
    
    if father and father in all_samples:
        edges.add((sample, father))
    if mother and mother in all_samples:
        edges.add((sample, mother))

# Add sibling edges (share at least one parent)
children_by_father = defaultdict(list)
children_by_mother = defaultdict(list)

for _, row in df.iterrows():
    sample = row['SampleID']
    father = row['FatherID'] if row['FatherID'] != '0' else None
    mother = row['MotherID'] if row['MotherID'] != '0' else None
    if father:
        children_by_father[father].append(sample)
    if mother:
        children_by_mother[mother].append(sample)

for parent, children in list(children_by_father.items()) + list(children_by_mother.items()):
    for i, c1 in enumerate(children):
        for c2 in children[i+1:]:
            edges.add((c1, c2))

# Union-find implementation
parent_map = {s: s for s in all_samples}

def find(x: str) -> str:
    if parent_map[x] != x:
        parent_map[x] = find(parent_map[x])
    return parent_map[x]

def union(x: str, y: str) -> None:
    px, py = find(x), find(y)
    if px != py:
        parent_map[px] = py

for a, b in edges:
    union(a, b)

# Group samples by component
components = defaultdict(set)
for s in all_samples:
    components[find(s)].add(s)

# Assign component ID to each sample
sample_to_component = {}
for comp_id, members in components.items():
    for m in members:
        sample_to_component[m] = comp_id

df['ComponentID'] = df['SampleID'].map(sample_to_component)

# Calculate component sizes
component_sizes = {comp_id: len(members) for comp_id, members in components.items()}
df['ComponentSize'] = df['ComponentID'].map(component_sizes)

# Component size distribution
print("\nComponent Size Distribution:")
size_counts = df.groupby('ComponentID')['ComponentSize'].first().value_counts().sort_index()
for size, count in size_counts.items():
    samples = size * count
    print(f"  Size {size}: {count:>4} components = {samples:>4} samples")

# Verify FamilyID vs ComponentID differences
families_per_component = df.groupby('ComponentID')['FamilyID'].nunique()
multi_family_components = families_per_component[families_per_component > 1]
print(f"\nComponents spanning multiple FamilyIDs: {len(multi_family_components)}")

# Derive trio role
sample_set = set(df['SampleID'].tolist())
father_set = set(df[df['FatherID'] != '0']['FatherID'].tolist()) & sample_set
mother_set = set(df[df['MotherID'] != '0']['MotherID'].tolist()) & sample_set
parent_set = father_set | mother_set

df['is_child'] = (df['FatherID'] != '0') | (df['MotherID'] != '0')
df['is_parent'] = df['SampleID'].isin(parent_set)
df['trio_role'] = 'unrelated'
df.loc[df['is_child'], 'trio_role'] = 'child'
df.loc[df['is_parent'], 'trio_role'] = 'parent'
df['Sex_label'] = df['Sex'].map({1: 'Male', 2: 'Female'})


def stratified_split_by_component(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int = 42
) -> pd.DataFrame:
    """
    Split by true connected components, stratified by Superpopulation × ComponentSize.
    """
    np.random.seed(seed)
    
    result = df.copy()
    result['split'] = ''
    
    # Build component-level info
    component_info = result.groupby('ComponentID').agg({
        'Superpopulation': 'first',
        'ComponentSize': 'first',
    }).reset_index()
    
    train_components = set()
    
    # Stratify by Superpopulation × ComponentSize
    for superpop in result['Superpopulation'].unique():
        for comp_size in sorted(result['ComponentSize'].unique()):
            # Get components in this stratum
            stratum_components = component_info[
                (component_info['Superpopulation'] == superpop) & 
                (component_info['ComponentSize'] == comp_size)
            ]['ComponentID'].values
            
            if len(stratum_components) == 0:
                continue
            
            # Shuffle
            np.random.shuffle(stratum_components)
            
            # Calculate target
            n_components = len(stratum_components)
            total_samples = n_components * comp_size
            target_train_samples = total_samples * train_ratio
            n_train_components = int(round(target_train_samples / comp_size))
            
            # Ensure at least 1 in each set if possible
            if n_components >= 2:
                n_train_components = max(1, min(n_train_components, n_components - 1))
            
            train_components.update(stratum_components[:n_train_components])
    
    # Assign splits
    result.loc[result['ComponentID'].isin(train_components), 'split'] = 'train'
    result.loc[~result['ComponentID'].isin(train_components), 'split'] = 'test'
    
    return result


# Create split
df_split = stratified_split_by_component(df, train_ratio=0.8, seed=42)

# Validation
print("\n" + "=" * 60)
print("1000 GENOMES 80/20 TRAIN/TEST SPLIT (COMPONENT-AWARE)")
print("=" * 60)

train_df = df_split[df_split['split'] == 'train']
test_df = df_split[df_split['split'] == 'test']

print(f"\nOverall Split:")
print(f"  Train: {len(train_df):,} samples ({100*len(train_df)/len(df_split):.2f}%)")
print(f"  Test:  {len(test_df):,} samples ({100*len(test_df)/len(df_split):.2f}%)")

# Component integrity check
component_split_counts = df_split.groupby('ComponentID')['split'].nunique()
components_split_across = (component_split_counts > 1).sum()
print(f"\nComponent Integrity: {'✓ PASSED' if components_split_across == 0 else '✗ FAILED'}")
print(f"  Components split across train/test: {components_split_across}")

# Stratification validation
print("\n" + "-" * 60)
print("STRATIFICATION VALIDATION")
print("-" * 60)

print("\nBy Superpopulation:")
print(f"  {'Superpop':<8} {'Train':>8} {'Test':>8} {'Train%':>8} {'Diff':>6}")
for sp in sorted(df_split['Superpopulation'].unique()):
    train_n = len(train_df[train_df['Superpopulation'] == sp])
    test_n = len(test_df[test_df['Superpopulation'] == sp])
    total = train_n + test_n
    train_pct = 100 * train_n / total
    diff = abs(train_pct - 80)
    print(f"  {sp:<8} {train_n:>8} {test_n:>8} {train_pct:>7.1f}% {diff:>5.1f}%")

print("\nBy Sex:")
print(f"  {'Sex':<8} {'Train':>8} {'Test':>8} {'Train%':>8} {'Diff':>6}")
for sex in ['Male', 'Female']:
    train_n = len(train_df[train_df['Sex_label'] == sex])
    test_n = len(test_df[test_df['Sex_label'] == sex])
    total = train_n + test_n
    train_pct = 100 * train_n / total
    diff = abs(train_pct - 80)
    print(f"  {sex:<8} {train_n:>8} {test_n:>8} {train_pct:>7.1f}% {diff:>5.1f}%")

print("\nBy Trio Role:")
print(f"  {'Role':<10} {'Train':>8} {'Test':>8} {'Train%':>8} {'Diff':>6}")
for role in ['unrelated', 'parent', 'child']:
    train_n = len(train_df[train_df['trio_role'] == role])
    test_n = len(test_df[test_df['trio_role'] == role])
    total = train_n + test_n
    train_pct = 100 * train_n / total
    diff = abs(train_pct - 80)
    print(f"  {role:<10} {train_n:>8} {test_n:>8} {train_pct:>7.1f}% {diff:>5.1f}%")

print("\nBy Component Size:")
print(f"  {'Size':<6} {'Train':>8} {'Test':>8} {'Train%':>8}")
for size in sorted(df_split['ComponentSize'].unique()):
    train_n = len(train_df[train_df['ComponentSize'] == size])
    test_n = len(test_df[test_df['ComponentSize'] == size])
    total = train_n + test_n
    train_pct = 100 * train_n / total if total > 0 else 0
    print(f"  {size:<6} {train_n:>8} {test_n:>8} {train_pct:>7.1f}%")

# Export
output_cols = ['SampleID', 'FamilyID', 'ComponentID', 'Sex', 'Sex_label', 'Population', 
               'Superpopulation', 'trio_role', 'ComponentSize', 'split']
df_split[output_cols].to_csv('/mnt/user-data/outputs/1kg_train_test_split.csv', index=False)

train_df[['SampleID']].to_csv('/mnt/user-data/outputs/1kg_train_samples.txt', 
                               index=False, header=False)
test_df[['SampleID']].to_csv('/mnt/user-data/outputs/1kg_test_samples.txt', 
                              index=False, header=False)

print("\n" + "=" * 60)
print("OUTPUT FILES")
print("=" * 60)
print(f"  Full split:    /mnt/user-data/outputs/1kg_train_test_split.csv")
print(f"  Train samples: /mnt/user-data/outputs/1kg_train_samples.txt ({len(train_df)} samples)")
print(f"  Test samples:  /mnt/user-data/outputs/1kg_test_samples.txt ({len(test_df)} samples)")
